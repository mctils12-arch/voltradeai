"""
[REPAIR] regression test, 2026-08-10 scheduled-routine session.

LIVE EVIDENCE this session: /api/diag/audit (production) showed
"Multiple API sources down: ['polygon', 'wikipedia', 'gdelt', 'fred']"
firing on every TIER3-DIAG entry across a 2+ day, 40+ hour window
spanning multiple daemon restarts and confirmed-active TIER2 scan cycles
(/api/diag/timings showed a real, non-trivial deep_score phase duration
each cycle, ruling out "deep_score never runs"). /api/diag/scanner's
dataSourceErrors stayed empty (no captured exception from macro/intel/
alt/social/finnhub), ruling out an exception-raising bug in the fetchers
themselves — confirmed locally too: `python3 -c "import macro_data;
macro_data.get_macro_snapshot()"` completes and writes its cache file
cleanly with zero network access (every branch degrades to a default).

READ BEFORE WRITE (this session) found the actual mechanism:
get_alt_data_score() (alt_data.py), the only alt-data caller of
get_wiki_attention/get_fred_macro/get_geopolitical_risk, makes ~19
SEQUENTIAL blocking HTTP requests in one call — 7 for FRED (one per
series in FRED_SERIES, each `timeout=10`) and 5 for GDELT (one per
RISK_QUERIES entry, each `timeout=8` PLUS an explicit `time.sleep(0.3)`
between them) among others. But bot_engine.py's deep_score() only gives
each of its 5 parallel fetchers a 15s total budget
(`_f_alt.result(timeout=15)`) before giving up and defaulting to `{}` —
a budget these two inner loops alone can only meet by coincidence, not
reliably, under any real-world per-call latency. This test proves the
fix: fetching each loop's items CONCURRENTLY instead of sequentially
turns "sum of N call latencies" into "roughly the slowest single call",
which is what actually keeps get_alt_data_score() inside its caller's
budget. See open_questions.md KNOWN BROKEN for the full evidence chain
and what remains unconfirmed (whether this alone fully explains 2+ days
of zero cache hits, or whether some fraction was also restart/rate-limit
related) — this session could not get a live traceback from the running
container, so the fix targets the one concretely-code-evidenced defect
found by reading the actual call graph, not a guess.
"""
import time
import unittest
from unittest.mock import patch

import alt_data


class _FakeResp:
    def __init__(self, text="", json_data=None):
        self.text = text
        self._json = json_data if json_data is not None else {}

    def json(self):
        return self._json


def _fred_csv(value):
    return "DATE,VALUE\n2026-01-01,1.0\n2026-01-02," + str(value) + "\n"


class TestFredMacroConcurrency(unittest.TestCase):
    def setUp(self):
        patch("alt_data._cache_get", return_value=None).start()
        patch("alt_data._cache_set", return_value=None).start()
        self.addCleanup(patch.stopall)

    def test_seven_series_fetched_concurrently_not_sequentially(self):
        """Each of the 7 FRED_SERIES calls sleeps 0.5s in this fake — a
        sequential loop would take >= 3.5s total; a concurrent fetch should
        take well under 1.5s (bounded by roughly one call's latency plus
        scheduling overhead, not the sum)."""
        per_call_delay = 0.5

        def _fake_get(url, timeout=None, **kw):
            time.sleep(per_call_delay)
            return _FakeResp(text=_fred_csv(3.14))

        with patch("alt_data.requests.get", side_effect=_fake_get):
            start = time.time()
            result = alt_data.get_fred_macro()
            elapsed = time.time() - start

        self.assertLess(
            elapsed, len(alt_data.FRED_SERIES) * per_call_delay * 0.75,
            "FRED_SERIES fetches must run concurrently — elapsed time should "
            "not scale with the number of series",
        )
        # Correctness: every series still resolves to the parsed value.
        for name in alt_data.FRED_SERIES:
            self.assertIn(name, result)
            self.assertEqual(result[name], 3.14)

    def test_partial_failure_still_fills_remaining_series(self):
        """One series erroring must not prevent the others from being
        recorded — each future is independent."""
        def _fake_get(url, timeout=None, **kw):
            if "T10Y2Y" in url:  # yield_curve series id
                raise ConnectionError("simulated network failure")
            return _FakeResp(text=_fred_csv(2.0))

        with patch("alt_data.requests.get", side_effect=_fake_get):
            result = alt_data.get_fred_macro()

        self.assertNotIn("yield_curve", result)
        self.assertIn("unemployment", result)
        self.assertEqual(result["unemployment"], 2.0)


class TestGeopoliticalRiskConcurrency(unittest.TestCase):
    def setUp(self):
        patch("alt_data._cache_get", return_value=None).start()
        patch("alt_data._cache_set", return_value=None).start()
        self.addCleanup(patch.stopall)

    def test_five_queries_fetched_concurrently_not_sequentially(self):
        """Each of the 5 RISK_QUERIES sleeps 0.5s in this fake. The old code
        also added a 0.3s sleep *between* each sequential call, so the old
        floor was ~5*(0.5+0.3)=4s; concurrent fetch should land well under
        2s."""
        per_call_delay = 0.5

        def _fake_get(url, timeout=None, **kw):
            time.sleep(per_call_delay)
            return _FakeResp(json_data={"articles": []})

        with patch("alt_data.requests.get", side_effect=_fake_get):
            start = time.time()
            alt_data.get_geopolitical_risk()
            elapsed = time.time() - start

        self.assertLess(
            elapsed, len(alt_data.RISK_QUERIES) * (per_call_delay + 0.3) * 0.6,
            "RISK_QUERIES fetches must run concurrently — elapsed time should "
            "not scale with the number of queries",
        )

    def test_result_assembly_order_independent_of_completion_order(self):
        """Even if queries complete out of order (simulated via variable
        per-query delay), active_risks/top_headline must come out identical
        to the deterministic RISK_QUERIES order — matching the old
        sequential behavior byte-for-byte, just faster."""
        cutoff_date = "20990101"  # always "recent" — before any real 48h cutoff check runs
        # Give every query 3+ recent articles so all 5 qualify for active_risks.
        articles = [{"seendate": cutoff_date, "title": "headline"} for _ in range(3)]

        # Reverse-order latency: the LAST query in RISK_QUERIES finishes
        # first, the FIRST query finishes last — completion order is the
        # exact opposite of RISK_QUERIES order.
        delays = {q: 0.05 * (len(alt_data.RISK_QUERIES) - i)
                  for i, q in enumerate(alt_data.RISK_QUERIES)}

        def _fake_get(url, timeout=None, **kw):
            for q, d in delays.items():
                if alt_data.requests.utils.quote(q) in url:
                    time.sleep(d)
                    break
            return _FakeResp(json_data={"articles": articles})

        with patch("alt_data.requests.get", side_effect=_fake_get):
            with patch("alt_data.datetime") as mock_dt:
                # Freeze "now" far enough in the future that cutoff_date
                # above always counts as within the last 48h.
                import datetime as _real_datetime
                mock_dt.now.return_value = _real_datetime.datetime(2099, 1, 2)
                mock_dt.side_effect = lambda *a, **kw: _real_datetime.datetime(*a, **kw)
                result = alt_data.get_geopolitical_risk()

        expected_first_risk = alt_data.RISK_QUERIES[0].split()[0]
        self.assertEqual(
            result["active_risks"][0], expected_first_risk,
            "active_risks must be ordered by RISK_QUERIES position, not by "
            "which concurrent fetch happened to finish first",
        )
        self.assertEqual(len(result["active_risks"]), len(alt_data.RISK_QUERIES))


if __name__ == "__main__":
    unittest.main()
