"""
Regression tests for scripts/usaspending_gate2.py — the ROOT VALIDATION
LADDER gate 2 (SIGNAL) screen for the USAspending contract-award archive.
Pure-function tests only: no network calls (no live diag probe, no SEC
EDGAR, no backtest_v2 Alpaca/Yahoo fetch).
"""
import importlib.util
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "usaspending_gate2",
    os.path.join(os.path.dirname(__file__), "scripts", "usaspending_gate2.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_events = _mod.build_events
significance_by_bucket = _mod.significance_by_bucket
summarize_by_bucket = _mod.summarize_by_bucket
compute_forward_returns = _mod.compute_forward_returns
compute_ticker_baseline = _mod.compute_ticker_baseline
HORIZONS = _mod.HORIZONS


def _row(aid="A1", mod="0", amt=100000.0, rt="2026-07-05", tkr="AAA",
         mm="name", ag="Department of the Treasury"):
    return {"aid": aid, "mod": mod, "amt": amt, "rt": rt, "tkr": tkr, "mm": mm, "ag": ag}


class TestBuildEvents(unittest.TestCase):
    def test_dod_agency_excluded(self):
        rows = [_row(ag="Department of Defense")]
        self.assertEqual(build_events(rows), {})

    def test_unmatched_ticker_excluded(self):
        rows = [_row(tkr=None)]
        self.assertEqual(build_events(rows), {})

    def test_parent_match_method_excluded(self):
        # mm="parent" carries the gate-1 CAVEAT-1 residual mismatch risk
        # (e.g. BALL CORPORATION as BAE Systems' FPDS-reported parent) and
        # is deliberately not trusted by this screen.
        rows = [_row(mm="parent")]
        self.assertEqual(build_events(rows), {})

    def test_name_and_cache_methods_kept(self):
        rows = [_row(mm="name", aid="A1", rt="2026-07-05"), _row(mm="cache", aid="A2", rt="2026-07-06")]
        events = build_events(rows)
        self.assertEqual(len(events["AAA"]), 2)

    def test_civilian_agency_kept(self):
        rows = [_row(ag="Department of the Treasury")]
        events = build_events(rows)
        self.assertEqual(len(events["AAA"]), 1)

    def test_duplicate_aid_mod_amt_deduped(self):
        # Same action re-polled on a later day (rt advances) is not a new
        # event — first sighting (earliest rt, given day-ordered input) wins.
        rows = [_row(rt="2026-07-05", amt=50000.0), _row(rt="2026-07-08", amt=50000.0)]
        events = build_events(rows)
        self.assertEqual(len(events["AAA"]), 1)
        self.assertEqual(events["AAA"][0]["filing_date"], "2026-07-05")
        self.assertEqual(events["AAA"][0]["amt"], 50000.0)  # not double-counted

    def test_different_mod_is_a_new_event_not_a_duplicate(self):
        rows = [_row(mod="0", amt=50000.0), _row(mod="1", amt=75000.0)]
        events = build_events(rows)
        self.assertEqual(len(events["AAA"]), 1)  # same day -> aggregated
        self.assertEqual(events["AAA"][0]["amt"], 125000.0)

    def test_same_ticker_same_day_awards_summed(self):
        rows = [_row(aid="A1", amt=10000.0), _row(aid="A2", amt=20000.0)]
        events = build_events(rows)
        self.assertEqual(len(events["AAA"]), 1)
        self.assertEqual(events["AAA"][0]["amt"], 30000.0)

    def test_events_sorted_chronologically(self):
        rows = [_row(aid="A2", rt="2026-07-10"), _row(aid="A1", rt="2026-07-05")]
        events = build_events(rows)
        self.assertEqual([e["filing_date"] for e in events["AAA"]], ["2026-07-05", "2026-07-10"])

    def test_missing_amt_or_rt_excluded(self):
        rows = [_row(amt=None), {"aid": "A2", "mod": "0", "rt": None, "tkr": "AAA",
                                  "mm": "name", "ag": "Department of the Treasury", "amt": 1000.0}]
        self.assertEqual(build_events(rows), {})


class TestComputeForwardReturnsUsesOwnHorizons(unittest.TestCase):
    # Regression guard for the exact bug found while first running this
    # script live: form4_gate2_test.py's compute_forward_returns/
    # compute_ticker_baseline close over THAT module's HORIZONS=(20,60),
    # not this module's (5,20) — reusing them unmodified silently computed
    # the wrong horizons. These two tests pin that this module's local
    # copies key off HORIZONS=(5,20), not (20,60).
    def test_forward_returns_populate_this_modules_horizons(self):
        self.assertEqual(HORIZONS, (5, 20))
        dates = [f"2026-01-{d:02d}" for d in range(1, 32)]
        closes = [100.0 + i for i in range(len(dates))]
        bars = {"date": dates, "close": closes}
        events = [{"filing_date": "2026-01-01", "amt": 1000.0}]
        rows = compute_forward_returns(events, bars)
        self.assertIn(5, rows[0]["forward_returns"])
        self.assertIn(20, rows[0]["forward_returns"])
        self.assertNotIn(60, rows[0]["forward_returns"])  # form4's horizon, not ours

    def test_baseline_keyed_by_this_modules_horizons(self):
        n = 100
        bars = {"date": [f"d{i}" for i in range(n)], "close": [100.0 + i for i in range(n)]}
        baseline = compute_ticker_baseline(bars, [])
        self.assertEqual(set(baseline.keys()), {5, 20})
        self.assertEqual(len(baseline[5]), n - 5)
        self.assertEqual(len(baseline[20]), n - 20)


class TestSummarizeAndSignificance(unittest.TestCase):
    def test_summarize_separates_high_low_ratio_buckets(self):
        rows = [
            {"high_ratio": True, "forward_returns": {5: 0.10}},
            {"high_ratio": False, "forward_returns": {5: 0.01}},
        ]
        baseline = {5: [0.0, 0.02], 20: []}
        summary = summarize_by_bucket(rows, baseline)
        self.assertEqual(summary["5"]["high_ratio"]["n"], 1)
        self.assertAlmostEqual(summary["5"]["high_ratio"]["mean_pct"], 10.0)
        self.assertEqual(summary["5"]["low_ratio"]["n"], 1)
        self.assertEqual(summary["5"]["baseline"]["n"], 2)

    def test_significance_none_for_thin_samples(self):
        rows = [{"high_ratio": True, "forward_returns": {5: 0.10}}]
        baseline = {5: [0.01, 0.02], 20: []}
        sig = significance_by_bucket(rows, baseline)
        self.assertIsNone(sig["5"]["high_ratio"])

    def test_significance_computes_for_adequate_samples(self):
        rows = [{"high_ratio": True, "forward_returns": {5: 0.20}} for _ in range(10)]
        baseline = {5: [0.0] * 10, 20: []}
        sig = significance_by_bucket(rows, baseline)
        self.assertIsNotNone(sig["5"]["high_ratio"])
        self.assertEqual(sig["5"]["high_ratio"]["n"], 10)
        self.assertAlmostEqual(sig["5"]["high_ratio"]["mean_diff_pct"], 20.0)

    def test_low_ratio_bucket_not_contaminated_by_high_ratio_rows(self):
        rows = [
            {"high_ratio": True, "forward_returns": {5: 1.0, 20: 1.0}},
            {"high_ratio": False, "forward_returns": {5: -1.0, 20: -1.0}},
        ]
        baseline = {5: [0.0] * 5, 20: [0.0] * 5}
        summary = summarize_by_bucket(rows, baseline)
        self.assertAlmostEqual(summary["5"]["low_ratio"]["mean_pct"], -100.0)
        self.assertAlmostEqual(summary["5"]["high_ratio"]["mean_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
