"""
KNOWN BROKEN #18, fourth-mechanism investigation (rate-limiter contention
during deep_score) — regression test for get_macro_snapshot()'s
self-duplicate SPY bars fetch, 2026-07-19.

READ BEFORE WRITE found get_macro_snapshot() (macro_data.py) fetched the
identical 300-day/220-limit SPY daily-bars request TWICE in the same
function call: once for the 200-day MA slow-bear detector, then again a
few lines later for the 50-day MA trend gauge — whose own comment already
said "We already fetched 300 days of SPY above for MA200 calc — reuse"
without the code actually doing so. Every real (non-cached) call to
get_macro_snapshot() was making one fully wasted Alpaca request, funneling
through the same shared alpaca_throttle bucket (alpaca_rate_limiter.py,
FROZEN, 180 req/min process-wide) implicated in the ongoing TIER2-ERROR
"Daemon timeout" recurrence. This test proves the second fetch is gone and
that both derived values (spy_below_200_days/spy_vs_ma200 from the first
block, spy_vs_ma50 from the second) are still computed correctly from the
single shared fetch.
"""
import unittest
from unittest.mock import patch, MagicMock

import macro_data


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def _spy_bars(n=220, start_price=400.0, step=0.1):
    return [{"c": round(start_price + i * step, 2)} for i in range(n)]


def _make_fake_get(call_counts, spy_bars):
    def _fake_get(url, params=None, headers=None, timeout=None, **kw):
        params = params or {}
        if params.get("symbols") == "SPY" and params.get("limit") == 220:
            call_counts["spy_300d"] += 1
            return _FakeResp({"bars": {"SPY": spy_bars}})
        if params.get("symbols") == "VXX":
            return _FakeResp({"bars": {"VXX": [{"c": 20.0 + i * 0.01} for i in range(40)]}})
        # Sector-momentum snapshot batch call has no `params` (symbols are
        # baked into the URL) — falls through here, empty dict is a
        # perfectly valid "no snapshot data" response for that code path.
        return _FakeResp({})

    return _fake_get


def _empty_yf_ticker(*a, **kw):
    m = MagicMock()
    m.history.return_value = MagicMock(empty=True)
    return m


class TestMacroSnapshotSpyDedup(unittest.TestCase):
    def setUp(self):
        # Bypass the 5-minute disk cache so each test exercises a real
        # (mocked-network) computation, and don't let a test write to the
        # real /tmp/voltrade_macro_cache.json the live system reads.
        patch("macro_data._load_cache", return_value=None).start()
        patch("macro_data._save_cache", return_value=None).start()
        patch("yfinance.Ticker", side_effect=_empty_yf_ticker).start()
        self.addCleanup(patch.stopall)

    def test_spy_300d_bars_fetched_exactly_once(self):
        call_counts = {"spy_300d": 0}
        spy_bars = _spy_bars()
        with patch("macro_data.requests.get", side_effect=_make_fake_get(call_counts, spy_bars)):
            result = macro_data.get_macro_snapshot()

        self.assertEqual(
            call_counts["spy_300d"], 1,
            "the 300-day/220-limit SPY bars request is identical in both "
            "the MA200 and MA50 blocks — the second block must reuse the "
            "first block's fetch instead of re-issuing it",
        )
        # Correctness preserved: both derived values are still populated
        # from the single shared fetch, not silently dropped.
        self.assertIn("spy_vs_ma200", result)
        self.assertIn("spy_below_200_days", result)
        self.assertIn("spy_vs_ma50", result)
        self.assertNotEqual(result["spy_vs_ma50"], 1.0,
                             "1.0 is the exception-path default — a real "
                             "value must come from the reused bars")

    def test_spy_vs_ma50_matches_a_direct_single_fetch_computation(self):
        """The reused-bars value must equal what a from-scratch fetch of the
        same window would have produced — proves the dedup didn't change
        WHICH data feeds the MA50 calc, only how many times it's fetched."""
        call_counts = {"spy_300d": 0}
        spy_bars = _spy_bars(start_price=450.0, step=0.25)
        with patch("macro_data.requests.get", side_effect=_make_fake_get(call_counts, spy_bars)):
            result = macro_data.get_macro_snapshot()

        closes = [b["c"] for b in spy_bars]
        expected_ma50 = sum(closes[-50:]) / 50
        expected_ratio = round(closes[-1] / expected_ma50, 4)
        self.assertEqual(result["spy_vs_ma50"], expected_ratio)

    def test_short_first_fetch_falls_back_to_a_live_refetch(self):
        """If the first (MA200) fetch comes back too short to reuse (<50
        bars — e.g. a partial/degraded response), the MA50 block's fallback
        fetch must still run rather than silently reporting the 1.0
        default. Guards against an over-eager dedup that trusts a fetch it
        shouldn't."""
        call_counts = {"spy_300d": 0}
        short_bars = _spy_bars(n=10)  # below both the >=200 and >=50 gates
        with patch("macro_data.requests.get", side_effect=_make_fake_get(call_counts, short_bars)):
            macro_data.get_macro_snapshot()

        self.assertEqual(
            call_counts["spy_300d"], 2,
            "a too-short first fetch must not be reused as-is — the MA50 "
            "block's own live-fetch fallback must still fire",
        )


if __name__ == "__main__":
    unittest.main()
