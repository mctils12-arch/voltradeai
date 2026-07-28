"""
Regression tests for scripts/illiquid_universe_probe_traintest.py.

split_bars() tests are pure (no network, no backtest_v2 import needed).
run_split_group() tests mock backtest_v2.fetch_bars only and exercise the
REAL backtest_v2.run_backtest against synthetic in-memory bars (it accepts
injected bars/spy/vxx and does no network I/O in that mode) — matches
test_backtest_v2_liquidity_cost.py's convention of real integration over
synthetic data rather than mocking run_backtest itself.
"""
import importlib.util
import os
import unittest
from datetime import date, timedelta
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    "illiquid_universe_probe_traintest",
    os.path.join(os.path.dirname(__file__), "scripts", "illiquid_universe_probe_traintest.py"))
traintest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(traintest)


def _synthetic_bars(n, volume=500_000, start=100.0, daily=0.0015, d0=date(2020, 1, 2)):
    """Deterministic oscillating-uptrend OHLCV bars (no randomness), long
    enough (n ~= 1000+) to produce real momentum/mean_reversion trades on
    both an early and a late half after the 252-day warmup."""
    dates, o, h, l, c, v = [], [], [], [], [], []
    px = start
    for i in range(n):
        wiggle = 0.02 * ((i % 11) - 5) / 5.0
        op = px
        cl = px * (1 + daily + wiggle)
        cl = max(cl, 0.01)
        dates.append((d0 + timedelta(days=i)).isoformat())
        o.append(round(op, 4)); c.append(round(cl, 4))
        h.append(round(max(op, cl) * 1.01, 4))
        l.append(round(min(op, cl) * 0.99, 4))
        v.append(volume)
        px = cl
    return {"date": dates, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _flat_vxx(n, d0=date(2020, 1, 2)):
    """VXX series whose close/30d-avg ratio hovers near 1.0 (NEUTRAL-ish
    input) — good enough for these tests, which only need SOME regime
    lookup to succeed by date, not a specific regime."""
    dates = [(d0 + timedelta(days=i)).isoformat() for i in range(n)]
    closes = [20.0 + 0.5 * ((i % 13) - 6) for i in range(n)]
    return {"date": dates, "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [2_000_000] * n}


class TestSplitBars(unittest.TestCase):
    def test_correct_split_point(self):
        bars = _synthetic_bars(200)
        early, late = traintest.split_bars(bars, warmup_days=50)
        self.assertEqual(len(early["date"]), 100)  # mid = 200 // 2
        # late_start = max(0, 100 - 50) = 50 -> late spans [50:200] = 150 bars
        self.assertEqual(len(late["date"]), 150)

    def test_late_window_prepend_is_min_warmup_and_mid(self):
        bars = _synthetic_bars(200)
        mid = 100
        # Case 1: warmup_days (50) < mid (100) -> prepend exactly 50 days.
        _, late = traintest.split_bars(bars, warmup_days=50)
        self.assertEqual(len(late["date"]), (200 - mid) + 50)

        # Case 2: warmup_days (500) > mid (100) -> prepend capped at mid,
        # i.e. late_start clamps to 0 and late == the full bars dict.
        _, late_capped = traintest.split_bars(bars, warmup_days=500)
        self.assertEqual(len(late_capped["date"]), 200)
        self.assertEqual(late_capped["date"], bars["date"])

    def test_fields_stay_aligned_no_off_by_one(self):
        n = 60
        bars = _synthetic_bars(n)
        early, late = traintest.split_bars(bars, warmup_days=10)
        mid = n // 2
        late_start = max(0, mid - 10)
        # early must be an exact prefix
        self.assertEqual(early["date"], bars["date"][:mid])
        self.assertEqual(early["close"], bars["close"][:mid])
        # late must be an exact suffix starting at late_start, and every
        # field must have moved together (same index -> same trading day)
        self.assertEqual(late["date"], bars["date"][late_start:])
        self.assertEqual(late["close"], bars["close"][late_start:])
        self.assertEqual(late["volume"], bars["volume"][late_start:])
        for field in ("date", "open", "high", "low", "close", "volume"):
            self.assertEqual(len(early[field]), len(early["date"]))
            self.assertEqual(len(late[field]), len(late["date"]))

    def test_no_lookahead_late_prepend_is_strictly_before_late_nominal_start(self):
        """Every day prepended into the late half must sit strictly before
        the late half's own nominal (post-midpoint) start date — proves the
        prepend is walk-forward warmup, not a peek into the future."""
        n = 300
        bars = _synthetic_bars(n)
        mid = n // 2
        early, late = traintest.split_bars(bars, warmup_days=100)
        late_start = max(0, mid - 100)
        nominal_late_start_date = bars["date"][mid]
        prepended_dates = late["date"][:mid - late_start]
        self.assertTrue(all(d < nominal_late_start_date for d in prepended_dates))
        # and the early half never sees anything past its own midpoint
        self.assertTrue(all(d < nominal_late_start_date for d in early["date"]))

    def test_raises_on_too_few_bars(self):
        bars = _synthetic_bars(3)
        with self.assertRaises(ValueError):
            traintest.split_bars(bars)

    def test_raises_on_mismatched_field_lengths(self):
        bars = _synthetic_bars(20)
        bars["close"] = bars["close"][:-1]  # desync one field
        with self.assertRaises(ValueError):
            traintest.split_bars(bars)


class TestRunSplitGroup(unittest.TestCase):
    def setUp(self):
        # ~1010 trading days (~4y) so both early and late halves clear the
        # 252-day momentum warmup with room to spare for real trades.
        self.n = 1010
        self.spy_bars = _synthetic_bars(self.n, volume=50_000_000, daily=0.0006)
        self.vxx_bars = _flat_vxx(self.n)
        self.ticker_bars = {
            "GOODTICK": _synthetic_bars(self.n, volume=400_000, daily=0.0012),
        }

    def _fake_fetch(self, symbol, days):
        if symbol == "SPY":
            return self.spy_bars
        if symbol == "VXX":
            return self.vxx_bars
        if symbol in self.ticker_bars:
            return self.ticker_bars[symbol]
        raise RuntimeError(f"no fake data for {symbol}")

    def test_produces_early_and_late_rows_for_a_good_ticker(self):
        with patch("backtest_v2.fetch_bars", side_effect=self._fake_fetch):
            early_rows, late_rows = traintest.run_split_group(["GOODTICK"], years=4)
        self.assertEqual(len(early_rows), 1)
        self.assertEqual(len(late_rows), 1)
        for row in (early_rows[0], late_rows[0]):
            self.assertNotIn("error", row)
            self.assertIn("momentum", row)
            self.assertIn("mean_reversion", row)
            self.assertIn("buy_and_hold_pct", row)
            self.assertIsInstance(row["momentum"]["sharpe"], float)

    def test_one_bad_ticker_does_not_kill_the_whole_group(self):
        def fetch_with_one_failure(symbol, days):
            if symbol == "BROKEN":
                raise RuntimeError("yahoo fetch failed")
            return self._fake_fetch(symbol, days)

        with patch("backtest_v2.fetch_bars", side_effect=fetch_with_one_failure):
            early_rows, late_rows = traintest.run_split_group(["BROKEN", "GOODTICK"], years=4)

        self.assertEqual(len(early_rows), 2)
        self.assertEqual(len(late_rows), 2)
        broken_early = next(r for r in early_rows if r["ticker"] == "BROKEN")
        broken_late = next(r for r in late_rows if r["ticker"] == "BROKEN")
        self.assertIn("error", broken_early)
        self.assertIn("error", broken_late)
        good_early = next(r for r in early_rows if r["ticker"] == "GOODTICK")
        self.assertNotIn("error", good_early)

    def test_spy_and_vxx_fetched_once_shared_across_tickers_and_halves(self):
        calls = []

        def counting_fetch(symbol, days):
            calls.append(symbol)
            return self._fake_fetch(symbol, days)

        self.ticker_bars["GOODTICK2"] = _synthetic_bars(self.n, volume=300_000, daily=0.0009)
        with patch("backtest_v2.fetch_bars", side_effect=counting_fetch):
            traintest.run_split_group(["GOODTICK", "GOODTICK2"], years=4)

        # SPY and VXX fetched exactly once each, regardless of ticker count
        # or the fact that both an early and a late run happen per ticker.
        self.assertEqual(calls.count("SPY"), 1)
        self.assertEqual(calls.count("VXX"), 1)
        self.assertEqual(calls.count("GOODTICK"), 1)
        self.assertEqual(calls.count("GOODTICK2"), 1)

    def test_summarize_is_reused_unchanged_on_each_half(self):
        import illiquid_universe_probe as orig
        with patch("backtest_v2.fetch_bars", side_effect=self._fake_fetch):
            early_rows, late_rows = traintest.run_split_group(["GOODTICK"], years=4)
        early_summary = orig.summarize(early_rows)
        late_summary = orig.summarize(late_rows)
        self.assertEqual(early_summary["n"], 1)
        self.assertEqual(late_summary["n"], 1)
        self.assertIn("mean_reversion", early_summary)
        self.assertIn("mean_reversion", late_summary)


if __name__ == "__main__":
    unittest.main()
