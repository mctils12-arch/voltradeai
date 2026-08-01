"""
Regression tests for scripts/illiquid_universe_probe_regime.py.

build_regime_lookup / tag_trades_with_regime / aggregate_by_regime are
pure (no network) and tested directly on hand-built inputs.
run_group_regime mocks backtest_v2.fetch_bars only and exercises the REAL
backtest_v2.run_backtest / regime_series against synthetic in-memory bars
(matches test_illiquid_universe_probe_traintest.py's convention of real
integration over synthetic data rather than mocking run_backtest itself).
"""
import importlib.util
import os
import unittest
from datetime import date, timedelta
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    "illiquid_universe_probe_regime",
    os.path.join(os.path.dirname(__file__), "scripts", "illiquid_universe_probe_regime.py"))
regime_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(regime_probe)


def _synthetic_bars(n, volume=500_000, start=100.0, daily=0.0015, d0=date(2020, 1, 2)):
    """Deterministic oscillating-uptrend OHLCV bars (no randomness), long
    enough to produce real mean_reversion trades after warmup."""
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


def _stepped_vxx(n, base=20.0, jump_at=None, jump_level=None, d0=date(2020, 1, 2)):
    """VXX close held FLAT at `base` (so vxx_ratio == close/sma30 converges
    to exactly 1.0 well within the flat run — sma of a constant series
    equals that constant), then STEPS to `jump_level` at index `jump_at`.
    Right at/just after the step, sma30 is still dominated by the
    pre-jump `base` values (the trailing 30-day window straddles the
    jump), so vxx_ratio is transiently pulled far from 1.0 in the jump's
    direction before decaying back to 1.0 once the window fully refills
    with `jump_level` (~30 days after the jump). This is what lets a
    single-index probe right at the jump reliably land in a KNOWN, chosen
    regime bucket, unlike a flat-forever series (which always converges
    to ratio==1.0 no matter what absolute level it's flat AT)."""
    dates = [(d0 + timedelta(days=i)).isoformat() for i in range(n)]
    closes = [base if (jump_at is None or i < jump_at) else jump_level for i in range(n)]
    return {"date": dates, "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [2_000_000] * n}


class TestBuildRegimeLookup(unittest.TestCase):
    def test_upward_vxx_step_produces_panic_right_at_the_jump(self):
        n = 400
        spy_bars = _synthetic_bars(n, volume=50_000_000, daily=0.0008)
        # base=20 flat for 300 days (ratio converges to 1.0), then jumps to
        # 100 at i=300: ratio at i=300 = 100*30/(29*20+100) = 4.41 -> PANIC
        vxx_bars = _stepped_vxx(n, base=20.0, jump_at=300, jump_level=100.0)
        lookup = regime_probe.build_regime_lookup(spy_bars, vxx_bars)
        self.assertEqual(len(lookup), n)
        self.assertEqual(lookup[spy_bars["date"][300]], "PANIC")

    def test_downward_vxx_step_produces_bull_right_at_the_jump(self):
        n = 400
        spy_bars = _synthetic_bars(n, volume=50_000_000, daily=0.0008)
        # base=20 flat for 300 days, then jumps to 5 at i=300: ratio at
        # i=300 = 5*30/(29*20+5) = 0.256 -> <=0.95, combined with the
        # uptrending/above-200d synthetic SPY -> BULL
        vxx_bars = _stepped_vxx(n, base=20.0, jump_at=300, jump_level=5.0)
        lookup = regime_probe.build_regime_lookup(spy_bars, vxx_bars)
        self.assertEqual(lookup[spy_bars["date"][300]], "BULL")

    def test_missing_vxx_degrades_to_a_valid_label_not_a_crash(self):
        n = 300
        spy_bars = _synthetic_bars(n, volume=50_000_000, daily=0.0008)
        lookup = regime_probe.build_regime_lookup(spy_bars, None)
        self.assertEqual(len(lookup), n)
        self.assertIn(lookup[spy_bars["date"][200]], regime_probe.REGIME_LEVELS)


class TestTagTradesWithRegime(unittest.TestCase):
    def test_tags_each_trade_from_lookup(self):
        trades = [
            {"date_entry": "2020-01-05", "net_pct": 1.0},
            {"date_entry": "2020-06-01", "net_pct": -2.0},
        ]
        lookup = {"2020-01-05": "BULL", "2020-06-01": "BEAR"}
        tagged = regime_probe.tag_trades_with_regime(trades, lookup)
        self.assertEqual(tagged[0]["regime"], "BULL")
        self.assertEqual(tagged[1]["regime"], "BEAR")
        # original fields preserved
        self.assertEqual(tagged[0]["net_pct"], 1.0)

    def test_missing_date_falls_back_to_neutral(self):
        trades = [{"date_entry": "1999-01-01", "net_pct": 0.5}]
        tagged = regime_probe.tag_trades_with_regime(trades, {})
        self.assertEqual(tagged[0]["regime"], "NEUTRAL")


class TestAggregateByRegime(unittest.TestCase):
    def test_win_rate_and_mean_net_pct_per_bucket(self):
        tagged = [
            {"regime": "BULL", "net_pct": 5.0},
            {"regime": "BULL", "net_pct": -1.0},
            {"regime": "BEAR", "net_pct": 2.0},
            {"regime": "BEAR", "net_pct": 4.0},
        ]
        out = regime_probe.aggregate_by_regime(tagged)
        self.assertEqual(out["BULL"]["n_trades"], 2)
        self.assertEqual(out["BULL"]["win_rate_pct"], 50.0)
        self.assertAlmostEqual(out["BULL"]["mean_net_pct"], 2.0)
        self.assertEqual(out["BEAR"]["n_trades"], 2)
        self.assertEqual(out["BEAR"]["win_rate_pct"], 100.0)
        self.assertAlmostEqual(out["BEAR"]["mean_net_pct"], 3.0)

    def test_empty_bucket_is_omitted_not_zero(self):
        tagged = [{"regime": "BULL", "net_pct": 1.0}]
        out = regime_probe.aggregate_by_regime(tagged)
        self.assertIn("BULL", out)
        self.assertNotIn("BEAR", out)
        self.assertNotIn("PANIC", out)

    def test_no_trades_returns_empty_dict(self):
        self.assertEqual(regime_probe.aggregate_by_regime([]), {})


class TestRunGroupRegime(unittest.TestCase):
    def setUp(self):
        self.n = 1010  # ~4y, clears warmup with room for real trades
        self.spy_bars = _synthetic_bars(self.n, volume=50_000_000, daily=0.0006)
        self.vxx_bars = _stepped_vxx(self.n, base=20.0, jump_at=self.n // 2, jump_level=35.0)
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

    def test_produces_regime_buckets_summing_to_total_trades(self):
        with patch("backtest_v2.fetch_bars", side_effect=self._fake_fetch):
            result = regime_probe.run_group_regime(["GOODTICK"], years=4)
        self.assertEqual(result["n_tickers"], 1)
        self.assertEqual(result["n_errors"], 0)
        bucket_total = sum(b["n_trades"] for b in result["by_regime"].values())
        self.assertEqual(bucket_total, result["total_trades"])
        for regime_name in result["by_regime"]:
            self.assertIn(regime_name, regime_probe.REGIME_LEVELS)

    def test_one_bad_ticker_does_not_kill_the_whole_group(self):
        def fetch_with_one_failure(symbol, days):
            if symbol == "BROKEN":
                raise RuntimeError("yahoo fetch failed")
            return self._fake_fetch(symbol, days)

        with patch("backtest_v2.fetch_bars", side_effect=fetch_with_one_failure):
            result = regime_probe.run_group_regime(["BROKEN", "GOODTICK"], years=4)

        self.assertEqual(result["n_tickers"], 2)
        self.assertEqual(result["n_errors"], 1)
        self.assertEqual(result["errors"][0]["ticker"], "BROKEN")

    def test_spy_and_vxx_fetched_once_shared_across_tickers(self):
        calls = []

        def counting_fetch(symbol, days):
            calls.append(symbol)
            return self._fake_fetch(symbol, days)

        self.ticker_bars["GOODTICK2"] = _synthetic_bars(self.n, volume=300_000, daily=0.0009)
        with patch("backtest_v2.fetch_bars", side_effect=counting_fetch):
            regime_probe.run_group_regime(["GOODTICK", "GOODTICK2"], years=4)

        self.assertEqual(calls.count("SPY"), 1)
        self.assertEqual(calls.count("VXX"), 1)
        self.assertEqual(calls.count("GOODTICK"), 1)
        self.assertEqual(calls.count("GOODTICK2"), 1)


if __name__ == "__main__":
    unittest.main()
