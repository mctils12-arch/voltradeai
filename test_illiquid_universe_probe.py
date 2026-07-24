"""
Regression tests for scripts/illiquid_universe_probe.py. Pure-function
tests only: no network calls, no dependency on live NASDAQ symbol
directory or backtest_v2's Alpaca/Yahoo fetch (matches the existing
test_gem_methane_gate2c.py / test_cot_gate2.py convention for research
probe scripts).
"""
import importlib.util
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "illiquid_universe_probe",
    os.path.join(os.path.dirname(__file__), "scripts", "illiquid_universe_probe.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class TestBuyAndHoldPct(unittest.TestCase):
    def test_positive_return(self):
        bars = {"close": [10.0, 12.0, 15.0]}
        self.assertEqual(probe.buy_and_hold_pct(bars), 50.0)

    def test_total_wipeout(self):
        bars = {"close": [10.0, 5.0, 0.01]}
        self.assertEqual(probe.buy_and_hold_pct(bars), -99.9)

    def test_flat(self):
        bars = {"close": [10.0, 10.0]}
        self.assertEqual(probe.buy_and_hold_pct(bars), 0.0)


class TestClassifyLiquidityTier(unittest.TestCase):
    """Pins classify_liquidity_tier's boundaries against backtest_v2's
    OWN liquidity_cost_pct() tiers (>5M / 1-5M / <=1M shares/day) —
    a drift in either place without updating the other would silently
    mislabel which tier this probe's tickers actually price at."""

    def test_matches_backtest_v2_tier_boundaries(self):
        from backtest_v2 import _LIQUIDITY_TIERS, _ILLIQUID_COST_PCT
        # First tier boundary in backtest_v2 must be 5,000,000 (this
        # probe only distinguishes >5M / 1-5M / <=1M, a coarser cut of
        # backtest_v2's 4 tiers — assert the shared boundary values.
        boundaries = [b for b, _ in _LIQUIDITY_TIERS]
        self.assertIn(5_000_000, boundaries)
        self.assertIn(1_000_000, boundaries)
        self.assertIsInstance(_ILLIQUID_COST_PCT, float)

    def test_illiquid_at_or_below_1m(self):
        self.assertEqual(probe.classify_liquidity_tier(1_000_000), "illiquid")
        self.assertEqual(probe.classify_liquidity_tier(4_655), "illiquid")

    def test_moderate_between_1m_and_5m(self):
        self.assertEqual(probe.classify_liquidity_tier(1_000_001), "moderate")
        self.assertEqual(probe.classify_liquidity_tier(5_000_000), "moderate")

    def test_liquid_above_5m(self):
        self.assertEqual(probe.classify_liquidity_tier(5_000_001), "liquid_5m_plus")


class TestSummarize(unittest.TestCase):
    def test_aggregates_sharpe_and_pos_frac(self):
        group = [
            {"ticker": "A", "buy_and_hold_pct": -50.0,
             "momentum": {"sharpe": -0.5, "total_return_pct": -5.0, "n_trades": 3},
             "mean_reversion": {"sharpe": 0.5, "total_return_pct": 5.0, "n_trades": 4}},
            {"ticker": "B", "buy_and_hold_pct": -90.0,
             "momentum": {"sharpe": 0.1, "total_return_pct": 1.0, "n_trades": 2},
             "mean_reversion": {"sharpe": 0.3, "total_return_pct": 3.0, "n_trades": 6}},
        ]
        s = probe.summarize(group)
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["buy_and_hold_mean_pct"], -70.0)
        self.assertEqual(s["momentum"]["pos_frac"], "1/2")
        self.assertEqual(s["mean_reversion"]["pos_frac"], "2/2")
        self.assertEqual(s["mean_reversion"]["total_trades"], 10)

    def test_skips_tickers_with_fetch_errors(self):
        group = [
            {"ticker": "BROKEN", "error": "yahoo fetch failed"},
            {"ticker": "OK", "buy_and_hold_pct": 10.0,
             "momentum": {"sharpe": 1.0, "total_return_pct": 10.0, "n_trades": 1}},
        ]
        s = probe.summarize(group)
        self.assertEqual(s["n"], 2)  # error rows still counted in n...
        self.assertEqual(s["buy_and_hold_mean_pct"], 10.0)  # ...but excluded from means
        self.assertEqual(s["momentum"]["pos_frac"], "1/1")

    def test_empty_group_is_null_not_a_crash(self):
        s = probe.summarize([])
        self.assertEqual(s["n"], 0)
        self.assertIsNone(s["buy_and_hold_mean_pct"])
        self.assertIsNone(s["momentum"])


class TestFrozenCandidateLists(unittest.TestCase):
    """The ILLIQUID/MODERATE/LIQUID lists are pinned for reproducibility
    (module docstring LIMITATION #2) — this just guards against an
    accidental edit silently changing the study population."""

    def test_no_overlap_between_groups(self):
        sets = [set(probe.ILLIQUID), set(probe.MODERATE), set(probe.LIQUID)]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertEqual(sets[i] & sets[j], set())

    def test_liquid_group_is_subset_of_t1_fallback_minus_etfs(self):
        from tiered_strategy import T1_TICKERS_FALLBACK
        etfs = {"SPY", "QQQ", "IWM"}
        allowed = set(T1_TICKERS_FALLBACK) - etfs
        self.assertTrue(set(probe.LIQUID).issubset(allowed))
        self.assertEqual(len(probe.LIQUID), 7)


if __name__ == "__main__":
    unittest.main()
