"""Unit tests for scripts/hurst_exponent_cross_sectional_probe.py — all
synthetic, no network, per this repo's established convention for
foreign-field/cross-sectional probe test files (test_hurst_exponent_probe.py,
test_illiquid_universe_probe.py, etc.). Does not re-test hurst_exponent_probe's
own math (rolling_hurst, continuation_scores, spearman, tertile_welch already
covered by test_hurst_exponent_probe.py) — only this file's OWN new logic:
the pinned ticker-list reuse, pooling, and the leave-one-out robustness check.
"""

import importlib.util
import os
import unittest
from unittest import mock

_ROOT = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location(
    "hurst_exponent_cross_sectional_probe",
    os.path.join(_ROOT, "scripts", "hurst_exponent_cross_sectional_probe.py"))
xsp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xsp)


class TestPinnedTickerLists(unittest.TestCase):
    """Regression-pins the three groups against the EXACT lists documented
    in scripts/illiquid_universe_probe.py's own module docstring (source
    of truth) — guards against either file silently drifting so the study
    population this script reports on no longer matches what was screened
    and pinned there 2026-07-24."""

    def test_illiquid_matches_documented_source(self):
        self.assertEqual(
            xsp.ILLIQUID,
            ["AXG", "CISO", "DYAI", "EPOW", "GALT", "NRXP", "PROF", "SNOA", "TRAW", "WNW"])

    def test_moderate_matches_documented_source(self):
        self.assertEqual(
            xsp.MODERATE,
            ["CRDF", "IMNM", "KTTA", "ONCY", "SXTP", "VIVO", "ZCMD"])

    def test_liquid_matches_documented_source(self):
        self.assertEqual(
            xsp.LIQUID,
            ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "CAT", "GE"])

    def test_lists_not_re_derived_from_a_different_source(self):
        # This script must import (not retype) illiquid_universe_probe's
        # lists — assert the actual imported module's lists are identical
        # to what this script exposes, so a hand-edited divergence here
        # would fail even if both copies happened to look plausible.
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "illiquid_universe_probe",
            os.path.join(_ROOT, "scripts", "illiquid_universe_probe.py"))
        iup = ilu.module_from_spec(spec)
        spec.loader.exec_module(iup)
        self.assertEqual(xsp.ILLIQUID, list(iup.ILLIQUID))
        self.assertEqual(xsp.MODERATE, list(iup.MODERATE))
        self.assertEqual(xsp.LIQUID, list(iup.LIQUID))

    def test_no_overlap_between_groups(self):
        sets = [set(xsp.ILLIQUID), set(xsp.MODERATE), set(xsp.LIQUID)]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertEqual(sets[i] & sets[j], set())


class TestDestride(unittest.TestCase):
    def test_every_stride_th_pair(self):
        pairs = [(float(i), float(i)) for i in range(20)]
        out = xsp.destride(pairs, horizon=4)
        self.assertEqual(out, [(0.0, 0.0), (4.0, 4.0), (8.0, 8.0), (12.0, 12.0), (16.0, 16.0)])

    def test_zero_horizon_is_empty(self):
        self.assertEqual(xsp.destride([(1.0, 1.0)] * 5, horizon=0), [])


class TestPoolGroup(unittest.TestCase):
    def test_concatenates_in_ticker_order(self):
        rows = [
            {"ticker": "A", "destrided_pairs": [(0.5, 1.0), (0.6, 2.0)]},
            {"ticker": "B", "destrided_pairs": [(0.7, 3.0)]},
            {"ticker": "C", "destrided_pairs": []},
        ]
        pooled = xsp.pool_group(rows)
        self.assertEqual(pooled, [(0.5, 1.0), (0.6, 2.0), (0.7, 3.0)])

    def test_missing_key_treated_as_empty(self):
        rows = [{"ticker": "A"}, {"ticker": "B", "destrided_pairs": [(0.1, 0.1)]}]
        pooled = xsp.pool_group(rows)
        self.assertEqual(pooled, [(0.1, 0.1)])

    def test_empty_rows_is_empty_pool(self):
        self.assertEqual(xsp.pool_group([]), [])


class TestPassesGate2(unittest.TestCase):
    def test_none_stat_fails(self):
        self.assertFalse(xsp._passes_gate2(None))

    def test_below_rho_floor_fails(self):
        self.assertFalse(xsp._passes_gate2({"rho": 0.10, "p_value": 0.001}))

    def test_above_p_ceiling_fails(self):
        self.assertFalse(xsp._passes_gate2({"rho": 0.5, "p_value": 0.10}))

    def test_clears_both_bars(self):
        self.assertTrue(xsp._passes_gate2({"rho": 0.35, "p_value": 0.01}))

    def test_negative_rho_uses_absolute_value(self):
        self.assertTrue(xsp._passes_gate2({"rho": -0.40, "p_value": 0.001}))


class TestLeaveOneOut(unittest.TestCase):
    """Synthetic construction: several tickers contribute noise pairs with
    ~zero rank correlation, and ONE ticker contributes a deliberately
    strong, clean monotonic relationship large enough to single-handedly
    carry the pooled Spearman across the GATE 2 bar. Excluding that one
    ticker should collapse the pooled effect back toward the noise
    tickers' level and should be correctly identified as the dominant
    contributor — the exact "does one name drive the result" check the
    task requires."""

    def _noise_row(self, ticker, seed):
        # Deterministic pseudo-noise via a simple LCG so this needs no
        # external RNG dependency and is 100% reproducible.
        pairs = []
        x = seed
        for i in range(30):
            x = (1103515245 * x + 12345) % (2 ** 31)
            h = 0.3 + (x % 1000) / 1000.0 * 0.4  # ~[0.3, 0.7)
            x = (1103515245 * x + 12345) % (2 ** 31)
            score = ((x % 2000) - 1000) / 1000.0  # ~[-1, 1), unrelated to h
            pairs.append((h, score))
        return {"ticker": ticker, "destrided_pairs": pairs}

    def _dominant_row(self, ticker, n=60):
        # Perfectly monotonic, large n relative to the noise tickers —
        # strong enough to carry a pooled Spearman across GATE 2 alone.
        return {"ticker": ticker,
                "destrided_pairs": [(0.3 + 0.01 * i, -1.0 + 0.05 * i) for i in range(n)]}

    def test_identifies_dominant_ticker_and_flips_conclusion(self):
        rows = [self._noise_row(f"NOISE{i}", seed=1000 + i) for i in range(4)]
        rows.append(self._dominant_row("DOMINANT"))

        result = xsp.leave_one_out(rows)

        self.assertEqual(result["largest_single_ticker_contribution"], "DOMINANT")
        self.assertTrue(result["full_pooled_gate2_pass"])
        # Find DOMINANT's own row in the per-exclusion breakdown and
        # confirm excluding it drops the pool below GATE 2.
        dominant_excl = next(r for r in result["per_exclusion"]
                              if r["excluded_ticker"] == "DOMINANT")
        self.assertFalse(dominant_excl["gate2_pass_without"])
        self.assertTrue(result["conclusion_flips_on_any_single_exclusion"])

    def test_skips_when_fewer_than_two_usable_tickers(self):
        result = xsp.leave_one_out([{"ticker": "ONLY", "destrided_pairs": [(0.5, 1.0)] * 20}])
        self.assertIn("note", result)
        self.assertEqual(result["per_exclusion"], [])

    def test_no_flip_when_all_tickers_agree(self):
        # Four independently-noisy tickers, none individually dominant —
        # excluding any one should not flip a (failing) pooled verdict.
        rows = [self._noise_row(f"NOISE{i}", seed=2000 + i) for i in range(4)]
        result = xsp.leave_one_out(rows)
        self.assertFalse(result["full_pooled_gate2_pass"])
        self.assertFalse(result["conclusion_flips_on_any_single_exclusion"])


class TestTickerPairsErrorHandling(unittest.TestCase):
    """Verifies ticker_pairs() reports failures instead of raising — a
    single obscure illiquid ticker failing to fetch must not crash the
    whole cross-sectional run. Mocks backtest_v2.fetch_bars; no network."""

    def test_fetch_exception_is_reported_not_raised(self):
        with mock.patch.object(xsp.backtest_v2, "fetch_bars", side_effect=RuntimeError("boom")):
            row = xsp.ticker_pairs("FAKE", days=2520)
        self.assertEqual(row["ticker"], "FAKE")
        self.assertEqual(row["pairs"], [])
        self.assertIn("fetch failed", row["error"])

    def test_insufficient_bars_is_reported_not_raised(self):
        with mock.patch.object(xsp.backtest_v2, "fetch_bars",
                                return_value={"close": [100.0] * 50, "date": ["d"] * 50}):
            row = xsp.ticker_pairs("THIN", days=2520)
        self.assertEqual(row["n_bars"], 50)
        self.assertEqual(row["pairs"], [])
        self.assertIn("insufficient bars", row["error"])


if __name__ == "__main__":
    unittest.main()
