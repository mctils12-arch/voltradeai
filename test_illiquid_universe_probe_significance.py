"""
Regression tests for scripts/illiquid_universe_probe_significance.py
(LADDER PATH step 3, the 2026-07-24 illiquid mean_reversion edge entry).
Pure-function tests only: no network calls, no dependency on live
backtest_v2/NASDAQ data — matches the existing
test_illiquid_universe_probe.py / test_illiquid_universe_probe_traintest.py
convention for this research-probe family. The bootstrap/permutation
machinery is verified against SYNTHETIC data with a known right answer
(clearly separated groups -> significant; identical distributions ->
not significant) before it is trusted on the one real comparison that
matters (CLAUDE.md READ BEFORE WRITE).
"""
import importlib.util
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "illiquid_universe_probe_significance",
    os.path.join(os.path.dirname(__file__), "scripts",
                 "illiquid_universe_probe_significance.py"))
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)


class TestBootstrapCiMeanDiff(unittest.TestCase):
    def test_clearly_separated_groups_excludes_zero(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.1, 0.9, 1.0, 1.05]
        b = [-1.0, -1.1, -0.9, -1.05, -0.95, -1.0, -1.1]
        result = sig.bootstrap_ci_mean_diff(a, b, n_boot=2000, seed=1)
        self.assertGreater(result["ci_low"], 0)
        self.assertTrue(result["excludes_zero"])
        self.assertAlmostEqual(result["observed_diff"],
                                sum(a) / len(a) - sum(b) / len(b), places=3)

    def test_identical_distribution_includes_zero(self):
        rng_vals = [0.1, -0.2, 0.05, 0.3, -0.1, 0.15, -0.05, 0.2, -0.15, 0.0]
        a = rng_vals[:5]
        b = rng_vals[5:] + [0.02]  # b has 6 members drawn from the same pool
        result = sig.bootstrap_ci_mean_diff(a, b, n_boot=2000, seed=2)
        self.assertFalse(result["excludes_zero"])
        self.assertLessEqual(result["ci_low"], 0)
        self.assertGreaterEqual(result["ci_high"], 0)

    def test_deterministic_with_same_seed(self):
        a = [1.0, 2.0, 1.5, 1.8, 2.2]
        b = [0.1, 0.4, 0.2, 0.3]
        r1 = sig.bootstrap_ci_mean_diff(a, b, n_boot=500, seed=42)
        r2 = sig.bootstrap_ci_mean_diff(a, b, n_boot=500, seed=42)
        self.assertEqual(r1, r2)

    def test_raises_on_single_observation_group(self):
        with self.assertRaises(ValueError):
            sig.bootstrap_ci_mean_diff([1.0], [1.0, 2.0, 3.0])

    def test_unequal_group_sizes_allowed(self):
        a = [1.0, 1.2, 0.8, 1.1, 1.3, 0.9, 1.0, 1.15, 0.95, 1.05]  # n=10
        b = [-0.5, -0.6, -0.4, -0.55, -0.45, -0.5, -0.6]  # n=7
        result = sig.bootstrap_ci_mean_diff(a, b, n_boot=1000, seed=3)
        self.assertEqual(result["n_a"], 10)
        self.assertEqual(result["n_b"], 7)


class TestPermutationTestMeanDiff(unittest.TestCase):
    def test_clearly_separated_groups_significant(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.1, 0.9, 1.0, 1.05]
        b = [-1.0, -1.1, -0.9, -1.05, -0.95, -1.0, -1.1]
        result = sig.permutation_test_mean_diff(a, b, n_perm=3000, seed=1)
        self.assertLess(result["p_value_two_sided"], 0.05)

    def test_identical_distribution_not_significant(self):
        a = [0.1, -0.2, 0.05, 0.3, -0.1]
        b = [0.15, -0.05, 0.2, -0.15, 0.0, 0.02]
        result = sig.permutation_test_mean_diff(a, b, n_perm=3000, seed=2)
        self.assertGreater(result["p_value_two_sided"], 0.05)

    def test_deterministic_with_same_seed(self):
        a = [1.0, 2.0, 1.5, 1.8, 2.2]
        b = [0.1, 0.4, 0.2, 0.3]
        r1 = sig.permutation_test_mean_diff(a, b, n_perm=500, seed=42)
        r2 = sig.permutation_test_mean_diff(a, b, n_perm=500, seed=42)
        self.assertEqual(r1, r2)

    def test_raises_on_single_observation_group(self):
        with self.assertRaises(ValueError):
            sig.permutation_test_mean_diff([1.0], [1.0, 2.0, 3.0])

    def test_permutation_does_not_mutate_input_lists(self):
        a = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.1, 0.9, 1.0, 1.05]
        b = [-1.0, -1.1, -0.9, -1.05, -0.95, -1.0, -1.1]
        a_copy, b_copy = list(a), list(b)
        sig.permutation_test_mean_diff(a, b, n_perm=200, seed=5)
        self.assertEqual(a, a_copy)
        self.assertEqual(b, b_copy)


class TestMeanReversionSharpesExtraction(unittest.TestCase):
    def test_skips_error_rows_and_missing_strategy_rows(self):
        rows = [
            {"ticker": "A", "mean_reversion": {"sharpe": 0.5}},
            {"ticker": "B", "error": "yahoo fetch failed"},
            {"ticker": "C", "momentum": {"sharpe": 0.3}},  # no mean_reversion key
            {"ticker": "D", "mean_reversion": {"sharpe": -0.2}},
        ]
        self.assertEqual(sig._mean_reversion_sharpes(rows), [0.5, -0.2])


class TestRunSignificance(unittest.TestCase):
    def _rows(self, sharpes):
        return [{"ticker": f"T{i}", "mean_reversion": {"sharpe": s}}
                for i, s in enumerate(sharpes)]

    def test_full_three_group_comparison_shape(self):
        group_rows = {
            "illiquid": self._rows([0.3, 0.4, 0.2, 0.35, 0.25, 0.3, 0.4, 0.2, 0.35, 0.3]),
            "moderate": self._rows([-0.2, -0.3, -0.1, -0.25, -0.15, -0.2, -0.3]),
            "liquid": self._rows([0.1, 0.2, 0.05, 0.15, 0.1, 0.2, 0.05]),
        }
        out = sig.run_significance(group_rows, n_boot=500, n_perm=500, seed=7)
        self.assertEqual(out["n_mean_reversion_tickers"],
                          {"illiquid": 10, "moderate": 7, "liquid": 7})
        self.assertIn("illiquid_vs_moderate", out["comparisons"])
        self.assertIn("illiquid_vs_liquid", out["comparisons"])
        for key in ("illiquid_vs_moderate", "illiquid_vs_liquid"):
            comp = out["comparisons"][key]
            self.assertIn("bootstrap_ci", comp)
            self.assertIn("permutation_test", comp)

    def test_gracefully_skips_comparison_with_insufficient_tickers(self):
        group_rows = {
            "illiquid": self._rows([0.3, 0.4]),
            "moderate": self._rows([-0.2]),  # only 1 -> can't bootstrap/permute
            "liquid": self._rows([0.1, 0.2]),
        }
        out = sig.run_significance(group_rows, n_boot=200, n_perm=200, seed=1)
        self.assertTrue(out["comparisons"]["illiquid_vs_moderate"]["skipped"])
        self.assertFalse(out["comparisons"]["illiquid_vs_liquid"].get("skipped", False))

    def test_error_rows_excluded_from_ticker_counts(self):
        group_rows = {
            "illiquid": self._rows([0.3, 0.4, 0.2]) + [{"ticker": "BAD", "error": "boom"}],
            "moderate": self._rows([-0.2, -0.3, -0.1]),
            "liquid": self._rows([0.1, 0.2, 0.05]),
        }
        out = sig.run_significance(group_rows, n_boot=200, n_perm=200, seed=1)
        self.assertEqual(out["n_mean_reversion_tickers"]["illiquid"], 3)


if __name__ == "__main__":
    unittest.main()
