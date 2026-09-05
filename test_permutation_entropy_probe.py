"""
Regression tests for scripts/permutation_entropy_probe.py. Pure-function
tests only, against synthetic/hand-computed series: no network calls, no
dependency on backtest_v2's Alpaca/Yahoo fetch (matches the existing
test_critical_slowing_down_probe.py / test_hazard_rate_probe.py convention
for research probe scripts in this repo).
"""
import importlib.util
import math
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "permutation_entropy_probe",
    os.path.join(os.path.dirname(__file__), "scripts", "permutation_entropy_probe.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class TestLogReturns(unittest.TestCase):
    def test_basic(self):
        out = probe.log_returns([100.0, 110.0, 99.0])
        self.assertAlmostEqual(out[0], math.log(1.1))
        self.assertAlmostEqual(out[1], math.log(99.0 / 110.0))

    def test_non_positive_price_yields_zero_not_crash(self):
        self.assertEqual(probe.log_returns([100.0, 0.0, 50.0]), [0.0, 0.0])

    def test_none_price_yields_zero(self):
        self.assertEqual(probe.log_returns([100.0, None, 50.0]), [0.0, 0.0])


class TestOrdinalPattern(unittest.TestCase):
    def test_strictly_increasing(self):
        self.assertEqual(probe.ordinal_pattern([1, 2, 3]), (0, 1, 2))

    def test_strictly_decreasing(self):
        self.assertEqual(probe.ordinal_pattern([3, 2, 1]), (2, 1, 0))

    def test_mixed(self):
        # values [2, 3, 1] -> ranks: 1 is smallest (rank0, index2),
        # 2 is middle (rank1, index0), 3 is largest (rank2, index1)
        self.assertEqual(probe.ordinal_pattern([2, 3, 1]), (1, 2, 0))

    def test_ties_broken_by_position_not_arbitrary(self):
        # all equal -> stable sort keeps original order -> identity pattern
        self.assertEqual(probe.ordinal_pattern([5, 5, 5]), (0, 1, 2))

    def test_partial_tie(self):
        # [1, 1, 2]: index0 and index1 tie at the lowest value; stable
        # sort keeps index0 before index1 -> ranks (0, 1, 2)
        self.assertEqual(probe.ordinal_pattern([1, 1, 2]), (0, 1, 2))


class TestPermutationEntropy(unittest.TestCase):
    def test_monotonic_series_is_fully_ordered_zero_entropy(self):
        # every length-3 window of a strictly increasing series is the
        # SAME ordinal pattern (0,1,2) -> single-outcome distribution -> H=0
        self.assertEqual(probe.permutation_entropy(list(range(30)), m=3), 0.0)

    def test_perfectly_alternating_binary_pattern_hits_max_entropy_exactly(self):
        # m=2: values alternate up/down every step with an EVEN number of
        # extracted windows (21 values -> 20 windows), so the two possible
        # ordinal patterns ((0,1) and (1,0)) occur exactly 10 times each
        # -> maximal, exactly-normalized entropy of 1.0 (hand-verified).
        seq = ([1, 2] * 10) + [1]
        self.assertEqual(len(seq), 21)
        self.assertEqual(probe.permutation_entropy(seq, m=2), 1.0)

    def test_returns_none_below_the_minimum_sample_floor(self):
        # m=3 needs at least 3!+1=7 extractable windows; 2 values can't
        # even form one length-3 window.
        self.assertIsNone(probe.permutation_entropy([1, 2], m=3))

    def test_returns_none_just_below_the_pattern_count_floor(self):
        # m=3 -> 6 possible patterns, floor is n_windows >= 7. A 7-value
        # series yields exactly 5 length-3 windows (7-2) -> still below
        # floor -> None, not a number computed from too few samples.
        self.assertIsNone(probe.permutation_entropy([1, 2, 3, 4, 5, 6, 7], m=3))

    def test_value_in_unit_interval_for_pseudo_random_input(self):
        import random
        rng = random.Random(7)
        series = [rng.random() for _ in range(200)]
        h = probe.permutation_entropy(series, m=3)
        self.assertIsNotNone(h)
        self.assertGreaterEqual(h, 0.0)
        self.assertLessEqual(h, 1.0)
        # pseudo-random input should land close to (not exactly) 1.0
        self.assertGreater(h, 0.9)

    def test_invalid_m_raises(self):
        with self.assertRaises(ValueError):
            probe.permutation_entropy([1, 2, 3], m=1)

    def test_invalid_tau_raises(self):
        with self.assertRaises(ValueError):
            probe.permutation_entropy([1, 2, 3], m=2, tau=0)


class TestRollingPermutationEntropy(unittest.TestCase):
    def test_length_matches_input_and_leading_nones(self):
        out = probe.rolling_permutation_entropy(list(range(30)), window=10, m=3)
        self.assertEqual(len(out), 30)
        self.assertTrue(all(v is None for v in out[:9]))
        self.assertIsNotNone(out[9])

    def test_matches_direct_call_at_a_given_index(self):
        import random
        rng = random.Random(3)
        series = [rng.random() for _ in range(80)]
        out = probe.rolling_permutation_entropy(series, window=20, m=3)
        direct = probe.permutation_entropy(series[10:30], m=3)
        self.assertEqual(out[29], direct)

    def test_too_short_series_is_all_none(self):
        out = probe.rolling_permutation_entropy([1, 2, 3], window=10, m=3)
        self.assertEqual(out, [None, None, None])


class TestPearsonCorrelation(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 4.0, 6.0, 8.0]
        self.assertAlmostEqual(probe.pearson_correlation(a, b), 1.0)

    def test_perfect_negative_correlation(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [4.0, 3.0, 2.0, 1.0]
        self.assertAlmostEqual(probe.pearson_correlation(a, b), -1.0)

    def test_skips_none_pairs(self):
        a = [1.0, None, 3.0, 4.0, 5.0]
        b = [1.0, 99.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(probe.pearson_correlation(a, b), 1.0)

    def test_none_below_minimum_points(self):
        self.assertIsNone(probe.pearson_correlation([1.0, 2.0], [1.0, 2.0]))

    def test_none_on_degenerate_zero_variance(self):
        self.assertIsNone(probe.pearson_correlation([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]))


class TestWelchVsControl(unittest.TestCase):
    def test_none_below_five_per_side(self):
        self.assertIsNone(probe.welch_vs_control([1.0, 2.0, 3.0, 4.0], [1.0] * 10))
        self.assertIsNone(probe.welch_vs_control([1.0] * 10, [1.0, 2.0, 3.0, 4.0]))

    def test_identical_distributions_high_p_value(self):
        vals = [0.98, 0.99, 0.97, 0.985, 0.975, 0.982]
        result = probe.welch_vs_control(vals, list(vals))
        self.assertAlmostEqual(result["t_stat"], 0.0)
        self.assertAlmostEqual(result["mean_diff"], 0.0)
        self.assertGreater(result["p_value"], 0.9)

    def test_clearly_separated_distributions_low_p_value(self):
        # two tight, non-overlapping clusters -> Welch t-test must reject
        # the null at a stringent bar, not just "point the right direction"
        onset = [0.10, 0.11, 0.09, 0.105, 0.095, 0.102]
        control = [0.90, 0.91, 0.89, 0.905, 0.895, 0.902]
        result = probe.welch_vs_control(onset, control)
        self.assertLess(result["p_value"], 0.001)
        self.assertLess(result["mean_diff"], 0)
        self.assertLess(result["t_stat"], 0)

    def test_mean_diff_sign_matches_direction(self):
        onset = [0.5, 0.6, 0.55, 0.58, 0.52]
        control = [0.2, 0.25, 0.22, 0.24, 0.21]
        result = probe.welch_vs_control(onset, control)
        self.assertGreater(result["mean_diff"], 0)
        self.assertGreater(result["t_stat"], 0)


class TestComputeEntropyLeadSignal(unittest.TestCase):
    def _synthetic_returns_and_onsets(self, n=400, seed=11):
        import random
        rng = random.Random(seed)
        returns = [rng.gauss(0, 0.01) for _ in range(n)]
        # hand-build 6 qualifying onsets, well spaced, each preceded by a
        # long stable run (mirrors CSD's own test-construction style)
        labels = ["BULL"] * n
        onset_starts = [60, 120, 180, 240, 300, 360]
        for s in onset_starts:
            for k in range(4):
                if s + k < n:
                    labels[s + k] = "PANIC"
        onsets = [{"index": s, "from": "BULL", "to": "PANIC", "severity_jump": 4}
                  for s in onset_starts]
        return returns, onsets, labels

    def test_insufficient_n_reported_not_fabricated(self):
        returns, _, _ = self._synthetic_returns_and_onsets()
        result = probe.compute_entropy_lead_signal(returns, onsets=[{"index": 100, "from": "BULL"}],
                                                     window=60)
        self.assertTrue(result["insufficient_n"])
        self.assertNotIn("by_lead_days", result) if False else None
        self.assertEqual(result["by_lead_days"], {})

    def test_shape_with_sufficient_onsets(self):
        returns, onsets, labels = self._synthetic_returns_and_onsets()
        result = probe.compute_entropy_lead_signal(
            returns, onsets, window=60, lead_offsets=(20, 10, 5, 1),
            regime_at=labels)
        self.assertEqual(result["n_onsets"], 6)
        for lead in (20, 10, 5, 1):
            entry = result["by_lead_days"][lead]
            if not entry.get("insufficient_n"):
                self.assertIn("onset_mean_entropy", entry)
                self.assertIn("control_mean_entropy", entry)
                self.assertIn("control_regime_matched", entry)
                self.assertIn("welch", entry)
                self.assertIn("t_stat", entry["welch"])
                self.assertIn("p_value", entry["welch"])

    def test_regime_matched_control_falls_back_when_pool_too_small(self):
        returns, onsets, labels = self._synthetic_returns_and_onsets()
        # a regime label that appears almost nowhere -> match pool too
        # small -> must fall back to the unconditional pool and say so
        sparse_labels = ["NEUTRAL"] * len(labels)
        for o in onsets:
            sparse_labels[o["index"] - 1] = "BULL"
        result = probe.compute_entropy_lead_signal(
            returns, onsets, window=60, lead_offsets=(1,), regime_at=sparse_labels)
        entry = result["by_lead_days"][1]
        if not entry.get("insufficient_n"):
            self.assertFalse(entry["control_regime_matched"])

    def test_reproducible_given_fixed_seed(self):
        returns, onsets, labels = self._synthetic_returns_and_onsets()
        r1 = probe.compute_entropy_lead_signal(returns, onsets, window=60,
                                                lead_offsets=(5,), regime_at=labels)
        r2 = probe.compute_entropy_lead_signal(returns, onsets, window=60,
                                                lead_offsets=(5,), regime_at=labels)
        self.assertEqual(r1, r2)


class TestRunProbeIntegrationShape(unittest.TestCase):
    """run_probe() itself requires network/Alpaca access this sandbox does
    not have (see module docstring) — not exercised here, matching every
    prior probe script's own established precedent
    (TestRunProbeIntegrationShape in test_critical_slowing_down_probe.py /
    test_hazard_rate_probe.py). These tests only confirm the module's
    import-by-path wiring into critical_slowing_down_probe.py is correct,
    since run_probe's find_transition_onsets/rolling_ar1 reuse is
    otherwise untested by the pure-function tests above."""

    def test_csd_module_loads_and_exposes_find_transition_onsets_and_rolling_ar1(self):
        csd = probe._load_csd_module()
        self.assertTrue(callable(csd.find_transition_onsets))
        self.assertTrue(callable(csd.rolling_ar1))
        onsets = csd.find_transition_onsets(
            ["BULL"] * 25 + ["PANIC"] * 5 + ["BULL"] * 10)
        self.assertEqual(len(onsets), 1)
        self.assertEqual(onsets[0]["index"], 25)

    def test_run_probe_defaults(self):
        import inspect
        sig = inspect.signature(probe.run_probe)
        self.assertEqual(sig.parameters["ticker"].default, "SPY")
        self.assertEqual(sig.parameters["window"].default, 60)
        self.assertEqual(sig.parameters["m"].default, 3)


if __name__ == "__main__":
    unittest.main()
