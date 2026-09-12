"""Unit tests for scripts/hurst_exponent_probe.py — all synthetic, no network,
per this repo's established convention for foreign-field probe test files
(test_permutation_entropy_probe.py, test_hazard_rate_probe.py, etc.)."""

import importlib.util
import math
import os
import unittest

spec = importlib.util.spec_from_file_location(
    "hurst_exponent_probe",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "hurst_exponent_probe.py"))
hep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hep)


class TestLogReturns(unittest.TestCase):
    def test_basic(self):
        closes = [100.0, 110.0, 99.0]
        rets = hep.log_returns(closes)
        self.assertEqual(len(rets), 2)
        self.assertAlmostEqual(rets[0], math.log(1.1), places=6)
        self.assertAlmostEqual(rets[1], math.log(99.0 / 110.0), places=6)

    def test_nonpositive_close_yields_none(self):
        rets = hep.log_returns([100.0, 0.0, 50.0])
        self.assertIsNone(rets[0])
        self.assertIsNone(rets[1])


class TestRsStat(unittest.TestCase):
    def test_too_short(self):
        self.assertIsNone(hep._rs_stat([1.0]))

    def test_zero_variance_is_none(self):
        self.assertIsNone(hep._rs_stat([0.01, 0.01, 0.01, 0.01]))

    def test_positive_for_varying_chunk(self):
        rs = hep._rs_stat([0.01, -0.02, 0.03, -0.01, 0.02])
        self.assertIsNotNone(rs)
        self.assertGreater(rs, 0)


class TestHurstRs(unittest.TestCase):
    def test_none_when_too_short(self):
        self.assertIsNone(hep.hurst_rs([0.01] * 5, min_chunk=8))

    def test_trending_series_has_high_hurst(self):
        # A slowly-varying, smooth series (values near each other change
        # direction rarely relative to the chunk sizes tested): locally
        # persistent, i.e. positively autocorrelated increments -> should
        # estimate H well above 0.5. (NOTE: a constant per-step drift is
        # NOT a valid fixture here — R/S mean-adjusts each chunk first, so
        # a pure constant offset is removed entirely and only the residual
        # noise pattern is measured; this caught a flawed first draft of
        # this test, not an algorithm bug.)
        rets = [0.01 * math.sin(2 * math.pi * i / 50) for i in range(300)]
        h = hep.hurst_rs(rets, min_chunk=8)
        self.assertIsNotNone(h)
        self.assertGreater(h, 0.5)

    def test_alternating_series_has_low_hurst(self):
        # Strict +x/-x alternation: every increment reverses the last
        # one (maximally anti-persistent) -> should estimate H well
        # below 0.5.
        rets = [0.02 * ((-1) ** i) for i in range(200)]
        h = hep.hurst_rs(rets, min_chunk=8)
        self.assertIsNotNone(h)
        self.assertLess(h, 0.5)

    def test_none_propagates_through_gaps(self):
        rets = [0.01 if i % 5 != 0 else None for i in range(200)]
        h = hep.hurst_rs(rets, min_chunk=8)
        # Should still compute (None values filtered), not raise.
        self.assertTrue(h is None or isinstance(h, float))


class TestRollingHurst(unittest.TestCase):
    def test_none_before_window(self):
        rets = [0.01] * 300
        out = hep.rolling_hurst(rets, window=252, min_chunk=8)
        self.assertTrue(all(v is None for v in out[:251]))
        self.assertIsNotNone(out[251])

    def test_no_lookahead(self):
        # Two series identical up to index 300, diverging only after.
        # H at index 260 (computed from a window entirely before 300)
        # must be identical in both, since rolling_hurst must never
        # look at future data.
        base = [0.001 * ((-1) ** i) for i in range(400)]
        a = list(base)
        b = list(base)
        for i in range(300, 400):
            b[i] = 0.05  # blow up the tail of b only

        out_a = hep.rolling_hurst(a, window=252, min_chunk=8)
        out_b = hep.rolling_hurst(b, window=252, min_chunk=8)
        self.assertEqual(out_a[260], out_b[260])
        self.assertEqual(out_a[299], out_b[299])


class TestContinuationScores(unittest.TestCase):
    def test_basic_shape_and_sign(self):
        # Flat trailing uptrend, then a clean forward continuation.
        rets = [0.01] * 20 + [0.02] * 20
        hurst = [None] * 19 + [0.7] + [None] * 20
        pairs = hep.continuation_scores(rets, hurst, lookback=19, horizon=15)
        self.assertEqual(len(pairs), 1)
        h, score = pairs[0]
        self.assertEqual(h, 0.7)
        # trailing 19 days of +0.01 -> trend=+1; forward 15 days of
        # +0.02 -> continuation_score should be strongly positive.
        self.assertGreater(score, 0)

    def test_zero_trailing_trend_is_skipped(self):
        rets = [0.01, -0.01] * 10 + [0.05] * 10
        hurst = [None] * 19 + [0.6] + [None] * 9
        pairs = hep.continuation_scores(rets, hurst, lookback=20, horizon=5)
        self.assertEqual(len(pairs), 0)

    def test_none_hurst_excluded(self):
        rets = [0.01] * 50
        hurst = [None] * 50
        pairs = hep.continuation_scores(rets, hurst, lookback=10, horizon=10)
        self.assertEqual(pairs, [])


class TestSpearman(unittest.TestCase):
    def test_none_below_floor(self):
        self.assertIsNone(hep.spearman([(0.5, 1.0), (0.6, 2.0)]))

    def test_perfect_monotonic(self):
        pairs = [(float(i), float(i) * 2) for i in range(10)]
        result = hep.spearman(pairs)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["rho"], 1.0, places=3)

    def test_inverse_monotonic(self):
        pairs = [(float(i), -float(i)) for i in range(10)]
        result = hep.spearman(pairs)
        self.assertAlmostEqual(result["rho"], -1.0, places=3)


class TestDestridedSpearman(unittest.TestCase):
    def test_selects_every_stride_th_pair(self):
        pairs = [(float(i), float(i) * 2) for i in range(20)]
        result = hep.destrided_spearman(pairs, stride=4)
        self.assertEqual(result["n"], 5)  # indices 0,4,8,12,16

    def test_none_for_zero_stride(self):
        self.assertIsNone(hep.destrided_spearman([(1.0, 1.0)] * 10, stride=0))

    def test_falls_below_floor_with_large_stride(self):
        pairs = [(float(i), float(i)) for i in range(10)]
        self.assertIsNone(hep.destrided_spearman(pairs, stride=5))  # only 2 pairs survive


class TestTertileWelch(unittest.TestCase):
    def test_none_below_floor(self):
        self.assertIsNone(hep.tertile_welch([(0.5, 1.0)] * 5))

    def test_structure_when_enough_data(self):
        pairs = [(0.3 + 0.001 * i, -1.0 + 0.01 * (i % 3)) for i in range(20)] + \
                [(0.7 + 0.001 * i, 1.0 + 0.01 * (i % 3)) for i in range(20)]
        result = hep.tertile_welch(pairs)
        self.assertIsNotNone(result)
        self.assertIn("t_stat", result)
        self.assertIn("p_value", result)
        self.assertGreater(result["high_h_mean_continuation"],
                            result["low_h_mean_continuation"])


if __name__ == "__main__":
    unittest.main()
