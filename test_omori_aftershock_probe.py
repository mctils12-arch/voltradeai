"""
Regression tests for scripts/omori_aftershock_probe.py. Pure-function tests
only, against synthetic return/shock/curve data: no network calls, no
dependency on backtest_v2's Alpaca/Yahoo fetch (matches the existing
test_critical_slowing_down_probe.py / test_hazard_rate_probe.py convention
for research probe scripts).
"""
import importlib.util
import math
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "omori_aftershock_probe",
    os.path.join(os.path.dirname(__file__), "scripts", "omori_aftershock_probe.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class TestTrailingSigma(unittest.TestCase):
    def test_none_before_window_fills(self):
        out = probe.trailing_sigma([1, 2, 3, 4, 5, 6, 7], window=3)
        self.assertEqual(out[:3], [None, None, None])

    def test_arithmetic_progression_has_constant_sigma(self):
        # every length-3 window of an AR-1 progression has the same
        # population variance (2/3 here), so every post-warmup sigma
        # should be identical
        out = probe.trailing_sigma([1, 2, 3, 4, 5, 6, 7], window=3)
        expected = math.sqrt(2 / 3)
        for v in out[3:]:
            self.assertAlmostEqual(v, expected, places=9)

    def test_strictly_prior_excludes_current_value(self):
        # a huge outlier AT t must not appear in sigma[t]'s own window
        returns = [1, 1, 1, 1, 1, 1, 100]
        out = probe.trailing_sigma(returns, window=3)
        self.assertEqual(out[6], 0.0)  # window is returns[3:6] = [1,1,1]


class TestShockFlags(unittest.TestCase):
    def test_none_sigma_is_never_a_shock(self):
        out = probe.shock_flags([0.1], [None], k=1.0)
        self.assertEqual(out, [False])

    def test_zero_sigma_is_never_a_shock(self):
        # guards against the degenerate abs(return) >= k*0 trivial-true case
        out = probe.shock_flags([0.05, 0.0], [0.0, 0.0], k=1.0)
        self.assertEqual(out, [False, False])

    def test_above_threshold_is_a_shock(self):
        out = probe.shock_flags([0.1, -0.1, 0.05], [None, 0.02, 0.0], k=2.0)
        self.assertEqual(out, [False, True, False])

    def test_below_threshold_is_not_a_shock(self):
        out = probe.shock_flags([0.01], [0.02], k=1.0)
        self.assertEqual(out, [False])


class TestAftershockRateCurve(unittest.TestCase):
    def test_hand_computed_curve(self):
        shocks = [True, False, True, False, False, True, False, False, False, True]
        curve = probe.aftershock_rate_curve(shocks, max_lag=3)
        self.assertEqual(len(curve), 3)
        lag1, lag2, lag3 = curve
        self.assertEqual(lag1["lag"], 1)
        self.assertEqual(lag1["n_mainshocks"], 3)
        self.assertAlmostEqual(lag1["rate"], 0.0, places=9)
        self.assertTrue(lag1.get("insufficient_n"))
        self.assertAlmostEqual(lag2["rate"], 1 / 3, places=9)
        self.assertAlmostEqual(lag3["rate"], 1 / 3, places=9)

    def test_no_mainshocks_yields_no_usable_lags(self):
        curve = probe.aftershock_rate_curve([False] * 5, max_lag=2)
        for entry in curve:
            self.assertEqual(entry["n_mainshocks"], 0)
            self.assertIsNone(entry["rate"])
            self.assertTrue(entry.get("insufficient_n"))

    def test_sufficient_n_omits_insufficient_flag(self):
        # 6 mainshocks spread far enough apart that lag=1 stays fully usable
        shocks = [False] * 60
        for i in (0, 10, 20, 30, 40, 50):
            shocks[i] = True
        curve = probe.aftershock_rate_curve(shocks, max_lag=1)
        self.assertEqual(curve[0]["n_mainshocks"], 6)
        self.assertNotIn("insufficient_n", curve[0])
        self.assertIsInstance(curve[0]["rate"], float)


class TestBaselineShockRate(unittest.TestCase):
    def test_empty_is_none(self):
        self.assertIsNone(probe.baseline_shock_rate([]))

    def test_known_fraction(self):
        self.assertAlmostEqual(
            probe.baseline_shock_rate([True, True, False, False]), 0.5, places=9)


def _synthetic_power_law_curve(base_rate, K, c, p, n_lags, n_mainshocks=50):
    curve = []
    for lag in range(1, n_lags + 1):
        rate = base_rate + K / (lag + c) ** p
        curve.append({"lag": lag, "rate": rate, "n_mainshocks": n_mainshocks})
    return curve


def _synthetic_exponential_curve(base_rate, K, lam, n_lags, n_mainshocks=50):
    curve = []
    for lag in range(1, n_lags + 1):
        rate = base_rate + K * math.exp(-lam * lag)
        curve.append({"lag": lag, "rate": rate, "n_mainshocks": n_mainshocks})
    return curve


class TestFitPowerLawDecay(unittest.TestCase):
    def test_none_when_base_rate_is_none(self):
        curve = _synthetic_power_law_curve(0.1, 0.5, 1.0, 1.2, 15)
        self.assertIsNone(probe.fit_power_law_decay(curve, None))

    def test_none_below_three_positive_excess_points(self):
        curve = [{"lag": 1, "rate": 0.5, "n_mainshocks": 50},
                 {"lag": 2, "rate": 0.1, "n_mainshocks": 50},
                 {"lag": 3, "rate": 0.1, "n_mainshocks": 50}]
        # only lag 1 has positive excess over base_rate=0.1
        self.assertIsNone(probe.fit_power_law_decay(curve, 0.1))

    def test_exact_power_law_is_recovered(self):
        base_rate, K, c, p = 0.1, 0.5, 1.0, 1.2
        curve = _synthetic_power_law_curve(base_rate, K, c, p, n_lags=15)
        fit = probe.fit_power_law_decay(curve, base_rate)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["c"], c, places=9)
        self.assertAlmostEqual(fit["p"], p, places=6)
        self.assertAlmostEqual(fit["K"], K, places=6)
        self.assertAlmostEqual(fit["r_squared"], 1.0, places=6)


class TestFitExponentialDecay(unittest.TestCase):
    def test_none_when_base_rate_is_none(self):
        curve = _synthetic_exponential_curve(0.05, 0.4, 0.3, 15)
        self.assertIsNone(probe.fit_exponential_decay(curve, None))

    def test_none_below_three_positive_excess_points(self):
        curve = [{"lag": 1, "rate": 0.5, "n_mainshocks": 50},
                 {"lag": 2, "rate": 0.1, "n_mainshocks": 50}]
        self.assertIsNone(probe.fit_exponential_decay(curve, 0.1))

    def test_exact_exponential_is_recovered(self):
        base_rate, K, lam = 0.05, 0.4, 0.3
        curve = _synthetic_exponential_curve(base_rate, K, lam, n_lags=15)
        fit = probe.fit_exponential_decay(curve, base_rate)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["lam"], lam, places=6)
        self.assertAlmostEqual(fit["K"], K, places=6)
        self.assertAlmostEqual(fit["r_squared"], 1.0, places=6)


class TestPowerLawBeatsExponentialDiscrimination(unittest.TestCase):
    """The actual hypothesis test: an exact power-law-generated curve should
    be correctly identified as power-law-favored, and vice versa for an
    exact exponential-generated curve — the discriminating test this
    probe's whole hypothesis depends on."""

    def test_power_law_data_favors_power_law_fit(self):
        curve = _synthetic_power_law_curve(0.1, 0.5, 1.0, 1.2, n_lags=15)
        pl = probe.fit_power_law_decay(curve, 0.1)
        exp = probe.fit_exponential_decay(curve, 0.1)
        self.assertGreater(pl["r_squared"], exp["r_squared"])
        self.assertTrue(probe.power_law_beats_exponential(pl, exp))

    def test_exponential_data_favors_exponential_fit(self):
        curve = _synthetic_exponential_curve(0.05, 0.4, 0.3, n_lags=15)
        pl = probe.fit_power_law_decay(curve, 0.05)
        exp = probe.fit_exponential_decay(curve, 0.05)
        self.assertGreater(exp["r_squared"], pl["r_squared"])
        self.assertFalse(probe.power_law_beats_exponential(pl, exp))


class TestPowerLawBeatsExponential(unittest.TestCase):
    def test_none_when_either_fit_missing(self):
        fit = {"r_squared": 0.9}
        self.assertIsNone(probe.power_law_beats_exponential(None, fit))
        self.assertIsNone(probe.power_law_beats_exponential(fit, None))
        self.assertIsNone(probe.power_law_beats_exponential(None, None))

    def test_higher_r_squared_wins(self):
        better = {"r_squared": 0.9}
        worse = {"r_squared": 0.5}
        self.assertTrue(probe.power_law_beats_exponential(better, worse))
        self.assertFalse(probe.power_law_beats_exponential(worse, better))


class TestCsdLogReturnsIntegrationShape(unittest.TestCase):
    """Confirms omori_aftershock_probe.run_probe()'s CSD-module import
    wiring (the one piece the pure-function tests above don't otherwise
    exercise) actually loads a working log_returns, the same way
    test_hazard_rate_probe.py verifies its own find_transition_onsets
    import."""

    def test_log_returns_importable_and_correct(self):
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "critical_slowing_down_probe",
            os.path.join(os.path.dirname(__file__), "scripts",
                         "critical_slowing_down_probe.py"))
        csd = ilu.module_from_spec(spec)
        spec.loader.exec_module(csd)
        out = csd.log_returns([100, 110, 121])
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0], math.log(1.1), places=9)
        self.assertAlmostEqual(out[1], math.log(1.1), places=9)


if __name__ == "__main__":
    unittest.main()
