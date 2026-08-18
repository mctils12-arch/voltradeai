"""
Regression tests for scripts/critical_slowing_down_probe.py. Pure-function
tests only, against synthetic series: no network calls, no dependency on
backtest_v2's Alpaca/Yahoo fetch (matches the existing
test_illiquid_universe_probe.py convention for research probe scripts).
"""
import importlib.util
import math
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "critical_slowing_down_probe",
    os.path.join(os.path.dirname(__file__), "scripts",
                 "critical_slowing_down_probe.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class TestLogReturns(unittest.TestCase):
    def test_basic(self):
        closes = [100.0, 110.0, 100.0]
        out = probe.log_returns(closes)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0], math.log(1.10), places=9)
        self.assertAlmostEqual(out[1], math.log(100.0 / 110.0), places=9)

    def test_non_positive_close_degrades_to_zero_not_crash(self):
        closes = [100.0, 0.0, 50.0]
        out = probe.log_returns(closes)
        self.assertEqual(out[0], 0.0)
        # second return uses prev=0.0 -> also degrades, never raises
        self.assertEqual(out[1], 0.0)


class TestRollingAr1(unittest.TestCase):
    def test_none_before_window_fills(self):
        returns = [0.01, -0.01, 0.02, -0.02, 0.01]
        out = probe.rolling_ar1(returns, window=5)
        self.assertTrue(all(v is None for v in out[:-1]))

    def test_perfectly_alternating_series_is_near_minus_one(self):
        # r[t] = (-1)^t * c is a textbook AR1 = -1 series
        returns = [0.01 * (1 if i % 2 == 0 else -1) for i in range(30)]
        out = probe.rolling_ar1(returns, window=10)
        last = out[-1]
        self.assertIsNotNone(last)
        self.assertLess(last, -0.95)

    def test_constant_series_is_none_not_zero(self):
        # zero variance in the window -> undefined correlation, must be
        # None (honest "can't compute"), never silently reported as 0.0
        returns = [0.005] * 20
        out = probe.rolling_ar1(returns, window=10)
        self.assertIsNone(out[-1])

    def test_strongly_trending_series_is_near_plus_one(self):
        returns = [0.001 * i for i in range(1, 31)]
        out = probe.rolling_ar1(returns, window=10)
        self.assertIsNotNone(out[-1])
        self.assertGreater(out[-1], 0.9)


class TestRollingVariance(unittest.TestCase):
    def test_matches_hand_computed_population_variance(self):
        returns = [1.0, 2.0, 3.0, 4.0]
        out = probe.rolling_variance(returns, window=4)
        # population variance of [1,2,3,4], mean=2.5 -> 1.25
        self.assertAlmostEqual(out[-1], 1.25, places=9)

    def test_none_before_window_fills(self):
        out = probe.rolling_variance([0.01, 0.02], window=5)
        self.assertTrue(all(v is None for v in out))

    def test_rising_amplitude_raises_variance(self):
        calm = [0.001 * ((-1) ** i) for i in range(20)]
        volatile = [0.05 * ((-1) ** i) for i in range(20)]
        v_calm = probe.rolling_variance(calm, window=20)[-1]
        v_volatile = probe.rolling_variance(volatile, window=20)[-1]
        self.assertLess(v_calm, v_volatile)


class TestFindTransitionOnsets(unittest.TestCase):
    def test_simple_persistent_transition_detected(self):
        labels = (["BULL"] * 25) + (["BEAR"] * 5)
        onsets = probe.find_transition_onsets(labels, persist_days=3,
                                               stable_days_before=20)
        self.assertEqual(len(onsets), 1)
        self.assertEqual(onsets[0]["index"], 25)
        self.assertEqual(onsets[0]["from"], "BULL")
        self.assertEqual(onsets[0]["to"], "BEAR")
        self.assertEqual(onsets[0]["severity_jump"], 3)

    def test_single_day_flap_is_not_an_onset(self):
        # one-day spike back to BULL immediately after -> fails persist_days
        labels = (["BULL"] * 25) + ["BEAR"] + (["BULL"] * 10)
        onsets = probe.find_transition_onsets(labels, persist_days=3,
                                               stable_days_before=20)
        self.assertEqual(onsets, [])

    def test_transition_without_enough_prior_stability_excluded(self):
        # regime only stable 5 days before jumping -> below stable_days_before=20
        labels = (["NEUTRAL"] * 5) + (["BULL"] * 5) + (["BEAR"] * 5)
        onsets = probe.find_transition_onsets(labels, persist_days=3,
                                               stable_days_before=20)
        self.assertEqual(onsets, [])

    def test_downgrade_is_not_an_onset(self):
        # severity DECREASING (BEAR -> BULL) must never register as an onset
        labels = (["BEAR"] * 25) + (["BULL"] * 10)
        onsets = probe.find_transition_onsets(labels, persist_days=3,
                                               stable_days_before=20)
        self.assertEqual(onsets, [])

    def test_unknown_label_defaults_to_neutral_severity_not_crash(self):
        labels = (["BULL"] * 25) + (["SOME_UNKNOWN_LABEL"] * 5)
        # NEUTRAL(1) < BULL(0)? no: BULL=0, NEUTRAL=1, so this IS a jump
        onsets = probe.find_transition_onsets(labels, persist_days=3,
                                               stable_days_before=20)
        self.assertEqual(len(onsets), 1)


class TestComputeLeadSignal(unittest.TestCase):
    def test_insufficient_onsets_reported_honestly(self):
        returns = [0.001 * ((-1) ** i) for i in range(100)]
        onsets = [{"index": 50, "from": "BULL", "to": "BEAR", "severity_jump": 3}]
        result = probe.compute_lead_signal(returns, onsets, window=10)
        self.assertTrue(result["insufficient_n"])
        self.assertEqual(result["n_onsets"], 1)
        self.assertEqual(result["by_lead_days"], {})

    def test_enough_onsets_produces_comparison_for_every_lead(self):
        # deterministic synthetic series: returns get visibly more volatile
        # and more trending in the 20 days right before each onset
        n = 400
        returns = [0.001 * ((-1) ** i) for i in range(n)]
        onset_positions = [80, 140, 200, 260, 320]
        for pos in onset_positions:
            for k in range(1, 21):
                if pos - k >= 0:
                    returns[pos - k] = 0.01 * (k / 20.0)
        onsets = [{"index": p, "from": "BULL", "to": "BEAR", "severity_jump": 3}
                  for p in onset_positions]
        result = probe.compute_lead_signal(returns, onsets, window=20,
                                            lead_offsets=(10, 5, 1))
        self.assertEqual(result["n_onsets"], 5)
        self.assertNotIn("insufficient_n", result)
        for lead in (10, 5, 1):
            cell = result["by_lead_days"][lead]
            self.assertNotIn("insufficient_n", cell)
            self.assertIsNotNone(cell["onset_mean_var"])
            self.assertIsNotNone(cell["control_mean_var"])
            # the engineered pre-onset volatility burst must read higher
            # variance than the flat control baseline
            self.assertGreater(cell["onset_mean_var"], cell["control_mean_var"])

    def test_reproducible_given_fixed_seed(self):
        n = 400
        returns = [0.001 * ((-1) ** i) for i in range(n)]
        onsets = [{"index": p, "from": "BULL", "to": "BEAR", "severity_jump": 3}
                  for p in (80, 140, 200, 260, 320)]
        r1 = probe.compute_lead_signal(returns, onsets, window=20,
                                        lead_offsets=(5,), rng_seed=42)
        r2 = probe.compute_lead_signal(returns, onsets, window=20,
                                        lead_offsets=(5,), rng_seed=42)
        self.assertEqual(r1["by_lead_days"][5]["control_mean_var"],
                          r2["by_lead_days"][5]["control_mean_var"])


if __name__ == "__main__":
    unittest.main()
