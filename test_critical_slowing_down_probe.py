"""
Regression tests for critical_slowing_down_probe.py — the FOREIGN-FIELD
IMPORT (ecology critical-slowing-down theory) screen testing whether
rolling return autocorrelation/variance predict forward realized
volatility. Pure-function tests only: no network calls, mirrors the
existing test_regime_detector_compare.py convention this script's sibling
already established.
"""
import math
import unittest

import numpy as np

from critical_slowing_down_probe import (
    compare_signals,
    compute_features,
    rolling_autocorr_lag1,
    rolling_std,
    _clean_pairs,
    _spearman,
)


class TestRollingAutocorrLag1(unittest.TestCase):
    def test_white_noise_autocorr_near_zero(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 1, 200)
        out = rolling_autocorr_lag1(returns, window=20)
        tail = out[~np.isnan(out)][-100:]
        # white noise: no true serial dependence, sample autocorr should
        # average close to 0 (loose bound — this is a statistical, not
        # exact, property, so the bound is generous on purpose)
        self.assertLess(abs(np.mean(tail)), 0.15)

    def test_strongly_autocorrelated_series_detected(self):
        rng = np.random.default_rng(7)
        n = 200
        phi = 0.8
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + rng.normal(0, 0.3)
        out = rolling_autocorr_lag1(x, window=40)
        tail = out[~np.isnan(out)][-50:]
        # an AR(1) series with phi=0.8 should show clearly positive,
        # elevated sample autocorrelation — not a precise phi recovery,
        # but must be well above a white-noise reading
        self.assertGreater(np.mean(tail), 0.4)

    def test_insufficient_history_is_nan(self):
        out = rolling_autocorr_lag1(np.array([1.0, 2.0, 3.0]), window=20)
        self.assertTrue(np.all(np.isnan(out)))

    def test_constant_window_is_nan_not_crash(self):
        # zero variance inside the window -> correlation is undefined;
        # must return NaN, not raise or silently return 0/1
        returns = np.array([0.0] * 30)
        out = rolling_autocorr_lag1(returns, window=20)
        self.assertTrue(np.all(np.isnan(out)))


class TestRollingStd(unittest.TestCase):
    def test_matches_manual_stdev(self):
        returns = np.arange(1.0, 31.0)  # 1..30
        out = rolling_std(returns, window=10)
        expected = float(np.std(returns[10:20], ddof=1))
        self.assertAlmostEqual(out[19], expected, places=9)

    def test_insufficient_history_is_nan(self):
        out = rolling_std(np.array([1.0, 2.0]), window=10)
        self.assertTrue(np.all(np.isnan(out)))


class TestComputeFeatures(unittest.TestCase):
    def _synthetic(self, n=150):
        rng = np.random.default_rng(1)
        spy = [100.0]
        for _ in range(n - 1):
            spy.append(spy[-1] * (1 + rng.normal(0, 0.01)))
        vix = [15.0 + rng.normal(0, 0.5) for _ in range(n)]
        dates = [f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)]
        return dates, spy, vix

    def test_no_lookahead_forward_vol_uses_only_future_days(self):
        dates, spy, vix = self._synthetic(n=150)
        rows = compute_features(dates, spy, vix, window=20)
        # inject a deterministic post-hoc spike into a fresh copy at day 100
        # and confirm ONLY rows whose forward window reaches it move
        spy2 = list(spy)
        for i in range(100, 105):
            spy2[i] = spy2[i - 1] * 1.05
        rows2 = compute_features(dates, spy2, vix, window=20)
        by_date1 = {r["date"]: r for r in rows}
        by_date2 = {r["date"]: r for r in rows2}
        # day index 70: forward 20d window (71..90) never touches day 100 -> unaffected
        d_far = dates[70]
        if d_far in by_date1 and d_far in by_date2:
            self.assertAlmostEqual(by_date1[d_far]["fwd_vol_20"], by_date2[d_far]["fwd_vol_20"], places=9)
        # day index 82: forward 20d window (83..102) touches the injected spike -> must differ
        d_near = dates[82]
        if d_near in by_date1 and d_near in by_date2:
            self.assertNotAlmostEqual(by_date1[d_near]["fwd_vol_20"], by_date2[d_near]["fwd_vol_20"], places=6)

    def test_drops_warmup_and_tail_rows(self):
        dates, spy, vix = self._synthetic(n=150)
        rows = compute_features(dates, spy, vix, window=20)
        warmup = 20 + 30
        self.assertEqual(len(rows), 150 - warmup - 20)
        self.assertEqual(rows[0]["date"], dates[warmup])

    def test_row_shape(self):
        dates, spy, vix = self._synthetic(n=150)
        rows = compute_features(dates, spy, vix, window=20)
        for key in ("date", "trailing_autocorr", "trailing_vol", "vix_ratio", "fwd_vol_5", "fwd_vol_20"):
            self.assertIn(key, rows[0])


class TestCompareSignals(unittest.TestCase):
    def _rows_with_known_correlation(self, n=200, rho_target=0.9):
        rng = np.random.default_rng(3)
        rows = []
        for i in range(n):
            ac = rng.normal(0, 1)
            fv5 = rho_target * ac + math.sqrt(1 - rho_target ** 2) * rng.normal(0, 1)
            fv20 = fv5
            rows.append({
                "date": f"d{i}", "trailing_autocorr": float(ac), "trailing_vol": float(abs(ac)),
                "vix_ratio": 1.0 + 0.1 * ac, "fwd_vol_5": float(fv5), "fwd_vol_20": float(fv20),
            })
        return rows

    def test_recovers_strong_known_correlation(self):
        rows = self._rows_with_known_correlation()
        out = compare_signals(rows)
        rho = out["predictive_power"]["trailing_autocorr"]["fwd_vol_5"]["rho"]
        self.assertGreater(rho, 0.7)

    def test_handles_missing_values_without_crashing(self):
        rows = [
            {"date": "d1", "trailing_autocorr": None, "trailing_vol": 1.0, "vix_ratio": 1.0,
             "fwd_vol_5": 0.5, "fwd_vol_20": 0.6},
            {"date": "d2", "trailing_autocorr": 0.1, "trailing_vol": None, "vix_ratio": None,
             "fwd_vol_5": None, "fwd_vol_20": 0.7},
        ] * 10
        out = compare_signals(rows)
        self.assertIsNotNone(out)  # must not raise

    def test_lead_lag_keys_present_for_all_horizons(self):
        rows = self._rows_with_known_correlation()
        out = compare_signals(rows)
        for k in (0, 5, 10, 20):
            self.assertIn(f"lag_{k}", out["lead_lag_autocorr_vs_future_vix_ratio"])

    def test_conditional_buckets_partition_by_tercile(self):
        rows = self._rows_with_known_correlation(n=300)
        out = compare_signals(rows)
        cond = out["conditional_on_vix_ratio_tercile"]
        total_n = sum(cond[b]["n"] for b in ("low_vix_ratio", "mid_vix_ratio", "high_vix_ratio"))
        self.assertEqual(total_n, 300)

    def test_base_rate_reported(self):
        rows = self._rows_with_known_correlation()
        out = compare_signals(rows)
        self.assertIsNotNone(out["fwd_vol_5_base_mean"])
        self.assertIsNotNone(out["fwd_vol_5_base_std"])


class TestCleanPairsAndSpearman(unittest.TestCase):
    def test_clean_pairs_drops_none_and_nan(self):
        xa, ya = _clean_pairs([1.0, None, float("nan"), 4.0], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(xa, [1.0, 4.0])
        self.assertEqual(ya, [1.0, 4.0])

    def test_spearman_below_min_n_returns_none(self):
        result = _spearman([1.0] * 5, [2.0] * 5)
        self.assertIsNone(result["rho"])


if __name__ == "__main__":
    unittest.main()
