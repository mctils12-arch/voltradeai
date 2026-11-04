"""Tests for forecast_range.py — the daily forecast's 68% range model
(fc-range-har-v1, Part C 2026-11-04, SHADOW).

Pinned: the HAR regressors use closes <= the base close only; implied vol is
converted and plausibility-checked; which coefficient set a name gets (VIX
only for SPY); the IV-missing fallback is labeled; the intraday remaining-
variance profile; the interval score (measurement code); the quantile fit
(non-negative, exact against an LP, scale-equivariant, hits its quantile);
and that the frozen coefficients / intraday table equal the study's json.
"""
import json
import math
import os
import random
import unittest

import numpy as np

import forecast_range as fr

HERE = os.path.dirname(os.path.abspath(__file__))


def _walk(n, seed=1, sd=0.012, start=100.0):
    rng = random.Random(seed)
    c = [start]
    for _ in range(n - 1):
        c.append(c[-1] * math.exp(rng.gauss(0, sd)))
    return c


class TestRegressors(unittest.TestCase):
    def test_har_components_by_hand(self):
        c = _walk(40)
        regs = fr.har_regressors(c)
        r = [math.log(c[k] / c[k - 1]) for k in range(len(c) - 22, len(c))]
        self.assertAlmostEqual(regs["v_d"], abs(r[-1]), places=12)
        self.assertAlmostEqual(regs["v_w"], math.sqrt(sum(x * x for x in r[-5:]) / 5), places=12)
        self.assertAlmostEqual(regs["v_m"], math.sqrt(sum(x * x for x in r) / 22), places=12)

    def test_no_future_closes_are_read(self):
        c = _walk(80, seed=3)
        at = 50
        a = fr.har_regressors(c, at)
        c2 = c[:at + 1] + [x * 5 for x in c[at + 1:]]
        self.assertEqual(a, fr.har_regressors(c2, at))
        self.assertEqual(a, fr.har_regressors(c[:at + 1]))

    def test_short_or_bad_history_is_none(self):
        self.assertIsNone(fr.har_regressors(_walk(22)))
        c = _walk(40)
        c[-3] = 0.0
        self.assertIsNone(fr.har_regressors(c))
        c = _walk(40)
        c[-5] = float("nan")
        self.assertIsNone(fr.har_regressors(c))

    def test_iv_regressor(self):
        self.assertAlmostEqual(fr.iv_regressor(0.16), 0.16 / math.sqrt(252), places=12)
        for bad in (None, 0.0, -0.2, float("nan"), 6.0, True, "0.2"):
            self.assertIsNone(fr.iv_regressor(bad), bad)


class TestBand(unittest.TestCase):
    COEF = {"index_iv": {"v_d": 0.1, "v_w": 0.2, "v_m": 0.3, "v_iv": 0.4},
            "stock_iv": {"v_d": 0.05, "v_w": 0.25, "v_m": 0.25, "v_iv": 0.5},
            "har": {"v_d": 0.1, "v_w": 0.3, "v_m": 0.6}}

    def test_halfwidth_is_the_weighted_sum(self):
        q = fr.band_halfwidth({"v_d": 0.5, "v_m": 0.25}, {"v_d": 0.02, "v_m": 0.04, "v_w": 9.0})
        self.assertAlmostEqual(q, 0.5 * 0.02 + 0.25 * 0.04, places=15)
        self.assertIsNone(fr.band_halfwidth({"v_iv": 1.0}, {"v_d": 0.01}))
        self.assertIsNone(fr.band_halfwidth({"v_d": 0.0}, {"v_d": 0.01}))

    def test_which_set_a_name_uses(self):
        self.assertEqual(fr.coef_set_for("SPY", "vix"), "index_iv")
        self.assertEqual(fr.coef_set_for("QQQ", "vix"), "har", "VIX is SPY's implied vol only")
        self.assertEqual(fr.coef_set_for("AAPL", "chain_iv30"), "stock_iv")
        self.assertEqual(fr.coef_set_for("SPY", "chain_iv30"), "stock_iv")
        self.assertEqual(fr.coef_set_for("AAPL", None), "har")

    def test_daily_range_uses_iv_when_plausible_and_labels_the_fallback(self):
        c = _walk(60, seed=5)
        regs = fr.har_regressors(c)
        spy = fr.daily_range(c, "SPY", 0.18, "vix", coef=self.COEF)
        self.assertEqual((spy["coef_set"], spy["range_source"]), ("index_iv", "har+vix"))
        want = 0.1 * regs["v_d"] + 0.2 * regs["v_w"] + 0.3 * regs["v_m"] + 0.4 * 0.18 / math.sqrt(252)
        self.assertAlmostEqual(spy["q"], want, places=12)
        self.assertAlmostEqual(spy["iv_weight"], 0.4, places=4)
        stale = fr.daily_range(c, "SPY", float("nan"), "vix", coef=self.COEF)
        self.assertEqual((stale["coef_set"], stale["range_source"]), ("har", "har"))
        self.assertNotIn("v_iv", stale["regressors"])
        name = fr.daily_range(c, "AAPL", 0.31, "chain_iv30", coef=self.COEF)
        self.assertEqual(name["range_source"], "har+chain_iv30")
        self.assertEqual(name["model_version"], fr.RANGE_MODEL_VERSION)
        self.assertIsNone(fr.daily_range(c[:10], "SPY", 0.2, "vix", coef=self.COEF))
        self.assertIsNone(fr.daily_range(c, "SPY", 0.2, "vix", coef={}))


class TestIntraday(unittest.TestCase):
    TABLE = {"index": ((570, 0.7), (630, 0.5), (930, 0.1)), "other": ((570, 0.8), (930, 0.2))}

    def test_profile_bounds_and_interpolation(self):
        f = fr.remaining_variance_fraction
        self.assertEqual(f(8 * 60, "SPY", self.TABLE), 1.0)
        self.assertEqual(f(16 * 60, "SPY", self.TABLE), 0.0)
        self.assertEqual(f(17 * 60, "AAPL", self.TABLE), 0.0)
        self.assertAlmostEqual(f(570, "SPY", self.TABLE), 0.7)
        self.assertAlmostEqual(f(600, "SPY", self.TABLE), 0.6)
        self.assertAlmostEqual(f(945, "SPY", self.TABLE), 0.05)
        self.assertAlmostEqual(f(570, "AAPL", self.TABLE), 0.8)
        self.assertAlmostEqual(f(750, "QQQ", self.TABLE), 0.5 + (0.1 - 0.5) * 120 / 300)
        self.assertIsNone(f(600, "SPY", {}))

    def test_profile_never_increases_within_the_session(self):
        if not fr.REMAINING_VARIANCE:
            self.skipTest("no frozen table loaded")
        for sym in ("SPY", "AAPL"):
            vals = [fr.remaining_variance_fraction(m, sym) for m in range(570, 961, 5)]
            self.assertTrue(all(0.0 <= v <= 1.0 for v in vals))
            self.assertTrue(all(b <= a + 1e-12 for a, b in zip(vals, vals[1:])), sym)


class TestIntervalScore(unittest.TestCase):
    def test_inside_is_the_width_outside_adds_the_penalty(self):
        a = fr.RANGE_ALPHA
        self.assertAlmostEqual(fr.interval_score(0.02, 0.01), 0.04)
        self.assertAlmostEqual(fr.interval_score(0.02, -0.02), 0.04)
        self.assertAlmostEqual(fr.interval_score(0.02, 0.05), 0.04 + (2 / a) * 0.03)
        self.assertAlmostEqual(fr.interval_score(0.02, -0.05), 0.04 + (2 / a) * 0.03)
        arr = fr.interval_score(np.array([0.02, 0.02]), np.array([0.01, -0.05]))
        self.assertTrue(np.allclose(arr, [0.04, 0.04 + (2 / a) * 0.03]))

    def test_the_tau_quantile_minimizes_the_expected_score(self):
        rng = np.random.default_rng(0)
        y = rng.standard_t(4, 200_000) * 0.01
        qs = np.linspace(0.004, 0.02, 81)
        best = qs[int(np.argmin([np.mean(fr.interval_score(np.full_like(y, q), y)) for q in qs]))]
        self.assertAlmostEqual(best, float(np.quantile(np.abs(y), fr.RANGE_TAU)), delta=0.0003)


class TestQuantileFit(unittest.TestCase):
    def _data(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        X = np.abs(rng.normal(0.01, 0.004, (n, 3)))
        scale = 0.2 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2]
        y = np.abs(rng.standard_t(5, n)) * scale
        return X, y

    def test_coverage_in_sample_is_tau_and_theta_nonnegative(self):
        X, y = self._data()
        th = fr.fit_quantile_nonneg(X, y)
        self.assertTrue(np.all(th >= 0))
        self.assertAlmostEqual(float(np.mean(y <= X @ th)), fr.RANGE_TAU, delta=0.01)

    def test_matches_an_exact_lp(self):
        from scipy.optimize import linprog
        X, y = self._data(n=400, seed=2)
        n, p = X.shape
        tau = fr.RANGE_TAU
        c = np.concatenate([np.zeros(p), tau * np.ones(n), (1 - tau) * np.ones(n)])
        A = np.hstack([X, np.eye(n), -np.eye(n)])
        lp = linprog(c, A_eq=A, b_eq=y, bounds=(0, None), method="highs")
        th = fr.fit_quantile_nonneg(X, y)
        self.assertLessEqual(fr.pinball(y, X @ th, tau), fr.pinball(y, X @ lp.x[:p], tau) * (1 + 1e-6))

    def test_scale_equivariant(self):
        X, y = self._data(seed=4)
        th = fr.fit_quantile_nonneg(X, y)
        th2 = fr.fit_quantile_nonneg(X * 7.0, y * 7.0)
        self.assertTrue(np.allclose(th, th2, rtol=1e-4, atol=1e-6))

    def test_the_sign_constraint_binds_on_a_regressor_that_wants_a_negative_weight(self):
        rng = np.random.default_rng(9)
        n = 6000
        X = np.abs(rng.normal(0.01, 0.003, (n, 3)))
        scale = 0.6 * X[:, 0] + 0.6 * X[:, 1] - 0.3 * X[:, 2]    # the third one narrows the quantile
        y = np.abs(rng.standard_normal(n)) * np.maximum(scale, 1e-4)
        th = fr.fit_quantile_nonneg(X, y)
        self.assertTrue(np.all(th >= 0))
        self.assertEqual(th[2], 0.0)
        self.assertGreater(th[0], 0.0)
        self.assertGreater(th[1], 0.0)

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            fr.fit_quantile_nonneg(np.zeros((3, 2)), np.zeros(4))


class TestFrozenModelMatchesTheStudy(unittest.TestCase):
    """The production coefficients and the intraday table are exactly the
    study's last walk-forward refit — a refit must come with a new study."""

    def _study(self):
        p = os.path.join(HERE, fr.RANGE_COEF_SOURCE)
        with open(p) as fh:
            return json.load(fh)

    def test_coefficients_equal_the_study_json(self):
        js = self._study()
        self.assertEqual(set(fr.RANGE_COEF), {"index_iv", "stock_iv", "har"})
        for name, key in fr.RANGE_COEF_KEYS.items():
            fit = js["coef_final"][key]
            cols = fr.RANGE_COEF_COLUMNS[name]
            self.assertEqual(list(fr.RANGE_COEF[name]), list(cols))
            for col, th in zip(cols, fit["theta"]):
                self.assertAlmostEqual(fr.RANGE_COEF[name][col], th, places=6, msg=f"{name}.{col}")
            self.assertTrue(all(v >= 0 for v in fr.RANGE_COEF[name].values()))

    def test_intraday_table_equals_the_study_json(self):
        # the shipped table is the study's QUANTILE-ratio estimate (fq_full) —
        # the declared post-hoc variant; the pre-registered variance ratio
        # (f_full) failed its coverage prior and stays in the json for the record
        js = self._study()
        marks = js["intraday"]["marks"]
        for grp in ("index", "other"):
            self.assertIn("f_full", js["intraday"]["groups"][grp])
            f = js["intraday"]["groups"][grp]["fq_full"]
            want = tuple((int(m[:2]) * 60 + int(m[3:]), round(v, 4)) for m, v in zip(marks, f))
            self.assertEqual(fr.REMAINING_VARIANCE[grp], want, grp)


if __name__ == "__main__":
    unittest.main()
