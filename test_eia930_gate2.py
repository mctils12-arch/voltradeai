"""
Regression tests for scripts/eia930_gate2.py — the ROOT VALIDATION LADDER
gate 2 (SIGNAL) screen for the EIA-930 grid demand archive. Pure-function
tests only: no network calls (no live EIA API fetch, no backtest_v2
Alpaca/Yahoo fetch).
"""
import importlib.util
import math
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "eia930_gate2",
    os.path.join(os.path.dirname(__file__), "scripts", "eia930_gate2.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_design_matrix = _mod.build_design_matrix
fit_and_residualize = _mod.fit_and_residualize
smooth_and_percentile = _mod.smooth_and_percentile
weekly_entries = _mod.weekly_entries
bucket_for = _mod.bucket_for
find_entry_index = _mod.find_entry_index
compute_forward_returns = _mod.compute_forward_returns
summarize = _mod.summarize
_newey_west_diff_test = _mod._newey_west_diff_test
evaluate_pass_bar = _mod.evaluate_pass_bar
EXTREME_HIGH = _mod.EXTREME_HIGH
EXTREME_LOW = _mod.EXTREME_LOW


class TestBuildDesignMatrix(unittest.TestCase):
    def test_row_shape_and_dummies(self):
        dd = {"2022-01-03": (10.0, 0.0)}  # Monday, January
        X, dates = build_design_matrix(["2022-01-03"], dd)
        self.assertEqual(dates, ["2022-01-03"])
        self.assertEqual(X.shape, (1, 20))  # intercept + HDD + CDD + 6 weekday + 11 month
        self.assertEqual(X[0, 0], 1.0)  # intercept
        self.assertEqual(X[0, 1], 10.0)  # HDD
        self.assertEqual(X[0, 2], 0.0)  # CDD
        # Monday + January are the baseline categories -> all dummy columns 0
        self.assertTrue((X[0, 3:] == 0.0).all())

    def test_tuesday_july_sets_correct_dummies(self):
        dd = {"2022-07-05": (0.0, 15.0)}  # Tuesday, July
        X, dates = build_design_matrix(["2022-07-05"], dd)
        # weekday dummies occupy columns 3-8 (Tue..Sun); Tuesday = weekday()==1 -> column 3
        self.assertEqual(X[0, 3], 1.0)
        self.assertEqual(sum(X[0, 3:9]), 1.0)
        # month dummies occupy columns 9-19; July = month 7 -> index 7-2=5 -> column 9+5=14
        self.assertEqual(X[0, 14], 1.0)
        self.assertEqual(sum(X[0, 9:19]), 1.0)

    def test_missing_degree_day_date_dropped(self):
        dd = {"2022-01-03": (10.0, 0.0)}
        X, dates = build_design_matrix(["2022-01-03", "2022-01-04"], dd)
        self.assertEqual(dates, ["2022-01-03"])
        self.assertEqual(X.shape[0], 1)


class TestFitAndResidualize(unittest.TestCase):
    def test_perfect_linear_relationship_gives_near_zero_residual(self):
        # demand = exp(1.0 + 0.01*HDD) exactly -> residual should vanish
        # everywhere once regressed on HDD (CDD=0, no weekday/month signal).
        demand, dd = {}, {}
        for i in range(400):
            from datetime import date, timedelta
            d = (date(2019, 1, 1) + timedelta(days=i)).isoformat()
            hdd = float(i % 30)
            demand[d] = math.exp(1.0 + 0.01 * hdd)
            dd[d] = (hdd, 0.0)
        resid, n_train, beta = fit_and_residualize(demand, dd)
        self.assertGreater(n_train, 200)
        self.assertTrue(all(abs(r) < 1e-6 for r in resid.values()))

    def test_raises_on_insufficient_training_data(self):
        demand = {"2019-01-01": 1000.0}
        dd = {"2019-01-01": (5.0, 0.0)}
        with self.assertRaises(SystemExit):
            fit_and_residualize(demand, dd)


class TestSmoothAndPercentile(unittest.TestCase):
    def test_percentile_none_before_lookback_seed(self):
        resid_by_date = {f"2022-01-{d:02d}": 0.0 for d in range(1, 20)}
        idx = smooth_and_percentile(resid_by_date)
        # fewer than 60 trailing days seeded -> pctile stays None
        self.assertTrue(all(v["pctile"] is None for v in idx.values()))

    def test_monotonic_trend_ranks_last_day_highest(self):
        resid_by_date = {}
        for i in range(120):
            d = f"2022-{1 + i // 28:02d}-{1 + i % 28:02d}"
            resid_by_date[d] = float(i)  # strictly increasing
        idx = smooth_and_percentile(resid_by_date)
        last_date = sorted(idx)[-1]
        self.assertEqual(idx[last_date]["pctile"], 100.0)


class TestWeeklyEntries(unittest.TestCase):
    def test_picks_last_scored_day_per_iso_week(self):
        # 2022-01-03 (Mon) .. 2022-01-09 (Sun) is ISO week 1 of 2022.
        index_by_date = {
            "2022-01-03": {"pctile": 50.0},
            "2022-01-05": {"pctile": 60.0},
            "2022-01-07": {"pctile": 70.0},
        }
        entries = weekly_entries(index_by_date, "2022-01-01")
        self.assertEqual(entries, ["2022-01-07"])

    def test_drops_unscored_days(self):
        index_by_date = {
            "2022-01-03": {"pctile": None},
            "2022-01-04": {"pctile": 50.0},
        }
        entries = weekly_entries(index_by_date, "2022-01-01")
        self.assertEqual(entries, ["2022-01-04"])

    def test_respects_validation_start(self):
        index_by_date = {"2021-12-31": {"pctile": 90.0}, "2022-01-03": {"pctile": 50.0}}
        entries = weekly_entries(index_by_date, "2022-01-01")
        self.assertEqual(entries, ["2022-01-03"])


class TestBucketFor(unittest.TestCase):
    def test_extreme_high(self):
        self.assertEqual(bucket_for(EXTREME_HIGH), "extreme_high")
        self.assertEqual(bucket_for(95.0), "extreme_high")

    def test_extreme_low(self):
        self.assertEqual(bucket_for(EXTREME_LOW), "extreme_low")
        self.assertEqual(bucket_for(2.0), "extreme_low")

    def test_mid(self):
        self.assertEqual(bucket_for(50.0), "mid")

    def test_none_passthrough(self):
        self.assertIsNone(bucket_for(None))


class TestFindEntryIndex(unittest.TestCase):
    def test_first_strictly_after(self):
        dates = ["2022-01-03", "2022-01-04", "2022-01-05"]
        self.assertEqual(find_entry_index(dates, "2022-01-03"), 1)

    def test_none_when_no_later_date(self):
        dates = ["2022-01-03"]
        self.assertIsNone(find_entry_index(dates, "2022-01-03"))


class TestComputeForwardReturns(unittest.TestCase):
    def test_no_lookahead_and_right_censoring(self):
        index_by_date = {
            "2022-01-03": {"pctile": 95.0},
            "2022-01-10": {"pctile": 5.0},
        }
        bars = {
            "date": ["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06", "2022-01-11"],
            "close": [100.0, 101.0, 102.0, 103.0, 110.0],
        }
        rows = compute_forward_returns(["2022-01-03", "2022-01-10"], index_by_date, bars)
        self.assertEqual(rows[0]["bucket"], "extreme_high")
        self.assertEqual(rows[0]["entry_date"], "2022-01-04")  # strictly after 01-03
        self.assertEqual(rows[1]["bucket"], "extreme_low")
        self.assertEqual(rows[1]["entry_date"], "2022-01-11")


class TestNeweyWestDiffTest(unittest.TestCase):
    def test_degenerate_bucket_returns_none(self):
        rows = [{"forward_returns": {20: 0.01}, "bucket": "mid"} for _ in range(20)]
        self.assertIsNone(_newey_west_diff_test(rows, 20, "extreme_high"))

    def test_too_few_observations_returns_none(self):
        rows = [{"forward_returns": {20: 0.01}, "bucket": "extreme_high"}]
        self.assertIsNone(_newey_west_diff_test(rows, 20, "extreme_high"))

    def test_clear_separation_yields_large_t_stat(self):
        rows = []
        for _ in range(30):
            rows.append({"forward_returns": {20: 0.10}, "bucket": "extreme_high"})
            rows.append({"forward_returns": {20: -0.10}, "bucket": "mid"})
        result = _newey_west_diff_test(rows, 20, "extreme_high")
        self.assertIsNotNone(result)
        self.assertGreater(result["t_stat"], 5.0)
        self.assertLess(result["p_value"], 0.001)


class TestEvaluatePassBar(unittest.TestCase):
    def test_fails_on_sign_disagreement(self):
        summary = {
            "20": {"extreme_high": {"mean_pct": -0.4}, "extreme_low": {"mean_pct": 2.5}},
            "60": {"extreme_high": {"mean_pct": -1.5}, "extreme_low": {"mean_pct": 1.7}},
        }
        significance = {
            "20": {"extreme_high": None, "extreme_low": None},
            "60": {"extreme_high": None, "extreme_low": None},
        }
        result = evaluate_pass_bar(summary, significance)
        self.assertFalse(result["PASSED"])

    def test_passes_when_sign_agrees_and_bonferroni_clears(self):
        summary = {
            "20": {"extreme_high": {"mean_pct": 3.0}, "extreme_low": {"mean_pct": -2.0}},
            "60": {"extreme_high": {"mean_pct": 4.0}, "extreme_low": {"mean_pct": -3.0}},
        }
        significance = {
            "20": {"extreme_high": {"p_value": 0.001}, "extreme_low": {"p_value": 0.5}},
            "60": {"extreme_high": {"p_value": 0.5}, "extreme_low": {"p_value": 0.5}},
        }
        result = evaluate_pass_bar(summary, significance)
        self.assertTrue(result["PASSED"])

    def test_fails_when_no_comparison_clears_bonferroni(self):
        summary = {
            "20": {"extreme_high": {"mean_pct": 3.0}, "extreme_low": {"mean_pct": -2.0}},
            "60": {"extreme_high": {"mean_pct": 4.0}, "extreme_low": {"mean_pct": -3.0}},
        }
        significance = {
            "20": {"extreme_high": {"p_value": 0.5}, "extreme_low": {"p_value": 0.5}},
            "60": {"extreme_high": {"p_value": 0.5}, "extreme_low": {"p_value": 0.5}},
        }
        result = evaluate_pass_bar(summary, significance)
        self.assertFalse(result["PASSED"])


if __name__ == "__main__":
    unittest.main()
