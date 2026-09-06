"""
Regression tests for scripts/crop_conditions_gate2.py — the ROOT VALIDATION
LADDER gate 2 (SIGNAL) screen for the crop_conditions_usda_nass root. Pure-
function tests only: no network calls (no live /api/data/crop-conditions/
history fetch, no backtest_v2 Alpaca/Yahoo fetch).
"""
import importlib.util
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "crop_conditions_gate2",
    os.path.join(os.path.dirname(__file__), "scripts", "crop_conditions_gate2.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

good_excellent_pct = _mod.good_excellent_pct
compute_weekly_deltas = _mod.compute_weekly_deltas
release_date_for = _mod.release_date_for
bucket_for = _mod.bucket_for
compute_forward_returns = _mod.compute_forward_returns
summarize = _mod.summarize
hac_significance = _mod.hac_significance
evaluate_pass_bar = _mod.evaluate_pass_bar
BONFERRONI_N = _mod.BONFERRONI_N


class TestGoodExcellentPct(unittest.TestCase):
    def test_sums_good_and_excellent(self):
        self.assertEqual(good_excellent_pct({"GOOD": 57, "EXCELLENT": 10, "FAIR": 28, "POOR": 4, "VERY POOR": 1}), 67)

    def test_missing_good_is_none(self):
        self.assertIsNone(good_excellent_pct({"EXCELLENT": 10}))

    def test_missing_excellent_is_none(self):
        self.assertIsNone(good_excellent_pct({"GOOD": 57}))

    def test_empty_row_is_none(self):
        self.assertIsNone(good_excellent_pct({}))


class TestComputeWeeklyDeltas(unittest.TestCase):
    def test_first_week_produces_no_row(self):
        trend = [{"week_ending": "2026-05-31", "corn": {"GOOD": 57, "EXCELLENT": 10}}]
        self.assertEqual(compute_weekly_deltas(trend, "corn"), [])

    def test_two_weeks_produces_one_delta(self):
        trend = [
            {"week_ending": "2026-05-31", "corn": {"GOOD": 57, "EXCELLENT": 10}},  # GE=67
            {"week_ending": "2026-06-07", "corn": {"GOOD": 55, "EXCELLENT": 12}},  # GE=67
        ]
        deltas = compute_weekly_deltas(trend, "corn")
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["week_ending"], "2026-06-07")
        self.assertEqual(deltas[0]["ge_pct"], 67)
        self.assertEqual(deltas[0]["delta_ge_pct"], 0)

    def test_negative_delta_on_worsening_conditions(self):
        trend = [
            {"week_ending": "2026-07-19", "corn": {"GOOD": 53, "EXCELLENT": 14}},  # GE=67
            {"week_ending": "2026-07-26", "corn": {"GOOD": 49, "EXCELLENT": 14}},  # GE=63
        ]
        deltas = compute_weekly_deltas(trend, "corn")
        self.assertEqual(deltas[0]["delta_ge_pct"], -4)

    def test_gap_week_does_not_corrupt_next_delta(self):
        """A week missing a commodity's data (e.g. GOOD/EXCELLENT absent) is
        skipped entirely -- the NEXT present week diffs against the last
        present week before the gap, not against a fabricated zero."""
        trend = [
            {"week_ending": "2026-06-01", "corn": {"GOOD": 50, "EXCELLENT": 10}},  # GE=60
            {"week_ending": "2026-06-08", "corn": {}},  # gap -- no row emitted, prev unchanged
            {"week_ending": "2026-06-15", "corn": {"GOOD": 52, "EXCELLENT": 10}},  # GE=62
        ]
        deltas = compute_weekly_deltas(trend, "corn")
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["week_ending"], "2026-06-15")
        self.assertEqual(deltas[0]["delta_ge_pct"], 2)

    def test_commodity_key_absent_from_row_is_skipped(self):
        trend = [
            {"week_ending": "2026-06-01", "corn": {"GOOD": 50, "EXCELLENT": 10}},
            {"week_ending": "2026-06-08", "soybeans": {"GOOD": 50, "EXCELLENT": 10}},
        ]
        self.assertEqual(compute_weekly_deltas(trend, "corn"), [])


class TestReleaseDateFor(unittest.TestCase):
    def test_default_buffer_is_two_days(self):
        self.assertEqual(release_date_for("2026-08-02"), "2026-08-04")

    def test_custom_buffer(self):
        self.assertEqual(release_date_for("2026-08-02", buffer_days=1), "2026-08-03")

    def test_crosses_month_boundary(self):
        self.assertEqual(release_date_for("2026-07-31"), "2026-08-02")


class TestBucketFor(unittest.TestCase):
    def test_positive_is_improving(self):
        self.assertEqual(bucket_for(2), "improving")

    def test_negative_is_worsening(self):
        self.assertEqual(bucket_for(-3), "worsening")

    def test_zero_is_none(self):
        self.assertIsNone(bucket_for(0))

    def test_none_is_none(self):
        self.assertIsNone(bucket_for(None))


class TestComputeForwardReturns(unittest.TestCase):
    def _bars(self):
        # 30 trading days, +1% each day from a base of 100.
        dates = [f"2026-08-{d:02d}" for d in range(1, 25)] + [f"2026-09-{d:02d}" for d in range(1, 7)]
        closes = [100.0 * (1.01 ** i) for i in range(len(dates))]
        return {"date": dates, "close": closes}

    def test_entry_strictly_after_publish_date(self):
        deltas = [{"week_ending": "2026-07-30", "delta_ge_pct": -2}]  # publish_date = 2026-08-01
        rows = compute_forward_returns(deltas, self._bars(), horizons=(5,))
        self.assertEqual(rows[0]["entry_date"], "2026-08-02")  # first bar strictly after 2026-08-01

    def test_forward_return_matches_close_ratio(self):
        deltas = [{"week_ending": "2026-07-30", "delta_ge_pct": -2}]
        bars = self._bars()
        rows = compute_forward_returns(deltas, bars, horizons=(5,))
        entry_idx = bars["date"].index(rows[0]["entry_date"])
        expected = bars["close"][entry_idx + 5] / bars["close"][entry_idx] - 1
        self.assertAlmostEqual(rows[0]["forward_returns"][5], expected)

    def test_bucket_carried_through(self):
        deltas = [{"week_ending": "2026-07-30", "delta_ge_pct": -2}, {"week_ending": "2026-07-23", "delta_ge_pct": 3}]
        rows = compute_forward_returns(deltas, self._bars(), horizons=(5,))
        self.assertEqual(rows[0]["bucket"], "worsening")
        self.assertEqual(rows[1]["bucket"], "improving")

    def test_no_lookahead_beyond_archive_tail_omits_horizon(self):
        deltas = [{"week_ending": "2026-08-28", "delta_ge_pct": -1}]  # publish_date near tail
        rows = compute_forward_returns(deltas, self._bars(), horizons=(20,))
        self.assertNotIn(20, rows[0]["forward_returns"])

    def test_missing_entry_when_publish_date_beyond_archive(self):
        deltas = [{"week_ending": "2026-12-01", "delta_ge_pct": -1}]
        rows = compute_forward_returns(deltas, self._bars(), horizons=(5,))
        self.assertIsNone(rows[0]["entry_date"])
        self.assertEqual(rows[0]["forward_returns"], {})


class TestSummarize(unittest.TestCase):
    def test_splits_by_bucket_and_reports_baseline(self):
        rows = [
            {"bucket": "worsening", "forward_returns": {5: 0.02}},
            {"bucket": "worsening", "forward_returns": {5: 0.04}},
            {"bucket": "improving", "forward_returns": {5: -0.01}},
        ]
        summary = summarize(rows, horizons=(5,))
        self.assertEqual(summary["5"]["worsening"]["n"], 2)
        self.assertAlmostEqual(summary["5"]["worsening"]["mean_pct"], 3.0)
        self.assertEqual(summary["5"]["improving"]["n"], 1)
        self.assertEqual(summary["5"]["baseline"]["n"], 3)

    def test_missing_horizon_row_excluded(self):
        rows = [{"bucket": "worsening", "forward_returns": {}}]
        summary = summarize(rows, horizons=(5,))
        self.assertEqual(summary["5"]["baseline"]["n"], 0)
        self.assertIsNone(summary["5"]["baseline"]["mean_pct"])


class TestHacSignificanceAndPassBar(unittest.TestCase):
    def _rows(self, n_worsening_pos, n_improving_neg, n_flat):
        """Builds a synthetic strongly-separated dataset: worsening bucket
        gets a positive forward return, improving bucket a negative one,
        flat (baseline-only, no bucket) near zero -- enough rows for
        gate2_stats' own n >= 2*lag+4 floor at horizon 5 (lag=1, n>=6)."""
        rows = []
        for i in range(n_worsening_pos):
            rows.append({"bucket": "worsening", "forward_returns": {5: 0.05 + 0.001 * i}})
        for i in range(n_improving_neg):
            rows.append({"bucket": "improving", "forward_returns": {5: -0.05 - 0.001 * i}})
        for i in range(n_flat):
            rows.append({"bucket": None, "forward_returns": {5: 0.0001 * i}})
        return rows

    def test_significant_result_in_hypothesized_direction_detected(self):
        rows = self._rows(6, 6, 2)
        significance = hac_significance(rows, horizons=(5,))
        worsening = significance["5"]["worsening"]
        self.assertIsNotNone(worsening)
        self.assertGreater(worsening["mean_diff_pct"], 0)  # worsening -> positive forward return, as hypothesized

    def test_too_few_observations_returns_none_not_fabricated(self):
        rows = self._rows(1, 1, 0)  # far below horizon-5's n>=6 floor
        significance = hac_significance(rows, horizons=(5,))
        self.assertIsNone(significance["5"]["worsening"])
        self.assertIsNone(significance["5"]["improving"])

    def test_pass_bar_requires_bonferroni_and_direction(self):
        rows = self._rows(8, 8, 2)
        significance = hac_significance(rows, horizons=(5,))
        result = evaluate_pass_bar({"corn": significance}, horizons=(5,))
        self.assertTrue(result["PASSED"])
        self.assertGreaterEqual(len(result["passing_comparisons"]), 1)
        for hit in result["passing_comparisons"]:
            if hit["bucket"] == "worsening":
                self.assertGreater(hit["mean_diff_pct"], 0)
            else:
                self.assertLess(hit["mean_diff_pct"], 0)

    def test_wrong_direction_significant_result_does_not_pass(self):
        """A bucket that is significant but in the WRONG direction (worsening
        conditions preceding a NEGATIVE forward return, opposite the
        pre-registered hypothesis) must not count as a pass."""
        rows = []
        for i in range(8):
            rows.append({"bucket": "worsening", "forward_returns": {5: -0.05 - 0.001 * i}})  # wrong sign
        for i in range(2):
            rows.append({"bucket": None, "forward_returns": {5: 0.0}})
        significance = hac_significance(rows, horizons=(5,))
        result = evaluate_pass_bar({"corn": significance}, horizons=(5,))
        self.assertFalse(result["PASSED"])

    def test_no_signal_does_not_pass(self):
        rows = self._rows(0, 0, 10)
        significance = hac_significance(rows, horizons=(5,))
        result = evaluate_pass_bar({"corn": significance}, horizons=(5,))
        self.assertFalse(result["PASSED"])

    def test_bonferroni_n_matches_commodities_horizons_buckets(self):
        self.assertEqual(BONFERRONI_N, 2 * 3 * 2)  # 2 commodities x 3 horizons x 2 buckets


if __name__ == "__main__":
    unittest.main()
