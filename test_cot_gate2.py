"""
Regression tests for cot_gate2_test.py — the ROOT VALIDATION LADDER gate 2
(SIGNAL) screen for the CFTC COT archive. Pure-function tests only: no
network calls, no dependency on cftc_cot's live Socrata fetch or
backtest_v2's Alpaca/Yahoo fetch.
"""
import math
import unittest

from cot_gate2_test import (
    _newey_west_diff_test,
    bucket_for,
    compute_forward_returns,
    find_entry_index,
    hac_significance,
    summarize,
)


class TestFindEntryIndex(unittest.TestCase):
    def test_finds_first_bar_strictly_after_publish_date(self):
        dates = ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"]
        self.assertEqual(find_entry_index(dates, "2026-01-02"), 2)

    def test_no_bar_after_publish_date_returns_none(self):
        dates = ["2026-01-01", "2026-01-02"]
        self.assertIsNone(find_entry_index(dates, "2026-01-02"))

    def test_never_returns_the_publish_date_itself_no_lookahead(self):
        # A bar dated exactly on the publish date must not be selected as
        # the entry — CFTC positions are published, not yet tradable, that
        # same instant; entry must be strictly after.
        dates = ["2026-01-02", "2026-01-03"]
        self.assertEqual(find_entry_index(dates, "2026-01-02"), 1)


class TestBucketFor(unittest.TestCase):
    def test_high_extreme(self):
        self.assertEqual(bucket_for(85.0), "extreme_high")
        self.assertEqual(bucket_for(80.0), "extreme_high")

    def test_low_extreme(self):
        self.assertEqual(bucket_for(15.0), "extreme_low")
        self.assertEqual(bucket_for(20.0), "extreme_low")

    def test_mid_is_neither(self):
        self.assertEqual(bucket_for(50.0), "mid")

    def test_none_passthrough(self):
        self.assertIsNone(bucket_for(None))


class TestComputeForwardReturns(unittest.TestCase):
    def _bars(self, dates, closes):
        return {"date": dates, "close": closes, "open": closes, "high": closes,
                "low": closes, "volume": [0] * len(closes)}

    def test_entry_is_after_friday_publish_not_the_tuesday_asof(self):
        # report_date Tuesday 2026-01-06 -> publish Friday 2026-01-09 ->
        # entry must be the first trading day AFTER 2026-01-09.
        rec = [{"report_date": "2026-01-06", "cot_index_noncomm": 90.0}]
        dates = ["2026-01-06", "2026-01-09", "2026-01-12", "2026-01-13"]
        closes = [100.0, 101.0, 102.0, 103.0]
        rows = compute_forward_returns(rec, self._bars(dates, closes))
        self.assertEqual(rows[0]["entry_date"], "2026-01-12")

    def test_forward_return_arithmetic(self):
        rec = [{"report_date": "2026-01-01", "cot_index_noncomm": 50.0}]
        dates = [f"2026-01-{d:02d}" for d in range(1, 32)] + \
                [f"2026-02-{d:02d}" for d in range(1, 29)]
        closes = [100.0 + i for i in range(len(dates))]
        rows = compute_forward_returns(rec, self._bars(dates, closes))
        entry_idx = dates.index(rows[0]["entry_date"])
        expected_20 = closes[entry_idx + 20] / closes[entry_idx] - 1
        self.assertAlmostEqual(rows[0]["forward_returns"][20], expected_20)

    def test_horizon_beyond_available_bars_is_dropped_not_zero_filled(self):
        rec = [{"report_date": "2026-01-01", "cot_index_noncomm": 50.0}]
        dates = ["2026-01-01", "2026-01-02", "2026-01-05"]
        closes = [100.0, 101.0, 102.0]
        rows = compute_forward_returns(rec, self._bars(dates, closes))
        self.assertNotIn(20, rows[0]["forward_returns"])
        self.assertNotIn(60, rows[0]["forward_returns"])

    def test_no_entry_found_yields_empty_forward_returns(self):
        rec = [{"report_date": "2026-01-01", "cot_index_noncomm": 50.0}]
        dates = ["2026-01-01"]
        closes = [100.0]
        rows = compute_forward_returns(rec, self._bars(dates, closes))
        self.assertIsNone(rows[0]["entry_date"])
        self.assertEqual(rows[0]["forward_returns"], {})


class TestSummarize(unittest.TestCase):
    def test_baseline_includes_all_buckets_extremes_isolated_separately(self):
        rows = [
            {"bucket": "extreme_high", "forward_returns": {20: 0.10}},
            {"bucket": "extreme_low", "forward_returns": {20: -0.05}},
            {"bucket": "mid", "forward_returns": {20: 0.02}},
        ]
        # Restrict HORIZONS view via monkeypatched summarize call: summarize
        # iterates the module-level HORIZONS constant, so use 20 only here
        # by ensuring 60 is simply absent from every row (dropped, not
        # counted as zero) rather than patching the constant.
        summary = summarize(rows)
        self.assertEqual(summary["20"]["baseline"]["n"], 3)
        self.assertAlmostEqual(summary["20"]["baseline"]["mean_pct"],
                                (0.10 - 0.05 + 0.02) / 3 * 100, places=2)
        self.assertEqual(summary["20"]["extreme_high"]["n"], 1)
        self.assertAlmostEqual(summary["20"]["extreme_high"]["mean_pct"], 10.0)
        self.assertEqual(summary["20"]["extreme_low"]["n"], 1)
        self.assertAlmostEqual(summary["20"]["extreme_low"]["mean_pct"], -5.0)

    def test_missing_horizon_not_counted(self):
        rows = [{"bucket": "mid", "forward_returns": {}}]
        summary = summarize(rows)
        self.assertEqual(summary["20"]["baseline"]["n"], 0)
        self.assertIsNone(summary["20"]["baseline"]["mean_pct"])


class TestNeweyWestDiffTest(unittest.TestCase):
    def test_beta_equals_conditional_mean_difference(self):
        # For a 0/1 dummy regressor, the OLS coefficient on the dummy is
        # ALWAYS exactly the conditional mean difference (basic OLS
        # algebra) — this must hold regardless of the HAC covariance
        # correction, which only affects the reported standard error.
        rows = [
            {"bucket": "extreme_high", "forward_returns": {20: 0.05}},
            {"bucket": "extreme_high", "forward_returns": {20: 0.07}},
            {"bucket": "mid", "forward_returns": {20: 0.01}},
            {"bucket": "extreme_low", "forward_returns": {20: -0.02}},
            {"bucket": "mid", "forward_returns": {20: 0.02}},
            {"bucket": "extreme_high", "forward_returns": {20: 0.03}},
        ]
        result = _newey_west_diff_test(rows, 20, "extreme_high", lag=1)
        self.assertIsNotNone(result)
        bucket_mean = (0.05 + 0.07 + 0.03) / 3
        complement_mean = (0.01 - 0.02 + 0.02) / 3
        # mean_diff_pct is rounded to 3dp by the function, so compare at
        # that precision rather than full float precision.
        self.assertAlmostEqual(result["mean_diff_pct"] / 100,
                                bucket_mean - complement_mean, places=3)
        self.assertEqual(result["n"], 6)
        self.assertEqual(result["lag_weeks"], 1)

    def test_too_few_observations_returns_none_not_a_fake_number(self):
        rows = [
            {"bucket": "extreme_high", "forward_returns": {20: 0.05}},
            {"bucket": "mid", "forward_returns": {20: 0.01}},
        ]
        self.assertIsNone(_newey_west_diff_test(rows, 20, "extreme_high"))

    def test_degenerate_all_bucket_dummy_returns_none(self):
        rows = [{"bucket": "extreme_high", "forward_returns": {20: 0.01 * i}}
                for i in range(10)]
        self.assertIsNone(_newey_west_diff_test(rows, 20, "extreme_high"))

    def test_degenerate_no_bucket_weeks_returns_none(self):
        rows = [{"bucket": "mid", "forward_returns": {20: 0.01 * i}}
                for i in range(10)]
        self.assertIsNone(_newey_west_diff_test(rows, 20, "extreme_high"))

    def test_hac_correction_inflates_se_under_autocorrelated_residuals(self):
        # The whole point of the fix (see open_questions.md's METHODOLOGICAL
        # FINDING): overlapping weekly windows create positively
        # autocorrelated residuals, which the naive (lag=0, White-only)
        # standard error understates. Construct a smoothly-varying
        # (strongly autocorrelated) noise series riding under a real dummy
        # effect and confirm the HAC-adjusted SE at a realistic lag is
        # LARGER than the lag=0 baseline for the identical data — i.e. the
        # correction moves in the direction the methodology finding
        # predicts, not an arbitrary direction.
        n = 40
        rows = []
        for t in range(n):
            # A CONTIGUOUS bucket block (not interleaved) matters here: if
            # bucket membership alternated every few steps, subtracting a
            # different per-group mean at alternating positions would
            # inject its own high-frequency, NEGATIVELY autocorrelated
            # jump into the residual and mask the effect under test.
            is_bucket = 15 <= t < 25
            # Smooth, slowly-varying (period ~42 steps) noise stays
            # strongly POSITIVELY autocorrelated across the lag-1..8
            # window under test.
            noise = math.sin(t * 0.15) * 0.02
            fr = 0.05 + (0.03 if is_bucket else 0.0) + noise
            rows.append({
                "bucket": "extreme_high" if is_bucket else "mid",
                "forward_returns": {10: fr},
            })
        naive = _newey_west_diff_test(rows, 10, "extreme_high", lag=0)
        hac = _newey_west_diff_test(rows, 10, "extreme_high", lag=8)
        self.assertIsNotNone(naive)
        self.assertIsNotNone(hac)
        # Same underlying data -> identical point estimate; only the SE
        # (and therefore t-stat/p-value) should move.
        self.assertAlmostEqual(naive["mean_diff_pct"], hac["mean_diff_pct"],
                                places=6)
        self.assertGreater(hac["hac_se_pct"], naive["hac_se_pct"])
        self.assertLess(abs(hac["t_stat"]), abs(naive["t_stat"]))


class TestHacSignificance(unittest.TestCase):
    def test_covers_every_horizon_and_extreme_bucket(self):
        rows = [
            {"bucket": ["extreme_high", "mid", "extreme_low", "mid"][i % 4],
             "forward_returns": {20: 0.01 * i, 60: -0.01 * i}}
            for i in range(20)
        ]
        sig = hac_significance(rows)
        self.assertEqual(set(sig.keys()), {"20", "60"})
        for h in sig.values():
            self.assertEqual(set(h.keys()), {"extreme_high", "extreme_low"})


if __name__ == "__main__":
    unittest.main()
