"""
Regression tests for cot_gate2_test.py — the ROOT VALIDATION LADDER gate 2
(SIGNAL) screen for the CFTC COT archive. Pure-function tests only: no
network calls, no dependency on cftc_cot's live Socrata fetch or
backtest_v2's Alpaca/Yahoo fetch.
"""
import unittest

from cot_gate2_test import (
    bucket_for,
    compute_forward_returns,
    find_entry_index,
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


if __name__ == "__main__":
    unittest.main()
