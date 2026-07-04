"""
Regression tests for bot_backtest_subperiods.py — the out-of-sample /
regime-split judgment harness for the Dual-momentum SPY/QQQ candidate
(research/open_questions.md). Pure-function tests only: no network calls,
no dependency on the .bt_cache/ Yahoo data used by main().
"""
import unittest

from bot_backtest_subperiods import split_dates, slice_data, judge, run_subperiod


class TestSplitDates(unittest.TestCase):
    def test_inclusive_boundaries(self):
        dates = ["2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02", "2021-06-01"]
        got = split_dates(dates, "2020-12-31", "2021-01-02")
        self.assertEqual(got, ["2020-12-31", "2021-01-01", "2021-01-02"])

    def test_excludes_outside_range(self):
        dates = ["2019-01-01", "2020-01-01", "2021-01-01"]
        got = split_dates(dates, "2020-01-01", "2020-12-31")
        self.assertEqual(got, ["2020-01-01"])

    def test_empty_when_no_overlap(self):
        dates = ["2016-01-01", "2016-06-01"]
        self.assertEqual(split_dates(dates, "2020-01-01", "2020-12-31"), [])


class TestSliceData(unittest.TestCase):
    def test_restricts_to_given_dates_and_handles_gaps(self):
        data = {
            "SPY": {"2020-01-01": 300.0, "2020-01-02": 301.0, "2020-06-01": 310.0},
            "QQQ": {"2020-01-01": 200.0, "2020-06-01": 210.0},  # missing 2020-01-02
        }
        sliced = slice_data(data, ["2020-01-01", "2020-01-02"])
        self.assertEqual(sliced["SPY"], {"2020-01-01": 300.0, "2020-01-02": 301.0})
        # QQQ has no 2020-01-02 entry — must not be synthesized
        self.assertEqual(sliced["QQQ"], {"2020-01-01": 200.0})


class TestRunSubperiodGuard(unittest.TestCase):
    def test_insufficient_history_returns_empty(self):
        # Fewer than the 252-trading-day momentum lookback + rebalance
        # window needed — must guard rather than crash inside backtest().
        dates = [f"2020-01-{d:02d}" for d in range(1, 29)]
        data = {"SPY": {d: 300.0 + i for i, d in enumerate(dates)},
                "QQQ": {d: 200.0 + i for i, d in enumerate(dates)}}
        self.assertEqual(run_subperiod(data, dates, "2020-01-01", "2020-01-28"), {})


class TestJudge(unittest.TestCase):
    def _r(self, spy_cagr, dual_cagr):
        return {"bench": {"cagr_pct": spy_cagr}, "dual": {"cagr_pct": dual_cagr}}

    def test_kill_when_two_non_covid_subperiods_negative(self):
        results = {
            "2016-2019 (pre-COVID bull)": self._r(14.75, 13.66),   # alpha -1.09 (neg)
            "2020-2021 (COVID crash+recovery, EXCLUDED from kill count)": self._r(22.9, 9.66),  # alpha -13.24, excluded
            "2022-2023 (rate-hike bear+recovery)": self._r(1.33, 20.77),  # alpha +19.44 (pos)
            "2024-2026 (recent)": self._r(17.25, 5.76),            # alpha -11.49 (neg)
        }
        v = judge(results)
        self.assertEqual(v["verdict"], "KILL")
        self.assertEqual(v["negative_subperiods"], 2)
        self.assertEqual(v["counted_subperiods"], 3)

    def test_2020_2021_label_never_counted_even_if_negative(self):
        # All three counted sub-periods positive; only the excluded label negative.
        results = {
            "2016-2019 (pre-COVID bull)": self._r(10.0, 11.0),
            "2020-2021 (COVID crash+recovery, EXCLUDED from kill count)": self._r(30.0, 1.0),
            "2022-2023 (rate-hike bear+recovery)": self._r(5.0, 6.0),
            "2024-2026 (recent)": self._r(10.0, 10.5),
        }
        v = judge(results)
        self.assertEqual(v["negative_subperiods"], 0)
        self.assertEqual(v["verdict"], "SURVIVES")

    def test_survives_with_only_one_negative(self):
        results = {
            "2016-2019 (pre-COVID bull)": self._r(10.0, 9.0),   # negative
            "2022-2023 (rate-hike bear+recovery)": self._r(5.0, 6.0),   # positive
            "2024-2026 (recent)": self._r(10.0, 10.5),          # positive
        }
        v = judge(results)
        self.assertEqual(v["negative_subperiods"], 1)
        self.assertEqual(v["verdict"], "SURVIVES")

    def test_missing_or_empty_subperiod_skipped_not_counted(self):
        results = {
            "2016-2019 (pre-COVID bull)": {},
            "2022-2023 (rate-hike bear+recovery)": self._r(5.0, 4.0),  # negative
            "2024-2026 (recent)": self._r(10.0, 9.0),                  # negative
        }
        v = judge(results)
        self.assertEqual(v["counted_subperiods"], 2)
        self.assertEqual(v["verdict"], "KILL")


if __name__ == "__main__":
    unittest.main()
