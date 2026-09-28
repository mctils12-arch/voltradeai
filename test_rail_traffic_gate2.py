"""
Regression tests for scripts/rail_traffic_gate2.py — the ROOT VALIDATION
LADDER gate 2 (SIGNAL) screen for the STB EP724 rail carload archive. Pure-
function tests only: no network calls, no dependency on backtest_v2's
Alpaca/Yahoo fetch.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from rail_traffic_gate2 import (  # noqa: E402
    bucket_count,
    bucket_for,
    build_basket_bars,
    build_events,
    compute_forward_returns,
    rolling4_growth,
    run,
    run_test,
    system_weekly_intermodal,
    trailing_percentile_index,
    wow_growth,
)


def _bars(dates, closes):
    return {"date": dates, "close": closes, "open": closes, "high": closes,
            "low": closes, "volume": [0] * len(closes)}


class TestSystemWeeklyIntermodal(unittest.TestCase):
    def test_sums_containers_and_trailers_across_railroads(self):
        doc = {
            "weeks": ["2020-01-04", "2020-01-11"],
            "series": {
                "BNSF|Weekly Carloads By 22 Commodity Categories|Containers": [100, 110],
                "BNSF|Weekly Carloads By 22 Commodity Categories|Trailers": [10, 12],
                "UP|Weekly Carloads By 22 Commodity Categories|Containers": [200, 210],
                "BNSF|Weekly Carloads By 22 Commodity Categories|Coal": [999, 999],  # not intermodal
                "BNSF|Cars On Line (Count)|Intermodal": [5000, 5000],  # wrong measure
            },
        }
        out = system_weekly_intermodal(doc)
        self.assertEqual(out, [310.0, 332.0])

    def test_week_with_no_reporting_railroad_is_none_not_zero(self):
        doc = {
            "weeks": ["2020-01-04", "2020-01-11"],
            "series": {
                "BNSF|Weekly Carloads By 22 Commodity Categories|Containers": [100, None],
            },
        }
        out = system_weekly_intermodal(doc)
        self.assertEqual(out[0], 100.0)
        self.assertIsNone(out[1])


class TestRollingGrowth(unittest.TestCase):
    def test_rolling4_growth_needs_seven_prior_weeks(self):
        series = [100.0] * 6
        out = rolling4_growth(series)
        self.assertTrue(all(v is None for v in out))

    def test_rolling4_growth_arithmetic(self):
        series = [100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0, 110.0]
        out = rolling4_growth(series)
        cur_sum = sum(series[4:8])
        prev_sum = sum(series[0:4])
        self.assertAlmostEqual(out[7], math.log(cur_sum / prev_sum))

    def test_rolling4_growth_none_propagates_from_missing_week(self):
        series = [100.0] * 7 + [None]
        out = rolling4_growth(series)
        self.assertIsNone(out[7])

    def test_wow_growth_arithmetic(self):
        series = [100.0, 110.0, None, 90.0]
        out = wow_growth(series)
        self.assertAlmostEqual(out[1], math.log(110.0 / 100.0))
        self.assertIsNone(out[2])
        self.assertIsNone(out[3])  # prior week (index 2) is None


class TestTrailingPercentileIndex(unittest.TestCase):
    def test_none_until_min_window_reached(self):
        values = [float(i) for i in range(10)]
        out = trailing_percentile_index(values, lookback=104, min_window=5)
        self.assertTrue(all(v is None for v in out[:4]))
        self.assertIsNotNone(out[4])

    def test_current_max_scores_100(self):
        values = [1.0, 2.0, 3.0, 4.0, 10.0]
        out = trailing_percentile_index(values, lookback=104, min_window=5)
        self.assertAlmostEqual(out[4], 100.0)

    def test_current_min_scores_0(self):
        values = [5.0, 4.0, 3.0, 2.0, 0.0]
        out = trailing_percentile_index(values, lookback=104, min_window=5)
        self.assertAlmostEqual(out[4], 0.0)

    def test_flat_window_scores_50(self):
        values = [3.0] * 6
        out = trailing_percentile_index(values, lookback=104, min_window=5)
        self.assertAlmostEqual(out[5], 50.0)

    def test_window_respects_lookback_cap(self):
        # a huge early outlier should fall out of the window once lookback
        # weeks have passed, letting a merely-average later value score high.
        values = [1000.0] + [10.0] * 10
        out = trailing_percentile_index(values, lookback=3, min_window=3)
        self.assertAlmostEqual(out[10], 50.0)  # flat window once the outlier ages out

    def test_none_input_stays_none_and_does_not_join_window(self):
        values = [1.0, 2.0, None, 3.0, 4.0]
        out = trailing_percentile_index(values, lookback=104, min_window=3)
        self.assertIsNone(out[2])


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


class TestBuildEvents(unittest.TestCase):
    def test_skips_weeks_without_a_valid_index(self):
        weeks = ["2020-01-04", "2020-01-11", "2020-01-18"]
        growth = [0.1, 0.2, 0.3]
        index = [None, 60.0, 90.0]
        events = build_events(weeks, growth, index, "growth4")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["week"], "2020-01-11")
        self.assertEqual(events[1]["bucket"], "extreme_high")
        self.assertTrue(all(e["measure"] == "growth4" for e in events))


class TestComputeForwardReturns(unittest.TestCase):
    def test_entry_is_after_publish_lag_not_the_week_itself(self):
        events = [{"week": "2026-01-03", "bucket": "extreme_high"}]
        # publish_date = 2026-01-03 + 5 days = 2026-01-08
        dates = ["2026-01-03", "2026-01-08", "2026-01-09", "2026-01-12"]
        closes = [100.0, 101.0, 102.0, 103.0]
        rows = compute_forward_returns(events, _bars(dates, closes))
        self.assertEqual(rows[0]["entry_date"], "2026-01-09")

    def test_forward_return_arithmetic(self):
        events = [{"week": "2026-01-03", "bucket": "extreme_high"}]
        dates = [f"2026-01-{d:02d}" for d in range(3, 32)] + \
                [f"2026-02-{d:02d}" for d in range(1, 20)]
        closes = [100.0 + i for i in range(len(dates))]
        rows = compute_forward_returns(events, _bars(dates, closes))
        entry_idx = dates.index(rows[0]["entry_date"])
        expected_5 = closes[entry_idx + 5] / closes[entry_idx] - 1
        self.assertAlmostEqual(rows[0]["forward_returns"][5], expected_5)

    def test_horizon_beyond_available_bars_dropped_not_zero_filled(self):
        events = [{"week": "2026-01-03", "bucket": "extreme_high"}]
        dates = ["2026-01-03", "2026-01-08", "2026-01-09"]
        closes = [100.0, 101.0, 102.0]
        rows = compute_forward_returns(events, _bars(dates, closes))
        self.assertNotIn(20, rows[0]["forward_returns"])

    def test_no_entry_found_yields_empty_forward_returns(self):
        events = [{"week": "2026-01-03", "bucket": "extreme_high"}]
        dates = ["2026-01-03"]
        closes = [100.0]
        rows = compute_forward_returns(events, _bars(dates, closes))
        self.assertIsNone(rows[0]["entry_date"])
        self.assertEqual(rows[0]["forward_returns"], {})


class TestBucketCountAndRunTest(unittest.TestCase):
    def test_bucket_count_only_counts_rows_with_that_horizon(self):
        rows = [
            {"bucket": "extreme_high", "forward_returns": {5: 0.01}},
            {"bucket": "extreme_high", "forward_returns": {}},
            {"bucket": "mid", "forward_returns": {5: 0.02}},
        ]
        self.assertEqual(bucket_count(rows, 5, "extreme_high"), 1)

    def test_waiting_when_below_min_n(self):
        rows = [{"bucket": "extreme_high", "forward_returns": {5: 0.01}}]
        result = run_test(rows, 5, "extreme_high", min_n=20)
        self.assertEqual(result["verdict"], "WAITING")

    def test_pass_when_significant_and_correctly_signed(self):
        n = 60
        rows = []
        for i in range(n):
            is_bucket = i % 3 == 0
            rows.append({
                "bucket": "extreme_high" if is_bucket else "mid",
                "forward_returns": {5: (0.05 if is_bucket else 0.0) + 0.0001 * math.sin(i)},
            })
        result = run_test(rows, 5, "extreme_high", min_n=10, expected_sign=1)
        self.assertEqual(result["verdict"], "PASS")

    def test_fail_when_wrong_sign(self):
        n = 60
        rows = []
        for i in range(n):
            is_bucket = i % 3 == 0
            rows.append({
                "bucket": "extreme_high" if is_bucket else "mid",
                "forward_returns": {5: (-0.05 if is_bucket else 0.0) + 0.0001 * math.sin(i)},
            })
        result = run_test(rows, 5, "extreme_high", min_n=10, expected_sign=1)
        self.assertEqual(result["verdict"], "FAIL")

    def test_fail_when_not_significant(self):
        n = 60
        rows = [{"bucket": "extreme_high" if i % 3 == 0 else "mid",
                  "forward_returns": {5: 0.001 * math.sin(i * 1.7)}}
                 for i in range(n)]
        result = run_test(rows, 5, "extreme_high", min_n=10)
        self.assertEqual(result["verdict"], "FAIL")


class TestBuildBasketBars(unittest.TestCase):
    def test_equal_weighted_average_return(self):
        all_bars = {
            "A": _bars(["2026-01-01", "2026-01-02"], [100.0, 110.0]),  # +10%
            "B": _bars(["2026-01-01", "2026-01-02"], [100.0, 90.0]),   # -10%
        }
        out = build_basket_bars(all_bars)
        self.assertEqual(out["date"], ["2026-01-01", "2026-01-02"])
        self.assertAlmostEqual(out["close"][0], 100.0)
        self.assertAlmostEqual(out["close"][1], 100.0)  # +10%/-10% average to flat

    def test_inner_join_drops_dates_not_common_to_all(self):
        all_bars = {
            "A": _bars(["2026-01-01", "2026-01-02", "2026-01-03"], [100.0, 101.0, 102.0]),
            "B": _bars(["2026-01-01", "2026-01-03"], [50.0, 51.0]),  # missing 01-02
        }
        out = build_basket_bars(all_bars)
        self.assertEqual(out["date"], ["2026-01-01", "2026-01-03"])

    def test_empty_input_returns_empty_series(self):
        out = build_basket_bars({})
        self.assertEqual(out, {"date": [], "close": []})


class TestRunOverallVerdict(unittest.TestCase):
    def _archive_and_bars(self, n_weeks=170):
        # Synthetic archive: two railroads' Containers series, with a
        # deliberate step up in growth partway through so a real
        # extreme_high bucket exists once the percentile window matures.
        weeks = [f"2020-{1 + (i // 4):02d}-{1 + (i % 4) * 7:02d}" for i in range(n_weeks)]
        # Use a clean, strictly increasing synthetic date axis instead
        # (avoids invalid calendar dates from the placeholder above).
        import datetime as _dt
        start = _dt.date(2018, 1, 6)
        weeks = [(start + _dt.timedelta(weeks=i)).isoformat() for i in range(n_weeks)]
        containers = []
        val = 1000.0
        for i in range(n_weeks):
            # normal ~0.5% weekly noise, with an engineered surprise block
            bump = 1.05 if 140 <= i < 150 else 1.0
            val = val * bump * (1.0 + 0.0005 * math.sin(i))
            containers.append(val)
        doc = {
            "weeks": weeks,
            "series": {
                "BNSF|Weekly Carloads By 22 Commodity Categories|Containers": containers,
            },
        }
        # Price bars: flat except a real positive drift starting exactly
        # where the surprise block's publish-lagged entries would land.
        import datetime as _dt2
        bar_dates = [(start + _dt2.timedelta(days=i)).isoformat() for i in range(700)]
        closes = []
        px = 100.0
        surprise_start_day = (start + _dt2.timedelta(weeks=140) - start).days
        for i in range(700):
            px *= 1.002 if i >= surprise_start_day else 1.0
            closes.append(px)
        bars = _bars(bar_dates, closes)
        return doc, bars

    def test_returns_a_string_verdict_and_covers_primary_key(self):
        doc, bars = self._archive_and_bars()
        result = run(doc, {"IYT": bars, "rail_basket": bars})
        self.assertIn(result["primary_key"], result["results"])
        self.assertIsInstance(result["overall_verdict"], str)
        self.assertTrue(result["results"][result["primary_key"]]["is_primary"])


if __name__ == "__main__":
    unittest.main()
