"""
Regression tests for scripts/contagion_reproduction_probe.py. Pure-function
tests only, against synthetic series: no network calls, no dependency on
backtest_v2's Alpaca/Yahoo fetch or bot_engine's SECTOR_MAP (matches the
existing test_critical_slowing_down_probe.py / test_illiquid_universe_probe.py
convention for research probe scripts).
"""
import importlib.util
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "contagion_reproduction_probe",
    os.path.join(os.path.dirname(__file__), "scripts",
                 "contagion_reproduction_probe.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class TestNewLowFlags(unittest.TestCase):
    def test_none_before_window_fills(self):
        closes = [10.0, 9.0, 8.0]
        out = probe.new_low_flags(closes, window=5)
        self.assertTrue(all(v is None for v in out))

    def test_new_low_flagged_true(self):
        closes = [10.0, 9.0, 8.0, 7.0, 6.0]
        out = probe.new_low_flags(closes, window=5)
        self.assertEqual(out[-1], 1)

    def test_bounce_not_flagged(self):
        closes = [10.0, 9.0, 8.0, 7.0, 9.5]
        out = probe.new_low_flags(closes, window=5)
        self.assertEqual(out[-1], 0)

    def test_flat_series_flags_every_day_as_a_tie_low(self):
        closes = [5.0] * 6
        out = probe.new_low_flags(closes, window=5)
        self.assertEqual(out[-1], 1)


class TestAlignCommonDates(unittest.TestCase):
    def test_intersects_and_reindexes(self):
        bars = {
            "A": {"date": ["d1", "d2", "d3"], "close": [1.0, 2.0, 3.0]},
            "B": {"date": ["d2", "d3", "d4"], "close": [20.0, 30.0, 40.0]},
        }
        common, aligned = probe.align_common_dates(bars)
        self.assertEqual(common, ["d2", "d3"])
        self.assertEqual(aligned["A"], [2.0, 3.0])
        self.assertEqual(aligned["B"], [20.0, 30.0])

    def test_empty_input(self):
        common, aligned = probe.align_common_dates({})
        self.assertEqual(common, [])
        self.assertEqual(aligned, {})


class TestCrossSectionalEventCounts(unittest.TestCase):
    def test_sums_across_tickers(self):
        aligned = {
            "A": [10.0, 9.0, 8.0, 7.0, 6.0],   # new low on last day -> 1
            "B": [10.0, 9.0, 8.0, 7.0, 9.0],   # bounce -> 0
        }
        out = probe.cross_sectional_event_counts(aligned, window=5)
        self.assertEqual(out[-1], 1)

    def test_none_when_any_ticker_undefined(self):
        # both tickers pre-aligned to the same 3-date grid (as
        # align_common_dates would produce), but window=5 needs 5 days of
        # history -> every date is still undefined for both.
        aligned = {
            "A": [10.0, 9.0, 8.0],
            "B": [10.0, 9.0, 7.0],
        }
        out = probe.cross_sectional_event_counts(aligned, window=5)
        self.assertTrue(all(v is None for v in out))

    def test_empty_input(self):
        self.assertEqual(probe.cross_sectional_event_counts({}, window=5), [])


class TestWindowedReproductionNumber(unittest.TestCase):
    def test_none_before_two_windows_available(self):
        counts = [1, 0, 1, 0, 1, 0, 1, 0, 1]
        out = probe.windowed_reproduction_number(counts, tau=5)
        self.assertTrue(all(v is None for v in out[:-1]))

    def test_growth_ratio_above_one_when_accelerating(self):
        # prior 5-day window sums to 2, current 5-day window sums to 8
        counts = [0, 1, 0, 1, 0, 2, 2, 2, 1, 1]
        out = probe.windowed_reproduction_number(counts, tau=5)
        self.assertAlmostEqual(out[-1], 8 / 2)

    def test_zero_prior_sum_is_none_not_infinity(self):
        counts = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        out = probe.windowed_reproduction_number(counts, tau=5)
        self.assertIsNone(out[-1])

    def test_none_propagates_through_undefined_counts(self):
        counts = [None, 1, 0, 1, 0, 2, 2, 2, 1, 1]
        out = probe.windowed_reproduction_number(counts, tau=5)
        self.assertIsNone(out[-1])


class TestForwardReturnPct(unittest.TestCase):
    def test_basic(self):
        closes = [100.0, 105.0, 110.0]
        self.assertAlmostEqual(probe.forward_return_pct(closes, 0, 2), 10.0)

    def test_out_of_range_is_none(self):
        closes = [100.0, 105.0]
        self.assertIsNone(probe.forward_return_pct(closes, 0, 5))

    def test_non_positive_base_is_none(self):
        closes = [0.0, 105.0, 110.0]
        self.assertIsNone(probe.forward_return_pct(closes, 0, 1))


class TestComputeLeadSignal(unittest.TestCase):
    def test_insufficient_n_reported_honestly(self):
        n = 20
        event_counts = [1 if i % 3 == 0 else 0 for i in range(n)]
        rt_series = [None] * n
        target_closes = [100.0 + i for i in range(n)]
        result = probe.compute_lead_signal(event_counts, rt_series,
                                            target_closes, horizons=(5,))
        self.assertTrue(result["h5"]["rt"]["insufficient_n"])

    def test_engineered_signal_detected_above_floor(self):
        # Deterministic synthetic archive, well above MIN_N_FOR_STATS: build
        # a target series where the next day's return is an exact linear,
        # decreasing function of R_t (higher R_t -> more negative next-day
        # return) at horizon=1, so overlapping multi-day windows can't blur
        # the relationship — a real rank correlation must appear (a smoke
        # test for wiring, not a claim about markets).
        n = 300
        event_counts = [(i % 7) for i in range(n)]
        rt_series = [1.0 + 0.1 * (i % 7) for i in range(n)]
        target_closes = [100.0]
        for i in range(n - 1):
            step = -0.05 * rt_series[i]
            target_closes.append(target_closes[-1] * (1 + step / 100.0))
        result = probe.compute_lead_signal(event_counts, rt_series,
                                            target_closes, horizons=(1,))
        cell = result["h1"]["rt"]
        self.assertNotIn("insufficient_n", cell)
        self.assertLess(cell["rho"], 0)  # higher R_t -> lower forward return


if __name__ == "__main__":
    unittest.main()
