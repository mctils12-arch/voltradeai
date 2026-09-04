"""
Regression tests for scripts/hazard_rate_probe.py. Pure-function tests only,
against synthetic onset/duration series: no network calls, no dependency on
backtest_v2's Alpaca/Yahoo fetch (matches the existing
test_critical_slowing_down_probe.py convention for research probe scripts).
"""
import importlib.util
import math
import os
import unittest

_spec = importlib.util.spec_from_file_location(
    "hazard_rate_probe",
    os.path.join(os.path.dirname(__file__), "scripts", "hazard_rate_probe.py"))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class TestInterOnsetGaps(unittest.TestCase):
    def test_basic_gaps(self):
        self.assertEqual(probe.inter_onset_gaps([10, 25, 40]), [15, 15])

    def test_unsorted_input_is_sorted_first(self):
        self.assertEqual(probe.inter_onset_gaps([40, 10, 25]), [15, 15])

    def test_fewer_than_two_onsets_yields_no_gaps(self):
        self.assertEqual(probe.inter_onset_gaps([]), [])
        self.assertEqual(probe.inter_onset_gaps([5]), [])


class TestGapCv(unittest.TestCase):
    def test_none_below_two_gaps(self):
        self.assertIsNone(probe.gap_cv([]))
        self.assertIsNone(probe.gap_cv([10]))

    def test_none_when_mean_is_zero(self):
        self.assertIsNone(probe.gap_cv([0, 0, 0]))

    def test_constant_gaps_have_zero_cv(self):
        # perfectly regular spacing -> zero spread -> CV = 0, the extreme
        # IFR/wear-out end of the scale
        self.assertAlmostEqual(probe.gap_cv([20, 20, 20, 20]), 0.0, places=9)

    def test_known_cv_value(self):
        gaps = [10, 20, 30]
        m = 20.0
        var = ((10 - m) ** 2 + (20 - m) ** 2 + (30 - m) ** 2) / 3
        expected = math.sqrt(var) / m
        self.assertAlmostEqual(probe.gap_cv(gaps), expected, places=9)

    def test_more_bursty_than_regular_has_higher_cv(self):
        regular = [20, 20, 20, 20, 20]
        bursty = [2, 2, 2, 90, 2]
        self.assertLess(probe.gap_cv(regular), probe.gap_cv(bursty))


class TestBootstrapCvRange(unittest.TestCase):
    def test_none_below_two_gaps(self):
        self.assertIsNone(probe.bootstrap_cv_range([5]))

    def test_reproducible_given_fixed_seed(self):
        gaps = [10, 22, 8, 35, 14, 19]
        a = probe.bootstrap_cv_range(gaps, n_boot=500, rng_seed=42)
        b = probe.bootstrap_cv_range(gaps, n_boot=500, rng_seed=42)
        self.assertEqual(a, b)

    def test_range_brackets_point_estimate_for_regular_gaps(self):
        # near-constant gaps -> CV point estimate is near 0 and the
        # bootstrap range should stay tight and near 0 too, not blow up
        gaps = [20, 21, 19, 20, 20, 19, 21]
        rng = probe.bootstrap_cv_range(gaps, n_boot=1000, rng_seed=7)
        self.assertIsNotNone(rng)
        self.assertLess(rng["hi"], 0.3)
        self.assertGreaterEqual(rng["lo"], 0.0)

    def test_valid_boot_count_matches_request_when_all_valid(self):
        gaps = [10, 22, 8, 35, 14, 19]
        rng = probe.bootstrap_cv_range(gaps, n_boot=300, rng_seed=1)
        self.assertEqual(rng["n_boot_valid"], 300)


class TestDurationSinceLastOnset(unittest.TestCase):
    def test_none_before_first_onset(self):
        out = probe.duration_since_last_onset([5], n=8)
        self.assertEqual(out[:5], [None, None, None, None, None])

    def test_zero_on_onset_day_itself(self):
        out = probe.duration_since_last_onset([5], n=8)
        self.assertEqual(out[5], 0)

    def test_increments_after_onset_and_resets_on_next(self):
        out = probe.duration_since_last_onset([2, 6], n=10)
        # after onset at 2: 0,1,2,3 at indices 2,3,4,5
        self.assertEqual(out[2:6], [0, 1, 2, 3])
        # onset at 6 resets to 0, then climbs again
        self.assertEqual(out[6:10], [0, 1, 2, 3])

    def test_unsorted_onset_input_still_correct(self):
        out = probe.duration_since_last_onset([6, 2], n=10)
        self.assertEqual(out[2:6], [0, 1, 2, 3])
        self.assertEqual(out[6:10], [0, 1, 2, 3])

    def test_empty_onsets_all_none(self):
        out = probe.duration_since_last_onset([], n=5)
        self.assertEqual(out, [None] * 5)


class TestForwardOnsetWithin(unittest.TestCase):
    def test_flags_true_strictly_after_current_day(self):
        out = probe.forward_onset_within([10], n=15, horizon=3)
        # onset at index 10 is visible to indices 7,8,9 (10-7=3,10-8=2,10-9=1)
        self.assertEqual(out[7:10], [True, True, True])
        # not visible further back than the horizon
        self.assertFalse(out[6])
        # not visible at or after the onset itself (strictly-after semantics)
        self.assertFalse(out[10])

    def test_no_onsets_all_false(self):
        out = probe.forward_onset_within([], n=10, horizon=5)
        self.assertEqual(out, [False] * 10)

    def test_horizon_edge_stays_within_series_bounds(self):
        # onset near the very end must not raise or index out of range
        out = probe.forward_onset_within([9], n=10, horizon=5)
        self.assertEqual(len(out), 10)


class TestBucketHazard(unittest.TestCase):
    def test_insufficient_n_flagged_below_min_days(self):
        durations = [0, 1, 2]
        forward = [True, False, True]
        out = probe.bucket_hazard(durations, forward, bucket_edges=(5,),
                                   min_days=10)
        self.assertTrue(out[0]["insufficient_n"])
        self.assertNotIn("hazard", out[0])

    def test_hazard_computed_when_enough_days(self):
        durations = [0] * 20 + [7] * 20
        forward = [True] * 10 + [False] * 10 + [False] * 5 + [True] * 15
        out = probe.bucket_hazard(durations, forward, bucket_edges=(5,),
                                   min_days=10)
        self.assertAlmostEqual(out[0]["hazard"], 0.5, places=9)
        self.assertAlmostEqual(out[1]["hazard"], 0.75, places=9)

    def test_none_durations_excluded(self):
        durations = [None, None, 0, 0, 0]
        forward = [True, True, False, False, False]
        out = probe.bucket_hazard(durations, forward, bucket_edges=(5,),
                                   min_days=2)
        self.assertEqual(out[0]["n_days"], 3)
        self.assertAlmostEqual(out[0]["hazard"], 0.0, places=9)

    def test_bucket_ranges_are_half_open_and_cover_last_as_unbounded(self):
        out = probe.bucket_hazard([0, 5, 10], [True, True, True],
                                   bucket_edges=(5, 10), min_days=1)
        self.assertEqual(out[0]["range"], [0, 5])
        self.assertEqual(out[1]["range"], [5, 10])
        self.assertEqual(out[2]["range"], [10, None])
        self.assertEqual(out[0]["n_days"], 1)  # only 0
        self.assertEqual(out[1]["n_days"], 1)  # only 5
        self.assertEqual(out[2]["n_days"], 1)  # only 10


class TestRunProbeIntegrationShape(unittest.TestCase):
    """run_probe() itself requires network/Alpaca access this sandbox does
    not have (see module docstring) — not exercised here. This test only
    confirms the module imports find_transition_onsets from the CSD probe
    correctly, since run_probe's import wiring is otherwise untested by
    the pure-function tests above."""

    def test_csd_module_loads_and_exposes_find_transition_onsets(self):
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "critical_slowing_down_probe",
            os.path.join(os.path.dirname(__file__), "scripts",
                         "critical_slowing_down_probe.py"))
        csd = ilu.module_from_spec(spec)
        spec.loader.exec_module(csd)
        self.assertTrue(callable(csd.find_transition_onsets))
        onsets = csd.find_transition_onsets(
            ["BULL"] * 25 + ["PANIC"] * 5 + ["BULL"] * 10)
        self.assertEqual(len(onsets), 1)
        self.assertEqual(onsets[0]["index"], 25)

    def test_run_probe_defaults_to_spy_and_accepts_a_ticker_param(self):
        """run_probe's broader-universe follow-up (2026-08-31 GATE 2
        provisional-positive entry) added a `ticker` param without a
        network call — assert the signature accepts it and still defaults
        to "SPY" for backward compatibility, without actually invoking
        run_probe (which needs network/Alpaca access this sandbox may not
        have, per the class docstring)."""
        import inspect
        sig = inspect.signature(probe.run_probe)
        self.assertIn("ticker", sig.parameters)
        self.assertEqual(sig.parameters["ticker"].default, "SPY")


class TestOnsetDates(unittest.TestCase):
    def test_maps_indices_to_dates(self):
        onsets = [{"index": 0}, {"index": 2}]
        dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
        self.assertEqual(probe.onset_dates(onsets, dates),
                          ["2020-01-01", "2020-01-03"])

    def test_out_of_range_index_is_skipped_not_raised(self):
        onsets = [{"index": 5}]
        dates = ["2020-01-01", "2020-01-02"]
        self.assertEqual(probe.onset_dates(onsets, dates), [])

    def test_empty_onsets_yields_empty_dates(self):
        self.assertEqual(probe.onset_dates([], ["2020-01-01"]), [])


class TestIdiosyncraticOnsetDates(unittest.TestCase):
    def test_filters_out_reference_shared_dates(self):
        ticker = ["2020-01-01", "2020-02-01", "2020-03-01"]
        reference = ["2020-01-01", "2020-03-01"]
        self.assertEqual(probe.idiosyncratic_onset_dates(ticker, reference),
                          ["2020-02-01"])

    def test_no_overlap_keeps_everything(self):
        ticker = ["2020-05-01", "2020-06-01"]
        reference = ["2020-01-01"]
        self.assertEqual(probe.idiosyncratic_onset_dates(ticker, reference),
                          ticker)

    def test_full_overlap_yields_empty(self):
        ticker = ["2020-01-01", "2020-02-01"]
        reference = ["2020-01-01", "2020-02-01", "2020-03-01"]
        self.assertEqual(probe.idiosyncratic_onset_dates(ticker, reference), [])

    def test_preserves_input_order_of_ticker_dates(self):
        # deliberately unsorted input -- filtering must not silently sort
        ticker = ["2020-03-01", "2020-01-01", "2020-02-01"]
        reference = ["2020-01-01"]
        self.assertEqual(probe.idiosyncratic_onset_dates(ticker, reference),
                          ["2020-03-01", "2020-02-01"])


class TestPoolIdiosyncraticOnsets(unittest.TestCase):
    def test_pools_and_sorts_across_tickers(self):
        per_ticker = {
            "AAPL": ["2020-03-01"],
            "MSFT": ["2020-01-01", "2020-06-01"],
        }
        out = probe.pool_idiosyncratic_onsets(per_ticker, min_days_apart=0)
        self.assertEqual(out["pooled_dates"],
                          ["2020-01-01", "2020-03-01", "2020-06-01"])
        self.assertEqual(out["pooled_tickers"], ["MSFT", "AAPL", "MSFT"])
        self.assertEqual(out["n_pooled_onsets"], 3)
        self.assertEqual(out["n_dropped_as_correlated_duplicate"], 0)

    def test_gaps_are_calendar_days_not_trading_days(self):
        per_ticker = {"A": ["2020-01-01", "2020-01-08"]}  # exactly one week
        out = probe.pool_idiosyncratic_onsets(per_ticker, min_days_apart=0)
        self.assertEqual(out["gaps_calendar_days"], [7])

    def test_near_duplicate_across_tickers_is_dropped(self):
        # two different tickers going idiosyncratic 2 days apart, under
        # min_days_apart=5 -- treated as one likely-correlated event, the
        # earlier one kept
        per_ticker = {
            "AAPL": ["2020-01-01"],
            "MSFT": ["2020-01-03"],
        }
        out = probe.pool_idiosyncratic_onsets(per_ticker, min_days_apart=5)
        self.assertEqual(out["pooled_dates"], ["2020-01-01"])
        self.assertEqual(out["n_dropped_as_correlated_duplicate"], 1)

    def test_min_days_apart_zero_recovers_full_undeduplicated_pool(self):
        per_ticker = {
            "AAPL": ["2020-01-01"],
            "MSFT": ["2020-01-02"],
        }
        out = probe.pool_idiosyncratic_onsets(per_ticker, min_days_apart=0)
        self.assertEqual(out["n_pooled_onsets"], 2)
        self.assertEqual(out["n_dropped_as_correlated_duplicate"], 0)

    def test_far_apart_dates_from_different_tickers_both_kept(self):
        per_ticker = {
            "AAPL": ["2020-01-01"],
            "MSFT": ["2020-06-01"],
        }
        out = probe.pool_idiosyncratic_onsets(per_ticker, min_days_apart=5)
        self.assertEqual(out["n_pooled_onsets"], 2)
        self.assertEqual(out["n_dropped_as_correlated_duplicate"], 0)

    def test_insufficient_n_flagged_below_min_gaps_floor(self):
        per_ticker = {"A": ["2020-01-01", "2020-02-01"]}
        out = probe.pool_idiosyncratic_onsets(per_ticker, min_days_apart=0)
        self.assertEqual(len(out["gaps_calendar_days"]), 1)
        self.assertTrue(out["gap_cv_insufficient_n"])

    def test_empty_input_yields_empty_pool(self):
        out = probe.pool_idiosyncratic_onsets({}, min_days_apart=5)
        self.assertEqual(out["n_pooled_onsets"], 0)
        self.assertEqual(out["gaps_calendar_days"], [])
        self.assertIsNone(out["gap_cv"])

    def test_single_onset_yields_no_gaps(self):
        out = probe.pool_idiosyncratic_onsets({"A": ["2020-01-01"]}, min_days_apart=5)
        self.assertEqual(out["n_pooled_onsets"], 1)
        self.assertEqual(out["gaps_calendar_days"], [])

    def test_min_days_apart_reported_back_in_output(self):
        out = probe.pool_idiosyncratic_onsets({"A": ["2020-01-01"]}, min_days_apart=7)
        self.assertEqual(out["min_days_apart"], 7)


class TestRunPooledProbeIntegrationShape(unittest.TestCase):
    """run_pooled_probe() requires network/Alpaca access this sandbox does
    not have (see module docstring) — not exercised end-to-end here, same
    precedent as TestRunProbeIntegrationShape above."""

    def test_signature_has_pooling_params_with_documented_defaults(self):
        import inspect
        sig = inspect.signature(probe.run_pooled_probe)
        self.assertEqual(sig.parameters["reference_ticker"].default, "SPY")
        self.assertEqual(sig.parameters["tickers"].default,
                          ("QQQ", "AAPL", "MSFT", "IWM"))
        self.assertEqual(sig.parameters["min_days_apart"].default, 5)

    def test_load_csd_module_helper_is_shared_and_reusable(self):
        """_load_csd_module() was factored out of run_probe so
        run_pooled_probe can reuse it without duplicating the import-by-
        path boilerplate — confirm it still loads a working module."""
        csd = probe._load_csd_module()
        self.assertTrue(callable(csd.find_transition_onsets))


if __name__ == "__main__":
    unittest.main()
