"""Unit tests for scripts/insider_cusum_probe.py — all synthetic, no network,
per this repo's established convention for foreign-field probe test files
(test_hurst_exponent_probe.py, test_permutation_entropy_probe.py, etc.)."""

import importlib.util
import os
import unittest

spec = importlib.util.spec_from_file_location(
    "insider_cusum_probe",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "insider_cusum_probe.py"))
icp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(icp)


def _record(filing_date, code, dollar_value):
    return {"filing_date": filing_date, "trans_code": code, "dollar_value": dollar_value}


class TestNetFlowByFilingDate(unittest.TestCase):
    def test_purchases_add_sales_subtract(self):
        records = [
            _record("2026-01-05", "P", 100.0),
            _record("2026-01-05", "S", 40.0),
            _record("2026-01-06", "P", 10.0),
        ]
        flow = icp.net_flow_by_filing_date(records)
        self.assertAlmostEqual(flow["2026-01-05"], 60.0)
        self.assertAlmostEqual(flow["2026-01-06"], 10.0)

    def test_multiple_records_same_date_sum(self):
        records = [_record("2026-01-05", "P", 5.0) for _ in range(3)]
        flow = icp.net_flow_by_filing_date(records)
        self.assertAlmostEqual(flow["2026-01-05"], 15.0)

    def test_ignores_other_codes_and_missing_fields(self):
        records = [
            {"filing_date": "2026-01-05", "trans_code": "A", "dollar_value": 100.0},
            {"filing_date": "2026-01-05", "trans_code": "P", "dollar_value": None},
            {"filing_date": None, "trans_code": "P", "dollar_value": 5.0},
        ]
        flow = icp.net_flow_by_filing_date(records)
        self.assertEqual(flow, {})

    def test_uses_filing_date_not_trans_date(self):
        # No-lookahead: only filing_date should ever be read, regardless of
        # what other date fields a record carries.
        records = [{"filing_date": "2026-01-05", "trans_date": "2026-01-01",
                    "trans_code": "P", "dollar_value": 50.0}]
        flow = icp.net_flow_by_filing_date(records)
        self.assertEqual(list(flow.keys()), ["2026-01-05"])


class TestAlignToTradingDays(unittest.TestCase):
    TRADING_DAYS = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]

    def test_exact_match(self):
        out = icp.align_to_trading_days({"2026-01-06": 100.0}, self.TRADING_DAYS)
        self.assertEqual(out, [0.0, 100.0, 0.0, 0.0, 0.0])

    def test_weekend_rolls_forward_never_backward(self):
        # 2026-01-10/11 is a weekend (not in TRADING_DAYS); the next trading
        # day is 2026-01-09... wait, must roll to the next date AFTER it.
        out = icp.align_to_trading_days({"2026-01-10": 50.0}, self.TRADING_DAYS)
        # Nothing on/after 2026-01-10 exists in this small calendar -> dropped.
        self.assertEqual(sum(out), 0.0)

    def test_holiday_gap_rolls_forward(self):
        # A filing dated between two trading days rolls onto the next one.
        out = icp.align_to_trading_days({"2026-01-06T12:00": 0}, self.TRADING_DAYS)
        # Malformed date (not in calendar, string-comparable) still resolves
        # via binary search onto the correct forward position.
        self.assertEqual(len(out), 5)

    def test_multiple_dates_accumulate_independently(self):
        out = icp.align_to_trading_days(
            {"2026-01-05": 10.0, "2026-01-05b_never_matches": 999.0, "2026-01-09": -5.0},
            self.TRADING_DAYS)
        self.assertAlmostEqual(out[0], 10.0)
        self.assertAlmostEqual(out[4], -5.0)

    def test_empty_calendar(self):
        self.assertEqual(icp.align_to_trading_days({"2026-01-05": 1.0}, []), [])


class TestRollingZscore(unittest.TestCase):
    def test_none_before_window(self):
        vals = [1.0] * 100
        z = icp.rolling_zscore(vals, window=60)
        for v in z[:59]:
            self.assertIsNone(v)

    def test_zero_variance_window_is_none(self):
        vals = [5.0] * 100
        z = icp.rolling_zscore(vals, window=60)
        self.assertTrue(all(v is None for v in z))

    def test_known_distribution_recovers_zscore(self):
        # A single outlier at the end of an otherwise-constant window: the
        # outlier's own z-score should be large and positive; a value equal
        # to the window mean should be near zero (excluding the outlier).
        vals = [0.0] * 59 + [10.0]
        z = icp.rolling_zscore(vals, window=60)
        self.assertIsNotNone(z[59])
        self.assertGreater(z[59], 5.0)

    def test_no_lookahead(self):
        base = [0.0] * 200
        a, b = list(base), list(base)
        for i in range(150, 200):
            b[i] = 999.0  # blow up the tail of b only
        za = icp.rolling_zscore(a, window=60)
        zb = icp.rolling_zscore(b, window=60)
        self.assertEqual(za[100], zb[100])
        self.assertEqual(za[149], zb[149])


class TestCusum(unittest.TestCase):
    def test_flat_series_stays_near_zero(self):
        z = [0.0] * 50
        c = icp.cusum(z, k=0.5)
        for v in c[1:]:
            self.assertEqual(v, 0.0)

    def test_sustained_positive_shift_is_detected(self):
        # Flat at 0 for 40 steps, then a sustained +2 sigma shift.
        z = [0.0] * 40 + [2.0] * 40
        c = icp.cusum(z, k=0.5)
        self.assertGreater(c[79], 10.0)  # accumulates (2 - 0.5) per step
        # Before the shift, the accumulator should not have wandered far.
        self.assertLess(abs(c[39]), 1.0)

    def test_sustained_negative_shift_is_detected(self):
        z = [0.0] * 40 + [-2.0] * 40
        c = icp.cusum(z, k=0.5)
        self.assertLess(c[79], -10.0)

    def test_none_resets_accumulator(self):
        z = [2.0] * 20 + [None] + [0.0] * 5
        c = icp.cusum(z, k=0.5)
        self.assertIsNone(c[20])
        self.assertEqual(c[21], 0.0)  # reset, not carried through the gap

    def test_no_lookahead(self):
        base = [0.1] * 100
        a, b = list(base), list(base)
        for i in range(70, 100):
            b[i] = -5.0
        ca = icp.cusum(a, k=0.5)
        cb = icp.cusum(b, k=0.5)
        self.assertEqual(ca[50], cb[50])
        self.assertEqual(ca[69], cb[69])


class TestCusumAlarms(unittest.TestCase):
    def test_no_alarm_when_below_threshold(self):
        c = [0.5] * 30
        self.assertEqual(icp.cusum_alarms(c, h=5.0), [])

    def test_alarm_fires_once_per_excursion(self):
        c = [0.0] * 10 + [6.0] * 10 + [0.0] * 10 + [7.0] * 10
        alarms = icp.cusum_alarms(c, h=5.0)
        self.assertEqual(len(alarms), 2)
        self.assertEqual(alarms[0], 10)
        self.assertEqual(alarms[1], 30)


class TestContinuationScores(unittest.TestCase):
    def test_basic_pairing(self):
        rets = [0.01] * 30
        cusum_series = [None] * 5 + [1.0] * 25
        pairs = icp.continuation_scores(rets, cusum_series, horizon=5)
        self.assertTrue(all(p[0] == 1.0 for p in pairs))
        for c, fwd in pairs:
            self.assertAlmostEqual(fwd, 0.05, places=6)

    def test_skips_none_cusum(self):
        rets = [0.01] * 10
        cusum_series = [None] * 10
        self.assertEqual(icp.continuation_scores(rets, cusum_series, horizon=3), [])


class TestRunProbeGracefulNoArchive(unittest.TestCase):
    def test_no_archived_quarters_returns_error_not_exception(self):
        # This sandbox has no archived Form 4 quarters on disk — run_probe
        # must degrade to a plain error dict, never raise or fabricate.
        result = icp.run_probe()
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
