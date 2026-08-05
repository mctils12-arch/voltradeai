"""
Regression tests for cftc_tff_gate2_test.py — the ROOT VALIDATION LADDER
gate 2 (SIGNAL) screen for the CFTC TFF archive. Pure-function tests only:
no network calls, no dependency on the live Socrata fetch or backtest_v2's
Alpaca/Yahoo fetch.
"""
import math
import unittest
from unittest.mock import MagicMock, patch

from cftc_tff_gate2_test import (
    _derive_fields,
    _newey_west_diff_test,
    bucket_for,
    compute_forward_returns,
    fetch_symbol_history,
    fetch_symbol_history_range,
    find_entry_index,
    hac_significance,
    summarize,
    validate_record,
)


def _real_record(**overrides):
    """A real TFF row shape (3 YEAR ERIS SOFR SWAP, report_date 2026-07-21 —
    the exact record cftcTff.ts's own tests were verified against
    2026-07-31, see server/cftcTff.test.ts) with every accounting identity
    satisfied by construction, so tests corrupt exactly one field at a time
    rather than starting from an already-wrong fixture."""
    row = {
        "report_date_as_yyyy_mm_dd": "2026-07-21T00:00:00.000",
        "open_interest_all": "78626",
        "dealer_positions_long_all": "10000",
        "dealer_positions_short_all": "9000",
        "dealer_positions_spread_all": "1000",
        "asset_mgr_positions_long": "20000",
        "asset_mgr_positions_short": "15000",
        "asset_mgr_positions_spread": "2000",
        "lev_money_positions_long": "15000",
        "lev_money_positions_short": "20000",
        "lev_money_positions_spread": "3000",
        "other_rept_positions_long": "5616",
        "other_rept_positions_short": "10625",
        "other_rept_positions_spread": "1000",
        "tot_rept_positions_long_all": "57616",  # dealer(10000+1000)+am(20000+2000)+lev(15000+3000)+other(5616+1000)
        "tot_rept_positions_short": "61625",      # dealer(9000+1000)+am(15000+2000)+lev(20000+3000)+other(10625+1000)
        "nonrept_positions_long_all": "21010",   # 78626 - 57616
        "nonrept_positions_short_all": "17001",  # 78626 - 61625
    }
    row.update(overrides)
    return row


class TestValidateRecord(unittest.TestCase):
    def test_real_record_passes(self):
        ok, reason = validate_record(_real_record())
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_zero_open_interest_rejected(self):
        ok, reason = validate_record(_real_record(open_interest_all="0"))
        self.assertFalse(ok)
        self.assertIn("open_interest_all", reason)

    def test_missing_open_interest_rejected(self):
        row = _real_record()
        del row["open_interest_all"]
        ok, _ = validate_record(row)
        self.assertFalse(ok)

    def test_corrupted_long_total_rejected(self):
        ok, reason = validate_record(_real_record(tot_rept_positions_long_all="99999"))
        self.assertFalse(ok)
        self.assertIn("long-side reported total", reason)

    def test_corrupted_short_total_rejected(self):
        ok, reason = validate_record(_real_record(tot_rept_positions_short="99999"))
        self.assertFalse(ok)
        self.assertIn("short-side reported total", reason)

    def test_corrupted_open_interest_denominator_rejected(self):
        # OI changes without the nonrept legs changing to compensate breaks
        # both open-interest identities (long AND short), same coupling
        # cftcTff.test.ts asserts for the TS validator.
        ok, reason = validate_record(_real_record(open_interest_all="99999"))
        self.assertFalse(ok)
        self.assertIn("open interest", reason)

    def test_small_rounding_delta_within_tolerance_passes(self):
        row = _real_record(tot_rept_positions_long_all="57619")  # +3, under _TOLERANCE=5
        ok, _ = validate_record(row)
        self.assertTrue(ok)


class TestDeriveFields(unittest.TestCase):
    def test_net_lev_money_and_pct_oi(self):
        rec = _derive_fields(_real_record())
        self.assertEqual(rec["report_date"], "2026-07-21")
        self.assertEqual(rec["lev_money_long"], 15000)
        self.assertEqual(rec["lev_money_short"], 20000)
        self.assertEqual(rec["net_lev_money"], -5000)
        self.assertAlmostEqual(rec["net_lev_money_pct_oi"], -5000 / 78626 * 100, places=2)

    def test_zero_open_interest_yields_zero_pct_not_a_crash(self):
        rec = _derive_fields(_real_record(open_interest_all="0"))
        self.assertEqual(rec["net_lev_money_pct_oi"], 0.0)


class TestFindEntryIndex(unittest.TestCase):
    def test_finds_first_bar_strictly_after_publish_date(self):
        dates = ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"]
        self.assertEqual(find_entry_index(dates, "2026-01-02"), 2)

    def test_no_bar_after_publish_date_returns_none(self):
        dates = ["2026-01-01", "2026-01-02"]
        self.assertIsNone(find_entry_index(dates, "2026-01-02"))

    def test_never_returns_the_publish_date_itself_no_lookahead(self):
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
        rec = [{"report_date": "2026-01-06", "lev_money_index": 90.0}]
        dates = ["2026-01-06", "2026-01-09", "2026-01-12", "2026-01-13"]
        closes = [100.0, 101.0, 102.0, 103.0]
        rows = compute_forward_returns(rec, self._bars(dates, closes))
        self.assertEqual(rows[0]["entry_date"], "2026-01-12")

    def test_forward_return_arithmetic(self):
        rec = [{"report_date": "2026-01-01", "lev_money_index": 50.0}]
        dates = [f"2026-01-{d:02d}" for d in range(1, 32)] + \
                [f"2026-02-{d:02d}" for d in range(1, 29)]
        closes = [100.0 + i for i in range(len(dates))]
        rows = compute_forward_returns(rec, self._bars(dates, closes))
        entry_idx = dates.index(rows[0]["entry_date"])
        expected_20 = closes[entry_idx + 20] / closes[entry_idx] - 1
        self.assertAlmostEqual(rows[0]["forward_returns"][20], expected_20)

    def test_horizon_beyond_available_bars_is_dropped_not_zero_filled(self):
        rec = [{"report_date": "2026-01-01", "lev_money_index": 50.0}]
        dates = ["2026-01-01", "2026-01-02", "2026-01-05"]
        closes = [100.0, 101.0, 102.0]
        rows = compute_forward_returns(rec, self._bars(dates, closes))
        self.assertNotIn(20, rows[0]["forward_returns"])
        self.assertNotIn(60, rows[0]["forward_returns"])

    def test_no_entry_found_yields_empty_forward_returns(self):
        rec = [{"report_date": "2026-01-01", "lev_money_index": 50.0}]
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
        n = 40
        rows = []
        for t in range(n):
            is_bucket = 15 <= t < 25
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
        self.assertAlmostEqual(naive["mean_diff_pct"], hac["mean_diff_pct"], places=6)
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


class TestFetchSymbolHistoryRange(unittest.TestCase):
    """fetch_symbol_history and fetch_symbol_history_range both route through
    the shared _fetch_and_validate helper (refactored 2026-08-05 to support
    the TLT disjoint-window replication) — these tests pin the ONE thing
    that should differ (the $where clause's date cutoff) and confirm the
    validate/derive/sort pipeline is still shared, not duplicated-and-drifted."""

    def _mock_response(self, rows):
        resp = MagicMock()
        resp.json.return_value = rows
        resp.raise_for_status.return_value = None
        return resp

    @patch("cftc_tff_gate2_test.requests.get")
    def test_fetch_symbol_history_has_no_date_cutoff(self, mock_get):
        mock_get.return_value = self._mock_response([])
        fetch_symbol_history("020601", limit=10)
        where = mock_get.call_args.kwargs["params"]["$where"]
        self.assertEqual(where, "cftc_contract_market_code='020601'")

    @patch("cftc_tff_gate2_test.requests.get")
    def test_fetch_symbol_history_range_adds_exclusive_date_cutoff(self, mock_get):
        mock_get.return_value = self._mock_response([])
        fetch_symbol_history_range("020601", "2023-08-01", limit=156)
        params = mock_get.call_args.kwargs["params"]
        self.assertIn("cftc_contract_market_code='020601'", params["$where"])
        self.assertIn("report_date_as_yyyy_mm_dd < '2023-08-01'", params["$where"])
        self.assertEqual(params["$limit"], 156)

    @patch("cftc_tff_gate2_test.requests.get")
    def test_range_still_validates_and_sorts_ascending(self, mock_get):
        good_early = _real_record(report_date_as_yyyy_mm_dd="2020-09-01T00:00:00.000")
        good_late = _real_record(report_date_as_yyyy_mm_dd="2023-07-25T00:00:00.000")
        bad = _real_record(
            report_date_as_yyyy_mm_dd="2021-01-01T00:00:00.000",
            open_interest_all="0",
        )
        # Socrata returns DESC order; the pipeline must re-sort ascending.
        mock_get.return_value = self._mock_response([good_late, bad, good_early])

        records, rejected = fetch_symbol_history_range("020601", "2023-08-01", limit=156)

        self.assertEqual(rejected, 1)
        self.assertEqual([r["report_date"] for r in records],
                          ["2020-09-01", "2023-07-25"])


if __name__ == "__main__":
    unittest.main()
