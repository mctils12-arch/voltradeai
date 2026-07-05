"""
test_cftc_cot.py
================
Offline, network-free tests for cftc_cot.py (CFTC Commitments of Traders
pipeline, EDGE DOCTRINE #1 free-data build). All fixtures are real field
shapes captured from the live Socrata dataset on 2026-07-04 so the
validation logic is exercised against the actual CFTC schema, including
its own field-name typo (noncomm_postions_spread_all).

Run with: python3 -m pytest test_cftc_cot.py -v
"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cftc_cot


# Real WTI crude row captured 2026-07-04 (report week 2026-06-23) — every
# accounting identity in this record holds exactly.
REAL_WTI_ROW = {
    "market_and_exchange_names": "CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE",
    "report_date_as_yyyy_mm_dd": "2026-06-23T00:00:00.000",
    "open_interest_all": "803067",
    "noncomm_positions_long_all": "90256",
    "noncomm_positions_short_all": "72832",
    "noncomm_postions_spread_all": "215779",
    "comm_positions_long_all": "481852",
    "comm_positions_short_all": "498099",
    "tot_rept_positions_long_all": "787887",
    "tot_rept_positions_short": "786710",
    "nonrept_positions_long_all": "15180",
    "nonrept_positions_short_all": "16357",
}


class TestValidateRecord(unittest.TestCase):
    """GATE 1 (DATA): the record-level accounting-identity check."""

    def test_real_record_passes(self):
        ok, reason = cftc_cot.validate_record(REAL_WTI_ROW)
        self.assertTrue(ok, reason)

    def test_corrupted_long_side_rejected(self):
        bad = dict(REAL_WTI_ROW)
        bad["tot_rept_positions_long_all"] = "999999"
        ok, reason = cftc_cot.validate_record(bad)
        self.assertFalse(ok)
        self.assertIn("long-side reported total", reason)

    def test_corrupted_open_interest_rejected(self):
        bad = dict(REAL_WTI_ROW)
        bad["open_interest_all"] = "1"
        ok, reason = cftc_cot.validate_record(bad)
        self.assertFalse(ok)

    def test_missing_open_interest_rejected(self):
        bad = dict(REAL_WTI_ROW)
        bad["open_interest_all"] = "0"
        ok, reason = cftc_cot.validate_record(bad)
        self.assertFalse(ok)
        self.assertIn("zero", reason)

    def test_small_rounding_tolerance_allowed(self):
        # CFTC occasionally revises with off-by-a-few-contracts artifacts —
        # should not be treated as corruption.
        near = dict(REAL_WTI_ROW)
        near["tot_rept_positions_long_all"] = str(int(REAL_WTI_ROW["tot_rept_positions_long_all"]) + 3)
        ok, reason = cftc_cot.validate_record(near)
        self.assertTrue(ok, reason)


class TestDeriveFields(unittest.TestCase):
    def test_net_positions_and_pct_oi(self):
        derived = cftc_cot._derive_fields(REAL_WTI_ROW)
        self.assertEqual(derived["net_noncomm"], 90256 - 72832)
        self.assertEqual(derived["net_comm"], 481852 - 498099)
        self.assertEqual(derived["report_date"], "2026-06-23")
        self.assertAlmostEqual(derived["net_noncomm_pct_oi"], (90256 - 72832) / 803067 * 100, places=2)


class TestCotIndex(unittest.TestCase):
    def test_index_at_series_high_is_100(self):
        values = [10, 20, 30, 100]
        self.assertEqual(cftc_cot._cot_index(values, lookback=4), 100.0)

    def test_index_at_series_low_is_0(self):
        values = [100, 50, 30, -10]
        self.assertEqual(cftc_cot._cot_index(values, lookback=4), 0.0)

    def test_flat_series_is_50(self):
        values = [42, 42, 42]
        self.assertEqual(cftc_cot._cot_index(values, lookback=3), 50.0)

    def test_lookback_window_respected(self):
        # Last value (5) is the max of the last 2 elements [1, 5] -> 100,
        # even though it's far from the max of the full series (100).
        values = [100, 1, 5]
        self.assertEqual(cftc_cot._cot_index(values, lookback=2), 100.0)

    def test_attach_cot_index_populates_every_record(self):
        records = [
            {"net_noncomm": 10, "net_comm": -5},
            {"net_noncomm": 20, "net_comm": -2},
            {"net_noncomm": -5, "net_comm": 8},
        ]
        out = cftc_cot.attach_cot_index(records, lookback=3)
        for rec in out:
            self.assertIn("cot_index_noncomm", rec)
            self.assertIn("cot_index_comm", rec)
        self.assertEqual(out[-1]["cot_index_noncomm"], 0.0)  # -5 is the min


class TestArchiveRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = os.path.join(os.path.dirname(__file__), "_tmp_test_cot")
        os.makedirs(self.tmpdir, exist_ok=True)
        self.archive_path = os.path.join(self.tmpdir, "archive.json")
        self.checkpoint_path = os.path.join(self.tmpdir, "checkpoint.json")
        self._patchers = [
            mock.patch.object(cftc_cot, "COT_ARCHIVE_PATH", self.archive_path),
            mock.patch.object(cftc_cot, "COT_CHECKPOINT_PATH", self.checkpoint_path),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        for f in (self.archive_path, self.checkpoint_path, self.archive_path + ".tmp"):
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.tmpdir)

    def test_load_archive_missing_file_returns_empty(self):
        self.assertEqual(cftc_cot.load_archive(), {})

    def test_save_then_load_round_trips(self):
        data = {"SPY": [{"report_date": "2026-06-23", "net_noncomm": 100}]}
        cftc_cot.save_archive(data)
        self.assertEqual(cftc_cot.load_archive(), data)

    def test_checkpoint_round_trips(self):
        self.assertEqual(cftc_cot._load_last_check(), 0)
        cftc_cot._save_last_check(1234.5)
        self.assertEqual(cftc_cot._load_last_check(), 1234.5)


class TestRunDailyUpdateGuard(unittest.TestCase):
    """The hourly Tier 3 call site must be near-free on the 23/24 hourly
    invocations that don't coincide with a real staleness check."""

    def setUp(self):
        self.tmpdir = os.path.join(os.path.dirname(__file__), "_tmp_test_cot_guard")
        os.makedirs(self.tmpdir, exist_ok=True)
        self.archive_path = os.path.join(self.tmpdir, "archive.json")
        self.checkpoint_path = os.path.join(self.tmpdir, "checkpoint.json")
        self._patchers = [
            mock.patch.object(cftc_cot, "COT_ARCHIVE_PATH", self.archive_path),
            mock.patch.object(cftc_cot, "COT_CHECKPOINT_PATH", self.checkpoint_path),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        for f in (self.archive_path, self.checkpoint_path, self.archive_path + ".tmp"):
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.tmpdir)

    def test_skips_network_when_checked_recently(self):
        cftc_cot._save_last_check(__import__("time").time())
        with mock.patch.object(cftc_cot, "fetch_symbol_history") as fetch_mock:
            result = cftc_cot.run_daily_update()
        fetch_mock.assert_not_called()
        self.assertEqual(result["status"], "skipped")

    def test_fetches_when_stale(self):
        cftc_cot._save_last_check(0)  # epoch -> definitely stale
        fake_records = [
            {"report_date": "2026-06-23", "net_noncomm": 10, "net_comm": -5,
             "open_interest": 100, "noncomm_long": 20, "noncomm_short": 10,
             "comm_long": 30, "comm_short": 35,
             "net_noncomm_pct_oi": 10.0, "net_comm_pct_oi": -5.0},
        ]
        with mock.patch.object(cftc_cot, "fetch_symbol_history", return_value=(fake_records, 0)) as fetch_mock:
            result = cftc_cot.run_daily_update()
        self.assertEqual(fetch_mock.call_count, len(cftc_cot.SYMBOLS))
        self.assertEqual(result["status"], "done")
        for symbol_result in result["symbols"].values():
            self.assertEqual(symbol_result["status"], "ok")

    def test_network_error_on_one_symbol_does_not_abort_others(self):
        import requests
        cftc_cot._save_last_check(0)
        fake_records = [
            {"report_date": "2026-06-23", "net_noncomm": 1, "net_comm": 1,
             "open_interest": 10, "noncomm_long": 2, "noncomm_short": 1,
             "comm_long": 5, "comm_short": 5,
             "net_noncomm_pct_oi": 10.0, "net_comm_pct_oi": 0.0},
        ]

        def flaky_fetch(code, limit=cftc_cot.LOOKBACK_WEEKS):
            if code == cftc_cot.SYMBOLS["GLD"]["code"]:
                raise requests.RequestException("boom")
            return fake_records, 0

        with mock.patch.object(cftc_cot, "fetch_symbol_history", side_effect=flaky_fetch):
            result = cftc_cot.run_daily_update()
        self.assertEqual(result["symbols"]["GLD"]["status"], "error")
        self.assertEqual(result["symbols"]["SPY"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
