"""
test_sec_form4_bulk.py
=======================
Offline, network-free tests for sec_form4_bulk.py (SEC bulk-historical
Form 3/4/5 insider-transaction pipeline, EDGE DOCTRINE #1 free-data build).
TSV column layouts and real row values below were captured live 2026-07-19
from https://www.sec.gov/files/structureddata/data/insider-transactions-
data-sets/2025q1_form345.zip (SUBMISSION.tsv / REPORTINGOWNER.tsv /
NONDERIV_TRANS.tsv), including a real single-owner officer/director P-code
buy (Oklo Inc., director John M. Jansen) and a real joint-filing accession
with 2 TenPercentOwner-only reporting owners (correctly excluded — the
gate-2 hypothesis is officer/director specifically).

Run with: python3 -m pytest test_sec_form4_bulk.py -v
"""
import io
import json
import os
import sys
import unittest
import zipfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sec_form4_bulk as f4

SUBMISSION_HEADER = (
    "ACCESSION_NUMBER\tFILING_DATE\tPERIOD_OF_REPORT\tDATE_OF_ORIG_SUB\t"
    "NO_SECURITIES_OWNED\tNOT_SUBJECT_SEC16\tFORM3_HOLDINGS_REPORTED\t"
    "FORM4_TRANS_REPORTED\tDOCUMENT_TYPE\tISSUERCIK\tISSUERNAME\t"
    "ISSUERTRADINGSYMBOL\tREMARKS\tAFF10B5ONE"
)
REPORTINGOWNER_HEADER = (
    "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\t"
    "RPTOWNER_TITLE\tRPTOWNER_TXT\tRPTOWNER_STREET1\tRPTOWNER_STREET2\t"
    "RPTOWNER_CITY\tRPTOWNER_STATE\tRPTOWNER_ZIPCODE\tRPTOWNER_STATE_DESC\t"
    "FILE_NUMBER"
)
NONDERIV_TRANS_HEADER = (
    "ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tSECURITY_TITLE\tSECURITY_TITLE_FN\t"
    "TRANS_DATE\tTRANS_DATE_FN\tDEEMED_EXECUTION_DATE\tDEEMED_EXECUTION_DATE_FN\t"
    "TRANS_FORM_TYPE\tTRANS_CODE\tEQUITY_SWAP_INVOLVED\tEQUITY_SWAP_TRANS_CD_FN\t"
    "TRANS_TIMELINESS\tTRANS_TIMELINESS_FN\tTRANS_SHARES\tTRANS_SHARES_FN\t"
    "TRANS_PRICEPERSHARE\tTRANS_PRICEPERSHARE_FN\tTRANS_ACQUIRED_DISP_CD\t"
    "TRANS_ACQUIRED_DISP_CD_FN\tSHRS_OWND_FOLWNG_TRANS\tSHRS_OWND_FOLWNG_TRANS_FN\t"
    "VALU_OWND_FOLWNG_TRANS\tVALU_OWND_FOLWNG_TRANS_FN\tDIRECT_INDIRECT_OWNERSHIP\t"
    "DIRECT_INDIRECT_OWNERSHIP_FN\tNATURE_OF_OWNERSHIP\tNATURE_OF_OWNERSHIP_FN"
)


def _make_quarter_zip(submission_rows, owner_rows, trans_rows) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SUBMISSION.tsv", SUBMISSION_HEADER + "\n" + "\n".join(submission_rows))
        zf.writestr("REPORTINGOWNER.tsv", REPORTINGOWNER_HEADER + "\n" + "\n".join(owner_rows))
        zf.writestr("NONDERIV_TRANS.tsv", NONDERIV_TRANS_HEADER + "\n" + "\n".join(trans_rows))
    return buf.getvalue()


# Real rows, 2025q1_form345.zip, captured 2026-07-19.
OKLO_ACC = "0001104659-25-030072"
OKLO_SUBMISSION = (
    f"{OKLO_ACC}\t31-MAR-2025\t27-MAR-2025\t\t\t\t\t\t4\t0001849056\t"
    "Oklo Inc.\tOKLO\t\t0"
)
OKLO_OWNER = (
    f"{OKLO_ACC}\t0002021774\tJansen John M\tDirector\t\t\t\t\t\t\t\t\t"
)
OKLO_TRANS = (
    f"{OKLO_ACC}\t1\tCommon Stock\t\t27-MAR-2025\t\t\t\t4\tP\t0\t\t\t\t"
    "6000.0\t\t24.57\t\tA\t\t\t\t\t\tD\t\t\t"
)

JOINT_ACC = "0002003074-25-000007"
JOINT_SUBMISSION = (
    f"{JOINT_ACC}\t01-APR-2025\t31-MAR-2025\t\t\t\t\t\t4\t0001900000\t"
    "Some Fund Target Inc.\tSFTI\t\t0"
)
JOINT_OWNER_FUND = f"{JOINT_ACC}\t0001111111\tSYLEBRA CAPITAL LLC\tTenPercentOwner\t\t\t\t\t\t\t\t\t"
JOINT_OWNER_PERSON = f"{JOINT_ACC}\t0001111112\tGibson Daniel Patrick\tTenPercentOwner\t\t\t\t\t\t\t\t\t"
JOINT_TRANS = (
    f"{JOINT_ACC}\t2\tCommon Stock\t\t31-MAR-2025\t\t\t\t4\tP\t0\t\t\t\t"
    "1000.0\t\t10.0\t\tA\t\t\t\t\t\tI\t\t\t"
)

# Same issuer as Oklo but a code-A grant (not discretionary) — must be excluded.
GRANT_ACC = "0001104659-25-030999"
GRANT_SUBMISSION = f"{GRANT_ACC}\t02-APR-2025\t01-APR-2025\t\t\t\t\t\t4\t0001849056\tOklo Inc.\tOKLO\t\t0"
GRANT_OWNER = f"{GRANT_ACC}\t0002021774\tJansen John M\tDirector\t\t\t\t\t\t\t\t\t"
GRANT_TRANS = (
    f"{GRANT_ACC}\t3\tCommon Stock\t\t01-APR-2025\t\t\t\t4\tA\t0\t\t\t\t"
    "500.0\t\t0.0\t\tA\t\t\t\t\t\tD\t\t\t"
)


class TestToIsoDate(unittest.TestCase):
    def test_parses_real_format(self):
        self.assertEqual(f4._to_iso_date("31-MAR-2025"), "2025-03-31")

    def test_single_digit_day_zero_padded(self):
        self.assertEqual(f4._to_iso_date("1-JAN-2020"), "2020-01-01")

    def test_none_input_returns_none(self):
        self.assertIsNone(f4._to_iso_date(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(f4._to_iso_date(""))

    def test_garbage_returns_none_not_raise(self):
        self.assertIsNone(f4._to_iso_date("not-a-date"))


class TestListAvailableQuarters(unittest.TestCase):
    def test_parses_both_url_base_generations(self):
        html = (
            '<a href="/files/datastandardsinnovation/data/insider-transactions-data-sets/2026q2_form345.zip">2026 Q2</a>'
            '<a href="/files/structureddata/data/insider-transactions-data-sets/2026q1_form345.zip">2026 Q1</a>'
            '<a href="/files/structureddata/data/insider-transactions-data-sets/2018q4_form345.zip">2018 Q4</a>'
        )
        fake_resp = mock.Mock(text=html, status_code=200)
        fake_resp.raise_for_status = mock.Mock()
        with mock.patch.object(f4.requests, "get", return_value=fake_resp) as get_mock:
            quarters = f4.list_available_quarters()
        get_mock.assert_called_once()
        self.assertEqual(
            quarters["2026q2"],
            "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/2026q2_form345.zip",
        )
        self.assertEqual(
            quarters["2026q1"],
            "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/2026q1_form345.zip",
        )
        self.assertEqual(len(quarters), 3)


class TestFetchQuarter(unittest.TestCase):
    """Real captured fixture rows — the join, role filter, and code filter
    all exercised against actual SEC field shapes, not guessed ones."""

    def _fetch(self, submission_rows, owner_rows, trans_rows):
        zip_bytes = _make_quarter_zip(submission_rows, owner_rows, trans_rows)
        fake_resp = mock.Mock(content=zip_bytes, status_code=200)
        fake_resp.raise_for_status = mock.Mock()
        with mock.patch.object(f4.requests, "get", return_value=fake_resp):
            return f4.fetch_quarter("https://example.com/fake.zip")

    def test_real_officer_director_buy_is_captured(self):
        out = self._fetch([OKLO_SUBMISSION], [OKLO_OWNER], [OKLO_TRANS])
        self.assertEqual(len(out), 1)
        rec = out[0]
        self.assertEqual(rec["ticker"], "OKLO")
        self.assertEqual(rec["trans_code"], "P")
        self.assertEqual(rec["shares"], 6000.0)
        self.assertEqual(rec["price"], 24.57)
        self.assertAlmostEqual(rec["dollar_value"], 6000.0 * 24.57, places=2)
        self.assertTrue(rec["is_director"])
        self.assertFalse(rec["is_officer"])
        self.assertEqual(rec["filing_date"], "2025-03-31")
        self.assertEqual(rec["owner_name"], "Jansen John M")

    def test_ten_percent_owner_only_is_excluded(self):
        # Real joint-filing accession where BOTH reporting owners are
        # TenPercentOwner-only (a fund + its principal) — the gate-2
        # hypothesis explicitly targets officer/director, not fund flows.
        out = self._fetch(
            [JOINT_SUBMISSION], [JOINT_OWNER_FUND, JOINT_OWNER_PERSON], [JOINT_TRANS]
        )
        self.assertEqual(out, [])

    def test_code_a_grant_is_excluded(self):
        out = self._fetch([GRANT_SUBMISSION], [GRANT_OWNER], [GRANT_TRANS])
        self.assertEqual(out, [])

    def test_joint_filing_with_one_qualifying_owner_emits_one_record(self):
        # 2 owners on the accession, only one is officer/director-qualifying.
        mixed_owner_1 = f"{JOINT_ACC}\t0001111111\tSYLEBRA CAPITAL LLC\tTenPercentOwner\t\t\t\t\t\t\t\t\t"
        mixed_owner_2 = f"{JOINT_ACC}\t0009999999\tReal Officer Person\tOfficer\t\t\t\t\t\t\t\t\t"
        out = self._fetch(
            [JOINT_SUBMISSION], [mixed_owner_1, mixed_owner_2], [JOINT_TRANS]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["owner_name"], "Real Officer Person")
        self.assertTrue(out[0]["is_officer"])

    def test_missing_ticker_is_excluded(self):
        sub_no_ticker = f"{OKLO_ACC}\t31-MAR-2025\t27-MAR-2025\t\t\t\t\t\t4\t0001849056\tOklo Inc.\t\t\t0"
        out = self._fetch([sub_no_ticker], [OKLO_OWNER], [OKLO_TRANS])
        self.assertEqual(out, [])

    def test_malformed_ticker_values_are_excluded(self):
        # Real garbage values captured live 2026-07-19 from SUBMISSION.tsv's
        # free-text ISSUERTRADINGSYMBOL field.
        for bad in ("N/A", "NONE", "-", "NASDAQ:SVC", "GEF, GEF-B", "JUSH/JUSHF", "1314152"):
            sub = f"{OKLO_ACC}\t31-MAR-2025\t27-MAR-2025\t\t\t\t\t\t4\t0001849056\tOklo Inc.\t{bad}\t\t0"
            out = self._fetch([sub], [OKLO_OWNER], [OKLO_TRANS])
            self.assertEqual(out, [], f"expected {bad!r} to be rejected")

    def test_dotted_class_ticker_is_accepted(self):
        sub = f"{OKLO_ACC}\t31-MAR-2025\t27-MAR-2025\t\t\t\t\t\t4\t0001849056\tOklo Inc.\tBRK.B\t\t0"
        out = self._fetch([sub], [OKLO_OWNER], [OKLO_TRANS])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["ticker"], "BRK.B")

    def test_zero_shares_excluded(self):
        zero_trans = (
            f"{OKLO_ACC}\t1\tCommon Stock\t\t27-MAR-2025\t\t\t\t4\tP\t0\t\t\t\t"
            "0.0\t\t24.57\t\tA\t\t\t\t\t\tD\t\t\t"
        )
        out = self._fetch([OKLO_SUBMISSION], [OKLO_OWNER], [zero_trans])
        self.assertEqual(out, [])


class TestPickQuartersToFetch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = os.path.join(os.path.dirname(__file__), "_tmp_test_f4bulk_pick")
        os.makedirs(self.tmpdir, exist_ok=True)
        self._patcher = mock.patch.object(f4, "FORM4_BULK_DIR", self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def _available(self, n=10):
        # 10 consecutive closed quarters ending at 2026q1
        keys = ["2023q3", "2023q4", "2024q1", "2024q2", "2024q3", "2024q4",
                "2025q1", "2025q2", "2025q3", "2025q4", "2026q1"]
        return {k: f"https://example.com/{k}.zip" for k in keys[-n:]}

    def test_fresh_backfill_picks_oldest_missing_first(self):
        available = self._available()
        target = f4.pick_quarters_to_fetch(available)
        window = sorted(available.keys())[-f4.BACKFILL_QUARTERS:]
        self.assertEqual(target, [window[0]])

    def test_partial_backfill_continues_from_oldest_missing(self):
        available = self._available()
        window = sorted(available.keys())[-f4.BACKFILL_QUARTERS:]
        f4.archive_quarter(window[0], [])
        target = f4.pick_quarters_to_fetch(available)
        self.assertEqual(target, [window[1]])

    def test_fully_backfilled_checks_for_new_quarter(self):
        available = self._available()
        window = sorted(available.keys())[-f4.BACKFILL_QUARTERS:]
        for q in window:
            f4.archive_quarter(q, [])
        target = f4.pick_quarters_to_fetch(available)
        self.assertEqual(target, [])

    def test_new_quarter_appears_after_full_backfill(self):
        available = self._available()
        window = sorted(available.keys())[-f4.BACKFILL_QUARTERS:]
        for q in window:
            f4.archive_quarter(q, [])
        available["2026q2"] = "https://example.com/2026q2.zip"
        target = f4.pick_quarters_to_fetch(available)
        self.assertEqual(target, ["2026q2"])


class TestArchiveRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = os.path.join(os.path.dirname(__file__), "_tmp_test_f4bulk_archive")
        os.makedirs(self.tmpdir, exist_ok=True)
        self._patcher = mock.patch.object(f4, "FORM4_BULK_DIR", self.tmpdir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_archive_then_load_round_trips(self):
        records = [{"ticker": "OKLO", "filing_date": "2025-03-31"}]
        f4.archive_quarter("2025q1", records)
        self.assertTrue(f4.is_archived("2025q1"))
        self.assertEqual(f4.load_quarter("2025q1"), records)

    def test_load_missing_quarter_returns_empty(self):
        self.assertEqual(f4.load_quarter("1999q1"), [])
        self.assertFalse(f4.is_archived("1999q1"))

    def test_archived_quarters_sorted(self):
        f4.archive_quarter("2025q2", [])
        f4.archive_quarter("2025q1", [])
        self.assertEqual(f4.archived_quarters(), ["2025q1", "2025q2"])

    def test_load_all_records_concatenates_sorted_by_filing_date(self):
        f4.archive_quarter("2025q1", [{"ticker": "B", "filing_date": "2025-02-01"}])
        f4.archive_quarter("2025q2", [{"ticker": "A", "filing_date": "2025-01-01"}])
        combined = f4.load_all_records()
        self.assertEqual([r["ticker"] for r in combined], ["A", "B"])


class TestRunUpdateGuard(unittest.TestCase):
    def setUp(self):
        self.tmpdir = os.path.join(os.path.dirname(__file__), "_tmp_test_f4bulk_guard")
        os.makedirs(self.tmpdir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.tmpdir, "checkpoint.json")
        self._patchers = [
            mock.patch.object(f4, "FORM4_BULK_DIR", self.tmpdir),
            mock.patch.object(f4, "FORM4_BULK_CHECKPOINT_PATH", self.checkpoint_path),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        for f in os.listdir(self.tmpdir):
            os.remove(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_skips_network_when_checked_recently(self):
        f4._save_last_check(__import__("time").time())
        with mock.patch.object(f4, "list_available_quarters") as list_mock:
            result = f4.run_update()
        list_mock.assert_not_called()
        self.assertEqual(result["status"], "skipped")

    def test_fetches_one_quarter_when_stale(self):
        f4._save_last_check(0)
        with mock.patch.object(f4, "list_available_quarters", return_value={"2025q1": "https://x/2025q1.zip"}), \
             mock.patch.object(f4, "fetch_quarter", return_value=[{"ticker": "OKLO"}]) as fetch_mock:
            result = f4.run_update()
        fetch_mock.assert_called_once()
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["quarters_checked"], ["2025q1"])
        self.assertTrue(f4.is_archived("2025q1"))

    def test_index_fetch_failure_does_not_raise(self):
        import requests
        f4._save_last_check(0)
        with mock.patch.object(f4, "list_available_quarters", side_effect=requests.RequestException("boom")):
            result = f4.run_update()
        self.assertEqual(result["status"], "error")

    def test_up_to_date_when_nothing_to_fetch(self):
        f4._save_last_check(0)
        with mock.patch.object(f4, "list_available_quarters", return_value={}):
            result = f4.run_update()
        self.assertEqual(result["status"], "up_to_date")
        self.assertEqual(result["quarters_checked"], [])


if __name__ == "__main__":
    unittest.main()
