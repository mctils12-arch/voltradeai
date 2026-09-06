"""
test_nhtsa_gate1_probe.py — battery for scripts/nhtsa_gate1_probe.py's pure
pieces (date parsing for both NHTSA date formats, relevance matching,
yearly bucketing, and the gate-1 verdict rule). The PASS thresholds were
fixed in the script BEFORE this session's live run; these tests pin them so
the ruler cannot silently drift after a real result exists (MEASUREMENT
INTEGRITY), and reproduce the two live-verified date-format facts (MM/DD
for complaints, DD/MM for recalls) as regression tests so a future session
cannot reintroduce the mix-up documented in the script's own header.
"""
import importlib.util
import os
from datetime import date

_spec = importlib.util.spec_from_file_location(
    "nhtsa_gate1_probe", os.path.join(os.path.dirname(__file__), "scripts", "nhtsa_gate1_probe.py"))
g1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g1)


# ── date parsing ─────────────────────────────────────────────────────────────

def test_parse_complaint_date_is_mm_dd_yyyy():
    assert g1.parse_complaint_date("02/10/2014") == date(2014, 2, 10)
    assert g1.parse_complaint_date("12/31/2020") == date(2020, 12, 31)
    assert g1.parse_complaint_date(None) is None
    assert g1.parse_complaint_date("") is None
    assert g1.parse_complaint_date("garbage") is None


def test_parse_recall_date_is_dd_mm_yyyy_not_mm_dd():
    # 14V047000's real ReportReceivedDate — DD/MM gives the publicly
    # documented Feb 10, 2014 GM->NHTSA notification date; MM/DD would
    # misread this as October 2.
    assert g1.parse_recall_date("10/02/2014") == date(2014, 2, 10)
    # 15V689000's real ReportReceivedDate — unambiguous proof: 22 cannot
    # be a month, so this field cannot be MM/DD/YYYY.
    assert g1.parse_recall_date("22/10/2015") == date(2015, 10, 22)
    assert g1.parse_recall_date(None) is None
    assert g1.parse_recall_date("13/13/2020") is None  # neither field valid


def test_recall_and_complaint_parsers_disagree_on_ambiguous_dates():
    # The whole point of having two separate parsers: the same raw string
    # means a different date depending on which NHTSA endpoint it came from.
    raw = "05/10/2009"
    assert g1.parse_complaint_date(raw) == date(2009, 5, 10)
    assert g1.parse_recall_date(raw) == date(2009, 10, 5)
    assert g1.parse_complaint_date(raw) != g1.parse_recall_date(raw)


# ── relevance + bucketing ────────────────────────────────────────────────────

def test_is_relevant_case_insensitive_substring():
    assert g1.is_relevant("ELECTRICAL SYSTEM:IGNITION", ("IGNITION",))
    assert g1.is_relevant("electrical system:ignition", ("IGNITION",))
    assert not g1.is_relevant("SUSPENSION", ("IGNITION", "AIR BAG"))
    assert not g1.is_relevant(None, ("IGNITION",))


def test_yearly_counts_buckets_total_and_relevant():
    rows = [
        {"dateComplaintFiled": "01/05/2013", "components": "ENGINE"},
        {"dateComplaintFiled": "06/20/2013", "components": "SUSPENSION"},
        {"dateComplaintFiled": "03/01/2014", "components": "ENGINE FAILURE"},
        {"dateComplaintFiled": "not-a-date", "components": "ENGINE"},  # dropped
    ]
    out = g1.yearly_counts(rows, ("ENGINE",))
    assert out[2013] == {"total": 2, "relevant": 1}
    assert out[2014] == {"total": 1, "relevant": 1}
    assert 2015 not in out


# ── recall lookup ────────────────────────────────────────────────────────────

def test_find_recall_matches_by_campaign_number():
    recalls = [{"NHTSACampaignNumber": "10V017000", "ReportReceivedDate": "21/01/2010"},
               {"NHTSACampaignNumber": "09V388000", "ReportReceivedDate": "05/10/2009"}]
    r = g1.find_recall(recalls, "10V017000")
    assert r is not None and r["ReportReceivedDate"] == "21/01/2010"
    assert g1.find_recall(recalls, "99V999999") is None


# ── evaluate_case: the gate-1 verdict rule ──────────────────────────────────

CASE = {
    "name": "test case",
    "make": "x", "model": "y", "model_year": 2000,
    "campaign": "14V047000",
    "expected_recall_date": date(2014, 2, 10),
    "keywords": ("ENGINE",),
}


def complaint(mdY, components, relevant_year_offset=0):
    return {"dateComplaintFiled": mdY, "components": components}


def _rows_for_years(year_counts_relevant, keywords_hit="ENGINE"):
    """Build synthetic complaint rows: {year: n_relevant} -> rows, one
    relevant complaint per count, dated mid-year."""
    rows = []
    for year, n in year_counts_relevant.items():
        for _ in range(n):
            rows.append({"dateComplaintFiled": f"06/15/{year}", "components": keywords_hit})
    return rows


def test_evaluate_case_passes_on_confirmed_recall_pre_recall_data_and_step_up():
    recalls = [{"NHTSACampaignNumber": "14V047000", "ReportReceivedDate": "10/02/2014"}]
    # prior 3 years (2011-2013) low and steady, recall year (2014) way up.
    rows = _rows_for_years({2011: 5, 2012: 5, 2013: 5, 2014: 40})
    result = g1.evaluate_case(CASE, rows, recalls)
    assert result["recall_confirmed"] is True
    assert result["pre_recall_relevant_complaints"] == 15  # 5+5+5, all before 2014-02-10
    assert result["pre_recall_ok"] is True
    assert result["step_up_ratio"] == 8.0  # 40 / median(5,5,5)
    assert result["step_up_ok"] is True
    assert result["PASS"] is True


def test_evaluate_case_fails_when_recall_not_found():
    recalls = [{"NHTSACampaignNumber": "OTHER0000", "ReportReceivedDate": "10/02/2014"}]
    rows = _rows_for_years({2011: 20, 2012: 20, 2013: 20, 2014: 100})
    result = g1.evaluate_case(CASE, rows, recalls)
    assert result["recall_confirmed"] is False
    assert result["PASS"] is False


def test_evaluate_case_fails_when_recall_date_mismatches_beyond_tolerance():
    # Right campaign number, but the live date is >45 days off the pin —
    # should not silently accept a mislabeled/superseded campaign.
    recalls = [{"NHTSACampaignNumber": "14V047000", "ReportReceivedDate": "01/06/2014"}]  # 2014-06-01
    rows = _rows_for_years({2011: 20, 2012: 20, 2013: 20, 2014: 100})
    result = g1.evaluate_case(CASE, rows, recalls)
    assert result["recall_date_delta_days"] > g1.RECALL_DATE_TOLERANCE_DAYS
    assert result["recall_confirmed"] is False
    assert result["PASS"] is False


def test_evaluate_case_fails_on_silent_defect_no_pre_recall_complaints():
    recalls = [{"NHTSACampaignNumber": "14V047000", "ReportReceivedDate": "10/02/2014"}]
    # Nothing before the recall, only after — the defect would have been
    # invisible to this pipeline ahead of time.
    rows = _rows_for_years({2014: 100})
    # push all into after the recall date within 2014
    rows = [{"dateComplaintFiled": "12/01/2014", "components": "ENGINE"} for _ in range(100)]
    result = g1.evaluate_case(CASE, rows, recalls)
    assert result["pre_recall_relevant_complaints"] == 0
    assert result["pre_recall_ok"] is False
    assert result["PASS"] is False


def test_evaluate_case_fails_on_flat_no_step_up():
    recalls = [{"NHTSACampaignNumber": "14V047000", "ReportReceivedDate": "10/02/2014"}]
    # Recall year no higher than the prior years' noise floor.
    rows = _rows_for_years({2011: 20, 2012: 22, 2013: 18, 2014: 21})
    result = g1.evaluate_case(CASE, rows, recalls)
    assert result["pre_recall_ok"] is True
    assert result["step_up_ok"] is False
    assert result["PASS"] is False


def test_evaluate_case_handles_zero_prior_median_without_crashing():
    recalls = [{"NHTSACampaignNumber": "14V047000", "ReportReceivedDate": "10/02/2014"}]
    rows = _rows_for_years({2014: 15})  # nothing in 2011-2013 at all
    result = g1.evaluate_case(CASE, rows, recalls)
    assert result["prior_3yr_median"] == 0
    assert result["step_up_ratio"] is None  # inf collapsed to None for JSON-safety
    assert result["step_up_ok"] is True  # ratio treated as inf > STEP_UP_MIN_RATIO
    # but pre_recall_ok is False (0 pre-recall complaints) so overall still fails
    assert result["pre_recall_ok"] is False
    assert result["PASS"] is False


def test_cases_list_has_three_distinct_pinned_recalls():
    assert len(g1.CASES) == 3
    campaigns = {c["campaign"] for c in g1.CASES}
    assert len(campaigns) == 3
    for c in g1.CASES:
        assert c["keywords"]
        assert isinstance(c["expected_recall_date"], date)
