"""
test_cdc_cancer_rates.py — NCI State Cancer Profiles county cancer-rate
battery (scripts/cdc_cancer_rates.py). Pure-function tests on synthetic
CSV text shaped exactly like the real NCI export (title block + header +
data rows + footnote block, live-probed 2026-08-11). Network-touching
`fetch`/`main` are never exercised here — only `parse_csv`/`row_to_record`/
`parse_name`/`gate1_check`/`build_artifact`, all pure.
"""
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "cdc_cancer_rates", os.path.join(os.path.dirname(__file__), "scripts", "cdc_cancer_rates.py"))
cdc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdc)

INCD_HEADER = (
    "Incidence Rate Report for United States by County\n\n"
    '"All Cancer Sites (All Stages^), 2018-2022"\n\n'
    "Sorted by Rate\n\n"
    'County,FIPS,2023 Rural-Urban Continuum Codes([rural urban note]),'
    '"Age-Adjusted Incidence Rate([rate note]) - cases per 100,000",'
    '"Lower 95% Confidence Interval","Upper 95% Confidence Interval",'
    '"CI*Rank([rank note])","Lower CI (CI*Rank)","Upper CI (CI*Rank)",'
    'Average Annual Count,Recent Trend,'
    '"Recent 5-Year Trend ([trend note]) in Incidence Rates",'
    '"Lower 95% Confidence Interval","Upper 95% Confidence Interval"\n'
)
DEATH_HEADER = (
    "Death Rate Report for United States by County\n\n"
    '"All Cancer Sites (All Stages^), 2018-2022"\n\n'
    "Sorted by Rate\n\n"
    'County,FIPS,2023 Rural-Urban Continuum Codes([rural urban note]),'
    'Met Healthy People Objective of 122.7?,'
    '"Age-Adjusted Death Rate([rate note]) - deaths per 100,000",'
    '"Lower CI (Rate)","Upper CI (Rate)","CI*Rank([rank note])",'
    '"Lower CI (CI*Rank)","Upper CI (CI*Rank)",'
    'Average Annual Count,Recent Trend,'
    '"Recent 5-Year Trend ([trend note]) in Death Rates",'
    '"Lower 95% Confidence Interval","Upper 95% Confidence Interval"\n'
)
FOOTNOTE_BLOCK = (
    '\n"* Data has been suppressed to ensure confidentiality."\n'
    '"Data for United States does not include Puerto Rico."\n'
)


def incd_row(name, fips, rate="450.0 ", ci_lo="440.0", ci_hi=" 460.0", count="300", trend="stable"):
    return f'"{name}",{fips},Rural,{rate},{ci_lo},{ci_hi},N/A , N/A , N/A,{count},{trend},0.1,-0.2, 0.4\n'


def death_row(name, fips, rate="146.0 ", ci_lo="140.0", ci_hi=" 152.0", count="100", trend="falling"):
    return f'"{name}",{fips},Rural,No,{rate},{ci_lo},{ci_hi},N/A , N/A , N/A,{count},{trend},-0.5,-0.9, -0.1\n'


def make_incd_csv(rows):
    return INCD_HEADER + "".join(rows) + FOOTNOTE_BLOCK


def make_death_csv(rows):
    return DEATH_HEADER + "".join(rows) + FOOTNOTE_BLOCK


# ── parse_name ───────────────────────────────────────────────────────────

def test_parse_name_strips_footnote_and_splits_state():
    assert cdc.parse_name("Union County, Florida(2)") == ("Union County", "Florida")


def test_parse_name_national_row_has_no_state():
    assert cdc.parse_name("US (SEER+NPCR)(1)") == ("US (SEER+NPCR)", None)
    assert cdc.parse_name("United States") == ("United States", None)


# ── parse_csv ────────────────────────────────────────────────────────────

def test_parse_csv_finds_header_and_stops_at_footnotes():
    text = make_incd_csv([incd_row("Autauga County, Alabama(2)", "01001")])
    rows = cdc.parse_csv(text)
    assert len(rows) == 1
    assert rows[0][1] == "01001"


# ── row_to_record: suppression + column-offset handling ────────────────

def test_row_to_record_incidence_normal():
    rows = cdc.parse_csv(make_incd_csv([incd_row("Autauga County, Alabama(2)", "01001", rate="459.8")]))
    rec = cdc.row_to_record(rows[0], "incidence")
    assert rec["fips"] == "01001"
    assert rec["rate"] == 459.8
    assert rec["suppressed"] is False


def test_row_to_record_suppressed_rate_becomes_null_not_zero():
    rows = cdc.parse_csv(make_incd_csv([incd_row("King County, Texas(7)", "48269", rate="* ", ci_lo="*", ci_hi=" *", count="3 or fewer", trend="*")]))
    rec = cdc.row_to_record(rows[0], "incidence")
    assert rec["rate"] is None
    assert rec["suppressed"] is True
    assert rec["avg_annual_count"] is None
    assert rec["avg_annual_count_note"] == "3 or fewer"
    assert rec["trend"] is None


def test_row_to_record_mortality_column_offset():
    # mortality rows carry one extra column (Healthy People objective)
    # before the rate — a wrong offset would silently read the wrong field.
    rows = cdc.parse_csv(make_death_csv([death_row("Autauga County, Alabama(2)", "01001", rate="153.8")]))
    rec = cdc.row_to_record(rows[0], "mortality")
    assert rec["rate"] == 153.8


# ── gate1_check ──────────────────────────────────────────────────────────

def test_gate1_passes_within_tolerance():
    ref = {"incidence_rate": 446.9, "mortality_rate": 146.0, "as_of": "x", "tolerance_pct": 10.0}
    g = cdc.gate1_check({"rate": 448.6}, {"rate": 145.4}, ref)
    assert g["passed"] is True
    assert g["incidence_pct_diff"] < 1
    assert g["mortality_pct_diff"] < 1


def test_gate1_fails_when_national_row_drifts_beyond_tolerance():
    ref = {"incidence_rate": 446.9, "mortality_rate": 146.0, "as_of": "x", "tolerance_pct": 10.0}
    g = cdc.gate1_check({"rate": 900.0}, {"rate": 145.4}, ref)
    assert g["passed"] is False


def test_gate1_fails_when_national_row_missing_entirely():
    ref = {"incidence_rate": 446.9, "mortality_rate": 146.0, "as_of": "x", "tolerance_pct": 10.0}
    g = cdc.gate1_check(None, None, ref)
    assert g["passed"] is False
    assert g["pulled_national_incidence_rate"] is None


# ── validate_fips ────────────────────────────────────────────────────────

def test_validate_fips():
    assert cdc.validate_fips("01001") is True
    assert cdc.validate_fips("1001") is False
    assert cdc.validate_fips("abcde") is False
    assert cdc.validate_fips("") is False


# ── build_artifact: end-to-end join + quarantine + caveat ───────────────

def _artifact():
    incd_rows = cdc.parse_csv(make_incd_csv([
        incd_row("US (SEER+NPCR)(1)", "00000", rate="448.6"),
        incd_row("Autauga County, Alabama(2)", "01001", rate="459.8"),
        incd_row("Baldwin County, Alabama(2)", "01003", rate="443.9"),
    ]))
    death_rows = cdc.parse_csv(make_death_csv([
        death_row("United States", "00000", rate="145.4"),
        death_row("Autauga County, Alabama(2)", "01001", rate="153.8"),
        death_row("Baldwin County, Alabama(2)", "01003", rate="143.9"),
    ]))
    return cdc.build_artifact(incd_rows, death_rows, "2026-08-11T00:00:00Z")


def test_build_artifact_excludes_national_row_from_counties():
    a = _artifact()
    fips = {c["fips"] for c in a["counties"]}
    assert "00000" not in fips
    assert a["county_count"] == 2


def test_build_artifact_joins_incidence_and_mortality_by_fips():
    a = _artifact()
    autauga = next(c for c in a["counties"] if c["fips"] == "01001")
    assert autauga["county"] == "Autauga County"
    assert autauga["state"] == "Alabama"
    assert autauga["incidence_rate"] == 459.8
    assert autauga["mortality_rate"] == 153.8


def test_build_artifact_gate1_uses_national_row():
    a = _artifact()
    assert a["gate1"]["passed"] is True
    assert a["gate1"]["pulled_national_incidence_rate"] == 448.6


def test_build_artifact_carries_ecological_fallacy_caveat():
    a = _artifact()
    assert "county" in a["caveat"].lower()
    assert "not" in a["caveat"].lower()
    assert a["predictive"] is False


def test_build_artifact_quarantines_bad_fips():
    incd_rows = cdc.parse_csv(make_incd_csv([
        incd_row("US (SEER+NPCR)(1)", "00000", rate="448.6"),
        incd_row("Bad Row County, Nowhere(9)", "9999", rate="400.0"),
    ]))
    death_rows = cdc.parse_csv(make_death_csv([
        death_row("United States", "00000", rate="145.4"),
    ]))
    a = cdc.build_artifact(incd_rows, death_rows, "2026-08-11T00:00:00Z")
    assert a["quarantined_count"] == 1
    assert a["quarantined"][0]["fips"] == "9999"
    assert a["quarantined"][0]["issue"] == "bad_fips_format"
    assert a["county_count"] == 0


def test_build_artifact_county_present_in_only_one_file_still_included():
    # A county that reports in incidence but is missing from the mortality
    # export (e.g. Puerto Rico-only asymmetries) must not silently vanish —
    # it should surface with the other metric honestly null, not dropped.
    incd_rows = cdc.parse_csv(make_incd_csv([
        incd_row("US (SEER+NPCR)(1)", "00000", rate="448.6"),
        incd_row("Solo County, Texas(7)", "48001", rate="400.0"),
    ]))
    death_rows = cdc.parse_csv(make_death_csv([
        death_row("United States", "00000", rate="145.4"),
    ]))
    a = cdc.build_artifact(incd_rows, death_rows, "2026-08-11T00:00:00Z")
    solo = next(c for c in a["counties"] if c["fips"] == "48001")
    assert solo["incidence_rate"] == 400.0
    assert solo["mortality_rate"] is None
    assert solo["mortality_suppressed"] is True
