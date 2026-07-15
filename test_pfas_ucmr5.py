"""
test_pfas_ucmr5.py — EPA UCMR 5 PFAS occurrence-data battery
(scripts/pfas_ucmr5.py). Pure-function tests on synthetic rows shaped
exactly like the real file (header + cp1252 units column verified live
2026-07-15). Network-touching functions (fetch_centroids/main) are tested
with an injected fake `post`, never a real request.
"""
import importlib.util
import io
import json
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "pfas_ucmr5", os.path.join(os.path.dirname(__file__), "scripts", "pfas_ucmr5.py"))
pfas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pfas)

HEADER = (
    "PWSID\tPWSName\tSize\tFacilityID\tFacilityName\tFacilityWaterType\t"
    "SamplePointID\tSamplePointName\tSamplePointType\tAssociatedFacilityID\t"
    "AssociatedSamplePointID\tCollectionDate\tSampleID\tContaminant\tMRL\t"
    "Units\tMethodID\tAnalyticalResultsSign\tAnalyticalResultValue\t"
    "SampleEventCode\tMonitoringRequirement\tRegion\tState\tUCMR1SampleType\n"
)


def _row(pwsid="010106001", name="Mashantucket Pequot Water System", contaminant="PFOA",
         mrl="0.004", sign="=", value="0.0074", date="9/27/2023", state="CT"):
    return (
        f"{pwsid}\t{name}\tL\t00006\tMPTN WTP\tGU\tTP1\tEntry point to Dist. System\tEP\t\t\t"
        f"{date}\t810-79458-1\t{contaminant}\t{mrl}\tµg/L\tEPA 533\t{sign}\t{value}\t"
        f"SE1\tAM\t1\t{state}\t\n"
    )


def _parse(rows: str):
    return pfas.parse_ucmr5(io.StringIO(HEADER + rows))


def test_header_change_raises_loudly():
    with pytest.raises(ValueError, match="source shape changed"):
        pfas.parse_ucmr5(io.StringIO("TOTALLY\tDIFFERENT\tHEADER\nA\tB\tC\n"))


def test_lithium_excluded_non_detect_excluded_only_detections_aggregated():
    rows = (
        _row(contaminant="lithium", sign="=", value="12.0")  # non-PFAS: dropped entirely
        + _row(contaminant="PFOA", sign="<", value="")        # non-detect: not a detection
        + _row(contaminant="PFOA", sign="=", value="0.0074")  # real detection
    )
    parsed = _parse(rows)
    assert parsed["total_rows"] == 3
    assert parsed["pfas_rows"] == 2          # lithium excluded
    assert parsed["detect_rows"] == 1        # only the '=' PFOA row
    sys = parsed["systems"]["010106001"]
    assert set(sys["analytes"]) == {"PFOA"}
    assert sys["analytes"]["PFOA"]["max_value"] == 0.0074


def test_per_contaminant_aggregation_tracks_max_value_and_date_range():
    rows = (
        _row(contaminant="PFOA", sign="=", value="0.0050", date="1/1/2024")
        + _row(contaminant="PFOA", sign="=", value="0.0090", date="6/1/2024")  # new max
        + _row(contaminant="PFOA", sign="=", value="0.0030", date="3/1/2024")  # neither max nor edge date
    )
    parsed = _parse(rows)
    a = parsed["systems"]["010106001"]["analytes"]["PFOA"]
    assert a["max_value"] == 0.0090
    assert a["first_detected"] == "1/1/2024"
    assert a["last_detected"] == "6/1/2024"
    assert a["n_detections"] == 3


def test_unparsable_value_counted_as_bad_never_silently_dropped():
    parsed = _parse(_row(sign="=", value="garbage"))
    assert parsed["bad_rows"] == 1
    assert parsed["systems"] == {}


def test_short_row_counted_as_bad():
    parsed = _parse(HEADER.rstrip("\n") + "\n" + "010106001\tShort Row\n")
    assert parsed["bad_rows"] == 1


def test_date_key_sorts_chronologically_and_unparsable_sorts_last():
    assert pfas._date_key("1/2/2020") < pfas._date_key("12/1/2020")
    assert pfas._date_key("not-a-date") > pfas._date_key("12/31/2099")


def test_geocode_batches_dedupes_sorts_and_chunks():
    ids = [f"P{i:04d}" for i in range(pfas.GEOCODE_BATCH + 5)] + ["P0000"]  # one duplicate
    batches = list(pfas.geocode_batches(ids))
    assert len(batches) == 2
    assert len(batches[0]) == pfas.GEOCODE_BATCH
    assert sum(len(b) for b in batches) == pfas.GEOCODE_BATCH + 5  # dup collapsed


def test_fetch_centroids_batches_and_merges_skips_unmatched():
    calls = []

    def fake_post(url, form):
        calls.append(form)
        return json.dumps({
            "features": [
                {"attributes": {"PWSID": "A1", "PWS_Name": "Sys A", "Population_Served_Count": 10},
                 "centroid": {"x": -70.0, "y": 40.0}},
                {"attributes": {"PWSID": "A2", "PWS_Name": "Sys B", "Population_Served_Count": None},
                 "centroid": None},  # no centroid: dropped
            ]
        })

    out = pfas.fetch_centroids(["A1", "A2"], post=fake_post)
    assert len(calls) == 1
    assert set(out) == {"A1"}
    assert out["A1"]["lat"] == 40.0 and out["A1"]["lon"] == -70.0


def test_build_artifact_refuses_empty_and_drops_ungeocoded_honestly():
    with pytest.raises(RuntimeError, match="refusing to write"):
        pfas.build_artifact({"systems": {}, "total_rows": 0, "pfas_rows": 0, "detect_rows": 0, "bad_rows": 0}, {}, None, "t")

    parsed = _parse(_row(pwsid="A1", sign="=", value="0.01") + _row(pwsid="A2", sign="=", value="0.02"))
    centroids = {"A1": {"lat": 40.0, "lon": -70.0, "population_served": 100, "boundary_name": "A1"}}
    art = pfas.build_artifact(parsed, centroids, "Thu, 25 Jun 2026 03:50:04 GMT", "2026-07-15T00:00:00Z")
    assert art["system_count"] == 1  # A2 dropped, ungeocoded
    assert art["totals"]["systems_ungeocoded"] == 1
    assert art["systems"][0]["pwsid"] == "A1"
    assert art["predictive"] is False
    assert "NOT an MCL-exceedance" in art["caveat"]


def test_committed_artifact_is_coherent():
    """The repo artifact (built from the real UCMR5 file) satisfies its
    own invariants — positive count, every system geocoded + has a
    detection, no non-PFAS analyte leaked through."""
    fp = os.path.join(os.path.dirname(__file__), "datacore", "pfas", "ucmr5_detections.json")
    art = json.load(open(fp))
    assert art["system_count"] == len(art["systems"]) > 1000
    assert art["totals"]["systems_geocoded"] == art["system_count"]
    seen_contaminants = set()
    for s in art["systems"][:200]:
        assert -90 <= s["lat"] <= 90 and -180 <= s["lon"] <= 180
        assert len(s["detections"]) == s["n_analytes_detected"] > 0
        for d in s["detections"]:
            seen_contaminants.add(d["contaminant"])
    assert "lithium" not in seen_contaminants
