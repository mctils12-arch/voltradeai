"""
test_pfas_ucmr5.py — UCMR5 PFAS occurrence pipeline battery
(scripts/pfas_ucmr5.py). Pure-function tests on synthetic rows shaped
exactly like the real UCMR5_All.txt (tab-delimited, verified header
2026-07-13). No network — geocode_pwsids (the ECHO two-hop join) is
exercised live only by the script itself, not here.
"""
import importlib.util
import io
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


def _row(pwsid="AL0000983", name="GRAND BAY WATER WORKS", contaminant="PFOA",
         sign="=", value="0.0314", mrl="0.004", date="8/12/2024",
         state="AL", region="4"):
    return (
        f"{pwsid}\t{name}\tL\t00001\tPlant\tGU\tEP1\tEntry point\tEP\t\t\t"
        f"{date}\t810-1\t{contaminant}\t{mrl}\tµg/L\tEPA 533\t{sign}\t{value}\t"
        f"SE1\tAM\t{region}\t{state}\t01\n"
    )


def _parse(rows: str):
    return pfas.parse_detections(io.StringIO(HEADER + rows))


def test_keeps_only_pfas_detections_above_mrl():
    rows = (
        _row(contaminant="PFOA", sign="=", value="0.0314")
        + _row(contaminant="PFOS", sign="<", value="")          # non-detect: excluded
        + _row(contaminant="lithium", sign="=", value="12.0")   # not PFAS: excluded
        + _row(pwsid="TX0000001", contaminant="PFHxS", sign="=", value="0.009", state="TX", region="6")
    )
    parsed = _parse(rows)
    assert set(parsed["by_pws"]) == {"AL0000983", "TX0000001"}
    assert parsed["monitored_pws"] == 2  # AL (PFOA detect + PFOS non-detect, one PWS) + TX
    al = parsed["by_pws"]["AL0000983"]
    assert al["state"] == "AL" and al["region"] == "4"
    assert len(al["dets"]) == 1
    assert al["dets"][0] == {"contaminant": "PFOA", "result": 0.0314, "mrl": 0.004,
                              "units": "µg/L", "date": "8/12/2024"}


def test_multiple_detections_per_system_all_kept():
    rows = (
        _row(contaminant="PFOA", value="0.03", date="8/12/2024")
        + _row(contaminant="PFOS", value="0.02", date="8/12/2024")
        + _row(contaminant="PFOA", value="0.028", date="4/8/2025")
    )
    parsed = _parse(rows)
    assert len(parsed["by_pws"]["AL0000983"]["dets"]) == 3


def test_bad_rows_counted_never_silent():
    rows = _row(contaminant="PFOA", value="not-a-number")
    parsed = _parse(rows)
    assert parsed["bad_rows"] == 1
    assert parsed["by_pws"] == {}


def test_missing_pwsid_counted_as_bad_row():
    rows = _row(pwsid="", contaminant="PFOA")
    parsed = _parse(rows)
    assert parsed["bad_rows"] == 1


def test_header_change_raises_loudly():
    with pytest.raises(ValueError, match="source shape changed"):
        pfas.parse_detections(io.StringIO("TOTALLY\tDIFFERENT\tHEADER\nA\tB\n"))


def test_build_artifact_refuses_empty():
    with pytest.raises(RuntimeError, match="refusing to write"):
        pfas.build_artifact({"by_pws": {}, "total_pfas_rows": 0, "monitored_pws": 0, "bad_rows": 0}, {}, "t")


def test_build_artifact_quarantines_missing_geocode_never_guesses():
    parsed = _parse(
        _row(pwsid="AL0000983", contaminant="PFOA", value="0.03")
        + _row(pwsid="TX0000001", contaminant="PFOS", value="0.02", state="TX", region="6")
    )
    geo = {"AL0000983": {"lat": 30.47, "lon": -88.33, "registry_id": "R1"}}  # TX missing on purpose
    art = pfas.build_artifact(parsed, geo, "2026-07-13T00:00:00Z")
    assert art["counts"]["pws_with_detection"] == 2
    assert art["counts"]["pws_geocoded"] == 1
    assert art["counts"]["pws_missing_geocode"] == 1
    assert [s["pws_id"] for s in art["systems"]] == ["AL0000983"]
    assert art["systems"][0]["max_result_ug_l"] == 0.03
    assert "not a regulatory violation" in art["caveat"]


def test_detections_sorted_highest_result_first():
    parsed = _parse(
        _row(contaminant="PFOA", value="0.01", date="1/1/2024")
        + _row(contaminant="PFOS", value="0.05", date="1/1/2024")
        + _row(contaminant="PFHxS", value="0.02", date="1/1/2024")
    )
    geo = {"AL0000983": {"lat": 30.0, "lon": -88.0, "registry_id": "R1"}}
    art = pfas.build_artifact(parsed, geo, "t")
    results = [d["result"] for d in art["systems"][0]["detections"]]
    assert results == sorted(results, reverse=True)


def test_committed_artifact_is_coherent():
    """The repo artifact (built from the real UCMR5 file + live ECHO
    geocode join) satisfies its own invariants."""
    import json
    fp = os.path.join(os.path.dirname(__file__), "datacore", "pfas", "pfas_ucmr5.json")
    art = json.load(open(fp))
    c = art["counts"]
    assert c["pws_with_detection"] == len(art["systems"]) + c["pws_missing_geocode"]
    assert c["pws_geocoded"] == len(art["systems"]) > 0
    assert c["pws_monitored_for_pfas"] >= c["pws_with_detection"] > 0
    for s in art["systems"][:50]:
        assert -90 <= s["lat"] <= 90 and -180 <= s["lon"] <= 180
        assert s["n_detections"] == len(s["detections"]) > 0
        assert s["max_result_ug_l"] == max(d["result"] for d in s["detections"])
        for d in s["detections"]:
            assert d["result"] >= d["mrl"] > 0
    assert "lithium" not in {d["contaminant"] for s in art["systems"] for d in s["detections"]}
