"""
test_un_comtrade_ingest.py — pure-function battery for
scripts/un_comtrade_ingest.py. No network (parse_response/merge_into_series/
build_artifact/month_range_desc are all pure); a separate coherence check
reads the real committed archive, same "test_committed_artifact_is_coherent"
convention as test_jodi_oil.py.
"""
import importlib.util
import json
import os

_spec = importlib.util.spec_from_file_location(
    "un_comtrade_ingest", os.path.join(os.path.dirname(__file__), "scripts", "un_comtrade_ingest.py"))
ingest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest)


def test_month_range_desc_walks_backward_across_year_boundary():
    assert ingest.month_range_desc("202602", 4) == ["202602", "202601", "202512", "202511"]


def test_month_range_desc_single_month():
    assert ingest.month_range_desc("202501", 1) == ["202501"]


def test_parse_response_extracts_known_partner_flow_rows():
    body = {
        "error": "",
        "data": [
            {"period": "202501", "partnerCode": 156, "flowCode": "M", "cifvalue": 100.0, "fobvalue": 90.0},
            {"period": "202501", "partnerCode": 156, "flowCode": "X", "cifvalue": None, "fobvalue": 50.0},
        ],
    }
    recs = ingest.parse_response(body, "202501")
    assert recs == [
        {"partner": 156, "flow": "M", "cif": 100.0, "fob": 90.0},
        {"partner": 156, "flow": "X", "cif": None, "fob": 50.0},
    ]


def test_parse_response_drops_rows_for_unknown_partner_or_flow():
    body = {"error": "", "data": [
        {"period": "202501", "partnerCode": 999, "flowCode": "M", "cifvalue": 1.0, "fobvalue": 1.0},
        {"period": "202501", "partnerCode": 156, "flowCode": "Z", "cifvalue": 1.0, "fobvalue": 1.0},
    ]}
    assert ingest.parse_response(body, "202501") == []


def test_parse_response_drops_rows_for_a_different_period():
    """Defensive: never trust a row echoed back for a period we didn't ask
    for (guards against a future API change silently mixing periods)."""
    body = {"error": "", "data": [
        {"period": "202412", "partnerCode": 156, "flowCode": "M", "cifvalue": 1.0, "fobvalue": 1.0},
    ]}
    assert ingest.parse_response(body, "202501") == []


def test_parse_response_raises_loudly_on_api_error():
    import pytest
    with pytest.raises(RuntimeError, match="UN Comtrade API error"):
        ingest.parse_response({"error": "Maximum number of periods for preview is 1"}, "202501")


def test_parse_response_empty_data_no_error_returns_empty_list():
    assert ingest.parse_response({"error": "", "data": []}, "202609") == []


def test_merge_into_series_adds_new_points_and_sorts():
    series = {}
    added1 = ingest.merge_into_series(series, "202502", [{"partner": 156, "flow": "M", "cif": 2.0, "fob": 1.0}])
    added2 = ingest.merge_into_series(series, "202501", [{"partner": 156, "flow": "M", "cif": 1.0, "fob": 0.5}])
    assert added1 == 1 and added2 == 1
    assert series["156|M"]["points"] == [["202501", 1.0, 0.5], ["202502", 2.0, 1.0]]


def test_merge_into_series_never_overwrites_an_already_archived_period():
    """Append-only, idempotent: re-running the ingest for a month it
    already has must not duplicate or overwrite that month's point."""
    series = {"156|M": {"points": [["202501", 999.0, 888.0]]}}
    added = ingest.merge_into_series(series, "202501", [{"partner": 156, "flow": "M", "cif": 1.0, "fob": 1.0}])
    assert added == 0
    assert series["156|M"]["points"] == [["202501", 999.0, 888.0]]


def test_build_artifact_reports_provenance_and_latest_period():
    series = {
        "156|M": {"points": [["202501", 1.0, 0.5], ["202502", 2.0, 1.0]]},
        "484|X": {"points": [["202412", 3.0, 3.0]]},
    }
    art = ingest.build_artifact(series, "2026-09-07T00:00:00+00:00")
    assert art["latest_period"] == "202502"
    assert art["series_count"] == 2
    assert art["series"]["156|M"] == {"points": [["202501", 1.0, 0.5], ["202502", 2.0, 1.0]], "n": 2, "first": "202501", "last": "202502"}
    assert art["reporter"] == {"code": 842, "name": "USA"}
    assert "China" in art["partners"].values()
    assert "UN Comtrade" in art["attribution"]


def test_build_artifact_empty_series_latest_period_is_none():
    art = ingest.build_artifact({}, "t")
    assert art["latest_period"] is None
    assert art["series_count"] == 0


def test_committed_archive_is_coherent():
    """The repo artifact (built live from the real API, 2026-09-07)
    satisfies its own invariants — same discipline as test_jodi_oil.py's
    test_committed_artifact_is_coherent."""
    fp = os.path.join(os.path.dirname(__file__), "datacore", "un_comtrade", "bilateral_trade.json")
    art = json.load(open(fp))
    assert art["series_count"] == len(art["series"]) == len(ingest.PARTNERS) * len(ingest.FLOWS)
    assert art["latest_period"] is not None
    for code in ingest.PARTNERS:
        for flow in ingest.FLOWS:
            s = art["series"][f"{code}|{flow}"]
            assert s["n"] == len(s["points"]) > 0
            periods = [p[0] for p in s["points"]]
            assert periods == sorted(periods), f"{code}|{flow} points not sorted"
            assert len(periods) == len(set(periods)), f"{code}|{flow} has a duplicate period"
            assert s["first"] == periods[0] and s["last"] == periods[-1]
            if flow == "M":
                assert all(p[1] is None or p[1] > 0 for p in s["points"]), f"{code}|M has a non-positive cif value"
