"""
test_jodi_oil.py — JODI primary closing-stocks battery (scripts/jodi_oil.py).
Pure-function tests on synthetic CSV shaped exactly like the real file
(header + markers verified by full-file scan 2026-07-06: 6,872,400 rows,
markers '-', 'x', 'N/A'; dense flow x product x unit grid). No network.
"""
import importlib.util
import io
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "jodi_oil", os.path.join(os.path.dirname(__file__), "scripts", "jodi_oil.py"))
jodi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jodi)

HEADER = "REF_AREA,TIME_PERIOD,ENERGY_PRODUCT,FLOW_BREAKDOWN,UNIT_MEASURE,OBS_VALUE,ASSESSMENT_CODE\n"


def _parse(rows: str):
    return jodi.parse_rows(io.StringIO(HEADER + rows))


def test_filters_to_clostlv_kbbl_and_known_products():
    parsed = _parse(
        "SA,2026-03,CRUDEOIL,CLOSTLV,KBBL,152645.0000,3\n"
        "SA,2026-03,CRUDEOIL,CLOSTLV,KBD,999.0,3\n"          # flow-rate unit: out
        "SA,2026-03,CRUDEOIL,TOTIMPSB,KBBL,555.0,3\n"        # import flow: out
        "SA,2026-03,SOMETHINGNEW,CLOSTLV,KBBL,1.0,3\n"        # unknown product: out
        "US,2026-04,TOTCRUDE,CLOSTLV,KBBL,784165.0000,1\n"
    )
    assert set(parsed["series"]) == {"SA|CRUDEOIL", "US|TOTCRUDE"}
    assert parsed["series"]["SA|CRUDEOIL"] == [["2026-03", 152645.0, "3"]]
    assert parsed["series"]["US|TOTCRUDE"] == [["2026-04", 784165.0, "1"]]


def test_markers_skipped_never_zeroed_and_counted():
    parsed = _parse(
        "AE,2002-01,CRUDEOIL,CLOSTLV,KBBL,-,3\n"
        "AE,2002-02,CRUDEOIL,CLOSTLV,KBBL,x,3\n"
        "AE,2002-03,CRUDEOIL,CLOSTLV,KBBL,N/A,3\n"
        "AE,2002-04,CRUDEOIL,CLOSTLV,KBBL,,3\n"
        "AE,2002-05,CRUDEOIL,CLOSTLV,KBBL,0.0000,3\n"  # a REAL zero level stays
    )
    assert parsed["skipped_markers"] == 4
    assert parsed["bad_rows"] == 0
    assert parsed["series"]["AE|CRUDEOIL"] == [["2002-05", 0.0, "3"]]


def test_garbage_value_counted_as_bad_never_silent():
    parsed = _parse("AE,2002-01,CRUDEOIL,CLOSTLV,KBBL,garbage,3\n")
    assert parsed["bad_rows"] == 1
    assert parsed["series"] == {}


def test_points_sorted_by_period_regardless_of_file_order():
    parsed = _parse(
        "IN,2020-12,CRUDEOIL,CLOSTLV,KBBL,2.0,2\n"
        "IN,2020-02,CRUDEOIL,CLOSTLV,KBBL,1.0,2\n"
    )
    assert [p[0] for p in parsed["series"]["IN|CRUDEOIL"]] == ["2020-02", "2020-12"]


def test_header_change_raises_loudly():
    with pytest.raises(ValueError, match="source shape changed"):
        jodi.parse_rows(io.StringIO("TOTALLY,DIFFERENT,HEADER\nA,B,C\n"))


def test_build_artifact_refuses_empty_and_reports_provenance():
    with pytest.raises(RuntimeError, match="refusing to write"):
        jodi.build_artifact({"series": {}, "skipped_markers": 0, "bad_rows": 0}, None, "t")
    parsed = _parse(
        "SA,2026-03,CRUDEOIL,CLOSTLV,KBBL,10.0,3\n"
        "SA,2026-04,CRUDEOIL,CLOSTLV,KBBL,11.0,3\n"
    )
    art = jodi.build_artifact(parsed, "Thu, 25 Jun 2026 03:50:04 GMT", "2026-07-06T00:00:00Z")
    s = art["series"]["SA|CRUDEOIL"]
    assert (s["first"], s["last"], s["n"]) == ("2026-03", "2026-04", 2)
    assert art["latest_period"] == "2026-04"
    assert art["series_count"] == 1
    assert art["source_last_modified"] == "Thu, 25 Jun 2026 03:50:04 GMT"
    assert "JODI" in art["attribution"]
    assert art["selection"]["markers_skipped_not_zeroed"] == 0


def test_committed_artifact_is_coherent():
    """The repo artifact (built from the real file) satisfies its own
    invariants — series key format, sorted points, count fields honest."""
    import json
    fp = os.path.join(os.path.dirname(__file__), "datacore", "jodi", "primary_stocks.json")
    art = json.load(open(fp))
    assert art["series_count"] == len(art["series"]) >= 300
    assert art["latest_period"] >= "2026-04"
    for key in ("SA|CRUDEOIL", "US|TOTCRUDE", "IN|CRUDEOIL"):
        s = art["series"][key]
        assert s["n"] == len(s["points"]) > 0
        periods = [p[0] for p in s["points"]]
        assert periods == sorted(periods)
        assert s["first"] == periods[0] and s["last"] == periods[-1]
