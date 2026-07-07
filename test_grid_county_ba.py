"""test_grid_county_ba.py — GRID VISION A1 step-2 battery
(scripts/grid_county_ba.py). Pure-function tests on the verified
EIA-861 contracts (state-preferred BA lookup — the Rio Grande
NM-row-smear fix; newline-wrapped headers) plus committed-artifact
coherence pinned to the workup's independently computed join."""
import importlib.util
import json
import os

_spec = importlib.util.spec_from_file_location(
    "grid_county_ba",
    os.path.join(os.path.dirname(__file__), "scripts", "grid_county_ba.py"))
gm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gm)


def test_bas_for_state_prefers_filing_state():
    u2ba = {"16057": {"NM": {"EPE"}, "TX": {"ERCO"}},   # Rio Grande EC
            "6198": {"NM": {"SWPP"}},                    # Farmers (no TX row)
            "5701": {"NM": {"EPE"}, "TX": {"EPE"}}}      # El Paso Electric
    assert gm.bas_for_state(u2ba, "16057", "TX") == {"ERCO"}, \
        "TX row must win — the NM-row EPE smear was the verified first-run bug"
    assert gm.bas_for_state(u2ba, "6198", "TX") == {"SWPP"}, \
        "no TX row -> any-state fallback (Farmers-NM case)"
    assert gm.bas_for_state(u2ba, "5701", "TX") == {"EPE"}
    assert gm.bas_for_state(u2ba, "99998", "TX") == set(), "unknown utility -> empty, never guessed"


def test_committed_crosswalk_matches_independent_join():
    path = os.path.join(os.path.dirname(__file__), "datacore", "gridvision",
                        "tx_county_ba.json")
    a = json.load(open(path))
    s = a["summary"]
    # pinned to the GV-A5 workup's independently computed 2024 join —
    # two implementations converging is the verification
    assert s["county_count"] == 254
    assert s["ba_county_counts"] == {"EPE": 3, "ERCO": 218, "MISO": 36,
                                     "PNM": 2, "SWPP": 78}
    assert s["multi_ba_counties"] == 80
    assert s["counties_with_no_ba"] == []
    assert s["ba_codes_not_in_eia930"] == [], "codes must join EIA-930 directly"
    assert s["unmatched_serving_utilities"] == []
    ep = a["counties"]["El Paso"]
    # El Paso is honestly multi-BA at 861 granularity: El Paso Electric
    # files EPE; Rio Grande EC also serves parts of the county and its
    # TX filing is ERCO — the set is the truth, no primary fabricated
    assert ep["bas"] == ["EPE", "ERCO"] and ep["multi_ba"]
    # Harris carries MISO via Entergy Texas (verified in the 861 join —
    # Entergy serves northeast slivers of the county); ERCOT metros
    # without such an edge utility are single-BA
    assert a["counties"]["Harris"]["bas"] == ["ERCO", "MISO"]
    assert a["counties"]["Travis"]["bas"] == ["ERCO"]
    assert a["counties"]["Dallas"]["bas"] == ["ERCO"]
    # honesty: no fabricated primary BA anywhere (861 cannot weight multi-BA)
    assert all("primary" not in str(k).lower()
               for c in a["counties"].values() for k in c), \
        "primary BA must not exist until a polygon arbiter lands"
    assert "RETAIL service, not grid topology" in a["provenance"]["method"]
    assert a["provenance"]["attribution"] == "U.S. Energy Information Administration"
