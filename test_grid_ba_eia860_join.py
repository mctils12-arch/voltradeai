"""test_grid_ba_eia860_join.py — pure-function tests for
scripts/grid_ba_eia860_join.py. No network, no real datacore files touched
(synthetic fixtures only); load_eia860_ba_directory (the only I/O function,
xlsx parsing) is exercised only by running the script live against a real
manually-downloaded EIA-860 file, same convention as every sibling
eia860_*.py script's test file in this repo.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "grid_ba_eia860_join", os.path.join(os.path.dirname(__file__), "scripts", "grid_ba_eia860_join.py"))
gbej = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gbej)


# ---- build_coord_index ----

def test_build_coord_index_groups_by_rounded_coordinate():
    directory = [
        {"plant_code": 1, "name": "A", "lat": 47.95751, "lon": -118.97732, "ba_code": "BPAT"},
        {"plant_code": 2, "name": "B", "lat": 26.6986, "lon": -80.3747, "ba_code": "FPL"},
    ]
    index = gbej.build_coord_index(directory)
    assert index[(47.9575, -118.9773)] == ["BPAT"]
    assert index[(26.6986, -80.3747)] == ["FPL"]


def test_build_coord_index_collects_multiple_codes_at_one_coordinate():
    directory = [
        {"plant_code": 1, "name": "A", "lat": 40.0, "lon": -90.0, "ba_code": "PJM"},
        {"plant_code": 2, "name": "B", "lat": 40.0, "lon": -90.0, "ba_code": "MISO"},
    ]
    index = gbej.build_coord_index(directory)
    assert sorted(index[(40.0, -90.0)]) == ["MISO", "PJM"]


# ---- assign_registry_ba ----

def test_assign_registry_ba_matched_single_code():
    plants = [["Grand Coulee", 6809.0, "hydro", "USBR", 47.9575, -118.9773, 1]]
    index = gbej.build_coord_index(
        [{"plant_code": 6163, "name": "Grand Coulee", "lat": 47.957511, "lon": -118.977323, "ba_code": "BPAT"}])
    out = gbej.assign_registry_ba(plants, index)
    assert out == [{"name": "Grand Coulee", "fuel": "hydro", "capacity_mw": 6809.0,
                     "ba_code": "BPAT", "status": "matched"}]


def test_assign_registry_ba_unmatched_no_coordinate_hit():
    plants = [["Nowhere Plant", 10.0, "gas", "Owner", 1.0, 1.0, 0]]
    index = gbej.build_coord_index([])
    out = gbej.assign_registry_ba(plants, index)
    assert out[0]["status"] == "unmatched"
    assert out[0]["ba_code"] is None


def test_assign_registry_ba_ambiguous_conflicting_codes_at_same_coordinate():
    plants = [["Border Plant", 50.0, "gas", "Owner", 40.0, -90.0, 1]]
    index = gbej.build_coord_index([
        {"plant_code": 1, "name": "X", "lat": 40.0, "lon": -90.0, "ba_code": "PJM"},
        {"plant_code": 2, "name": "Y", "lat": 40.0, "lon": -90.0, "ba_code": "MISO"},
    ])
    out = gbej.assign_registry_ba(plants, index)
    assert out[0]["status"] == "ambiguous"
    assert out[0]["ba_code"] is None


def test_assign_registry_ba_same_coordinate_agreeing_codes_is_matched_not_ambiguous():
    # Two EIA-860 records at one rounded coordinate (a multi-unit site
    # carrying more than one Plant Code) that AGREE on the BA code is not
    # a real collision — must resolve to matched, not ambiguous.
    plants = [["Multi-Unit Site", 200.0, "gas", "Owner", 40.0, -90.0, 1]]
    index = gbej.build_coord_index([
        {"plant_code": 1, "name": "X", "lat": 40.0, "lon": -90.0, "ba_code": "PJM"},
        {"plant_code": 2, "name": "Y", "lat": 40.0, "lon": -90.0, "ba_code": "PJM"},
    ])
    out = gbej.assign_registry_ba(plants, index)
    assert out[0] == {"name": "Multi-Unit Site", "fuel": "gas", "capacity_mw": 200.0,
                       "ba_code": "PJM", "status": "matched"}


# ---- registry_capacity_by_ba ----

def test_registry_capacity_by_ba_sums_matched_only():
    assignments = [
        {"name": "A", "fuel": "gas", "capacity_mw": 500.0, "ba_code": "CISO", "status": "matched"},
        {"name": "B", "fuel": "gas", "capacity_mw": 300.0, "ba_code": "CISO", "status": "matched"},
        {"name": "C", "fuel": "solar", "capacity_mw": 100.0, "ba_code": "ERCO", "status": "matched"},
        {"name": "D", "fuel": "gas", "capacity_mw": 999.0, "ba_code": None, "status": "unmatched"},
        {"name": "E", "fuel": "gas", "capacity_mw": 999.0, "ba_code": None, "status": "ambiguous"},
    ]
    cap, excluded = gbej.registry_capacity_by_ba(assignments)
    assert cap == {"CISO": {"gas": 800.0}, "ERCO": {"solar": 100.0}}
    # same 2-tuple interface as grid_ba_polygon_join.py's function of the
    # same name, but this join has no per-region excluded bucket (see
    # module docstring) — always empty.
    assert excluded == {}


# ---- summarize ----

def test_summarize_counts_and_capacity_by_status():
    assignments = [
        {"name": "A", "fuel": "gas", "capacity_mw": 500.0, "ba_code": "CISO", "status": "matched"},
        {"name": "B", "fuel": "gas", "capacity_mw": 100.0, "ba_code": None, "status": "unmatched"},
        {"name": "C", "fuel": "gas", "capacity_mw": 50.0, "ba_code": None, "status": "ambiguous"},
    ]
    summary = gbej.summarize(assignments)
    assert summary == {
        "total": 3, "matched": 1, "unmatched": 1, "ambiguous": 1,
        "unmatched_capacity_mw": 100.0, "ambiguous_capacity_mw": 50.0,
    }


# ---- compare_to_polygon_join ----

def test_compare_to_polygon_join_resolves_and_corroborates():
    eia860_assignments = [
        {"name": "A", "fuel": "nuclear", "capacity_mw": 100.0, "ba_code": "SRP", "status": "matched"},
        {"name": "B", "fuel": "gas", "capacity_mw": 50.0, "ba_code": "PJM", "status": "matched"},
        {"name": "C", "fuel": "gas", "capacity_mw": 20.0, "ba_code": None, "status": "unmatched"},
    ]
    polygon_assignments = [
        {"name": "A", "fuel": "nuclear", "capacity_mw": 100.0, "ba_codes": ["WALC", "AZPS", "SRP"]},
        {"name": "B", "fuel": "gas", "capacity_mw": 50.0, "ba_codes": ["PJM", "MISO"]},
        {"name": "C", "fuel": "gas", "capacity_mw": 20.0, "ba_codes": ["ERCO", "SWPP"]},
    ]
    result = gbej.compare_to_polygon_join(eia860_assignments, polygon_assignments)
    assert result == {
        "polygon_ambiguous_plants": 3,
        "resolved_by_eia860_join": 2,
        "resolved_matches_a_polygon_candidate": 2,
        "resolved_to_a_different_ba_than_any_polygon_candidate": 0,
        "still_unresolved": 1,
    }


def test_compare_to_polygon_join_flags_override_when_eia860_disagrees_with_every_candidate():
    eia860_assignments = [
        {"name": "A", "fuel": "gas", "capacity_mw": 10.0, "ba_code": "BPAT", "status": "matched"},
    ]
    polygon_assignments = [
        {"name": "A", "fuel": "gas", "capacity_mw": 10.0, "ba_codes": ["WALC", "AZPS"]},
    ]
    result = gbej.compare_to_polygon_join(eia860_assignments, polygon_assignments)
    assert result["resolved_matches_a_polygon_candidate"] == 0
    assert result["resolved_to_a_different_ba_than_any_polygon_candidate"] == 1


def test_compare_to_polygon_join_ignores_non_ambiguous_polygon_plants():
    # polygon already had exactly one candidate — not part of the
    # ambiguous-plant question this comparison answers.
    eia860_assignments = [
        {"name": "A", "fuel": "gas", "capacity_mw": 10.0, "ba_code": "CISO", "status": "matched"},
    ]
    polygon_assignments = [
        {"name": "A", "fuel": "gas", "capacity_mw": 10.0, "ba_codes": ["CISO"]},
    ]
    result = gbej.compare_to_polygon_join(eia860_assignments, polygon_assignments)
    assert result["polygon_ambiguous_plants"] == 0
    assert result["resolved_by_eia860_join"] == 0
