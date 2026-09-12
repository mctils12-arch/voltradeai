"""test_eia860_regional_capacity_check.py — pure-function tests for
scripts/eia860_regional_capacity_check.py. No network, no real EIA-860/
registry files touched (synthetic fixtures only); load_eia860_plant_ba_rows
(the only I/O function, xlsx parsing) is exercised only by running the
script live against a real manually-downloaded EIA-860 file, same
convention as every sibling eia860_*.py script's test file in this repo.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "eia860_regional_capacity_check",
    os.path.join(os.path.dirname(__file__), "scripts", "eia860_regional_capacity_check.py"))
erc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(erc)


# ---- eia860_plant_ba_map ----

def test_eia860_plant_ba_map_builds_direct_code_to_ba_dict():
    rows = [(1001, "ERCO"), (1002, "SWPP")]
    plant_to_ba, skipped = erc.eia860_plant_ba_map(rows)
    assert plant_to_ba == {1001: "ERCO", 1002: "SWPP"}
    assert skipped == 0


def test_eia860_plant_ba_map_counts_rows_with_no_reported_ba_code():
    rows = [(1001, "ERCO"), (1002, None), (1003, ""), (1004, "  ")]
    plant_to_ba, skipped = erc.eia860_plant_ba_map(rows)
    assert plant_to_ba == {1001: "ERCO"}
    assert skipped == 3


def test_eia860_plant_ba_map_strips_whitespace():
    plant_to_ba, skipped = erc.eia860_plant_ba_map([(1, " SWPP ")])
    assert plant_to_ba == {1: "SWPP"}
    assert skipped == 0


# ---- eia860_capacity_by_ba ----

def test_eia860_capacity_by_ba_sums_by_mapped_region():
    capacity_by_code = {1: 100.0, 2: 200.0, 3: 50.0}
    plant_to_ba = {1: "ERCO", 2: "ERCO", 3: "SWPP"}
    cap, unmapped_mw, unmapped_plants = erc.eia860_capacity_by_ba(capacity_by_code, plant_to_ba)
    assert cap == {"ERCO": 300.0, "SWPP": 50.0}
    assert unmapped_mw == 0.0
    assert unmapped_plants == 0


def test_eia860_capacity_by_ba_reports_unmapped_plants_never_drops_silently():
    capacity_by_code = {1: 100.0, 2: 999.0}
    plant_to_ba = {1: "ERCO"}  # plant 2 has capacity but no Schedule-2 BA row
    cap, unmapped_mw, unmapped_plants = erc.eia860_capacity_by_ba(capacity_by_code, plant_to_ba)
    assert cap == {"ERCO": 100.0}
    assert unmapped_mw == 999.0
    assert unmapped_plants == 1


def test_eia860_capacity_by_ba_empty_inputs():
    cap, unmapped_mw, unmapped_plants = erc.eia860_capacity_by_ba({}, {})
    assert cap == {}
    assert unmapped_mw == 0.0
    assert unmapped_plants == 0


# ---- compare_regional_capacity ----

def test_compare_regional_capacity_near_parity_small_ratio():
    eia860_ba_capacity = {"ERCO": 30016.3}
    registry_ba_capacity = {"ERCO": {"solar": 30050.7}}
    result = erc.compare_regional_capacity(eia860_ba_capacity, registry_ba_capacity, "ERCO", "solar")
    assert result == {
        "ba": "ERCO", "fuel": "solar",
        "eia860_reported_mw": 30016.3, "registry_matched_mw": 30050.7,
        "registry_to_eia860_ratio": 1.001, "gap_mw": -34.4,
    }


def test_compare_regional_capacity_real_residual_gap_ratio_below_one():
    eia860_ba_capacity = {"SWPP": 1362.8}
    registry_ba_capacity = {"SWPP": {"solar": 1345.0}}
    result = erc.compare_regional_capacity(eia860_ba_capacity, registry_ba_capacity, "SWPP", "solar")
    assert result["registry_to_eia860_ratio"] == 0.987
    assert result["gap_mw"] == 17.8


def test_compare_regional_capacity_zero_eia860_capacity_yields_none_ratio_not_divide_by_zero():
    result = erc.compare_regional_capacity({}, {"ERCO": {"solar": 100.0}}, "ERCO", "solar")
    assert result["eia860_reported_mw"] == 0.0
    assert result["registry_to_eia860_ratio"] is None


def test_compare_regional_capacity_missing_registry_fuel_bucket_is_zero_not_a_crash():
    result = erc.compare_regional_capacity({"ERCO": 100.0}, {"ERCO": {}}, "ERCO", "solar")
    assert result["registry_matched_mw"] == 0.0
    assert result["registry_to_eia860_ratio"] == 0.0


# ---- default constants ----

def test_default_bas_and_fuels_match_the_filed_next_item():
    assert erc.DEFAULT_BAS == ("ERCO", "SWPP")
    assert erc.DEFAULT_FUELS == ("solar", "wind")
