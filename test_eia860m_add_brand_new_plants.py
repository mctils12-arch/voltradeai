"""
test_eia860m_add_brand_new_plants.py — pure-function battery for
scripts/eia860m_add_brand_new_plants.py (eia860m_plants_by_fuel/
build_brand_new_plant_rows). No network, no xlsx, no csv —
load_eia860m_operating_rows (the file-reading function) is exercised only
by running the script live against a real downloaded file, same convention
as every sibling EIA-860 script in this repo.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "add_brand_new_plants",
    os.path.join(os.path.dirname(__file__), "scripts", "eia860m_add_brand_new_plants.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_eia860m_plants_by_fuel_sums_multiple_generators_same_plant():
    rows = [
        (1, "Foo Solar", "Foo Co", "ERCO", "SUN", "(OP) Operating", 10.0, 30.0, -97.0),
        (1, "Foo Solar", "Foo Co", "ERCO", "SUN", "(OP) Operating", 5.0, 30.0, -97.0),
    ]
    out = mod.eia860m_plants_by_fuel(rows)
    assert out == {"solar": {1: ("Foo Solar", "Foo Co", "ERCO", 30.0, -97.0, 15.0)}}


def test_eia860m_plants_by_fuel_excludes_unmapped_energy_source():
    rows = [(1, "Coal Plant", "Co", "ERCO", "BIT", "(OP) Operating", 100.0, 30.0, -97.0)]
    assert mod.eia860m_plants_by_fuel(rows) == {}


def test_eia860m_plants_by_fuel_excludes_non_operating_status():
    rows = [(1, "Wind Farm", "Co", "SWPP", "WND", "(RE) Retired", 50.0, 33.0, -95.0)]
    assert mod.eia860m_plants_by_fuel(rows) == {}


def test_eia860m_plants_by_fuel_excludes_missing_plant_id():
    rows = [(None, "Foo", "Co", "ERCO", "SUN", "(OP) Operating", 10.0, 30.0, -97.0)]
    assert mod.eia860m_plants_by_fuel(rows) == {}


def test_eia860m_plants_by_fuel_keeps_first_seen_metadata():
    rows = [
        (1, "Name A", "Entity A", "ERCO", "SUN", "(OP) Operating", 10.0, 30.0, -97.0),
        (1, "Name B (typo)", "Entity A", "ERCO", "SUN", "(OP) Operating", 5.0, 30.1, -97.1),
    ]
    out = mod.eia860m_plants_by_fuel(rows)
    assert out["solar"][1][0] == "Name A"
    assert out["solar"][1][3:5] == (30.0, -97.0)
    assert out["solar"][1][5] == 15.0


def test_eia860m_plants_by_fuel_multiple_fuels_and_plants():
    rows = [
        (1, "Solar A", "Co", "ERCO", "SUN", "(OP) Operating", 10.0, 30.0, -97.0),
        (2, "Wind B", "Co", "SWPP", "WND", "(OP) Operating", 20.0, 33.0, -95.0),
    ]
    out = mod.eia860m_plants_by_fuel(rows)
    assert set(out.keys()) == {"solar", "wind"}
    assert 1 in out["solar"] and 2 in out["wind"]


def test_build_brand_new_plant_rows_basic():
    plants = {1: ("Foo Solar", "Foo Co", "ERCO", 30.0, -97.0, 12.3)}
    rows, skip_cap, skip_coords = mod.build_brand_new_plant_rows(
        "solar", plants, annual_codes=set(), gppd_codes=set())
    assert rows == [["Foo Solar", 12.3, "solar", "Foo Co", 30.0, -97.0, 0]]
    assert skip_cap == 0
    assert skip_coords == 0


def test_build_brand_new_plant_rows_excludes_code_in_annual():
    plants = {1: ("Foo", "Co", "ERCO", 30.0, -97.0, 12.3)}
    rows, skip_cap, skip_coords = mod.build_brand_new_plant_rows(
        "solar", plants, annual_codes={1}, gppd_codes=set())
    assert rows == []
    assert skip_cap == 0
    assert skip_coords == 0


def test_build_brand_new_plant_rows_excludes_code_in_gppd():
    plants = {1: ("Foo", "Co", "ERCO", 30.0, -97.0, 12.3)}
    rows, skip_cap, skip_coords = mod.build_brand_new_plant_rows(
        "solar", plants, annual_codes=set(), gppd_codes={1})
    assert rows == []
    assert skip_cap == 0
    assert skip_coords == 0


def test_build_brand_new_plant_rows_skips_zero_or_negative_capacity():
    plants = {1: ("Foo", "Co", "ERCO", 30.0, -97.0, 0.0), 2: ("Bar", "Co", "ERCO", 30.0, -97.0, -5.0)}
    rows, skip_cap, skip_coords = mod.build_brand_new_plant_rows(
        "solar", plants, annual_codes=set(), gppd_codes=set())
    assert rows == []
    assert skip_cap == 2
    assert skip_coords == 0


def test_build_brand_new_plant_rows_skips_missing_coords():
    plants = {1: ("Foo", "Co", "ERCO", None, -97.0, 12.3), 2: ("Bar", "Co", "ERCO", 30.0, None, 5.0)}
    rows, skip_cap, skip_coords = mod.build_brand_new_plant_rows(
        "solar", plants, annual_codes=set(), gppd_codes=set())
    assert rows == []
    assert skip_cap == 0
    assert skip_coords == 2


def test_build_brand_new_plant_rows_falls_back_to_synthetic_name():
    plants = {1: (None, None, "ERCO", 30.0, -97.0, 12.3)}
    rows, _, _ = mod.build_brand_new_plant_rows(
        "solar", plants, annual_codes=set(), gppd_codes=set())
    assert rows[0][0] == "EIA Plant 1"
    assert rows[0][3] == ""


def test_split_top_n_unverifiable_holds_back_row_above_cutoff():
    existing = [["A", 100.0, "solar", "O", 1.0, 1.0, 1]] * 99 + [["B", 50.0, "solar", "O", 1.0, 1.0, 1]]
    new_rows = [["New Big", 75.0, "solar", "O", 2.0, 2.0, 0], ["New Small", 10.0, "solar", "O", 3.0, 3.0, 0]]
    safe, held_back = mod.split_top_n_unverifiable(existing, new_rows, top_n=100)
    assert safe == [["New Small", 10.0, "solar", "O", 3.0, 3.0, 0]]
    assert held_back == [["New Big", 75.0, "solar", "O", 2.0, 2.0, 0]]


def test_split_top_n_unverifiable_all_safe_when_registry_below_top_n():
    existing = [["A", 100.0, "solar", "O", 1.0, 1.0, 1]] * 5
    new_rows = [["New Big", 9999.0, "solar", "O", 2.0, 2.0, 0]]
    safe, held_back = mod.split_top_n_unverifiable(existing, new_rows, top_n=100)
    assert safe == new_rows
    assert held_back == []


def test_split_top_n_unverifiable_empty_new_rows():
    existing = [["A", 100.0, "solar", "O", 1.0, 1.0, 1]] * 100
    safe, held_back = mod.split_top_n_unverifiable(existing, [], top_n=100)
    assert safe == []
    assert held_back == []


def test_build_brand_new_plant_rows_sorted_by_plant_id_deterministic():
    plants = {
        3: ("C", "Co", "ERCO", 30.0, -97.0, 1.0),
        1: ("A", "Co", "ERCO", 30.0, -97.0, 1.0),
        2: ("B", "Co", "ERCO", 30.0, -97.0, 1.0),
    }
    rows, _, _ = mod.build_brand_new_plant_rows(
        "solar", plants, annual_codes=set(), gppd_codes=set())
    assert [r[0] for r in rows] == ["A", "B", "C"]
