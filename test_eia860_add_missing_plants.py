"""
test_eia860_add_missing_plants.py — pure-function battery for
scripts/eia860_add_missing_plants.py (gppd_all_usa_codes/
build_missing_plant_rows/merge_registry). No network, no xlsx, no csv —
load_gppd_country_idnr_rows/load_eia860_plant_directory (the two
file-reading functions) are exercised only by running the script live
against real downloaded files, same convention as every sibling EIA-860
script in this repo.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "add_missing_plants",
    os.path.join(os.path.dirname(__file__), "scripts", "eia860_add_missing_plants.py"))
add = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(add)


def test_gppd_all_usa_codes_ignores_fuel_collects_any():
    rows = [
        ("USA", "USA0000001"),  # some fuel, doesn't matter here
        ("USA", "USA0000002"),
        ("CAN", "USA0000099"),  # non-USA country -> excluded
        ("USA", "WRI1000005"),  # unparseable -> excluded
    ]
    assert add.gppd_all_usa_codes(rows) == {1, 2}


def test_gppd_all_usa_codes_empty_input():
    assert add.gppd_all_usa_codes([]) == set()


def test_build_missing_plant_rows_basic():
    directory = {1: ("Foo Solar", "TX", 30.0, -97.0, "Foo Utility")}
    rows, skip_cap, skip_coords = add.build_missing_plant_rows(
        "solar", {1}, {1: 12.3}, directory)
    assert rows == [["Foo Solar", 12.3, "solar", "Foo Utility", 30.0, -97.0, 0]]
    assert skip_cap == 0
    assert skip_coords == 0


def test_build_missing_plant_rows_skips_zero_or_negative_capacity():
    directory = {1: ("Foo", "TX", 30.0, -97.0, "U")}
    rows, skip_cap, skip_coords = add.build_missing_plant_rows(
        "solar", {1}, {1: 0.0}, directory)
    assert rows == []
    assert skip_cap == 1
    assert skip_coords == 0


def test_build_missing_plant_rows_skips_missing_directory_entry():
    rows, skip_cap, skip_coords = add.build_missing_plant_rows(
        "wind", {1}, {1: 5.0}, {})
    assert rows == []
    assert skip_cap == 0
    assert skip_coords == 1


def test_build_missing_plant_rows_skips_null_coords():
    directory = {1: ("Foo", "TX", None, -97.0, "U")}
    rows, skip_cap, skip_coords = add.build_missing_plant_rows(
        "wind", {1}, {1: 5.0}, directory)
    assert rows == []
    assert skip_cap == 0
    assert skip_coords == 1


def test_build_missing_plant_rows_falls_back_to_synthetic_name():
    directory = {1: (None, "TX", 30.0, -97.0, "U")}
    rows, _, _ = add.build_missing_plant_rows("solar", {1}, {1: 5.0}, directory)
    assert rows[0][0] == "EIA Plant 1"


def test_build_missing_plant_rows_truncates_name_and_owner_to_60_chars():
    long_name = "X" * 90
    long_owner = "Y" * 90
    directory = {1: (long_name, "TX", 30.0, -97.0, long_owner)}
    rows, _, _ = add.build_missing_plant_rows("solar", {1}, {1: 5.0}, directory)
    assert len(rows[0][0]) == 60
    assert len(rows[0][3]) == 60


def test_build_missing_plant_rows_verified_is_always_zero():
    directory = {1: ("Foo", "TX", 30.0, -97.0, "U")}
    rows, _, _ = add.build_missing_plant_rows("solar", {1}, {1: 5.0}, directory)
    assert rows[0][6] == 0


def test_build_missing_plant_rows_sorted_by_code_deterministic():
    directory = {
        2: ("B", "TX", 1.0, 1.0, "U"),
        1: ("A", "TX", 1.0, 1.0, "U"),
    }
    rows, _, _ = add.build_missing_plant_rows("wind", {2, 1}, {1: 5.0, 2: 5.0}, directory)
    assert [r[0] for r in rows] == ["A", "B"]


def test_merge_registry_sorts_by_descending_capacity():
    existing = [["A", 10.0, "solar", "", 0.0, 0.0, 1], ["B", 1.0, "wind", "", 0.0, 0.0, 1]]
    new = [["C", 5.0, "solar", "", 0.0, 0.0, 0]]
    merged = add.merge_registry(existing, new)
    assert [p[0] for p in merged] == ["A", "C", "B"]


def test_merge_registry_does_not_mutate_inputs():
    existing = [["A", 10.0, "solar", "", 0.0, 0.0, 1]]
    new = [["B", 20.0, "solar", "", 0.0, 0.0, 0]]
    add.merge_registry(existing, new)
    assert existing == [["A", 10.0, "solar", "", 0.0, 0.0, 1]]
    assert new == [["B", 20.0, "solar", "", 0.0, 0.0, 0]]


def test_merge_registry_empty_new_rows_is_noop_besides_copy():
    existing = [["A", 10.0, "solar", "", 0.0, 0.0, 1]]
    merged = add.merge_registry(existing, [])
    assert merged == existing
    assert merged is not existing
