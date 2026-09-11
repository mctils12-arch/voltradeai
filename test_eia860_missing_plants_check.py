"""
test_eia860_missing_plants_check.py — pure-function battery for
scripts/eia860_missing_plants_check.py (gppd_plant_code/gppd_codes_by_fuel/
eia860_capacity_by_code/missing_plants_report). No network, no xlsx, no csv
— load_gppd_rows/load_eia860_generator_rows (the two file-reading functions)
are exercised only by running the script live against real downloaded
files, same convention as eia860_registry_capacity_check.py.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "missing_plants_check",
    os.path.join(os.path.dirname(__file__), "scripts", "eia860_missing_plants_check.py"))
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def test_gppd_plant_code_parses_usa_prefix():
    assert check.gppd_plant_code("USA0063292") == 63292
    assert check.gppd_plant_code("USA0000001") == 1


def test_gppd_plant_code_none_for_non_usa_prefixed_synthetic_ids():
    assert check.gppd_plant_code("WRI1026808") is None
    assert check.gppd_plant_code("WKS0072975") is None


def test_gppd_plant_code_none_for_empty_or_none():
    assert check.gppd_plant_code("") is None
    assert check.gppd_plant_code(None) is None


def test_gppd_codes_by_fuel_filters_country_and_fuel():
    rows = [
        ("USA", "Solar", "USA0000001"),
        ("USA", "Wind", "USA0000002"),
        ("USA", "Coal", "USA0000003"),  # not solar/wind -> excluded
        ("CAN", "Solar", "USA0000004"),  # not USA country -> excluded
        ("USA", "Solar", "WRI1000005"),  # no parseable code -> excluded from the set
    ]
    out = check.gppd_codes_by_fuel(rows)
    assert out == {"solar": {1}, "wind": {2}}


def test_gppd_codes_by_fuel_empty_input():
    assert check.gppd_codes_by_fuel([]) == {}


def test_eia860_capacity_by_code_sums_multiple_generators_op_only():
    rows = [
        ("OP", 100, 5.0),
        ("OP", 100, 3.0),   # same plant code, second generator -> summed
        ("OS", 100, 50.0),  # out of service -> excluded
        ("OP", 200, 10.0),
    ]
    out = check.eia860_capacity_by_code(rows)
    assert out == {100: 8.0, 200: 10.0}


def test_eia860_capacity_by_code_none_capacity_counts_as_zero():
    out = check.eia860_capacity_by_code([("OP", 1, None), ("OP", 1, 4.0)])
    assert out == {1: 4.0}


def test_missing_plants_report_splits_matched_vs_missing():
    gppd_codes = {1, 2}
    eia_cap = {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}
    out = check.missing_plants_report("solar", gppd_codes, eia_cap)
    assert out["fuel"] == "solar"
    assert out["eia860_plant_codes_total"] == 4
    assert out["present_in_gppd"] == 2
    assert out["missing_from_gppd"] == 2
    assert out["matched_capacity_mw"] == 30.0
    assert out["missing_plant_capacity_mw"] == 70.0


def test_missing_plants_report_all_present_means_zero_missing_capacity():
    out = check.missing_plants_report("wind", {1, 2}, {1: 5.0, 2: 5.0})
    assert out["missing_from_gppd"] == 0
    assert out["missing_plant_capacity_mw"] == 0.0


def test_missing_plants_report_all_missing_means_zero_matched_capacity():
    out = check.missing_plants_report("wind", set(), {1: 5.0, 2: 5.0})
    assert out["present_in_gppd"] == 0
    assert out["matched_capacity_mw"] == 0.0
    assert out["missing_plant_capacity_mw"] == 10.0


def test_reuses_operating_status_constant_from_eia860_registry_capacity_check():
    # EDGE DOCTRINE #3 — reuse, not re-derive, the "OP" status literal.
    assert check.OPERATING_STATUS is check._chk.OPERATING_STATUS


def test_reuses_fuel_code_mapping_from_build_powerplants():
    assert check.FUEL_CODE is check._bpp.FUEL_CODE
