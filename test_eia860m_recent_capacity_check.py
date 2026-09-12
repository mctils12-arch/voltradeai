"""test_eia860m_recent_capacity_check.py — pure-function tests for
scripts/eia860m_recent_capacity_check.py. No network, no real EIA-860M/
registry files touched (synthetic fixtures only); load_eia860m_operating_rows
(the only I/O function, xlsx parsing) is exercised only by running the
script live against a real manually-downloaded EIA-860M file, same
convention as every sibling eia860_*.py test file in this repo.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "eia860m_recent_capacity_check",
    os.path.join(os.path.dirname(__file__), "scripts", "eia860m_recent_capacity_check.py"))
mrc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mrc)


# ---- eia860m_capacity_by_ba ----

def test_eia860m_capacity_by_ba_sums_operating_rows_by_ba_and_fuel():
    rows = [
        ("ERCO", "SUN", "(OP) Operating", 100.0),
        ("ERCO", "SUN", "(OP) Operating", 50.0),
        ("SWPP", "SUN", "(OP) Operating", 25.0),
        ("ERCO", "WND", "(OP) Operating", 10.0),
    ]
    cap = mrc.eia860m_capacity_by_ba(rows)
    assert cap == {"solar": {"ERCO": 150.0, "SWPP": 25.0}, "wind": {"ERCO": 10.0}}


def test_eia860m_capacity_by_ba_excludes_non_operating_status():
    rows = [
        ("ERCO", "SUN", "(OP) Operating", 100.0),
        ("ERCO", "SUN", "(OA) Out of service but expected to return to service in next calendar year", 999.0),
        ("ERCO", "SUN", "(OS) Out of service and NOT expected to return", 999.0),
        ("ERCO", "SUN", "(RE) Retired", 999.0),
    ]
    cap = mrc.eia860m_capacity_by_ba(rows)
    assert cap == {"solar": {"ERCO": 100.0}}


def test_eia860m_capacity_by_ba_excludes_unmapped_energy_source():
    rows = [("ERCO", "SUN", "(OP) Operating", 100.0), ("ERCO", "NG", "(OP) Operating", 500.0)]
    cap = mrc.eia860m_capacity_by_ba(rows)
    assert cap == {"solar": {"ERCO": 100.0}}


def test_eia860m_capacity_by_ba_excludes_missing_or_blank_ba():
    rows = [("ERCO", "SUN", "(OP) Operating", 100.0), (None, "SUN", "(OP) Operating", 50.0),
            ("", "SUN", "(OP) Operating", 25.0), ("  ", "SUN", "(OP) Operating", 25.0)]
    cap = mrc.eia860m_capacity_by_ba(rows)
    assert cap == {"solar": {"ERCO": 100.0}}


def test_eia860m_capacity_by_ba_treats_none_capacity_as_zero_not_a_crash():
    cap = mrc.eia860m_capacity_by_ba([("ERCO", "SUN", "(OP) Operating", None)])
    assert cap == {"solar": {"ERCO": 0.0}}


def test_eia860m_capacity_by_ba_empty_input():
    assert mrc.eia860m_capacity_by_ba([]) == {}


# ---- parse_as_of_period ----

def test_parse_as_of_period_extracts_month_year():
    assert mrc.parse_as_of_period("Inventory of Operating Generators as of July 2026") == "July 2026"


def test_parse_as_of_period_none_on_unexpected_format():
    assert mrc.parse_as_of_period("Some other title") is None
    assert mrc.parse_as_of_period(None) is None
    assert mrc.parse_as_of_period("") is None


# ---- compare_staleness ----

def test_compare_staleness_erco_solar_gate1_fail_fully_explained_by_growth():
    result = mrc.compare_staleness(32269.4, 30050.7, "ERCO", "solar", eia930_max_mwh=32327.0)
    assert result["capacity_growth_ratio"] == 1.074
    assert result["gate1_ratio_vs_stale_registry"] == 1.076
    assert result["gate1_ratio_vs_current_eia860m"] == 1.002


def test_compare_staleness_swpp_solar_gate1_fail_partially_explained_by_growth():
    result = mrc.compare_staleness(2078.1, 1345.0, "SWPP", "solar", eia930_max_mwh=2650.0)
    assert result["capacity_growth_ratio"] == 1.545
    assert result["gate1_ratio_vs_stale_registry"] == 1.97
    assert result["gate1_ratio_vs_current_eia860m"] == 1.275


def test_compare_staleness_without_eia930_max_omits_gate1_ratio_keys():
    result = mrc.compare_staleness(100.0, 90.0, "ERCO", "solar")
    assert "eia930_max_mwh" not in result
    assert "gate1_ratio_vs_stale_registry" not in result
    assert "gate1_ratio_vs_current_eia860m" not in result


def test_compare_staleness_zero_registry_mw_yields_none_ratio_not_divide_by_zero():
    result = mrc.compare_staleness(100.0, 0.0, "ERCO", "solar")
    assert result["capacity_growth_ratio"] is None


def test_compare_staleness_zero_eia860m_mw_yields_none_gate1_ratio_vs_current():
    result = mrc.compare_staleness(0.0, 100.0, "ERCO", "solar", eia930_max_mwh=50.0)
    assert result["gate1_ratio_vs_current_eia860m"] is None
    assert result["gate1_ratio_vs_stale_registry"] == 0.5


# ---- _parse_eia_max_arg ----

def test_parse_eia_max_arg_single_fuel_multiple_bas():
    out = mrc._parse_eia_max_arg("solar:ERCO=32327.0,SWPP=2650.0")
    assert out == {"solar": {"ERCO": 32327.0, "SWPP": 2650.0}}


def test_parse_eia_max_arg_multiple_fuels():
    out = mrc._parse_eia_max_arg("solar:ERCO=1;wind:ISNE=3")
    assert out == {"solar": {"ERCO": 1.0}, "wind": {"ISNE": 3.0}}


def test_parse_eia_max_arg_none_or_empty_returns_empty_dict():
    assert mrc._parse_eia_max_arg(None) == {}
    assert mrc._parse_eia_max_arg("") == {}


# ---- default constants ----

def test_default_bas_match_the_filed_next_item():
    assert mrc.DEFAULT_BAS == ("ERCO", "SWPP")


def test_energy_source_to_fuel_covers_solar_and_wind():
    assert mrc.ENERGY_SOURCE_TO_FUEL == {"SUN": "solar", "WND": "wind"}
