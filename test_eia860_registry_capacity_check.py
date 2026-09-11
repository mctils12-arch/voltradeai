"""
test_eia860_registry_capacity_check.py — pure-function battery for
scripts/eia860_registry_capacity_check.py (sum_operable_nameplate/
compare_fuel). No network, no xlsx — load_eia860_nameplate_rows (the one
file-reading function) is exercised only by running the script live against
a real downloaded EIA-860 file, same convention as grid_generation_gate1.py's
fetch_window.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "eia860_check", os.path.join(os.path.dirname(__file__), "scripts", "eia860_registry_capacity_check.py"))
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def test_sum_operable_nameplate_counts_only_op_status():
    rows = [("OP", 0.2), ("OP", 1.2), ("OS", 5.0), ("OA", 3.0)]
    total, counted, skipped = check.sum_operable_nameplate(rows)
    assert total == 1.4
    assert counted == 2
    assert skipped == 2


def test_sum_operable_nameplate_none_capacity_counts_as_zero_not_skipped():
    rows = [("OP", None), ("OP", 2.0)]
    total, counted, skipped = check.sum_operable_nameplate(rows)
    assert total == 2.0
    assert counted == 2
    assert skipped == 0


def test_sum_operable_nameplate_empty_input():
    total, counted, skipped = check.sum_operable_nameplate([])
    assert total == 0.0
    assert counted == 0
    assert skipped == 0


def test_compare_fuel_ratio_and_gap():
    out = check.compare_fuel("solar", registry_mw=37468.0, eia860_mw=154251.1)
    assert out["fuel"] == "solar"
    assert out["registry_capacity_mw"] == 37468.0
    assert out["eia860_nameplate_mw"] == 154251.1
    assert out["ratio_eia860_over_registry"] == round(154251.1 / 37468.0, 3)
    assert out["gap_mw"] == round(154251.1 - 37468.0, 1)


def test_compare_fuel_zero_registry_capacity_reports_none_ratio_not_a_crash():
    out = check.compare_fuel("solar", registry_mw=0.0, eia860_mw=100.0)
    assert out["ratio_eia860_over_registry"] is None
    assert out["gap_mw"] == 100.0


def test_compare_fuel_ratio_below_one_when_registry_overcounts():
    out = check.compare_fuel("wind", registry_mw=1000.0, eia860_mw=900.0)
    assert out["ratio_eia860_over_registry"] == 0.9
    assert out["gap_mw"] == -100.0


def test_registry_capacity_by_fuel_reused_from_grid_generation_gate1():
    # This module must reuse grid_generation_gate1.py's own bucket-sum
    # function rather than re-deriving it (EDGE DOCTRINE #3) — pin that by
    # identity, not just by matching behavior.
    assert check.registry_capacity_by_fuel is check._gg1.registry_capacity_by_fuel
