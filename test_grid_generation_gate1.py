"""
test_grid_generation_gate1.py — pure-function battery for
scripts/grid_generation_gate1.py (registry_capacity_by_fuel/
aggregate_max_by_fueltype/bucket_generation_max/reconcile). No network —
fetch_window is the one networked function and is exercised only by
running the script live, same convention as test_un_comtrade_gate1.py.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "grid_generation_gate1", os.path.join(os.path.dirname(__file__), "scripts", "grid_generation_gate1.py"))
gate1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate1)


def test_registry_capacity_by_fuel_sums_within_conus_bbox():
    plants = [
        ["Plant A", 1000.0, "nuclear", "Op A", 35.0, -80.0, 1],   # in CONUS
        ["Plant B", 500.0, "nuclear", "Op B", 36.0, -81.0, 1],    # in CONUS
        ["Kahe", 609.7, "oil", "Hawaiian Electric", 21.3, -158.1, 0],  # HI, excluded
    ]
    cap, excluded = gate1.registry_capacity_by_fuel(plants)
    assert cap == {"nuclear": 1500.0}
    assert excluded == 1


def test_registry_capacity_by_fuel_excludes_missing_coordinates():
    plants = [["Plant A", 100.0, "gas", "Op A", None, None, 0]]
    cap, excluded = gate1.registry_capacity_by_fuel(plants)
    assert cap == {}
    assert excluded == 1


def test_aggregate_max_by_fueltype_takes_max_ignores_blank_values():
    rows = [
        {"fueltype": "NUC", "value": "100"},
        {"fueltype": "NUC", "value": "150"},
        {"fueltype": "NUC", "value": "120"},
        {"fueltype": "COL", "value": ""},   # blank, skipped
        {"fueltype": "COL", "value": None},  # missing, skipped
    ]
    out = gate1.aggregate_max_by_fueltype(rows)
    assert out["NUC"] == 150.0
    # COL had zero usable rows (both blank/missing) -> never counted as seen, absent entirely
    assert "COL" not in out


def test_bucket_generation_max_sums_wat_and_ps_into_hydro_and_floors_negative_at_zero():
    eia_max = {"WAT": 40000.0, "PS": 6000.0, "BAT": -11000.0, "SUN": 100000.0}
    bucket = gate1.bucket_generation_max(eia_max)
    assert bucket["hydro"] == 46000.0
    assert bucket["solar"] == 100000.0
    assert bucket["other"] == 0.0  # BAT floored at 0, not -11000


def test_reconcile_pass_when_generation_within_capacity_plus_tolerance():
    cap = {"nuclear": 100000.0}
    gen = {"nuclear": 95000.0}
    out = gate1.reconcile(cap, gen, tolerance=0.05)
    assert out["nuclear"]["verdict"] == "PASS"
    assert out["nuclear"]["ratio_of_capacity"] == 0.95


def test_reconcile_fail_when_generation_exceeds_capacity_ceiling():
    cap = {"solar": 38000.0}
    gen = {"solar": 115000.0}
    out = gate1.reconcile(cap, gen, tolerance=0.05)
    assert out["solar"]["verdict"] == "FAIL"
    assert out["solar"]["ratio_of_capacity"] > 1.05


def test_reconcile_pass_at_exact_tolerance_boundary_fail_just_above():
    cap = {"gas": 100000.0}
    assert gate1.reconcile(cap, {"gas": 105000.0}, tolerance=0.05)["gas"]["verdict"] == "PASS"
    assert gate1.reconcile(cap, {"gas": 105000.01}, tolerance=0.05)["gas"]["verdict"] == "FAIL"


def test_reconcile_inconclusive_when_bucket_missing_on_either_side():
    out = gate1.reconcile({}, {"nuclear": 1000.0}, tolerance=0.05)
    assert out["nuclear"]["verdict"] == "INCONCLUSIVE"
    out2 = gate1.reconcile({"nuclear": 1000.0}, {}, tolerance=0.05)
    assert out2["nuclear"]["verdict"] == "INCONCLUSIVE"


def test_reconcile_only_covers_verdicted_buckets_not_other():
    out = gate1.reconcile({"other": 1.0}, {"other": 1.0}, tolerance=0.05)
    assert "other" not in out
    assert set(out.keys()) == set(gate1.VERDICTED_BUCKETS)
