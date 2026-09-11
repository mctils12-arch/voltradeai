"""test_grid_generation_gate1_ba.py — pure-function battery for
scripts/grid_generation_gate1_ba.py's own new logic (registry_capacity_by_ba).
The per-fuel reconcile/aggregate/bucket machinery is reused, not
reimplemented, from grid_generation_gate1.py and is already covered by
test_grid_generation_gate1.py — not re-tested here. No network — fetch_window
is exercised only by running the script live, same convention as
test_grid_generation_gate1.py/test_un_comtrade_gate1.py.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "grid_generation_gate1_ba", os.path.join(os.path.dirname(__file__), "scripts", "grid_generation_gate1_ba.py"))
gate1_ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate1_ba)


def test_registry_capacity_by_ba_sums_unambiguous_plants_only():
    assignments = [
        {"name": "A", "fuel": "gas", "capacity_mw": 500.0, "ba_codes": ["CISO"]},
        {"name": "B", "fuel": "gas", "capacity_mw": 300.0, "ba_codes": ["CISO"]},
        {"name": "C", "fuel": "solar", "capacity_mw": 100.0, "ba_codes": ["ERCO"]},
    ]
    cap, excluded = gate1_ba.registry_capacity_by_ba(assignments)
    assert cap == {"CISO": {"gas": 800.0}, "ERCO": {"solar": 100.0}}
    assert excluded == {}


def test_registry_capacity_by_ba_excludes_ambiguous_multi_ba_plants_from_every_region():
    assignments = [
        {"name": "Palo Verde", "fuel": "nuclear", "capacity_mw": 4209.6, "ba_codes": ["WALC", "AZPS", "SRP"]},
    ]
    cap, excluded = gate1_ba.registry_capacity_by_ba(assignments)
    # not counted toward ANY region's capacity sum
    assert cap == {}
    # but its capacity IS surfaced against every region it overlaps, so the
    # gap is visible rather than silently disappearing
    assert excluded == {"WALC": 4209.6, "AZPS": 4209.6, "SRP": 4209.6}


def test_registry_capacity_by_ba_drops_unmatched_plants_entirely():
    assignments = [
        {"name": "Chugach", "fuel": "gas", "capacity_mw": 200.0, "ba_codes": []},
    ]
    cap, excluded = gate1_ba.registry_capacity_by_ba(assignments)
    assert cap == {}
    assert excluded == {}


def test_registry_capacity_by_ba_mixed_ambiguous_and_unambiguous_same_region():
    assignments = [
        {"name": "Unambig", "fuel": "wind", "capacity_mw": 1000.0, "ba_codes": ["SWPP"]},
        {"name": "Ambig", "fuel": "wind", "capacity_mw": 250.0, "ba_codes": ["SWPP", "MISO"]},
    ]
    cap, excluded = gate1_ba.registry_capacity_by_ba(assignments)
    # SWPP's counted capacity is ONLY the unambiguous plant
    assert cap == {"SWPP": {"wind": 1000.0}}
    assert excluded == {"SWPP": 250.0, "MISO": 250.0}


def test_excluded_capacity_fraction_basic():
    assert gate1_ba.excluded_capacity_fraction(counted_mw=800.0, excluded_mw=200.0) == 0.2


def test_excluded_capacity_fraction_bounded_zero_to_one_even_when_excluded_dominates():
    # a real observed case (FPL): ambiguous capacity overlapping the region
    # (23,211.4 MW) actually exceeds its unambiguous capacity (15,743.5 MW) —
    # the fraction is still bounded in [0, 1) because it is excluded /
    # (excluded + counted), not excluded / counted
    frac = gate1_ba.excluded_capacity_fraction(counted_mw=15743.5, excluded_mw=23211.4)
    assert 0.0 < frac < 1.0
    assert frac == 0.596


def test_excluded_capacity_fraction_none_when_region_has_no_known_capacity():
    assert gate1_ba.excluded_capacity_fraction(counted_mw=0.0, excluded_mw=0.0) is None


def test_default_respondents_match_the_literal_ground_truth_region_list():
    # CLAUDE.md / open_questions.md name CISO/ERCO/MISO/PJM/NYIS/ISNE/SWPP/FPL
    # explicitly; SE/NW/SW are EIA-930 rollup regions with no HIFLD polygon,
    # deliberately excluded (module docstring explains why).
    assert gate1_ba.DEFAULT_RESPONDENTS == ("CISO", "ERCO", "MISO", "PJM", "NYIS", "ISNE", "SWPP", "FPL")
