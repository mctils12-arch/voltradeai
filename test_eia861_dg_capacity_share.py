"""test_eia861_dg_capacity_share.py — pure-function tests for
scripts/eia861_dg_capacity_share.py. No network; the four load_* I/O
functions (openpyxl file reads) are exercised only by running the script
live against real downloaded EIA-861/EIA-860 files, same convention as
every sibling EIA-860/861/930 gate-1 script's test file in this repo.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "eia861_dg_capacity_share",
    os.path.join(os.path.dirname(__file__), "scripts", "eia861_dg_capacity_share.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _nm_row(year, state, pv_total, extra_len=8):
    """A synthetic Net Metering 'States- State Level' row: index0=caveat
    text, index1=year, index2=state, index7=PV total capacity — the same
    positions the module's NET_METERING_* column constants point at."""
    row = [None] * extra_len
    row[0] = "caveat text"
    row[1] = year
    row[2] = state
    row[7] = pv_total
    return tuple(row)


def _nnm_row(year, state, all_tech_total, utility_owned, pv_total, extra_len=14):
    """A synthetic Non-Net-Metering 'States- State Level' row: index1=year,
    index2=state, index4=all-tech total capacity, index6=all-tech
    utility-owned capacity, index13=PV total capacity."""
    row = [None] * extra_len
    row[0] = "caveat text"
    row[1] = year
    row[2] = state
    row[4] = all_tech_total
    row[6] = utility_owned
    row[13] = pv_total
    return tuple(row)


# ---- _numeric ----

def test_numeric_treats_dot_as_zero():
    assert mod._numeric(".") == 0.0


def test_numeric_treats_none_as_zero():
    assert mod._numeric(None) == 0.0


def test_numeric_parses_real_value():
    assert mod._numeric(161.562) == 161.562


# ---- find_state_level_rows ----

def test_find_state_level_rows_filters_by_year_and_state():
    rows = [
        _nm_row(2024, "OK", 100.0),   # wrong year
        _nm_row(2025, "OK", 161.562),
        _nm_row(2025, "TX", 999.0),   # not a requested state
        _nm_row(2025, "KS", 104.618),
    ]
    out = mod.find_state_level_rows(rows, 2025, ("OK", "KS", "NE"), mod.NET_METERING_YEAR_COL, mod.NET_METERING_STATE_COL)
    assert set(out) == {"OK", "KS"}
    assert out["OK"][7] == 161.562


def test_find_state_level_rows_skips_caveat_and_header_rows():
    caveat_row = (None,) * 5  # year_col holds None, not an int — must not raise or match
    rows = [caveat_row, _nm_row(2025, "NE", 27.283)]
    out = mod.find_state_level_rows(rows, 2025, ("NE",), mod.NET_METERING_YEAR_COL, mod.NET_METERING_STATE_COL)
    assert set(out) == {"NE"}


# ---- net_metering_pv_capacity_mw / non_net_metering_pv_capacity_mw ----

def test_net_metering_pv_capacity_mw_reads_correct_column():
    row = _nm_row(2025, "OK", 161.562)
    assert mod.net_metering_pv_capacity_mw(row) == 161.562


def test_non_net_metering_pv_capacity_mw_reads_correct_column():
    row = _nnm_row(2025, "OK", 8.158, 5.442, 1.733)
    assert mod.non_net_metering_pv_capacity_mw(row) == 1.733


def test_non_net_metering_pv_capacity_mw_handles_dot():
    row = _nnm_row(2025, "KS", 22.493, ".", 22.064)
    assert mod.non_net_metering_pv_capacity_mw(row) == 22.064


# ---- combined_dg_pv_capacity ----

def test_combined_dg_pv_capacity_sums_both_tables_matching_live_ok_reading():
    nm = {"OK": _nm_row(2025, "OK", 161.562)}
    nnm = {"OK": _nnm_row(2025, "OK", 8.158, 5.442, 1.733)}
    per_state, total = mod.combined_dg_pv_capacity(nm, nnm, ("OK",))
    assert per_state["OK"]["total_mw"] == 163.295
    assert total == 163.295


def test_combined_dg_pv_capacity_handles_one_table_missing_a_state():
    nm = {"NE": _nm_row(2025, "NE", 27.283)}
    nnm = {}  # NE absent from non-net-metering table entirely
    per_state, total = mod.combined_dg_pv_capacity(nm, nnm, ("NE",))
    assert per_state["NE"]["net_metering_mw"] == 27.283
    assert per_state["NE"]["non_net_metering_mw"] is None
    assert per_state["NE"]["total_mw"] == 27.283
    assert total == 27.283


def test_combined_dg_pv_capacity_state_absent_from_both_is_none_not_zero():
    per_state, total = mod.combined_dg_pv_capacity({}, {}, ("ND",))
    assert per_state["ND"] is None
    assert total == 0.0


def test_combined_dg_pv_capacity_matches_full_three_state_live_reading():
    nm = {
        "OK": _nm_row(2025, "OK", 161.562),
        "KS": _nm_row(2025, "KS", 104.618),
        "NE": _nm_row(2025, "NE", 27.283),
    }
    nnm = {
        "OK": _nnm_row(2025, "OK", 8.158, 5.442, 1.733),
        "KS": _nnm_row(2025, "KS", 22.493, ".", 22.064),
        "NE": _nnm_row(2025, "NE", 19.689, 1.065, 10.537),
    }
    _, total = mod.combined_dg_pv_capacity(nm, nnm, ("OK", "KS", "NE"))
    assert total == 327.797  # matches this session's live production run exactly


# ---- all_tech_utility_owned_bound ----

def test_all_tech_utility_owned_bound_pooled_share():
    nnm = {
        "OK": _nnm_row(2025, "OK", 8.158, 5.442, 1.733),
        "KS": _nnm_row(2025, "KS", 22.493, ".", 22.064),
        "NE": _nnm_row(2025, "NE", 19.689, 1.065, 10.537),
    }
    per_state, pooled = mod.all_tech_utility_owned_bound(nnm, ("OK", "KS", "NE"))
    assert per_state["OK"]["utility_owned_mw"] == 5.442
    assert per_state["KS"]["utility_owned_mw"] == 0.0
    assert pooled == 0.1293  # (5.442+0+1.065) / (8.158+22.493+19.689), matches live run


def test_all_tech_utility_owned_bound_missing_state_is_none():
    per_state, pooled = mod.all_tech_utility_owned_bound({}, ("ND",))
    assert per_state["ND"] is None
    assert pooled is None


# ---- utility_scale_solar_capacity_by_state ----

def test_utility_scale_solar_capacity_by_state_sums_op_status_only():
    rows = [
        ("OP", 1, 100.0),
        ("RE", 2, 999.0),  # retired — excluded
        ("OP", 3, 50.0),
    ]
    plant_state = {1: "OK", 2: "OK", 3: "OK"}
    out, unmatched = mod.utility_scale_solar_capacity_by_state(rows, plant_state)
    assert out == {"OK": 150.0}
    assert unmatched == 0


def test_utility_scale_solar_capacity_by_state_groups_by_state():
    rows = [("OP", 1, 10.0), ("OP", 2, 20.0), ("OP", 3, 5.0)]
    plant_state = {1: "OK", 2: "KS", 3: "OK"}
    out, unmatched = mod.utility_scale_solar_capacity_by_state(rows, plant_state)
    assert out == {"OK": 15.0, "KS": 20.0}
    assert unmatched == 0


def test_utility_scale_solar_capacity_by_state_counts_unmatched_plant_codes():
    rows = [("OP", 1, 10.0), ("OP", 99, 5.0)]  # 99 has no directory entry
    plant_state = {1: "OK"}
    out, unmatched = mod.utility_scale_solar_capacity_by_state(rows, plant_state)
    assert out == {"OK": 10.0}
    assert unmatched == 1


# ---- dg_capacity_share ----

def test_dg_capacity_share_matches_live_reading():
    assert mod.dg_capacity_share(327.797, 961.7) == 0.2542


def test_dg_capacity_share_zero_denominator_returns_none():
    assert mod.dg_capacity_share(0.0, 0.0) is None


# ---- implied_overshoot_from_capacity_share ----

def test_implied_overshoot_from_capacity_share_matches_manual_calc():
    # 1 / (1 - 0.2) = 1.25 exactly
    assert mod.implied_overshoot_from_capacity_share(0.2) == 1.25


def test_implied_overshoot_from_capacity_share_matches_live_reading():
    assert mod.implied_overshoot_from_capacity_share(0.2542) == 1.3408


def test_implied_overshoot_from_capacity_share_none_input():
    assert mod.implied_overshoot_from_capacity_share(None) is None


def test_implied_overshoot_from_capacity_share_undefined_at_or_above_one():
    assert mod.implied_overshoot_from_capacity_share(1.0) is None
    assert mod.implied_overshoot_from_capacity_share(1.2) is None
