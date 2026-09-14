"""test_eia_dpv_btm_solar_share.py — pure-function tests for
scripts/eia_dpv_btm_solar_share.py. No network; fetch_state_generation (the
only I/O function) is exercised only by running the script live, same
convention as every sibling eia930/eia860 gate-1 script's test file.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "eia_dpv_btm_solar_share",
    os.path.join(os.path.dirname(__file__), "scripts", "eia_dpv_btm_solar_share.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


# ---- parse_generation_rows ----

def test_parse_generation_rows_parses_numeric_strings():
    rows = [{"location": "OK", "fueltypeid": "SUN", "period": "2026-06", "generation": "198.08416"}]
    gen, unparseable = mod.parse_generation_rows(rows)
    assert gen == {("OK", "SUN", "2026-06"): 198.08416}
    assert unparseable == 0


def test_parse_generation_rows_skips_missing_without_counting_as_error():
    rows = [{"location": "OK", "fueltypeid": "SUN", "period": "2026-06", "generation": None},
            {"location": "OK", "fueltypeid": "SUN", "period": "2026-05", "generation": ""}]
    gen, unparseable = mod.parse_generation_rows(rows)
    assert gen == {}
    assert unparseable == 0


def test_parse_generation_rows_counts_non_numeric_as_unparseable():
    rows = [{"location": "OK", "fueltypeid": "SUN", "period": "2026-06", "generation": "n/a"}]
    gen, unparseable = mod.parse_generation_rows(rows)
    assert gen == {}
    assert unparseable == 1


def test_parse_generation_rows_empty_input():
    assert mod.parse_generation_rows([]) == ({}, 0)


# ---- latest_common_period ----

def test_latest_common_period_picks_newest_fully_populated_period():
    gen = {
        ("OK", "DPV", "2026-06"): 1.0,
        ("OK", "SUN", "2026-06"): 2.0,
        ("KS", "DPV", "2026-06"): 1.0,
        # KS SUN missing for 2026-06 -> that period is not "common"
        ("OK", "DPV", "2026-05"): 1.0,
        ("OK", "SUN", "2026-05"): 2.0,
        ("KS", "DPV", "2026-05"): 1.0,
        ("KS", "SUN", "2026-05"): 2.0,
    }
    assert mod.latest_common_period(gen, ["OK", "KS"]) == "2026-05"


def test_latest_common_period_none_when_no_period_is_complete():
    gen = {("OK", "DPV", "2026-06"): 1.0}
    assert mod.latest_common_period(gen, ["OK", "KS"]) is None


# ---- dpv_btm_share ----

def test_dpv_btm_share_computes_per_state_and_aggregate():
    gen = {
        ("OK", "DPV", "2026-06"): 30.0,
        ("OK", "SUN", "2026-06"): 170.0,
        ("KS", "DPV", "2026-06"): 20.0,
        ("KS", "SUN", "2026-06"): 80.0,
    }
    result = mod.dpv_btm_share(gen, ["OK", "KS"], "2026-06")
    assert result["per_state"]["OK"]["dpv_share"] == 0.15
    assert result["per_state"]["KS"]["dpv_share"] == 0.2
    assert result["aggregate"]["total_dpv_mwh"] == 50.0
    assert result["aggregate"]["total_sun_mwh"] == 250.0
    assert result["aggregate"]["dpv_share"] == round(50.0 / 300.0, 4)


def test_dpv_btm_share_reports_missing_state_data_as_none_not_zero():
    result = mod.dpv_btm_share({}, ["OK"], "2026-06")
    assert result["per_state"]["OK"] == {"dpv_mwh": None, "sun_mwh": None, "dpv_share": None}
    assert result["aggregate"]["dpv_share"] is None


# ---- implied_overshoot_from_btm_share ----

def test_implied_overshoot_from_btm_share_arithmetic():
    assert mod.implied_overshoot_from_btm_share(0.2) == 1.25
    assert mod.implied_overshoot_from_btm_share(0.0) == 1.0


def test_implied_overshoot_from_btm_share_none_when_share_is_none_or_full():
    assert mod.implied_overshoot_from_btm_share(None) is None
    assert mod.implied_overshoot_from_btm_share(1.0) is None
    assert mod.implied_overshoot_from_btm_share(1.5) is None
