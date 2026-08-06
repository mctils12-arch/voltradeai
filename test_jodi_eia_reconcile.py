"""
test_jodi_eia_reconcile.py — battery for scripts/jodi_eia_reconcile.py.
Pure-function tests on small synthetic series (same shape as the real
git-committed archives: EIA weekly points, JODI monthly points). No
network, no dependency on the live datacore/ artifacts staying a
particular size — only monthly_avg/reconcile/verdict's own math is
under test.
"""
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "jodi_eia_reconcile", os.path.join(os.path.dirname(__file__), "scripts", "jodi_eia_reconcile.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_monthly_avg_averages_multiple_weekly_points_per_month():
    points = [
        ["2026-01-05", 100.0],
        ["2026-01-12", 200.0],
        ["2026-02-02", 300.0],
    ]
    out = mod.monthly_avg(points)
    assert out == {"2026-01": 150.0, "2026-02": 300.0}


def test_mom_delta_correlation_perfect_positive():
    eia = [100.0, 110.0, 105.0, 125.0, 120.0]   # deltas: +10, -5, +20, -5
    jodi = [50.0, 55.0, 52.5, 62.5, 60.0]        # deltas exactly half of eia's, same sign
    corr = mod.mom_delta_correlation(eia, jodi)
    assert corr == pytest.approx(1.0, abs=1e-9)


def test_mom_delta_correlation_zero_variance_returns_none():
    eia = [100.0, 100.0, 100.0]
    jodi = [50.0, 60.0, 40.0]
    assert mod.mom_delta_correlation(eia, jodi) is None


def test_reconcile_flags_insufficient_sample_below_two_months():
    eia_series = {
        "crude_stocks_us": {"points": [["2026-01-05", 400.0]]},
        "spr_crude_stocks": {"points": [["2026-01-05", 300.0]]},
    }
    jodi_series = {"US|CRUDEOIL": {"points": [["2026-01", 650.0]]}}
    r = mod.reconcile(eia_series, jodi_series, "US|CRUDEOIL")
    assert r["insufficient"] is True
    assert mod.verdict(r) == "INSUFFICIENT_SAMPLE"


def test_reconcile_close_match_passes_gate1():
    # 6 months, JODI tracks EIA total closely (small stable gap, same
    # direction of month-to-month moves) -> should pass both bars.
    months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    eia_crude = [400.0, 410.0, 420.0, 415.0, 425.0, 430.0]
    eia_spr = [300.0] * 6  # flat SPR
    eia_totals = [c + s for c, s in zip(eia_crude, eia_spr)]
    # JODI = 97% of EIA total each month -> ~3% gap, moves in lockstep
    jodi_vals = [t * 0.97 for t in eia_totals]

    eia_series = {
        "crude_stocks_us": {"points": [[f"{m}-05", v] for m, v in zip(months, eia_crude)]},
        "spr_crude_stocks": {"points": [[f"{m}-05", v] for m, v in zip(months, eia_spr)]},
    }
    jodi_series = {"US|TOTCRUDE": {"points": [[m, v] for m, v in zip(months, jodi_vals)]}}

    r = mod.reconcile(eia_series, jodi_series, "US|TOTCRUDE")
    assert r["n_months"] == 6
    assert abs(r["diff_pct_of_eia_mean"]) < mod.LEVEL_GAP_PASS_PCT
    assert r["mom_corr_full"] == pytest.approx(1.0, abs=1e-6)
    assert mod.verdict(r) == "GATE1_PASS"


def test_reconcile_large_stable_offset_fails_gate1():
    # JODI is a flat 40% BELOW the EIA total every month -> gap far
    # exceeds LEVEL_GAP_PASS_PCT even though correlation is perfect.
    months = ["2026-01", "2026-02", "2026-03", "2026-04"]
    eia_crude = [400.0, 410.0, 420.0, 430.0]
    eia_spr = [300.0] * 4
    eia_totals = [c + s for c, s in zip(eia_crude, eia_spr)]
    jodi_vals = [t * 0.60 for t in eia_totals]

    eia_series = {
        "crude_stocks_us": {"points": [[f"{m}-05", v] for m, v in zip(months, eia_crude)]},
        "spr_crude_stocks": {"points": [[f"{m}-05", v] for m, v in zip(months, eia_spr)]},
    }
    jodi_series = {"US|CRUDEOIL": {"points": [[m, v] for m, v in zip(months, jodi_vals)]}}

    r = mod.reconcile(eia_series, jodi_series, "US|CRUDEOIL")
    assert abs(r["diff_pct_of_eia_mean"]) > mod.LEVEL_GAP_PASS_PCT
    assert mod.verdict(r) == "GATE1_FAIL"


def test_reconcile_uncorrelated_moves_fails_gate1_even_with_small_gap():
    # Small average level gap but month-to-month moves are unrelated
    # (JODI held flat while EIA moves around) -> must still fail on the
    # correlation bar, not just the level-gap bar.
    months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
    eia_crude = [400.0, 380.0, 430.0, 390.0, 425.0]
    eia_spr = [300.0] * 5
    eia_totals = [c + s for c, s in zip(eia_crude, eia_spr)]
    avg_total = sum(eia_totals) / len(eia_totals)
    jodi_vals = [avg_total] * 5  # flat -> zero variance in JODI deltas

    eia_series = {
        "crude_stocks_us": {"points": [[f"{m}-05", v] for m, v in zip(months, eia_crude)]},
        "spr_crude_stocks": {"points": [[f"{m}-05", v] for m, v in zip(months, eia_spr)]},
    }
    jodi_series = {"US|TOTCRUDE": {"points": [[m, v] for m, v in zip(months, jodi_vals)]}}

    r = mod.reconcile(eia_series, jodi_series, "US|TOTCRUDE")
    assert abs(r["diff_pct_of_eia_mean"]) < mod.LEVEL_GAP_PASS_PCT
    assert r["mom_corr_full"] is None  # zero-variance JODI deltas
    assert mod.verdict(r) == "GATE1_FAIL"
