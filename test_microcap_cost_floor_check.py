"""Tests for scripts/microcap_cost_floor_check.py — the EDGE DOCTRINE axis (b)
costs-and-frictions-first workup (REASONING STANDARD #6) pricing tick-size
spread floor and LULD halt-band exposure for the pinned illiquid/moderate
microcap tickers. Pure structural-math tests; no network, no live data."""
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

spec = importlib.util.spec_from_file_location(
    "microcap_cost_floor_check",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "microcap_cost_floor_check.py"),
)
mccf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mccf)

import backtest_v2


def test_tick_floor_below_one_dollar_is_none():
    assert mccf.tick_floor_pct(0.99) is None
    assert mccf.tick_floor_pct(0.26) is None


def test_tick_floor_at_exactly_one_dollar_applies():
    # Reg NMS Rule 612's $0.01 minimum variation applies AT $1.00, not just above.
    assert mccf.tick_floor_pct(1.00) is not None
    assert mccf.tick_floor_pct(1.00) == pytest.approx(0.5, abs=1e-9)


def test_tick_floor_scales_inversely_with_price():
    assert mccf.tick_floor_pct(1.00) > mccf.tick_floor_pct(10.00) > mccf.tick_floor_pct(100.00)


def test_tick_floor_formula_is_half_tick_over_price():
    # half a cent / $2.00 * 100 = 0.25%
    assert mccf.tick_floor_pct(2.00) == pytest.approx(0.25, abs=1e-9)


def test_luld_band_sub_75_cents_uses_lesser_of_15c_or_75pct():
    # $0.10 stock: 0.15/0.10 = 1.5 -> capped at 0.75 -> 75%
    assert mccf.luld_band_pct(0.10) == pytest.approx(75.0, abs=1e-9)
    # $0.50 stock: 0.15/0.50 = 0.30 -> 30%, below the 75% cap
    assert mccf.luld_band_pct(0.50) == pytest.approx(30.0, abs=1e-9)


def test_luld_band_boundary_at_75_cents_and_3_dollars():
    assert mccf.luld_band_pct(0.75) == pytest.approx(20.0, abs=1e-9)
    assert mccf.luld_band_pct(3.00) == pytest.approx(20.0, abs=1e-9)
    assert mccf.luld_band_pct(3.01) == pytest.approx(10.0, abs=1e-9)  # tier2 default


def test_luld_band_tier1_above_3_dollars_is_5pct():
    assert mccf.luld_band_pct(50.0, tier="tier1") == pytest.approx(5.0, abs=1e-9)
    assert mccf.luld_band_pct(50.0, tier="tier2") == pytest.approx(10.0, abs=1e-9)


def test_luld_band_doubles_near_close_for_tier1_and_sub_3_tier2():
    assert mccf.luld_band_pct(50.0, tier="tier1", near_close=True) == pytest.approx(10.0, abs=1e-9)
    assert mccf.luld_band_pct(2.00, tier="tier2", near_close=True) == pytest.approx(40.0, abs=1e-9)


def test_luld_band_does_not_double_near_close_for_tier2_above_3():
    assert mccf.luld_band_pct(50.0, tier="tier2", near_close=True) == pytest.approx(10.0, abs=1e-9)


def test_model_cost_pct_is_imported_live_not_copied():
    report = mccf.build_report({"GALT": 3.12})
    assert report["model_cost_pct_per_side"] == pytest.approx(backtest_v2._ILLIQUID_COST_PCT * 100.0)


def test_undercharged_flag_matches_tick_floor_vs_model_comparison():
    # SNOA $1.29 -> tick floor 0.5/1.29*100 = 0.388%, well above the live
    # 0.185% model cost -> must be flagged undercharged.
    report = mccf.build_report({"SNOA": 1.29})
    row = report["rows"][0]
    assert row["tick_floor_pct"] > report["model_cost_pct_per_side"]
    assert row["model_undercharges"] is True
    assert "SNOA" in report["undercharged_tickers"]


def test_high_priced_name_is_not_undercharged():
    # PROF $7.34 -> tick floor 0.068%, well below the model's 0.185% floor.
    report = mccf.build_report({"PROF": 7.34})
    row = report["rows"][0]
    assert row["model_undercharges"] is False
    assert "PROF" not in report["undercharged_tickers"]


def test_sub_penny_names_excluded_from_undercharged_count_not_silently_dropped():
    report = mccf.build_report({"CISO": 0.26, "GALT": 3.12})
    assert "CISO" in report["sub_penny_tickers"]
    assert report["n_priced"] == 1  # only GALT is priceable
    assert len(report["rows"]) == 2  # but both rows are still reported


def test_pinned_snapshot_covers_the_exact_2026_08_11_illiquid_and_sub5_moderate_tickers():
    # Reuses illiquid_universe_probe's own pinned lists by identity — this
    # test would fail if the snapshot silently drifted from the tickers the
    # live ladder-step-5 run actually evaluated.
    import illiquid_universe_probe as orig
    illiquid_priced = set(mccf.PRICE_SNAPSHOT) & set(orig.ILLIQUID)
    assert illiquid_priced == set(orig.ILLIQUID)


def test_default_report_runs_end_to_end():
    report = mccf.build_report()
    assert report["n_priced"] + report["n_sub_penny_unpriceable"] == len(report["rows"])
    assert len(report["rows"]) == len(mccf.PRICE_SNAPSHOT)


def test_format_report_is_nonempty_and_mentions_flagged_tickers():
    report = mccf.build_report()
    text = mccf.format_report(report)
    assert "UNDERCHARGED" in text
    for t in report["undercharged_tickers"]:
        assert t in text
