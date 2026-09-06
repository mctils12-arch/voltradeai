"""
test_nhtsa_gate2_probe.py — battery for scripts/nhtsa_gate2_probe.py's pure
pieces: ticker grouping, calendar-day aggregation/zero-filling, the
calendar->trading-day causal mapping, the forward-return metric, the
per-ticker evaluation pipeline, cross-ticker pooling, and the pre-registered
gate-2 verdict rule. Written before the live run this same session (PROMOTION
RULE discipline: pin the ruler's behavior on synthetic fixtures with a known
right answer before trusting it on real data).
"""
import importlib.util
import os
from datetime import date

_spec = importlib.util.spec_from_file_location(
    "nhtsa_gate2_probe", os.path.join(os.path.dirname(__file__), "scripts", "nhtsa_gate2_probe.py"))
g2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g2)


# ── watchlist grouping ───────────────────────────────────────────────────────

def test_group_by_ticker_pools_multiple_rows():
    vehicles = [
        {"ticker": "TSLA", "make": "tesla", "model": "model 3", "modelYear": 2024},
        {"ticker": "TSLA", "make": "tesla", "model": "model y", "modelYear": 2024},
        {"ticker": "F", "make": "ford", "model": "f-150", "modelYear": 2024},
    ]
    grouped = g2.group_by_ticker(vehicles)
    assert set(grouped.keys()) == {"TSLA", "F"}
    assert len(grouped["TSLA"]) == 2
    assert len(grouped["F"]) == 1


# ── calendar aggregation ─────────────────────────────────────────────────────

def test_aggregate_daily_counts_sums_same_day_dupes():
    dates = [date(2026, 1, 1), date(2026, 1, 1), date(2026, 1, 3)]
    out = g2.aggregate_daily_counts(dates)
    assert out == {date(2026, 1, 1): 2, date(2026, 1, 3): 1}


def test_build_calendar_daily_series_zero_fills_gaps():
    counts = {date(2026, 1, 1): 3, date(2026, 1, 3): 1}
    dates, values = g2.build_calendar_daily_series(counts, date(2026, 1, 1), date(2026, 1, 4))
    assert dates == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    assert values == [3, 0, 1, 0]  # day 2 and day 4 are real zeros, not missing data


def test_build_calendar_daily_series_single_day_range():
    dates, values = g2.build_calendar_daily_series({}, date(2026, 1, 1), date(2026, 1, 1))
    assert dates == ["2026-01-01"]
    assert values == [0]


# ── calendar -> trading-day causal mapping ──────────────────────────────────

def test_map_calendar_to_trading_idx_exact_match():
    trading = ["2026-01-02", "2026-01-05", "2026-01-06"]
    assert g2.map_calendar_to_trading_idx("2026-01-05", trading) == 1


def test_map_calendar_to_trading_idx_weekend_maps_to_next_trading_day():
    # A Saturday complaint (2026-01-03) must map to the NEXT trading day
    # (Monday 2026-01-05), never to a day strictly before it — that would
    # leak a signal backward in time.
    trading = ["2026-01-02", "2026-01-05", "2026-01-06"]
    assert g2.map_calendar_to_trading_idx("2026-01-03", trading) == 1


def test_map_calendar_to_trading_idx_beyond_calendar_is_none():
    trading = ["2026-01-02", "2026-01-05"]
    assert g2.map_calendar_to_trading_idx("2026-01-10", trading) is None


# ── forward return ───────────────────────────────────────────────────────────

def test_forward_return_basic():
    closes = [100.0, 101.0, 102.0, 99.0, 110.0]
    assert round(g2.forward_return(closes, 0, 2), 6) == round(102.0 / 100.0 - 1, 6)


def test_forward_return_none_when_horizon_exceeds_series():
    closes = [100.0, 101.0]
    assert g2.forward_return(closes, 0, 5) is None


def test_forward_return_none_on_zero_entry_price():
    closes = [0.0, 101.0, 102.0]
    assert g2.forward_return(closes, 0, 1) is None


def test_forward_return_none_on_negative_idx():
    closes = [100.0, 101.0]
    assert g2.forward_return(closes, -1, 1) is None


# ── per-ticker evaluation pipeline (synthetic, deterministic) ───────────────

def _synthetic_trading_days(n: int, start_ordinal: int = date(2026, 1, 2).toordinal()) -> list:
    """n consecutive calendar days as ISO strings, standing in for trading
    days (weekends irrelevant to the pure-function contract under test)."""
    return [date.fromordinal(start_ordinal + i).isoformat() for i in range(n)]


def test_evaluate_ticker_separates_spike_from_baseline_forward_returns():
    # 250 calendar days: flat complaint noise (0-1/day) except 5 clean,
    # widely-spaced spikes (count=10, spacing 40 days >> the 5-day horizon
    # so no spike's forward window can bleed into another's), each
    # engineered so the trailing 30-day window is well past its warm-up
    # (first spike at index 80). welch_vs_baseline's own n>=5/side floor
    # (reused unmodified from wikiattention_gate2) needs >=5 spike-day
    # observations, hence 5 spikes rather than 1. Price is flat at 100
    # everywhere except an isolated -10% close exactly 5 days after each
    # spike — a hand-verifiable case where the spike-day sample's forward
    # return must equal exactly -10% and the (untouched, flat) baseline
    # must equal exactly 0%.
    n = 250
    spike_idxs = [80, 120, 160, 200, 240]
    counts = [0 if i % 3 else 1 for i in range(n)]  # low, noisy baseline
    for i in spike_idxs:
        counts[i] = 10  # unambiguous spike, way above trailing baseline
    cal_dates = _synthetic_trading_days(n)
    trading_dates = cal_dates  # 1:1 calendar/trading mapping for this synthetic case
    closes = [100.0] * n
    for i in spike_idxs:
        closes[i + 5] = 90.0  # isolated -10% dip, 5 days after this spike only
    result = g2.evaluate_ticker(cal_dates, counts, trading_dates, closes,
                                 threshold=2.0, window=30, horizons=(5,))
    assert result["n_spike_calendar_days"] == len(spike_idxs)
    fr = result["horizons"][5]["forward_return"]
    assert fr is not None
    assert fr["n"] == len(spike_idxs)
    # spike-day entries (close=100) -> exit 5 days later (close=90): exactly -10%
    assert round(fr["mean"], 4) == -0.1
    # baseline is flat 100->100 everywhere except the 5 "recovery" days
    # immediately after each dip (dip_idx -> dip_idx+5, back to 100: +11.1%,
    # itself a real, correctly-computed forward_return, not a bug) — with
    # ~240 baseline observations diluting 5 such outliers, the baseline mean
    # stays an order of magnitude below the spike sample's -10%.
    assert abs(fr["baseline_mean"]) < 0.01
    assert fr["mean"] < fr["baseline_mean"]


def test_evaluate_ticker_no_spikes_gives_null_forward_return_when_undersized():
    n = 40
    counts = [0] * n  # zero variance -> zscore_series returns None everywhere
    cal_dates = _synthetic_trading_days(n)
    closes = [100.0 + i * 0.1 for i in range(n)]
    result = g2.evaluate_ticker(cal_dates, counts, cal_dates, closes,
                                 threshold=2.0, window=30, horizons=(5,))
    assert result["n_spike_calendar_days"] == 0
    assert result["horizons"][5]["forward_return"] is None  # sample n=0 < 5 floor


# ── pooling ──────────────────────────────────────────────────────────────────

def test_pool_forward_return_combines_raw_samples_across_tickers():
    per_ticker = {
        "AAA": {"horizons": {5: {"_raw": {"sample": [-0.1, -0.12, -0.09, -0.11, -0.1],
                                           "baseline": [0.01] * 10}}}},
        "BBB": {"horizons": {5: {"_raw": {"sample": [-0.08, -0.07, -0.09, -0.1, -0.11],
                                           "baseline": [0.02] * 10}}}},
    }
    pooled = g2.pool_forward_return(per_ticker, ["AAA", "BBB"], [5])
    row = pooled[5]
    assert row["n_tickers_pooled"] == 2
    assert row["forward_return"]["n"] == 10
    assert row["forward_return"]["n_baseline"] == 20
    assert row["forward_return"]["mean_diff"] < 0


def test_pool_forward_return_skips_tickers_with_errors():
    per_ticker = {
        "AAA": {"error": "price fetch failed: boom"},
        "BBB": {"horizons": {5: {"_raw": {"sample": [-0.1, -0.1, -0.1, -0.1, -0.1],
                                           "baseline": [0.0] * 5}}}},
    }
    pooled = g2.pool_forward_return(per_ticker, ["AAA", "BBB"], [5])
    assert pooled[5]["n_tickers_pooled"] == 1


# ── verdict rule ─────────────────────────────────────────────────────────────

def _fake_result(primary_horizons: dict) -> dict:
    return {
        "horizons": list(primary_horizons.keys()),
        "pooled": {
            "primary_edge_group": {h: {"forward_return": v} for h, v in primary_horizons.items()},
            "secondary_mega_comparison": {h: {"forward_return": None} for h in primary_horizons},
        },
    }


def test_gate2_verdict_passes_on_significant_negative_primary_effect():
    result = _fake_result({5: {"p_value": 0.001, "mean_diff": -0.05, "n": 20, "n_baseline": 200}})
    v = g2.gate2_verdict(result)
    assert v["PASS"] is True
    assert len(v["passing_horizons"]) == 1


def test_gate2_verdict_fails_on_wrong_direction_even_if_significant():
    # A significant POSITIVE effect must not pass — the pre-registered rule
    # requires both significance AND the hypothesized negative direction.
    result = _fake_result({5: {"p_value": 0.0001, "mean_diff": 0.05, "n": 20, "n_baseline": 200}})
    v = g2.gate2_verdict(result)
    assert v["PASS"] is False


def test_gate2_verdict_fails_when_not_below_bonferroni_bar():
    result = _fake_result({5: {"p_value": 0.02, "mean_diff": -0.05, "n": 20, "n_baseline": 200}})
    v = g2.gate2_verdict(result)
    assert v["PASS"] is False


def test_gate2_verdict_fails_on_null_result():
    result = _fake_result({5: None, 10: None, 20: None})
    v = g2.gate2_verdict(result)
    assert v["PASS"] is False
    assert v["passing_horizons"] == []
