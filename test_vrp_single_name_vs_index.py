"""scripts/vrp_single_name_vs_index.py — the pure parts (research test 2 gate 2,
[RULE-REVIEW] 2026-10-20): realized variance ahead, Newey–West SE, the
IV30/RV pairing, aggregation and the report. The live run is evidence
(research/vrp_single_name_vs_index_<date>.md), not a test."""
import importlib.util
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vrp", os.path.join(HERE, "scripts", "vrp_single_name_vs_index.py"))
vrp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vrp)


def _bars(closes, start="2026-07-01"):
    from datetime import date, timedelta
    d0 = date.fromisoformat(start)
    return {"date": [(d0 + timedelta(days=i)).isoformat() for i in range(len(closes))], "close": list(closes)}


def test_realized_variance_ahead_is_the_annualized_sum_of_squared_log_returns():
    flat = [100.0] * 30
    assert vrp.realized_var_ahead(flat, 0, 21) == 0.0
    alt = [100.0 * (1.01 if i % 2 else 1.0) for i in range(30)]
    rv2 = vrp.realized_var_ahead(alt, 0, 21)
    r = math.log(1.01)
    assert abs(rv2 - 21 * r * r * 252 / 21) < 1e-12
    assert vrp.realized_var_ahead(alt, 9, 21) is None, "needs 21 bars after i"
    assert vrp.realized_var_ahead([100.0, 0.0, 100.0] + [100.0] * 25, 0, 21) is None, "a zero close is unusable"


def test_newey_west_lag_zero_is_the_plain_se_and_positive_autocorrelation_widens_it():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    plain = math.sqrt(sum((v - 3.5) ** 2 for v in x) / 6 / 6)
    assert abs(vrp.newey_west_se(x, 0) - plain) < 1e-12
    assert vrp.newey_west_se(x, 2) > plain, "a trending series has positive autocovariance"
    assert math.isnan(vrp.newey_west_se([1.0], 2))


def test_pair_joins_iv30_with_the_forward_realized_variance_and_drops_days_without_a_window():
    bars = {"SPY": _bars([100.0 * (1.001 ** i) for i in range(40)]), "AAA": _bars([50.0] * 40)}
    rows = [
        {"u": "SPY", "day": bars["SPY"]["date"][0], "iv30": 0.20, "method": "interp"},
        {"u": "SPY", "day": bars["SPY"]["date"][30], "iv30": 0.20, "method": "interp"},   # only 9 forward bars: dropped
        {"u": "AAA", "day": bars["AAA"]["date"][2], "iv30": 0.30, "method": "nearest"},
        {"u": "AAA", "day": "2026-01-01", "iv30": 0.30, "method": "interp"},               # no such bar day
        {"u": "AAA", "day": bars["AAA"]["date"][3], "iv30": None, "method": None},         # no IV
        {"u": "ZZZ", "day": bars["AAA"]["date"][3], "iv30": 0.5, "method": "interp"},      # no bars
    ]
    pairs = vrp.pair(rows, bars, 21, {bars["SPY"]["date"][0]: "BULL"})
    assert [(p["u"], p["day"]) for p in pairs] == [("SPY", bars["SPY"]["date"][0]), ("AAA", bars["AAA"]["date"][2])]
    spy = pairs[0]
    r = math.log(1.001)
    assert abs(spy["rv"] - math.sqrt(r * r * 252)) < 1e-4  # rv is rounded to 4 dp
    assert abs(spy["vrp_var"] - (0.04 - r * r * 252)) < 1e-5  # vrp_var is rounded to 5 dp
    assert spy["regime"] == "BULL" and pairs[1]["regime"] is None
    assert pairs[1]["rv"] == 0.0 and pairs[1]["vrp_vol"] == 0.3


def test_aggregate_splits_index_from_pooled_singles_per_day_and_by_regime():
    pairs = []
    for i in range(5):
        day = f"2026-08-{i + 1:02d}"
        pairs.append({"u": "SPY", "day": day, "iv30": 0.2, "rv": 0.1, "method": "interp", "vrp_var": 0.03, "vrp_vol": 0.1, "regime": "BULL"})
        pairs.append({"u": "AAA", "day": day, "iv30": 0.4, "rv": 0.5, "method": "interp", "vrp_var": -0.09, "vrp_vol": -0.1, "regime": "BULL" if i < 3 else "CAUTION"})
        pairs.append({"u": "BBB", "day": day, "iv30": 0.4, "rv": 0.3, "method": "nearest", "vrp_var": 0.07, "vrp_vol": 0.1, "regime": "BULL" if i < 3 else "CAUTION"})
    agg = vrp.aggregate(pairs, 21)
    g = agg["groups"]
    assert g["SPY"]["pairs"] == 5 and g["SPY"]["vrp_var"]["mean"] == 0.03 and g["SPY"]["vrp_var"]["se"] == 0.0
    assert g["single_pooled"]["names"] == 2 and g["single_pooled"]["days"] == 5
    assert abs(g["single_pooled"]["vrp_var"]["mean"] - (-0.01)) < 1e-9, "per-day cross-sectional mean first"
    assert g["single_pooled"]["share_iv_above_rv"] == 0.5
    assert agg["per_underlying"]["AAA"]["group"] == "single" and agg["per_underlying"]["SPY"]["group"] == "index"
    assert agg["per_underlying"]["AAA"]["share_iv_above_rv"] == 0.0
    assert set(agg["by_regime"]) == {"BULL", "CAUTION"}
    assert agg["by_regime"]["CAUTION"]["single_pooled"]["days"] == 2
    assert agg["by_regime"]["CAUTION"]["SPY"]["pairs"] == 0
    assert agg["lag"] == 20


def test_render_states_the_prior_the_verdict_and_the_honesty_section():
    pairs = [{"u": "SPY", "day": f"2026-08-{i + 1:02d}", "iv30": 0.2, "rv": 0.1, "method": "interp", "vrp_var": 0.03 + 0.001 * (i % 3), "vrp_vol": 0.1, "regime": None} for i in range(10)]
    agg = vrp.aggregate(pairs, 21)
    md = vrp.render(agg, {"date": "2026-10-20", "generated_at": "t", "iv_rows": 10, "iv_days": 10, "iv_underlyings": 1, "source": "test"})
    assert "PRIOR (stated before the run)" in md and vrp.PRIOR[:30] in md
    assert "## VERDICT" in md and "single-name VRP: no pairs" in md
    assert "| SPY | index |" in md and "HONESTY" in md and "SURVIVORSHIP" in md
