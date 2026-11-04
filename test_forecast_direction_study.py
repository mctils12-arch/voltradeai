"""Tests for scripts/forecast_direction_study.py — the pre-registered daily
direction feature study (Part C, 2026-11-04; research/forecast_direction_study_
2026-11-04.md). Synthetic data only, no network.

Load-bearing: every feature at origin t reads data <= t (the rest-of-day
gap reads the target day's open only); a walk-forward refit trains only on
labels that end before its embargo; the verdict needs all three conditions;
and the committed verdict (no feature passed) is pinned to the code that
would use a feature — forecast.DAILY_DIRECTION_FEATURES.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# APPENDED, never inserted at 0: scripts/ holds a source_validation.py that
# would shadow the root module of the same name for every later test.
sys.path.append(os.path.join(HERE, "scripts"))
import forecast_direction_study as ds  # noqa: E402


def _ohlc(n, seed=0):
    rng = np.random.default_rng(seed)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = c * np.exp(rng.normal(0, 0.004, n))
    return o, c


def test_name_features_are_point_in_time():
    o, c = _ohlc(120, seed=1)
    f = ds.name_features(o, c)
    t = 80
    o2, c2 = o.copy(), c.copy()
    o2[t + 1:] *= 2.0
    c2[t + 1:] *= 3.0
    g = ds.name_features(o2, c2)
    for k in ("F1", "F2", "sd20"):
        assert f[k][t] == g[k][t], k
    assert abs(f["F1"][t] - np.clip(math.log(o[t] / c[t - 1]) / f["sd20"][t], -4, 4)) < 1e-12
    assert abs(f["F2"][t] - np.clip(math.log(c[t] / o[t]) / f["sd20"][t], -4, 4)) < 1e-12


def test_targets_and_the_rest_of_day_gap_line_up():
    o = np.array([99.5, 100.5, 102.5, 101.5, 102.0] + [100.0] * 30)
    c = np.array([100.0, 101.0, 102.0, 101.0, 103.0] + [100.0] * 30)
    f = ds.name_features(o, c)
    assert list(f["y"][:4]) == [1.0, 1.0, 0.0, 1.0]
    assert list(f["y_rod"][:4]) == [1.0, 0.0, 0.0, 1.0]          # C_{t+1} > O_{t+1}
    assert np.isnan(f["y"][-1]) and np.isnan(f["y_rod"][-1])
    # G1 at t is the gap INTO t+1 (known at the t+1 open, used only against y_rod)
    o2, c2 = _ohlc(60, seed=3)
    g = ds.name_features(o2, c2)
    t = 40
    assert abs(g["G1"][t] - np.clip(math.log(o2[t + 1] / c2[t]) / g["sd20"][t], -4, 4)) < 1e-12


def test_breadth_is_the_regime_engine_definition():
    n = 260
    up = np.linspace(100, 200, n)
    down = np.linspace(200, 100, n)
    closes = {f"S{k}": (up if k < 5 else down) for k in range(8)}
    b = ds.breadth_feature(closes)
    assert np.isnan(b[198]) and abs(b[-1] - 2 * (5 / 8 - 0.5)) < 1e-12
    few = {f"S{k}": up for k in range(5)}                          # under 6 members -> None
    assert np.all(np.isnan(ds.breadth_feature(few)))


def test_gex_rank_matches_p1():
    import edge_p1_dealer_gamma as p1   # scripts/ is on the path (appended above)
    vals = list(np.random.default_rng(4).normal(size=300))
    assert ds.trailing_pct_rank(vals) == p1.trailing_pct_rank(vals)


def test_walk_forward_never_trains_past_its_embargo():
    n = 1400
    rng = np.random.default_rng(7)
    x = {"A": rng.normal(size=n)}
    y = {"A": (rng.random(n) < 0.53).astype(float)}
    first, last = 900, 940
    pm, pb, fits = ds.walk_forward({"F": x}, y, ["A"], first, last, n, ["F"])
    assert fits and fits[0]["refit_at"] == first
    y2 = {"A": y["A"].copy()}
    y2["A"][first - ds.EMBARGO:] = 1.0 - y2["A"][first - ds.EMBARGO:]   # labels the first refit must not see
    pm2, pb2, _ = ds.walk_forward({"F": x}, y2, ["A"], first, last, n, ["F"])
    seg = slice(first, first + ds.REFIT_EVERY)
    assert np.allclose(pm["A"][seg], pm2["A"][seg]) and np.allclose(pb["A"][seg], pb2["A"][seg])
    # the base-rate reference is the training up-frequency
    assert abs(pb["A"][first] - y["A"][:first - ds.EMBARGO].mean()) < 1e-12


def test_verdict_needs_holdout_gain_significance_and_selection_sign():
    good = {"d_brier": -1e-4}
    assert ds.verdict(good, {"d_brier": -1e-5}, 0.001) == "PASS"
    assert ds.verdict(good, {"d_brier": +1e-5}, 0.001) == "FAIL"     # selection sign flips
    assert ds.verdict(good, {"d_brier": -1e-5}, ds.P_BAR) == "FAIL"  # not under the Bonferroni bar
    assert ds.verdict({"d_brier": +1e-6}, {"d_brier": -1e-5}, 0.0) == "FAIL"
    assert ds.verdict(good, None, 0.0) == "FAIL"
    assert abs(ds.P_BAR - 0.05 / 14) < 1e-15


def test_bootstrap_p_is_small_only_for_a_consistent_gain():
    D = 600
    cnt = np.full(D, 10.0)
    gain = np.full(D, -1e-3) + np.random.default_rng(1).normal(0, 1e-4, D)
    lo, hi, p = ds.bootstrap_mean(gain, cnt, n_boot=500)
    assert hi < 0 and p == 0.0
    noise = np.random.default_rng(2).normal(0, 1e-3, D)
    _, _, p2 = ds.bootstrap_mean(noise, cnt, n_boot=500)
    assert p2 > ds.P_BAR


def test_committed_verdict_no_feature_ships_and_the_daily_direction_stays_at_the_base_rate():
    """The study's committed verdict is that no feature passed — so the
    daily direction must stay at the shrunk base rate. If forecast.py ever
    grows a daily direction feature, it must come with a study that passes."""
    with open(os.path.join(HERE, "research", "forecast_direction_study_2026-11-04.json")) as fh:
        js = json.load(fh)
    assert js["shipped_features"] == []
    assert all(t.get("verdict") != "PASS" for t in js["tests"].values())
    assert js["spec"]["m_tests"] == 14 and abs(js["spec"]["p_bar"] - 0.05 / 14) < 1e-6
    import forecast
    assert tuple(getattr(forecast, "DAILY_DIRECTION_FEATURES", ())) == tuple(js["shipped_features"])
