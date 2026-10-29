# Pure-computation tests for scripts/edge_p1_dealer_gamma.py (pre-registered
# test P1, research/edge_preregistration.md). Synthetic data only, no network.
# The load-bearing one is the no-lookahead pair: GEX on day t may only be
# compared against SPY returns t+1..t+5, and its rank may only use GEX <= t.
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import edge_p1_dealer_gamma as p1  # noqa: E402


def _spy(n, seed=0):
    rng = np.random.default_rng(seed)
    closes = list(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    return {"date": [f"D{i:05d}" for i in range(n)], "close": closes,
            "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes]}


def test_forward_realized_var_uses_only_t_plus_1_to_t_plus_5():
    closes = [100.0, 101.0, 100.0, 102.0, 101.0, 103.0, 104.0, 90.0]
    lr = p1.log_returns(closes)
    assert lr[0] is None and abs(lr[1] - math.log(1.01)) < 1e-12
    want = sum(math.log(closes[j] / closes[j - 1]) ** 2 for j in range(2, 7)) * 252 / 5
    assert abs(p1.forward_realized_var(lr, 1) - want) < 1e-12
    # a shock on day t (index 1) or day t+6 (index 7) must not move RV_t
    lr2 = list(lr)
    lr2[1] = 5.0
    lr2[7] = 5.0
    assert abs(p1.forward_realized_var(lr2, 1) - want) < 1e-12
    # window running past the data -> None, never a short window
    assert p1.forward_realized_var(lr, 3) is None


def test_trailing_realized_var_uses_only_data_through_t():
    closes = [100.0, 101.0, 100.0, 102.0, 101.0, 103.0, 104.0]
    lr = p1.log_returns(closes)
    want = sum(lr[j] ** 2 for j in range(1, 6)) * 252 / 5
    assert abs(p1.trailing_realized_var(lr, 5) - want) < 1e-12
    assert p1.trailing_realized_var(lr, 4) is None


def test_parkinson_constant_range():
    h, l = [101.0] * 7, [99.0] * 7
    one = math.log(101 / 99) ** 2 / (4 * math.log(2))
    assert abs(p1.forward_parkinson_var(h, l, 0) - one * 252) < 1e-12
    assert p1.forward_parkinson_var(h, l, 2) is None


def test_implied_var_from_vix():
    assert abs(p1.implied_var_from_vix(20.0) - 0.04) < 1e-15
    # constant 20% daily-equivalent vol realized == VIX 20 -> log ratio 0
    d = 0.20 / math.sqrt(252)
    lr = [None] + [d, -d, d, -d, d]
    assert abs(math.log(p1.forward_realized_var(lr, 0) / p1.implied_var_from_vix(20.0))) < 1e-12


def test_trailing_pct_rank_burn_in_bounds_and_ties():
    r = p1.trailing_pct_rank([1, 2, 3, 4, 5], window=3)
    assert r[:2] == [None, None]
    assert r[2:] == [1.0, 1.0, 1.0]
    assert p1.trailing_pct_rank([5, 4, 3], window=3)[2] == 0.0
    assert p1.trailing_pct_rank([2, 2, 2], window=3)[2] == 0.5


def test_rank_ignores_future_gex():
    vals = list(np.random.default_rng(1).normal(size=300))
    a = p1.trailing_pct_rank(vals, window=252)
    b = p1.trailing_pct_rank(vals[:270] + [1e9] * 30, window=252)
    assert a[:270] == b[:270]


def test_build_panel_no_lookahead_end_to_end():
    n = 400
    spy = _spy(n)
    gex = [(d, float(i % 37)) for i, d in enumerate(spy["date"])]
    vix = {d: 18.0 for d in spy["date"]}
    rows, counts = p1.build_panel(spy, gex, vix, end="D99999")
    k = 300
    base = {r["date"]: r for r in rows}[spy["date"][k]]
    assert base["R"] is not None and base["idx"] == k
    # perturb GEX after t and SPY after t+5: row t unchanged
    gex2 = [(d, (1e6 if i > k else g)) for i, (d, g) in enumerate(gex)]
    spy2 = dict(spy, close=spy["close"][:k + 6] + [c * 3 for c in spy["close"][k + 6:]])
    r2 = {r["date"]: r for r in p1.build_panel(spy2, gex2, vix, end="D99999")[0]}[spy["date"][k]]
    assert r2["R"] == base["R"] and abs(r2["y"] - base["y"]) < 1e-12
    # perturbing SPY at t+5 MUST move y_t (the window really reaches t+5)
    spy3 = dict(spy, close=spy["close"][:k + 5] + [c * 3 for c in spy["close"][k + 5:]])
    r3 = {r["date"]: r for r in p1.build_panel(spy3, gex, vix, end="D99999")[0]}[spy["date"][k]]
    assert abs(r3["y"] - base["y"]) > 0.1
    # last usable t has a full forward window; burn-in rows carry no rank
    assert rows[-1]["idx"] == n - 1 - p1.HORIZON
    assert counts["skip_no_rank"] == p1.RANK_WINDOW - 1
    # R_lag1 is the previous day's rank
    prev = {r["date"]: r for r in rows}[spy["date"][k - 1]]
    assert base["R_lag1"] == prev["R"]


def test_end_cap_truncates_forward_windows():
    spy = _spy(300)
    gex = [(d, float(i)) for i, d in enumerate(spy["date"])]
    vix = {d: 18.0 for d in spy["date"]}
    rows, _ = p1.build_panel(spy, gex, vix, end=spy["date"][280])
    assert rows[-1]["idx"] == 280 - p1.HORIZON


def test_non_overlapping_windows_never_share_days():
    rows = [{"idx": i} for i in range(3, 40)]
    for off in range(5):
        sub = p1.non_overlapping(rows, 5, off)
        ids = [r["idx"] for r in sub]
        assert all(b - a >= 5 for a, b in zip(ids, ids[1:]))
        assert all((i - off) % 5 == 0 for i in ids)


def test_ols_nw_recovers_slope_and_lag0_equals_white():
    rng = np.random.default_rng(3)
    x = rng.uniform(size=2000)
    y = 1.0 - 0.5 * x + rng.normal(0, 0.1, 2000)
    X = [[1.0, v] for v in x]
    f = p1.ols_nw(y, X, lag=5)
    assert abs(f["beta"][1] + 0.5) < 0.03 and f["t"][1] < -20
    # lag 0 == White heteroskedasticity-robust SE
    Xm = np.array(X)
    e = y - Xm @ np.array(f["beta"])
    inv = np.linalg.inv(Xm.T @ Xm)
    white = np.sqrt(np.diag(inv @ (Xm.T * e ** 2) @ Xm @ inv))
    assert np.allclose(p1.ols_nw(y, X, lag=0)["se"], white)
    # gappy time index: rows 10 days apart never get paired at lag 5
    gappy = p1.ols_nw(y[:200], X[:200], lag=5, time_index=[10 * i for i in range(200)])
    assert np.allclose(gappy["se"], p1.ols_nw(y[:200], X[:200], lag=0)["se"])


def test_verdict_maps_preregistered_rule():
    f = lambda b, t: {"b": b, "b_t": t}  # noqa: E731
    assert p1.verdict(f(-0.3, -3.0), f(-0.2, -2.0))[0] == "PASS"
    assert p1.verdict(f(-0.3, -2.4), f(-0.2, -2.0))[0] == "FAIL"
    assert p1.verdict(f(0.3, 3.0), f(0.2, 2.0))[0] == "FAIL"
    assert p1.verdict(f(-0.3, -3.0), f(0.1, 0.5))[0] == "FAIL"      # holdout sign flip
    assert p1.verdict(f(-0.3, -3.0), f(-0.1, -1.0))[0] == "INCONCLUSIVE"


def test_base_rate_and_quintiles():
    rows = [{"R": i / 99, "y": (-1.0 if i % 4 else 0.5)} for i in range(100)]
    br = p1.base_rate(rows)
    assert br["n"] == 100 and abs(br["share_rv_gt_iv"] - 0.25) < 1e-12
    q = p1.quintile_table(rows)
    assert sum(b["n"] for b in q) == 100
