# Pure-computation tests for scripts/edge_p1_gate3_rule_sim.py (Edge P1 ladder
# gate 3: "no new short premium when R_{t-1} < 0.2" replayed on SPY with the
# live manager exits). Synthetic data only, no network. The load-bearing ones:
# the rule reads the PREVIOUS session's rank (no lookahead), every trade's
# daily P&L path sums exactly to its P&L, and the constants are the live ones.
import json
import math
import os
import re
import sys
from datetime import date, timedelta

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "scripts"))   # appended: never shadow a root module (source_validation)
import edge_p1_dealer_gamma as p1  # noqa: E402
import edge_p1_gate3_rule_sim as g3  # noqa: E402
import csp_manager_rule_sim as sim  # noqa: E402


def _src(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


def _days(n, start=date(2024, 1, 2)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


# ── constants are the live ones ─────────────────────────────────────────────
def test_constants_are_the_live_tier1_and_manager_values():
    src = _src("tiered_strategy.py")
    val = lambda n: float(re.search(rf"^{n}\s*=\s*([\d.]+)", src, re.M).group(1))  # noqa: E731
    assert g3.TARGET_PUT_DELTA == val("T1_TARGET_DELTA")
    # every expiry the sim picks (DTE 35..41) sits inside Tier 1's window
    assert val("T1_TARGET_DTE_MIN") <= g3.MIN_DTE_CAL and g3.MIN_DTE_CAL + 6 <= val("T1_TARGET_DTE_MAX")
    assert g3.RULES is sim.RULES          # the manager constants pinned by test_csp_manager_rule_sim.py
    assert g3.LOW_RANK == p1.GATE3_LOW_RANK == 0.2
    assert (g3.FULL_START, g3.HOLDOUT_START) == ("2012-05-01", "2019-01-01")


# ── pricing / contract selection ────────────────────────────────────────────
def test_strike_for_put_delta_hits_minus_thirty_delta_on_the_dollar_grid():
    S, T, sig = 412.37, 38 / 365, 0.18
    K = g3.strike_for_put_delta(S, T, sig)
    assert K == round(K) and K < S
    assert abs(g3.put_delta(S, K, T, sig) + 0.30) < 0.02


def test_bs_call_parity_and_limits():
    S, K, T, sig, r = 100.0, 103.0, 0.1, 0.25, g3.R_RISK_FREE
    lhs = g3.bs_call(S, K, T, sig) - sim.bs_put(S, K, T, sig)
    assert abs(lhs - (S - K * math.exp(-r * T))) < 1e-9
    assert g3.bs_call(100, 90, 0, 0.3) == 10.0 and g3.bs_call(100, 110, 0, 0.3) == 0.0


def test_strangle_strikes_are_one_sd_out():
    kp, kc = g3.strangle_strikes(400.0, 36 / 365, 0.20)
    m = 0.20 * math.sqrt(36 / 365)
    assert kp == round(400 * math.exp(-m)) and kc == round(400 * math.exp(m)) and kp < 400 < kc


def test_expiry_is_first_friday_at_least_35_days_out():
    assert g3.expiry_for(date(2024, 1, 8)) == date(2024, 2, 16)    # Mon + 35 = Mon -> Fri (39 DTE)
    assert g3.expiry_for(date(2024, 1, 12)) == date(2024, 2, 16)   # Fri + 35 = Fri (35 DTE)
    for k in range(14):
        d = date(2024, 3, 1) + timedelta(days=k)
        e = g3.expiry_for(d)
        assert e.weekday() == 4 and 35 <= (e - d).days <= 41


def test_half_spread_floor_and_pct():
    assert g3.half_spread(0.5) == 0.02 and abs(g3.half_spread(5.0) - 0.05) < 1e-12


# ── one position: accounting identity and the exit order ───────────────────
def _flat_series(n=60, S=400.0, vix=18.0):
    return _days(n), [S] * n, [vix] * n


def test_path_sums_exactly_to_pnl_for_both_books():
    dates, closes, vix = _flat_series()
    rng = np.random.default_rng(5)
    closes = list(400 * np.exp(np.cumsum(rng.normal(0, 0.012, len(dates)))))
    vix = list(18 + np.cumsum(rng.normal(0, 0.6, len(dates))).clip(-8, 30))
    for book in g3.BOOKS:
        for i0 in (0, 3, 9):
            tr = g3.simulate_position(book, i0, dates, closes, vix)
            assert tr is not None
            assert abs(sum(tr["path"].values()) - tr["pnl"]) < 1e-6
            assert min(tr["path"]) == i0 and max(tr["path"]) == tr["exit_i"]
            assert abs(tr["pnl_pct"] - tr["pnl"] / tr["collateral"] * 100) < 1e-12
            assert tr["collateral"] == tr["legs"][0][1] * 100


def test_vol_crush_takes_the_profit_target():
    dates, closes, vix = _flat_series()
    vix = [30.0] + [10.0] * (len(dates) - 1)
    tr = g3.simulate_position("csp", 0, dates, closes, vix)
    assert tr["exit_reason"] == "profit_target" and tr["exit_i"] == 1
    assert tr["pnl"] >= g3.RULES["profit_target_pct"] * tr["credit"]


def test_crash_takes_the_loss_limit():
    dates, closes, vix = _flat_series()
    closes = closes[:1] + [c * 0.85 for c in closes[1:]]
    vix = vix[:1] + [45.0] * (len(dates) - 1)
    tr = g3.simulate_position("csp", 0, dates, closes, vix)
    assert tr["exit_reason"] == "loss_limit" and tr["exit_i"] == 1
    assert tr["pnl"] <= -g3.RULES["loss_limit_mult"] * tr["credit"]


def test_quiet_tape_closes_at_21_dte():
    dates, closes, vix = _flat_series()
    tr = g3.simulate_position("csp", 0, dates, closes, vix)
    exp = date.fromisoformat(tr["exp"])
    assert tr["exit_reason"] == "dte_exit"
    assert (exp - date.fromisoformat(tr["exit_day"])).days <= g3.RULES["dte_exit"]
    prev = date.fromisoformat(dates[tr["exit_i"] - 1])
    assert (exp - prev).days > g3.RULES["dte_exit"]       # the FIRST session at or under 21 DTE


def test_data_ending_before_an_exit_is_censored_not_guessed():
    dates, closes, vix = _flat_series(n=8)
    assert g3.simulate_position("csp", 0, dates, closes, vix) is None
    assert g3.simulate_position("csp", 0, dates, closes, [None] * 8) is None


# ── the rule: previous session's rank, missing -> kept ─────────────────────
def test_rule_skips_strictly_below_threshold_and_keeps_missing():
    assert g3.rule_skips(0.199) and not g3.rule_skips(0.2) and not g3.rule_skips(None)


def test_rule_reads_the_previous_sessions_rank_no_lookahead():
    n = 340
    dates = _days(n, date(2023, 1, 2))
    rng = np.random.default_rng(11)
    closes = list(400 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    spy = {"date": dates, "close": closes, "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes]}
    gex = [(d, float(rng.normal())) for d in dates]
    vixd = {d: 18.0 for d in dates}
    meta = g3.entry_meta_from_panel(p1.build_panel(spy, gex, vixd, end=dates[-1])[0])
    k = 300
    # same-day GEX moves R_t but never the entry on day t (it reads R_{t-1})
    gex2 = [(d, (-1e9 if i == k else g)) for i, (d, g) in enumerate(gex)]
    meta2 = g3.entry_meta_from_panel(p1.build_panel(spy, gex2, vixd, end=dates[-1])[0])
    assert meta2[dates[k]]["R_lag1"] == meta[dates[k]]["R_lag1"]
    assert meta2[dates[k + 1]]["R_lag1"] == 0.0               # ... and moves the NEXT session's entry
    trades, counts = g3.run_book("csp", dates, closes, [18.0] * n, meta2, start=dates[260])
    assert counts["entries"] == len([d for d in dates[260:] if d in meta2])
    by_day = {t["day"]: t for t in trades}
    assert by_day[dates[k + 1]]["skipped"] is True
    for t in trades:
        assert t["skipped"] == g3.rule_skips(meta2[t["day"]]["R_lag1"])


# ── books: ladder, metrics, bootstrap, verdict ──────────────────────────────
def _trade(i0, exit_i, pnl, skipped, collateral=1000.0):
    path = {i0: pnl / 2.0, exit_i: pnl / 2.0}
    return {"i0": i0, "exit_i": exit_i, "pnl": pnl, "pnl_pct": pnl / collateral * 100, "path": path,
            "collateral": collateral, "skipped": skipped, "day": f"D{i0:03d}", "exit_reason": "dte_exit",
            "regime": "BULL"}


def test_ladder_depth_returns_and_deployment():
    trades = [_trade(0, 4, 10.0, False), _trade(2, 6, -20.0, True)]
    L = g3.ladder_depth(trades)
    assert L == (4 + 4) / 6
    r_all = g3.ladder_returns(trades, 8, L, lambda t: True)
    assert abs(sum(r_all) - (10.0 - 20.0) / 1000.0 / L) < 1e-12
    r_on = g3.ladder_returns(trades, 8, L, lambda t: not t["skipped"])
    assert abs(sum(r_on) - 10.0 / 1000.0 / L) < 1e-12
    dep = g3.deployment(trades, 8, L, lambda t: True)
    assert dep[3] == 2 / L and dep[5] == 1 / L and dep[7] == 0.0


def test_portfolio_metrics_drawdown_and_worst_windows():
    r = [0.01, -0.02, -0.01, 0.005] + [0.0] * 30
    m = g3.portfolio_metrics(r)
    assert abs(m["max_dd_pct"] - 3.0) < 1e-9
    assert abs(m["total_pct"] - (-1.5)) < 1e-9 and abs(m["worst_day_pct"] + 2.0) < 1e-12


def test_bootstrap_is_seeded_and_signed():
    rng = np.random.default_rng(2)
    trades = []
    for i in range(400):
        sk = (i // 20) % 5 == 0                      # skipped trades come in blocks
        pnl = (-3.0 if sk else 1.0) + float(rng.normal(0, 0.5))
        trades.append(_trade(i, i + 5, pnl, sk))
    a, b = g3.boot_mean_diff(trades), g3.boot_mean_diff(trades)
    assert a == b and a["point"] < 0 and a["ci95"][1] < 0 and a["p_lt0"] > 0.99
    assert g3.boot_mean_diff(trades[:3]) is None


def _ev(kept_mean, skip_mean, kept_q05, skip_q05, sh_on, sh_off, dd_on, dd_off, ci_hi=-0.1):
    return {"trades_kept": {"n": 100, "mean": kept_mean, "q05": kept_q05},
            "trades_skipped": {"n": 20, "mean": skip_mean, "q05": skip_q05},
            "portfolio_on": {"sharpe": sh_on, "max_dd_pct": dd_on},
            "portfolio_off": {"sharpe": sh_off, "max_dd_pct": dd_off},
            "boot_trade_diff": {"ci95": [ci_hi - 1.0, ci_hi]}}


def test_verdict_maps_the_prestated_rule():
    # skipped trades at least as good AND no worse tail -> HURTS
    assert g3.verdict(_ev(0.1, 0.3, -3.0, -2.0, 0.6, 0.6, 10, 10, ci_hi=0.5))[0] == "HURTS"
    # lower Sharpe AND deeper drawdown with the rule -> HURTS (the 2026-11-06 csp result's shape)
    assert g3.verdict(_ev(0.15, 0.27, -2.7, -3.7, 0.58, 0.63, 17.9, 15.9, ci_hi=0.5))[0] == "HURTS"
    # skipped clearly worse, CI below 0, drawdown no deeper -> HELPS
    assert g3.verdict(_ev(0.3, -0.5, -2.0, -6.0, 0.8, 0.6, 8, 12))[0] == "HELPS"
    # worse skipped mean but CI straddles 0 -> MIXED (not HURTS: ships ON under the human rule)
    assert g3.verdict(_ev(0.15, 0.07, -1.7, -3.4, 0.79, 0.63, 5.5, 12.5, ci_hi=0.27))[0] == "MIXED"
    # forgone premium for a smaller tail and shallower drawdown -> MIXED
    assert g3.verdict(_ev(0.1, 0.3, -2.0, -4.0, 0.55, 0.6, 9, 12, ci_hi=0.5))[0] == "MIXED"
    empty = _ev(0.1, 0.3, -2.0, -4.0, 0.55, 0.6, 9, 12)
    empty["trades_skipped"] = {"n": 0}
    assert g3.verdict(empty)[0] == "MIXED"


# ── loading ─────────────────────────────────────────────────────────────────
def test_load_spy_raw_keeps_the_calendar_and_raw_closes(tmp_path):
    ts0 = 1704205800   # 2024-01-02 14:30Z
    doc = {"chart": {"result": [{
        "timestamp": [ts0, ts0 + 86400, ts0 + 2 * 86400],
        "indicators": {"quote": [{"close": [470.0, None, 472.0], "high": [471.0, 1.0, 473.0],
                                  "low": [469.0, 1.0, 471.0]}],
                       "adjclose": [{"adjclose": [460.0, 1.0, 462.0]}]}}]}}
    fp = tmp_path / "spy.json"
    fp.write_text(json.dumps(doc))
    raw, adj = g3.load_spy_raw(str(fp)), p1.load_spy(str(fp))
    assert raw["date"] == adj["date"] == ["2024-01-02", "2024-01-04"]
    assert raw["close"] == [470.0, 472.0] and adj["close"] == [460.0, 462.0]
