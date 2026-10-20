"""scripts/floor_basket_vs_index.py — the pure parts (research test 4,
[RULE-REVIEW] 2026-10-20): the live floor policy's targets, the trend
override, the rebalance mechanics and the statistics. The run itself is the
evidence (research/floor_basket_vs_index_<date>.md), not a test."""
import importlib.util
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fbi", os.path.join(HERE, "scripts", "floor_basket_vs_index.py"))
fbi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fbi)
emr = fbi._load_emr()


def test_targets_follow_the_live_regime_schedule_trend_caps_and_the_gld_leg():
    t = fbi.targets("BULL", None, fbi.ARMS["basket"])
    assert abs(t["QQQ"] - 0.35) < 1e-9 and abs(t["SMH"] - 0.105) < 1e-9 and abs(t["KWEB"] - 0.07) < 1e-9 and abs(t["VXUS"] - 0.105) < 1e-9
    assert abs(t["GLD"] - 0.08) < 1e-9
    assert abs(sum(t.values()) - (0.70 * 0.90 + 0.08)) < 1e-9, "the basket's CASH slice stays cash"
    assert fbi.targets("BEAR", None, fbi.ARMS["basket"]) == {"GLD": 0.30}
    assert fbi.targets("PANIC", None, fbi.ARMS["qqq_only"]) == {"GLD": 0.40}
    assert abs(fbi.targets("NEUTRAL", None, fbi.ARMS["qqq_only"])["QQQ"] - 0.90) < 1e-9
    assert abs(fbi.targets("BULL", "early", fbi.ARMS["qqq_only"])["QQQ"] - 0.50) < 1e-9, "early warning caps the floor at .50"
    assert abs(fbi.targets("NEUTRAL", "death", fbi.ARMS["spy_core"])["SPY"] - 0.20 * 0.65) < 1e-9, "death cross caps it at .20"
    assert abs(fbi.targets("CAUTION", "early", fbi.ARMS["basket"])["QQQ"] - 0.35 * 0.5) < 1e-9, "a cap above the target changes nothing"
    assert fbi.targets("BEAR", "death", fbi.ARMS["basket"], buy_hold=True) == {"SPY": 1.0}
    for name, w in fbi.ARMS.items():
        assert abs(sum(w.values()) - (1.0 if name == "qqq_only" else 0.90)) < 1e-9, name


def test_qqq_trend_needs_200_bars_and_classifies_early_warning_vs_death_cross():
    flat = [100.0] * 260
    assert fbi.qqq_trend(flat, 150) is None
    assert fbi.qqq_trend(flat, 259) is None
    # a long decline: price below both MAs and the 50d below the 200d
    down = [300.0 - i for i in range(260)]
    assert fbi.qqq_trend(down, 259) == "death"
    # a rise then a sharp dip: price below the 200d (104.3) while the 50d (113.2) is still above it
    dip = [100.0] * 200 + [120.0] * 40 + [103.0] * 20
    assert fbi.qqq_trend(dip, 259) == "early"
    # the same dip twenty days deeper into a decline pulls the 50d under the 200d
    assert fbi.qqq_trend([100.0] * 240 + [100.0 - 0.1 * i for i in range(1, 21)], 259) == "death"


def _prices(n, **series):
    base = {s: [100.0] * n for s in ("SPY", "QQQ", "SMH", "KWEB", "VXUS", "GLD")}
    base.update(series)
    return base


def test_simulate_holds_the_targets_earns_the_next_days_return_and_rebalances_on_drift_or_regime_change():
    n = 6
    qqq = [100.0, 100.0, 200.0, 200.0, 200.0, 200.0]        # doubles on day 2
    prices = _prices(n, QQQ=qqq)
    regimes = ["NEUTRAL"] * 3 + ["BEAR"] * 3
    s = fbi.simulate(prices, regimes, fbi.ARMS["qqq_only"], cost_bps=0.0)
    eq = s["equity"]
    assert eq[0] == 1.0 and abs(eq[1] - 1.0) < 1e-12
    assert abs(eq[2] - 1.90) < 1e-9, "90% in QQQ (NEUTRAL) doubled: 0.10 + 0.90*2"
    assert abs(eq[3] - 1.90) < 1e-9
    # day 2: QQQ is now 1.80/1.90 = 94.7% of equity vs the 90% target: drift > 3% → rebalanced back; BEAR at day 3 → QQQ sold, GLD 30%
    assert s["rebalances"] == 3, "day 0 (initial), day 2 (drift), day 3 (regime change)"
    with_cost = fbi.simulate(prices, regimes, fbi.ARMS["qqq_only"], cost_bps=10.0)
    assert with_cost["equity"][-1] < eq[-1] and with_cost["costs"] > 0
    assert len(s["alloc_regime"]) == n and s["alloc_regime"][3] == "BEAR"
    bh = fbi.simulate(prices, regimes, {"SPY": 1.0}, buy_hold=True, cost_bps=0.0)
    assert bh["rebalances"] == 1 and abs(bh["equity"][-1] - 1.0) < 1e-12


def test_statistics_on_known_series():
    r = [0.01, -0.005, 0.02, 0.0, 0.01, -0.01, 0.015, 0.005]
    beta, alpha = fbi.beta_alpha(r, r)
    assert abs(beta - 1.0) < 1e-12 and abs(alpha) < 1e-12
    te, ir = fbi.te_ir(r, r)
    assert te == 0.0 and math.isnan(ir)
    assert abs(fbi.corr(r, r) - 1.0) < 1e-12
    diff, lo, hi = fbi.lw_sharpe_diff_ci(r, r)
    assert abs(diff) < 1e-12 and abs(lo) < 1e-9 and abs(hi) < 1e-9, "identical series: a zero-width interval at 0"
    eq = [1.0, 1.1, 1.0, 1.2]
    assert abs(fbi.max_drawdown(eq) - (1.0 / 1.1 - 1.0)) < 1e-12
    assert abs(fbi.cagr([1.0] * 253) - 0.0) < 1e-12
    assert abs(fbi.cagr([1.0, 2.0]) - (2.0 ** 252 - 1.0)) < 1e-6
    s = fbi.sharpe_annual([0.01] * 10)
    assert math.isnan(s), "zero-variance returns have no Sharpe"
    br = fbi.by_regime([1.0, 1.01, 1.02, 1.0], ["BULL", "BULL", "BEAR", "BEAR"])
    assert br["BULL"]["days"] == 2 and br["BEAR"]["days"] == 1 and "CAUTION" not in br


def test_align_carries_a_missing_close_forward_and_starts_every_arm_on_the_same_day():
    bars = {
        "SPY": {"date": ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"], "close": [1, 2, 3, 4]},
        "QQQ": {"date": ["2026-01-05", "2026-01-07"], "close": [10, 12]},
        "SMH": {"date": ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"], "close": [5, 5, 5, 5]},
        "KWEB": {"date": ["2026-01-05", "2026-01-06", "2026-01-07"], "close": [7, 7, 7]},
        "VXUS": {"date": ["2026-01-05", "2026-01-06", "2026-01-07"], "close": [8, 8, 8]},
        "GLD": {"date": ["2026-01-05", "2026-01-06", "2026-01-07"], "close": [9, 9, 9]},
    }
    dates, prices, carried = fbi.align(bars)
    assert dates == ["2026-01-05", "2026-01-06", "2026-01-07"], "starts at the latest first close"
    assert prices["QQQ"] == [10.0, 10.0, 12.0] and carried["QQQ"] == 1
    assert prices["SPY"] == [2.0, 3.0, 4.0] and carried["SPY"] == 0


def test_run_and_render_state_the_prior_the_arms_and_the_honesty_section():
    n = 300
    import random
    rng = random.Random(3)
    def walk(drift):
        p = [100.0]
        for _ in range(n - 1):
            p.append(p[-1] * (1 + drift + rng.gauss(0, 0.01)))
        return p
    bars = {}
    dates = [f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n)]
    for s, d in (("SPY", 0.0004), ("QQQ", 0.0006), ("SMH", 0.0007), ("KWEB", 0.0), ("VXUS", 0.0002), ("GLD", 0.0003)):
        bars[s] = {"date": dates, "close": walk(d)}
    regimes = {d: ("BULL" if i < 200 else "CAUTION") for i, d in enumerate(dates)}
    res = fbi.run(bars, regimes, emr)
    assert set(res["arms"]) == set(fbi.ARMS) | {"spy_same_exposure", "spy_buy_hold"}
    assert res["days"] == n and res["regime_days"]["BULL"] == 200
    assert abs(res["arms"]["spy_buy_hold"]["beta_spy"] - 1.0) < 1e-9
    assert res["arms"]["spy_same_exposure"]["te_pct"] == 0.0
    assert res["dsr"]["n_arms"] == 4 and res["dsr"]["best_arm"] in fbi.ARMS
    md = fbi.render(res, {"date": "2026-10-20", "generated_at": "t", "cost_bps": 5.0, "data_source": "test"})
    assert "PRIOR (stated before the run)" in md and fbi.PRIOR[:30] in md
    for name in fbi.ARMS:
        assert f"| {name} |" in md
    assert "## VERDICT" in md and "HONESTY" in md and "By regime" in md
