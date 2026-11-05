"""Tests for scripts/floor_study.py (the pre-registered floor study, 2026-11-05).

They pin the study's timing (no lookahead), its financing arithmetic, its metrics,
and the committed verdict, so the research write-up cannot drift from the code
or the data it reports."""
import importlib.util
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("floor_study", os.path.join(HERE, "scripts", "floor_study.py"))
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)


def test_lever_ret_cash_and_financing():
    rf = 0.0001
    assert math.isclose(fs.lever_ret(1.0, 0.01, rf), 0.01)
    assert math.isclose(fs.lever_ret(0.5, 0.01, rf), 0.5 * 0.01 + 0.5 * rf)            # idle half earns T-bills
    borrow = rf + fs.FIN_SPREAD / fs.TRADING_DAYS
    assert math.isclose(fs.lever_ret(1.5, 0.01, rf), 1.5 * 0.01 - 0.5 * borrow)        # the extra 0.5x pays rf + spread


def _series(n, seed=7):
    x, out = seed, []
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7fffffff
        out.append((x % 2001 - 1000) / 1e5)          # -1% .. +1%
    return out


def test_v1_exposure_never_sees_the_day_it_is_applied_to():
    r = _series(120)
    rf = [0.0] * len(r)
    _, w = fs.run_v1(r, rf)
    shocked = list(r)
    shocked[60] = -0.20                               # a crash on day 60
    _, w2 = fs.run_v1(shocked, rf)
    assert w2[:61] == w[:61], "the weight held over day 60 (and before) cannot react to day 60's return"
    assert w2[61] < w[61] or w[61] == w2[61] == w[60], "from day 61 the shock may lower exposure"
    assert min(w2[61:61 + fs.V1_WIN]) < max(w[61:61 + fs.V1_WIN]), "a -20% day inside the window cuts exposure"


def test_v1_cap_and_band():
    calm = [0.0005 * (1 if i % 2 else -1) for i in range(60)]     # ~0.8% annualised vol -> capped
    _, w = fs.run_v1(calm, [0.0] * len(calm))
    assert max(w) == fs.V1_CAP
    # band: a change smaller than V1_BAND is not traded
    assert all(abs(w[i] - w[i - 1]) == 0 or abs(w[i] - w[i - 1]) > fs.V1_BAND for i in range(1, len(w)))


def test_v2_signal_uses_close_t_and_never_close_t_plus_1():
    px = [100.0 + i * 0.1 for i in range(260)]                    # steady uptrend: on
    r = fs.returns(px)
    rf = [0.0] * len(r)
    _, w = fs.run_v2(px, r, rf)
    i = 230
    px2 = list(px)
    px2[i + 1] = 1.0                                              # a collapse at close i+1
    r2 = fs.returns(px2)
    _, w2 = fs.run_v2(px2, r2, rf)
    assert w2[i] == w[i] == fs.V2_LEV, "exposure over r[i] (close i -> i+1) is decided at close i"
    assert w2[i + 1] == 0.0, "the next decision sees the collapse and goes to T-bills"


def test_v2_off_means_tbills_and_switching_costs():
    px = [100.0 - i * 0.1 for i in range(260)]                    # downtrend: off after warm-up
    r = fs.returns(px)
    rf = [0.0001] * len(r)
    out, w = fs.run_v2(px, r, rf)
    assert w[-1] == 0.0 and math.isclose(out[-1], rf[-1])


def test_metrics_on_known_paths():
    r = [0.01] * 252
    m = fs.metrics(r, [0.0] * 252)
    assert math.isclose(m["cagr_pct"], round(((1.01 ** 252) - 1) * 100, 2))
    assert m["max_dd_pct"] == 0.0
    dd = fs.metrics([0.1, -0.5, 0.2], [0.0] * 3)
    assert math.isclose(dd["max_dd_pct"], -50.0)


def test_committed_verdict_matches_the_write_up():
    js = json.load(open(os.path.join(HERE, "research", "floor_study_2026-11-05.json")))
    assert js["chosen"] == "V1_volmanaged"
    assert js["verdict"]["V1_volmanaged"]["passes"] is True
    assert js["verdict"]["V2_leverage_rotation"]["passes"] is False
    assert js["verdict"]["V3_constant_1_25x"]["passes"] is False
    assert js["V1_volmanaged"]["sharpe"] > js["B0_SPY"]["sharpe"]
    assert js["V1_volmanaged"]["max_dd_pct"] > js["B0_SPY"]["max_dd_pct"]
    # the pre-registered parameters are the ones the live floor will use
    assert (fs.V1_TARGET, fs.V1_CAP, fs.V1_BAND, fs.V1_WIN) == (0.16, 1.5, 0.10, 20)
    # the robustness grid is reported, and every point beats SPY's Sharpe
    grid = js["robustness_grid_v1"]["grid"]
    assert len(grid) == 15 and all(v["sharpe"] > js["B0_SPY"]["sharpe"] for v in grid.values())
