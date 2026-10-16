"""backtest_v2 exit models (2026-10-16, [RULE-REVIEW] measurement change).

KNOWN BROKEN #48: backtest_v2's exits (1.5xATR stop clamped 2-6%, 2R TP,
regime time stop — a copy of system_config's never-read exit block) are not
the exits the bot runs. The live stock exit engine is server/bot.ts
checkPositionOnTick, mirrored in live_exit_policy.py (same rule set as PR
#1355's reviewed shadow-labeler mirror). These tests pin:
  1. every live exit path through backtest_v2.simulate(exit_model="live") on
     hand-built bars (stop, gap-through-stop, clamp bounds, same-bar
     stop-first, scale-out -> breakeven, bullish phase-4 trail, bearish TP,
     7-calendar-day time stop from a mid-week entry, regime/ATR refresh with
     no lookahead, integer tranches, partial-fill accounting);
  2. exit_model="legacy" (the default) is byte-identical to the
     pre-change engine on identical inputs (sha256 of the full simulate()
     output, captured from origin/main before the change);
  3. live_exit_policy matches #1355's shadow_portfolio mirror (pinned values
     from that PR's own tests, plus a fuzz comparison whenever
     shadow_portfolio carries the mirror);
  4. the mirrored literals still exist in server/bot.ts / bot_engine.py.
Every test here fails on the pre-change code (exit_model, live_exit_policy
and live_atr_pct do not exist there).
"""
import hashlib
import json
import os
import sys
from datetime import date, datetime, timedelta

import pytest

import backtest_v2 as bt
import live_exit_policy as lep

HERE = os.path.dirname(os.path.abspath(__file__))


# ── deterministic synthetic series (fixed LCG: identical on every Python) ────
class _LCG:
    """Knuth MMIX constants; uniform() in [0, 1)."""

    def __init__(self, seed: int):
        self.s = (seed * 2862933555777941757 + 3037000493) % (1 << 64)

    def uniform(self) -> float:
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (self.s >> 11) / float(1 << 53)

    def normal(self) -> float:
        # Irwin-Hall(12) - 6: deterministic, ~N(0,1), no transcendental calls
        return sum(self.uniform() for _ in range(12)) - 6.0


def weekday_dates(n: int, start: date = date(2021, 1, 4)) -> list:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def synth_bars(n: int = 900, seed: int = 7, drift: float = 0.0006,
               vol: float = 0.018, start_px: float = 50.0,
               volume: float = 6_000_000) -> dict:
    """Random-walk daily bars on a weekday calendar with overnight gaps and
    intrabar ranges (so stops, gaps, scale-outs and trails all occur)."""
    rng = _LCG(seed)
    dates = weekday_dates(n)
    o, h, lo, c, v = [], [], [], [], []
    px = start_px
    for _ in range(n):
        gap = rng.normal() * vol * 0.35
        op = px * (1 + gap)
        cl = op * (1 + drift + rng.normal() * vol)
        hi = max(op, cl) * (1 + abs(rng.normal()) * vol * 0.5)
        low = min(op, cl) * (1 - abs(rng.normal()) * vol * 0.5)
        o.append(round(op, 4)); c.append(round(cl, 4))
        h.append(round(hi, 4)); lo.append(round(low, 4))
        # volume swells on down days (capitulation), so mean-reversion's
        # volume-ratio leg can fire; deterministic like everything else here
        v.append(round(volume * (1 + max(0.0, op / cl - 1) / vol * 0.9)))
        px = cl
    return {"date": dates, "open": o, "high": h, "low": lo, "close": c, "volume": v}


def synth_vxx(dates: list, seed: int = 11, calm_days: int = 120,
              spike_days: int = 12, spike_rate: float = 0.02) -> dict:
    """VXX-like series: calm decays broken by periodic spikes, so BULL and
    CAUTION (the two entry-permitting regimes) both occur."""
    rng = _LCG(seed)
    px, out = 30.0, []
    cycle = calm_days + spike_days
    for i in range(len(dates)):
        phase = i % cycle
        if phase < calm_days:
            px *= 1 - 0.004 + rng.normal() * 0.01
        else:
            px *= 1 + spike_rate + rng.normal() * 0.02
        out.append(round(px, 4))
    return {"date": list(dates), "open": out, "high": out, "low": out,
            "close": out, "volume": [1_000_000] * len(dates)}


def _fixture_set():
    spy = synth_bars(900, seed=3, drift=0.0009, vol=0.009, start_px=300)
    vxx = synth_vxx(spy["date"])
    return spy, vxx


# ── hand-built single-trade scenarios through the REAL simulate() ───────────
HISTORY = 30   # flat bars at 100 (+/-1): Wilder ATR == 2.0 exactly


def _case(monkeypatch, path, regimes=None, atr=2.0, cost=0.0,
          start=date(2021, 1, 4), exit_model="live"):
    """One entry: score 100 on the last history bar only -> fills at the open
    of path[0]. path = [(o, h, l, c), ...]; a trailing bar at the last close
    is appended (simulate never walks bar n-1, it only closes there).
    regimes: per-bar labels (default all BULL). atr: a fixed live ATR% (None
    = the real live_atr_pct). cost: per-side cost (0 isolates the exits)."""
    rows = [(100.0, 101.0, 99.0, 100.0)] * HISTORY + list(path) + [(path[-1][3],) * 4]
    n = len(rows)
    dates = weekday_dates(n, start)
    bars = {"date": dates, "open": [r[0] for r in rows], "high": [r[1] for r in rows],
            "low": [r[2] for r in rows], "close": [r[3] for r in rows],
            "volume": [30_000_000] * n}
    labels = list(regimes) if regimes is not None else ["BULL"] * n
    assert len(labels) == n
    monkeypatch.setattr(bt, "regime_series", lambda spy, vxx: (labels, "ok"))
    monkeypatch.setattr(bt, "score_candidate",
                        lambda strategy, b, i: 100 if i == HISTORY - 1 else 0)
    monkeypatch.setattr(bt, "liquidity_cost_pct", lambda b, i: cost)
    if atr is not None:
        monkeypatch.setattr(bt, "live_atr_pct", lambda b, k: atr)
    out = bt.simulate("TEST", "momentum", bars, bars, None, exit_model=exit_model)
    return out, bars


def _only_trade(out):
    assert out["n_trades"] == 1, out["trades"]
    return out["trades"][0]


def test_phase1_stop_fills_at_the_2x_atr_level(monkeypatch):
    out, _ = _case(monkeypatch, [(100, 100.5, 99.0, 99.5), (99.5, 100, 95.0, 96.5)])
    t = _only_trade(out)
    assert (t["exit_reason"], t["net_pct"], t["holding_days"]) == ("stop_loss", -4.0, 1)
    assert t["exit_price"] == 96.0 and t["scale_outs"] == 0


def test_gap_through_the_stop_fills_at_the_open_not_the_stop(monkeypatch):
    out, _ = _case(monkeypatch, [(100, 100.5, 99.0, 99.5), (90.0, 91.0, 89.0, 90.5)])
    t = _only_trade(out)
    assert (t["exit_reason"], t["exit_price"], t["net_pct"]) == ("stop_loss", 90.0, -10.0)


@pytest.mark.parametrize("atr,stop_pct", [
    (0.5, 1.5),   # 2 x 0.5 = 1.0 -> clamped UP to 1.5
    (6.0, 8.0),   # 2 x 6.0 = 12  -> clamped DOWN to 8.0
    (3.0, 6.0),   # inside the clamp: 2 x ATR
])
def test_phase1_stop_clamp_at_both_bounds(monkeypatch, atr, stop_pct):
    low = 100 - stop_pct - 1.0
    out, _ = _case(monkeypatch, [(100, 100.2, low, low + 0.5)], atr=atr)
    t = _only_trade(out)
    assert (t["exit_reason"], t["net_pct"], t["holding_days"]) == ("stop_loss", -stop_pct, 0)
    p = lep.LivePosition(100.0, atr, "bullish")
    assert p.p1_stop == stop_pct and p.risk == 2 * atr, "R stays UNclamped 2 x ATR%"


def test_same_bar_stop_and_profit_resolves_stop_first(monkeypatch):
    """O -> L -> H -> C: the adverse leg first, so a bar touching both the stop
    and 1R is a stop — the same priority the legacy close check gives stops."""
    out, _ = _case(monkeypatch, [(100, 105.0, 95.0, 104.0)])
    t = _only_trade(out)
    assert (t["exit_reason"], t["net_pct"], t["scale_outs"]) == ("stop_loss", -4.0, 0)


def test_scale_out_at_1r_then_breakeven_on_the_rest(monkeypatch):
    # ATR 0.6 -> R 1.2%; phase-2 trail max(1.5, 0.9)=1.5 sits below entry, so breakeven binds
    out, _ = _case(monkeypatch, [(100, 101.3, 99.9, 101.0), (100.8, 100.9, 99.0, 99.5)], atr=0.6)
    t = _only_trade(out)
    assert (t["exit_reason"], t["scale_outs"]) == ("breakeven_stop", 1)
    # 150 shares: 50 at +1.2%, 100 at ~0 -> +60 on a 15,000 position
    assert t["pnl"] == pytest.approx(60.0, abs=0.01) and t["net_pct"] == 0.4
    assert out["final_capital"] == pytest.approx(bt.STARTING_CAPITAL + 60.0, abs=0.01)


def test_bullish_runner_trails_3x_atr_from_the_highest_price(monkeypatch):
    path = [(100, 104.5, 100, 104), (104, 113, 103.5, 112.5),
            (113, 130, 112, 129), (128, 128.5, 120, 121)]
    out, bars = _case(monkeypatch, path)
    t = _only_trade(out)
    # 50 sh at 1R (+4), 50 at 3R (+12), last 50 trail 3 x 2% = 6% below the 130 high -> 122.2
    assert (t["exit_reason"], t["scale_outs"]) == ("trailing_stop", 2)
    assert t["pnl"] == pytest.approx(50 * 4 + 50 * 12 + 50 * 22.2, abs=0.01)
    assert t["date_exit"] == bars["date"][HISTORY + 3]


def test_bearish_takes_profit_right_after_the_second_scale_out(monkeypatch):
    n = HISTORY + 2 + 1
    out, _ = _case(monkeypatch, [(100, 104.5, 100, 104), (104, 107, 103.5, 106.5)],
                   regimes=["CAUTION"] * n)
    t = _only_trade(out)
    # CAUTION sizes 10% -> 100 shares, thirds of floor(100/3) = 33:
    # 33 at 1R (+4), 33 at 1.5R (+6), then TP = max(4, min(3 x 2, 15)) = 6% on the last 34
    assert (t["exit_reason"], t["scale_outs"]) == ("take_profit", 2)
    assert t["pnl"] == pytest.approx(33 * 4 + 33 * 6 + 34 * 6, abs=0.01)


def test_time_stop_counts_7_calendar_days_from_a_mid_week_entry(monkeypatch):
    """A Wednesday entry puts calendar days 5 and 6 on trading sessions (Mon,
    Tue), so a 5- or 6-day rule would fire there; live fires on day 7 only,
    at that session's first tick (the open)."""
    wed = date(2021, 1, 6)
    drift = [(101.0, 101.3, 100.7, 101.0)] * 7
    out, bars = _case(monkeypatch, [(100, 101.2, 99.8, 101.0)] + drift, start=wed)
    entry_day = bars["date"][HISTORY]
    assert datetime.strptime(entry_day, "%Y-%m-%d").weekday() == 2
    t = _only_trade(out)
    assert t["exit_reason"] == "time_stop" and t["net_pct"] == 1.0
    held = (datetime.strptime(t["date_exit"], "%Y-%m-%d")
            - datetime.strptime(entry_day, "%Y-%m-%d")).days
    assert held == 7 and t["holding_days"] == 5


def test_flat_outside_the_band_is_never_time_stopped(monkeypatch):
    """|pnl| < 2 is required: +3% (under 1R = 4%) is held to the end."""
    out, _ = _case(monkeypatch, [(100, 103.2, 99.8, 103.0)] + [(103.0, 103.3, 102.7, 103.0)] * 15)
    assert _only_trade(out)["exit_reason"] == "end_of_backtest"


def test_regime_refresh_uses_the_prior_bars_label_only(monkeypatch):
    """Live reads stop_state.regime, rewritten every manage cycle. Bar k uses
    the label of bar k-1 (its close is the last thing known at k's open)."""
    path = [(100, 104.5, 100, 104), (104, 107, 103.5, 106.5), (106.5, 106.6, 106.4, 106.5)]
    n = HISTORY + len(path) + 1
    e = HISTORY  # entry bar index
    # CAUTION from the entry bar on: bar e+1 sees CAUTION (bearish, scale 2 at 1.5R, TP 6%)
    early = ["BULL"] * e + ["CAUTION"] * (n - e)
    t = _only_trade(_case(monkeypatch, path, regimes=early)[0])
    assert (t["exit_reason"], t["scale_outs"]) == ("take_profit", 2)
    # CAUTION only from bar e+1 on: bar e+1 still sees bar e's BULL -> no 1.5R scale there
    late = ["BULL"] * (e + 1) + ["CAUTION"] * (n - e - 1)
    t = _only_trade(_case(monkeypatch, path, regimes=late)[0])
    assert t["date_exit"] != t["date_entry"]
    assert t["scale_outs"] == 2 and t["exit_reason"] == "take_profit"
    assert t["date_exit"] == weekday_dates(n)[e + 2], "the bearish exit waits for bar e+2"


def test_entry_atr_comes_only_from_bars_before_the_entry_bar(monkeypatch):
    """Real live_atr_pct: the flat history gives ATR 2.0 -> stop 96 on the
    entry bar, however wide the entry bar itself is."""
    out, _ = _case(monkeypatch, [(100, 100.5, 70.0, 75.0)], atr=None)
    t = _only_trade(out)
    assert (t["exit_reason"], t["net_pct"]) == ("stop_loss", -4.0)


def test_live_atr_pct_is_bot_engine_get_atr_on_prior_bars_only():
    bars = synth_bars(120, seed=5)
    k = 90
    got = bt.live_atr_pct(bars, k)
    mutated = {key: list(v) for key, v in bars.items()}
    for j in range(k, 120):            # bars at and after k must not matter
        mutated["high"][j] *= 3
        mutated["low"][j] *= 0.3
        mutated["close"][j] *= 2
    assert bt.live_atr_pct(mutated, k) == got
    # window: bars dated within 60 calendar days before bar k, first 42 (limit=period*3)
    cutoff = (datetime.strptime(bars["date"][k], "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    win = [j for j in range(k) if bars["date"][j] >= cutoff][:42]
    trs = [max(bars["high"][j] - bars["low"][j], abs(bars["high"][j] - bars["close"][j - 1]),
               abs(bars["low"][j] - bars["close"][j - 1])) for j in win[1:]]
    atr = sum(trs[:14]) / 14
    for tr in trs[14:]:
        atr = (atr * 13 + tr) / 14
    assert got == pytest.approx(atr / bars["close"][k - 1] * 100, rel=1e-12)
    assert bt.live_atr_pct(bars, 5) == lep.ATR_DEFAULT_PCT == 2.0, "too little history -> 2.0"


def test_two_share_position_never_scales_out_integer_tranches():
    """bot.ts: thirdQty = floor(originalQty / 3) must be >= 1 and remainingQty > 1."""
    p = lep.LivePosition(100.0, 2.0, "bullish", original_qty=2)
    assert not lep.walk_bar(p, 100, 104.5, 100, 104, False)
    assert p.scales == 0 and not p.breakeven and p.remaining_qty == 2
    q = lep.LivePosition(100.0, 2.0, "bullish", original_qty=7)
    lep.walk_bar(q, 100, 104.5, 100, 104, False)
    assert (q.scales, q.remaining_qty, q.fills[0][0]) == (1, 5, 2)


def test_costs_apply_per_tranche_on_the_exit_bar(monkeypatch):
    path = [(100, 104.5, 100, 104), (104, 104.2, 101.0, 101.5)]
    out, _ = _case(monkeypatch, path, cost=0.001)
    t = _only_trade(out)
    entry = 100 * 1.001
    assert t["entry_price"] == round(entry, 2)
    # P&L is measured against the cost-inclusive FILL (as live's
    # avg_entry_price): scale 1 at entry*1.04 (R = 4%), then the phase-2 trail
    # (max(1.5, 1.5 x 2) = 3 P&L points below the 104.5 peak) closes the rest
    shares = int(bt.STARTING_CAPITAL * 0.15 / entry)        # 149 -> thirds of 49
    third = shares // 3
    peak = (104.5 / entry - 1) * 100
    proceeds = (third * entry * 1.04 * 0.999
                + (shares - third) * entry * (1 + (peak - 3.0) / 100) * 0.999)
    assert (shares, t["exit_reason"]) == (149, "trailing_stop")
    assert t["pnl"] == pytest.approx(proceeds - shares * entry, abs=0.02)


def test_live_equity_is_marked_after_the_bars_exits(monkeypatch):
    out, _ = _case(monkeypatch, [(100, 100.5, 99.0, 99.5), (99.5, 100, 95.0, 99.0)])
    assert _only_trade(out)["pnl"] == -600.0
    # equity[0] = start, equity[i+1] = bar i -> the stop bar is HISTORY+1
    assert out["final_capital"] == bt.STARTING_CAPITAL - 600.0


def test_unknown_exit_model_is_refused():
    with pytest.raises(ValueError):
        bt.simulate("T", "momentum", synth_bars(50), synth_bars(50), None, exit_model="bogus")
    with pytest.raises(ValueError):
        bt.run_backtest("T", "momentum", 1, bars=synth_bars(50), exit_model="bogus")
    assert bt.DEFAULT_EXIT_MODEL == "legacy" and bt.EXIT_MODELS == ("legacy", "live")


# ── whole-series runs on the synthetic fixtures ─────────────────────────────
# sha256 of json.dumps(simulate(...), sort_keys=True) captured from origin/main
# (66408b5) BEFORE this change: legacy must stay byte-identical.
LEGACY_PINS = {
    (7, "momentum"): ("3d87b5d259108fecf9b2a8b4a6b05ac7db4dfe20d98dbdf9e28e20080ca5fb8b", 32, -0.525, -6.34, 10.38, 43.8),
    (7, "mean_reversion"): ("514b48393bdde3221127f832426134dffa277a567b11925ece936d7ee4bbfcba", 3, -0.068, -0.29, 1.56, 33.3),
    (21, "momentum"): ("8b35f8dbd4d0d7452721520d6123000f4e20c1fb3ef6cf4909cfb2a5230cc48e", 6, -0.56, -3.43, 4.76, 33.3),
    (21, "mean_reversion"): ("be5180f7359bf4e309bb11391f9f1542b142bf60b72076a1ceface2cee11d911", 7, -0.068, -0.48, 1.59, 42.9),
    (99, "momentum"): ("51612e6aa4fcdba4b01462fb4d7cf888690684761ef5a60daa64af9e56d963cd", 20, -0.49, -4.48, 6.58, 45.0),
    (99, "mean_reversion"): ("12d5c22b9bd092bfbb3193c0b12e44b762fa25970d89c5de664de60c7d1c44e7", 4, 0.025, 0.08, 2.46, 50.0),
}


@pytest.mark.parametrize("key", sorted(LEGACY_PINS))
def test_legacy_exit_model_byte_identical_to_the_pre_change_engine(key):
    seed, strategy = key
    spy, vxx = _fixture_set()
    bars = synth_bars(900, seed=seed, vol=0.028)
    sha, n, sharpe, ret, dd, wr = LEGACY_PINS[key]
    for r in (bt.simulate("SYN", strategy, bars, spy, vxx),                       # default
              bt.simulate("SYN", strategy, bars, spy, vxx, exit_model="legacy")):
        assert hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() == sha
        assert (r["n_trades"], r["sharpe"], r["total_return_pct"],
                r["max_drawdown_pct"], r["win_rate_pct"]) == (n, sharpe, ret, dd, wr)
        assert "exit_model" not in r


@pytest.mark.parametrize("seed", [7, 21, 99])
@pytest.mark.parametrize("exit_model", ["legacy", "live"])
def test_cash_accounting_closes_on_every_model(seed, exit_model):
    """final capital == start + the sum of every round trip's pnl (scale-out
    tranches included), so partial fills can neither leak nor double-count."""
    spy, vxx = _fixture_set()
    r = bt.simulate("SYN", "momentum", synth_bars(900, seed=seed, vol=0.028), spy, vxx,
                    exit_model=exit_model)
    assert r["n_trades"] > 0
    total = sum(t["pnl"] for t in r["trades"])
    assert r["final_capital"] == pytest.approx(bt.STARTING_CAPITAL + total, abs=0.01 * r["n_trades"])


def test_live_model_uses_its_own_exit_vocabulary_and_scales_out():
    spy, vxx = _fixture_set()
    r = bt.simulate("SYN", "momentum", synth_bars(900, seed=7, vol=0.028), spy, vxx, exit_model="live")
    assert r["exit_model"] == "live"
    reasons = {t["exit_reason"] for t in r["trades"]}
    assert reasons <= {"stop_loss", "trailing_stop", "breakeven_stop", "take_profit",
                       "time_stop", "end_of_backtest"}
    assert "trailing_stop" in reasons and sum(t["scale_outs"] for t in r["trades"]) > 0
    legacy = bt.simulate("SYN", "momentum", synth_bars(900, seed=7, vol=0.028), spy, vxx)
    assert r["n_trades"] != legacy["n_trades"], "the exit model must actually change the run"


def test_run_backtest_labels_the_exit_model():
    bars = synth_bars(500, seed=9)
    assert bt.run_backtest("T", "momentum", 1, bars=bars)["exit_model"] == "legacy"
    out = bt.run_backtest("T", "momentum", 1, bars=bars, exit_model="live")
    assert out["exit_model"] == "live" and out["results"][0]["exit_model"] == "live"


def test_cli_passes_an_optional_fourth_arg_and_defaults_to_the_engine(monkeypatch, capsys):
    import backtest
    seen = []

    def fake(ticker, strategy, years, **kw):
        seen.append(kw)
        return {"ok": True, "results": [], "all_trades": {}}
    monkeypatch.setattr(bt, "run_backtest", fake)
    monkeypatch.setattr(sys, "argv", ["backtest.py", "SPY", "all", "2"])
    backtest.main()
    monkeypatch.setattr(sys, "argv", ["backtest.py", "SPY", "all", "2", "live"])
    backtest.main()
    assert seen == [{}, {"exit_model": "live"}], "bot.ts's 3-arg call keeps the engine default"
    capsys.readouterr()
    monkeypatch.undo()
    monkeypatch.setattr(sys, "argv", ["backtest.py", "SPY", "all", "2", "bogus"])
    backtest.main()
    resp = json.loads(capsys.readouterr().out)
    assert resp["ok"] is False and "exit_model" in resp["message"]


# ── 3. identity with PR #1355's shadow-labeler mirror ───────────────────────
def _label_like_1355(entry, fwd_bars, atr_pct, cls, entry_date, cap=20):
    """shadow_portfolio._label_exit_aligned's loop, driven by live_exit_policy."""
    pos = lep.LivePosition(entry, atr_pct, cls)
    used, last = 0, None
    for b in fwd_bars:
        if used >= cap:
            break
        o, h, lo, c = b["o"], b["h"], b["l"], b["c"]
        day = b["t"][:10]
        used += 1
        last = (c, day)
        if lep.walk_bar(pos, o, h, lo, c, lep.time_eligible(entry_date, day)):
            return _res(pos, day, used)
    if used < cap or last is None:
        return {"exit_reason": "incomplete_path"}
    pos.max_px, pos.min_px = max(pos.max_px, last[0]), min(pos.min_px, last[0])
    pos._close(last[0], "open_at_cap")
    return _res(pos, last[1], used)


def _res(pos, day, used):
    ret = sum(w * (px - pos.entry) / pos.entry * 100.0 for w, px in pos.fills)
    return {"exit_reason": pos.exit_reason, "return_pct": round(ret, 3) + 0.0,
            "scale_outs": pos.scales, "exit_date": day, "bars_held": used}


def B(day, o, h, lo, c):
    return {"t": f"{day}T04:00:00Z", "o": o, "h": h, "l": lo, "c": c, "v": 1000}


ENTRY = "2026-06-01"  # a Monday — #1355's fixture convention


@pytest.mark.parametrize("bars,atr,cls,reason,ret", [
    # values copied from #1355's test_shadow_exit_aligned.py
    ([B("2026-06-02", 100, 100.5, 95, 96)], 2.0, "neutral", "stop", -4.0),
    ([B("2026-06-02", 90, 91, 89, 90.5)], 2.0, "neutral", "stop", -10.0),
    ([B("2026-06-02", 100, 100.2, 91.0, 91.5)], 6.0, "neutral", "stop", -8.0),
    ([B("2026-06-02", 100, 105, 95, 104)], 2.0, "neutral", "stop", -4.0),
    ([B("2026-06-02", 100.2, 101.3, 100.1, 101.0), B("2026-06-03", 100.8, 100.9, 99.0, 99.5)],
     0.6, "neutral", "breakeven", 0.4),
    ([B("2026-06-02", 100.5, 104.5, 100, 104), B("2026-06-03", 104, 113, 103.5, 112.5),
      B("2026-06-04", 113, 130, 112, 129), B("2026-06-05", 128, 128.5, 120, 121)],
     2.0, "bullish", "trail", round((4 + 12 + 22.2) / 3, 3)),
    ([B("2026-06-02", 100.5, 104.5, 100, 104), B("2026-06-03", 104, 107, 103.5, 106.5)],
     2.0, "bearish", "take_profit", round((4 + 6 + 6) / 3, 3)),
    ([B("2026-06-02", 100.5, 104.5, 100, 104), B("2026-06-03", 104, 109, 103.5, 108.5),
      B("2026-06-04", 108.5, 109.0, 105.0, 105.5)], 2.0, "neutral", "trail", round((4 + 8 + 7) / 3, 3)),
])
def test_matches_the_1355_mirror_on_its_own_fixtures(bars, atr, cls, reason, ret):
    out = _label_like_1355(100.0, bars, atr, cls, ENTRY)
    assert out["exit_reason"] == reason
    assert out["return_pct"] == pytest.approx(ret, abs=1e-3)


def test_matches_the_1355_mid_week_time_stop():
    wed = "2026-06-03"
    days = []
    d = date(2026, 6, 3)
    while len(days) < 6:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days.append(d.isoformat())
    out = _label_like_1355(100.0, [B(x, 101.0, 101.3, 100.7, 101.0) for x in days], 2.0, "neutral", wed)
    assert (out["exit_reason"], out["exit_date"], out["bars_held"], out["return_pct"]) == \
        ("time_stop", "2026-06-10", 5, 1.0)


def _fuzz_paths(n_paths=400, seed=2026):
    rng = _LCG(seed)
    for _ in range(n_paths):
        atr = 0.4 + rng.uniform() * 5.6
        cls = ("bullish", "bearish", "neutral")[int(rng.uniform() * 3)]
        start = date(2026, 6, 1) + timedelta(days=int(rng.uniform() * 5))
        px, bars, d = 100.0, [], start
        while len(bars) < 25:
            d += timedelta(days=1)
            if d.weekday() >= 5:
                continue
            o = px * (1 + rng.normal() * atr / 300)
            c = o * (1 + rng.normal() * atr / 100)
            h = max(o, c) * (1 + abs(rng.normal()) * atr / 200)
            lo = min(o, c) * (1 - abs(rng.normal()) * atr / 200)
            bars.append(B(d.isoformat(), round(o, 3), round(h, 3), round(lo, 3), round(c, 3)))
            px = c
        yield bars, atr, cls, start.isoformat()


def test_fuzz_identical_to_shadow_portfolio_mirror_when_present():
    """#1355's mirror is merged: it MUST exist, and both copies must give the
    same exits on every fuzzed path (the only drift guard until they are
    de-duplicated)."""
    import shadow_portfolio as sp
    assert hasattr(sp, "_label_exit_aligned"), "shadow_portfolio's exit-aligned mirror (#1355) is missing"
    for bars, atr, cls, start in _fuzz_paths():
        theirs = sp._label_exit_aligned(100.0, bars, atr, cls, start)
        ours = _label_like_1355(100.0, bars, atr, cls, start)
        if theirs["exit_reason"] == "incomplete_path":
            assert ours["exit_reason"] == "incomplete_path"
            continue
        for k in ("exit_reason", "return_pct", "scale_outs", "exit_date", "bars_held"):
            assert ours[k] == theirs[k], (k, ours, theirs)


def test_fuzz_exercises_every_exit_path():
    """The fuzz above is only evidence if its paths reach every rule."""
    seen = {_label_like_1355(100.0, b, a, c, s)["exit_reason"] for b, a, c, s in _fuzz_paths()}
    assert {"stop", "trail", "breakeven", "take_profit", "time_stop"} <= seen, seen


# ── 4. drift pin: the mirrored literals must still exist in the live code ───
BOT_TS_LITERALS = [
    "ps.initial_risk_pct || atrPct * 2.0",
    "const atrPct = ps.current_stop_pct ? ps.current_stop_pct : 2.0;",
    'const regime = ps.regime || "NEUTRAL";',
    "else stopPct = Math.max(1.5, Math.min(atrPct * 2.0, 8.0));",
    "const tpPct = phase < 3 ? Math.max(4.0, Math.min(atrPct * 3.0, 15.0)) : 999;",
    "if (peakR >= 3 && pos.phase < 4) pos.phase = 4;",
    "else if (peakR >= 2 && pos.phase < 3) pos.phase = 3;",
    "else if (peakR >= 1 && pos.phase < 2) pos.phase = 2;",
    'const isBullish = regime === "BULL" || regime === "NEUTRAL_BULL";',
    'const isBearish = regime === "BEAR" || regime === "PANIC" || regime === "CAUTION";',
    "const scaleOut1R = 1.0;",
    "const scaleOut2R = isBullish ? 3.0 : (isBearish ? 1.5 : 2.0);",
    "const trailAtrMult = isBullish ? 3.0 : (isBearish ? 1.5 : 2.0);",
    "if (pos.scalesCompleted < 2 && pos.remainingQty > 1) {",
    "const thirdQty = Math.floor(pos.originalQty / 3);",
    "return; // Don't evaluate full exit on the same tick as a scale-out",
    "stopPct = Math.max(1.0, atr * 1.0);",
    "stopPct = Math.max(1.5, atr * 1.5);",
    "const drawdownFromHighPrice = ((pos.highestPrice - currentPrice) / pos.highestPrice) * 100;",
    "const drawdownFromPeak = pos.highestPnl - pnlPct;",
    "if (!shouldStop && breakevenFloor && pnlPct < 0) {",
    "if (pnlPct <= -stopPct) {",
    "if (!shouldStop && pnlPct >= pos.tpPct && pos.phase < 3 && pos.scalesCompleted >= 2) {",
    "if (daysHeld >= 7 && Math.abs(pnlPct) < 2.0) {",
]
BOT_ENGINE_LITERALS = [
    'initial_risk_pct = ps.get("initial_risk_pct", atr_pct * 2.0)',
    "atr_pct = (atr / current * 100) if current > 0 and atr else 2.0",
    '"current_stop_pct": round(atr_pct, 2),',
    "lookback_days = max(60, period * 4)",
    "limit={period * 3}",
]


@pytest.mark.parametrize("literal", BOT_TS_LITERALS)
def test_bot_ts_exit_literals_still_mirrored(literal):
    src = open(os.path.join(HERE, "server", "bot.ts")).read()
    assert literal in src, (
        f"server/bot.ts no longer contains {literal!r}. The live exit policy changed: update "
        "live_exit_policy.py to match, or backtest_v2's exit_model='live' silently stops "
        "describing live exits.")


@pytest.mark.parametrize("literal", BOT_ENGINE_LITERALS)
def test_bot_engine_literals_still_mirrored(literal):
    assert literal in open(os.path.join(HERE, "bot_engine.py")).read(), literal


def test_python_constants_match_the_pinned_literals():
    assert (lep.RISK_ATR_MULT, lep.P1_STOP_ATR_MULT, lep.P1_STOP_MIN_PCT, lep.P1_STOP_MAX_PCT) == (2.0, 2.0, 1.5, 8.0)
    assert (lep.TP_MIN_PCT, lep.TP_ATR_MULT, lep.TP_MAX_PCT) == (4.0, 3.0, 15.0)
    assert (lep.TIME_STOP_DAYS, lep.TIME_STOP_BAND_PCT, lep.SCALE1_R) == (7, 2.0, 1.0)
    assert lep.SCALE2_R == lep.TRAIL4_ATR_MULT == {"bullish": 3.0, "bearish": 1.5, "neutral": 2.0}
    assert (lep.P2_TRAIL_MIN_PCT, lep.P2_TRAIL_ATR_MULT, lep.P3_TRAIL_MIN_PCT, lep.P3_TRAIL_ATR_MULT) == (1.5, 1.5, 1.0, 1.0)
    assert (lep.ATR_PERIOD, lep.ATR_LOOKBACK_DAYS, lep.ATR_LIMIT_BARS, lep.ATR_DEFAULT_PCT) == (14, 60, 42, 2.0)
    assert [lep.regime_class(r) for r in ("BULL", "NEUTRAL_BULL", "NEUTRAL", "CAUTION", "BEAR", "PANIC", None)] == \
        ["bullish", "bullish", "neutral", "bearish", "bearish", "bearish", "neutral"]


def test_live_exit_policy_is_pure():
    """No trading/datacore imports: a measurement mirror must not pull in
    the code it measures (MEASUREMENT INTEGRITY)."""
    src = open(os.path.join(HERE, "live_exit_policy.py")).read()
    imports = [ln.strip() for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert imports == ["from __future__ import annotations", "from datetime import datetime",
                       "from typing import List, Optional, Tuple"], imports
