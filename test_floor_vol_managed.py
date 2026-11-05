#!/usr/bin/env python3
"""
The volatility-managed SPY floor (2026-11-05, research/floor_study_2026-11-05.md).

Pins:
- the pure rule (floor_vol_managed.py): the study's estimator, cap, band, and
  the no-margin SPY/SSO split;
- the engine pass (bot_engine._manage_vol_managed_floor), with all broker I/O
  mocked: legacy floor tickers are sold, SPY/SSO are sized to the rule,
  nothing trades outside 09:45-15:54 ET or on top of a working order, and only
  marketable limit orders are sent (never market);
- the wiring: FLOOR_MODE routes the scan, both new tickers are exempt from the
  active-exit machinery (Python and bot.ts), and ownership follows the mode.

Run: python3 -m pytest -q test_floor_vol_managed.py
"""
import math
import os
import re
import sys
from unittest.mock import patch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import floor_vol_managed as fvm
import system_config
import bot_engine

CFG = {"FLOOR_VM_WINDOW": 20, "FLOOR_VM_TARGET_VOL": 0.16, "FLOOR_VM_CAP": 1.5,
       "FLOOR_VM_BAND": 0.10, "FLOOR_VM_CAPITAL_PCT": 0.85}


def _closes_with_daily_move(move: float, n: int = 21, start: float = 500.0):
    """Alternating +move / -move closes: a known sample volatility."""
    out, px = [start], start
    for i in range(n - 1):
        px *= 1 + (move if i % 2 == 0 else -move)
        out.append(px)
    return out


# ── the pure rule ───────────────────────────────────────────────────────────

def test_realized_vol_is_the_studys_estimator():
    closes = _closes_with_daily_move(0.01)
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    mu = sum(rets) / 20
    expect = math.sqrt(sum((r - mu) ** 2 for r in rets) / 19) * math.sqrt(252)
    assert math.isclose(fvm.realized_vol(closes, 20), expect, rel_tol=1e-12)
    assert fvm.realized_vol(closes[:20], 20) is None, "needs window + 1 closes"
    assert fvm.realized_vol(closes[:-1] + [0.0], 20) is None, "a bad close is refused, not guessed"


def test_target_exposure_is_capped_and_scales_down_with_volatility():
    assert fvm.target_exposure(0.08, 0.16, 1.5) == 1.5            # calm: capped
    assert math.isclose(fvm.target_exposure(0.32, 0.16, 1.5), 0.5)  # volatile: half exposure
    assert fvm.target_exposure(0.0, 0.16, 1.5) == 1.5
    assert fvm.target_exposure(None, 0.16, 1.5) is None


def test_holdings_never_use_margin():
    for e in (0.0, 0.4, 1.0, 1.25, 1.5, 2.0, 3.0):
        h = fvm.holdings_for(e, 100_000)
        spent = h["base"] + h["levered"]
        exposure = h["base"] + 2 * h["levered"]
        assert spent <= 100_000 + 1e-6, f"E={e}: spends more than the floor's capital"
        assert math.isclose(exposure, min(e, 2.0) * 100_000, rel_tol=1e-12, abs_tol=1e-6), f"E={e}: wrong exposure"
    assert fvm.holdings_for(0.6, 100_000) == {"base": 60_000.0, "levered": 0.0}
    assert fvm.holdings_for(1.5, 100_000) == {"base": 50_000.0, "levered": 50_000.0}


def test_plan_band_and_legacy():
    calm = _closes_with_daily_move(0.004)          # ~6% vol -> E* capped at 1.5
    held = fvm.holdings_for(1.5, 0.85 * 100_000)
    p = fvm.plan(calm, 100_000, held["base"], held["levered"], 0.0, CFG)
    assert p["ok"] and p["target_exposure"] == 1.5 and not p["rebalance"], "on target: no trade"
    p = fvm.plan(calm, 100_000, held["base"], held["levered"], 5_000.0, CFG)
    assert p["rebalance"] and p["legacy_to_sell"], "a legacy floor ticker held forces the pass"
    p = fvm.plan(calm, 100_000, 0.0, 0.0, 0.0, CFG)
    assert p["rebalance"] and p["target_base_dollars"] == 42_500.0 and p["target_levered_dollars"] == 42_500.0
    small = fvm.plan(calm, 100_000, held["base"] * 0.95, held["levered"], 0.0, CFG)
    assert not small["rebalance"], "a drift inside the 0.10 band is not traded"
    short = fvm.plan(calm[:10], 100_000, 0, 0, 0, CFG)
    assert not short["ok"] and not short["rebalance"], "too little history: hold, never trade blind"


def test_config_is_the_preregistered_v1():
    c = system_config.BASE_CONFIG
    assert c["FLOOR_MODE"] == "vol_managed_spy"
    assert (c["FLOOR_VM_TARGET_VOL"], c["FLOOR_VM_CAP"], c["FLOOR_VM_BAND"], c["FLOOR_VM_WINDOW"]) == (0.16, 1.5, 0.10, 20)
    assert (c["FLOOR_VM_TICKER"], c["FLOOR_VM_LEVERED_TICKER"]) == ("SPY", "SSO")
    assert 0 < c["FLOOR_VM_CAPITAL_PCT"] <= 0.90


# ── the engine pass, broker I/O mocked ──────────────────────────────────────

class _Broker:
    def __init__(self, positions, quotes, working=()):
        self.positions, self.quotes, self.working = positions, quotes, set(working)
        self.orders = []

    def order(self, symbol, qty, side, limit_px):
        self.orders.append((symbol, qty, side, round(limit_px, 2)))
        return {"ok": True, "order_id": f"o{len(self.orders)}", "error": None}


def _run(broker, closes, hour=11.0, regular=True, equity=100_000.0, clock=(True, "open")):
    with patch.object(bot_engine, "_daily_closes_completed", return_value=closes), \
         patch.object(bot_engine, "_floor_vm_broker_clock", return_value=clock), \
         patch.object(bot_engine, "get_alpaca_account", return_value={"equity": str(equity)}), \
         patch.object(bot_engine, "get_alpaca_positions", return_value=broker.positions), \
         patch.object(bot_engine, "_latest_quote", side_effect=lambda s: broker.quotes[s]), \
         patch.object(bot_engine, "_open_order_symbols", return_value=broker.working), \
         patch.object(bot_engine, "_floor_vm_order", side_effect=broker.order), \
         patch.object(bot_engine, "_unwind_covered_calls", return_value=[]), \
         patch.object(bot_engine, "_floor_vm_save_state"), \
         patch.object(bot_engine, "_is_regular_hours", return_value=regular), \
         patch.object(bot_engine, "_et_now_hour", return_value=hour):
        return bot_engine._manage_vol_managed_floor({})


def _pos(sym, qty, px):
    return {"symbol": sym, "qty": str(qty), "qty_available": str(qty), "market_value": str(qty * px)}


QUOTES = {"SPY": (499.9, 500.1), "SSO": (99.95, 100.05), "QQQ": (599.9, 600.1), "SMH": (299.9, 300.1),
          "KWEB": (24.99, 25.01), "VXUS": (69.99, 70.01), "GLD": (379.9, 380.1)}


def test_first_pass_sells_the_old_floor_and_buys_spy_and_sso_at_the_nbbo():
    b = _Broker([_pos("QQQ", 60, 600), _pos("SMH", 24, 300), _pos("GLD", 21, 380), _pos("AAPL", 10, 200)], QUOTES)
    r = _run(b, _closes_with_daily_move(0.004))
    sells = {s: (q, px) for s, q, side, px in b.orders if side == "sell"}
    buys = {s: (q, px) for s, q, side, px in b.orders if side == "buy"}
    assert sells == {"GLD": (21, 379.9), "QQQ": (60, 599.9), "SMH": (24, 299.9)}, "every legacy share, at the bid"
    assert "AAPL" not in sells, "satellites are not the floor's to touch"
    assert buys == {"SPY": (84, 500.1), "SSO": (424, 100.05)}, "42.5k each at the ask (calm market: 1.5x of 85k)"
    assert r["status"] == "ok" and r["target_exposure"] == 1.5


def test_volatile_market_holds_less_and_no_sso():
    b = _Broker([], QUOTES)
    r = _run(b, _closes_with_daily_move(0.02))        # ~32% vol -> E* = 0.5
    assert math.isclose(r["target_exposure"], 0.16 / fvm.realized_vol(_closes_with_daily_move(0.02), 20), rel_tol=1e-3)
    assert [o for o in b.orders if o[0] == "SSO"] == []
    spy = [o for o in b.orders if o[0] == "SPY"][0]
    assert spy[2] == "buy" and abs(spy[1] * 500.1 - r["target_base_dollars"]) < 500.1


def test_nothing_trades_outside_the_window_or_on_a_working_order():
    for hour, regular in ((9.6, True), (15.95, True), (11.0, False)):
        b = _Broker([_pos("QQQ", 60, 600)], QUOTES)
        r = _run(b, _closes_with_daily_move(0.004), hour=hour, regular=regular)
        assert b.orders == [] and r["status"] == "deferred", f"traded at {hour} ET (regular={regular})"
    b = _Broker([_pos("QQQ", 60, 600)], QUOTES, working={"QQQ", "SPY"})
    _run(b, _closes_with_daily_move(0.004))
    assert [o[0] for o in b.orders] == ["SSO"], "QQQ and SPY have orders working: only SSO trades"


def test_the_brokers_clock_gates_every_trade():
    """Weekends, holidays and half-day afternoons: the Tier 2 scan still runs
    (04:00-20:00 ET daily) and _is_regular_hours reads only the time of day."""
    for why in ("broker clock: market closed", "broker clock: 4 min to the close",
                "broker clock unreadable: timeout"):
        b = _Broker([_pos("QQQ", 60, 600)], QUOTES)
        r = _run(b, _closes_with_daily_move(0.004), clock=(False, why))
        assert b.orders == [] and r["status"] == "deferred", why
        assert r["actions"][-1] == {"type": "floor_vm_deferred", "reason": why}


class _Resp:
    def __init__(self, body=None, exc=None):
        self.body, self.exc = body, exc

    def json(self):
        if self.exc:
            raise self.exc
        return self.body


def _clock(body=None, exc=None, get_exc=None):
    def fake_get(*_a, **_k):
        if get_exc:
            raise get_exc
        return _Resp(body, exc)
    with patch.object(bot_engine.requests, "get", side_effect=fake_get):
        return bot_engine._floor_vm_broker_clock()


def test_broker_clock_reading():
    open_mid = {"is_open": True, "timestamp": "2026-11-05T11:02:03.123456789-05:00",
                "next_close": "2026-11-05T16:00:00-05:00"}
    assert _clock(open_mid) == (True, "open")
    half_day_end = {"is_open": True, "timestamp": "2026-11-27T12:55:00-05:00",
                    "next_close": "2026-11-27T13:00:00-05:00"}
    ok, why = _clock(half_day_end)
    assert ok is False and "min to the close" in why
    assert _clock({"is_open": False, "timestamp": "2026-11-07T11:00:00-05:00",
                   "next_close": "2026-11-09T16:00:00-05:00"}) == (False, "broker clock: market closed")
    for bad in (dict(get_exc=bot_engine.requests.ConnectionError("down")),
                dict(body=None, exc=ValueError("not json")),
                dict(body=[{"symbol": "AAPL"}]),
                dict(body={"is_open": True}),
                dict(body={"is_open": True, "timestamp": "garbage", "next_close": "2026-11-05T16:00:00-05:00"})):
        ok, why = _clock(**bad)
        assert ok is False and why.startswith("broker clock unreadable"), (bad, why)


def test_on_target_does_nothing():
    cap = 0.85 * 100_000
    h = fvm.holdings_for(1.5, cap)
    b = _Broker([_pos("SPY", h["base"] / 500.0, 500.0), _pos("SSO", h["levered"] / 100.0, 100.0)], QUOTES)
    r = _run(b, _closes_with_daily_move(0.004))
    assert b.orders == [] and r["rebalance"] is False


# ── wiring ──────────────────────────────────────────────────────────────────

def test_scan_routes_the_floor_through_one_step():
    src = open(os.path.join(_ROOT, "bot_engine.py")).read()
    scan = src[src.index("def _scan_market_inner("):]
    scan = scan[:scan.index("\ndef ")]
    assert "_run_floor_step(_macro, _floor_regime)" in scan
    for direct in ("_manage_vol_managed_floor(_macro", "_manage_spy_floor(_macro", "_manage_defensive_floor(_macro"):
        assert direct not in scan, f"{direct} is called only through _run_floor_step"


def _floor_step(mode):
    calls = []
    with patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": mode}), \
         patch.object(bot_engine, "_manage_vol_managed_floor",
                      side_effect=lambda m: calls.append("vm") or {"mode": "vol_managed_spy", "actions": []}), \
         patch.object(bot_engine, "_manage_spy_floor",
                      side_effect=lambda m, r: calls.append("basket") or {"actions": [], "status": "ok"}), \
         patch.object(bot_engine, "_manage_defensive_floor",
                      side_effect=lambda m, r: calls.append("gld") or {"actions": [], "status": "ok"}):
        spy, gld = bot_engine._run_floor_step({"vxx_ratio": 1.0}, {"regime": "NEUTRAL"})
    return calls, spy, gld


def test_rollback_is_one_setting():
    """FLOOR_MODE "basket" runs the old basket floor and the GLD floor exactly as
    before and never the volatility-managed one; "vol_managed_spy" the reverse."""
    calls, spy, gld = _floor_step("vol_managed_spy")
    assert calls == ["vm"] and spy["mode"] == "vol_managed_spy" and gld["status"].startswith("off")
    calls, spy, gld = _floor_step("basket")
    assert calls == ["basket", "gld"] and gld["status"] == "ok"
    # the rest of the rollback follows the same one setting
    with patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": "basket"}):
        assert {"QQQ", "SMH", "KWEB", "VXUS", "GLD"} <= bot_engine._floor_owned_tickers(system_config.BASE_CONFIG)
    assert system_config.BASE_CONFIG["FLOOR_MODE"] == "vol_managed_spy", "the shipped mode"


def test_a_failing_floor_never_breaks_the_scan():
    with patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": "vol_managed_spy"}), \
         patch.object(bot_engine, "_manage_vol_managed_floor", side_effect=RuntimeError("broker down")):
        spy, _ = bot_engine._run_floor_step({}, None)
    assert spy["status"].startswith("error: broker down")
    with patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": "basket"}), \
         patch.object(bot_engine, "_manage_spy_floor", side_effect=RuntimeError("x")), \
         patch.object(bot_engine, "_manage_defensive_floor", side_effect=RuntimeError("y")):
        spy, gld = bot_engine._run_floor_step({}, None)
    assert spy["status"] == "error: x" and gld["status"] == "error: y"


def test_the_vm_manager_never_sends_a_market_order():
    src = open(os.path.join(_ROOT, "bot_engine.py")).read()
    fn = src[src.index("def _floor_vm_order("):src.index("def _floor_vm_save_state(")]
    assert '"type": "limit"' in fn and '"type": "market"' not in fn


def test_new_tickers_are_exempt_from_active_exits_everywhere():
    assert {"SPY", "SSO"} <= bot_engine.FLOOR_AND_LEG_TICKERS
    bot_ts = open(os.path.join(_ROOT, "server", "bot.ts")).read()
    decl = re.search(r"const FLOOR_AND_LEG_TICKERS = new Set\(\[([^\]]*)\]\);", bot_ts).group(1)
    assert '"SSO"' in decl and '"SPY"' in decl


def test_ownership_follows_the_mode():
    vm = dict(system_config.BASE_CONFIG, FLOOR_MODE="vol_managed_spy")
    basket = dict(system_config.BASE_CONFIG, FLOOR_MODE="basket")
    assert bot_engine._floor_owned_tickers(vm) == {"SPY", "SSO"}
    assert {"QQQ", "SMH", "KWEB", "VXUS", "GLD"} <= bot_engine._floor_owned_tickers(basket)
    assert bot_engine._floor_vm_legacy_tickers(vm) == {"QQQ", "SMH", "KWEB", "VXUS", "GLD"}


# ── reduce-only (the drawdown halt) ─────────────────────────────────────────

def test_reduce_only_sells_down_and_never_buys():
    # volatile market (E* = 0.5) while the floor holds 1.5x and some QQQ
    cap = 0.85 * 100_000
    h = fvm.holdings_for(1.5, cap)
    b = _Broker([_pos("SPY", h["base"] / 500.0, 500.0), _pos("SSO", h["levered"] / 100.0, 100.0),
                 _pos("QQQ", 10, 600)], QUOTES)
    with patch.object(bot_engine, "_daily_closes_completed", return_value=_closes_with_daily_move(0.02)), \
         patch.object(bot_engine, "get_alpaca_account", return_value={"equity": "100000"}), \
         patch.object(bot_engine, "get_alpaca_positions", return_value=b.positions), \
         patch.object(bot_engine, "_latest_quote", side_effect=lambda s_: b.quotes[s_]), \
         patch.object(bot_engine, "_open_order_symbols", return_value=set()), \
         patch.object(bot_engine, "_floor_vm_order", side_effect=b.order), \
         patch.object(bot_engine, "_unwind_covered_calls", return_value=[]), \
         patch.object(bot_engine, "_floor_vm_save_state"), \
         patch.object(bot_engine, "_is_regular_hours", return_value=True), \
         patch.object(bot_engine, "_floor_vm_broker_clock", return_value=(True, "open")), \
         patch.object(bot_engine, "_et_now_hour", return_value=11.0):
        r = bot_engine._manage_vol_managed_floor({}, reduce_only=True)
    assert r["reduce_only"] is True
    assert all(side == "sell" for _, _, side, _ in b.orders), b.orders
    # E* = 0.16 / 0.3257 = 0.491: SSO's target is 0 (sell all 425), the legacy
    # QQQ goes, and SPY is trimmed from 42.5k to E* x 85k at the bid
    spy_cut = int((h["base"] - r["target_base_dollars"]) // 499.9)
    assert spy_cut >= 1
    assert sorted((s_, q) for s_, q, _, _ in b.orders) == [("QQQ", 10), ("SPY", spy_cut), ("SSO", 425)], b.orders


def test_reduce_only_never_buys_even_when_under_target():
    b = _Broker([], QUOTES)
    with patch.object(bot_engine, "_daily_closes_completed", return_value=_closes_with_daily_move(0.004)), \
         patch.object(bot_engine, "get_alpaca_account", return_value={"equity": "100000"}), \
         patch.object(bot_engine, "get_alpaca_positions", return_value=[]), \
         patch.object(bot_engine, "_latest_quote", side_effect=lambda s_: b.quotes[s_]), \
         patch.object(bot_engine, "_open_order_symbols", return_value=set()), \
         patch.object(bot_engine, "_floor_vm_order", side_effect=b.order), \
         patch.object(bot_engine, "_floor_vm_save_state"), \
         patch.object(bot_engine, "_is_regular_hours", return_value=True), \
         patch.object(bot_engine, "_floor_vm_broker_clock", return_value=(True, "open")), \
         patch.object(bot_engine, "_et_now_hour", return_value=11.0):
        bot_engine._manage_vol_managed_floor({}, reduce_only=True)
    assert b.orders == []


def test_drawdown_halt_runs_the_reduce_only_floor():
    src = open(os.path.join(_ROOT, "bot_engine.py")).read()
    i = src.index('f"[DD_HALT] scan_market returning empty')
    block = src[i:i + 3000]
    assert '_manage_vol_managed_floor({}, reduce_only=True)' in block
    assert 'BASE_CONFIG.get("FLOOR_MODE", "basket") == "vol_managed_spy"' in block
    assert '"spy_floor": _halt_floor' in block
    assert "_manage_spy_floor(" not in block and "_manage_defensive_floor(" not in block, "the basket floors stay skipped"


# ── tier 3 stands down under the volatility-managed floor ──────────────────

def test_tier3_does_not_buy_spy_or_qqq_under_the_vm_floor():
    import tiered_strategy as ts
    ctx = ts.TierContext(equity=100_000, peak_equity=100_000, buying_power=200_000, positions=[],
                         regime="BULL", vxx_ratio=0.9, spy_vs_ma50=1.05)
    with patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": "basket"}), \
         patch.object(ts, "get_regime_caps", return_value={"MAX_TOTAL_EXPOSURE": 0.95, "MAX_POSITION_PCT": 0.08, "regime": "BULL"}):
        basket_actions = ts.tier3_trend_capture(ctx)
    with patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": "vol_managed_spy"}), \
         patch.object(ts, "get_regime_caps", return_value={"MAX_TOTAL_EXPOSURE": 0.95, "MAX_POSITION_PCT": 0.08, "regime": "BULL"}):
        vm_actions = ts.tier3_trend_capture(ctx)
    assert {a.ticker for a in basket_actions} == {"SPY", "QQQ"}, "premise: T3 buys in this context under the basket floor"
    assert vm_actions == [], "under the volatility-managed floor T3 must not buy what the floor owns or sells"
