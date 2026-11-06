#!/usr/bin/env python3
"""
The floor's idle cash in a T-bill fund (2026-11-06, system_config FLOOR_VM_CASH_*).

Why: the floor study's V1 credited idle cash with the T-bill rate; Alpaca
paper cash earns 0, and with idle cash at 0 V1's 1993-2026 log growth falls
below SPY's. The fund must pay its return in price (the paper account credits
no dividends), so it is BOXX, not SGOV/BIL.

Pins:
- the pure plan (floor_vol_managed.cash_plan): idle capital, the cash reserve
  (non-floor slice + short-put collateral), the band, sells first, buys only
  when allowed, the off switch sells everything;
- the engine pass (bot_engine._manage_floor_cash), broker I/O mocked: window,
  broker clock, working orders, quote checks, marketable limits only, a
  rejected buy blocks buys for the day, no I/O without a floor plan;
- the wiring: runs after the core in _run_floor_step only (never in the
  reduce-only passes), and the fund counts as cash everywhere exposure is
  measured (tier caps, third leg, sizing heat) and as a floor ticker
  everywhere stops are applied (Python and bot.ts).

Run: python3 -m pytest -q test_floor_cash_parking.py
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

CFG = {"FLOOR_VM_CASH_BAND": 0.05, "FLOOR_VM_CASH_MIN_TRADE_DOLLARS": 2000}
CAP = 85_000.0          # the floor's capital at 100k equity
RESERVE = 15_000.0      # the non-floor 15%, no short puts


# ── the pure plan ────────────────────────────────────────────────────────────

def test_idle_capital_is_parked_up_to_the_reserve():
    # E* = 0.5: SPY 42.5k of the floor's 85k; the account has 57.5k cash
    p = fvm.cash_plan(CAP, 42_500, 0.0, 57_500, RESERVE, CFG)
    assert (p["idle"], p["room"], p["target"]) == (42_500.0, 42_500.0, 42_500.0)
    assert p["action"] == "buy" and p["dollars"] == 42_500.0
    assert "idle" in p["reason"] and "earn 0" in p["reason"]


def test_the_reserve_binds_when_other_sleeves_use_cash():
    # satellites/options already hold 27.5k of the cash: only 15k can be parked
    p = fvm.cash_plan(CAP, 42_500, 0.0, 30_000, RESERVE, CFG)
    assert p["target"] == 15_000.0 and p["action"] == "buy" and p["dollars"] == 15_000.0


def test_small_gaps_are_not_traded():
    band = max(2000, 0.05 * CAP)                       # 4,250
    p = fvm.cash_plan(CAP, 42_500, 42_500 - band + 1, 57_500 - (42_500 - band + 1), RESERVE, CFG)
    assert p["action"] == "hold" and p["band"] == band
    p = fvm.cash_plan(CAP, 42_500, 42_500 + band - 1, 50_000, RESERVE, CFG)
    assert p["action"] == "hold", "a small overshoot is not sold either"


def test_buys_wait_while_the_core_trades_and_on_unknown_cash():
    p = fvm.cash_plan(CAP, 42_500, 0.0, 57_500, RESERVE, CFG, buys_allowed=False)
    assert p["action"] == "hold" and "waits" in p["reason"]
    p = fvm.cash_plan(CAP, 42_500, 0.0, None, RESERVE, CFG)
    assert p["action"] == "hold" and p["room"] == 0.0 and "unreadable" in p["reason"]


def test_sells_come_first_when_cash_runs_below_its_reserve():
    p = fvm.cash_plan(CAP, 42_500, 40_000, 5_000, RESERVE, CFG)
    assert p["room"] == 30_000.0 and p["target"] == 30_000.0
    assert p["action"] == "sell" and p["dollars"] == 10_000.0 and "reserve" in p["reason"]
    # sells are never held back by buys_allowed
    assert fvm.cash_plan(CAP, 42_500, 40_000, 5_000, RESERVE, CFG, buys_allowed=False)["action"] == "sell"


def test_the_floor_takes_its_capital_back_when_exposure_rises():
    # E* back near 1: SPY 80k, so only 5k of the floor's capital is idle
    p = fvm.cash_plan(CAP, 80_000, 40_000, 20_000, RESERVE, CFG)
    assert p["target"] == 5_000.0 and p["action"] == "sell" and p["dollars"] == 35_000.0
    assert "floor needs the capital back" in p["reason"]


def test_the_off_switch_sells_everything():
    p = fvm.cash_plan(CAP, 42_500, 12_345.67, 40_000, RESERVE, CFG, enabled=False)
    assert p["action"] == "sell" and p["sell_all"] and p["dollars"] == 12_345.67
    p = fvm.cash_plan(CAP, 42_500, 0.0, 40_000, RESERVE, CFG, enabled=False)
    assert p["action"] == "hold"


def test_no_capital_no_trade():
    assert fvm.cash_plan(0.0, 0.0, 0.0, 10_000, 0.0, CFG)["action"] == "hold"


def test_the_target_never_exceeds_idle_capital_or_room():
    for core in (0, 20_000, 42_500, 85_000, 120_000):
        for held in (0, 10_000, 50_000):
            for cash in (0, 10_000, 60_000, 100_000):
                for reserve in (0, 15_000, 40_000):
                    p = fvm.cash_plan(CAP, core, held, cash, reserve, CFG)
                    assert 0 <= p["target"] <= max(0.0, CAP - core) + 1e-6
                    assert p["target"] <= max(0.0, held + cash - reserve) + 1e-6
                    if p["action"] == "buy":
                        assert held + p["dollars"] <= p["target"] + 1e-6
                    if p["action"] == "sell":
                        assert p["dollars"] <= held + 1e-6


# ── config ───────────────────────────────────────────────────────────────────

def test_config_ships_on_with_a_fund_that_accrues_in_price():
    c = system_config.BASE_CONFIG
    assert c["FLOOR_VM_CASH_ENABLED"] is True, "ON by default (human directive: paper, test it)"
    assert c["FLOOR_VM_CASH_TICKER"] == "BOXX"
    distributing = {"SGOV", "BIL", "SHV", "TBIL", "USFR", "TFLO", "CLIP", "XBIL", "GBIL", "BILS"}
    assert c["FLOOR_VM_CASH_TICKER"] not in distributing, (
        "a T-bill fund that pays its yield as distributions earns ~0 in a paper account that credits "
        "no dividends (SGOV price-only +0.04%/yr 2024-11..2026-11)")
    assert 0 < c["FLOOR_VM_CASH_BAND"] <= 0.10 and c["FLOOR_VM_CASH_MIN_TRADE_DOLLARS"] > 0
    assert 0 < c["FLOOR_VM_CASH_MAX_SPREAD_PCT"] <= 0.005


def test_the_fund_ticker_must_not_clash_with_the_floor():
    vm = dict(system_config.BASE_CONFIG)
    assert bot_engine.floor_cash_ticker(vm) == "BOXX"
    for clash in ("SPY", "SSO", "QQQ", "GLD", "", None):
        assert bot_engine.floor_cash_ticker(dict(vm, FLOOR_VM_CASH_TICKER=clash)) is None, clash
    assert bot_engine.floor_cash_ticker(dict(vm, FLOOR_VM_CASH_TICKER=" boxx ")) == "BOXX"


# ── the engine pass, broker I/O mocked ───────────────────────────────────────

QUOTES = {"BOXX": (118.90, 118.92), "SPY": (499.9, 500.1), "SSO": (99.95, 100.05)}


class _Broker:
    def __init__(self, positions, cash=57_500.0, equity=100_000.0, quotes=None, working=(), reject=False):
        self.positions, self.cash, self.equity = positions, cash, equity
        self.quotes = dict(QUOTES if quotes is None else quotes)
        self.working, self.reject, self.orders = set(working), reject, []

    def account(self):
        a = {"equity": str(self.equity)}
        if self.cash is not None:
            a["cash"] = str(self.cash)
        return a

    def order(self, symbol, qty, side, limit_px):
        self.orders.append((symbol, qty, side, round(limit_px, 2)))
        if self.reject:
            return {"ok": False, "order_id": None, "error": "asset BOXX is not tradable"}
        return {"ok": True, "order_id": f"c{len(self.orders)}", "error": None}


def _pos(sym, qty, px, asset_class="us_equity"):
    return {"symbol": sym, "qty": str(qty), "qty_available": str(abs(qty)), "market_value": str(qty * px),
            "asset_class": asset_class}


def _run(broker, core_actions=(), hour=11.0, regular=True, clock=(True, "open"), core_ok=True, cfg=None):
    core = {"mode": "vol_managed_spy", "ok": core_ok, "actions": list(core_actions), "status": "ok"}
    with patch.dict(system_config.BASE_CONFIG, cfg or {}), \
         patch.dict(bot_engine._FLOOR_CASH_BUY_BLOCK, {"day": None, "error": None}), \
         patch.object(bot_engine, "get_alpaca_account", side_effect=broker.account) as acct, \
         patch.object(bot_engine, "get_alpaca_positions", return_value=broker.positions), \
         patch.object(bot_engine, "_latest_quote", side_effect=lambda s: broker.quotes[s]), \
         patch.object(bot_engine, "_open_order_symbols", return_value=broker.working), \
         patch.object(bot_engine, "_floor_vm_order", side_effect=broker.order), \
         patch.object(bot_engine, "_floor_cash_save_state"), \
         patch.object(bot_engine, "_floor_vm_broker_clock", return_value=clock), \
         patch.object(bot_engine, "_is_regular_hours", return_value=regular), \
         patch.object(bot_engine, "_et_now_hour", return_value=hour):
        plan = bot_engine._manage_floor_cash(core)
        return plan, core, acct


SPY_HALF = _pos("SPY", 85, 500.0)       # 42.5k: E* = 0.5 on target


def test_volatile_market_parks_the_idle_half_at_the_ask():
    b = _Broker([SPY_HALF])
    plan, core, _ = _run(b)
    assert b.orders == [("BOXX", int(42_500 // 118.92), "buy", 118.92)], b.orders
    assert plan["status"] == "traded" and plan["target"] == 42_500.0
    act = core["actions"][-1]
    assert act["type"] == "floor_vm_cash_buy" and act["ok"] and act["order_id"] == "c1"
    assert core["cash_parking"] is plan


def test_calm_market_fully_invested_floor_parks_nothing():
    b = _Broker([_pos("SPY", 85, 500.0), _pos("SSO", 425, 100.0)], cash=15_000.0)   # 1.5x: no idle capital
    plan, _, _ = _run(b)
    assert b.orders == [] and plan["action"] == "hold"


def test_no_buy_in_a_pass_where_the_core_traded_or_a_floor_order_works():
    b = _Broker([SPY_HALF])
    plan, _, _ = _run(b, core_actions=[{"type": "floor_vm_sell", "ticker": "SSO", "qty": 10, "ok": True}])
    assert b.orders == [] and plan["action"] == "hold" and "waits" in plan["reason"]
    b = _Broker([SPY_HALF], working={"SPY"})
    plan, _, _ = _run(b)
    assert b.orders == [] and plan["status"] == "wait: a floor order is working"
    b = _Broker([SPY_HALF], working={"BOXX"})
    plan, _, _ = _run(b)
    assert b.orders == [] and plan["status"] == "wait: an order for the fund is working"


def test_cash_below_its_reserve_sells_the_fund_at_the_bid():
    b = _Broker([SPY_HALF, _pos("BOXX", 336, 119.0)], cash=5_000.0)   # 39,984 parked, cash 5k < 15k
    plan, core, _ = _run(b)
    need = plan["dollars"]
    assert plan["action"] == "sell" and math.isclose(need, 39_984 - 29_984, abs_tol=0.01)
    assert b.orders == [("BOXX", math.ceil(need / 118.90), "sell", 118.90)]
    assert core["actions"][-1]["type"] == "floor_vm_cash_sell"


def test_a_core_buy_pass_may_still_sell_the_fund_for_cash():
    # E* rose: the core bought SPY this pass; the fund gives the floor its capital back
    b = _Broker([_pos("SPY", 160, 500.0), _pos("BOXX", 336, 119.0)], cash=-20_000.0)
    plan, _, _ = _run(b, core_actions=[{"type": "floor_vm_buy", "ticker": "SPY", "qty": 75, "ok": True}])
    assert plan["action"] == "sell" and b.orders and b.orders[0][2] == "sell"


def test_short_put_collateral_stays_in_cash():
    put = _pos("CMCSA261218P00024000", -10, 0.97, asset_class="us_option")   # 10 x $24 strike = 24k
    b = _Broker([SPY_HALF, put], cash=57_500.0)
    plan, _, _ = _run(b)
    assert plan["reserve"] == RESERVE + 24_000.0
    assert plan["target"] == 57_500.0 - RESERVE - 24_000.0
    assert b.orders == [("BOXX", int(plan["target"] // 118.92), "buy", 118.92)]


def test_nothing_trades_outside_the_window_or_while_the_broker_says_closed():
    for hour, regular in ((9.6, True), (15.95, True), (11.0, False)):
        b = _Broker([SPY_HALF])
        plan, _, _ = _run(b, hour=hour, regular=regular)
        assert b.orders == [] and plan["status"].startswith("deferred"), (hour, regular)
    b = _Broker([SPY_HALF])
    plan, _, _ = _run(b, clock=(False, "broker clock: market closed"))
    assert b.orders == [] and plan["status"] == "deferred: broker clock: market closed"


def test_a_wide_or_missing_quote_is_not_traded():
    b = _Broker([SPY_HALF], quotes={"BOXX": (118.0, 119.0)})          # 0.84% wide
    plan, core, _ = _run(b)
    assert b.orders == [] and plan["status"] == "skipped: quote too wide"
    assert core["actions"][-1]["type"] == "floor_vm_cash_wide_spread"
    b = _Broker([SPY_HALF], quotes={"BOXX": (0.0, 0.0)})
    plan, core, _ = _run(b)
    assert b.orders == [] and core["actions"][-1] == {"type": "floor_vm_no_quote", "ticker": "BOXX"}


def test_the_off_switch_sells_every_share_held():
    b = _Broker([SPY_HALF, _pos("BOXX", 336, 119.0)], cash=20_000.0)
    plan, _, _ = _run(b, cfg={"FLOOR_VM_CASH_ENABLED": False})
    assert b.orders == [("BOXX", 336, "sell", 118.90)] and plan["sell_all"]


def test_a_rejected_buy_blocks_buys_for_the_rest_of_the_day():
    b = _Broker([SPY_HALF], reject=True)
    core = {"mode": "vol_managed_spy", "ok": True, "actions": [], "status": "ok"}
    common = dict(get_alpaca_account=b.account, get_alpaca_positions=lambda: b.positions,
                  _latest_quote=lambda s: b.quotes[s], _open_order_symbols=lambda: set(),
                  _floor_vm_order=b.order, _floor_cash_save_state=lambda st: None,
                  _floor_vm_broker_clock=lambda: (True, "open"), _is_regular_hours=lambda: True,
                  _et_now_hour=lambda: 11.0)
    with patch.dict(bot_engine._FLOOR_CASH_BUY_BLOCK, {"day": None, "error": None}), \
         patch.multiple(bot_engine, **common):
        first = bot_engine._manage_floor_cash(dict(core, actions=[]))
        second = bot_engine._manage_floor_cash(dict(core, actions=[]))
    assert first["status"] == "order rejected" and len(b.orders) == 1
    assert second["status"].startswith("buys blocked today") and "not tradable" in second["status"]
    assert len(b.orders) == 1, "no resubmission every scan"


def test_a_malformed_put_quantity_is_an_error_not_a_guess():
    bad_put = {"symbol": "CMCSA261218P00024000", "qty": "n/a", "market_value": "-97", "asset_class": "us_option"}
    b = _Broker([SPY_HALF, bad_put])
    core = {"mode": "vol_managed_spy", "ok": True, "actions": [], "status": "ok"}
    with patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": "vol_managed_spy"}), \
         patch.object(bot_engine, "_manage_vol_managed_floor", return_value=core), \
         patch.object(bot_engine, "get_alpaca_account", side_effect=b.account), \
         patch.object(bot_engine, "get_alpaca_positions", return_value=b.positions), \
         patch.object(bot_engine, "_floor_vm_order", side_effect=b.order):
        spy_floor, _ = bot_engine._run_floor_step({}, None)
    assert spy_floor["cash_parking"]["status"].startswith("error: could not convert")
    assert b.orders == [] and spy_floor["status"] == "ok", "the core's own result is untouched"


def test_without_a_floor_plan_there_is_no_broker_call():
    b = _Broker([SPY_HALF])
    plan, _, acct = _run(b, core_ok=False)
    assert plan["status"].startswith("skipped") and acct.call_count == 0 and b.orders == []


def test_marketable_limits_only():
    src = open(os.path.join(_ROOT, "bot_engine.py")).read()
    fn = src[src.index("def _manage_floor_cash("):src.index("def _run_floor_step(")]
    assert "_floor_vm_order(ticker, qty, side, px)" in fn, "orders go through the floor's limit-order helper"
    assert "requests.post" not in fn and '"market"' not in fn


# ── wiring ──────────────────────────────────────────────────────────────────

def _floor_step(mode, core):
    seen = []
    with patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": mode}), \
         patch.object(bot_engine, "_manage_vol_managed_floor", return_value=core), \
         patch.object(bot_engine, "_manage_spy_floor", return_value={"actions": [], "status": "ok"}), \
         patch.object(bot_engine, "_manage_defensive_floor", return_value={"actions": [], "status": "ok"}), \
         patch.object(bot_engine, "_manage_floor_cash", side_effect=lambda r: seen.append(r) or {}):
        bot_engine._run_floor_step({}, None)
    return seen


def test_the_cash_step_runs_after_the_core_only_in_the_vm_mode():
    core = {"mode": "vol_managed_spy", "ok": True, "actions": [], "status": "ok"}
    assert _floor_step("vol_managed_spy", core) == [core], "it gets the core's own result"
    assert _floor_step("basket", core) == []
    assert _floor_step("vol_managed_spy", dict(core, status="error: broker down")) == []


def test_the_reduce_only_passes_never_park_cash():
    """The drawdown halt and the kill switch: sells only, the result is cash
    (human decision 2026-11-05). Neither calls the cash step."""
    src = open(os.path.join(_ROOT, "bot_engine.py")).read()
    kill = src[src.index("def floor_reduce_only_pass("):src.index("\ndef ", src.index("def floor_reduce_only_pass("))]
    halt = src[src.index('f"[DD_HALT] scan_market returning empty'):][:3000]
    vm = src[src.index("def _manage_vol_managed_floor("):src.index("def floor_reduce_only_pass(")]
    for name, body in (("kill pass", kill), ("drawdown halt", halt), ("core manager", vm)):
        assert "_manage_floor_cash" not in body, name


def test_the_fund_is_a_floor_ticker_for_stops_and_ledgers():
    import entry_shadow
    import position_sizing as ps
    assert "BOXX" in bot_engine.FLOOR_AND_LEG_TICKERS, "manage_positions must never stop it out"
    assert entry_shadow.ledger_sleeve("BOXX") == "floor"
    assert "BOXX" in ps._passive_floor_tickers()
    bot_ts = open(os.path.join(_ROOT, "server", "bot.ts")).read()
    decl = re.search(r"const FLOOR_AND_LEG_TICKERS = new Set\(\[([^\]]*)\]\);", bot_ts).group(1)
    assert '"BOXX"' in decl, "bot.ts: floor sleeve, no WS stop, no POS-KILL, not heat"


def test_the_fund_never_shrinks_a_satellite_trade():
    import position_sizing as ps
    sat = [{"symbol": "AAPL", "market_value": "5000"}]
    book = [{"symbol": "SPY", "market_value": "42500"}, {"symbol": "BOXX", "market_value": "42500"}] + sat
    assert ps._portfolio_heat_scalar(book, 100_000) == ps._portfolio_heat_scalar(sat, 100_000) == 0.9


def test_the_tier_engine_counts_the_fund_as_cash():
    """Without the exclusion, a parked floor reads as 85% invested and the
    CAUTION (60%), BEAR (50%) and PANIC (30%) ceilings would block every tier,
    Tier-1 CSPs included, exactly when E* < 1."""
    import tiered_strategy as ts
    book = [{"symbol": "SPY", "market_value": "42500"}, {"symbol": "BOXX", "market_value": "42500"}]
    caution = {"MAX_TOTAL_EXPOSURE": 0.60, "regime": "CAUTION"}

    def killed(positions):
        ctx = ts.TierContext(equity=100_000, peak_equity=100_000, buying_power=200_000, positions=positions,
                             regime="CAUTION", vxx_ratio=1.1, spy_vs_ma50=0.99)
        with patch.object(ts, "get_regime_caps", return_value=caution):
            return ts.master_kill_switch(ctx)[0]

    assert killed(book) is True, "premise: the raw book trips the 60% ceiling"
    assert killed(bot_engine.without_floor_cash(book)) is False
    assert bot_engine.without_floor_cash(book) == [book[0]]
    err = {"code": 40010001, "message": "request is not authorized"}
    assert bot_engine.without_floor_cash(err) is err, "a failed read fails as before, never as an empty book"
    assert bot_engine.without_floor_cash(None) is None
    src = open(os.path.join(_ROOT, "bot_engine.py")).read()
    tier = src[src.index("ctx = TierContext("):][:600]
    assert "positions=without_floor_cash(positions)" in tier


def test_the_third_leg_room_counts_the_fund_as_cash():
    src = open(os.path.join(_ROOT, "bot_engine.py")).read()
    leg = src[src.index("def _run_third_leg("):]
    leg = leg[:leg.index("\ndef ")]
    assert "for p in without_floor_cash(positions_raw)) / equity" in leg


def test_the_floor_audit_knows_the_cash_actions():
    ts_src = open(os.path.join(_ROOT, "server", "floorAudit.ts")).read()
    for t in ("floor_vm_cash_buy", "floor_vm_cash_sell", "floor_vm_cash_wide_spread"):
        assert t in ts_src, t
