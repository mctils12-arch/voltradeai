#!/usr/bin/env python3
"""
KILL-SWITCH FLOOR (human decision 2026-11-05): "while it's tripped, the floor
may only reduce exposure (sell or de-lever SSO->SPY->cash). It may not make any
buys or add leverage. Log every action taken while tripped."

Pins:
- bot_engine.floor_reduce_only_pass runs the volatility-managed floor in
  reduce-only mode, and does nothing under the basket floor (rollback mode);
- reduce-only sells SSO first, then SPY, then any legacy floor ticker, and never
  buys, even under target;
- the normal pass keeps its order (legacy first, then SPY, then SSO);
- the daemon registers the method and bot.ts never runs it in a fresh
  interpreter (HEAVY_DAEMON_ONLY), only from tier2HaltedExits under the kill latch.

Run: python3 -m pytest -q test_floor_kill_switch.py
"""
import os
import sys
from unittest.mock import patch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import bot_engine
import floor_vol_managed as fvm
import system_config
import pytest


@pytest.fixture(autouse=True)
def _flows_estimate_neutral():
    """The floor pass reads the day's flows estimate from DATA_DIR since
    2026-11-06 (flows_model.py); held neutral here like every other input.
    test_flows_model.py pins the reduce-only pass WITH a flows cut."""
    with patch.object(bot_engine, "_floor_flows_adjustment",
                      return_value={"multiplier": 1.0, "applied": False, "reason": "test: neutral"}):
        yield


QUOTES = {"SPY": (499.9, 500.1), "SSO": (99.95, 100.05), "QQQ": (599.9, 600.1)}


def _closes(move, n=21, start=500.0):
    out, px = [start], start
    for i in range(n - 1):
        px *= 1 + (move if i % 2 == 0 else -move)
        out.append(px)
    return out


def _pos(sym, qty, px):
    return {"symbol": sym, "qty": str(qty), "qty_available": str(qty), "market_value": str(qty * px)}


def _run(positions, move, reduce_only, mode="vol_managed_spy"):
    orders = []

    def order(symbol, qty, side, limit_px):
        orders.append((symbol, qty, side))
        return {"ok": True, "order_id": f"o{len(orders)}", "error": None}

    with patch.dict(system_config.BASE_CONFIG, {"FLOOR_MODE": mode}), \
         patch.object(bot_engine, "_daily_closes_completed", return_value=_closes(move)), \
         patch.object(bot_engine, "get_alpaca_account", return_value={"equity": "100000"}), \
         patch.object(bot_engine, "get_alpaca_positions", return_value=positions), \
         patch.object(bot_engine, "_latest_quote", side_effect=lambda s: QUOTES[s]), \
         patch.object(bot_engine, "_open_order_symbols", return_value=set()), \
         patch.object(bot_engine, "_floor_vm_order", side_effect=order), \
         patch.object(bot_engine, "_unwind_covered_calls", return_value=[]), \
         patch.object(bot_engine, "_floor_vm_save_state"), \
         patch.object(bot_engine, "_is_regular_hours", return_value=True), \
         patch.object(bot_engine, "_floor_vm_broker_clock", return_value=(True, "open")), \
         patch.object(bot_engine, "_et_now_hour", return_value=11.0):
        if reduce_only:
            result = bot_engine.floor_reduce_only_pass()
        else:
            result = bot_engine._manage_vol_managed_floor({})
    return result, orders


def _levered_book():
    """1.5x held (SPY + SSO) plus a leftover QQQ, on $100k equity."""
    h = fvm.holdings_for(1.5, 0.85 * 100_000)
    return [_pos("SPY", h["base"] / 500.0, 500.0), _pos("SSO", h["levered"] / 100.0, 100.0), _pos("QQQ", 10, 600)]


def test_kill_pass_delevers_sso_then_spy_then_legacy_and_never_buys():
    # a crash market (E* = 0.16 / sigma ~ 0.49): SSO goes entirely, SPY is trimmed, QQQ is sold
    result, orders = _run(_levered_book(), 0.02, reduce_only=True)
    assert result["reduce_only"] is True
    assert [o[0] for o in orders] == ["SSO", "SPY", "QQQ"], orders
    assert all(side == "sell" for _, _, side in orders)
    assert orders[0][1] == 425, "all of SSO goes first"


def test_kill_pass_never_buys_even_under_target():
    result, orders = _run([], 0.004, reduce_only=True)          # empty floor, calm market: E* = cap
    assert orders == [] and result["rebalance"] is True, "under target is not a reason to buy while killed"


def test_kill_pass_under_the_basket_floor_does_nothing():
    with patch.object(bot_engine, "_manage_vol_managed_floor",
                      side_effect=AssertionError("the VM floor must not run in basket mode")):
        result, orders = _run(_levered_book(), 0.02, reduce_only=True, mode="basket")
    assert orders == [] and result["actions"] == [] and result["status"].startswith("skipped")


def test_normal_pass_keeps_its_order_legacy_first():
    _result, orders = _run(_levered_book(), 0.02, reduce_only=False)
    assert [o[0] for o in orders] == ["QQQ", "SPY", "SSO"], orders


def test_daemon_and_bot_ts_wiring():
    daemon = open(os.path.join(_ROOT, "voltrade_daemon.py")).read()
    assert '"floor_reduce_only": ("bot_engine", "floor_reduce_only_pass")' in daemon
    bot = open(os.path.join(_ROOT, "server", "bot.ts")).read()
    heavy = bot[bot.index("const HEAVY_DAEMON_ONLY = new Set(["):]
    heavy = heavy[:heavy.index("]);")]
    assert '"floor_reduce_only"' in heavy, "a bot_engine import in a fresh interpreter is the OOM cold start"
    halted = bot[bot.index("async function tier2HaltedExits(): Promise<void> {"):bot.index("async function scheduleTier2() {")]
    assert halted.index("if (state.killSwitch) await killSwitchFloorReduce();") > halted.index('"manage_options_exits"')
    assert 'floorAuditRows(r.result, "kill"' in halted, "every kill-pass action is audited"
