"""
Scale-out by thirds — what manage_positions persists (2026-11-06, refocus spec
section 4 "scale out systematically (e.g. thirds)").

VERIFIED BROKEN, FIXED HERE: server/bot.ts checkPositionOnTick places every
scale-out and persists scales_completed / remaining_qty / breakeven_active in
voltrade_stop_state.json; bot_engine.manage_positions only RECOMMENDS (Tier 1
never executes its actions, server/daemonRouting.posMonitorSyncLines). It
wrote its recommendation back as if executed, so when it saw a 1R crossing
first the WS monitor read scales_completed=1 (skipping the real first-third
sale), turned breakeven on, and sized its final exit from the reduced
remaining_qty — leaving shares unsold. Also pinned: before any scale-out the
whole live position is the position (scale-in tranches grow it), a scale-out
the monitor persisted mid-pass survives manage_positions' write, and unbought
tranches are held by the cash reserve (position_mgmt.pending_scale_in_dollars).
"""
import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

import bot_engine
import position_mgmt as pm


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(bot_engine, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(bot_engine, "_get_atr", lambda *a, **k: 2.0)
    with patch("macro_data.get_macro_snapshot", return_value={
            "vxx_ratio": 1.0, "spy_vs_ma50": 1.05, "spy_below_200_days": 0, "spy_above_200d": True}):
        yield tmp_path


def seed(tmp_path, rows):
    (tmp_path / "voltrade_stop_state.json").write_text(json.dumps(rows))


def state(tmp_path):
    return json.loads((tmp_path / "voltrade_stop_state.json").read_text())


def pos(sym, qty, entry, cur):
    return {"symbol": sym, "asset_class": "us_equity", "side": "long", "qty": str(qty),
            "avg_entry_price": str(entry), "current_price": str(cur),
            "unrealized_plpc": str(cur / entry - 1), "market_value": str(qty * cur)}


ENTRY_DATE = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")


def test_a_recommended_scale_out_is_not_recorded_as_done(env):
    # ATR 2% -> 1R = 4%; +6% is past 1R, no scale-out executed yet
    seed(env, {"AAPL": {"entry_date": ENTRY_DATE, "initial_risk_pct": 4.0, "original_qty": 30,
                        "remaining_qty": 30, "scales_completed": 0, "breakeven_active": False}})
    with patch.object(bot_engine, "get_alpaca_positions", return_value=[pos("AAPL", 30, 100.0, 106.0)]):
        res = bot_engine.manage_positions()
    rec = [a for a in res["actions"] if a.get("action") == "SCALE_OUT"]
    assert rec and rec[0]["qty"] == 10                          # still recommended (audit / UI)
    row = state(env)["AAPL"]
    assert (row["scales_completed"], row["remaining_qty"], row["original_qty"], row["breakeven_active"]) == (0, 30, 30, False)


def test_executed_scale_outs_persisted_by_the_monitor_are_kept(env):
    seed(env, {"AAPL": {"entry_date": ENTRY_DATE, "initial_risk_pct": 4.0, "original_qty": 30,
                        "remaining_qty": 20, "scales_completed": 1, "breakeven_active": True}})
    with patch.object(bot_engine, "get_alpaca_positions", return_value=[pos("AAPL", 20, 100.0, 112.0)]):
        bot_engine.manage_positions()                           # +12% = 3R: a second scale-out is recommended
    row = state(env)["AAPL"]
    assert (row["scales_completed"], row["remaining_qty"], row["original_qty"], row["breakeven_active"]) == (1, 20, 30, True)


def test_before_any_scale_out_the_live_position_is_the_position(env):
    # first tranche 10 shares recorded; two more tranches filled since -> live 30
    seed(env, {"AAPL": {"entry_date": ENTRY_DATE, "initial_risk_pct": 4.0, "original_qty": 10,
                        "remaining_qty": 10, "scales_completed": 0}})
    with patch.object(bot_engine, "get_alpaca_positions", return_value=[pos("AAPL", 30, 100.0, 101.0)]):
        bot_engine.manage_positions()
    row = state(env)["AAPL"]
    assert (row["original_qty"], row["remaining_qty"]) == (30, 30)


def test_a_scale_out_persisted_during_the_pass_survives_the_write(tmp_path):
    path = tmp_path / "voltrade_stop_state.json"
    path.write_text(json.dumps({"AAPL": {"scales_completed": 1, "original_qty": 30, "remaining_qty": 20,
                                         "breakeven_active": True, "phase": 2}}))
    pass_state = {"AAPL": {"scales_completed": 0, "original_qty": 30, "remaining_qty": 30,
                           "breakeven_active": False, "phase": 2},
                  "MSFT": {"scales_completed": 0, "original_qty": 9, "remaining_qty": 9}}
    bot_engine._merge_executed_scale_outs(pass_state, str(path))
    assert pass_state["AAPL"] == {"scales_completed": 1, "original_qty": 30, "remaining_qty": 20,
                                  "breakeven_active": True, "phase": 2}
    assert pass_state["MSFT"]["scales_completed"] == 0
    bot_engine._merge_executed_scale_outs(pass_state, str(tmp_path / "missing.json"))   # unreadable: no change


def plan_file(tmp_path, plans):
    (tmp_path / pm.SCALE_IN_STATE_FILE).write_text(json.dumps({"version": 1, "plans": plans}))


def test_unbought_tranches_are_held_by_the_cash_reserve(tmp_path, monkeypatch):
    import regime_engine
    monkeypatch.setattr(pm, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pm, "POSITION_MGMT_LOG_PATH", str(tmp_path / "pm.jsonl"))
    monkeypatch.setattr(pm, "floor_plan_target_ratio", lambda *a, **k: 0.85)
    monkeypatch.setattr(regime_engine, "risk_budget_for", lambda *a, **k: None, raising=False)
    assert pm.pending_scale_in_dollars() == 0.0
    plan_file(tmp_path, {
        "AAPL": {"status": "active", "ref_price": 100.0, "tranches": [
            {"n": 1, "qty": 10, "status": "bought"}, {"n": 2, "qty": 10, "status": "pending"},
            {"n": 3, "qty": 10, "status": "pending"}]},
        "MSFT": {"status": "cancelled", "ref_price": 400.0, "tranches": [{"n": 2, "qty": 5, "status": "cancelled"}]},
        "NVDA": {"status": "active", "ref_price": 50.0, "tranches": [
            {"n": 2, "qty": 4, "status": "ordered"}, {"n": 3, "qty": 4, "status": "pending"}]},
    })
    assert pm.pending_scale_in_dollars() == pytest.approx(2_000 + 200)
    snap = pm.live_headroom(100_000, [{"symbol": "SPY", "qty": "100", "market_value": "85000", "asset_class": "us_equity"}], "BULL")
    assert snap["pending_other"] == pytest.approx(2_200) and snap["headroom_dollars"] == pytest.approx(10_000 - 2_200)
    # an explicit pending amount overrides the file (the tranche executor releases its own)
    assert pm.live_headroom(100_000, [], "BULL", pending_dollars=0.0)["pending_other"] == 0.0
    (tmp_path / pm.SCALE_IN_STATE_FILE).write_text("{torn")
    assert pm.pending_scale_in_dollars() == 0.0


def test_file_name_matches_the_node_module():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", "scaleIn.ts")).read()
    assert f'SCALE_IN_STATE_FILE = "{pm.SCALE_IN_STATE_FILE}"' in src
