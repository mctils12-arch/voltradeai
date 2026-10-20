"""Regression tests: the scanner-direct options path must not wipe the entry
state register_options_entry() records (REPAIR 2026-10-20).

LIVE FAILURE: TSLA / META / GOOGL iron condors opened 2026-10-19/20 by the
bot.ts scanner-direct path were market-closed 4-10 minutes after the opening
fill (-$66 / -$70 / -$84), despite MIN_HOLD_MINUTES = 60. The inline Python
in server/bot.ts registered each leg via register_options_entry() and then
rewrote the same state keys as a bare {strategy, setup, ticker} dict for the
dashboard — deleting entry_timestamp (so _manage_strategy_group's minimum
hold gate saw no timestamp and let every exit rule run on the very next
cycle) and max_loss (so the iron-condor 50%-of-max-loss exit never armed).

These tests run the REAL inline script extracted from server/bot.ts, with
select_contract / submit_options_order stubbed (no network, no orders).
"""
import json
import os
import re
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import options_manager
import storage_config

BOT_TS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", "bot.ts")


def _scanner_direct_script() -> str:
    src = open(BOT_TS, encoding="utf-8").read()
    anchor = src.index("SCANNER PATH: Direct to select_contract + submit")
    start = src.index('python3 -c "', anchor) + len('python3 -c "')
    end = src.index('"`', start)
    script = src[start:end]
    assert "${scanTmpPath}" in script and "register_options_entry" in script, "scanner-direct script moved"
    return script


def _occ(root: str, days: int, cp: str, strike: float) -> str:
    exp = (datetime.now() + timedelta(days=days)).strftime("%y%m%d")
    return f"{root}{exp}{cp}{int(round(strike * 1000)):08d}"


DTE = 10  # inside the 21-DTE exit zone, the live condors' shape
LEGS = {
    "short_call": _occ("TSLA", DTE, "C", 470.0),
    "long_call": _occ("TSLA", DTE, "C", 485.0),
    "short_put": _occ("TSLA", DTE, "P", 420.0),
    "long_put": _occ("TSLA", DTE, "P", 405.0),
}
CONDOR = {"strategy": "iron_condor", **LEGS, "qty": 1, "net_credit": 4.59,
          "limit_price": 4.48, "max_loss": 1041.0}


def _alpaca_legs():
    sides = {"short_call": "short", "long_call": "long", "short_put": "short", "long_put": "long"}
    px = {"short_call": 6.10, "long_call": 3.20, "short_put": 5.40, "long_put": 3.71}
    return [{"symbol": LEGS[k], "qty": "1" if sides[k] == "long" else "-1", "side": sides[k],
             "avg_entry_price": str(px[k]), "current_price": str(px[k] * (1.03 if sides[k] == "short" else 0.97)),
             "unrealized_pl": "-16.5"} for k in LEGS]


class ScannerDirectStateTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.state_path = os.path.join(self.dir, "voltrade_options_state.json")
        self.payload = os.path.join(self.dir, "payload.json")
        with open(self.payload, "w") as f:
            json.dump({"ticker": "TSLA", "strategy": "iron_condor", "price": 445.0,
                       "equity": 100000, "setup": "high_iv_premium_sale"}, f)
        self.patches = [
            patch.object(options_manager, "OPTIONS_STATE_PATH", self.state_path),
            # production: storage_config.DATA_DIR is the directory both
            # writers resolve, so the bot.ts block sees the same file
            patch.object(storage_config, "DATA_DIR", self.dir),
            patch.dict(sys.modules, {"options_execution": types.SimpleNamespace(
                select_contract=lambda *a, **k: dict(CONDOR),
                submit_options_order=lambda c: {"status": "pending_new", "id": "x"},
            )}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def _run_bot_ts_script(self):
        code = _scanner_direct_script().replace("${scanTmpPath}", self.payload)
        with patch("builtins.print"):
            exec(compile(code, "bot.ts:scanner-direct", "exec"), {"__name__": "__bot_ts__"})
        with open(self.state_path) as f:
            return json.load(f)

    def test_every_leg_keeps_entry_timestamp_and_max_loss(self):
        state = self._run_bot_ts_script()
        for key, occ in LEGS.items():
            leg = state.get(occ)
            self.assertIsInstance(leg, dict, key)
            self.assertIn("entry_timestamp", leg, f"{key}: minimum-hold clock wiped")
            self.assertEqual(leg.get("max_loss"), 1041.0, f"{key}: IC max-loss exit disarmed")
            self.assertEqual(leg.get("strategy"), "iron_condor")
            self.assertEqual(leg.get("setup"), "high_iv_premium_sale")
            self.assertEqual(leg.get("ticker"), "TSLA")
        self.assertEqual(state["TSLA"].get("options_strategy"), "iron_condor", "dashboard ticker row kept")

    def test_fresh_condor_is_not_closed_inside_min_hold(self):
        state = self._run_bot_ts_script()
        closes = []
        group = {"legs": _alpaca_legs(), "strategy": "iron_condor", "ticker": "TSLA", "setup": "high_iv_premium_sale"}
        with patch.object(options_manager, "_close_strategy_mleg", lambda legs, t: closes.append(t) or {"status": "submitted"}), \
             patch.dict(sys.modules, {"vol_surface": types.SimpleNamespace(get_surface_score=lambda t: {"vrp": {"vrp_20d": 0.05}})}):
            actions = options_manager._manage_strategy_group(group, 100000, state)
            self.assertEqual(closes, [], f"closed inside MIN_HOLD_MINUTES: {actions}")
            # control: once the hold has elapsed the same position IS closed
            # (here by the 21-DTE rule) — the gate is what held it above.
            aged = (datetime.now() - timedelta(minutes=options_manager.MIN_HOLD_MINUTES + 1)).isoformat()
            for occ in LEGS.values():
                state[occ]["entry_timestamp"] = aged
            options_manager._manage_strategy_group(group, 100000, state)
            self.assertEqual(closes, ["TSLA"])


class MergeOptionsMetaTest(unittest.TestCase):
    def test_merge_keeps_existing_fields_and_skips_non_string_keys(self):
        with tempfile.TemporaryDirectory() as d, \
             patch.object(options_manager, "OPTIONS_STATE_PATH", os.path.join(d, "s.json")):
            options_manager.register_options_entry("XLF261120P00050000", 0.9, "sell", "sell_cash_secured_put", ticker="XLF")
            options_manager.merge_options_meta(["XLF261120P00050000", "", None, {"x": 1}, "NEW"], {"setup": "csp"})
            s = options_manager._load_options_state()
            self.assertIn("entry_timestamp", s["XLF261120P00050000"])
            self.assertEqual(s["XLF261120P00050000"]["setup"], "csp")
            self.assertEqual(s["NEW"], {"setup": "csp"})
            self.assertEqual(set(s), {"XLF261120P00050000", "NEW"})


if __name__ == "__main__":
    unittest.main()
