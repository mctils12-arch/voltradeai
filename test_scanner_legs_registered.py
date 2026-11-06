"""[REPAIR] 2026-11-06 — the bot.ts scanner-direct script registers a vertical's
legs (short_leg / long_leg) through the same leg walk evaluate_and_execute uses.

Found while tracing the mleg sign fix. The scanner-direct inline script in
server/bot.ts kept its own leg key list ('occ_symbol', 'short_call', 'long_call',
'short_put', 'long_put', 'call_leg', 'put_leg'), the list #1672 fixed in
evaluate_and_execute only. A scanner bull_put_credit_spread (the
vxx_panic_put_sale setup) names its legs short_leg / long_leg, so neither leg
was registered: options_manager managed two "unknown" single legs (no group
profit target, max-loss exit or 21-DTE close; the long wing could be sold
alone). The same list put every condor leg on the contract's "side" (a condor
contract has none, so "buy"): its sold legs were recorded as bought.

The tests run the REAL inline script extracted from server/bot.ts, with
select_contract / submit_options_order stubbed (no network, no orders), then a
real manage_options_positions pass on the registered state.
"""
import inspect
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_execution as real_oe  # noqa: E402
import options_manager as om  # noqa: E402
import storage_config  # noqa: E402
from test_options_exit_market_hours import Broker, _ManagerCase, occ, position  # noqa: E402

BOT_TS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", "bot.ts")
ANCHOR = "SCANNER PATH: Direct to select_contract + submit"


def _scanner_direct_script() -> str:
    src = open(BOT_TS, encoding="utf-8").read()
    start = src.index('python3 -c "', src.index(ANCHOR)) + len('python3 -c "')
    return src[start:src.index('"`', start)]


PUT_SHORT, PUT_LONG = occ("SPY", 35, "P", 560.0), occ("SPY", 35, "P", 532.0)
CREDIT_PUT_SPREAD = {  # _select_bull_put_credit_spread's keys (vxx_panic_put_sale)
    "strategy": "bull_put_credit_spread", "short_leg": PUT_SHORT, "long_leg": PUT_LONG, "qty": 2,
    "net_credit": 1.35, "limit_price": 1.35, "max_loss": 5330.0, "short_delta": -0.24, "long_delta": -0.11,
    "error": None}
CONDOR_LEGS = {"short_call": occ("TSLA", 40, "C", 470.0), "long_call": occ("TSLA", 40, "C", 485.0),
               "short_put": occ("TSLA", 40, "P", 420.0), "long_put": occ("TSLA", 40, "P", 405.0)}
CONDOR = {"strategy": "iron_condor", **CONDOR_LEGS, "qty": 1, "net_credit": 4.59, "max_loss": 1041.0, "error": None}
CSP = {"strategy": "sell_cash_secured_put", "occ_symbol": occ("AMD", 40, "P", 140.0), "side": "sell", "qty": 1,
       "limit_price": 3.2, "max_loss": 13680.0, "error": None}


class _ScannerCase(_ManagerCase):
    def run_script(self, contract, ticker, strategy, setup):
        payload = os.path.join(os.path.dirname(self.state_path), "payload.json")
        with open(payload, "w") as f:
            json.dump({"ticker": ticker, "strategy": strategy, "price": 100.0, "equity": 100000, "setup": setup}, f)
        code = _scanner_direct_script().replace("${scanTmpPath}", payload)
        printed = []
        with patch.dict(os.environ, {"VOLTRADE_SHADOW_ENTRIES": "off"}), \
                patch.object(storage_config, "DATA_DIR", os.path.dirname(self.state_path)), \
                patch.dict(sys.modules, {"options_execution": types.SimpleNamespace(
                    select_contract=lambda *a, **k: dict(contract),
                    submit_options_order=lambda c: {"status": "pending_new", "order_id": "o-1"},
                    register_contract_entry=real_oe.register_contract_entry)}), \
                patch("builtins.print", side_effect=lambda *a, **k: printed.append(a[0] if a else "")):
            exec(compile(code, "bot.ts:scanner-direct", "exec"), {"__name__": "__bot_ts__"})
        self.assertEqual(json.loads(printed[-1])["order"]["status"], "pending_new")
        return self.read_state()


class TestScannerVerticalLegsAreRegistered(_ScannerCase):
    def test_a_scanner_credit_put_spread_registers_both_legs_on_their_sides(self):
        state = self.run_script(CREDIT_PUT_SPREAD, "SPY", "bull_put_credit_spread", "vxx_panic_put_sale")
        for sym, side, delta in ((PUT_SHORT, "short", -0.24), (PUT_LONG, "long", -0.11)):
            with self.subTest(leg=sym):
                st = state.get(sym)
                self.assertIsInstance(st, dict, f"{sym} not registered")
                self.assertEqual((st["strategy"], st["side"], st["ticker"], st["setup"], st["max_loss"], st["qty"]),
                                 ("bull_put_credit_spread", side, "SPY", "vxx_panic_put_sale", 5330.0, 2))
                self.assertEqual(st["entry_delta"], delta)
                self.assertIn("entry_timestamp", st)
        self.assertEqual(state["SPY"]["options_strategy"], "bull_put_credit_spread", "dashboard ticker row kept")

    def test_the_manager_then_manages_the_two_legs_as_one_group(self):
        self.run_script(CREDIT_PUT_SPREAD, "SPY", "bull_put_credit_spread", "vxx_panic_put_sale")
        groups = []
        broker = Broker([position(PUT_SHORT, 2, "short", 3.10, 3.00), position(PUT_LONG, 2, "long", 1.75, 1.70)])
        with patch.object(om, "_manage_strategy_group", side_effect=lambda g, *a, **k: groups.append(g) or []):
            self.run_manager(broker)
        self.assertEqual([(g["strategy"], g["ticker"], sorted(p["symbol"] for p in g["legs"])) for g in groups],
                         [("bull_put_credit_spread", "SPY", sorted([PUT_SHORT, PUT_LONG]))])
        self.assertEqual(self.submits, [], "no single-leg rule touched either leg")

    def test_a_scanner_condor_records_its_sold_legs_as_sold(self):
        state = self.run_script(CONDOR, "TSLA", "iron_condor", "high_iv_premium_sale")
        self.assertEqual({k: state[sym]["side"] for k, sym in CONDOR_LEGS.items()},
                         {"short_call": "short", "long_call": "long", "short_put": "short", "long_put": "long"})
        self.assertTrue(all(state[sym]["strategy"] == "iron_condor" and state[sym]["max_loss"] == 1041.0
                            for sym in CONDOR_LEGS.values()))

    def test_a_single_leg_scanner_entry_is_registered_as_before(self):
        state = self.run_script(CSP, "AMD", "sell_cash_secured_put", "vxx_panic_put_sale")
        st = state[CSP["occ_symbol"]]
        self.assertEqual((st["strategy"], st["side"], st["entry_price"], st["initial_credit"], st["setup"]),
                         ("sell_cash_secured_put", "short", 3.2, 3.2, "vxx_panic_put_sale"))
        self.assertIn("pending_entry_feedback", st, "single-leg entries keep their deferred feedback marker")


class TestOneLegWalk(unittest.TestCase):
    def test_the_script_calls_the_shared_walk_and_keeps_no_key_list_of_its_own(self):
        script = _scanner_direct_script()
        self.assertIn("from options_execution import register_contract_entry", script)
        self.assertNotIn("_leg_keys", script)
        self.assertNotIn("'short_call','long_call'", script)

    def test_evaluate_and_execute_uses_the_same_function(self):
        src = inspect.getsource(real_oe.evaluate_and_execute)
        self.assertIn("register_contract_entry(contract, decision[\"strategy\"]", src)
        self.assertNotIn("_contract_leg_entries(", src, "the walk lives in register_contract_entry only")

    def test_register_contract_entry_returns_what_it_registered(self):
        with tempfile.TemporaryDirectory() as d, patch.object(om, "OPTIONS_STATE_PATH", os.path.join(d, "s.json")):
            self.assertEqual(real_oe.register_contract_entry(dict(CREDIT_PUT_SPREAD), "bull_put_credit_spread", "SPY"),
                             [PUT_SHORT, PUT_LONG])
            self.assertEqual(real_oe.register_contract_entry(dict(CSP), "sell_cash_secured_put", "AMD"), [CSP["occ_symbol"]])
            self.assertEqual(real_oe.register_contract_entry({"strategy": "x"}, "x", "Y"), [])


if __name__ == "__main__":
    unittest.main()
