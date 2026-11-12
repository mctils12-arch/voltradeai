#!/usr/bin/env python3
"""
KILL LATCH AT THE ORDER (2026-11-12, KNOWN BROKEN #51): Python's direct order
paths read Node's persisted kill latch immediately before an OPENING order.

On 2026-11-05 a Tier-2 scan that began unlatched bought GLD 33 s after Node's
-20% kill re-tripped: only Node-dispatched entries were re-checked, at the loop
top; the floor rebalance, the floor's cash parking, the SVXY third leg and the
Python option opens never read the latch.

Pinned here:
  - the reader (kill_latch.py): no file -> not latched; Node's record
    {killSwitch, reason, savedAt} read as Node reads it; the primary path wins
    over the fallback; a corrupt or shapeless file is NOT latched (Node's own
    boot behaviour) and names the error;
  - the gate: EntryBlocked (an OrderBlocked, ORDER-BLOCKED prefix) while
    latched, silence while not;
  - the mid-scan flip, at each path: the floor's order helper sends a buy
    before the latch and refuses it after, sells still go; the cash parking
    pass does not order and does not arm its rejected-buy day block; the SVXY
    third leg records a third_leg_blocked action with the price it would have
    paid; evaluate_and_execute returns the blocked status without submitting;
    a tax-harvest swap is gated;
  - on the source: every opening-order POST in bot_engine sits behind the
    latch gate in its own block (the broker-clock gate test's AST walk), the
    three dispatcher scripts in bot.ts read it before submit_options_order,
    bot.ts handles the blocked status (never a stock fallback) and re-reads
    state.killSwitch before the stock order, and the status string is one
    string on both sides.
All broker I/O is mocked; the latch file lives in a temp dir on both paths.
"""
import ast
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import bot_engine  # noqa: E402
import broker_clock  # noqa: E402
import kill_latch  # noqa: E402
import options_execution  # noqa: E402
import system_config  # noqa: E402


def _write(path, payload):
    with open(path, "w") as f:
        if isinstance(payload, str):
            f.write(payload)
        else:
            json.dump(payload, f)


class LatchDirs(unittest.TestCase):
    """Both of Node's paths point into fresh temp dirs."""

    def setUp(self):
        self.primary = tempfile.mkdtemp(prefix="vt_latch_primary_")
        self.fallback = tempfile.mkdtemp(prefix="vt_latch_fallback_")
        for p in (patch.object(kill_latch, "DATA_DIR", self.primary),
                  patch.object(kill_latch, "FALLBACK_DIR", self.fallback)):
            p.start()
            self.addCleanup(p.stop)

    def latch(self, on=True, where="primary", reason="DRAWDOWN-KILL: -20.0% from peak", saved_at="2026-11-05T17:32:42.000Z"):
        d = self.primary if where == "primary" else self.fallback
        _write(os.path.join(d, kill_latch.KILL_SWITCH_FILE),
               {"killSwitch": on, "reason": reason, "savedAt": saved_at})


class TestReader(LatchDirs):
    def test_no_file_is_not_latched(self):
        st = kill_latch.read_latch()
        self.assertEqual((st["latched"], st["source"], st["error"]), (False, None, None))
        self.assertFalse(kill_latch.is_latched())

    def test_nodes_record_is_read_as_node_reads_it(self):
        self.latch(True)
        st = kill_latch.read_latch()
        self.assertTrue(st["latched"])
        self.assertEqual(st["reason"], "DRAWDOWN-KILL: -20.0% from peak")
        self.assertEqual(st["saved_at"], "2026-11-05T17:32:42.000Z")
        self.assertEqual(st["source"], os.path.join(self.primary, kill_latch.KILL_SWITCH_FILE))
        self.latch(False)
        self.assertFalse(kill_latch.read_latch()["latched"], "killSwitch false (owner toggle OFF / auto-resume) clears it")

    def test_primary_path_wins_over_the_fallback(self):
        self.latch(False, where="primary")
        self.latch(True, where="fallback")
        self.assertFalse(kill_latch.read_latch()["latched"], "the volume's file is Node's record; /tmp is the fallback")
        os.remove(os.path.join(self.primary, kill_latch.KILL_SWITCH_FILE))
        self.assertTrue(kill_latch.read_latch()["latched"], "without the primary the fallback is read")

    def test_paths_mirror_bot_ts(self):
        paths = kill_latch.latch_paths()
        self.assertEqual(paths, (os.path.join(self.primary, "voltrade_kill_switch.json"),
                                 os.path.join(self.fallback, "voltrade_kill_switch.json")))
        with patch.object(kill_latch, "FALLBACK_DIR", self.primary):
            self.assertEqual(len(kill_latch.latch_paths()), 1, "DATA_DIR /tmp (local) is one path, read once")
        bot_ts = open(os.path.join(_ROOT, "server", "bot.ts")).read()
        self.assertIn('const KILL_SWITCH_PATH = "/data/voltrade/voltrade_kill_switch.json";', bot_ts)
        self.assertIn('const KILL_SWITCH_FALLBACK = "/tmp/voltrade_kill_switch.json";', bot_ts)
        self.assertEqual(kill_latch.KILL_SWITCH_FILE, "voltrade_kill_switch.json")

    def test_corrupt_or_shapeless_file_is_not_latched_and_names_the_error(self):
        path = os.path.join(self.primary, kill_latch.KILL_SWITCH_FILE)
        _write(path, "{not json")
        with self.assertLogs("kill_latch", level="WARNING") as logs:
            st = kill_latch.read_latch()
        self.assertFalse(st["latched"])
        self.assertIn("JSONDecodeError", st["error"])
        self.assertTrue(any("treated as not latched like Node does" in m for m in logs.output))
        _write(path, {"killSwitch": "yes"})
        st = kill_latch.read_latch()
        self.assertFalse(st["latched"])
        self.assertIn("no boolean killSwitch field", st["error"])
        _write(path, {"killSwitch": True, "reason": 7, "savedAt": None})
        st = kill_latch.read_latch()
        self.assertEqual((st["latched"], st["reason"], st["saved_at"]), (True, None, None), "non-string meta reads as absent, the boolean still rules")


class TestGate(LatchDirs):
    def test_not_latched_returns(self):
        self.assertIsNone(kill_latch.check_opening_allowed("SPY", "floor buy"))

    def test_latched_raises_an_order_blocked_with_the_prefix_and_logs(self):
        self.latch(True)
        with self.assertLogs("kill_latch", level="WARNING") as logs:
            with self.assertRaises(kill_latch.EntryBlocked) as ctx:
                kill_latch.check_opening_allowed("SPY", "floor buy")
        msg = str(ctx.exception)
        self.assertTrue(msg.startswith(broker_clock.ORDER_BLOCKED_PREFIX + ": Node kill latch set"), msg)
        self.assertIn("DRAWDOWN-KILL: -20.0% from peak", msg)
        self.assertIn("floor buy SPY not sent", msg)
        self.assertIsInstance(ctx.exception, broker_clock.OrderBlocked, "every existing OrderBlocked handler covers it")
        self.assertIsInstance(ctx.exception, RuntimeError)
        self.assertTrue(any(kill_latch.ENTRY_BLOCKED_TAG in m for m in logs.output))

    def test_blocked_order_status_is_the_shared_string_never_error(self):
        self.latch(True)
        try:
            kill_latch.check_opening_allowed("X", "test")
        except kill_latch.EntryBlocked as e:
            order = kill_latch.blocked_order(e, cash_required=0)
        self.assertEqual(order["status"], kill_latch.BLOCKED_ORDER_STATUS)
        self.assertNotEqual(order["status"], "error", "an 'error' option open makes bot.ts fall back to a stock buy")
        self.assertEqual(order["cash_required"], 0)
        self.assertIn("Node kill latch set", order["detail"])
        halt_scope = open(os.path.join(_ROOT, "server", "haltScope.ts")).read()
        self.assertIn(f'export const KILL_BLOCKED_ORDER_STATUS = "{kill_latch.BLOCKED_ORDER_STATUS}";', halt_scope,
                      "one string on both sides")


class TestFloorOrderHelperMidScanFlip(LatchDirs):
    def _post_ok(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"id": "ord-1"}
        return MagicMock(return_value=resp)

    def test_buy_sent_before_the_latch_refused_after_it_sells_still_go(self):
        post = self._post_ok()
        with patch.object(bot_engine.requests, "post", post), patch.object(bot_engine.alpaca_throttle, "acquire"):
            first = bot_engine._floor_vm_order("SPY", 10, "buy", 500.0)
            self.assertTrue(first["ok"]) and self.assertEqual(post.call_count, 1)
            self.latch(True)                                     # the latch trips mid-scan
            second = bot_engine._floor_vm_order("SPY", 10, "buy", 500.0)
            self.assertFalse(second["ok"])
            self.assertIsNone(second["order_id"])
            self.assertIn("Node kill latch set", second["error"])
            self.assertEqual(post.call_count, 1, "the second buy never reached the broker")
            sell = bot_engine._floor_vm_order("SSO", 5, "sell", 100.0)
            self.assertTrue(sell["ok"])
            self.assertEqual(post.call_count, 2, "a reduce-only sell is not an entry")
            self.assertEqual(post.call_args.kwargs["json"]["side"], "sell")

    def test_cash_parking_buy_waits_without_arming_the_day_block(self):
        self.latch(True)
        orders = []

        def order(symbol, qty, side, px):
            orders.append((symbol, qty, side))
            return {"ok": True, "order_id": "o1", "error": None}

        core = {"mode": "vol_managed_spy", "ok": True, "actions": [], "status": "ok"}
        positions = [{"symbol": "SPY", "qty": "85", "qty_available": "85", "market_value": "42500", "asset_class": "us_equity"}]
        with patch.dict(system_config.BASE_CONFIG, {"FLOOR_VM_CASH_BAND": 0.05, "FLOOR_VM_CASH_MIN_TRADE_DOLLARS": 2000}), \
             patch.dict(bot_engine._FLOOR_CASH_BUY_BLOCK, {"day": None, "error": None}), \
             patch.object(bot_engine, "get_alpaca_account", return_value={"equity": "100000", "cash": "57500"}), \
             patch.object(bot_engine, "get_alpaca_positions", return_value=positions), \
             patch.object(bot_engine, "_latest_quote", return_value=(110.0, 110.02)), \
             patch.object(bot_engine, "_open_order_symbols", return_value=set()), \
             patch.object(bot_engine, "_floor_vm_order", side_effect=order), \
             patch.object(bot_engine, "_floor_cash_save_state"), \
             patch.object(bot_engine, "_floor_vm_broker_clock", return_value=(True, "open")), \
             patch.object(bot_engine, "_is_regular_hours", return_value=True), \
             patch.object(bot_engine, "_et_now_hour", return_value=11.0):
            plan = bot_engine._manage_floor_cash(core)
            self.assertEqual(plan["action"], "buy", "the test case is a parking buy")
            self.assertEqual(orders, [], "nothing ordered while latched")
            self.assertTrue(plan["status"].startswith("blocked: Node kill latch"))
            self.assertIsNone(bot_engine._FLOOR_CASH_BUY_BLOCK["day"], "a latch block is not a rejected buy: parking resumes with the latch")
            blocked = [a for a in core["actions"] if a.get("type") == "floor_vm_cash_blocked"]
            self.assertEqual(len(blocked), 1)
            self.assertIn("Node kill latch set", blocked[0]["reason"])
            self.assertGreater(blocked[0]["qty"], 0)
            # the latch clears: the same pass buys
            self.latch(False)
            plan = bot_engine._manage_floor_cash(core)
            self.assertEqual([o[2] for o in orders], ["buy"])
            self.assertEqual(plan["status"], "traded")


class TestThirdLegMidScanFlip(LatchDirs):
    SVXY_PX = 50.0

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="vt_latch_leg3_")
        for p in (patch.object(bot_engine, "DATA_DIR", self.tmp),
                  patch.dict(os.environ, {"VOLTRADE_SHADOW_ENTRIES": "off"}),
                  patch.dict(system_config.BASE_CONFIG, {"CASH_RESERVE_ENABLED": False})):
            p.start()
            self.addCleanup(p.stop)

    def run_leg(self):
        def _resp(payload):
            m = MagicMock()
            m.json.return_value = payload
            m.ok = True
            return m

        def fake_get(url, *a, **kw):
            if url.endswith("/v2/account"):
                return _resp({"equity": "100000"})
            if url.endswith("/v2/positions") or url.endswith("/v2/orders"):
                return _resp([])
            if url.endswith("/v2/stocks/bars"):
                return _resp({"bars": {"VXX": [{"c": c} for c in (30, 29, 28, 27, 26, 25)]}})
            if url.endswith("/v2/stocks/snapshots"):
                return _resp({"SVXY": {"latestTrade": {"p": self.SVXY_PX}}})
            return _resp({})

        post = MagicMock(return_value=_resp({"id": "ord-1"}))
        macro = {"vxx_ratio": 1.15, "spy_vs_ma50": 1.0, "spy_below_200_days": 0, "spy_above_200d": True,
                 "vxx_closes": [20.0] * 30, "vxx_latest": 23.0, "timestamp": 2_000_000.0}
        with patch("requests.get", side_effect=fake_get), patch("requests.post", post):
            result = bot_engine._run_third_leg(macro)
        return result, [c.kwargs.get("json") or {} for c in post.call_args_list]

    def test_unlatched_buys_latched_records_the_block_with_its_price(self):
        result, posted = self.run_leg()
        self.assertEqual([o.get("symbol") for o in posted], ["SVXY"], "the control: an unlatched pass buys")
        self.assertTrue(any(a.get("type") == "vrp_harvest" for a in result["actions"]))
        self.latch(True)
        result, posted = self.run_leg()
        self.assertEqual(posted, [], "no order while Node's latch holds")
        blocked = [a for a in result["actions"] if a.get("type") == "third_leg_blocked"]
        self.assertEqual(len(blocked), 1, result["actions"])
        self.assertEqual((blocked[0]["symbol"], blocked[0]["stage"], blocked[0]["price"]), ("SVXY", "buy", self.SVXY_PX))
        self.assertGreater(blocked[0]["shares"], 0)
        self.assertIn("Node kill latch set", blocked[0]["reason"])
        self.assertEqual([a for a in result["actions"] if a.get("type") in ("third_leg_error", "vrp_harvest")], [],
                         "a latch block is neither a failed order nor a buy")


class TestOptionOpenAndHarvest(LatchDirs):
    def test_evaluate_and_execute_returns_the_blocked_status_without_submitting(self):
        self.latch(True)
        contract = {"occ_symbol": "SPY261218P00500000", "strategy": "buy_put", "side": "buy", "qty": 1, "limit_price": 5.0}
        with patch.object(options_execution, "should_use_options", return_value={"use_options": True, "strategy": "buy_put", "edge_pct": 3.0, "reason": "test"}), \
             patch.object(options_execution, "select_contract", return_value=dict(contract)), \
             patch.dict(os.environ, {"VOLTRADE_SHADOW_ENTRIES": "off"}), \
             patch.object(options_execution, "submit_options_order") as submit, \
             patch.object(options_execution, "_mid_first_entry", side_effect=AssertionError("the entry ladder must not run for a blocked open")):
            out = options_execution.evaluate_and_execute({"ticker": "SPY", "price": 500.0, "deep_score": 70}, 100_000.0, [])
        submit.assert_not_called()
        self.assertEqual(out["instrument"], "options")
        self.assertEqual(out["order"]["status"], kill_latch.BLOCKED_ORDER_STATUS)
        self.assertTrue(out["reasoning"].startswith("blocked: "))
        self.assertEqual(out["contract"], contract)

    def test_evaluate_and_execute_unlatched_submits(self):
        contract = {"occ_symbol": "SPY261218P00500000", "strategy": "buy_put", "side": "buy", "qty": 1, "limit_price": 5.0}
        with patch.object(options_execution, "should_use_options", return_value={"use_options": True, "strategy": "buy_put", "edge_pct": 3.0, "reason": "test"}), \
             patch.object(options_execution, "select_contract", return_value=dict(contract)), \
             patch.dict(os.environ, {"VOLTRADE_SHADOW_ENTRIES": "off"}), \
             patch.object(options_execution, "_mid_first_entry", side_effect=lambda c: c), \
             patch.object(options_execution, "submit_options_order", return_value={"status": "submitted", "order_id": "o1"}) as submit:
            out = options_execution.evaluate_and_execute({"ticker": "SPY", "price": 500.0, "deep_score": 70}, 100_000.0, [])
        submit.assert_called_once()
        self.assertEqual(out["order"]["status"], "submitted")

    def test_harvest_step_reads_the_latch_in_its_killed_input(self):
        src = ast.get_source_segment(open(os.path.join(_ROOT, "bot_engine.py")).read(),
                                     self._harvest_call()) or ""
        self.assertIn('"killed": kill_status is None or bool(kill_status.get("killed")) or kill_latch.is_latched()', src)

    @staticmethod
    def _harvest_call():
        tree = ast.parse(open(os.path.join(_ROOT, "bot_engine.py")).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(".harvest_step"):
                return node
        raise AssertionError("tax_filter.harvest_step call not found in bot_engine.py")


# ── on the source ─────────────────────────────────────────────────────────────

OPENING_LABELS = {"floor basket buy", "floor buy", "defensive floor buy", "third-leg VRP buy"}


def _order_posts(tree):
    """(post call, its block's statements, index) for every POST to /v2/orders — the
    broker-clock gate test's walk."""
    found = []
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody", "handlers"):
            stmts = getattr(parent, field, None)
            if not isinstance(stmts, list):
                continue
            for i, stmt in enumerate(stmts):
                if not isinstance(stmt, ast.stmt) or isinstance(stmt, (ast.If, ast.For, ast.While, ast.Try, ast.With,
                                                                        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                for node in ast.walk(stmt):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "post"
                            and node.args and "v2/orders" in ast.unparse(node.args[0])):
                        found.append((node, stmts, i))
    return found


def _calls(stmts, name):
    out = []
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Call) and ast.unparse(n.func) == name:
                out.append(n)
    return out


class TestEveryOpeningPostIsGated(unittest.TestCase):
    def test_each_opening_order_post_in_bot_engine_reads_the_latch_in_its_own_block(self):
        tree = ast.parse(open(os.path.join(_ROOT, "bot_engine.py")).read())
        opening_posts = 0
        for call, stmts, i in _order_posts(tree):
            clock = _calls(stmts[:i], "broker_clock.check_order_allowed")
            self.assertTrue(clock, f"bot_engine.py:{call.lineno}: an order POST without the broker-clock gate")
            labels = {ast.literal_eval(c.args[1]) for c in clock if len(c.args) > 1 and isinstance(c.args[1], ast.Constant)}
            if not labels & OPENING_LABELS:
                continue
            opening_posts += 1
            latch = _calls(stmts[:i], "kill_latch.check_opening_allowed")
            self.assertTrue(latch, f"bot_engine.py:{call.lineno}: an OPENING order POST ({labels}) without kill_latch.check_opening_allowed before it")
            self.assertEqual({ast.literal_eval(c.args[1]) for c in latch}, labels, "the latch gate carries the same label as the clock gate")
        self.assertEqual(opening_posts, 4, "basket floor buy, old floor buy, GLD defensive buy, SVXY buy: a new opening POST must be gated and counted here")

    def test_the_floor_order_helper_gates_buys_only(self):
        src = open(os.path.join(_ROOT, "bot_engine.py")).read()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_floor_vm_order")
        body = ast.get_source_segment(src, fn)
        self.assertIn('if side == "buy":\n            # KILL LATCH', body)
        self.assertIn('kill_latch.check_opening_allowed(symbol, f"floor {side}")', body)
        self.assertLess(body.index("kill_latch.check_opening_allowed"), body.index("broker_clock.check_order_allowed"))
        self.assertLess(body.index("broker_clock.check_order_allowed"), body.index("requests.post"))

    def test_bot_ts_scripts_read_the_latch_before_submit_and_handle_the_status(self):
        bot_ts = open(os.path.join(_ROOT, "server", "bot.ts")).read()
        for label in ("tier SELL_CSP open", "tier BUY_PUT open", "scanner-direct option open"):
            at = bot_ts.index(f"'{label}')")
            block = bot_ts[at - 400: at + 400]
            self.assertIn("kill_latch.check_opening_allowed(contract.get('occ_symbol') or data['ticker'], '" + label + "')", block)
            self.assertIn("submit_options_order(contract)", bot_ts[at: at + 200], f"{label}: the gate sits immediately before the submit")
            self.assertIn("except kill_latch.EntryBlocked as _kb:", block)
            self.assertIn("kill_latch.blocked_order(_kb)", block)
        self.assertIn("import { haltStateFrom, haltAllows, haltMode, KILL_BLOCKED_ORDER_STATUS, type HaltState } from \"./haltScope\";", bot_ts)
        self.assertEqual(bot_ts.count("=== KILL_BLOCKED_ORDER_STATUS"), 3, "SELL_CSP, BUY_PUT and executeTrades each read the status")
        exec_at = bot_ts.index('optExec.order?.status === KILL_BLOCKED_ORDER_STATUS')
        self.assertIn('noteRemaining(trade, "rejected_killswitch", "Node kill switch on — read at the order, loop ended");\n              break;',
                      bot_ts[exec_at: exec_at + 900], "a blocked option open ends the batch — never a stock fallback")
        fallback_at = bot_ts.index('optExec.order?.status === "error"')
        self.assertLess(exec_at, fallback_at, "the blocked branch is read before the error-fallback branch")
        stock_at = bot_ts.index("// ── Stock execution (default or fallback from options) ──")
        self.assertIn("if (state.killSwitch) {", bot_ts[stock_at: stock_at + 200], "the latch is re-read right before the stock order")
        self.assertIn('noteRemaining(trade, "rejected_killswitch", "Node kill switch on — read before the stock order, loop ended");\n          break;',
                      bot_ts[stock_at: stock_at + 700])

    def test_the_latch_module_posts_no_orders(self):
        tree = ast.parse(open(os.path.join(_ROOT, "kill_latch.py")).read())
        self.assertEqual(_order_posts(tree), [])
        self.assertNotIn("risk_kill_switch", open(os.path.join(_ROOT, "kill_latch.py")).read(),
                         "the module reads Node's file; the frozen Python mechanism is untouched")


if __name__ == "__main__":
    unittest.main()
