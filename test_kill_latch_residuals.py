"""
KILL LATCH RESIDUALS (2026-11-12 [RULE-REVIEW], KNOWN BROKEN #51 follow-up to
kill_latch.py). The first fix gated the Python opens it listed; these are the
opening paths it left, and two ratchets it lacked.

Pinned here:
  - a ROLL's re-open is a new entry: while Node's latch is set the options
    manager closes and does not re-open, even when the scan passed
    allow_rolls=True (it started unlatched);
  - EVERY order POST in the four engine files is classified by its broker-clock
    label as opening, closing, or a frozen submitter. Every opening POST sits
    behind kill_latch.check_opening_allowed with the same label; no closing
    POST and no frozen submitter reads the latch; an unclassified POST fails
    (the first fix's scan skipped any label it did not list, which is how the
    convexity overlay's put buy and the intraday short entry went ungated);
  - bot.ts re-reads the latch right before its ETF substitution order, as it
    does before its stock order;
  - the suite itself never reads the machine's real latch file (conftest).
All broker I/O is mocked.
"""
import ast
import json
import os
import re
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import kill_latch
import options_manager as om

_ROOT = os.path.dirname(os.path.abspath(__file__))
_REASON = "Portfolio DD -20.0% <= -20%"


class _LatchDirs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vt_killlatch_res_")
        for p in (patch.object(kill_latch, "DATA_DIR", self.tmp), patch.object(kill_latch, "FALLBACK_DIR", self.tmp)):
            p.start()
            self.addCleanup(p.stop)

    def latch(self, on=True):
        with open(os.path.join(self.tmp, kill_latch.KILL_SWITCH_FILE), "w") as f:
            json.dump({"killSwitch": on, "reason": _REASON if on else "owner toggle OFF",
                       "savedAt": "2026-11-05T17:32:42.000Z"}, f)


class TestRollIsAnEntry(_LatchDirs):
    def test_a_roll_closes_and_does_not_reopen_while_latched(self):
        self.latch(True)
        with patch.object(om, "_attempt_roll", side_effect=AssertionError("must not roll")):
            r = om._roll_or_skip(True, "X", 1, "sell", {}, 100000)
        self.assertFalse(r["rolled"])
        self.assertTrue(r["detail"].startswith(om.ROLL_SKIPPED_KILL_LATCH_DETAIL), r)
        self.assertIn(_REASON, r["detail"])

    def test_the_halt_and_unlatched_paths_are_unchanged(self):
        self.latch(True)
        with patch.object(om, "_attempt_roll", side_effect=AssertionError("must not roll")):
            r = om._roll_or_skip(False, "X", 1, "sell", {}, 100000)
        self.assertEqual(r, {"rolled": False, "detail": om.ROLL_SKIPPED_HALTED_DETAIL})
        self.latch(False)
        with patch.object(om, "_attempt_roll", return_value={"rolled": True, "detail": "ok"}) as roll:
            r = om._roll_or_skip(True, "X", 2, "sell", {"ticker": "X"}, 5.0)
        roll.assert_called_once_with("X", 2, "sell", {"ticker": "X"}, 5.0)
        self.assertTrue(r["rolled"])

    @patch("ml_model_v2.track_fill")
    @patch("options_manager._save_options_state")
    @patch("options_manager._load_options_state")
    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    @patch("options_manager._attempt_roll")
    def test_the_manager_closes_instead_of_rolling_while_latched(self, mock_roll, mock_close, mock_snap, mock_get,
                                                                 mock_load, _save, _track):
        """test_options_halt_exits_only's 15-DTE short put with rolls ALLOWED
        (the scan started unlatched) and the latch on: the roll point takes its
        close branch."""
        exp = (datetime.now() + timedelta(days=15)).strftime("%y%m%d")
        occ = f"HALTQ{exp}P00050000"
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [{
            "symbol": occ, "qty": "-1", "side": "short", "avg_entry_price": "2.0", "current_price": "1.8",
            "market_value": "-180", "unrealized_pl": "20", "asset_class": "option"}])
        mock_snap.return_value = {"bid": 1.75, "ask": 1.85, "mid": 1.80, "delta": -0.30,
                                  "gamma": 0.02, "theta": -0.05, "vega": 0.05, "iv": 0.30}
        mock_close.return_value = {"status": "submitted", "order_id": "t1"}
        mock_roll.side_effect = AssertionError("a roll re-opens: never while latched")
        mock_load.return_value = {occ: {
            "entry_price": 2.00, "entry_delta": -0.30, "side": "short", "entry_date": "2026-09-01",
            "entry_timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
            "strategy": "sell_cash_secured_put", "highest_value": 2.00, "qty": 1,
        }}
        self.latch(True)
        result = om.manage_options_positions(100000, allow_rolls=True)
        mock_roll.assert_not_called()
        self.assertEqual([a["action"] for a in result["actions"]], ["CLOSE"])
        self.assertIn(om.ROLL_SKIPPED_KILL_LATCH_DETAIL, result["actions"][0]["reason"])
        mock_close.assert_called_once()
        self.assertEqual(mock_close.call_args[0][2], "buy", "buy-to-close the short put")


# ── on the source: every order POST is classified ───────────────────────────

ENGINE_FILES = ("bot_engine.py", "intraday_shorts.py", "options_manager.py", "options_execution.py")
OPENING = {"'third-leg VRP buy'", "'convexity put buy'", "'floor basket buy'", "'floor buy'",
           "'defensive floor buy'", "'intraday short entry'"}
SIDE_HELPER = "f'floor {side}'"          # _floor_vm_order: gated when side == "buy"
CLOSING = {"f'third-leg exit ({trigger})'", "'legacy SQQQ cleanup'", "'covered-call buy-back'",
           "'defensive floor exit'", "'defensive floor sell'", "'convexity put roll close'", "'floor sell'",
           "'floor zero-target exit'", "'floor basket sell'", "'intraday short cover'",
           "f'option close ({side})'", "'strategy close (mleg)'"}
# FROZEN order-submission internals: gated at their callers (evaluate_and_execute, bot.ts's scripts)
FROZEN_SUBMITTERS = {"f'option order ({side})'", "f'mleg order ({label})'"}


def _posts_with_gates(tree):
    """(post lineno, clock-gate label, latch-gate labels in the POST's own block) per order POST."""
    out = []
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
                    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "post"
                            and node.args and "v2/orders" in ast.unparse(node.args[0])):
                        continue
                    clock, latch = None, []
                    for s in stmts[:i]:
                        for n in ast.walk(s):
                            if isinstance(n, ast.Call) and len(n.args) > 1:
                                fn = ast.unparse(n.func)
                                if fn == "broker_clock.check_order_allowed":
                                    clock = ast.unparse(n.args[1])
                                elif fn == "kill_latch.check_opening_allowed":
                                    latch.append(ast.unparse(n.args[1]))
                    out.append((node.lineno, clock, latch))
    return out


class TestEveryOrderPostIsClassified(unittest.TestCase):
    def test_every_opening_post_reads_the_latch_and_no_exit_does(self):
        seen = 0
        for name in ENGINE_FILES:
            tree = ast.parse(open(os.path.join(_ROOT, name)).read())
            for lineno, clock, latch in _posts_with_gates(tree):
                seen += 1
                where = f"{name}:{lineno} ({clock})"
                if clock in OPENING:
                    self.assertEqual(latch, [clock], f"{where}: an opening order without the latch gate")
                elif clock == SIDE_HELPER:
                    self.assertEqual(latch, [SIDE_HELPER], f"{where}: the floor helper's buys are not latched")
                elif clock in CLOSING or clock in FROZEN_SUBMITTERS:
                    self.assertEqual(latch, [], f"{where}: an exit (or a frozen submitter) must never read the latch")
                else:
                    self.fail(f"{where}: an order POST with an unclassified label — list it as opening or closing here")
        self.assertEqual(seen, 21, "the same 21 POST sites test_broker_clock_gate.py counts")

    def test_bot_ts_rechecks_the_latch_right_before_its_etf_order(self):
        src = open(os.path.join(_ROOT, "server", "bot.ts"), encoding="utf-8").read()
        start = src.index("async function executeTrades(")
        post = src.index('const etfOrderResult = await alpaca("/v2/orders", {', start)
        shadow = src.rindex('source: "executeTrades.etf"', start, post)
        window = src[shadow:post]
        m = re.search(r'if \(state\.killSwitch\) \{\s*noteRemaining\(trade, "rejected_killswitch", "Node kill switch on — read before the ETF order, loop ended"\);\s*break;\s*\}',
                      window)
        self.assertIsNotNone(m, "the ETF substitution order re-checks the latch after this trade's awaits")
        self.assertNotIn("await ", window[m.end():], "no await between the re-check and the ETF order")


class TestTheSuiteIsHermetic(unittest.TestCase):
    def test_by_default_the_reader_looks_in_an_empty_directory(self):
        """conftest._hermetic_kill_latch: no test reads the machine's latch
        file (deploy_gate_smoke writes a LATCHED one under /tmp or
        /data/voltrade while it runs)."""
        paths = kill_latch.latch_paths()
        self.assertEqual(len(paths), 1)
        self.assertNotIn(os.path.dirname(paths[0]), ("/tmp", "/data/voltrade"))
        self.assertFalse(os.path.exists(paths[0]))
        self.assertFalse(kill_latch.is_latched())


if __name__ == "__main__":
    unittest.main()
