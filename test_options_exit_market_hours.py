"""REPAIR 2026-11-04 — a single-leg option close waits for regular hours, like
a group close.

What went wrong live (2026-10-29 11:31:35Z = 07:31 ET): Phase 2 of
manage_options_positions (the standalone single-leg rules) read
XLF261127P00055500's pre-market mark, fired LOSS LIMIT ("Cost to close $5.00
is 6.9x the credit"), and submitted a $5.00 DAY limit at the pre-market ask.
Options do not trade outside regular hours: the order sat until the open and
filled there at $1.61 against a fair value near $0.49. The group path
(_request_group_close) already deferred while the market was closed; the
single-leg path had no market-hours check at all.

Pinned here, for every single-leg rule (CRITICAL DTE, assignment risk, 21 DTE
sold and bought, profit target, sold and bought loss limit, gamma):
  (1) market closed: no order, no roll attempt, no exit feedback, the leg's
      state kept as it was, and one CLOSE action whose order is
      {"status": "deferred", "method": "none"} (the group path's shape, which
      the audit formatter already renders as "order deferred");
  (2) market open: the close is submitted exactly as before (same contract,
      qty, side and limit price), the state popped, the exit recorded;
  (3) the XLF case end to end: the pre-market LOSS LIMIT is deferred, and at
      the open the rule is judged again on regular-hours marks (no close at
      fair value);
  (4) a failed clock read still counts as open (_market_is_open semantics);
  (5) structurally, every _submit_close_order / _roll_or_skip call in
      manage_options_positions sits behind the regular-hours gate.
Mocked: no network (requests.get/post stand-ins), state in a temp dir.
"""
import ast
import inspect
import json
import os
import sys
import tempfile
import textwrap
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_manager as om  # noqa: E402


def occ(root, days, cp, strike):
    exp = (datetime.now() + timedelta(days=days)).strftime("%y%m%d")
    return f"{root}{exp}{cp}{int(round(strike * 1000)):08d}"


def position(symbol, qty, side, entry, cur):
    sign = 1 if side == "long" else -1
    return {"symbol": symbol, "qty": str(qty * sign), "side": side, "avg_entry_price": str(entry),
            "current_price": str(cur), "market_value": str(cur * qty * 100 * sign),
            "unrealized_pl": str((cur - entry) * qty * 100 * sign), "asset_class": "us_option"}


def leg(root, days, cp, strike, qty, side, entry, cur, bid, ask, delta, gamma, strategy, expect):
    """One standalone leg: its Alpaca position, its snapshot, the state
    register_options_entry() left (held 10 days, past the 60-minute hold
    gate), and the (close side, limit price) the rule submits at."""
    symbol = occ(root, days, cp, strike)
    held = (datetime.now() - timedelta(days=10)).isoformat()
    return {
        "occ": symbol, "qty": qty, "expect": expect,
        "pos": position(symbol, qty, side, entry, cur),
        "snap": {"bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 2), "delta": delta, "gamma": gamma,
                 "theta": -0.03, "vega": 0.05, "iv": 0.30},
        "state": {"entry_price": entry, "entry_delta": delta, "side": side, "qty": qty, "strategy": strategy,
                  "ticker": root, "entry_date": held[:10], "entry_timestamp": held,
                  "initial_credit": entry if side == "short" else 0,
                  "max_profit_target": entry * om.PROFIT_TARGET_PCT if side == "short" else 0,
                  "highest_value": entry},
    }


def book():
    """One leg per single-leg rule, each built so exactly that rule fires
    (Phase 2 order: CRITICAL DTE, assignment, 21 DTE, profit target, loss
    limit, gamma)."""
    return {
        # Check 1: <= 5 DTE, force close at the mid (ignores the hold gate)
        "dte_critical": leg("CRIT", 3, "P", 50, 1, "short", 2.10, 2.00, 1.90, 2.10, -0.40, 0.05,
                            "sell_cash_secured_put", ("buy", 2.00)),
        # Check 2: sold, |delta| > 0.80 at <= 10 DTE: roll, else buy back at the ask
        "assignment_close": leg("ASGN", 7, "P", 60, 1, "short", 2.00, 6.00, 5.90, 6.10, -0.90, 0.05,
                                "sell_cash_secured_put", ("buy", 6.10)),
        # Check 3: sold at <= 21 DTE: roll, else buy back at the ask
        "dte_close": leg("DTES", 15, "P", 50, 2, "short", 2.00, 1.80, 1.75, 1.85, -0.30, 0.02,
                         "sell_cash_secured_put", ("buy", 1.85)),
        # Check 3: bought at <= 21 DTE: sell at the bid
        "dte_close_bought": leg("DTEB", 18, "C", 500, 2, "long", 5.00, 4.00, 3.80, 4.20, 0.40, 0.04,
                                "buy_call", ("sell", 3.80)),
        # Check 4: sold at >= 50% of max profit: buy back at the mid
        "profit_target": leg("PRFT", 40, "P", 175, 1, "short", 3.00, 1.20, 1.10, 1.30, -0.15, 0.01,
                             "sell_cash_secured_put", ("buy", 1.20)),
        # Check 5: sold, cost to close >= 2x the credit: buy back at the ask.
        # The live case: a 0/5.00 pre-market quote on a $0.72 credit.
        "loss_limit": leg("XLF", 29, "P", 55.5, 1, "short", 0.72, 5.00, 0.00, 5.00, -0.20, 0.03,
                          "sell_cash_secured_put", ("buy", 5.00)),
        # Check 5: bought, down >= 50%: sell at the bid
        "bought_loss_limit": leg("BLOS", 40, "C", 520, 2, "long", 5.00, 1.80, 1.60, 2.00, 0.15, 0.01,
                                 "buy_call", ("sell", 1.60)),
        # Check 6: |gamma| > 0.08 inside 30 DTE: close at the mid
        "gamma_risk": leg("GAMA", 25, "C", 185, 1, "long", 4.00, 4.50, 4.30, 4.70, 0.55, 0.12,
                          "buy_call", ("sell", 4.50)),
    }


def quiet_leg():
    """60 DTE, 20% of max profit, low gamma, no delta drift: no rule fires."""
    return leg("QUIT", 60, "P", 40, 1, "short", 2.00, 1.60, 1.55, 1.65, -0.20, 0.01, "sell_cash_secured_put", None)


# every rule's reason starts with the same headline whether it is submitted or deferred
HEADLINE = {"dte_critical": "CRITICAL DTE", "assignment_close": "ASSIGNMENT RISK", "dte_close": "21 DTE",
            "dte_close_bought": "21 DTE", "profit_target": "PROFIT TARGET", "loss_limit": "LOSS LIMIT",
            "bought_loss_limit": "LOSS LIMIT", "gamma_risk": "GAMMA RISK"}


class Broker:
    """requests.get / requests.post stand-in: positions and the clock (the
    shape test_options_group_close.Broker serves). Nothing may be POSTed:
    closes go through the patched _submit_close_order."""

    def __init__(self, positions, is_open=True, clock_error=False):
        self.positions, self.is_open, self.clock_error = positions, is_open, clock_error
        self.gets, self.posts = [], []

    def get(self, url, **kw):
        self.gets.append(url)
        if "/v2/positions" in url:
            return MagicMock(status_code=200, json=lambda: self.positions)
        if "/v2/clock" in url:
            if self.clock_error:
                raise ConnectionError("clock unreachable")
            return MagicMock(status_code=200, json=lambda: {"is_open": self.is_open})
        return MagicMock(status_code=404, json=lambda: {}, text="")

    def post(self, url, **kw):
        self.posts.append((url, kw.get("json")))
        return MagicMock(status_code=500, text="no orders in tests", json=lambda: {})


class _ManagerCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state_path = os.path.join(tmp.name, "voltrade_options_state.json")
        self.snaps = {}
        self.submits = []
        self.track_fill = MagicMock()
        self.roll = MagicMock(return_value={"rolled": False, "detail": "no roll in tests"})

        def fake_submit(occ_symbol, qty, side, limit_price):
            self.submits.append((occ_symbol, qty, side, limit_price))
            return {"status": "submitted", "order_id": f"close-{len(self.submits)}", "detail": "test"}

        for p in (patch.object(om, "DATA_DIR", tmp.name),
                  patch.object(om, "OPTIONS_STATE_PATH", self.state_path),
                  patch.object(om, "_get_option_snapshot", side_effect=lambda s: dict(self.snaps.get(s, {}))),
                  patch.object(om, "_attempt_roll", self.roll),
                  patch.object(om, "_submit_close_order", side_effect=fake_submit),
                  patch.dict(sys.modules, {"ml_model_v2": MagicMock(track_fill=self.track_fill)})):
            p.start()
            self.addCleanup(p.stop)
        # _CLOCK_CACHE is module-wide: start cold and leave it cold, so no
        # "closed" reading from these tests leaks into another test's run.
        self._cold_clock()
        self.addCleanup(self._cold_clock)

    @staticmethod
    def _cold_clock():
        om._CLOCK_CACHE.update({"at": 0.0, "is_open": None})

    def load(self, legs, *extra):
        all_legs = list(legs.values()) + list(extra)
        self.snaps.update({l["occ"]: l["snap"] for l in all_legs})
        with open(self.state_path, "w") as f:
            json.dump({l["occ"]: dict(l["state"]) for l in all_legs}, f)

    def read_state(self):
        with open(self.state_path) as f:
            return json.load(f)

    def run_manager(self, broker, **kw):
        self._cold_clock()  # every run reads its own broker's clock
        with patch.object(om.requests, "get", side_effect=broker.get), \
                patch.object(om.requests, "post", side_effect=broker.post):
            return om.manage_options_positions(100000, **kw)


class TestMarketClosedDefersEverySingleLegRule(_ManagerCase):
    def test_no_order_no_roll_no_exit_state_kept_one_deferred_close_per_rule(self):
        legs, quiet = book(), quiet_leg()
        self.load(legs, quiet)
        before = self.read_state()
        broker = Broker([l["pos"] for l in legs.values()] + [quiet["pos"]], is_open=False)
        result = self.run_manager(broker, allow_rolls=True)

        self.assertEqual(self.submits, [], "a single-leg close was submitted outside regular hours")
        self.assertEqual(broker.posts, [], "nothing is POSTed to the broker outside regular hours")
        self.roll.assert_not_called()  # a roll is a close + a sell-to-open: neither outside regular hours
        self.track_fill.assert_not_called()  # nothing was closed, so no exit is recorded
        self.assertEqual(result["positions_checked"], 9)
        saved = self.read_state()
        for rule, l in legs.items():
            with self.subTest(rule=rule):
                mine = [a for a in result["actions"] if a.get("occ_symbol") == l["occ"]]
                self.assertEqual([a["action"] for a in mine], ["CLOSE"], mine)
                act = mine[0]
                self.assertEqual(act["type"], rule)
                self.assertEqual(act["order"]["status"], "deferred", act)
                self.assertTrue(act["reason"].startswith(HEADLINE[rule]), act["reason"])
                self.assertIn(l["occ"], saved, "a deferred leg's state is kept, never popped")
                for k in ("entry_timestamp", "entry_price", "initial_credit", "strategy"):
                    self.assertEqual(saved[l["occ"]][k], before[l["occ"]][k], k)
        self.assertEqual([a for a in result["actions"] if a.get("occ_symbol") == quiet["occ"]], [],
                         "a leg no rule fires on gets no deferred row")
        self.assertIn(quiet["occ"], saved)

    def test_rolls_off_too(self):
        """exits-only / shadow mode (allow_rolls=False) defers the same way."""
        legs = book()
        self.load(legs)
        result = self.run_manager(Broker([l["pos"] for l in legs.values()], is_open=False), allow_rolls=False)
        self.assertEqual(self.submits, [])
        self.assertEqual(sorted(a["type"] for a in result["actions"] if a["action"] == "CLOSE"), sorted(legs))
        self.assertTrue(all(a["order"]["status"] == "deferred" for a in result["actions"] if a["action"] == "CLOSE"))


class TestMarketOpenSubmitsExactlyAsBefore(_ManagerCase):
    def test_each_rule_submits_the_same_order_pops_state_and_records_the_exit(self):
        legs, quiet = book(), quiet_leg()
        self.load(legs, quiet)
        broker = Broker([l["pos"] for l in legs.values()] + [quiet["pos"]], is_open=True)
        result = self.run_manager(broker, allow_rolls=True)

        self.assertEqual(sorted(self.submits), sorted((l["occ"], l["qty"]) + l["expect"] for l in legs.values()))
        self.assertEqual(self.roll.call_count, 2, "the assignment and 21-DTE sold legs still try to roll first")
        saved = self.read_state()
        for rule, l in legs.items():
            with self.subTest(rule=rule):
                mine = [a for a in result["actions"] if a.get("occ_symbol") == l["occ"]]
                self.assertEqual([a["action"] for a in mine], ["CLOSE"], mine)
                self.assertEqual(mine[0]["type"], rule)
                self.assertEqual(mine[0]["order"]["status"], "submitted")
                self.assertTrue(mine[0]["reason"].startswith(HEADLINE[rule]), mine[0]["reason"])
                self.assertNotIn(l["occ"], saved, "a submitted close drops the leg's state, as before")
        self.assertEqual(sorted(c.args[0]["exit_reason"] for c in self.track_fill.call_args_list), sorted(legs))
        self.assertIn(quiet["occ"], saved)
        self.assertEqual([a for a in result["actions"] if a.get("occ_symbol") == quiet["occ"]], [])


class TestXlfPreMarketLossLimit(_ManagerCase):
    """The live case, 2026-10-29: a pre-market LOSS LIMIT on a 0/5.00 quote."""

    def test_premarket_mark_defers_and_the_open_judges_again_on_regular_hours_marks(self):
        xlf = book()["loss_limit"]
        self.assertTrue(xlf["occ"].startswith("XLF") and xlf["occ"].endswith("P00055500"))
        self.load({"loss_limit": xlf})
        before = self.read_state()[xlf["occ"]]

        # 07:31 ET: the pre-market mark ($5.00, 6.9x the $0.72 credit) trips LOSS LIMIT
        result = self.run_manager(Broker([xlf["pos"]], is_open=False))
        acts = [a for a in result["actions"] if a["action"] == "CLOSE"]
        self.assertEqual(len(acts), 1, result["actions"])
        self.assertIn("6.9x", acts[0]["reason"])
        self.assertEqual(acts[0]["order"]["status"], "deferred")
        self.assertEqual(self.submits, [], "the $5.00 DAY limit is never sent")
        self.track_fill.assert_not_called()
        kept = self.read_state()[xlf["occ"]]
        self.assertEqual(kept["entry_timestamp"], before["entry_timestamp"],
                         "state not re-initialized, so the 60-minute hold gate does not restart at the open")
        self.assertEqual(kept["initial_credit"], 0.72)

        # 09:31 ET: regular-hours marks near fair value; the rule is judged again and does not fire
        self.snaps[xlf["occ"]] = {"bid": 0.47, "ask": 0.51, "mid": 0.49, "delta": -0.18, "gamma": 0.03,
                                  "theta": -0.02, "vega": 0.04, "iv": 0.22}
        result = self.run_manager(Broker([position(xlf["occ"], 1, "short", 0.72, 0.49)], is_open=True))
        self.assertEqual([a for a in result["actions"] if a["action"] == "CLOSE"], [], "no close at fair value")
        self.assertEqual(self.submits, [])
        self.assertIn(xlf["occ"], self.read_state())

    def test_a_failed_clock_read_counts_as_open(self):
        xlf = book()["loss_limit"]
        self.load({"loss_limit": xlf})
        broker = Broker([xlf["pos"]], clock_error=True)
        result = self.run_manager(broker)
        self.assertTrue(any("/v2/clock" in u for u in broker.gets), "the single-leg path consults the clock")
        self.assertEqual(self.submits, [(xlf["occ"], 1, "buy", 5.00)], "a clock outage never withholds a close")
        self.assertEqual([a["order"]["status"] for a in result["actions"] if a["action"] == "CLOSE"], ["submitted"])


class TestDeferredActionShape(_ManagerCase):
    def test_group_deferral_shape_and_no_phantom_shadow_roll_open(self):
        legs = book()
        self.load(legs)
        result = self.run_manager(Broker([l["pos"] for l in legs.values()], is_open=False),
                                  allow_rolls=False, roll_skip_detail=om.ROLL_SKIPPED_SHADOW_DETAIL)
        deferred = [a for a in result["actions"] if a["action"] == "CLOSE"]
        self.assertEqual(len(deferred), len(legs))
        for a in deferred:
            with self.subTest(rule=a["type"]):
                # _request_group_close's deferral: {"status": "deferred", "method": "none", "detail": "market closed — ..."}
                self.assertEqual((a["order"]["status"], a["order"]["method"]), ("deferred", "none"))
                self.assertIn("market closed", a["order"]["detail"])
                self.assertEqual(a["ticker"], om._parse_occ_symbol(a["occ_symbol"])["ticker"])
                self.assertIsInstance(a["dte"], int)
                # bot_engine hands these actions to entry_shadow.roll_skip_entries: a
                # deferred roll site never reached the roll, so it is no skipped roll-open
                self.assertNotIn(om.ROLL_SKIPPED_SHADOW_DETAIL, a["reason"])
                self.assertNotIn(om.ROLL_SKIPPED_HALTED_DETAIL, a["reason"])
        import entry_shadow
        with patch.object(entry_shadow, "_underlying_price", return_value=50.0):
            self.assertEqual(entry_shadow.roll_skip_entries(result["actions"]), [])


class TestEveryCloseSitsBehindTheGate(unittest.TestCase):
    """A future single-leg rule that submits (or rolls) without the
    regular-hours gate fails here: in manage_options_positions every
    _submit_close_order / _roll_or_skip call must come after an
    `if _defer_single_leg_close(...): continue` in the same or an enclosing
    block."""

    ORDERING = ("_submit_close_order", "_roll_or_skip")

    def test_no_close_or_roll_outside_the_gate(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(om.manage_options_positions)))
        seen, ungated = [], []

        def is_gate(stmt):
            return (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Call)
                    and isinstance(stmt.test.func, ast.Name) and stmt.test.func.id == "_defer_single_leg_close"
                    and len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Continue) and not stmt.orelse)

        def flag(node, gated):
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.ORDERING:
                    seen.append(n.func.id)
                    if not gated:
                        ungated.append(f"{n.func.id} (line {n.lineno} of manage_options_positions)")

        def visit(stmts, gated):
            for stmt in stmts:
                if is_gate(stmt):
                    gated = True
                    continue
                blocks = [v for _f, v in ast.iter_fields(stmt) if isinstance(v, list) and v and isinstance(v[0], ast.stmt)]
                handlers = getattr(stmt, "handlers", None) or []
                if blocks or handlers:
                    for _f, v in ast.iter_fields(stmt):
                        if isinstance(v, ast.expr):
                            flag(v, gated)
                    for block in blocks:
                        visit(block, gated)
                    for h in handlers:
                        visit(h.body, gated)
                else:
                    flag(stmt, gated)

        visit(tree.body, False)
        self.assertGreaterEqual(seen.count("_submit_close_order"), 8, "the walk must see every close site")
        self.assertGreaterEqual(seen.count("_roll_or_skip"), 2, "the walk must see both roll sites")
        self.assertEqual(ungated, [], "a single-leg close or roll that can run outside regular hours")


if __name__ == "__main__":
    unittest.main()
