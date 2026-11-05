"""MID-FIRST EXIT LADDER (2026-11-05): a single-leg option exit that crosses the
spread starts at the mid and steps to the touch (order_ladder.py).

Human directive: build it behind a config flag; then "as much on as possible
... we have a paper account so use it", so the default is ON in paper too.
Alpaca paper fills a limit only once it is marketable at the NBBO, so in paper
the ladder tests the mechanics and fills at the touch after the last step; no
fill-improvement claim is made anywhere.

Pinned here (the default config, i.e. ON, unless a test says otherwise):
  - mode resolution: on / off / auto (auto = on only against the live host),
    the env override, the bounds on steps / step_seconds;
  - the price schedule: mid first, valid increments ($0.01 under $3, $0.05 at
    or above), never backing off, never past the touch or the far side, the
    last step the touch exactly;
  - only a close whose own price crosses the spread is laddered: mid-priced
    rules, a quote with no bid, rolls and group per-leg closes go out exactly
    as before; switched off, every rule sends today's single order;
  - tick by tick through manage_options_positions: step 1 at the mid; a rule
    firing again sends no second order and records no second exit; a step
    every step_seconds (never more than one per tick); the touch is held, never
    a market order; the fill is logged with its step;
  - races: a replace that loses to a fill, a refused replace (cancel, then the
    touch next tick), a partial fill (cancel, the rest at the next step), the
    position gone while the order works (canceled), the market closed or the
    opening window (no step), a stale ladder (dropped untouched), an order read
    failure (judged again), and a switched-off ladder finishing its order.
Existing exit tests (test_options_exit_market_hours.py and the rest) patch
_submit_close_order, so they pin the rules' prices and are unchanged.
"""
import ast
import inspect
import json
import math
import os
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_manager as om  # noqa: E402
import order_ladder  # noqa: E402
import system_config  # noqa: E402
from test_options_exit_market_hours import Broker, _ManagerCase, book, leg, position  # noqa: E402
from test_options_exit_market_hours import occ as make_occ  # noqa: E402

REAL_SUBMIT = om._submit_close_order
REAL_ROLL = om._attempt_roll
PAPER = "https://paper-api.alpaca.markets"
LIVE = "https://api.alpaca.markets"


# ── pure: mode, schedule, plan, pause ───────────────────────────────────────

class TestMode(unittest.TestCase):
    def test_default_is_on_in_paper(self):
        with patch.dict(os.environ, {}):
            os.environ.pop(order_ladder.ENV_VAR, None)
            s = order_ladder.settings(PAPER)
        self.assertEqual(system_config.BASE_CONFIG["OPTIONS_EXIT_LADDER"], {"mode": "on", "steps": 3, "step_seconds": 60})
        self.assertEqual((s["on"], s["steps"], s["step_seconds"]), (True, 3, 60.0))

    def test_auto_is_on_only_against_the_live_host(self):
        r = order_ladder.resolve_mode
        self.assertEqual(r("auto", None, PAPER), "off")
        self.assertEqual(r("auto", None, LIVE), "on")
        self.assertEqual(r("auto", None, LIVE + "/"), "on")
        for other in ("", None, "http://localhost:8080", "https://api.alpaca.markets.evil.com", "not a url"):
            self.assertEqual(r("auto", None, other), "off", other)

    def test_the_env_override_wins_and_anything_unknown_is_off(self):
        r = order_ladder.resolve_mode
        self.assertEqual(r("on", "off", LIVE), "off")
        self.assertEqual(r("off", "on", PAPER), "on")
        self.assertEqual(r("on", "auto", PAPER), "off")
        self.assertEqual(r("off", " AUTO ", LIVE), "on")
        self.assertEqual(r("on", "   ", PAPER), "on", "a blank env value is no override")
        self.assertEqual(r("on", "yes", PAPER), "off", "an unreadable override fails closed")
        self.assertEqual(r("sometimes", None, LIVE), "off")
        with patch.dict(os.environ, {order_ladder.ENV_VAR: "off"}):
            self.assertFalse(order_ladder.settings(PAPER)["on"])

    def test_bounds_and_a_missing_key(self):
        cfg = system_config.BASE_CONFIG["OPTIONS_EXIT_LADDER"]
        with patch.dict(os.environ, {}), patch.dict(cfg, {"steps": 99, "step_seconds": -5}):
            os.environ.pop(order_ladder.ENV_VAR, None)
            s = order_ladder.settings(PAPER)
            self.assertEqual((s["steps"], s["step_seconds"]), (order_ladder.MAX_STEPS, 0.0))
        with patch.dict(os.environ, {}), patch.dict(cfg, {"steps": "x", "step_seconds": float("nan")}):
            os.environ.pop(order_ladder.ENV_VAR, None)
            s = order_ladder.settings(PAPER)
            self.assertEqual((s["steps"], s["step_seconds"]), (3, 60.0))
        with patch.dict(os.environ, {}), patch.dict(system_config.BASE_CONFIG, {"OPTIONS_EXIT_LADDER": None}):
            os.environ.pop(order_ladder.ENV_VAR, None)
            self.assertFalse(order_ladder.settings(PAPER)["on"], "no config: off")


class TestSchedule(unittest.TestCase):
    P = staticmethod(order_ladder.ladder_prices)

    def test_mid_first_last_step_is_the_touch(self):
        self.assertEqual(self.P("sell", 1.20, 1.30, 3), [1.25, 1.23, 1.20])
        self.assertEqual(self.P("buy", 1.20, 1.30, 3), [1.25, 1.27, 1.30])
        self.assertEqual(self.P("sell", 0.47, 0.57, 5), [0.52, 0.51, 0.50, 0.49, 0.47])
        self.assertEqual(self.P("sell", 1.20, 1.30, 2), [1.25, 1.20])
        self.assertEqual(self.P("buy", 1.20, 1.30, 1), [1.30], "one step is today's order")

    def test_nickel_increments_at_and_above_three_dollars(self):
        self.assertEqual(self.P("sell", 3.10, 3.40, 3), [3.25, 3.20, 3.10])
        self.assertEqual(self.P("buy", 3.10, 3.40, 3), [3.25, 3.30, 3.40])
        self.assertEqual(self.P("sell", 2.90, 3.30, 3), [3.10, 3.00, 2.90], "across $3.00")
        self.assertEqual(self.P("buy", 2.80, 3.20, 3), [3.00, 3.10, 3.20])

    def test_every_schedule_is_valid_monotone_and_inside_the_quote(self):
        for bid_c in range(5, 700, 7):
            for width in (1, 2, 3, 5, 10, 17, 40):
                for steps in (2, 3, 4, 6):
                    for side in ("sell", "buy"):
                        bid, ask = bid_c / 100, (bid_c + width) / 100
                        prices = self.P(side, bid, ask, steps)
                        cents = [round(p * 100) for p in prices]
                        with self.subTest(side=side, bid=bid, ask=ask, steps=steps):
                            self.assertEqual(len(cents), steps)
                            self.assertEqual(cents[-1], bid_c if side == "sell" else bid_c + width, "last step = touch")
                            self.assertTrue(all(bid_c <= c <= bid_c + width for c in cents))
                            pairs = list(zip(cents, cents[1:]))
                            self.assertTrue(all((b <= a) if side == "sell" else (b >= a) for a, b in pairs), "never backs off")
                            for c in cents[:-1]:
                                self.assertTrue(c < 300 or c % 5 == 0 or c == bid_c + width or c == bid_c,
                                                f"{c} is not a valid increment")

    def test_unusable_quotes_give_no_schedule(self):
        for bid, ask in ((0, 1.0), (1.0, 0), (1.2, 1.1), (None, 1.0), (float("nan"), 1.0), ("x", 1.0)):
            self.assertEqual(self.P("sell", bid, ask, 3), [], (bid, ask))
        self.assertEqual(self.P("hold", 1.0, 1.1, 3), [])


class TestPlan(unittest.TestCase):
    def test_only_a_spread_crossing_close_is_laddered(self):
        p = order_ladder.plan
        self.assertEqual(p("sell", 1.20, 1.20, 1.30, 3)["prices"], [1.25, 1.23, 1.20])
        self.assertEqual(p("sell", 1.10, 1.20, 1.30, 3)["prices"][-1], 1.20, "a price under the bid still crosses")
        self.assertEqual(p("buy", 1.30, 1.20, 1.30, 3)["prices"], [1.25, 1.27, 1.30])
        self.assertIn("inside the spread", p("sell", 1.25, 1.20, 1.30, 3)["skip"], "a mid-priced rule")
        self.assertIn("inside the spread", p("buy", 1.29, 1.20, 1.30, 3)["skip"])
        self.assertIn("no two-sided quote", p("buy", 5.00, 0.00, 5.00, 3)["skip"], "XLF's 0/5.00 quote")
        self.assertIn("strictly inside", p("sell", 1.00, 1.00, 1.01, 3)["skip"], "a one-tick market has no mid")
        self.assertIn("strictly inside", p("sell", 3.00, 3.00, 3.05, 3)["skip"])
        self.assertIn("1 step", p("sell", 1.20, 1.20, 1.30, 1)["skip"])


class TestPause(unittest.TestCase):
    def test_paused_nests_and_always_releases(self):
        self.assertFalse(order_ladder.is_paused())
        with order_ladder.paused():
            with order_ladder.paused():
                self.assertTrue(order_ladder.is_paused())
            self.assertTrue(order_ladder.is_paused())
        self.assertFalse(order_ladder.is_paused())
        with self.assertRaises(RuntimeError):
            with order_ladder.paused():
                raise RuntimeError("boom")
        self.assertFalse(order_ladder.is_paused())


# ── a broker with an order book ─────────────────────────────────────────────

class LadderBroker(Broker):
    """Broker plus the order endpoints the ladder uses, in Alpaca's shapes:
    POST /v2/orders, GET /v2/orders/{id}, PATCH (replace) and DELETE (cancel)."""

    def __init__(self, positions, is_open=True):
        super().__init__(positions, is_open=is_open)
        self.orders, self.patches, self.deletes, self.chain = {}, [], [], {}
        self.refuse_patch = False           # PATCH answers 422 while the order stays working
        self.fill_before_replace = False    # the old order fills first; its replacement is rejected
        self.order_read_error = False

    def _book(self, payload, replaces=None):
        oid = f"ord-{len(self.orders) + 1}"
        self.orders[oid] = {"id": oid, "symbol": payload.get("symbol"), "qty": str(payload.get("qty")),
                            "side": payload.get("side"), "type": "limit",
                            "time_in_force": payload.get("time_in_force", "day"),
                            "limit_price": str(payload.get("limit_price")), "status": "new", "filled_qty": "0",
                            "filled_avg_price": None, "replaces": replaces, "replaced_by": None}
        return self.orders[oid]

    def get(self, url, **kw):
        if "/v2/orders/" in url:
            self.gets.append(url)
            if self.order_read_error:
                raise ConnectionError("orders endpoint down")
            o = self.orders.get(url.rsplit("/", 1)[1])
            if o is None:
                return MagicMock(status_code=404, text="order not found", json=lambda: {"message": "order not found"})
            body = dict(o)
            return MagicMock(status_code=200, json=lambda: body)
        if "/v1beta1/options/snapshots/" in url:
            self.gets.append(url)
            return MagicMock(status_code=200, json=lambda: {"snapshots": self.chain})
        return super().get(url, **kw)

    def post(self, url, **kw):
        payload = kw.get("json") or {}
        self.posts.append((url, payload))
        if payload.get("order_class") == "mleg":
            return MagicMock(status_code=422, text='{"message":"mleg refused in tests"}', json=lambda: {})
        body = dict(self._book(payload))
        return MagicMock(status_code=200, json=lambda: body, text="")

    def patch(self, url, **kw):
        oid, payload = url.rsplit("/", 1)[1], kw.get("json") or {}
        self.patches.append((oid, payload))
        old = self.orders.get(oid)
        if old is None or old["status"] != "new" or self.refuse_patch:
            return MagicMock(status_code=422, text='{"message":"order is not replaceable"}', json=lambda: {})
        new = self._book({**old, "limit_price": payload.get("limit_price")}, replaces=oid)
        old.update({"status": "replaced", "replaced_by": new["id"]})
        if self.fill_before_replace:
            old.update({"status": "filled", "filled_qty": old["qty"], "filled_avg_price": old["limit_price"]})
            new["status"] = "rejected"
            self.positions = [p for p in self.positions if p["symbol"] != old["symbol"]]
        body = {**new, "status": "pending_new"}   # what the replace call itself answers
        return MagicMock(status_code=200, json=lambda: body, text="")

    def delete(self, url, **kw):
        oid = url.rsplit("/", 1)[1]
        self.deletes.append(oid)
        o = self.orders.get(oid)
        if o and o["status"] in ("new", "partially_filled"):
            o["status"] = "canceled"
            return MagicMock(status_code=204, text="")
        return MagicMock(status_code=422, text="not cancelable")

    # test controls
    def working(self, symbol):
        return [o for o in self.orders.values() if o["symbol"] == symbol and o["status"] in ("new", "partially_filled")]

    def fill(self, symbol, price=None):
        (o,) = self.working(symbol)
        o.update({"status": "filled", "filled_qty": o["qty"],
                  "filled_avg_price": str(price if price is not None else o["limit_price"])})
        self.positions = [p for p in self.positions if p["symbol"] != symbol]

    def partial(self, symbol, filled):
        (o,) = self.working(symbol)
        o.update({"status": "partially_filled", "filled_qty": str(filled), "filled_avg_price": o["limit_price"]})
        for p in self.positions:
            if p["symbol"] == symbol:
                sign = -1 if p["side"] == "short" else 1
                p["qty"] = str((abs(int(float(p["qty"]))) - filled) * sign)


def sent(broker):
    """(symbol, side, qty, limit) of every non-mleg POST."""
    return [(p["symbol"], p["side"], int(p["qty"]), float(p["limit_price"]))
            for _u, p in broker.posts if p.get("order_class") != "mleg"]


class _LadderCase(_ManagerCase):
    """_ManagerCase with the REAL _submit_close_order (the ladder lives in it),
    a controllable ladder clock and the default config (ON)."""

    def setUp(self):
        super().setUp()
        self.now = [1_800_000_000.0]
        for p in (patch.object(om, "_submit_close_order", REAL_SUBMIT),
                  patch.object(order_ladder, "wall_clock", lambda: self.now[0]),
                  patch.dict(os.environ, {})):
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop(order_ladder.ENV_VAR, None)

    def broker_calls(self, broker):
        return (patch.object(om.requests, "get", side_effect=broker.get),
                patch.object(om.requests, "post", side_effect=broker.post),
                patch.object(om.requests, "patch", side_effect=broker.patch),
                patch.object(om.requests, "delete", side_effect=broker.delete))

    def tick(self, broker, seconds=0.0, **kw):
        self.now[0] += seconds
        self._cold_clock()
        g, p, pa, d = self.broker_calls(broker)
        with g, p, pa, d:
            return om.manage_options_positions(100000, **kw)

    def ladders(self):
        path = os.path.join(om.DATA_DIR, order_ladder.STATE_FILE)
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def ladder_log(self):
        path = os.path.join(om.DATA_DIR, order_ladder.LOG_FILE)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def of(result, action):
        return [a for a in result["actions"] if a["action"] == action]

    def start_one(self):
        """BLOS: 2 bought calls down 64% (bid 1.60 / ask 2.00). The bought LOSS
        LIMIT sells at the bid; the ladder starts at the mid, 1.80."""
        lg = book()["bought_loss_limit"]
        self.load({"x": lg})
        broker = LadderBroker([lg["pos"]])
        self.tick(broker)
        self.assertEqual(sent(broker), [(lg["occ"], "sell", 2, 1.80)])
        return lg, broker


# ── the default (ON) through the manager ────────────────────────────────────

class TestDefaultOnThroughTheManager(_LadderCase):
    LADDERED = {"assignment_close": 6.00, "dte_close": 1.80, "dte_close_bought": 4.00, "bought_loss_limit": 1.80}

    def test_spread_crossing_rules_start_at_the_mid_and_the_rest_go_out_as_before(self):
        legs = book()
        self.load(legs)
        broker = LadderBroker([l["pos"] for l in legs.values()])
        result = self.tick(broker, allow_rolls=True)

        by_occ = {s[0]: s[1:] for s in sent(broker)}
        self.assertEqual(len(sent(broker)), len(legs), "one order per contract")
        for rule, l in legs.items():
            side, rule_px = l["expect"]
            with self.subTest(rule=rule):
                # mid-priced rules (CRITICAL DTE, profit target, gamma) and XLF's 0/5.00
                # quote (no bid) send the rule's own price, exactly as before
                self.assertEqual(by_occ[l["occ"]], (side, l["qty"], self.LADDERED.get(rule, rule_px)))
        self.assertEqual(sorted(self.ladders()), sorted(legs[r]["occ"] for r in self.LADDERED))
        for rule in self.LADDERED:
            (act,) = [a for a in self.of(result, "CLOSE") if a["occ_symbol"] == legs[rule]["occ"]]
            self.assertEqual(act["order"]["status"], "submitted")
            self.assertEqual((act["order"]["ladder"]["step"], act["order"]["ladder"]["of"]), (1, 3))
            self.assertEqual(act["order"]["ladder"]["plan"][-1], legs[rule]["expect"][1], "the last step is the rule's touch")
        self.assertEqual(sorted(c.args[0]["exit_reason"] for c in self.track_fill.call_args_list), sorted(legs),
                         "one exit record per rule, as before")

    def test_switched_off_every_rule_sends_todays_single_order(self):
        os.environ[order_ladder.ENV_VAR] = "off"
        legs = book()
        self.load(legs)
        broker = LadderBroker([l["pos"] for l in legs.values()])
        self.tick(broker, allow_rolls=True)
        self.assertEqual(sorted(sent(broker)), sorted((l["occ"], l["expect"][0], l["qty"], l["expect"][1]) for l in legs.values()))
        for _u, payload in broker.posts:
            self.assertEqual(sorted(payload), ["limit_price", "qty", "side", "symbol", "time_in_force", "type"])
        self.assertEqual((broker.patches, broker.deletes), ([], []))
        self.assertFalse(os.path.exists(os.path.join(om.DATA_DIR, order_ladder.STATE_FILE)))


class TestLadderTickByTick(_LadderCase):
    def test_mid_then_steps_then_the_touch_then_the_fill(self):
        lg, broker = self.start_one()
        occ = lg["occ"]
        self.assertEqual(self.track_fill.call_count, 1)
        self.assertEqual(self.ladders()[occ]["step"], 1)

        # 30 s later: nothing to do (the rule's state was popped; the hold gate keeps it quiet)
        self.tick(broker, 30)
        self.assertEqual((len(broker.posts), broker.patches), (1, []))

        # the rule fires again before step_seconds: no second order, no second exit record
        self.load({"x": lg})
        r = self.tick(broker, 10)
        (close,) = self.of(r, "CLOSE")
        self.assertEqual(close["order"]["status"], "working")
        self.assertEqual((close["order"]["ladder"]["step"], close["order"]["order_id"]), (1, "ord-1"))
        self.assertEqual((len(broker.posts), broker.patches), (1, []))
        self.assertEqual(self.track_fill.call_count, 1, "a re-fire records no second exit")

        # 61 s after step 1: step 2, by replacing the order
        r = self.tick(broker, 21)
        self.assertEqual([(oid, float(p["limit_price"])) for oid, p in broker.patches], [("ord-1", 1.70)])
        (step,) = self.of(r, "LADDER")
        self.assertEqual((step["type"], step["order"]["status"], step["order"]["ladder"]["step"]), ("exit_ladder_step", "replaced", 2))

        # one step per tick at most: a tick right after does nothing
        self.tick(broker, 1)
        self.assertEqual(len(broker.patches), 1)

        # step 3 is the touch
        self.tick(broker, 60)
        self.assertEqual([float(p["limit_price"]) for _o, p in broker.patches], [1.70, 1.60])
        self.assertEqual(self.ladders()[occ]["step"], 3)

        # the touch is held: no chase, no market order, no second order
        self.tick(broker, 900)
        self.assertEqual(len(broker.patches), 2)
        self.assertEqual(len(broker.working(occ)), 1)
        self.assertTrue(all(o["type"] == "limit" for o in broker.orders.values()))
        self.assertEqual(len(broker.posts), 1)

        # it fills between ticks; the position is gone
        broker.fill(occ, 1.60)
        r = self.tick(broker, 60)
        (done,) = self.of(r, "LADDER")
        self.assertEqual((done["type"], done["order"]["status"]), ("exit_ladder_filled", "filled"))
        self.assertIn("FILLED at step 3/3 (the touch) at $1.60", done["reason"])
        (rec,) = self.ladder_log()
        self.assertEqual((rec["outcome"], rec["filled_step"], rec["steps"], rec["fill_price"]), ("filled", 3, 3, 1.60))
        self.assertEqual([h["price"] for h in rec["history"]], [1.80, 1.70, 1.60])
        self.assertEqual([h["via"] for h in rec["history"]], ["post", "replace", "replace"])
        self.assertEqual(self.ladders(), {})
        self.assertEqual(self.track_fill.call_count, 1, "one close, one exit record")

    def test_a_step_never_backs_off_when_the_quote_moves_our_way(self):
        lg, broker = self.start_one()          # selling at 1.80 (bid 1.60 / ask 2.00)
        self.snaps[lg["occ"]].update({"bid": 1.90, "ask": 2.30, "mid": 2.10})
        r = self.tick(broker, 61)              # step 2 would be 2.00 on the new quote: it stays 1.80
        (step,) = self.of(r, "LADDER")
        self.assertIn("limit stays $1.80", step["reason"])
        self.tick(broker, 60)                  # step 3 would be the new bid 1.90: still 1.80
        self.assertEqual(broker.patches, [], "a sell is never re-priced up")
        self.assertEqual((self.ladders()[lg["occ"]]["step"], self.ladders()[lg["occ"]]["price"]), (3, 1.80))

    def test_a_fill_before_the_last_step_is_logged_with_its_step(self):
        lg, broker = self.start_one()
        broker.fill(lg["occ"], 1.80)
        r = self.tick(broker, 30)
        (done,) = self.of(r, "LADDER")
        self.assertIn("FILLED at step 1/3 (the mid) at $1.80", done["reason"])
        (rec,) = self.ladder_log()
        self.assertEqual((rec["filled_step"], rec["fill_price"], rec["fills"][0]["step"]), (1, 1.80, 1))
        self.assertEqual((broker.patches, self.ladders()), ([], {}))

    def test_an_order_that_ends_unfilled_ends_the_ladder_and_the_rule_starts_a_new_one(self):
        lg, broker = self.start_one()
        broker.orders["ord-1"]["status"] = "canceled"     # e.g. a deploy's cancel-all
        r = self.tick(broker, 30)
        (ended,) = self.of(r, "LADDER")
        self.assertEqual((ended["type"], ended["order"]["status"]), ("exit_ladder_ended", "canceled"))
        self.assertEqual(self.ladders(), {})
        self.load({"x": lg})                                # the rule judges again
        self.tick(broker, 30)
        self.assertEqual(sent(broker), [(lg["occ"], "sell", 2, 1.80)] * 2, "a fresh ladder starts at the mid")
        self.assertEqual(self.track_fill.call_count, 2)


class TestLadderRaces(_LadderCase):
    def test_a_replace_that_loses_to_a_fill_reports_the_earlier_step(self):
        lg, broker = self.start_one()
        broker.fill_before_replace = True
        self.tick(broker, 61)                  # the replace answers 200, but step 1 filled first
        self.assertEqual(len(broker.patches), 1)
        r = self.tick(broker, 60)              # the replacement reads rejected; the position is gone
        (done,) = self.of(r, "LADDER")
        self.assertEqual(done["type"], "exit_ladder_filled")
        (rec,) = self.ladder_log()
        self.assertEqual((rec["outcome"], rec["filled_step"], rec["fill_price"]), ("filled", 1, 1.80))
        self.assertEqual((len(broker.posts), self.ladders()), (1, {}), "nothing re-sent")

    def test_a_refused_replace_cancels_and_sends_the_touch_next_tick(self):
        lg, broker = self.start_one()
        broker.refuse_patch = True
        r = self.tick(broker, 61)
        (act,) = self.of(r, "LADDER")
        self.assertEqual((act["type"], act["order"]["status"]), ("exit_ladder_replace_refused", "cancel_requested"))
        self.assertEqual((broker.deletes, len(broker.posts)), (["ord-1"], 1), "never a new order in the cancel's tick")
        broker.is_open = False
        self.tick(broker, 60)
        self.assertEqual(len(broker.posts), 1, "the re-send waits for regular hours")
        broker.is_open = True
        r = self.tick(broker, 60)
        self.assertEqual(sent(broker)[-1], (lg["occ"], "sell", 2, 1.60), "the touch, for the whole position")
        self.assertEqual(len(broker.working(lg["occ"])), 1)
        self.assertEqual(self.ladders()[lg["occ"]]["step"], 3)
        broker.fill(lg["occ"])
        self.tick(broker, 60)
        self.assertEqual(self.ladder_log()[0]["filled_step"], 3)
        self.assertEqual(self.track_fill.call_count, 1)

    def test_a_partial_fill_is_canceled_and_the_rest_sent_at_the_next_step(self):
        lg, broker = self.start_one()
        broker.partial(lg["occ"], 1)           # one of two filled at the mid
        self.tick(broker, 61)
        self.assertEqual((broker.patches, broker.deletes), ([], ["ord-1"]), "a partly filled order is never replaced in place")
        self.tick(broker, 60)
        self.assertEqual(sent(broker)[-1], (lg["occ"], "sell", 1, 1.70), "the remaining one, at step 2")
        broker.fill(lg["occ"])
        self.tick(broker, 60)
        (rec,) = self.ladder_log()
        self.assertEqual([(f["step"], f["qty"]) for f in rec["fills"]], [(1, 1.0), (2, 1.0)])
        self.assertEqual((rec["filled_step"], rec["filled_qty"]), (2, 2.0))

    def test_a_working_order_whose_position_is_gone_is_canceled(self):
        lg, broker = self.start_one()
        broker.positions = []                  # closed by something else
        r = self.tick(broker, 30)
        self.assertEqual(broker.deletes, ["ord-1"])
        (ended,) = self.of(r, "LADDER")
        self.assertEqual(ended["order"]["status"], "position_gone")
        self.assertEqual(self.ladders(), {})

    def test_no_step_while_the_market_is_closed_or_in_the_opening_window(self):
        lg, broker = self.start_one()
        broker.is_open = False
        self.tick(broker, 120)
        with patch.object(om, "_in_opening_window", return_value=True):
            broker.is_open = True
            self.tick(broker, 60)
        self.assertEqual(broker.patches, [])
        self.tick(broker, 1)
        self.assertEqual([float(p["limit_price"]) for _o, p in broker.patches], [1.70])

    def test_a_ladder_from_an_earlier_session_is_dropped_untouched(self):
        lg, broker = self.start_one()
        broker.gets.clear()
        self.tick(broker, order_ladder.MAX_LADDER_AGE_SECONDS + 1)
        self.assertEqual((broker.patches, broker.deletes), ([], []))
        self.assertFalse(any("/v2/orders/" in u for u in broker.gets), "its order is not even read")
        self.assertEqual((self.ladders(), self.ladder_log()[0]["outcome"]), ({}, "stale"))

    def test_an_unreadable_order_is_judged_again_next_tick(self):
        lg, broker = self.start_one()
        broker.order_read_error = True
        self.tick(broker, 61)
        self.assertEqual((broker.patches, self.ladders()[lg["occ"]]["step"]), ([], 1))
        broker.order_read_error = False
        self.tick(broker, 1)
        self.assertEqual(len(broker.patches), 1)

    def test_switched_off_no_new_ladder_but_the_working_one_finishes(self):
        lg, broker = self.start_one()
        os.environ[order_ladder.ENV_VAR] = "off"
        other = book()["dte_close_bought"]     # sells at the bid 3.80; ON would start at 4.00
        self.load({"o": other})
        broker.positions.append(other["pos"])
        self.tick(broker, 61)
        self.assertEqual(sent(broker)[-1], (other["occ"], "sell", 2, 3.80), "off: today's single order")
        self.assertEqual([float(p["limit_price"]) for _o, p in broker.patches], [1.70], "the working ladder still steps")
        self.assertEqual(sorted(self.ladders()), [lg["occ"]])


class TestNeverLaddered(_LadderCase):
    def test_a_roll_closes_and_reopens_at_the_touch(self):
        old = make_occ("ROLL", 15, "P", 50)
        new = make_occ("ROLL", 45, "P", 50)
        self.snaps[old] = {"bid": 1.75, "ask": 1.85, "mid": 1.80}
        self.snaps[new] = {"bid": 2.10, "ask": 2.30, "mid": 2.20}
        broker = LadderBroker([])
        broker.chain = {new: {"latestQuote": {"bp": 2.10, "ap": 2.30}}}
        g, p, pa, d = self.broker_calls(broker)
        with g, p, pa, d:
            res = REAL_ROLL(old, 1, "sell", om._parse_occ_symbol(old), 100000)
        self.assertTrue(res["rolled"], res)
        self.assertEqual(sent(broker), [(old, "buy", 1, 1.85), (new, "sell", 1, 2.10)])
        self.assertEqual(self.ladders(), {})

    def test_a_group_close_fallback_sends_each_leg_at_its_touch(self):
        short, wing = make_occ("GRP", 30, "P", 100), make_occ("GRP", 30, "P", 95)
        legs = [position(short, 1, "short", 2.00, 1.50), position(wing, 1, "long", 1.00, 0.60)]
        self.snaps[short] = {"bid": 1.40, "ask": 1.60, "mid": 1.50}
        self.snaps[wing] = {"bid": 0.50, "ask": 0.70, "mid": 0.60}
        broker = LadderBroker(legs)
        g, p, pa, d = self.broker_calls(broker)
        with g, p, pa, d:
            res = om._close_strategy_mleg(legs, "GRP")
        self.assertEqual(res["method"], "individual_legs")
        self.assertEqual(sent(broker), [(short, "buy", 1, 1.60)], "the short bought back at the ask; the wing waits")
        self.assertEqual(self.ladders(), {})

    def test_every_close_outside_the_single_leg_rules_runs_paused(self):
        for fn in (REAL_ROLL, om._close_strategy_mleg, om._post_close_unladdered):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_submit_close_order"]
            inside = []
            for node in ast.walk(tree):
                if isinstance(node, ast.With) and any(
                        isinstance(i.context_expr, ast.Call) and getattr(i.context_expr.func, "attr", "") == "paused"
                        for i in node.items):
                    inside += [n for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_submit_close_order"]
            with self.subTest(fn=fn.__name__):
                self.assertTrue(calls)
                self.assertEqual(len(calls), len(inside), "every close here is wrapped in order_ladder.paused()")

    def test_the_pursuit_runs_before_every_rule_and_before_the_no_positions_return(self):
        src = inspect.getsource(om.manage_options_positions)
        at = src.index("actions.extend(_pursue_exit_ladders(options_positions))")
        self.assertLess(at, src.index("if not options_positions:"))
        self.assertLess(at, src.index("_manage_strategy_group(group, equity, state"))   # (the earnings close adds an argument)
        self.assertIn('return {"actions": actions, "positions_checked": 0', src)


class TestOffIsTodaysOrder(unittest.TestCase):
    """Switched off (env off, or auto against paper) the close is the one POST it
    always was: same payload, same result, no quote read, no ladder file."""

    def test_off_and_auto_on_paper(self):
        for env in ("off", "auto"):
            with self.subTest(env=env), tempfile.TemporaryDirectory() as tmp, \
                    patch.dict(os.environ, {order_ladder.ENV_VAR: env}), \
                    patch.object(om, "DATA_DIR", tmp), patch.object(om, "ALPACA_BASE", PAPER), \
                    patch.object(om, "_get_option_snapshot") as snap, \
                    patch.object(om.requests, "post", return_value=MagicMock(status_code=200, json=lambda: {"id": "o-1"})) as post:
                res = om._submit_close_order("BLOS261218C00520000", 2, "sell", 1.60)
                snap.assert_not_called()
                self.assertEqual(post.call_args.args[0], f"{PAPER}/v2/orders")
                self.assertEqual(post.call_args.kwargs["json"], {"symbol": "BLOS261218C00520000", "qty": "2", "side": "sell",
                                                                  "type": "limit", "limit_price": "1.6", "time_in_force": "day"})
                self.assertEqual(res, {"status": "submitted", "order_id": "o-1",
                                       "detail": "Close sell 2x BLOS261218C00520000 @ $1.60"})
                self.assertEqual(os.listdir(tmp), [])

    def test_a_step_one_the_broker_refuses_falls_back_to_the_rules_order(self):
        calls = []

        def post(url, **kw):
            calls.append(float(kw["json"]["limit_price"]))
            ok = len(calls) > 1
            return MagicMock(status_code=200 if ok else 422, text="invalid price increment", json=lambda: {"id": "o-2"})

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {order_ladder.ENV_VAR: "on"}), \
                patch.object(om, "DATA_DIR", tmp), \
                patch.object(om, "_get_option_snapshot", return_value={"bid": 1.60, "ask": 2.00, "mid": 1.80}), \
                patch.object(om.requests, "post", side_effect=post):
            res = om._submit_close_order("BLOS261218C00520000", 2, "sell", 1.60)
            self.assertEqual(calls, [1.80, 1.60], "the mid refused, then the rule's own order")
            self.assertEqual((res["status"], res["order_id"]), ("submitted", "o-2"))
            self.assertNotIn("ladder", res)
            self.assertFalse(os.path.exists(os.path.join(tmp, order_ladder.STATE_FILE)))


if __name__ == "__main__":
    unittest.main()
