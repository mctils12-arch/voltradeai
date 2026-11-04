"""REPAIR 2026-11-02 — a multi-leg strategy is closed AS A UNIT and the close
is PURSUED until the broker no longer holds the legs (the GOOGL Nov-27 iron
condor regression, research/experiments.md 2026-11-02).

What went wrong live: every group exit submitted one close and popped the
legs' state on the spot. The 10-22 close's mleg was rejected (reason never
logged), the fallback placed per-leg DAY limits at a 0 after-hours mark, the
orders died, and from then on the four legs were "unknown" standalone
options: the single-leg rules tried to SELL the long wings (Alpaca:
"uncovered option contracts" / "insufficient options buying power") and to
buy back the shorts at 2× credit — one at a time, into an earnings gap.

Pinned here: (1) the fallback buys back the SHORT legs first at the ask and
defers the wings; (2) the legs keep their strategy and carry the close
request — never popped; (3) the next tick re-submits when nothing is working
at the broker, waits while something is; (4) nothing is submitted outside
regular hours; (5) a wings-only remainder is sold at the bid; (6) a leg that
lost its label is adopted by its siblings and never runs the single-leg
rules; (7) the mleg POST itself is unchanged.
"""
import inspect
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_manager as om  # noqa: E402


def occ(root, days, cp, strike):
    exp = (datetime.now() + timedelta(days=days)).strftime("%y%m%d")
    return f"{root}{exp}{cp}{int(round(strike * 1000)):08d}"


def pos(symbol, qty, side, entry, cur):
    sign = 1 if side == "long" else -1
    return {"symbol": symbol, "qty": str(qty * sign), "side": side, "avg_entry_price": str(entry),
            "current_price": str(cur), "market_value": str(cur * qty * 100 * sign),
            "unrealized_pl": str((cur - entry) * qty * 100 * sign), "asset_class": "us_option"}


SNAP = {"bid": 1.00, "ask": 1.20, "mid": 1.10, "delta": -0.3, "gamma": 0.01, "theta": -0.02, "vega": 0.05, "iv": 0.4}


def condor(days=3, strategy="iron_condor"):
    root = "COND"
    sp, lp, sc, lc = occ(root, days, "P", 310), occ(root, days, "P", 300), occ(root, days, "C", 345), occ(root, days, "C", 355)
    positions = [pos(sp, 2, "short", 6.95, 27.6), pos(lp, 2, "long", 4.70, 20.7),
                 pos(sc, 2, "short", 9.00, 0.95), pos(lc, 2, "long", 5.15, 0.06)]
    ts = (datetime.now() - timedelta(days=10)).isoformat()
    state = {}
    for s, e, sd in ((sp, 6.95, "short"), (lp, 4.70, "long"), (sc, 9.00, "short"), (lc, 5.15, "long")):
        state[s] = {"entry_price": e, "entry_delta": 0.2, "side": sd, "entry_date": "2026-10-20", "entry_timestamp": ts,
                    "strategy": strategy, "ticker": root, "setup": "high_iv_premium_sale", "qty": 2, "max_loss": 1140.0,
                    "initial_credit": e if sd == "short" else 0, "highest_value": e}
    return root, (sp, lp, sc, lc), positions, state


class Broker:
    """requests.get / requests.post stand-in: positions, open orders, clock, mleg verdict."""

    def __init__(self, positions, open_orders=None, is_open=True, mleg_status=422, mleg_text='{"code":40310000,"message":"position intent mismatch"}'):
        self.positions, self.open_orders, self.is_open = positions, open_orders or [], is_open
        self.mleg_status, self.mleg_text = mleg_status, mleg_text
        self.posts, self.gets = [], []

    def get(self, url, **kw):
        self.gets.append((url, kw.get("params")))
        if "/v2/positions" in url:
            return MagicMock(status_code=200, json=lambda: self.positions)
        if "/v2/orders" in url:
            return MagicMock(status_code=200, json=lambda: self.open_orders)
        if "/v2/clock" in url:
            return MagicMock(status_code=200, json=lambda: {"is_open": self.is_open})
        return MagicMock(status_code=404, json=lambda: {}, text="")

    def post(self, url, **kw):
        self.posts.append((url, kw.get("json")))
        return MagicMock(status_code=self.mleg_status, text=self.mleg_text, json=lambda: {"id": "mleg-1"})


class GroupCloseCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state_path = os.path.join(self.tmp.name, "voltrade_options_state.json")
        self.patches = [
            patch.object(om, "DATA_DIR", self.tmp.name),
            patch.object(om, "OPTIONS_STATE_PATH", self.state_path),
            patch.object(om, "_get_option_snapshot", return_value=dict(SNAP)),
            patch.object(om, "_attempt_roll", return_value={"rolled": False, "detail": "no roll in tests"}),
            patch.dict(sys.modules, {"vol_surface": MagicMock(get_surface_score=lambda t: {"vrp": {"vrp_20d": 0.0}}),
                                     "ml_model_v2": MagicMock(track_fill=lambda *a, **k: None)}),
        ]
        for p in self.patches:
            p.start()
        om._CLOCK_CACHE.update({"at": 0.0, "is_open": None})
        self.submits = []

        def fake_submit(occ_symbol, qty, side, limit_price):
            self.submits.append((occ_symbol, qty, side, limit_price))
            return {"status": "submitted", "order_id": f"leg-{len(self.submits)}"}
        self.submit_patch = patch.object(om, "_submit_close_order", side_effect=fake_submit)
        self.submit_patch.start()

    def tearDown(self):
        self.submit_patch.stop()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()
        # 2026-11-04: single-leg closes read this module-wide clock cache too
        # (test_options_exit_market_hours.py); a "closed" reading left by
        # test_market_closed_defers_without_an_order must not leak into them.
        om._CLOCK_CACHE.update({"at": 0.0, "is_open": None})

    def write_state(self, state):
        with open(self.state_path, "w") as f:
            json.dump(state, f)

    def read_state(self):
        with open(self.state_path) as f:
            return json.load(f)

    def run_manager(self, broker):
        om._CLOCK_CACHE.update({"at": 0.0, "is_open": None})
        with patch.object(om.requests, "get", side_effect=broker.get), patch.object(om.requests, "post", side_effect=broker.post):
            return om.manage_options_positions(100000, allow_rolls=False)


class TestGroupCloseIsPursued(GroupCloseCase):
    def test_regression_shorts_first_state_kept(self):
        root, (sp, lp, sc, lc), positions, state = condor(days=3)
        self.write_state(state)
        broker = Broker(positions)
        result = self.run_manager(broker)
        acts = [a for a in result["actions"] if a["action"] == "CLOSE_STRATEGY"]
        self.assertEqual(len(acts), 1, result["actions"])
        self.assertEqual(acts[0]["type"], "strategy_dte_critical")
        self.assertEqual(len(broker.posts), 1, "one mleg attempt first")
        self.assertEqual(broker.posts[0][1]["order_class"], "mleg")
        # (1) fallback: the two SHORT legs are bought back at the ASK; the long wings wait
        self.assertEqual(sorted((s, side, px) for s, _q, side, px in self.submits), sorted([(sp, "buy", SNAP["ask"]), (sc, "buy", SNAP["ask"])]))
        order = acts[0]["order"]
        self.assertIn("position intent mismatch", order["mleg_error"], "the broker's rejection is no longer swallowed")
        self.assertEqual(sorted(order["deferred"]), sorted([lp, lc]), "the wings are deferred until no short leg remains")
        self.assertEqual(order["status"], "partial")
        # (2) the legs keep their strategy and carry the close request — never popped
        saved = self.read_state()
        for s in (sp, lp, sc, lc):
            self.assertIn(s, saved, "state popped — the GOOGL orphan bug")
            self.assertEqual(saved[s]["strategy"], "iron_condor")
            self.assertTrue(saved[s]["close_requested_at"])
            self.assertEqual(saved[s]["close_rule"], "strategy_dte_critical")
            self.assertEqual(saved[s]["close_attempts"], 1)
        # no single-leg rule touched a structure leg
        self.assertFalse([a for a in result["actions"] if a["action"] == "CLOSE"], "single-leg CLOSE on a structure leg")

    def test_next_ticks_resubmit_when_nothing_is_working_and_wait_while_something_is(self):
        root, legs, positions, state = condor(days=3)
        old = (datetime.now() - timedelta(minutes=10)).isoformat()
        for s in legs:
            state[s].update({"close_requested_at": old, "close_rule": "strategy_dte_critical", "close_reason": "CRITICAL DTE", "close_attempts": 1,
                             "close_last_at": old, "close_last_result": {"status": "partial"}})
        self.write_state(state)
        # (3a) nothing working at the broker, last attempt 10 min ago → re-submit
        result = self.run_manager(Broker(positions))
        acts = [a for a in result["actions"] if a["action"] == "CLOSE_STRATEGY"]
        self.assertEqual(len(acts), 1, result["actions"])
        self.assertEqual(acts[0]["retry"], 2)
        self.assertEqual(acts[0]["type"], "strategy_dte_critical")
        self.assertTrue(self.submits, "the retry submitted leg closes")
        self.assertEqual(self.read_state()[legs[0]]["close_attempts"], 2)
        # (3b) a close order working at the broker → WAIT, no submission
        self.submits.clear()
        result = self.run_manager(Broker(positions, open_orders=[{"id": "o1", "symbol": legs[0], "status": "new"}]))
        waits = [a for a in result["actions"] if a["action"] == "WAIT"]
        self.assertEqual([w["type"] for w in waits], ["strategy_close_working"])
        self.assertFalse(self.submits)
        self.assertFalse([a for a in result["actions"] if a["action"] in ("CLOSE", "CLOSE_STRATEGY")])

    def test_a_just_submitted_close_settles_before_any_resubmit(self):
        root, legs, positions, state = condor(days=3)
        now = datetime.now().isoformat()
        for s in legs:
            state[s].update({"close_requested_at": now, "close_rule": "strategy_dte_critical", "close_reason": "CRITICAL DTE", "close_attempts": 1,
                             "close_last_at": now, "close_last_result": {"status": "submitted"}})
        self.write_state(state)
        result = self.run_manager(Broker(positions))
        self.assertEqual([a["type"] for a in result["actions"] if a["action"] == "WAIT"], ["strategy_close_settling"])
        self.assertFalse(self.submits)

    def test_market_closed_defers_without_an_order(self):
        root, legs, positions, state = condor(days=3)
        self.write_state(state)
        broker = Broker(positions, is_open=False)
        result = self.run_manager(broker)
        acts = [a for a in result["actions"] if a["action"] == "CLOSE_STRATEGY"]
        self.assertEqual(len(acts), 1)
        self.assertEqual(acts[0]["order"]["status"], "deferred")
        self.assertFalse(broker.posts, "no mleg POST outside regular hours")
        self.assertFalse(self.submits, "no per-leg order outside regular hours")
        saved = self.read_state()
        self.assertTrue(all(saved[s]["close_requested_at"] for s in legs))
        self.assertTrue(all(saved[s]["close_attempts"] == 0 for s in legs), "a deferred close is not an attempt")

    def test_wings_only_remainder_is_sold_at_the_bid(self):
        root, (sp, lp, sc, lc), positions, state = condor(days=3)
        old = (datetime.now() - timedelta(minutes=10)).isoformat()
        for s in (lp, lc):
            state[s].update({"close_requested_at": old, "close_rule": "strategy_dte_critical", "close_reason": "CRITICAL DTE", "close_attempts": 1,
                             "close_last_at": old, "close_last_result": {"status": "partial"}})
        self.write_state(state)
        wings = [p for p in positions if p["symbol"] in (lp, lc)]
        result = self.run_manager(Broker(wings))
        self.assertEqual(sorted((s, side, px) for s, _q, side, px in self.submits), sorted([(lp, "sell", SNAP["bid"]), (lc, "sell", SNAP["bid"])]))
        self.assertEqual([a["action"] for a in result["actions"] if a["action"] != "WARNING"], ["CLOSE_STRATEGY"])

    def test_a_leg_that_lost_its_label_is_adopted_by_its_siblings(self):
        root, (sp, lp, sc, lc), positions, state = condor(days=30)
        # the long call wing's state lost its strategy (what the old pop did); it is down 98 %
        state[lc] = {"entry_price": 5.15, "entry_delta": 0.2, "side": "long", "entry_date": "2026-10-20",
                     "entry_timestamp": (datetime.now() - timedelta(days=10)).isoformat(), "highest_value": 5.15, "qty": 2}
        self.write_state(state)
        result = self.run_manager(Broker(positions))
        adopted = [a for a in result["actions"] if a.get("type") == "group_adopted"]
        self.assertEqual(len(adopted), 1, result["actions"])
        self.assertEqual(adopted[0]["occ_symbol"], lc)
        self.assertFalse([a for a in result["actions"] if a["action"] == "CLOSE"], "the single-leg −50 % rule must not sell a spread wing")
        self.assertEqual(self.read_state()[lc]["strategy"], "iron_condor")
        self.assertFalse(self.submits, "a healthy 30-DTE condor is not closed")

    def test_mleg_success_path_unchanged(self):
        root, legs, positions, state = condor(days=3)
        self.write_state(state)
        broker = Broker(positions, mleg_status=200)
        result = self.run_manager(broker)
        order = [a for a in result["actions"] if a["action"] == "CLOSE_STRATEGY"][0]["order"]
        self.assertEqual(order["method"], "mleg")
        self.assertEqual(order["order_id"], "mleg-1")
        self.assertFalse(self.submits, "no per-leg fallback after an accepted mleg")
        payload = broker.posts[0][1]
        self.assertEqual((payload["order_class"], payload["type"], payload["time_in_force"], payload["qty"]), ("mleg", "market", "day", "2"))
        self.assertEqual(sorted(l["position_intent"] for l in payload["legs"]), ["buy_to_close", "buy_to_close", "sell_to_close", "sell_to_close"])
        self.assertTrue(all(self.read_state()[s]["close_requested_at"] for s in legs))


class TestSourcePins(unittest.TestCase):
    def test_every_group_exit_goes_through_the_pursued_close_and_nothing_pops_state(self):
        src = inspect.getsource(om._manage_strategy_group)
        self.assertNotIn("state.pop(", src, "a group exit must never drop its legs' state")
        self.assertNotIn("_close_strategy_mleg(legs, ticker)", src, "every exit submits through _request_group_close")
        self.assertEqual(src.count("_request_group_close(legs, ticker, state,"), 7)
        self.assertIn("return _pursue_group_close(group, state)", src)

    def test_mleg_transmission_is_the_same_post(self):
        src = inspect.getsource(om._close_strategy_mleg)
        self.assertIn('"order_class": "mleg"', src)
        self.assertIn('"type": "market",', src)
        self.assertIn('f"{ALPACA_BASE}/v2/orders"', src)
        self.assertIn("mleg_error", src)


if __name__ == "__main__":
    unittest.main()
