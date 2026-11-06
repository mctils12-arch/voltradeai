"""MID-FIRST MULTI-LEG CLOSES (2026-11-06, mleg_ladder.py): a strategy group's
close starts as a LIMIT mleg at the combination's net mid and is worked one
step per tick toward the natural, then today's market mleg.

Human refocus spec §5: "CBOE complex order book for multi-leg fills, work
limits near mid". Before: _close_strategy_mleg sent a MARKET mleg at once.

Pinned here:
  - the combination's quote from its legs (a debit when the close pays, a
    credit when it receives), Alpaca's signed limit (credit negative), the
    schedule (mid first, natural last, never backing off), when a step is
    due, held, or the natural has rested too long;
  - settings: on by default, env VOLTRADE_OPTIONS_MLEG_LADDER wins, auto is
    off on paper, a broken entry is off;
  - _close_strategy_mleg: the limit mleg at the net mid first (the plan in
    the result); a refused limit, no usable quote, the ladder off, or a
    ladder that ran out -> today's market mleg (then the per-leg fallback);
  - the pursued close: the working order is read by its id and re-priced
    (PATCH) step by step in regular hours after 09:45 only; at the natural
    for touch_seconds it is canceled and the next tick sends the market mleg;
    a refused replace cancels and falls back the same way;
  - a debit group's partial scale-out is laddered the same way.
Paper fills a limit only once it is marketable at the NBBO: the ladder's
value (price improvement on the complex order book) exists only live.
"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mleg_ladder  # noqa: E402
import options_manager as om  # noqa: E402
import system_config  # noqa: E402
from test_options_exit_ladder import LadderBroker, PAPER, LIVE  # noqa: E402
from test_options_exit_market_hours import _ManagerCase, occ, position  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402


def leg(side, bid, ask, ratio=1):
    return {"side": side, "bid": bid, "ask": ask, "ratio": ratio}


class TestComboQuote(unittest.TestCase):
    def test_closing_a_condor_pays_a_debit(self):
        # buy back the shorts (2.00/2.20, 1.50/1.70), sell the wings (0.40/0.50, 0.30/0.40)
        q = mleg_ladder.combo_quote([leg("buy", 2.00, 2.20), leg("buy", 1.50, 1.70), leg("sell", 0.40, 0.50), leg("sell", 0.30, 0.40)])
        self.assertEqual(q, {"side": "buy", "bid": 2.6, "ask": 3.2, "mid": 2.9})   # far 3.5-0.9, natural 3.9-0.7

    def test_closing_a_debit_spread_receives_a_credit(self):
        q = mleg_ladder.combo_quote([leg("sell", 5.00, 5.20), leg("buy", 2.00, 2.10)])
        self.assertEqual(q, {"side": "sell", "bid": 2.9, "ask": 3.2, "mid": 3.05})  # natural 5.00-2.10, far 5.20-2.00

    def test_unusable_quotes(self):
        self.assertIsNone(mleg_ladder.combo_quote([]))
        self.assertIsNone(mleg_ladder.combo_quote([leg("buy", 1.0, 0.0)]), "no ask")
        self.assertIsNone(mleg_ladder.combo_quote([leg("buy", 1.2, 1.0)]), "crossed")
        self.assertIsNone(mleg_ladder.combo_quote([leg("buy", 1.0, 1.2), leg("sell", 1.0, 1.2)]), "a mid at zero: no side to work")
        self.assertIsNone(mleg_ladder.combo_quote([leg("buy", 0.05, 0.30), leg("sell", 0.10, 0.20)]),
                          "the far side is below zero")
        q = mleg_ladder.combo_quote([leg("buy", 1.00, 1.10), leg("sell", 0.0, 0.05)])
        self.assertEqual(q["side"], "buy", "a worthless wing (0 bid) still prices")

    def test_signed_limit(self):
        self.assertEqual(mleg_ladder.signed_limit("buy", 2.9), 2.9)
        self.assertEqual(mleg_ladder.signed_limit("sell", 3.05), -3.05)

    def test_plan_is_mid_first_and_natural_last(self):
        p = mleg_ladder.plan([leg("buy", 2.00, 2.20), leg("buy", 1.50, 1.70), leg("sell", 0.40, 0.50), leg("sell", 0.30, 0.40)], 3)
        self.assertEqual(p["prices"], [2.9, 3.05, 3.2])
        p = mleg_ladder.plan([leg("sell", 5.00, 5.20), leg("buy", 2.00, 2.10)], 3)
        self.assertEqual(p["prices"], [3.05, 2.98, 2.9])
        self.assertIn("skip", mleg_ladder.plan([leg("buy", 1.00, 1.01)], 3), "a one-tick combination has no inside price")


class TestNextStep(unittest.TestCase):
    CFG = {"step_seconds": 60, "touch_seconds": 120}
    LEGS = [leg("buy", 2.00, 2.20), leg("buy", 1.50, 1.70), leg("sell", 0.40, 0.50), leg("sell", 0.30, 0.40)]

    def lad(self, step=1, price=2.9, step_at=1000.0):
        return {"side": "buy", "steps": 3, "step": step, "price": price, "step_at": step_at}

    def test_waits_for_step_seconds_then_steps(self):
        self.assertEqual(mleg_ladder.next_step(self.lad(), self.LEGS, 1059.0, self.CFG), {"do": "wait"})
        self.assertEqual(mleg_ladder.next_step(self.lad(), self.LEGS, 1060.0, self.CFG),
                         {"do": "step", "step": 2, "price": 3.05, "bid": 2.6, "ask": 3.2})

    def test_never_backs_off_and_holds_an_unchanged_price(self):
        d = mleg_ladder.next_step(self.lad(price=3.10), self.LEGS, 2000.0, self.CFG)
        self.assertEqual((d["do"], d["price"]), ("hold", 3.10))

    def test_the_natural_rests_touch_seconds_then_cancels(self):
        self.assertEqual(mleg_ladder.next_step(self.lad(step=3, price=3.2), self.LEGS, 1119.0, self.CFG), {"do": "wait"})
        d = mleg_ladder.next_step(self.lad(step=3, price=3.2), self.LEGS, 1120.0, self.CFG)
        self.assertEqual(d["do"], "cancel")
        self.assertIn("rested 120s unfilled", d["why"])

    def test_no_usable_quote_or_a_flipped_side_waits(self):
        self.assertEqual(mleg_ladder.next_step(self.lad(), [leg("buy", 0, 0)], 2000.0, self.CFG), {"do": "wait"})
        self.assertEqual(mleg_ladder.next_step(self.lad(), [leg("sell", 5.00, 5.20), leg("buy", 2.00, 2.10)], 2000.0, self.CFG),
                         {"do": "wait"})


class TestSettings(unittest.TestCase):
    def test_default_on_env_wins_auto_off_on_paper(self):
        with patch.dict(os.environ, {}):
            os.environ.pop(mleg_ladder.ENV_VAR, None)
            self.assertEqual(system_config.BASE_CONFIG["OPTIONS_MLEG_LADDER"],
                             {"mode": "on", "steps": 3, "step_seconds": 60, "touch_seconds": 120})
            self.assertEqual(mleg_ladder.settings(PAPER), {"on": True, "steps": 3, "step_seconds": 60.0, "touch_seconds": 120.0})
        with patch.dict(os.environ, {mleg_ladder.ENV_VAR: "off"}):
            self.assertFalse(mleg_ladder.settings(PAPER)["on"])
        with patch.dict(os.environ, {mleg_ladder.ENV_VAR: "auto"}):
            self.assertFalse(mleg_ladder.settings(PAPER)["on"])
            self.assertTrue(mleg_ladder.settings(LIVE)["on"])
        with patch.dict(os.environ, {}), patch.dict(system_config.BASE_CONFIG, {"OPTIONS_MLEG_LADDER": None}):
            os.environ.pop(mleg_ladder.ENV_VAR, None)
            self.assertFalse(mleg_ladder.settings(PAPER)["on"])


# ── through options_manager ─────────────────────────────────────────────────

class MlegBroker(LadderBroker):
    """LadderBroker that also books mleg orders (or refuses limit / market ones)."""

    def __init__(self, positions, is_open=True):
        super().__init__(positions, is_open=is_open)
        self.refuse_limit_mleg = False
        self.refuse_market_mleg = False

    def post(self, url, **kw):
        payload = kw.get("json") or {}
        if payload.get("order_class") == "mleg":
            self.posts.append((url, payload))
            if (payload.get("type") == "limit" and self.refuse_limit_mleg) or (payload.get("type") == "market" and self.refuse_market_mleg):
                return MagicMock(status_code=422, text='{"message":"refused in tests"}', json=lambda: {})
            oid = f"mleg-{len(self.orders) + 1}"
            self.orders[oid] = {"id": oid, "symbol": "", "order_class": "mleg", "qty": payload.get("qty"),
                                "type": payload.get("type"), "limit_price": payload.get("limit_price"),
                                "status": "new", "filled_qty": "0", "replaced_by": None}
            body = dict(self.orders[oid])
            return MagicMock(status_code=200, json=lambda: body, text="")
        return super().post(url, **kw)

    def mlegs(self):
        return [p for _u, p in self.posts if p.get("order_class") == "mleg"]


def condor(days=40, strategy="iron_condor", held_days=10):
    root = "MLGC"
    sp, lp, sc, lc = occ(root, days, "P", 90), occ(root, days, "P", 85), occ(root, days, "C", 110), occ(root, days, "C", 115)
    positions = [position(sp, 2, "short", 3.0, 2.1), position(lp, 2, "long", 1.0, 0.45),
                 position(sc, 2, "short", 2.5, 1.6), position(lc, 2, "long", 0.8, 0.35)]
    ts = (datetime.now() - timedelta(days=held_days)).isoformat()
    state = {}
    for s, e, sd in ((sp, 3.0, "short"), (lp, 1.0, "long"), (sc, 2.5, "short"), (lc, 0.8, "long")):
        state[s] = {"entry_price": e, "side": sd, "entry_date": ts[:10], "entry_timestamp": ts, "strategy": strategy,
                    "ticker": root, "setup": "", "qty": 2, "max_loss": 0, "initial_credit": e if sd == "short" else 0,
                    "highest_value": e}
    snaps = {sp: {"bid": 2.00, "ask": 2.20, "mid": 2.10}, lp: {"bid": 0.40, "ask": 0.50, "mid": 0.45},
             sc: {"bid": 1.50, "ask": 1.70, "mid": 1.60}, lc: {"bid": 0.30, "ask": 0.40, "mid": 0.35}}
    return (sp, lp, sc, lc), positions, state, snaps


class _MlegCase(_ManagerCase):
    def setUp(self):
        super().setUp()
        p = patch.dict(sys.modules, {"vol_surface": MagicMock(get_surface_score=lambda t: {"vrp": {"vrp_20d": 0.0}})})
        p.start()
        self.addCleanup(p.stop)
        e = patch.dict(os.environ, {})
        e.start()
        self.addCleanup(e.stop)
        os.environ.pop(mleg_ladder.ENV_VAR, None)

    def calls(self, broker):
        return (patch.object(om.requests, "get", side_effect=broker.get), patch.object(om.requests, "post", side_effect=broker.post),
                patch.object(om.requests, "patch", side_effect=broker.patch), patch.object(om.requests, "delete", side_effect=broker.delete))

    def tick(self, broker):
        self._cold_clock()
        g, p, pa, d = self.calls(broker)
        with g, p, pa, d:
            return om.manage_options_positions(100000)

    def close_now(self, legs, ticker, broker, **kw):
        self._cold_clock()
        g, p, pa, d = self.calls(broker)
        with g, p, pa, d:
            return om._close_strategy_mleg(legs, ticker, **kw)

    def setup_condor(self, days=40):
        legs, positions, state, snaps = condor(days=days)
        self.snaps.update(snaps)
        with open(self.state_path, "w") as f:
            json.dump(state, f)
        return legs, positions

    def ladder_state(self, occ_symbol):
        return self.read_state()[occ_symbol]["close_last_result"]

    def age_ladder(self, occs, seconds):
        st = self.read_state()
        for o in occs:
            st[o]["close_last_result"]["ladder"]["step_at"] -= seconds
        with open(self.state_path, "w") as f:
            json.dump(st, f)


class TestCloseStartsAtTheMid(_MlegCase):
    def test_the_first_order_is_a_limit_mleg_at_the_net_mid(self):
        legs, positions = self.setup_condor()
        broker = MlegBroker(positions)
        res = self.close_now(positions, "MLGC", broker)
        (first,) = broker.mlegs()
        self.assertEqual((first["type"], first["limit_price"], first["qty"], first["time_in_force"]), ("limit", "2.9", "2", "day"))
        self.assertEqual(sorted(l["position_intent"] for l in first["legs"]), ["buy_to_close", "buy_to_close", "sell_to_close", "sell_to_close"])
        self.assertEqual((res["status"], res["method"]), ("submitted", "mleg_limit"))
        self.assertEqual(res["ladder"]["plan"], [2.9, 3.05, 3.2])
        self.assertEqual((res["ladder"]["side"], res["ladder"]["natural"]), ("buy", 3.2))
        self.assertIn("limit at the net mid $2.90 debit (step 1/3; natural $3.20)", res["detail"])

    def test_closing_for_a_credit_sends_a_negative_limit(self):
        lo, sh = occ("MLGD", 40, "C", 100), occ("MLGD", 40, "C", 110)
        positions = [position(lo, 1, "long", 4.0, 5.1), position(sh, 1, "short", 1.5, 2.05)]
        self.snaps.update({lo: {"bid": 5.00, "ask": 5.20}, sh: {"bid": 2.00, "ask": 2.10}})
        broker = MlegBroker(positions)
        res = self.close_now(positions, "MLGD", broker)
        self.assertEqual(broker.mlegs()[0]["limit_price"], "-3.05")
        self.assertEqual(res["ladder"]["side"], "sell")

    def test_a_refused_limit_falls_back_to_todays_market_mleg_in_the_same_call(self):
        legs, positions = self.setup_condor()
        broker = MlegBroker(positions)
        broker.refuse_limit_mleg = True
        res = self.close_now(positions, "MLGC", broker)
        self.assertEqual([m["type"] for m in broker.mlegs()], ["limit", "market"])
        self.assertEqual(res["method"], "mleg")
        self.assertIn("mid limit refused (HTTP 422", res["detail"])

    def test_off_no_quote_or_a_spent_ladder_send_todays_market_mleg(self):
        legs, positions = self.setup_condor()
        with patch.dict(os.environ, {mleg_ladder.ENV_VAR: "off"}):
            broker = MlegBroker(positions)
            self.close_now(positions, "MLGC", broker)
            self.assertEqual([m["type"] for m in broker.mlegs()], ["market"])
        broker = MlegBroker(positions)
        res = self.close_now(positions, "MLGC", broker, ladder_ok=False)
        self.assertEqual([m["type"] for m in broker.mlegs()], ["market"])
        self.assertIn("the mid-first ladder ran out", res["detail"])
        for o in legs:
            self.snaps[o] = {"bid": 0.0, "ask": 0.0}
        broker = MlegBroker(positions)
        res = self.close_now(positions, "MLGC", broker)
        self.assertEqual([m["type"] for m in broker.mlegs()], ["market"])
        self.assertIn("no mid limit", res["detail"])

    def test_both_refused_still_reach_the_per_leg_fallback(self):
        legs, positions = self.setup_condor()
        broker = MlegBroker(positions)
        broker.refuse_limit_mleg = broker.refuse_market_mleg = True
        res = self.close_now(positions, "MLGC", broker)
        self.assertEqual(res["method"], "individual_legs")
        self.assertEqual(sorted(s[0] for s in self.submits), sorted([legs[0], legs[2]]), "the shorts first, as before")


class TestThePursuitWorksTheLadder(_MlegCase):
    def test_mid_then_steps_then_the_natural_then_the_market_mleg(self):
        legs, positions = self.setup_condor(days=15)          # 21-DTE rule closes the condor
        broker = MlegBroker(positions)
        r = self.tick(broker)
        self.assertEqual([m["limit_price"] for m in broker.mlegs()], ["2.9"])
        self.assertEqual([a["type"] for a in r["actions"] if a["action"] == "CLOSE_STRATEGY"], ["strategy_dte_exit"])
        self.assertEqual(self.ladder_state(legs[0])["ladder"]["step"], 1)
        # too soon: the order works, nothing changes
        r = self.tick(broker)
        self.assertEqual(broker.patches, [])
        self.assertEqual([a["type"] for a in r["actions"] if a["action"] == "WAIT"], ["strategy_close_working"])
        # step 2
        self.age_ladder(legs, 61)
        r = self.tick(broker)
        self.assertEqual([p["limit_price"] for _o, p in broker.patches], ["3.05"])
        (lad,) = [a for a in r["actions"] if a["action"] == "LADDER"]
        self.assertEqual((lad["type"], lad["order"]["mleg_ladder"]), ("mleg_ladder_step", {"step": 2, "of": 3, "price": 3.05}))
        self.assertNotIn("ladder", lad["order"], "the audit's single-leg 'exit ladder step' suffix is not this row's")
        new_id = self.ladder_state(legs[0])["order_id"]
        self.assertEqual(broker.orders[new_id]["limit_price"], "3.05")
        # step 3: the natural
        self.age_ladder(legs, 61)
        self.tick(broker)
        self.assertEqual([p["limit_price"] for _o, p in broker.patches], ["3.05", "3.2"])
        # resting at the natural: canceled after touch_seconds
        self.age_ladder(legs, 60)
        self.tick(broker)
        self.assertEqual(broker.deletes, [])
        self.age_ladder(legs, 61)
        r = self.tick(broker)
        self.assertEqual(len(broker.deletes), 1)
        self.assertEqual(self.ladder_state(legs[0])["ladder"]["done"], "touch_timeout")
        self.assertIn("today's market mleg goes out next tick", [a for a in r["actions"] if a["action"] == "LADDER"][0]["reason"])
        # next tick (the close settled long ago): today's market mleg
        st = self.read_state()
        for o in legs:
            st[o]["close_last_at"] = (datetime.now() - timedelta(seconds=om.CLOSE_RETRY_SECONDS + 5)).isoformat()
        with open(self.state_path, "w") as f:
            json.dump(st, f)
        r = self.tick(broker)
        self.assertEqual([m["type"] for m in broker.mlegs()], ["limit", "market"])
        self.assertEqual([a["order"]["method"] for a in r["actions"] if a["action"] == "CLOSE_STRATEGY"], ["mleg"])

    def test_no_step_outside_regular_hours_or_in_the_opening_window(self):
        legs, positions = self.setup_condor(days=15)
        broker = MlegBroker(positions)
        self.tick(broker)
        self.age_ladder(legs, 300)
        broker.is_open = False
        self.tick(broker)
        with patch.object(om, "_in_opening_window", return_value=True):
            broker.is_open = True
            self.tick(broker)
        self.assertEqual(broker.patches, [])

    def test_a_refused_replace_cancels_and_falls_back(self):
        legs, positions = self.setup_condor(days=15)
        broker = MlegBroker(positions)
        self.tick(broker)
        broker.refuse_patch = True
        self.age_ladder(legs, 61)
        r = self.tick(broker)
        self.assertEqual(len(broker.deletes), 1)
        self.assertEqual(self.ladder_state(legs[0])["ladder"]["done"], "replace_refused")
        self.assertIn("replace refused", [a for a in r["actions"] if a["action"] == "LADDER"][0]["reason"])

    def test_a_filled_order_ends_the_close(self):
        legs, positions = self.setup_condor(days=15)
        broker = MlegBroker(positions)
        self.tick(broker)
        oid = self.ladder_state(legs[0])["order_id"]
        broker.orders[oid]["status"] = "filled"
        broker.positions = []
        self.tick(broker)
        self.assertEqual(self.read_state(), {})


class TestPartialScaleOutIsLaddered(_MlegCase):
    def test_a_debit_groups_half_goes_out_at_the_mid_and_steps(self):
        lo, sh = occ("MLGS", 40, "C", 100), occ("MLGS", 40, "C", 110)
        held = (datetime.now() - timedelta(days=10)).isoformat()
        positions = [position(lo, 2, "long", 5.00, 11.10), position(sh, 2, "short", 2.00, 5.00)]
        state = {o: {"entry_price": e, "side": sd, "qty": 2, "strategy": "bull_call_spread", "ticker": "MLGS", "setup": "",
                     "entry_date": held[:10], "entry_timestamp": held, "initial_credit": 0, "highest_value": e, "max_loss": 0}
                 for o, e, sd in ((lo, 5.00, "long"), (sh, 2.00, "short"))}
        with open(self.state_path, "w") as f:
            json.dump(state, f)
        self.snaps.update({lo: {"bid": 11.00, "ask": 11.20}, sh: {"bid": 4.90, "ask": 5.10}})   # +103% on a 3.00 net debit
        broker = MlegBroker(positions)
        r = self.tick(broker)
        (m,) = broker.mlegs()
        self.assertEqual((m["type"], m["qty"], m["limit_price"]), ("limit", "1", "-6.1"), "half the units, a credit at the net mid")
        self.assertEqual([a["type"] for a in r["actions"] if a["action"] == "CLOSE_STRATEGY"], ["strategy_scale_out"])
        so = self.read_state()[lo]["group_profit_taking"]["scale_out"]
        self.assertEqual((so["status"], so["ladder"]["step"]), ("submitted", 1))
        st = self.read_state()
        for o in (lo, sh):
            st[o]["group_profit_taking"]["scale_out"]["ladder"]["step_at"] -= 61
        with open(self.state_path, "w") as f:
            json.dump(st, f)
        r = self.tick(broker)
        self.assertEqual(len(broker.patches), 1)
        self.assertTrue([a for a in r["actions"] if a["action"] == "LADDER" and a["reason"].startswith("SCALE-OUT mleg ladder step 2/3")])
        self.assertEqual(len(broker.mlegs()), 1, "no second order while the first works")


# ── entries: a debit vertical starts at the net mid and is walked ───────────

def vertical(long_q=(5.00, 5.20), short_q=(2.00, 2.10), root="MLGE"):
    lo, sh = occ(root, 40, "C", 100), occ(root, 40, "C", 110)
    natural = round(long_q[1] - short_q[0], 2)
    return {"strategy": "bull_call_spread", "long_leg": lo, "short_leg": sh, "qty": 2, "net_debit": natural,
            "limit_price": natural, "max_cost": natural * 200, "max_loss": natural * 200, "error": None,
            "leg_quotes": [{"symbol": lo, "side": "buy", "bid": long_q[0], "ask": long_q[1]},
                           {"symbol": sh, "side": "sell", "bid": short_q[0], "ask": short_q[1]}]}


class TestMidFirstEntry(unittest.TestCase):
    def setUp(self):
        e = patch.dict(os.environ, {})
        e.start()
        self.addCleanup(e.stop)
        os.environ.pop(mleg_ladder.ENV_VAR, None)

    def test_a_debit_vertical_goes_out_at_the_net_mid_with_its_walk_planned(self):
        import options_execution as oe
        c = oe._mid_first_entry(vertical())
        self.assertEqual((c["net_debit"], c["limit_price"], c["net_debit_natural"]), (3.05, 3.05, 3.2))
        self.assertEqual(c["entry_ladder"]["plan"], [3.05, 3.1, 3.2])
        self.assertEqual(c["entry_ladder"]["legs"], [{"symbol": c["long_leg"], "order_side": "buy"},
                                                     {"symbol": c["short_leg"], "order_side": "sell"}])
        self.assertEqual((c["max_cost"], c["max_loss"]), (640.0, 640.0), "sizing and max loss stay on the natural")

    def test_off_a_credit_structure_or_no_quotes_are_left_as_they_were(self):
        import options_execution as oe
        with patch.dict(os.environ, {mleg_ladder.ENV_VAR: "off"}):
            self.assertEqual(oe._mid_first_entry(vertical())["limit_price"], 3.2)
        condor = {"strategy": "iron_condor", "net_credit": 1.2, "limit_price": 1.2, "error": None}
        self.assertEqual(oe._mid_first_entry(dict(condor)), condor)
        self.assertNotIn("entry_ladder", oe._mid_first_entry(vertical(long_q=(5.00, 5.01), short_q=(4.99, 5.00))))

    def test_evaluate_and_execute_sends_the_mid_and_registers_the_walk(self):
        import options_execution as oe
        sent = []
        with patch.object(oe, "should_use_options", return_value={"use_options": True, "strategy": "bull_call_spread",
                                                                   "edge_pct": 1.0, "reason": "test"}), \
                patch.object(oe, "select_contract", return_value=vertical()), \
                patch.object(oe, "submit_options_order", side_effect=lambda c: sent.append(dict(c)) or {"status": "submitted", "order_id": "e-1"}), \
                patch("entry_shadow.is_shadowed", return_value=False), \
                patch("options_manager.register_entry_ladder") as reg:
            out = oe.evaluate_and_execute({"ticker": "MLGE", "price": 100.0}, 100000, [])
        self.assertEqual(sent[0]["limit_price"], 3.05)
        reg.assert_called_once()
        self.assertEqual(reg.call_args.args[0], "e-1")
        self.assertEqual(reg.call_args.args[1]["plan"], [3.05, 3.1, 3.2])
        self.assertEqual(out["instrument"], "options")


class TestEntryWalk(_MlegCase):
    def _register(self, broker):
        c = vertical()
        import options_execution as oe
        c = oe._mid_first_entry(c)
        self.snaps.update({c["long_leg"]: {"bid": 5.00, "ask": 5.20}, c["short_leg"]: {"bid": 2.00, "ask": 2.10}})
        oid = "mleg-entry"
        broker.orders[oid] = {"id": oid, "symbol": "", "order_class": "mleg", "qty": "2", "type": "limit",
                              "limit_price": "3.05", "status": "new", "filled_qty": "0", "replaced_by": None}
        om.register_entry_ladder(oid, c["entry_ladder"])
        return c

    def _age(self, seconds):
        path = os.path.join(om.DATA_DIR, om.ENTRY_LADDER_FILE)
        with open(path) as f:
            data = json.load(f)
        for v in data.values():
            v["step_at"] -= seconds
        with open(path, "w") as f:
            json.dump(data, f)

    def entries(self):
        path = os.path.join(om.DATA_DIR, om.ENTRY_LADDER_FILE)
        with open(path) as f:
            return json.load(f)

    def test_the_entry_walks_to_the_natural_rests_there_and_finishes_on_the_fill(self):
        broker = MlegBroker([])
        self._register(broker)
        r = self.tick(broker)
        self.assertEqual((broker.patches, r["actions"]), ([], []), "too soon to step")
        self._age(61)
        r = self.tick(broker)
        self.assertEqual([p["limit_price"] for _o, p in broker.patches], ["3.1"])
        (act,) = r["actions"]
        self.assertEqual((act["action"], act["type"]), ("LADDER", "mleg_ladder_entry_step"))
        self.assertTrue(act["reason"].startswith("ENTRY mleg ladder step 2/3: limit $3.10 debit"))
        self._age(61)
        self.tick(broker)
        self.assertEqual([p["limit_price"] for _o, p in broker.patches], ["3.1", "3.2"])
        self._age(3600)
        self.tick(broker)
        self.assertEqual(broker.deletes, [], "an entry rests at the natural like today's order; never canceled here")
        (oid,) = [v["order_id"] for v in self.entries().values()]
        broker.orders[oid].update(status="filled", filled_avg_price="3.2")
        r = self.tick(broker)
        self.assertEqual(self.entries(), {})
        (end,) = r["actions"]
        self.assertIn("ended (filled) at step 3/3, filled at $3.20", end["reason"])
        with open(os.path.join(om.DATA_DIR, om.ENTRY_LADDER_LOG_FILE)) as f:
            (line,) = [json.loads(l) for l in f if l.strip()]
        self.assertEqual((line["outcome"], line["step"], line["fill_price"]), ("filled", 3, "3.2"))

    def test_no_step_while_closed_and_an_old_entry_is_dropped(self):
        broker = MlegBroker([], is_open=False)
        self._register(broker)
        self._age(120)
        self.tick(broker)
        self.assertEqual(broker.patches, [])
        path = os.path.join(om.DATA_DIR, om.ENTRY_LADDER_FILE)
        with open(path) as f:
            data = json.load(f)
        for v in data.values():
            v["started_at"] -= 8 * 3600
        with open(path, "w") as f:
            json.dump(data, f)
        broker.is_open = True
        self.tick(broker)
        self.assertEqual((self.entries(), broker.patches), ({}, []))


if __name__ == "__main__":
    unittest.main()
