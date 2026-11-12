"""RESTING MULTI-LEG ENTRIES (2026-11-06): a multi-leg entry's leg state survives
the stale-state prune until its opening order resolves.

Before: manage_options_positions deleted every state key that is not a held
position. A multi-leg entry is registered at SUBMISSION, so the first pass on
which its order had not filled dropped the legs' strategy, setup and max_loss.
The mid-first debit entries rest by design, and since the mleg limit-sign fix a
credit condor rests whenever the market moved against it. A leg that filled
after that pass came back as an "unknown" single leg: no group profit target,
max-loss exit or 21-DTE close.

Pinned here (mocked broker: positions, clock, orders by id; state in a temp dir):
  - a resting condor's legs survive a pass with nothing held, and a pass with
    other positions held;
  - a late fill is managed as the group an immediate fill would be: the same
    registered fields, the entry clock moved to the fill (MIN_HOLD counts from
    there), the marker cleared, then the 21-DTE close, the registered max-loss
    exit and the profit target all fire on the group;
  - expired (the close) / canceled / rejected / not found: the legs' state is
    dropped; a filled order whose legs never appear is dropped after the grace;
  - an unreadable order keeps the legs until the age limit;
  - the mid-first debit entries: evaluate_and_execute hands the order id over,
    and the ladder's replacements are followed to the working order;
  - the scanner-direct script hands the order id over, and registers a 'new' order;
  - single-leg entries are pruned exactly as before.
"""
import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mleg_ladder  # noqa: E402
import options_execution as oe  # noqa: E402
import options_manager as om  # noqa: E402
from test_options_exit_market_hours import Broker, _ManagerCase, occ, position  # noqa: E402
from test_scanner_legs_registered import CREDIT_PUT_SPREAD, PUT_LONG, PUT_SHORT, _ScannerCase  # noqa: E402

ROOT = "RSTG"


def condor(dte, max_loss=1041.0):
    legs = {"short_call": occ(ROOT, dte, "C", 120.0), "long_call": occ(ROOT, dte, "C", 125.0),
            "short_put": occ(ROOT, dte, "P", 90.0), "long_put": occ(ROOT, dte, "P", 85.0)}
    return legs, {"strategy": "iron_condor", **legs, "qty": 1, "net_credit": 2.5, "max_loss": max_loss, "error": None}


def held_legs(legs, marks):
    """Alpaca positions for the condor: marks = {key: (entry, current)}."""
    side = {"short_call": "short", "long_call": "long", "short_put": "short", "long_put": "long"}
    return [position(legs[k], 1, side[k], *marks[k]) for k in legs]


FLAT = {"short_call": (2.00, 2.00), "long_call": (0.80, 0.80), "short_put": (2.20, 2.20), "long_put": (0.90, 0.90)}
PROFIT = {"short_call": (2.00, 0.80), "long_call": (0.80, 0.30), "short_put": (2.20, 0.90), "long_put": (0.90, 0.40)}  # +150 of 250
LOSS = {"short_call": (2.00, 9.00), "long_call": (0.80, 3.00), "short_put": (2.20, 0.40), "long_put": (0.90, 0.10)}    # -380


def utc_iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class OrderBroker(Broker):
    """+ GET /v2/orders/{id} from self.orders ("error" raises; a missing id is a 404)."""

    def __init__(self, positions, orders=None, **kw):
        super().__init__(positions, **kw)
        self.orders = dict(orders or {})

    def get(self, url, **kw):
        if "/v2/orders/" in url:
            oid = url.rsplit("/", 1)[-1]
            o = self.orders.get(oid)
            if o == "error":
                raise ConnectionError("order read failed")
            if o is None:
                return MagicMock(status_code=404, json=lambda: {}, text="")
            return MagicMock(status_code=200, json=lambda o=o: {"id": oid, **o})
        return super().get(url, **kw)


class _RestingCase(_ManagerCase):
    def setUp(self):
        super().setUp()
        self.closes = []
        p = patch.object(om, "_close_strategy_mleg", side_effect=lambda legs, t, **k: self.closes.append(
            (t, sorted(l["symbol"] for l in legs))) or {"status": "submitted", "order_id": "close-1", "method": "mleg"})
        p.start()
        self.addCleanup(p.stop)

    def register(self, contract, order_id, strategy="iron_condor"):
        return oe.register_contract_entry(dict(contract), strategy, ROOT, "high_iv_premium_sale", order_id=order_id)

    def age(self, syms, minutes, key="entry_timestamp"):
        st = self.read_state()
        for s in syms:
            st[s][key] = (datetime.now() - timedelta(minutes=minutes)).isoformat()
        with open(self.state_path, "w") as f:
            json.dump(st, f)

    def close_types(self, result):
        return [a.get("type") for a in result["actions"] if a.get("action") == "CLOSE_STRATEGY"]


class TestTheLegsSurviveWhileTheOrderRests(_RestingCase):
    def test_nothing_held_and_the_order_working(self):
        legs, c = condor(40)
        self.register(c, "o-1")
        for status in ("new", "accepted", "pending_new", "partially_filled", "pending_replace"):
            with self.subTest(status=status):
                self.run_manager(OrderBroker([], {"o-1": {"status": status}}))
                st = self.read_state()
                self.assertEqual(sorted(st), sorted(legs.values()))
                self.assertTrue(all(st[s]["strategy"] == "iron_condor" and st[s][om.OPENING_ORDER_KEY] == "o-1"
                                    for s in legs.values()))

    def test_other_positions_held_and_the_order_working(self):
        legs, c = condor(40)
        self.register(c, "o-1")
        other = occ("OTHR", 40, "P", 50.0)
        om.register_options_entry(other, 1.0, "sell", "sell_cash_secured_put", ticker="OTHR")
        self.run_manager(OrderBroker([position(other, 1, "short", 1.0, 1.0)], {"o-1": {"status": "new"}}))
        self.assertEqual(sorted(self.read_state()), sorted(list(legs.values()) + [other]))


class TestALateFillIsTheGroupAnImmediateFillWouldBe(_RestingCase):
    def test_registered_fields_kept_entry_clock_moved_to_the_fill_marker_cleared(self):
        legs, c = condor(10)
        syms = list(legs.values())
        self.register(c, "o-1")
        registered = {s: dict(v) for s, v in self.read_state().items()}
        self.age(syms, 180)                                  # it rested three hours
        self.run_manager(OrderBroker([], {"o-1": {"status": "new"}}))
        filled_at = datetime.now() - timedelta(seconds=60)
        r = self.run_manager(OrderBroker(held_legs(legs, FLAT), {"o-1": {"status": "filled", "filled_at": utc_iso(filled_at)}}))
        st = self.read_state()
        same = ("strategy", "setup", "ticker", "side", "qty", "max_loss", "entry_price", "entry_delta", "initial_credit")
        for s in syms:
            with self.subTest(leg=s):
                self.assertEqual({k: st[s][k] for k in same}, {k: registered[s][k] for k in same})
                self.assertNotIn(om.OPENING_ORDER_KEY, st[s])
                moved = datetime.fromisoformat(st[s]["entry_timestamp"])
                self.assertLess(abs((moved - filled_at).total_seconds()), 2, "the entry clock is the fill time")
        self.assertEqual(self.close_types(r), [], "MIN_HOLD counts from the fill, not from the 3-hour-old submission")
        self.age(syms, om.MIN_HOLD_MINUTES + 1)
        r = self.run_manager(OrderBroker(held_legs(legs, FLAT), {}))
        self.assertEqual(self.close_types(r), ["strategy_dte_exit"])
        self.assertEqual(self.closes, [(ROOT, sorted(syms))], "the group's 21-DTE close, all four legs in one order")

    def test_the_registered_max_loss_exit_and_the_profit_target_fire_after_a_late_fill(self):
        for marks, max_loss, want in ((LOSS, 600.0, "strategy_ic_max_loss_exit"), (PROFIT, 1041.0, "strategy_profit_target")):
            with self.subTest(rule=want):
                self.closes.clear()
                with open(self.state_path, "w") as f:
                    json.dump({}, f)
                legs, c = condor(40, max_loss=max_loss)
                syms = list(legs.values())
                self.register(c, "o-9")
                self.run_manager(OrderBroker([], {"o-9": {"status": "accepted"}}))
                self.run_manager(OrderBroker(held_legs(legs, marks),
                                             {"o-9": {"status": "filled", "filled_at": utc_iso(datetime.now())}}))
                self.age(syms, om.MIN_HOLD_MINUTES + 1)
                r = self.run_manager(OrderBroker(held_legs(legs, marks), {}))
                acts = [a for a in r["actions"] if a.get("action") == "CLOSE_STRATEGY"]
                self.assertEqual([a["type"] for a in acts], [want])
                if want == "strategy_ic_max_loss_exit":
                    self.assertEqual(acts[0]["max_loss"], 600.0, "the max_loss registered at entry, not one rebuilt from the legs")

    def test_the_legs_reach_the_group_manager_after_resting(self):
        """On the old prune this fails: pass 1 deleted the legs' state, so pass 2 saw four unknown single legs."""
        legs, c = condor(10)
        syms = list(legs.values())
        self.register(c, "o-1")
        self.age(syms, om.MIN_HOLD_MINUTES + 1)
        groups = []
        real = om._manage_strategy_group
        with patch.object(om, "_manage_strategy_group", side_effect=lambda g, *a, **k: groups.append(g["strategy"]) or real(g, *a, **k)):
            self.run_manager(OrderBroker([], {"o-1": {"status": "new"}}))
            self.run_manager(OrderBroker(held_legs(legs, FLAT), {"o-1": {"status": "filled"}}))
        self.assertEqual(groups, ["iron_condor"])


class TestAnOrderThatEndsUnfilledDropsTheLegs(_RestingCase):
    def test_expired_canceled_rejected_not_found(self):
        for status in ("expired", "canceled", "rejected", "done_for_day", None):
            with self.subTest(status=status):
                with open(self.state_path, "w") as f:
                    json.dump({}, f)
                legs, c = condor(40)
                self.register(c, "o-2")
                orders = {} if status is None else {"o-2": {"status": status}}
                self.run_manager(OrderBroker([], orders))
                self.assertEqual(self.read_state(), {})

    def test_a_filled_order_whose_legs_never_appear_is_dropped_after_the_grace(self):
        legs, c = condor(40)
        self.register(c, "o-3")
        recent = utc_iso(datetime.now() - timedelta(seconds=30))
        self.run_manager(OrderBroker([], {"o-3": {"status": "filled", "filled_at": recent}}))
        self.assertEqual(sorted(self.read_state()), sorted(legs.values()), "kept: the positions follow the fill")
        old = utc_iso(datetime.now() - timedelta(seconds=om.OPENING_FILL_GRACE_SECONDS + 60))
        self.run_manager(OrderBroker([], {"o-3": {"status": "filled", "filled_at": old}}))
        self.assertEqual(self.read_state(), {})

    def test_an_unreadable_order_keeps_the_legs_until_the_age_limit(self):
        legs, c = condor(40)
        self.register(c, "o-4")
        self.run_manager(OrderBroker([], {"o-4": "error"}))
        self.assertEqual(sorted(self.read_state()), sorted(legs.values()))
        st = self.read_state()
        for s in legs.values():
            st[s]["opening_order_at"] = st[s]["opening_order_at"] - om.OPENING_ORDER_MAX_AGE_SECONDS - 60
        with open(self.state_path, "w") as f:
            json.dump(st, f)
        self.run_manager(OrderBroker([], {"o-4": "error"}))
        self.assertEqual(self.read_state(), {})


class TestEveryEntryPathHandsTheOrderIdOver(_RestingCase):
    def vertical(self):
        lo, sh = occ(ROOT, 40, "C", 100.0), occ(ROOT, 40, "C", 110.0)
        return lo, sh, {"strategy": "bull_call_spread", "long_leg": lo, "short_leg": sh, "qty": 2, "net_debit": 3.2,
                        "limit_price": 3.2, "max_cost": 640.0, "max_loss": 640.0, "error": None,
                        "leg_quotes": [{"symbol": lo, "side": "buy", "bid": 5.00, "ask": 5.20},
                                       {"symbol": sh, "side": "sell", "bid": 2.00, "ask": 2.10}]}

    def test_the_mid_first_debit_entry_is_followed_through_the_ladder_replacements(self):
        lo, sh, c = self.vertical()
        with patch.object(oe, "should_use_options", return_value={"use_options": True, "strategy": "bull_call_spread",
                                                                   "edge_pct": 1.0, "reason": "test"}), \
                patch.object(oe, "select_contract", return_value=c), \
                patch.object(oe, "submit_options_order", return_value={"status": "pending_new", "order_id": "e-1"}), \
                patch.object(mleg_ladder, "settings", return_value={"on": True, "steps": 3, "step_seconds": 60.0,
                                                                      "touch_seconds": 120.0}), \
                patch("entry_shadow.is_shadowed", return_value=False), \
                patch.object(om, "register_entry_ladder"):
            oe.evaluate_and_execute({"ticker": ROOT, "price": 100.0}, 100000, [])
        self.assertEqual({s: self.read_state()[s][om.OPENING_ORDER_KEY] for s in (lo, sh)}, {lo: "e-1", sh: "e-1"})
        ladder = {"e-1": {"status": "replaced", "replaced_by": "e-2"}, "e-2": {"status": "replaced", "replaced_by": "e-3"},
                  "e-3": {"status": "new"}}
        self.run_manager(OrderBroker([], ladder))
        self.assertEqual({s: self.read_state()[s][om.OPENING_ORDER_KEY] for s in (lo, sh)}, {lo: "e-3", sh: "e-3"},
                         "the marker follows the ladder to its working order")
        groups = []
        with patch.object(om, "_manage_strategy_group", side_effect=lambda g, *a, **k: groups.append(g["strategy"]) or []):
            self.run_manager(OrderBroker([position(lo, 2, "long", 5.1, 5.1), position(sh, 2, "short", 2.05, 2.05)],
                                         {**ladder, "e-3": {"status": "filled", "filled_at": utc_iso(datetime.now())}}))
        self.assertEqual(groups, ["bull_call_spread"])
        self.assertTrue(all(om.OPENING_ORDER_KEY not in self.read_state()[s] for s in (lo, sh)))


class TestTheScannerHandsTheOrderIdOver(_ScannerCase):
    def test_the_scanner_legs_carry_the_order_id(self):
        state = self.run_script(CREDIT_PUT_SPREAD, "SPY", "bull_put_credit_spread", "vxx_panic_put_sale",
                                order={"status": "pending_new", "order_id": "o-7"})
        self.assertEqual((state[PUT_SHORT][om.OPENING_ORDER_KEY], state[PUT_LONG][om.OPENING_ORDER_KEY]), ("o-7", "o-7"))

    def test_an_order_alpaca_reports_as_new_is_registered(self):
        state = self.run_script(CREDIT_PUT_SPREAD, "SPY", "bull_put_credit_spread", "vxx_panic_put_sale",
                                order={"status": "new", "order_id": "o-8"})
        self.assertEqual(state[PUT_SHORT]["strategy"], "bull_put_credit_spread")


class TestSingleLegEntriesUnchanged(_RestingCase):
    def test_a_single_leg_entry_carries_no_marker_and_is_pruned_as_before(self):
        csp = occ(ROOT, 40, "P", 90.0)
        oe.register_contract_entry({"occ_symbol": csp, "side": "sell", "qty": 1, "limit_price": 1.2}, "sell_cash_secured_put",
                                   ROOT, order_id="s-1")
        self.assertNotIn(om.OPENING_ORDER_KEY, self.read_state()[csp])
        self.run_manager(OrderBroker([], {"s-1": {"status": "new"}}))
        self.assertEqual(self.read_state(), {})


if __name__ == "__main__":
    unittest.main()
