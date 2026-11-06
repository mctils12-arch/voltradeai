"""BOUGHT-OPTION PROFIT TAKING (2026-11-06): a bought option is scaled out
(half at +100%) and the rest trailed (35% off the peak once it has been +50%).

Human refocus spec §5 (2026-11-05): "hard profit-taking rules (take gains when
up), time-based exits ahead of theta decay, stop losses". Filed as
research/open_questions.md "RULE-REVIEW CANDIDATE 2026-11-02 — bought options
have no profit-taking exit", candidate (a); ON by the human's decision with
every trigger recorded (the published evidence does not support targets on
option premium, so the results are what will judge it).

Pinned here:
  - settings: the system_config defaults, bounds, the off switch;
  - the mark: the quote mid, only in regular hours after the opening window,
    only from a two-sided quote within peak_max_spread (no false peaks);
  - scale-out: floor(qty/2) sold at the bid, the leg's state KEPT (rest and
    peak), no exit recorded until the broker's quantity shows the fill; then
    one exit record (bought_scale_out) for the closed part and one open entry
    for the remainder; never a second scale-out; a lot of 1 is trailed whole;
  - a scale-out still settling holds the leg's other rules (no stacked close);
    one still working after scale_out_work_seconds is canceled and re-armed;
    one that ended unfilled re-arms; a refused one is not re-sent every tick;
  - trail: armed at a +50% peak, fires 35% below it, sells the whole rest at
    the bid, records bought_trail_stop once, keeps the state so an unfilled
    close is sent again (the peak is never lost); judged before the scale-out;
  - a bought leg that may cover a sold option of the same type on its
    underlying is never touched; the 21-DTE exit still wins;
  - debit groups: half the units in ONE partial mleg (no per-leg fallback),
    the trail through the pursued group close, the partial settles first;
  - order_ladder re-sends only what is left of its own order (a ladder
    working half a position never sells the other half).
"""
import json
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_manager as om  # noqa: E402
import order_ladder  # noqa: E402
import system_config  # noqa: E402
from ml_model_v2 import OPTIONS_EXIT_REASONS  # noqa: E402
from test_options_exit_market_hours import Broker, _ManagerCase, leg, occ, position  # noqa: E402

DTE = 40   # outside the 21-DTE exit and the 30-DTE gamma window


class PTBroker(Broker):
    """Broker plus open orders (GET /v2/orders) and the mleg verdict."""

    def __init__(self, positions, is_open=True, open_orders=None, mleg_status=500):
        super().__init__(positions, is_open=is_open)
        self.open_orders = open_orders or []
        self.mleg_status = mleg_status

    def get(self, url, **kw):
        if url.endswith("/v2/orders"):
            self.gets.append(url)
            syms = set(str((kw.get("params") or {}).get("symbols", "")).split(","))
            return MagicMock(status_code=200, json=lambda: [o for o in self.open_orders if o.get("symbol") in syms])
        return super().get(url, **kw)

    def post(self, url, **kw):
        self.posts.append((url, kw.get("json")))
        if (kw.get("json") or {}).get("order_class") == "mleg":
            return MagicMock(status_code=self.mleg_status, text='{"message":"mleg verdict in tests"}',
                             json=lambda: {"id": "mleg-1"})
        return MagicMock(status_code=500, text="no orders in tests", json=lambda: {})

    def set_qty(self, symbol, qty):
        for p in self.positions:
            if p["symbol"] == symbol:
                sign = -1 if p["side"] == "short" else 1
                p["qty"] = str(qty * sign)


def call(qty=2, entry=2.00, bid=4.00, ask=4.10, strategy="buy_call", root="PTKG", days=DTE, cp="C", strike=50):
    """A bought leg held 10 days. The mark (Alpaca current_price) follows the mid."""
    mid = round((bid + ask) / 2, 2)
    return leg(root, days, cp, strike, qty, "long", entry, mid, bid, ask, 0.45, 0.02, strategy, None)


class _PTCase(_ManagerCase):
    def setUp(self):
        super().setUp()
        self.cancels = []
        p = patch.object(om, "_cancel_order", side_effect=lambda oid: self.cancels.append(oid) or True)
        p.start()
        self.addCleanup(p.stop)

    def tick(self, broker, **kw):
        self._cold_clock()
        with patch.object(om.requests, "get", side_effect=broker.get), \
                patch.object(om.requests, "post", side_effect=broker.post):
            return om.manage_options_positions(100000, **kw)

    def quote(self, lg, bid, ask, broker=None):
        self.snaps[lg["occ"]] = {**self.snaps.get(lg["occ"], {}), "bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 2)}
        if broker is not None:
            for p in broker.positions:
                if p["symbol"] == lg["occ"]:
                    q = abs(int(float(p["qty"])))
                    entry = float(p["avg_entry_price"])
                    mid = (bid + ask) / 2
                    p.update(current_price=str(mid), unrealized_pl=str((mid - entry) * q * 100))

    def pt(self, lg):
        return self.read_state().get(lg["occ"], {}).get("profit_taking", {})

    def age(self, lg, key, seconds):
        """Make the leg's pending scale-out / trail record `seconds` old."""
        st = self.read_state()
        rec = st[lg["occ"]]["profit_taking"][key]
        stamp = (datetime.now() - timedelta(seconds=seconds)).isoformat()
        rec["submitted_at" if "submitted_at" in rec else "at"] = stamp
        with open(self.state_path, "w") as f:
            json.dump(st, f)

    def log(self):
        path = os.path.join(om.DATA_DIR, om.PROFIT_TAKING_LOG_FILE)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def closes(self, result):
        return [a for a in result["actions"] if a.get("action") in ("CLOSE", "CLOSE_STRATEGY")]


class TestSettings(unittest.TestCase):
    def test_defaults_are_the_config_and_on(self):
        self.assertEqual(system_config.BASE_CONFIG["OPTIONS_PROFIT_TAKING"],
                         {"enabled": True, "scale_out_gain": 1.00, "scale_out_fraction": 0.5, "trail_arm_gain": 0.50,
                          "trail_giveback": 0.35, "peak_max_spread": 0.50, "scale_out_work_seconds": 600})
        cfg = om._profit_taking_settings()
        self.assertEqual((cfg["enabled"], cfg["scale_out_gain"], cfg["scale_out_fraction"], cfg["trail_arm_gain"],
                          cfg["trail_giveback"]), (True, 1.0, 0.5, 0.5, 0.35))

    def test_out_of_bounds_values_keep_the_default(self):
        bad = {"enabled": True, "scale_out_gain": -1, "scale_out_fraction": 1.5, "trail_giveback": "x",
               "trail_arm_gain": float("nan"), "peak_max_spread": 0.3}
        with patch.dict(system_config.BASE_CONFIG, {"OPTIONS_PROFIT_TAKING": bad}):
            cfg = om._profit_taking_settings()
        self.assertEqual((cfg["scale_out_gain"], cfg["scale_out_fraction"], cfg["trail_giveback"], cfg["trail_arm_gain"]),
                         (1.0, 0.5, 0.35, 0.5))
        self.assertEqual(cfg["peak_max_spread"], 0.3)
        with patch.dict(system_config.BASE_CONFIG, {"OPTIONS_PROFIT_TAKING": None}):
            self.assertTrue(om._profit_taking_settings()["enabled"], "a broken entry runs the defaults")

    def test_the_mark_is_a_sane_two_sided_mid(self):
        self.assertEqual(om._quote_mid({"bid": 1.0, "ask": 1.2}, 0.5), 1.1)
        self.assertIsNone(om._quote_mid({"bid": 0.0, "ask": 5.0}, 0.5), "the 0 / 5.00 pre-market XLF quote is no mark")
        self.assertIsNone(om._quote_mid({"bid": 0.2, "ask": 2.0}, 0.5), "a 164%-wide quote is no mark")
        self.assertIsNone(om._quote_mid({"bid": 1.3, "ask": 1.2}, 0.5), "crossed")
        self.assertIsNone(om._quote_mid({}, 0.5))

    def test_exit_reasons_are_options_exit_reasons(self):
        self.assertIn("bought_scale_out", OPTIONS_EXIT_REASONS)
        self.assertIn("bought_trail_stop", OPTIONS_EXIT_REASONS)


class TestScaleOut(_PTCase):
    def test_half_is_sold_at_the_bid_and_the_state_kept_no_exit_recorded_yet(self):
        lg = call(qty=4, entry=2.00, bid=4.00, ask=4.10)   # mid 4.05: +102.5%
        self.load({"c": lg})
        broker = PTBroker([lg["pos"]])
        r = self.tick(broker)
        self.assertEqual(self.submits, [(lg["occ"], 2, "sell", 4.00)])
        (act,) = self.closes(r)
        self.assertEqual((act["type"], act["qty"], act["of_qty"], act["partial"]), ("bought_scale_out", 2, 4, True))
        self.assertIn("PROFIT SCALE-OUT", act["reason"])
        so = self.pt(lg)["scale_out"]
        self.assertEqual((so["status"], so["qty"], so["qty_before"]), ("submitted", 2, 4))
        self.assertIn(lg["occ"], self.read_state(), "the rest rides: state kept")
        self.track_fill.assert_not_called()
        self.assertEqual([e["event"] for e in self.log()], ["trail_armed", "scale_out_submitted"],
                         "+102.5% also arms the trail (+50%) for the rest")

    def test_the_fill_records_one_exit_and_reopens_the_remainder_and_never_scales_again(self):
        lg = call(qty=4, entry=2.00, bid=4.00, ask=4.10)
        self.load({"c": lg})
        broker = PTBroker([lg["pos"]])
        self.tick(broker)
        broker.set_qty(lg["occ"], 2)          # the half filled
        self.submits.clear()
        r = self.tick(broker)
        self.assertEqual(self.submits, [], "filled: nothing new while the rest sits below any trigger")
        calls = [c.args[0] for c in self.track_fill.call_args_list]
        self.assertEqual(len(calls), 2)
        self.assertEqual((calls[0]["exit_reason"], calls[0]["qty"]), ("bought_scale_out", 2))
        self.assertAlmostEqual(calls[0]["exit_context"]["pnl_pct"], 102.5, places=1)
        self.assertNotIn("exit_reason", calls[1])
        self.assertEqual((calls[1]["qty"], calls[1]["fill_price"], calls[1]["side"]), (2, 2.00, "buy"))
        self.assertEqual(self.pt(lg)["scale_out"]["status"], "filled")
        self.assertEqual(self.read_state()[lg["occ"]]["qty"], 2)
        self.assertEqual(self.closes(r), [])
        self.quote(lg, 6.00, 6.10, broker)    # +202%: a second scale-out never fires
        self.tick(broker)
        self.assertEqual(self.submits, [])
        events = [e["event"] for e in self.log()]
        self.assertEqual(events.count("scale_out_submitted"), 1)
        self.assertLess(events.index("scale_out_submitted"), events.index("scale_out_filled"))

    def test_a_lot_of_one_is_not_split_and_the_trail_manages_it(self):
        lg = call(qty=1, entry=2.00, bid=4.00, ask=4.10)
        self.load({"c": lg})
        broker = PTBroker([lg["pos"]])
        self.tick(broker)
        self.assertEqual(self.submits, [])
        self.assertEqual(self.pt(lg)["scale_out"]["status"], "not_possible")
        self.assertIn("scale_out_not_possible", [e["event"] for e in self.log()])
        self.tick(broker)
        self.assertEqual([e["event"] for e in self.log()].count("scale_out_not_possible"), 1, "logged once")
        self.quote(lg, 2.60, 2.66, broker)    # mid 2.63 <= 4.05 * 0.65 = 2.63: the trail
        self.tick(broker)
        self.assertEqual(self.submits, [(lg["occ"], 1, "sell", 2.60)])

    def test_three_contracts_sell_one(self):
        lg = call(qty=3, entry=2.00, bid=4.00, ask=4.10)
        self.load({"c": lg})
        self.tick(PTBroker([lg["pos"]]))
        self.assertEqual(self.submits, [(lg["occ"], 1, "sell", 4.00)], "floor(3 x 0.5) = 1")

    def test_below_the_target_nothing_happens(self):
        lg = call(qty=2, entry=2.00, bid=3.90, ask=4.00)  # mid 3.95: +97.5%
        self.load({"c": lg})
        r = self.tick(PTBroker([lg["pos"]]))
        self.assertEqual((self.submits, self.closes(r)), ([], []))

    def test_off_switch(self):
        lg = call(qty=2, entry=2.00, bid=4.00, ask=4.10)
        self.load({"c": lg})
        with patch.dict(system_config.BASE_CONFIG["OPTIONS_PROFIT_TAKING"], {"enabled": False}):
            r = self.tick(PTBroker([lg["pos"]]))
        self.assertEqual((self.submits, self.closes(r), self.log()), ([], [], []))


class TestTheMark(_PTCase):
    def test_no_mark_outside_regular_hours_in_the_opening_window_or_from_a_bad_quote(self):
        lg = call(qty=2, entry=2.00, bid=4.00, ask=4.10)
        self.load({"c": lg})
        self.tick(PTBroker([lg["pos"]], is_open=False))
        with patch.object(om, "_in_opening_window", return_value=True):
            self.tick(PTBroker([lg["pos"]]))
        self.quote(lg, 0.0, 9.0)
        self.tick(PTBroker([lg["pos"]]))
        self.quote(lg, 1.0, 8.0)              # 156% wide
        self.tick(PTBroker([lg["pos"]]))
        self.assertEqual(self.submits, [])
        self.assertNotIn("peak_ratio", self.pt(lg), "no peak from any of these")

    def test_the_peak_ignores_the_brokers_old_high_water_mark(self):
        lg = call(qty=2, entry=2.00, bid=2.30, ask=2.36)     # mid 2.33
        lg["state"]["highest_value"] = 9.99                    # e.g. a pre-market mark
        self.load({"c": lg})
        self.tick(PTBroker([lg["pos"]]))
        self.assertEqual(self.submits, [], "an old highest_value never arms the trail")
        self.assertAlmostEqual(self.pt(lg)["peak_ratio"], 1.165, places=3)


class TestTrail(_PTCase):
    def _armed(self, qty=2):
        lg = call(qty=qty, entry=2.00, bid=3.50, ask=3.60)   # mid 3.55: +77.5%, armed
        self.load({"c": lg})
        broker = PTBroker([lg["pos"]])
        self.tick(broker)
        self.assertEqual(self.submits, [])
        self.assertIn("trail_armed", [e["event"] for e in self.log()])
        return lg, broker

    def test_fires_35_percent_below_the_peak_sells_the_rest_records_once_keeps_state(self):
        lg, broker = self._armed()
        self.quote(lg, 2.34, 2.38, broker)    # mid 2.36 > 3.55 * 0.65 = 2.3075: holds
        self.tick(broker)
        self.assertEqual(self.submits, [])
        self.quote(lg, 2.28, 2.32, broker)    # mid 2.30: fires
        r = self.tick(broker)
        self.assertEqual(self.submits, [(lg["occ"], 2, "sell", 2.28)])
        (act,) = self.closes(r)
        self.assertEqual((act["type"], act["partial"]), ("bought_trail_stop", False))
        self.assertIn("PROFIT TRAIL", act["reason"])
        self.assertEqual([c.args[0]["exit_reason"] for c in self.track_fill.call_args_list], ["bought_trail_stop"])
        self.assertEqual(self.pt(lg)["trail"]["status"], "submitted")
        self.assertIn(lg["occ"], self.read_state(), "kept, so an unfilled close is sent again")
        # working at the broker: wait, nothing new, no second record
        broker.open_orders = [{"id": "o-9", "symbol": lg["occ"], "side": "sell"}]
        r = self.tick(broker)
        self.assertEqual(len(self.submits), 1)
        self.assertEqual([a["type"] for a in r["actions"] if a["action"] == "WAIT"], ["bought_trail_stop_working"])
        # it ended unfilled (a restart canceled it): sent again, not recorded again
        broker.open_orders = []
        self.age(lg, "trail", om.CLOSE_RETRY_SECONDS + 5)
        self.tick(broker)
        self.assertEqual(len(self.submits), 2)
        self.assertEqual(self.track_fill.call_count, 1)
        # filled: the position is gone and so is its state
        broker.positions = []
        self.tick(broker)
        self.assertNotIn(lg["occ"], self.read_state())

    def test_a_refused_trail_close_is_not_resent_every_tick(self):
        lg, broker = self._armed()
        self.quote(lg, 2.20, 2.24, broker)
        with patch.object(om, "_submit_close_order", return_value={"status": "error", "detail": "Alpaca rejected"}) as sub:
            self.tick(broker)
            self.tick(broker)
            self.assertEqual(sub.call_count, 1)
            self.assertEqual(self.pt(lg)["trail"]["status"], "refused")
            self.age(lg, "trail", om.CLOSE_RETRY_SECONDS + 5)
            self.tick(broker)
            self.assertEqual(sub.call_count, 2)
        self.track_fill.assert_not_called()

    def test_the_trail_is_judged_before_the_scale_out(self):
        lg = call(qty=4, entry=2.00, bid=6.40, ask=6.50)     # peak mid 6.45 (+222%)
        self.load({"c": lg})
        broker = PTBroker([lg["pos"]])
        with patch.dict(system_config.BASE_CONFIG["OPTIONS_PROFIT_TAKING"], {"scale_out_gain": 5.0}):
            self.tick(broker)                  # no scale-out yet (target out of reach)
        self.quote(lg, 4.10, 4.20, broker)    # +107.5%, and 4.15 <= 6.45 * 0.65 = 4.19
        self.tick(broker)
        self.assertEqual(self.submits, [(lg["occ"], 4, "sell", 4.10)], "the trail sells all, not half")


class TestScaleOutSettling(_PTCase):
    def _sent(self):
        lg = call(qty=4, entry=2.00, bid=4.00, ask=4.10)
        self.load({"c": lg})
        broker = PTBroker([lg["pos"]])
        self.tick(broker)
        self.assertEqual(len(self.submits), 1)
        return lg, broker

    def test_while_its_order_works_the_legs_other_rules_wait(self):
        lg, broker = self._sent()
        broker.open_orders = [{"id": "o-1", "symbol": lg["occ"], "side": "sell"}]
        self.snaps[lg["occ"]]["gamma"] = 0.50          # gamma risk would close the whole leg
        with patch.object(om, "_days_to_expiry", return_value=25):
            r = self.tick(broker)
        self.assertEqual(len(self.submits), 1, "no full close stacked on the partial one")
        self.assertEqual([a["type"] for a in r["actions"] if a["action"] == "WAIT"], ["bought_scale_out_working"])
        self.assertEqual(self.cancels, [])

    def test_still_working_after_the_work_window_it_is_canceled_and_re_armed(self):
        lg, broker = self._sent()
        broker.open_orders = [{"id": "o-1", "symbol": lg["occ"], "side": "sell"}]
        self.age(lg, "scale_out", om._profit_taking_settings()["scale_out_work_seconds"] + 5)
        self.tick(broker)
        self.assertEqual(self.cancels, ["o-1"])
        self.assertEqual(len(self.submits), 1, "nothing else is sent in the tick of the cancel")
        self.assertEqual(self.pt(lg)["scale_out"]["status"], "unfilled")
        broker.open_orders = []
        self.tick(broker)
        self.assertEqual(len(self.submits), 2, "re-armed: the scale-out fires again")

    def test_the_exit_ladder_walking_the_sale_is_never_cut_by_the_work_window(self):
        """The mid-first exit ladder (order_ladder.py) may need ~2 ticks (up to 14
        min) to reach the bid: the work window starts when the sale reaches it."""
        lg, broker = self._sent()
        broker.open_orders = [{"id": "o-1", "symbol": lg["occ"], "side": "sell"}]
        self.age(lg, "scale_out", om._profit_taking_settings()["scale_out_work_seconds"] + 300)
        ladder_path = os.path.join(om.DATA_DIR, order_ladder.STATE_FILE)
        now = order_ladder.wall_clock()
        with open(ladder_path, "w") as f:
            json.dump({lg["occ"]: {"occ": lg["occ"], "side": "sell", "qty": 2, "steps": 3, "step": 2,
                                   "price": 4.03, "order_id": "o-1", "started_at": now - 900, "step_at": now - 60}}, f)
        with patch.object(om, "_pursue_exit_ladders", return_value=[]):
            self.tick(broker)
            self.assertEqual(self.cancels, [], "step 2 of 3: the ladder is still walking")
            with open(ladder_path, "w") as f:
                json.dump({lg["occ"]: {"occ": lg["occ"], "side": "sell", "qty": 2, "steps": 3, "step": 3,
                                       "price": 4.00, "order_id": "o-1", "started_at": now - 900, "step_at": now - 120}}, f)
            self.tick(broker)
            self.assertEqual(self.cancels, [], "at the touch for 120 s: inside the work window")
            with open(ladder_path, "w") as f:
                json.dump({lg["occ"]: {"occ": lg["occ"], "side": "sell", "qty": 2, "steps": 3, "step": 3,
                                       "price": 4.00, "order_id": "o-1", "started_at": now - 1500, "step_at": now - 700}}, f)
            self.tick(broker)
        self.assertEqual(self.cancels, ["o-1"], "at the touch past the work window: canceled and re-armed")
        self.assertEqual(self.pt(lg)["scale_out"]["status"], "unfilled")

    def test_an_order_that_ended_unfilled_re_arms(self):
        lg, broker = self._sent()
        self.tick(broker)                               # just sent: settling, nothing new
        self.assertEqual(len(self.submits), 1)
        self.age(lg, "scale_out", om.CLOSE_RETRY_SECONDS + 5)
        self.tick(broker)
        self.assertEqual(len(self.submits), 2)
        self.assertIn("scale_out_unfilled", [e["event"] for e in self.log()])
        self.track_fill.assert_not_called()

    def test_a_partial_fill_that_ended_records_what_closed(self):
        lg, broker = self._sent()
        broker.set_qty(lg["occ"], 3)                   # 1 of 2 filled, order gone
        self.tick(broker)
        exit_rec = self.track_fill.call_args_list[0].args[0]
        self.assertEqual((exit_rec["exit_reason"], exit_rec["qty"]), ("bought_scale_out", 1))
        self.assertEqual(self.track_fill.call_args_list[1].args[0]["qty"], 3)
        self.assertEqual(self.pt(lg)["scale_out"]["status"], "filled")

    def test_a_refused_scale_out_is_not_resent_every_tick(self):
        lg = call(qty=4, entry=2.00, bid=4.00, ask=4.10)
        self.load({"c": lg})
        broker = PTBroker([lg["pos"]])
        with patch.object(om, "_submit_close_order", return_value={"status": "error", "detail": "rejected"}) as sub:
            self.tick(broker)
            self.tick(broker)
            self.assertEqual(sub.call_count, 1)
            self.age(lg, "scale_out", om.CLOSE_RETRY_SECONDS + 5)
            self.tick(broker)
            self.assertEqual(sub.call_count, 2)


class TestWhoIsManaged(_PTCase):
    def test_a_long_that_may_cover_a_sold_option_of_its_type_is_never_touched(self):
        wing = call(qty=2, entry=1.00, bid=2.40, ask=2.50, strategy="unknown", cp="P", strike=45)
        short = leg("PTKG", DTE, "P", 50, 2, "short", 3.00, 2.95, 2.90, 3.00, -0.40, 0.02, "unknown", None)
        self.load({"w": wing, "s": short})
        self.tick(PTBroker([wing["pos"], short["pos"]]))
        self.assertNotIn(wing["occ"], [s[0] for s in self.submits])
        self.assertIn("covers a sold option", self.pt(wing)["blocked"])
        self.tick(PTBroker([wing["pos"], short["pos"]]))
        self.assertEqual([e["event"] for e in self.log()].count("not_applied"), 1, "logged once")

    def test_a_sold_option_of_the_other_type_does_not_block(self):
        lg = call(qty=2, entry=2.00, bid=4.00, ask=4.10)
        short_put = leg("PTKG", DTE, "P", 40, 1, "short", 1.00, 0.95, 0.90, 1.00, -0.20, 0.01, "sell_cash_secured_put", None)
        self.load({"c": lg, "s": short_put})
        self.tick(PTBroker([lg["pos"], short_put["pos"]]))
        self.assertIn((lg["occ"], 1, "sell", 4.00), self.submits)

    def test_the_21_dte_exit_still_wins(self):
        lg = call(qty=2, entry=2.00, bid=4.00, ask=4.10, days=15)
        self.load({"c": lg})
        r = self.tick(PTBroker([lg["pos"]]))
        self.assertEqual(self.submits, [(lg["occ"], 2, "sell", 4.00)])
        self.assertEqual([a["type"] for a in self.closes(r)], ["dte_close_bought"])

    def test_sold_options_are_untouched(self):
        csp = leg("PTKS", DTE, "P", 40, 2, "short", 1.00, 0.80, 0.78, 0.82, -0.20, 0.01, "sell_cash_secured_put", None)
        self.load({"s": csp})
        self.tick(PTBroker([csp["pos"]]))
        self.assertNotIn("profit_taking", self.read_state()[csp["occ"]])


# ── debit groups ────────────────────────────────────────────────────────────

def spread(qty_long=2, qty_short=2, long_entry=5.00, short_entry=2.00, root="PTSP"):
    """A bull call spread (net debit 3.00/unit) held 10 days, 40 DTE."""
    lo, sh = occ(root, DTE, "C", 100), occ(root, DTE, "C", 110)
    held = (datetime.now() - timedelta(days=10)).isoformat()
    positions = [position(lo, qty_long, "long", long_entry, long_entry), position(sh, qty_short, "short", short_entry, short_entry)]
    state = {}
    for s, e, sd, q in ((lo, long_entry, "long", qty_long), (sh, short_entry, "short", qty_short)):
        state[s] = {"entry_price": e, "side": sd, "qty": q, "strategy": "bull_call_spread", "ticker": root, "setup": "",
                    "entry_date": held[:10], "entry_timestamp": held, "initial_credit": e if sd == "short" else 0,
                    "highest_value": e, "max_loss": 0}
    return lo, sh, positions, state


class TestDebitGroups(_PTCase):
    def setUp(self):
        super().setUp()
        p = patch.dict(sys.modules, {"vol_surface": MagicMock(get_surface_score=lambda t: {"vrp": {"vrp_20d": 0.0}})})
        p.start()
        self.addCleanup(p.stop)

    def _load(self, lo, sh, positions, state, long_q=(11.0, 11.2), short_q=(4.9, 5.1)):
        with open(self.state_path, "w") as f:
            json.dump(state, f)
        self.snaps[lo] = {"bid": long_q[0], "ask": long_q[1], "mid": sum(long_q) / 2, "gamma": 0.01, "delta": 0.6}
        self.snaps[sh] = {"bid": short_q[0], "ask": short_q[1], "mid": sum(short_q) / 2, "gamma": 0.01, "delta": 0.3}

    def test_half_the_units_in_one_partial_mleg_then_it_settles(self):
        lo, sh, positions, state = spread()
        self._load(lo, sh, positions, state)          # value 11.1 - 5.0 = 6.1/unit on 3.0: +103%
        broker = PTBroker(positions, mleg_status=200)
        r = self.tick(broker)
        (act,) = self.closes(r)
        self.assertEqual((act["type"], act["units"], act["of_units"], act["partial"]), ("strategy_scale_out", 1, 2, True))
        (payload,) = [p for _u, p in broker.posts if p and p.get("order_class") == "mleg"]
        self.assertEqual(payload["qty"], "1")
        self.assertEqual(sorted(l["position_intent"] for l in payload["legs"]), ["buy_to_close", "sell_to_close"])
        st = self.read_state()
        self.assertTrue(all(st[o]["group_profit_taking"]["scale_out"]["status"] == "submitted" for o in (lo, sh)))
        self.assertFalse(any("close_requested_at" in st[o] for o in (lo, sh)), "a partial close is no group close request")
        # working: every other group rule waits
        broker.open_orders = [{"id": "mleg-1", "symbol": lo, "side": "sell"}]
        r = self.tick(broker)
        self.assertEqual([a["type"] for a in r["actions"] if a["action"] == "WAIT"], ["strategy_scale_out_working"])
        self.assertEqual(len([p for _u, p in broker.posts if p and p.get("order_class") == "mleg"]), 1)
        # filled
        broker.open_orders = []
        broker.set_qty(lo, 1)
        broker.set_qty(sh, 1)
        self.tick(broker)
        self.assertEqual(self.read_state()[lo]["group_profit_taking"]["scale_out"]["status"], "filled")
        self.assertIn("group_scale_out_filled", [e["event"] for e in self.log()])

    def test_a_refused_partial_mleg_is_never_split_into_leg_orders(self):
        lo, sh, positions, state = spread()
        self._load(lo, sh, positions, state)
        broker = PTBroker(positions, mleg_status=422)
        r = self.tick(broker)
        (act,) = self.closes(r)
        self.assertEqual((act["order"]["status"], act["order"]["method"]), ("error", "mleg"))
        self.assertEqual(self.submits, [], "no per-leg fallback for a partial close")
        self.assertEqual(self.read_state()[lo]["group_profit_taking"]["scale_out"]["status"], "refused")

    def test_the_trail_closes_the_whole_group_through_the_pursued_close(self):
        lo, sh, positions, state = spread()
        self._load(lo, sh, positions, state, long_q=(9.5, 9.7), short_q=(4.4, 4.6))   # 5.1 on 3.0: +70%, armed
        broker = PTBroker(positions, mleg_status=200)
        self.tick(broker)
        self.assertEqual(self.closes(self.tick(broker)), [])
        self.snaps[lo].update(bid=7.0, ask=7.2)        # 7.1 - 4.5 = 2.6... below 5.1 * 0.65 = 3.315
        r = self.tick(broker)
        (act,) = self.closes(r)
        self.assertEqual(act["type"], "strategy_trail_stop")
        st = self.read_state()
        self.assertTrue(all(st[o]["close_rule"] == "strategy_trail_stop" for o in (lo, sh)))
        (payload,) = [p for _u, p in broker.posts if p and p.get("order_class") == "mleg"]
        self.assertEqual(payload["qty"], "2", "the whole group")

    def test_unequal_legs_cannot_be_halved(self):
        lo, sh, positions, state = spread(qty_long=3, qty_short=2)
        self._load(lo, sh, positions, state)
        r = self.tick(PTBroker(positions, mleg_status=200))
        self.assertEqual(self.closes(r), [])
        self.assertEqual(self.read_state()[lo]["group_profit_taking"]["scale_out"]["status"], "not_possible")

    def test_credit_groups_carry_no_profit_taking_record(self):
        lo, sh, positions, state = spread()
        for s in state.values():
            s["strategy"] = "bull_put_credit_spread"
        self._load(lo, sh, positions, state)
        self.tick(PTBroker(positions, mleg_status=200))
        self.assertFalse(any("group_profit_taking" in v for v in self.read_state().values()))


# ── the exit ladder working half a position ─────────────────────────────────

class TestLadderResendsOnlyItsOwnRest(unittest.TestCase):
    OCC = "PTKG261218C00050000"

    def _resend(self, held_qty):
        """A 2-contract sell ladder, 1 filled at step 1, then canceled for the
        rest (pending step 2): what does the next tick re-send?"""
        posts = []
        entry = {"occ": self.OCC, "side": "sell", "qty": 2, "steps": 3, "step": 1, "price": 4.05,
                 "order_id": "o-1", "started_at": 1000.0, "step_at": 1000.0, "filled_qty": 1.0,
                 "pending": {"step": 2, "at": 1000.0, "why": "partial fill"}}
        io = order_ladder.LadderIO(
            quote=lambda occ: {"bid": 4.00, "ask": 4.10},
            post=lambda occ, qty, side, px: posts.append(qty) or {"status": "submitted", "order_id": "o-2"},
            replace=lambda oid, px: (None, "x"), get_order=lambda oid: {"id": oid, "status": "canceled", "filled_qty": "1"},
            cancel=lambda oid: True, market_ok=lambda: True, state_path="unused", log_path="unused", now=lambda: 1100.0)
        io.load = lambda: {self.OCC: dict(entry)}
        io.save = lambda s: None
        io.record = lambda rec: None
        order_ladder.pursue(io, {"steps": 3, "step_seconds": 60}, {self.OCC: {"symbol": self.OCC, "qty": str(held_qty)}})
        return posts

    def test_a_partly_filled_half_close_resends_its_own_rest_not_the_position(self):
        # a scale-out of 2 out of 4: 1 sold, 3 still held
        self.assertEqual(self._resend(held_qty=3), [1], "the ladder's own rest (2 - 1), never the 3 still held")

    def test_a_full_close_still_resends_the_whole_rest(self):
        self.assertEqual(self._resend(held_qty=1), [1])


if __name__ == "__main__":
    unittest.main()
