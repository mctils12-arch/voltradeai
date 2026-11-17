#!/usr/bin/env python3
"""
OPENING-WINDOW COUNTERFACTUAL (2026-11-12), the Python half. Measurement only.

research/open_questions.md, "opening-window exits: the counterfactual to log":
no option exit is sent before 09:45 ET, and the rule's rollback trigger needs,
for each exit it held, the quote at the moment a rule first fired against what
the exit got after 09:45. Until now only the count of held exits was kept.

Pinned here:
  (1) FIRST FIRED. The first scan on which the window holds an exit writes one
      record (the rule, the time, per leg the close side, quantity, mark and
      bid / ask / mid) under the broker's ET day. Later scans in the window
      count and never replace the first quote. A group's legs are read once,
      and that read never reaches the fill ledger's quote memo.
  (2) THE LINK. Once the exit is sent, after a process restart too, the fill
      ledger row of every order for it carries the record. A contract the
      window never held writes the row it always wrote.
  (3) NEVER SENT / REFUSED. A rule that no longer holds after 09:45 leaves the
      record and no row. A close the broker refuses is counted on the record.
  (4) NO DECISION CHANGES. The actions and the order POSTs are what they were;
      a corrupt, missing or unwritable file costs the record and nothing else.

Mocked: no network (requests.get / post / patch stand-ins); DATA_DIR in a temp
dir. Run: python3 -m pytest -q test_option_opening_first_fired.py
"""
import contextlib
import copy
import inspect
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import option_fill_ledger as ofl  # noqa: E402
import options_manager as om  # noqa: E402
import storage_config  # noqa: E402
from test_options_exit_market_hours import book, leg, position  # noqa: E402

ET = ZoneInfo("America/New_York")
IN_WINDOW = "2026-11-03T09:35:12.123456789-05:00"
LATER_IN_WINDOW = "2026-11-03T09:41:00-05:00"
AFTER_WINDOW = "2026-11-03T09:46:00-05:00"
OLD_ROW_KEYS = {"ts", "order_id", "symbol", "side", "qty", "bid", "ask", "mid", "source", "limit", "quote_age_s"}
LADDER_OFF = {"on": False, "setting": "off", "steps": 3, "step_seconds": 60.0}
LADDER_ON = {"on": True, "setting": "on", "steps": 3, "step_seconds": 60.0}
MLEG_OFF = {"on": False, "setting": "off", "steps": 3, "step_seconds": 60.0, "touch_seconds": 120.0}


def _resp(status, body):
    r = MagicMock(status_code=status, text=json.dumps(body))
    r.json.return_value = body
    return r


def condor(days=15):
    """An iron condor at 15 DTE, flat on the day: only the group's 21-DTE exit fires."""
    return [leg("COND", days, "P", 290, 2, "long", 3.00, 3.10, 3.00, 3.20, -0.10, 0.01, "iron_condor", None),
            leg("COND", days, "P", 300, 2, "short", 4.70, 4.80, 4.70, 4.90, -0.20, 0.01, "iron_condor", None),
            leg("COND", days, "C", 355, 2, "short", 4.50, 4.40, 4.30, 4.50, 0.20, 0.01, "iron_condor", None),
            leg("COND", days, "C", 365, 2, "long", 2.90, 2.80, 2.70, 2.90, 0.10, 0.01, "iron_condor", None)]


class Broker:
    """requests.get / post / patch stand-in: positions, the clock with Alpaca's
    ET timestamp (the opening window reads it), one option snapshot per
    contract, no open orders, and an order endpoint that accepts every order
    except those for a contract in `reject`."""

    def __init__(self, legs, stamp, is_open=True, reject=()):
        self.positions = [l["pos"] for l in legs]
        self.quotes = {l["occ"]: l["snap"] for l in legs}
        self.stamp, self.is_open, self.reject = stamp, is_open, set(reject)
        self.posts, self.snapshot_reads = [], []

    def get(self, url, **kw):
        if "/v2/positions" in url:
            return _resp(200, self.positions)
        if "/v2/clock" in url:
            return _resp(200, {"is_open": self.is_open, "timestamp": self.stamp})
        if "/options/snapshots/" in url:
            occ = (kw.get("params") or {}).get("symbols", "")
            self.snapshot_reads.append(occ)
            q = self.quotes.get(occ)
            if q is None:
                return _resp(200, {"snapshots": {}})
            return _resp(200, {"snapshots": {occ: {
                "latestQuote": {"bp": q["bid"], "ap": q["ask"]}, "impliedVolatility": q["iv"],
                "greeks": {"delta": q["delta"], "gamma": q["gamma"], "theta": q["theta"], "vega": q["vega"]}}}})
        if "/v2/orders" in url:
            return _resp(200, [])
        return _resp(404, {})

    def post(self, url, **kw):
        body = kw.get("json") or {}
        self.posts.append(body)
        if body.get("symbol") in self.reject:
            return _resp(422, {"message": "insufficient options buying power"})
        return _resp(200, {"id": f"ord-{len(self.posts)}"})

    def patch(self, url, **kw):
        return _resp(422, {"message": "no replace in this test"})


class _Case(unittest.TestCase):
    """The real manager, snapshot reader and close paths; only the network is a stand-in."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.state_path = os.path.join(self.dir, "voltrade_options_state.json")
        inactive = {"active": False, "today": None, "by_root": {}, "exempt": {}}
        for p in (patch.object(om, "DATA_DIR", self.dir),
                  patch.object(om, "OPTIONS_STATE_PATH", self.state_path),
                  patch.object(ofl, "DATA_DIR", self.dir),
                  patch.object(om.alpaca_feed, "options_feed", return_value="indicative"),
                  patch.object(om.order_ladder, "settings", return_value=dict(LADDER_OFF)),
                  patch.object(om.mleg_ladder, "settings", return_value=dict(MLEG_OFF)),
                  patch.object(om, "_earnings_close_context", return_value=inactive),
                  patch.dict(sys.modules, {"ml_model_v2": MagicMock(track_fill=MagicMock())})):
            p.start()
            self.addCleanup(p.stop)
        self.restart()
        self.addCleanup(self.restart)

    @staticmethod
    def restart():
        """What a daemon recycle or the subprocess fallback starts from: no
        quote memo and no clock reading. Only the files remain."""
        ofl._quotes.clear()
        om._CLOCK_CACHE.update({"at": 0.0, "is_open": None, "et": None, "et_at": 0.0})

    def load(self, legs):
        with open(self.state_path, "w") as f:
            json.dump({l["occ"]: dict(l["state"]) for l in legs}, f)

    def run_manager(self, broker):
        om._CLOCK_CACHE.update({"at": 0.0, "is_open": None})   # every run reads its own broker's clock
        with patch.object(om.requests, "get", side_effect=broker.get), \
                patch.object(om.requests, "post", side_effect=broker.post), \
                patch.object(om.requests, "patch", side_effect=broker.patch):
            return om.manage_options_positions(100000, allow_rolls=False)

    def records(self, day="2026-11-03"):
        path = os.path.join(self.dir, "voltrade_option_opening_first_fired.json")
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)["days"].get(day, {})

    def rows(self):
        path = os.path.join(self.dir, ofl.LEDGER_FILENAME)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]


def requoted(l, bid, ask, cur=None):
    """The same leg at a later quote (and mark)."""
    out = copy.deepcopy(l)
    out["snap"].update(bid=bid, ask=ask, mid=round((bid + ask) / 2, 2))
    if cur is not None:
        out["pos"] = position(l["occ"], l["qty"], l["pos"]["side"], float(l["pos"]["avg_entry_price"]), cur)
    return out


class TestFirstFiredRecord(_Case):
    def test_each_held_exit_records_its_rule_time_side_and_quote(self):
        legs = book()
        self.load(legs.values())
        broker = Broker(legs.values(), IN_WINDOW)
        result = self.run_manager(broker)
        self.assertEqual(broker.posts, [], "nothing is sent inside the window")
        held = [a for a in result["actions"] if a.get("order", {}).get("detail") == om.OPENING_WINDOW_DEFERRED_DETAIL]
        self.assertEqual(len(held), len(legs))
        recs = self.records()
        self.assertEqual(sorted(recs), sorted(l["occ"] for l in legs.values()), "one record per contract, under the broker's ET day")
        for rule, l in legs.items():
            with self.subTest(rule=rule):
                rec = recs[l["occ"]]
                self.assertEqual((rec["kind"], rec["rule"], rec["rules"], rec["fires"], rec["refused"]),
                                 ("single_leg", rule, [rule], 1, 0))
                self.assertRegex(rec["first_fired_at"], r"^2026-11-03T09:35:1\d-05:00$", "the broker's clock, in New York time")
                self.assertEqual(rec["last_fired_at"], rec["first_fired_at"])
                (one,) = rec["legs"]
                self.assertEqual((one["symbol"], one["side"], one["qty"]), (l["occ"], l["expect"][0], l["qty"]),
                                 "the side is the one the close takes")
                self.assertEqual((one["bid"], one["ask"]), (l["snap"]["bid"], l["snap"]["ask"]),
                                 "the snapshot the rule was judged on")
                self.assertEqual(one["mid"], ofl.mid_of(l["snap"]["bid"], l["snap"]["ask"]))
                self.assertEqual(one["mark"], abs(float(l["pos"]["current_price"])))
                self.assertLessEqual(one["quote_age_s"], 60.0)
        xlf = recs[legs["loss_limit"]["occ"]]["legs"][0]
        self.assertEqual((xlf["bid"], xlf["ask"], xlf["mid"]), (0.0, 5.0, None), "a one-sided quote is kept and has no mid")
        self.assertEqual(len(broker.snapshot_reads), len(legs), "a single leg's quote is not fetched a second time")

    def test_a_later_scan_in_the_window_counts_and_never_replaces_the_first_quote(self):
        l = book()["bought_loss_limit"]
        self.load([l])
        self.run_manager(Broker([l], IN_WINDOW))
        first = self.records()[l["occ"]]
        self.restart()
        self.run_manager(Broker([requoted(l, 1.20, 1.50, cur=1.30)], LATER_IN_WINDOW))
        rec = self.records()[l["occ"]]
        self.assertEqual(rec["legs"], first["legs"], "the first quote stands")
        self.assertEqual((rec["first_fired_at"], rec["fires"]), (first["first_fired_at"], 2))
        self.assertRegex(rec["last_fired_at"], r"^2026-11-03T09:41:0\d-05:00$")

    def test_a_second_rule_on_the_same_contract_is_listed_on_the_one_record(self):
        l = book()["profit_target"]                      # sold at 3.00, now 1.20: 60% of max profit
        self.load([l])
        self.run_manager(Broker([l], IN_WINDOW))
        self.restart()
        # the opening quote turns: cost to close 6.50 is 2.2x the credit, so the loss limit fires
        self.run_manager(Broker([requoted(l, 6.40, 6.60, cur=6.50)], LATER_IN_WINDOW))
        recs = self.records()
        self.assertEqual(list(recs), [l["occ"]], "one exit: had the first close been sent the second rule could not fire")
        rec = recs[l["occ"]]
        self.assertEqual((rec["rule"], rec["rules"], rec["fires"]), ("profit_target", ["profit_target", "loss_limit"], 2))
        self.assertEqual((rec["legs"][0]["bid"], rec["legs"][0]["ask"]), (1.10, 1.30))

    def test_only_a_deferral_inside_the_window_is_recorded(self):
        legs = book()
        self.load(legs.values())
        self.run_manager(Broker(legs.values(), "2026-11-03T07:31:00-05:00", is_open=False))
        self.assertEqual(self.records(), {}, "a market-closed deferral is not held by the window")
        self.restart()
        self.run_manager(Broker(legs.values(), IN_WINDOW))
        self.assertEqual(len(self.records()), len(legs), "the same rules at 09:35 are")
        os.remove(om._opening_first_fired_path())
        self.restart()
        broker = Broker(legs.values(), AFTER_WINDOW)
        self.run_manager(broker)
        self.assertEqual(self.records(), {}, "nothing is held after 09:45")
        self.assertEqual(len(broker.posts), len(legs), "every close goes out")
        self.assertTrue(all("first_fired" not in r for r in self.rows()))

    def test_a_held_group_records_every_legs_quote_once_and_leaves_the_ledger_memo_alone(self):
        legs = condor()
        self.load(legs)
        ofl.note_quote(legs[0]["occ"], 9.90, 10.10)          # a quote some order path noted a moment ago
        broker = Broker(legs, IN_WINDOW)
        result = self.run_manager(broker)
        (act,) = [a for a in result["actions"] if a["action"] == "CLOSE_STRATEGY"]
        self.assertEqual((act["type"], act["order"]["status"]), ("strategy_dte_exit", "deferred"))
        key = ",".join(sorted(l["occ"] for l in legs))
        rec = self.records()[key]
        self.assertEqual((rec["kind"], rec["rule"], rec["fires"]), ("group", "strategy_dte_exit", 1))
        by_occ = {x["symbol"]: x for x in rec["legs"]}
        for l in legs:
            with self.subTest(leg=l["occ"]):
                x = by_occ[l["occ"]]
                self.assertEqual(x["side"], "sell" if l["pos"]["side"] == "long" else "buy")
                self.assertEqual((x["qty"], x["bid"], x["ask"]), (2, l["snap"]["bid"], l["snap"]["ask"]))
                self.assertEqual(x["mid"], ofl.mid_of(l["snap"]["bid"], l["snap"]["ask"]))
        self.assertEqual(sorted(broker.snapshot_reads), sorted(l["occ"] for l in legs), "each leg read once")
        memo = ofl.quote_for(legs[0]["occ"])
        self.assertEqual((memo["bid"], memo["ask"]), (9.90, 10.10), "what the memo held before is put back")
        for l in legs[1:]:
            self.assertIsNone(ofl.quote_for(l["occ"])["bid"], "the record's read is not left in the ledger's memo")
        # the pursued close is held again on the next scan: counted, no second read
        self.restart()
        again = Broker(legs, LATER_IN_WINDOW)
        self.run_manager(again)
        self.assertEqual(again.snapshot_reads, [])
        self.assertEqual(self.records()[key]["fires"], 2)
        self.assertEqual(self.records()[key]["legs"], rec["legs"])

    def test_the_read_budget_stops_further_group_quotes_and_the_record_is_still_written(self):
        legs = condor()
        self.load(legs)
        broker = Broker(legs, IN_WINDOW)
        with patch.object(om, "OPENING_FIRST_FIRED_READ_BUDGET_S", -1.0):
            self.run_manager(broker)
        rec = self.records()[",".join(sorted(l["occ"] for l in legs))]
        self.assertEqual(broker.snapshot_reads, [])
        self.assertEqual([(x["bid"], x["ask"], x["mid"]) for x in rec["legs"]], [(None, None, None)] * 4)
        self.assertEqual([x["side"] for x in rec["legs"]].count("buy"), 2)

    def test_days_and_records_are_bounded(self):
        def held(occ):
            return {"action": "CLOSE", "occ_symbol": occ, "type": "profit_target",
                    "order": {"status": "deferred", "detail": om.OPENING_WINDOW_DEFERRED_DETAIL}}
        total = om.OPENING_FIRST_FIRED_KEEP_DAYS + 5
        for i in range(total):
            when = datetime(2026, 9, 1, 9, 35, tzinfo=ET) + timedelta(days=i)
            self.assertEqual(om._note_opening_first_fired([held("X261127P00010000")], [], now_et=when), 1)
        with open(om._opening_first_fired_path()) as f:
            days = sorted(json.load(f)["days"])
        self.assertEqual(len(days), om.OPENING_FIRST_FIRED_KEEP_DAYS)
        self.assertEqual(days[-1], "2026-10-05")
        with patch.object(om, "OPENING_FIRST_FIRED_MAX_PER_DAY", 3):
            when = datetime(2026, 10, 6, 9, 35, tzinfo=ET)
            added = om._note_opening_first_fired([held(f"Y{i}261127P00010000") for i in range(5)], [], now_et=when)
        self.assertEqual(added, 3)
        self.assertEqual(len(self.records("2026-10-06")), 3)

    def test_the_file_is_storage_configs_in_the_managers_data_dir(self):
        name = os.path.basename(storage_config.OPTION_OPENING_FIRST_FIRED_PATH)
        self.assertEqual(name, "voltrade_option_opening_first_fired.json")
        self.assertEqual(os.path.dirname(storage_config.OPTION_OPENING_FIRST_FIRED_PATH), storage_config.DATA_DIR)
        self.assertEqual(om._opening_first_fired_path(), os.path.join(self.dir, name))
        ts = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", "openingCounterfactual.ts")).read()
        self.assertIn(f'export const OPENING_FIRST_FIRED_FILENAME = "{name}";', ts, "the daily line reads the same file")


class TestTheLinkOnTheLedgerRow(_Case):
    def test_the_later_close_carries_the_record_across_a_process_restart(self):
        legs = book()
        self.load(legs.values())
        self.run_manager(Broker(legs.values(), IN_WINDOW))
        recs = self.records()
        self.restart()                                   # the daemon is recycled inside the window
        later = {rule: requoted(l, round(l["snap"]["bid"] + 0.02, 2), round(l["snap"]["ask"] + 0.02, 2))
                 for rule, l in legs.items()}
        broker = Broker(later.values(), AFTER_WINDOW)
        self.run_manager(broker)
        self.assertEqual(len(broker.posts), len(legs))
        rows = {r["symbol"]: r for r in self.rows()}
        self.assertEqual(len(rows), len(legs))
        self.assertEqual(sorted(recs), sorted(rows), "a record for every close that was held")
        for rule, l in legs.items():
            with self.subTest(rule=rule):
                rec, row = recs[l["occ"]], rows[l["occ"]]
                first = rec["legs"][0]
                self.assertEqual(row["first_fired"], {"key": l["occ"], "at": rec["first_fired_at"], "rule": rule,
                                                      "kind": "single_leg", "bid": first["bid"], "ask": first["ask"],
                                                      "mid": first["mid"]})
                self.assertEqual(set(row) - {"first_fired"}, OLD_ROW_KEYS, "nothing else about the row changed")
                self.assertEqual((row["side"], row["qty"]), (l["expect"][0], float(l["qty"])))
                self.assertEqual((row["bid"], row["ask"]), (later[rule]["snap"]["bid"], later[rule]["snap"]["ask"]),
                                 "the row's own quote is still the one at submission")
        # the POST itself is the one the rule always sent (the XLF loss limit: buy back at the ask)
        xlf = later["loss_limit"]
        self.assertIn({"symbol": xlf["occ"], "qty": "1", "side": "buy", "type": "limit", "limit_price": "5.02",
                       "time_in_force": "day"}, broker.posts)

    def test_a_contract_the_window_never_held_writes_the_row_it_always_did(self):
        legs = book()
        held, other = legs["gamma_risk"], legs["profit_target"]
        self.load([held])
        self.run_manager(Broker([held], IN_WINDOW))
        self.restart()
        self.load([held, other])
        self.run_manager(Broker([held, other], AFTER_WINDOW))
        rows = {r["symbol"]: r for r in self.rows()}
        self.assertIn("first_fired", rows[held["occ"]])
        self.assertEqual(set(rows[other["occ"]]), OLD_ROW_KEYS)

    def test_yesterdays_record_is_not_carried_by_todays_close(self):
        l = book()["gamma_risk"]
        self.load([l])
        self.run_manager(Broker([l], IN_WINDOW))
        self.restart()
        self.run_manager(Broker([l], "2026-11-04T10:00:00-05:00"))
        (row,) = self.rows()
        self.assertNotIn("first_fired", row)
        self.assertEqual(list(self.records("2026-11-04")), [])
        self.assertEqual(list(self.records("2026-11-03")), [l["occ"]], "the 11-03 record stays what it was: never sent")

    def test_an_exit_the_rule_drops_after_the_window_leaves_its_record_and_no_row(self):
        xlf = book()["loss_limit"]                       # 0 / 5.00 at the open: "6.9x the credit"
        self.load([xlf])
        self.run_manager(Broker([xlf], IN_WINDOW))
        self.restart()
        broker = Broker([requoted(xlf, 0.47, 0.51, cur=0.49)], AFTER_WINDOW)
        result = self.run_manager(broker)
        self.assertEqual([a for a in result["actions"] if a["action"] == "CLOSE"], [], "at fair value the rule does not fire")
        self.assertEqual((broker.posts, self.rows()), ([], []))
        rec = self.records()[xlf["occ"]]
        self.assertEqual((rec["rule"], rec["fires"], rec["refused"]), ("loss_limit", 1, 0))

    def test_a_close_the_broker_refuses_is_counted_on_the_record(self):
        l = book()["bought_loss_limit"]
        self.load([l])
        self.run_manager(Broker([l], IN_WINDOW))
        self.restart()
        broker = Broker([l], AFTER_WINDOW, reject=[l["occ"]])
        result = self.run_manager(broker)
        (act,) = [a for a in result["actions"] if a["action"] == "CLOSE"]
        self.assertEqual(act["order"]["status"], "error")
        self.assertEqual(self.rows(), [], "a refused order writes no row")
        rec = self.records()[l["occ"]]
        self.assertEqual(rec["refused"], 1)
        self.assertRegex(rec["last_refused_at"], r"^2026-11-03T09:46:0\d-05:00$")
        self.assertEqual(rec["fires"], 1, "a refusal is not a fire")

    def test_the_exit_ladders_first_order_and_its_replace_both_carry_the_record(self):
        l = book()["bought_loss_limit"]                  # sells at the bid: the ladder starts at the mid
        self.load([l])
        self.run_manager(Broker([l], IN_WINDOW))
        rec = self.records()[l["occ"]]
        self.restart()
        broker = Broker([l], AFTER_WINDOW)
        with patch.object(om.order_ladder, "settings", return_value=dict(LADDER_ON)):
            self.run_manager(broker)
            self.assertEqual([p["limit_price"] for p in broker.posts], ["1.8"], "step 1 is the mid of 1.60 / 2.00")
            replaced = _resp(200, {"id": "ord-2", "symbol": l["occ"], "side": "sell", "qty": "2"})
            with patch.object(om.requests, "patch", MagicMock(return_value=replaced)):
                om._replace_order_price("ord-1", 1.70)
        rows = self.rows()
        self.assertEqual([(r["order_id"], r["limit"], r["source"]) for r in rows],
                         [("ord-1", 1.8, "options_manager._submit_close_order"),
                          ("ord-2", 1.7, "options_manager:exit ladder replace")])
        for r in rows:
            self.assertEqual((r["first_fired"]["key"], r["first_fired"]["at"], r["first_fired"]["mid"]),
                             (l["occ"], rec["first_fired_at"], 1.8))

    def test_a_group_close_after_the_window_links_each_leg_to_its_own_first_quote(self):
        legs = condor()
        self.load(legs)
        self.run_manager(Broker(legs, IN_WINDOW))
        key = ",".join(sorted(l["occ"] for l in legs))
        rec = self.records()[key]
        self.restart()
        broker = Broker(legs, AFTER_WINDOW)
        result = self.run_manager(broker)
        (act,) = [a for a in result["actions"] if a["action"] == "CLOSE_STRATEGY"]
        self.assertEqual((act["order"]["status"], act["order"]["method"]), ("submitted", "mleg"))
        (mleg,) = broker.posts
        self.assertEqual((mleg["order_class"], mleg["type"], mleg["qty"], len(mleg["legs"])), ("mleg", "market", "2", 4),
                         "the group close itself is unchanged")
        rows = self.rows()
        self.assertEqual(len(rows), 4)
        first = {x["symbol"]: x for x in rec["legs"]}
        for r in rows:
            with self.subTest(leg=r["symbol"]):
                f = first[r["symbol"]]
                self.assertEqual(r["first_fired"], {"key": key, "at": rec["first_fired_at"], "rule": "strategy_dte_exit",
                                                    "kind": "group", "bid": f["bid"], "ask": f["ask"], "mid": f["mid"]})
                self.assertEqual((r["order_id"], r["order_class"], r["side"]), ("ord-1", "mleg", f["side"]))
                self.assertIsNone(r["mid"], "a market mleg's rows have no mid, as before: the record's quote read "
                                            "is not the quote at submission")

    def test_a_failed_lookup_never_changes_the_order_or_its_result(self):
        occ = "XLF261127P00055500"
        path = os.path.join(self.dir, "voltrade_option_opening_first_fired.json")
        posts = []

        def close():
            post = MagicMock(return_value=_resp(200, {"id": f"close-{len(posts) + 1}"}))
            with patch.object(om.requests, "post", post):
                res = om._submit_close_order(occ, 1, "buy", 0.51)
            posts.append(post.call_args.kwargs["json"])
            return res

        baseline = close()                               # no file at all
        with open(path, "w") as f:
            f.write("{not json")
        corrupt = close()
        with open(path, "w") as f:
            json.dump({"version": 1, "days": {"2026-11-03": ["not", "records"], "x": 3}}, f)
        odd = close()
        with patch.object(om, "_load_opening_first_fired", side_effect=RuntimeError("disk gone")):
            raising = close()
            self.assertIsNone(om._opening_first_fired_link(occ))
            self.assertEqual(om._opening_first_fired_links([occ, None]), {})
        for res in (corrupt, odd, raising):
            self.assertEqual((res["status"], res["detail"]), (baseline["status"], baseline["detail"]))
        self.assertEqual(posts, [posts[0]] * 4, "the same POST every time")
        self.assertEqual([set(r) for r in self.rows()], [OLD_ROW_KEYS] * 4, "the rows are written, without a record")

    def test_a_failed_lookup_never_sends_an_accepted_group_close_to_its_fallback(self):
        legs = [{"symbol": "IREN261204C00024000", "qty": "-2", "side": "short", "current_price": "0.30"},
                {"symbol": "IREN261204C00026000", "qty": "2", "side": "long", "current_price": "0.10"}]
        post = MagicMock(return_value=_resp(200, {"id": "mleg-1"}))
        with patch.object(om, "_load_opening_first_fired", side_effect=RuntimeError("disk gone")), \
                patch.object(om.requests, "post", post):
            res = om._close_strategy_mleg(legs, "IREN")
        self.assertEqual((res["status"], res["method"], post.call_count), ("submitted", "mleg", 1))
        self.assertEqual([set(r) - {"order_class"} for r in self.rows()], [OLD_ROW_KEYS] * 2)
        replaced = _resp(200, {"id": "lad-2", "symbol": "IREN261204C00024000", "side": "buy", "qty": "2"})
        with patch.object(om, "_load_opening_first_fired", side_effect=RuntimeError("disk gone")), \
                patch.object(om.requests, "patch", MagicMock(return_value=replaced)):
            order, detail = om._replace_order_price("lad-1", 0.31)
        self.assertEqual((order["id"], detail), ("lad-2", "replaced"))


class TestNoDecisionChanges(_Case):
    def test_the_actions_and_orders_are_the_same_with_and_without_the_record(self):
        legs = {**book(), **{f"condor{i}": l for i, l in enumerate(condor())}}
        runs = {}
        for name, recorder in (("with", contextlib.nullcontext()),
                               ("without", patch.object(om, "_note_opening_first_fired", MagicMock(return_value=0)))):
            self.restart()
            for f in os.listdir(self.dir):
                os.remove(os.path.join(self.dir, f))
            self.load(legs.values())
            with recorder:
                held = self.run_manager(Broker(legs.values(), IN_WINDOW))["actions"]
                self.restart()
                after = Broker(legs.values(), AFTER_WINDOW)
                sent = self.run_manager(after)["actions"]
            runs[name] = {"held": held, "sent": sent, "posts": after.posts,
                          "recorded": os.path.exists(os.path.join(self.dir, "voltrade_option_opening_first_fired.json"))}
        self.assertEqual((runs["with"]["recorded"], runs["without"]["recorded"]), (True, False))
        self.assertEqual(runs["with"]["held"], runs["without"]["held"], "the same deferrals inside the window")
        self.assertEqual(runs["with"]["sent"], runs["without"]["sent"], "the same closes after it")
        self.assertEqual(runs["with"]["posts"], runs["without"]["posts"], "the same order POSTs")
        self.assertEqual(len(runs["with"]["held"]), len(book()) + 1, "eight single legs and one group held")
        self.assertEqual(len(runs["with"]["posts"]), len(book()) + 1, "eight single-leg closes and one mleg")

    def test_a_corrupt_or_unwritable_file_costs_the_record_only(self):
        legs = book()
        self.load(legs.values())
        path = os.path.join(self.dir, "voltrade_option_opening_first_fired.json")
        with open(path, "w") as f:
            f.write("{not json")
        result = self.run_manager(Broker(legs.values(), IN_WINDOW))
        self.assertEqual(sum(1 for a in result["actions"] if a.get("order", {}).get("status") == "deferred"), len(legs))
        self.assertEqual(len(self.records()), len(legs), "an unreadable file is replaced by a new one")
        self.assertEqual([f for f in os.listdir(self.dir) if f.endswith(".tmp")], [], "write-then-rename leaves no temp file")
        # an unwritable directory: the run is the same run, the record is lost
        blocker = os.path.join(self.dir, "not-a-dir")
        open(blocker, "w").close()
        held = {"action": "CLOSE", "occ_symbol": "X261127P00010000", "type": "profit_target",
                "order": {"status": "deferred", "detail": om.OPENING_WINDOW_DEFERRED_DETAIL}}
        with patch.object(om, "DATA_DIR", blocker):
            self.assertEqual(om._note_opening_first_fired([held], []), 0)
            self.assertIsNone(om._opening_first_fired_link("X261127P00010000"))
        for junk in (None, [], [None, 3, "x", {"action": "CLOSE"}, {"action": "CLOSE_STRATEGY", "legs": None, "order": 5}]):
            self.assertEqual(om._note_opening_first_fired(junk, None), 0)

    def test_every_close_path_passes_the_record_only_after_its_order_was_accepted(self):
        for fn, send in ((om._submit_close_order, "requests.post("), (om._replace_order_price, "requests.patch("),
                         (om._close_strategy_mleg, "requests.post(")):
            src = inspect.getsource(fn)
            with self.subTest(fn=fn.__name__):
                self.assertEqual(src.count("first_fired=_opening_first_fired_link"), 1)
                self.assertLess(src.index(send), src.index("first_fired=_opening_first_fired_link"))
        body = inspect.getsource(om.manage_options_positions)
        self.assertLess(body.index("_save_options_state(state)\n    # 2026-11-05"), body.index("_note_opening_first_fired(actions, options_positions)"),
                        "the record is written after every decision of the run is made and saved")
        self.assertEqual(body.count("_note_opening_first_fired("), 1)
        for fn in (om._note_opening_first_fired, om._opening_first_fired_links):
            self.assertIn("except Exception as e:", inspect.getsource(fn), "neither can raise into the manager")


class TestLedgerAdditions(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        p = patch.object(ofl, "DATA_DIR", self.dir)
        p.start()
        self.addCleanup(p.stop)
        ofl._quotes.clear()
        self.addCleanup(ofl._quotes.clear)

    def _rows(self):
        with open(os.path.join(self.dir, ofl.LEDGER_FILENAME)) as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_first_fired_is_written_only_when_a_record_is_passed(self):
        link = {"key": "A261218C00010000", "at": "2026-11-03T09:35:12-05:00", "rule": "gamma_risk", "kind": "single_leg",
                "bid": 1.0, "ask": 1.2, "mid": 1.1}
        self.assertTrue(ofl.record_order("o-1", "A261218C00010000", "sell", 1, "unit", limit=1.0, first_fired=link))
        for i, nothing in enumerate((None, {}, "x", 0, [link])):
            self.assertTrue(ofl.record_order(f"n-{i}", "A261218C00010000", "sell", 1, "unit", limit=1.0, first_fired=nothing))
        rows = self._rows()
        self.assertEqual(rows[0]["first_fired"], link)
        self.assertEqual([set(r) for r in rows[1:]], [OLD_ROW_KEYS] * 5)
        link["mid"] = 99
        self.assertEqual(self._rows()[0]["first_fired"]["mid"], 1.1)

    def test_an_mleg_order_gives_each_leg_its_own_record(self):
        legs = [{"symbol": "A261218C00010000", "side": "sell", "ratio_qty": "1"},
                {"symbol": "A261218C00012000", "side": "buy", "ratio_qty": "1"}]
        links = {"A261218C00010000": {"key": "k", "at": "t", "mid": 1.1}}
        self.assertEqual(ofl.record_mleg_order("m-1", legs, 2, "unit", first_fired=links), 2)
        self.assertEqual(ofl.record_mleg_order("m-2", legs, 2, "unit"), 2)
        self.assertEqual(ofl.record_mleg_order("m-3", legs, 2, "unit", first_fired="not a map"), 2)
        rows = self._rows()
        self.assertEqual(rows[0]["first_fired"], {"key": "k", "at": "t", "mid": 1.1})
        self.assertTrue(all("first_fired" not in r for r in rows[1:]))

    def test_a_rows_quote_age_limit_is_unchanged_and_a_longer_one_is_explicit(self):
        ofl.note_quote("B261218C00010000", 1.0, 1.2, now=1000.0)
        late = 1000.0 + ofl.QUOTE_MAX_AGE_S + 30
        self.assertIsNone(ofl.quote_for("B261218C00010000", now=late)["mid"], "a ledger row's limit is still 120 s")
        got = ofl.quote_for("B261218C00010000", now=late, max_age_s=600)
        self.assertEqual((got["bid"], got["ask"], got["mid"], got["quote_age_s"]), (1.0, 1.2, 1.1, 150.0))
        self.assertIsNone(ofl.quote_for("B261218C00010000", now=1000.0 + 601, max_age_s=600)["mid"])

    def test_quotes_read_inside_the_block_are_not_kept(self):
        now = time.time()
        ofl.note_quote("KEPT261218C00010000", 2.0, 2.2, now=now)
        with ofl.quotes_not_kept(["KEPT261218C00010000", "NEW261218C00010000", "", None]):
            ofl.forget_quote("KEPT261218C00010000")
            ofl.note_quote("KEPT261218C00010000", 7.0, 7.2)
            ofl.note_quote("NEW261218C00010000", 3.0, 3.2)
            ofl.note_quote("OTHER261218C00010000", 4.0, 4.2)
            self.assertEqual(ofl.quote_for("NEW261218C00010000")["mid"], 3.1, "readable inside the block")
        self.assertEqual(ofl.quote_for("KEPT261218C00010000")["bid"], 2.0)
        self.assertIsNone(ofl.quote_for("NEW261218C00010000")["bid"])
        self.assertEqual(ofl.quote_for("OTHER261218C00010000")["bid"], 4.0, "a contract the block was not given is untouched")
        with self.assertRaises(ValueError):
            with ofl.quotes_not_kept(["NEW261218C00010000"]):
                ofl.note_quote("NEW261218C00010000", 3.0, 3.2)
                raise ValueError("the read failed")
        self.assertIsNone(ofl.quote_for("NEW261218C00010000")["bid"], "restored on an error too")


if __name__ == "__main__":
    unittest.main()
