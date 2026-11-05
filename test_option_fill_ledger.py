#!/usr/bin/env python3
"""
DAILY LINE (2026-11-05), the Python half — measurement only.

(1) The option FILL LEDGER (option_fill_ledger.py): one row per option order
    the broker accepted, written AFTER the order POST returned, carrying the
    quote the order was priced from — so the daily line can split option P&L
    into fills/slippage (fill - mid at submission) and direction. A path with
    no quote writes a null mid (the daily line counts it unattributed, never
    guesses one). The order POSTs themselves are pinned unchanged.
(2) The OPENING-WINDOW BLOCK COUNT (options_manager): every option exit the
    09:30-09:45 window held back, once per (contract, rule) per broker-ET day.

Mocked: no network (requests.get/post stand-ins); DATA_DIR in a temp dir.
Run: python3 -m pytest -q test_option_fill_ledger.py
"""
import inspect
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import option_fill_ledger as ofl  # noqa: E402
import options_execution as oe  # noqa: E402
import options_manager as om  # noqa: E402
from test_options_exit_market_hours import Broker, _ManagerCase, book  # noqa: E402


def _rows(data_dir, suffix=""):
    path = os.path.join(data_dir, ofl.LEDGER_FILENAME + suffix)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class _LedgerDir(unittest.TestCase):
    """DATA_DIR in a temp dir, an empty quote memo, a cold broker clock."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        p = patch.object(ofl, "DATA_DIR", self.dir)
        p.start()
        self.addCleanup(p.stop)
        ofl._quotes.clear()
        self.addCleanup(ofl._quotes.clear)


class TestLedgerRows(_LedgerDir):
    def test_a_row_carries_the_noted_quote_its_mid_and_the_order_identity(self):
        ofl.note_quote("XLF261127P00055500", 0.47, 0.51, now=1000.0)
        self.assertTrue(ofl.record_order("ord-1", "XLF261127P00055500", "BUY", 2, "unit", limit=0.51, now=1030.0))
        (row,) = _rows(self.dir)
        self.assertEqual((row["order_id"], row["symbol"], row["side"], row["qty"]), ("ord-1", "XLF261127P00055500", "buy", 2.0))
        self.assertEqual((row["bid"], row["ask"], row["mid"], row["limit"]), (0.47, 0.51, 0.49, 0.51))
        self.assertEqual((row["source"], row["quote_age_s"]), ("unit", 30.0))
        self.assertIn("ts", row)

    def test_no_stale_one_sided_or_crossed_quote_becomes_a_mid(self):
        ofl.record_order("o-none", "AAA261218C00010000", "sell", 1, "unit", now=1000.0)
        ofl.note_quote("BBB261218C00010000", 1.0, 1.2, now=1000.0)
        ofl.record_order("o-stale", "BBB261218C00010000", "sell", 1, "unit", now=1000.0 + ofl.QUOTE_MAX_AGE_S + 1)
        ofl.note_quote("CCC261218P00020000", 0.0, 5.00, now=1000.0)          # the live XLF pre-market 0/5.00
        ofl.record_order("o-onesided", "CCC261218P00020000", "buy", 1, "unit", now=1001.0)
        ofl.note_quote("DDD261218P00020000", 1.10, 1.00, now=1000.0)
        ofl.record_order("o-crossed", "DDD261218P00020000", "buy", 1, "unit", now=1001.0)
        rows = {r["order_id"]: r for r in _rows(self.dir)}
        self.assertEqual({k: rows[k]["mid"] for k in rows}, {"o-none": None, "o-stale": None, "o-onesided": None, "o-crossed": None})
        self.assertEqual((rows["o-none"]["bid"], rows["o-stale"]["bid"]), (None, None))
        self.assertEqual((rows["o-onesided"]["bid"], rows["o-onesided"]["ask"]), (0.0, 5.0), "the quote is kept, only the mid is refused")

    def test_a_failed_fresh_read_leaves_no_older_quote_behind(self):
        ofl.note_quote("EEE261218C00030000", 2.0, 2.2, now=time.time())
        ofl.forget_quote("EEE261218C00030000")
        self.assertIsNone(ofl.quote_for("EEE261218C00030000")["mid"])

    def test_no_order_id_no_row(self):
        self.assertFalse(ofl.record_order("", "FFF261218C00030000", "buy", 1, "unit"))
        self.assertEqual(_rows(self.dir), [])

    def test_mleg_rows_one_per_leg_under_the_parent_id(self):
        legs = [{"symbol": "IREN261204C00024000", "side": "sell", "ratio_qty": "1"},
                {"symbol": "IREN261204C00026000", "side": "buy", "ratio_qty": 2}]
        ofl.note_quote("IREN261204C00024000", 1.00, 1.10, now=1000.0)
        self.assertEqual(ofl.record_mleg_order("parent-9", legs, 3, "unit", now=1000.0), 2)
        rows = _rows(self.dir)
        self.assertEqual([(r["order_id"], r["symbol"], r["side"], r["qty"], r["order_class"]) for r in rows],
                         [("parent-9", "IREN261204C00024000", "sell", 3.0, "mleg"),
                          ("parent-9", "IREN261204C00026000", "buy", 6.0, "mleg")])
        self.assertEqual([r["mid"] for r in rows], [1.05, None], "a leg without a noted quote is null, not guessed")

    def test_rotation_keeps_one_old_generation(self):
        with patch.object(ofl, "LEDGER_MAX_BYTES", 300):
            for i in range(8):
                ofl.record_order(f"o-{i}", "GGG261218C00030000", "buy", 1, "unit")
        self.assertTrue(os.path.exists(os.path.join(self.dir, ofl.LEDGER_FILENAME + ".1")))
        ids = [r["order_id"] for r in _rows(self.dir, ".1") + _rows(self.dir)]
        self.assertEqual(ids[-1], "o-7")
        self.assertLess(os.path.getsize(os.path.join(self.dir, ofl.LEDGER_FILENAME)), 300 + 400)

    def test_an_unwritable_dir_returns_false_and_never_raises(self):
        blocker = os.path.join(self.dir, "not-a-dir")
        open(blocker, "w").close()
        with patch.object(ofl, "DATA_DIR", blocker):
            self.assertFalse(ofl.record_order("o-x", "HHH261218C00030000", "buy", 1, "unit"))
            self.assertEqual(ofl.record_mleg_order("o-y", [{"symbol": "HHH261218C00030000", "side": "buy"}], 1, "unit"), 0)

    def test_the_quote_memo_stays_bounded(self):
        with patch.object(ofl, "QUOTE_MEMO_MAX", 5):
            for i in range(12):
                ofl.note_quote(f"M{i:02d}261218C00010000", 1, 1.1, now=1000.0 + i * 1000)
            self.assertLessEqual(len(ofl._quotes), 5)
            self.assertIn("M11261218C00010000", ofl._quotes, "the newest quote is always kept")


def _resp(status, body):
    r = MagicMock(status_code=status, text=json.dumps(body))
    r.json.return_value = body
    r.headers = {"content-type": "application/json"}
    return r


class TestOrderPathsWriteAfterThePost(_LedgerDir):
    """The real order functions, network mocked: the POST body is exactly what
    it was, and the ledger row lands only once the broker accepted it."""

    OCC = "XLF261127P00055500"

    def _snapshot_get(self, bid, ask):
        body = {"snapshots": {self.OCC: {"latestQuote": {"bp": bid, "ap": ask}, "greeks": {"delta": -0.2}}}}
        return MagicMock(return_value=_resp(200, body))

    def test_a_single_leg_close_records_the_snapshot_quote_it_was_priced_from(self):
        with patch.object(om.requests, "get", self._snapshot_get(0.47, 0.51)), \
                patch.object(om.alpaca_feed, "options_feed", return_value="indicative"):
            snap = om._get_option_snapshot(self.OCC)
        post = MagicMock(return_value=_resp(200, {"id": "close-1"}))
        with patch.object(om.requests, "post", post):
            res = om._submit_close_order(self.OCC, 1, "buy", snap["ask"])
        self.assertEqual(res["status"], "submitted")
        self.assertEqual(post.call_args.kwargs["json"], {"symbol": self.OCC, "qty": "1", "side": "buy", "type": "limit",
                                                         "limit_price": "0.51", "time_in_force": "day"},
                         "the close order itself is unchanged")
        (row,) = _rows(self.dir)
        self.assertEqual((row["order_id"], row["symbol"], row["side"], row["qty"]), ("close-1", self.OCC, "buy", 1.0))
        self.assertEqual((row["bid"], row["ask"], row["mid"], row["limit"]), (0.47, 0.51, 0.49, 0.51))
        self.assertEqual(row["source"], "options_manager._submit_close_order")

    def test_a_rejected_close_writes_no_row(self):
        with patch.object(om.requests, "post", MagicMock(return_value=_resp(422, {"message": "no"}))):
            res = om._submit_close_order(self.OCC, 1, "buy", 0.51)
        self.assertEqual(res["status"], "error")
        self.assertEqual(_rows(self.dir), [])

    def test_a_failed_snapshot_read_leaves_the_close_unattributed(self):
        ofl.note_quote(self.OCC, 0.40, 0.44)                      # an older read
        with patch.object(om.requests, "get", MagicMock(side_effect=ConnectionError("down"))), \
                patch.object(om.alpaca_feed, "options_feed", return_value="indicative"):
            self.assertEqual(om._get_option_snapshot(self.OCC), {})
        with patch.object(om.requests, "post", MagicMock(return_value=_resp(200, {"id": "close-2"}))):
            om._submit_close_order(self.OCC, 1, "buy", 0.60)
        self.assertIsNone(_rows(self.dir)[0]["mid"])

    def test_a_ledger_failure_never_changes_the_order_result(self):
        with patch.object(ofl, "append_row", side_effect=RuntimeError("disk gone")), \
                patch.object(om.requests, "post", MagicMock(return_value=_resp(200, {"id": "close-3"}))):
            res = om._submit_close_order(self.OCC, 1, "buy", 0.51)
        self.assertEqual((res["status"], res["order_id"]), ("submitted", "close-3"))

    def test_the_mleg_close_writes_one_row_per_leg_and_a_ledger_failure_never_triggers_the_fallback(self):
        legs = [{"symbol": "IREN261204C00024000", "qty": "-2", "side": "short", "current_price": "0.30"},
                {"symbol": "IREN261204C00026000", "qty": "2", "side": "long", "current_price": "0.10"}]
        post = MagicMock(return_value=_resp(200, {"id": "mleg-1"}))
        with patch.object(om.requests, "post", post):
            res = om._close_strategy_mleg(legs, "IREN")
        self.assertEqual((res["status"], res["method"]), ("submitted", "mleg"))
        rows = _rows(self.dir)
        self.assertEqual([(r["order_id"], r["symbol"], r["side"], r["qty"], r["mid"]) for r in rows],
                         [("mleg-1", "IREN261204C00024000", "buy", 2.0, None),
                          ("mleg-1", "IREN261204C00026000", "sell", 2.0, None)])
        post2 = MagicMock(return_value=_resp(200, {"id": "mleg-2"}))
        with patch.object(ofl, "append_row", side_effect=RuntimeError("disk gone")), patch.object(om.requests, "post", post2):
            res2 = om._close_strategy_mleg(legs, "IREN")
        self.assertEqual((res2["status"], res2["method"]), ("submitted", "mleg"))
        self.assertEqual(post2.call_count, 1, "an accepted mleg close is never followed by fallback leg closes")

    def test_a_single_leg_entry_records_the_chain_quote_it_was_priced_from(self):
        chain = {"snapshots": {
            "SPY261218P00500000": {"latestQuote": {"bp": 4.10, "ap": 4.30, "bs": 10, "as": 10}, "greeks": {"delta": -0.3}},
            "SPY261218P00490000": {"latestQuote": {"bp": 3.00, "ap": 3.20}, "greeks": {"delta": -0.2}},
        }}
        with patch.object(oe.requests, "get", MagicMock(return_value=_resp(200, chain))), \
                patch.object(oe.alpaca_feed, "options_feed", return_value="indicative"):
            contracts = oe._fetch_option_chain("SPY", 520.0, option_type="put")
        self.assertEqual(len(contracts), 2)
        post = MagicMock(return_value=_resp(200, {"id": "entry-1"}))
        contract = {"strategy": "buy_put", "occ_symbol": "SPY261218P00500000", "side": "buy", "qty": 2, "limit_price": 4.26}
        with patch.object(oe.requests, "post", post):
            res = oe.submit_options_order(contract)
        self.assertEqual(res["status"], "submitted")
        self.assertEqual(post.call_args.kwargs["json"], {"symbol": "SPY261218P00500000", "qty": "2", "side": "buy",
                                                         "type": "limit", "limit_price": "4.26", "time_in_force": "day"},
                         "the FROZEN single-leg POST is unchanged")
        (row,) = _rows(self.dir)
        self.assertEqual((row["order_id"], row["side"], row["qty"], row["bid"], row["ask"], row["mid"], row["limit"]),
                         ("entry-1", "buy", 2.0, 4.10, 4.30, 4.2, 4.26))
        self.assertEqual(row["source"], "options_execution.submit_options_order")

    def test_an_mleg_entry_records_each_legs_chain_quote(self):
        oe_quotes = {"SPY261218C00560000": (1.00, 1.10), "SPY261218C00570000": (0.40, 0.50)}
        for sym, (b, a) in oe_quotes.items():
            ofl.note_quote(sym, b, a)
        post = MagicMock(return_value=_resp(200, {"id": "mleg-entry", "status": "pending_new"}))
        with patch.object(oe.requests, "post", post):
            res = oe._submit_mleg_order([{"symbol": "SPY261218C00560000", "side": "sell", "ratio_qty": 1},
                                         {"symbol": "SPY261218C00570000", "side": "buy", "ratio_qty": 1}],
                                        qty=1, limit_price=-0.55, label="bear call")
        self.assertEqual(res["order_id"], "mleg-entry")
        payload = post.call_args.kwargs["json"]
        self.assertEqual((payload["order_class"], payload["limit_price"], payload["type"]), ("mleg", "0.55", "limit"))
        rows = _rows(self.dir)
        self.assertEqual([(r["order_id"], r["symbol"], r["side"], r["mid"]) for r in rows],
                         [("mleg-entry", "SPY261218C00560000", "sell", 1.05), ("mleg-entry", "SPY261218C00570000", "buy", 0.45)])

    def test_a_rolls_new_leg_is_recorded_with_the_chain_quote_the_roll_priced_it_from(self):
        old = "KMI261120P00025000"
        new = "KMI261218P00025000"
        def get(url, **kw):
            if kw.get("params", {}).get("symbols") == old:
                return _resp(200, {"snapshots": {old: {"latestQuote": {"bp": 0.20, "ap": 0.24}, "greeks": {}}}})
            return _resp(200, {"snapshots": {new: {"latestQuote": {"bp": 0.60, "ap": 0.66}}}})
        ids = iter(["roll-close", "roll-open"])
        post = MagicMock(side_effect=lambda *a, **k: _resp(200, {"id": next(ids)}))
        parsed = om._parse_occ_symbol(old)
        with patch.object(om.requests, "get", side_effect=get), patch.object(om.requests, "post", post), \
                patch.object(om.alpaca_feed, "options_feed", return_value="indicative"):
            res = om._attempt_roll(old, 1, "sell", parsed, 100000)
        self.assertTrue(res["rolled"], res)
        rows = {r["order_id"]: r for r in _rows(self.dir)}
        self.assertEqual((rows["roll-close"]["symbol"], rows["roll-close"]["mid"]), (old, 0.22))
        self.assertEqual((rows["roll-open"]["symbol"], rows["roll-open"]["side"], rows["roll-open"]["mid"]), (new, "sell", 0.63))

    def test_every_covered_order_path_writes_after_its_post(self):
        for fn in (om._submit_close_order, om._close_strategy_mleg, oe.submit_options_order, oe._submit_mleg_order):
            src = inspect.getsource(fn)
            with self.subTest(fn=fn.__name__):
                self.assertIn("_fill_ledger.record_", src)
                self.assertLess(src.index("requests.post("), src.index("_fill_ledger.record_"),
                                "the ledger row is written only after the order POST returned")


class _WindowBroker(Broker):
    """Broker whose /v2/clock carries Alpaca's ET timestamp (the opening window reads it)."""

    def __init__(self, positions, stamp, is_open=True):
        super().__init__(positions, is_open=is_open)
        self.stamp = stamp

    def get(self, url, **kw):
        if "/v2/clock" in url:
            self.gets.append(url)
            return MagicMock(status_code=200, json=lambda: {"is_open": self.is_open, "timestamp": self.stamp})
        return super().get(url, **kw)


class TestOpeningWindowBlockCount(_ManagerCase):
    def setUp(self):
        super().setUp()
        saved = dict(om._CLOCK_CACHE)

        def restore():
            om._CLOCK_CACHE.clear()
            om._CLOCK_CACHE.update(saved)
            om._CLOCK_CACHE.update({"at": 0.0, "is_open": None, "et": None, "et_at": 0.0})
        self.addCleanup(restore)
        self.path = os.path.join(om.DATA_DIR, om.OPENING_DEFERRALS_FILE)

    def counted(self):
        if not os.path.exists(self.path):
            return {}
        with open(self.path) as f:
            return json.load(f)["days"]

    def test_every_exit_held_by_the_window_counts_once_per_contract_and_rule(self):
        legs = book()
        self.load(legs)
        broker = _WindowBroker([l["pos"] for l in legs.values()], "2026-11-03T09:35:12.123456789-05:00")
        result = self.run_manager(broker)
        held = [a for a in result["actions"] if a.get("order", {}).get("detail") == om.OPENING_WINDOW_DEFERRED_DETAIL]
        self.assertEqual(len(held), len(legs), "every rule is held by the window")
        self.assertEqual(self.submits, [])
        want = sorted(f"{l['occ']}|{rule}" for rule, l in legs.items())
        self.assertEqual(self.counted(), {"2026-11-03": {"single_leg": want, "group": []}},
                         "keyed by the BROKER's ET date, not the machine's")
        # the rules fire again on the next scan inside the window: still one per (contract, rule)
        self.run_manager(_WindowBroker([l["pos"] for l in legs.values()], "2026-11-03T09:41:00-05:00"))
        self.assertEqual(self.counted()["2026-11-03"]["single_leg"], want)

    def test_market_closed_and_after_the_window_are_not_counted(self):
        legs = book()
        self.load(legs)
        self.run_manager(_WindowBroker([l["pos"] for l in legs.values()], "2026-11-03T07:31:00-05:00", is_open=False))
        self.assertEqual(self.counted(), {}, "a market-closed deferral is not an opening-window block")
        self.load(legs)
        self.run_manager(_WindowBroker([l["pos"] for l in legs.values()], "2026-11-03T09:46:00-05:00"))
        self.assertEqual(self.counted(), {})
        self.assertEqual(len(self.submits), len(legs), "after 09:45 the closes go out exactly as before")

    def test_a_group_close_held_in_the_window_counts_once_by_its_legs_and_rule(self):
        legs = [{"symbol": "IREN261204P00020000"}, {"symbol": "IREN261204C00024000"}]
        state = {}
        with patch.object(om, "_market_is_open", return_value=True), \
                patch.object(om, "_in_opening_window", return_value=True), \
                patch.object(om, "_close_strategy_mleg", side_effect=AssertionError("never sent in the window")):
            res = om._request_group_close(legs, "IREN", state, "strategy_ic_max_loss_exit", "IC MAX-LOSS EXIT")
        action = {"action": "CLOSE_STRATEGY", "ticker": "IREN", "legs": [l["symbol"] for l in legs],
                  "type": "strategy_ic_max_loss_exit", "order": res}
        now_et = datetime(2026, 11, 3, 9, 36, tzinfo=ZoneInfo("America/New_York"))
        self.assertEqual(om._note_opening_window_deferrals([action], now_et=now_et), 1)
        self.assertEqual(om._note_opening_window_deferrals([action, dict(action)], now_et=now_et), 0)
        self.assertEqual(self.counted()["2026-11-03"]["group"],
                         ["IREN261204C00024000,IREN261204P00020000|strategy_ic_max_loss_exit"])

    def test_only_the_last_days_are_kept_and_a_bad_file_never_raises(self):
        action = {"action": "CLOSE", "occ_symbol": "X261127P00010000", "type": "profit_target",
                  "order": {"status": "deferred", "detail": om.OPENING_WINDOW_DEFERRED_DETAIL}}
        for day in range(1, 20):
            om._note_opening_window_deferrals([action], now_et=datetime(2026, 10, day, 9, 35, tzinfo=ZoneInfo("America/New_York")))
        days = sorted(self.counted())
        self.assertEqual(len(days), om.OPENING_DEFERRALS_KEEP_DAYS)
        self.assertEqual(days[-1], "2026-10-19")
        with open(self.path, "w") as f:
            f.write("{not json")
        self.assertEqual(om._note_opening_window_deferrals([action], now_et=datetime(2026, 10, 20, 9, 35, tzinfo=ZoneInfo("America/New_York"))), 1)
        with patch.object(om, "DATA_DIR", os.path.join(self.path, "nope")):
            self.assertEqual(om._note_opening_window_deferrals([action]), 0)


if __name__ == "__main__":
    unittest.main()
