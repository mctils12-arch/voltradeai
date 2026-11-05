"""EARNINGS CLOSE (2026-11-05): an already-open sold option is closed on the
last regular session before its underlying reports.

Human directive, 2026-11-05: "Existing short options into earnings: add a rule
that closes any already-open sold option before the underlying's earnings (last
session before the report, after 9:45 ET), not just one that blocks new ones.
Backfill a check against current positions on deploy."

Live: IREN261211P00018000 (1 sold at $1.68 on 11-04) was sold the day before
IREN's report (2026-11-05 after the close; a calendar bug, since fixed). The
entry gates block NEW short premium into a report; nothing closed one already
open.

Pinned:
- the close session: the report date for "amc", the previous trading day for
  "bmo" / "dmh" / unannounced, across a weekend and a holiday (pure);
- a sold single leg is bought back at the ask after 09:45 on its close session,
  its exit recorded as earnings_close, even inside the 60-minute hold gate;
- a report after the option's expiry, or not yet in its close session, closes
  nothing; bought legs are never touched and cost no lookup;
- a strategy group with a sold leg is closed whole, in one group close request;
- a failed lookup closes nothing: one WARNING per underlying per ET day,
  retried every tick, and one failing symbol does not starve the others;
- market closed or the opening window: deferred, nothing sent;
- still held on report day after a BMO report: closed at the first eligible
  tick; the session after, nothing;
- the deploy backfill: one summary action, on the first pass only;
- one Finnhub request per underlying per ET day;
- no FINNHUB_KEY, or the flag off: inert.
Mocked: no network (EarningsBroker answers Alpaca and Finnhub), state in a temp
dir, and the New York clock fixed at Thursday 2026-11-05 10:30 unless a test
moves it.
"""
import os
import sys
import unittest
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_manager as om  # noqa: E402
import options_scanner  # noqa: E402
import system_config  # noqa: E402
from ml_model_v2 import OPTIONS_EXIT_REASONS  # noqa: E402
from test_options_exit_market_hours import Broker, _ManagerCase, leg  # noqa: E402

ET = ZoneInfo("America/New_York")
THU = datetime(2026, 11, 5, 10, 30, tzinfo=ET)    # IREN reports today after the close


class EarningsBroker(Broker):
    """Broker plus Finnhub's per-symbol earnings calendar, answering only rows
    inside the requested from/to like the real endpoint. `reports` maps a
    symbol to [(ISO date, hour)]; a symbol in `fail` gets HTTP 429. Optional: a
    clock timestamp (the opening window reads it) and open orders."""

    def __init__(self, positions, reports=None, fail=(), is_open=True, clock_et=None, open_orders=()):
        super().__init__(positions, is_open=is_open)
        self.reports, self.fail, self.clock_et = dict(reports or {}), set(fail), clock_et
        self.open_orders = list(open_orders)
        self.finnhub = []      # the symbol of every Finnhub request, in order

    def get(self, url, **kw):
        params = kw.get("params") or {}
        if "finnhub.io/api/v1/calendar/earnings" in url:
            sym = params.get("symbol")
            self.finnhub.append(sym)
            if sym in self.fail:
                return MagicMock(status_code=429, json=lambda: {"error": "API limit reached"})
            rows = [{"symbol": sym, "date": d, "hour": h} for d, h in self.reports.get(sym, [])
                    if params["from"] <= d <= params["to"]]
            return MagicMock(status_code=200, json=lambda: {"earningsCalendar": rows})
        if "/v2/clock" in url and self.clock_et:
            self.gets.append(url)
            return MagicMock(status_code=200, json=lambda: {"is_open": self.is_open, "timestamp": self.clock_et})
        if "/v2/orders" in url:
            self.gets.append(url)
            return MagicMock(status_code=200, json=lambda: list(self.open_orders))
        return super().get(url, **kw)


def iren_put(days=36):
    """The live leg: 1 IREN 18P sold at $1.68, quoted 2.55/2.70. Its expiry is
    machine-relative (36 days) so CRITICAL DTE and the 21-DTE rule stay away."""
    return leg("IREN", days, "P", 18, 1, "short", 1.68, 2.60, 2.55, 2.70, -0.35, 0.02,
               "sell_cash_secured_put", ("buy", 2.70))


def csp(root, days, strike):
    """A quiet sold put (10% of max profit, low gamma): no other rule fires."""
    return leg(root, days, "P", strike, 1, "short", 1.00, 0.90, 0.88, 0.92, -0.25, 0.01,
               "sell_cash_secured_put", ("buy", 0.92))


def long_put(root="IREN", days=36):
    return leg(root, days, "P", 15, 1, "long", 0.50, 0.60, 0.58, 0.62, -0.10, 0.01, "buy_put", None)


def condor(root="GOOGL", days=35, entered_minutes_ago=10):
    """An iron condor entered minutes ago: inside the 60-minute hold gate."""
    legs = [leg(root, days, "P", 290, 2, "long", 3.00, 3.10, 3.00, 3.20, -0.10, 0.01, "iron_condor", None),
            leg(root, days, "P", 300, 2, "short", 4.70, 4.80, 4.70, 4.90, -0.20, 0.01, "iron_condor", None),
            leg(root, days, "C", 355, 2, "short", 4.50, 4.40, 4.30, 4.50, 0.20, 0.01, "iron_condor", None),
            leg(root, days, "C", 365, 2, "long", 2.90, 2.80, 2.70, 2.90, 0.10, 0.01, "iron_condor", None)]
    entered = (datetime.now() - timedelta(minutes=entered_minutes_ago)).isoformat()
    for lg in legs:
        lg["state"]["entry_timestamp"] = entered
    return legs


def setup_condor(setup="earnings_iv_crush"):
    """condor() whose legs record a scanner setup ("earnings_iv_crush" is what
    Setup 1 registers: options_scanner -> bot_engine -> register_options_entry)."""
    legs = condor()
    for lg in legs:
        lg["state"]["setup"] = setup
    return legs


def expiry_of(lg):
    return om._expiry_date(om._parse_occ_symbol(lg["occ"]))


class TestCloseSession(unittest.TestCase):
    """Pure: market_calendar.py's weekends and 2026 holidays."""

    def test_the_hour_picks_the_session_across_a_weekend_and_a_holiday(self):
        cases = [
            ((date(2026, 11, 6), "bmo"), date(2026, 11, 5)),    # Fri before the open -> Thu
            ((date(2026, 11, 5), "amc"), date(2026, 11, 5)),    # Thu after the close -> Thu itself (IREN, live)
            ((date(2026, 11, 9), "bmo"), date(2026, 11, 6)),    # Mon before the open -> Fri, over the weekend
            ((date(2026, 11, 9), "dmh"), date(2026, 11, 6)),    # during market hours: the session before
            ((date(2026, 11, 9), ""), date(2026, 11, 6)),       # unannounced: as before the open
            ((date(2026, 11, 9), "tbd"), date(2026, 11, 6)),    # unrecognised: the same
            ((date(2026, 11, 9), "amc"), date(2026, 11, 9)),
            ((date(2026, 11, 9), "AMC"), date(2026, 11, 9)),
            ((date(2026, 11, 7), "amc"), date(2026, 11, 6)),    # dated on a Saturday -> Fri
            ((date(2026, 11, 27), "bmo"), date(2026, 11, 25)),  # Fri after Thanksgiving (Thu 11-26 closed) -> Wed
            ((date(2026, 11, 27), "amc"), date(2026, 11, 27)),  # the 1 pm half day is a session
            ((date(2026, 11, 26), "amc"), date(2026, 11, 25)),  # dated on the holiday itself -> Wed
            ((date(2026, 9, 8), "bmo"), date(2026, 9, 4)),      # Tue after Labor Day: Mon closed + weekend -> Fri
        ]
        for (report_date, hour), session in cases:
            with self.subTest(report=report_date, hour=hour):
                self.assertEqual(om.earnings_close_session(report_date, hour), session)

    def test_due_from_the_close_session_through_the_report_date(self):
        monday = date(2026, 11, 9)
        for today, due in ((date(2026, 11, 5), False), (date(2026, 11, 6), True), (date(2026, 11, 7), True),
                           (date(2026, 11, 8), True), (monday, True), (date(2026, 11, 10), False)):
            with self.subTest(today=today):
                self.assertIs(om.earnings_close_due(today, monday, "bmo"), due)
        self.assertTrue(om.earnings_close_due(date(2026, 11, 5), date(2026, 11, 5), "amc"))
        self.assertFalse(om.earnings_close_due(date(2026, 11, 4), date(2026, 11, 5), "amc"))
        self.assertFalse(om.earnings_close_due(date(2026, 11, 6), date(2026, 11, 5), "amc"))


class _EarningsCase(_ManagerCase):
    def setUp(self):
        super().setUp()
        self.now = THU
        memos = (om._EARNINGS_REPORT_CACHE, om._EARNINGS_FAILED, om._EARNINGS_WARNED, om._EARNINGS_RUN)
        saved = [dict(m) for m in memos]

        def restore():
            for m, old in zip(memos, saved):
                m.clear()
                m.update(old)
            om._CLOCK_CACHE.update({"et": None, "et_at": 0.0})

        self.addCleanup(restore)
        for m in memos[:3]:
            m.clear()
        om._EARNINGS_RUN["backfill_done"] = True       # the backfill has its own tests
        for p in (patch.object(options_scanner, "FINNHUB_KEY", "test-key"),
                  patch.object(om, "_now_et", side_effect=lambda: self.now)):
            p.start()
            self.addCleanup(p.stop)

    def run_book(self, legs, **broker_kw):
        """Fresh state for `legs`, then one manage tick against a new broker."""
        self.load({str(i): lg for i, lg in enumerate(legs)})
        self.broker = EarningsBroker([lg["pos"] for lg in legs], **broker_kw)
        return self.run_manager(self.broker, allow_rolls=True)

    def run_again(self):
        """The next tick: same broker (its Finnhub log keeps counting), same state file."""
        return self.run_manager(self.broker, allow_rolls=True)

    @staticmethod
    def of_type(result, *types):
        return [a for a in result["actions"] if a.get("type") in types]


class TestSoldSingleLeg(_EarningsCase):
    def test_bmo_tomorrow_closes_today_after_0945_at_the_ask(self):
        put = iren_put()
        result = self.run_book([put], reports={"IREN": [("2026-11-06", "bmo")]})
        self.assertEqual(self.submits, [(put["occ"], 1, "buy", 2.70)])
        [act] = self.of_type(result, "earnings_close")
        self.assertEqual((act["action"], act["order"]["status"]), ("CLOSE", "submitted"))
        self.assertEqual((act["report_date"], act["report_hour"], act["close_session"]),
                         ("2026-11-06", "bmo", "2026-11-05"))
        self.assertTrue(act["reason"].startswith("EARNINGS CLOSE: IREN reports 2026-11-06 before the open; "
                                                 "2026-11-05 is the last session before it"), act["reason"])
        self.assertNotIn(put["occ"], self.read_state(), "an accepted close drops the state, like every single-leg close")
        self.track_fill.assert_called_once()
        self.assertEqual(self.track_fill.call_args[0][0]["exit_reason"], "earnings_close")

    def test_the_live_iren_case_amc_today_closes_today(self):
        put = iren_put()
        result = self.run_book([put], reports={"IREN": [("2026-11-05", "amc")]})
        self.assertEqual(self.submits, [(put["occ"], 1, "buy", 2.70)])
        [act] = self.of_type(result, "earnings_close")
        self.assertEqual(act["close_session"], "2026-11-05")
        self.assertIn("2026-11-05 after the close", act["reason"])

    def test_a_leg_sold_30_minutes_ago_is_not_held_by_the_hold_gate(self):
        put = iren_put()
        put["state"]["entry_timestamp"] = (datetime.now() - timedelta(minutes=30)).isoformat()
        self.run_book([put], reports={"IREN": [("2026-11-05", "amc")]})
        self.assertEqual(self.submits, [(put["occ"], 1, "buy", 2.70)])

    def test_nothing_before_the_close_session_after_expiry_or_without_a_report(self):
        put = iren_put(days=40)
        for label, rows in (("next week: not yet its close session", [("2026-11-09", "bmo")]),
                            ("the day after expiry", [((expiry_of(put) + timedelta(days=1)).isoformat(), "bmo")]),
                            ("no report", [])):
            with self.subTest(label):
                om._EARNINGS_REPORT_CACHE.clear()
                result = self.run_book([put], reports={"IREN": rows})
                self.assertEqual(self.submits, [])
                self.assertEqual(self.of_type(result, "earnings_close"), [])

    def test_only_the_leg_whose_life_spans_the_report_is_closed(self):
        near, far = csp("IREN", 30, 17), csp("IREN", 60, 16)
        report = expiry_of(near) + timedelta(days=1)                 # after near's expiry, inside far's
        session = om.earnings_close_session(report, "bmo")
        self.now = datetime(session.year, session.month, session.day, 10, 30, tzinfo=ET)
        result = self.run_book([near, far], reports={"IREN": [(report.isoformat(), "bmo")]})
        self.assertEqual(self.submits, [(far["occ"], 1, "buy", 0.92)])
        self.assertEqual([a["occ_symbol"] for a in self.of_type(result, "earnings_close")], [far["occ"]])

    def test_bought_legs_are_never_touched_and_cost_no_lookup(self):
        result = self.run_book([long_put()], reports={"IREN": [("2026-11-05", "amc")]})
        self.assertEqual(self.submits, [])
        self.assertEqual(self.broker.finnhub, [], "no sold option, no lookup")
        self.assertEqual(self.of_type(result, "earnings_close", "strategy_earnings_close"), [])

    def test_still_held_on_report_day_after_a_bmo_report_closes_at_the_first_eligible_tick(self):
        put = iren_put()
        self.now = datetime(2026, 11, 6, 10, 0, tzinfo=ET)       # Fri: the report came out before the open
        result = self.run_book([put], reports={"IREN": [("2026-11-06", "bmo")]})
        self.assertEqual(self.submits, [(put["occ"], 1, "buy", 2.70)])
        [act] = self.of_type(result, "earnings_close")
        self.assertIn("its last session before the report (2026-11-05) has passed — closing the sold option at "
                      "the first eligible tick", act["reason"])
        self.submits.clear()
        self.now = datetime(2026, 11, 9, 10, 0, tzinfo=ET)       # Mon: the report is behind it
        self.run_book([put], reports={"IREN": [("2026-11-06", "bmo")]})
        self.assertEqual(self.submits, [])

    def test_market_closed_and_the_opening_window_defer_without_an_order(self):
        put = iren_put()
        for label, now, broker_kw, detail in (
                ("pre-market", datetime(2026, 11, 5, 8, 0, tzinfo=ET), {"is_open": False},
                 om.SINGLE_LEG_DEFERRED_DETAIL),
                ("09:35 ET", datetime(2026, 11, 5, 9, 35, tzinfo=ET), {"clock_et": "2026-11-05T09:35:00-05:00"},
                 om.OPENING_WINDOW_DEFERRED_DETAIL)):
            with self.subTest(label):
                self.now = now
                result = self.run_book([put], reports={"IREN": [("2026-11-05", "amc")]}, **broker_kw)
                self.assertEqual(self.submits, [])
                [act] = self.of_type(result, "earnings_close")
                self.assertEqual((act["action"], act["order"]), ("CLOSE", {"status": "deferred", "method": "none",
                                                                           "detail": detail}))
                self.assertIn(put["occ"], self.read_state(), "a deferred leg keeps its state")
                self.track_fill.assert_not_called()

    def test_a_working_close_order_waits_and_no_ask_holds(self):
        put = iren_put()
        result = self.run_book([put], reports={"IREN": [("2026-11-05", "amc")]},
                               open_orders=[{"id": "o-1", "symbol": put["occ"], "side": "buy", "status": "new"}])
        self.assertEqual(self.submits, [], "never a second buy-to-close while one is working")
        self.assertEqual([a["action"] for a in self.of_type(result, "earnings_close_working")], ["WAIT"])
        result = self.run_book([put], reports={"IREN": [("2026-11-05", "amc")]},
                               open_orders=[{"id": "o-2", "symbol": put["occ"], "side": "sell", "status": "new"}])
        self.assertEqual(self.submits, [(put["occ"], 1, "buy", 2.70)], "a working sell is no close: it never holds the leg")
        self.submits.clear()
        put["snap"]["ask"] = 0.0
        result = self.run_book([put], reports={"IREN": [("2026-11-05", "amc")]})
        self.assertEqual(self.submits, [])
        self.assertEqual([a["action"] for a in self.of_type(result, "earnings_close_no_ask")], ["HOLD"])
        self.assertIn(put["occ"], self.read_state())

    def test_a_rejected_close_keeps_the_legs_state_and_records_no_exit(self):
        put = iren_put()
        with patch.object(om, "_submit_close_order", return_value={"status": "error", "detail": "Alpaca rejected"}):
            result = self.run_book([put], reports={"IREN": [("2026-11-05", "amc")]})
        [act] = self.of_type(result, "earnings_close")
        self.assertEqual(act["order"]["status"], "error")
        saved = self.read_state()[put["occ"]]
        self.assertEqual((saved["strategy"], saved["entry_timestamp"]),
                         (put["state"]["strategy"], put["state"]["entry_timestamp"]))
        self.track_fill.assert_not_called()

    def test_the_exit_reason_is_an_options_exit_reason(self):
        self.assertIn("earnings_close", OPTIONS_EXIT_REASONS)


class TestStrategyGroup(_EarningsCase):
    def test_a_group_with_a_sold_leg_is_closed_whole_in_one_request(self):
        legs = condor()
        straddle = [leg("KMI", 35, "C", 30, 1, "long", 1.00, 1.00, 0.95, 1.05, 0.50, 0.05, "buy_straddle", None),
                    leg("KMI", 35, "P", 30, 1, "long", 1.00, 1.00, 0.95, 1.05, -0.50, 0.05, "buy_straddle", None)]
        with patch.object(om, "_close_strategy_mleg",
                          return_value={"status": "submitted", "method": "mleg", "order_id": "m-1"}) as mleg:
            result = self.run_book(legs + straddle, reports={"GOOGL": [("2026-11-06", "bmo")],
                                                             "KMI": [("2026-11-06", "bmo")]})
        self.assertEqual(mleg.call_count, 1)
        self.assertEqual(sorted(p["symbol"] for p in mleg.call_args[0][0]), sorted(lg["occ"] for lg in legs))
        acts = [a for a in result["actions"] if a["action"] == "CLOSE_STRATEGY"]
        self.assertEqual([(a["type"], a["strategy"], a["ticker"]) for a in acts],
                         [("strategy_earnings_close", "iron_condor", "GOOGL")], "the bought straddle is untouched")
        self.assertEqual((acts[0]["report_date"], acts[0]["close_session"]), ("2026-11-06", "2026-11-05"))
        self.assertEqual(self.submits, [], "no single-leg close of a condor leg")
        self.track_fill.assert_not_called()   # multi-leg exits are not attributed (KNOWN BROKEN #12(c))
        saved = self.read_state()
        self.assertTrue(all(saved[lg["occ"]]["close_rule"] == "strategy_earnings_close" for lg in legs),
                        "the request is stamped, so it is pursued until the legs are gone")
        self.assertEqual(self.broker.finnhub, ["GOOGL"], "a bought straddle has no sold leg to look up")

    def test_the_group_close_waits_out_the_opening_window_then_is_pursued(self):
        legs = condor()
        self.now = datetime(2026, 11, 5, 9, 35, tzinfo=ET)
        with patch.object(om, "_close_strategy_mleg", side_effect=AssertionError("nothing is sent before 09:45")):
            result = self.run_book(legs, reports={"GOOGL": [("2026-11-06", "bmo")]},
                                   clock_et="2026-11-05T09:35:00-05:00")
        [act] = [a for a in result["actions"] if a["action"] == "CLOSE_STRATEGY"]
        self.assertEqual((act["type"], act["order"]["status"]), ("strategy_earnings_close", "deferred"))
        self.assertTrue(all(self.read_state()[lg["occ"]].get("close_requested_at") for lg in legs))
        self.now, self.broker.clock_et = datetime(2026, 11, 5, 9, 46, tzinfo=ET), "2026-11-05T09:46:00-05:00"
        with patch.object(om, "_close_strategy_mleg", return_value={"status": "submitted", "method": "mleg"}) as mleg:
            result = self.run_again()
        self.assertEqual(mleg.call_count, 1)
        [act] = [a for a in result["actions"] if a["action"] == "CLOSE_STRATEGY"]
        self.assertEqual((act["type"], act["order"]["status"]), ("strategy_earnings_close", "submitted"))


class TestSetupExemption(_EarningsCase):
    """A defined-risk group opened FOR its report (Setup 1's earnings_iv_crush
    condor) is not closed before it. Everything else still is."""

    GOOGL_BMO_TOMORROW = {"GOOGL": [("2026-11-06", "bmo")]}

    def run_groups(self, legs, reports=None):
        with patch.object(om, "_close_strategy_mleg",
                          return_value={"status": "submitted", "method": "mleg", "order_id": "m-1"}) as mleg:
            result = self.run_book(legs, reports=reports or self.GOOGL_BMO_TOMORROW)
        return result, mleg

    @staticmethod
    def group_closes(result):
        return [a["type"] for a in result["actions"] if a["action"] == "CLOSE_STRATEGY"]

    def test_a_setup1_condor_with_its_report_tomorrow_bmo_is_not_closed(self):
        legs = setup_condor()
        result, mleg = self.run_groups(legs)
        mleg.assert_not_called()
        self.assertEqual(self.group_closes(result), [])
        self.assertFalse([lg for lg in legs if "close_requested_at" in self.read_state()[lg["occ"]]])
        self.assertEqual(self.broker.finnhub, [], "an exempt group's legs are not looked up")

    def test_still_exempt_when_its_underlying_is_looked_up_for_another_sold_leg(self):
        legs, googl_put = setup_condor(), csp("GOOGL", 40, 250)    # another expiry: not adopted into the condor
        result, mleg = self.run_groups(legs + [googl_put])
        mleg.assert_not_called()
        self.assertEqual(self.submits, [(googl_put["occ"], 1, "buy", 0.92)], "the CSP closes, the condor does not")
        self.assertEqual(self.broker.finnhub, ["GOOGL"])

    def test_the_same_condor_with_any_other_setup_is_closed(self):
        for setup in ("high_iv_premium_sale", "vrp_premium_harvest", ""):
            with self.subTest(setup=setup):
                result, mleg = self.run_groups(setup_condor(setup))
                self.assertEqual(mleg.call_count, 1)
                self.assertEqual(self.group_closes(result), ["strategy_earnings_close"])

    def test_a_lone_short_leg_recorded_as_earnings_iv_crush_is_closed(self):
        lone_short = setup_condor()[1]       # the short 300P; its state still records iron_condor + the setup
        result, mleg = self.run_groups([lone_short])
        self.assertEqual(mleg.call_count, 1)
        self.assertEqual([p["symbol"] for p in mleg.call_args[0][0]], [lone_short["occ"]])
        self.assertEqual(self.group_closes(result), ["strategy_earnings_close"])

    def test_an_emptied_tuple_makes_the_rule_literal(self):
        with patch.dict(system_config.BASE_CONFIG, {"OPTIONS_EARNINGS_CLOSE_EXEMPT_SETUPS": ()}):
            result, mleg = self.run_groups(setup_condor())
        self.assertEqual(mleg.call_count, 1)
        self.assertEqual(self.group_closes(result), ["strategy_earnings_close"])

    def test_defined_risk_means_every_sold_contract_is_covered_by_a_bought_one(self):
        legs = setup_condor()
        group = lambda ls, setup="earnings_iv_crush": {"setup": setup, "legs": [lg["pos"] for lg in ls]}
        self.assertTrue(om._earnings_close_exempt(group(legs)), "the condor")
        self.assertTrue(om._earnings_close_exempt(group(legs[:2])), "a put credit spread with its wing")
        self.assertFalse(om._earnings_close_exempt(group(legs, "high_iv_premium_sale")), "another setup")
        self.assertFalse(om._earnings_close_exempt(group([legs[1]])), "a lone short leg")
        self.assertFalse(om._earnings_close_exempt(group(legs[:3])), "the call wing gone: the short call is naked")
        adopted = leg("GOOGL", 35, "P", 280, 2, "short", 1.00, 1.00, 0.95, 1.05, -0.10, 0.01, "iron_condor", None)
        self.assertFalse(om._earnings_close_exempt(group(legs + [adopted])), "4 sold puts against 2 bought")

    def test_the_backfill_lists_an_exempt_group(self):
        om._EARNINGS_RUN["backfill_done"] = False
        legs, put = setup_condor(), iren_put()
        result, _ = self.run_groups(legs + [put], reports={**self.GOOGL_BMO_TOMORROW, "IREN": [("2026-11-05", "amc")]})
        [summary] = self.of_type(result, "earnings_backfill")
        rows = {r["occ_symbol"]: r for r in summary["legs"]}
        sold = [lg["occ"] for lg in legs if lg["pos"]["side"] == "short"]
        self.assertEqual({occ: rows[occ]["status"] for occ in rows},
                         {**{occ: "exempt_setup" for occ in sold}, put["occ"]: "due"}, "sold legs only")
        self.assertTrue(all(rows[occ]["setup"] == "earnings_iv_crush" for occ in sold))
        self.assertIn(f"{sold[0]}: exempt (setup earnings_iv_crush, a defined-risk structure opened for its report)",
                      summary["reason"])
        self.assertEqual(self.submits, [(put["occ"], 1, "buy", 2.70)])


class TestLookupFailure(_EarningsCase):
    def test_no_close_on_a_guess_one_warning_per_underlying_per_et_day_retried_every_tick(self):
        put = iren_put()
        first = self.run_book([put], reports={"IREN": [("2026-11-05", "amc")]}, fail={"IREN"})
        self.assertEqual(self.submits, [])
        [warn] = self.of_type(first, "earnings_lookup_failed")
        self.assertEqual((warn["action"], warn["ticker"], warn["legs"]), ("WARNING", "IREN", [put["occ"]]))
        self.assertIn("not closed on a guess", warn["reason"])
        second = self.run_again()
        self.assertEqual(self.of_type(second, "earnings_lookup_failed"), [], "once per underlying per ET day")
        self.assertEqual(self.broker.finnhub, ["IREN", "IREN"], "retried on the next tick")
        self.assertEqual(self.submits, [])
        self.broker.fail.clear()                                   # Finnhub answers again
        self.run_again()
        self.assertEqual(self.submits, [(put["occ"], 1, "buy", 2.70)])
        self.broker.fail.add("IREN")
        self.now = datetime(2026, 11, 6, 10, 30, tzinfo=ET)        # a new ET day warns again
        self.assertEqual(len(self.of_type(self.run_again(), "earnings_lookup_failed")), 1)

    def test_one_failing_symbol_does_not_starve_the_others(self):
        aal, put = csp("AAL", 29, 13), iren_put()
        result = self.run_book([aal, put], reports={"IREN": [("2026-11-05", "amc")]}, fail={"AAL"})
        self.assertEqual(self.broker.finnhub, ["AAL"], "after one failed request the pass sends no more")
        self.assertEqual(self.submits, [])
        warned = {a["ticker"]: a["reason"] for a in self.of_type(result, "earnings_lookup_failed")}
        self.assertEqual(sorted(warned), ["AAL", "IREN"])
        self.assertIn("another lookup failed first", warned["IREN"])
        self.run_again()   # AAL failed today, so IREN is asked first; AAL is asked again after it
        self.assertEqual(self.broker.finnhub, ["AAL", "IREN", "AAL"])
        self.assertEqual(self.submits, [(put["occ"], 1, "buy", 2.70)])


class TestDeployBackfill(_EarningsCase):
    def test_the_first_pass_lists_every_sold_option_and_closes_the_ones_due(self):
        om._EARNINGS_RUN["backfill_done"] = False
        put, aal, googl, cmcsa, zzz = iren_put(), csp("AAL", 29, 13), csp("GOOGL", 40, 250), csp("CMCSA", 43, 24), \
            csp("ZZZ", 30, 10)
        result = self.run_book([put, aal, googl, cmcsa, zzz, long_put("MSFT")], fail={"ZZZ"}, reports={
            "IREN": [("2026-11-05", "amc")],
            "GOOGL": [("2026-11-09", "bmo")],
            "CMCSA": [((expiry_of(cmcsa) + timedelta(days=5)).isoformat(), "bmo")],    # after its expiry
        })
        [summary] = self.of_type(result, "earnings_backfill")
        self.assertEqual(summary["action"], "INFO")
        rows = {r["occ_symbol"]: r for r in summary["legs"]}
        self.assertEqual({occ: r["status"] for occ, r in rows.items()},
                         {put["occ"]: "due", aal["occ"]: "none_before_expiry", googl["occ"]: "scheduled",
                          cmcsa["occ"]: "none_before_expiry", zzz["occ"]: "lookup_failed"}, "sold options only")
        g = rows[googl["occ"]]
        self.assertEqual((g["underlying"], g["report_date"], g["report_hour"], g["close_session"]),
                         ("GOOGL", "2026-11-09", "bmo", "2026-11-06"))
        self.assertTrue(summary["reason"].startswith("EARNINGS BACKFILL (first pass since start): 5 sold option(s) "
                                                     "checked"), summary["reason"])
        self.assertIn(f"{put['occ']}: IREN reports 2026-11-05 after the close, close session 2026-11-05 (due now)",
                      summary["reason"])
        self.assertEqual(self.submits, [(put["occ"], 1, "buy", 2.70)], "the leg already due closes in the same pass")
        self.assertEqual(self.broker.finnhub, ["AAL", "CMCSA", "GOOGL", "IREN", "ZZZ"], "one request per underlying")
        self.assertEqual([a["ticker"] for a in self.of_type(result, "earnings_lookup_failed")], ["ZZZ"])
        self.assertEqual(self.of_type(self.run_again(), "earnings_backfill"), [], "the first pass only")

    def test_a_first_pass_with_no_sold_option_reports_nothing_and_no_later_pass_backfills(self):
        om._EARNINGS_RUN["backfill_done"] = False
        self.assertEqual(self.of_type(self.run_book([long_put()]), "earnings_backfill"), [])
        result = self.run_book([iren_put()], reports={"IREN": [("2026-11-05", "amc")]})
        self.assertEqual(self.of_type(result, "earnings_backfill"), [])


class TestLookupCache(_EarningsCase):
    def test_one_request_per_underlying_per_et_day(self):
        near, far = csp("IREN", 30, 17), csp("IREN", 60, 16)
        self.run_book([near, far])
        self.run_again()
        self.assertEqual(self.broker.finnhub, ["IREN"], "two sold legs, two ticks, one request")
        self.assertEqual(om._EARNINGS_REPORT_CACHE["IREN"][1], expiry_of(far), "the window reaches the latest expiry")
        self.now = datetime(2026, 11, 6, 10, 30, tzinfo=ET)
        self.run_again()
        self.assertEqual(self.broker.finnhub, ["IREN", "IREN"], "a new ET day asks again")


class TestInert(_EarningsCase):
    def test_no_finnhub_key_or_the_flag_off_does_nothing(self):
        om._EARNINGS_RUN["backfill_done"] = False
        for label, off in (("no FINNHUB_KEY", patch.object(options_scanner, "FINNHUB_KEY", "")),
                           ("flag off", patch.dict(system_config.BASE_CONFIG,
                                                   {"OPTIONS_EARNINGS_CLOSE_ENABLED": False}))):
            with self.subTest(label), off:
                result = self.run_book([iren_put()], reports={"IREN": [("2026-11-05", "amc")]})
                self.assertEqual(self.broker.finnhub, [])
                self.assertEqual(self.submits, [])
                self.assertEqual(self.of_type(result, "earnings_close", "earnings_backfill",
                                              "earnings_lookup_failed"), [])
        self.assertFalse(om._EARNINGS_RUN["backfill_done"], "an inert pass does not use up the deploy backfill")

    def test_the_flag_ships_on(self):
        self.assertIs(system_config.BASE_CONFIG["OPTIONS_EARNINGS_CLOSE_ENABLED"], True)


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class TestFetchSymbolEarningsReports(unittest.TestCase):
    """options_scanner._fetch_symbol_earnings_reports: dates AND hours, one
    company's rows, an explicit window; failures are never "no report"."""

    LO, HI = date(2026, 11, 5), date(2026, 12, 11)

    def fetch(self, body=None, status=200, raises=None, symbol="iren", lo=LO, hi=HI):
        calls = []

        def fake_get(url, params=None, timeout=None, **kw):
            calls.append(dict(params or {}))
            if raises:
                raise raises
            return _Resp(status, body)

        with patch.object(options_scanner.requests, "get", side_effect=fake_get):
            return options_scanner._fetch_symbol_earnings_reports(symbol, lo, hi), calls

    def test_dates_and_hours_inside_the_window_one_company(self):
        (reports, ok), calls = self.fetch({"earningsCalendar": [
            {"symbol": "IREN", "date": "2026-11-05", "hour": "amc"},
            {"symbol": "IREN", "date": "2026-12-10", "hour": "BMO"},    # case-folded
            {"symbol": "IREN", "date": "2026-11-20", "hour": "dmh"},
            {"symbol": "IREN", "date": "2026-11-25", "hour": None},     # unannounced
            {"symbol": "IREN", "date": "2026-11-27", "hour": "tbd"},    # unrecognised -> ""
            {"symbol": "IREN", "date": "2026-11-04", "hour": "bmo"},    # before the window
            {"symbol": "IREN", "date": "2026-12-12", "hour": "bmo"},    # after it
            {"symbol": "OTHER", "date": "2026-11-06", "hour": "bmo"},   # never another name's row
            {"symbol": "IREN", "date": "soon", "hour": "bmo"},
            "junk",
        ]})
        self.assertTrue(ok)
        self.assertEqual(reports, [(date(2026, 11, 5), "amc"), (date(2026, 11, 20), "dmh"), (date(2026, 11, 25), ""),
                                   (date(2026, 11, 27), ""), (date(2026, 12, 10), "bmo")])
        self.assertEqual({k: calls[0][k] for k in ("from", "to", "symbol")},
                         {"from": "2026-11-05", "to": "2026-12-11", "symbol": "IREN"})

    def test_two_rows_on_one_date_keep_the_earlier_closing_reading(self):
        for hours, kept in ((("amc", "bmo"), "bmo"), (("bmo", "amc"), "bmo"), (("", "amc"), ""),
                            (("amc", "amc"), "amc")):
            with self.subTest(hours=hours):
                (reports, ok), _ = self.fetch({"earningsCalendar": [
                    {"symbol": "IREN", "date": "2026-11-05", "hour": h} for h in hours]})
                self.assertEqual((reports, ok), ([(date(2026, 11, 5), kept)], True))

    def test_no_row_is_a_real_answer(self):
        self.assertEqual(self.fetch({"earningsCalendar": []})[0], ([], True))

    def test_a_failure_is_never_read_as_no_report(self):
        for label, kw in (("HTTP 429", {"status": 429, "body": {"error": "API limit reached"}}),
                          ("no calendar list", {"body": {"error": "Invalid API key"}}),
                          ("not a dict", {"body": ["IREN"]}),
                          ("network", {"raises": ConnectionError("down")})):
            with self.subTest(label):
                self.assertEqual(self.fetch(**kw)[0], (None, False))
        out, calls = self.fetch({"earningsCalendar": []}, lo=self.HI, hi=self.LO)
        self.assertEqual((out, calls), ((None, False), []), "an empty window sends nothing")


if __name__ == "__main__":
    unittest.main()
