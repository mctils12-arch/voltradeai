"""REPAIR 2026-11-12 — EARNINGS CLOSE: a defined-risk structure closed with its
setup unknown is named in the pass.

What was wrong: the earnings close exempts a Setup-1 ("earnings_iv_crush")
condor, which is opened FOR its report. The setup is recorded only in each
leg's options state. With that state lost the condor is not a group at all:
its legs are managed one by one and its two sold legs are bought back before
the very report it was opened for, in rows that read like any accident's
(research/open_questions.md, 2026-11-05 follow-ups, "[EARNINGS CLOSE]
Residuals").

The exemption is NOT inferred. Nothing but the lost record identifies an
earnings play: every iron_condor is built by one builder whatever the setup,
the entry time is part of the same lost state, and the broker's orders carry
no setup. Exempting a condor that is not an earnings play would be worse than
closing one that is. So the rule still closes the structure, and says so.

Pinned:
- a condor (or a credit spread with its wing) whose state is lost is closed
  exactly as before, and the pass carries one "earnings_setup_unknown" WARNING
  naming the underlying, the expiry, every leg and the sold legs being closed;
- once per structure per ET day, also while the close is deferred;
- no notice, and the same closes as before, for: a recorded group (exempt or
  not), a partly lost condor (adopted, still exempt), a lone sold leg, an
  uncovered structure, recorded single legs in a spread shape, a report that
  is not due, a failed lookup, and an emptied exemption list;
- a failing notice never stops the close.
Mocked like test_options_earnings_close.py: no network, state in a temp dir,
the New York clock fixed at Thursday 2026-11-05 10:30.
"""
import json
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_manager as om  # noqa: E402
import system_config  # noqa: E402
from test_options_earnings_close import EarningsBroker, _EarningsCase, csp, expiry_of, setup_condor  # noqa: E402
from test_options_exit_market_hours import leg  # noqa: E402

ET = ZoneInfo("America/New_York")
GOOGL_BMO_TOMORROW = {"GOOGL": [("2026-11-06", "bmo")]}
NOTICE = "earnings_setup_unknown"


def occs(legs, side=None):
    return sorted(lg["occ"] for lg in legs if side is None or lg["pos"]["side"] == side)


class _LostStateCase(_EarningsCase):
    def setUp(self):
        super().setUp()
        noted = getattr(om, "_EARNINGS_SETUP_NOTED", None)      # the once-per-day memo, cold for every test
        if noted is not None:
            saved = dict(noted)
            noted.clear()
            self.addCleanup(lambda: (noted.clear(), noted.update(saved)))

    def run_lost(self, legs, keep=(), reports=None, **broker_kw):
        """One manage tick over `legs` whose recorded state is gone, except
        for the legs in `keep`."""
        self.load({str(i): lg for i, lg in enumerate(legs)})
        kept = {lg["occ"]: lg["state"] for lg in keep}
        with open(self.state_path, "w") as f:
            json.dump(kept, f)
        self.broker = EarningsBroker([lg["pos"] for lg in legs],
                                     reports=GOOGL_BMO_TOMORROW if reports is None else reports, **broker_kw)
        return self.tick()

    def run_recorded(self, legs, reports=None):
        self.load({str(i): lg for i, lg in enumerate(legs)})
        self.broker = EarningsBroker([lg["pos"] for lg in legs],
                                     reports=GOOGL_BMO_TOMORROW if reports is None else reports)
        return self.tick()

    def tick(self):
        with patch.object(om, "_close_strategy_mleg",
                          return_value={"status": "submitted", "method": "mleg", "order_id": "m-1"}) as self.mleg:
            return self.run_manager(self.broker, allow_rolls=True)

    def sold_closes(self, legs):
        """What the rule submits for the sold legs, in the broker's order: each
        bought back at its ask."""
        return [(lg["occ"], lg["qty"], "buy", lg["snap"]["ask"]) for lg in legs if lg["pos"]["side"] == "short"]


class TestALostStructureIsClosedAndNamed(_LostStateCase):
    def test_a_condor_whose_state_is_lost_is_closed_as_before_and_named(self):
        legs = setup_condor()
        result = self.run_lost(legs)
        # the rule itself is unchanged: with no state the four legs are single legs, the two sold ones are bought back
        self.assertEqual(self.submits, self.sold_closes(legs))
        self.mleg.assert_not_called()
        self.assertEqual([a["occ_symbol"] for a in self.of_type(result, "earnings_close")],
                         [lg["occ"] for lg in legs if lg["pos"]["side"] == "short"])
        [note] = self.of_type(result, NOTICE)
        expiry = expiry_of(legs[0]).isoformat()
        self.assertEqual((note["action"], note["ticker"], note["expiry"]), ("WARNING", "GOOGL", expiry))
        self.assertEqual((note["legs"], note["sold_legs"]), (occs(legs), occs(legs, "short")))
        self.assertEqual((note["report_date"], note["report_hour"], note["close_session"]),
                         ("2026-11-06", "bmo", "2026-11-05"))
        self.assertEqual(note["reason"], (
            f"EARNINGS CLOSE: GOOGL's 4 option legs expiring {expiry} ({', '.join(occs(legs))}) form a defined-risk "
            f"structure, but their state records no strategy or setup (lost, or never written) — it cannot be "
            f"exempted as an earnings play, so its sold option(s) ({', '.join(occs(legs, 'short'))}) are closed for "
            f"GOOGL's 2026-11-06 report like any other; noted once per ET day"))
        kinds = [a.get("type") for a in result["actions"]]
        self.assertLess(kinds.index(NOTICE), kinds.index("earnings_close"), "the notice comes before the closes")

    def test_the_state_the_manager_rebuilt_as_unknown_counts_as_lost(self):
        legs = setup_condor()
        self.run_lost(legs, reports={"GOOGL": []})        # no report: the manager records each leg as "unknown"
        self.assertEqual({st["strategy"] for st in self.read_state().values()}, {"unknown"})
        om._EARNINGS_REPORT_CACHE.clear()
        self.broker.reports = dict(GOOGL_BMO_TOMORROW)
        result = self.tick()
        self.assertEqual(len(self.of_type(result, NOTICE)), 1)
        self.assertEqual(self.submits, self.sold_closes(legs))

    def test_a_lost_credit_spread_with_its_wing_is_named_too(self):
        spread = setup_condor()[:2]                        # the bought 290 put and the sold 300 put
        result = self.run_lost(spread)
        [note] = self.of_type(result, NOTICE)
        self.assertEqual((note["legs"], note["sold_legs"]), (occs(spread), occs(spread, "short")))
        self.assertIn("GOOGL's 2 option legs", note["reason"])
        self.assertEqual(self.submits, self.sold_closes(spread))

    def test_two_lost_structures_are_named_one_each(self):
        googl = setup_condor()
        tsla = [leg("TSLA", 35, "P", 380, 1, "long", 3.00, 3.10, 3.00, 3.20, -0.10, 0.01, "iron_condor", None),
                leg("TSLA", 35, "P", 400, 1, "short", 4.70, 4.80, 4.70, 4.90, -0.20, 0.01, "iron_condor", None)]
        result = self.run_lost(googl + tsla, reports={**GOOGL_BMO_TOMORROW, "TSLA": [("2026-11-05", "amc")]})
        notes = {a["ticker"]: a for a in self.of_type(result, NOTICE)}
        self.assertEqual(sorted(notes), ["GOOGL", "TSLA"])
        self.assertEqual((notes["TSLA"]["legs"], notes["TSLA"]["report_date"], notes["TSLA"]["close_session"]),
                         (occs(tsla), "2026-11-05", "2026-11-05"))
        self.assertEqual(self.submits, self.sold_closes(googl + tsla))


class TestOncePerStructurePerEtDay(_LostStateCase):
    def test_one_notice_a_day_while_the_structure_is_due(self):
        legs = setup_condor()
        self.assertEqual(len(self.of_type(self.run_lost(legs), NOTICE)), 1)
        self.assertEqual(self.of_type(self.tick(), NOTICE), [], "the same ET day: said once")
        self.now = datetime(2026, 11, 6, 10, 0, tzinfo=ET)        # report day, still held: still due
        self.assertEqual(len(self.of_type(self.tick(), NOTICE)), 1, "a new ET day says it again")
        self.assertEqual(self.of_type(self.tick(), NOTICE), [])
        self.submits.clear()
        self.now = datetime(2026, 11, 9, 10, 0, tzinfo=ET)        # the report is behind it: nothing is closed
        self.assertEqual(self.of_type(self.tick(), NOTICE), [])
        self.assertEqual(self.submits, [])

    def test_a_deferred_close_is_named_once_too(self):
        legs = setup_condor()
        self.now = datetime(2026, 11, 5, 8, 0, tzinfo=ET)
        result = self.run_lost(legs, is_open=False)
        self.assertEqual(self.submits, [])
        self.assertEqual([a["order"]["status"] for a in self.of_type(result, "earnings_close")], ["deferred"] * 2)
        self.assertEqual(len(self.of_type(result, NOTICE)), 1)
        self.assertEqual(self.of_type(self.tick(), NOTICE), [])


class TestNoNoticeAndTheSameClosesAsBefore(_LostStateCase):
    def test_a_recorded_setup1_condor_is_exempt_and_silent(self):
        result = self.run_recorded(setup_condor())
        self.mleg.assert_not_called()
        self.assertEqual((self.submits, self.of_type(result, NOTICE, "earnings_close", "strategy_earnings_close")),
                         ([], []))

    def test_a_recorded_group_with_another_or_no_setup_is_closed_whole_and_silent(self):
        for setup in ("high_iv_premium_sale", ""):
            with self.subTest(setup=setup):
                result = self.run_recorded(setup_condor(setup))
                self.assertEqual(self.mleg.call_count, 1)
                self.assertEqual([a["type"] for a in result["actions"] if a["action"] == "CLOSE_STRATEGY"],
                                 ["strategy_earnings_close"])
                self.assertEqual(self.of_type(result, NOTICE), [], "its state is recorded: nothing is unknown")

    def test_a_partly_lost_condor_is_adopted_and_stays_exempt(self):
        legs = setup_condor()
        result = self.run_lost(legs, keep=[legs[2]])       # one sold leg still records the strategy and the setup
        self.mleg.assert_not_called()
        self.assertEqual(self.submits, [])
        self.assertEqual(len(self.of_type(result, "group_adopted")), 3)
        self.assertEqual(self.of_type(result, NOTICE, "earnings_close", "strategy_earnings_close"), [])

    def test_a_lone_sold_leg_or_an_uncovered_structure_is_closed_and_silent(self):
        condor_legs = setup_condor()
        for label, legs in (("a lone sold put", [condor_legs[1]]),
                            ("the call wing gone: the sold call is uncovered", condor_legs[:3])):
            with self.subTest(label):
                self.submits.clear()
                result = self.run_lost(legs)
                self.assertEqual(self.submits, self.sold_closes(legs))
                self.assertEqual(self.of_type(result, NOTICE), [])

    def test_recorded_single_legs_in_a_spread_shape_are_silent(self):
        sold = csp("GOOGL", 35, 300)
        bought = leg("GOOGL", 35, "P", 290, 1, "long", 0.50, 0.60, 0.58, 0.62, -0.10, 0.01, "buy_put", None)
        result = self.run_recorded([bought, sold])
        self.assertEqual(self.submits, [(sold["occ"], 1, "buy", 0.92)])
        self.assertEqual(self.of_type(result, NOTICE), [], "both legs record what opened them")

    def test_a_report_that_is_not_due_names_nothing_and_closes_nothing(self):
        legs = setup_condor()
        for label, rows in (("no report", []), ("next week: not yet its close session", [("2026-11-12", "bmo")]),
                            ("the day after expiry", [((expiry_of(legs[0]) + timedelta(days=1)).isoformat(), "bmo")])):
            with self.subTest(label):
                om._EARNINGS_REPORT_CACHE.clear()
                result = self.run_lost(legs, reports={"GOOGL": rows})
                self.assertEqual(self.submits, [])
                self.assertEqual(self.of_type(result, NOTICE, "earnings_close"), [])

    def test_a_failed_lookup_names_nothing_and_closes_nothing(self):
        result = self.run_lost(setup_condor(), fail={"GOOGL"})
        self.assertEqual(self.submits, [])
        self.assertEqual(self.of_type(result, NOTICE), [])
        self.assertEqual(len(self.of_type(result, "earnings_lookup_failed")), 1)

    def test_with_no_exempt_setup_configured_there_is_nothing_to_miss(self):
        legs = setup_condor()
        with patch.dict(system_config.BASE_CONFIG, {"OPTIONS_EARNINGS_CLOSE_EXEMPT_SETUPS": ()}):
            result = self.run_lost(legs)
        self.assertEqual(self.submits, self.sold_closes(legs))
        self.assertEqual(self.of_type(result, NOTICE), [])


class TestTheNoticeDecidesNothing(_LostStateCase):
    def test_a_failing_notice_never_stops_the_close(self):
        legs = setup_condor()
        with patch.object(om, "_unrecorded_structure_notices", side_effect=RuntimeError("notice broke")):
            result = self.run_lost(legs)
        self.assertEqual(self.submits, self.sold_closes(legs))
        self.assertEqual(len(self.of_type(result, "earnings_close")), 2)
        self.assertEqual(self.of_type(result, NOTICE), [])


if __name__ == "__main__":
    unittest.main()
