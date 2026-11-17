"""REPAIR 2026-11-12 — EARNINGS CLOSE: a sold option on an adjusted OCC root is
looked up under its underlying.

What was wrong: after a corporate action a contract's OCC root carries a number
(OCC Information Memo #50867: "Option Symbol: ROOT changes to ROOT1", which
still delivers ROOT shares). The earnings close asked Finnhub for the raw root
("IONQ1"), which is no company, got "no report", and left the sold option open
through IONQ's report (research/open_questions.md, 2026-11-05 follow-ups,
"[EARNINGS CLOSE] Residuals").

Pinned:
- the one OCC parser (options_manager._parse_occ_symbol) names the underlying:
  the letters of a root that is letters followed only by digits; every other
  root, itself. `ticker` (the root) and the other fields are unchanged;
- a sold leg on an adjusted root is looked up as its underlying and bought back
  on its close session, with or without recorded state (a renamed contract has
  none under its new symbol). The order names the contract unchanged;
- a strategy group on an adjusted root is closed whole;
- the standard and the adjusted series of one company share one lookup, one
  cache entry and one lookup-failure warning;
- the deploy backfill names the underlying;
- a standard root is asked, closed and reported exactly as before.
Mocked like test_options_earnings_close.py: no network, state in a temp dir,
the New York clock fixed at Thursday 2026-11-05 10:30.
"""
import json
import os
import sys
import unittest
from datetime import timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_manager as om  # noqa: E402
from test_options_earnings_close import EarningsBroker, _EarningsCase, condor, csp, expiry_of, iren_put  # noqa: E402

IONQ_BMO_TOMORROW = {"IONQ": [("2026-11-06", "bmo")]}


class TestTheParserNamesTheUnderlying(unittest.TestCase):
    def test_an_adjusted_root_gives_its_letters_and_keeps_every_other_field(self):
        for root, underlying in (("IONQ1", "IONQ"), ("ROOT1", "ROOT"), ("CSGP1", "CSGP"), ("BB2", "BB"),
                                 ("F1", "F"), ("ABCDE1", "ABCDE"), ("AAPL12", "AAPL")):
            with self.subTest(root=root):
                parsed = om._parse_occ_symbol(f"{root}260418P00037500")
                self.assertEqual(parsed["underlying"], underlying)
                self.assertEqual((parsed["ticker"], parsed["expiry_date"], parsed["option_type"], parsed["strike"]),
                                 (root, "2026-04-18", "put", 37.5), "the root still names the contract")

    def test_a_ticker_is_never_cut(self):
        for root in ("AAPL", "SPY", "GOOGL", "F", "BRKB", "SPXW", "ABCDEF"):
            with self.subTest(root=root):
                parsed = om._parse_occ_symbol(f"{root}260418C00250000")
                self.assertEqual((parsed["ticker"], parsed["underlying"]), (root, root))

    def test_a_root_that_is_not_letters_then_digits_is_left_alone(self):
        for root in ("2SPX", "1AAPL", "A1B", "123", "ionq1"):
            with self.subTest(root=root):
                self.assertEqual(om._parse_occ_symbol(f"{root}260418C00250000")["underlying"], root)

    def test_an_unparseable_symbol_is_still_an_empty_dict(self):
        for bad in ("", "XX", "260418C00250000"):
            with self.subTest(symbol=bad):
                self.assertEqual(om._parse_occ_symbol(bad), {})


class TestAdjustedRootIsClosedBeforeItsReport(_EarningsCase):
    def test_a_sold_leg_is_looked_up_as_its_underlying_and_bought_back(self):
        put = csp("IONQ1", 36, 18)
        result = self.run_book([put], reports=IONQ_BMO_TOMORROW)
        self.assertEqual(self.broker.finnhub, ["IONQ"], "the company, not the adjusted root")
        self.assertEqual(self.submits, [(put["occ"], 1, "buy", 0.92)], "the contract symbol is untouched")
        [act] = self.of_type(result, "earnings_close")
        self.assertEqual((act["action"], act["occ_symbol"], act["ticker"]), ("CLOSE", put["occ"], "IONQ1"))
        self.assertEqual((act["report_date"], act["report_hour"], act["close_session"]),
                         ("2026-11-06", "bmo", "2026-11-05"))
        self.assertTrue(act["reason"].startswith("EARNINGS CLOSE: IONQ reports 2026-11-06 before the open; "
                                                 "2026-11-05 is the last session before it"), act["reason"])
        self.assertEqual(self.track_fill.call_args[0][0]["exit_reason"], "earnings_close")

    def test_a_renamed_contract_with_no_recorded_state_is_closed_too(self):
        # a corporate action renames the contract, so nothing is recorded under its new symbol
        put = csp("IONQ1", 36, 18)
        self.load({"0": put})
        with open(self.state_path, "w") as f:
            json.dump({}, f)
        self.broker = EarningsBroker([put["pos"]], reports=IONQ_BMO_TOMORROW)
        result = self.run_manager(self.broker, allow_rolls=True)
        self.assertEqual(self.broker.finnhub, ["IONQ"])
        self.assertEqual(self.submits, [(put["occ"], 1, "buy", 0.92)])
        self.assertEqual([a["occ_symbol"] for a in self.of_type(result, "earnings_close")], [put["occ"]])

    def test_nothing_closes_when_the_underlying_has_no_report_in_the_options_life(self):
        put = csp("IONQ1", 36, 18)
        for label, rows in (("no report", []), ("next week", [("2026-11-12", "bmo")]),
                            ("the day after expiry", [((expiry_of(put) + timedelta(days=1)).isoformat(), "bmo")])):
            with self.subTest(label):
                om._EARNINGS_REPORT_CACHE.clear()
                result = self.run_book([put], reports={"IONQ": rows})
                self.assertEqual(self.broker.finnhub, ["IONQ"])
                self.assertEqual(self.submits, [])
                self.assertEqual(self.of_type(result, "earnings_close"), [])

    def test_a_group_on_an_adjusted_root_is_closed_whole(self):
        legs = condor(root="IONQ1")
        with patch.object(om, "_close_strategy_mleg",
                          return_value={"status": "submitted", "method": "mleg", "order_id": "m-1"}) as mleg:
            result = self.run_book(legs, reports=IONQ_BMO_TOMORROW)
        self.assertEqual(self.broker.finnhub, ["IONQ"])
        self.assertEqual(mleg.call_count, 1)
        self.assertEqual(sorted(p["symbol"] for p in mleg.call_args[0][0]), sorted(lg["occ"] for lg in legs))
        [act] = [a for a in result["actions"] if a["action"] == "CLOSE_STRATEGY"]
        self.assertEqual((act["type"], act["ticker"], act["close_session"]),
                         ("strategy_earnings_close", "IONQ1", "2026-11-05"))
        self.assertTrue(act["reason"].startswith("EARNINGS CLOSE: IONQ reports 2026-11-06 before the open"),
                        act["reason"])
        self.assertEqual(self.submits, [], "no single-leg close of a condor leg")

    def test_the_standard_and_the_adjusted_series_share_one_lookup(self):
        adjusted, standard = csp("IONQ1", 30, 17), csp("IONQ", 60, 40)
        result = self.run_book([adjusted, standard], reports=IONQ_BMO_TOMORROW)
        self.assertEqual(self.broker.finnhub, ["IONQ"], "two series, one company, one request")
        self.assertEqual(sorted(self.submits), sorted([(adjusted["occ"], 1, "buy", 0.92),
                                                       (standard["occ"], 1, "buy", 0.92)]))
        self.assertEqual(sorted(a["ticker"] for a in self.of_type(result, "earnings_close")), ["IONQ", "IONQ1"])
        self.assertEqual(list(om._EARNINGS_REPORT_CACHE), ["IONQ"])
        self.assertEqual(om._EARNINGS_REPORT_CACHE["IONQ"][1], expiry_of(standard), "the window reaches the later expiry")
        self.run_again()
        self.assertEqual(self.broker.finnhub, ["IONQ"], "cached for the ET day under the underlying")

    def test_a_failed_lookup_warns_once_under_the_underlying_and_lists_both_series(self):
        adjusted, standard = csp("IONQ1", 30, 17), csp("IONQ", 60, 40)
        result = self.run_book([adjusted, standard], reports=IONQ_BMO_TOMORROW, fail={"IONQ"})
        self.assertEqual(self.submits, [], "nothing is closed on a guess")
        [warn] = self.of_type(result, "earnings_lookup_failed")
        self.assertEqual((warn["ticker"], warn["legs"]), ("IONQ", sorted([adjusted["occ"], standard["occ"]])))

    def test_the_deploy_backfill_names_the_underlying(self):
        om._EARNINGS_RUN["backfill_done"] = False
        put = csp("IONQ1", 36, 18)
        result = self.run_book([put], reports=IONQ_BMO_TOMORROW)
        [summary] = self.of_type(result, "earnings_backfill")
        [row] = summary["legs"]
        self.assertEqual((row["occ_symbol"], row["underlying"], row["status"], row["report_date"], row["close_session"]),
                         (put["occ"], "IONQ", "due", "2026-11-06", "2026-11-05"))
        self.assertIn(f"{put['occ']}: IONQ reports 2026-11-06 before the open, close session 2026-11-05 (due now)",
                      summary["reason"])


class TestStandardRootIsUnchanged(_EarningsCase):
    def test_the_live_iren_leg_is_asked_closed_and_reported_as_before(self):
        put = iren_put()
        result = self.run_book([put], reports={"IREN": [("2026-11-05", "amc")]})
        self.assertEqual(self.broker.finnhub, ["IREN"])
        self.assertEqual(self.submits, [(put["occ"], 1, "buy", 2.70)])
        [act] = self.of_type(result, "earnings_close")
        self.assertEqual((act["ticker"], act["occ_symbol"], act["close_session"]), ("IREN", put["occ"], "2026-11-05"))
        self.assertTrue(act["reason"].startswith("EARNINGS CLOSE: IREN reports 2026-11-05 after the close; "
                                                 "2026-11-05 is the last session before it"), act["reason"])
        self.assertEqual(list(om._EARNINGS_REPORT_CACHE), ["IREN"])


if __name__ == "__main__":
    unittest.main()
