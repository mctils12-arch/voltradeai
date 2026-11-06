"""LEFTOVER LONG LEG (2026-11-05): a bought option whose structure is gone is closed.

Live: GOOGL261127P00300000, the long put wing of an earnings iron condor whose
other three legs were closed 10-29, sat alone for a week with no rule able to
exit it ($4.70 -> $22.49 -> sold at $10.65 by the 21-DTE rule on 11-05).

Long options carry a negative expected return (the variance risk premium), and
no peer-reviewed test supports profit targets or trailing stops on premium. The
research-backed rule is to close leftover wings.

Pinned:
- a leftover (a long leg, the only option on its underlying, not a deliberate
  standalone long) is sold at the bid during regular hours, after the hold gate,
  whether its state says "unknown" or still records the condor it came from;
- a long leg whose short partner is still open is NEVER closed by this rule:
  closing the wing would leave the short naked;
- buy_call / buy_put / convexity_hedge legs and a straddle's two long legs are
  not leftovers;
- market closed -> deferred, no order; no bid -> HOLD, no order; short legs are
  untouched;
- trade_feedback: an unknown-strategy leftover records its exit like every
  single-leg rule; a wing still recording its multi-leg strategy does not
  (KNOWN BROKEN #12(c): multi-leg exits are not attributed).
"""
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options_manager as om  # noqa: E402
from ml_model_v2 import OPTIONS_EXIT_REASONS  # noqa: E402
from test_options_exit_market_hours import Broker, _ManagerCase, leg  # noqa: E402

DTE = 35   # past the 21-DTE rule, so only the leftover rule can act


def _held(lg, days=7):
    held = (datetime.now() - timedelta(days=days)).isoformat()
    lg["state"].update({"entry_timestamp": held, "entry_date": held[:10]})
    return lg


def googl_wing(strategy="unknown", bid=10.60, ask=10.90, held_days=7):
    """The live case's leg (2 x GOOGL 300P bought at 4.70, quoted ~10.75),
    moved out to 35 DTE so the 21-DTE rule stays out of the way."""
    return _held(leg("GOOGL", DTE, "P", 300, 2, "long", 4.70, 10.75, bid, ask, -0.30, 0.01, strategy,
                     ("sell", bid)), held_days)


def googl_short(strategy="unknown"):
    return _held(leg("GOOGL", DTE, "P", 310, 2, "short", 9.00, 14.00, 13.90, 14.10, -0.40, 0.01,
                     strategy, None))


class TestLeftoverLongLeg(_ManagerCase):
    def _run(self, legs, is_open=True):
        self.load({str(i): l for i, l in enumerate(legs)})
        broker = Broker([l["pos"] for l in legs], is_open=is_open)
        return self.run_manager(broker, allow_rolls=True)

    def _leftover_actions(self, result):
        return [a for a in result["actions"] if a.get("type") == "leftover_long_close"]

    def test_the_googl_case_is_sold_at_the_bid(self):
        wing = googl_wing()
        result = self._run([wing])
        self.assertEqual(self.submits, [(wing["occ"], 2, "sell", 10.60)])
        act = self._leftover_actions(result)
        self.assertEqual(len(act), 1)
        self.assertIn("LEFTOVER LONG LEG", act[0]["reason"])
        self.assertNotIn(wing["occ"], self.read_state(), "state popped like every single-leg close")
        self.track_fill.assert_called_once()
        self.assertEqual(self.track_fill.call_args[0][0]["exit_reason"], "leftover_long_close")

    def test_a_lone_wing_still_recording_its_condor_is_a_leftover_not_a_one_leg_group(self):
        """As a one-leg iron_condor "group" no group rule could exit a long wing."""
        wing = googl_wing(strategy="iron_condor")
        result = self._run([wing])
        self.assertEqual(self.submits, [(wing["occ"], 2, "sell", 10.60)])
        self.assertEqual(len(self._leftover_actions(result)), 1)
        self.track_fill.assert_not_called()  # multi-leg exits are not attributed (KNOWN BROKEN #12(c))

    def test_a_long_leg_whose_short_partner_is_open_is_never_closed_by_this_rule(self):
        for strategy in ("unknown", "iron_condor"):
            self.submits.clear()
            wing, short = googl_wing(strategy=strategy), googl_short(strategy=strategy)
            result = self._run([wing, short])
            self.assertNotIn(wing["occ"], [s[0] for s in self.submits],
                             f"[{strategy}] closing the wing would leave the short 310P naked")
            self.assertEqual(self._leftover_actions(result), [])

    def test_deliberate_standalone_longs_are_exempt(self):
        # The wing is up 129% (4.70 -> 10.75): with BOUGHT-OPTION PROFIT TAKING
        # (2026-11-06) on, a deliberate long that far up is scaled out by THAT
        # rule (next test). Off here, so this test isolates the leftover rule.
        import system_config
        with patch.dict(system_config.BASE_CONFIG["OPTIONS_PROFIT_TAKING"], {"enabled": False}):
            for strategy in om.STANDALONE_LONG_STRATEGIES:
                self.submits.clear()
                result = self._run([googl_wing(strategy=strategy)])
                self.assertEqual(self.submits, [], f"{strategy} is a deliberate long, not a leftover")
                self.assertEqual(self._leftover_actions(result), [])

    def test_a_deliberate_long_far_up_is_scaled_out_by_profit_taking_not_closed_as_a_leftover(self):
        """BOUGHT-OPTION PROFIT TAKING (2026-11-06, default on): the same buy_call
        wing up 129% sells half (1 of 2) at the bid; the leftover rule stays out."""
        wing = googl_wing(strategy="buy_call")
        result = self._run([wing])
        self.assertEqual(self.submits, [(wing["occ"], 1, "sell", 10.60)])
        self.assertEqual(self._leftover_actions(result), [])
        self.assertEqual([a["type"] for a in result["actions"] if a.get("action") == "CLOSE"], ["bought_scale_out"])

    def test_a_straddles_two_long_legs_are_not_leftovers(self):
        call = _held(leg("KMI", DTE, "C", 30, 1, "long", 1.00, 1.10, 1.05, 1.15, 0.50, 0.05, "buy_straddle", None))
        put = _held(leg("KMI", DTE, "P", 30, 1, "long", 1.00, 0.90, 0.85, 0.95, -0.50, 0.05, "buy_straddle", None))
        result = self._run([call, put])
        self.assertEqual(self._leftover_actions(result), [])

    def test_market_closed_defers_without_an_order(self):
        wing = googl_wing()
        result = self._run([wing], is_open=False)
        self.assertEqual(self.submits, [])
        deferred = self._leftover_actions(result)
        self.assertEqual(deferred[0]["order"]["status"], "deferred")
        self.assertIn(wing["occ"], self.read_state(), "the leg's state is kept while deferred")
        self.track_fill.assert_not_called()

    def test_no_bid_holds_and_judges_again(self):
        wing = googl_wing(bid=0.0, ask=11.0)
        result = self._run([wing])
        self.assertEqual(self.submits, [])
        self.assertTrue([a for a in result["actions"] if a.get("type") == "leftover_long_no_bid"])
        self.assertIn(wing["occ"], self.read_state())

    def test_the_hold_gate_still_applies(self):
        wing = googl_wing()
        wing["state"]["entry_timestamp"] = (datetime.now() - timedelta(minutes=10)).isoformat()
        self._run([wing])
        self.assertEqual(self.submits, [])

    def test_a_lone_short_leg_is_untouched(self):
        csp = _held(leg("PFE", 40, "P", 24, 1, "short", 0.80, 0.70, 0.68, 0.72, -0.20, 0.01,
                        "sell_cash_secured_put", None))
        result = self._run([csp])
        self.assertEqual(self.submits, [])
        self.assertEqual(self._leftover_actions(result), [])

    def test_the_exit_reason_is_an_options_exit_reason(self):
        self.assertIn("leftover_long_close", OPTIONS_EXIT_REASONS)
