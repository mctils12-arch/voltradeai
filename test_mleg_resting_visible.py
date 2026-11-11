"""RESTING MULTI-LEG ORDERS (2026-11-06), Python half: a resting multi-leg
OPENING order reserves its max-loss collateral in the cash reserve, and the
options scanner treats its underlying and legs as taken (the Node half, the
slot count / duplicate check / CSP gate / 12-minute sweep, is
server/mlegOpenOrders.test.ts).

An Alpaca mleg order has an empty top-level side and symbol; with nested=true
its legs come back. position_mgmt.open_order_commitments counted it as $0, and
bot_engine's scan read the open orders without legs.
"""
import inspect
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import position_mgmt as pm  # noqa: E402


def leg(symbol, side, intent):
    return {"symbol": symbol, "side": side, "position_intent": intent, "ratio_qty": "1"}


def mleg(legs, qty=3, filled=0, limit="-0.71"):
    return {"id": "m-1", "order_class": "mleg", "symbol": "", "side": "", "type": "limit", "qty": str(qty),
            "filled_qty": str(filled), "limit_price": limit, "status": "new", "legs": legs}


CONDOR = [leg("IREN261218C00016000", "sell", "sell_to_open"), leg("IREN261218C00017000", "buy", "buy_to_open"),
          leg("IREN261218P00012000", "sell", "sell_to_open"), leg("IREN261218P00011000", "buy", "buy_to_open")]
PUT_SPREAD = [leg("SPY261211P00560000", "sell", "sell_to_open"), leg("SPY261211P00532000", "buy", "buy_to_open")]
CALL_DEBIT = [leg("XYZ261218C00100000", "buy", "buy_to_open"), leg("XYZ261218C00110000", "sell", "sell_to_open")]
CALENDAR = [leg("XYZ270115C00100000", "buy", "buy_to_open"), leg("XYZ261218C00100000", "sell", "sell_to_open")]
SHORT_STRADDLE = [leg("XYZ261218C00100000", "sell", "sell_to_open"), leg("XYZ261218P00100000", "sell", "sell_to_open")]


class TestCollateral(unittest.TestCase):
    def test_a_resting_credit_condor_reserves_its_max_loss(self):
        # wings 1 wide: the worst expiry loses $1 a share on one side, x 100 x 3 units
        self.assertEqual(pm.mleg_order_commitment(mleg(CONDOR, qty=3)), 300.0)
        self.assertEqual(pm.mleg_order_commitment(mleg(PUT_SPREAD, qty=2, limit="-1.35")), 28 * 100 * 2)

    def test_the_limit_sign_does_not_change_it(self):
        """The same number before (+0.71, the abs bug) and after (-0.71) the limit-sign fix."""
        self.assertEqual(pm.mleg_order_commitment(mleg(CONDOR, limit="0.71")),
                         pm.mleg_order_commitment(mleg(CONDOR, limit="-0.71")))

    def test_a_debit_structure_reserves_its_debit(self):
        self.assertAlmostEqual(pm.mleg_order_commitment(mleg(CALL_DEBIT, qty=2, limit="3.05")), 610.0)
        self.assertAlmostEqual(pm.mleg_order_commitment(mleg(CALENDAR, qty=1, limit="1.40")), 140.0)

    def test_uncapped_short_calls_commit_every_short_strike(self):
        self.assertEqual(pm.mleg_order_commitment(mleg(SHORT_STRADDLE, qty=1)), 2 * 100 * 100)

    def test_only_the_unfilled_units_and_only_opening_orders_count(self):
        self.assertEqual(pm.mleg_order_commitment(mleg(CONDOR, qty=3, filled=2)), 100.0)
        close = mleg([dict(l, position_intent=l["position_intent"].replace("_open", "_close")) for l in CONDOR])
        self.assertEqual(pm.mleg_order_commitment(close), 0.0, "a group close frees cash")
        self.assertEqual(pm.mleg_order_commitment(dict(mleg(CONDOR), legs=None)), 0.0, "no legs (not nested): unpriced")

    def test_the_reserve_counts_it_as_a_working_order(self):
        o = mleg(CONDOR, qty=3)
        self.assertEqual(pm.open_order_commitments([o], []), 300.0)
        snap = pm.satellite_headroom(100000.0, [], "NEUTRAL", open_orders=[o])
        self.assertEqual(snap["pending_orders"], 300.0)
        self.assertEqual(pm.satellite_headroom(100000.0, [], "NEUTRAL", open_orders=[])["pending_orders"], 0.0)

    def test_the_resting_underlyings(self):
        self.assertEqual(pm.mleg_opening_underlyings([mleg(CONDOR), mleg(PUT_SPREAD), {"symbol": "AAPL", "side": "buy"}]),
                         ["IREN", "SPY"])
        self.assertEqual(len(pm.mleg_opening_legs(mleg(CONDOR))), 4)


class TestTheScannerSeesIt(unittest.TestCase):
    def test_open_orders_are_read_with_their_legs(self):
        import bot_engine
        resp = MagicMock()
        resp.json.return_value = []
        with patch.object(bot_engine.requests, "get", return_value=resp) as get:
            bot_engine._open_order_rows()
        self.assertEqual(get.call_args.kwargs["params"], {"status": "open", "limit": 500, "nested": "true"})

    def test_the_options_scanner_skips_a_resting_structures_underlying_and_counts_its_legs(self):
        import bot_engine
        src = inspect.getsource(bot_engine._scan_market_inner)
        self.assertIn("_resting_underlyings = _pm.mleg_opening_underlyings(_scan_orders)", src)
        self.assertIn('current_tickers=current_tickers + [t["ticker"] for t in trades] + _resting_underlyings,', src)
        self.assertIn(") + _resting_legs", src)
        self.assertLess(src.index("_scan_orders = []"), src.index("_scan_orders = _open_order_rows()"))

    def test_get_options_trades_skips_a_listed_underlying(self):
        import options_scanner as osc
        opp = {"ticker": "IREN", "setup": "high_iv_premium_sale", "options_strategy": "iron_condor", "score": 90,
               "price": 14.0, "action_label": "SELL IC", "side": "sell"}
        with patch.object(osc, "scan_options", return_value={"opportunities": [dict(opp)]}), \
                patch.dict(sys.modules, {"vol_surface": MagicMock(get_surface_score=lambda t: {})}), \
                patch.object(osc, "_dealer_gamma_gate", return_value={"action": "allow"}):
            self.assertEqual(osc.get_options_trades(100000, current_tickers=["IREN"]), [])
            self.assertEqual([t["ticker"] for t in osc.get_options_trades(100000, current_tickers=[])], ["IREN"])


class TestEvaluateAndExecuteReportsItsLegs(unittest.TestCase):
    def test_the_registered_legs_are_returned(self):
        import options_execution as oe
        contract = {"strategy": "bear_put_spread", "long_leg": "XYZ261218P00100000", "short_leg": "XYZ261218P00095000",
                    "qty": 1, "net_debit": 2.0, "limit_price": 2.0, "max_loss": 200.0, "error": None}
        with patch.object(oe, "should_use_options", return_value={"use_options": True, "strategy": "bear_put_spread",
                                                                   "edge_pct": 1.0, "reason": "t"}), \
                patch.object(oe, "select_contract", return_value=contract), \
                patch.object(oe, "submit_options_order", return_value={"status": "new", "order_id": "e-1"}), \
                patch("entry_shadow.is_shadowed", return_value=False), \
                patch.object(oe, "register_contract_entry", return_value=["XYZ261218P00095000", "XYZ261218P00100000"]) as reg:
            out = oe.evaluate_and_execute({"ticker": "XYZ", "price": 100.0}, 100000, [])
        self.assertEqual(out["legs"], ["XYZ261218P00095000", "XYZ261218P00100000"])
        self.assertEqual(reg.call_args.kwargs["order_id"], "e-1")


if __name__ == "__main__":
    unittest.main()
