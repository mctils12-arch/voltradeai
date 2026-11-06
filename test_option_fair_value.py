"""FAIR VALUE BEFORE PAYING (2026-11-06, option_fair_value.py): an option entry
is compared with a realized-volatility fair value before it is sent.

Human refocus spec §5: "estimate fair value before paying, do not overpay on
IV". options_execution.select_contract calls option_fair_value.apply() last,
so every entry path (tier dispatcher, scanner, evaluate_and_execute) is judged
before submit_options_order (frozen) ever sees the contract.

Pinned here:
  - settings: the system_config defaults (enforce, +25% over fair for a
    bought leg, 100% of fair for a sold one, 50/50 RV20/RV60), bounds, modes;
  - fair vol = 0.5 x RV20 + 0.5 x RV60 from the selector's vol surface, else
    from daily bars; either alone when only one exists;
  - bought premium over the limit is refused (enforce) with a plain-English
    reason naming the IV, the fair vol, both RVs and the prices; at or under
    the limit it passes; sold premium below fair is refused; the judged legs
    of each structure (a condor's shorts, a debit spread's long leg, a
    straddle's two legs), never the wings or the financing leg;
  - an IV missing from the quote is solved from the mid; no IV and no quote,
    or no realized-vol history: allowed, unpriced, and said so;
  - flag mode records and never refuses; off mode does nothing;
  - every judged entry is one JSONL line; a contract already in error is
    untouched;
  - select_contract runs the check last, on every strategy, and its error is
    what every caller already treats as "no order".
"""
import inspect
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import option_fair_value as fv  # noqa: E402
import system_config  # noqa: E402
import vol_surface  # noqa: E402

SURFACE = {"vrp": {"realized_vol_20d": 0.28, "realized_vol_60d": 0.32}}   # fair = 0.30
SPOT = 100.0


def single(side="buy", iv=0.33, limit=None, strike=105.0, dte=30, opt="call"):
    """A single-leg contract as select_contract builds it. Its limit is the
    Black-Scholes price at `iv` (so the IV the limit implies is `iv`) unless
    given."""
    if limit is None:
        limit = round(vol_surface.bs_price(SPOT, strike, dte / 365.0, vol_surface.RISK_FREE_RATE, iv, opt), 2)
    cp = "C" if opt == "call" else "P"
    return {"occ_symbol": f"ABC261218{cp}{int(strike * 1000):08d}", "strike": strike, "expiry": "2026-12-18",
            "option_type": opt, "side": side, "qty": 1, "limit_price": limit, "iv": iv, "days_to_expiry": dte,
            "error": None}


def row(occ, iv, bid, ask, strike, opt, dte=35):
    return {"occ_symbol": occ, "iv": iv, "bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 2), "strike": strike,
            "option_type": opt, "days_to_expiry": dte}


class _LogCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = patch.object(fv, "DATA_DIR", tmp.name)
        p.start()
        self.addCleanup(p.stop)
        self.dir = tmp.name

    def lines(self):
        path = os.path.join(self.dir, fv.LOG_FILE)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]


class TestSettings(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(system_config.BASE_CONFIG["OPTIONS_FAIR_VALUE"],
                         {"mode": "enforce", "max_buy_premium": 0.25, "min_sell_ratio": 1.00, "rv_short_weight": 0.5})
        self.assertEqual(fv.settings(), {"mode": "enforce", "max_buy_premium": 0.25, "min_sell_ratio": 1.0, "rv_short_weight": 0.5})

    def test_bad_values_keep_the_default(self):
        with patch.dict(system_config.BASE_CONFIG, {"OPTIONS_FAIR_VALUE": {"mode": "sometimes", "max_buy_premium": -1,
                                                                           "min_sell_ratio": "x", "rv_short_weight": 0.8}}):
            self.assertEqual(fv.settings(), {"mode": "enforce", "max_buy_premium": 0.25, "min_sell_ratio": 1.0, "rv_short_weight": 0.8})
        with patch.dict(system_config.BASE_CONFIG, {"OPTIONS_FAIR_VALUE": None}):
            self.assertEqual(fv.settings()["mode"], "enforce")

    def test_fair_vol_blend(self):
        self.assertAlmostEqual(fv.fair_vol(0.28, 0.32, 0.5), 0.30)
        self.assertAlmostEqual(fv.fair_vol(0.28, None, 0.5), 0.28)
        self.assertAlmostEqual(fv.fair_vol(None, 0.32, 0.5), 0.32)
        self.assertIsNone(fv.fair_vol(None, None, 0.5))


class TestSingleLegs(_LogCase):
    def test_a_bought_call_within_25_percent_of_fair_passes(self):
        c = single(iv=0.37)
        out = fv.check(c, "buy_call", "ABC", SPOT, SURFACE)
        self.assertEqual((out["verdict"], out["allowed"]), ("fair", True))
        self.assertAlmostEqual(out["fair_vol"], 0.30)
        self.assertIn("FAIR VALUE OK: buying ABC261218C00105000 at IV 37% vs fair 30% (+23%) — paying "
                      f"${c['limit_price']:.2f} vs fair $", out["reason"])
        self.assertEqual(out["legs"][0]["iv_source"], f"at the ${c['limit_price']:.2f} limit")

    def test_a_bought_call_more_than_25_percent_over_fair_is_refused_in_plain_english(self):
        c = single(iv=0.45)
        limit = c["limit_price"]
        fair_px = vol_surface.bs_price(SPOT, 105.0, 30 / 365.0, vol_surface.RISK_FREE_RATE, 0.30, "call")
        out = fv.apply(c, "buy_call", "ABC", SPOT, SURFACE)
        self.assertEqual(out["fair_value"]["verdict"], "overpriced")
        self.assertEqual(out["error"], "FAIR VALUE: refused — buying ABC261218C00105000 at IV 45% is 50% over the realized-vol "
                                       "fair 30% (20-day 28%, 60-day 32%; limit +25%) — paying "
                                       f"${limit:.2f} vs fair ${fair_px:.2f}")
        self.assertEqual(self.lines()[0]["verdict"], "overpriced")
        self.assertFalse(self.lines()[0]["allowed"])

    def test_the_limit_not_the_quote_is_what_is_judged(self):
        """The quote says 30% (fair) but the order would pay a price that implies 45%."""
        c = single(iv=0.45)
        c["iv"] = 0.30
        self.assertEqual(fv.check(c, "buy_call", "ABC", SPOT, SURFACE)["verdict"], "overpriced")

    def test_a_sold_put_below_fair_is_refused_and_above_fair_passes(self):
        low = fv.apply(single(side="sell", iv=0.26, opt="put", strike=95.0), "sell_cash_secured_put", "ABC", SPOT, SURFACE)
        self.assertEqual(low["fair_value"]["verdict"], "underpriced")
        self.assertIn("FAIR VALUE: refused — selling ABC261218P00095000 at IV 26% is below the realized-vol fair 30% "
                      "(20-day 28%, 60-day 32%; floor 100% of fair) — receiving $", low["error"])
        ok = fv.apply(single(side="sell", iv=0.31, opt="put", strike=95.0), "sell_cash_secured_put", "ABC", SPOT, SURFACE)
        self.assertIsNone(ok["error"])
        self.assertEqual(ok["fair_value"]["verdict"], "fair")

    def test_a_limit_the_solver_cannot_price_falls_back_to_the_quoted_iv(self):
        c = single(iv=0.45, limit=150.0)   # above the stock price: no volatility prices it
        out = fv.check(c, "buy_call", "ABC", SPOT, SURFACE)
        self.assertEqual(out["legs"][0]["iv_source"], "quoted")
        self.assertEqual(out["verdict"], "overpriced")

    def test_a_structure_leg_without_an_iv_is_solved_from_its_mid(self):
        chain = [row("XYZ261218P00100000", 0.0, 3.00, 3.20, 100, "put"), row("XYZ261218P00095000", 0.2, 1.0, 1.1, 95, "put")]
        c = {"strategy": "bear_put_spread", "long_leg": "XYZ261218P00100000", "short_leg": "XYZ261218P00095000",
             "net_debit": 2.1, "limit_price": 2.1, "qty": 1, "error": None}
        with patch("vol_surface.implied_vol", return_value=0.50) as solve:
            out = fv.check(c, "bear_put_spread", "XYZ", SPOT, SURFACE, chain)
        solve.assert_called_once()
        self.assertAlmostEqual(solve.call_args.args[0], 3.10)
        self.assertEqual(out["legs"][0]["iv_source"], "solved from the mid")
        self.assertEqual(out["verdict"], "overpriced")

    def test_no_iv_and_no_quote_is_allowed_unpriced(self):
        c = single(iv=0.0, limit=0)
        out = fv.apply(c, "buy_call", "ABC", 100.0, SURFACE)
        self.assertIsNone(out["error"])
        self.assertEqual(out["fair_value"]["verdict"], "unpriced")
        self.assertIn("entry allowed unpriced", out["fair_value"]["reason"])

    def test_no_realized_vol_history_is_allowed_unpriced(self):
        with patch("vol_surface._fetch_historical_bars", return_value=[]), \
                patch("vol_surface._compute_realized_vol", return_value=0.0):
            out = fv.apply(single(iv=0.90), "buy_call", "ABC", 100.0, {"error": "no chain"})
        self.assertIsNone(out["error"])
        self.assertEqual(out["fair_value"]["verdict"], "unpriced")
        self.assertIn("not enough daily bars", out["fair_value"]["reason"])

    def test_without_a_surface_the_bars_give_the_fair_vol(self):
        with patch("vol_surface._fetch_historical_bars", return_value=[{"c": 1}] * 90), \
                patch("vol_surface._compute_realized_vol", side_effect=lambda bars, w: 0.20 if w == 20 else 0.40):
            out = fv.check(single(iv=0.33), "buy_call", "ABC", 100.0, {})
        self.assertEqual(out["rv_source"], "daily bars")
        self.assertAlmostEqual(out["fair_vol"], 0.30)


class TestStructures(_LogCase):
    CHAIN = [
        row("XYZ261218C00110000", 0.24, 1.00, 1.10, 110, "call"),   # short call
        row("XYZ261218C00115000", 0.55, 0.40, 0.50, 115, "call"),   # long call wing (IV never judged)
        row("XYZ261218P00090000", 0.36, 1.20, 1.30, 90, "put"),     # short put
        row("XYZ261218P00085000", 0.60, 0.50, 0.60, 85, "put"),     # long put wing
    ]

    def condor(self):
        return {"strategy": "iron_condor", "short_call": "XYZ261218C00110000", "long_call": "XYZ261218C00115000",
                "short_put": "XYZ261218P00090000", "long_put": "XYZ261218P00085000", "net_credit": 1.2,
                "limit_price": 1.2, "qty": 1, "error": None}

    def test_a_condor_is_judged_on_its_two_short_legs_only(self):
        out = fv.apply(self.condor(), "iron_condor", "XYZ", 100.0, SURFACE, self.CHAIN)
        self.assertEqual([l["occ_symbol"] for l in out["fair_value"]["legs"]], ["XYZ261218C00110000", "XYZ261218P00090000"])
        self.assertEqual(out["fair_value"]["verdict"], "underpriced", "the 24% short call sells below the 30% fair")
        self.assertIn("selling XYZ261218C00110000 at IV 24%", out["error"])
        self.assertNotIn("C00115000", out["error"], "the wings' rich IV is insurance, never judged")

    def test_a_debit_spread_is_judged_on_its_long_leg_only(self):
        chain = [row("XYZ261218P00100000", 0.33, 3.00, 3.20, 100, "put"), row("XYZ261218P00095000", 0.10, 1.00, 1.10, 95, "put")]
        c = {"strategy": "bear_put_spread", "long_leg": "XYZ261218P00100000", "short_leg": "XYZ261218P00095000",
             "net_debit": 2.1, "limit_price": 2.1, "qty": 1, "error": None}
        out = fv.apply(c, "bear_put_spread", "XYZ", 100.0, SURFACE, chain)
        self.assertIsNone(out["error"])
        self.assertEqual([l["occ_symbol"] for l in out["fair_value"]["legs"]], ["XYZ261218P00100000"])
        self.assertEqual(out["fair_value"]["legs"][0]["price"], 3.1, "a structure's leg is priced at its mid")

    def test_a_long_straddle_is_judged_on_both_legs(self):
        chain = [row("XYZ261218C00100000", 0.30, 3.0, 3.2, 100, "call"), row("XYZ261218P00100000", 0.45, 3.0, 3.2, 100, "put")]
        c = {"strategy": "buy_straddle", "call_leg": "XYZ261218C00100000", "put_leg": "XYZ261218P00100000",
             "limit_price": 6.2, "qty": 1, "error": None}
        out = fv.apply(c, "buy_straddle", "XYZ", 100.0, SURFACE, chain)
        self.assertEqual(out["fair_value"]["verdict"], "overpriced")
        self.assertIn("buying XYZ261218P00100000 at IV 45% is 50% over", out["error"])

    def test_a_leg_missing_from_the_chain_and_without_a_quote_is_allowed_unpriced(self):
        c = self.condor()
        out = fv.apply(c, "iron_condor", "XYZ", 100.0, SURFACE, [])
        self.assertIsNone(out["error"])
        self.assertEqual(out["fair_value"]["verdict"], "unpriced")

    def test_every_strategy_select_contract_dispatches_is_known(self):
        import options_execution
        src = inspect.getsource(options_execution.select_contract)
        dispatched = set(__import__("re").findall(r'strategy == "([a-z_]+)"', src))
        self.assertTrue(dispatched)
        self.assertEqual(dispatched - set(fv.JUDGED_LEGS), set(), "a strategy the fair-value check cannot judge")


class TestModes(_LogCase):
    def test_flag_records_and_never_refuses(self):
        with patch.dict(system_config.BASE_CONFIG, {"OPTIONS_FAIR_VALUE": {**system_config.BASE_CONFIG["OPTIONS_FAIR_VALUE"], "mode": "flag"}}):
            out = fv.apply(single(iv=0.60), "buy_call", "ABC", 100.0, SURFACE)
        self.assertIsNone(out["error"])
        self.assertEqual(out["fair_value"]["verdict"], "overpriced")
        self.assertIn("flagged (not enforced)", out["fair_value"]["reason"])
        self.assertTrue(self.lines()[0]["allowed"])

    def test_off_does_nothing(self):
        with patch.dict(system_config.BASE_CONFIG, {"OPTIONS_FAIR_VALUE": {**system_config.BASE_CONFIG["OPTIONS_FAIR_VALUE"], "mode": "off"}}):
            c = single(iv=0.60)
            out = fv.apply(c, "buy_call", "ABC", 100.0, SURFACE)
        self.assertNotIn("fair_value", out)
        self.assertIsNone(out["error"])
        self.assertEqual(self.lines(), [])

    def test_a_contract_already_in_error_is_untouched(self):
        c = {"error": "No liquid options contracts"}
        self.assertIs(fv.apply(c, "buy_call", "ABC", 100.0, SURFACE), c)
        self.assertEqual(c, {"error": "No liquid options contracts"})
        self.assertEqual(self.lines(), [])


class TestWiredIntoSelectContract(_LogCase):
    def test_select_contract_runs_the_check_last_on_the_built_contract(self):
        import options_execution as oe
        built = single(iv=0.60)
        with patch.object(oe, "_fetch_option_chain", return_value=[{"occ_symbol": "x"}]), \
                patch.object(oe, "_resolve_options_size_pct", return_value=0.05), \
                patch.object(oe, "_is_liquid", return_value=True), \
                patch.object(oe, "_select_buy_call", return_value=built), \
                patch("vol_surface.get_surface_score", return_value=SURFACE):
            out = oe.select_contract("ABC", "buy_call", 100.0, 100000)
        self.assertTrue(out["error"].startswith("FAIR VALUE: refused"), out["error"])
        self.assertEqual(self.lines()[0]["strategy"], "buy_call")

    def test_a_crash_in_the_check_never_blocks_the_entry(self):
        import options_execution as oe
        built = single(iv=0.60)
        with patch.object(oe, "_fetch_option_chain", return_value=[{"occ_symbol": "x"}]), \
                patch.object(oe, "_resolve_options_size_pct", return_value=0.05), \
                patch.object(oe, "_is_liquid", return_value=True), \
                patch.object(oe, "_select_buy_call", return_value=built), \
                patch("vol_surface.get_surface_score", return_value=SURFACE), \
                patch.object(fv, "check", side_effect=RuntimeError("boom")):
            out = oe.select_contract("ABC", "buy_call", 100.0, 100000)
        self.assertIsNone(out["error"])


if __name__ == "__main__":
    unittest.main()
