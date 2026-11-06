"""
[RULE-REVIEW] 2026-11-06 — the tax-aware entry filter (tax_filter.py; refocus
spec §6, pipeline step 8): the math, the verdicts (a marginal short-term trade
skipped, a strong one allowed), the off switch, the decision log, the three
gates and their wiring into bot_engine.scan_market and
options_execution.select_contract, and the parity of every constant the
filter mirrors from elsewhere.
"""
import json
import os
import re
import statistics
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone

import tax_filter as tf

HERE = os.path.dirname(os.path.abspath(__file__))

# the Kelly priors (position_sizing._get_historical_stats defaults), as bucket_stats shapes them
PRIOR_STATS = {
    "overall": {"win_rate": 0.55, "avg_win": 2.0, "avg_loss": 2.0, "total_trades": 0},
    "by_strategy": {
        "etf": {"win_rate": 0.558, "avg_win": 3.82, "avg_loss": 3.55, "trades": 0},
        "stocks": {"win_rate": 0.49, "avg_win": 4.19, "avg_loss": 4.25, "trades": 0},
        "csp_options": {"win_rate": 0.716, "avg_win": 0.69, "avg_loss": 0.46, "trades": 0},
        "vrp": {"win_rate": 0.60, "avg_win": 2.0, "avg_loss": 2.0, "trades": 0},
    },
}


def _stats(win_rate, avg_win, avg_loss, n=0):
    return {"bucket": "x", "winRate": win_rate, "avgWin": avg_win, "avgLoss": avg_loss, "n": n, "source": "test"}


def _read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TempLog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, tf.DECISIONS_FILE)

    def tearDown(self):
        self._tmp.cleanup()


class TestMath(unittest.TestCase):
    def test_expectancy_is_win_rate_times_win_minus_loss_rate_times_loss(self):
        self.assertAlmostEqual(tf.expectancy_pct(0.716, 0.69, 0.46), 0.716 * 0.69 - 0.284 * 0.46, places=12)
        # the loss is a magnitude either way round
        self.assertAlmostEqual(tf.expectancy_pct(0.5, 2.0, -1.0), 0.5, places=12)

    def test_assess_numbers_by_hand(self):
        cfg = tf.config({"ENABLED": True})
        a = tf.assess(1.0, 36.5, cost_pct=0.2, cfg=cfg)
        # net 0.8, short-term 24 % -> 0.608; SPY 10 %/yr x 0.1 yr = 1.0 -> 0.85 after 15 %
        self.assertEqual(a["netPct"], 0.8)
        self.assertEqual(a["afterTaxPct"], 0.608)
        self.assertEqual(a["spyPct"], 1.0)
        self.assertEqual(a["spyAfterTaxPct"], 0.85)
        self.assertEqual(a["edgePp"], round(0.608 - 0.85, 4))
        self.assertEqual(a["marginPp"], 0.2)      # 2 %/yr x 0.1 yr
        self.assertEqual(a["term"], "short")
        self.assertFalse(a["wouldAllow"])

    def test_more_than_a_year_is_long_term_and_taxed_at_the_long_rate(self):
        self.assertEqual(tf.holding_term(365), "short")
        self.assertEqual(tf.holding_term(366), "long")
        a = tf.assess(20.0, 400, cfg=tf.config({"SUBTRACT_COSTS": True}))
        self.assertEqual(a["term"], "long")
        self.assertEqual(a["taxRate"], 0.15)
        self.assertEqual(a["afterTaxPct"], 17.0)

    def test_the_premium_caps_a_sold_option_and_costs_can_be_turned_off(self):
        a = tf.assess(3.0, 10, cost_pct=0.5, cfg=tf.config({"SUBTRACT_COSTS": False}), max_gain_pct=1.5)
        self.assertTrue(a["capped"])
        self.assertEqual(a["netPct"], 1.5)
        self.assertEqual(a["costPct"], 0.0)


class TestVerdicts(unittest.TestCase):
    def test_a_marginal_short_term_trade_is_skipped(self):
        # +0.30 % in 7 days, 0.10 % costs: 0.20 x 0.76 = 0.152 after tax vs SPY 0.163 -> skipped
        row = tf.decide(ticker="mar", side="buy", instrument="stock", bucket="stocks",
                        stats=_stats(0.5, 1.0, 0.4), cost_pct=0.10, holding_days=7, source="test",
                        cfg=tf.config({"ENABLED": True}))
        self.assertFalse(row["allowed"])
        self.assertFalse(row["wouldAllow"])
        self.assertEqual(row["ticker"], "MAR")
        self.assertTrue(row["reason"].startswith("Skipped: "), row["reason"])
        self.assertIn("24% short-term tax", row["reason"])
        self.assertIn("SPY", row["reason"])
        self.assertIn("margin", row["reason"])

    def test_a_strong_trade_is_allowed(self):
        row = tf.decide(ticker="STR", side="buy", instrument="etf", bucket="etf",
                        stats=_stats(0.6, 4.0, 2.0), cost_pct=0.10, holding_days=7, source="test",
                        cfg=tf.config({"ENABLED": True}))
        self.assertTrue(row["allowed"])
        self.assertGreater(row["edgePp"], row["marginPp"])
        self.assertTrue(row["reason"].startswith("Allowed: "), row["reason"])

    def test_a_negative_expectancy_is_skipped_as_a_loss_before_tax(self):
        row = tf.decide(ticker="NEG", side="buy", instrument="stock", bucket="stocks",
                        stats=tf.bucket_stats("stocks", PRIOR_STATS), cost_pct=0.04, source="test",
                        cfg=tf.config({"ENABLED": True}))
        self.assertFalse(row["allowed"])
        self.assertLess(row["expectedPct"], 0)
        self.assertIn("a loss before tax", row["reason"])
        self.assertEqual(row["holdingDays"], 6.36)       # the stocks bucket's measured hold

    def test_the_off_switch_logs_the_verdict_but_skips_nothing(self):
        row = tf.decide(ticker="OFF", side="buy", instrument="stock", bucket="stocks",
                        stats=_stats(0.3, 1.0, 1.0), cost_pct=0.0, holding_days=7, source="test",
                        cfg=tf.config({"ENABLED": False}))
        self.assertTrue(row["allowed"])
        self.assertFalse(row["wouldAllow"])
        self.assertFalse(row["enforced"])
        self.assertTrue(row["reason"].startswith("Filter off"), row["reason"])

    def test_today_the_priors_allow_etfs_and_skip_stocks(self):
        cfg = tf.config({"ENABLED": True})
        etf = tf.decide(ticker="SSO", side="buy", instrument="etf", bucket="etf",
                        stats=tf.bucket_stats("etf", PRIOR_STATS), cost_pct=0.04, source="t", cfg=cfg)
        stock = tf.decide(ticker="AMD", side="buy", instrument="stock", bucket="stocks",
                          stats=tf.bucket_stats("stocks", PRIOR_STATS), cost_pct=0.04, source="t", cfg=cfg)
        self.assertTrue(etf["allowed"])
        self.assertFalse(stock["allowed"])

    def test_bucket_stats_label_the_source_and_fall_back_to_overall(self):
        s = tf.bucket_stats("csp_options", PRIOR_STATS)
        self.assertEqual((s["winRate"], s["avgWin"], s["avgLoss"], s["n"]), (0.716, 0.69, 0.46, 0))
        self.assertIn("Kelly prior", s["source"])
        live = tf.bucket_stats("stocks", {"by_strategy": {"stocks": {"win_rate": 0.28, "avg_win": 3, "avg_loss": -2, "trades": 40}}})
        self.assertEqual(live["avgLoss"], 2.0)
        self.assertIn("40 closed trades", live["source"])
        self.assertEqual(tf.bucket_stats("nope", PRIOR_STATS)["bucket"], "overall")


class TestDecisionLog(TempLog):
    def _row(self, ticker="AAA", allowed=True, ts=None):
        r = tf.decide_unpriced(ticker=ticker, side="buy", instrument="stock", why="w", source="test")
        r["allowed"] = r["wouldAllow"] = allowed
        if ts:
            r["ts"] = ts
        return r

    def test_rows_are_appended_once_per_day_per_ticker_and_verdict(self):
        self.assertTrue(tf.log_decision(self._row(), path=self.path))
        self.assertFalse(tf.log_decision(self._row(), path=self.path), "same ticker, verdict and day: deduped")
        self.assertTrue(tf.log_decision(self._row(allowed=False), path=self.path), "a changed verdict is a new row")
        self.assertTrue(tf.log_decision(self._row(ticker="BBB"), path=self.path))
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(timespec="seconds")
        self.assertTrue(tf.log_decision(self._row(ts=tomorrow), path=self.path))
        rows = _read(self.path)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["schema"], tf.SCHEMA)

    def test_rotation_above_the_size_cap(self):
        with open(self.path, "w") as f:
            f.write("x" * 200 + "\n")
        self.assertTrue(tf.log_decision(self._row(), path=self.path, max_bytes=100))
        self.assertTrue(os.path.exists(self.path + ".1"))
        self.assertEqual(len(_read(self.path)), 1)

    def test_recent_decisions_newest_first_skipping_torn_lines(self):
        tf.log_decision(self._row("AAA"), path=self.path)
        tf.log_decision(self._row("BBB"), path=self.path)
        with open(self.path, "a") as f:
            f.write('{"ticker": "torn"\n')
        got = tf.recent_decisions(10, path=self.path)
        self.assertEqual([r["ticker"] for r in got], ["BBB", "AAA"])

    def test_a_failing_write_is_reported_not_raised(self):
        self.assertFalse(tf.log_decision(self._row(), path=os.path.join(self._tmp.name, "missing", "x.jsonl")))


CSP_CHEAP = {"occ_symbol": "CMCSA261218P00024000", "strike": 24.0, "expiry": "2026-12-18", "side": "sell", "qty": 1,
             "limit_price": 0.79, "premium_received": 79.0, "max_loss": 2321.0, "cash_required": 2400.0,
             "days_to_expiry": 42}
CSP_RICH = dict(CSP_CHEAP, occ_symbol="AAPL261218P00150000", strike=150.0, limit_price=2.5, premium_received=250.0,
                cash_required=15000.0, max_loss=14750.0)


class TestOptionGate(TempLog):
    def gate(self, contract, strategy="sell_cash_secured_put", **kw):
        return tf.gate_option_contract(contract, kw.pop("ticker", "CMCSA"), strategy, path=self.path,
                                       stats=PRIOR_STATS, shadow=False, cfg=tf.config({"ENABLED": True}), **kw)

    def test_a_cheap_strike_csp_is_skipped_on_its_own_costs(self):
        row = self.gate(CSP_CHEAP)
        self.assertFalse(row["allowed"])
        self.assertEqual(row["notional"], 2400.0)
        # fees 0.65 x 1 x 2 + entry walk 2 x 0.30 x 0.05 x $79, over $2,400
        self.assertAlmostEqual(row["costPct"], (1.30 + 2 * 0.30 * 0.05 * 79.0) / 2400.0 * 100, places=3)
        self.assertEqual(row["holdingDays"], 6.39)
        self.assertIn("Option-selling (sell_cash_secured_put)", row["reason"])
        err = tf.skip_error(row)
        self.assertTrue(err.startswith("tax filter: Skipped"))
        self.assertNotRegex(err.lower(), r"budget|capital", "the tier dispatcher keys rejected_capital on these words")

    def test_a_rich_strike_csp_is_allowed(self):
        self.assertTrue(self.gate(CSP_RICH, ticker="AAPL")["allowed"])

    def test_the_hold_never_runs_past_expiry(self):
        row = self.gate(dict(CSP_RICH, days_to_expiry=3), ticker="AAPL")
        self.assertEqual(row["holdingDays"], 3.0)

    def test_long_premium_and_hedges_are_logged_unpriced_and_allowed(self):
        lp = self.gate(CSP_RICH, strategy="buy_call", ticker="AAPL")
        self.assertTrue(lp["allowed"])
        self.assertFalse(lp["priced"])
        self.assertIn("does not guess", lp["reason"])
        hedge = self.gate(CSP_RICH, strategy="buy_put", ticker="SPY", metadata={"tax_role": "hedge"})
        self.assertTrue(hedge["allowed"])
        self.assertIn("hedge", hedge["reason"])
        self.assertEqual(len(_read(self.path)), 2)

    def test_a_broken_contract_fails_open_with_the_error_logged(self):
        row = self.gate({"qty": 1}, ticker="BRK")
        self.assertTrue(row["allowed"])
        self.assertIn("ValueError", row["error"])
        self.assertEqual(_read(self.path)[-1]["ticker"], "BRK")

    def test_notional_and_premium_per_structure(self):
        self.assertEqual(tf.option_notional({"cash_required": 2400, "strike": 24, "qty": 1}, "sell_cash_secured_put"), 2400)
        self.assertEqual(tf.option_notional({"strike": 100, "qty": 2}, "covered_call"), 20000)
        self.assertEqual(tf.option_notional({"short_put_strike": 425, "short_call_strike": 465, "qty": 3}, "iron_condor"), 127500)
        self.assertEqual(tf.option_notional({"short_strike": 50, "qty": 1}, "bull_put_credit_spread"), 5000)
        self.assertEqual(tf.option_premium({"premium_received": 79.0, "limit_price": 0.79, "qty": 1}), 79.0)
        self.assertEqual(tf.option_premium({"total_credit": 3318.0, "net_credit": 11.06, "qty": 3}), 3318.0)
        self.assertAlmostEqual(tf.option_premium({"net_credit": 1.5, "qty": 2}), 300.0)

    def test_a_condor_pays_fees_on_four_legs_and_the_walk_once_on_its_net_credit(self):
        condor = {"strategy": "iron_condor", "short_call": "A", "long_call": "B", "short_put": "C", "long_put": "D",
                  "short_put_strike": 425.0, "short_call_strike": 465.0, "expiry": "2026-12-06", "qty": 3,
                  "net_credit": 11.06, "total_credit": 3318.0, "max_loss": 1182.0, "days_to_expiry": 30}
        self.assertAlmostEqual(tf.option_round_trip_cost(condor), 0.65 * 3 * 4 * 2 + 2 * 0.30 * 0.05 * 3318.0, places=6)
        row = self.gate(condor, strategy="iron_condor", ticker="TSLA")
        self.assertTrue(row["allowed"], row["reason"])
        self.assertEqual(row["notional"], 127500.0)
        self.assertFalse(row["capped"])

    def test_a_new_skip_writes_one_rejected_tax_shadow_record(self):
        calls = []
        orig = tf._shadow_reject
        tf._shadow_reject = lambda *a, **k: calls.append(a)
        try:
            for _ in range(2):
                tf.gate_option_contract(CSP_CHEAP, "CMCSA", "sell_cash_secured_put", path=self.path,
                                        stats=PRIOR_STATS, shadow=True, cfg=tf.config({"ENABLED": True}))
        finally:
            tf._shadow_reject = orig
        self.assertEqual(len(calls), 1, "the second identical skip is deduped, so no second record")


class TestScanAndTierGates(TempLog):
    def test_a_stock_trade_on_the_negative_prior_is_skipped(self):
        row = tf.gate_scan_trade({"ticker": "AMD", "side": "buy", "score": 70, "instrument": "stock"},
                                 {"costs": {"total_cost_pct": 0.04}}, sizing_input={"ticker": "AMD", "momentum_score": 80},
                                 stats=PRIOR_STATS, cfg=tf.config({"ENABLED": True}), path=self.path)
        self.assertFalse(row["allowed"])
        self.assertEqual((row["bucket"], row["instrument"], row["costPct"]), ("stocks", "stock", 0.04))

    def test_an_etf_substitution_is_judged_as_the_etf(self):
        row = tf.gate_scan_trade({"ticker": "AAPL", "side": "buy", "instrument": "etf", "instrument_ticker": "AAPU",
                                  "etf_shares": 10}, {"costs": {"total_cost_pct": 0.04}}, sizing_input={"ticker": "AAPL"},
                                 stats=PRIOR_STATS, cfg=tf.config({"ENABLED": True}), path=self.path)
        self.assertEqual((row["ticker"], row["bucket"], row["underlying"]), ("AAPU", "etf", "AAPL"))
        self.assertTrue(row["allowed"])

    def test_an_options_choice_is_judged_on_its_stock_fallback(self):
        row = tf.gate_scan_trade({"ticker": "AMD", "side": "buy", "instrument": "options"}, None,
                                 sizing_input={"ticker": "AMD", "momentum_score": 80},
                                 stats=PRIOR_STATS, cfg=tf.config({"ENABLED": True}), path=self.path)
        self.assertEqual(row["instrument"], "stock_fallback")
        self.assertEqual(row["bucket"], "stocks")

    def test_tier_buys_are_gated_and_option_opens_pass_through(self):
        class A:  # tiered_strategy.TierAction's shape
            def __init__(self, action, ticker, strategy, tier):
                self.action, self.ticker, self.strategy, self.tier = action, ticker, strategy, tier
        actions = [A("SELL_CSP", "XYZ", "csp_core", 1), A("BUY", "SSO", "trend_long", 3)]
        kept, skipped = tf.filter_tier_buys(actions, stats=PRIOR_STATS, cfg=tf.config({"ENABLED": True}), path=self.path)
        self.assertEqual([a.ticker for a in kept], ["XYZ", "SSO"])
        self.assertEqual(skipped, [])
        bad = {"by_strategy": {"etf": {"win_rate": 0.3, "avg_win": 1.0, "avg_loss": 1.0, "trades": 50}}}
        kept, skipped = tf.filter_tier_buys(actions, stats=bad, cfg=tf.config({"ENABLED": True}), path=self.path)
        self.assertEqual([a.ticker for a in kept], ["XYZ"])
        self.assertEqual(skipped[0]["ticker"], "SSO")
        self.assertEqual(skipped[0]["source"], "tier3.BUY")


class TestSelectContractGate(TempLog):
    def test_a_skip_becomes_the_no_contract_error_and_an_allow_carries_the_summary(self):
        import options_execution as oe
        orig_stats, orig_path = tf.sizing_stats, tf.decisions_path
        tf.sizing_stats = lambda: PRIOR_STATS
        tf.decisions_path = lambda: self.path
        try:
            skipped = oe._tax_gate_contract(dict(CSP_CHEAP), "CMCSA", "sell_cash_secured_put", {}, 24.5, shadow=False)
            allowed = oe._tax_gate_contract(dict(CSP_RICH), "AAPL", "sell_cash_secured_put", {}, 160.0, shadow=False)
        finally:
            tf.sizing_stats, tf.decisions_path = orig_stats, orig_path
        self.assertTrue(skipped["error"].startswith("tax filter: Skipped"))
        self.assertFalse(skipped["tax"]["allowed"])
        self.assertEqual(skipped["price"], 24.5)
        self.assertIsNone(allowed.get("error"))
        self.assertTrue(allowed["tax"]["allowed"])
        self.assertEqual(allowed["occ_symbol"], CSP_RICH["occ_symbol"])
        self.assertEqual(len(_read(self.path)), 2)


class TestWiring(unittest.TestCase):
    def _src(self, name):
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            return f.read()

    def test_scan_market_gates_every_stock_loop_trade_after_sizing_before_it_is_returned(self):
        src = self._src("bot_engine.py")
        resolve = src.index("instrument_decision, _etf_fields = _resolve_etf_execution(")
        gate = src.index("_tax = _tf.gate_scan_trade(")
        append = src.index("trades.append({", gate)
        self.assertLess(resolve, gate)
        self.assertLess(gate, append)
        block = src[gate:append]
        self.assertIn('_note_gate(ticker, "rejected_tax"', block)
        self.assertIn("_sizing_blocks.append(", block)
        self.assertIn("sizing_input={**stock, \"score\": final_score, \"side\": side}", block,
                      "the bucket must be the one calculate_position sized with")
        self.assertIn('trades[-1]["trade_type"] = "options"', src[append:append + 4000])

    def test_tier_buys_are_filtered_where_the_tier_actions_are_read(self):
        src = self._src("bot_engine.py")
        i = src.index('tiered_actions = tier_result["actions"]')
        self.assertIn("filter_tier_buys(tiered_actions)", src[i:i + 800])

    def test_select_contract_judges_every_open_before_returning_it(self):
        src = self._src("options_execution.py")
        start = src.index("def select_contract(")
        end = src.index("\ndef ", start + 10)
        body = src[start:end]
        self.assertIn("_tax_gate_contract(result, ticker, strategy, metadata, price, shadow=trade is None)", body)
        self.assertLess(body.index("_tax_gate_contract("), body.rindex("return result"))
        submit = src[src.index("def submit_options_order("):src.index("def _submit_mleg_order(")]
        self.assertNotIn("tax", submit, "the frozen submit path is untouched")

    def test_the_tier4_hedge_is_tagged_so_the_filter_never_prices_it(self):
        src = self._src("tiered_strategy.py")
        t4 = src[src.index("def tier4_tail_hedge("):src.index("def _enforce_exposure_cap(")]
        self.assertIn('"tax_role": "hedge"', t4)


class TestParity(unittest.TestCase):
    def test_defaults_mirror_system_config(self):
        from system_config import BASE_CONFIG
        self.assertEqual(BASE_CONFIG["TAX_FILTER"], tf.DEFAULTS)

    def test_rates_mirror_the_tax_ledger(self):
        with open(os.path.join(HERE, "server", "taxLedger.ts"), encoding="utf-8") as f:
            ts = f.read()
        m = re.search(r"DEFAULT_TAX_RATES: TaxRates = \{ shortTerm: ([0-9.]+), longTerm: ([0-9.]+)", ts)
        self.assertIsNotNone(m)
        self.assertEqual((float(m.group(1)), float(m.group(2))),
                         (tf.DEFAULTS["SHORT_TERM_RATE"], tf.DEFAULTS["LONG_TERM_RATE"]))

    def test_hold_days_are_the_backtest_samples_mean_calendar_hold(self):
        with open(os.path.join(HERE, "backtest_10yr_results.json")) as f:
            trades = json.load(f)["all_trades"]["NEW_CSP_ONLY"]

        def mean_days(rows):
            return round(statistics.mean((date.fromisoformat(t["date_exit"]) - date.fromisoformat(t["date_entry"])).days
                                         for t in rows), 2)
        by = {"csp_options": "options", "stocks": "stock", "etf": "etf"}
        for bucket, inst in by.items():
            self.assertEqual(tf.DEFAULTS["HOLD_DAYS"][bucket], mean_days([t for t in trades if t["instrument"] == inst]), bucket)
        self.assertEqual(tf.DEFAULTS["HOLD_DAYS"]["vrp"], mean_days(trades))

    def test_option_cost_constants_are_the_repos(self):
        src = open(os.path.join(HERE, "options_execution.py"), encoding="utf-8").read()
        m = re.search(r"WALK_PCT = ([0-9.]+)", src[src.index("def _optimized_limit_price("):])
        self.assertEqual(float(m.group(1)), tf.OPTION_ENTRY_WALK)

    def test_premium_selling_strategies_are_the_sizers_csp_bucket(self):
        from position_sizing import _infer_strategy
        for s in tf.PREMIUM_SELLING:
            self.assertEqual(_infer_strategy({"options_strategy": s}), "csp_options", s)


if __name__ == "__main__":
    unittest.main()
