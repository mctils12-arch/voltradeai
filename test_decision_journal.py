#!/usr/bin/env python3
"""
DECISION JOURNAL (2026-11-06, decision_journal.py; human refocus spec §7 "log
every signal, decision, and outcome").

Pins:
- the row contract (the keys the UI's per-pick cards read; every stage key
  present; plain-English summary; non-finite numbers never written);
- the size cap (rotation to .1) and the per-row cap;
- the off switch (env wins when present, config default ON);
- each Python producer, through its REAL call path where a harness exists:
  scan_market's gate skips (dedupe once per ticker/gate/ET day, including
  across a fresh process), tiered_strategy's CSP refusals (shadow-only
  would_reject_* excluded) and master kill, options_manager's exits (deferred
  closes excluded), the volatility-managed floor's orders;
- deep_score's sub-score drivers (points vs a neutral 50 in the rules score);
- parity with server/decisionJournal.ts (file name, size cap, env var, the
  plain-English gate table, the sub-score weights);
- the wiring: every producer hook exists and none can raise into its caller.

Run: bash pytest_locked.sh -q test_decision_journal.py
"""
import json
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import decision_journal as dj  # noqa: E402

TS_SRC = open(os.path.join(_ROOT, "server", "decisionJournal.ts"), encoding="utf-8").read()
CONTRACT_KEYS = {"ts", "ticker", "action", "instrument", "chosen", "rejected", "stages", "sizing", "summary"}


class _JournalCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        p = patch.object(dj, "DATA_DIR", tmp.name)
        p.start()
        self.addCleanup(p.stop)
        env = patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop(dj.ENV_VAR, None)
        dj._reset_dedupe()
        self.addCleanup(dj._reset_dedupe)

    def rows(self):
        return dj.read_rows()


CAND = {
    "ticker": "AMD", "deep_score": 61.2, "trade_type": "stock", "sector": "Technology", "regime_label": "BULL",
    "reasons": ["Momentum strong", "ADX 31 — trending"],
    "score_breakdown": {"momentum": 80, "mean_reversion": 40, "vrp": 56, "squeeze": 50, "volume": 60,
                        "macro": -2, "news": 3, "adx": 3, "insider_mspr": 4, "rules_raw": 64.0},
}


class TestContract(_JournalCase):
    def test_normalize_fills_every_contract_key_and_stage(self):
        r = dj.normalize({"action": "BUY", "ticker": "amd", "summary": "  Bought   AMD  ", "stages": {"flows": {"pressure": 0.2}, "custom": {"x": 1}}})
        self.assertTrue(CONTRACT_KEYS <= set(r))
        self.assertEqual(r["action"], "buy")
        self.assertEqual(r["ticker"], "AMD")
        self.assertEqual(r["summary"], "Bought AMD")
        self.assertEqual(set(dj.STAGE_KEYS) - set(r["stages"]), set())
        self.assertEqual(r["stages"]["flows"], {"pressure": 0.2}, "another module's stage is kept as-is")
        self.assertEqual(r["stages"]["custom"], {"x": 1})
        self.assertIsNone(r["stages"]["tax"])
        self.assertEqual(r["rejected"], [])
        self.assertEqual(r["v"], dj.SCHEMA_VERSION)

    def test_invalid_rows_are_refused(self):
        for bad in ({"action": "hold", "ticker": "AMD"}, {"action": "buy"}, {"action": "buy", "ticker": ""}, None, "x"):
            self.assertIsNone(dj.normalize(bad), bad)

    def test_skip_defaults_chosen_to_none_and_non_finite_numbers_are_dropped(self):
        r = dj.normalize({"action": "skip", "ticker": "X", "sizing": {"a": float("nan"), "b": float("inf"), "c": 1.5}})
        self.assertEqual(r["chosen"], "none")
        self.assertEqual(r["sizing"], {"a": None, "b": None, "c": 1.5})
        dj.append_rows([{"action": "skip", "ticker": "X", "price": float("nan")}])
        line = open(dj.journal_path()).read()
        self.assertNotIn("NaN", line)
        json.loads(line)

    def test_an_oversized_row_is_cut_but_keeps_the_contract(self):
        huge = {"action": "buy", "ticker": "AMD", "summary": "s" * 5000,
                "rejected": [{"instrument": f"i{i}", "reason": "r" * 900} for i in range(6)],
                "stages": {"selection": {"reasons": ["x" * 300] * 6, "subscores": {f"k{i}": i for i in range(60)},
                                         "blob": "b" * 7000}}}
        self.assertEqual(dj.append_rows([huge]), 1)
        raw = open(dj.journal_path(), "rb").read()
        self.assertLessEqual(len(raw), dj.ROW_MAX_BYTES + 1)
        row = json.loads(raw)
        self.assertTrue(row["truncated"])
        self.assertTrue(CONTRACT_KEYS <= set(row))


class TestSwitchAndSize(_JournalCase):
    def test_env_wins_when_present_and_config_defaults_on(self):
        self.assertTrue(dj.enabled(env={}, config={}))
        self.assertFalse(dj.enabled(env={}, config={"enabled": False}))
        for off in ("off", "0", "false", "", "none", "NO"):
            self.assertFalse(dj.enabled(env={dj.ENV_VAR: off}, config={"enabled": True}), off)
        self.assertTrue(dj.enabled(env={dj.ENV_VAR: "on"}, config={"enabled": False}))

    def test_off_writes_nothing(self):
        with patch.dict(os.environ, {dj.ENV_VAR: "off"}):
            self.assertEqual(dj.append_rows([{"action": "buy", "ticker": "AMD"}]), 0)
            self.assertEqual(dj.record_scan_skips([{"ticker": "AMD", "decision": "rejected_score"}], [CAND]), 0)
        self.assertFalse(os.path.exists(dj.journal_path()))

    def test_rotates_to_dot_one_above_the_cap(self):
        cap = dj.MIN_MAX_BYTES
        with patch.object(dj, "_config", return_value={"max_bytes": cap}):
            row = {"action": "skip", "ticker": "AMD", "summary": "x" * 900}
            while not os.path.exists(dj.journal_path()) or os.path.getsize(dj.journal_path()) <= cap:
                dj.append_rows([row])
            dj.append_rows([{"action": "buy", "ticker": "NEW"}])
        self.assertTrue(os.path.exists(dj.journal_path() + ".1"))
        self.assertGreater(os.path.getsize(dj.journal_path() + ".1"), cap)
        current = open(dj.journal_path()).read().splitlines()
        self.assertEqual([json.loads(x)["ticker"] for x in current], ["NEW"])
        rows = dj.read_rows()
        self.assertEqual(rows[-1]["ticker"], "NEW", "both generations are read, oldest first")
        self.assertGreater(len(rows), 1)

    def test_config_floor_and_default(self):
        self.assertEqual(dj.max_bytes(config={}), dj.DEFAULT_MAX_BYTES)
        self.assertEqual(dj.max_bytes(config={"max_bytes": 5}), dj.MIN_MAX_BYTES)
        self.assertEqual(dj.max_bytes(config={"max_bytes": "x"}), dj.DEFAULT_MAX_BYTES)


class TestSelection(unittest.TestCase):
    def test_drivers_are_points_in_the_rules_score_vs_neutral(self):
        d = {x["name"]: x["points"] for x in dj.score_drivers(CAND["score_breakdown"])}
        self.assertEqual(d["momentum"], 7.5)          # (80 - 50) x 0.25
        self.assertEqual(d["mean_reversion"], -2.0)   # (40 - 50) x 0.20
        self.assertEqual(d["vrp"], 1.5)               # (56 - 50) x 0.25
        self.assertEqual(d["squeeze"], 0.0)
        self.assertEqual(d["volume"], 1.5)
        self.assertEqual((d["macro"], d["news"], d["adx"]), (-2.0, 3.0, 3.0))
        self.assertNotIn("insider_mspr", d, "inside intel: a sub-score, not a separate driver")
        self.assertNotIn("rules_raw", d)
        self.assertEqual(dj.dominant_driver(dj.score_drivers(CAND["score_breakdown"]))["name"], "momentum")
        self.assertEqual(dj.score_drivers(None), [])

    def test_dominant_is_the_largest_positive_else_the_largest(self):
        neg = dj.score_drivers({"momentum": 20, "volume": 45})
        self.assertEqual(dj.dominant_driver(neg)["name"], "momentum")
        self.assertIsNone(dj.dominant_driver([]))

    def test_selection_stage_and_signal(self):
        st = dj.selection_stage(CAND, threshold=63)
        self.assertEqual((st["score"], st["threshold"]), (61.2, 63.0))
        self.assertEqual(st["subscores"]["momentum"], 80.0)
        self.assertEqual(st["dominant"]["label"], "momentum")
        self.assertEqual(st["reasons"][0], "Momentum strong")
        self.assertEqual(dj.signal_for_candidate(CAND), "deep_score scan, momentum-led")
        self.assertEqual(dj.signal_for_candidate({"regime_at_entry": "OPTIONS_SCANNER", "setup": "iv_crush"}),
                         "options scanner: iv_crush")


class TestScanSkips(_JournalCase):
    def test_one_row_per_ticker_gate_day_held_left_out(self):
        skips = [{"ticker": "AMD", "decision": "rejected_score", "reason": "score 61.2 < adaptive MIN_SCORE 63 (BULL)"},
                 {"ticker": "AMD", "decision": "rejected_score", "reason": "again"},
                 {"ticker": "XOM", "decision": "rejected_held", "reason": "already holding"},
                 {"ticker": "NVDA", "decision": "rejected_regime_gate", "reason": "no new stock longs in BEAR"}]
        n = dj.record_scan_skips(skips, [CAND], scan_regime="BULL", threshold=63, vxx_ratio=0.93)
        self.assertEqual(n, 2)
        rows = self.rows()
        self.assertEqual([(r["ticker"], r["rule"]) for r in rows], [("AMD", "rejected_score"), ("NVDA", "rejected_regime_gate")])
        amd = rows[0]
        self.assertEqual((amd["action"], amd["instrument"], amd["chosen"]), ("skip", "stock", "none"))
        self.assertEqual(amd["stages"]["regime"]["label"], "BULL")
        self.assertIn("VXX 0.93x", amd["stages"]["regime"]["reason"])
        self.assertEqual(amd["stages"]["sector"]["name"], "Technology")
        self.assertEqual(amd["stages"]["selection"]["dominant"]["name"], "momentum")
        self.assertEqual(amd["signal"], "deep_score scan, momentum-led")
        self.assertEqual(amd["summary"], "Skipped AMD: its score was below the bar (score 61.2 < adaptive MIN_SCORE 63 (BULL)).")
        self.assertIn("a stock buy was refused because its score was below the bar", amd["rejected"][0]["reason"])
        self.assertIsNone(rows[1]["stages"]["selection"], "a skip with no scored candidate has no selection stage")
        # a later scan the same day writes nothing new for the same gate
        self.assertEqual(dj.record_scan_skips(skips, [CAND]), 0)

    def test_dedupe_survives_a_fresh_process(self):
        dj.record_scan_skips([{"ticker": "AMD", "decision": "rejected_score", "reason": "r"}], [CAND])
        dj._reset_dedupe()   # a new subprocess: no memory, the journal's tail seeds it
        self.assertEqual(dj.record_scan_skips([{"ticker": "AMD", "decision": "rejected_score", "reason": "r"}], [CAND]), 0)
        self.assertEqual(dj.record_scan_skips([{"ticker": "AMD", "decision": "rejected_sizing", "reason": "kelly<=0"}], [CAND]), 1)

    def test_a_new_et_day_records_again(self):
        d1 = datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc)
        d2 = datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)
        self.assertEqual(dj.record_scan_skips([{"ticker": "AMD", "decision": "rejected_score"}], now=d1), 1)
        self.assertEqual(dj.record_scan_skips([{"ticker": "AMD", "decision": "rejected_score"}], now=d1), 0)
        self.assertEqual(dj.record_scan_skips([{"ticker": "AMD", "decision": "rejected_score"}], now=d2), 1)

    def test_extra_stages_from_other_modules_ride_along(self):
        cand = {**CAND, "decision_stages": {"flows": {"direction": "sell", "pressure": -0.4}, "tax": {"allowed": False}}}
        dj.record_scan_skips([{"ticker": "AMD", "decision": "rejected_score"}], [cand])
        st = self.rows()[0]["stages"]
        self.assertEqual(st["flows"]["direction"], "sell")
        self.assertFalse(st["tax"]["allowed"])

    def test_a_write_failure_never_raises(self):
        with patch.object(dj, "append_rows", side_effect=RuntimeError("disk gone")):
            self.assertEqual(dj.record_scan_skips([{"ticker": "AMD", "decision": "rejected_score"}], [CAND]), 0)
        with patch.object(dj, "journal_path", return_value="/proc/nonexistent/dir/journal.jsonl"):
            self.assertEqual(dj.append_rows([{"action": "buy", "ticker": "AMD"}]), 0)


class TestRegimeBudgetAndFlowsStages(_JournalCase):
    """The finished agents' inputs: the scan's regime risk budget (regime_engine.risk_budget_view;
    its `summary` is the regime stage's plain-English reason) and flows_model's `flows` blocks."""

    BUDGET = {"budget": 0.62, "band": "reduced", "summary": "Risk budget 0.62 (reduced): VIX 22 above its 1-year median; breadth 41%.",
              "drivers": ["VIX 22.1 (+1.2 sd)", "breadth 41%"], "applied": True, "stale": False, "as_of": "2026-11-06"}

    def test_regime_stage_uses_the_budget_summary(self):
        st = dj.regime_stage("BULL", "NEUTRAL", 1.04, budget=self.BUDGET)
        self.assertEqual(st["reason"], "Risk budget 0.62 (reduced): VIX 22 above its 1-year median; breadth 41%. "
                                       "(NEUTRAL market (VXX 1.04x its 30-day average); the scorer's regime read was BULL)")
        self.assertEqual(st["risk_budget"]["budget"], 0.62)
        self.assertEqual(st["risk_budget"]["drivers"][1], "breadth 41%")
        self.assertTrue(st["risk_budget"]["applied"])
        self.assertEqual(dj.regime_stage(budget=self.BUDGET)["reason"], self.BUDGET["summary"])
        self.assertNotIn("risk_budget", dj.regime_stage("BULL", budget={"error": "boom"}))
        self.assertIsNone(dj.budget_view({"budget": None}))

    def test_scan_skips_read_the_budget_off_the_regime_state(self):
        import regime_engine
        state = {"as_of": "2026-11-06", "risk_budget": 0.62}
        with patch.object(regime_engine, "risk_budget_view", create=True, side_effect=lambda st: self.BUDGET if st is state else None):
            dj.record_scan_skips([{"ticker": "AMD", "decision": "rejected_score"}], [CAND], scan_regime="BULL", regime_state=state)
        self.assertEqual(self.rows()[0]["stages"]["regime"]["risk_budget"]["band"], "reduced")
        with patch.object(regime_engine, "risk_budget_view", create=True, side_effect=RuntimeError("no state")):
            self.assertIsNone(dj.regime_budget_from_state(state), "a failing view never raises")
        self.assertIsNone(dj.regime_budget_from_state(None))

    def test_flows_stage_from_a_candidate_and_the_floor(self):
        flows = {"applied": True, "multiplier": 0.92, "sector": "Technology", "etf": "XLK", "pressure": -0.4,
                 "reason": "Technology: expected net selling (pension rebalancing) — size x0.92", "label": "ESTIMATE"}
        dj.record_scan_skips([{"ticker": "AMD", "decision": "rejected_volume_share"}], [{**CAND, "flows": flows}])
        self.assertEqual(self.rows()[0]["stages"]["flows"]["etf"], "XLK")
        dj.record_floor({"sigma": 0.12, "target_exposure": 1.2, "current_exposure": 1.0,
                         "flows": {"applied": True, "multiplier": 0.95, "reason": "quarter-end pension selling expected"},
                         "actions": [{"type": "floor_vm_buy", "ticker": "SSO", "qty": 3, "limit": 95.0, "ok": True, "order_id": "o"}]})
        floor = self.rows()[-1]
        self.assertEqual(floor["stages"]["flows"]["multiplier"], 0.95)
        self.assertIn("; institutional flows: quarter-end pension selling expected.", floor["summary"])
        self.assertIsNone(dj.flows_stage({}))


class TestTierProducers(_JournalCase):
    def test_csp_gate_rejections_journal_real_refusals_only(self):
        import tiered_strategy as ts
        with patch.object(ts, "_layer2_price_lookup", return_value={"AAPL": 150.0}), \
                patch("shadow_portfolio.log_candidate"):
            ts.log_csp_gate_shadow([
                {"ticker": "aapl", "decision": "rejected_options_slots", "reason": "options slots full (6/6)",
                 "metadata": {"vxx_ratio": 1.12, "regime_label": "CAUTION", "target_delta": 0.3}},
                {"ticker": "MSFT", "decision": "would_reject_regime_vrp", "reason": "shadow only"},
            ])
        rows = self.rows()
        self.assertEqual([(r["ticker"], r["rule"]) for r in rows], [("AAPL", "rejected_options_slots")])
        r = rows[0]
        self.assertEqual((r["action"], r["instrument"], r["signal"]), ("skip", "option", "tier1 CSP harvest"))
        self.assertEqual(r["stages"]["regime"]["label"], "CAUTION")
        self.assertEqual(r["stages"]["options"]["target_delta"], 0.3)
        self.assertEqual(r["summary"], "Skipped AAPL: every options slot was taken (options slots full (6/6)).")

    def test_master_kill_journals_each_blocked_csp(self):
        import tiered_strategy as ts
        acts = [ts.TierAction(tier=1, action="SELL_CSP", ticker=t, strategy="csp_core", size_pct=0.05, reason="T1",
                              metadata={"regime_label": "NEUTRAL", "vxx_ratio": 1.0}) for t in ("AMD", "AAPL")]
        with patch.object(ts, "tier1_csp_core", return_value=acts), \
                patch("csp_universe._load_layer2_cache", return_value=None):
            ctx = ts.TierContext(equity=100_000)
            ts.log_masterkill_csp_shadow(ctx, "Portfolio invested 96% >= 95%")
        rows = self.rows()
        self.assertEqual([r["ticker"] for r in rows], ["AMD", "AAPL"])
        self.assertTrue(all(r["rule"] == "rejected_masterkill" for r in rows))
        self.assertIn("Portfolio invested 96% >= 95%", rows[0]["summary"])


class TestOptionExits(_JournalCase):
    def test_submitted_closes_are_journaled_deferred_ones_are_not(self):
        acts = [
            {"action": "CLOSE", "ticker": "TLT", "occ_symbol": "TLT261219P00085000", "type": "profit_target",
             "reason": "PROFIT TARGET: 55% of max profit reached", "order": {"status": "submitted", "order_id": "o1"}},
            {"action": "CLOSE", "ticker": "XLF", "occ_symbol": "XLF261127P00055500", "type": "loss_limit",
             "reason": "LOSS LIMIT — deferred: market closed", "order": {"status": "deferred"}},
            {"action": "CLOSE_STRATEGY", "ticker": "IREN", "strategy": "iron_condor", "type": "strategy_profit_target",
             "legs": ["IREN261204P00020000", "IREN261204P00021000", "IREN261204C00023000", "IREN261204C00024000"],
             "reason": "50% of credit kept", "combined_pnl": 85.25, "order": {"status": "error", "detail": "mleg rejected"}},
            {"action": "ROLL", "ticker": "KO", "occ_symbol": "KO261219P00060000", "type": "dte_roll", "reason": "21 DTE",
             "order": {"status": "submitted"}},
            {"action": "WARNING", "ticker": "KO", "type": "delta_drift"},
            {"action": "WAIT", "ticker": "IREN", "type": "strategy_close_working"},
        ]
        self.assertEqual(dj.record_option_exits(acts), 3)
        rows = self.rows()
        self.assertEqual([(r["ticker"], r["action"], r["chosen"]) for r in rows],
                         [("TLT", "sell", "profit_target"), ("IREN", "sell", "strategy_profit_target"), ("KO", "rotate", "dte_roll")])
        self.assertEqual(rows[0]["summary"],
                         "Closed the TLT option (TLT $85 put expiring 2026-12-19): the profit target was reached. "
                         "PROFIT TARGET: 55% of max profit reached. Order submitted.")
        self.assertIn("Unrealized P&L at the decision: $85.25.", rows[1]["summary"])
        self.assertIn("Order error: mleg rejected.", rows[1]["summary"])
        self.assertEqual(rows[1]["stages"]["options"]["legs"][0], "IREN261204P00020000")
        self.assertEqual(rows[0]["order"]["id"], "o1")
        # the same close on the next scan (still working) is not a new decision
        self.assertEqual(dj.record_option_exits(acts), 0)

    def test_through_manage_options_positions(self):
        """The REAL manager pass (test_options_exit_market_hours's harness): every
        single-leg rule that submits is journaled; a market-closed pass journals nothing."""
        import test_options_exit_market_hours as h

        class Case(h._ManagerCase):
            def runTest(self):
                pass

        c = Case()
        c.setUp()
        try:
            legs = h.book()
            c.load(legs)
            c.run_manager(h.Broker([l["pos"] for l in legs.values()], is_open=False), allow_rolls=False)
            self.assertEqual(self.rows(), [], "deferred closes are not decisions yet")
            c.run_manager(h.Broker([l["pos"] for l in legs.values()], is_open=True), allow_rolls=False)
            rows = self.rows()
            self.assertEqual(sorted(r["chosen"] for r in rows), sorted(legs))
            self.assertTrue(all(r["action"] == "sell" and r["instrument"] == "option" for r in rows))
            self.assertTrue(all(r["order"]["status"] == "submitted" for r in rows))
        finally:
            c.doCleanups()

    def test_manager_survives_a_failing_journal(self):
        import options_manager as om
        src = open(om.__file__, encoding="utf-8").read()
        fn = src[src.index("def manage_options_positions("):src.index("def register_options_entry(")]
        self.assertIn("decision_journal.record_option_exits(actions)", fn)
        hook = fn[fn.index("decision_journal.record_option_exits(actions)") - 200:]
        self.assertIn("try:", hook[:220])
        self.assertIn("except Exception as _dj_err:", hook)


class TestFloor(_JournalCase):
    def test_the_vm_floor_journals_each_order_with_its_plan(self):
        import test_floor_vol_managed as fh
        b = fh._Broker([fh._pos("QQQ", 60, 600), fh._pos("GLD", 21, 380)], fh.QUOTES)
        fh._run(b, fh._closes_with_daily_move(0.004))
        rows = self.rows()
        got = sorted((r["ticker"], r["action"], r["chosen"], r["qty"]) for r in rows)
        self.assertEqual(got, [("GLD", "scale_out", "sell_legacy_floor", 21), ("QQQ", "scale_out", "sell_legacy_floor", 60),
                               ("SPY", "scale_in", "vol_managed_spy", 84), ("SSO", "scale_in", "vol_managed_spy", 424)])
        spy = next(r for r in rows if r["ticker"] == "SPY")
        self.assertEqual(spy["instrument"], "floor")
        self.assertEqual(spy["sizing"]["target_exposure"], 1.5)
        self.assertEqual(spy["price"], 500.1)
        self.assertIn("against a 16% target puts the target exposure at 1.50x", spy["summary"])
        self.assertEqual(spy["order"]["status"], "submitted")

    def test_refused_orders_and_non_orders(self):
        n = dj.record_floor({"sigma": 0.2, "target_exposure": 0.8, "current_exposure": 1.0, "actions": [
            {"type": "floor_vm_sell", "ticker": "SPY", "qty": 5, "limit": 500.0, "ok": False, "order_id": None, "error": "market closed"},
            {"type": "floor_vm_wait", "ticker": "SSO", "reason": "order working"},
            {"type": "floor_vm_no_quote", "ticker": "SPY"}]})
        self.assertEqual(n, 1)
        r = self.rows()[0]
        self.assertEqual((r["action"], r["order"]["status"]), ("scale_out", "error"))
        self.assertIn("The order was refused: market closed.", r["summary"])
        self.assertEqual(dj.record_floor(None), 0)


class TestParityWithNode(unittest.TestCase):
    def test_file_cap_env_and_tables_match_the_ts_twin(self):
        import system_config
        self.assertEqual(system_config.BASE_CONFIG["DECISION_JOURNAL"], {"enabled": True, "max_bytes": dj.DEFAULT_MAX_BYTES})
        self.assertIn(f'export const DECISIONS_FILENAME = "{dj.JOURNAL_FILENAME}";', TS_SRC)
        self.assertIn(f'export const DECISION_JOURNAL_ENV = "{dj.ENV_VAR}";', TS_SRC)
        m = re.search(r"export const DECISION_JOURNAL_MAX_BYTES = ([\d_]+);", TS_SRC)
        self.assertEqual(int(m.group(1).replace("_", "")), dj.DEFAULT_MAX_BYTES)
        m = re.search(r"export const DECISION_ROW_MAX_BYTES = ([\d_]+);", TS_SRC)
        self.assertEqual(int(m.group(1).replace("_", "")), dj.ROW_MAX_BYTES)
        block = TS_SRC[TS_SRC.index("export const RULE_TEXT"):TS_SRC.index("export function ruleText")]
        ts_rules = dict(re.findall(r'\n  (\w+): "([^"]+)",', block))
        self.assertEqual(ts_rules, dj.RULE_TEXT, "the two plain-English gate tables must say the same thing")
        m = re.search(r"export const SUBSCORE_WEIGHTS[^=]*= \{([^}]*)\}", TS_SRC)
        ts_w = {k: float(v) for k, v in re.findall(r"(\w+): ([\d.]+)", m.group(1))}
        self.assertEqual(ts_w, dj.SUBSCORE_WEIGHTS)
        m = re.search(r"export const SUBSCORE_ADDITIVE = \[([^\]]*)\]", TS_SRC)
        self.assertEqual(tuple(re.findall(r'"(\w+)"', m.group(1))), dj.SUBSCORE_ADDITIVE)
        m = re.search(r"export const STAGE_KEYS = \[([^\]]*)\]", TS_SRC)
        self.assertEqual(tuple(re.findall(r'"(\w+)"', m.group(1))), dj.STAGE_KEYS)
        m = re.search(r"export const DECISION_ACTIONS = \[([^\]]*)\]", TS_SRC)
        self.assertEqual(tuple(re.findall(r'"(\w+)"', m.group(1))), dj.ACTIONS)

    def test_the_weights_are_deep_scores_own(self):
        src = open(os.path.join(_ROOT, "bot_engine.py"), encoding="utf-8").read()
        combo = src[src.index("combined_score = (\n        momentum_score"):]
        combo = combo[:combo.index(") + vol_premium_bonus")]
        found = dict(re.findall(r"(\w+?)(?:_score_val|_score) \* ([\d.]+)", combo))
        self.assertEqual({k: float(v) for k, v in found.items()}, dj.SUBSCORE_WEIGHTS)


class TestWiring(unittest.TestCase):
    def setUp(self):
        self.bot_engine = open(os.path.join(_ROOT, "bot_engine.py"), encoding="utf-8").read()

    def test_deep_score_returns_its_breakdown_and_the_trade_carries_it(self):
        ds = self.bot_engine[self.bot_engine.index("def deep_score("):self.bot_engine.index("def check_sector_correlation(")]
        self.assertIn("\"score_breakdown\": _score_breakdown if '_score_breakdown' in locals() else None,", ds)
        scan = self.bot_engine[self.bot_engine.index("def _scan_market_inner("):]
        for key in ('"score_breakdown": stock.get("score_breakdown")', '"sector": stock.get("sector")',
                    '"scan_regime": _scan_regime', '"score_threshold": _min_score_threshold',
                    '"decision_stages": stock.get("decision_stages")'):
            self.assertIn(key, scan)

    def test_every_scan_gate_reaches_the_journal(self):
        scan = self.bot_engine[self.bot_engine.index("def _scan_market_inner("):]
        note = scan[scan.index("def _note_gate("):scan.index("for _di, stock in enumerate(deep_scored):")]
        self.assertIn("_journal_skips.append(_gate_relabels[-1])", note)
        self.assertIn('"decision": "rejected_heat"', scan)
        self.assertIn('"decision": "rejected_spread"', scan)
        call = scan[scan.index("_dj.record_scan_skips("):][:700]
        self.assertIn("_sizing_blocks", call, "sizing refusals are journaled too")
        self.assertIn("deep_scored", call)
        self.assertIn("regime_state=_regime_state if '_regime_state' in dir() else None", call,
                      "the regime state (its risk budget) reaches the skip rows")

    def test_the_floor_and_tier_hooks(self):
        vm = self.bot_engine[self.bot_engine.index("def _manage_vol_managed_floor("):self.bot_engine.index("def floor_reduce_only_pass(")]
        self.assertIn("_dj_floor.record_floor(result)", vm)
        ts = open(os.path.join(_ROOT, "tiered_strategy.py"), encoding="utf-8").read()
        self.assertIn("decision_journal.record_tier_kill(shadow_actions, kill_reason", ts)
        self.assertIn("decision_journal.record_tier_rejections(rejections)", ts)

    def test_hooks_never_print_to_stdout(self):
        src = open(dj.__file__, encoding="utf-8").read()
        self.assertNotRegex(src, r"(?m)^\s*print\(", "bot.ts parses subprocess stdout")


if __name__ == "__main__":
    unittest.main()
