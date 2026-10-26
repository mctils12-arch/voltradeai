"""
[RULE-REVIEW] 2026-10-26: bot.ts's executeTrades admission loop applies
~15 gates after scan_market returns, and every candidate they dropped stayed
labelled "taken" in the shadow log. update_decisions_batch() is the ONE call
per admission loop that relabels those records with the real gate, so each
Node rule earns a prevention-P&L the same way scan_market's own
sector/spread/sizing filters already do through update_last_decision().
"""
import inspect
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import shadow_portfolio as sp


class _TmpLog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = (sp.SHADOW_LOG_PATH, sp.SHADOW_LOCK_PATH)
        sp.SHADOW_LOG_PATH = os.path.join(self._tmp.name, "shadow.json")
        sp.SHADOW_LOCK_PATH = sp.SHADOW_LOG_PATH + ".lock"

    def tearDown(self):
        sp.SHADOW_LOG_PATH, sp.SHADOW_LOCK_PATH = self._orig
        self._tmp.cleanup()

    def _seed(self, ticker, decision="taken", age_seconds=1.0):
        records = sp._load_shadow_log()
        ts = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
        records.append({
            "ticker": ticker, "timestamp": ts.isoformat(), "score": 70.0,
            "decision": decision, "decision_reason": "seed", "entry_price": 10.0,
            "outcomes": {"+5d": None, "+10d": None, "+20d": None},
        })
        sp._save_shadow_log(records)

    def _by_ticker(self):
        return {r["ticker"]: r for r in sp._load_shadow_log()}


class TestUpdateDecisionsBatch(_TmpLog):
    def test_relabels_every_fresh_taken_record_in_one_save(self):
        for t in ("AMD", "NVDA", "TSLA"):
            self._seed(t)
        saves = []
        real_save = sp._save_shadow_log
        sp._save_shadow_log = lambda recs: saves.append(len(recs)) or real_save(recs)
        try:
            out = sp.update_decisions_batch([
                {"ticker": "AMD", "decision": "rejected_size_cap", "reason": "$9000 exceeds ceiling 8.0%"},
                {"ticker": "nvda", "decision": "rejected_corr_cap", "reason": "tech 45% > 40%"},
                {"ticker": "TSLA", "decision": "rejected_exposure_cap", "reason": "loop ended"},
            ])
        finally:
            sp._save_shadow_log = real_save
        self.assertEqual(out["received"], 3)
        self.assertEqual(out["updated"], 3)
        self.assertEqual(out["skipped"], 0)
        self.assertTrue(out["saved"])
        self.assertEqual(saves, [3], "three relabels must cost exactly one save")
        recs = self._by_ticker()
        self.assertEqual(recs["AMD"]["decision"], "rejected_size_cap")
        self.assertEqual(recs["AMD"]["decision_reason"], "$9000 exceeds ceiling 8.0%")
        self.assertEqual(recs["NVDA"]["decision"], "rejected_corr_cap")
        self.assertEqual(recs["TSLA"]["decision"], "rejected_exposure_cap")

    def test_never_moves_a_record_off_a_resolved_bucket(self):
        self._seed("MSFT", decision="rejected_heat")
        out = sp.update_decisions_batch([{"ticker": "MSFT", "decision": "rejected_held", "reason": "x"}])
        self.assertEqual((out["updated"], out["skipped"]), (0, 1))
        self.assertFalse(out["saved"])
        self.assertEqual(self._by_ticker()["MSFT"]["decision"], "rejected_heat")

    def test_respects_the_age_window(self):
        self._seed("AAPL", age_seconds=3600.0)
        out = sp.update_decisions_batch(
            [{"ticker": "AAPL", "decision": "rejected_node_score", "reason": "x"}],
            max_age_seconds=1800.0,
        )
        self.assertEqual(out["updated"], 0)
        self.assertEqual(self._by_ticker()["AAPL"]["decision"], "taken")
        # Inside the window the same record is relabelled.
        out = sp.update_decisions_batch(
            [{"ticker": "AAPL", "decision": "rejected_node_score", "reason": "x"}],
            max_age_seconds=7200.0,
        )
        self.assertEqual(out["updated"], 1)

    def test_only_touches_the_freshest_record_for_a_ticker(self):
        self._seed("AMD", decision="rejected_score", age_seconds=90.0)
        self._seed("AMD", decision="taken", age_seconds=1.0)
        sp.update_decisions_batch([{"ticker": "AMD", "decision": "rejected_dup_order", "reason": "x"}])
        amd = [r for r in sp._load_shadow_log() if r["ticker"] == "AMD"]
        self.assertEqual([r["decision"] for r in amd], ["rejected_score", "rejected_dup_order"])

    def test_malformed_entries_are_skipped_not_raised(self):
        self._seed("AMD")
        out = sp.update_decisions_batch([
            "not a dict", {"decision": "rejected_held"}, {"ticker": "AMD"},
            {"ticker": "AMD", "decision": "rejected_held"},
        ])
        self.assertEqual(out["received"], 4)
        self.assertEqual(out["skipped"], 3)
        self.assertEqual(out["updated"], 1)
        self.assertEqual(self._by_ticker()["AMD"]["decision_reason"], "")

    def test_empty_or_non_list_input_is_a_noop(self):
        self._seed("AMD")
        for bad in ([], None, "AMD", {"ticker": "AMD"}):
            out = sp.update_decisions_batch(bad)
            self.assertEqual((out["received"], out["updated"], out["saved"]), (0, 0, False))
        self.assertEqual(self._by_ticker()["AMD"]["decision"], "taken")

    def test_reason_is_truncated_like_update_last_decision(self):
        self._seed("AMD")
        sp.update_decisions_batch([{"ticker": "AMD", "decision": "rejected_held", "reason": "r" * 500}])
        self.assertEqual(len(self._by_ticker()["AMD"]["decision_reason"]), 200)

    def test_update_last_decision_still_works_through_the_shared_helper(self):
        self._seed("AMD")
        self.assertTrue(sp.update_last_decision("AMD", "rejected_heat", "sector"))
        self.assertEqual(self._by_ticker()["AMD"]["decision"], "rejected_heat")


class TestDaemonRoute(unittest.TestCase):
    def test_route_targets_the_batch_function_and_binds_bot_ts_args(self):
        from voltrade_daemon import RPCDispatcher
        module_name, attr_name = RPCDispatcher()._routes["shadow_update_decisions"]
        self.assertEqual((module_name, attr_name), ("shadow_portfolio", "update_decisions_batch"))
        fn = getattr(sp, attr_name)
        # The dispatcher calls fn(**args) when the signature binds — this is
        # the exact args shape bot.ts sends.
        inspect.signature(fn).bind(updates=[{"ticker": "AMD", "decision": "rejected_held", "reason": ""}],
                                   max_age_seconds=1800)

    def test_default_window_matches_bot_ts_constant(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "server", "bot.ts"), encoding="utf8") as f:
            bot = f.read()
        self.assertIn(f"const SHADOW_GATE_RELABEL_MAX_AGE_S = {int(sp.NODE_GATE_RELABEL_MAX_AGE_S)};", bot)


if __name__ == "__main__":
    unittest.main()
