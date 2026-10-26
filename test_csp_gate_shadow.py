"""
[RULE-REVIEW] 2026-10-26: the last blind rule set on the counterfactual
program. bot.ts's tier dispatcher drops SELL_CSP actions at OPTIONS-SLOT-FULL
and CSP-SKIP (and the remainder after a kill-switch break), and a CSP never
had a shadow record to relabel — tier1_csp_core proposes it, deep_score never
sees it. tiered_strategy.log_csp_gate_shadow logs one record per dropped
action in log_masterkill_csp_shadow's convention; this pins its contract.
"""
import json
import os
import tempfile
import unittest

import shadow_portfolio as sp
import tiered_strategy as ts


class TestLogCspGateShadow(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_paths = (sp.SHADOW_LOG_PATH, sp.SHADOW_LOCK_PATH)
        sp.SHADOW_LOG_PATH = os.path.join(self._tmp.name, "shadow.json")
        sp.SHADOW_LOCK_PATH = sp.SHADOW_LOG_PATH + ".lock"
        self._orig_lookup = ts._layer2_price_lookup
        ts._layer2_price_lookup = lambda: {"AAPL": 150.25, "MSFT": 410.0}

    def tearDown(self):
        sp.SHADOW_LOG_PATH, sp.SHADOW_LOCK_PATH = self._orig_paths
        ts._layer2_price_lookup = self._orig_lookup
        self._tmp.cleanup()

    def _records(self):
        if not os.path.exists(sp.SHADOW_LOG_PATH):
            return []
        with open(sp.SHADOW_LOG_PATH) as f:
            return json.load(f)

    def test_logs_one_record_per_priced_rejection_in_the_csp_convention(self):
        out = ts.log_csp_gate_shadow([
            {"ticker": "aapl", "decision": "rejected_options_slots", "reason": "options slots full (6/6)",
             "metadata": {"vxx_ratio": 1.12, "regime_label": "CAUTION"}},
            {"ticker": "MSFT", "decision": "rejected_csp_resubmit", "reason": "swept 3m ago", "metadata": {}},
        ])
        self.assertEqual(out, {"received": 2, "logged": 2, "skipped_no_price": 0, "skipped": 0})
        recs = {r["ticker"]: r for r in self._records()}
        aapl = recs["AAPL"]
        self.assertEqual(aapl["decision"], "rejected_options_slots")
        self.assertEqual(aapl["decision_reason"], "options slots full (6/6)")
        self.assertEqual(aapl["entry_price"], 150.25)
        self.assertEqual(aapl["vxx_ratio"], 1.12)
        self.assertEqual(aapl["regime_label"], "CAUTION")
        # Marked as what the bot would have done, so the long-stock labeler's
        # proxy nature is visible to every reader of this bucket.
        self.assertEqual((aapl["side"], aapl["action_label"], aapl["trade_type"]), ("sell", "SELL CSP", "options"))
        self.assertEqual(aapl["score"], 0.0)
        msft = recs["MSFT"]
        self.assertEqual(msft["decision"], "rejected_csp_resubmit")
        self.assertEqual(msft["regime_label"], "unknown")
        self.assertEqual(msft["vxx_ratio"], 1.0)

    def test_unpriced_and_malformed_rejections_are_counted_not_logged(self):
        out = ts.log_csp_gate_shadow([
            {"ticker": "ZZZZ", "decision": "rejected_options_slots", "reason": "no layer-2 price"},
            "not a dict",
            {"decision": "rejected_options_slots"},
            {"ticker": "AAPL"},
            {"ticker": "AAPL", "decision": "rejected_killswitch", "reason": "r" * 500, "metadata": "bad"},
        ])
        self.assertEqual(out, {"received": 5, "logged": 1, "skipped_no_price": 1, "skipped": 3})
        recs = self._records()
        self.assertEqual(len(recs), 1)
        self.assertEqual(len(recs[0]["decision_reason"]), 200)
        self.assertEqual(recs[0]["regime_label"], "unknown")

    def test_empty_or_non_list_is_a_noop(self):
        for bad in ([], None, {"ticker": "AAPL"}, "AAPL"):
            self.assertEqual(ts.log_csp_gate_shadow(bad)["logged"], 0)
        self.assertEqual(self._records(), [])

    def test_price_lookup_failure_never_raises(self):
        def boom():
            raise RuntimeError("layer-2 cache unreadable")
        ts._layer2_price_lookup = boom
        out = ts.log_csp_gate_shadow([{"ticker": "AAPL", "decision": "rejected_options_slots", "reason": "x"}])
        self.assertEqual(out["logged"], 0)
        self.assertIn("layer-2 cache unreadable", out["error"])
        self.assertEqual(self._records(), [])

    def test_default_price_lookup_reads_the_layer2_cache_like_masterkill_does(self):
        import inspect
        src = inspect.getsource(self._orig_lookup)
        self.assertIn("_load_layer2_cache", src)
        self.assertIn('e.get("price", 0) > 0', src)


class TestDaemonRoute(unittest.TestCase):
    def test_route_targets_the_logger_and_binds_bot_ts_args(self):
        import inspect
        from voltrade_daemon import RPCDispatcher
        module_name, attr_name = RPCDispatcher()._routes["csp_gate_shadow"]
        self.assertEqual((module_name, attr_name), ("tiered_strategy", "log_csp_gate_shadow"))
        inspect.signature(getattr(ts, attr_name)).bind(rejections=[{"ticker": "AAPL", "decision": "x", "reason": ""}])


if __name__ == "__main__":
    unittest.main()
