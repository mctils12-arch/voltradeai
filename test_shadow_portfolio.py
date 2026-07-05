#!/usr/bin/env python3
"""
Regression tests for shadow_portfolio.update_last_decision() and its two
call sites in bot_engine.py's scan_market() loop.

WHY THIS EXISTS (BUILD ORDER 4 #6, T-BOT half, 2026-07-05): log_candidate()
runs inside deep_score(), before scan_market()'s downstream per-candidate
filters (sector/correlation, spread) get a chance to reject a candidate
that already cleared MIN_SCORE. Before this fix, every such candidate was
permanently mislabeled "taken" in the shadow log even though it was never
traded — silently corrupting the exact RULE COST AUDIT questions
(open_questions.md) about correlation-block and spread-filter cost.
update_last_decision() corrects the label immediately after the real
rejection, without touching either filter's actual trading behavior.
"""
import importlib
import inspect
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import shadow_portfolio


class TestUpdateLastDecision(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_log_path = shadow_portfolio.SHADOW_LOG_PATH
        self._orig_lock_path = shadow_portfolio.SHADOW_LOCK_PATH
        shadow_portfolio.SHADOW_LOG_PATH = os.path.join(self._tmpdir.name, "shadow.json")
        shadow_portfolio.SHADOW_LOCK_PATH = shadow_portfolio.SHADOW_LOG_PATH + ".lock"

    def tearDown(self):
        shadow_portfolio.SHADOW_LOG_PATH = self._orig_log_path
        shadow_portfolio.SHADOW_LOCK_PATH = self._orig_lock_path
        self._tmpdir.cleanup()

    def _seed(self, ticker, decision="taken", age_seconds=1.0):
        ts = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat()
        records = shadow_portfolio._load_shadow_log()
        records.append({
            "ticker": ticker, "timestamp": ts, "score": 70.0,
            "decision": decision, "decision_reason": "seed",
            "entry_price": 100.0, "vxx_ratio": 1.0, "regime_label": "NEUTRAL",
            "features": {}, "outcomes": {"+5d": None, "+10d": None, "+20d": None},
            "code_version": "test-shadow",
        })
        shadow_portfolio._save_shadow_log(records)

    def test_updates_fresh_taken_record(self):
        self._seed("AMD", decision="taken", age_seconds=1.0)
        ok = shadow_portfolio.update_last_decision("AMD", "rejected_heat", "sector block")
        self.assertTrue(ok)
        records = shadow_portfolio._load_shadow_log()
        self.assertEqual(records[-1]["decision"], "rejected_heat")
        self.assertEqual(records[-1]["decision_reason"], "sector block")

    def test_noop_when_already_resolved(self):
        self._seed("MSFT", decision="rejected_score", age_seconds=1.0)
        ok = shadow_portfolio.update_last_decision("MSFT", "rejected_heat", "sector block")
        self.assertFalse(ok)
        records = shadow_portfolio._load_shadow_log()
        self.assertEqual(records[-1]["decision"], "rejected_score")

    def test_noop_when_record_is_stale(self):
        self._seed("NVDA", decision="taken", age_seconds=300.0)
        ok = shadow_portfolio.update_last_decision(
            "NVDA", "rejected_other", "spread", max_age_seconds=120.0,
        )
        self.assertFalse(ok)
        records = shadow_portfolio._load_shadow_log()
        self.assertEqual(records[-1]["decision"], "taken")

    def test_noop_when_no_matching_ticker(self):
        self._seed("TSLA", decision="taken", age_seconds=1.0)
        ok = shadow_portfolio.update_last_decision("AAPL", "rejected_heat", "sector block")
        self.assertFalse(ok)

    def test_only_touches_most_recent_record_for_ticker(self):
        # An older AMD record already resolved must be untouched; only the
        # freshest "taken" record for the ticker may be corrected.
        self._seed("AMD", decision="rejected_score", age_seconds=90.0)
        self._seed("AMD", decision="taken", age_seconds=1.0)
        ok = shadow_portfolio.update_last_decision("AMD", "rejected_heat", "sector block")
        self.assertTrue(ok)
        records = shadow_portfolio._load_shadow_log()
        amd_records = [r for r in records if r["ticker"] == "AMD"]
        self.assertEqual(amd_records[0]["decision"], "rejected_score")
        self.assertEqual(amd_records[1]["decision"], "rejected_heat")

    def test_empty_ticker_is_noop(self):
        self.assertFalse(shadow_portfolio.update_last_decision("", "rejected_heat"))


class TestBotEngineCallSites(unittest.TestCase):
    """
    Source-inspection: confirms scan_market()'s correlation and spread
    rejection sites call update_last_decision() with the correct decision
    bucket, wrapped defensively so a shadow-logging failure can never
    break the trading loop. Mirrors test_voltrade_daemon.py's pattern of
    pinning wiring via source inspection rather than a live scan.
    """

    @classmethod
    def setUpClass(cls):
        import bot_engine
        # The rejection sites live in _scan_market_inner(), the closure
        # scan_market() dispatches to under its timeout guard.
        cls.source = inspect.getsource(bot_engine._scan_market_inner)

    def test_correlation_block_logs_rejected_heat(self):
        idx = self.source.find("check_sector_correlation(")
        self.assertNotEqual(idx, -1, "check_sector_correlation call site not found")
        window = self.source[idx:idx + 800]
        self.assertIn("update_last_decision", window)
        self.assertIn('"rejected_heat"', window)
        self.assertIn("except Exception", window)

    def test_spread_filter_logs_rejected_other(self):
        idx = self.source.find("_spread_pct > 0.005")
        self.assertNotEqual(idx, -1, "spread filter call site not found")
        window = self.source[idx:idx + 600]
        self.assertIn("update_last_decision", window)
        self.assertIn('"rejected_other"', window)
        self.assertIn("except Exception", window)


if __name__ == "__main__":
    unittest.main()
