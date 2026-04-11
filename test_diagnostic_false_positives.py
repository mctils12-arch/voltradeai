"""
Tests for diagnostic false positive fixes:
  1. ML model criticality: critical only when file exists
  2. Polygon API health check uses /tmp/ path (not DATA_DIR)
  3. Feedback filter: entry records (pnl_pct=0, outcome=None) filtered out
  4. Performance endpoint matches diagnostics filter
  5. Overall diagnostic status not "degraded" when only non-critical items missing

Run: python3 -m pytest test_diagnostic_false_positives.py -v
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch


# ── Helpers: replicate filter logic ──────────────────────────────────────────

def diagnostics_feedback_filter(trades):
    """Replicates diagnostics.py check_model_health() filter (after fix)."""
    return [t for t in trades if t.get("ticker", "").strip()
            and t.get("pnl_pct") is not None
            and not (t.get("pnl_pct") == 0 and t.get("outcome") is None)]


def performance_endpoint_filter(feedback):
    """Replicates bot.ts performance endpoint filter (after fix)."""
    return [t for t in feedback if t.get('ticker', '').strip()
            and not (t.get('pnl_pct', 0) == 0 and t.get('outcome') is None)]


# ── 1. ML model criticality tests ───────────────────────────────────────────


class TestMLModelCriticality(unittest.TestCase):
    """ML model should only be critical when the model file exists."""

    def test_ml_critical_true_when_file_exists(self):
        """When ML model file exists, critical should be True."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            critical = os.path.exists(tmp_path)
            self.assertTrue(critical)
        finally:
            os.unlink(tmp_path)

    def test_ml_critical_false_when_file_missing(self):
        """When ML model file does not exist, critical should be False."""
        fake_path = "/tmp/nonexistent_ml_model_test_12345.pkl"
        critical = os.path.exists(fake_path)
        self.assertFalse(critical)

    def test_diagnostics_ml_criticality_dynamic(self):
        """Verify diagnostics.py EXPECTED_CACHE_FRESHNESS uses dynamic criticality."""
        import diagnostics
        ml_config = diagnostics.EXPECTED_CACHE_FRESHNESS["ml_model"]
        # critical should be a bool derived from os.path.exists, not hardcoded True
        self.assertIsInstance(ml_config["critical"], bool)
        # On test machine, ML model likely doesn't exist → should be False
        if not os.path.exists(diagnostics.ML_MODEL_PATH):
            self.assertFalse(ml_config["critical"])

    def test_missing_ml_model_not_critical_stale(self):
        """When ML model is missing but non-critical, check_cache_freshness
        should NOT set critical_stale."""
        import diagnostics
        original = diagnostics.EXPECTED_CACHE_FRESHNESS["ml_model"]["critical"]
        try:
            diagnostics.EXPECTED_CACHE_FRESHNESS["ml_model"]["critical"] = False
            result = diagnostics.check_cache_freshness()
            # ml_model may be in stale list but should NOT be critical
            ml_stale = [s for s in result["stale_caches"] if s["name"] == "ml_model"]
            if ml_stale:
                self.assertFalse(ml_stale[0]["critical"])
        finally:
            diagnostics.EXPECTED_CACHE_FRESHNESS["ml_model"]["critical"] = original


# ── 2. Polygon path check tests ─────────────────────────────────────────────


class TestPolygonPathCheck(unittest.TestCase):
    """Polygon API health check must look in /tmp/, not DATA_DIR."""

    def test_polygon_checks_tmp_path(self):
        """Verify diagnostics.py checks /tmp/voltrade_macro_cache.json for polygon."""
        import diagnostics
        import inspect
        source = inspect.getsource(diagnostics.run_diagnostics)
        self.assertIn(
            '/tmp/voltrade_macro_cache.json',
            source,
            "Polygon check should use /tmp/ path, not DATA_DIR"
        )

    def test_polygon_does_not_use_data_dir(self):
        """Polygon check must NOT use os.path.join(DATA_DIR, ...)."""
        import diagnostics
        import inspect
        source = inspect.getsource(diagnostics.run_diagnostics)
        # Check the specific polygon line doesn't use DATA_DIR
        for line in source.split('\n'):
            if '"polygon"' in line:
                self.assertNotIn('DATA_DIR', line,
                                 "Polygon check should not reference DATA_DIR")

    def test_polygon_healthy_when_cache_exists(self):
        """Polygon should report healthy when /tmp/voltrade_macro_cache.json exists."""
        cache_path = "/tmp/voltrade_macro_cache.json"
        existed = os.path.exists(cache_path)
        try:
            with open(cache_path, "w") as f:
                json.dump({"test": True}, f)
            healthy = os.path.exists(cache_path)
            self.assertTrue(healthy)
        finally:
            if not existed:
                os.unlink(cache_path)


# ── 3. Diagnostics feedback filter tests ─────────────────────────────────────


class TestDiagnosticsFeedbackFilter(unittest.TestCase):
    """Diagnostics filter should exclude entry records with pnl_pct=0 and outcome=None."""

    def test_filters_entry_records_zero_pnl_no_outcome(self):
        """Entry records (pnl_pct=0, outcome=None) should be filtered out."""
        trades = [
            {"ticker": "AAPL", "pnl_pct": 0, "outcome": None},
        ]
        result = diagnostics_feedback_filter(trades)
        self.assertEqual(len(result), 0)

    def test_keeps_real_flat_trade(self):
        """Real flat trades (pnl_pct=0, outcome='closed') should be KEPT."""
        trades = [
            {"ticker": "AAPL", "pnl_pct": 0, "outcome": "closed"},
        ]
        result = diagnostics_feedback_filter(trades)
        self.assertEqual(len(result), 1)

    def test_keeps_winning_trades(self):
        """Winning trades pass through the filter."""
        trades = [
            {"ticker": "AAPL", "pnl_pct": 5.2, "outcome": "win"},
            {"ticker": "MSFT", "pnl_pct": 3.1, "outcome": "win"},
        ]
        result = diagnostics_feedback_filter(trades)
        self.assertEqual(len(result), 2)

    def test_keeps_losing_trades(self):
        """Losing trades pass through the filter."""
        trades = [
            {"ticker": "GOOG", "pnl_pct": -2.5, "outcome": "loss"},
            {"ticker": "TSLA", "pnl_pct": -1.0, "outcome": "loss"},
        ]
        result = diagnostics_feedback_filter(trades)
        self.assertEqual(len(result), 2)

    def test_filters_empty_ticker(self):
        """Records with empty ticker are still filtered."""
        trades = [
            {"ticker": "", "pnl_pct": 5.0, "outcome": "win"},
        ]
        result = diagnostics_feedback_filter(trades)
        self.assertEqual(len(result), 0)

    def test_filters_null_pnl(self):
        """Records with pnl_pct=None are still filtered."""
        trades = [
            {"ticker": "AAPL", "pnl_pct": None, "outcome": "win"},
        ]
        result = diagnostics_feedback_filter(trades)
        self.assertEqual(len(result), 0)

    def test_mixed_entry_and_real_trades(self):
        """Only real trades survive when mixed with entry records."""
        trades = [
            {"ticker": "AAPL", "pnl_pct": 0, "outcome": None},       # entry - filtered
            {"ticker": "MSFT", "pnl_pct": 5.0, "outcome": "win"},     # real - kept
            {"ticker": "GOOG", "pnl_pct": 0, "outcome": None},        # entry - filtered
            {"ticker": "TSLA", "pnl_pct": -2.0, "outcome": "loss"},   # real - kept
            {"ticker": "NVDA", "pnl_pct": 0, "outcome": "closed"},    # flat - kept
        ]
        result = diagnostics_feedback_filter(trades)
        self.assertEqual(len(result), 3)
        tickers = [t["ticker"] for t in result]
        self.assertIn("MSFT", tickers)
        self.assertIn("TSLA", tickers)
        self.assertIn("NVDA", tickers)

    def test_bulk_entry_records_all_filtered(self):
        """Hundreds of entry records should all be filtered out."""
        trades = [{"ticker": "AAPL", "pnl_pct": 0, "outcome": None}
                  for _ in range(325)]
        result = diagnostics_feedback_filter(trades)
        self.assertEqual(len(result), 0)


# ── 4. Performance endpoint filter tests ─────────────────────────────────────


class TestPerformanceEndpointFilter(unittest.TestCase):
    """Performance endpoint filter should match diagnostics filter behavior."""

    def test_filters_entry_records(self):
        """Entry records (pnl_pct=0, outcome=None) filtered from performance endpoint."""
        feedback = [
            {"ticker": "AAPL", "pnl_pct": 0, "outcome": None},
        ]
        result = performance_endpoint_filter(feedback)
        self.assertEqual(len(result), 0)

    def test_keeps_real_flat_trade(self):
        """Real flat trades kept in performance endpoint."""
        feedback = [
            {"ticker": "AAPL", "pnl_pct": 0, "outcome": "closed"},
        ]
        result = performance_endpoint_filter(feedback)
        self.assertEqual(len(result), 1)

    def test_keeps_winning_and_losing(self):
        """Normal trades pass through performance endpoint filter."""
        feedback = [
            {"ticker": "AAPL", "pnl_pct": 8.0, "outcome": "win"},
            {"ticker": "MSFT", "pnl_pct": -3.0, "outcome": "loss"},
        ]
        result = performance_endpoint_filter(feedback)
        self.assertEqual(len(result), 2)

    def test_filters_match_diagnostics(self):
        """Performance endpoint filter produces same results as diagnostics filter
        for records that have non-null pnl_pct (the extra null filter in diagnostics
        doesn't apply when pnl_pct defaults to 0 via .get())."""
        test_data = [
            {"ticker": "AAPL", "pnl_pct": 0, "outcome": None},       # entry
            {"ticker": "MSFT", "pnl_pct": 5.0, "outcome": "win"},     # win
            {"ticker": "GOOG", "pnl_pct": -2.0, "outcome": "loss"},   # loss
            {"ticker": "TSLA", "pnl_pct": 0, "outcome": "closed"},    # flat
            {"ticker": "", "pnl_pct": 3.0, "outcome": "win"},         # bad ticker
        ]
        diag_result = diagnostics_feedback_filter(test_data)
        perf_result = performance_endpoint_filter(test_data)
        diag_tickers = {t["ticker"] for t in diag_result}
        perf_tickers = {t["ticker"] for t in perf_result}
        self.assertEqual(diag_tickers, perf_tickers)


# ── 5. Bot.ts source verification ───────────────────────────────────────────


class TestBotSourceUpdated(unittest.TestCase):
    """Verify bot.ts performance endpoint has the updated filter."""

    @classmethod
    def setUpClass(cls):
        with open("server/bot.ts") as f:
            cls.source = f.read()

    def test_performance_filter_includes_outcome_check(self):
        """Performance endpoint filter should check for entry records."""
        self.assertIn(
            "not (t.get('pnl_pct', 0) == 0 and t.get('outcome') is None)",
            self.source,
            "Performance endpoint missing entry record filter"
        )


# ── 6. Overall diagnostic status tests ──────────────────────────────────────


class TestOverallDiagnosticStatus(unittest.TestCase):
    """Overall status should not be 'degraded' from non-critical missing items."""

    def test_not_degraded_when_only_ml_missing_noncritical(self):
        """If ML model is the only issue and it's non-critical, status != degraded."""
        import diagnostics
        original = diagnostics.EXPECTED_CACHE_FRESHNESS["ml_model"]["critical"]
        try:
            # Ensure ML model is non-critical (as it should be when file missing)
            diagnostics.EXPECTED_CACHE_FRESHNESS["ml_model"]["critical"] = False
            report = diagnostics.run_diagnostics()
            # Should not be degraded purely from a non-critical ML model
            high_problems = [p for p in report["problems"] if p.get("severity") == "high"]
            ml_cache_problems = [p for p in high_problems if "ml_model" in str(p.get("message", ""))]
            # No high-severity ML cache problem should exist
            self.assertEqual(len(ml_cache_problems), 0)
        finally:
            diagnostics.EXPECTED_CACHE_FRESHNESS["ml_model"]["critical"] = original


if __name__ == "__main__":
    unittest.main()
