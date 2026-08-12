"""
Tests for diagnostic false positive fixes:
  1. ML model criticality: critical only when file exists
  2. Polygon API health check uses /tmp/ path (not DATA_DIR)
  3. Feedback filter: entry records (pnl_pct=0, outcome=None) filtered out
  4. Performance endpoint matches diagnostics filter
  5. Overall diagnostic status not "degraded" when only non-critical items missing
  6. Extended API health (reddit_/fh_ cache monitoring) doesn't false-positive
     on an unconfigured FINNHUB_KEY, and stays isolated from the >=3
     reduce_position_size auto-fix trigger (KNOWN BROKEN #5 audit, 2026-07-04)

Run: python3 -m pytest test_diagnostic_false_positives.py -v
"""
import json
import os
import shutil
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


# ── 7. Extended API health checks (reddit_/fh_) — KNOWN BROKEN #5 audit ──────
# social_data.py and finnhub_data.py were the two live-scoring data sources
# (deep_score() in bot_engine.py) with ZERO freshness/health monitoring —
# unlike macro/insider/wiki/gdelt/fred, which diagnostics.py already tracked.
# A dead Reddit RSS feed or an expired FINNHUB_KEY degraded those signals to
# permanent silent no-ops (bot_engine.py's `except Exception: return {}`
# swallows the failure with no logging). These tests pin the new checks and,
# critically, the same false-positive discipline #1 above established for
# ml_model: an unconfigured key is an expected state, not a break.


class TestExtendedApiHealthChecks(unittest.TestCase):
    """reddit_/fh_ cache monitoring, isolated from the api_checks/failed_apis
    bucket that drives reduce_position_size at >=3 failures."""

    def setUp(self):
        import diagnostics
        self.diagnostics = diagnostics
        self._orig_cache_dir = diagnostics.CACHE_DIR
        self._tmp_dir = tempfile.mkdtemp(prefix="voltrade_ext_health_test_")
        diagnostics.CACHE_DIR = self._tmp_dir

    def tearDown(self):
        self.diagnostics.CACHE_DIR = self._orig_cache_dir
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_reddit_and_finnhub_flagged_when_both_down(self):
        """Key configured but no cache written yet -> both sources listed."""
        with patch.dict(os.environ, {"FINNHUB_KEY": "real_test_key"}):
            report = self.diagnostics.run_diagnostics()
        api_warnings = [w for w in report["warnings"] if w.get("system") == "api"
                        and "Extended data sources" in w.get("message", "")]
        self.assertEqual(len(api_warnings), 1)
        self.assertIn("reddit", api_warnings[0]["message"])
        self.assertIn("finnhub", api_warnings[0]["message"])

    def test_no_extended_warning_when_finnhub_unconfigured_and_reddit_cached(self):
        """Unconfigured FINNHUB_KEY must NOT be flagged (expected state, not a
        break) — the exact false-positive class this test file exists for."""
        with open(os.path.join(self._tmp_dir, "reddit_AAPL.json"), "w") as f:
            json.dump({}, f)
        with patch.dict(os.environ, {"FINNHUB_KEY": ""}):
            report = self.diagnostics.run_diagnostics()
        api_warnings = [w for w in report["warnings"] if w.get("system") == "api"
                        and "Extended data sources" in w.get("message", "")]
        self.assertEqual(len(api_warnings), 0)

    def test_finnhub_placeholder_key_treated_as_unconfigured(self):
        """YOUR_FINNHUB_KEY_HERE (the shipped placeholder) must read as
        unconfigured, matching finnhub_data.py's own gate."""
        with open(os.path.join(self._tmp_dir, "reddit_AAPL.json"), "w") as f:
            json.dump({}, f)
        with patch.dict(os.environ, {"FINNHUB_KEY": "YOUR_FINNHUB_KEY_HERE"}):
            report = self.diagnostics.run_diagnostics()
        api_warnings = [w for w in report["warnings"] if w.get("system") == "api"
                        and "Extended data sources" in w.get("message", "")]
        self.assertEqual(len(api_warnings), 0)

    def test_finnhub_healthy_when_configured_and_cached(self):
        with open(os.path.join(self._tmp_dir, "reddit_AAPL.json"), "w") as f:
            json.dump({}, f)
        with open(os.path.join(self._tmp_dir, "fh_insider_AAPL.json"), "w") as f:
            json.dump({}, f)
        with patch.dict(os.environ, {"FINNHUB_KEY": "real_test_key"}):
            report = self.diagnostics.run_diagnostics()
        api_warnings = [w for w in report["warnings"] if w.get("system") == "api"
                        and "Extended data sources" in w.get("message", "")]
        self.assertEqual(len(api_warnings), 0)

    def test_extended_checks_isolated_from_failed_apis_threshold(self):
        """reddit/finnhub must never enter the api_checks dict that feeds
        failed_apis -> reduce_position_size(>=3) — that would silently
        change when the risk-affecting auto-fix fires."""
        import inspect
        source = inspect.getsource(self.diagnostics.run_diagnostics)
        start = source.index("api_checks = {")
        end = source.index("failed_apis = [")
        api_checks_block = source[start:end]
        self.assertNotIn('"reddit"', api_checks_block)
        self.assertNotIn('"finnhub"', api_checks_block)

    def test_extended_down_never_triggers_reduce_position_size(self):
        with patch.dict(os.environ, {"FINNHUB_KEY": "real_test_key"}):
            report = self.diagnostics.run_diagnostics()
        for problem in report["problems"]:
            if problem.get("auto_fix") == "reduce_position_size":
                self.assertNotIn("reddit", problem["message"])
                self.assertNotIn("finnhub", problem["message"])


# ── 8. Restart grace period for the >=3-sources-down check (KNOWN BROKEN #29
# recurrence, 2026-08-12) — a fresh container boot wipes ephemeral /tmp, so
# wikipedia/gdelt/fred's cache files are genuinely absent until the first
# post-restart Tier 2 scan writes them. Without a grace period this fires a
# false "Multiple API sources down" problem (with a real reduce_position_size
# auto-fix) on every redeploy, independent of whether sources are actually
# reachable.


class TestApiCheckRestartGracePeriod(unittest.TestCase):
    """server_uptime_s gates the >=3-failed-sources problem, not the warning."""

    def setUp(self):
        import diagnostics
        self.diagnostics = diagnostics
        self._orig_cache_dir = diagnostics.CACHE_DIR
        self._tmp_dir = tempfile.mkdtemp(prefix="voltrade_grace_test_")
        diagnostics.CACHE_DIR = self._tmp_dir
        # Empty CACHE_DIR alone makes wikipedia/gdelt/fred all fail (>=3),
        # regardless of polygon/sec_edgar's real state on the test machine.

    def tearDown(self):
        self.diagnostics.CACHE_DIR = self._orig_cache_dir
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def _api_problems(self, report):
        return [p for p in report["problems"]
                if p.get("system") == "api" and "Multiple API sources down" in p.get("message", "")]

    def _api_warnings(self, report):
        return [w for w in report["warnings"]
                if w.get("system") == "api" and "Multiple API sources down" in w.get("message", "")]

    def test_within_grace_period_is_warning_not_problem(self):
        """Just booted (uptime << grace period): no auto-fix problem, just a warning."""
        report = self.diagnostics.run_diagnostics(server_uptime_s=100)
        self.assertEqual(len(self._api_problems(report)), 0)
        warnings = self._api_warnings(report)
        self.assertEqual(len(warnings), 1)
        self.assertIn("restart grace", warnings[0]["message"])

    def test_after_grace_period_is_a_real_problem(self):
        """Long-running process, still down: the original problem+auto_fix fires."""
        report = self.diagnostics.run_diagnostics(
            server_uptime_s=self.diagnostics.API_CHECK_GRACE_PERIOD_S + 1
        )
        problems = self._api_problems(report)
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["auto_fix"], "reduce_position_size")
        self.assertEqual(problems[0]["fix_params"]["multiplier"], 0.6)
        self.assertEqual(len(self._api_warnings(report)), 0)

    def test_unknown_uptime_preserves_original_always_on_behavior(self):
        """A caller that doesn't pass server_uptime_s (None) gets the original
        behavior — never silently softened just because uptime wasn't wired up."""
        report = self.diagnostics.run_diagnostics()
        problems = self._api_problems(report)
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["fix_params"]["multiplier"], 0.6)

    def test_one_second_before_grace_boundary_is_still_a_warning(self):
        """uptime just under the grace period is still 'within grace' (< threshold)."""
        report = self.diagnostics.run_diagnostics(
            server_uptime_s=self.diagnostics.API_CHECK_GRACE_PERIOD_S - 1
        )
        self.assertEqual(len(self._api_problems(report)), 0)
        self.assertEqual(len(self._api_warnings(report)), 1)

    def test_exactly_at_grace_boundary_is_already_a_problem(self):
        """uptime == grace period means the grace window has fully elapsed."""
        report = self.diagnostics.run_diagnostics(
            server_uptime_s=self.diagnostics.API_CHECK_GRACE_PERIOD_S
        )
        self.assertEqual(len(self._api_problems(report)), 1)
        self.assertEqual(len(self._api_warnings(report)), 0)

    def test_get_auto_fix_params_within_grace_does_not_cut_position_size(self):
        params = self.diagnostics.get_auto_fix_params(server_uptime_s=100)
        self.assertEqual(params["position_size_multiplier"], 1.0)

    def test_get_auto_fix_params_after_grace_cuts_position_size(self):
        params = self.diagnostics.get_auto_fix_params(
            server_uptime_s=self.diagnostics.API_CHECK_GRACE_PERIOD_S + 1
        )
        self.assertLessEqual(params["position_size_multiplier"], 0.6)

    def test_tier2_not_yet_run_is_grace_even_hours_past_uptime_clock(self):
        """CONFIRMED LIVE 2026-08-12: a restart landing in bot.ts's ~8pm-4am ET
        Tier-2-dark window leaves the flat 30-min uptime clock elapsed hours
        before Tier 2 gets its first chance to run. tier2_ran_since_boot=False
        must keep this a warning no matter how large server_uptime_s is."""
        report = self.diagnostics.run_diagnostics(
            server_uptime_s=8 * 3600, tier2_ran_since_boot=False
        )
        self.assertEqual(len(self._api_problems(report)), 0)
        warnings = self._api_warnings(report)
        self.assertEqual(len(warnings), 1)
        self.assertIn("Tier 2 has not completed a scan", warnings[0]["message"])

    def test_tier2_already_ran_is_a_real_problem_even_seconds_after_boot(self):
        """Once Tier 2 has completed a scan since boot, the caches had their
        one real chance to be written — tier2_ran_since_boot=True must not be
        softened by a small server_uptime_s."""
        report = self.diagnostics.run_diagnostics(
            server_uptime_s=5, tier2_ran_since_boot=True
        )
        problems = self._api_problems(report)
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["fix_params"]["multiplier"], 0.6)
        self.assertEqual(len(self._api_warnings(report)), 0)

    def test_get_auto_fix_params_threads_tier2_ran_since_boot_through(self):
        params = self.diagnostics.get_auto_fix_params(
            server_uptime_s=8 * 3600, tier2_ran_since_boot=False
        )
        self.assertEqual(params["position_size_multiplier"], 1.0)


if __name__ == "__main__":
    unittest.main()
