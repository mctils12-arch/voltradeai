"""
[REPAIR 2026-07-17] ml_retrain_safe.py runs as a FRESH subprocess every
hourly/4am retrain and only ever imports `ml_model_v2` (never `bot_engine`,
which is where the Alpaca rate limiter used to get installed). Its
_fetch_training_bars() therefore made completely unthrottled bursts of
Alpaca calls, concurrently with the daemon's own throttled Tier-2 scan
traffic — and any resulting error (429, timeout, etc.) was swallowed by a
bare `except: continue`, collapsing to the content-free "Could not fetch
training bars" seen recurring live in /api/diag/audit on 2026-07-17
(5 TIER3-ML-ERROR entries in ~90 minutes, model stuck at 26+ hours stale).

This battery pins two independent fixes:
  1. Importing ml_model_v2 alone (the exact ml_retrain_safe.py scenario)
     installs the global Alpaca throttle.
  2. _fetch_training_bars() surfaces the real failure cause instead of
     silently discarding it, and train_model() propagates that cause into
     its "reason" field.

Run: python3 -m pytest test_ml_retrain_throttle.py -q
"""
import os
import subprocess
import sys
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.abspath(__file__))


class TestFreshSubprocessInstallsThrottle(unittest.TestCase):
    def test_importing_ml_model_v2_alone_patches_requests(self):
        """Reproduces ml_retrain_safe.py's exact import shape: a brand-new
        Python process that imports ONLY ml_model_v2, never bot_engine.
        A/B: on the pre-fix code this prints False (unpatched); post-fix
        prints True."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import ml_model_v2, alpaca_rate_limiter; print(alpaca_rate_limiter._patched)"],
            cwd=REPO, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-1000:])
        self.assertEqual(result.stdout.strip(), "True",
            "ml_model_v2 must install the global Alpaca throttle on import "
            "so ml_retrain_safe.py's fresh subprocess never makes unthrottled "
            "bursts of Alpaca calls")


class TestFetchTrainingBarsErrorVisibility(unittest.TestCase):
    def setUp(self):
        import ml_model_v2
        self.ml = ml_model_v2

    def test_http_error_surfaces_as_last_error(self):
        """Every batch returns a non-200 -> bars empty, error carries the
        real HTTP status instead of being swallowed."""
        class FakeResp:
            status_code = 429
            text = "rate limit exceeded"
            def json(self): return {}

        with mock.patch.object(self.ml, "requests") as mock_requests:
            mock_requests.get.return_value = FakeResp()
            bars, error = self.ml._fetch_training_bars(days=30, max_tickers=3)

        self.assertEqual(bars, {})
        self.assertIsNotNone(error)
        self.assertIn("429", error)

    def test_exception_surfaces_as_last_error(self):
        with mock.patch.object(self.ml, "requests") as mock_requests:
            mock_requests.get.side_effect = ConnectionError("boom")
            bars, error = self.ml._fetch_training_bars(days=30, max_tickers=3)

        self.assertEqual(bars, {})
        self.assertIsNotNone(error)
        self.assertIn("boom", error)

    def test_success_returns_no_error(self):
        class FakeResp:
            status_code = 200
            def json(self): return {"bars": {"AAPL": [{"c": 1}]}}

        with mock.patch.object(self.ml, "requests") as mock_requests:
            mock_requests.get.return_value = FakeResp()
            bars, error = self.ml._fetch_training_bars(days=30, max_tickers=3)

        self.assertIn("AAPL", bars)
        self.assertIsNone(error)


class TestTrainModelPropagatesFetchReason(unittest.TestCase):
    def test_reason_includes_the_real_fetch_error(self):
        import ml_model_v2
        if not ml_model_v2.HAS_SKLEARN:
            self.skipTest("sklearn not installed in this environment")
        with mock.patch.object(ml_model_v2, "_fetch_training_bars",
                                return_value=({}, "HTTP 429: rate limit exceeded")):
            result = ml_model_v2.train_model(fast_mode=True)

        self.assertEqual(result["status"], "failed")
        self.assertIn("Could not fetch training bars", result["reason"])
        self.assertIn("429", result["reason"],
            "the real cause must survive into train_model()'s reason field, "
            "not collapse back to a bare 'Could not fetch training bars'")


if __name__ == "__main__":
    unittest.main()
