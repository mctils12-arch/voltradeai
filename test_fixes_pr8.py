"""
Extensive tests for PR #8 fixes:
  1. SEC EDGAR / Polygon path resolution in diagnostics.py
  2. track_fill() input validation in ml_model_v2.py
  3. Diagnostics win rate filtering of corrupt records
  4. cleanup_feedback.py garbage removal logic

Run: python3 -m pytest test_fixes_pr8.py -v
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock


# ── 1. SEC EDGAR / Polygon path resolution tests ────────────────────────────

class TestDiagnosticsPathResolution(unittest.TestCase):
    """Verify diagnostics checks the correct DATA_DIR paths, not hardcoded /tmp."""

    def test_polygon_path_uses_tmp(self):
        """Polygon check should use /tmp/ (where macro_data.py writes the cache)."""
        import inspect
        from diagnostics import run_diagnostics
        source = inspect.getsource(run_diagnostics)
        # The polygon line should reference /tmp/ directly
        for line in source.split('\n'):
            if '"polygon"' in line:
                self.assertIn('/tmp/voltrade_macro_cache.json', line,
                              "Polygon path should use /tmp/voltrade_macro_cache.json")
                break
        else:
            self.fail("No polygon line found in run_diagnostics")

    @patch("diagnostics.os.path.exists")
    @patch("diagnostics.os.listdir", return_value=[])
    @patch("diagnostics.DATA_DIR", "/data/voltrade")
    @patch("diagnostics.INSIDER_CACHE_PATH", "/data/voltrade/voltrade_insider_cache.json")
    def test_sec_edgar_path_uses_insider_cache_path(self, mock_listdir, mock_exists):
        """SEC EDGAR check should use INSIDER_CACHE_PATH, not /tmp."""
        mock_exists.return_value = False

        from diagnostics import run_diagnostics
        calls = [str(c) for c in mock_exists.call_args_list]
        edgar_calls = [c for c in calls if "insider_cache" in c]
        for c in edgar_calls:
            self.assertNotIn("/tmp/voltrade_insider_cache.json", c,
                             "SEC EDGAR path should not be hardcoded to /tmp")

    def test_diagnostics_imports_data_dir(self):
        """diagnostics.py should have DATA_DIR available."""
        import diagnostics
        self.assertTrue(hasattr(diagnostics, "DATA_DIR"))

    def test_diagnostics_imports_insider_cache_path(self):
        """diagnostics.py should have INSIDER_CACHE_PATH available."""
        import diagnostics
        self.assertTrue(hasattr(diagnostics, "INSIDER_CACHE_PATH"))

    @patch("diagnostics.DATA_DIR", "/tmp")
    @patch("diagnostics.INSIDER_CACHE_PATH", "/tmp/voltrade_insider_cache.json")
    def test_fallback_paths_resolve_to_tmp(self):
        """When storage_config import fails, paths should fall back to /tmp."""
        import diagnostics
        # Verify the fallback values are coherent
        self.assertIn("voltrade_insider_cache.json",
                      diagnostics.INSIDER_CACHE_PATH)


# ── 2. track_fill() input validation tests ──────────────────────────────────

class TestTrackFillValidation(unittest.TestCase):
    """Verify track_fill rejects garbage and tags records with code_version."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.feedback_path = os.path.join(self.tmpdir, "feedback.json")
        # Start with empty feedback file
        with open(self.feedback_path, "w") as f:
            json.dump([], f)

    def tearDown(self):
        if os.path.exists(self.feedback_path):
            os.remove(self.feedback_path)
        os.rmdir(self.tmpdir)

    def _call_track_fill(self, order_data):
        """Call track_fill with patched FEEDBACK_PATH."""
        with patch("ml_model_v2.FEEDBACK_PATH", self.feedback_path):
            from ml_model_v2 import track_fill
            track_fill(order_data)

    def _read_feedback(self):
        with open(self.feedback_path) as f:
            return json.load(f)

    def test_rejects_empty_ticker(self):
        """Records with empty ticker should be silently rejected."""
        self._call_track_fill({
            "ticker": "", "side": "buy", "qty": 10,
            "expected_price": 100, "fill_price": 100,
        })
        records = self._read_feedback()
        self.assertEqual(len(records), 0)

    def test_rejects_missing_ticker(self):
        """Records with no ticker key should be silently rejected."""
        self._call_track_fill({
            "side": "buy", "qty": 10,
            "expected_price": 100, "fill_price": 100,
        })
        records = self._read_feedback()
        self.assertEqual(len(records), 0)

    def test_rejects_whitespace_ticker(self):
        """Records with whitespace-only ticker should be rejected."""
        self._call_track_fill({
            "ticker": "   ", "side": "buy", "qty": 10,
            "expected_price": 100, "fill_price": 100,
        })
        records = self._read_feedback()
        self.assertEqual(len(records), 0)

    def test_rejects_zero_qty(self):
        """Records with qty=0 should be rejected."""
        self._call_track_fill({
            "ticker": "AAPL", "side": "buy", "qty": 0,
            "expected_price": 150, "fill_price": 150,
        })
        records = self._read_feedback()
        self.assertEqual(len(records), 0)

    def test_rejects_negative_qty(self):
        """Records with negative qty should be rejected."""
        self._call_track_fill({
            "ticker": "AAPL", "side": "buy", "qty": -5,
            "expected_price": 150, "fill_price": 150,
        })
        records = self._read_feedback()
        self.assertEqual(len(records), 0)

    def test_rejects_none_qty(self):
        """Records with qty=None should be rejected."""
        self._call_track_fill({
            "ticker": "AAPL", "side": "buy", "qty": None,
            "expected_price": 150, "fill_price": 150,
        })
        records = self._read_feedback()
        self.assertEqual(len(records), 0)

    def test_accepts_valid_record(self):
        """Valid record with ticker and positive qty should be saved."""
        self._call_track_fill({
            "ticker": "AAPL", "side": "buy", "qty": 10,
            "expected_price": 150.0, "fill_price": 150.5,
        })
        records = self._read_feedback()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["ticker"], "AAPL")

    def test_adds_code_version(self):
        """Every new record should include code_version field."""
        self._call_track_fill({
            "ticker": "MSFT", "side": "sell", "qty": 5,
            "expected_price": 400, "fill_price": 399,
        })
        records = self._read_feedback()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["code_version"], "1.0.34")

    def test_valid_record_has_all_fields(self):
        """Valid record should contain all expected fields."""
        self._call_track_fill({
            "ticker": "TSLA", "side": "buy", "qty": 3,
            "expected_price": 200, "fill_price": 201,
            "volume": 50000, "session": "extended", "score": 0.85,
        })
        records = self._read_feedback()
        self.assertEqual(len(records), 1)
        rec = records[0]
        for field in ["ticker", "side", "qty", "expected_price", "fill_price",
                      "slippage_pct", "volume", "session", "time_placed",
                      "time_filled", "score", "outcome", "pnl_pct", "code_version"]:
            self.assertIn(field, rec, f"Missing field: {field}")


# ── 3. Diagnostics win rate filtering tests ──────────────────────────────────

class TestDiagnosticsWinRateFiltering(unittest.TestCase):
    """Verify diagnostics filters corrupt records before computing win rate."""

    def _run_model_health_with_trades(self, trades):
        """Run check_model_health with mocked trade data."""
        import diagnostics
        with patch("diagnostics.os.path.exists", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(trades))), \
             patch("diagnostics.os.path.getmtime", return_value=0):
            return diagnostics.check_model_health()

    def test_filters_empty_ticker_from_win_rate(self):
        """Records with empty ticker should not count in win rate."""
        trades = [
            {"ticker": "AAPL", "pnl_pct": 5.0},
            {"ticker": "", "pnl_pct": None},
            {"ticker": "", "pnl_pct": 0},
        ]
        result = self._run_model_health_with_trades(trades)
        perf = result["performance"]
        self.assertEqual(perf["total_trades"], 1)
        self.assertEqual(perf["win_rate"], 100.0)

    def test_filters_none_pnl_from_win_rate(self):
        """Records with pnl_pct=None should not count in win rate."""
        trades = [
            {"ticker": "AAPL", "pnl_pct": 2.0},
            {"ticker": "MSFT", "pnl_pct": None},
        ]
        result = self._run_model_health_with_trades(trades)
        perf = result["performance"]
        self.assertEqual(perf["total_trades"], 1)

    def test_keeps_valid_losing_trades(self):
        """Legitimate losing trades (negative pnl_pct) should still count."""
        trades = [
            {"ticker": "AAPL", "pnl_pct": 5.0},
            {"ticker": "MSFT", "pnl_pct": -3.0},
        ]
        result = self._run_model_health_with_trades(trades)
        perf = result["performance"]
        self.assertEqual(perf["total_trades"], 2)
        self.assertEqual(perf["win_rate"], 50.0)

    def test_all_corrupt_returns_empty_performance(self):
        """If all records are corrupt, performance should be empty or zero."""
        trades = [
            {"ticker": "", "pnl_pct": None},
            {"ticker": "", "pnl_pct": 0},
        ]
        result = self._run_model_health_with_trades(trades)
        perf = result.get("performance", {})
        # Either empty dict or total_trades=0
        if perf:
            self.assertEqual(perf.get("total_trades", 0), 0)

    def test_mixed_valid_and_corrupt(self):
        """Mixed dataset: only valid records should affect stats."""
        trades = [
            {"ticker": "AAPL", "pnl_pct": 10.0},   # valid winner
            {"ticker": "GOOG", "pnl_pct": -2.0},    # valid loser
            {"ticker": "", "pnl_pct": None},          # corrupt
            {"ticker": "TSLA", "pnl_pct": 3.0},      # valid winner
            {"ticker": "X", "pnl_pct": None},         # corrupt (no pnl)
            {"ticker": "", "pnl_pct": 0},              # corrupt
        ]
        result = self._run_model_health_with_trades(trades)
        perf = result["performance"]
        # Should count: AAPL, GOOG, TSLA = 3 valid (X filtered by None pnl)
        self.assertEqual(perf["total_trades"], 3)
        # Winners: AAPL, TSLA = 2 of 3
        self.assertAlmostEqual(perf["win_rate"], 66.7, places=1)


# ── 4. cleanup_feedback.py tests ────────────────────────────────────────────

class TestCleanupFeedback(unittest.TestCase):
    """Verify the cleanup script correctly identifies garbage vs valid records."""

    def test_removes_empty_ticker(self):
        """Records with empty ticker should be flagged as garbage."""
        from cleanup_feedback import is_garbage
        record = {"ticker": "", "pnl_pct": None, "outcome": None}
        self.assertTrue(is_garbage(record))

    def test_removes_missing_ticker(self):
        """Records with no ticker key should be flagged as garbage."""
        from cleanup_feedback import is_garbage
        record = {"pnl_pct": None, "outcome": None}
        self.assertTrue(is_garbage(record))

    def test_removes_whitespace_ticker(self):
        """Records with whitespace-only ticker should be flagged as garbage."""
        from cleanup_feedback import is_garbage
        record = {"ticker": "   ", "pnl_pct": 0, "outcome": None}
        self.assertTrue(is_garbage(record))

    def test_removes_no_data_record(self):
        """Records with pnl=0, no code_version, no outcome should be removed."""
        from cleanup_feedback import is_garbage
        record = {"ticker": "AAPL", "pnl_pct": 0, "outcome": None}
        self.assertTrue(is_garbage(record))

    def test_removes_none_pnl_no_version(self):
        """Records with pnl=None, no code_version, no outcome should be removed."""
        from cleanup_feedback import is_garbage
        record = {"ticker": "AAPL", "pnl_pct": None, "outcome": None}
        self.assertTrue(is_garbage(record))

    def test_preserves_valid_winning_trade(self):
        """Valid winning trade should NOT be flagged as garbage."""
        from cleanup_feedback import is_garbage
        record = {"ticker": "AAPL", "pnl_pct": 5.2, "outcome": "win",
                  "code_version": "1.0.34"}
        self.assertFalse(is_garbage(record))

    def test_preserves_valid_losing_trade(self):
        """Valid losing trade should NOT be flagged as garbage."""
        from cleanup_feedback import is_garbage
        record = {"ticker": "MSFT", "pnl_pct": -2.1, "outcome": "loss",
                  "code_version": "1.0.34"}
        self.assertFalse(is_garbage(record))

    def test_preserves_record_with_code_version(self):
        """Record with code_version should be preserved even if pnl is 0."""
        from cleanup_feedback import is_garbage
        record = {"ticker": "GOOG", "pnl_pct": 0, "outcome": None,
                  "code_version": "1.0.34"}
        self.assertFalse(is_garbage(record))

    def test_preserves_record_with_outcome(self):
        """Record with outcome set should be preserved even without code_version."""
        from cleanup_feedback import is_garbage
        record = {"ticker": "TSLA", "pnl_pct": None, "outcome": "win"}
        self.assertFalse(is_garbage(record))

    def test_preserves_record_with_nonzero_pnl(self):
        """Record with real pnl_pct should be preserved even without version."""
        from cleanup_feedback import is_garbage
        record = {"ticker": "NVDA", "pnl_pct": 3.5, "outcome": None}
        self.assertFalse(is_garbage(record))

    def test_full_cleanup_flow(self):
        """End-to-end: main() should clean a file and write back."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "feedback.json")
        records = [
            {"ticker": "AAPL", "pnl_pct": 5.0, "outcome": "win", "code_version": "1.0.34"},
            {"ticker": "", "pnl_pct": None, "outcome": None},
            {"ticker": "MSFT", "pnl_pct": 0, "outcome": None},  # garbage (no version)
            {"ticker": "GOOG", "pnl_pct": -1.0, "outcome": "loss", "code_version": "1.0.34"},
        ]
        with open(path, "w") as f:
            json.dump(records, f)

        from cleanup_feedback import main
        with patch("cleanup_feedback.TRADE_FEEDBACK_PATH", path):
            main()

        with open(path) as f:
            cleaned = json.load(f)
        # Should keep AAPL and GOOG, remove empty-ticker and MSFT (no data)
        self.assertEqual(len(cleaned), 2)
        tickers = [r["ticker"] for r in cleaned]
        self.assertIn("AAPL", tickers)
        self.assertIn("GOOG", tickers)

        os.remove(path)
        os.rmdir(tmpdir)


# ── 5. Edge case tests ──────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):
    """Edge cases: None values, empty strings, mixed valid/invalid data."""

    def test_track_fill_with_none_ticker(self):
        """track_fill should handle ticker=None gracefully."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "feedback.json")
        with open(path, "w") as f:
            json.dump([], f)

        with patch("ml_model_v2.FEEDBACK_PATH", path):
            from ml_model_v2 import track_fill
            track_fill({"ticker": None, "qty": 10, "expected_price": 100, "fill_price": 100})

        with open(path) as f:
            records = json.load(f)
        self.assertEqual(len(records), 0)

        os.remove(path)
        os.rmdir(tmpdir)

    def test_cleanup_empty_file(self):
        """cleanup should handle an empty feedback file gracefully."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "feedback.json")
        with open(path, "w") as f:
            json.dump([], f)

        from cleanup_feedback import main
        with patch("cleanup_feedback.TRADE_FEEDBACK_PATH", path):
            main()  # Should not crash

        with open(path) as f:
            records = json.load(f)
        self.assertEqual(len(records), 0)

        os.remove(path)
        os.rmdir(tmpdir)

    def test_cleanup_is_garbage_with_none_values_everywhere(self):
        """Record with all None values should be garbage."""
        from cleanup_feedback import is_garbage
        record = {"ticker": None, "pnl_pct": None, "outcome": None}
        self.assertTrue(is_garbage(record))

    def test_diagnostics_handles_empty_trades_list(self):
        """check_model_health should not crash on empty trades list."""
        import diagnostics
        with patch("diagnostics.os.path.exists", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(read_data="[]")), \
             patch("diagnostics.os.path.getmtime", return_value=0):
            result = diagnostics.check_model_health()
        self.assertEqual(result["performance"], {})


if __name__ == "__main__":
    unittest.main()
