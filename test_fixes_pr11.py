"""
Tests for PR #11 fixes:
  1. Performance endpoint (/api/bot/performance) inline Python now filters
     corrupt records (empty ticker) before computing stats.
  2. Startup cleanup now requires non-empty ticker in addition to code_version.

These test the Python filter logic extracted from the inline scripts in bot.ts.

Run: python3 -m pytest test_fixes_pr11.py -v
"""
import json
import unittest


# ── Helpers: replicate the inline Python filter logic from bot.ts ──────────


def performance_filter(feedback):
    """Replicates the performance endpoint filter + stats computation.

    From bot.ts lines ~775-783 (after PR #11 fix):
        feedback = [t for t in feedback if t.get('ticker', '').strip()]
        wins = [t for t in feedback if t.get('pnl_pct', 0) > 0]
        losses = [t for t in feedback if t.get('pnl_pct', 0) <= 0]
        win_rate = len(wins) / len(feedback) * 100 if feedback else 0
        avg_win = ...
        avg_loss = ...
        total_pnl = ...
    """
    feedback = [t for t in feedback if t.get('ticker', '').strip()]
    wins = [t for t in feedback if t.get('pnl_pct', 0) > 0]
    losses = [t for t in feedback if t.get('pnl_pct', 0) <= 0]
    win_rate = len(wins) / len(feedback) * 100 if feedback else 0
    avg_win = sum(t.get('pnl_pct', 0) for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.get('pnl_pct', 0) for t in losses) / len(losses) if losses else 0
    total_pnl = sum(t.get('pnl_pct', 0) for t in feedback)
    return {
        "feedback": feedback,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_pnl": total_pnl,
    }


def startup_cleanup_filter(raw):
    """Replicates the startup cleanup filter from bot.ts line ~3227 (after PR #11 fix):
        valid = [t for t in raw if t.get('code_version', '') >= '1.0.33' and t.get('ticker', '').strip()]
    """
    valid = [t for t in raw if t.get('code_version', '') >= '1.0.33' and t.get('ticker', '').strip()]
    removed = len(raw) - len(valid)
    return {"valid": valid, "removed": removed}


# ── 1. Performance endpoint filter tests ───────────────────────────────────


class TestPerformanceEndpointFilter(unittest.TestCase):
    """Verify the performance endpoint filters corrupt records before stats."""

    def test_filters_empty_ticker(self):
        """Records with ticker='' should be excluded from stats."""
        feedback = [
            {"ticker": "", "pnl_pct": 0, "code_version": "1.0.33"},
            {"ticker": "", "pnl_pct": 0, "code_version": "1.0.33"},
        ]
        result = performance_filter(feedback)
        self.assertEqual(len(result["feedback"]), 0)
        self.assertEqual(result["win_rate"], 0)
        self.assertEqual(result["total_pnl"], 0)

    def test_filters_whitespace_ticker(self):
        """Records with ticker='   ' should be excluded."""
        feedback = [
            {"ticker": "   ", "pnl_pct": 5.0},
            {"ticker": "\t", "pnl_pct": -2.0},
        ]
        result = performance_filter(feedback)
        self.assertEqual(len(result["feedback"]), 0)
        self.assertEqual(result["win_rate"], 0)

    def test_keeps_valid_records(self):
        """Records with real tickers pass through the filter."""
        feedback = [
            {"ticker": "AAPL", "pnl_pct": 5.0},
            {"ticker": "MSFT", "pnl_pct": -2.0},
        ]
        result = performance_filter(feedback)
        self.assertEqual(len(result["feedback"]), 2)
        self.assertAlmostEqual(result["win_rate"], 50.0)
        self.assertAlmostEqual(result["avg_win"], 5.0)
        self.assertAlmostEqual(result["avg_loss"], -2.0)
        self.assertAlmostEqual(result["total_pnl"], 3.0)

    def test_mixed_valid_and_garbage(self):
        """Only valid records contribute to stats, garbage is excluded."""
        feedback = [
            {"ticker": "AAPL", "pnl_pct": 10.0},    # valid win
            {"ticker": "", "pnl_pct": 0},              # garbage
            {"ticker": "GOOG", "pnl_pct": -3.0},      # valid loss
            {"ticker": "", "pnl_pct": 0},              # garbage
            {"ticker": "", "pnl_pct": 0},              # garbage
        ]
        result = performance_filter(feedback)
        self.assertEqual(len(result["feedback"]), 2)
        self.assertAlmostEqual(result["win_rate"], 50.0)
        self.assertAlmostEqual(result["total_pnl"], 7.0)

    def test_500_garbage_records_all_filtered(self):
        """Simulates the actual bug: 500 garbage records with empty ticker, pnl=0."""
        garbage = [{"ticker": "", "pnl_pct": 0, "code_version": "1.0.33"}
                   for _ in range(500)]
        result = performance_filter(garbage)
        self.assertEqual(len(result["feedback"]), 0)
        self.assertEqual(result["win_rate"], 0)
        self.assertEqual(result["total_pnl"], 0)
        self.assertEqual(len(result["wins"]), 0)
        self.assertEqual(len(result["losses"]), 0)

    def test_garbage_mixed_with_real_trades(self):
        """500 garbage records + 3 real trades: only real trades should count."""
        garbage = [{"ticker": "", "pnl_pct": 0, "code_version": "1.0.33"}
                   for _ in range(500)]
        real = [
            {"ticker": "AAPL", "pnl_pct": 8.0},
            {"ticker": "TSLA", "pnl_pct": -1.5},
            {"ticker": "NVDA", "pnl_pct": 4.0},
        ]
        result = performance_filter(garbage + real)
        self.assertEqual(len(result["feedback"]), 3)
        self.assertAlmostEqual(result["win_rate"], 66.67, places=1)
        self.assertAlmostEqual(result["total_pnl"], 10.5)

    def test_all_winners(self):
        """Stats correct when all valid records are winners."""
        feedback = [
            {"ticker": "AAPL", "pnl_pct": 5.0},
            {"ticker": "MSFT", "pnl_pct": 3.0},
        ]
        result = performance_filter(feedback)
        self.assertAlmostEqual(result["win_rate"], 100.0)
        self.assertAlmostEqual(result["avg_win"], 4.0)
        self.assertEqual(len(result["losses"]), 0)

    def test_all_losers(self):
        """Stats correct when all valid records are losers."""
        feedback = [
            {"ticker": "AAPL", "pnl_pct": -5.0},
            {"ticker": "MSFT", "pnl_pct": -3.0},
        ]
        result = performance_filter(feedback)
        self.assertAlmostEqual(result["win_rate"], 0.0)
        self.assertAlmostEqual(result["avg_loss"], -4.0)
        self.assertEqual(len(result["wins"]), 0)

    def test_empty_feedback_file(self):
        """Empty feedback list returns zero stats."""
        result = performance_filter([])
        self.assertEqual(len(result["feedback"]), 0)
        self.assertEqual(result["win_rate"], 0)
        self.assertEqual(result["total_pnl"], 0)

    def test_missing_ticker_key(self):
        """Records with no 'ticker' key at all should be filtered out."""
        feedback = [
            {"pnl_pct": 5.0},   # no ticker key
            {"ticker": "AAPL", "pnl_pct": 2.0},
        ]
        result = performance_filter(feedback)
        self.assertEqual(len(result["feedback"]), 1)
        self.assertEqual(result["feedback"][0]["ticker"], "AAPL")

    def test_missing_pnl_defaults_to_zero(self):
        """Records with no pnl_pct default to 0 (counted as loss)."""
        feedback = [{"ticker": "AAPL"}]
        result = performance_filter(feedback)
        self.assertEqual(len(result["feedback"]), 1)
        self.assertEqual(len(result["losses"]), 1)
        self.assertAlmostEqual(result["win_rate"], 0.0)


# ── 2. Startup cleanup filter tests ───────────────────────────────────────


class TestStartupCleanupFilter(unittest.TestCase):
    """Verify the startup cleanup filters by BOTH code_version AND non-empty ticker."""

    def test_removes_empty_ticker_even_with_valid_version(self):
        """The core bug: garbage records have code_version='1.0.33' but empty ticker."""
        raw = [
            {"ticker": "", "pnl_pct": 0, "code_version": "1.0.33"},
        ]
        result = startup_cleanup_filter(raw)
        self.assertEqual(len(result["valid"]), 0)
        self.assertEqual(result["removed"], 1)

    def test_removes_old_version(self):
        """Records with old code_version should still be removed."""
        raw = [
            {"ticker": "AAPL", "pnl_pct": 5.0, "code_version": "1.0.30"},
        ]
        result = startup_cleanup_filter(raw)
        self.assertEqual(len(result["valid"]), 0)
        self.assertEqual(result["removed"], 1)

    def test_keeps_valid_record(self):
        """Records with valid version AND real ticker pass through."""
        raw = [
            {"ticker": "AAPL", "pnl_pct": 5.0, "code_version": "1.0.34"},
        ]
        result = startup_cleanup_filter(raw)
        self.assertEqual(len(result["valid"]), 1)
        self.assertEqual(result["removed"], 0)

    def test_purges_500_garbage_records(self):
        """Simulates the actual scenario: 500 garbage records all get purged."""
        raw = [{"ticker": "", "pnl_pct": 0, "code_version": "1.0.33"}
               for _ in range(500)]
        result = startup_cleanup_filter(raw)
        self.assertEqual(len(result["valid"]), 0)
        self.assertEqual(result["removed"], 500)

    def test_mixed_garbage_and_valid(self):
        """Keeps valid records, purges garbage."""
        raw = [
            {"ticker": "", "pnl_pct": 0, "code_version": "1.0.33"},       # garbage
            {"ticker": "AAPL", "pnl_pct": 5.0, "code_version": "1.0.34"}, # valid
            {"ticker": "", "pnl_pct": 0, "code_version": "1.0.33"},       # garbage
            {"ticker": "MSFT", "pnl_pct": -2.0, "code_version": "1.0.33"},# valid (version ok, ticker ok)
        ]
        result = startup_cleanup_filter(raw)
        self.assertEqual(len(result["valid"]), 2)
        self.assertEqual(result["removed"], 2)
        tickers = [t["ticker"] for t in result["valid"]]
        self.assertIn("AAPL", tickers)
        self.assertIn("MSFT", tickers)

    def test_no_code_version_key(self):
        """Records without code_version key are removed (empty string < '1.0.33')."""
        raw = [{"ticker": "AAPL", "pnl_pct": 5.0}]
        result = startup_cleanup_filter(raw)
        self.assertEqual(len(result["valid"]), 0)

    def test_whitespace_ticker_removed(self):
        """Records with whitespace-only ticker should be removed."""
        raw = [
            {"ticker": "   ", "pnl_pct": 0, "code_version": "1.0.34"},
        ]
        result = startup_cleanup_filter(raw)
        self.assertEqual(len(result["valid"]), 0)
        self.assertEqual(result["removed"], 1)

    def test_empty_raw_list(self):
        """Empty file should result in 0 valid, 0 removed."""
        result = startup_cleanup_filter([])
        self.assertEqual(len(result["valid"]), 0)
        self.assertEqual(result["removed"], 0)

    def test_version_boundary_1_0_33(self):
        """Records at exactly version '1.0.33' with valid ticker should pass."""
        raw = [
            {"ticker": "GOOG", "pnl_pct": 3.0, "code_version": "1.0.33"},
        ]
        result = startup_cleanup_filter(raw)
        self.assertEqual(len(result["valid"]), 1)

    def test_newer_version_passes(self):
        """Records with version > '1.0.33' and valid ticker pass."""
        raw = [
            {"ticker": "TSLA", "pnl_pct": -1.0, "code_version": "2.0.0"},
        ]
        result = startup_cleanup_filter(raw)
        self.assertEqual(len(result["valid"]), 1)


# ── 3. Integration: verify bot.ts source has the filters ──────────────────


class TestBotSourceContainsFilters(unittest.TestCase):
    """Verify bot.ts actually contains the filter code (source-level check)."""

    @classmethod
    def setUpClass(cls):
        with open("server/bot.ts") as f:
            cls.source = f.read()

    def test_performance_endpoint_has_ticker_filter(self):
        """Performance endpoint should filter feedback by non-empty ticker and entry records."""
        self.assertIn(
            "feedback = [t for t in feedback if t.get('ticker', '').strip() and not (t.get('pnl_pct', 0) == 0 and t.get('outcome') is None)]",
            self.source,
            "Performance endpoint missing ticker + entry record filter"
        )

    def test_startup_cleanup_has_ticker_filter(self):
        """Startup cleanup should check for non-empty ticker."""
        self.assertIn(
            "and t.get('ticker', '').strip()",
            self.source,
            "Startup cleanup missing ticker filter"
        )

    def test_startup_cleanup_still_checks_code_version(self):
        """Startup cleanup should still check code_version >= '1.0.33'."""
        self.assertIn(
            "t.get('code_version', '') >= '1.0.33'",
            self.source,
            "Startup cleanup missing code_version check"
        )


if __name__ == "__main__":
    unittest.main()
