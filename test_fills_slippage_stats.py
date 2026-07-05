"""
Regression test for the FILLS_PATH measurement bug (2026-07-05, [RULE-REVIEW]).

BUG: server/bot.ts's /api/bot/performance, /api/bot/ml-status, and
/api/diag/ml all computed "realistic P&L net of slippage" / fill counts by
reading storage_config.FILLS_PATH (voltrade_fills.json). Nothing has
written that file since the legacy ml_model.py.track_fill() was retired —
ml_model_v2.track_fill() (the live one, called by bot.ts on every order
fill) has written expected_price/fill_price/slippage_pct straight into the
trade_feedback store since v1.0.34. Reading FILLS_PATH therefore always
returns an empty list, silently zeroing avgSlippagePct/totalSlippageCost/
slippageGapPct/totalFills on every live deploy regardless of real trading
activity — exactly the HONESTY METRIC risk CLAUDE.md's MEASUREMENT
INTEGRITY section warns about (a performance dashboard reporting "0%
slippage gap" when real slippage data existed all along, just in a
different file).

fills_slippage_stats() (ml_model_v2.py) is the fix: it derives the same
stats directly from trade_feedback records, which is where track_fill
actually writes them.

Run: python3 -m pytest test_fills_slippage_stats.py -v
"""
import unittest

from ml_model_v2 import fills_slippage_stats


class TestFillsSlippageStats(unittest.TestCase):
    def test_empty_feedback_returns_zeroed_stats(self):
        stats = fills_slippage_stats([])
        self.assertEqual(stats, {"count": 0, "avg_slippage_pct": 0.0, "total_slippage_cost": 0.0})

    def test_none_feedback_returns_zeroed_stats(self):
        stats = fills_slippage_stats(None)
        self.assertEqual(stats["count"], 0)

    def test_real_entry_fill_records_are_counted_and_aggregated(self):
        """This is the case the old FILLS_PATH read got permanently wrong:
        real entry-fill records exist (written by track_fill), but the old
        code looked in a file nothing writes to and always saw zero."""
        feedback = [
            {
                "ticker": "AAPL", "side": "buy", "qty": 10,
                "expected_price": 200.0, "fill_price": 200.10,
                "slippage_pct": 0.05, "outcome": "win", "pnl_pct": 3.2,
            },
            {
                "ticker": "MSFT", "side": "buy", "qty": 5,
                "expected_price": 400.0, "fill_price": 400.80,
                "slippage_pct": 0.20, "outcome": "loss", "pnl_pct": -1.1,
            },
        ]
        stats = fills_slippage_stats(feedback)
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["avg_slippage_pct"], (0.05 + 0.20) / 2, places=6)

    def test_avg_slippage_and_cost_match_hand_computation(self):
        feedback = [
            {"ticker": "AAPL", "qty": 10, "expected_price": 200.0, "slippage_pct": 0.05},
            {"ticker": "MSFT", "qty": 5, "expected_price": 400.0, "slippage_pct": 0.20},
        ]
        stats = fills_slippage_stats(feedback)
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["avg_slippage_pct"], (0.05 + 0.20) / 2, places=6)
        expected_cost = (200.0 * 10 * 0.05 / 100) + (400.0 * 5 * 0.20 / 100)
        self.assertAlmostEqual(stats["total_slippage_cost"], expected_cost, places=6)

    def test_orphan_exit_records_excluded(self):
        """Exit fills with no matching entry (ml_model_v2._find_entry_record
        miss) are appended WITHOUT expected_price/slippage_pct — they must
        not be miscounted as fills."""
        feedback = [
            {"ticker": "TSLA", "side": "sell", "qty": 3, "fill_price": 250.0,
             "outcome": "orphan_exit", "pnl_pct": None},
        ]
        stats = fills_slippage_stats(feedback)
        self.assertEqual(stats["count"], 0)

    def test_seeded_backtest_records_without_slippage_excluded(self):
        """Seed records (seed_feedback_from_backtest.py) carry pnl_pct but
        no real fill slippage data — must not inflate the fill count."""
        feedback = [
            {"ticker": "SPY", "_seed": True, "pnl_pct": 1.0, "outcome": "win"},
        ]
        stats = fills_slippage_stats(feedback)
        self.assertEqual(stats["count"], 0)

    def test_mixed_real_and_non_fill_records(self):
        feedback = [
            {"ticker": "AAPL", "qty": 10, "expected_price": 200.0, "slippage_pct": 0.05},
            {"ticker": "TSLA", "side": "sell", "qty": 3, "fill_price": 250.0,
             "outcome": "orphan_exit", "pnl_pct": None},
            {"ticker": "SPY", "_seed": True, "pnl_pct": 1.0},
        ]
        stats = fills_slippage_stats(feedback)
        self.assertEqual(stats["count"], 1)


if __name__ == "__main__":
    unittest.main()
