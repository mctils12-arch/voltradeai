"""[RULE-REVIEW] 2026-10-27 — REAL BROKER FILLS (audit §7.1 / §7.2.5).

Before: bot.ts sent fill_price = a random volume-tiered haircut (regular
path: on top of Alpaca's filled_avg_price; morning queue: on top of the
queued quote, no fill read at all). track_fill stored only that number, so
slippage_pct, fills_slippage_stats and /api/bot/performance realisticPnlPct
measured the bot's own random draw.

After: fill_price is the broker's filled_avg_price; the old haircut travels
as synthetic_fill_price and is stored beside it (with the old ruler's
synthetic_slippage_pct), and fills_slippage_stats averages real fills only
once any exist.

Run: python3 -m pytest test_real_broker_fills.py -q
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ml_model_v2 import fills_slippage_basis, fills_slippage_stats


class TestTrackFillStoresRealAndSyntheticFill(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "feedback.json")
        with open(self.path, "w") as f:
            json.dump([], f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _track(self, payload):
        with patch("ml_model_v2.FEEDBACK_PATH", self.path):
            from ml_model_v2 import track_fill
            track_fill(payload)
        with open(self.path) as f:
            return json.load(f)

    def test_entry_record_keeps_broker_fill_and_synthetic_side_by_side(self):
        # Payload shape produced by server/entryFill.ts buildEntryFillPayload
        # for a regular-hours entry: decision quote 100.00, broker fill 100.02,
        # prior haircut 0.10% on the fill -> 100.12.
        rec = self._track({
            "ticker": "abc", "side": "buy", "qty": 10, "qty_requested": 12,
            "partial_fill": True, "expected_price": 100.0, "fill_price": 100.02,
            "fill_source": "broker", "synthetic_fill_price": 100.12,
            "synthetic_slippage_applied_pct": 0.10, "order_status": "filled",
            "order_id": "oid-1", "session": "regular", "volume": 3_000_000,
            "score": 70, "code_version": "1.0.1169",
        })[-1]
        self.assertEqual(rec["fill_price"], 100.02)
        self.assertEqual(rec["fill_source"], "broker")
        # slippage_pct now measures the real fill against the quote ...
        self.assertAlmostEqual(rec["slippage_pct"], 0.02, places=4)
        # ... and the old ruler's number for the same trade is still there.
        self.assertEqual(rec["synthetic_fill_price"], 100.12)
        self.assertAlmostEqual(rec["synthetic_slippage_pct"], 0.12, places=4)
        self.assertEqual(rec["qty"], 10)
        self.assertEqual(rec["qty_requested"], 12)
        self.assertTrue(rec["partial_fill"])
        self.assertEqual(rec["order_status"], "filled")
        self.assertEqual(rec["order_id"], "oid-1")
        self.assertIsNone(rec["outcome"])

    def test_legacy_payload_without_new_fields_is_unchanged(self):
        rec = self._track({
            "ticker": "XYZ", "side": "buy", "qty": 5,
            "expected_price": 50.0, "fill_price": 50.05, "code_version": "1.0.900",
        })[-1]
        self.assertNotIn("fill_source", rec)
        self.assertNotIn("synthetic_fill_price", rec)
        self.assertAlmostEqual(rec["slippage_pct"], 0.1, places=4)


class TestFillsSlippageStatsPrefersBrokerFills(unittest.TestCase):
    LEGACY = {"ticker": "OLD", "qty": 10, "expected_price": 100.0, "slippage_pct": 0.20}
    BROKER = {"ticker": "NEW", "qty": 10, "expected_price": 100.0, "slippage_pct": 0.02,
              "fill_source": "broker", "synthetic_slippage_pct": 0.12}

    def test_average_uses_only_real_fills_once_any_exist(self):
        stats = fills_slippage_stats([self.LEGACY, self.BROKER])
        # count stays "entry-fill records" (what /api/diag/ml fills_count means)
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["avg_slippage_pct"], 0.02, places=6)
        self.assertAlmostEqual(stats["total_slippage_cost"], 100.0 * 10 * 0.02 / 100, places=6)

    def test_legacy_only_feedback_keeps_old_numbers(self):
        stats = fills_slippage_stats([self.LEGACY])
        self.assertAlmostEqual(stats["avg_slippage_pct"], 0.20, places=6)

    def test_basis_labels_the_population(self):
        self.assertEqual(fills_slippage_basis([]), "none")
        self.assertEqual(fills_slippage_basis([self.LEGACY]), "legacy_synthetic")
        self.assertEqual(fills_slippage_basis([self.LEGACY, self.BROKER]), "broker")


if __name__ == "__main__":
    unittest.main()
