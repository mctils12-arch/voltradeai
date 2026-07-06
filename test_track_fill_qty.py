"""
REPAIR 2026-07-06 (R12) — regression battery for the track_fill qty guard
that silently killed the live ML feedback pipeline for 2.5 months.

THE INCIDENT: bot.ts's regular-hours fill payload switched to
qty_requested/qty_filled on 2026-04-23 (commit 2479df0) and stopped sending
"qty". track_fill's `if qty <= 0: return` guard then dropped every
regular-hours entry fill with zero trace. Diagnosed live via /api/diag/ml
shape aggregates (v1.0.152): all 500 feedback records are pre-2026-04-20
trackClosedTrades fossils (code_version 1.0.33, pnl_pct 0), fills_count 0,
newest record 2026-04-20 — nothing had appended since the payload change.

These tests pin the payload CONTRACT (fixtures copy the exact key sets
bot.ts sends today — the R6 lesson: fixtures must match the real contract
shape, not a convenient one).

Run: python3 -m pytest test_track_fill_qty.py -q
"""
import json
import os
import tempfile
import unittest

import ml_model_v2


def _regular_hours_payload(**overrides):
    """The EXACT regular-hours fill payload shape from server/bot.ts
    (batch fill confirmation, post-2026-04-23): qty_requested/qty_filled,
    no 'qty' key."""
    p = {
        "ticker": "MSFT", "order_type": "market", "side": "buy",
        "qty_requested": 12,
        "qty_filled": 12,
        "partial_fill": False,
        "expected_price": 100.0,
        "alpaca_fill_price": 100.02,
        "fill_price": 100.05,
        "slippage_applied_pct": 0.05,
        "order_status": "filled",
        "time_placed": "2026-07-06T17:00:00Z",
        "session": "regular", "volume": 2_000_000, "score": 71,
        "instrument": "stock", "entry_features": {"rsi": 55.0},
    }
    p.update(overrides)
    return p


def _morning_queue_payload(**overrides):
    """The morning-queue payload shape from server/bot.ts (has 'qty')."""
    p = {
        "ticker": "NVDA", "order_type": "market", "side": "buy",
        "qty": 5,
        "expected_price": 200.0, "fill_price": 200.30,
        "slippage_applied_pct": 0.15,
        "time_placed": "2026-07-06T13:30:00Z",
        "session": "morning_queue", "volume": 30_000_000, "score": 80,
        "instrument": "stock",
    }
    p.update(overrides)
    return p


class TestTrackFillQtyContract(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="tfq_")
        self._orig_path = ml_model_v2.FEEDBACK_PATH
        ml_model_v2.FEEDBACK_PATH = os.path.join(self._tmpdir, "fb.json")

    def tearDown(self):
        ml_model_v2.FEEDBACK_PATH = self._orig_path

    def _records(self):
        if not os.path.exists(ml_model_v2.FEEDBACK_PATH):
            return []
        with open(ml_model_v2.FEEDBACK_PATH) as f:
            return json.load(f)

    def test_regular_hours_payload_writes_a_record(self):
        """THE regression: the exact post-2026-04-23 bot.ts payload (no
        'qty' key) must produce a feedback record."""
        ml_model_v2.track_fill(_regular_hours_payload())
        recs = self._records()
        self.assertEqual(len(recs), 1,
            "regular-hours fill payload was silently dropped — the exact "
            "2.5-month feedback blackout this test exists to prevent")
        r = recs[0]
        self.assertEqual(r["ticker"], "MSFT")
        self.assertEqual(r["qty"], 12.0)
        self.assertEqual(r["expected_price"], 100.0)
        self.assertIsNotNone(r["slippage_pct"])
        self.assertIsNone(r["outcome"], "entry records open with outcome=None")

    def test_partial_fill_uses_filled_quantity(self):
        """qty_filled (what actually executed) wins over qty_requested."""
        ml_model_v2.track_fill(_regular_hours_payload(
            qty_requested=12, qty_filled=7, partial_fill=True))
        self.assertEqual(self._records()[0]["qty"], 7.0)

    def test_zero_filled_falls_back_to_requested(self):
        """The FILL-CHECK-WARN path records expected values when Alpaca
        confirmation fails — qty_filled may be 0 with a real request."""
        ml_model_v2.track_fill(_regular_hours_payload(qty_filled=0))
        self.assertEqual(self._records()[0]["qty"], 12.0)

    def test_morning_queue_payload_still_writes(self):
        """The pre-existing 'qty' contract keeps working unchanged."""
        ml_model_v2.track_fill(_morning_queue_payload())
        recs = self._records()
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["qty"], 5.0)

    def test_no_quantity_at_all_drops_without_writing(self):
        """A genuinely quantity-less payload still refuses to write a
        garbage record (and the guard now logs the drop instead of being
        silent — asserted via the logger)."""
        payload = _regular_hours_payload()
        del payload["qty_requested"], payload["qty_filled"]
        with self.assertLogs("voltrade.ml", level="WARNING") as cm:
            ml_model_v2.track_fill(payload)
        self.assertEqual(self._records(), [])
        self.assertTrue(any("track_fill DROP" in m for m in cm.output))

    def test_exit_shaped_payload_closes_the_entry(self):
        """End-to-end: entry via the regular payload, then an exit payload
        carrying exit_context updates the SAME record (Bug #13 machinery,
        exercised with today's entry contract)."""
        ml_model_v2.track_fill(_regular_hours_payload())
        ml_model_v2.track_fill({
            "ticker": "MSFT", "side": "sell",
            "qty_filled": 12,
            "fill_price": 104.0, "expected_price": 104.0,
            "exit_context": {"pnl_pct": 3.95, "exit_reason": "take_profit",
                             "days_held": 2},
            "session": "regular",
        })
        recs = self._records()
        self.assertEqual(len(recs), 1, "exit must update, not append")
        self.assertEqual(recs[0]["outcome"], "win")
        self.assertAlmostEqual(recs[0]["pnl_pct"], 3.95)


if __name__ == "__main__":
    unittest.main()
