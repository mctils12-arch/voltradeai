"""[RULE-REVIEW] 2026-10-27 — ORPHAN EXITS (audit §7.2.4).

Live /api/diag/ml (2026-10-27): 342 of 398 live trade-feedback records are
orphan_exit and 362 have pnl_pct None. Mechanisms found in code:

1. The Node tier dispatcher's SELL_CSP / BUY_PUT branches call
   submit_options_order() directly and never register the entry, so
   options_manager has no pending_entry_feedback to resolve; every later
   close of those positions (the bulk of the CSP sleeve) found no entry.
2. options_manager records an exit right after _submit_close_order() even
   when Alpaca REJECTED the close; the position stays open, its state is
   popped, the next cycle re-submits — one more orphan per rejected attempt.
3. The POS-KILL backstop records option exits under the OCC symbol while
   option entries are keyed by the underlying (TS side, exitFill.ts —
   pinned in server/exitFill.test.ts).

These tests reproduce 1 and 2 end to end through the real track_fill.
Historical records are not rewritten.

Run: python3 -m pytest test_orphan_exit_matching.py -q
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _occ(root: str, right: str = "P", strike_thousandths: int = 50000, days_out: int = 180) -> str:
    exp = (datetime.now() + timedelta(days=days_out)).strftime("%y%m%d")
    return f"{root}{exp}{right}{strike_thousandths:08d}"


SNAP = {"bid": 1.95, "ask": 2.05, "mid": 2.00, "delta": -0.25, "gamma": 0.02,
        "theta": -0.01, "vega": 0.03, "iv": 0.30}


class _FeedbackSandbox(unittest.TestCase):
    def setUp(self):
        from options_manager import OPTIONS_STATE_PATH
        self.state_path = OPTIONS_STATE_PATH
        if os.path.exists(self.state_path):
            os.remove(self.state_path)
        self.tmpdir = tempfile.mkdtemp()
        self.fb_path = os.path.join(self.tmpdir, "feedback.json")
        with open(self.fb_path, "w") as f:
            json.dump([], f)
        self._fb_patch = patch("ml_model_v2.FEEDBACK_PATH", self.fb_path)
        self._fb_patch.start()

    def tearDown(self):
        self._fb_patch.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        if os.path.exists(self.state_path):
            os.remove(self.state_path)

    def feedback(self):
        with open(self.fb_path) as f:
            return json.load(f)


class TestTierDispatchedCspEntryIsRecorded(_FeedbackSandbox):
    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    def test_tier_csp_entry_then_close_attributes_instead_of_orphaning(self, mock_snap, mock_get):
        from options_manager import (register_pending_entry_feedback, manage_options_positions,
                                     _record_options_exit_feedback)
        occ = _occ("XYZ")
        # What the tier dispatcher's inline Python now does right after a
        # submitted SELL_CSP (bot.ts SELL_CSP branch).
        register_pending_entry_feedback(occ, "XYZ", "sell")

        mock_snap.return_value = SNAP
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [{
            "symbol": occ, "qty": "-1", "side": "short", "avg_entry_price": "2.10",
            "current_price": "2.00", "market_value": "-200", "unrealized_pl": "10",
            "asset_class": "us_option",
        }])
        manage_options_positions(100000)  # first scan sees the filled position

        fb = self.feedback()
        self.assertEqual(len(fb), 1)
        self.assertEqual(fb[0]["ticker"], "XYZ")
        self.assertIsNone(fb[0]["outcome"], "an open entry record must exist")
        self.assertEqual(fb[0]["fill_price"], 2.10, "entry recorded at Alpaca's avg_entry_price")

        _record_options_exit_feedback("XYZ", "short", 1, 2.10, 1.00, 110.0, "profit_target",
                                      None, order_result={"status": "submitted", "order_id": "c1"})
        fb = self.feedback()
        self.assertEqual(len(fb), 1, "the exit must close the entry, not append an orphan")
        self.assertEqual(fb[0]["outcome"], "win")
        self.assertIsNotNone(fb[0]["pnl_pct"])
        self.assertEqual([r for r in fb if r.get("outcome") == "orphan_exit"], [])

    def test_unregistered_position_close_is_still_an_orphan(self):
        """The mismatch itself: with no entry record, a close can only be
        logged as orphan_exit (the pre-fix state of every tier CSP)."""
        from options_manager import _record_options_exit_feedback
        _record_options_exit_feedback("XYZ", "short", 1, 2.10, 1.00, 110.0, "profit_target",
                                      None, order_result={"status": "submitted"})
        fb = self.feedback()
        self.assertEqual(len(fb), 1)
        self.assertEqual(fb[0]["outcome"], "orphan_exit")

    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    def test_marker_only_state_initializes_like_an_unregistered_position(self, mock_snap, mock_get):
        """Exit thresholds must not change: the marker-only entry gets the same
        fresh state (entry_timestamp, initial_credit from Alpaca) the manager
        always built for these positions."""
        from options_manager import register_pending_entry_feedback, manage_options_positions, _load_options_state
        occ = _occ("XYZ")
        register_pending_entry_feedback(occ, "XYZ", "sell")
        mock_snap.return_value = SNAP
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [{
            "symbol": occ, "qty": "-1", "side": "short", "avg_entry_price": "2.10",
            "current_price": "2.00", "market_value": "-200", "unrealized_pl": "10",
            "asset_class": "us_option",
        }])
        manage_options_positions(100000)
        st = _load_options_state()[occ]
        self.assertEqual(st["entry_price"], 2.10)
        self.assertEqual(st["initial_credit"], 2.10)
        self.assertEqual(st["strategy"], "unknown")
        self.assertIn("entry_timestamp", st)
        self.assertNotIn("pending_entry_feedback", st, "resolved exactly once")

    def test_marker_never_overwrites_a_registered_entry(self):
        from options_manager import register_options_entry, register_pending_entry_feedback, _load_options_state
        occ = _occ("XYZ")
        register_options_entry(occ, 2.00, "sell", "sell_cash_secured_put", delta=-0.25, qty=1, ticker="XYZ")
        before = _load_options_state()[occ]
        register_pending_entry_feedback(occ, "XYZ", "sell")
        self.assertEqual(_load_options_state()[occ], before)


class TestRejectedCloseWritesNoExit(_FeedbackSandbox):
    def test_rejected_close_records_nothing(self):
        from options_manager import _record_options_exit_feedback
        _record_options_exit_feedback("XYZ", "short", 1, 2.10, 1.00, 110.0, "dte_critical",
                                      None, order_result={"status": "error", "detail": "Alpaca rejected: 403"})
        self.assertEqual(self.feedback(), [], "a rejected close is not an exit")

    @patch("options_manager.requests.get")
    @patch("options_manager._get_option_snapshot")
    @patch("options_manager._submit_close_order")
    def test_repeated_rejected_closes_do_not_accumulate_orphans(self, mock_close, mock_snap, mock_get):
        from options_manager import manage_options_positions
        occ = _occ("XYZ", days_out=3)  # inside DTE_CRITICAL: force close every scan
        mock_close.return_value = {"status": "error", "detail": "Alpaca rejected: insufficient qty"}
        mock_snap.return_value = SNAP
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [{
            "symbol": occ, "qty": "-1", "side": "short", "avg_entry_price": "2.10",
            "current_price": "2.00", "market_value": "-200", "unrealized_pl": "10",
            "asset_class": "us_option",
        }])
        for _ in range(3):
            manage_options_positions(100000)
        self.assertEqual(mock_close.call_count, 3)
        self.assertEqual([r for r in self.feedback() if r.get("outcome") == "orphan_exit"], [])


if __name__ == "__main__":
    unittest.main()
