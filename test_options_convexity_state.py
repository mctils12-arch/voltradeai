"""Convexity-marker preservation ratchet (repair 2026-08-06, full-code-review
finding, adversarially verified): manage_options_positions' empty-branch
cleanup used to wipe the WHOLE options state file whenever no manager-owned
options were held — including the convexity overlay's strategy/managed_by
markers, even though the hedge puts were still open (they are filtered OUT
of options_positions by design, which is exactly what made the branch fire).
The QQQ tail-hedge legs were thereby orphaned from _run_convexity_overlay
every quiet cycle. The repair preserves convexity-managed entries and drops
only truly stale state."""
import json
import unittest
from unittest import mock

import options_manager


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._p = payload
    def json(self): return self._p


CONVEXITY_STATE = {
    "QQQ260918P00400000": {
        "strategy": "convexity_hedge", "managed_by": "convexity_overlay",
        "entry_price": 3.10, "qty": 1,
    },
    "STALEAAPL260101C00100000": {"strategy": "csp", "entry_price": 1.0},
}


class TestConvexityMarkersSurviveQuietCycles(unittest.TestCase):
    def _run_cycle(self, tmp, held_positions):
        state_path = f"{tmp}/voltrade_options_state.json"
        with open(state_path, "w") as f:
            json.dump(CONVEXITY_STATE, f)
        with mock.patch.object(options_manager, "DATA_DIR", tmp), \
             mock.patch.object(options_manager, "OPTIONS_STATE_PATH", state_path), \
             mock.patch.object(options_manager.requests, "get",
                               return_value=_Resp(held_positions)):
            result = options_manager.manage_options_positions(equity=100_000)
        with open(state_path) as f:
            return result, json.load(f)

    def test_hedge_marker_survives_when_only_hedge_puts_are_held(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            held = [{"symbol": "QQQ260918P00400000", "qty": "-1", "asset_class": "option"}]
            result, saved = self._run_cycle(tmp, held)
            self.assertIn("QQQ260918P00400000", saved,
                          "the convexity marker must survive the quiet-cycle cleanup")
            self.assertEqual(saved["QQQ260918P00400000"]["strategy"], "convexity_hedge")
            self.assertNotIn("STALEAAPL260101C00100000", saved,
                             "truly stale entries are still cleaned")

    def test_no_positions_at_all_still_cleans_stale_but_keeps_hedge_markers(self):
        # even with zero live positions the marker file entry is the overlay's
        # record; the overlay owns its lifecycle, the manager must not reap it
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result, saved = self._run_cycle(tmp, [])
            self.assertIn("QQQ260918P00400000", saved)
            self.assertNotIn("STALEAAPL260101C00100000", saved)


if __name__ == "__main__":
    unittest.main()
