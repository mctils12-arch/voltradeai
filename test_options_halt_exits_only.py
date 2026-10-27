"""HALT SCOPE (2026-10-27, audit N15/N9) — options_manager exits-only mode.

While a halt blocks entries, the options manager's protective exits must keep
running, but a ROLL is close + sell-to-open: its open half is a new entry. So
manage_options_positions(allow_rolls=False) takes each roll point's existing
"roll failed -> close" branch instead. Mocked: no network, no state file IO.
"""
import inspect
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import options_manager  # noqa: E402


def _short_put_position(occ: str, qty: int, entry: float, current: float) -> dict:
    return {
        "symbol": occ,
        "qty": str(-qty),
        "side": "short",
        "avg_entry_price": str(entry),
        "current_price": str(current),
        "market_value": str(-current * qty * 100),
        "unrealized_pl": str((entry - current) * qty * 100),
        "asset_class": "option",
    }


class TestRollOrSkip(unittest.TestCase):
    def test_skip_never_calls_attempt_roll(self):
        with patch("options_manager._attempt_roll", side_effect=AssertionError("must not roll")):
            r = options_manager._roll_or_skip(False, "X", 1, "sell", {}, 100000)
        self.assertEqual(r, {"rolled": False, "detail": options_manager.ROLL_SKIPPED_HALTED_DETAIL})

    def test_allowed_delegates_to_attempt_roll(self):
        with patch("options_manager._attempt_roll", return_value={"rolled": True, "detail": "ok"}) as m:
            r = options_manager._roll_or_skip(True, "X", 2, "sell", {"ticker": "X"}, 5.0)
        m.assert_called_once_with("X", 2, "sell", {"ticker": "X"}, 5.0)
        self.assertTrue(r["rolled"])

    def test_every_roll_point_goes_through_the_gate(self):
        src = inspect.getsource(options_manager.manage_options_positions)
        self.assertNotIn("_attempt_roll(", src, "a direct roll call would bypass allow_rolls")
        self.assertGreaterEqual(src.count("_roll_or_skip(allow_rolls,"), 2)
        sig = inspect.signature(options_manager.manage_options_positions)
        self.assertIs(sig.parameters["allow_rolls"].default, True, "normal scans keep rolling")


class TestDaemonRoute(unittest.TestCase):
    def test_manage_options_exits_route_binds_allow_rolls_as_kwarg(self):
        from voltrade_daemon import RPCDispatcher
        module_name, attr_name = RPCDispatcher()._routes["manage_options_exits"]
        self.assertEqual((module_name, attr_name), ("options_manager", "manage_options_positions"))
        fn = getattr(__import__(module_name), attr_name)
        # the dispatcher calls fn(**args) when the args bind to the signature
        bound = inspect.signature(fn).bind(**{"allow_rolls": False})
        self.assertIs(bound.arguments["allow_rolls"], False)


@patch("ml_model_v2.track_fill")
@patch("options_manager._save_options_state")
@patch("options_manager._load_options_state")
@patch("options_manager.requests.get")
@patch("options_manager._get_option_snapshot")
@patch("options_manager._submit_close_order")
@patch("options_manager._attempt_roll")
class TestManageOptionsExitsOnly(unittest.TestCase):
    """A short put at 15 DTE hits Check 3 (21 DTE management): rolls normally,
    closes instead when allow_rolls=False."""

    def _run(self, allow_rolls, mock_roll, mock_close, mock_snap, mock_get, mock_load, mock_save, _track):
        exp = (datetime.now() + timedelta(days=15)).strftime("%y%m%d")
        occ = f"HALTQ{exp}P00050000"
        mock_get.return_value = MagicMock(status_code=200, json=lambda: [_short_put_position(occ, 1, 2.00, 1.80)])
        mock_snap.return_value = {"bid": 1.75, "ask": 1.85, "mid": 1.80, "delta": -0.30,
                                  "gamma": 0.02, "theta": -0.05, "vega": 0.05, "iv": 0.30}
        mock_close.return_value = {"status": "submitted", "order_id": "t1"}
        mock_roll.return_value = {"rolled": True, "detail": "rolled", "new_symbol": None, "new_credit": 1.0}
        mock_load.return_value = {occ: {
            "entry_price": 2.00, "entry_delta": -0.30, "side": "short", "entry_date": "2026-09-01",
            "entry_timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
            "strategy": "sell_cash_secured_put", "highest_value": 2.00, "qty": 1,
        }}
        return occ, options_manager.manage_options_positions(100000, allow_rolls=allow_rolls)

    def test_default_rolls(self, mock_roll, mock_close, mock_snap, mock_get, mock_load, mock_save, _track):
        _, result = self._run(True, mock_roll, mock_close, mock_snap, mock_get, mock_load, mock_save, _track)
        mock_roll.assert_called_once()
        self.assertEqual([a["action"] for a in result["actions"]], ["ROLL"])

    def test_exits_only_closes_instead_of_rolling(self, mock_roll, mock_close, mock_snap, mock_get, mock_load, mock_save, _track):
        occ, result = self._run(False, mock_roll, mock_close, mock_snap, mock_get, mock_load, mock_save, _track)
        mock_roll.assert_not_called()
        self.assertEqual([a["action"] for a in result["actions"]], ["CLOSE"])
        self.assertEqual(result["actions"][0]["type"], "dte_close")
        self.assertIn(options_manager.ROLL_SKIPPED_HALTED_DETAIL, result["actions"][0]["reason"])
        mock_close.assert_called_once()
        self.assertEqual(mock_close.call_args[0][0], occ)
        self.assertEqual(mock_close.call_args[0][2], "buy", "buy-to-close the short put")


if __name__ == "__main__":
    unittest.main()
