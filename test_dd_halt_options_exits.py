"""HALT SCOPE (2026-10-27, audit N9) — the Python portfolio DD halt gates
entries only.

bot_engine._scan_market_inner's DD-halt early return used to skip the options
manager as well, although its own comment says "only entries are gated, not
exits": an 18% drawdown froze every options exit. Now the halted return runs
manage_options_positions(allow_rolls=False) first and hands its result to
bot.ts as `options_management` (audited by server/optionsMgmtAudit.ts).
Mocked: no broker or data calls.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bot_engine  # noqa: E402

HALTED = {
    "halted": True, "halt_reason": "DD 19.0% >= 18%", "peak_equity": 100000.0,
    "current_equity": 81000.0, "dd_pct": 19.0,
}


def _no_macro():
    raise RuntimeError("no macro in tests")


class TestDdHaltRunsOptionsExits(unittest.TestCase):
    def setUp(self):
        os.environ.pop("VOLTRADE_ULTRA_SURVIVAL_MODE", None)

    @patch("macro_data.get_macro_snapshot", side_effect=_no_macro)
    @patch("bot_engine.update_equity_peak", return_value=dict(HALTED))
    @patch("bot_engine.get_alpaca_account", return_value={"equity": "81000"})
    @patch("bot_engine._get_full_universe", side_effect=AssertionError("halted scan must not reach the universe"))
    @patch("options_manager.manage_options_positions")
    def test_halted_scan_runs_options_exits_without_rolls(self, mock_mgr, _univ, _acct, _peak, _macro):
        mgmt = {"actions": [{"action": "CLOSE", "ticker": "XYZ", "type": "loss_limit"}], "positions_checked": 1}
        mock_mgr.return_value = mgmt
        res = bot_engine._scan_market_inner()
        self.assertTrue(res["halted"])
        self.assertEqual(res["new_trades"], [])
        self.assertEqual(res["trades"], [])
        mock_mgr.assert_called_once_with(allow_rolls=False)
        self.assertEqual(res["options_management"], mgmt)

    @patch("macro_data.get_macro_snapshot", side_effect=_no_macro)
    @patch("bot_engine.update_equity_peak", return_value=dict(HALTED))
    @patch("bot_engine.get_alpaca_account", return_value={"equity": "81000"})
    @patch("bot_engine._get_full_universe", side_effect=AssertionError("halted scan must not reach the universe"))
    @patch("options_manager.manage_options_positions", side_effect=RuntimeError("boom"))
    def test_options_manager_failure_still_returns_the_halt(self, _mgr, _univ, _acct, _peak, _macro):
        res = bot_engine._scan_market_inner()
        self.assertTrue(res["halted"], "a manager crash must not fall through into a full (entry) scan")
        self.assertEqual(res["new_trades"], [])
        self.assertIn("boom", res["options_management"]["error"])

    def test_bot_ts_audits_options_management_before_the_halt_line(self):
        """bot.ts must read options_management on a halted result (it does so
        before the DD-HALT audit, with no early return in between)."""
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", "bot.ts")).read()
        a = src.index("for (const row of formatOptionsMgmt(result.options_management)) audit(row.type, row.message);")
        b = src.index("if (result.halted) {", a)
        self.assertNotIn("return;", src[a:b])


if __name__ == "__main__":
    unittest.main()
