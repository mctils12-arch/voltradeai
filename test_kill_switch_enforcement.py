"""Kill-switch ENFORCEMENT ratchet (repair 2026-08-06, full-code-review
finding, adversarially verified before fixing).

The defect: check_kill_switches() gated ONLY the tier engine — the
scanner-built candidate trades were returned in new_trades unconditionally,
and the Node side executes new_trades before it ever reads kill_status
(which it only audits). A halt that does not halt new entries is a
mechanism bypass. The repair suppresses scanner ENTRY trades when the kill
switch has fired, while management/exit actions stay untouched (closing
risk during a halt is required behavior — liquidation depends on it).

These tests pin (1) the pure suppression contract and (2) that scan_market's
killed branch actually calls it — a source-level ratchet in the repo's
established style (test_alpaca_feed precedent), so a refactor cannot
silently disconnect the enforcement again.
"""
import re
import unittest

from bot_engine import suppress_entries_on_kill


class TestSuppressEntriesOnKill(unittest.TestCase):
    TRADES = [{"ticker": "AAPL", "qty": 10}, {"ticker": "MSFT", "qty": 5}]

    def test_fired_kill_switch_suppresses_all_entries(self):
        trades, suppressed = suppress_entries_on_kill(
            list(self.TRADES), {"killed": True, "kill_reason": "drawdown halt"}
        )
        self.assertEqual(trades, [], "a fired kill switch must return ZERO entry trades")
        self.assertEqual(suppressed, ["AAPL", "MSFT"], "suppressed tickers surfaced for the audit log")

    def test_not_fired_passes_trades_through_unchanged(self):
        trades, suppressed = suppress_entries_on_kill(list(self.TRADES), {"killed": False})
        self.assertEqual(trades, self.TRADES)
        self.assertEqual(suppressed, [])

    def test_no_kill_status_passes_through(self):
        # kill_status None (tier machinery unavailable) must not invent a halt
        trades, suppressed = suppress_entries_on_kill(list(self.TRADES), None)
        self.assertEqual(trades, self.TRADES)
        self.assertEqual(suppressed, [])

    def test_fired_with_no_trades_is_a_clean_noop(self):
        trades, suppressed = suppress_entries_on_kill([], {"killed": True})
        self.assertEqual(trades, [])
        self.assertEqual(suppressed, [])

    def test_malformed_trade_rows_never_crash_the_suppression(self):
        trades, suppressed = suppress_entries_on_kill(
            [{}, {"ticker": "X"}], {"killed": True, "kill_reason": "halt"}
        )
        self.assertEqual(trades, [])
        self.assertEqual(suppressed, ["?", "X"])


class TestKilledBranchIsWired(unittest.TestCase):
    """Source ratchet: scan_market's kill-fired branch must route trades
    through suppress_entries_on_kill, and the scan result must surface
    kill_suppressed_trades so the Node audit can see what a halt blocked."""

    def setUp(self):
        with open("bot_engine.py", encoding="utf-8") as f:
            self.src = f.read()

    def test_killed_branch_calls_the_suppressor(self):
        # the [TIERS] BLOCKED branch and the suppression call must coexist
        # within the same ~30-line window — a refactor that drops either
        # (or moves suppression out of the killed path) fails here.
        m = re.search(r"\[TIERS\] BLOCKED.{0,2000}?suppress_entries_on_kill\(trades, kill_status\)", self.src, re.S)
        self.assertIsNotNone(m, "the kill-fired branch no longer suppresses scanner entry trades")

    def test_scan_result_surfaces_suppressed_tickers(self):
        self.assertIn('"kill_suppressed_trades": kill_suppressed', self.src,
                      "scan result must carry what the halt suppressed — silent suppression is undiagnosable")


if __name__ == "__main__":
    unittest.main()
