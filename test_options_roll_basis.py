"""Roll-basis ratchet (repair 2026-08-06, full-code-review finding,
adversarially verified): rolling a short option previously cloned the OLD
leg's state — initial_credit / max_profit_target / highest_value — onto the
new symbol, so the 50%-profit target and trailing logic computed against a
credit the fresh leg never received. The repair restarts economics at the
price the new leg actually sold for (surfaced by _attempt_roll as
new_credit); identity fields carry over."""
import re
import unittest

from options_manager import rolled_position_state, PROFIT_TARGET_PCT


OLD = {
    "ticker": "QQQ", "qty": 2, "side": "short", "strategy": "csp",
    "entry_date": "2026-07-01", "entry_timestamp": "2026-07-01T14:30:00",
    "entry_price": 6.40, "initial_credit": 6.40,
    "max_profit_target": 3.20, "highest_value": 6.40,
}


class TestRolledPositionState(unittest.TestCase):
    def test_fresh_leg_gets_its_own_economics(self):
        fresh = rolled_position_state(dict(OLD), 2.15, now_date="2026-08-06", now_iso="2026-08-06T15:00:00")
        self.assertEqual(fresh["initial_credit"], 2.15, "old credit basis must not survive the roll")
        self.assertEqual(fresh["entry_price"], 2.15)
        self.assertAlmostEqual(fresh["max_profit_target"], 2.15 * PROFIT_TARGET_PCT, places=6)
        self.assertEqual(fresh["highest_value"], 2.15, "trailing high restarts at the fresh credit")
        self.assertEqual(fresh["entry_date"], "2026-08-06")
        self.assertEqual(fresh["ticker"], "QQQ", "identity fields carry over")
        self.assertEqual(fresh["strategy"], "csp")

    def test_profit_target_now_reachable_for_the_new_leg(self):
        # THE consequence: old basis 6.40 -> target closes at 3.20, but a leg
        # sold for 2.15 can NEVER decay to 3.20-profit against that basis.
        fresh = rolled_position_state(dict(OLD), 2.15)
        current_price = 1.05  # new leg has decayed >50%
        profit_pct = (fresh["initial_credit"] - current_price) / fresh["initial_credit"]
        self.assertGreater(profit_pct, 0.50, "50% target must be computable against the FRESH basis")
        stale_profit_pct = (OLD["initial_credit"] - current_price) / OLD["initial_credit"]
        self.assertNotAlmostEqual(profit_pct, stale_profit_pct, places=2)

    def test_unknown_credit_keeps_old_economics_rather_than_inventing_zero(self):
        fresh = rolled_position_state(dict(OLD), None)
        self.assertEqual(fresh["initial_credit"], 6.40, "no fabricated basis on a missing fill price")
        fresh0 = rolled_position_state(dict(OLD), 0)
        self.assertEqual(fresh0["initial_credit"], 6.40)


class TestRollSitesWired(unittest.TestCase):
    def setUp(self):
        with open("options_manager.py", encoding="utf-8") as f:
            self.src = f.read()

    def test_both_roll_sites_use_the_fresh_state_builder(self):
        n = self.src.count('rolled_position_state(pos_state, roll_result.get("new_credit"))')
        self.assertEqual(n, 2, "both roll paths (assignment + 21-DTE) must build fresh state")

    def test_old_inline_clone_is_gone(self):
        self.assertNotIn('state[new_sym] = {**pos_state, "entry_date"', self.src,
                         "the stale-economics clone pattern must not return")

    def test_attempt_roll_surfaces_the_new_leg_price(self):
        self.assertIn('"new_credit": new_limit', self.src)


if __name__ == "__main__":
    unittest.main()
