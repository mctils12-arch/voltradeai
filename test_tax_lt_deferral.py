"""
[RULE-REVIEW] 2026-11-06 — hold winners past one year (tax_filter.lt_deferral;
refocus spec §6 "prefer holding winners past one year where sensible").

A winning stock/ETF position within LT_DEFER_WINDOW_DAYS of its long-term date
keeps its profit-taking exits (scale-out, take-profit, time stop) on hold while
the tax saving of waiting beats the expected give-back; stops and risk exits
are never held. Pins: the rule's math and window, the log row, and
manage_positions end to end (the hold blocks a scale-out, leaves a trailing
stop alone, and rides in the stop state bot.ts reads).
"""
import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta
from unittest.mock import patch

import bot_engine
import tax_filter as tf

TODAY = date(2026, 11, 6)


def _entry(days_held: int) -> str:
    return (TODAY - timedelta(days=days_held)).isoformat()


class TempLog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, tf.DECISIONS_FILE)

    def tearDown(self):
        self._tmp.cleanup()

    def rows(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(x) for x in f if x.strip()]

    def hold(self, gain, days_held, atr=2.0, side="long", cfg=None):
        return tf.lt_deferral(ticker="aapl", side=side, gain_pct=gain, entry_date=_entry(days_held), atr_pct=atr,
                              today=TODAY, cfg=cfg or tf.config({"LT_DEFER_ENABLED": True}), path=self.path)


class TestRule(TempLog):
    def test_long_term_means_more_than_one_year(self):
        self.assertEqual(tf.days_to_long_term(365), 1)
        self.assertEqual(tf.days_to_long_term(366), 0)
        r = self.hold(40.0, 340)
        self.assertEqual(r["daysToLongTerm"], 26)
        self.assertEqual(r["longTermDate"], (TODAY - timedelta(days=340) + timedelta(days=366)).isoformat())

    def test_a_big_winner_near_its_long_term_date_is_held(self):
        r = self.hold(40.0, 340, atr=2.0)
        self.assertTrue(r["eligible"])
        self.assertTrue(r["defer"])
        # saving 0.4/1.4 x 9 % = 2.571 %; give-back 0.3989 x (2/1.596) x sqrt(26 x 252/365) x 0.85
        self.assertAlmostEqual(r["taxSavingPct"], 2.571, places=3)
        expect = 0.3989 * (2.0 / 1.596) * ((26 * 252 / 365) ** 0.5) * 0.85
        self.assertAlmostEqual(r["giveBackPct"], round(expect, 3), places=3)
        self.assertTrue(r["reason"].startswith("Deferred: AAPL is 26 days from long-term"), r["reason"])

    def test_a_small_winner_is_not_held_when_the_give_back_is_bigger(self):
        r = self.hold(10.0, 340, atr=2.0)
        self.assertTrue(r["eligible"])
        self.assertFalse(r["defer"])
        self.assertLess(r["taxSavingPct"], r["giveBackPct"])
        self.assertTrue(r["reason"].startswith("Not deferred: "))

    def test_outside_the_window_already_long_term_losers_and_shorts_are_never_held(self):
        for gain, days, side in ((40.0, 300, "long"), (40.0, 366, "long"), (40.0, 400, "long"),
                                 (-8.0, 350, "long"), (0.0, 350, "long"), (40.0, 350, "short")):
            with self.subTest(gain=gain, days=days, side=side):
                r = self.hold(gain, days, side=side)
                self.assertFalse(r["eligible"])
                self.assertFalse(r["defer"])
        self.assertEqual(self.rows(), [], "an ineligible position logs nothing")

    def test_the_off_switch_keeps_the_verdict_but_holds_nothing(self):
        r = self.hold(40.0, 340, cfg=tf.config({"LT_DEFER_ENABLED": False}))
        self.assertFalse(r["defer"])
        self.assertIn("LT_DEFER_ENABLED off", r["reason"])
        self.assertEqual(self.rows()[-1]["wouldAllow"], False)

    def test_each_eligible_verdict_is_logged_once_a_day(self):
        self.hold(40.0, 340)
        self.hold(40.0, 340)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["kind"], row["ticker"], row["side"], row["allowed"]), ("exit_deferral", "AAPL", "sell", False))
        self.assertEqual(row["daysToLongTerm"], 26)
        self.assertIn("Deferred", row["reason"])

    def test_a_bad_date_is_no_hold_never_a_crash(self):
        r = tf.lt_deferral(ticker="X", side="long", gain_pct=40, entry_date="not-a-date", atr_pct=2.0, path=self.path)
        self.assertFalse(r["defer"])
        self.assertIn("ValueError", r["reason"])

    def test_only_profit_taking_exits_can_be_held(self):
        self.assertEqual(set(tf.LT_DEFER_EXITS), {"scale_out", "take_profit", "time_stop"})


class TestManagePositions(TempLog):
    """manage_positions end to end: a winner 340 days held, +40 %, scale-out due."""

    def setUp(self):
        super().setUp()
        self._dir = tempfile.mkdtemp(prefix="vt_lt_defer_")
        self._patches = [
            patch.object(bot_engine, "DATA_DIR", self._dir),
            patch.object(bot_engine, "_get_atr", return_value=2.0),   # ATR 2.0 on a $140 price = 1.43 %
            patch("macro_data.get_macro_snapshot", return_value={"vxx_ratio": 1.0, "spy_vs_ma50": 1.05,
                                                                 "spy_below_200_days": 0, "spy_above_200d": True}),
            patch.object(tf, "decisions_path", lambda: self.path),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        super().tearDown()

    def run_with(self, days_held, highest_price=None, pnl=0.40):
        entry = (datetime.now() - timedelta(days=days_held)).strftime("%Y-%m-%d")
        ps = {"entry_date": entry, "entry_timestamp": (datetime.now() - timedelta(days=days_held)).isoformat(),
              "initial_risk_pct": 4.0, "highest_pnl": pnl * 100, "highest_price": highest_price or 140.0,
              "original_qty": 30, "remaining_qty": 30, "scales_completed": 0}
        with open(os.path.join(self._dir, "voltrade_stop_state.json"), "w") as f:
            json.dump({"AAPL": ps}, f)
        pos = {"symbol": "AAPL", "asset_class": "us_equity", "side": "long", "qty": "30",
               "current_price": "140.0", "avg_entry_price": str(round(140.0 / (1 + pnl), 4)),
               "unrealized_plpc": str(pnl), "market_value": "4200.0"}
        with patch.object(bot_engine, "get_alpaca_positions", return_value=[pos]):
            out = bot_engine.manage_positions()
        with open(os.path.join(self._dir, "voltrade_stop_state.json")) as f:
            state = json.load(f)["AAPL"]
        return out["actions"], state

    def test_the_hold_blocks_the_scale_out_and_rides_in_the_stop_state(self):
        actions, state = self.run_with(340)
        self.assertEqual([a for a in actions if a["type"] == "scale_out"], [])
        self.assertEqual(state["scales_completed"], 0, "a held scale-out must not advance the scale state")
        self.assertEqual(state["remaining_qty"], 30)
        self.assertTrue(state["tax_defer_until"])
        self.assertTrue(state["tax_defer_reason"].startswith("Deferred: AAPL"))

    def test_without_the_hold_the_same_position_scales_out(self):
        actions, state = self.run_with(100)
        self.assertEqual(len([a for a in actions if a["type"] == "scale_out"]), 1)
        self.assertEqual(state["scales_completed"], 1)
        self.assertIsNone(state["tax_defer_until"])

    def test_a_stop_is_never_held(self):
        # peak far above: phase 4 trailing stop fires while the hold is active
        actions, state = self.run_with(340, highest_price=160.0)
        self.assertTrue(state["tax_defer_until"], "the hold is active")
        stops = [a for a in actions if a["action"] == "CLOSE"]
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0]["type"], "trailing_stop")


if __name__ == "__main__":
    unittest.main()
