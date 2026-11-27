"""The suite's "today", as market_calendar sees it, is one fixed ordinary
trading day (conftest._hermetic_market_calendar).

WHY (2026-11-27). test_mleg_resting_visible.py's
test_get_options_trades_skips_a_listed_underlying called
options_scanner.get_options_trades() without naming the day. That function
returns [] on a half day and before a long weekend, by market_calendar's
date.today(). On 2026-11-27, the half day after Thanksgiving by the CI runner's
UTC date, the test failed and every pull request's required suite was red. The
same fault had been repaired test by test in test_options_v134_fixes.py; this
is the repair of the class.
"""
import os
import sys
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conftest  # noqa: E402
import market_calendar as mc  # noqa: E402

PINNED = date(*conftest.MARKET_CALENDAR_TEST_TODAY)
HALF_DAY = date(2026, 11, 27)        # the day after Thanksgiving
PRE_LONG_WEEKEND = date(2026, 4, 2)  # the Thursday before Good Friday

OPP = {"ticker": "IREN", "setup": "high_iv_premium_sale", "options_strategy": "iron_condor", "score": 90,
       "price": 14.0, "action_label": "SELL IC", "side": "sell"}


def _trades(osc):
    with patch.object(osc, "scan_options", return_value={"opportunities": [dict(OPP)]}), \
            patch.dict(sys.modules, {"vol_surface": MagicMock(get_surface_score=lambda t: {})}), \
            patch.object(osc, "_dealer_gamma_gate", return_value={"action": "allow"}):
        return [t["ticker"] for t in osc.get_options_trades(100000, current_tickers=[])]


class TestTheSuiteHasOneToday(unittest.TestCase):
    def test_the_default_day_is_the_pinned_day(self):
        self.assertEqual(mc.date.today(), PINNED)
        self.assertEqual(mc.next_trading_day(), mc.next_trading_day(PINNED))

    def test_the_pinned_day_is_an_ordinary_trading_day(self):
        self.assertEqual(PINNED.weekday(), 2, "a Wednesday")
        self.assertFalse(mc.is_market_holiday(PINNED))
        self.assertFalse(mc.is_half_day(PINNED))
        self.assertFalse(mc.is_pre_long_weekend(PINNED))
        self.assertFalse(mc.is_short_week(PINNED))
        self.assertEqual(mc.trading_days_this_week(PINNED), 5)
        self.assertEqual(mc.should_skip_new_options(PINNED), (False, ""))

    def test_with_no_day_named_the_calendar_answers_for_the_pinned_day(self):
        self.assertEqual(mc.should_skip_new_options(), (False, ""))
        self.assertFalse(mc.is_short_week())
        self.assertEqual(mc.trading_days_this_week(), 5)

    def test_a_day_passed_in_is_used_as_given(self):
        # the tables and the rules are the real ones: only the default day moved
        self.assertTrue(mc.is_half_day(HALF_DAY))
        self.assertTrue(mc.should_skip_new_options(HALF_DAY)[0])
        self.assertTrue(mc.should_skip_new_options(PRE_LONG_WEEKEND)[0])
        self.assertTrue(mc.is_market_holiday(date(2026, 11, 26)))


class TestTheScannerOnACalendarDay(unittest.TestCase):
    """The failure of 2026-11-27, reproduced by naming the day."""

    def _on(self, day):
        real = mc.date.__mro__[1] if getattr(mc.date, "_pinned_for_tests", False) else mc.date

        class _Day(real):
            @classmethod
            def today(cls):
                return day

        return patch.object(mc, "date", _Day)

    def test_on_the_pinned_day_the_scanner_returns_the_trade(self):
        import options_scanner as osc
        self.assertEqual(_trades(osc), ["IREN"])

    def test_on_a_half_day_the_scanner_opens_nothing(self):
        import options_scanner as osc
        with self._on(HALF_DAY):
            self.assertEqual(_trades(osc), [])

    def test_before_a_long_weekend_the_scanner_opens_nothing(self):
        import options_scanner as osc
        with self._on(PRE_LONG_WEEKEND):
            self.assertEqual(_trades(osc), [])

    def test_a_test_that_names_its_own_day_is_not_overridden(self):
        with self._on(HALF_DAY):
            self.assertEqual(mc.date.today(), HALF_DAY)
            self.assertTrue(mc.should_skip_new_options()[0])
        self.assertEqual(mc.date.today(), PINNED)


if __name__ == "__main__":
    unittest.main()
