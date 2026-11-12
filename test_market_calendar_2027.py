#!/usr/bin/env python3
"""
[REPAIR] 2026-11-06 CALENDAR YEAR-ADD: 2027 NYSE holidays and early closes.

Pins the 2027 dates against the official NYSE "Holidays & Trading Hours" table
(https://www.nyse.com/markets/hours-calendars, 2027 column, read 2026-11-06),
checks every date is a weekday (an observed holiday never lands on a weekend),
and that both readers see them: Python (market_calendar's helpers) and Node
(server/optionsChainArchive.parseMarketHolidays reads every MARKET_HOLIDAYS_<year>
/ HALF_DAYS_<year> block of this file).

Run: python3 -m pytest -q test_market_calendar_2027.py
"""
import os
import re
import sys
from datetime import date

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import market_calendar as mc

NYSE_2027 = {
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31),
    date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24),
}


def test_2027_holidays_match_the_official_table():
    assert mc.MARKET_HOLIDAYS_2027 == NYSE_2027
    assert mc.HALF_DAYS_2027 == {date(2027, 11, 26)}


def test_every_2027_closure_is_a_weekday():
    for d in NYSE_2027 | mc.HALF_DAYS_2027:
        assert d.weekday() < 5, f"{d} falls on a weekend"


def test_the_helpers_see_2027_and_2026_is_unchanged():
    assert mc.is_market_holiday(date(2027, 7, 5))
    assert not mc.is_market_holiday(date(2027, 7, 2)), "Friday before the observed Monday is a full session"
    assert mc.is_half_day(date(2027, 11, 26))
    assert mc.is_market_holiday(date(2026, 11, 26)) and mc.is_half_day(date(2026, 12, 24))
    assert mc.ALL_HOLIDAYS == mc.MARKET_HOLIDAYS_2026 | mc.MARKET_HOLIDAYS_2027
    assert mc.next_trading_day(date(2027, 7, 2)) == date(2027, 7, 6)


def test_the_node_reader_pattern_finds_the_2027_blocks():
    src = open(os.path.join(_ROOT, "market_calendar.py"), encoding="utf-8").read()
    blocks = re.findall(r"MARKET_HOLIDAYS_\d{4}\s*=\s*\{([\s\S]*?)\}", src)
    found = {tuple(map(int, m)) for b in blocks for m in re.findall(r"date\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2})\)", b)}
    assert {(d.year, d.month, d.day) for d in NYSE_2027} <= found
