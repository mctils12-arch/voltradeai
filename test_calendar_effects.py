#!/usr/bin/env python3
"""
calendar_effects.py (2026-11-06, refocus spec step 2): known 2026/2027 dates.

Pins:
- the NYSE holiday rules reproduce market_calendar.py's official 2026 list
  exactly, and give the published-pattern 2027 dates (Juneteenth and Christmas
  2027 fall on Saturdays: observed Fridays June 18 and December 24);
- index events: S&P 500 quarterly rebalances (third Fridays; June moves to
  Thursday when the Friday is Juneteenth), Russell (June 26 and December 11,
  2026 as announced; 2027 estimated), Nasdaq-100 (third Friday of December);
- the Dash for Cash windows move with the settlement cycle (T+3, T+2, T+1);
- the rebalancing window, tax and earnings windows, and the calendar list.

Run: python3 -m pytest -q test_calendar_effects.py
"""
import os
import sys
from datetime import date, timedelta

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import calendar_effects as ce
import market_calendar as mc

D = date


def test_rules_reproduce_the_official_2026_calendar():
    assert set(ce.nyse_rule_holidays(2026)) == set(mc.MARKET_HOLIDAYS_2026)
    # 2026-11-06: 2027 added to market_calendar.py (CALENDAR YEAR-ADD), so the official file takes over 2027
    assert ce.covered_years() == {2026, 2027}, "add 2028 to market_calendar.py in December and it takes over"
    assert set(ce.nyse_rule_holidays(2027)) == set(mc.MARKET_HOLIDAYS_2027), "the rules reproduce the official 2027 calendar too"
    for d in mc.MARKET_HOLIDAYS_2026:
        assert not ce.is_trading_day(d) and ce.is_holiday(d)


def test_2027_holidays_by_rule():
    assert sorted(ce.nyse_rule_holidays(2027)) == [
        D(2027, 1, 1), D(2027, 1, 18), D(2027, 2, 15), D(2027, 3, 26), D(2027, 5, 31),
        D(2027, 6, 18), D(2027, 7, 5), D(2027, 9, 6), D(2027, 11, 25), D(2027, 12, 24)]
    assert ce.holiday_source(D(2027, 1, 1)) == "market_calendar.py"   # official since 2026-11-06 (was the rule estimate)
    assert ce.holiday_source(D(2028, 1, 17)) == "estimated (NYSE holiday rules)"   # MLK Day 2028: still by rule
    assert ce.holiday_source(D(2026, 12, 25)) == "market_calendar.py"
    # New Year's Day 2028 is a Saturday: NYSE does not close the Friday before
    assert ce.is_trading_day(D(2027, 12, 31))
    assert not ce.is_trading_day(D(2027, 12, 24)) and ce.is_half_day(D(2027, 11, 26))


def test_session_arithmetic_across_holidays():
    assert ce.sessions_after(D(2026, 11, 25), 2) == [D(2026, 11, 27), D(2026, 11, 30)]
    assert ce.prev_session(D(2026, 11, 27)) == D(2026, 11, 25)
    assert ce.add_sessions(D(2026, 12, 31), 1) == D(2027, 1, 4)
    assert ce.add_sessions(D(2027, 1, 4), -1) == D(2026, 12, 31)
    assert ce.month_last_session(2026, 11) == D(2026, 11, 30)
    assert ce.month_last_session(2027, 1) == D(2027, 1, 29)
    assert ce.month_first_session(2027, 1) == D(2027, 1, 4)
    assert ce.sessions_to_month_end(D(2026, 11, 23)) == 4
    assert ce.session_of_month(D(2026, 12, 3)) == 2


def test_sp500_quarterly_rebalance_dates():
    got = [d for d, _ in ce.sp500_rebalance_dates(2026)]
    assert got == [D(2026, 3, 20), D(2026, 6, 18), D(2026, 9, 18), D(2026, 12, 18)]
    assert "holiday" in ce.sp500_rebalance_dates(2026)[1][1], "June 19, 2026 is Juneteenth"
    got27 = [d for d, _ in ce.sp500_rebalance_dates(2027)]
    assert got27 == [D(2027, 3, 19), D(2027, 6, 17), D(2027, 9, 17), D(2027, 12, 17)]


def test_russell_and_nasdaq100_reconstitutions():
    assert ce.russell_reconstitution_dates(2026) == [
        (D(2026, 6, 26), "FTSE Russell 2026 reconstitution schedule"),
        (D(2026, 12, 11), "FTSE Russell December 2026 reconstitution schedule")]
    r27 = ce.russell_reconstitution_dates(2027)
    assert [d for d, _ in r27] == [D(2027, 6, 25), D(2027, 12, 10)]
    assert all("estimated" in s for _, s in r27)
    assert [d for d, _ in ce.russell_reconstitution_dates(2025)] == [D(2025, 6, 27)], "annual before 2026"
    assert ce.nasdaq100_reconstitution_date(2026)[0] == D(2026, 12, 18)
    assert ce.nasdaq100_reconstitution_date(2027)[0] == D(2027, 12, 17)
    # real past dates: effective before the Monday open after these Fridays
    assert ce.nasdaq100_reconstitution_date(2024)[0] == D(2024, 12, 20)
    assert ce.nasdaq100_reconstitution_date(2025)[0] == D(2025, 12, 19)


def test_settlement_cycle_dates():
    assert ce.settlement_days(D(1995, 6, 6)) == 5 and ce.settlement_days(D(1995, 6, 7)) == 3
    assert ce.settlement_days(D(2017, 9, 1)) == 3 and ce.settlement_days(D(2017, 9, 5)) == 2
    assert ce.settlement_days(D(2024, 5, 24)) == 2 and ce.settlement_days(D(2024, 5, 28)) == 1


def _phases(start, end):
    out, d = {}, start
    while d <= end:
        if ce.is_trading_day(d):
            out[d] = ce.month_end_phase(d)
        d += timedelta(days=1)
    return out


def test_dash_for_cash_windows_follow_settlement():
    # T+3 (the paper's sample): sell T-8..T-4, reversal T-3..T. June 2010, T = June 30.
    p = _phases(D(2010, 6, 17), D(2010, 6, 30))
    assert [d for d, v in p.items() if v == "cash_sell"] == [D(2010, 6, 18), D(2010, 6, 21), D(2010, 6, 22),
                                                               D(2010, 6, 23), D(2010, 6, 24)]
    assert [d for d, v in p.items() if v == "reversal"] == [D(2010, 6, 25), D(2010, 6, 28), D(2010, 6, 29),
                                                              D(2010, 6, 30)]
    # T+2: sell T-7..T-3. March 2019, T = March 29.
    p = _phases(D(2019, 3, 19), D(2019, 3, 29))
    assert [d for d, v in p.items() if v == "cash_sell"][0] == D(2019, 3, 20)
    assert [d for d, v in p.items() if v == "reversal"] == [D(2019, 3, 27), D(2019, 3, 28), D(2019, 3, 29)]
    # T+1 (now): sell T-6..T-2. November 2026, T = November 30 (Thanksgiving week).
    p = _phases(D(2026, 11, 18), D(2026, 12, 4))
    assert [d for d, v in p.items() if v == "cash_sell"] == [D(2026, 11, 19), D(2026, 11, 20), D(2026, 11, 23),
                                                               D(2026, 11, 24), D(2026, 11, 25)]
    assert [d for d, v in p.items() if v == "reversal"] == [D(2026, 11, 27), D(2026, 11, 30)]
    assert [d for d, v in p.items() if v == "new_month_buy"] == [D(2026, 12, 1), D(2026, 12, 2), D(2026, 12, 3)]
    assert p[D(2026, 12, 4)] is None and p[D(2026, 11, 18)] is None


def test_rebalance_window_is_the_last_five_sessions():
    w = ce.rebalance_window(D(2026, 11, 23))
    assert w == {"month_end": D(2026, 11, 30), "k": 4, "quarter_end": False, "year_end": False}
    assert ce.rebalance_window(D(2026, 11, 20)) is None
    q = ce.rebalance_window(D(2026, 12, 28))
    assert q["quarter_end"] and q["year_end"] and q["month_end"] == D(2026, 12, 31)
    assert ce.rebalance_window(D(2026, 12, 31))["k"] == 0


def test_tax_and_earnings_windows():
    assert ce.tax_window(D(2026, 10, 15)) == "fund_tax_loss"
    assert ce.tax_window(D(2026, 11, 2)) == "fund_tax_loss_reversal"
    assert ce.tax_window(D(2026, 12, 10)) == "december_tax_loss"
    assert ce.tax_window(D(2026, 12, 28)) == "turn_of_year" and ce.tax_window(D(2027, 1, 5)) == "turn_of_year"
    assert ce.tax_window(D(2027, 1, 8)) is None and ce.tax_window(D(2026, 9, 30)) is None
    assert ce.earnings_season(D(2026, 11, 6)) == "peak"       # 37 days after Sep 30
    assert ce.earnings_season(D(2026, 11, 20)) == "season"    # 51 days
    assert ce.earnings_season(D(2026, 12, 1)) is None
    assert ce.earnings_season(D(2026, 10, 5)) is None
    assert ce.earnings_season(D(2027, 1, 19)) == "peak"


def test_calendar_window_rows():
    rows = ce.calendar_window(D(2026, 11, 5))
    assert rows == sorted(rows, key=lambda r: (r["date"], r["kind"]))
    for r in rows:
        assert set(r) >= {"date", "event", "effect", "kind", "source"}
        assert r["event"] and r["effect"] and len(r["date"]) == 10
    by = {r["kind"]: r for r in rows}
    assert by["cash_sell"]["date"] == "2026-11-19" and by["cash_sell"]["end"] == "2026-11-25"
    assert "T+1" in by["cash_sell"]["effect"]
    assert by["rebalance"]["date"] == "2026-11-24" and by["rebalance"]["end"] == "2026-12-01"
    assert by["holiday"]["date"] == "2026-11-26" and by["half_day"]["date"] == "2026-11-27"
    assert by["short_week"]["date"] == "2026-11-23" and by["short_week"]["effect"].startswith("4 sessions")
    assert by["russell_recon"]["date"] == "2026-12-11"
    assert by["ndx_recon"]["date"] == "2026-12-18" and by["sp500_rebalance"]["date"] == "2026-12-18"
    yearend = {r["kind"]: r for r in ce.calendar_window(D(2026, 12, 20), 30)}
    assert yearend["rebalance_quarter"]["date"] == "2026-12-28" and yearend["rebalance_quarter"]["end"] == "2027-01-04"
    assert yearend["year_end"]["date"] == "2026-12-31"
    hol = [r for r in ce.calendar_window(D(2026, 12, 20), 30) if r["kind"] == "holiday"]
    assert {r["date"]: r["source"] for r in hol} == {"2026-12-25": "market_calendar.py",
                                                    "2027-01-01": "market_calendar.py",
                                                    "2027-01-18": "market_calendar.py"}


def test_module_is_pure():
    src = open(os.path.join(_ROOT, "calendar_effects.py")).read()
    for banned in ("import requests", "urllib", "open(", "subprocess"):
        assert banned not in src, banned
