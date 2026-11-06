#!/usr/bin/env python3
"""
calendar_effects.py — step 2 of the decision pipeline (refocus spec, 2026-11-05:
"calendar effects: holiday/short weeks, quarter-end/year-end, index rebalance
dates, earnings season"). Pure: no network, no orders, no files.

For any date it answers: is the market open, how many sessions remain to the
month end, which month-end window (cash raising, reversal, new-month inflows)
a session falls in, and which scheduled flow events are near. flows_model.py
turns these into pressure estimates; this module makes no prediction itself.

Sources (every event carries one; "estimated" means computed by a rule, not
read from an official calendar):
- Holidays and half days: market_calendar.py (FROZEN data, 2026 only). For a
  year it does not cover, the standard NYSE holiday rules below estimate them.
  When next year's official dates are added there each December, they take
  over automatically (covered_years()).
- Month-end cash needs: Etula, Rinne, Suominen & Vaittinen (2020), "Dash for
  Cash: Monthly Market Impact of Institutional Liquidity Needs", Review of
  Financial Studies 33(1):75-111. With 3-day settlement, an institution that
  needs cash on the morning of the last trading day T must sell by the close of
  T-4; US stock returns were low on T-8..T-4 and high on T-3..T+3 (CRSP,
  1995-2013). The windows here move with the settlement cycle (T+2 from
  2017-09-05, T+1 from 2024-05-28). That shift is our inference from the
  paper's own mechanism; the paper did not test it.
- Month- and quarter-end rebalancing: Harvey, Mazzoleni & Melone, "The
  Unintended Consequences of Rebalancing" (NBER w33554, 2025, revised 2026):
  calendar rebalancers trade back to target at the month (quarter) end, and the
  calendar signal predicts the next day's return in the last week of the month.
- S&P 500 quarterly rebalance: after the close of the third Friday of March,
  June, September and December (S&P Dow Jones Indices). When that Friday is a
  holiday, the prior trading day is used (estimated).
- Russell US reconstitution: after the close of the fourth Friday of June
  (2026: June 26, FTSE Russell schedule). Semi-annual from 2026: December 2026
  is December 11 (FTSE Russell schedule); later Decembers are estimated as the
  second Friday, the 2026 pattern.
- Nasdaq-100 annual reconstitution: after the close of the third Friday of
  December (Nasdaq-100 methodology).
- Mutual-fund tax-loss window: since the 1986 Tax Reform Act most funds close
  their tax year on October 31 and sell losing positions before it. Gibson,
  Safieddine & Titman (2000), JFQA 35(3):369-386, found the sales and a strong
  November rebound in the first year, and that funds later spread the sales
  out, which weakens the effect.
- December tax-loss selling by individuals and calendar-year funds, and the
  early-January rebound: a long-studied hypothesis; no paper was read for it
  tonight, so it is listed as an event with no claimed size.
- Earnings season: ESTIMATED from the calendar (most S&P 500 companies report
  roughly 2-7 weeks after the quarter end), not read from an earnings calendar.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import market_calendar as mc

# US equity settlement cycle: T+5 -> T+3 on 1995-06-07 (SEC Rule 15c6-1),
# T+2 on 2017-09-05, T+1 on 2024-05-28.
SETTLEMENT_CHANGES: Tuple[Tuple[date, int], ...] = (
    (date(1995, 6, 7), 3), (date(2017, 9, 5), 2), (date(2024, 5, 28), 1))

# Dash for Cash windows, in sessions (Etula et al. 2020, Figure 4): the cash
# raise is the 5 sessions ending on the last day a sale still settles before T;
# the reversal runs from the next session through T; new-month inflows are the
# first 3 sessions of the month.
CASH_SELL_SESSIONS = 5
NEW_MONTH_BUY_SESSIONS = 3

# Month-end rebalancing window (Harvey, Mazzoleni & Melone: the calendar signal
# times a last-week dummy predicts the next day's return): the signal day t is
# one of the last REBALANCE_WINDOW_SESSIONS sessions of its month (k = 0..4),
# so the affected sessions are T-3 .. T+1.
REBALANCE_WINDOW_SESSIONS = 5

QUARTER_END_MONTHS = (3, 6, 9, 12)

# Announced reconstitution dates (FTSE Russell schedules for 2026).
RUSSELL_ANNOUNCED = {
    date(2026, 6, 26): "FTSE Russell 2026 reconstitution schedule",
    date(2026, 12, 11): "FTSE Russell December 2026 reconstitution schedule",
}

# Earnings season, estimated: days after the quarter end.
EARNINGS_SEASON_DAYS = (10, 55)
EARNINGS_PEAK_DAYS = (18, 42)


# ── NYSE holiday rules (used only for years market_calendar.py does not cover) ──

def _easter(year: int) -> date:
    """Gregorian Easter Sunday (the anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    lval = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lval) // 451
    month = (h + lval - 7 * m + 114) // 31
    day = ((h + lval - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th (1-based) `weekday` (Mon=0) of a month."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def third_friday(year: int, month: int) -> date:
    return nth_weekday(year, month, 4, 3)


def _observed(d: date) -> date:
    """NYSE observance: a Saturday holiday moves to Friday, a Sunday one to Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def nyse_rule_holidays(year: int) -> frozenset:
    """Full-day NYSE closures by rule (current rule set, Juneteenth from 2022).
    New Year's Day on a Saturday is not moved to the Friday before (NYSE rule).
    Unscheduled closures (mourning days, storms) cannot be predicted."""
    hol = set()
    ny = date(year, 1, 1)
    if ny.weekday() == 6:
        hol.add(date(year, 1, 2))
    elif ny.weekday() < 5:
        hol.add(ny)
    hol.add(nth_weekday(year, 1, 0, 3))          # Martin Luther King Jr. Day
    hol.add(nth_weekday(year, 2, 0, 3))          # Washington's Birthday
    hol.add(_easter(year) - timedelta(days=2))   # Good Friday
    hol.add(last_weekday(year, 5, 0))            # Memorial Day
    if year >= 2022:
        hol.add(_observed(date(year, 6, 19)))    # Juneteenth
    hol.add(_observed(date(year, 7, 4)))         # Independence Day
    hol.add(nth_weekday(year, 9, 0, 1))          # Labor Day
    hol.add(nth_weekday(year, 11, 3, 4))         # Thanksgiving
    hol.add(_observed(date(year, 12, 25)))       # Christmas
    return frozenset(h for h in hol if h.year == year)


def covered_years() -> set:
    """Years market_calendar.py has official dates for."""
    return {h.year for h in mc.ALL_HOLIDAYS}


def holiday_source(d: date) -> str:
    return "market_calendar.py" if d.year in covered_years() else "estimated (NYSE holiday rules)"


def is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    if d.year in covered_years():
        return not mc.is_market_holiday(d)
    return d not in nyse_rule_holidays(d.year)


def is_holiday(d: date) -> bool:
    """A weekday the market is closed."""
    return d.weekday() < 5 and not is_trading_day(d)


def is_half_day(d: date) -> bool:
    """1:00 PM ET close. Official for covered years; otherwise estimated: the
    day after Thanksgiving, and July 3 / December 24 when they are sessions."""
    if d.year in covered_years():
        return mc.is_half_day(d)
    if not is_trading_day(d):
        return False
    thanks = nth_weekday(d.year, 11, 3, 4)
    return d in (thanks + timedelta(days=1), date(d.year, 12, 24), date(d.year, 7, 3))


# ── session arithmetic ────────────────────────────────────────────────────────

def next_session(d: date) -> date:
    x = d + timedelta(days=1)
    while not is_trading_day(x):
        x += timedelta(days=1)
    return x


def prev_session(d: date) -> date:
    x = d - timedelta(days=1)
    while not is_trading_day(x):
        x -= timedelta(days=1)
    return x


def on_or_before_session(d: date) -> date:
    return d if is_trading_day(d) else prev_session(d)


def on_or_after_session(d: date) -> date:
    return d if is_trading_day(d) else next_session(d)


def add_sessions(d: date, n: int) -> date:
    """The n-th session after d (n >= 1), or before it (n <= -1); n == 0 is d."""
    x = d
    for _ in range(abs(n)):
        x = next_session(x) if n > 0 else prev_session(x)
    return x


def sessions_after(d: date, n: int) -> List[date]:
    out, x = [], d
    for _ in range(max(0, n)):
        x = next_session(x)
        out.append(x)
    return out


def month_last_session(year: int, month: int) -> date:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return on_or_before_session(nxt - timedelta(days=1))


def month_first_session(year: int, month: int) -> date:
    return on_or_after_session(date(year, month, 1))


def sessions_to_month_end(d: date) -> int:
    """k such that d is the session T-k, T = the last session of d's month (d a session)."""
    end = month_last_session(d.year, d.month)
    k, x = 0, d
    while x < end:
        x = next_session(x)
        k += 1
    return k


def session_of_month(d: date) -> int:
    """0 for the first session of d's month, 1 for the second, ..."""
    start = month_first_session(d.year, d.month)
    j, x = 0, start
    while x < d:
        x = next_session(x)
        j += 1
    return j


def settlement_days(d: date) -> int:
    s = 5
    for since, days in SETTLEMENT_CHANGES:
        if d >= since:
            s = days
    return s


def month_end_phase(x: date) -> Optional[str]:
    """Which Dash for Cash window session x is in: "cash_sell" (the 5 sessions
    ending on the last day a sale still settles before the last session T),
    "reversal" (from the next session through T), "new_month_buy" (the first 3
    sessions of a month), or None."""
    k = sessions_to_month_end(x)
    last_sell = settlement_days(x) + 1        # T+3 -> T-4, T+2 -> T-3, T+1 -> T-2
    if last_sell <= k < last_sell + CASH_SELL_SESSIONS:
        return "cash_sell"
    if 0 <= k < last_sell:
        return "reversal"
    if session_of_month(x) < NEW_MONTH_BUY_SESSIONS:
        return "new_month_buy"
    return None


def rebalance_window(signal_day: date) -> Optional[dict]:
    """When signal_day t is one of the last REBALANCE_WINDOW_SESSIONS sessions of
    its month, the month end T it points at (the session after t is one of
    T-3 .. T+1); else None."""
    k = sessions_to_month_end(signal_day)
    if k >= REBALANCE_WINDOW_SESSIONS:
        return None
    end = month_last_session(signal_day.year, signal_day.month)
    return {"month_end": end, "k": k, "quarter_end": end.month in QUARTER_END_MONTHS,
            "year_end": end.month == 12}


# ── index events ──────────────────────────────────────────────────────────────

def sp500_rebalance_dates(year: int) -> List[Tuple[date, str]]:
    out = []
    for m in QUARTER_END_MONTHS:
        f = third_friday(year, m)
        d = on_or_before_session(f)
        out.append((d, "S&P DJI rule: third Friday" if d == f
                    else "S&P DJI rule: third Friday is a holiday, prior session (estimated)"))
    return out


def russell_reconstitution_dates(year: int) -> List[Tuple[date, str]]:
    out = []
    june = on_or_before_session(nth_weekday(year, 6, 4, 4))
    out.append((june, RUSSELL_ANNOUNCED.get(june, "estimated: fourth Friday of June")))
    if year >= 2026:
        dec = date(2026, 12, 11) if year == 2026 else on_or_before_session(nth_weekday(year, 12, 4, 2))
        out.append((dec, RUSSELL_ANNOUNCED.get(dec, "estimated: second Friday of December (2026 pattern)")))
    return out


def nasdaq100_reconstitution_date(year: int) -> Tuple[date, str]:
    f = third_friday(year, 12)
    d = on_or_before_session(f)
    return d, ("Nasdaq-100 methodology: third Friday of December" if d == f
               else "third Friday of December is a holiday, prior session (estimated)")


def tax_window(d: date) -> Optional[str]:
    """"fund_tax_loss" (October: funds with an October 31 year end sell losers),
    "fund_tax_loss_reversal" (November), "december_tax_loss" (Dec 1-23),
    "turn_of_year" (Dec 24 - Jan 7), else None."""
    if d.month == 10:
        return "fund_tax_loss"
    if d.month == 11:
        return "fund_tax_loss_reversal"
    if d.month == 12 and d.day <= 23:
        return "december_tax_loss"
    if (d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 7):
        return "turn_of_year"
    return None


def earnings_season(d: date) -> Optional[str]:
    """"peak" / "season" / None, ESTIMATED from days since the last quarter end."""
    q_end = None
    for y in (d.year, d.year - 1):
        for m in (12, 9, 6, 3):
            end = date(y + 1, 1, 1) - timedelta(days=1) if m == 12 else date(y, m + 1, 1) - timedelta(days=1)
            if end < d:
                q_end = end
                break
        if q_end:
            break
    days = (d - q_end).days
    if EARNINGS_PEAK_DAYS[0] <= days <= EARNINGS_PEAK_DAYS[1]:
        return "peak"
    if EARNINGS_SEASON_DAYS[0] <= days <= EARNINGS_SEASON_DAYS[1]:
        return "season"
    return None


# ── events for the UI and the drivers ────────────────────────────────────────

EFFECT_TEXT = {
    "cash_sell": ("Month-end cash raise",
                  "Pensions and funds sell to raise cash for payments due at the month end (Etula et al. 2020); "
                  "historically a weak window for stocks. Window moved for T+{s} settlement (our inference)."),
    "reversal": ("Month-end reversal",
                 "After the cash raise, prices historically recovered through the last session of the month."),
    "new_month_buy": ("New-month inflows",
                      "Payments received at the month end are reinvested in the first sessions of the month; "
                      "historically a strong window for stocks."),
    "rebalance": ("Month-end rebalancing",
                  "Balanced funds and pensions trade back to their stock/bond targets at the month end "
                  "(Harvey, Mazzoleni & Melone 2025): they sell stocks after stocks beat bonds, and buy after bonds won."),
    "rebalance_quarter": ("Quarter-end rebalancing",
                          "Monthly AND quarterly rebalancers trade at this month end, so the rebalancing flow is larger."),
    "year_end": ("Year end",
                 "Calendar-year rebalancing and tax-loss selling meet at the December close."),
    "sp500_rebalance": ("S&P 500 quarterly rebalance",
                        "Index funds trade share-count and membership changes at this close; heavy closing volume. "
                        "No broad-market direction is assumed."),
    "russell_recon": ("Russell reconstitution",
                      "Russell index funds rebalance at this close, one of the heaviest volume days of the year, "
                      "concentrated in small caps. No large-cap direction is assumed."),
    "ndx_recon": ("Nasdaq-100 reconstitution",
                  "Nasdaq-100 funds trade additions and deletions at this close; the flow lands in "
                  "technology, communication and semiconductor names. Direction depends on the changes."),
    "fund_tax_loss": ("Mutual-fund tax-loss window",
                      "Funds with an October 31 tax year end sell losing positions before it "
                      "(Gibson, Safieddine & Titman 2000); later years showed the selling spread out."),
    "fund_tax_loss_reversal": ("Tax-loss rebound window",
                               "Losers sold before October 31 historically rebounded in November "
                               "(strongest in the first year after the 1986 tax change)."),
    "december_tax_loss": ("December tax-loss selling",
                          "Individuals and calendar-year funds sell this year's losers before December 31 "
                          "(long-studied hypothesis; no size is claimed here)."),
    "turn_of_year": ("Turn of the year",
                     "This year's losers often rebound in early January after the tax selling ends "
                     "(no size is claimed here)."),
    "earnings_peak": ("Earnings season (peak, estimated)",
                      "Most companies report now (estimated from the calendar, not an earnings calendar): "
                      "single-stock gap risk is highest."),
    "earnings_season": ("Earnings season (estimated)",
                        "Companies are reporting (estimated from the calendar): single-stock gap risk."),
    "holiday": ("Market holiday", "The market is closed."),
    "half_day": ("Early close (1:00 PM ET)", "A short session: thin liquidity and wider spreads."),
    "short_week": ("Short week", "Fewer sessions this week: thinner liquidity; no direction is assumed."),
    "pre_long_weekend": ("Before a long weekend", "Three or more days closed after this session."),
}


def _ev(kind: str, start: date, end: Optional[date] = None, source: str = "", **fmt) -> dict:
    title, effect = EFFECT_TEXT[kind]
    row = {"date": start.isoformat(), "event": title, "effect": effect.format(**fmt) if fmt else effect,
           "kind": kind, "source": source}
    if end and end != start:
        row["end"] = end.isoformat()
    return row


def _group(days: List[date], label_of) -> List[Tuple[str, date, date]]:
    """Consecutive sessions with the same label -> (label, first, last)."""
    out: List[Tuple[str, date, date]] = []
    for d in days:
        lab = label_of(d)
        if lab is None:
            continue
        if out and out[-1][0] == lab and next_session(out[-1][2]) == d:
            out[-1] = (lab, out[-1][1], d)
        else:
            out.append((lab, d, d))
    return out


def calendar_window(as_of: date, lookahead_days: int = 45) -> List[dict]:
    """Every calendar event in the sessions after as_of up to as_of +
    lookahead_days: month-end windows, rebalancing weeks, index events, the
    tax and earnings windows, holidays, half days and short weeks. Sorted by
    date; each row {date, event, effect, kind, source[, end]}. A window that
    starts inside the span is shown whole (its end may lie beyond it)."""
    horizon_end = as_of + timedelta(days=lookahead_days)
    nxt = []
    x = next_session(as_of)
    while x <= horizon_end:
        nxt.append(x)
        x = next_session(x)
    rows: List[dict] = []
    if not nxt:
        return rows
    # extend the phase scan a little past the span so the last window is shown whole
    tail = sessions_after(nxt[-1], CASH_SELL_SESSIONS + NEW_MONTH_BUY_SESSIONS)
    for lab, a, b in _group(nxt + tail, month_end_phase):
        if a <= horizon_end:
            rows.append(_ev(lab, a, b, source="Etula et al. 2020 (settlement-adjusted windows)",
                            s=settlement_days(a)))
    # rebalancing: the sessions T-3 .. T+1 of each month end in range
    seen_ends = set()
    for d in [as_of] + nxt:
        win = rebalance_window(d)
        if not win or win["month_end"] in seen_ends:
            continue
        seen_ends.add(win["month_end"])
        end = win["month_end"]
        first = max(next_session(d), add_sessions(end, -(REBALANCE_WINDOW_SESSIONS - 2)))
        kind = "rebalance_quarter" if win["quarter_end"] else "rebalance"
        rows.append(_ev(kind, first, next_session(end), source="Harvey, Mazzoleni & Melone 2025"))
        if win["year_end"]:
            rows.append(_ev("year_end", end, source="calendar"))
    for y in {as_of.year, horizon_end.year}:
        for d, src in sp500_rebalance_dates(y):
            if as_of < d <= horizon_end:
                rows.append(_ev("sp500_rebalance", d, source=src))
        for d, src in russell_reconstitution_dates(y):
            if as_of < d <= horizon_end:
                rows.append(_ev("russell_recon", d, source=src))
        d, src = nasdaq100_reconstitution_date(y)
        if as_of < d <= horizon_end:
            rows.append(_ev("ndx_recon", d, source=src))
    # tax and earnings windows: the current one, or the next one to start within the horizon
    span = [as_of + timedelta(days=i) for i in range(1, lookahead_days + 1)]
    for lab, a, b in _group([d for d in span if is_trading_day(d)], tax_window):
        rows.append(_ev(lab, a, b, source="see calendar_effects.py"))
    for lab, a, b in _group([d for d in span if is_trading_day(d)], earnings_season):
        rows.append(_ev("earnings_peak" if lab == "peak" else "earnings_season", a, b,
                        source="estimated from the calendar"))
    # holidays, half days, short weeks in the span
    day = as_of + timedelta(days=1)
    weeks = set()
    while day <= horizon_end:
        if is_holiday(day):
            rows.append(_ev("holiday", day, source=holiday_source(day)))
        elif is_trading_day(day):
            if is_half_day(day):
                rows.append(_ev("half_day", day, source=holiday_source(day)))
            monday = day - timedelta(days=day.weekday())
            if monday not in weeks:
                weeks.add(monday)
                n = sum(1 for i in range(5) if is_trading_day(monday + timedelta(days=i)))
                if n < 5:
                    rows.append(_ev("short_week", monday, monday + timedelta(days=4),
                                    source=holiday_source(day)))
                    rows[-1]["effect"] = f"{n} sessions this week: thinner liquidity; no direction is assumed."
            closed_after = 0
            probe = day + timedelta(days=1)
            while not is_trading_day(probe) and closed_after < 6:
                closed_after += 1
                probe += timedelta(days=1)
            if closed_after >= 3:
                rows.append(_ev("pre_long_weekend", day, source=holiday_source(day)))
        day += timedelta(days=1)
    rows.sort(key=lambda r: (r["date"], r["kind"]))
    return rows


def session_context(x: date) -> Dict:
    """Plain facts about one session, for the pressure drivers."""
    return {"date": x.isoformat(), "phase": month_end_phase(x), "settlement_days": settlement_days(x),
            "sessions_to_month_end": sessions_to_month_end(x), "session_of_month": session_of_month(x),
            "tax_window": tax_window(x), "earnings": earnings_season(x), "half_day": is_half_day(x),
            "holiday_source": holiday_source(x)}
