"""
Market Calendar Utility (v1.0.34)

NYSE/NASDAQ holiday and half-day schedule for 2026.
Used by the options scanner to:
  1. Skip opening new options positions on half-days (reduced liquidity)
  2. Adjust theta decay expectations around long weekends
  3. Avoid selling premium before 3-day or 4-day weekends when gamma risk
     is unmonitored

Source: https://www.nasdaqtrader.com/trader.aspx?id=calendar
Updated annually — add 2027 dates when available.
"""

from datetime import date, timedelta
from typing import Optional


# ── 2026 NYSE/NASDAQ Full Closures ──────────────────────────────────────────
MARKET_HOLIDAYS_2026 = {
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents' Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth (observed)
    date(2026, 7, 3),    # Independence Day (observed — July 4 is Saturday)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
}

# ── 2026 Early Close Days (1:00 PM ET) ──────────────────────────────────────
HALF_DAYS_2026 = {
    date(2026, 11, 27),  # Day after Thanksgiving
    date(2026, 12, 24),  # Christmas Eve
}

# Combine all non-standard days
ALL_HOLIDAYS = MARKET_HOLIDAYS_2026
ALL_HALF_DAYS = HALF_DAYS_2026


def is_market_holiday(d: Optional[date] = None) -> bool:
    """True if the given date is a full market closure."""
    d = d or date.today()
    if d.weekday() >= 5:  # Saturday/Sunday
        return True
    return d in ALL_HOLIDAYS


def is_half_day(d: Optional[date] = None) -> bool:
    """True if the market closes early (1:00 PM ET)."""
    d = d or date.today()
    return d in ALL_HALF_DAYS


def is_short_week(d: Optional[date] = None) -> bool:
    """True if this calendar week has < 5 trading days.

    Short weeks matter for options because:
      - Less time for theta to work in your favor
      - Compressed trading increases gap risk relative to DTE
      - Options with Friday expiry on a 4-day week have accelerated
        gamma near expiry
    """
    d = d or date.today()
    # Find Monday of this week
    monday = d - timedelta(days=d.weekday())
    trading_days = 0
    for i in range(5):  # Mon-Fri
        day = monday + timedelta(days=i)
        if not is_market_holiday(day):
            trading_days += 1
    return trading_days < 5


def trading_days_this_week(d: Optional[date] = None) -> int:
    """Count trading days in the week containing date d."""
    d = d or date.today()
    monday = d - timedelta(days=d.weekday())
    count = 0
    for i in range(5):
        day = monday + timedelta(days=i)
        if not is_market_holiday(day):
            count += 1
    return count


def next_trading_day(d: Optional[date] = None) -> date:
    """Return the next open market day after d."""
    d = d or date.today()
    candidate = d + timedelta(days=1)
    while is_market_holiday(candidate):
        candidate += timedelta(days=1)
    return candidate


def days_until_next_holiday(d: Optional[date] = None) -> int:
    """Calendar days until the next market holiday (including weekends)."""
    d = d or date.today()
    for days_ahead in range(1, 365):
        check = d + timedelta(days=days_ahead)
        if check in ALL_HOLIDAYS:
            return days_ahead
    return 365


def is_pre_long_weekend(d: Optional[date] = None) -> bool:
    """True if tomorrow starts a 3+ day market closure.

    Examples: Thursday before Good Friday (3 days off),
    Wednesday before Thanksgiving (Thu closed + Fri half-day + weekend).

    Risk: Selling premium before a long weekend means unmonitored gamma
    exposure while markets are closed.
    """
    d = d or date.today()
    consecutive_closed = 0
    check = d + timedelta(days=1)
    while is_market_holiday(check):
        consecutive_closed += 1
        check += timedelta(days=1)
    return consecutive_closed >= 3


def should_skip_new_options(d: Optional[date] = None) -> tuple:
    """Determine if we should skip opening new options positions today.

    Returns: (should_skip: bool, reason: str)

    Skip conditions:
      1. Half-day: reduced liquidity, wider spreads, shorter monitoring window
      2. Pre-long-weekend: unmonitored gamma risk over 3+ days
    """
    d = d or date.today()

    if is_half_day(d):
        return True, "Half-day trading (1:00 PM close) — reduced liquidity and monitoring window"

    if is_pre_long_weekend(d):
        return True, "Pre-long-weekend — unmonitored gamma exposure over 3+ day closure"

    return False, ""
