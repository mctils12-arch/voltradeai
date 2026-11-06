#!/usr/bin/env python3
"""
[REPAIR] 2026-11-06 N17: the options helpers' regular-hours test asks the broker clock too.

instrument_selector._is_regular_hours (options candidates, the options score,
the instrument choice) and options_scanner._is_regular_hours (every options
setup and the scan itself) read only the time of day, so a Saturday or a
holiday at 11:00 ET counted as market hours. Orphan PR #1506 fixed the hour
test with the NYSE calendar; #1641 then gave bot_engine the broker clock
instead, and this applies the same rule to these two: inside 09:30-16:00 ET the
broker clock must also say open (broker_clock.orders_allowed). Outside that
window the clock is never asked.

Run: python3 -m pytest -q test_regular_hours_broker_clock.py
"""
import os
import sys
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import broker_clock
import instrument_selector
import options_scanner

_ET = ZoneInfo("America/New_York")


def _fixed_datetime(hour, minute=0):
    """A datetime class whose now() is a fixed ET wall time (instrument_selector
    computes its hour from datetime.now)."""
    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 7, hour, minute, tzinfo=_ET).astimezone(tz) if tz else datetime(2026, 11, 7, hour, minute)
    return _DT


def _clock(allowed):
    calls = []

    def fake(now=None):
        calls.append(now)
        return (allowed, "broker says open" if allowed else "broker says closed")
    return fake, calls


def test_instrument_selector_inside_hours_follows_the_broker_clock():
    for allowed in (True, False):
        fake, calls = _clock(allowed)
        with patch.object(instrument_selector, "datetime", _fixed_datetime(11, 0)), \
             patch.object(broker_clock, "orders_allowed", side_effect=fake):
            assert instrument_selector._is_regular_hours() is allowed
        assert len(calls) == 1


def test_instrument_selector_outside_hours_never_asks_the_clock():
    for hour, minute in ((9, 29), (16, 0), (20, 30), (4, 0)):
        fake, calls = _clock(True)
        with patch.object(instrument_selector, "datetime", _fixed_datetime(hour, minute)), \
             patch.object(broker_clock, "orders_allowed", side_effect=fake):
            assert instrument_selector._is_regular_hours() is False
        assert calls == []


def test_options_scanner_inside_hours_follows_the_broker_clock():
    for allowed in (True, False):
        fake, calls = _clock(allowed)
        with patch.object(options_scanner, "_et_now_hour", return_value=10.5), \
             patch.object(broker_clock, "orders_allowed", side_effect=fake):
            assert options_scanner._is_regular_hours() is allowed
        assert len(calls) == 1


def test_options_scanner_outside_hours_never_asks_the_clock():
    for hour in (9.49, 16.0, 19.0, 4.0):
        fake, calls = _clock(True)
        with patch.object(options_scanner, "_et_now_hour", return_value=hour), \
             patch.object(broker_clock, "orders_allowed", side_effect=fake):
            assert options_scanner._is_regular_hours() is False
        assert calls == []


def test_a_closed_clock_stops_the_options_scan():
    """A weekday-hour holiday: the scan returns no opportunities and touches no
    data, the same as after hours."""
    def _no_io(*_a, **_k):
        raise AssertionError("the scan fetched data while the broker said closed")
    with patch.object(options_scanner, "_et_now_hour", return_value=11.0), \
         patch.object(broker_clock, "orders_allowed", return_value=(False, "broker says closed")), \
         patch("requests.get", side_effect=_no_io):
        res = options_scanner.scan_options()
    assert res["opportunities"] == [] and res["regime"] == "CLOSED" and res["scanned"] == 0


def test_an_unreadable_clock_keeps_the_time_of_day_rule():
    """The suite's conftest stubs the clock read to fail: no reading at all
    counts as open, so inside the window the answer is the time of day alone."""
    broker_clock.reset_cache()
    with patch.object(options_scanner, "_et_now_hour", return_value=12.0):
        assert options_scanner._is_regular_hours() is True
    with patch.object(instrument_selector, "datetime", _fixed_datetime(12, 0)):
        assert instrument_selector._is_regular_hours() is True
