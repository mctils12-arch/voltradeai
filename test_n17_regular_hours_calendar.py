"""N17: the _is_regular_hours helpers must respect weekends, holidays and half-days."""
from datetime import datetime
from zoneinfo import ZoneInfo
import pytest
import bot_engine
import instrument_selector
import options_scanner

ET = ZoneInfo("America/New_York")


def _fake(y, m, d, hh, mm):
    real = datetime(y, m, d, hh, mm, tzinfo=ET)

    class _DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return real.astimezone(tz) if tz else real
    return _DT


MODS = [bot_engine, instrument_selector, options_scanner]
# (y, m, d, h, m, expected)
CASES = [
    (2026, 10, 28, 11, 0, True),    # Wed mid-session
    (2026, 10, 28, 9, 0, False),    # before open
    (2026, 10, 28, 16, 0, False),   # after close
    (2026, 10, 31, 11, 0, False),   # Saturday
    (2026, 11, 1, 11, 0, False),    # Sunday
    (2026, 11, 26, 11, 0, False),   # Thanksgiving
    (2026, 11, 27, 11, 0, True),    # half-day, before 13:00
    (2026, 11, 27, 14, 0, False),   # half-day, after 13:00 close
]


@pytest.mark.parametrize("mod", MODS, ids=lambda m: m.__name__)
@pytest.mark.parametrize("y,m,d,hh,mm,expected", CASES)
def test_regular_hours_respects_calendar(monkeypatch, mod, y, m, d, hh, mm, expected):
    monkeypatch.setattr(mod, "datetime", _fake(y, m, d, hh, mm))
    assert mod._is_regular_hours() is expected
