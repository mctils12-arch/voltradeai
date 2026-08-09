# Regression test for the 2026-08-06 full-code-review finding
# "earnings-day full sizing".
#
# Root cause: get_earnings_date() caught every exception from the yfinance
# lookup (import failure, network error, rate limit, unrecognized response
# shape) and returned None — exactly the same None a *confirmed* "no
# earnings scheduled" result produced. _earnings_scalar() then treated both
# identically: `if earnings_date is None: return 1.0  # Unknown = no
# adjustment`. So any yfinance hiccup silently gave a position full size,
# indistinguishable from a genuinely quiet week, right before whatever the
# earnings-proximity scalar exists to protect against. This scalar feeds
# calculate_position()'s Kelly sizing, options_execution._dynamic_options_
# size(), and options_execution._check_earnings_guard()'s blocking gate —
# all three inherited the same fail-open behavior from one shared function.
#
# Fix: _fetch_earnings_date_with_status() now returns (date, lookup_ok) so
# _earnings_scalar() can tell "confirmed nothing scheduled" (still 1.0x)
# apart from "could not determine" (now 0.85x, the same mild caution as the
# 5-7-day-out tier — not full size, not the harshest tier either). Offline-
# safe: yfinance is replaced with a fake module in sys.modules, no network
# access.
import sys
import types
from datetime import datetime, timedelta

import position_sizing


def _install_fake_yfinance(monkeypatch, ticker_cls):
    fake_mod = types.ModuleType("yfinance")
    fake_mod.Ticker = ticker_cls
    monkeypatch.setitem(sys.modules, "yfinance", fake_mod)


def _clear_cache(monkeypatch):
    monkeypatch.setattr(position_sizing, "_earnings_cache", {})


def test_yfinance_failure_no_longer_gives_full_size_end_to_end(monkeypatch):
    _clear_cache(monkeypatch)

    class RaisingTicker:
        def __init__(self, ticker):
            raise RuntimeError("SSL error / connection reset")

    _install_fake_yfinance(monkeypatch, RaisingTicker)
    # Pre-fix this was 1.0 (full size) — identical to a confirmed quiet week.
    assert position_sizing._earnings_scalar("ZZZZ") == 0.85


def test_lookup_failure_distinguished_from_confirmed_clean_in_status_fn(monkeypatch):
    _clear_cache(monkeypatch)

    class RaisingTicker:
        def __init__(self, ticker):
            raise RuntimeError("network down")

    _install_fake_yfinance(monkeypatch, RaisingTicker)
    date, ok = position_sizing._fetch_earnings_date_with_status("ZZZZ")
    assert date is None
    assert ok is False


def test_confirmed_no_earnings_scheduled_is_ok_true(monkeypatch):
    _clear_cache(monkeypatch)

    class EmptyCalTicker:
        def __init__(self, ticker):
            self.calendar = {}

    _install_fake_yfinance(monkeypatch, EmptyCalTicker)
    date, ok = position_sizing._fetch_earnings_date_with_status("ZZZZ")
    assert date is None
    assert ok is True


def test_confirmed_clean_result_still_gives_full_size(monkeypatch):
    monkeypatch.setattr(
        position_sizing, "_fetch_earnings_date_with_status", lambda ticker: (None, True)
    )
    assert position_sizing._earnings_scalar("ZZZZ") == 1.0


def test_lookup_failure_gives_mild_caution_not_full_size(monkeypatch):
    monkeypatch.setattr(
        position_sizing, "_fetch_earnings_date_with_status", lambda ticker: (None, False)
    )
    # Pre-fix this branch didn't exist — get_earnings_date collapsed both
    # cases to the same None, so _earnings_scalar always returned 1.0 here.
    assert position_sizing._earnings_scalar("ZZZZ") == 0.85


def test_known_date_tiers_unchanged(monkeypatch):
    def _mk(days):
        d = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        return lambda ticker: (d, True)

    monkeypatch.setattr(position_sizing, "_fetch_earnings_date_with_status", _mk(1))
    assert position_sizing._earnings_scalar("ZZZZ") == 0.35
    monkeypatch.setattr(position_sizing, "_fetch_earnings_date_with_status", _mk(3))
    assert position_sizing._earnings_scalar("ZZZZ") == 0.60
    monkeypatch.setattr(position_sizing, "_fetch_earnings_date_with_status", _mk(6))
    assert position_sizing._earnings_scalar("ZZZZ") == 0.85
    monkeypatch.setattr(position_sizing, "_fetch_earnings_date_with_status", _mk(20))
    assert position_sizing._earnings_scalar("ZZZZ") == 1.0


def test_malformed_date_string_fails_closed_not_open(monkeypatch):
    monkeypatch.setattr(
        position_sizing, "_fetch_earnings_date_with_status", lambda ticker: ("not-a-date", True)
    )
    # Pre-fix the parse-exception branch returned 1.0 (full size).
    assert position_sizing._earnings_scalar("ZZZZ") == 0.85


def test_get_earnings_date_stays_backward_compatible(monkeypatch):
    monkeypatch.setattr(
        position_sizing, "_fetch_earnings_date_with_status",
        lambda ticker: ("2026-09-01", True),
    )
    assert position_sizing.get_earnings_date("AAPL") == "2026-09-01"
