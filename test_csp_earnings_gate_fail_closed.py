# Regression test for the "CSP earnings gate fails open" finding, filed
# 2026-08-06 alongside "earnings-day full sizing" (fixed in v1.0.635,
# position_sizing.py) and left explicitly unclaimed as its own sibling item
# in that fix's NEXT note — same shape, different file/data source.
#
# Root cause: options_scanner._fetch_earnings_calendar() caught every
# exception (network error, timeout, bad Finnhub response) and returned {},
# indistinguishable from "the calendar was fetched fine and genuinely
# nobody in the universe reports soon". csp_universe._score_earnings() then
# did `earnings_cal.get(ticker, 99)` -> days=99 -> score=100.0 ("no earnings
# concern") for EVERY ticker whenever the fetch failed — the earnings-safety
# component of the CSP composite score silently went to its most permissive
# value on a data outage instead of failing closed.
#
# Fix: _fetch_earnings_calendar_with_status() returns (calendar, lookup_ok)
# so csp_universe._layer2_score() can tell "confirmed calendar, no one
# reporting soon" (still 100.0 per-ticker via the normal day-count logic)
# apart from "could not fetch the calendar at all" (now 80.0 for every
# ticker that scan — the same mild-caution tier the 15-30-day-out case
# already uses, not the harshest tier and not full permissiveness).
import types

import pytest

import csp_universe
import options_scanner
from csp_universe import _score_earnings


# ---------------------------------------------------------------------------
# _score_earnings: lookup_ok plumbing
# ---------------------------------------------------------------------------

def test_lookup_failure_gives_mild_caution_not_full_permissiveness():
    # Pre-fix this branch didn't exist — a missing ticker always fell
    # through to days=99 -> 100.0 regardless of why it was missing.
    assert _score_earnings("ZZZZ", {}, lookup_ok=False) == 80.0


def test_lookup_failure_applies_even_if_ticker_happens_to_be_in_a_stale_dict():
    # lookup_ok=False means "don't trust this scan's calendar at all" —
    # a stale/partial dict from a prior successful call must not leak
    # tickers through as if this scan's fetch had succeeded.
    assert _score_earnings("AAPL", {"AAPL": 45}, lookup_ok=False) == 80.0


def test_confirmed_empty_calendar_still_scores_no_concern():
    # A genuinely successful fetch that found nobody reporting soon is
    # unchanged — this is NOT the failure case.
    assert _score_earnings("ZZZZ", {}, lookup_ok=True) == 100.0


def test_lookup_ok_defaults_true_for_existing_callers():
    # Backward compatibility: any caller not yet updated to pass lookup_ok
    # keeps today's behavior for a confirmed-empty/missing-ticker calendar.
    assert _score_earnings("ZZZZ", {}) == 100.0


def test_known_day_tiers_unchanged_when_lookup_ok():
    cal = {"AAPL": 1, "MSFT": 5, "NVDA": 10, "TSLA": 25, "SPY": 60}
    assert _score_earnings("AAPL", cal, lookup_ok=True) == 0.0
    assert _score_earnings("MSFT", cal, lookup_ok=True) == 10.0
    assert _score_earnings("NVDA", cal, lookup_ok=True) == 50.0
    assert _score_earnings("TSLA", cal, lookup_ok=True) == 80.0
    assert _score_earnings("SPY", cal, lookup_ok=True) == 100.0


# ---------------------------------------------------------------------------
# options_scanner._fetch_earnings_calendar_with_status: fetch-level status
# ---------------------------------------------------------------------------

def test_fetch_with_status_returns_ok_false_on_request_exception(monkeypatch):
    def _raise(*args, **kwargs):
        raise ConnectionError("network down")
    monkeypatch.setattr(options_scanner.requests, "get", _raise)
    cal, ok = options_scanner._fetch_earnings_calendar_with_status(days_ahead=30)
    assert cal == {}
    assert ok is False


def test_fetch_with_status_returns_ok_true_on_success(monkeypatch):
    class FakeResp:
        def json(self):
            return {"earningsCalendar": []}
    monkeypatch.setattr(options_scanner.requests, "get", lambda *a, **k: FakeResp())
    cal, ok = options_scanner._fetch_earnings_calendar_with_status(days_ahead=30)
    assert cal == {}
    assert ok is True


def test_fetch_with_status_ok_true_even_with_real_rows(monkeypatch):
    from datetime import datetime, timedelta
    in3 = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")

    class FakeResp:
        def json(self):
            return {"earningsCalendar": [{"symbol": "AAPL", "date": in3}]}
    monkeypatch.setattr(options_scanner.requests, "get", lambda *a, **k: FakeResp())
    cal, ok = options_scanner._fetch_earnings_calendar_with_status(days_ahead=30)
    assert ok is True
    # datetime.now() carries a time-of-day component the source date string
    # doesn't, so the day-count can land on 2 or 3 depending on wall-clock
    # time of day — that fencepost behavior predates this fix and isn't
    # what this test is checking; only that a real row survives with ok=True.
    assert cal.get("AAPL") in (2, 3)


def test_backward_compatible_wrapper_still_returns_bare_dict(monkeypatch):
    def _raise(*args, **kwargs):
        raise ConnectionError("network down")
    monkeypatch.setattr(options_scanner.requests, "get", _raise)
    cal = options_scanner._fetch_earnings_calendar(days_ahead=30)
    assert cal == {}


# ---------------------------------------------------------------------------
# csp_universe._layer2_score wiring: a fetch failure must not silently
# score every candidate as "no earnings concern".
# ---------------------------------------------------------------------------

def test_layer2_score_wires_lookup_failure_through_to_score_earnings(monkeypatch, tmp_path):
    # A fresh, never-before-written cache path per test run — reusing a
    # fixed path across runs hits _layer2_score's own on-disk score cache
    # and short-circuits before _score_earnings is ever called.
    monkeypatch.setattr(csp_universe, "SCORES_CACHE_PATH", str(tmp_path / "scores_cache.json"))
    monkeypatch.setattr(
        options_scanner, "_fetch_earnings_calendar_with_status",
        lambda days_ahead=30: ({}, False),
    )
    captured = {}
    real_score_earnings = csp_universe._score_earnings

    def _spy(ticker, earnings_cal, lookup_ok=True):
        captured["lookup_ok"] = lookup_ok
        return real_score_earnings(ticker, earnings_cal, lookup_ok=lookup_ok)
    monkeypatch.setattr(csp_universe, "_score_earnings", _spy)

    csp_universe._layer2_score([("AAPL", 100.0, 1_000_000, 100_000_000.0)])
    assert captured.get("lookup_ok") is False
