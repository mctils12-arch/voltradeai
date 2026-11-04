"""
[REPAIR] 2026-11-04 structural follow-up to KB #45 (c) (CSP earnings guard,
third fix on the gate): a cut Finnhub window page is COMPLETED by chunked
re-fetches of its missing near end, so every calendar consumer sees the
reports of the next few days instead of each confirming per symbol.

RangeFinnhub models the real server: it honours from/to, returns rows
NEWEST first and cuts the page at ROW_CAP. A server that ignores the range
is also modelled (ignore_range=True) to pin that such answers are distrusted.
"""
from datetime import datetime, timedelta
from unittest.mock import patch

import options_scanner as osc

WINDOW = 47


def _d(n):
    return (datetime.now().date() + timedelta(days=n))


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class RangeFinnhub:
    def __init__(self, rows, ignore_range=False, fail_after=None, status=200):
        self.rows, self.ignore_range, self.fail_after, self.status = rows, ignore_range, fail_after, status
        self.calls = []

    def __call__(self, url, params=None, timeout=None, **kw):
        params = dict(params or {})
        self.calls.append(params)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            return _Resp(429, {"error": "API limit reached"})
        lo, hi = params["from"], params["to"]
        sel = [r for r in self.rows if self.ignore_range or lo <= r["date"] <= hi]
        sel.sort(key=lambda r: r["date"], reverse=True)          # newest first
        return _Resp(200, {"earningsCalendar": sel[:osc.EARNINGS_CALENDAR_ROW_CAP]})


def busy_calendar(per_day=60, extra=None):
    """per_day reports on every day 0..WINDOW so that the window overflows one page
    (60 x 48 = 2880 > 1500) while every single day fits a page."""
    rows = [{"symbol": f"Z{d:02d}{i:03d}", "date": _d(d).strftime("%Y-%m-%d")}
            for d in range(WINDOW + 1) for i in range(per_day)]
    for sym, day in (extra or {}).items():
        rows.append({"symbol": sym, "date": _d(day).strftime("%Y-%m-%d")})
    return rows


def fetch(fake):
    with patch.object(osc.requests, "get", fake):
        cal, ok = osc._fetch_earnings_calendar_with_status(days_ahead=WINDOW)
    return cal, ok, osc.earnings_calendar_coverage(WINDOW)


def test_root_cause_the_one_page_fetch_alone_misses_tomorrows_report():
    """Pins WHY completion exists: page 1 of this calendar starts ~day 22."""
    fake = RangeFinnhub(busy_calendar(extra={"IREN": 1}))
    page = fake({}, params={"from": _d(0).strftime("%Y-%m-%d"), "to": _d(WINDOW).strftime("%Y-%m-%d")})
    names = {r["symbol"] for r in page.json()["earningsCalendar"]}
    assert "IREN" not in names


def test_cut_page_is_completed_and_tomorrows_report_is_found():
    fake = RangeFinnhub(busy_calendar(extra={"IREN": 1, "BAC": 0}))
    cal, ok, cov = fetch(fake)
    assert ok is True
    assert cal["IREN"] == 1 and cal["BAC"] == 0
    assert cov["primary_truncated"] is True and cov["completed_by_chunks"] is True
    assert cov["truncated"] is False                     # the calendar now vouches for absent names
    assert cov["chunk_requests"] >= 1
    assert len(fake.calls) == 1 + cov["chunk_requests"] <= 1 + osc.EARNINGS_CHUNK_MAX_REQUESTS
    # the primary-page evidence is preserved, not overwritten
    assert cov["at_row_cap"] is True and cov["calendar_entries"] == osc.EARNINGS_CALENDAR_ROW_CAP


def test_names_in_both_primary_and_chunk_keep_the_nearest_date():
    fake = RangeFinnhub(busy_calendar(extra={"DUP": 30}) + [{"symbol": "DUP", "date": _d(2).strftime("%Y-%m-%d")}])
    cal, _ok, _cov = fetch(fake)
    assert cal["DUP"] == 2


def test_a_server_that_ignores_the_range_is_not_trusted():
    fake = RangeFinnhub(busy_calendar(extra={"IREN": 1}), ignore_range=True)
    _cal, ok, cov = fetch(fake)
    assert ok is True and cov["truncated"] is True and cov["completed_by_chunks"] is False


def test_a_failing_chunk_leaves_the_calendar_truncated():
    fake = RangeFinnhub(busy_calendar(), fail_after=2)
    cal, ok, cov = fetch(fake)
    assert ok is True and cov["truncated"] is True and cov["completed_by_chunks"] is False


def test_a_single_day_that_fills_a_page_cannot_be_completed():
    rows = [{"symbol": f"A{i:04d}", "date": _d(1).strftime("%Y-%m-%d")} for i in range(1500)]
    rows += [{"symbol": f"B{i:04d}", "date": _d(30).strftime("%Y-%m-%d")} for i in range(10)]
    _cal, _ok, cov = fetch(RangeFinnhub(rows))
    assert cov["truncated"] is True


def test_request_budget_is_bounded():
    fake = RangeFinnhub(busy_calendar(per_day=1600 // 2))   # every day overflows when halved? still bounded
    fetch(fake)
    assert len(fake.calls) <= 1 + osc.EARNINGS_CHUNK_MAX_REQUESTS


def test_a_complete_page_triggers_no_chunk_requests():
    fake = RangeFinnhub(busy_calendar(per_day=5, extra={"NVDA": 15}))
    cal, ok, cov = fetch(fake)
    assert cal["NVDA"] == 15 and cov["truncated"] is False
    assert len(fake.calls) == 1 and cov["chunk_requests"] == 0
