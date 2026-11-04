"""
[REPAIR] 2026-11-04: the Tier-1 CSP earnings guard fails CLOSED when the
window calendar cannot vouch for a candidate (KB #45 path (c), recurrence).

LIVE: the tier engine sold IREN261211P00018000 (1 @ $1.68) at
2026-11-04 18:41:56Z ("T1 CSP harvest"). IREN reports 2026-11-05 after the
close (Nasdaq). Same shape as the BAC put of 2026-10-13 (KB #45). A
shadow-log read of 10-29 -> 11-04 found 6 of 31 proposed names 0-3 days
before earnings.

ROOT CAUSE: one unfiltered Finnhub /calendar/earnings request per window
([today, today+47] for T1, [today, today+30] for csp_universe Layer 2).
Finnhub answers ONE page of at most 1,500 rows, NEWEST dates first, so a
busy window silently loses its EARLIEST dates. With from=today those are
the next few days, which is where the guards look. A live October month
held 3,166 releases, about two pages. The cut response still read as
lookup_ok=True, and tier1_csp_core passed cal.get(ticker) -> None to
regime_engine.csp_verdict, which reads None as "no earnings". The guard
was active and blocking live (regime_counterfactual: 2-8 earnings blocks a
day, all on names reporting later in the window). It was blind only where
the page was cut.

FIX: the window fetch records its coverage (rows, window, earliest/latest
date returned, truncated). When the calendar is not authoritative (lookup
failed, or truncated: a full page or no row in the window's first 4 days),
an ABSENT candidate is confirmed with Finnhub's per-symbol calendar before
it may be sold. If that also fails, the candidate is refused as
rejected_regime_earnings_unverified (shadow record + counterfactual row).

Mocks follow the existing tests: options_scanner.requests.get is faked
(test_csp_earnings_gate_fail_closed.py), and the tier engine is driven the
way test_regime_options.py's run_t1 drives it. Here the REAL
_t1_earnings_calendar -> options_scanner fetch path runs, so the cut page
reaches the guard exactly as it does live.
"""
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

import options_scanner
import regime_engine as re_
import tiered_strategy as ts

UNIVERSE = ["IREN", "AMD", "AAPL", "MSFT", "NVDA"]
WINDOW = ts.T1_TARGET_DTE_MAX + int(re_.OPTIONS_PARAMS["earnings_buffer_days"])   # 45 + 2 = 47


def _d(n):
    """ISO date n days from today (the fetchers count from datetime.now())."""
    return (datetime.now().date() + timedelta(days=n)).strftime("%Y-%m-%d")


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


def cut_page(rows=1500, first_day=8, last_day=WINDOW, extra=None):
    """What Finnhub returns for a busy window: one full page, newest dates
    first, the near end (today .. first_day-1) cut off. `extra` maps
    symbol -> day for real names placed inside the page."""
    out = []
    span = last_day - first_day
    for i in range(rows):
        out.append({"symbol": f"Z{i:04d}", "date": _d(last_day - (i * (span + 1)) // rows), "hour": "amc"})
    for j, (sym, day) in enumerate(sorted((extra or {}).items())):
        out[j] = {"symbol": sym, "date": _d(day), "hour": "bmo"}
    out.sort(key=lambda r: r["date"], reverse=True)
    return out


class FakeFinnhub:
    """requests.get stand-in: the all-company window request vs per-symbol
    requests (symbol= filter), each with its own status / body / exception."""

    def __init__(self, window_rows=None, window_status=200, window_raise=False,
                 symbols=None, symbol_status=200, symbol_raise=False):
        self.window_rows = window_rows if window_rows is not None else []
        self.window_status = window_status
        self.window_raise = window_raise
        self.symbols = symbols or {}
        self.symbol_status = symbol_status
        self.symbol_raise = symbol_raise
        self.calls = []

    def __call__(self, url, params=None, timeout=None, **kw):
        params = dict(params or {})
        self.calls.append(params)
        if "symbol" in params:
            if self.symbol_raise:
                raise ConnectionError("finnhub unreachable")
            if self.symbol_status != 200:
                return _Resp(self.symbol_status, {"error": "API limit reached"})
            return _Resp(200, {"earningsCalendar": list(self.symbols.get(params["symbol"], []))})
        if self.window_raise:
            raise ConnectionError("finnhub unreachable")
        if self.window_status != 200:
            return _Resp(self.window_status, {"error": "API limit reached"})
        return _Resp(200, {"earningsCalendar": list(self.window_rows)})

    @property
    def symbol_calls(self):
        return [c["symbol"] for c in self.calls if "symbol" in c]

    @property
    def window_calls(self):
        """The primary all-company window requests (full [today, today+WINDOW]
        range). Chunk completion requests (narrower ranges) are counted by
        chunk_calls, so 'one window request' stays an exact assertion."""
        return [c for c in self.calls if "symbol" not in c
                and (c.get("from"), c.get("to")) == (_d(0), _d(WINDOW))]

    @property
    def chunk_calls(self):
        return [c for c in self.calls if "symbol" not in c
                and (c.get("from"), c.get("to")) != (_d(0), _d(WINDOW))]


def state(**over):
    base = {"as_of": "2026-11-04", "composite_regime": "CALM_BULL", "vol_state": "NORMAL",
            "stress_flags": {"vrp_negative": False, "vix_vix3m_inverted": False, "vrp_points": 3.1, "vix_vix3m": 0.9},
            "ebp": {"ebp": -0.2}}
    base.update(over)
    return base


def ctx():
    return ts.TierContext(equity=100_000.0, peak_equity=100_000.0, buying_power=100_000.0,
                          positions=[], vxx_ratio=1.0, spy_vs_ma50=1.0)


def _reset():
    ts._T1_EARNINGS_CACHE.update(at=0.0, days_ahead=0, cal={}, ok=False)
    for name in ("_T1_SYMBOL_EARNINGS_CACHE",):
        getattr(ts, name, {}).clear()
    re_._SHADOW_ONCE.update(day=None, keys=set())


@pytest.fixture(autouse=True)
def _clean_caches():
    _reset()
    yield
    _reset()


def run_t1(fake, flag="all", universe=UNIVERSE, st=None):
    """tier1_csp_core with the engine on and the REAL calendar fetch path:
    only requests.get (Finnhub) and the shadow/counterfactual sinks are faked."""
    shadow, cf_rows = [], []

    def _cf(kind, key, legacy, engine, source, **kw):
        if kind == "options":
            cf_rows.append({"ticker": key, "decision": engine.get("decision"),
                            "shadow_decision": kw.get("shadow_decision")})
        return False

    with patch("tiered_strategy._get_t1_universe", return_value=list(universe)), \
         patch("csp_universe._load_layer2_cache", return_value={}), \
         patch("regime_engine.engine_allowed", return_value=True), \
         patch("regime_engine.cached_state", return_value=st or state()), \
         patch("regime_engine.log_counterfactual", side_effect=_cf), \
         patch("tiered_strategy.log_csp_gate_shadow", side_effect=lambda rej: shadow.extend(rej) or {}), \
         patch.object(options_scanner.requests, "get", fake), \
         patch.dict(os.environ, {"VOLTRADE_REGIME_OPTIONS": flag, "VOLTRADE_REGIME_SIZING": "off"}):
        actions = ts.tier1_csp_core(ctx())
    return actions, shadow, cf_rows


# ── (a) the live IREN shape: a cut page omits IREN, the per-symbol calendar knows ──

def test_iren_absent_from_a_cut_page_is_confirmed_per_symbol_and_blocked():
    fake = FakeFinnhub(window_rows=cut_page(extra={"NVDA": 15}),
                       symbols={"IREN": [{"symbol": "IREN", "date": _d(1), "hour": "amc"}]})
    actions, shadow, cf_rows = run_t1(fake)
    tickers = [a.ticker for a in actions]
    assert "IREN" not in tickers                       # unmodified code: IREN is proposed FIRST
    assert tickers == ["AMD", "AAPL", "MSFT"]          # NVDA blocked by its own row at +15; slots kept
    by_ticker = {r["ticker"]: r for r in shadow}
    assert by_ticker["IREN"]["decision"] == "rejected_regime_earnings"
    assert "earnings in 1 days" in by_ticker["IREN"]["reason"]
    assert by_ticker["NVDA"]["decision"] == "rejected_regime_earnings"
    assert {"ticker": "IREN", "decision": "block", "shadow_decision": "rejected_regime_earnings"} in cf_rows
    # one per-symbol call per absent candidate that reached the gate; none for a name the page holds
    assert fake.symbol_calls == ["IREN", "AMD", "AAPL", "MSFT"]
    assert len(fake.window_calls) == 1


def test_per_symbol_request_carries_the_symbol_and_the_guard_window():
    fake = FakeFinnhub(window_rows=cut_page(), symbols={"IREN": [{"symbol": "IREN", "date": _d(1)}]})
    run_t1(fake, universe=["IREN"])
    (call,) = [c for c in fake.calls if "symbol" in c]
    assert (call["symbol"], call["from"], call["to"]) == ("IREN", _d(0), _d(WINDOW))


# ── (b) both lookups fail -> refused (fail closed), with a distinct label ──

@pytest.mark.parametrize("window", [dict(window_status=429), dict(window_raise=True)])
@pytest.mark.parametrize("symbol", [dict(symbol_status=429), dict(symbol_raise=True)])
def test_window_and_symbol_lookup_both_failing_refuses_every_absent_name(window, symbol):
    fake = FakeFinnhub(**window, **symbol)
    actions, shadow, cf_rows = run_t1(fake)
    assert actions == []                               # unmodified code: 3 names sold blind
    assert {r["decision"] for r in shadow} == {"rejected_regime_earnings_unverified"}
    assert [r["ticker"] for r in shadow] == UNIVERSE[:3]   # bounded like every other bucket (3 per scan)
    assert "unverified" in shadow[0]["reason"].lower()
    assert all(r["decision"] == "block" for r in cf_rows if r["ticker"] in UNIVERSE[:3])
    assert {r["shadow_decision"] for r in cf_rows} == {"rejected_regime_earnings_unverified"}
    assert len(fake.symbol_calls) == 1                 # the first failure ends this scan's lookups


def test_cut_page_and_symbol_failure_fails_closed():
    fake = FakeFinnhub(window_rows=cut_page(extra={"NVDA": 15}), symbol_raise=True)
    actions, shadow, _ = run_t1(fake)
    assert actions == []
    decisions = {r["ticker"]: r["decision"] for r in shadow}
    assert decisions == {"IREN": "rejected_regime_earnings_unverified",
                         "AMD": "rejected_regime_earnings_unverified",
                         "AAPL": "rejected_regime_earnings_unverified",
                         "NVDA": "rejected_regime_earnings"}


# ── (c) unchanged where the facts are known ──

def test_a_report_60_days_out_is_allowed_as_before():
    fake = FakeFinnhub(window_rows=cut_page(),
                       symbols={"IREN": [{"symbol": "IREN", "date": _d(60)}]})
    actions, shadow, _ = run_t1(fake, universe=["IREN"])
    assert [a.ticker for a in actions] == ["IREN"]
    assert shadow == []


def test_complete_calendar_makes_no_per_symbol_calls():
    rows = [{"symbol": "NVDA", "date": _d(15)}] + [{"symbol": f"Z{i:03d}", "date": _d(i % 40)} for i in range(300)]
    fake = FakeFinnhub(window_rows=rows)
    actions, shadow, _ = run_t1(fake)
    assert [a.ticker for a in actions] == ["IREN", "AMD", "AAPL"]   # absent from a COMPLETE page = no report
    assert [(r["ticker"], r["decision"]) for r in shadow] == [("NVDA", "rejected_regime_earnings")]
    assert fake.symbol_calls == []


# ── (d) root cause: the cut page read as a complete, successful lookup ──

def test_root_cause_a_full_page_with_its_near_end_cut_is_flagged_truncated():
    fake = FakeFinnhub(window_rows=cut_page())
    with patch.object(options_scanner.requests, "get", fake):
        cal, ok = options_scanner._fetch_earnings_calendar_with_status(days_ahead=WINDOW)
    assert ok is True and "IREN" not in cal            # the fetch "succeeds", silently missing the next week
    cov = options_scanner.earnings_calendar_coverage(WINDOW)
    assert cov["truncated"] is True
    assert (cov["at_row_cap"], cov["near_edge_gap"]) == (True, True)
    assert cov["calendar_entries"] == options_scanner.EARNINGS_CALENDAR_ROW_CAP == 1500
    assert (cov["window_from"], cov["window_to"]) == (_d(0), _d(WINDOW))
    assert (cov["earliest_date_returned"], cov["latest_date_returned"]) == (_d(8), _d(WINDOW))
    assert cov["lookup_ok"] is True and cov["http_status"] == 200


@pytest.mark.parametrize("first_day,rows,truncated", [
    (0, 300, False),     # a normal page that starts today
    (3, 300, False),     # a Friday window with no Friday rows: Monday is day 3
    (4, 300, True),      # nothing in the window's first 4 days: the near end was cut (cap-agnostic)
    (6, 900, True),      # below the 1,500 cap, near end still empty -> a smaller page is cut too
    (0, 1500, True),     # a full page is cut even when it reaches today
])
def test_truncation_signal(first_day, rows, truncated):
    fake = FakeFinnhub(window_rows=cut_page(rows=rows, first_day=first_day))
    with patch.object(options_scanner.requests, "get", fake):
        options_scanner._fetch_earnings_calendar_with_status(days_ahead=WINDOW)
    assert options_scanner.earnings_calendar_coverage(WINDOW)["truncated"] is truncated


def test_an_empty_page_for_a_47_day_window_is_not_authoritative():
    fake = FakeFinnhub(window_rows=[])
    with patch.object(options_scanner.requests, "get", fake):
        cal, ok = options_scanner._fetch_earnings_calendar_with_status(days_ahead=WINDOW)
    assert (cal, ok) == ({}, True)                     # unchanged contract for csp_universe / existing callers
    cov = options_scanner.earnings_calendar_coverage(WINDOW)
    assert cov["calendar_entries"] == 0 and cov["truncated"] is True


def test_root_cause_end_to_end_the_cut_page_used_to_sell_the_reporting_name():
    """The exact live chain: real fetch -> real T1 cache -> real csp_verdict.
    IREN reports tomorrow, the page is cut at day 8, IREN is not on it."""
    fake = FakeFinnhub(window_rows=cut_page(),
                       symbols={"IREN": [{"symbol": "IREN", "date": _d(1), "hour": "amc"}]})
    actions, shadow, _ = run_t1(fake, universe=["IREN"])
    assert actions == []
    assert [(r["ticker"], r["decision"]) for r in shadow] == [("IREN", "rejected_regime_earnings")]


def test_coverage_is_recorded_on_every_failure_path():
    for fake in (FakeFinnhub(window_status=429), FakeFinnhub(window_raise=True)):
        with patch.object(options_scanner.requests, "get", fake):
            cal, ok = options_scanner._fetch_earnings_calendar_with_status(days_ahead=WINDOW)
        cov = options_scanner.earnings_calendar_coverage(WINDOW)
        assert ok is False and cov["lookup_ok"] is False
        assert cov["window_from"] == _d(0) and cov["window_to"] == _d(WINDOW)


# ── per-symbol fetch ──

def test_symbol_fetch_returns_the_nearest_report_inside_the_window():
    fake = FakeFinnhub(symbols={"IREN": [{"symbol": "IREN", "date": _d(30)},
                                         {"symbol": "IREN", "date": _d(1)},
                                         {"symbol": "IREN", "date": _d(-3)},     # already past
                                         {"symbol": "IREN", "date": _d(60)},     # outside the window
                                         {"symbol": "OTHER", "date": _d(0)}]})   # never another name's row
    with patch.object(options_scanner.requests, "get", fake):
        assert options_scanner._fetch_symbol_earnings_with_status("iren", days_ahead=WINDOW) == (1, True)
        assert options_scanner._fetch_symbol_earnings_with_status("AMD", days_ahead=WINDOW) == (None, True)


@pytest.mark.parametrize("fake", [FakeFinnhub(symbol_status=429), FakeFinnhub(symbol_raise=True)])
def test_symbol_fetch_failure_is_not_read_as_no_report(fake):
    with patch.object(options_scanner.requests, "get", fake):
        assert options_scanner._fetch_symbol_earnings_with_status("IREN", days_ahead=WINDOW) == (None, False)


def test_symbol_fetch_missing_calendar_key_is_a_failure():
    class _BadBody(FakeFinnhub):
        def __call__(self, url, params=None, timeout=None, **kw):
            self.calls.append(dict(params or {}))
            return _Resp(200, {"error": "bad token"})
    with patch.object(options_scanner.requests, "get", _BadBody()):
        assert options_scanner._fetch_symbol_earnings_with_status("IREN", days_ahead=WINDOW) == (None, False)


# ── the verdict ──

def test_verdict_blocks_an_unverified_name_and_only_an_unverified_one():
    v = re_.csp_verdict(state(), {"days_to_earnings": None, "earnings_unverified": True, "dte_max": 45})
    assert v["flags"]["earnings"] == 1 and v["would_block"] == ["earnings"]
    assert "unverified" in v["reasons"]["earnings"].lower()
    known = re_.csp_verdict(state(), {"days_to_earnings": 60, "earnings_unverified": True, "dte_max": 45})
    assert known["flags"]["earnings"] == 0                       # a confirmed date decides
    assert re_.csp_verdict(state(), {"days_to_earnings": None, "dte_max": 45})["flags"]["earnings"] == 0


# ── bounds: budget, breaker, cache ──

def test_per_scan_lookup_budget_then_fail_closed():
    names = [f"N{i:02d}" for i in range(12)]
    fake = FakeFinnhub(window_rows=cut_page(), symbols={n: [{"symbol": n, "date": _d(1)}] for n in names})
    actions, shadow, _ = run_t1(fake, universe=names)
    assert actions == []
    assert len(fake.symbol_calls) == ts.T1_EARNINGS_CONFIRM_MAX
    decisions = [r["decision"] for r in shadow]
    assert decisions.count("rejected_regime_earnings") == 3
    assert decisions.count("rejected_regime_earnings_unverified") == 3


def test_per_symbol_answers_are_cached_across_scans():
    fake = FakeFinnhub(window_rows=cut_page())
    first, _, _ = run_t1(fake)
    calls_after_first = list(fake.symbol_calls)
    second, _, _ = run_t1(fake)
    assert [a.ticker for a in second] == [a.ticker for a in first] == ["IREN", "AMD", "AAPL"]
    assert fake.symbol_calls == calls_after_first          # second scan: cache hits only
    assert len(fake.window_calls) == 1                     # the window calendar is cached too (15 min)


def test_a_failed_lookup_is_not_cached():
    run_t1(FakeFinnhub(window_rows=cut_page(), symbol_raise=True), universe=["IREN"])
    fake = FakeFinnhub(window_rows=cut_page())
    actions, _, _ = run_t1(fake, universe=["IREN"])
    assert [a.ticker for a in actions] == ["IREN"] and fake.symbol_calls == ["IREN"]


# ── the flag still rolls the gate back (shadow) ──

def test_shadow_mode_proposes_and_logs_would_reject_unverified():
    fake = FakeFinnhub(window_status=429, symbol_status=429)
    actions, shadow, _ = run_t1(fake, flag="")
    assert [a.ticker for a in actions] == UNIVERSE[:3]
    assert {r["decision"] for r in shadow} == {"would_reject_regime_earnings_unverified"}


# ── metadata + diagnostic ──

def test_action_metadata_says_where_the_earnings_answer_came_from():
    actions, _, _ = run_t1(FakeFinnhub(window_rows=cut_page()), universe=["AMD"])
    g = actions[0].metadata["regime_guards"]
    assert (g["earnings_source"], g["earnings_calendar_truncated"], g["earnings_lookup_ok"]) == ("symbol", True, True)
    _reset()
    rows = [{"symbol": f"Z{i:03d}", "date": _d(i % 40)} for i in range(300)]
    actions, _, _ = run_t1(FakeFinnhub(window_rows=rows), universe=["AMD"])
    g = actions[0].metadata["regime_guards"]
    assert (g["earnings_source"], g["earnings_calendar_truncated"]) == ("calendar", False)


def test_coverage_diagnostic_after_a_scan():
    fake = FakeFinnhub(window_rows=cut_page(extra={"NVDA": 15}),
                       symbols={"IREN": [{"symbol": "IREN", "date": _d(1)}]})
    run_t1(fake)
    cov = ts.get_t1_earnings_coverage()
    assert cov["engine_active"] is True and cov["window_days"] == WINDOW
    assert (cov["calendar_entries"], cov["truncated"], cov["lookup_ok"], cov["authoritative"]) == (1500, True, True, False)
    assert (cov["window_from"], cov["window_to"]) == (_d(0), _d(WINDOW))
    assert (cov["earliest_date_returned"], cov["latest_date_returned"]) == (_d(8), _d(WINDOW))
    assert cov["symbol_lookups"] == 4 and cov["symbol_lookup_failures"] == 0
    assert cov["confirmed_in_window"] == 1 and cov["earnings_unverified"] == 0
    assert str(WINDOW) in cov["windows"] and cov["windows"][str(WINDOW)]["truncated"] is True


def test_run_tiers_surfaces_the_coverage_in_tier_timings():
    """Same wiring as csp_layer2_prefetch: tier_timings -> bot_engine's
    tier_engine_breakdown phase -> the timings file -> /api/diag/timings."""
    with patch("tiered_strategy._get_t1_universe", return_value=["AAPL", "MSFT"]):
        result = ts.TieredStrategy().run_tiers(ctx())
    cov = result["tier_timings"]["t1_earnings_coverage"]
    for key in ("calendar_entries", "window_from", "window_to", "earliest_date_returned",
                "latest_date_returned", "truncated", "lookup_ok", "engine_active"):
        assert key in cov
    assert "actions" in result and "tier_stats" in result


def test_t1_calendar_still_unpacks_and_compares_as_a_pair():
    fake = FakeFinnhub(window_rows=cut_page())
    with patch.object(options_scanner.requests, "get", fake):
        lookup = ts._t1_earnings_calendar(WINDOW)
    cal, ok = lookup
    assert ok is True and lookup == (cal, True)
    assert lookup.coverage["truncated"] is True
