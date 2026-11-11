"""
Tests for live_price.py and its three display callers (2026-11-11 [REPAIR],
human report: "on some of the new pages we have issues with ticker being just
what they were at the start of the day not what they actually are").

Root cause pinned here: the Analyze panel, the Ticker Dossier and the ETF
Builder took "the price" from daily bars requested with an end of today's date
(00:00 UTC to Alpaca), so the newest bar was the previous session all day. A
snapshot of a NEWER session now supplies the price; the bar helpers are
unchanged (backtest_v2 / regime_engine read them).
"""
import os
import random
from datetime import date, datetime, timedelta, timezone

import pytest

import live_price as LP
import ticker_dossier as TD

SNAP = {
    "latestTrade": {"p": 412.37, "t": "2026-11-11T19:58:01.5Z"},
    "dailyBar": {"t": "2026-11-11T05:00:00Z", "o": 405.0, "h": 413.0, "l": 404.2, "c": 412.1, "v": 1_234_567},
    "prevDailyBar": {"t": "2026-11-10T05:00:00Z", "c": 401.5},
}


def test_parse_prefers_the_latest_trade_and_keeps_todays_bar():
    p = LP.parse_snapshot("abc", SNAP)
    assert p["symbol"] == "ABC" and p["price"] == 412.37 and p["source"] == "latest_trade"
    assert p["session_date"] == "2026-11-11" and p["prev_close"] == 401.5
    assert p["asof"] == "2026-11-11T19:58:01.5Z"
    assert p["bar"] == {"t": "2026-11-11T05:00:00Z", "o": 405.0, "h": 413.0, "l": 404.2, "c": 412.37, "v": 1234567.0}


def test_parse_falls_back_to_the_daily_bar_and_rejects_no_price():
    snap = {**SNAP, "latestTrade": {"p": 0}}
    p = LP.parse_snapshot("ABC", snap)
    assert p["price"] == 412.1 and p["source"] == "daily_bar"
    assert LP.parse_snapshot("ABC", {"latestTrade": {}, "dailyBar": {}}) is None
    assert LP.parse_snapshot("ABC", None) is None
    assert LP.parse_snapshot("ABC", {"latestTrade": {"p": "nan"}}) is None


def test_the_bar_range_always_contains_the_price():
    snap = {**SNAP, "latestTrade": {"p": 420.0, "t": "2026-11-11T20:00:00Z"}}
    b = LP.parse_snapshot("ABC", snap)["bar"]
    assert b["h"] == 420.0 and b["l"] == 404.2 and b["c"] == 420.0


@pytest.mark.parametrize("last_bar,newer", [("2026-11-10", True), ("2026-11-11", False), ("2026-11-12", False), (None, True),
                                            ("2026-11-10T05:00:00Z", True)])
def test_only_a_newer_session_replaces_the_bar_close(last_bar, newer):
    assert LP.is_newer_session(LP.parse_snapshot("ABC", SNAP), last_bar) is newer
    assert LP.is_newer_session(None, last_bar) is False


class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def json(self):
        return self._b


def test_fetch_uses_the_feed_and_never_raises(monkeypatch):
    monkeypatch.setenv("ALPACA_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET", "s")
    seen = {}

    def get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params, headers=headers)
        return _Resp({"ABC": SNAP, "BAD": {"latestTrade": {}}})

    out = LP.fetch_snapshots(["abc", "bad"], get=get, feed="iex")
    assert seen["url"] == LP.SNAPSHOT_URL and seen["params"] == {"symbols": "ABC,BAD", "feed": "iex"}
    assert seen["headers"]["APCA-API-KEY-ID"] == "k"
    assert list(out) == ["ABC"] and out["ABC"]["price"] == 412.37
    assert LP.fetch_snapshots(["ABC"], get=lambda *a, **k: _Resp({}, 403), feed="iex") == {}

    def boom(*a, **k):
        raise OSError("network down")

    assert LP.fetch_snapshots(["ABC"], get=boom, feed="iex") == {}


def test_fetch_without_keys_makes_no_call(monkeypatch):
    monkeypatch.delenv("ALPACA_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET", raising=False)

    def get(*a, **k):
        raise AssertionError("no call without keys")

    assert LP.fetch_snapshots(["ABC"], get=get, feed="iex") == {}


# ── the Ticker Dossier: today's partial bar arrives, the completed bars stay ──
def _bars(n=800, vol=0.03, p0=20.0, seed=1, end=date(2026, 11, 10)):
    rng = random.Random(seed)
    days, d = [], end
    while len(days) < n:                      # the n weekdays ending at `end`
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    out, p = [], p0
    for d in reversed(days):
        p *= 1 + rng.gauss(0.0004, vol)
        out.append({"t": f"{d.isoformat()}T05:00:00Z", "o": p, "h": p * 1.01, "l": p * 0.99, "c": p, "v": 1e6})
    return out


def _io(live):
    bars = {"RKLB": _bars(seed=1), "SPY": _bars(vol=0.01, p0=500, seed=2), "UFO": _bars(vol=0.015, seed=3)}
    return {
        "bars": lambda syms, a, b: ({k: bars[k] for k in syms if k in bars}, set()),
        "live": live,
        "regime": lambda: {"as_of": "2026-11-10", "composite_regime": "RISK_ON", "exposure_multiplier": 1.0, "inputs": {}},
        "sector": lambda s: None, "edgar": lambda s: None, "news": lambda s: [],
        "chain": lambda s, p: [], "earnings": lambda s: {"status": "unavailable", "reason": "no key"},
        "forecast_store": lambda: ([], [], {}), "shadow": lambda: None,
        "record": lambda rows: len(rows), "ledger_report": lambda t: {"all": {}, "ticker": {}},
    }, bars


NOW = datetime(2026, 11, 11, 19, 30, tzinfo=timezone.utc)   # 14:30 ET, session open


def test_dossier_price_is_the_live_price_mid_session():
    seen = []
    live = LP.parse_snapshot("RKLB", {**SNAP, "latestTrade": {"p": 31.25, "t": "2026-11-11T19:29:59Z"}})

    def live_fn(sym, last_bar_date):
        seen.append((sym, last_bar_date))
        return live if LP.is_newer_session(live, last_bar_date) else None

    io, bars = _io(live_fn)
    o = TD.build("RKLB", now=NOW, io=io)
    st = o["stock"]
    assert seen == [("RKLB", "2026-11-10")], "asked with the newest bar's date"
    assert st["price"] == 31.25 and st["price_day"] == "2026-11-11" and st["price_is_partial_session"] is True
    assert st["base_date"] == "2026-11-10", "ranges and indicators still stand on the completed session"
    assert o["plan"]["stop"] < 31.25 < o["plan"]["target"], "the plan is anchored on the live price"


def test_dossier_without_a_newer_snapshot_keeps_the_last_close():
    io, bars = _io(lambda sym, d: None)
    o = TD.build("RKLB", now=NOW, io=io)
    assert o["stock"]["price"] == round(bars["RKLB"][-1]["c"], 4)
    assert o["stock"]["price_is_partial_session"] is False


def test_dossier_live_failure_is_reported_not_fatal():
    def live_fn(sym, d):
        raise RuntimeError("snapshot timeout")

    io, _ = _io(live_fn)
    o = TD.build("RKLB", now=NOW, io=io)
    assert o["stock"]["status"] == "live" and "snapshot timeout" in o["errors"]["live_price"]
    assert o["stock"]["price_is_partial_session"] is False


def test_dossier_default_io_wires_the_live_source():
    assert TD.DEFAULT_IO["live"] is TD._io_live


# ── Analyze and the ETF Builder: the wiring (both run network-heavy engines) ──
_HERE = os.path.dirname(os.path.abspath(__file__))


def _src(name):
    return open(os.path.join(_HERE, name), encoding="utf-8").read()


def test_analyze_overlays_the_live_price_and_measures_the_day_from_the_last_close():
    s = _src("analyze.py")
    assert "_live = live_price.latest(ticker_symbol, _last_bar_date)" in s
    assert "spot = float(_live[\"price\"])" in s
    assert "prev_close = _last_close" in s, "with a live price the day's change is vs the last completed close"
    assert "**_price_meta," in s, "the result says where its price came from and when"
    # the shared bar helper is unchanged: backtest_v2 reads it
    assert '"end":       end.strftime("%Y-%m-%dT00:00:00Z"),' in s


def test_etf_builder_overlays_the_live_price():
    s = _src("etf_analyzer.py")
    assert 'live = live_price.latest(ticker, etf_pkg.get("last_date"))' in s
    assert '"current_price": etf_price,\n        **price_meta,' in s
    assert '"last_date": str(hist[-1][0])[:10],' in _src("etf_data_sources.py")
