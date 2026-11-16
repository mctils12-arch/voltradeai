"""
live_price.py — the CURRENT price of a symbol for display paths, from Alpaca's
snapshot endpoint (2026-11-11 [REPAIR], human report: "on some of the new pages
we have issues with ticker being just what they were at the start of the day
not what they actually are").

WHY: three display paths take "the price" from daily-bar history requested with
an end of today's date, which Alpaca reads as 00:00 UTC — so the newest bar is
the PREVIOUS session and the "current" price is yesterday's close all day:
  - analyze.analyze_ticker (_fetch_alpaca_bars: end = today T00:00:00Z) — the
    Analyze panel of the Trade page;
  - ticker_dossier.build (bars fetched with end = today's date) — the Ticker
    Dossier page, whose trade plan is "from the live price" and which already
    has a partial-session flag that could never turn on;
  - etf_analyzer (etf_data_sources.compute_returns_for_symbols, same end) — the
    ETF Builder's current price.
The bar helpers themselves stay as they are: backtest_v2, regime_engine and
flows_model read them, and a partial bar in a backtest or a regime read is a
different (measurement) change. This module only supplies the newer price, and
callers use it only when its session is NEWER than their last bar.

Snapshot fields used: latestTrade {p, t}, dailyBar {t, o, h, l, c, v},
prevDailyBar {c}. The feed is alpaca_feed.data_feed() (SIP, IEX, or the
15-minute-delayed SIP the account is entitled to), the same one the site's
/api/market/snapshots route serves the watchlist from.

REAL-TIME (2026-11-12, post-open check of the fix above): this account's plan
rejects recent SIP data, so data_feed() is "delayed_sip" and every price was
15 minutes old (SPY's latest trade at 10:08 ET read 09:53). The plan does
include IEX in real time. Under delayed_sip, fetch_snapshots() also reads the
IEX tape and freshest() takes its trade when it is a NEWER REGULAR-HOURS print:
the consolidated snapshot stays the base (its daily bar's volume is the whole
tape's), only the price moves. Every result names its tape (feed).

Never raises: any failure returns {} and the caller keeps its bar close.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger("live_price")

SNAPSHOT_URL = "https://data.alpaca.markets/v2/stocks/snapshots"
MAX_SYMBOLS = 50
DELAYED_FEED = "delayed_sip"   # alpaca_feed's fallback: the consolidated tape, 15 minutes late
REALTIME_FEED = "iex"          # the real-time tape every plan carries (one exchange's prints)
_ET = ZoneInfo("America/New_York")
_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})$")


def _pos(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 and f == f and f != float("inf") else None


def parse_snapshot(symbol: str, snap: Any) -> Optional[Dict[str, Any]]:
    """One Alpaca snapshot -> {symbol, price, asof, source, session_date,
    prev_close, bar}; None when it carries no usable price. Pure.

    price: the latest trade when there is one, else the daily bar's close.
    session_date: the daily bar's date (YYYY-MM-DD) — the session the price
    belongs to. bar: today's partial daily bar {t, o, h, l, c, v} (c = price),
    for callers that keep a bar series."""
    if not isinstance(snap, dict):
        return None
    lt = snap.get("latestTrade") if isinstance(snap.get("latestTrade"), dict) else {}
    db = snap.get("dailyBar") if isinstance(snap.get("dailyBar"), dict) else {}
    pdb = snap.get("prevDailyBar") if isinstance(snap.get("prevDailyBar"), dict) else {}
    trade_px, bar_c = _pos(lt.get("p")), _pos(db.get("c"))
    price = trade_px if trade_px is not None else bar_c
    if price is None:
        return None
    session = str(db.get("t") or "")[:10] or None
    bar = None
    if session and bar_c is not None:
        o, h, lo = _pos(db.get("o")) or bar_c, _pos(db.get("h")) or bar_c, _pos(db.get("l")) or bar_c
        bar = {"t": str(db.get("t")), "o": o, "h": max(h, price), "l": min(lo, price), "c": price,
               "v": float(db.get("v") or 0)}
    return {
        "symbol": str(symbol).upper(),
        "price": price,
        "asof": str(lt.get("t") or db.get("t") or "") or None,
        "source": "latest_trade" if trade_px is not None else "daily_bar",
        "session_date": session,
        "prev_close": _pos(pdb.get("c")),
        "bar": bar,
    }


def is_newer_session(parsed: Optional[Dict[str, Any]], last_bar_date: Optional[str]) -> bool:
    """True when the snapshot's session is after the caller's newest bar —
    the only case in which its price replaces the bar close. A snapshot of
    the SAME session (weekends, holidays, after the history already has
    today's bar) changes nothing. Pure."""
    if not parsed or not parsed.get("session_date"):
        return False
    if not last_bar_date:
        return True
    return str(parsed["session_date"]) > str(last_bar_date)[:10]


def trade_time(value: Any) -> Optional[datetime]:
    """An RFC 3339 trade time (Alpaca sends nanoseconds) as an aware datetime;
    None when absent or unparseable. Pure."""
    m = _TS.match(str(value or "").strip())
    if not m:
        return None
    try:
        return datetime.fromisoformat(m.group(1) + (m.group(2) or "")[:7]
                                      + ("+00:00" if m.group(3) == "Z" else m.group(3)))
    except ValueError:
        return None


def in_regular_hours(t: Optional[datetime]) -> bool:
    """09:30-16:00 ET on a weekday. Pure."""
    if t is None:
        return False
    et = t.astimezone(_ET)
    return et.weekday() < 5 and 9 * 60 + 30 <= et.hour * 60 + et.minute < 16 * 60


def freshest(delayed: Optional[Dict[str, Any]], realtime: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The display snapshot for one symbol from the 15-minute-delayed
    consolidated tape and the real-time IEX tape (parse_snapshot() results or
    None). Pure.

    The IEX trade is used only when it is a regular-hours print NEWER than the
    consolidated one: a pre-market or after-hours IEX print never replaces a
    session's price, and an IEX print older than the delayed tape's (a thinly
    traded name) never replaces it either.
      - same session: the consolidated snapshot with the IEX price, time and
        tape; its bar's close is the IEX price, its range widened to include
        it, its volume (the whole tape's) kept;
      - IEX already on a NEWER session (the open's first ~15 minutes, before
        the delayed tape has today's bar): the IEX snapshot, tape "iex".
    Otherwise the consolidated snapshot, tape "delayed_sip"."""
    if delayed is None and realtime is None:
        return None
    rt_t = trade_time((realtime or {}).get("asof")) if realtime and realtime.get("source") == "latest_trade" else None
    usable = rt_t is not None and in_regular_hours(rt_t)
    if delayed is None:
        return {**realtime, "feed": REALTIME_FEED} if usable else None
    base = {**delayed, "feed": DELAYED_FEED, "source": f"{delayed.get('source')}, 15-min delayed"}
    if not usable:
        return base
    d_t = trade_time(delayed.get("asof"))
    if d_t is not None and rt_t <= d_t:
        return base
    d_session, rt_session = str(delayed.get("session_date") or ""), str(realtime.get("session_date") or "")
    if rt_session and rt_session > d_session:
        return {**realtime, "feed": REALTIME_FEED, "source": "latest_trade, iex real-time"}
    if rt_session != d_session:
        return base
    price = float(realtime["price"])
    out = {**base, "price": price, "asof": realtime.get("asof"), "feed": REALTIME_FEED,
           "source": "latest_trade, iex real-time"}
    if isinstance(delayed.get("bar"), dict):
        bar = dict(delayed["bar"])
        bar.update(c=price, h=max(float(bar.get("h") or price), price), l=min(float(bar.get("l") or price), price))
        out["bar"] = bar
    return out


def _fetch_parsed(syms: list, feed: str, key: str, secret: str, timeout: float,
                  get: Callable[..., Any]) -> Dict[str, Dict[str, Any]]:
    """{SYMBOL: parse_snapshot(...)} from one feed. Never raises."""
    try:
        r = get(SNAPSHOT_URL, params={"symbols": ",".join(syms), "feed": feed},
                headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}, timeout=timeout)
        if getattr(r, "status_code", 200) != 200:
            log.warning("live_price: snapshots HTTP %s for %s (feed %s)", getattr(r, "status_code", "?"), ",".join(syms), feed)
            return {}
        body = r.json()
    except Exception as e:  # network, JSON — the caller keeps what it has
        log.warning("live_price: snapshots failed for %s (feed %s): %s", ",".join(syms), feed, str(e)[:160])
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for sym, snap in (body.items() if isinstance(body, dict) else []):
        parsed = parse_snapshot(sym, snap)
        if parsed:
            out[str(sym).upper()] = parsed
    return out


def fetch_snapshots(symbols: Iterable[str], timeout: float = 5.0,
                    get: Optional[Callable[..., Any]] = None, feed: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """{SYMBOL: parse_snapshot(...)} for up to MAX_SYMBOLS symbols. Never raises.
    Under the delayed consolidated feed the IEX tape is read too and each
    symbol's price is freshest() of the two."""
    syms = [str(s).upper().strip() for s in symbols if s and str(s).strip()][:MAX_SYMBOLS]
    key, secret = os.environ.get("ALPACA_KEY", ""), os.environ.get("ALPACA_SECRET", "")
    if not syms or not key or not secret:
        return {}
    try:
        if feed is None:
            from alpaca_feed import data_feed
            feed = data_feed()
        if get is None:
            import requests
            get = requests.get
    except Exception as e:  # the feed probe — the caller keeps its bar close
        log.warning("live_price: no feed for %s: %s", ",".join(syms), str(e)[:160])
        return {}
    out = _fetch_parsed(syms, feed, key, secret, timeout, get)
    if feed != DELAYED_FEED:
        return out
    realtime = _fetch_parsed(syms, REALTIME_FEED, key, secret, timeout, get)
    merged = {sym: freshest(out.get(sym), realtime.get(sym)) for sym in syms}
    return {sym: snap for sym, snap in merged.items() if snap}


def latest(symbol: str, last_bar_date: Optional[str], **kw: Any) -> Optional[Dict[str, Any]]:
    """The snapshot for `symbol` when it is newer than `last_bar_date`, else None."""
    s = fetch_snapshots([symbol], **kw).get(str(symbol).upper())
    return s if is_newer_session(s, last_bar_date) else None
