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

Never raises: any failure returns {} and the caller keeps its bar close.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Iterable, Optional

log = logging.getLogger("live_price")

SNAPSHOT_URL = "https://data.alpaca.markets/v2/stocks/snapshots"
MAX_SYMBOLS = 50


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


def fetch_snapshots(symbols: Iterable[str], timeout: float = 5.0,
                    get: Optional[Callable[..., Any]] = None, feed: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """{SYMBOL: parse_snapshot(...)} for up to MAX_SYMBOLS symbols. Never raises."""
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
        r = get(SNAPSHOT_URL, params={"symbols": ",".join(syms), "feed": feed},
                headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}, timeout=timeout)
        if getattr(r, "status_code", 200) != 200:
            log.warning("live_price: snapshots HTTP %s for %s", getattr(r, "status_code", "?"), ",".join(syms))
            return {}
        body = r.json()
    except Exception as e:  # network, JSON, feed probe — the caller keeps its bar close
        log.warning("live_price: snapshots failed for %s: %s", ",".join(syms), str(e)[:160])
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for sym, snap in (body.items() if isinstance(body, dict) else []):
        parsed = parse_snapshot(sym, snap)
        if parsed:
            out[str(sym).upper()] = parsed
    return out


def latest(symbol: str, last_bar_date: Optional[str], **kw: Any) -> Optional[Dict[str, Any]]:
    """The snapshot for `symbol` when it is newer than `last_bar_date`, else None."""
    s = fetch_snapshots([symbol], **kw).get(str(symbol).upper())
    return s if is_newer_session(s, last_bar_date) else None
