"""
VolTradeAI — the broker clock and the order-hours gate (2026-11-05)
===================================================================
Human-directed: "Market-hours check: apply the Alpaca clock/calendar check
(is_open, holidays, half-day early close) to every order path, including the
old floor (so a rollback is safe), options exits and the backstop. Make the
4am-8pm scan unable to send orders when the market is closed."

Why. The Tier 2 scan runs 04:00-20:00 ET every day, weekends and holidays
included (server/bot.ts scheduleTier2), and the overnight research runs the
full scan at 20:00-22:00 ET. The engine's own hours check
(bot_engine._is_regular_hours) read only the time of day, and several order
paths had no check at all (the GLD defensive floor, the third leg, the
convexity overlay, intraday shorts). A DAY order sent while the market is
closed is queued by Alpaca for the next opening print.

ONE cached reader of Alpaca GET /v2/clock {timestamp, is_open, next_open,
next_close} serves every Python order path. Alpaca's clock already reflects
exchange holidays and half-day early closes. The policy:
  * the broker says open   -> orders allowed;
  * the broker says closed -> refused: check_order_allowed raises OrderBlocked;
  * the read failed        -> extrapolate from the last good reading: it said
    open and its next_close is still ahead -> open; it said closed and its
    next_open is still ahead -> closed; it said open, its next_close has
    passed and its next_open is still ahead -> closed (the broker's own
    schedule); otherwise unknown;
  * unknown (no usable reading at all) -> allowed, "clock unknown": a clock
    outage never blocks a protective exit (the policy of
    options_manager._market_is_open and of POS-KILL's clock in bot.ts).
A reading is cached for BROKER_CLOCK_TTL_SECONDS (system_config) or until its
own next_open / next_close, whichever comes first, so a half-day close at
13:00 takes effect at 13:00 and not a TTL later. A failed read is cached for
the TTL too (one clock request per TTL however many orders ask).

check_order_allowed(symbol, what) is called immediately before every order
POST in bot_engine.py, intraday_shorts.py, options_manager.py and
options_execution.py (test_broker_clock_gate.py checks this on the source).
It RAISES rather than returning a fake rejected response: several callers
never read the POST's response (the old floors book "buy"/"sell", the third
leg books an exit and stamps its re-entry cooldown, the intraday-short entry
books a trade with order id "?"), while every caller already treats an
exception from its POST as a failed order. OrderBlocked is a RuntimeError so
that bot_engine._unwind_covered_calls passes it up as "the buy-back failed:
do not sell the stock". Cancels are never gated. Python sends no
extended-hours orders, so the extended-hours stock-exit exception in
server/brokerClockGate.ts has no Python counterpart.
"""
import logging
import os
import re
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from alpaca_rate_limiter import alpaca_throttle
from system_config import BASE_CONFIG

logger = logging.getLogger("broker_clock")

ALPACA_BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
CLOCK_READ_TIMEOUT_S = 8
ORDER_BLOCKED_PREFIX = "ORDER-BLOCKED"
_ET = ZoneInfo("America/New_York")

_LOCK = threading.Lock()
_CACHE = {"reading": None, "valid_until": 0.0}
_LAST_GOOD = {"reading": None}
_BLOCK_LOGGED = {}   # symbol -> ET day of its last blocked-order log line


class OrderBlocked(RuntimeError):
    """Raised instead of sending an order while the broker's clock says the
    market is closed. The message starts with ORDER_BLOCKED_PREFIX."""


def reset_cache() -> None:
    """Forget every reading and every blocked-order log mark (tests; a fresh
    process starts here)."""
    with _LOCK:
        _CACHE.update({"reading": None, "valid_until": 0.0})
        _LAST_GOOD["reading"] = None
        _BLOCK_LOGGED.clear()


def _ttl_seconds() -> float:
    return float(BASE_CONFIG.get("BROKER_CLOCK_TTL_SECONDS", 30))


def _headers() -> dict:
    return {"APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ""),
            "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", "")}


def _fetch_clock_body():
    """One GET of the broker's clock: its decoded JSON body, or None when the
    request or the decode fails (logged, never raised). The only network I/O
    in this module; the root conftest stubs it so the test suite never reaches
    the network (tests of the reader patch it themselves)."""
    try:
        alpaca_throttle.acquire()
        resp = requests.get(f"{ALPACA_BASE}/v2/clock", headers=_headers(), timeout=CLOCK_READ_TIMEOUT_S)
        return resp.json()
    except Exception as e:  # a broken clock must never raise into an order path
        logger.warning(f"[BROKER-CLOCK] clock read failed: {str(e)[:160]}")
        return None


def _parse_ts(raw):
    """An Alpaca clock timestamp ('2026-11-05T16:00:00-05:00', nanoseconds
    allowed) as epoch seconds; None when missing, unreadable or without an
    offset."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        dt = datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", raw.replace("Z", "+00:00")))
    except ValueError:
        return None
    return dt.timestamp() if dt.tzinfo is not None else None


def _reading_from_body(body, read_at: float):
    """A usable reading needs a boolean is_open; the times are optional."""
    if not isinstance(body, dict) or not isinstance(body.get("is_open"), bool):
        return None
    return {"is_open": body["is_open"], "next_open": _parse_ts(body.get("next_open")),
            "next_close": _parse_ts(body.get("next_close")), "read_at": float(read_at)}


def _et(ts) -> str:
    return datetime.fromtimestamp(ts, _ET).strftime("%a %Y-%m-%d %H:%M ET")


def _live_reading(now: float):
    """The cached reading while it is valid, else one fresh read. None when
    the read failed (a failure is cached for the TTL as well)."""
    with _LOCK:
        if now < _CACHE["valid_until"]:
            return _CACHE["reading"]
        reading = _reading_from_body(_fetch_clock_body(), now)
        valid_until = now + _ttl_seconds()
        if reading is not None:
            _LAST_GOOD["reading"] = reading
            edge = reading["next_close"] if reading["is_open"] else reading["next_open"]
            if edge is not None and edge > now:
                valid_until = min(valid_until, edge)
        _CACHE.update({"reading": reading, "valid_until": valid_until})
        return reading


def _state(name: str, source: str, detail: str) -> dict:
    return {"state": name, "source": source, "detail": detail}


def _from_reading(r: dict, now: float, live: bool) -> dict:
    """The market state a reading implies at `now`. A live reading's is_open
    is trusted as read; an old one (the read failed) only within the times
    it states."""
    source = "live" if live else "extrapolated"
    nxt_open, nxt_close = r["next_open"], r["next_close"]
    if r["is_open"]:
        if nxt_close is None:
            if live:
                return _state("open", source, "broker clock: open")
            return _state("unknown", source, "the last reading said open but gave no close time")
        if now < nxt_close:
            return _state("open", source, f"broker clock: open until {_et(nxt_close)}")
        if nxt_open is not None and now < nxt_open:
            return _state("closed", source, f"broker clock: closed at {_et(nxt_close)}, next open {_et(nxt_open)}")
        return _state("unknown", source, "the last reading is out of date")
    if nxt_open is None:
        if live:
            return _state("closed", source, "broker clock: closed")
        return _state("unknown", source, "the last reading said closed but gave no open time")
    if now < nxt_open:
        return _state("closed", source, f"broker clock: closed until {_et(nxt_open)}")
    if nxt_close is not None and now < nxt_close:
        return _state("open", source, f"broker clock: opened at {_et(nxt_open)} on its own schedule")
    return _state("unknown", source, "the last reading is out of date")


def clock_state(now=None) -> dict:
    """{"state": "open" | "closed" | "unknown", "source": "live" |
    "extrapolated" | "none", "detail": str} at `now` (epoch seconds)."""
    now = time.time() if now is None else float(now)
    live = _live_reading(now)
    if live is not None:
        return _from_reading(live, now, live=True)
    good = _LAST_GOOD["reading"]
    if good is None:
        return _state("unknown", "none", "no usable broker clock reading")
    st = _from_reading(good, now, live=False)
    st["detail"] = f"clock read failed; {st['detail']} (last good reading {_et(good['read_at'])})"
    return st


def orders_allowed(now=None) -> tuple:
    """(allowed, reason). Allowed while the broker says open, and while the
    clock is unknown (a clock outage never blocks an exit)."""
    st = clock_state(now)
    if st["state"] == "closed":
        return False, f"market closed ({st['detail']})"
    if st["state"] == "open":
        return True, f"market open ({st['detail']})"
    return True, f"clock unknown, order allowed ({st['detail']})"


def check_order_allowed(symbol, what: str = "order", now=None) -> None:
    """Call immediately before an order POST. Returns when the order may be
    sent; raises OrderBlocked while the broker's clock says the market is
    closed. One warning line per symbol per ET day (the scan asks every few
    minutes)."""
    ok, why = orders_allowed(now)
    if ok:
        return
    sym = str(symbol or "?")
    msg = f"{ORDER_BLOCKED_PREFIX}: {why} — {what} {sym} not sent"
    day = datetime.fromtimestamp(time.time() if now is None else float(now), _ET).strftime("%Y-%m-%d")
    with _LOCK:
        first = _BLOCK_LOGGED.get(sym) != day
        if first:
            for stale in [s for s, d in _BLOCK_LOGGED.items() if d != day]:
                del _BLOCK_LOGGED[stale]
            _BLOCK_LOGGED[sym] = day
    if first:
        logger.warning(f"[ORDER-BLOCKED-CLOSED] {msg}")
    raise OrderBlocked(msg)
