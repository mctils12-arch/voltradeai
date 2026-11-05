#!/usr/bin/env python3
"""
option_fill_ledger.py — the option FILL LEDGER (2026-11-05, measurement only).

WHY: the human asked for a daily line whose option P&L is split into
FILLS/SLIPPAGE vs DIRECTION. The fills half is, per filled option order,
(fill - mid at submission) x qty x 100 — and the mid at submission was never
recorded anywhere. This module records it: one JSON row per option order the
broker ACCEPTED, appended right after the order POST returned, to
DATA_DIR/voltrade_option_fill_ledger.jsonl:

    {"ts", "order_id", "symbol", "side", "qty", "bid", "ask", "mid",
     "source", "limit", "quote_age_s"}            (+ "order_class": "mleg")

A multi-leg (mleg) order writes one row per leg under the PARENT order id.
server/dailyLine.ts joins the rows with Alpaca's filled orders of the day by
order id and leg symbol. server/bot.ts's own option order paths append the
same row shape to the same file.

THE QUOTE. Nothing here fetches anything. The paths that price an order
already read a quote: options_manager._get_option_snapshot (every single-leg
close, a roll's close leg, the mleg fallback's legs) and
options_execution._fetch_option_chain (every entry's contract selection) pass
each quote they read to note_quote(); a row takes the latest quote noted for
its contract if it is at most QUOTE_MAX_AGE_S old (a roll's new leg is noted
from the chain quote the roll priced it with). A path without a quote writes
bid/ask/mid null — the daily line counts such fills as "unattributed" and
never guesses a mid. A one-sided or crossed quote has no mid.

Rotation: above LEDGER_MAX_BYTES the file moves to <file>.1 (one old
generation kept; a day's rows are a few kB, so .1 always covers the line's
window).

MEASUREMENT ONLY: nothing here changes an order, a price, a size or a
decision, nothing prints to stdout (bot.ts parses the subprocess stdout), and
nothing raises into an order path — a failed write is logged and dropped.
"""
import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

logger = logging.getLogger("option_fill_ledger")

LEDGER_FILENAME = "voltrade_option_fill_ledger.jsonl"
LEDGER_MAX_BYTES = 2_000_000      # ~10k rows; rotated to <file>.1 above this
QUOTE_MAX_AGE_S = 120.0           # an older noted quote is not "the quote at submission"
QUOTE_MEMO_MAX = 4000             # contracts remembered at once (chain fetches note ~100 each)

_quotes = {}                      # symbol -> (noted_at_epoch_s, bid, ask)
_lock = threading.Lock()


def _num(x):
    """A finite float, else None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def mid_of(bid, ask):
    """(bid + ask) / 2 of a two-sided, uncrossed quote; None otherwise."""
    b, a = _num(bid), _num(ask)
    if b is None or a is None or b <= 0 or a <= 0 or a < b:
        return None
    return round((b + a) / 2.0, 4)


def note_quote(symbol, bid, ask, now=None):
    """Remember the quote an order path just read for `symbol`. Bounded: past
    QUOTE_MEMO_MAX contracts the stale ones are dropped first."""
    if not symbol:
        return
    t = time.time() if now is None else float(now)
    with _lock:
        if symbol not in _quotes and len(_quotes) >= QUOTE_MEMO_MAX:
            cutoff = t - QUOTE_MAX_AGE_S
            for k in [k for k, v in _quotes.items() if v[0] < cutoff]:
                del _quotes[k]
            if len(_quotes) >= QUOTE_MEMO_MAX:
                _quotes.clear()
        _quotes[symbol] = (t, _num(bid), _num(ask))


def forget_quote(symbol):
    """A fresh read of `symbol` is starting: an older quote must not stand in
    for a read that then fails."""
    with _lock:
        _quotes.pop(symbol, None)


def quote_for(symbol, now=None):
    """The latest quote noted for `symbol`, if at most QUOTE_MAX_AGE_S old:
    {"bid", "ask", "mid", "quote_age_s"} (all None when there is none)."""
    t = time.time() if now is None else float(now)
    with _lock:
        hit = _quotes.get(symbol)
    if not hit or t - hit[0] > QUOTE_MAX_AGE_S:
        return {"bid": None, "ask": None, "mid": None, "quote_age_s": None}
    noted_at, bid, ask = hit
    return {"bid": bid, "ask": ask, "mid": mid_of(bid, ask), "quote_age_s": round(max(0.0, t - noted_at), 1)}


def ledger_path():
    return os.path.join(DATA_DIR, LEDGER_FILENAME)


def append_row(row):
    """Append one row, rotating the file above LEDGER_MAX_BYTES. True when written."""
    try:
        p = ledger_path()
        line = json.dumps(row, default=str) + "\n"
        with _lock:
            if os.path.exists(p) and os.path.getsize(p) > LEDGER_MAX_BYTES:
                os.replace(p, p + ".1")
            with open(p, "a") as f:
                f.write(line)
        return True
    except (OSError, TypeError, ValueError) as e:
        logger.warning(f"option fill ledger: row for order {row.get('order_id')} not written: {e}")
        return False


def record_order(order_id, symbol, side, qty, source, limit=None, order_class=None, now=None):
    """One row for an option order the broker accepted (call it AFTER the
    POST returned, with the id from its response). The quote is the latest
    one noted for `symbol` (quote_for). Returns whether a row was written;
    never raises."""
    try:
        if not order_id or not symbol:
            return False
        q = quote_for(symbol, now)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "order_id": str(order_id),
            "symbol": str(symbol),
            "side": str(side or "").lower(),
            "qty": _num(qty),
            "bid": q["bid"],
            "ask": q["ask"],
            "mid": q["mid"],
            "source": str(source),
            "limit": _num(limit),
            "quote_age_s": q["quote_age_s"],
        }
        if order_class:
            row["order_class"] = str(order_class)
        return append_row(row)
    except Exception as e:  # an order path must never see a measurement failure
        logger.warning(f"option fill ledger: record_order({order_id}, {symbol}) failed: {e}")
        return False


def record_mleg_order(order_id, legs, qty, source, now=None):
    """One row per leg of an accepted mleg order, all under the PARENT order
    id; a leg's qty is the order qty x its ratio. Returns rows written."""
    written = 0
    try:
        units = _num(qty) or 0.0
        for leg in legs or []:
            ratio = _num((leg or {}).get("ratio_qty")) or 1.0
            if record_order(order_id, (leg or {}).get("symbol"), (leg or {}).get("side"), units * ratio,
                            source, order_class="mleg", now=now):
                written += 1
    except Exception as e:
        logger.warning(f"option fill ledger: record_mleg_order({order_id}) failed: {e}")
    return written
