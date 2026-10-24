#!/usr/bin/env python3
"""
VolTradeAI — Look-up ledger
===========================
The human's observation (2026-10-22): "it seems like a stock looks good when
it's down big on bad news." The Watch card's look-up (lookup_ticker.py) shows
what the bot thinks of one symbol right now; this ledger turns that
impression into a measured rule. Every fresh look-up is recorded with its
verdict, score and the day's move; a nightly job fills the +1d / +5d / +20d
close-to-close outcomes AND the same window for SPY, and `report()` groups
them by verdict and by the size of the day's drop.

HONESTY (read before believing any number it prints)
  * NOT an unbiased sample. A human looked these symbols up because something
    caught their eye. The ledger answers "what happened after the bot said X
    about a name I was curious about", not "does the scan's X predict".
    The scan's own rejected/admitted candidates live in shadow_portfolio.
  * Repeat look-ups of one symbol within DEDUP_HOURS are one observation.
  * Returns are close-to-close from the last close on or before the look-up
    date (`excess_pct` = symbol minus SPY over the same bars); `ret_pct`
    starts at the price the look-up saw. A look-up made after the close has
    the two nearly equal; one made mid-session differs by the session's
    remainder, and only `excess_pct` removes the market leg.
  * Small n is reported as small n: a bucket under MIN_N is flagged
    `thin`, never ranked.

Storage: voltrade_lookup_ledger.json (atomic write; capped at MAX_RECORDS).
Nothing here touches order flow or any trading threshold.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

try:
    from storage_config import DATA_DIR
except ImportError:  # pragma: no cover
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

log = logging.getLogger("voltrade.lookup_ledger")

LEDGER_PATH = os.path.join(DATA_DIR, "voltrade_lookup_ledger.json")
MAX_RECORDS = 5000
DEDUP_HOURS = 6
HORIZONS = {"+1d": 1, "+5d": 5, "+20d": 20}
GIVE_UP_DAYS = 70            # past this age a missing horizon gets a terminal sentinel
MIN_N = 20                   # under this a bucket is `thin`
# The day's move buckets (percent change on the day at look-up time).
MOVE_BUCKETS = (("down >=5%", None, -5.0), ("down 2-5%", -5.0, -2.0), ("flat/up", -2.0, None))


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def load() -> List[dict]:
    try:
        with open(LEDGER_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    except Exception as e:
        log.warning("lookup ledger unreadable: %s", e)
        return []


def _save(records: List[dict]) -> bool:
    try:
        d = os.path.dirname(LEDGER_PATH) or "."
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".lookup_ledger.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as tf:
                json.dump(records[-MAX_RECORDS:], tf)
            os.replace(tmp, LEDGER_PATH)
            return True
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                log.debug("lookup ledger tmp cleanup failed: %s", tmp)
            return False
    except Exception as e:
        log.warning("lookup ledger save failed: %s", e)
        return False


# ── recording ───────────────────────────────────────────────────────────────
def entry_from_lookup(out: dict) -> Optional[dict]:
    """The compact ledger row for one look-up result, or None when the
    look-up produced nothing worth recording (no price, no decision)."""
    if not isinstance(out, dict) or out.get("errors", {}).get("symbol"):
        return None
    quick = out.get("quick") or {}
    decision = out.get("decision") or {}
    price = quick.get("price")
    if not isinstance(price, (int, float)) or price <= 0 or not decision.get("verdict"):
        return None
    deep = out.get("deep") or {}
    score = deep.get("score")
    regime = out.get("regime") or {}
    return {
        "symbol": out.get("symbol"),
        "at": out.get("at"),
        "price": float(price),
        "change_pct": quick.get("change_pct") if isinstance(quick.get("change_pct"), (int, float)) else None,
        "score": round(float(score), 2) if isinstance(score, (int, float)) else None,
        "min_score": regime.get("min_score"),
        "regime": regime.get("label"),
        "verdict": decision.get("verdict"),
        "stopped_at": decision.get("stopped_at"),
        "held": bool(out.get("held")),
        "outcomes": {},
    }


def record(out: dict) -> bool:
    """Append a look-up to the ledger. Never raises (a look-up must not fail
    because its bookkeeping did). Repeats of a symbol inside DEDUP_HOURS are
    one observation and are skipped."""
    try:
        row = entry_from_lookup(out)
        if row is None:
            return False
        records = load()
        t = _parse_iso(row["at"])
        if t is not None:
            for prev in reversed(records[-200:]):
                if prev.get("symbol") != row["symbol"]:
                    continue
                pt = _parse_iso(prev.get("at"))
                if pt is not None and abs((t - pt).total_seconds()) < DEDUP_HOURS * 3600:
                    return False
        records.append(row)
        return _save(records)
    except Exception as e:
        log.warning("lookup ledger record failed: %s", e)
        return False


# ── outcomes ────────────────────────────────────────────────────────────────
def _bar_date(bar: dict) -> str:
    return str(bar.get("t", ""))[:10]


def outcomes_from_bars(row: dict, bars: List[dict], spy_bars: List[dict]) -> Dict[str, dict]:
    """Pure: the horizons that `bars` (daily, ascending) are long enough to
    settle for this row. Day 0 is the last bar dated on or before the
    look-up's UTC date; +Nd is the Nth bar after it."""
    at = _parse_iso(row.get("at"))
    if at is None or not bars:
        return {}
    d0 = at.date().isoformat()

    def split(bs: List[dict]):
        base_i = None
        for i, b in enumerate(bs):
            if _bar_date(b) <= d0:
                base_i = i
            else:
                break
        return base_i

    bi = split(bars)
    si = split(spy_bars) if spy_bars else None
    res: Dict[str, dict] = {}
    for key, n in HORIZONS.items():
        if key in (row.get("outcomes") or {}):
            continue
        # Without a bar on/before the look-up date the base close is unknown.
        if bi is None or bi + n >= len(bars):
            continue
        base, end = bars[bi].get("c"), bars[bi + n].get("c")
        if not base or not end:
            continue
        ret = (end / row["price"] - 1) * 100
        cc = (end / base - 1) * 100
        o: Dict[str, Any] = {"ret_pct": round(ret, 3), "cc_pct": round(cc, 3), "end_date": _bar_date(bars[bi + n])}
        if si is not None and si + n < len(spy_bars):
            sb, se = spy_bars[si].get("c"), spy_bars[si + n].get("c")
            if sb and se:
                o["spy_cc_pct"] = round((se / sb - 1) * 100, 3)
                o["excess_pct"] = round(cc - o["spy_cc_pct"], 3)
        res[key] = o
    return res


def _fetch_daily(sym: str, start: str, end: str) -> List[dict]:
    from shadow_portfolio import _fetch_historical_bars_batch
    return (_fetch_historical_bars_batch([sym], start, end) or {}).get(sym) or []


def backfill(max_records: int = 60, fetch=None, now: Optional[datetime] = None) -> dict:
    """Settle every horizon the market has now produced. `fetch(sym, start,
    end)` is injectable for tests. Returns counts."""
    fetch = fetch or _fetch_daily
    now = now or datetime.now(timezone.utc)
    stats = {"records": 0, "updated": 0, "pending": 0, "gave_up": 0, "fetch_errors": 0}
    records = load()
    stats["records"] = len(records)
    spy_cache: Dict[str, List[dict]] = {}
    done = 0
    for row in records:
        missing = [k for k in HORIZONS if k not in (row.get("outcomes") or {})]
        if not missing:
            continue
        at = _parse_iso(row.get("at"))
        if at is None:
            continue
        age_days = (now - at).total_seconds() / 86400
        if age_days < 1:
            stats["pending"] += 1
            continue
        if done >= max_records:
            stats["pending"] += 1
            continue
        done += 1
        start = (at - timedelta(days=6)).date().isoformat()
        end = min(now, at + timedelta(days=45)).date().isoformat()
        try:
            bars = fetch(row["symbol"], start, end)
            key = f"{start}|{end}"
            if key not in spy_cache:
                spy_cache[key] = fetch("SPY", start, end)
            spy = spy_cache[key]
        except Exception as e:
            log.warning("lookup ledger fetch failed for %s: %s", row.get("symbol"), e)
            stats["fetch_errors"] += 1
            continue
        got = outcomes_from_bars(row, bars, spy)
        if got:
            row.setdefault("outcomes", {}).update(got)
            stats["updated"] += len(got)
        if age_days > GIVE_UP_DAYS:
            for k in missing:
                if k not in row.get("outcomes", {}):
                    row.setdefault("outcomes", {})[k] = {"no_price_data": True}
                    stats["gave_up"] += 1
        elif any(k not in row.get("outcomes", {}) for k in HORIZONS):
            stats["pending"] += 1
    if stats["updated"] or stats["gave_up"]:
        _save(records)
    return stats


# ── the report ──────────────────────────────────────────────────────────────
def _move_bucket(change_pct: Optional[float]) -> str:
    if change_pct is None:
        return "unknown"
    for name, lo, hi in MOVE_BUCKETS:
        if (lo is None or change_pct >= lo) and (hi is None or change_pct < hi):
            return name
    return "unknown"


def _cell(rows: List[dict], horizon: str) -> dict:
    rets, exc = [], []
    for r in rows:
        o = (r.get("outcomes") or {}).get(horizon) or {}
        if "ret_pct" in o:
            rets.append(o["ret_pct"])
        if "excess_pct" in o:
            exc.append(o["excess_pct"])
    n = len(rets)
    return {
        "n": n,
        "thin": n < MIN_N,
        "mean_ret_pct": round(sum(rets) / n, 3) if n else None,
        "win_rate": round(100 * sum(1 for x in rets if x > 0) / n, 1) if n else None,
        "n_excess": len(exc),
        "mean_excess_pct": round(sum(exc) / len(exc), 3) if exc else None,
    }


def report(records: Optional[List[dict]] = None) -> dict:
    records = load() if records is None else records
    groups: Dict[str, Dict[str, List[dict]]] = {"by_verdict": {}, "by_move": {}, "admitted_by_move": {}}
    for r in records:
        v = r.get("stopped_at") and f"stopped at {r['stopped_at']}" or r.get("verdict") or "unknown"
        mv = _move_bucket(r.get("change_pct"))
        groups["by_verdict"].setdefault(v, []).append(r)
        groups["by_move"].setdefault(mv, []).append(r)
        # The human's question: of the names the bot LIKED (score cleared its
        # bar), how did the ones down hard on the day do?
        sc, ms = r.get("score"), r.get("min_score")
        if isinstance(sc, (int, float)) and isinstance(ms, (int, float)) and sc >= ms:
            groups["admitted_by_move"].setdefault(mv, []).append(r)
    out: Dict[str, Any] = {
        "total": len(records),
        "pending": sum(1 for r in records if len(r.get("outcomes") or {}) < len(HORIZONS)),
        "min_n": MIN_N,
        "caveat": ("Hand-picked look-ups, not a random sample; repeats within "
                   f"{DEDUP_HOURS}h are one observation; buckets under {MIN_N} are thin — read as anecdote."),
    }
    for name, g in groups.items():
        out[name] = {k: {"count": len(rows), **{h: _cell(rows, h) for h in HORIZONS}} for k, rows in sorted(g.items())}
    return out


if __name__ == "__main__":  # python3 lookup_ledger.py
    print(json.dumps(report(), indent=1))
