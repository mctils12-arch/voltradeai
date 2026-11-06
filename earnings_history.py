#!/usr/bin/env python3
"""
earnings_history.py — EARNINGS HISTORY, logged forward (2026-11-06; human
refocus spec §3 step 5 "stock selection: earnings history (beats last 3
quarters)" and §7 "build our own dataset going forward; log every signal").

LOGGING ONLY. Nothing here changes a score, a size, a gate or a decision: the
block it builds rides the shadow record (`earnings_history`), the deep-scored
candidate and the decision journal's selection stage, so the signal can be
measured forward once outcome labels accrue (signal_attribution lists it as
the `earnings_beats` row).

Two halves, so deep_score never spends its 15 s fetch budget on Finnhub:

  FILL (Tier 3, hourly; server/bot.ts runs refresh_wanted() as a python3 -c
  subprocess in the mutex lane, server/earningsHistory.ts): for the names
  deep_score asked about in the last WANT_WINDOW_H hours (the want-list),
  fetch Finnhub /stock/earnings — EPS actual vs consensus estimate for the last
  4 reported quarters (free tier) — through finnhub_data's rate-limited GET
  (its 55/min token bucket), spaced call_interval_s apart so this job alone
  stays under half of the free tier's 60 calls/min, at most max_fetch_per_run
  names and run_budget_s seconds per run. Entries older than ttl_h (12 h) are
  refreshed. A failed refresh keeps the last good entry (its error is noted).
  401/403/429 stop the run (the key or the rate limit, not the symbol).

  READ (bot_engine.deep_score, every candidate): read_for_candidate() reads
  that cache file ONLY — no network, no import of the fetch path — and returns
    {"source": "finnhub",
     "beats_last_3":        quarters, of the 3 most recent reported, whose
                            actual EPS beat the consensus estimate (0-3), or
                            None unless all 3 are on record,
     "surprise_pct_last_3": (actual - estimate) / |estimate| x 100 per quarter,
                            newest first (None for a quarter missing either
                            figure or with a zero estimate),
     "quarters_seen":       reported quarters on record (<= 3; 0 = Finnhub
                            has none; None = no fetch has succeeded),
     "latest_period":       the newest quarter's fiscal period end (YYYY-MM-DD),
     "cache_age_h":         hours since the fetch,
     "reason":              None when beats_last_3 is set, else WHY (REASONS)}
  Missing data is None with a reason, never a zero. The only side effect is
  adding the name to the want-list when its entry is missing, failed or due.

POINT IN TIME: Finnhub's endpoint lists reported quarters only (a quarter whose
period has not ended is dropped), and every record is written AFTER the fetch
it reads, so no quarter can postdate its record; cache_age_h bounds how far a
quarter reported after the fetch can lag the record. The endpoint carries no
report dates, so there is no history to backtest: forward logging is the test.

FILES (storage_config.DATA_DIR, the volume): voltrade_earnings_history.json
(written by the fill only; entries a week old are pruned) and
voltrade_earnings_wanted.json (written by deep_score's read only; names older
than WANT_WINDOW_H are pruned). Both atomic (temp file + os.replace).

OFF SWITCH: system_config BASE_CONFIG["EARNINGS_HISTORY"]["enabled"]; the env
var VOLTRADE_EARNINGS_HISTORY wins whenever it is PRESENT ("off"/"0"/"false"
= off: deep_score reads nothing and the record has no earnings_history key,
the fill does nothing). Nothing here prints to stdout except the caller's
JSON (bot.ts parses it).
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from datetime import date, datetime, timedelta, timezone

try:
    from storage_config import DATA_DIR
except ImportError:  # storage_config is the sanctioned resolver; this only covers a bare checkout
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

log = logging.getLogger("voltrade.earnings_history")

CONFIG_KEY = "EARNINGS_HISTORY"
ENV_VAR = "VOLTRADE_EARNINGS_HISTORY"
OFF_VALUES = ("", "0", "off", "false", "no", "none")
CACHE_FILENAME = "voltrade_earnings_history.json"
WANTED_FILENAME = "voltrade_earnings_wanted.json"
SOURCE = "finnhub"
SCHEMA_VERSION = 1

QUARTERS = 3              # the refocus spec's "last 3 quarters"
FETCH_LIMIT = 4           # Finnhub's free tier returns the last 4 reported quarters
WANT_WINDOW_H = 24        # "that day's deep-scored names": the fill serves names asked about within 24 h
WANT_REWRITE_S = 900      # deep_score re-stamps a wanted name at most every 15 minutes
MAX_WANTED = 400          # want-list bound (newest kept); a day deep-scores a few dozen names
PRUNE_AFTER_H = 168       # cache entries not refreshed for a week are dropped
STOP_HTTP = (401, 403, 429)  # account-level answers: stop the run, write nothing for the name

# Mirrors system_config BASE_CONFIG["EARNINGS_HISTORY"] (test_earnings_history.py pins the parity).
DEFAULTS = {
    "enabled": True,
    "ttl_h": 12.0,             # the fill refreshes an entry older than this
    "max_age_h": 48.0,         # deep_score logs an entry up to this old; older = None, reason "stale"
    "max_fetch_per_run": 15,   # Finnhub calls per hourly run
    "call_interval_s": 2.0,    # spacing between this job's calls: <= 30/min, half the free tier's 60/min
    "run_budget_s": 40.0,      # no new call starts after this (the Node kill is at 60 s)
}

# Why a block has no beats_last_3 (decision_journal.EARNINGS_REASON_TEXT words them).
REASONS = ("not_cached", "no_api_key", "fetch_error", "stale", "no_reports",
           "fewer_than_3", "incomplete", "cache_unreadable")

_want_lock = threading.Lock()


# ── switch + config ───────────────────────────────────────────────────────────
def _config_raw() -> dict:
    try:
        from system_config import BASE_CONFIG
        c = BASE_CONFIG.get(CONFIG_KEY) or {}
        return c if isinstance(c, dict) else {}
    except ImportError as e:  # never break deep_score on a config read
        log.warning(f"[EARNINGS-HISTORY] system_config unavailable ({e}); defaults used")
        return {}


def _bounded(v, default, lo, hi):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return default if not math.isfinite(x) else min(hi, max(lo, x))


def config(raw=None) -> dict:
    """The effective settings: system_config over DEFAULTS, each kept inside the
    range the Tier-3 lane can honour (run budget + one 10 s call < the 60 s kill)."""
    c = _config_raw() if raw is None else (raw if isinstance(raw, dict) else {})
    ttl = _bounded(c.get("ttl_h"), DEFAULTS["ttl_h"], 1.0, 48.0)
    return {
        "enabled": bool(c.get("enabled", DEFAULTS["enabled"])),
        "ttl_h": ttl,
        "max_age_h": _bounded(c.get("max_age_h"), DEFAULTS["max_age_h"], ttl, 168.0),
        "max_fetch_per_run": int(_bounded(c.get("max_fetch_per_run"), DEFAULTS["max_fetch_per_run"], 0, 50)),
        "call_interval_s": _bounded(c.get("call_interval_s"), DEFAULTS["call_interval_s"], 1.0, 10.0),
        "run_budget_s": _bounded(c.get("run_budget_s"), DEFAULTS["run_budget_s"], 5.0, 45.0),
    }


def enabled(env=None, cfg=None) -> bool:
    """The env var when PRESENT (any value wins), else the config flag (default on)."""
    env = os.environ if env is None else env
    v = env.get(ENV_VAR)
    if v is not None:
        return str(v).strip().lower() not in OFF_VALUES
    return bool((config() if cfg is None else cfg).get("enabled", True))


def cache_path() -> str:
    return os.path.join(DATA_DIR, CACHE_FILENAME)


def wanted_path() -> str:
    return os.path.join(DATA_DIR, WANTED_FILENAME)


# ── small helpers ─────────────────────────────────────────────────────────────
def _num(x):
    """A finite float, else None (bools are not numbers here)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_ts(s):
    """A UTC datetime from our own ISO stamps, else None."""
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as e:
        log.debug(f"[EARNINGS-HISTORY] unparseable timestamp {s!r}: {e}")
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _period_date(s):
    try:
        return datetime.strptime(str(s or "")[:10], "%Y-%m-%d").date()
    except ValueError as e:
        log.debug(f"[EARNINGS-HISTORY] quarter without a usable period {s!r}: {e}")
        return None


def _now(now=None) -> datetime:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc)


def _read_json(path: str):
    """(dict, error): {} with no error when the file is absent; {} and the error when unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except FileNotFoundError:
        return {}, None
    except (OSError, ValueError) as e:
        log.warning(f"[EARNINGS-HISTORY] {os.path.basename(path)} unreadable: {e}")
        return {}, f"{type(e).__name__}: {str(e)[:120]}"
    if not isinstance(d, dict):
        return {}, "not a JSON object"
    return d, None


def _write_json(path: str, data: dict) -> bool:
    """Atomic write (temp file in the same directory + os.replace). Never raises."""
    folder = os.path.dirname(path) or "."
    try:
        os.makedirs(folder, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".earnings.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, separators=(",", ":"))
            os.replace(tmp, path)
            tmp = None
        finally:
            if tmp is not None and os.path.exists(tmp):
                os.unlink(tmp)
        return True
    except (OSError, TypeError, ValueError) as e:
        log.warning(f"[EARNINGS-HISTORY] {os.path.basename(path)} not written: {e}")
        return False


# ── per-quarter arithmetic ────────────────────────────────────────────────────
def surprise_pct(q: dict):
    """(actual - estimate) / |estimate| x 100, 2 decimals; None without both or with a zero estimate."""
    a, e = _num((q or {}).get("actual")), _num((q or {}).get("estimate"))
    if a is None or e is None or e == 0:
        return None
    return round((a - e) / abs(e) * 100.0, 2)


def beat(q: dict):
    """True when actual EPS > estimate, False when not (an in-line quarter is not a beat), None without both."""
    a, e = _num((q or {}).get("actual")), _num((q or {}).get("estimate"))
    if a is None or e is None:
        return None
    return a > e


def parse_quarters(rows, today: date) -> list:
    """Finnhub /stock/earnings rows -> [{period, actual, estimate}], newest first, at most
    FETCH_LIMIT. A row with no usable period, a duplicate period, or a period that has not
    ended yet (cannot have been reported) is dropped; missing figures stay None."""
    out, seen = [], set()
    for item in rows if isinstance(rows, list) else []:
        if not isinstance(item, dict):
            continue
        pd = _period_date(item.get("period"))
        if pd is None or pd > today or pd.isoformat() in seen:
            continue
        seen.add(pd.isoformat())
        out.append({"period": pd.isoformat(), "actual": _num(item.get("actual")),
                    "estimate": _num(item.get("estimate"))})
    out.sort(key=lambda q: q["period"], reverse=True)
    return out[:FETCH_LIMIT]


# ── READ: what deep_score logs ────────────────────────────────────────────────
def empty_block(reason: str) -> dict:
    return {"source": SOURCE, "beats_last_3": None, "surprise_pct_last_3": None, "quarters_seen": None,
            "latest_period": None, "cache_age_h": None, "reason": reason}


def build_block(entry, now: datetime, cfg: dict, run_status=None, read_error=None):
    """(block, due) for one cache entry — pure. due = the fill should fetch the name
    (missing, failed, or older than ttl_h)."""
    if read_error:
        return empty_block("cache_unreadable"), True
    if not isinstance(entry, dict):
        return empty_block("no_api_key" if run_status == "no_key" else "not_cached"), True
    fetched = _parse_ts(entry.get("fetched_at"))
    if fetched is None:
        return empty_block("not_cached"), True
    blk = empty_block(None)
    age_h = max(0.0, (now - fetched).total_seconds() / 3600.0)
    blk["cache_age_h"] = round(age_h, 1)
    due = age_h >= cfg["ttl_h"]
    if entry.get("status") == "error":
        blk["reason"] = "fetch_error"
        err = str(entry.get("error") or "").strip()
        if err:
            blk["error"] = err[:120]
        return blk, True
    if age_h > cfg["max_age_h"]:
        blk["reason"] = "stale"
        return blk, True
    quarters = [q for q in (entry.get("quarters") or []) if isinstance(q, dict)]
    if entry.get("status") == "empty" or not quarters:
        blk["quarters_seen"] = 0
        blk["reason"] = "no_reports"
        return blk, due
    q3 = quarters[:QUARTERS]
    flags = [beat(q) for q in q3]
    blk["quarters_seen"] = len(q3)
    blk["surprise_pct_last_3"] = [surprise_pct(q) for q in q3]
    blk["latest_period"] = str(q3[0].get("period") or "")[:10] or None
    if len(q3) == QUARTERS and all(f is not None for f in flags):
        blk["beats_last_3"] = sum(1 for f in flags if f)
    else:
        blk["reason"] = "fewer_than_3" if len(q3) < QUARTERS else "incomplete"
    return blk, due


def mark_wanted(ticker: str, now=None) -> bool:
    """Add (or re-stamp) a name on the want-list the fill reads. Names asked about more
    than WANT_WINDOW_H ago are pruned; a name stamped within WANT_REWRITE_S is left
    as is (no write). Never raises; True when the file was written."""
    sym = str(ticker or "").upper().strip()
    if not sym:
        return False
    now = _now(now)
    with _want_lock:
        doc, err = _read_json(wanted_path())
        names = doc.get("names") if isinstance(doc.get("names"), dict) and not err else {}
        prev = _parse_ts(names.get(sym))
        if prev is not None and 0 <= (now - prev).total_seconds() < WANT_REWRITE_S:
            return False
        cutoff = now - timedelta(hours=WANT_WINDOW_H)
        kept = {}
        for k, v in names.items():
            t = _parse_ts(v)
            if isinstance(k, str) and t is not None and t >= cutoff:
                kept[k] = v
        kept[sym] = _iso(now)
        if len(kept) > MAX_WANTED:
            kept = dict(sorted(kept.items(), key=lambda kv: kv[1], reverse=True)[:MAX_WANTED])
        return _write_json(wanted_path(), {"v": SCHEMA_VERSION, "names": kept})


def read_for_candidate(ticker: str, now=None, mark=True):
    """The block deep_score attaches to a candidate, or None when the feature is off.
    Reads the cache file only (no network). Never raises."""
    try:
        cfg = config()
        if not enabled(cfg=cfg):
            return None
        sym = str(ticker or "").upper().strip()
        if not sym:
            return None
        now = _now(now)
        cache, err = _read_json(cache_path())
        entries = cache.get("entries") if isinstance(cache.get("entries"), dict) else {}
        run_status = (cache.get("last_run") or {}).get("status") if isinstance(cache.get("last_run"), dict) else None
        blk, due = build_block(entries.get(sym), now, cfg, run_status, err)
        if due and mark:
            mark_wanted(sym, now)
        return blk
    except Exception as e:  # logging only: a failure here must never reach the score
        log.warning(f"[EARNINGS-HISTORY] read failed for {ticker} (nothing scored reads it): {e}")
        return empty_block("cache_unreadable")


# ── FILL: the Tier-3 job ──────────────────────────────────────────────────────
def _http_status(err: str):
    m = re.search(r"HTTP (\d{3})", err or "")
    return int(m.group(1)) if m else None


def _finnhub_configured() -> bool:
    import finnhub_data
    k = getattr(finnhub_data, "FINNHUB_KEY", "") or ""
    return bool(k) and k != "YOUR_FINNHUB_KEY_HERE"


def due_names(wanted: dict, entries: dict, now: datetime, cfg: dict) -> list:
    """Names to fetch, in order: never fetched, then failed, then older than ttl_h;
    most recently asked-about first within each group. Only names asked about within
    WANT_WINDOW_H count."""
    cutoff = now - timedelta(hours=WANT_WINDOW_H)
    rows = []
    for sym, stamp in (wanted or {}).items():
        t = _parse_ts(stamp)
        if not isinstance(sym, str) or t is None or t < cutoff:
            continue
        e = (entries or {}).get(sym)
        if not isinstance(e, dict):
            rank = 0
        elif e.get("status") == "error":
            rank = 1
        else:
            f = _parse_ts(e.get("fetched_at"))
            if f is not None and (now - f).total_seconds() / 3600.0 < cfg["ttl_h"]:
                continue
            rank = 2
        rows.append((rank, -t.timestamp(), sym))
    rows.sort()
    return [s for _, _, s in rows]


def _prune(entries: dict, now: datetime) -> dict:
    cutoff = now - timedelta(hours=PRUNE_AFTER_H)
    out = {}
    for k, e in (entries or {}).items():
        f = _parse_ts(e.get("fetched_at")) if isinstance(e, dict) else None
        if f is not None and f >= cutoff:
            out[k] = e
    return out


def _record_response(entries: dict, sym: str, data, now: datetime) -> str:
    """Store one fetch's outcome on `entries`; returns "ok" | "empty" | "failed".
    A failed refresh keeps the last good entry (with the error noted on it)."""
    err = str(data.get("error") or "") if isinstance(data, dict) else ""
    if not err and isinstance(data, dict) and data:
        err = f"unexpected response ({len(data)} keys, no list)"
    if err:
        prev = entries.get(sym)
        if isinstance(prev, dict) and prev.get("status") in ("ok", "empty"):
            entries[sym] = {**prev, "last_error": err[:160], "last_error_at": _iso(now)}
        else:
            entries[sym] = {"fetched_at": _iso(now), "status": "error", "error": err[:160], "quarters": []}
        return "failed"
    quarters = parse_quarters(data if isinstance(data, list) else [], now.date())
    entries[sym] = {"fetched_at": _iso(now), "status": "ok" if quarters else "empty", "quarters": quarters}
    return "ok" if quarters else "empty"


def refresh_wanted(now=None, fetch=None, sleep=None, clock=None) -> dict:
    """Tier 3's hourly fill (see the module docstring). Returns the summary bot.ts
    audits (server/earningsHistory.ts): status "off" | "no_key" | "idle" | "done" |
    "error"; wanted / due / attempted / ok / empty / failed / left counts; `stopped`
    (why a run ended early); `first_error`; `status_changed` (vs the last run).
    Never raises. fetch / sleep / clock are injectable for tests."""
    tick = clock or time.monotonic
    pause = sleep or time.sleep
    t0 = tick()
    now = _now(now)
    out = {"status": "error", "wanted": 0, "due": 0, "attempted": 0, "ok": 0, "empty": 0, "failed": 0,
           "left": 0, "stopped": None, "first_error": None, "status_changed": False, "elapsed_s": 0.0}
    try:
        cfg = config()
        if not enabled(cfg=cfg):
            out["status"] = "off"
            return out
        cache, cache_err = _read_json(cache_path())
        if cache_err:
            log.warning(f"[EARNINGS-HISTORY] cache unreadable ({cache_err}); rebuilding it")
        entries = dict(cache.get("entries")) if isinstance(cache.get("entries"), dict) and not cache_err else {}
        prev_run = cache.get("last_run") if isinstance(cache.get("last_run"), dict) else None
        wdoc, _werr = _read_json(wanted_path())
        wanted = wdoc.get("names") if isinstance(wdoc.get("names"), dict) else {}
        cutoff = now - timedelta(hours=WANT_WINDOW_H)
        out["wanted"] = sum(1 for v in wanted.values() if (t := _parse_ts(v)) is not None and t >= cutoff)
        queue = due_names(wanted, entries, now, cfg)
        out["due"] = out["left"] = len(queue)

        if not _finnhub_configured():
            out["status"] = "no_key"
        elif not queue:
            out["status"] = "idle"
        else:
            if fetch is None:
                from finnhub_data import get_earnings_history_raw as fetch
            for sym in queue[:cfg["max_fetch_per_run"]]:
                if tick() - t0 > cfg["run_budget_s"]:
                    out["stopped"] = f"run budget of {cfg['run_budget_s']:g}s reached"
                    break
                if out["attempted"]:
                    pause(cfg["call_interval_s"])   # this job's own spacing, on top of the token bucket
                out["attempted"] += 1
                try:
                    data = fetch(sym)
                except Exception as e:  # e.g. a non-JSON body: recorded on the name, the run goes on
                    log.warning(f"[EARNINGS-HISTORY] fetch raised for {sym}: {e}")
                    data = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
                err = str(data.get("error") or "") if isinstance(data, dict) else ""
                code = _http_status(err) if err else None
                if code in STOP_HTTP:         # the key or the rate limit, not the symbol: write nothing
                    out["first_error"] = out["first_error"] or f"{sym}: {err[:160]}"
                    out["stopped"] = f"Finnhub answered HTTP {code}"
                    break
                kind = _record_response(entries, sym, data, now)
                if kind == "failed" and not out["first_error"]:
                    e = entries.get(sym) or {}
                    out["first_error"] = f"{sym}: {str(e.get('last_error') or e.get('error') or '')[:160]}"
                out[kind] += 1
                out["left"] -= 1
                _write_json(cache_path(), {"v": SCHEMA_VERSION, "entries": _prune(entries, now), "last_run": prev_run})
            out["status"] = "done"

        out["status_changed"] = out["status"] != (prev_run or {}).get("status")
        if out["status"] == "done" or out["status_changed"]:
            _write_json(cache_path(), {"v": SCHEMA_VERSION, "entries": _prune(entries, now),
                                       "last_run": {"at": _iso(now), "status": out["status"]}})
    except Exception as e:  # the job must report, not raise: bot.ts audits this dict
        log.warning(f"[EARNINGS-HISTORY] refresh failed: {e}")
        out["status"] = "error"
        out["first_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    out["elapsed_s"] = round(tick() - t0, 1)
    return out


if __name__ == "__main__":  # python3 earnings_history.py — one fill, JSON on stdout
    print(json.dumps(refresh_wanted(), default=str))
