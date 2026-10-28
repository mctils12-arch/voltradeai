#!/usr/bin/env python3
"""
VolTradeAI — Shadow Portfolio Logger
=====================================
The learning-data multiplier. Logs EVERY scanned candidate (not just the ones
we traded), then backfills hypothetical outcomes from real historical prices
later. Gives the ML model 50-100x more training data without placing a single
additional order.

WHY THIS EXISTS
  The existing feedback loop in ml_model_v2.py only learns from trades the
  bot actually took. If the bot scans 80 candidates, takes 2, and rejects
  78, only the 2 contribute to learning. The 78 rejections are thrown away
  — meaning the system can't learn:
    - Were my rejections correct? (would those 78 have lost money?)
    - Am I rejecting winners? (would some have been big winners?)
    - Is my MIN_SCORE threshold calibrated to reality?

HOW IT WORKS
  1. Every time deep_score() finishes, we log the candidate + features +
     decision to voltrade_shadow_candidates.json
  2. A nightly backfill job (called from bot.ts or cron) looks up the
     actual forward 5/10/20-day return via Alpaca's free historical bars
     endpoint — one batch request handles ~50 tickers at a time
  3. Labels each shadow record: would it have won or lost under the bot's
     normal exit rules (+2% PT, -4% SL, 5d timeout)
  4. A meta-model in ml_model_v2 can optionally train on shadow records
     at reduced weight (e.g. 0.3x) to learn the "rejection decision" layer

API DESIGN
  from shadow_portfolio import (
      log_candidate,      # call from deep_score() — non-blocking
      backfill_outcomes,  # call once per day — batch Alpaca lookup
      load_shadow_data,   # call from ml_model_v2 training — returns X/y/weights
      get_shadow_stats,   # call from dashboard — for visibility
  )

RATE LIMIT BUDGET
  Designed for paid Alpaca (200/min SIP quota, $100/mo tier):
    - log_candidate: zero API calls (pure file I/O)
    - backfill_outcomes: ~3 batch calls per run (50 tickers each) = 150 calls
      total, spread over the backfill window. Safe margin.
  Uses thread-safe token bucket in case multiple workers ever hit it.

STORAGE
  voltrade_shadow_candidates.json — hot log, capped at 20,000 records
  Auto-rotates oldest out once cap is hit. Uses atomic write (temp + rename)
  and POSIX file locking for crash safety across parallel scan workers.

CODE VERSION INTEROP
  Shadow records use code_version = f"{CODE_VERSION}-shadow" so the existing
  _load_trade_feedback() filter doesn't accidentally pull them into the
  primary-model training set. They stay isolated unless load_shadow_data()
  is explicitly called.
"""

import os
import json
import time
import logging
import threading
from alpaca_feed import data_feed, bars_feed  # [REPAIR 2026-07-06/2026-07-18] central feed w/ SIP-403 + bars-endpoint fallback
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

logger = logging.getLogger("voltrade.shadow")

# ── Paths ─────────────────────────────────────────────────────────────────────
SHADOW_LOG_PATH = os.path.join(DATA_DIR, "voltrade_shadow_candidates.json")
SHADOW_STATS_PATH = os.path.join(DATA_DIR, "voltrade_shadow_stats.json")
SHADOW_LOCK_PATH = SHADOW_LOG_PATH + ".lock"

# ── Config ────────────────────────────────────────────────────────────────────
MAX_SHADOW_RECORDS = 20_000          # rotation cap (~6 months of scans)
FORWARD_HORIZONS_DAYS = [5, 10, 20]  # when to check hypothetical outcomes
WIN_THRESHOLD_PCT = 2.0              # matches bot's take-profit
LOSS_THRESHOLD_PCT = -4.0            # matches bot's stop-loss
BATCH_SIZE = 50                       # tickers per Alpaca batch request
CODE_VERSION = "1.0.34-shadow"

# REPAIR 2026-08-17 (KNOWN BROKEN #20 follow-up): backfill_outcomes() had no
# way to give up on a (record, horizon) pair whose price data will never
# arrive (delisted ticker, bad symbol, permanent Alpaca gap) — a record with
# outcomes[horizon_key] still None gets re-queued and re-fetched EVERY night
# forever, since nothing ever gets written to outcomes on failure. Live
# evidence (/api/diag/shadow + /api/diag/audit?type=SHADOW-BACKFILL,
# 2026-08-16 run): missing_price=1469 out of 3002 processed jobs (49%) in a
# single night, while rejected_masterkill/rejected_heat/rejected_other — the
# exact evidence #20 needs — still show zero labeled outcomes despite being
# weeks old. Permanently-stuck jobs re-consuming ~half of every night's
# max_records budget is sufficient on its own to explain the frontier never
# reaching those buckets, regardless of how many nights this has run.
MAX_BACKFILL_ATTEMPTS = 3            # give up on a (record, horizon) after this many failed nights

ALPACA_KEY = os.environ.get("ALPACA_KEY", "")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET", "")
ALPACA_DATA_URL = "https://data.alpaca.markets"


# ── Token bucket — same pattern as finnhub_data.py (already proven) ──────────
class _TokenBucket:
    """Blocking token bucket for rate-limit safety on backfill calls."""
    def __init__(self, rate: int = 180, period: float = 60.0):
        # Use 180/min (not 200) to leave headroom for trading calls
        self._rate = rate
        self._period = period
        self._tokens = float(rate)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                float(self._rate),
                self._tokens + elapsed * (self._rate / self._period),
            )
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait_time = (1.0 - self._tokens) * (self._period / self._rate)
        time.sleep(wait_time)
        with self._lock:
            self._tokens = max(0.0, self._tokens - 1.0 +
                               wait_time * (self._rate / self._period))


_alpaca_bucket = _TokenBucket(rate=180, period=60.0)


# ── Headers (function, not captured — avoids the vol_surface.py bug) ─────────
def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": os.environ.get("ALPACA_KEY", ALPACA_KEY),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", ALPACA_SECRET),
    }


# ── File I/O — POSIX lock + atomic write, matches the sector cache fix ──────
def _load_shadow_log() -> List[dict]:
    """Load shadow records with shared read lock. Returns [] if not yet created."""
    try:
        import fcntl
        with open(SHADOW_LOG_PATH, "r") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                return json.load(f)
            finally:
                try: fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception: pass
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    except ImportError:
        try:
            with open(SHADOW_LOG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    except Exception:
        return []


def _save_shadow_log(records: List[dict]) -> bool:
    """Save with exclusive lock + atomic write. Returns True on success."""
    try:
        import fcntl
        use_lock = True
    except ImportError:
        use_lock = False

    try:
        lock_f = open(SHADOW_LOCK_PATH, "a+")
        try:
            if use_lock:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

            # Atomic write via temp file + rename
            dirname = os.path.dirname(SHADOW_LOG_PATH) or "."
            fd, tmp_path = tempfile.mkstemp(
                dir=dirname, prefix=".shadow.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as tf:
                    json.dump(records, tf)
                os.replace(tmp_path, SHADOW_LOG_PATH)
                return True
            except Exception:
                try: os.unlink(tmp_path)
                except Exception: pass
                return False
        finally:
            try:
                if use_lock:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            except Exception: pass
            lock_f.close()
    except Exception as e:
        logger.debug(f"Shadow log save failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

# PIPELINE 2026-10-16: per-candidate model trace persisted on every shadow
# record. Keys and types are fixed here so the page and the band reports can
# rely on them; anything else deep_score hands over is dropped.
ML_TRACE_FLOAT_KEYS = ("win_probability", "ml_score", "ml_confidence", "rule_weight",
                       "ml_weight", "rules_only_score", "ml_only_score", "final_score")
ML_TRACE_STR_KEYS = ("model_type", "regime_used", "ml_signal", "vxx_source", "error")
ML_TRACE_BOOL_KEYS = ("gated", "blended")


def sanitize_ml_trace(ml) -> dict | None:
    """Coerce a deep_score ML trace into the persisted shape (floats, short
    strings, bools, None). Returns None for a missing/invalid trace so an
    absent model output is stored as absence, never as zeros."""
    if not isinstance(ml, dict) or not ml:
        return None
    out = {}
    for k in ML_TRACE_FLOAT_KEYS:
        v = ml.get(k)
        if isinstance(v, bool):
            out[k] = None
        elif isinstance(v, (int, float)) and v == v:
            out[k] = round(float(v), 4)
        else:
            out[k] = None
    for k in ML_TRACE_STR_KEYS:
        v = ml.get(k)
        out[k] = str(v)[:200] if v not in (None, "") else None
    for k in ML_TRACE_BOOL_KEYS:
        v = ml.get(k)
        out[k] = bool(v) if v is not None else None
    return out


SCORE_BREAKDOWN_KEYS = (
    "momentum", "mean_reversion", "vrp", "squeeze", "volume",
    "vol_premium", "macro", "news", "adx", "obv",
    "intel", "alt", "social", "fundamental",
)
# REGIME ENGINE selection guards (2026-10-28, docs/regime/FINDINGS.md step 4):
# 0/1 per guard (knife / dip / panic / rs / breadth) and the score points a
# live 'dip' neutralization would remove — persisted beside the sub-scores so
# each guard's prevention-P&L is readable BEFORE it gates (RULE REVIEW). Kept
# as a separate tuple so the 14 sub-score keys above stay exactly what the
# 2026-10-26 breakdown contract (test_shadow_score_breakdown.py) pins.
REGIME_GUARD_KEYS = ("regime_knife", "regime_dip", "regime_panic", "regime_rs", "regime_breadth", "regime_mr_delta")


def _sanitize_score_breakdown(bd) -> dict | None:
    """Coerce a deep_score sub-score breakdown into the persisted shape.
    Returns None for a missing breakdown (older callers)."""
    if not isinstance(bd, dict) or not bd:
        return None
    out = {}
    for k in SCORE_BREAKDOWN_KEYS + REGIME_GUARD_KEYS:
        v = bd.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
            out[k] = round(float(v), 1)
        else:
            out[k] = None
    return out


def log_candidate(
    ticker: str,
    features: dict,
    score: float,
    decision: str,
    decision_reason: str = "",
    entry_price: float = 0.0,
    vxx_ratio: float = 1.0,
    regime_label: str = "neutral",
    side: str = "",
    action_label: str = "",
    trade_type: str = "",
    ml: dict = None,
    score_breakdown: dict = None,
) -> None:
    """
    Log a scored candidate to the shadow portfolio.

    Called from bot_engine.deep_score() AFTER score is finalized.
    Non-blocking: any failure is swallowed silently (logging must never
    break the trading loop).

    Args:
        ticker:          "AMD"
        features:        the 34-feature dict passed to ml_score()
        score:           the final combined_score from deep_score
        decision:        one of "taken" | "rejected_score" | "rejected_heat" |
                         "rejected_halt" | "rejected_earnings" | "rejected_other" |
                         "rejected_masterkill" | "rejected_capital"
        decision_reason: human-readable explanation (for debugging)
        entry_price:     stock price at scan time (for forward-return math)
        vxx_ratio:       VXX regime context
        regime_label:    regime string from _classify_regime
        side / action_label / trade_type: what deep_score() would actually DO
                         with the candidate ("buy"/"BUY"/"stock", "sell"/
                         "SELL OPTIONS"/"options", "sell"/"SELL"). Added
                         2026-10-06: "taken" only means score >= MIN_SCORE,
                         and many high scorers are option SALES or overbought
                         SELLs, not stock buys — without this, outcome labels
                         (long-stock returns) mix decisions the bot never made.

    Rate cost: ZERO API calls. Pure file I/O.
    """
    try:
        if not ticker:
            return

        record = {
            "ticker":          str(ticker).upper(),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "score":           float(score),
            "decision":        str(decision),
            "decision_reason": str(decision_reason)[:200],  # truncate long reasons
            "entry_price":     float(entry_price) if entry_price else 0.0,
            "vxx_ratio":       float(vxx_ratio),
            "regime_label":    str(regime_label),
            "side":            str(side or ""),
            "action_label":    str(action_label or ""),
            "trade_type":      str(trade_type or ""),
            "features":        {k: float(v) if isinstance(v, (int, float)) else 0.0
                                for k, v in features.items()} if features else {},
            "outcomes":        {f"+{h}d": None for h in FORWARD_HORIZONS_DAYS},
            "code_version":    CODE_VERSION,
            # PIPELINE 2026-10-16 (cockpit audit §4 layer 4): what the model
            # said about THIS candidate, which was returned by ml_score() and
            # dropped on the floor until now. None when deep_score passed no
            # trace (older callers); see sanitize_ml_trace().
            "ml":              sanitize_ml_trace(ml),
            "score_breakdown": _sanitize_score_breakdown(score_breakdown),
        }

        # Load, append, rotate, save
        records = _load_shadow_log()
        records.append(record)
        if len(records) > MAX_SHADOW_RECORDS:
            records = records[-MAX_SHADOW_RECORDS:]  # keep newest
        _save_shadow_log(records)
    except Exception as e:
        logger.debug(f"log_candidate failed for {ticker}: {e}")


def update_last_decision(
    ticker: str,
    decision: str,
    decision_reason: str = "",
    max_age_seconds: float = 120.0,
) -> bool:
    """
    Correct a just-logged candidate's decision bucket.

    log_candidate() runs inside deep_score(), before scan_market()'s
    downstream per-candidate filters (sector/correlation, spread) have a
    chance to reject a candidate that cleared MIN_SCORE. Those filters
    call this immediately after rejecting, so the shadow record reflects
    the REAL reason the candidate was never traded (decision buckets:
    "rejected_heat" for correlation/sector blocks, "rejected_other" for
    spread — see CLAUDE.md RULE REVIEW / RULE COST AUDIT). Mechanisms
    (the filters themselves) are untouched — this only corrects the
    learning-data label.

    Only ever moves a record OFF "taken", and only the most recent
    matching record within max_age_seconds — never touches an older
    record for the same ticker from a prior scan, so a slow caller can
    never mislabel unrelated history.

    Rate cost: ZERO API calls. Pure file I/O. Non-blocking: any failure
    is swallowed silently (logging must never break the trading loop).
    """
    try:
        if not ticker:
            return False
        records = _load_shadow_log()
        if not _relabel_latest_taken(records, ticker, decision, decision_reason,
                                     datetime.now(timezone.utc), max_age_seconds):
            return False
        return _save_shadow_log(records)
    except Exception as e:
        logger.debug(f"update_last_decision failed for {ticker}: {e}")
        return False


def _relabel_latest_taken(records: List[dict], ticker: str, decision: str,
                          decision_reason: str, now: datetime,
                          max_age_seconds: float) -> bool:
    """In-place relabel of the freshest "taken" record for ticker; no I/O."""
    ticker_u = str(ticker).upper()
    for record in reversed(records):
        if record.get("ticker") != ticker_u:
            continue
        if record.get("decision") != "taken":
            return False  # most recent record already resolved elsewhere
        try:
            ts = datetime.fromisoformat(str(record["timestamp"]).replace("Z", "+00:00"))
        except Exception:
            return False
        if (now - ts).total_seconds() > max_age_seconds:
            return False  # too old — not this scan's candidate
        record["decision"] = str(decision)
        record["decision_reason"] = str(decision_reason)[:200]
        return True
    return False


# [RULE-REVIEW] 2026-10-26: bot.ts's executeTrades admission loop applies
# ~15 more gates AFTER scan_market returns (Node score threshold, slots,
# size/exposure caps, sector correlation, exchange halt, retry/dup/BP,
# options-engine drops) and every one of them left the record "taken" —
# the bucket's win rate mixed never-ordered candidates with real orders,
# and none of those rules had a prevention-P&L. Node collects its
# rejections and sends them here in ONE call so the ~45MB log is loaded
# and saved once per admission loop, not once per gate. The window is
# wide (scan SIGALRM ~55s + per-candidate RPCs) but the guards in
# _relabel_latest_taken (freshest record only, "taken" only) still hold.
NODE_GATE_RELABEL_MAX_AGE_S = 1800.0


def update_decisions_batch(updates: list, max_age_seconds: float = NODE_GATE_RELABEL_MAX_AGE_S) -> dict:
    """
    Relabel several just-logged "taken" records in one load/save.

    updates: [{"ticker": "AMD", "decision": "rejected_size_cap",
               "reason": "..."}, ...] — entries missing a ticker or a
    decision are counted as skipped, never raised on. Same per-record
    semantics as update_last_decision(). Returns counts so the caller can
    audit the relabel; never raises (logging must not break the loop).
    """
    out = {"received": 0, "updated": 0, "skipped": 0, "saved": False}
    try:
        if not isinstance(updates, list) or not updates:
            return out
        out["received"] = len(updates)
        records = _load_shadow_log()
        now = datetime.now(timezone.utc)
        changed = 0
        for u in updates:
            if not isinstance(u, dict):
                out["skipped"] += 1
                continue
            ticker = u.get("ticker")
            decision = u.get("decision")
            if not ticker or not decision:
                out["skipped"] += 1
                continue
            if _relabel_latest_taken(records, ticker, str(decision),
                                     str(u.get("reason") or ""), now, max_age_seconds):
                changed += 1
            else:
                out["skipped"] += 1
        out["updated"] = changed
        if changed:
            out["saved"] = bool(_save_shadow_log(records))
        return out
    except Exception as e:
        logger.debug(f"update_decisions_batch failed: {e}")
        out["error"] = str(e)[:200]
        return out


def _fetch_historical_bars_batch(
    tickers: List[str], start_date: str, end_date: str
) -> Dict[str, List[dict]]:
    """
    Batch-fetch daily bars for many tickers in one Alpaca call.
    Returns {ticker: [bars]}. Alpaca endpoint supports up to ~100 symbols per call.
    Respects token bucket.
    """
    import requests
    if not tickers:
        return {}

    results: Dict[str, List[dict]] = {}
    # Chunk into BATCH_SIZE
    for i in range(0, len(tickers), BATCH_SIZE):
        chunk = tickers[i:i + BATCH_SIZE]
        _alpaca_bucket.acquire()  # rate-limit gate

        try:
            resp = requests.get(
                f"{ALPACA_DATA_URL}/v2/stocks/bars",
                params={
                    "symbols": ",".join(chunk),
                    "timeframe": "1Day",
                    "start": start_date,
                    "end": end_date,
                    "limit": 100,
                    "adjustment": "split",
                    "feed": bars_feed(),
                },
                headers=_alpaca_headers(),
                timeout=15,
            )

            if resp.status_code == 429:
                # Hard rate limit — back off hard and retry once
                logger.warning("Alpaca 429 during shadow backfill — sleeping 30s")
                time.sleep(30)
                _alpaca_bucket.acquire()
                resp = requests.get(
                    f"{ALPACA_DATA_URL}/v2/stocks/bars",
                    params={
                        "symbols": ",".join(chunk),
                        "timeframe": "1Day",
                        "start": start_date,
                        "end": end_date,
                        "limit": 100,
                        "adjustment": "split",
                        "feed": bars_feed(),
                    },
                    headers=_alpaca_headers(),
                    timeout=15,
                )

            if resp.status_code != 200:
                logger.warning(f"Alpaca bars batch failed: {resp.status_code}")
                continue

            data = resp.json().get("bars", {})
            for sym, bars in data.items():
                results[sym] = bars or []

        except Exception as e:
            logger.warning(f"Alpaca bars batch exception: {e}")
            continue

    return results


def _label_from_return(pct_return: float) -> int:
    """
    LEGACY: close-only labeling. Kept for backward compat.

    DEPRECATED 2026-05-03 (audit Finding #10): this function only looks at
    the close at horizon day, not the bot's actual stop (-4%) / target (+2%)
    rules. Use _label_from_path() below for new labeling.
    """
    if pct_return is None:
        return -1  # unknown
    if pct_return >= WIN_THRESHOLD_PCT:
        return 1
    else:
        return 0


def _label_from_path(entry_price: float, bars: List[dict],
                     win_pct: float = WIN_THRESHOLD_PCT,
                     stop_pct: float = LOSS_THRESHOLD_PCT,
                     max_bars: int = 5) -> tuple:
    """
    PATH-DEPENDENT LABELING 2026-05-03 (audit Finding #10).

    Walks daily OHLC bars from entry forward and determines the outcome
    using the bot's ACTUAL exit rules, not just the close at day N:

      win_pct:   take-profit threshold (+2.0% default)
      stop_pct:  stop-loss threshold   (-4.0% default, NEGATIVE)
      max_bars:  hold for at most this many TRADING DAYS (not calendar)

    On each trading day:
      - If HIGH crosses entry*(1+win_pct/100)  → label = 1 (win)
      - If LOW  crosses entry*(1+stop_pct/100) → label = 0 (loss)
      - If both on same day: conservative — label as 0 (we can't know
        intraday order from daily bars, so assume worst case)

    If neither threshold is hit within max_bars, label by the close price
    of the last bar:
      - close > entry*(1+win_pct/100) → 1
      - else                          → 0

    Returns (label, exit_price, exit_reason, exit_bar_date)
      label:     1 / 0 / -1
      exit_reason: "target" / "stop" / "timeout" / "ambiguous"

    This matches what manage_positions() actually does in production —
    so labels can train ML on signals the live bot can actually capture.
    """
    if not bars or entry_price <= 0:
        return (-1, None, "no_data", None)

    target_price = entry_price * (1 + win_pct / 100.0)
    stop_price = entry_price * (1 + stop_pct / 100.0)  # stop_pct is negative

    # Walk at most max_bars trading days (each bar = 1 trading day)
    for i, bar in enumerate(bars[:max_bars]):
        high = float(bar.get("h", 0) or 0)
        low = float(bar.get("l", 0) or 0)
        close = float(bar.get("c", 0) or 0)
        bar_date = (bar.get("t") or "")[:10]

        if high <= 0 or low <= 0:
            continue

        hit_target = high >= target_price
        hit_stop = low <= stop_price

        if hit_target and hit_stop:
            # Ambiguous — both touched on same day. Conservative: stop first.
            return (0, stop_price, "ambiguous", bar_date)
        if hit_stop:
            return (0, stop_price, "stop", bar_date)
        if hit_target:
            return (1, target_price, "target", bar_date)

    # Neither hit within max_bars → label by close of last bar (timeout)
    last_bar = bars[max_bars - 1] if len(bars) >= max_bars else bars[-1]
    last_close = float(last_bar.get("c", 0) or 0)
    last_date = (last_bar.get("t") or "")[:10]
    if last_close <= 0:
        return (-1, None, "no_close", last_date)
    pct_at_close = (last_close - entry_price) / entry_price * 100
    label = 1 if pct_at_close >= win_pct else 0
    return (label, last_close, "timeout", last_date)  # anything below threshold counts as non-win (conservative)


def _record_missing_price_attempt(rec: dict, horizon_key: str, stats: dict) -> None:
    """
    Track a failed backfill attempt for one (record, horizon) pair.

    Stored under "_bf_attempts" — a field distinct from "outcomes" so a
    single failed attempt does NOT get treated as "already_filled" (that
    would kill retries after one transient Alpaca hiccup). Only once
    MAX_BACKFILL_ATTEMPTS is reached does this write a terminal outcomes[]
    sentinel, which the normal "already_filled" skip then picks up on every
    future run — permanently retiring the pair without ever recording a
    fabricated win/loss (label=-1 is excluded from every win/loss tally).
    """
    attempts = rec.setdefault("_bf_attempts", {})
    n = attempts.get(horizon_key, 0) + 1
    attempts[horizon_key] = n
    if n >= MAX_BACKFILL_ATTEMPTS:
        rec.setdefault("outcomes", {})[horizon_key] = {
            "return_pct":   None,
            "label":        -1,
            "exit_price":   None,
            "exit_date":    None,
            "exit_reason":  "no_price_data",
            "label_method": "path_dependent_v2",
        }
        attempts.pop(horizon_key, None)
        stats["permanently_unlabelable"] += 1


def backfill_outcomes(max_records: int = 500) -> dict:
    """
    Fill in forward-return outcomes for shadow records that are old enough.

    For each record with at least horizon_days since its timestamp AND no
    outcome set for that horizon, look up the actual price on that date
    and compute hypothetical P&L.

    Args:
        max_records: cap on how many to process per call (rate-limit safety)

    Returns:
        dict with counts: {updated, already_filled, missing_price,
        permanently_unlabelable, skipped}

    Rate cost: ~3-5 batch Alpaca calls for 500 records (~50 tickers per batch).
    Well within the 180/min token budget.

    A (record, horizon) pair that fails MAX_BACKFILL_ATTEMPTS times (missing
    price data every attempt) gets a terminal outcomes[horizon_key] sentinel
    ({"label": -1, "exit_reason": "no_price_data"}) instead of being retried
    forever — see the MAX_BACKFILL_ATTEMPTS comment above. label=-1 is
    already excluded from win/loss tallies everywhere it's consumed
    (get_shadow_stats, load_shadow_data), so this only stops wasted retries;
    it never fabricates a win/loss.
    """
    stats = {"updated": 0, "already_filled": 0, "missing_price": 0,
             "permanently_unlabelable": 0, "skipped": 0, "total_records": 0}

    records = _load_shadow_log()
    stats["total_records"] = len(records)
    if not records:
        return stats

    now_utc = datetime.now(timezone.utc)

    # ── BUFFER MULTIPLIER 2026-05-03 (audit Finding #54) ───────────────────
    # Old code used calendar-day arithmetic for horizon (timedelta(days=N)).
    # That makes "5-day forward" mean 5 calendar days = 3-5 trading days
    # depending on weekends. ML feature horizons drifted ~25-30% through
    # the year. Now we fetch a wider window and walk by *trading days*
    # (i.e. just consecutive bars) using _label_from_path.
    #
    # Window: from entry to entry + horizon_calendar_days * 2 (cushion for
    # weekends + holidays). Then we feed the first `horizon` bars to the
    # path labeler, which is now horizon-in-TRADING-DAYS as intended.
    HORIZON_CALENDAR_BUFFER = 2.5  # trading days × this = calendar days to fetch

    processed = 0
    backfill_jobs = []  # list of (idx, horizon_key, horizon_trading_days, ticker, entry_price, ts)
    for idx, rec in enumerate(records):
        if processed >= max_records:
            break

        try:
            ts_str = rec.get("timestamp", "")
            if not ts_str:
                continue
            rec_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            continue

        for horizon in FORWARD_HORIZONS_DAYS:
            horizon_key = f"+{horizon}d"
            # Skip if already filled
            if rec.get("outcomes", {}).get(horizon_key) is not None:
                stats["already_filled"] += 1
                continue

            # We need at least horizon_trading_days * ~1.4 calendar days
            # to have observed the full window. Skip if too recent.
            min_observable = rec_time + timedelta(
                days=int(horizon * HORIZON_CALENDAR_BUFFER) + 1
            )
            if now_utc < min_observable:
                stats["skipped"] += 1
                continue

            ticker = rec.get("ticker", "")
            entry_price = float(rec.get("entry_price", 0) or 0)
            if not ticker or entry_price <= 0:
                continue

            backfill_jobs.append((idx, horizon_key, horizon, ticker, entry_price, rec_time))
            processed += 1

    if not backfill_jobs:
        return stats

    # REPAIR 2026-08-07 (KNOWN BROKEN #10/#20 follow-up): this used to group
    # jobs by TICKER and fetch one Alpaca call per unique ticker, ignoring
    # _fetch_historical_bars_batch's own ability to fetch up to BATCH_SIZE
    # (50) tickers in a single call. With ~13k archived shadow records and
    # only `max_records` new horizon-jobs processed per nightly run, that
    # per-ticker-call pattern made ticker count (not the Alpaca rate budget)
    # the effective throughput ceiling, and the array-order scan (oldest
    # record first) meant decision buckets that started existing later in
    # the archive's timeline — e.g. rejected_masterkill, first logged
    # 2026-07-11 per KNOWN BROKEN #3/#20 — sat unreached for months at that
    # rate. Grouping by WINDOW instead (each job's own window is already
    # day-granularity via strftime, so jobs entered on the same calendar
    # day at the same horizon collapse to one key) lets many tickers from
    # the same scan cycle share one batched call, cutting call count
    # roughly by the number of tickers sharing a scan cycle instead of
    # doing one call each. Label OUTPUT is unchanged for every record: each
    # job still only reads bars strictly after its own entry date, capped
    # at its own horizon — this only changes how many Alpaca calls it takes
    # to fetch the same bars, not what _label_from_path computes from them.
    jobs_by_window: Dict[tuple, List[tuple]] = defaultdict(list)
    for job in backfill_jobs:
        _, _, horizon, _, _, rec_time = job
        window_start = (rec_time - timedelta(days=2)).strftime("%Y-%m-%d")
        window_end = (rec_time + timedelta(
            days=int(horizon * HORIZON_CALENDAR_BUFFER) + 3
        )).strftime("%Y-%m-%d")
        jobs_by_window[(window_start, window_end)].append(job)

    for (window_start, window_end), jobs in jobs_by_window.items():
        tickers = sorted(set(j[3] for j in jobs))
        bars_by_ticker = _fetch_historical_bars_batch(
            tickers, window_start, window_end
        )

        for idx, horizon_key, horizon, ticker, entry_price, rec_time in jobs:
            all_bars = bars_by_ticker.get(ticker, [])
            if not all_bars:
                stats["missing_price"] += 1
                _record_missing_price_attempt(records[idx], horizon_key, stats)
                continue

            entry_date_str = rec_time.strftime("%Y-%m-%d")
            # Bars at or after entry date — we want trading-day forward window
            forward_bars = [
                b for b in all_bars
                if (b.get("t") or "")[:10] > entry_date_str
            ]
            if not forward_bars:
                stats["missing_price"] += 1
                _record_missing_price_attempt(records[idx], horizon_key, stats)
                continue

            label, exit_price, exit_reason, exit_date = _label_from_path(
                entry_price=entry_price,
                bars=forward_bars,
                win_pct=WIN_THRESHOLD_PCT,
                stop_pct=LOSS_THRESHOLD_PCT,
                max_bars=horizon,  # trading days, not calendar
            )

            if label == -1:
                stats["missing_price"] += 1
                _record_missing_price_attempt(records[idx], horizon_key, stats)
                continue

            # Compute return (entry → exit)
            pct_return = ((exit_price - entry_price) / entry_price * 100
                          if exit_price else 0.0)

            records[idx].setdefault("outcomes", {})
            records[idx]["outcomes"][horizon_key] = {
                "return_pct":  round(pct_return, 3),
                "label":       label,
                "exit_price":  round(float(exit_price), 4) if exit_price else None,
                "exit_date":   exit_date,
                "exit_reason": exit_reason,  # "target" / "stop" / "timeout" / "ambiguous"
                "label_method": "path_dependent_v2",  # vs legacy "close_only"
            }
            stats["updated"] += 1

    # Save updated records atomically. missing_price also mutates records
    # (via _record_missing_price_attempt's retry counter / terminal
    # sentinel) even when "updated" stays 0, so it must persist too — a
    # nightly run with 100% missing_price would otherwise reset every
    # attempt counter back to 0 on the next boot and never reach
    # MAX_BACKFILL_ATTEMPTS.
    if stats["updated"] > 0 or stats["missing_price"] > 0:
        _save_shadow_log(records)

    # Also write a stats summary for dashboards
    try:
        summary = get_shadow_stats()
        with open(SHADOW_STATS_PATH, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception:
        pass

    logger.info(f"Shadow backfill: {stats}")
    return stats


def load_shadow_data(
    horizon_days: int = 10,
    min_records: int = 50,
    shadow_weight: float = 0.3,
) -> Optional[Tuple]:
    """
    Build training data from shadow records for a meta-model.

    Returns (X, y, sample_weights) suitable for LightGBM training, or None
    if not enough labeled data yet.

    Args:
        horizon_days:   which outcome horizon to use as label (5, 10, or 20)
        min_records:    minimum labeled records required (default 50)
        shadow_weight:  per-sample weight multiplier. Default 0.3 reflects
                        that shadow trades didn't actually execute — they're
                        useful signal but less reliable than real fills.

    Usage from ml_model_v2.py:
        from shadow_portfolio import load_shadow_data
        shadow = load_shadow_data(horizon_days=10)
        if shadow is not None:
            X_shadow, y_shadow, w_shadow = shadow
            # Concatenate with real feedback data
            X_combined = np.vstack([X_real, X_shadow])
            y_combined = np.concatenate([y_real, y_shadow])
            w_combined = np.concatenate([w_real, w_shadow])
    """
    import numpy as np

    horizon_key = f"+{horizon_days}d"
    records = _load_shadow_log()

    # Filter to records with this horizon filled in
    labeled = []
    for rec in records:
        outcome = rec.get("outcomes", {}).get(horizon_key)
        if outcome is None or not isinstance(outcome, dict):
            continue
        label = outcome.get("label")
        if label not in (0, 1):
            continue
        features = rec.get("features", {})
        if not features:
            continue
        labeled.append((features, label, rec.get("timestamp", "")))

    if len(labeled) < min_records:
        return None

    # Need to import FEATURE_COLS from ml_model_v2 for consistent ordering
    try:
        from ml_model_v2 import FEATURE_COLS
    except ImportError:
        return None

    # Build X/y/weights
    X_rows, y_rows, w_rows = [], [], []
    current_ts = time.time()
    for features, label, ts_str in labeled:
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = current_ts
        age_days = (current_ts - ts) / 86400
        decay = 2 ** (-age_days / 30.0)  # 30-day half-life, matches main loop

        row = [float(features.get(col, 0) or 0) for col in FEATURE_COLS]
        X_rows.append(row)
        y_rows.append(label)
        w_rows.append(shadow_weight * decay)

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=np.int8),
        np.array(w_rows, dtype=np.float32),
    )


def _change_pct_distribution(abs_values: List[float]) -> dict:
    """
    Descriptive stats over |change_pct_today| for "taken" shadow records.

    Built because _change_pct_band_stats()'s over_threshold band sat at
    candidate_count: 0 across 10,000+ taken records for multiple days
    (research/open_questions.md KNOWN BROKEN #10) — without this, a future
    session re-checking that band can't tell "genuinely never happens
    among taken candidates" from "not enough data yet, keep waiting."
    max/p99 answer that directly instead of requiring another blind
    wait-and-recheck cycle.
    """
    n = len(abs_values)
    if n == 0:
        return {"n": 0, "max": None, "p50": None, "p95": None, "p99": None,
                "count_over_20": 0, "count_over_30": 0}

    ordered = sorted(abs_values)

    def _pct(p: float) -> float:
        idx = min(n - 1, int(round(p * (n - 1))))
        return ordered[idx]

    return {
        "n": n,
        "max": round(ordered[-1], 2),
        "p50": round(_pct(0.50), 2),
        "p95": round(_pct(0.95), 2),
        "p99": round(_pct(0.99), 2),
        "count_over_20": sum(1 for v in abs_values if v > 20),
        "count_over_30": sum(1 for v in abs_values if v > 30),
    }


def _change_pct_band_stats(records: List[dict], min_n: int = 5) -> dict:
    """
    KNOWN BROKEN #10's queued evidence check (research/open_questions.md):
    system_config.py's MAX_CHANGE_PCT ("Skip stocks already up/down 35%+")
    is read nowhere in the trading path — a dead gate that LOOKS live to
    anyone reading system_config.py. Rather than guess, split "taken"
    shadow records by whether |change_pct_today| exceeds that same
    threshold (imported live, not re-hardcoded, so a future threshold
    edit updates this query too) and compare win rate on each side, per
    horizon. Only "taken" records carry a real outcome under the bot's
    actual exit rules — this is the natural experiment of what the dead
    gate would have skipped, using data that already exists because the
    gate never fired.
    """
    try:
        from system_config import BASE_CONFIG
        threshold = float(BASE_CONFIG.get("MAX_CHANGE_PCT", 35.0))
    except Exception:
        threshold = 35.0

    bands: Dict[str, list] = {"over_threshold": [], "at_or_under_threshold": []}
    taken_abs_change_pcts: List[float] = []
    for r in records:
        if r.get("decision") != "taken":
            continue
        cp = abs(float(r.get("features", {}).get("change_pct_today", 0) or 0))
        taken_abs_change_pcts.append(cp)
        bands["over_threshold" if cp > threshold else "at_or_under_threshold"].append(r)

    by_band = {}
    for band_name, recs in bands.items():
        win_rate = {}
        for h in FORWARD_HORIZONS_DAYS:
            key = f"+{h}d"
            wins = sum(1 for r in recs
                       if isinstance(r.get("outcomes", {}).get(key), dict)
                       and r["outcomes"][key].get("label") == 1)
            losses = sum(1 for r in recs
                         if isinstance(r.get("outcomes", {}).get(key), dict)
                         and r["outcomes"][key].get("label") == 0)
            total = wins + losses
            if total >= min_n:
                win_rate[key] = {"win_rate": round(wins / total * 100, 1), "n": total}
        by_band[band_name] = {"candidate_count": len(recs), "win_rate": win_rate}

    return {
        "threshold": threshold,
        "by_band": by_band,
        "taken_distribution": _change_pct_distribution(taken_abs_change_pcts),
    }


# REPAIR 2026-09-01 (KNOWN BROKEN item #20 follow-up, RULE-REVIEW evidence
# gate): tiered_strategy.master_kill_switch() has THREE independent kill
# conditions (portfolio drawdown, daily loss limit, whole-portfolio invested
# ceiling) but log_masterkill_csp_shadow() logs every rejected Tier-1 CSP
# candidate under one flat "rejected_masterkill" decision regardless of
# which condition fired. Live evidence this session
# (win_rate_by_decision["rejected_masterkill"]: 85.8% @ +5d n=127, 91.2% @
# +10d n=114 — far above "taken"'s 42-45%) looks like a strong loosening
# signal for item #20's (b) finding (the exposure ceiling kills CSP even
# though CSP holds none of the exposure that trips it) — but REASONING
# STANDARD #1 (variables interact, don't reason about an aggregate in
# isolation) applies directly: that same aggregate also includes drawdown
# and daily-loss kills, where continuing to sell CSPs would be a real,
# not spurious, risk decision. Acting on the flat number would risk
# loosening a genuine safety trigger on evidence that actually only
# supports loosening the unrelated exposure-ceiling one. This function
# splits the SAME already-collected win/loss labels by which kill reason
# produced them (parsed from decision_reason's existing, already-fixed
# message prefixes in master_kill_switch — no new logging field needed,
# no re-run required), so a future session can check whether the high win
# rate survives when isolated to reason="exposure_ceiling" before treating
# it as evidence for a threshold/design change. Visibility only — reused
# by get_shadow_stats() below, never consulted by the kill switch itself.
_MASTERKILL_REASON_PREFIXES = {
    "Portfolio DD kill": "drawdown",
    "Daily loss limit": "daily_loss",
    "Portfolio invested": "exposure_ceiling",
}


def _masterkill_reason_category(decision_reason: str) -> str:
    """Classify a rejected_masterkill record's decision_reason string into
    the kill condition that produced it. See _MASTERKILL_REASON_PREFIXES'
    docstring above for why this exists. Falls back to "unknown" for any
    reason text that doesn't match a known master_kill_switch() message
    (e.g. records logged before a future wording change)."""
    reason = decision_reason or ""
    for prefix, category in _MASTERKILL_REASON_PREFIXES.items():
        if reason.startswith(prefix):
            return category
    return "unknown"


def _win_rate_by_masterkill_reason(records: List[dict], min_n: int = 5) -> dict:
    by_category: Dict[str, list] = defaultdict(list)
    for r in records:
        if r.get("decision") != "rejected_masterkill":
            continue
        by_category[_masterkill_reason_category(r.get("decision_reason", ""))].append(r)

    result = {}
    for category, recs in by_category.items():
        win_rate = {}
        for h in FORWARD_HORIZONS_DAYS:
            key = f"+{h}d"
            wins = sum(1 for r in recs
                       if isinstance(r.get("outcomes", {}).get(key), dict)
                       and r["outcomes"][key].get("label") == 1)
            losses = sum(1 for r in recs
                         if isinstance(r.get("outcomes", {}).get(key), dict)
                         and r["outcomes"][key].get("label") == 0)
            total = wins + losses
            if total >= min_n:
                win_rate[key] = {"win_rate": round(wins / total * 100, 1), "n": total}
        result[category] = {"candidate_count": len(recs), "win_rate": win_rate}
    return result


def get_shadow_stats() -> dict:
    """
    Summary stats for dashboards / health checks.
    Shows learning velocity, label distribution, decision breakdown.
    """
    records = _load_shadow_log()
    if not records:
        return {
            "total_records": 0,
            "labeled": {},
            "by_decision": {},
            "by_regime": {},
            "win_rate_shadow": {},
            "note": "No shadow data yet. Backfill runs nightly.",
        }

    by_decision = defaultdict(int)
    by_regime = defaultdict(int)
    labeled_by_horizon = {f"+{h}d": {"wins": 0, "losses": 0, "unknown": 0}
                          for h in FORWARD_HORIZONS_DAYS}

    for rec in records:
        by_decision[rec.get("decision", "unknown")] += 1
        by_regime[rec.get("regime_label", "unknown")] += 1

        outcomes = rec.get("outcomes", {})
        for h in FORWARD_HORIZONS_DAYS:
            key = f"+{h}d"
            out = outcomes.get(key)
            if out is None:
                labeled_by_horizon[key]["unknown"] += 1
            elif isinstance(out, dict):
                lbl = out.get("label")
                if lbl == 1:
                    labeled_by_horizon[key]["wins"] += 1
                elif lbl == 0:
                    labeled_by_horizon[key]["losses"] += 1
                else:
                    labeled_by_horizon[key]["unknown"] += 1

    # Compute win rate per horizon AND per decision bucket
    win_rate_by_decision = {}
    for decision in by_decision:
        taken_recs = [r for r in records if r.get("decision") == decision]
        for h in FORWARD_HORIZONS_DAYS:
            key = f"+{h}d"
            wins = sum(1 for r in taken_recs
                       if isinstance(r.get("outcomes", {}).get(key), dict)
                       and r["outcomes"][key].get("label") == 1)
            losses = sum(1 for r in taken_recs
                         if isinstance(r.get("outcomes", {}).get(key), dict)
                         and r["outcomes"][key].get("label") == 0)
            total = wins + losses
            if total >= 5:
                win_rate_by_decision.setdefault(decision, {})[key] = {
                    "win_rate": round(wins / total * 100, 1),
                    "n": total,
                }

    # REPAIR 2026-08-19 (KNOWN BROKEN #20 follow-up): win_rate_by_decision's
    # n>=5 floor makes a starved decision bucket indistinguishable from a
    # slow-but-progressing one — both just show up as "absent". Live evidence
    # this session: rejected_masterkill/rejected_heat/rejected_other (156 +
    # 493 + 350 = 999 records, oldest dating to 2026-04-20) have carried ZERO
    # real win/loss labels at any horizon for weeks, across two separate
    # nightly-backfill throughput fixes (2026-08-07 windowed batching,
    # 2026-08-17 permanent-failure retirement) — but the aggregate stats gave
    # no way to tell WHY: never reached by the array-order scan frontier,
    # actively failing on missing price data (pre-retirement), or already
    # given up on (label=-1, correctly excluded from win_rate forever). This
    # breaks that ambiguity apart per decision bucket per horizon so the next
    # check of this evidence gate is diagnostic, not just another "still
    # empty".
    backfill_progress_by_decision = {}
    for decision in by_decision:
        recs = [r for r in records if r.get("decision") == decision]
        for h in FORWARD_HORIZONS_DAYS:
            key = f"+{h}d"
            labeled = 0
            permanently_unlabelable = 0
            pending = 0
            not_yet_attempted = 0
            for r in recs:
                out = r.get("outcomes", {}).get(key)
                if isinstance(out, dict):
                    lbl = out.get("label")
                    if lbl in (0, 1):
                        labeled += 1
                    elif lbl == -1:
                        permanently_unlabelable += 1
                    else:
                        not_yet_attempted += 1
                elif r.get("_bf_attempts", {}).get(key, 0) > 0:
                    pending += 1
                else:
                    not_yet_attempted += 1
            backfill_progress_by_decision.setdefault(decision, {})[key] = {
                "labeled": labeled,
                "permanently_unlabelable": permanently_unlabelable,
                "pending_retry": pending,
                "not_yet_attempted": not_yet_attempted,
            }

    return {
        "total_records":         len(records),
        "oldest":                records[0].get("timestamp") if records else None,
        "newest":                records[-1].get("timestamp") if records else None,
        "by_decision":           dict(by_decision),
        "by_regime":             dict(by_regime),
        "labeled_by_horizon":    labeled_by_horizon,
        "win_rate_by_decision":  win_rate_by_decision,
        "backfill_progress_by_decision": backfill_progress_by_decision,
        "win_rate_by_change_pct_band": _change_pct_band_stats(records),
        "win_rate_by_masterkill_reason": _win_rate_by_masterkill_reason(records),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EXIT-ALIGNED LABELER ("exit_aligned_v1") — a SECOND ruler, stored next to
#  the old one. [RULE-REVIEW] measurement change, 2026-10-14.
# ══════════════════════════════════════════════════════════════════════════════
# WHY. The old ruler (_label_from_path: +2% target / -4% stop / N-bar
# timeout) says it "matches bot's take-profit" / "stop-loss" (the
# WIN_THRESHOLD_PCT / LOSS_THRESHOLD_PCT comments above). It does not, and
# neither does system_config.py's ATR_STOP_MULTIPLIER / TIME_STOP_DAYS block
# (read by no exit code) nor backtest_v2.py's 1.5xATR / 2R model. The ONLY
# code that places stock exits live is server/bot.ts checkPositionOnTick
# (fed by syncMonitoredPositions' stop/TP levels): a 2xATR R-ladder with
# scale-outs in thirds, breakeven, phase-dependent trails, a conditional
# take-profit and a 7-calendar-day flat-position time stop. This block
# mirrors THAT policy on daily bars.
#
# MEASUREMENT INTEGRITY. The old ruler is untouched: _label_from_path,
# backfill_outcomes, _fetch_historical_bars_batch, the 'outcomes' and
# '_bf_attempts' record keys, get_shadow_stats(), load_shadow_data() (the ML
# training input) and every pre-existing shadow_band_report() key. This
# ruler writes ONLY rec[XA_KEY] and its own retry counter rec[XA_ATTEMPTS_KEY]
# and is surfaced ONLY under shadow_band_report()["exit_aligned"].
#
# Every mirrored constant names the bot.ts expression it copies;
# test_shadow_exit_aligned.py's drift pin asserts those literals still exist
# in server/bot.ts, so a live exit change fails CI instead of silently
# leaving this ruler stale (which is exactly how the old ruler's
# "matches bot's take-profit" comment went wrong).
XA_KEY = "outcome_exit_aligned"
XA_ATTEMPTS_KEY = "_bf_attempts_xa"
XA_LABEL_METHOD = "exit_aligned_v1"

XA_CAP_BARS = 20                 # censoring cap (trading bars after the entry ET session); live has no horizon
XA_RISK_ATR_MULT = 2.0           # bot.ts: `ps.initial_risk_pct || atrPct * 2.0` (bot_engine.py: atr_pct * 2.0, UNclamped)
XA_P1_STOP_ATR_MULT = 2.0        # bot.ts: `Math.max(1.5, Math.min(atrPct * 2.0, 8.0))`
XA_P1_STOP_MIN_PCT = 1.5         #   ... lower clamp
XA_P1_STOP_MAX_PCT = 8.0         #   ... upper clamp
XA_SCALE1_R = 1.0                # bot.ts: `const scaleOut1R = 1.0`
XA_SCALE2_R = {"bullish": 3.0, "bearish": 1.5, "neutral": 2.0}         # bot.ts: `isBullish ? 3.0 : (isBearish ? 1.5 : 2.0)` (scaleOut2R)
XA_TRAIL4_ATR_MULT = {"bullish": 3.0, "bearish": 1.5, "neutral": 2.0}  # bot.ts: same literal (trailAtrMult), from the highest PRICE
XA_P2_TRAIL_MIN_PCT, XA_P2_TRAIL_ATR_MULT = 1.5, 1.5   # bot.ts: `Math.max(1.5, atr * 1.5)`, from peak P&L
XA_P3_TRAIL_MIN_PCT, XA_P3_TRAIL_ATR_MULT = 1.0, 1.0   # bot.ts: `Math.max(1.0, atr * 1.0)`, from peak P&L
XA_TP_MIN_PCT, XA_TP_ATR_MULT, XA_TP_MAX_PCT = 4.0, 3.0, 15.0  # bot.ts: `Math.max(4.0, Math.min(atrPct * 3.0, 15.0))`
XA_TIME_STOP_DAYS = 7            # bot.ts: `daysHeld >= 7 && Math.abs(pnlPct) < 2.0` (CALENDAR days)
XA_TIME_STOP_BAND_PCT = 2.0      #   ... the |pnl| band
XA_TRANCHE = 1.0 / 3.0           # bot.ts: `Math.floor(pos.originalQty / 3)` per scale-out
XA_ATR_PERIOD = 14               # bot_engine._get_atr: Wilder ATR(14), seeded with the mean of the first 14 TRs
XA_ATR_LOOKBACK_DAYS = 60        # bot_engine._get_atr: lookback_days = max(60, period * 4)
XA_ATR_FEATURE_DEFAULT = 3.0     # bot_engine.deep_score: `_atr_pct = 3.0  # default` when ATR was unavailable
# ATR source order. Both are decision-time (no lookahead): features.atr_pct is
# deep_score's scan-time Wilder ATR (its bars include the session's partial
# bar up to the scan — exactly what live's _get_atr sees on the first manage
# cycle, so it is the closest analogue of the live R unit); the prior-bars
# ATR uses only sessions strictly before the entry session. False = always
# use prior bars (features.atr_pct is then ignored). atr_pct_prior is stored
# either way, so the two can be compared on the archive (a DATA-gate check).
XA_ATR_PREFER_FEATURE = True
XA_SCALE_GUARD = (0.85, 1.15)    # entry_price must sit inside the split-adjusted entry/next-session range
# Calendar-day eligibility mirrors the old +20d convention exactly: rec_time +
# int(h * HORIZON_CALENDAR_BUFFER) + 1 with h = XA_CAP_BARS and the buffer
# 2.5 that backfill_outcomes() keeps as a LOCAL (HORIZON_CALENDAR_BUFFER).
# Not shared on purpose: tightening that buffer (trading-day-aware
# eligibility + a bar-completeness guard) is its own [RULE-REVIEW] change.
XA_CALENDAR_BUFFER = 2.5
XA_ELIGIBLE_CALENDAR_DAYS = int(XA_CAP_BARS * XA_CALENDAR_BUFFER) + 1      # 51
XA_WINDOW_END_DAYS = int(XA_CAP_BARS * XA_CALENDAR_BUFFER) + 3             # 53
XA_FETCH_LIMIT = 10_000          # data points per page (Alpaca's multi-symbol limit counts ALL symbols)
XA_MAX_PAGES = 10                # next_page_token pages per 50-symbol chunk
XA_ROUND_TRIP_COST_PCT = 0.10    # system_config SLIPPAGE_PCT 0.0005 = 0.05% per side, x2 sides
# ── Nightly-job sizing: sized to the cap that ACTUALLY applies ──────────────
# server/bot.ts checkShadowBackfillExitAligned runs this job through
# execPythonSerialized. A command isTrivialPython() calls trivial (<= 150
# chars, no heavy import) skips the Python mutex and is SIGKILLed at 10s; this
# job's command is deliberately longer, so it holds the mutex and is SIGKILLed
# at min(timeout, 90000) = 90s (server/shadowBackfillExitAligned.test.ts
# derives that cap from the real command string). Budget: XA_JOB_TIME_BUDGET_S
# of wall time from function entry (log load + fetching + checkpoints; no
# Alpaca request STARTS unless its timeout fits before that deadline, and the
# 30s 429 back-off only runs if it fits too), then at most one merge-save,
# for which XA_SAVE_RESERVE_S is reserved: 60 + 15 = 75s < 90s. Even so, a
# kill must never lose finished work, so results are merge-saved every
# XA_CHECKPOINT_WINDOWS windows or XA_CHECKPOINT_SECONDS, whichever is first
# (an os.replace()d atomic write: a kill mid-save leaves the previous file).
XA_REQUEST_TIMEOUT_S = 15        # per Alpaca call (same as _fetch_historical_bars_batch)
XA_429_BACKOFF_S = 30            # same back-off as _fetch_historical_bars_batch
XA_JOB_TIME_BUDGET_S = 60.0
XA_SAVE_RESERVE_S = 15.0         # one re-read + json.dump of a ~45MB log: ~2s measured locally, x7 headroom
XA_CHECKPOINT_WINDOWS = 5
XA_CHECKPOINT_SECONDS = 20.0
_XA_TOL = 1e-9                  # pct-point slack so a level's own float round-trip still counts as touched
_XA_NUDGE = 1e-9                 # relative step past a level crossed INSIDE a continuous leg


def _xa_regime_class(rec: dict) -> str:
    """bot.ts isBullish/isBearish class of the regime AT ENTRY.

    Live reads stop_state.regime, which bot_engine.manage_positions()
    rewrites every cycle from system_config.get_market_regime() — the
    5-level taxonomy (PANIC/BEAR/CAUTION/NEUTRAL/BULL), NOT the Markov
    rec['regime_label']. Decision-time inputs only: the record's vxx_ratio
    and features.spy_vs_ma50 (1.0 when the record has no features).
    """
    regime = "NEUTRAL"
    try:
        from system_config import get_market_regime
        vxx = float(rec.get("vxx_ratio", 1.0) or 1.0)
        spy = float((rec.get("features") or {}).get("spy_vs_ma50", 1.0) or 1.0)
        regime = str(get_market_regime(vxx, spy))
    except Exception as e:
        logger.debug(f"xa regime classification fell back to NEUTRAL: {e}")
    if regime in ("BULL", "NEUTRAL_BULL"):
        return "bullish"
    if regime in ("BEAR", "PANIC", "CAUTION"):
        return "bearish"
    return "neutral"


def _xa_entry_date_et(ts_str: str) -> Optional[str]:
    """America/New_York calendar date of a record's UTC timestamp.

    Alpaca daily-bar 't' values are midnight ET, so t[:10] is the ET session
    date. The old backfill compares against the UTC date instead, which
    skips one session for scans between 20:00 and 24:00 ET — this ruler
    uses the ET date (same 'strictly after the entry session' rule)."""
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
    except Exception as e:
        logger.debug(f"xa: unparseable timestamp {ts_str!r}: {e}")
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception as e:
        logger.debug(f"xa: zoneinfo unavailable ({e}); using fixed UTC-5")
        return (dt.astimezone(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%d")


def _xa_prior_atr_pct(prior_bars: List[dict], entry_price: float,
                      period: int = XA_ATR_PERIOD) -> Optional[float]:
    """Wilder ATR(period) as % of entry_price, from bars STRICTLY BEFORE the
    entry session only (the caller filters) — bot_engine._get_atr's exact
    method. None with fewer than period+1 bars."""
    if entry_price <= 0 or len(prior_bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(prior_bars)):
        h = float(prior_bars[i].get("h", 0) or 0)
        lo = float(prior_bars[i].get("l", 0) or 0)
        pc = float(prior_bars[i - 1].get("c", 0) or 0)
        trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    if atr <= 0:
        return None
    return atr / entry_price * 100.0


class _XaPosition:
    """One long position under the live exit policy. tick() is a line-by-line
    mirror of server/bot.ts checkPositionOnTick for a long stock position
    (P&L in percent, 'R' = initialRiskPct)."""

    def __init__(self, entry: float, atr_pct: float, regime_class: str):
        self.entry = entry
        self.atr = atr_pct
        self.risk = atr_pct * XA_RISK_ATR_MULT
        self.p1_stop = max(XA_P1_STOP_MIN_PCT, min(atr_pct * XA_P1_STOP_ATR_MULT, XA_P1_STOP_MAX_PCT))
        self.tp = max(XA_TP_MIN_PCT, min(atr_pct * XA_TP_ATR_MULT, XA_TP_MAX_PCT))
        self.regime_class = regime_class
        self.scale2_r = XA_SCALE2_R[regime_class]
        self.trail4 = XA_TRAIL4_ATR_MULT[regime_class]
        self.phase = 1
        self.peak_pnl = 0.0          # bot.ts addPositionToMonitor: highestPnl: 0
        self.peak_price = entry
        self.scales = 0
        self.breakeven = False
        self.fills: List[Tuple[float, float]] = []   # (weight, price) of each tranche exit
        self.exit_reason: Optional[str] = None
        self.max_px = entry
        self.min_px = entry

    def _pnl(self, px: float) -> float:
        return (px - self.entry) / self.entry * 100.0

    def _close(self, px: float, reason: str) -> None:
        self.fills.append((1.0 - XA_TRANCHE * self.scales, px))
        self.exit_reason = reason

    def tick(self, px: float, time_eligible: bool) -> Optional[str]:
        """Returns 'scale' (a tranche sold — the caller re-ticks at the same
        price, standing in for the next 1-minute bar), an exit reason, or None."""
        self.max_px = max(self.max_px, px)
        self.min_px = min(self.min_px, px)
        pnl = self._pnl(px)
        self.peak_pnl = max(self.peak_pnl, pnl)
        self.peak_price = max(self.peak_price, px)
        # phases only move up, from the PEAK R multiple
        if self.peak_pnl >= 3.0 * self.risk - _XA_TOL and self.phase < 4:
            self.phase = 4
        elif self.peak_pnl >= 2.0 * self.risk - _XA_TOL and self.phase < 3:
            self.phase = 3
        elif self.peak_pnl >= 1.0 * self.risk - _XA_TOL and self.phase < 2:
            self.phase = 2
        # scale-outs (bot.ts returns right after one: no full exit on that tick)
        if self.scales < 2:
            if self.scales == 0 and pnl >= XA_SCALE1_R * self.risk - _XA_TOL:
                self.fills.append((XA_TRANCHE, px))
                self.scales = 1
                self.breakeven = True
                return "scale"
            if self.scales == 1 and pnl >= self.scale2_r * self.risk - _XA_TOL:
                self.fills.append((XA_TRANCHE, px))
                self.scales = 2
                return "scale"
        reason: Optional[str] = None
        if self.phase >= 2:
            if self.phase >= 4:
                dd_from_high = (self.peak_price - px) / self.peak_price * 100.0
                if dd_from_high >= self.atr * self.trail4 - _XA_TOL:
                    reason = "trail"
            else:
                trail = (max(XA_P3_TRAIL_MIN_PCT, self.atr * XA_P3_TRAIL_ATR_MULT) if self.phase >= 3
                         else max(XA_P2_TRAIL_MIN_PCT, self.atr * XA_P2_TRAIL_ATR_MULT))
                if self.peak_pnl - pnl >= trail - _XA_TOL:
                    reason = "trail"
            if reason is None and self.breakeven and pnl < -_XA_TOL:
                reason = "breakeven"
        elif pnl <= -self.p1_stop + _XA_TOL:
            reason = "stop"
        if reason is None and pnl >= self.tp - _XA_TOL and self.phase < 3 and self.scales >= 2:
            reason = "take_profit"
        if reason is None and time_eligible and abs(pnl) < XA_TIME_STOP_BAND_PCT - _XA_TOL:
            reason = "time_stop"
        if reason is not None:
            self._close(px, reason)
        return reason

    def settle(self, px: float, time_eligible: bool) -> bool:
        """tick() at px, re-ticking after each scale-out. True once closed."""
        for _ in range(3):
            if self.tick(px, time_eligible) != "scale":
                break
        return self.exit_reason is not None

    def _levels(self, time_eligible: bool) -> List[float]:
        """Every price at which some tick() condition can flip, given the
        current state (on a falling leg the peak is fixed; on a rising leg the
        trail levels sit below the price and cannot fire)."""
        e = self.entry
        lv = [e, e * (1 + self.tp / 100.0), e * (1 - self.p1_stop / 100.0)]
        lv += [e * (1 + k * self.risk / 100.0) for k in (1.0, 2.0, 3.0, self.scale2_r)]
        if time_eligible:
            lv += [e * (1 - XA_TIME_STOP_BAND_PCT / 100.0), e * (1 + XA_TIME_STOP_BAND_PCT / 100.0)]
        lv.append(e * (1 + (self.peak_pnl - max(XA_P2_TRAIL_MIN_PCT, self.atr * XA_P2_TRAIL_ATR_MULT)) / 100.0))
        lv.append(e * (1 + (self.peak_pnl - max(XA_P3_TRAIL_MIN_PCT, self.atr * XA_P3_TRAIL_ATR_MULT)) / 100.0))
        lv.append(self.peak_price * (1 - self.atr * self.trail4 / 100.0))
        return [x for x in lv if x > 0]

    def leg(self, a: float, b: float, time_eligible: bool) -> bool:
        """Continuous move a -> b: tick just past every level crossed, in
        order, then at b. A level crossed inside the leg fills at the level
        (+/- a 1e-9 relative nudge). True once closed."""
        cur = a
        for _ in range(64):
            if b == cur:
                return False
            up = b > cur
            # A level exactly AT the current point counts: a strict condition
            # (|pnl| < 2, pnl < 0) that did not fire AT the level fires on the
            # very next tick past it. The nudge keeps every step strictly
            # moving toward b, so no level is visited twice.
            inside = [x for x in self._levels(time_eligible) if (cur <= x < b if up else b < x <= cur)]
            if inside:
                nxt = min(inside) if up else max(inside)
                px = nxt * (1 + _XA_NUDGE) if up else nxt * (1 - _XA_NUDGE)
                if (up and px > b) or (not up and px < b):
                    px = b
            else:
                px = b
            if self.settle(px, time_eligible):
                return True
            cur = px
        return self.settle(b, time_eligible)


def _label_exit_aligned(entry_price: float, fwd_bars: List[dict], atr_pct: float,
                        regime_class: str, entry_date_et: str,
                        cap_bars: int = XA_CAP_BARS) -> dict:
    """PURE exit-aligned label for one long entry (see the block comment above).

    fwd_bars: daily bars STRICTLY AFTER the entry ET session, oldest first.
    Each bar is walked OPEN -> LOW -> HIGH -> CLOSE (adverse move first for a
    long, so a bar touching both a stop and a profit level resolves STOP
    FIRST — the old ruler's 'ambiguous = stop' rule, generalised). The jump
    from the previous close (the entry price for bar 1) to the open is a gap:
    anything it skips past fills AT THE OPEN. Inside a leg, a level fills at
    the level. Tranches: 1/3 at each scale-out, the rest at the final exit;
    return_pct is the gross weighted return; label = 1 iff return_pct > 0.

    A path still open after cap_bars is marked to that bar's close
    (exit_reason 'open_at_cap', censored=True). Fewer than cap_bars valid
    bars without an exit -> label -1 'incomplete_path' (the caller retries;
    never labeled off a truncated path).
    """
    if entry_price <= 0 or not atr_pct or atr_pct <= 0:
        return {"label": -1, "exit_reason": "no_atr", "label_method": XA_LABEL_METHOD}
    pos = _XaPosition(entry_price, atr_pct, regime_class)
    try:
        entry_day = datetime.strptime(entry_date_et, "%Y-%m-%d").date()
    except Exception as e:
        logger.debug(f"xa: bad entry date {entry_date_et!r}: {e}")
        return {"label": -1, "exit_reason": "no_price_data", "label_method": XA_LABEL_METHOD}

    used = 0
    last_close, last_date = None, None
    for bar in fwd_bars:
        if used >= cap_bars:
            break
        try:
            o, h, lo, c = (float(bar.get(k, 0) or 0) for k in ("o", "h", "l", "c"))
            bar_date = str(bar.get("t") or "")[:10]
            bar_day = datetime.strptime(bar_date, "%Y-%m-%d").date()
        except Exception as e:
            logger.debug(f"xa: skipping malformed bar {bar!r}: {e}")
            continue
        if min(o, h, lo, c) <= 0:
            continue
        hi, lo = max(o, h, lo, c), min(o, h, lo, c)
        used += 1
        last_close, last_date = c, bar_date
        te = (bar_day - entry_day).days >= XA_TIME_STOP_DAYS
        if pos.settle(o, te) or pos.leg(o, lo, te) or pos.leg(lo, hi, te) or pos.leg(hi, c, te):
            return _xa_result(pos, bar_date, used, censored=False)

    if used < cap_bars or last_close is None:
        return {"label": -1, "exit_reason": "incomplete_path", "label_method": XA_LABEL_METHOD}
    pos.max_px = max(pos.max_px, last_close)
    pos.min_px = min(pos.min_px, last_close)
    pos._close(last_close, "open_at_cap")
    return _xa_result(pos, last_date, used, censored=True)


def _xa_result(pos: "_XaPosition", exit_date: str, bars_held: int, censored: bool) -> dict:
    ret = sum(w * (px - pos.entry) / pos.entry * 100.0 for w, px in pos.fills)
    return {
        "label":        1 if ret > 0 else 0,
        "return_pct":   round(ret, 3) + 0.0,
        "r_multiple":   round(ret / pos.risk, 3) + 0.0 if pos.risk > 0 else None,
        "exit_reason":  pos.exit_reason,
        "exit_date":    exit_date,
        "bars_held":    bars_held,
        "censored":     bool(censored),
        "scale_outs":   pos.scales,
        "atr_pct":      round(pos.atr, 4),
        "risk_pct":     round(pos.risk, 4),
        "stop_pct":     round(pos.p1_stop, 4),
        "regime_class": pos.regime_class,
        "mfe_pct":      round((pos.max_px - pos.entry) / pos.entry * 100.0, 3) + 0.0,  # +0.0: no "-0.0"
        "mae_pct":      round((pos.min_px - pos.entry) / pos.entry * 100.0, 3) + 0.0,
        "label_method": XA_LABEL_METHOD,
    }


def _exit_aligned_outcome(rec: dict, bars: List[dict]) -> Optional[dict]:
    """Exit-aligned outcome for one shadow record from its ticker's daily
    bars (split-adjusted, covering entry-60d .. entry+53d). None = no usable
    price data (retryable). A dict with label -1 is terminal ('no_atr',
    'scale_mismatch') or retryable ('incomplete_path').

    NO LOOKAHEAD: the path uses only bars dated AFTER the entry ET session;
    ATR is decision-time — features.atr_pct (deep_score's scan-time ATR,
    the live R unit's closest analogue) when present and not the 3.0
    default, else Wilder ATR14 from bars dated BEFORE the entry session.
    atr_pct_prior (prior-bars ATR) is stored either way as a DATA-gate
    diagnostic. regime_class reads only record fields.
    """
    entry = float(rec.get("entry_price", 0) or 0)
    entry_et = _xa_entry_date_et(rec.get("timestamp", ""))
    if entry <= 0 or not entry_et or not bars:
        return None
    bars = sorted(bars, key=lambda b: str(b.get("t") or ""))
    prior = [b for b in bars if str(b.get("t") or "")[:10] < entry_et]
    fwd = [b for b in bars if str(b.get("t") or "")[:10] > entry_et]
    if not fwd:
        return None
    # SCALE GUARD: bars are split-adjusted, entry_price is the raw scan price.
    # A split after entry rescales the whole path (reverse splits in small caps
    # would read as +900% / -90%), so refuse to label instead.
    ref = [b for b in bars if str(b.get("t") or "")[:10] <= entry_et][-1:] + fwd[:1]
    lows = [float(b.get("l", 0) or 0) for b in ref if float(b.get("l", 0) or 0) > 0]
    highs = [float(b.get("h", 0) or 0) for b in ref if float(b.get("h", 0) or 0) > 0]
    if not lows or not highs:
        return None
    if not (XA_SCALE_GUARD[0] * min(lows) <= entry <= XA_SCALE_GUARD[1] * max(highs)):
        return {"label": -1, "exit_reason": "scale_mismatch", "label_method": XA_LABEL_METHOD}

    atr_prior = _xa_prior_atr_pct(prior, entry)
    feat = (rec.get("features") or {}).get("atr_pct")
    try:
        feat = float(feat) if feat is not None else None
    except (TypeError, ValueError):
        feat = None
    if XA_ATR_PREFER_FEATURE and feat is not None and feat > 0 and feat != XA_ATR_FEATURE_DEFAULT:
        atr, src = feat, "feature"
    elif atr_prior is not None:
        atr, src = atr_prior, "prior_bars"
    else:
        return {"label": -1, "exit_reason": "no_atr", "label_method": XA_LABEL_METHOD,
                "atr_pct_prior": None}

    out = _label_exit_aligned(entry, fwd, atr, _xa_regime_class(rec), entry_et)
    if out.get("label") in (0, 1):
        out["atr_src"] = src
        out["atr_pct_prior"] = round(atr_prior, 4) if atr_prior is not None else None
    return out


def _xa_time_left(deadline: Optional[float]) -> float:
    return float("inf") if deadline is None else deadline - time.monotonic()


def _xa_bars_get(params: dict, deadline: Optional[float] = None):
    """One rate-limited GET of /v2/stocks/bars (same bucket + 429 back-off as
    _fetch_historical_bars_batch). With a deadline, the 30s back-off and its
    retry run only if both fit before it; otherwise the 429 is returned and
    the caller treats the chunk as NOT fetched (retried next run, no strike)."""
    import requests
    _alpaca_bucket.acquire()
    resp = requests.get(f"{ALPACA_DATA_URL}/v2/stocks/bars", params=params,
                        headers=_alpaca_headers(), timeout=XA_REQUEST_TIMEOUT_S)
    if resp.status_code == 429:
        if _xa_time_left(deadline) < XA_429_BACKOFF_S + XA_REQUEST_TIMEOUT_S:
            logger.warning("Alpaca 429 during exit-aligned backfill — no time left in the budget "
                           "for the back-off; chunk deferred to the next run")
            return resp
        logger.warning(f"Alpaca 429 during exit-aligned backfill — sleeping {XA_429_BACKOFF_S}s")
        time.sleep(XA_429_BACKOFF_S)
        _alpaca_bucket.acquire()
        resp = requests.get(f"{ALPACA_DATA_URL}/v2/stocks/bars", params=params,
                            headers=_alpaca_headers(), timeout=XA_REQUEST_TIMEOUT_S)
    return resp


def _fetch_bars_paginated(tickers: List[str], start_date: str, end_date: str,
                          deadline: Optional[float] = None,
                          unfetched: Optional[set] = None) -> Dict[str, List[dict]]:
    """Daily bars for many tickers, FOLLOWING next_page_token.

    Separate from _fetch_historical_bars_batch on purpose: that helper sends
    limit=100 and never pages (Alpaca's multi-symbol limit counts data
    points across ALL symbols, so a 50-symbol batch is truncated) — fixing it
    would change the OLD ruler's inputs and is its own change. Same
    _alpaca_bucket, same 429 handling, same feed and split adjustment.

    FETCH FAILURE IS NOT MISSING DATA. When a 50-symbol chunk does not
    complete — an exception, a non-200 (outage, bad key, 429 after the
    back-off), an unparseable body, or no time left before `deadline` (no
    request starts unless XA_REQUEST_TIMEOUT_S fits before it) — every
    symbol of that chunk is added to `unfetched`, even if an earlier page
    succeeded (its data would be partial). The backfill skips those records
    WITHOUT a retry strike, so an Alpaca outage can never retire records as
    'no_price_data'. Only a chunk that completed but holds no usable bars for
    a record counts as missing data. Hitting the XA_MAX_PAGES cap is a
    completed (deterministic) fetch, not a failure."""
    out: Dict[str, List[dict]] = defaultdict(list)
    for i in range(0, len(tickers), BATCH_SIZE):
        chunk = tickers[i:i + BATCH_SIZE]
        token = None
        complete = True
        for _page in range(XA_MAX_PAGES):
            if _xa_time_left(deadline) < XA_REQUEST_TIMEOUT_S:
                logger.warning("exit-aligned bars fetch: time budget reached — chunk deferred to the next run")
                complete = False
                break
            params = {
                "symbols": ",".join(chunk),
                "timeframe": "1Day",
                "start": start_date,
                "end": end_date,
                "limit": XA_FETCH_LIMIT,
                "adjustment": "split",
                "feed": bars_feed(),
            }
            if token:
                params["page_token"] = token
            try:
                resp = _xa_bars_get(params, deadline)
            except Exception as e:
                logger.warning(f"exit-aligned bars fetch exception: {e}")
                complete = False
                break
            if resp.status_code != 200:
                logger.warning(f"exit-aligned bars fetch failed: {resp.status_code}")
                complete = False
                break
            try:
                body = resp.json() or {}
            except ValueError as e:
                logger.warning(f"exit-aligned bars fetch: unparseable body: {e}")
                complete = False
                break
            for sym, bars in (body.get("bars") or {}).items():
                out[sym].extend(bars or [])
            token = body.get("next_page_token")
            if not token:
                break
        else:
            logger.warning(f"exit-aligned bars fetch hit the {XA_MAX_PAGES}-page cap — "
                           f"later pages dropped (short paths retry as incomplete_path)")
        if not complete and unfetched is not None:
            unfetched.update(chunk)
    return {sym: sorted(bars, key=lambda b: str(b.get("t") or "")) for sym, bars in out.items()}


def _xa_apply_results(records: List[dict], results: Dict[Tuple[str, str], tuple],
                      stats: dict) -> int:
    """Write computed results into `records` (a FRESH read of the log), keyed
    by (ticker, timestamp). Returns how many records changed."""
    index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        index[(str(r.get("ticker", "")), str(r.get("timestamp", "")))].append(i)
    changed = 0
    for key, (kind, payload) in results.items():
        idxs = index.get(key)
        if not idxs:
            stats["rotated_out"] += 1
            continue
        for i in idxs:
            rec = records[i]
            if isinstance(rec.get(XA_KEY), dict):
                continue  # labeled meanwhile (another run) — never overwrite
            changed += 1
            if kind == "label":
                rec[XA_KEY] = payload
                rec.pop(XA_ATTEMPTS_KEY, None)
                lbl = payload.get("label")
                if lbl in (0, 1):
                    stats["updated"] += 1
                    stats["censored"] += 1 if payload.get("censored") else 0
                elif payload.get("exit_reason") in ("no_atr", "scale_mismatch"):
                    stats[payload["exit_reason"]] += 1
                continue
            # kind == "retry": payload is the failure reason
            stats["missing_price"] += 1
            n = int(rec.get(XA_ATTEMPTS_KEY, 0) or 0) + 1
            if n >= MAX_BACKFILL_ATTEMPTS:
                rec[XA_KEY] = {"label": -1, "exit_reason": str(payload),
                               "label_method": XA_LABEL_METHOD}
                rec.pop(XA_ATTEMPTS_KEY, None)
                stats["permanently_unlabelable"] += 1
            else:
                rec[XA_ATTEMPTS_KEY] = n
    return changed


def _xa_merge_save(results: Dict[Tuple[str, str], tuple], stats: dict) -> bool:
    """SAVE BY MERGE: exclusive lock, RE-READ the log, apply results by
    (ticker, timestamp), atomic replace — all under the lock. Writing back
    the list loaded at the start would drop every log_candidate() append
    made during the network phase (backfill_outcomes still has that race;
    filed separately). The apply counts go into `stats` only once the write
    has landed, so a failed checkpoint that the caller retries later is
    never double-counted."""
    try:
        import fcntl
    except ImportError:
        fcntl = None
    try:
        lock_f = open(SHADOW_LOCK_PATH, "a+")
    except OSError as e:
        logger.warning(f"exit-aligned save: cannot open lock file: {e}")
        return False
    delta: Dict[str, int] = defaultdict(int)

    def _commit() -> bool:
        for k, v in delta.items():
            stats[k] = stats.get(k, 0) + v
        return True

    try:
        if fcntl is not None:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            with open(SHADOW_LOG_PATH, "r") as f:
                records = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"exit-aligned save: cannot re-read the log: {e}")
            return False
        if not _xa_apply_results(records, results, delta):
            return _commit()
        dirname = os.path.dirname(SHADOW_LOG_PATH) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dirname, prefix=".shadow.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as tf:
                json.dump(records, tf)
            os.replace(tmp_path, SHADOW_LOG_PATH)
            return _commit()
        except Exception as e:
            logger.warning(f"exit-aligned save failed: {e}")
            try:
                os.unlink(tmp_path)
            except OSError as ue:
                logger.debug(f"exit-aligned save: tmp cleanup failed: {ue}")
            return False
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            except OSError as e:
                logger.debug(f"exit-aligned save: unlock failed: {e}")
        lock_f.close()


def backfill_exit_aligned(max_records: int = 3000, max_windows: int = 40,
                          time_budget_s: float = XA_JOB_TIME_BUDGET_S) -> dict:
    """Nightly exit-aligned labeling (own job, runs AFTER backfill_outcomes).

    Every record (any decision, old-labeled or not) that lacks XA_KEY, has an
    entry price, and is >= XA_ELIGIBLE_CALENDAR_DAYS old is grouped by entry
    ET day. ONE paginated fetch per day-window labels EVERY pending record of
    that day in the same pass; days go NEWEST FIRST so the steady-state new
    day is never starved behind the backlog (the KNOWN BROKEN #20 lesson),
    and the backlog drains over the following nights. Bounded three ways —
    max_windows fetch windows, max_records record-jobs, and time_budget_s of
    WALL TIME FROM ENTRY (log load included) — all through the shared
    180/min _alpaca_bucket. See the job-sizing block at XA_JOB_TIME_BUDGET_S:
    the caller's real cap is execPythonSerialized's 90s SIGKILL, and no
    Alpaca request starts unless it can finish before the deadline, so the
    run ends within time_budget_s + one merge-save (XA_SAVE_RESERVE_S).

    KILL-SAFE: results are merge-saved every XA_CHECKPOINT_WINDOWS windows or
    XA_CHECKPOINT_SECONDS, whichever comes first, and once more at the end —
    a SIGKILL loses at most the work since the last checkpoint.

    Missing data (a COMPLETED fetch with no usable bars for the record, or a
    path too short to resolve) counts up in XA_ATTEMPTS_KEY, and
    MAX_BACKFILL_ATTEMPTS of those write a terminal label -1. A fetch that did
    NOT complete (exception, non-200, out of time) strikes nothing: those
    records count in stats['fetch_failed'] and are retried on the next run.
    """
    t0 = time.monotonic()
    deadline = t0 + float(time_budget_s)
    stats = {"updated": 0, "already_filled": 0, "not_eligible": 0, "missing_price": 0,
             "permanently_unlabelable": 0, "no_atr": 0, "scale_mismatch": 0, "censored": 0,
             "rotated_out": 0, "windows_fetched": 0, "windows_deferred": 0, "jobs": 0,
             "fetch_failed": 0, "checkpoints": 0, "save_failures": 0,
             "time_budget_s": float(time_budget_s), "total_records": 0, "saved": False}
    records = _load_shadow_log()
    stats["total_records"] = len(records)
    if not records:
        return stats

    now_utc = datetime.now(timezone.utc)
    by_day: Dict[str, List[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        if isinstance(rec.get(XA_KEY), dict):
            stats["already_filled"] += 1
            continue
        try:
            rec_time = datetime.fromisoformat(str(rec.get("timestamp", "")).replace("Z", "+00:00"))
            if rec_time.tzinfo is None:
                rec_time = rec_time.replace(tzinfo=timezone.utc)
        except Exception as e:
            logger.debug(f"xa backfill: unparseable timestamp: {e}")
            continue
        if not rec.get("ticker") or float(rec.get("entry_price", 0) or 0) <= 0:
            continue
        if now_utc < rec_time + timedelta(days=XA_ELIGIBLE_CALENDAR_DAYS):
            stats["not_eligible"] += 1
            continue
        day = _xa_entry_date_et(rec.get("timestamp", ""))
        if day:
            by_day[day].append(idx)

    days = sorted(by_day, reverse=True)
    # Plan the run up front under the window/record caps, then drop the full
    # list: only the planned records stay referenced, so a checkpoint's fresh
    # re-read is the only full copy of the log in memory (not two).
    plan: List[Tuple[str, List[dict]]] = []
    planned = 0
    for day in days:
        if len(plan) >= max_windows or planned >= max_records:
            break
        day_recs = [records[i] for i in by_day[day][: max_records - planned]]
        plan.append((day, day_recs))
        planned += len(day_recs)
    del records, by_day

    results: Dict[Tuple[str, str], tuple] = {}
    since = {"t": time.monotonic(), "windows": 0}

    def _checkpoint() -> None:
        since["t"], since["windows"] = time.monotonic(), 0
        if not results:
            return
        stats["checkpoints"] += 1
        if _xa_merge_save(results, stats):
            results.clear()
        else:
            stats["save_failures"] += 1   # kept in `results`; retried at the next checkpoint

    for day, day_recs in plan:
        if _xa_time_left(deadline) < XA_REQUEST_TIMEOUT_S:
            break   # no request could start: the rest stay windows_deferred, not fetch_failed
        d = datetime.strptime(day, "%Y-%m-%d")
        start = (d - timedelta(days=XA_ATR_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end = min(d + timedelta(days=XA_WINDOW_END_DAYS), now_utc.replace(tzinfo=None)).strftime("%Y-%m-%d")
        tickers = sorted({str(r["ticker"]) for r in day_recs})
        unfetched: set = set()
        bars_by_ticker = _fetch_bars_paginated(tickers, start, end, deadline=deadline, unfetched=unfetched)
        stats["windows_fetched"] += 1
        stats["jobs"] += len(day_recs)
        for rec in day_recs:
            ticker = str(rec["ticker"])
            if ticker in unfetched:
                stats["fetch_failed"] += 1   # the fetch failed, not the data: no strike
                continue
            key = (ticker, str(rec.get("timestamp", "")))
            out = _exit_aligned_outcome(rec, bars_by_ticker.get(ticker, []))
            if out is None:
                results[key] = ("retry", "no_price_data")
            elif out.get("exit_reason") == "incomplete_path":
                results[key] = ("retry", "incomplete_path")
            else:
                results[key] = ("label", out)
        since["windows"] += 1
        if (since["windows"] >= XA_CHECKPOINT_WINDOWS
                or time.monotonic() - since["t"] >= XA_CHECKPOINT_SECONDS):
            _checkpoint()
    _checkpoint()
    stats["windows_deferred"] = max(0, len(days) - stats["windows_fetched"])
    stats["saved"] = stats["checkpoints"] > 0 and not results
    stats["elapsed_s"] = round(time.monotonic() - t0, 1)
    logger.info(f"Shadow exit-aligned backfill: {stats}")
    return stats


# ── CLI — run backfill from command line or cron ─────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"

    if cmd == "backfill":
        max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        result = backfill_outcomes(max_records=max_n)
        print(json.dumps(result, indent=2))
    elif cmd == "backfill_xa":
        max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
        print(json.dumps(backfill_exit_aligned(max_records=max_n), indent=2))
    elif cmd == "stats":
        print(json.dumps(get_shadow_stats(), indent=2, default=str))
    elif cmd == "clear":
        if os.path.exists(SHADOW_LOG_PATH):
            os.remove(SHADOW_LOG_PATH)
            print(f"Cleared {SHADOW_LOG_PATH}")
    else:
        print("Usage: python3 shadow_portfolio.py [backfill [N] | backfill_xa [N] | stats | clear]")
        sys.exit(1)


# ── Band report (2026-10-06, human-directed "fix the issues") ────────────────
# Where does the scan lose? get_shadow_stats() showed TAKEN candidates
# winning LESS than score-REJECTED ones (+5d 47.2% vs 51.0%, +20d 44.5% vs
# 56.4%) and the taken median |change_pct_today| at 18%. This view cuts the
# same labeled records by decision x SIGNED same-day move x score quintile x
# regime, reporting n / win rate / MEAN return_pct (the path-dependent exit
# return each record already stores) so expectancy, not just hit rate, is
# visible. Aggregate-only: no ticker, no price, no timestamp leaves here.
SHADOW_CHANGE_BANDS = [(-1e9, -10.0, "<-10"), (-10.0, -3.0, "-10..-3"), (-3.0, 3.0, "-3..3"),
                       (3.0, 10.0, "3..10"), (10.0, 20.0, "10..20"), (20.0, 30.0, "20..30"),
                       (30.0, 1e9, ">=30")]


def _band_of(change_pct: float) -> str:
    for lo, hi, name in SHADOW_CHANGE_BANDS:
        if lo <= change_pct < hi:
            return name
    return ">=30"


def _cell_stats(rets: List[float], wins: int, min_n: int) -> dict:
    n = len(rets)
    if n < min_n:
        return {"n": n}
    srt = sorted(rets)
    return {"n": n, "win_rate": round(100.0 * wins / n, 1),
            "mean_return_pct": round(sum(rets) / n, 3),
            "median_return_pct": round(srt[n // 2], 3)}


def _exit_aligned_labeling_note() -> dict:
    return {
        "method": XA_LABEL_METHOD,
        "mirrors": "server/bot.ts checkPositionOnTick (the only code that places live stock exits) "
                   "with syncMonitoredPositions' stop/TP levels; drift-pinned by test_shadow_exit_aligned.py",
        "constants": {
            "risk_unit_pct": f"{XA_RISK_ATR_MULT} x ATR% (unclamped)",
            "phase1_stop_pct": f"clamp({XA_P1_STOP_ATR_MULT} x ATR%, {XA_P1_STOP_MIN_PCT}, {XA_P1_STOP_MAX_PCT})",
            "scale_out_1": f"1/3 at {XA_SCALE1_R}R, arms breakeven (rest exits once P&L < 0)",
            "scale_out_2_r": dict(XA_SCALE2_R),
            "phases": "peak R >= 1 / 2 / 3 -> phase 2 / 3 / 4 (only move up)",
            "phase2_trail_from_peak_pnl": f"max({XA_P2_TRAIL_MIN_PCT}, {XA_P2_TRAIL_ATR_MULT} x ATR%)",
            "phase3_trail_from_peak_pnl": f"max({XA_P3_TRAIL_MIN_PCT}, {XA_P3_TRAIL_ATR_MULT} x ATR%)",
            "phase4_trail_from_high_price_atr_mult": dict(XA_TRAIL4_ATR_MULT),
            "take_profit_pct": f"max({XA_TP_MIN_PCT}, min({XA_TP_ATR_MULT} x ATR%, {XA_TP_MAX_PCT})), "
                               "only while phase < 3 and both scale-outs are done",
            "time_stop": f"remaining shares exit once >= {XA_TIME_STOP_DAYS} calendar days after the entry "
                         f"session AND |P&L| < {XA_TIME_STOP_BAND_PCT}%",
            "cap_bars": XA_CAP_BARS,
            "eligible_after_calendar_days": XA_ELIGIBLE_CALENDAR_DAYS,
            "round_trip_cost_pct": XA_ROUND_TRIP_COST_PCT,
        },
        "same_bar_rule": "each daily bar is walked OPEN -> LOW -> HIGH -> CLOSE (adverse move first for a long, "
                         "so a bar touching a stop and a profit level resolves STOP FIRST, as the old ruler's "
                         "'ambiguous'); a level crossed inside a leg fills at the level, a gap past it fills "
                         "at the open",
        "atr": "features.atr_pct (decision-time, scan-time ATR) when present and not the 3.0 default, else "
               "Wilder ATR14 from daily bars strictly before the entry ET session; held fixed at entry",
        "regime_class": "system_config.get_market_regime(record vxx_ratio, features.spy_vs_ma50) at entry: "
                        "BULL -> bullish, CAUTION/BEAR/PANIC -> bearish, else neutral; held fixed at entry",
        "horizon": "one label per record; path = trading bars strictly after the entry ET session; still "
                   "open after cap_bars -> marked to that close, censored=true (see censored_pct)",
        "returns": "gross; three 1/3 tranches; mean_net_return_pct subtracts round_trip_cost_pct",
        "note": "Labels are LONG-stock outcomes: they mean what they say only for action BUY. The keys "
                "outside 'exit_aligned' are the OLD ruler, unchanged; compare the two rulers only on "
                "paired_vs_old (identical records). paired_vs_old compares DIFFERENT forward paths for "
                "scans made 20:00-24:00 ET: the old ruler starts after the UTC entry date (it skips one "
                "session), this one after the ET entry session. Not modelled: see not_modelled (each "
                "item names its bias direction).",
        "not_modelled": [
            "PHANTOM FIRST SCALE-OUT (live race, unverified): bot_engine.manage_positions (45s Tier-1 "
            "cycle) writes scales_completed=1, remaining_qty -= 1/3 and breakeven_active to "
            "voltrade_stop_state.json when it sees R >= 1, WITHOUT placing an order; "
            "syncMonitoredPositions adopts them via Math.max/Math.min, so when Python sees 1R before a "
            "1-minute bar close does, checkPositionOnTick never sells the first third and the final exit "
            "sells only remainingQty. This label always banks 1/3 at +1R: OPTIMISTIC on 1R-touch-then-"
            "breakeven paths, PESSIMISTIC on big runners (live keeps more size on the trail).",
            "REGIME INPUTS: live calls get_market_regime(vxx, spy_vs_ma50, spy_below_200_days, "
            "spy_above_200d); records carry no 200-day inputs, so this labels with the defaults (0, "
            "True). When SPY is below its 200-day MA, live can read NEUTRAL where this reads BULL, or "
            "BEAR where this reads BULL/NEUTRAL, changing scale-out-2 R and the phase-4 trail; bias "
            "direction mixed. Live also re-reads the regime every cycle; this holds it fixed at entry.",
            "SAMPLED PEAKS: every ~45s sync overwrites live highestPnl / highestPrice / phase with "
            "Python's 45s-sampled values (max with the current price), so live trails measure from a "
            "lower peak than the 1-minute closes saw, and far lower than a daily high. Trails here run "
            "from the daily high: OPTIMISTIC on trail exits.",
            "DAILY BARS: daily highs/lows overstate what live's 1-minute IEX bar closes touch, so "
            "scale-outs, phase promotions and trail peaks are reached more often here: OPTIMISTIC. "
            "Checking the adverse leg first and filling gaps at the open is PESSIMISTIC.",
            "ALSO: entry-session moves after the scan; live ATR refresh (live R is fixed at the first "
            "manage cycle, but the phase-2/3/4 trails and the TP use the CURRENT ATR%; this holds the "
            "entry ATR); sub-3-share positions (never scale out); paused/kill-switched exits; unfilled "
            "limit exits; slippage beyond the flat round-trip cost.",
        ],
    }


def _exit_aligned_tables(records: List[dict], edges: List[float], min_n: int) -> dict:
    """The exit-aligned ruler's view of the same records (aggregate-only: no
    ticker, price or timestamp leaves here). Reuses the old report's score
    quintile edges and _cell_stats under one pseudo-horizon key 'xa'."""
    def quint(s: float) -> str:
        return f"Q{1 + sum(1 for e in edges if s >= e)}" if edges else "Q?"

    cells: Dict[Tuple[str, str], List[Tuple[float, int, bool]]] = defaultdict(list)
    coverage: Dict[str, dict] = {}
    unique_days: Dict[str, set] = defaultdict(set)
    paired: Dict[str, dict] = {}
    for r in records:
        dec = str(r.get("decision") or "?")
        cov = coverage.setdefault(dec, {"labeled": 0, "censored": 0, "pending_retry": 0,
                                        "not_yet_labeled": 0, "unlabelable": defaultdict(int)})
        out = r.get(XA_KEY)
        if not isinstance(out, dict):
            if int(r.get(XA_ATTEMPTS_KEY, 0) or 0) > 0:
                cov["pending_retry"] += 1
            else:
                cov["not_yet_labeled"] += 1
            continue
        lbl = out.get("label")
        ret = out.get("return_pct")
        if lbl not in (0, 1) or ret is None:
            cov["unlabelable"][str(out.get("exit_reason") or "?")] += 1
            continue
        censored = bool(out.get("censored"))
        cov["labeled"] += 1
        cov["censored"] += 1 if censored else 0
        unique_days[dec].add((r.get("ticker"), _xa_entry_date_et(r.get("timestamp", ""))))
        act = r.get("action_label") or "unrecorded"
        cp = float((r.get("features") or {}).get("change_pct_today", 0) or 0)
        rc = str(out.get("regime_class") or "?")
        keys = {
            ("decision", dec),
            ("decision_x_action", f"{dec}|{act}"),
            ("action_x_band", f"{act}|{_band_of(cp)}"),
            ("score_quintile", quint(float(r.get("score", 0) or 0))),
            ("decision_x_regime", f"{dec}|{rc}"),
            ("regime_class", rc),
            ("exit_reason", str(out.get("exit_reason") or "?")),
            ("all", "all"),
        }
        for dim, k in keys:
            cells[(dim, k)].append((float(ret), int(lbl), censored))
        old = (r.get("outcomes") or {}).get("+10d")
        if isinstance(old, dict) and old.get("label") in (0, 1) and old.get("return_pct") is not None:
            for pk in ("all", f"{dec}|{act}"):
                p = paired.setdefault(pk, {"matrix": {"old1_xa1": 0, "old1_xa0": 0, "old0_xa1": 0, "old0_xa0": 0},
                                           "old": [], "xa": []})
                p["matrix"][f"old{int(old['label'])}_xa{int(lbl)}"] += 1
                p["old"].append(float(old["return_pct"]))
                p["xa"].append(float(ret))

    tables: Dict[str, dict] = defaultdict(dict)
    for (dim, k), vals in cells.items():
        rets = [v[0] for v in vals]
        cell = _cell_stats(rets, sum(v[1] for v in vals), min_n)
        if "mean_return_pct" in cell:
            cell["mean_net_return_pct"] = round(cell["mean_return_pct"] - XA_ROUND_TRIP_COST_PCT, 3)
            cell["censored_pct"] = round(100.0 * sum(1 for v in vals if v[2]) / len(vals), 1)
        tables[dim][k] = {"xa": cell}

    paired_out = {}
    for pk, p in paired.items():
        n = len(p["xa"])
        if n < min_n:
            paired_out[pk] = {"n": n}
            continue
        paired_out[pk] = {
            "n": n,
            "matrix": p["matrix"],
            "win_rate_old_plus10d": round(100.0 * (p["matrix"]["old1_xa1"] + p["matrix"]["old1_xa0"]) / n, 1),
            "win_rate_xa": round(100.0 * (p["matrix"]["old1_xa1"] + p["matrix"]["old0_xa1"]) / n, 1),
            "mean_return_pct_old_plus10d": round(sum(p["old"]) / n, 3),
            "mean_return_pct_xa": round(sum(p["xa"]) / n, 3),
        }

    for dec, cov in coverage.items():
        cov["unlabelable"] = dict(cov["unlabelable"])
        cov["unique_ticker_entry_days"] = len(unique_days.get(dec, ()))
    return {
        "labeling": _exit_aligned_labeling_note(),
        "coverage": coverage,
        "paired_vs_old": paired_out,
        **{dim: dict(v) for dim, v in tables.items()},
    }


def _score_quintile_edges(records: List[dict]) -> List[float]:
    """Score quintile edges over every labeled record that has a score."""
    scores = sorted(float(r.get("score", 0) or 0) for r in records
                    if isinstance(r.get("outcomes"), dict))
    return [scores[int(len(scores) * q / 5)] for q in range(1, 5)] if len(scores) >= 5 else []


def shadow_band_report(records: List[dict], min_n: int = 20,
                       edges: Optional[List[float]] = None) -> dict:
    """Pure aggregation over shadow records (see block comment above).
    edges=None computes the score quintile edges from these records (the
    original behaviour); the deduped probe passes the FULL population's edges
    so its quintiles are comparable."""
    horizons = [f"+{h}d" for h in FORWARD_HORIZONS_DAYS]
    if edges is None:
        edges = _score_quintile_edges(records)

    def quint(s: float) -> str:
        return f"Q{1 + sum(1 for e in edges if s >= e)}" if edges else "Q?"

    cells: Dict[Tuple[str, str, str], List[Tuple[float, int]]] = defaultdict(list)
    for r in records:
        outs = r.get("outcomes")
        if not isinstance(outs, dict):
            continue
        dec = str(r.get("decision") or "?")
        cp = float((r.get("features") or {}).get("change_pct_today", 0) or 0)
        keys = {
            ("decision", dec),
            ("decision_x_band", f"{dec}|{_band_of(cp)}"),
            ("band_all", _band_of(cp)),
            ("score_quintile", quint(float(r.get("score", 0) or 0))),
            ("decision_x_regime", f"{dec}|{r.get('regime_label') or '?'}"),
            ("decision_x_action", f"{dec}|{r.get('action_label') or 'unrecorded'}"),
            ("action_x_band", f"{r.get('action_label') or 'unrecorded'}|{_band_of(cp)}"),
            ("all", "all"),
        }
        for h in horizons:
            o = outs.get(h)
            if not isinstance(o, dict) or o.get("label") not in (0, 1):
                continue
            ret = o.get("return_pct")
            if ret is None:
                continue
            for dim, k in keys:
                cells[(dim, k, h)].append((float(ret), int(o["label"])))

    out: Dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    for (dim, k, h), vals in cells.items():
        rets = [v[0] for v in vals]
        out[dim][k][h] = _cell_stats(rets, sum(v[1] for v in vals), min_n)
    return {
        "labeling": {"method": "path_dependent_v2", "win_pct": WIN_THRESHOLD_PCT,
                     "stop_pct": LOSS_THRESHOLD_PCT,
                     "note": "return_pct is entry->exit under the labeler's target/stop/timeout rules, "
                             "not a raw forward return; 'all' is the base rate across every scored candidate. "
                             "Labels are LONG-stock outcomes: they mean what they say only for action BUY. "
                             "Records before 2026-10-06 carry no action ('unrecorded')."},
        "score_quintile_edges": [round(e, 3) for e in edges],
        "change_bands": [b[2] for b in SHADOW_CHANGE_BANDS],
        "min_n": min_n,
        **{dim: {k: dict(v) for k, v in d.items()} for dim, d in out.items()},
        # 2026-10-14 [RULE-REVIEW]: the exit-aligned ruler, under ONE new key —
        # every key above is the old ruler and keeps its exact values.
        "exit_aligned": _exit_aligned_tables(records, edges, min_n),
    }


def _first_seen_per_ticker_day(records: List[dict], has_label) -> List[dict]:
    """2026-10-27 [RULE-REVIEW]: the shadow logger writes the same candidate
    on every scan, so one (ticker, ET entry day) can appear dozens of times
    (2026-10-20: 5,192 labeled 'taken' records covered only 148 ticker-days),
    and a cell's n counts copies, not events. Keep ONE record per (ticker, ET
    entry session): the FIRST-SEEN one (earliest timestamp) — the first scan
    at which the system could have acted — among records where
    has_label(rec) is true. Pure; records are not modified."""
    first: Dict[Tuple[str, Optional[str]], dict] = {}
    for r in records:
        if not has_label(r):
            continue
        k = (str(r.get("ticker") or ""), _xa_entry_date_et(r.get("timestamp", "")))
        cur = first.get(k)
        if cur is None or str(r.get("timestamp", "")) < str(cur.get("timestamp", "")):
            first[k] = r
    return list(first.values())


def _has_old_label(r: dict) -> bool:
    outs = r.get("outcomes")
    return isinstance(outs, dict) and any(
        isinstance(o, dict) and o.get("label") in (0, 1) for o in outs.values())


def _has_xa_label(r: dict) -> bool:
    out = r.get(XA_KEY)
    return isinstance(out, dict) and out.get("label") in (0, 1) and out.get("return_pct") is not None


def shadow_band_report_deduped(records: List[dict], min_n: int = 20) -> dict:
    """2026-10-27 [RULE-REVIEW]: the shadow_bands tables for BOTH rulers over
    one first-seen record per (ticker, ET entry day), so min_n counts distinct
    events. Score quintile edges are the FULL population's (same as
    shadow_bands), so quintiles are comparable across the two probes.
    Aggregate-only, like shadow_band_report. Purely additive: shadow_bands is
    unchanged."""
    edges = _score_quintile_edges(records)
    old_dd = _first_seen_per_ticker_day(records, _has_old_label)
    xa_dd = _first_seen_per_ticker_day(records, _has_xa_label)
    old_rep = shadow_band_report(old_dd, min_n=min_n, edges=edges)
    old_rep.pop("exit_aligned", None)
    xa_rep = _exit_aligned_tables(xa_dd, edges, min_n)
    xa_rep.pop("labeling", None)
    return {
        "method": "first-seen record per (ticker, ET entry session) among records labeled by that ruler; "
                  "score quintile edges from the full population",
        "min_n": min_n,
        "old": {"n_records": sum(1 for r in records if _has_old_label(r)),
                "n_unique": len(old_dd), **old_rep},
        "exit_aligned": {"n_records": sum(1 for r in records if _has_xa_label(r)),
                         "n_unique": len(xa_dd), **xa_rep},
    }


def get_shadow_band_report_deduped() -> dict:
    return shadow_band_report_deduped(_load_shadow_log())


def get_shadow_band_report() -> dict:
    return shadow_band_report(_load_shadow_log())
