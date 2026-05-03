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

def log_candidate(
    ticker: str,
    features: dict,
    score: float,
    decision: str,
    decision_reason: str = "",
    entry_price: float = 0.0,
    vxx_ratio: float = 1.0,
    regime_label: str = "neutral",
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
                         "rejected_halt" | "rejected_earnings" | "rejected_other"
        decision_reason: human-readable explanation (for debugging)
        entry_price:     stock price at scan time (for forward-return math)
        vxx_ratio:       VXX regime context
        regime_label:    regime string from _classify_regime

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
            "features":        {k: float(v) if isinstance(v, (int, float)) else 0.0
                                for k, v in features.items()} if features else {},
            "outcomes":        {f"+{h}d": None for h in FORWARD_HORIZONS_DAYS},
            "code_version":    CODE_VERSION,
        }

        # Load, append, rotate, save
        records = _load_shadow_log()
        records.append(record)
        if len(records) > MAX_SHADOW_RECORDS:
            records = records[-MAX_SHADOW_RECORDS:]  # keep newest
        _save_shadow_log(records)
    except Exception as e:
        logger.debug(f"log_candidate failed for {ticker}: {e}")


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
                    "feed": "sip",
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
                        "feed": "sip",
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


def backfill_outcomes(max_records: int = 500) -> dict:
    """
    Fill in forward-return outcomes for shadow records that are old enough.

    For each record with at least horizon_days since its timestamp AND no
    outcome set for that horizon, look up the actual price on that date
    and compute hypothetical P&L.

    Args:
        max_records: cap on how many to process per call (rate-limit safety)

    Returns:
        dict with counts: {updated, already_filled, missing_price, skipped}

    Rate cost: ~3-5 batch Alpaca calls for 500 records (~50 tickers per batch).
    Well within the 180/min token budget.
    """
    stats = {"updated": 0, "already_filled": 0, "missing_price": 0, "skipped": 0,
             "total_records": 0}

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

    # Group by ticker so we batch-fetch one window per ticker (covers all
    # horizons for that ticker in one Alpaca call).
    jobs_by_ticker: Dict[str, List[tuple]] = defaultdict(list)
    for job in backfill_jobs:
        jobs_by_ticker[job[3]].append(job)

    for ticker, jobs in jobs_by_ticker.items():
        # Fetch one wide window covering the longest horizon needed
        max_horizon = max(j[2] for j in jobs)
        earliest_entry = min(j[5] for j in jobs)
        latest_entry = max(j[5] for j in jobs)
        window_start = (earliest_entry - timedelta(days=2)).strftime("%Y-%m-%d")
        window_end = (latest_entry + timedelta(
            days=int(max_horizon * HORIZON_CALENDAR_BUFFER) + 3
        )).strftime("%Y-%m-%d")

        bars_by_ticker = _fetch_historical_bars_batch(
            [ticker], window_start, window_end
        )
        all_bars = bars_by_ticker.get(ticker, [])
        if not all_bars:
            for job in jobs:
                stats["missing_price"] += 1
            continue

        for idx, horizon_key, horizon, _, entry_price, rec_time in jobs:
            entry_date_str = rec_time.strftime("%Y-%m-%d")
            # Bars at or after entry date — we want trading-day forward window
            forward_bars = [
                b for b in all_bars
                if (b.get("t") or "")[:10] > entry_date_str
            ]
            if not forward_bars:
                stats["missing_price"] += 1
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

    # Save updated records atomically
    if stats["updated"] > 0:
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

    return {
        "total_records":         len(records),
        "oldest":                records[0].get("timestamp") if records else None,
        "newest":                records[-1].get("timestamp") if records else None,
        "by_decision":           dict(by_decision),
        "by_regime":             dict(by_regime),
        "labeled_by_horizon":    labeled_by_horizon,
        "win_rate_by_decision":  win_rate_by_decision,
    }


# ── CLI — run backfill from command line or cron ─────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"

    if cmd == "backfill":
        max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        result = backfill_outcomes(max_records=max_n)
        print(json.dumps(result, indent=2))
    elif cmd == "stats":
        print(json.dumps(get_shadow_stats(), indent=2, default=str))
    elif cmd == "clear":
        if os.path.exists(SHADOW_LOG_PATH):
            os.remove(SHADOW_LOG_PATH)
            print(f"Cleared {SHADOW_LOG_PATH}")
    else:
        print("Usage: python3 shadow_portfolio.py [backfill [N] | stats | clear]")
        sys.exit(1)
