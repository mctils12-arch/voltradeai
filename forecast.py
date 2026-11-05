#!/usr/bin/env python3
"""
VolTradeAI — Weekly forecast screen (SHADOW, experimental model)
================================================================
The human's Phase 7 ask (2026-10-27): a weekly call for SPY, every open
position and the top scan candidates — direction (up/down), its probability,
and an expected price range for the next 5 trading days — plus the calls'
accuracy and calibration on the same page. "No defaulting to neutral; calls
must be meaningful."

SHADOW: display and tracking only. Nothing in the trading loop reads these
files; no order, threshold or size depends on them.

DAILY HORIZON (Part C, 2026-11-04): the next session's close for the same
universe, alongside the weekly call — generate_daily / score_daily /
daily_report, below the weekly code. Same files, horizon="daily"; read the
"daily horizon" section's header for what a daily call is.

WHAT A CALL IS (stated so it can be audited)
  * Base close d0 = the last COMPLETED daily session at the time the batch is
    made (a bar dated today counts only after SESSION_DONE_ET, 16:15 ET /
    13:15 ET on a half day). spot = the d0 close (split-adjusted Alpaca bars).
  * Horizon = the 5 trading sessions after d0. The planned end date comes from
    market_calendar; the scorer settles on the 5th actual bar after d0, so a
    calendar gap (2027 dates are not in market_calendar yet) can only move
    the label, never fake it.
  * P(up) = P(close at d0+5 > close at d0) from a ridge-regularized logistic
    regression on four point-in-time features, all computed from closes on or
    before d0 (FEATURES below). It is fit each week on a pooled panel — a
    fixed basket of 15 index/sector ETFs plus that week's forecast names —
    sampled every 5th session (non-overlapping labels). Validation is a
    TIME-ORDERED split: the model is fit on samples whose label date falls
    before the split date (purged — no label overlaps the holdout) and scored
    on the later 30%; those holdout numbers ship with the batch. The live
    coefficients are then refit on all history with the same settings.
  * The call is ALWAYS emitted: direction = up if P(up) >= 0.5 else down, with
    P(called direction) = max(P, 1-P). That is "no defaulting to neutral" — it
    does NOT mean inflating confidence: a 0.52 stays 0.52.
  * Range = the 68.3% band of a zero-drift lognormal: spot * exp(+/- s_h).
    s_h = chain ATM IV30 (options-chain archive, server/chainIv.ts, passed in
    by bot.ts) * sqrt(calendar days to horizon end / 365) when the archive has
    the name on a day within IV_MAX_AGE_DAYS of d0; else 20-session realized
    vol * sqrt(5). The source is stored per call (`vol_source`).

HONESTY
  * Experimental. The training panel is names that exist today (survivorship
    bias); the ETF basket dominates the sample; momentum/reversal edges at a
    1-week horizon are small, so most P(up) values sit near the base rate.
    The holdout Brier score vs. the climatology Brier is printed beside every
    batch so the model's skill (or lack of it) is visible.
  * Every batch is stored append-only, including late ones (made after the
    first horizon session opened — `on_time: false`, counted separately).

STORAGE (storage_config): FORECASTS_PATH — one JSON line per call
(kind="forecast") plus one per batch (kind="batch"), written in one append;
FORECAST_SCORES_PATH — one line per resolved call (kind="score"). Neither file
is ever rewritten. Readers dedupe by id, first line wins.

Entry points (voltrade_daemon.py routes forecast_generate / forecast_score /
forecast_report; bot.ts runs generate/score as python3 -c subprocesses behind
execPythonSerialized's hard cap, and serves report() on /api/forecast).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from storage_config import FORECASTS_PATH, FORECAST_SCORES_PATH, FORECAST_DAILY_LIVE_PATH
except ImportError:  # pragma: no cover — storage_config is always present in the repo
    _d = "/data/voltrade" if os.path.isdir("/data") else "/tmp"
    FORECASTS_PATH = os.path.join(_d, "voltrade_forecasts.jsonl")
    FORECAST_SCORES_PATH = os.path.join(_d, "voltrade_forecast_scores.jsonl")
    FORECAST_DAILY_LIVE_PATH = os.path.join(_d, "voltrade_forecast_daily_live.json")

import forecast_range as FR

log = logging.getLogger("voltrade.forecast")

MODEL_VERSION = "fc-logit-v1"
HORIZON_DAYS = 5                    # trading sessions
BAND_Z = 1.0                        # +/- 1 sigma = 68.27% two-sided
BAND_COVERAGE = 0.6827
HISTORY_CALENDAR_DAYS = 3 * 365 + 40
SAMPLE_STRIDE = 5                   # one sample per 5 sessions per name: non-overlapping labels
TRAIN_FRACTION = 0.70               # time-ordered split point (by sample date)
RIDGE_L2 = 1.0                      # on the slope weights only
MIN_TRAIN_SAMPLES = 300
FEATURE_CLIP = 4.0
RV_WINDOW = 20                      # sessions of daily log returns
SPY_TREND_WINDOW = 50
IV_MAX_AGE_DAYS = 7                 # chain IV older than this vs d0 is not used
MAX_TICKERS = 30
SESSION_DONE_ET = (16, 15)          # bars dated today count after this (ET)
HALF_DAY_DONE_ET = (13, 15)
SCORE_GIVE_UP_DAYS = 30             # calendar days past the planned horizon end
MIN_N_FOR_RATES = 30                # under this an accuracy figure is `thin`
CAL_EDGES = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0)
FEATURES = ("mom5_z", "mom20_z", "rev1_z", "spy_trend_z")
# Fixed training basket (index + SPDR sector ETFs): long histories, little
# survivorship, liquid. That week's forecast names are pooled in with it.
TRAIN_BASKET = ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV",
                "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC")
GROUPS = ("index", "position", "candidate")

DAILY_MODEL_VERSION = "fc-daily-v1"             # base-rate direction + fc-range-har-v1 band
DAILY_DIRECTION_MODEL = "base-rate-v1"
# Features the daily direction uses. EMPTY: none of the pre-registered
# candidates (overnight gap, intraday momentum, VIX change, breadth, dealer
# gamma, first-hour momentum) beat the base-rate Brier on the purged
# walk-forward holdout after the multiple-testing bar
# (research/forecast_direction_study_2026-11-04.md). A feature may enter only
# with a study that passes; test_forecast.py pins this tuple to that verdict.
DAILY_DIRECTION_FEATURES: Tuple[str, ...] = ()
DAILY_BASE_RATE_SESSIONS = 756          # trailing daily returns per basket name (~3 years)
DAILY_BASE_RATE_MIN = 2000              # pooled returns needed for a base rate
DAILY_CLIM_SESSIONS = 756               # constant-band reference: the name's own |r| quantile
DAILY_CLIM_MIN = 250
DAILY_PREOPEN_FROM_ET = (18, 45)        # after the options archive's 16:30-18:30 ET capture
DAILY_VIX_CUTOFF_ET = (8, 0)            # on D: stop waiting for the d0 VIX close
DAILY_LIVE_FETCH_DAYS = 60
DAILY_REPORT_SESSIONS = 30
SESSION_OPEN_ET = (9, 30)
SESSION_CLOSE_ET = (16, 0)
HALF_DAY_CLOSE_ET = (13, 0)


# ── time / calendar ─────────────────────────────────────────────────────────
def _et(dt: datetime) -> datetime:
    from zoneinfo import ZoneInfo
    return dt.astimezone(ZoneInfo("America/New_York"))


def _is_trading_day(d: date) -> bool:
    from market_calendar import is_market_holiday
    return not is_market_holiday(d)


def _next_trading_day(d: date) -> date:
    from market_calendar import next_trading_day
    return next_trading_day(d)


def _prev_trading_day(d: date) -> date:
    c = d - timedelta(days=1)
    while not _is_trading_day(c):
        c -= timedelta(days=1)
    return c


def _session_done(d: date, now: datetime) -> bool:
    """True when the session dated `d` (ET) has closed and its daily bar is final."""
    from market_calendar import is_half_day
    n = _et(now)
    if d < n.date():
        return True
    if d > n.date():
        return False
    hh, mm = HALF_DAY_DONE_ET if is_half_day(d) else SESSION_DONE_ET
    return (n.hour, n.minute) >= (hh, mm)


def expected_base_date(now: datetime) -> date:
    """The calendar's view of the last completed session (the bars confirm it)."""
    d = _et(now).date()
    if _is_trading_day(d) and _session_done(d, now):
        return d
    return _prev_trading_day(d)


def horizon_end(d0: date, n: int = HORIZON_DAYS) -> date:
    d = d0
    for _ in range(n):
        d = _next_trading_day(d)
    return d


def week_key(d0: date) -> str:
    """ISO week of the first horizon session — one batch per key."""
    y, w, _ = _next_trading_day(d0).isocalendar()
    return f"{y}-W{w:02d}"


def _open_after(d0: date) -> datetime:
    """09:30 ET of the first horizon session, as an aware datetime."""
    from zoneinfo import ZoneInfo
    d1 = _next_trading_day(d0)
    return datetime(d1.year, d1.month, d1.day, 9, 30, tzinfo=ZoneInfo("America/New_York"))


def _bar_day(bar: dict) -> str:
    """Alpaca daily bars are stamped at 04:00/05:00 UTC of the session date."""
    return str(bar.get("t", ""))[:10]


def completed_bars(bars: Sequence[dict], now: datetime) -> List[dict]:
    """Ascending, deduped by date, only sessions that have closed by `now`."""
    out: Dict[str, dict] = {}
    for b in bars or []:
        ds = _bar_day(b)
        c = b.get("c")
        if len(ds) != 10 or not isinstance(c, (int, float)) or c <= 0:
            continue
        try:
            d = date.fromisoformat(ds)
        except ValueError:
            log.debug("forecast: skipping bar with malformed date %r", ds)
            continue
        if _session_done(d, now):
            out[ds] = b
    return [out[k] for k in sorted(out)]


# ── features + model (pure) ─────────────────────────────────────────────────
def _std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _clip(x: float) -> float:
    return max(-FEATURE_CLIP, min(FEATURE_CLIP, x))


def daily_vol(closes: Sequence[float], i: int, window: int = RV_WINDOW) -> Optional[float]:
    """Sample std of the `window` daily log returns ending at index i."""
    if i < window:
        return None
    rets = [math.log(closes[k] / closes[k - 1]) for k in range(i - window + 1, i + 1)]
    sd = _std(rets)
    return sd if sd > 0 and math.isfinite(sd) else None


def feature_row(closes: Sequence[float], i: int, spy: Sequence[float], j: int) -> Optional[List[float]]:
    """The four point-in-time features at index i of `closes` (spy index j is
    the same date). Uses closes[..i] and spy[..j] only. None when history is
    too short. Every momentum is scaled by the name's own daily vol, so one
    pooled model reads a utility and a biotech on the same scale."""
    if i < RV_WINDOW or j < SPY_TREND_WINDOW - 1 or j < RV_WINDOW:
        return None
    sd = daily_vol(closes, i)
    ssd = daily_vol(spy, j)
    if sd is None or ssd is None:
        return None
    c = closes[i]
    mom5 = math.log(c / closes[i - 5]) / (sd * math.sqrt(5))
    mom20 = math.log(c / closes[i - 20]) / (sd * math.sqrt(20))
    rev1 = math.log(c / closes[i - 1]) / sd
    sma = sum(spy[j - SPY_TREND_WINDOW + 1:j + 1]) / SPY_TREND_WINDOW
    trend = math.log(spy[j] / sma) / (ssd * math.sqrt(SPY_TREND_WINDOW))
    return [_clip(mom5), _clip(mom20), _clip(rev1), _clip(trend)]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_logistic(X: Sequence[Sequence[float]], y: Sequence[int], l2: float = RIDGE_L2,
                 iters: int = 60) -> List[float]:
    """Ridge logistic regression by Newton/IRLS. Returns [intercept, w...];
    the intercept is not penalized."""
    import numpy as np
    A = np.asarray(X, dtype=float)
    t = np.asarray(y, dtype=float)
    Xb = np.column_stack([np.ones(len(A)), A])
    w = np.zeros(Xb.shape[1])
    P = l2 * np.eye(Xb.shape[1])
    P[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        W = p * (1.0 - p)
        H = Xb.T @ (Xb * W[:, None]) + P
        g = Xb.T @ (p - t) + P @ w
        step = np.linalg.solve(H, g)
        w = w - step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return [float(v) for v in w]


def predict(coef: Sequence[float], x: Sequence[float]) -> float:
    return _sigmoid(coef[0] + sum(c * v for c, v in zip(coef[1:], x)))


def calibration_bins(pairs: Iterable[Tuple[float, int]], edges: Sequence[float] = CAL_EDGES) -> List[dict]:
    """Reliability table: per predicted-P(up) bin, the count, the mean
    prediction and the realized up-frequency (None when the bin is empty)."""
    acc = [[0, 0.0, 0] for _ in range(len(edges) - 1)]
    for p, o in pairs:
        for k in range(len(edges) - 1):
            last = k == len(edges) - 2
            if edges[k] <= p < edges[k + 1] or (last and p == edges[-1]):
                acc[k][0] += 1
                acc[k][1] += p
                acc[k][2] += int(o)
                break
    return [{"lo": edges[k], "hi": edges[k + 1], "n": n,
             "mean_p": round(sp / n, 4) if n else None,
             "freq": round(so / n, 4) if n else None}
            for k, (n, sp, so) in enumerate(acc)]


def brier(pairs: Sequence[Tuple[float, int]]) -> Optional[float]:
    return round(sum((p - o) ** 2 for p, o in pairs) / len(pairs), 5) if pairs else None


def build_samples(series: Dict[str, List[dict]], spy_bars: List[dict]) -> List[dict]:
    """Non-overlapping (stride SAMPLE_STRIDE) labeled samples from every name
    in `series`, aligned to SPY by date: {sym, date, label_date, x, y}."""
    spy_idx = {_bar_day(b): k for k, b in enumerate(spy_bars)}
    spy_c = [float(b["c"]) for b in spy_bars]
    out: List[dict] = []
    for sym, bars in series.items():
        closes = [float(b["c"]) for b in bars]
        days = [_bar_day(b) for b in bars]
        last = len(bars) - 1 - HORIZON_DAYS
        for i in range(last, RV_WINDOW - 1, -SAMPLE_STRIDE):
            j = spy_idx.get(days[i])
            if j is None:
                continue
            x = feature_row(closes, i, spy_c, j)
            if x is None:
                continue
            out.append({"sym": sym, "date": days[i], "label_date": days[i + HORIZON_DAYS], "x": x,
                        "y": 1 if closes[i + HORIZON_DAYS] > closes[i] else 0})
    out.sort(key=lambda s: (s["date"], s["sym"]))
    return out


def fit_and_validate(samples: List[dict]) -> Optional[dict]:
    """Time-ordered, purged split -> holdout metrics; then the live fit on
    everything. None when there are too few samples to fit honestly."""
    if not samples:
        return None
    dates = sorted({s["date"] for s in samples})
    split = dates[min(len(dates) - 1, int(len(dates) * TRAIN_FRACTION))]
    train = [s for s in samples if s["label_date"] < split]
    test = [s for s in samples if s["date"] >= split]
    if len(train) < MIN_TRAIN_SAMPLES or not test:
        return None
    coef_tr = fit_logistic([s["x"] for s in train], [s["y"] for s in train])
    base = sum(s["y"] for s in train) / len(train)
    pairs = [(predict(coef_tr, s["x"]), s["y"]) for s in test]
    eps = 1e-12
    logloss = -sum(o * math.log(max(p, eps)) + (1 - o) * math.log(max(1 - p, eps)) for p, o in pairs) / len(pairs)
    b_model = brier(pairs)
    b_clim = brier([(base, o) for _, o in pairs])
    validation = {
        "split_date": split, "n_train": len(train), "n_test": len(test),
        "train_base_rate": round(base, 4),
        "test_base_rate": round(sum(o for _, o in pairs) / len(pairs), 4),
        "brier": b_model, "brier_climatology": b_clim,
        "brier_skill": round(1 - b_model / b_clim, 4) if b_clim else None,
        "log_loss": round(logloss, 5),
        "hit_rate": round(sum(1 for p, o in pairs if (p >= 0.5) == (o == 1)) / len(pairs), 4),
        "calibration": calibration_bins(pairs),
    }
    coef = fit_logistic([s["x"] for s in samples], [s["y"] for s in samples])
    return {"coef": [round(c, 6) for c in coef], "features": list(FEATURES),
            "n_samples": len(samples), "first_date": dates[0], "last_date": dates[-1],
            "validation": validation}


def band(spot: float, sigma_h: float, z: float = BAND_Z) -> Tuple[float, float]:
    return spot * math.exp(-z * sigma_h), spot * math.exp(z * sigma_h)


def sigma_for(bars_closes: Sequence[float], d0: date, h_end: date, iv: Optional[float],
              iv_day: Optional[str]) -> Tuple[Optional[float], str, Optional[float]]:
    """(horizon sigma, source, annualized vol used). Chain IV when it is fresh
    enough, else 20-session realized vol."""
    iv_ok = isinstance(iv, (int, float)) and math.isfinite(iv) and 0.01 < iv < 5.0
    if iv_ok and iv_day:
        try:
            age = (d0 - date.fromisoformat(iv_day)).days
        except ValueError:
            age = None
        if age is not None and -1 <= age <= IV_MAX_AGE_DAYS:
            cal = max(1, (h_end - d0).days)
            return float(iv) * math.sqrt(cal / 365.0), "chain_iv30", float(iv)
    sd = daily_vol(bars_closes, len(bars_closes) - 1)
    if sd is None:
        return None, "none", None
    return sd * math.sqrt(HORIZON_DAYS), "realized_20d", sd * math.sqrt(252)


# ── persistence ─────────────────────────────────────────────────────────────
def _read_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    v = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("forecast store: skipped a corrupt line in %s", path)
                    continue
                if isinstance(v, dict):
                    out.append(v)
    except FileNotFoundError:
        return []
    return out


def _append_lines(path: str, rows: List[dict]) -> None:
    """One write() of every line, then fsync: a batch lands whole or not at all
    (bar a torn final line, which the reader skips)."""
    if not rows:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    blob = "".join(json.dumps(r, separators=(",", ":"), default=str) + "\n" for r in rows)
    with open(path, "a") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())


def load_store(path: Optional[str] = None, scores_path: Optional[str] = None,
               horizon: str = "weekly") -> Tuple[List[dict], List[dict], Dict[str, dict]]:
    """(batches, forecasts, scores_by_id) of ONE horizon, each deduped by id
    (first wins). The daily horizon shares the files: its lines carry
    horizon="daily"; a line without the field is weekly (every line written
    before 2026-11-04). The default keeps every weekly reader — the weekly
    report and scorer, ticker_dossier, brain_evidence — exactly as it was."""
    def mine(r: dict) -> bool:
        return (r.get("horizon") or "weekly") == horizon
    batches: Dict[str, dict] = {}
    forecasts: Dict[str, dict] = {}
    for r in _read_jsonl(path or FORECASTS_PATH):
        rid = r.get("id")
        if not rid or not mine(r):
            continue
        if r.get("kind") == "batch":
            batches.setdefault(rid, r)
        elif r.get("kind") == "forecast":
            forecasts.setdefault(rid, r)
    scores: Dict[str, dict] = {}
    for r in _read_jsonl(scores_path or FORECAST_SCORES_PATH):
        if r.get("kind") == "score" and r.get("id") and mine(r):
            scores.setdefault(r["id"], r)
    return list(batches.values()), list(forecasts.values()), scores


# ── bars ────────────────────────────────────────────────────────────────────
def _default_fetch(symbols: List[str], start: str, end: str, deadline: Optional[float]) -> Tuple[Dict[str, List[dict]], set]:
    """The existing paginated Alpaca bars helper (shadow_portfolio: shared
    180/min token bucket, split-adjusted, central feed). Returns (bars, unfetched)."""
    from shadow_portfolio import _fetch_bars_paginated
    unfetched: set = set()
    bars = _fetch_bars_paginated(symbols, start, end, deadline=deadline, unfetched=unfetched)
    return bars, unfetched


FetchFn = Callable[[List[str], str, str, Optional[float]], Tuple[Dict[str, List[dict]], set]]


def _norm_symbol(s: Any) -> Optional[str]:
    v = str(s or "").strip().upper()
    return v if v and len(v) <= 6 and v.replace(".", "").isalpha() else None


def normalize_universe(tickers: Optional[Sequence[Any]]) -> List[dict]:
    """[{symbol, group}] with SPY first as `index`, deduped (index > position >
    candidate), capped at MAX_TICKERS. Accepts strings or {symbol, group}."""
    rank = {g: k for k, g in enumerate(GROUPS)}
    best: Dict[str, str] = {"SPY": "index"}
    order: List[str] = ["SPY"]
    for t in tickers or []:
        sym = _norm_symbol(t.get("symbol") if isinstance(t, dict) else t)
        grp = (t.get("group") if isinstance(t, dict) else None) or "candidate"
        if sym is None or grp not in rank:
            continue
        if sym not in best:
            best[sym] = grp
            order.append(sym)
        elif rank[grp] < rank[best[sym]]:
            best[sym] = grp
    return [{"symbol": s, "group": best[s]} for s in order[:MAX_TICKERS]]


# ── generation ──────────────────────────────────────────────────────────────
def generate_weekly(tickers: Optional[Sequence[Any]] = None, iv: Optional[Dict[str, float]] = None,
                    iv_day: Optional[str] = None, candidates_scan_at: Optional[str] = None,
                    candidates_available: Optional[bool] = None, time_budget_s: float = 60.0,
                    now: Optional[datetime] = None, fetch: Optional[FetchFn] = None,
                    path: Optional[str] = None) -> dict:
    """Make this week's batch unless it already exists. Idempotent: one batch
    per week_key (ISO week of the first horizon session), checked on disk
    before any network call and again after the bars confirm d0. Writes
    nothing on a data or fit failure (the next hourly attempt retries)."""
    now = now or datetime.now(timezone.utc)
    path = path or FORECASTS_PATH
    fetch = fetch or _default_fetch
    deadline = time.monotonic() + max(5.0, float(time_budget_s))
    batches, _, _ = load_store(path, os.devnull)
    have = {b.get("week_key") for b in batches}
    pre_key = week_key(expected_base_date(now))
    if pre_key in have:
        return {"status": "skipped", "reason": "exists", "week_key": pre_key}

    universe = normalize_universe(tickers)
    syms = [u["symbol"] for u in universe]
    fetch_syms = list(dict.fromkeys(syms + list(TRAIN_BASKET)))
    start = (now - timedelta(days=HISTORY_CALENDAR_DAYS)).date().isoformat()
    end = now.date().isoformat()
    raw, unfetched = fetch(fetch_syms, start, end, deadline)
    series = {s: completed_bars(raw.get(s) or [], now) for s in fetch_syms if s not in unfetched}
    spy = series.get("SPY") or []
    if len(spy) < SPY_TREND_WINDOW + HORIZON_DAYS + 10:
        return {"status": "error", "reason": "spy_bars_unavailable", "unfetched": sorted(unfetched)}
    d0 = date.fromisoformat(_bar_day(spy[-1]))
    key = week_key(d0)
    if key in have:
        return {"status": "skipped", "reason": "exists", "week_key": key}

    samples = build_samples({s: b for s, b in series.items() if b}, spy)
    model = fit_and_validate(samples)
    if model is None:
        return {"status": "error", "reason": "too_few_training_samples", "samples": len(samples)}

    h_end = horizon_end(d0)
    made_at = now.astimezone(timezone.utc).isoformat()
    spy_idx = {_bar_day(b): k for k, b in enumerate(spy)}
    spy_c = [float(b["c"]) for b in spy]
    iv = {str(k).upper(): v for k, v in (iv or {}).items()}
    rows: List[dict] = []
    skipped: List[dict] = []
    for u in universe:
        sym = u["symbol"]
        if sym in unfetched:
            skipped.append({"symbol": sym, "reason": "bars_not_fetched"})
            continue
        bars = series.get(sym) or []
        if not bars or _bar_day(bars[-1]) != d0.isoformat():
            skipped.append({"symbol": sym, "reason": "no_bar_on_base_date"})
            continue
        closes = [float(b["c"]) for b in bars]
        x = feature_row(closes, len(closes) - 1, spy_c, spy_idx[d0.isoformat()])
        if x is None:
            skipped.append({"symbol": sym, "reason": "insufficient_history"})
            continue
        sig, src, ann = sigma_for(closes, d0, h_end, iv.get(sym), iv_day)
        if sig is None:
            skipped.append({"symbol": sym, "reason": "no_volatility"})
            continue
        p_up = predict(model["coef"], x)
        spot = closes[-1]
        lo, hi = band(spot, sig)
        rows.append({
            "kind": "forecast", "id": f"{key}:{sym}", "week_key": key, "symbol": sym, "group": u["group"],
            "made_at": made_at, "base_date": d0.isoformat(), "horizon_end": h_end.isoformat(),
            "spot": round(spot, 4), "p_up": round(p_up, 4),
            "direction": "up" if p_up >= 0.5 else "down",
            "p_call": round(max(p_up, 1 - p_up), 4),
            "range_lo": round(lo, 4), "range_hi": round(hi, 4), "band_coverage": BAND_COVERAGE,
            "sigma_h": round(sig, 6), "vol_source": src, "vol_annual": round(ann, 4) if ann else None,
            "features": dict(zip(FEATURES, [round(v, 4) for v in x])),
            "model_version": MODEL_VERSION,
        })
    if not rows:
        return {"status": "error", "reason": "no_forecastable_tickers", "skipped": skipped}
    batch = {
        "kind": "batch", "id": f"batch:{key}", "week_key": key, "made_at": made_at,
        "base_date": d0.isoformat(), "horizon_end": h_end.isoformat(),
        "on_time": now < _open_after(d0), "model_version": MODEL_VERSION,
        "model": model, "train_tickers": sorted(s for s, b in series.items() if b),
        "n_forecasts": len(rows), "skipped": skipped,
        "candidates_available": bool(candidates_available) if candidates_available is not None
        else any(u["group"] == "candidate" for u in universe),
        "candidates_scan_at": candidates_scan_at, "iv_day": iv_day,
        "iv_names": sum(1 for r in rows if r["vol_source"] == "chain_iv30"),
    }
    _append_lines(path, rows + [batch])
    return {"status": "ok", "week_key": key, "base_date": d0.isoformat(), "n": len(rows),
            "skipped": len(skipped), "holdout_brier_skill": model["validation"]["brier_skill"]}


# ── daily horizon: the next session's close (Part C, 2026-11-04, SHADOW) ────
# One call per name per target session D (the next session to close),
# alongside the weekly call. Stored in the SAME append-only files as the weekly
# (kind forecast/batch/score) with horizon="daily"; load_store() filters by
# horizon, so every weekly reader (the weekly report, ticker_dossier,
# brain_evidence, the weekly scorer) is unchanged.
#
# WHAT A DAILY CALL IS
#   * Base d0 = the last completed session; target D = the next session. The
#     reference price `spot` is the d0 close for a PRE-OPEN call (made after
#     the d0 close, before D opens — the normal case: the first hourly attempt
#     from DAILY_PREOPEN_FROM_ET, once the d0 VIX close is in the payload or
#     DAILY_VIX_CUTOFF_ET passes) and the latest Alpaca price for an INTRADAY
#     call (a name that joins the universe during D, or a batch first made
#     after D opened — `on_time: false`, scored in its own bucket).
#   * Direction: P(up) = P(close_D > spot) = the point-in-time BASE RATE —
#     the up-frequency of the fixed TRAIN_BASKET's last DAILY_BASE_RATE_SESSIONS
#     daily returns — shrunk all the way: no candidate feature beat it on the
#     purged walk-forward holdout (research/forecast_direction_study_2026-11-04.md);
#     DAILY_DIRECTION_FEATURES is empty and a test pins it. Intraday calls use
#     0.5 + (base - 0.5) * sqrt(f): the drift still to come scales with time
#     while the noise scales with its square root.
#   * Range: forecast_range (fc-range-har-v1): spot * exp(+/- q), q = the HAR +
#     implied-vol 68.27% half-width for the full session from closes <= d0 (VIX
#     for SPY when the payload's VIX is the d0 close; chain IV30 for a name when
#     iv_day == d0; else HAR alone, labeled), times sqrt(f) for an intraday call.
#   * Live band (display only, never scored): during D every hourly run
#     re-centres each call's band on the latest price with the variance still
#     to come, q_1d * sqrt(f(now)) — written to FORECAST_DAILY_LIVE_PATH
#     (overwritten; a display cache, not a record).
#   * Scored once, after D's close (the daily scorer inside score_matured):
#     hit, Brier (and the base-rate reference Brier), inside/outside the band,
#     the interval score and the constant-band reference (q_clim) for range
#     skill. Realized = base_close * (C_D / C_d0) from one fetched series, so a
#     split between making and scoring cannot fake a result.


def _hm(n: datetime) -> Tuple[int, int]:
    return (n.hour, n.minute)


def daily_target(now: datetime) -> Tuple[date, date, str]:
    """(d0, D, phase). d0 = the last completed session, D = the next one (the
    close a daily call forecasts). phase: "pre_open" (before D's 09:30 ET),
    "open" (D's regular session), "closing" (D's session has ended but its
    daily bar is not final yet — nothing is made then)."""
    d0 = expected_base_date(now)
    D = _next_trading_day(d0)
    n = _et(now)
    if n.date() < D:
        return d0, D, "pre_open"
    from market_calendar import is_half_day
    close = HALF_DAY_CLOSE_ET if is_half_day(D) else SESSION_CLOSE_ET
    if _hm(n) < SESSION_OPEN_ET:
        return d0, D, "pre_open"
    return d0, D, ("open" if _hm(n) < close else "closing")


def session_minutes(now: datetime, D: date) -> float:
    """ET minute-of-day on the regular-session clock: a half day's 3.5 hours
    are stretched onto 09:30-16:00 so the remaining-variance table applies."""
    from market_calendar import is_half_day
    n = _et(now)
    m = n.hour * 60 + n.minute + n.second / 60.0
    o = SESSION_OPEN_ET[0] * 60 + SESSION_OPEN_ET[1]
    if is_half_day(D) and m >= o:
        m = o + (m - o) * 390.0 / 210.0
    return m


def preopen_ready(now: datetime, d0: date) -> bool:
    """A pre-open batch waits for the options archive's post-close capture
    (chain IV30 for the d0 close) on the evening of d0."""
    n = _et(now)
    return n.date() > d0 or _hm(n) >= DAILY_PREOPEN_FROM_ET


def vix_cutoff_passed(now: datetime, D: date) -> bool:
    n = _et(now)
    return n.date() > D or (n.date() == D and _hm(n) >= DAILY_VIX_CUTOFF_ET)


def base_rate(series: Dict[str, List[dict]], d0: str, basket: Sequence[str] = TRAIN_BASKET,
              window: int = DAILY_BASE_RATE_SESSIONS) -> Optional[dict]:
    """The pooled up-frequency (close > previous close) of each basket name's
    last `window` daily returns ending at d0 — closes on or before d0 only."""
    ups = n = 0
    first, used = None, []
    for s in basket:
        bars = [b for b in (series.get(s) or []) if _bar_day(b) <= d0]
        if len(bars) < 2 or _bar_day(bars[-1]) != d0:
            continue
        tail = bars[-(window + 1):]
        c = [float(b["c"]) for b in tail]
        n += len(c) - 1
        ups += sum(1 for a, b in zip(c, c[1:]) if b > a)
        f = _bar_day(tail[0])
        first = f if first is None or f < first else first
        used.append(s)
    if n < DAILY_BASE_RATE_MIN:
        return None
    return {"p": round(ups / n, 4), "n": n, "first": first, "last": d0, "basket": used}


def clim_q(closes: Sequence[float], window: int = DAILY_CLIM_SESSIONS) -> Optional[float]:
    """The constant-band reference: the 68.27% quantile of the name's own
    |daily log return| over its last `window` sessions (no conditioning)."""
    c = [float(x) for x in closes[-(window + 1):]]
    if len(c) - 1 < DAILY_CLIM_MIN or any(x <= 0 for x in c):
        return None
    import numpy as np
    r = np.abs(np.diff(np.log(np.asarray(c))))
    q = float(np.quantile(r, FR.RANGE_TAU))
    return q if q > 0 else None


def daily_p_up(base_p: float, f: float = 1.0) -> float:
    """The daily direction = the base rate, shrunk all the way (no feature
    passed). For a call made with a fraction f of the session's variance still
    to come, the drift's z-score shrinks by sqrt(f)."""
    return 0.5 + (float(base_p) - 0.5) * math.sqrt(max(0.0, min(1.0, float(f))))


def _latest_price(raw_bars: Sequence[dict], D: date) -> Optional[float]:
    """The latest Alpaca price for session D: the close of D's (partial)
    daily bar — the same reading ticker_dossier uses for its live price."""
    for b in reversed(list(raw_bars or [])):
        if _bar_day(b) == D.isoformat():
            c = b.get("c")
            return float(c) if isinstance(c, (int, float)) and c > 0 else None
        if _bar_day(b) < D.isoformat():
            return None
    return None


def daily_call(sym: str, group: str, closes: Sequence[float], d0: date, D: date, spot: float, made_at: str,
               timing: str, f: float, base_p: float, iv_val: Optional[float], iv_src: Optional[str],
               iv_day: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
    """One daily call from the closes up to the d0 close, or (None, reason)."""
    rng = FR.daily_range(closes, sym, iv_val, iv_src)
    if rng is None:
        return None, "insufficient_history"
    q1 = rng["q"]
    q = q1 * math.sqrt(f)
    if not (q > 0):
        return None, "session_over"
    qc = clim_q(closes)
    p = daily_p_up(base_p, f)
    pb = p                                  # the base-rate reference IS the call (no feature shipped)
    lo, hi = band(spot, q)
    used_iv = rng["range_source"] != "har"
    return {
        "kind": "forecast", "horizon": "daily", "id": f"D:{D.isoformat()}:{sym}", "session": D.isoformat(),
        "symbol": sym, "group": group, "made_at": made_at, "timing": timing, "on_time": timing == "pre_open",
        "base_date": d0.isoformat(), "base_close": round(float(closes[-1]), 4), "spot": round(float(spot), 4),
        "horizon_end": D.isoformat(), "p_up": round(p, 4), "p_base": round(pb, 4),
        "direction": "up" if p >= 0.5 else "down", "p_call": round(max(p, 1 - p), 4),
        "base_rate": round(float(base_p), 4), "direction_model": DAILY_DIRECTION_MODEL,
        "remaining_fraction": round(f, 4), "q": round(q, 6), "q_1d": round(q1, 6),
        "q_clim": round(qc * math.sqrt(f), 6) if qc else None,
        "range_lo": round(lo, 4), "range_hi": round(hi, 4), "band_coverage": BAND_COVERAGE,
        "range_source": rng["range_source"], "range_model_version": FR.RANGE_MODEL_VERSION,
        "iv": round(float(iv_val), 4) if used_iv and iv_val is not None else None,
        "iv_day": iv_day if used_iv else None,
        "regressors": rng["regressors"], "model_version": DAILY_MODEL_VERSION,
    }, None


def live_row(call: dict, price: float, f: float, at: str) -> Optional[dict]:
    """The display-only live band for a call: centred on the latest price,
    with the variance still to come (q_1d * sqrt(f))."""
    q1 = float(call.get("q_1d") or 0.0)
    if not (q1 > 0) or not (price > 0) or not (f > 0):
        return None
    q = q1 * math.sqrt(f)
    lo, hi = band(price, q)
    return {"price": round(price, 4), "price_day": call["session"], "at": at, "remaining_fraction": round(f, 4),
            "q": round(q, 6), "lo": round(lo, 4), "hi": round(hi, 4),
            "vs_spot_pct": round((price / float(call["spot"]) - 1) * 100, 3)}


def _write_json_atomic(path: str, obj: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"), default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_live(path: Optional[str] = None) -> Optional[dict]:
    try:
        with open(path or FORECAST_DAILY_LIVE_PATH) as fh:
            v = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        log.warning("forecast daily live file unreadable: %s", e)
        return None
    return v if isinstance(v, dict) else None


def _vix_decimal(vix: Any) -> Optional[float]:
    v = vix.get("value") if isinstance(vix, dict) else None
    return float(v) / 100.0 if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def generate_daily(tickers: Optional[Sequence[Any]] = None, iv: Optional[Dict[str, float]] = None,
                   iv_day: Optional[str] = None, vix: Optional[dict] = None, candidates_scan_at: Optional[str] = None,
                   candidates_available: Optional[bool] = None, time_budget_s: float = 60.0,
                   now: Optional[datetime] = None, fetch: Optional[FetchFn] = None, path: Optional[str] = None,
                   live_path: Optional[str] = None) -> dict:
    """Make (or extend) the daily calls for the next session to close and,
    during that session, refresh the live band. One call per (session,
    symbol), first wins, checked on disk before any network call. Writes
    nothing on a data failure — the next hourly attempt retries. `vix` is
    {value, day} from the Node side's Cboe archive; `iv` / `iv_day` the
    chain IV30 map of the newest archived day (the weekly payload's)."""
    now = now or datetime.now(timezone.utc)
    path = path or FORECASTS_PATH
    live_path = live_path or FORECAST_DAILY_LIVE_PATH
    fetch = fetch or _default_fetch
    deadline = time.monotonic() + max(5.0, float(time_budget_s))
    d0, D, phase = daily_target(now)
    sess, d0s = D.isoformat(), d0.isoformat()
    if phase == "closing":
        return {"status": "skipped", "reason": "session_closing", "session": sess}
    batches, calls, _ = load_store(path, os.devnull, horizon="daily")
    batch = next((b for b in batches if b.get("session") == sess), None)
    mine = [c for c in calls if c.get("session") == sess]
    have = {c.get("symbol") for c in mine}
    universe = normalize_universe(tickers)
    missing = [u for u in universe if u["symbol"] not in have]
    timing = "pre_open" if phase == "pre_open" else "intraday"
    vix_dec = _vix_decimal(vix)
    vix_ok = isinstance(vix, dict) and vix.get("day") == d0s and FR.iv_regressor(vix_dec) is not None
    if phase == "pre_open":
        if not missing:
            return {"status": "skipped", "reason": "exists", "session": sess}
        if not preopen_ready(now, d0):
            return {"status": "waiting", "reason": "options_capture_window", "session": sess, "base_date": d0s}
        if batch is None and not vix_ok and not vix_cutoff_passed(now, D):
            return {"status": "waiting", "reason": "vix_not_published", "session": sess, "base_date": d0s,
                    "vix_day": vix.get("day") if isinstance(vix, dict) else None}
    new_syms = [u["symbol"] for u in missing]
    long_syms = list(dict.fromkeys(new_syms + (list(TRAIN_BASKET) if batch is None else [])))
    short_syms = sorted(s for s in have if s and s not in long_syms) if phase == "open" else []
    end = now.date().isoformat()
    raw: Dict[str, List[dict]] = {}
    unfetched: set = set()
    if long_syms:
        r1, u1 = fetch(long_syms, (now - timedelta(days=HISTORY_CALENDAR_DAYS)).date().isoformat(), end, deadline)
        raw.update(r1 or {})
        unfetched |= set(u1 or ())
    if short_syms:
        r2, u2 = fetch(short_syms, (now - timedelta(days=DAILY_LIVE_FETCH_DAYS)).date().isoformat(), end, deadline)
        raw.update(r2 or {})
        unfetched |= set(u2 or ())
    series = {s: completed_bars(raw.get(s) or [], now) for s in raw}
    made_at = now.astimezone(timezone.utc).isoformat()
    mins = session_minutes(now, D)
    br = None
    if batch is None:
        spy = series.get("SPY") or []
        if not spy or _bar_day(spy[-1]) != d0s:
            return {"status": "error", "reason": "spy_base_bar_missing", "session": sess, "base_date": d0s,
                    "unfetched": sorted(unfetched)}
        br = base_rate(series, d0s)
        if br is None:
            return {"status": "error", "reason": "base_rate_unavailable", "session": sess}
        base_p = br["p"]
    else:
        bp = (batch.get("base_rate") or {}).get("p")
        if not isinstance(bp, (int, float)):
            return {"status": "error", "reason": "batch_without_base_rate", "session": sess}
        base_p = float(bp)
    ivmap = {str(k).upper(): v for k, v in (iv or {}).items()}
    rows: List[dict] = []
    skipped: List[dict] = []
    for u in missing:
        sym = u["symbol"]
        if sym in unfetched:
            skipped.append({"symbol": sym, "reason": "bars_not_fetched"})
            continue
        bars = series.get(sym) or []
        if not bars or _bar_day(bars[-1]) != d0s:
            skipped.append({"symbol": sym, "reason": "no_bar_on_base_date"})
            continue
        closes = [float(b["c"]) for b in bars]
        if timing == "pre_open":
            f, spot = 1.0, closes[-1]
        else:
            f = FR.remaining_variance_fraction(mins, sym)
            spot = _latest_price(raw.get(sym) or [], D)
            if f is None or spot is None or f <= 0:
                skipped.append({"symbol": sym, "reason": "no_live_price"})
                continue
        if sym == FR.INDEX_IV_SYMBOL and vix_ok:
            iv_val, iv_src, ivd = vix_dec, "vix", d0s
        elif iv_day == d0s and isinstance(ivmap.get(sym), (int, float)):
            iv_val, iv_src, ivd = float(ivmap[sym]), "chain_iv30", iv_day
        else:
            iv_val, iv_src, ivd = None, None, None
        call, why = daily_call(sym, u["group"], closes, d0, D, spot, made_at, timing, f, base_p, iv_val, iv_src, ivd)
        if call is None:
            skipped.append({"symbol": sym, "reason": why})
            continue
        rows.append(call)
    if batch is None and not rows:
        return {"status": "error", "reason": "no_forecastable_tickers", "session": sess, "skipped": skipped}
    if batch is None:
        srcs: Dict[str, int] = {}
        for r in rows:
            srcs[r["range_source"]] = srcs.get(r["range_source"], 0) + 1
        rows.append({
            "kind": "batch", "horizon": "daily", "id": f"batch:D:{sess}", "session": sess, "made_at": made_at,
            "base_date": d0s, "timing": timing, "on_time": timing == "pre_open", "model_version": DAILY_MODEL_VERSION,
            "range_model_version": FR.RANGE_MODEL_VERSION, "direction_model": DAILY_DIRECTION_MODEL,
            "direction_features": list(DAILY_DIRECTION_FEATURES), "base_rate": br,
            "vix": {"value": vix.get("value"), "day": vix.get("day")} if isinstance(vix, dict) else None,
            "vix_used": bool(vix_ok), "iv_day": iv_day, "n_forecasts": len(rows), "skipped": skipped,
            "candidates_available": bool(candidates_available) if candidates_available is not None
            else any(u["group"] == "candidate" for u in universe),
            "candidates_scan_at": candidates_scan_at, "range_sources": srcs,
        })
    _append_lines(path, rows)
    new_calls = [r for r in rows if r["kind"] == "forecast"]
    live_n = 0
    if phase == "open":
        live: Dict[str, dict] = {}
        for c in mine + new_calls:
            sym = c.get("symbol")
            price = _latest_price(raw.get(sym) or [], D)
            f = FR.remaining_variance_fraction(mins, sym)
            if price is None or f is None:
                continue
            lr = live_row(c, price, f, made_at)
            if lr:
                live[sym] = lr
        if live:
            _write_json_atomic(live_path, {"session": sess, "made_at": made_at, "et_minutes": round(mins, 1), "rows": live})
        live_n = len(live)
    return {"status": "ok", "session": sess, "base_date": d0s, "timing": timing, "new_calls": len(new_calls),
            "batch": "new" if batch is None else "existing", "live_rows": live_n, "skipped": len(skipped)}


# ── scoring (measurement code — keep it plain) ──────────────────────────────
def score_one(fc: dict, bars: List[dict]) -> Optional[dict]:
    """The score for one forecast from completed, ascending bars, or None when
    the 5th session after the base date has not closed yet. Realized move is a
    RATIO of the fetched bars (d0+5 / d0), applied to the stored spot, so a
    split between making and scoring cannot fake a hit."""
    days = [_bar_day(b) for b in bars]
    try:
        i = days.index(fc["base_date"])
    except ValueError:
        return None
    if i + HORIZON_DAYS >= len(bars):
        return None
    c0, c5 = float(bars[i]["c"]), float(bars[i + HORIZON_DAYS]["c"])
    realized = float(fc["spot"]) * (c5 / c0)
    up = 1 if c5 > c0 else 0
    p = float(fc["p_up"])
    return {
        "kind": "score", "id": fc["id"], "status": "resolved",
        "resolved_date": days[i + HORIZON_DAYS], "realized_close": round(realized, 4),
        "realized_ret_pct": round((c5 / c0 - 1) * 100, 4),
        "outcome_up": up, "hit": int((fc["direction"] == "up") == (up == 1)),
        "in_range": int(float(fc["range_lo"]) <= realized <= float(fc["range_hi"])),
        "brier": round((p - up) ** 2, 6),
    }


def score_matured(now: Optional[datetime] = None, fetch: Optional[FetchFn] = None,
                  time_budget_s: float = 60.0, path: Optional[str] = None,
                  scores_path: Optional[str] = None) -> dict:
    """Resolve every WEEKLY forecast whose 5th session has closed, then every
    DAILY call whose session has closed (score_daily, stats under "daily").
    Append-only; a fetch failure is retried next run (never a strike); a name
    with no price data SCORE_GIVE_UP_DAYS past its horizon gets a terminal
    no_price_data line, excluded from every statistic. The top-level stats
    stay the weekly ones."""
    now = now or datetime.now(timezone.utc)
    fetch = fetch or _default_fetch
    deadline = time.monotonic() + max(5.0, float(time_budget_s))
    stats = _score_weekly(now, fetch, deadline, path, scores_path)
    stats["daily"] = score_daily(now, fetch, deadline, path, scores_path)
    return stats


def _score_weekly(now: datetime, fetch: FetchFn, deadline: float, path: Optional[str],
                  scores_path: Optional[str]) -> dict:
    _, forecasts, scores = load_store(path, scores_path)
    stats = {"open": 0, "resolved": 0, "pending": 0, "gave_up": 0, "fetch_errors": 0}
    todo = [f for f in forecasts if f["id"] not in scores]
    stats["open"] = len(todo)
    # The planned end can only be EARLY (a holiday missing from the calendar),
    # never late, so nothing due is skipped; an early one just stays pending.
    due = [f for f in todo if date.fromisoformat(f["horizon_end"]) <= _et(now).date()]
    stats["pending"] = len(todo) - len(due)
    if not due:
        return stats
    start = (min(date.fromisoformat(f["base_date"]) for f in due) - timedelta(days=7)).isoformat()
    syms = sorted({f["symbol"] for f in due})
    raw, unfetched = fetch(syms, start, now.date().isoformat(), deadline)
    out: List[dict] = []
    for f in due:
        if f["symbol"] in unfetched:
            stats["fetch_errors"] += 1
            continue
        s = score_one(f, completed_bars(raw.get(f["symbol"]) or [], now))
        if s is None:
            late = (_et(now).date() - date.fromisoformat(f["horizon_end"])).days
            if late > SCORE_GIVE_UP_DAYS:
                out.append({"kind": "score", "id": f["id"], "status": "no_price_data"})
                stats["gave_up"] += 1
            else:
                stats["pending"] += 1
            continue
        s["scored_at"] = now.astimezone(timezone.utc).isoformat()
        out.append(s)
        stats["resolved"] += 1
    _append_lines(scores_path or FORECAST_SCORES_PATH, out)
    return stats


# ── daily scoring (measurement code — keep it plain) ────────────────────────
def score_daily_one(fc: dict, bars: List[dict]) -> Optional[dict]:
    """The score of one daily call from completed, ascending bars, or None
    while the session's bar is not final. realized = base_close * (C_D /
    C_d0) from ONE fetched series (split-robust), compared with the stored
    spot (the d0 close for a pre-open call, the live price for an intraday one)."""
    days = [_bar_day(b) for b in bars]
    try:
        i0 = days.index(fc["base_date"])
        i1 = days.index(fc["session"])
    except ValueError:
        return None
    if i1 <= i0:
        return None
    c0, c1 = float(bars[i0]["c"]), float(bars[i1]["c"])
    realized = float(fc["base_close"]) * (c1 / c0)
    spot = float(fc["spot"])
    up = int(c1 > c0) if fc.get("timing") == "pre_open" else int(realized > spot)
    y = math.log(realized / spot)
    p = float(fc["p_up"])
    pb = float(fc.get("p_base", p))
    qc = fc.get("q_clim")
    return {
        "kind": "score", "horizon": "daily", "id": fc["id"], "status": "resolved", "resolved_date": fc["session"],
        "realized_close": round(realized, 4), "realized_ret_pct": round((realized / spot - 1) * 100, 4),
        "outcome_up": up, "hit": int((fc["direction"] == "up") == (up == 1)),
        "in_range": int(float(fc["range_lo"]) <= realized <= float(fc["range_hi"])),
        "brier": round((p - up) ** 2, 6), "brier_base": round((pb - up) ** 2, 6),
        "interval_score": round(FR.interval_score(float(fc["q"]), y) * 100, 5),
        "interval_score_clim": round(FR.interval_score(float(qc), y) * 100, 5) if isinstance(qc, (int, float)) and qc > 0 else None,
    }


def score_daily(now: datetime, fetch: FetchFn, deadline: float, path: Optional[str] = None,
                scores_path: Optional[str] = None) -> dict:
    """Resolve every daily call whose session has closed (append-only; a
    fetch failure is retried; no data SCORE_GIVE_UP_DAYS after the session
    gives a terminal no_price_data line, excluded from every statistic)."""
    _, calls, scores = load_store(path, scores_path, horizon="daily")
    todo = [c for c in calls if c["id"] not in scores]
    stats = {"open": len(todo), "resolved": 0, "pending": 0, "gave_up": 0, "fetch_errors": 0}
    due = [c for c in todo if _session_done(date.fromisoformat(c["session"]), now)]
    stats["pending"] = len(todo) - len(due)
    if not due:
        return stats
    start = (min(date.fromisoformat(c["base_date"]) for c in due) - timedelta(days=7)).isoformat()
    syms = sorted({c["symbol"] for c in due})
    raw, unfetched = fetch(syms, start, now.date().isoformat(), deadline)
    out: List[dict] = []
    today = _et(now).date()
    for c in due:
        if c["symbol"] in unfetched:
            stats["fetch_errors"] += 1
            continue
        s = score_daily_one(c, completed_bars(raw.get(c["symbol"]) or [], now))
        if s is None:
            if (today - date.fromisoformat(c["session"])).days > SCORE_GIVE_UP_DAYS:
                out.append({"kind": "score", "horizon": "daily", "id": c["id"], "status": "no_price_data"})
                stats["gave_up"] += 1
            else:
                stats["pending"] += 1
            continue
        s["scored_at"] = now.astimezone(timezone.utc).isoformat()
        out.append(s)
        stats["resolved"] += 1
    _append_lines(scores_path or FORECAST_SCORES_PATH, out)
    return stats


# ── the report (/api/forecast) ──────────────────────────────────────────────
def _rate(k: int, n: int) -> Optional[float]:
    return round(k / n, 4) if n else None


def accuracy(rows: List[Tuple[dict, dict]]) -> dict:
    """Summary over resolved (forecast, score) pairs."""
    n = len(rows)
    pairs = [(float(f["p_up"]), int(s["outcome_up"])) for f, s in rows]
    b = brier(pairs)
    hits = sum(int(s["hit"]) for _, s in rows)
    inr = sum(int(s["in_range"]) for _, s in rows)
    return {"n": n, "thin": n < MIN_N_FOR_RATES, "hits": hits, "hit_rate": _rate(hits, n),
            "brier": b, "brier_coinflip": 0.25,
            "brier_skill_vs_coinflip": round(1 - b / 0.25, 4) if b is not None else None,
            "in_range": inr, "in_range_rate": _rate(inr, n), "in_range_target": BAND_COVERAGE,
            "up_rate": _rate(sum(o for _, o in pairs), n)}


def report(path: Optional[str] = None, scores_path: Optional[str] = None,
           now: Optional[datetime] = None, max_weeks: int = 26, live_path: Optional[str] = None) -> dict:
    """The weekly report (top-level keys, unchanged) plus the daily horizon
    under "daily" (daily_report)."""
    now = now or datetime.now(timezone.utc)
    batches, forecasts, scores = load_store(path, scores_path)
    batches.sort(key=lambda b: (b.get("week_key") or "", b.get("made_at") or ""))
    by_week: Dict[str, List[dict]] = {}
    for f in forecasts:
        by_week.setdefault(f.get("week_key") or "", []).append(f)
    resolved = [(f, scores[f["id"]]) for f in forecasts
                if scores.get(f["id"], {}).get("status") == "resolved"]
    latest = None
    if batches:
        b = batches[-1]
        order = {g: k for k, g in enumerate(GROUPS)}
        calls = sorted(by_week.get(b["week_key"], []),
                       key=lambda f: (order.get(f.get("group"), 9), -float(f.get("p_call") or 0), f.get("symbol")))
        latest = {k: v for k, v in b.items() if k not in ("kind", "id")}
        latest["forecasts"] = [{**{k: v for k, v in f.items() if k not in ("kind",)},
                                "score": scores.get(f["id"])} for f in calls]
    weeks = []
    for b in reversed(batches[-max_weeks:]):
        fs = by_week.get(b["week_key"], [])
        rs = [(f, scores[f["id"]]) for f in fs if scores.get(f["id"], {}).get("status") == "resolved"]
        a = accuracy(rs)
        weeks.append({"week_key": b["week_key"], "made_at": b.get("made_at"), "base_date": b.get("base_date"),
                      "horizon_end": b.get("horizon_end"), "on_time": b.get("on_time"),
                      "model_version": b.get("model_version"), "n": len(fs), "resolved": a["n"],
                      "hits": a["hits"], "hit_rate": a["hit_rate"], "brier": a["brier"],
                      "in_range_rate": a["in_range_rate"]})
    by_group = {g: accuracy([r for r in resolved if r[0].get("group") == g]) for g in GROUPS}
    by_vol = {v: accuracy([r for r in resolved if r[0].get("vol_source") == v])
              for v in ("chain_iv30", "realized_20d")}
    return {
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION, "experimental": True, "shadow": True,
        "horizon_days": HORIZON_DAYS, "band_coverage": BAND_COVERAGE,
        "latest": latest,
        "accuracy": {**accuracy(resolved),
                     "on_time": accuracy([r for r in resolved if batch_on_time(batches, r[0])]),
                     "by_group": by_group, "by_vol_source": by_vol},
        "calibration": {"edges": list(CAL_EDGES),
                        "bins": calibration_bins((float(f["p_up"]), int(s["outcome_up"])) for f, s in resolved)},
        "weeks": weeks,
        "open": sum(1 for f in forecasts if f["id"] not in scores),
        "method": ("P(up) = ridge logistic regression on vol-scaled 5- and 20-session momentum, 1-session "
                   "reversal and SPY's distance from its 50-day average, fit weekly on 3 years of daily bars "
                   "(15 index/sector ETFs + the week's names, one sample per 5 sessions), validated on a "
                   "time-ordered 30% holdout. Range = 68% band of a zero-drift lognormal from chain IV30 "
                   "when archived, else 20-session realized vol."),
        "caveats": [
            "Experimental model, shadow only: nothing trades on these calls.",
            "Training names exist today (survivorship bias); edges at a 1-week horizon are small, so most "
            "probabilities sit near the base rate — read the holdout Brier skill before trusting any call.",
            f"Accuracy figures under {MIN_N_FOR_RATES} resolved calls are marked thin.",
        ],
        "daily": daily_report(path, scores_path, live_path, now),
    }


def batch_on_time(batches: List[dict], fc: dict) -> bool:
    for b in batches:
        if b.get("week_key") == fc.get("week_key"):
            return bool(b.get("on_time"))
    return False


# ── the daily report (/api/forecast `daily`) ────────────────────────────────
def daily_accuracy(rows: List[Tuple[dict, dict]]) -> dict:
    """Summary over resolved (daily call, score) pairs: direction (hits,
    Brier vs the base-rate reference and vs a coin flip) and range (inside
    the band, interval score, skill vs the name's constant band)."""
    n = len(rows)
    pairs = [(float(f["p_up"]), int(s["outcome_up"])) for f, s in rows]
    b = brier(pairs)
    bb = brier([(float(f.get("p_base", f["p_up"])), int(s["outcome_up"])) for f, s in rows])
    hits = sum(int(s["hit"]) for _, s in rows)
    inr = sum(int(s["in_range"]) for _, s in rows)
    isv = [float(s["interval_score"]) for _, s in rows if isinstance(s.get("interval_score"), (int, float))]
    paired = [(float(s["interval_score"]), float(s["interval_score_clim"])) for _, s in rows
              if isinstance(s.get("interval_score"), (int, float)) and isinstance(s.get("interval_score_clim"), (int, float))]
    clim_sum = sum(c for _, c in paired)
    widths = [200.0 * float(f["q"]) for f, _ in rows if isinstance(f.get("q"), (int, float))]
    return {"n": n, "thin": n < MIN_N_FOR_RATES, "hits": hits, "hit_rate": _rate(hits, n),
            "up_rate": _rate(sum(o for _, o in pairs), n),
            "brier": b, "brier_base_rate": bb, "brier_coinflip": 0.25,
            "bss_vs_base_rate": round(1 - b / bb, 4) if b is not None and bb else None,
            "bss_vs_coinflip": round(1 - b / 0.25, 4) if b is not None else None,
            "in_range": inr, "in_range_rate": _rate(inr, n), "in_range_target": BAND_COVERAGE,
            "interval_score_pct": round(sum(isv) / len(isv), 5) if isv else None,
            "interval_score_clim_pct": round(clim_sum / len(paired), 5) if paired else None,
            "range_skill_vs_constant": round(1 - sum(a for a, _ in paired) / clim_sum, 4) if paired and clim_sum > 0 else None,
            "range_skill_n": len(paired),
            "width_pct": round(sum(widths) / len(widths), 4) if widths else None}


def daily_report(path: Optional[str] = None, scores_path: Optional[str] = None, live_path: Optional[str] = None,
                 now: Optional[datetime] = None, max_sessions: int = DAILY_REPORT_SESSIONS) -> dict:
    now = now or datetime.now(timezone.utc)
    batches, calls, scores = load_store(path, scores_path, horizon="daily")
    by_sess: Dict[str, List[dict]] = {}
    for c in calls:
        by_sess.setdefault(c.get("session") or "", []).append(c)
    sessions = sorted(s for s in by_sess if s)
    resolved = [(c, scores[c["id"]]) for c in calls if scores.get(c["id"], {}).get("status") == "resolved"]
    latest = None
    live_meta = None
    if sessions:
        s = sessions[-1]
        b = next((x for x in sorted(batches, key=lambda x: x.get("made_at") or "") if x.get("session") == s), None)
        live = read_live(live_path)
        live_rows = live.get("rows") if isinstance(live, dict) and live.get("session") == s and isinstance(live.get("rows"), dict) else {}
        if live_rows:
            live_meta = {"session": s, "made_at": live.get("made_at"), "et_minutes": live.get("et_minutes")}
        order = {g: k for k, g in enumerate(GROUPS)}
        cs = sorted(by_sess[s], key=lambda c: (order.get(c.get("group"), 9), c.get("made_at") or "", c.get("symbol") or ""))
        latest = {k: v for k, v in (b or {"session": s, "base_date": cs[0].get("base_date"), "made_at": cs[0].get("made_at"),
                                          "timing": cs[0].get("timing"), "on_time": cs[0].get("on_time")}).items()
                  if k not in ("kind", "id", "horizon")}
        latest["n_forecasts"] = len(cs)
        srcs: Dict[str, int] = {}
        for c in cs:                      # over every call of the session, intraday additions included
            srcs[c.get("range_source") or "unknown"] = srcs.get(c.get("range_source") or "unknown", 0) + 1
        latest["range_sources"] = srcs
        latest["forecasts"] = [{**{k: v for k, v in c.items() if k not in ("kind", "regressors")},
                                "score": scores.get(c["id"]), "live": live_rows.get(c.get("symbol"))} for c in cs]
    rows = []
    for s in reversed(sessions[-max_sessions:]):
        cs = by_sess[s]
        rs = [(c, scores[c["id"]]) for c in cs if scores.get(c["id"], {}).get("status") == "resolved"]
        a = daily_accuracy(rs)
        rows.append({"session": s, "n": len(cs), "resolved": a["n"], "hits": a["hits"], "hit_rate": a["hit_rate"],
                     "brier": a["brier"], "in_range_rate": a["in_range_rate"], "interval_score_pct": a["interval_score_pct"],
                     "range_skill_vs_constant": a["range_skill_vs_constant"]})
    acc = daily_accuracy(resolved)
    return {
        "model_version": DAILY_MODEL_VERSION, "range_model_version": FR.RANGE_MODEL_VERSION,
        "direction_model": DAILY_DIRECTION_MODEL, "direction_features": list(DAILY_DIRECTION_FEATURES),
        "horizon": "next session close", "band_coverage": BAND_COVERAGE,
        "latest": latest, "live": live_meta,
        "accuracy": {**acc,
                     "by_timing": {t: daily_accuracy([r for r in resolved if r[0].get("timing") == t]) for t in ("pre_open", "intraday")},
                     "by_group": {g: daily_accuracy([r for r in resolved if r[0].get("group") == g]) for g in GROUPS},
                     "by_range_source": {v: daily_accuracy([r for r in resolved if r[0].get("range_source") == v])
                                         for v in ("har+vix", "har+chain_iv30", "har")}},
        "calibration": {"edges": list(CAL_EDGES),
                        "bins": calibration_bins((float(c["p_up"]), int(s["outcome_up"])) for c, s in resolved)},
        "sessions": rows,
        "open": sum(1 for c in calls if c["id"] not in scores),
        "study": DAILY_STUDY,
        "method": DAILY_METHOD,
        "caveats": DAILY_CAVEATS,
    }


# The research numbers the page shows beside the live record (which starts at
# n = 0): the two pre-registered studies' holdout results, 2019-01-02 ->
# 2026-11-02. test_forecast.py pins every number to the committed study jsons.
DAILY_STUDY = {
    "range": {"doc": "research/forecast_daily_study_2026-11-04.md", "holdout": "2019-01-02 → 2026-11-02",
              "spy_coverage": 0.7081, "spy_skill_vs_constant": 0.0793, "spy_skill_vs_current": 0.0604,
              "iv_names_coverage": 0.6838, "iv_names_skill_vs_constant": 0.0445,
              "names_coverage": 0.6579, "names_skill_vs_constant": 0.0386},
    "direction": {"doc": "research/forecast_direction_study_2026-11-04.md", "holdout": "2019-01-02 → 2026-11-02",
                  "tests": 14, "passed": 0, "spy_base_rate": 0.5508, "spy_brier_base_rate": 0.24767,
                  "spy_skill_vs_coinflip": 0.0093},
}

DAILY_METHOD = (
    "Daily call for the next session's close. Direction: P(up) = the point-in-time base rate — the up-frequency of "
    "15 index/sector ETFs' last 756 daily returns — shrunk all the way: none of 13 pre-registered candidate features "
    "(overnight gap, intraday momentum, VIX change, breadth, dealer gamma, first-hour momentum) beat it on a purged "
    "walk-forward holdout (research/forecast_direction_study_2026-11-04.md). Range: the 68% band of fc-range-har-v1 — "
    "HAR-RV daily/weekly/monthly realized vol blended with implied vol (VIX for SPY, chain IV30 for names the options "
    "archive covers, else HAR alone), fit as a 68.27% quantile regression walk-forward "
    "(research/forecast_daily_study_2026-11-04.md). A call made during the session starts from the live price and covers "
    "the variance still to come. Scored after each close: Brier against the base rate and against a coin flip, hit rate, "
    "inside-the-range rate, and the interval score against a constant-width band.")
DAILY_CAVEATS = [
    "Experimental model, shadow only: nothing trades on these calls.",
    "The direction is the base rate: its only edge over a coin flip is the market's drift, and it cannot tell one name "
    "from another — read its Brier skill against both references.",
    "The live band is display only and never scored. Its implied-vol input is the base close's (the repo's VIX and "
    "chain-IV feeds are end-of-day), so intraday it follows price and time, not intraday implied vol.",
    "Names without implied vol get no warning of a scheduled event (earnings) inside the session; their bands are HAR only.",
    f"Accuracy figures under {MIN_N_FOR_RATES} resolved calls are marked thin.",
]


if __name__ == "__main__":  # python3 forecast.py — print the report
    print(json.dumps(report(), indent=1, default=str))
