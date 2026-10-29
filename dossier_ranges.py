#!/usr/bin/env python3
"""
dossier_ranges.py — the Ticker Dossier's measurement half ([PRODUCT]
2026-10-29, human-asked: "like weather — accurate because it looks at a vast
area, then the region, then the spot").

What is forecastable with weather-like reliability is the RANGE (volatility),
not the direction. This module owns everything the dossier claims with a
number and must therefore be able to score:

  1. VOL MODELS (pure). Daily log returns from split-adjusted closes.
       * EWMA (RiskMetrics, lambda 0.94) — the always-available baseline.
       * HAR-RV on daily squared returns (Corsi 2009, daily/weekly/monthly
         components), OLS-fit on the name's own last <= HAR_FIT_WINDOW
         sessions, one fit per horizon (1 and 5 sessions; target = the mean
         squared return over the next h sessions). Daily squared returns are
         a noisy RV proxy (no intraday data), so the HAR forecast is used only
         when it is finite, positive and within [1/4, 4] x the EWMA forecast;
         otherwise EWMA is used and the reason is reported.
  2. BANDS (pure). Zero-drift lognormal: spot * exp(+/- z * sigma_h) with
     z = 1.0 (68.27%) and z = 1.6449 (90%). Daily returns are fat-tailed, so a
     Gaussian 68% band usually CONTAINS MORE than 68% and a 90% band slightly
     less — the walk-forward backtest and the live ledger measure exactly that.
  3. WALK-FORWARD COVERAGE (pure, no lookahead). HAR coefficients are frozen
     from data that ends BEFORE the evaluation window (the last training
     target uses returns up to the window start); every evaluation day then
     forecasts from closes on or before that day only.
  4. TRADE-PLAN MATH (pure). Stop = the 90% 5-session LOWER band from the
     entry reference price; target = the 68% 5-session UPPER band (and the
     90% upper as a stretch target). P(target touched before stop within the
     horizon) = Monte Carlo of DRIFTLESS log-price paths on the fitted daily
     vol (PLAN_MC_PATHS paths, PLAN_STEPS_PER_SESSION steps per session, fixed
     seed). No drift is assumed — the dossier has no validated direction
     signal to supply one — so P(target first) x reward ~= P(stop first) x
     risk before costs, and the UI says so.
  5. CALIBRATION LEDGER (append-only JSONL, storage_config paths). Every
     served range (ticker x base date x horizon) and every served plan
     (ticker x serve date) is recorded once (first served wins). The nightly
     scorer marks:
       * range: the close h sessions after the base close inside the 68% /
         90% band (ratio of fetched bars applied to the stored spot, so a split
         between serving and scoring cannot fake a hit);
       * plan: over the h FULL sessions strictly after the serve date (the
         rest of the serve day is ignored — daily bars cannot split it), the
         first session whose high >= target or low <= stop decides
         target_first / stop_first; both in the same session = ambiguous
         (daily bars cannot order them); none within h sessions = neither.
     Measurement code: keep it plain, never tune it together with the models.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from storage_config import DOSSIER_LEDGER_PATH, DOSSIER_SCORES_PATH
except ImportError:  # pragma: no cover — storage_config is always present in the repo
    _d = "/data/voltrade" if os.path.isdir("/data") else "/tmp"
    DOSSIER_LEDGER_PATH = os.path.join(_d, "voltrade_dossier_ledger.jsonl")
    DOSSIER_SCORES_PATH = os.path.join(_d, "voltrade_dossier_scores.jsonl")

log = logging.getLogger("voltrade.dossier_ranges")

MODEL_VERSION = "dr-har-ewma-v1"
EWMA_LAMBDA = 0.94
EWMA_SEED_N = 20
HAR_FIT_WINDOW = 750          # sessions of history the HAR fit may use
HAR_MIN_ROWS = 250            # fewer training rows -> EWMA
HAR_SANITY = (0.25, 4.0)      # HAR/EWMA variance ratio outside this -> EWMA
Z68 = 1.0
Z90 = 1.6449
HORIZONS = (1, 5)             # trading sessions
BACKTEST_SESSIONS = 250
PLAN_HORIZON = 5
PLAN_MC_PATHS = 4000
PLAN_STEPS_PER_SESSION = 78   # 5-minute steps in a 6.5h session
PLAN_SEED = 20261029
SCORE_GIVE_UP_DAYS = 30


# ── returns / vol models (pure) ─────────────────────────────────────────────
def log_returns(closes: Sequence[float]) -> np.ndarray:
    c = np.asarray([float(x) for x in closes], dtype=float)
    if len(c) < 2 or np.any(c <= 0):
        return np.zeros(0)
    return np.diff(np.log(c))


def ewma_var(rets: Sequence[float], lam: float = EWMA_LAMBDA) -> Optional[float]:
    """Next-day variance: seeded with the mean square of the first EWMA_SEED_N
    returns, then the RiskMetrics recursion over the rest."""
    r = np.asarray(rets, dtype=float)
    if len(r) < EWMA_SEED_N + 5:
        return None
    var = float(np.mean(r[:EWMA_SEED_N] ** 2))
    for x in r[EWMA_SEED_N:]:
        var = lam * var + (1.0 - lam) * float(x) ** 2
    return var if var > 0 and math.isfinite(var) else None


def _har_features(r2: np.ndarray, t: int) -> List[float]:
    """HAR regressors at return index t (uses r2[..t] only)."""
    return [1.0, float(r2[t]), float(np.mean(r2[t - 4:t + 1])), float(np.mean(r2[t - 21:t + 1]))]


def fit_har(rets: Sequence[float], h: int, end: Optional[int] = None,
            window: int = HAR_FIT_WINDOW) -> Optional[np.ndarray]:
    """OLS HAR coefficients for horizon h using returns r[..end] only (end
    inclusive; default the last). Training rows t satisfy t + h <= end, so no
    target reaches past `end`."""
    r = np.asarray(rets, dtype=float)
    if end is None:
        end = len(r) - 1
    r2 = r ** 2
    lo = max(21, end - window)
    rows = range(lo, end - h + 1)
    if len(rows) < HAR_MIN_ROWS:
        return None
    X = np.array([_har_features(r2, t) for t in rows])
    y = np.array([float(np.mean(r2[t + 1:t + h + 1])) for t in rows])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError as e:
        log.debug("dossier: HAR fit failed (%s)", e)
        return None
    return coef if np.all(np.isfinite(coef)) else None


def har_var(coef: Optional[np.ndarray], rets: Sequence[float], t: Optional[int] = None) -> Optional[float]:
    """Per-session variance forecast for the next h sessions from return index t."""
    if coef is None:
        return None
    r = np.asarray(rets, dtype=float)
    t = len(r) - 1 if t is None else t
    if t < 21:
        return None
    v = float(np.dot(coef, _har_features(r ** 2, t)))
    return v if v > 0 and math.isfinite(v) else None


def choose_var(har_v: Optional[float], ewma_v: Optional[float]) -> Tuple[Optional[float], str, Optional[str]]:
    """(per-session variance, model used, why HAR was not used)."""
    if ewma_v is None:
        return (har_v, "har", None) if har_v else (None, "none", "not enough history")
    if har_v is None:
        return ewma_v, "ewma", "HAR fit unavailable (short history or invalid fit)"
    ratio = har_v / ewma_v
    if not (HAR_SANITY[0] <= ratio <= HAR_SANITY[1]):
        return ewma_v, "ewma", f"HAR/EWMA variance ratio {ratio:.2f} outside sanity bounds"
    return har_v, "har", None


def sigma_forecast(closes: Sequence[float]) -> dict:
    """Per-horizon sigma of the log return (horizon total, not per day)."""
    r = log_returns(closes)
    e = ewma_var(r)
    out: Dict[str, Any] = {"model_version": MODEL_VERSION, "n_returns": int(len(r)),
                           "ewma_sigma_1d": math.sqrt(e) if e else None, "horizons": {}}
    for h in HORIZONS:
        coef = fit_har(r, h)
        hv = har_var(coef, r)
        v, model, why = choose_var(hv, e)
        out["horizons"][str(h)] = {
            "h": h, "model": model, "fallback_reason": why,
            "sigma": math.sqrt(v * h) if v else None,
            "sigma_daily": math.sqrt(v) if v else None,
            "har_sigma_daily": math.sqrt(hv) if hv else None,
        }
    return out


def band(spot: float, sigma_h: float, z: float) -> Tuple[float, float]:
    return spot * math.exp(-z * sigma_h), spot * math.exp(z * sigma_h)


def bands_for(spot: float, sigma_h: Optional[float]) -> Optional[dict]:
    if not sigma_h or spot <= 0:
        return None
    lo68, hi68 = band(spot, sigma_h, Z68)
    lo90, hi90 = band(spot, sigma_h, Z90)
    return {"lo68": round(lo68, 4), "hi68": round(hi68, 4), "lo90": round(lo90, 4), "hi90": round(hi90, 4),
            "move68_pct": round((math.exp(Z68 * sigma_h) - 1) * 100, 3),
            "move90_pct": round((math.exp(Z90 * sigma_h) - 1) * 100, 3)}


def walk_forward_coverage(closes: Sequence[float], h: int, n_eval: int = BACKTEST_SESSIONS) -> Optional[dict]:
    """Out-of-sample coverage of the 68%/90% bands over the last n_eval
    forecast origins. Return index k is the return INTO close k+1. An origin at
    close index c knows returns r[..c-1]; its outcome is log(C[c+h]/C[c])."""
    c = np.asarray([float(x) for x in closes], dtype=float)
    r = log_returns(c)
    if len(r) == 0:
        return None
    last_origin = len(c) - 1 - h
    first_origin = last_origin - n_eval + 1
    if first_origin < EWMA_SEED_N + 30:
        return None
    # frozen HAR fit: every training target ends at return index first_origin-1
    coef = fit_har(r, h, end=first_origin - 1)
    lam = EWMA_LAMBDA
    var = float(np.mean(r[:EWMA_SEED_N] ** 2))
    for x in r[EWMA_SEED_N:first_origin]:
        var = lam * var + (1 - lam) * float(x) ** 2
    n = in68 = in90 = 0
    used = {"har": 0, "ewma": 0}
    for o in range(first_origin, last_origin + 1):
        # var now = EWMA forecast using r[..o-1]
        hv = har_var(coef, r, o - 1)
        v, model, _ = choose_var(hv, var)
        if v:
            s = math.sqrt(v * h)
            realized = math.log(c[o + h] / c[o])
            n += 1
            used[model] = used.get(model, 0) + 1
            in68 += int(abs(realized) <= Z68 * s)
            in90 += int(abs(realized) <= Z90 * s)
        var = lam * var + (1 - lam) * float(r[o]) ** 2
    if not n:
        return None
    return {"h": h, "n": n, "in68": in68, "in90": in90, "rate68": round(in68 / n, 4), "rate90": round(in90 / n, 4),
            "target68": 0.6827, "target90": 0.90, "overlapping": h > 1, "models_used": used}


def expected_move_context(closes: Sequence[float], sigma_1d: Optional[float], lookback: int = 252) -> Optional[dict]:
    """Today's expected 1-session move vs the name's own past year: the median
    absolute daily move and where today's sigma sits among trailing 20-session
    realized vols."""
    r = log_returns(closes)
    if not sigma_1d or len(r) < 60:
        return None
    tail = r[-lookback:]
    med_abs = float(np.median(np.abs(tail)))
    rv = [float(np.std(r[i - 20:i], ddof=1)) for i in range(max(20, len(r) - lookback), len(r) + 1)]
    pct = sum(1 for x in rv if x <= sigma_1d) / len(rv) if rv else None
    return {"sigma_1d_pct": round(sigma_1d * 100, 3), "median_abs_move_pct": round(med_abs * 100, 3),
            "ratio_vs_median": round(sigma_1d / med_abs, 3) if med_abs > 0 else None,
            "sigma_pctile_1y": round(pct, 3) if pct is not None else None, "lookback_sessions": int(len(tail))}


# ── trade-plan math (pure) ──────────────────────────────────────────────────
def touch_probs(sigma_daily: float, sessions: int, up_log: float, down_log: float,
                paths: int = PLAN_MC_PATHS, steps_per_session: int = PLAN_STEPS_PER_SESSION,
                seed: int = PLAN_SEED, drift_daily: float = 0.0) -> dict:
    """P(log price reaches +up_log before -down_log) within `sessions`, driftless
    Gaussian increments at steps_per_session steps per session (Monte Carlo,
    fixed seed). up_log, down_log > 0."""
    rng = np.random.default_rng(seed)
    steps = max(1, int(sessions * steps_per_session))
    dt_sd = sigma_daily / math.sqrt(steps_per_session)
    mu = drift_daily / steps_per_session
    x = np.zeros(paths)
    state = np.zeros(paths, dtype=np.int8)  # 0 alive, 1 target, -1 stop
    for _ in range(steps):
        alive = state == 0
        if not alive.any():
            break
        x[alive] += mu + dt_sd * rng.standard_normal(int(alive.sum()))
        state[alive & (x >= up_log)] = 1
        state[alive & (x <= -down_log)] = -1
    t = float(np.mean(state == 1))
    s = float(np.mean(state == -1))
    return {"p_target_first": round(t, 4), "p_stop_first": round(s, 4), "p_neither": round(1 - t - s, 4),
            "paths": paths, "steps_per_session": steps_per_session, "seed": seed, "drift_daily": drift_daily}


def trade_plan(entry: float, sigma_daily_5: Optional[float], sessions: int = PLAN_HORIZON) -> Optional[dict]:
    """Levels and touch probabilities from the 5-session vol forecast.
    sigma_daily_5 = the per-session sigma of the 5-session forecast."""
    if not sigma_daily_5 or entry <= 0:
        return None
    s_h = sigma_daily_5 * math.sqrt(sessions)
    stop = entry * math.exp(-Z90 * s_h)
    t1 = entry * math.exp(Z68 * s_h)
    t2 = entry * math.exp(Z90 * s_h)
    risk = entry - stop
    p1 = touch_probs(sigma_daily_5, sessions, math.log(t1 / entry), math.log(entry / stop))
    p2 = touch_probs(sigma_daily_5, sessions, math.log(t2 / entry), math.log(entry / stop))
    return {
        "entry_ref": round(entry, 4), "horizon_sessions": sessions, "sigma_h": round(s_h, 6),
        "stop": round(stop, 4), "target": round(t1, 4), "stretch_target": round(t2, 4),
        "risk_per_share": round(risk, 4), "reward_per_share": round(t1 - entry, 4),
        "reward_risk": round((t1 - entry) / risk, 3) if risk > 0 else None,
        "stretch_reward_risk": round((t2 - entry) / risk, 3) if risk > 0 else None,
        "stop_pct": round((stop / entry - 1) * 100, 3), "target_pct": round((t1 / entry - 1) * 100, 3),
        "stretch_pct": round((t2 / entry - 1) * 100, 3),
        "touch": p1, "touch_stretch": p2,
        "rules": {
            "stop": "the 90% band's lower edge for the next 5 sessions, from the entry reference price — outside ~95% of driftless 5-session closes",
            "target": "the 68% band's upper edge for the next 5 sessions (stretch: the 90% band's upper edge)",
            "probabilities": f"Monte Carlo of {PLAN_MC_PATHS:,} driftless paths at 5-minute steps on the fitted daily vol; no drift assumed (no validated direction signal)",
        },
    }


# ── ledger I/O ──────────────────────────────────────────────────────────────
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
                    log.warning("dossier ledger: skipped a corrupt line in %s", path)
                    continue
                if isinstance(v, dict):
                    out.append(v)
    except FileNotFoundError:
        return []
    return out


def _append(path: str, rows: List[dict]) -> None:
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


def range_rows(ticker: str, base_date: str, spot: float, sig: dict, served_at: str) -> List[dict]:
    rows = []
    for h in HORIZONS:
        hz = (sig.get("horizons") or {}).get(str(h)) or {}
        b = bands_for(spot, hz.get("sigma"))
        if not b:
            continue
        rows.append({"kind": "range", "id": f"range|{ticker}|{base_date}|{h}", "ticker": ticker, "base_date": base_date,
                     "h": h, "spot": round(spot, 4), "sigma": round(float(hz["sigma"]), 6), "model": hz.get("model"),
                     "model_version": MODEL_VERSION, "served_at": served_at, **{k: b[k] for k in ("lo68", "hi68", "lo90", "hi90")}})
    return rows


def plan_row(ticker: str, serve_date: str, base_date: str, base_close: float, plan: dict, served_at: str) -> dict:
    return {"kind": "plan", "id": f"plan|{ticker}|{serve_date}", "ticker": ticker, "serve_date": serve_date,
            "base_date": base_date, "base_close": round(base_close, 4), "entry_ref": plan["entry_ref"],
            "stop": plan["stop"], "target": plan["target"], "h": plan["horizon_sessions"],
            "p_target_first": plan["touch"]["p_target_first"], "p_stop_first": plan["touch"]["p_stop_first"],
            "model_version": MODEL_VERSION, "served_at": served_at}


def record(rows: List[dict], path: Optional[str] = None) -> int:
    """Append rows whose id is not in the ledger yet (first served wins). Never raises."""
    path = path or DOSSIER_LEDGER_PATH
    try:
        have = {r.get("id") for r in _read_jsonl(path)}
        new = [r for r in rows if r.get("id") not in have]
        _append(path, new)
        return len(new)
    except OSError as e:
        log.warning("dossier ledger: record failed (%s)", e)
        return 0


# ── scoring (measurement code — keep it plain) ──────────────────────────────
def _bar_day(b: dict) -> str:
    return str(b.get("t", ""))[:10]


def score_range(row: dict, bars: List[dict]) -> Optional[dict]:
    """Completed ascending bars -> score, or None while the h-th session after
    the base date has not closed."""
    days = [_bar_day(b) for b in bars]
    try:
        i = days.index(row["base_date"])
    except ValueError:
        return None
    h = int(row["h"])
    if i + h >= len(bars):
        return None
    c0, ch = float(bars[i]["c"]), float(bars[i + h]["c"])
    realized = float(row["spot"]) * (ch / c0)
    return {"kind": "score", "id": row["id"], "status": "resolved", "resolved_date": days[i + h],
            "realized_close": round(realized, 4), "in68": int(row["lo68"] <= realized <= row["hi68"]),
            "in90": int(row["lo90"] <= realized <= row["hi90"])}


def score_plan(row: dict, bars: List[dict]) -> Optional[dict]:
    """The h full sessions strictly after serve_date decide; None until they close.
    Levels are rescaled by the fetched/stored base close ratio (split-safe)."""
    days = [_bar_day(b) for b in bars]
    if row["base_date"] not in days:
        return None
    scale = float(bars[days.index(row["base_date"])]["c"]) / float(row["base_close"])
    after = [b for b in bars if _bar_day(b) > row["serve_date"]]
    h = int(row["h"])
    if len(after) < h:
        return None
    tgt, stp = float(row["target"]) * scale, float(row["stop"]) * scale
    outcome, when = "neither", _bar_day(after[h - 1])
    for b in after[:h]:
        hi = float(b.get("h", b["c"]))
        lo = float(b.get("l", b["c"]))
        up, dn = hi >= tgt, lo <= stp
        if up and dn:
            outcome, when = "ambiguous", _bar_day(b)
            break
        if up:
            outcome, when = "target_first", _bar_day(b)
            break
        if dn:
            outcome, when = "stop_first", _bar_day(b)
            break
    return {"kind": "score", "id": row["id"], "status": "resolved", "outcome": outcome, "resolved_date": when}


FetchFn = Callable[[List[str], str, str, Optional[float]], Tuple[Dict[str, List[dict]], set]]


def _default_fetch(symbols: List[str], start: str, end: str, deadline: Optional[float]):
    from shadow_portfolio import _fetch_bars_paginated
    unfetched: set = set()
    bars = _fetch_bars_paginated(symbols, start, end, deadline=deadline, unfetched=unfetched)
    return bars, unfetched


def _completed(bars: List[dict], now: datetime) -> List[dict]:
    from forecast import completed_bars
    return completed_bars(bars, now)


def score_ledger(now: Optional[datetime] = None, fetch: Optional[FetchFn] = None, time_budget_s: float = 60.0,
                 path: Optional[str] = None, scores_path: Optional[str] = None,
                 completed: Optional[Callable[[List[dict], datetime], List[dict]]] = None) -> dict:
    """Score every ledger row whose horizon has passed. Append-only; a fetch
    failure is retried next run; a row with no price data SCORE_GIVE_UP_DAYS
    after serving gets a terminal no_price_data line."""
    now = now or datetime.now(timezone.utc)
    fetch = fetch or _default_fetch
    completed = completed or _completed
    deadline = time.monotonic() + max(5.0, float(time_budget_s))
    rows = _read_jsonl(path or DOSSIER_LEDGER_PATH)
    done = {s.get("id") for s in _read_jsonl(scores_path or DOSSIER_SCORES_PATH)}
    seen: set = set()
    todo = []
    for r in rows:
        if r.get("id") and r["id"] not in done and r["id"] not in seen and r.get("kind") in ("range", "plan"):
            seen.add(r["id"])
            todo.append(r)
    stats = {"open": len(todo), "resolved": 0, "pending": 0, "gave_up": 0, "fetch_errors": 0}
    if not todo:
        return stats
    today = now.date()
    start = (min(date.fromisoformat(r["base_date"]) for r in todo) - timedelta(days=7)).isoformat()
    raw, unfetched = fetch(sorted({r["ticker"] for r in todo}), start, today.isoformat(), deadline)
    out = []
    for r in todo:
        if r["ticker"] in unfetched:
            stats["fetch_errors"] += 1
            continue
        bars = completed(raw.get(r["ticker"]) or [], now)
        s = score_range(r, bars) if r["kind"] == "range" else score_plan(r, bars)
        if s is None:
            age = (today - date.fromisoformat(r["base_date"])).days
            if age > SCORE_GIVE_UP_DAYS + 10:
                out.append({"kind": "score", "id": r["id"], "status": "no_price_data"})
                stats["gave_up"] += 1
            else:
                stats["pending"] += 1
            continue
        s["scored_at"] = now.astimezone(timezone.utc).isoformat()
        out.append(s)
        stats["resolved"] += 1
    _append(scores_path or DOSSIER_SCORES_PATH, out)
    return stats


def _rate(k: int, n: int) -> Optional[float]:
    return round(k / n, 4) if n else None


def ledger_report(ticker: Optional[str] = None, path: Optional[str] = None, scores_path: Optional[str] = None) -> dict:
    """Running record: range hit rates per horizon and plan outcomes — for all
    names and for `ticker`."""
    rows: Dict[str, dict] = {}
    for r in _read_jsonl(path or DOSSIER_LEDGER_PATH):
        if r.get("id"):
            rows.setdefault(r["id"], r)
    scores: Dict[str, dict] = {}
    for s in _read_jsonl(scores_path or DOSSIER_SCORES_PATH):
        if s.get("id") and s.get("status") == "resolved":
            scores.setdefault(s["id"], s)

    def summarize(filter_ticker: Optional[str]) -> dict:
        rng: Dict[str, dict] = {}
        for h in HORIZONS:
            sel = [(r, scores[r["id"]]) for r in rows.values() if r.get("kind") == "range" and int(r.get("h", 0)) == h
                   and r["id"] in scores and (filter_ticker is None or r.get("ticker") == filter_ticker)]
            n = len(sel)
            k68 = sum(int(s["in68"]) for _, s in sel)
            k90 = sum(int(s["in90"]) for _, s in sel)
            rng[str(h)] = {"n": n, "in68": k68, "in90": k90, "rate68": _rate(k68, n), "rate90": _rate(k90, n)}
        pl = [s for r, s in ((r, scores.get(r["id"])) for r in rows.values() if r.get("kind") == "plan"
                             and (filter_ticker is None or r.get("ticker") == filter_ticker)) if s]
        counts = {k: sum(1 for s in pl if s.get("outcome") == k) for k in ("target_first", "stop_first", "neither", "ambiguous")}
        n = len(pl)
        return {"range": rng, "plan": {"n": n, **counts, **{f"{k}_rate": _rate(v, n) for k, v in counts.items()}}}

    open_n = sum(1 for r in rows.values() if r["id"] not in scores)
    return {"model_version": MODEL_VERSION, "all": summarize(None),
            "ticker": summarize(ticker) if ticker else None, "open": open_n, "min_n": 30}


if __name__ == "__main__":  # python3 dossier_ranges.py — print the ledger report
    print(json.dumps(ledger_report(), indent=1, default=str))
