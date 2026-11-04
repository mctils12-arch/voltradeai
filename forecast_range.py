#!/usr/bin/env python3
"""
forecast_range.py — the DAILY forecast's 68% range model, fc-range-har-v1
(Part C of the human-approved forecast batch, 2026-11-04; SHADOW — nothing
trades on it). Pure functions + frozen coefficients; no I/O, no network.

WHAT IT COMPUTES (stated so it can be audited)
  The half-width q of a symmetric band in LOG-return space for the next
  session's close: spot * exp(+/- q). q is the 68.27% quantile of
  |log(C_next / C_base)| conditional on what is known at the base close:

      q = th_d * v_d + th_w * v_w + th_m * v_m [+ th_iv * v_iv]

      v_d  = |r_t|                         (the base session's log return)
      v_w  = sqrt(mean r^2 over t-4..t)    (HAR weekly component)
      v_m  = sqrt(mean r^2 over t-21..t)   (HAR monthly component)
      v_iv = IV_t / sqrt(252)              (implied vol as a daily vol)

  i.e. Corsi's (2009) HAR-RV with daily/weekly/monthly components, blended
  with implied vol, fit as a linear QUANTILE regression at tau = 0.6827
  (no intercept, th >= 0). Minimizing that pinball loss is the same as
  minimizing the expected interval score of the band (Gneiting & Raftery
  2007), so the fit targets the metric the scorer reports. No intercept
  keeps the model scale-equivariant: one fit serves names of any vol level.
  Blend weight on implied vol = th_iv / sum(th).

  Three frozen coefficient sets (RANGE_COEF), from the LAST walk-forward refit
  of research/forecast_daily_study_2026-11-04.md (data through 2026-11-02;
  the study's json holds the same numbers and test_forecast_range.py pins
  them together):
    index_iv  SPY with VIX (the Cboe VIX close of the base day — the same
              series server/cboeVix.ts archives; fit on SPY/QQQ/IWM/DIA with
              VIX/VXN/RVX/VXD)
    stock_iv  any other name with a chain IV30 (server/chainIv.ts, options
              archive, indicative feed) for the base day (fit on AAPL AMZN
              GOOG IBM GS with Cboe's single-stock vol indices — a 30-day
              VIX-method index, not delta-ATM IV30: a stated transfer risk)
    har       no usable implied vol (fit on the whole 41-name panel)
  The source is stored per call so the scorer splits coverage by it.

  INTRADAY (the live band, display only): during the session the band for
  today's close is re-centred on the latest price S and shrunk to the
  variance still to come: q_live = q * sqrt(f(tau)), f = the share of
  close-to-close variance remaining at ET time tau (REMAINING_VARIANCE,
  measured on 60-minute bars in the same study; 1 before the open, 0 at
  16:00). The IV input is NOT updated intraday: the repo's VIX and chain-IV
  sources are end-of-day series (read the study's "intraday IV" note).

Refitting = a new model version + a rerun of the study, never an in-place edit.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

RANGE_MODEL_VERSION = "fc-range-har-v1"
RANGE_TAU = 0.6827                  # two-sided 68.27% band = the tau-quantile of |log return|
RANGE_ALPHA = 1.0 - RANGE_TAU       # the interval score's alpha
HAR_WEEK = 5
HAR_MONTH = 22
RANGE_TRADING_DAYS = 252            # implied vol -> daily vol
IV_PLAUSIBLE = (0.01, 5.0)          # same plausibility window as forecast.sigma_for
REGRESSORS = ("v_d", "v_w", "v_m", "v_iv")

# Frozen coefficients: research/forecast_daily_study_2026-11-04.json `coef_final`
# — the LAST walk-forward refit (refit index 6986 = 2026-10-13, training labels
# through 2026-10-09). test_forecast_range.py pins these to the json; a refit is
# a new model version plus a rerun of the study, never an edit here.
RANGE_COEF_SOURCE = "research/forecast_daily_study_2026-11-04.json"
RANGE_COEF_KEYS = {"index_iv": "index_h1_V4", "stock_iv": "stock_iv_h1_V4", "har": "har_h1_V3"}
RANGE_COEF_COLUMNS = {"index_iv": ("v_d", "v_w", "v_m", "v_iv"), "stock_iv": ("v_d", "v_w", "v_m", "v_iv"),
                      "har": ("v_d", "v_w", "v_m")}
RANGE_COEF: Dict[str, Dict[str, float]] = {
    # SPY with VIX — fit on SPY/QQQ/IWM/DIA with VIX/VXN/RVX/VXD (implied-vol weight 0.71)
    "index_iv": {"v_d": 0.0, "v_w": 0.219303, "v_m": 0.028169, "v_iv": 0.604132},
    # any name with a chain IV30 — fit on AAPL/AMZN/GOOG/IBM/GS with Cboe's stock vol indices (IV weight 0.78)
    "stock_iv": {"v_d": 0.02079, "v_w": 0.097044, "v_m": 0.061147, "v_iv": 0.647347},
    # no implied vol — fit on the 41-name panel
    "har": {"v_d": 0.022428, "v_w": 0.26203, "v_m": 0.634399},
}

# The only name the index_iv set is applied to (its IV input is VIX).
INDEX_IV_SYMBOL = "SPY"
# Names that use the index remaining-variance profile intraday.
INTRADAY_INDEX_SYMBOLS = ("SPY", "QQQ", "IWM", "DIA")
# ET minute-of-day -> f, the share of the session's band (squared) still to
# come: q_live = q * sqrt(f). The study json's `intraday.groups.*.fq_full` — the
# QUANTILE-ratio estimate (a declared post-hoc variant: the pre-registered
# variance ratio was dominated by a few event hours and over-covered intraday;
# research/forecast_daily_study_2026-11-04.md § Intraday). Pinned by the tests.
REMAINING_VARIANCE: Dict[str, Tuple[Tuple[int, float], ...]] = {
    "index": ((570, 0.5988), (630, 0.3503), (690, 0.2552), (750, 0.174), (810, 0.1232), (870, 0.0752), (930, 0.0345)),
    "other": ((570, 0.6155), (630, 0.3058), (690, 0.2031), (750, 0.148), (810, 0.1005), (870, 0.0634), (930, 0.0299)),
}
SESSION_OPEN_MIN = 9 * 60 + 30
SESSION_CLOSE_MIN = 16 * 60

Number = Union[float, int]


# ── regressors (pure; use closes <= i only) ─────────────────────────────────
def har_regressors(closes: Sequence[float], i: Optional[int] = None) -> Optional[Dict[str, float]]:
    """v_d, v_w, v_m at index i of `closes` (default: the last), from the 22
    log returns ending at i — closes[i-22..i]. None when history is short or
    a close is not a positive finite number."""
    if i is None:
        i = len(closes) - 1
    if i < HAR_MONTH or i >= len(closes):
        return None
    window = closes[i - HAR_MONTH:i + 1]
    try:
        c = [float(x) for x in window]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(x) or x <= 0 for x in c):
        return None
    r = [math.log(c[k] / c[k - 1]) for k in range(1, len(c))]
    r2 = [x * x for x in r]
    return {"v_d": abs(r[-1]),
            "v_w": math.sqrt(sum(r2[-HAR_WEEK:]) / HAR_WEEK),
            "v_m": math.sqrt(sum(r2) / HAR_MONTH)}


def iv_regressor(iv: Optional[Number]) -> Optional[float]:
    """Annualized implied vol (decimal, e.g. 0.18) -> a daily vol; None when
    missing or outside the plausibility window."""
    if not isinstance(iv, (int, float)) or isinstance(iv, bool) or not math.isfinite(iv):
        return None
    if not (IV_PLAUSIBLE[0] < float(iv) < IV_PLAUSIBLE[1]):
        return None
    return float(iv) / math.sqrt(RANGE_TRADING_DAYS)


def band_halfwidth(coef: Dict[str, float], regs: Dict[str, float]) -> Optional[float]:
    """q = sum(theta_k * v_k) over the coefficient set's regressors; None if a
    needed regressor is missing or q is not a positive finite number."""
    q = 0.0
    for k, th in coef.items():
        v = regs.get(k)
        if v is None or not math.isfinite(v):
            return None
        q += float(th) * float(v)
    return q if q > 0 and math.isfinite(q) else None


def coef_set_for(symbol: str, iv_source: Optional[str]) -> str:
    """Which frozen set a name uses: SPY with VIX -> index_iv; any name with a
    chain IV30 -> stock_iv; otherwise har."""
    if iv_source == "vix" and symbol == INDEX_IV_SYMBOL:
        return "index_iv"
    if iv_source == "chain_iv30":
        return "stock_iv"
    return "har"


def daily_range(closes: Sequence[float], symbol: str, iv: Optional[Number] = None,
                iv_source: Optional[str] = None, coef: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[dict]:
    """The next-session band half-width from the closes up to the base close.
    `iv` is the implied vol of the base day (decimal) and `iv_source` says
    which series it is ("vix" for SPY, "chain_iv30" for names). Falls back
    to the har set — and says so — when the IV is missing or implausible."""
    coef = RANGE_COEF if coef is None else coef
    regs = har_regressors(closes)
    if regs is None:
        return None
    v_iv = iv_regressor(iv) if iv_source in ("vix", "chain_iv30") else None
    src = iv_source if v_iv is not None else None
    if v_iv is not None:
        regs = {**regs, "v_iv": v_iv}
    name = coef_set_for(symbol, src)
    if name not in coef:
        return None
    q = band_halfwidth(coef[name], regs)
    if q is None:
        return None
    th = coef[name]
    tot = sum(th.values())
    return {"q": q, "coef_set": name, "range_source": {"index_iv": "har+vix", "stock_iv": "har+chain_iv30",
                                                       "har": "har"}[name],
            "regressors": {k: round(v, 6) for k, v in regs.items()},
            "iv_weight": round(th.get("v_iv", 0.0) / tot, 4) if tot > 0 else 0.0,
            "model_version": RANGE_MODEL_VERSION}


# ── intraday: the variance still to come ────────────────────────────────────
def remaining_variance_fraction(et_minutes: Number, symbol: str,
                                table: Optional[Dict[str, Tuple[Tuple[int, float], ...]]] = None) -> Optional[float]:
    """Share of the close-to-close variance still to come at ET minute-of-day
    `et_minutes` on a regular session: 1 before 09:30, linear between the
    measured marks, 0 from 16:00. None when no table is loaded."""
    table = REMAINING_VARIANCE if table is None else table
    prof = table.get("index" if symbol in INTRADAY_INDEX_SYMBOLS else "other")
    if not prof:
        return None
    m = float(et_minutes)
    if m < SESSION_OPEN_MIN:
        return 1.0
    if m >= SESSION_CLOSE_MIN:
        return 0.0
    pts = list(prof) + [(SESSION_CLOSE_MIN, 0.0)]
    for (m0, f0), (m1, f1) in zip(pts, pts[1:]):
        if m0 <= m < m1:
            return max(0.0, min(1.0, f0 + (f1 - f0) * (m - m0) / (m1 - m0)))
    return 0.0


# ── measurement (keep plain) ────────────────────────────────────────────────
def interval_score(q, y, alpha: float = RANGE_ALPHA):
    """Interval score of the band [-q, +q] for the log return y (Gneiting &
    Raftery 2007): 2q + (2/alpha) * max(|y| - q, 0). Lower is better; works
    on floats or numpy arrays; same units as q and y."""
    if isinstance(q, np.ndarray) or isinstance(y, np.ndarray):
        qa, ya = np.asarray(q, float), np.abs(np.asarray(y, float))
        return 2.0 * qa + (2.0 / alpha) * np.maximum(ya - qa, 0.0)
    return 2.0 * float(q) + (2.0 / alpha) * max(abs(float(y)) - float(q), 0.0)


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    u = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.where(u >= 0, tau * u, (tau - 1.0) * u)))


# ── fitting (used by the study; production uses the frozen sets) ───────────
def _solve(H: np.ndarray, g: np.ndarray) -> Optional[np.ndarray]:
    """H^-1 g, or None for a singular sub-system (that active set is skipped)."""
    try:
        return np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        return None


def _nnqp(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    """argmin th'H th - 2 g'th subject to th >= 0, for a handful of
    parameters, by enumerating active sets (exact)."""
    p = len(g)
    best, best_obj = np.zeros(p), 0.0
    for k in range(1, p + 1):
        for S in itertools.combinations(range(p), k):
            idx = list(S)
            t = _solve(H[np.ix_(idx, idx)], g[idx])
            if t is None or np.any(t < 0) or not np.all(np.isfinite(t)):
                continue
            th = np.zeros(p)
            th[idx] = t
            obj = float(th @ H @ th - 2.0 * g @ th)
            if obj < best_obj - 1e-18:
                best, best_obj = th, obj
    return best


def fit_quantile_nonneg(X: Iterable, y: Iterable, tau: float = RANGE_TAU, theta0: Optional[Sequence[float]] = None,
                        iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Linear quantile regression y ~ X theta at level tau, no intercept,
    theta >= 0. Majorize-minimize (Hunter & Lange 2000): each step is a
    weighted least-squares problem with weights 1/(eps + |residual|), solved
    under the sign constraint exactly (_nnqp); eps = 1e-6 * median|y|.
    Agrees with an exact LP to ~1e-9 in the pinball loss on 20k rows."""
    Xa = np.asarray(X, float)
    ya = np.asarray(y, float)
    if Xa.ndim != 2 or len(Xa) != len(ya) or len(ya) == 0:
        raise ValueError("X must be n x p and y length n (n > 0)")
    med = float(np.median(np.abs(ya)))
    eps = 1e-6 * med if med > 0 else 1e-12
    th = (np.asarray(theta0, float) if theta0 is not None and len(theta0) == Xa.shape[1]
          else _nnqp(Xa.T @ Xa, Xa.T @ ya))
    s1 = (2.0 * tau - 1.0) * Xa.sum(axis=0)
    prev = pinball(ya, Xa @ th, tau)
    for _ in range(iters):
        r = ya - Xa @ th
        w = 1.0 / (eps + np.abs(r))
        H = (Xa * w[:, None]).T @ Xa
        g = Xa.T @ (w * ya) + s1
        th = _nnqp(H, g)
        cur = pinball(ya, Xa @ th, tau)
        if abs(prev - cur) <= tol * max(1e-12, abs(prev)):
            break
        prev = cur
    return th
