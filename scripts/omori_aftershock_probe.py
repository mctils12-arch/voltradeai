#!/usr/bin/env python3
"""
Omori-law aftershock-decay probe: after a large single-day price shock, does
the elevated rate of FURTHER shocks decay as a POWER LAW in time (the
seismology signature) rather than as an ordinary EXPONENTIAL (the shape
already implicit in this codebase's GARCH-adjacent regime/volatility
machinery)?

FOREIGN-FIELD IMPORT (EDGE DOCTRINE #4, CLAUDE.md) from seismology: Omori
(1894) and the modified Omori-Utsu law (Utsu, Ogata & Matsu'ura 1995, "The
Centenary of the Omori Formula for a Decay Law of Aftershock Activity,"
J. Phys. Earth 43(1):1-33) — the standard empirical law that the rate of
aftershocks n(t) following a mainshock decays as
    n(t) = K / (t + c)^p
with p typically near 1, NOT as a simple exponential. Power-law-vs-
exponential is the field's own standard diagnostic for whether a process
is self-exciting/heavy-tailed (long memory of the shock) versus merely
relaxing (short, constant-half-life memory) — genuinely different physics,
not two labels for the same curve.

This is a DIFFERENT statistical technique from the three prior
foreign-field imports on file: 2026-08-18/21 (ecology, critical slowing
down) measured a single index's OWN rolling autocorrelation/variance ahead
of a transition; 2026-08-26 (epidemiology, R_t) measured a cross-sectional
breakdown-count growth ratio across a peer basket; 2026-08-29 (reliability
engineering, hazard rate) measured the coefficient of variation of
INTER-FAILURE GAP TIMING (regular vs. bursty spacing). None of the three
fit a parametric decay curve to the rate of subsequent events following a
reference event and asked "heavy-tailed or short-memory" — that is what
this one does. It is also the first of the four that directly targets an
assumption (exponential-ish volatility decay) already built into this
codebase's own machinery, rather than adding an unrelated new lens.

WHY THIS MATTERS BEYOND "shocks cluster" (already known, already used via
the regime classifier): a power-law fit that clearly beats an exponential
fit AND has a fitted exponent p meaningfully below 1 (heavy tail) implies
the elevated-risk window after a shock persists measurably LONGER than a
standard exponential/GARCH decay predicts — an actionable duration edge
(how long to stay defensive after a shock), not merely "shocks cluster."

HYPOTHESIS (pre-registered, see the matching research/experiments.md and
research/open_questions.md entries): following an SPY daily log-return
shock (|return| >= k trailing-sigma, sigma computed from STRICTLY PRIOR
trading days only — never including the shock day itself, so a shock's own
magnitude cannot inflate the sigma used to classify it), the empirical rate
of a further shock at forward lag t (averaged across every mainshock, a
standard "superposed epoch analysis") fits a power-law decay BETTER
(higher log-log R^2) than an exponential decay, with fitted exponent p < 1.

PRIOR (stated before ever running this against real data): my prior is
that the exponential fit does about AS WELL AS, or better than, the power
law — i.e. I expect a CLEAN NEGATIVE for "seismology adds something GARCH
doesn't," the same disposition the CSD and R_t entries landed on.
REASONING STANDARD #5 (second-order): if a heavy-tailed power-law
aftershock structure in equity-index shocks were both real and this easy
to detect with a superposed-epoch analysis, GARCH variants already built to
fit fat-tailed/long-memory volatility (FIGARCH, HAR-RV) would already have
absorbed it — this is about as well-trodden a corner of quant finance as
exists, so the base rate for "found something new" here is low. Finding
p < 1 WITH a power-law R^2 that clearly beats the exponential fit's R^2
would be the genuinely interesting result; the exponential fit doing as
well or better is the expected, unglamorous, and equally valid-to-log
outcome (REASONING STANDARD #10 — state the prior, then update, don't only
write conclusions after seeing data).

SIMPLIFICATION STATED EXPLICITLY (REASONING STANDARD #4): "aftershock" here
reuses the SAME shock definition as "mainshock" (any day clearing the same
k-sigma bar counts as both a candidate mainshock and a candidate
aftershock), not a lower/graded threshold the way real seismic aftershock
catalogs sometimes use. This is the simplest faithful reading of the law,
stated as a limitation rather than the whole design space.

LADDER PATH: gate-2 SIGNAL test (statistical predictive power only, no
trading), built directly on `backtest_v2.fetch_bars` (already gate-1
verified elsewhere in this codebase) and `critical_slowing_down_probe.
log_returns` reused verbatim (EDGE DOCTRINE #3: don't re-derive already-
verified data plumbing). Note `critical_slowing_down_probe.rolling_
variance` is NOT reused for the sigma estimate: that helper's window is
INCLUSIVE of the current day, which is fine for CSD's own approach-to-
transition use but wrong here — this probe's sigma must be strictly prior
to the day it classifies, so `trailing_sigma` below is a new, narrower
function, not a re-derivation of existing plumbing.

WHY THIS SESSION DID NOT RUN IT AGAINST REAL DATA: this sandbox has
neither ALPACA_KEY/ALPACA_SECRET in its environment nor working network
access to Yahoo Finance (query2/guce/fc.finance.yahoo.com connections were
reset by the egress proxy this session; no pre-existing .bt_cache either)
— the identical constraint the three prior foreign-field-import entries
hit. The statistical core below is pure and unit-tested against synthetic
data in test_omori_aftershock_probe.py, so a future session with real data
access can call run_probe() directly.

USAGE (future session, once ALPACA_KEY/SECRET or working Yahoo access
exists):
    python3 scripts/omori_aftershock_probe.py [--days 2520] [--k 2.5]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Sequence

MIN_MAINSHOCKS_FOR_STATS = 5  # same n>=5 reporting floor used elsewhere in this repo
DEFAULT_SIGMA_WINDOW = 60
DEFAULT_SHOCK_K = 2.5
DEFAULT_MAX_LAG = 20
_C_GRID = (0.0, 0.5, 1.0, 2.0, 5.0)


def trailing_sigma(returns: Sequence[float], window: int) -> list[float | None]:
    """Population std of `returns[t-window:t]` — STRICTLY PRIOR to t, never
    including returns[t] itself. None until `window` strictly-prior
    observations exist (t < window)."""
    n = len(returns)
    out: list[float | None] = [None] * n
    for t in range(n):
        if t < window:
            continue
        w = returns[t - window:t]
        m = sum(w) / window
        var = sum((v - m) ** 2 for v in w) / window
        out[t] = math.sqrt(var)
    return out


def shock_flags(returns: Sequence[float], sigmas: Sequence[float | None],
                 k: float = DEFAULT_SHOCK_K) -> list[bool]:
    """True at t iff sigmas[t] is known, strictly positive, and
    abs(returns[t]) >= k * sigmas[t]. The `sigma > 0` guard matters: a
    degenerate zero-variance window would otherwise make abs(return) >= 0
    trivially true for every non-flat day, misclassifying constant/near-
    constant stretches as perpetual shocks."""
    out = []
    for r, s in zip(returns, sigmas):
        out.append(s is not None and s > 0 and abs(r) >= k * s)
    return out


def aftershock_rate_curve(shocks: Sequence[bool], max_lag: int) -> list[dict]:
    """Superposed epoch analysis: for each forward lag t in 1..max_lag,
    the fraction of mainshocks (shocks[i] True) for which shocks[i+t] is
    ALSO True, averaged over every mainshock with i+t still inside the
    series. One dict per lag: {lag, rate, n_mainshocks, insufficient_n}."""
    mainshock_idx = [i for i, s in enumerate(shocks) if s]
    n = len(shocks)
    out = []
    for lag in range(1, max_lag + 1):
        usable = [i for i in mainshock_idx if i + lag < n]
        entry: dict = {"lag": lag, "n_mainshocks": len(usable)}
        if len(usable) < MIN_MAINSHOCKS_FOR_STATS:
            entry["insufficient_n"] = True
            entry["rate"] = (sum(1 for i in usable if shocks[i + lag]) / len(usable)
                              if usable else None)
        else:
            entry["rate"] = sum(1 for i in usable if shocks[i + lag]) / len(usable)
        out.append(entry)
    return out


def baseline_shock_rate(shocks: Sequence[bool]) -> float | None:
    """Unconditional fraction of days that are shocks — the asymptote the
    aftershock rate curve should decay toward at long lag. None on an
    empty series."""
    if not shocks:
        return None
    return sum(shocks) / len(shocks)


def _ols_loglog(xs: Sequence[float], ys: Sequence[float]) -> dict | None:
    """OLS of log(ys) on log(xs) — the power-law model log(y) = log(K) -
    p*log(x). Returns slope (= -p), intercept (= log K), and R^2. None if
    fewer than 2 strictly-positive (x, y) pairs remain (log undefined at
    or below zero) or x has zero spread."""
    pts = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    return _ols(([math.log(x) for x, _ in pts]), [math.log(y) for _, y in pts])


def _ols_linlog(xs: Sequence[float], ys: Sequence[float]) -> dict | None:
    """OLS of log(ys) on RAW xs — the exponential-decay model
    log(y) = log(K) - lambda*x. Returns slope (= -lambda), intercept
    (= log K), and R^2. None if fewer than 2 strictly-positive ys remain."""
    pts = [(x, y) for x, y in zip(xs, ys) if y > 0]
    if len(pts) < 2:
        return None
    return _ols([x for x, _ in pts], [math.log(y) for _, y in pts])


def _ols(lx: list[float], ly: list[float]) -> dict | None:
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    sxx = sum((v - mx) ** 2 for v in lx)
    if sxx == 0:
        return None
    sxy = sum((lx[i] - mx) * (ly[i] - my) for i in range(n))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((ly[i] - (intercept + slope * lx[i])) ** 2 for i in range(n))
    ss_tot = sum((v - my) ** 2 for v in ly)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return {"slope": slope, "intercept": intercept, "r_squared": r2, "n_points": n}


def _excess_pairs(curve: Sequence[dict], base_rate: float) -> list[tuple[float, float]]:
    return [(pt["lag"], pt["rate"] - base_rate) for pt in curve if pt["rate"] is not None]


def fit_power_law_decay(curve: Sequence[dict], base_rate: float | None,
                         c_grid: Sequence[float] = _C_GRID) -> dict | None:
    """Fits the Omori-Utsu excess-rate decay e(t) = K/(t+c)^p by grid-
    searching `c` and OLS-regressing log(e(t)) on log(t+c) for each
    candidate, keeping the c with the best R^2 (avoids a full 3-parameter
    nonlinear solver while still fitting all of K, c, p). e(t) = rate(t) -
    base_rate must be strictly positive to log-transform; lags where the
    aftershock rate has already fallen to/below baseline are excluded, not
    clipped. None if base_rate is None or fewer than 3 positive-excess
    lags remain — a 2-point "fit" is not a fit."""
    if base_rate is None:
        return None
    pairs = _excess_pairs(curve, base_rate)
    if len([e for _, e in pairs if e > 0]) < 3:
        return None
    lags = [lag for lag, _ in pairs]
    excess = [e for _, e in pairs]
    best = None
    for c in c_grid:
        fit = _ols_loglog([lag + c for lag in lags], excess)
        if fit is None or fit["r_squared"] is None:
            continue
        if best is None or fit["r_squared"] > best["r_squared"]:
            best = {"c": c, "p": -fit["slope"], "K": math.exp(fit["intercept"]),
                    "r_squared": fit["r_squared"], "n_points_fit": fit["n_points"]}
    return best


def fit_exponential_decay(curve: Sequence[dict], base_rate: float | None) -> dict | None:
    """Fits the exponential excess-rate decay e(t) = K*exp(-lambda*t), the
    ordinary-mean-reversion/GARCH-adjacent null this probe's hypothesis
    tests the power-law fit against. Same e(t) > 0 requirement and >=3
    fittable-point floor as fit_power_law_decay, for a fair comparison."""
    if base_rate is None:
        return None
    pairs = _excess_pairs(curve, base_rate)
    if len([e for _, e in pairs if e > 0]) < 3:
        return None
    lags = [lag for lag, _ in pairs]
    excess = [e for _, e in pairs]
    fit = _ols_linlog(lags, excess)
    if fit is None or fit["r_squared"] is None:
        return None
    return {"lam": -fit["slope"], "K": math.exp(fit["intercept"]),
            "r_squared": fit["r_squared"], "n_points_fit": fit["n_points"]}


def power_law_beats_exponential(pl_fit: dict | None, exp_fit: dict | None) -> bool | None:
    """True iff both fits exist and the power-law fit's R^2 exceeds the
    exponential fit's R^2 — the discriminating test this probe's
    hypothesis hinges on. None (unknown), not a silent False, if either
    fit could not be computed."""
    if pl_fit is None or exp_fit is None:
        return None
    return pl_fit["r_squared"] > exp_fit["r_squared"]


def run_probe(days: int = 2520, sigma_window: int = DEFAULT_SIGMA_WINDOW,
              k: float = DEFAULT_SHOCK_K, max_lag: int = DEFAULT_MAX_LAG) -> dict:
    """Orchestrates the full gate-2 probe against REAL data. Requires
    network/Alpaca access this sandbox did not have — see module
    docstring. Reuses backtest_v2.fetch_bars and the CSD entry's
    log_returns verbatim (EDGE DOCTRINE #3)."""
    import backtest_v2 as bt

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "critical_slowing_down_probe",
        os.path.join(os.path.dirname(__file__), "critical_slowing_down_probe.py"))
    csd = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(csd)

    spy = bt.fetch_bars("SPY", days)
    returns = csd.log_returns(spy["close"])
    sigmas = trailing_sigma(returns, sigma_window)
    shocks = shock_flags(returns, sigmas, k)
    n_mainshocks = sum(shocks)

    curve = aftershock_rate_curve(shocks, max_lag)
    base_rate = baseline_shock_rate(shocks)
    pl_fit = fit_power_law_decay(curve, base_rate)
    exp_fit = fit_exponential_decay(curve, base_rate)

    return {
        "n_days": len(returns),
        "date_range": [spy["date"][0], spy["date"][-1]] if spy.get("date") else [],
        "sigma_window": sigma_window,
        "shock_k": k,
        "n_mainshocks": n_mainshocks,
        "n_mainshocks_insufficient": n_mainshocks < MIN_MAINSHOCKS_FOR_STATS,
        "base_rate": base_rate,
        "aftershock_rate_curve": curve,
        "power_law_fit": pl_fit,
        "exponential_fit": exp_fit,
        "power_law_beats_exponential": power_law_beats_exponential(pl_fit, exp_fit),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=2520,
                     help="calendar days of SPY history to pull (default ~10y)")
    ap.add_argument("--sigma-window", type=int, default=DEFAULT_SIGMA_WINDOW,
                     help="trailing trading days used to estimate the shock-day sigma")
    ap.add_argument("--k", type=float, default=DEFAULT_SHOCK_K,
                     help="trailing-sigma multiple defining a 'shock' day")
    ap.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG,
                     help="forward trading-day window over which the aftershock rate curve is measured")
    args = ap.parse_args()
    try:
        out = run_probe(days=args.days, sigma_window=args.sigma_window,
                         k=args.k, max_lag=args.max_lag)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(out, indent=2, default=str))
