#!/usr/bin/env python3
"""
FOREIGN-FIELD IMPORT (EDGE DOCTRINE #4, axis c): Kuramoto phase
synchronization (nonlinear physics / neuroscience — coupled oscillators
lock phase before a population "fires" in unison) applied to the nine
SPDR sector ETFs. Hypothesis: when sector oscillation phases lock
(order parameter R -> 1) the market has lost its internal diversification
and a common shock hits all sectors at once, so forward SPY returns are
lower / forward SPY volatility higher than R's trailing-volatility
correlate alone explains.

WHY THIS IS NOT A COLD SEVENTH-IN-A-ROW REPEAT (REASONING STANDARD #4):
the six prior same-family imports (CSD, R_t, hazard rate, Omori, permutation
entropy, Hurst) each tested a SINGLE-SERIES statistic. This one is
cross-sectional in construction (a coupling measure across 9 series) and is
built around the one control that matters: R is mechanically correlated with
trailing volatility and mean pairwise correlation (the known "absorption
ratio" / correlation-surprise literature), so the ONLY claim worth testing is
incremental information AFTER partialling those out. Raw correlation with
forward vol is expected and is NOT evidence.

PRIOR, stated BEFORE any real-data run (REASONING STANDARD #10): raw
Spearman(R, fwd-20d SPY vol) positive (~+0.2) but |partial rho| < 0.15 after
controlling trailing 20d realized vol + 60d mean pairwise correlation;
Spearman(R, fwd-20d SPY return) |rho| < 0.15 (efficient-market null). Most
likely outcome: clean negative, gate 2 fails. Single pre-registered spec, run
once; no parameter re-tuning after seeing results.

PHASE CONSTRUCTION (causal, no lookahead — a Hilbert transform over a window
has end effects and would leak): per ETF, x_t = EMA10(log p) - EMA30(log p)
(trend oscillator); y_t = x_t - x_{t-5} (discrete derivative); for x =
A cos(theta), dx/dt ~ -A sin(theta), so theta_t = atan2(-y_t, x_t). Order
parameter R_t = |mean_j exp(i theta_{j,t})|, smoothed by a 10-day mean (S_t).

LADDER PATH: GATE 2 (SIGNAL) only. Bar = this repo's established
|rho| >= 0.30 AND p < 0.05 on the de-strided (every 20th day) sample of the
PARTIAL correlation with forward return. Does not touch signal_ladder.json
or any trading code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
TARGET = "SPY"
DEFAULT_DAYS = 2520
EMA_FAST, EMA_SLOW = 10, 30
DERIV_LAG = 5
SMOOTH = 10
HORIZON = 20
VOL_WIN = 20
CORR_WIN = 60
GATE2_RHO_FLOOR = 0.30
GATE2_P_CEILING = 0.05
MIN_PAIRS = 30


def ema(xs: Sequence[float], n: int) -> list[float]:
    a = 2.0 / (n + 1)
    out: list[float] = []
    for i, v in enumerate(xs):
        out.append(v if i == 0 else a * v + (1 - a) * out[-1])
    return out


def phases(closes: Sequence[float]) -> list[float | None]:
    """Causal oscillator phase per bar; None until the oscillator has a
    derivative (and the slow EMA has warmed up)."""
    lp = [math.log(c) for c in closes]
    x = [f - s for f, s in zip(ema(lp, EMA_FAST), ema(lp, EMA_SLOW))]
    out: list[float | None] = []
    for t in range(len(x)):
        if t < EMA_SLOW + DERIV_LAG:
            out.append(None)
        else:
            out.append(math.atan2(-(x[t] - x[t - DERIV_LAG]), x[t]))
    return out


def order_parameter(thetas: Sequence[float | None]) -> float | None:
    """Kuramoto R = |mean exp(i theta)| over the oscillators present."""
    ps = [t for t in thetas if t is not None]
    if len(ps) < 2:
        return None
    c = sum(math.cos(t) for t in ps) / len(ps)
    s = sum(math.sin(t) for t in ps) / len(ps)
    return math.hypot(c, s)


def sync_series(phase_by_etf: Sequence[Sequence[float | None]]) -> list[float | None]:
    """R_t smoothed by a trailing SMOOTH-day mean (all inputs same length)."""
    n = len(phase_by_etf[0])
    raw = [order_parameter([p[t] for p in phase_by_etf]) for t in range(n)]
    out: list[float | None] = []
    for t in range(n):
        w = raw[max(0, t - SMOOTH + 1): t + 1]
        out.append(sum(w) / len(w) if len(w) == SMOOTH and None not in w else None)
    return out


def log_returns(closes: Sequence[float]) -> list[float]:
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]


def trailing_vol(rets: Sequence[float], t: int, win: int) -> float | None:
    """Std of the `win` returns ending at return index t (inclusive)."""
    if t + 1 < win:
        return None
    w = rets[t + 1 - win: t + 1]
    m = sum(w) / win
    return math.sqrt(sum((r - m) ** 2 for r in w) / (win - 1))


def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if sa == 0 or sb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def mean_pairwise_corr(rets_by_etf: Sequence[Sequence[float]], t: int, win: int) -> float | None:
    if t + 1 < win:
        return None
    ws = [r[t + 1 - win: t + 1] for r in rets_by_etf]
    cs = [_corr(ws[i], ws[j]) for i in range(len(ws)) for j in range(i + 1, len(ws))]
    return sum(cs) / len(cs)


def build_rows(closes_by_etf: Sequence[Sequence[float]], spy: Sequence[float],
               horizon: int = HORIZON) -> list[dict]:
    """One row per decision day t (bar index), all inputs known at close of t,
    targets strictly after t. Rows are de-strided by the caller."""
    n = len(spy)
    assert all(len(c) == n for c in closes_by_etf)
    S = sync_series([phases(c) for c in closes_by_etf])
    sec_rets = [log_returns(c) for c in closes_by_etf]
    spy_rets = log_returns(spy)  # spy_rets[k] = return into bar k+1
    rows = []
    for t in range(1, n - horizon):
        if S[t] is None:
            continue
        k = t - 1  # return index for bar t
        tv = trailing_vol(spy_rets, k, VOL_WIN)
        mc = mean_pairwise_corr(sec_rets, k, CORR_WIN)
        if tv is None or mc is None:
            continue
        fwd = spy_rets[k + 1: k + 1 + horizon]
        fm = sum(fwd) / horizon
        rows.append({
            "t": t, "S": S[t], "trail_vol": tv, "mean_corr": mc,
            "fwd_ret": sum(fwd),
            "fwd_vol": math.sqrt(sum((r - fm) ** 2 for r in fwd) / (horizon - 1)),
        })
    return rows


def _rank(xs: Sequence[float]) -> list[float]:
    from scipy.stats import rankdata
    return list(rankdata(xs))


def partial_spearman(x: Sequence[float], y: Sequence[float],
                     controls: Sequence[Sequence[float]]) -> dict | None:
    """Spearman partial correlation of x,y given controls: rank everything,
    OLS-residualize ranks of x and y on [1, ranked controls], Pearson of the
    residuals. p-value uses n - 2 - k dof (t-dist)."""
    n = len(x)
    k = len(controls)
    if n < MIN_PAIRS or n - 2 - k < 5:
        return None
    import numpy as np
    from scipy.stats import t as tdist
    Z = np.column_stack([np.ones(n)] + [np.array(_rank(c)) for c in controls])
    def resid(v):
        r = np.array(_rank(v))
        beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
        return r - Z @ beta
    rx, ry = resid(x), resid(y)
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    if den == 0:
        return None
    rho = float(rx @ ry) / den
    dof = n - 2 - k
    tt = rho * math.sqrt(dof / max(1e-12, 1 - rho * rho))
    p = float(2 * tdist.sf(abs(tt), dof))
    return {"rho": round(rho, 4), "p_value": round(p, 5), "n": n}


def raw_spearman(x: Sequence[float], y: Sequence[float]) -> dict | None:
    return partial_spearman(x, y, [])


def passes_gate2(stat: dict | None) -> bool:
    return bool(stat) and abs(stat["rho"]) >= GATE2_RHO_FLOOR and stat["p_value"] < GATE2_P_CEILING


def analyze(rows: Sequence[dict], horizon: int = HORIZON) -> dict:
    ds = list(rows)[::horizon]  # de-stride: non-overlapping forward windows
    S = [r["S"] for r in ds]
    ctrl = [[r["trail_vol"] for r in ds], [r["mean_corr"] for r in ds]]
    fr = [r["fwd_ret"] for r in ds]
    fv = [r["fwd_vol"] for r in ds]
    half = len(ds) // 2

    def sp(sl, y):
        return partial_spearman(S[sl], y[sl], [c[sl] for c in ctrl])

    res = {
        "n_destrided": len(ds),
        "fwd_ret_raw": raw_spearman(S, fr),
        "fwd_ret_partial": partial_spearman(S, fr, ctrl),
        "fwd_vol_raw": raw_spearman(S, fv),
        "fwd_vol_partial": partial_spearman(S, fv, ctrl),
        "S_vs_trail_vol_raw": raw_spearman(S, ctrl[0]),
        "S_vs_mean_corr_raw": raw_spearman(S, ctrl[1]),
        "halves_fwd_ret_partial": [sp(slice(0, half), fr), sp(slice(half, None), fr)],
    }
    res["gate2_pass_fwd_ret_partial"] = passes_gate2(res["fwd_ret_partial"])
    h = res["halves_fwd_ret_partial"]
    res["halves_same_sign"] = bool(h[0] and h[1] and h[0]["rho"] * h[1]["rho"] > 0)
    return res


def run_probe(days: int = DEFAULT_DAYS) -> dict:
    import backtest_v2
    bars = {t: backtest_v2.fetch_bars(t, days) for t in SECTORS + [TARGET]}
    dates = None
    for t, b in bars.items():
        ds = list(b["date"])
        dates = set(ds) if dates is None else dates & set(ds)
    keep = sorted(dates)
    def aligned(t):
        m = dict(zip(bars[t]["date"], bars[t]["close"]))
        return [float(m[d]) for d in keep]
    out = analyze(build_rows([aligned(t) for t in SECTORS], aligned(TARGET)))
    out["window"] = [keep[0], keep[-1]]
    out["n_aligned_bars"] = len(keep)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    print(json.dumps(run_probe(ap.parse_args().days), indent=2))
