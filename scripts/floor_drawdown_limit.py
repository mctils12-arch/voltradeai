#!/usr/bin/env python3
"""
floor_drawdown_limit.py — the number the floor's second rollback leg points at.

The volatility-managed floor (research/floor_study_2026-11-05.md) was
pre-registered with a two-leg rollback trigger. The second leg reads "account
drawdown beyond the backtest's 95th-percentile 3-month drawdown". The study
never computed that percentile, so the leg had no limit and
server/floorTracking.ts reported it as "not evaluable" (2026-11-12).

This script computes it from the study's OWN code and inputs: it imports
scripts/floor_study.py (load, returns, daily_rf, run_v1, metrics) and adds one
measurement on top of V1's daily returns. It changes nothing in the study.

DEFINITION (written 2026-11-22, before the account was set against it)
  3 months            63 sessions (252 / 4).
  a window            every run of 63 consecutive sessions of the backtest:
                      64 closing values, the close before the first session
                      and the close of each session.
  its drawdown        the largest fall from a high to a later low INSIDE the
                      window, as a fraction of the high (0 when the window
                      never falls).
  95th percentile     of the windows' drawdowns, by severity: 95% of the
                      windows fell less. Linear interpolation between ranks.
This is the PRIMARY reading and the one the server tests. Two other readings
of the same words are computed and written beside it, so the choice is
visible and its effect measurable (`alternatives`):
  from_window_high    on each day, the fall from the highest close of the
                      last 63 sessions (the drawdown the account is IN, not
                      the worst one it had);
  window_return       the 63-session return, 5th percentile.

REPRODUCTION CHECK. The result records V1's CAGR, Sharpe, max drawdown and
log growth as recomputed here, beside the study's published figures. Yahoo
re-states adjusted closes when a dividend is paid, so a later download can
differ in the last digit; `reproduces_study` is true when all four are within
0.02 of the study's.

Usage:
    python3 scripts/floor_drawdown_limit.py <dir with h_SPY.json h_IRX.json> <out.json>
The two files are Yahoo chart-API JSON, daily, 1993-01-29 onward (the study's
inputs; they are not committed).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floor_study as fs  # noqa: E402

WINDOW_SESSIONS = 63
PERCENTILE = 0.95
STUDY_END = "2026-11-04"          # the study's last session (its `window.to`)
STUDY_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "floor_study_2026-11-05.json")
REPRODUCTION_TOL = 0.02


def equity_path(rets: Sequence[float]) -> List[float]:
    """Closing values from daily returns, starting at 1.0 (one more value than returns)."""
    out = [1.0]
    for r in rets:
        out.append(out[-1] * (1.0 + r))
    return out


def max_drawdown(values: Sequence[float]) -> float:
    """Largest fall from a high to a later low, as a fraction of the high: 0 or negative."""
    peak, worst = None, 0.0
    for v in values:
        if peak is None or v > peak:
            peak = v
        if peak and peak > 0:
            worst = min(worst, v / peak - 1.0)
    return worst


def window_drawdowns(rets: Sequence[float], window: int = WINDOW_SESSIONS) -> List[float]:
    """The drawdown inside every run of `window` consecutive sessions."""
    eq = equity_path(rets)
    return [max_drawdown(eq[e - window:e + 1]) for e in range(window, len(eq))]


def from_window_high(rets: Sequence[float], window: int = WINDOW_SESSIONS) -> List[float]:
    """On each day, the fall from the highest close of the last `window` sessions."""
    eq = equity_path(rets)
    return [eq[e] / max(eq[e - window:e + 1]) - 1.0 for e in range(window, len(eq))]


def window_returns(rets: Sequence[float], window: int = WINDOW_SESSIONS) -> List[float]:
    eq = equity_path(rets)
    return [eq[e] / eq[e - window] - 1.0 for e in range(window, len(eq))]


def severity_percentile(values: Sequence[float], q: float) -> float:
    """The value that a share q of `values` is ABOVE (less severe than): the (1 - q) quantile, interpolated."""
    xs = sorted(values)
    if not xs:
        raise ValueError("no windows")
    k = (len(xs) - 1) * (1.0 - q)
    lo = int(math.floor(k))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(rets: Sequence[float], window: int = WINDOW_SESSIONS) -> Dict:
    pct = lambda xs, q: round(100.0 * severity_percentile(xs, q), 2)  # noqa: E731
    d1, d3, d2 = window_drawdowns(rets, window), from_window_high(rets, window), window_returns(rets, window)
    return {
        "window_sessions": window,
        "windows": len(d1),
        "p95_pct": pct(d1, 0.95), "p99_pct": pct(d1, 0.99), "worst_pct": round(100.0 * min(d1), 2),
        "alternatives": {
            "from_window_high": {"p95_pct": pct(d3, 0.95), "p99_pct": pct(d3, 0.99)},
            "window_return": {"p05_pct": pct(d2, 0.95), "p01_pct": pct(d2, 0.99)},
        },
    }


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(src: str, out_path: str) -> None:
    spy_file, irx_file = os.path.join(src, "h_SPY.json"), os.path.join(src, "h_IRX.json")
    spy_px = fs.load(spy_file)
    irx = fs.load(irx_file, adjusted=False)
    days = sorted(d for d in spy_px if d <= STUDY_END)
    px = [spy_px[d] for d in days]
    r = fs.returns(px)
    rdays = days[1:]
    rf = fs.daily_rf(irx, rdays)
    v1, _ = fs.run_v1(r, rf)

    study = json.load(open(STUDY_RESULTS))
    keys = ("cagr_pct", "sharpe", "max_dd_pct", "log_growth_pct")
    here = fs.metrics(v1, rf)
    published = study["V1_volmanaged"]
    reproduces = (len(r) == study["window"]["sessions"] and rdays[0] == study["window"]["from"] and rdays[-1] == study["window"]["to"]
                  and all(abs(here[k] - published[k]) <= REPRODUCTION_TOL for k in keys))

    out = {
        "what": "the volatility-managed floor's 95th-percentile 3-month drawdown in its pre-registered backtest (V1), "
                "the limit of the rollback trigger's second leg",
        "computed": "2026-11-22",
        "study": "research/floor_study_2026-11-05.md (pre-registration and results unchanged by this file)",
        "script": "scripts/floor_drawdown_limit.py over scripts/floor_study.py run_v1",
        "definition": "63 sessions = 3 months; a window = 63 consecutive sessions (64 closes); its drawdown = the largest "
                      "fall from a high to a later low inside the window; the limit = the drawdown 95% of the windows stayed above",
        "window": {"from": rdays[0], "to": rdays[-1], "sessions": len(r)},
        "V1_volmanaged": summarize(v1),
        "B0_SPY": summarize(r),
        "reproduction": {
            "recomputed": {k: here[k] for k in keys},
            "study": {k: published[k] for k in keys},
            "tolerance": REPRODUCTION_TOL,
            "reproduces_study": reproduces,
        },
        "inputs": {
            "source": "Yahoo chart API, daily, adjusted close (SPY) and close (^IRX)",
            "h_SPY.json": {"sha256": sha256_of(spy_file), "first": days[0], "last_used": days[-1]},
            "h_IRX.json": {"sha256": sha256_of(irx_file)},
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    v = out["V1_volmanaged"]
    print(f"V1 {v['windows']} windows of {WINDOW_SESSIONS} sessions: 95th-percentile drawdown {v['p95_pct']}% "
          f"(99th {v['p99_pct']}%, worst {v['worst_pct']}%); reproduces the study: {reproduces}")
    if not reproduces:
        sys.exit(2)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
