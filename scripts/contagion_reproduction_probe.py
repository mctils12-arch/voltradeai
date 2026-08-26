#!/usr/bin/env python3
"""
Sector contagion "reproduction number" probe.

FOREIGN-FIELD IMPORT (EDGE DOCTRINE #4, CLAUDE.md) from epidemiology:
Cori et al. 2013 ("A New Framework and Software to Estimate Time-Varying
Reproduction Numbers During Epidemics", Am J Epidemiol 178(9)) estimates
R_t — how fast an outbreak is currently spreading — as the ratio of new
cases in a recent window to new cases in the window before it (the full
method additionally convolves by a serial-interval distribution; this
probe uses the simpler uniform-window special case of that same ratio,
stated as an honest simplification, not the full parametric model).
grepped experiments.md/open_questions.md for "reproduction number",
"epidemiology", "contagion", "R_t"/"Rt", "SIR" before writing this file:
zero prior hits — genuinely new foreign-field import, distinct from the
2026-08-18/21 ecology (critical-slowing-down) import already on file
(that one measured a SINGLE index's own autocorrelation/variance; this
one is cross-sectional — how fast is a "case" — a fresh technical
breakdown — spreading ACROSS a basket of sector peers).

HYPOTHESIS (pre-registered, testable form; full ladder entry in
research/open_questions.md): define a daily "distress case" for a ticker
as its close making a new N-trading-day low (a discrete, threshold-free
event, mirroring epidemiological case counting). Sum cases across a
sector's constituent stocks each day to get I_t (today's case count).
R_t = sum(I_t over the last tau days) / sum(I_t over the tau days before
that). CLAIM: R_t crossing above 1 (case count still ACCELERATING) carries
predictive information about the sector's forward index return beyond
what the raw case LEVEL (I_t) already carries — i.e. the epidemiological
insight (spread RATE matters, not just case count) transfers to market
breakdowns.

PRIOR (stated before running against real data, REASONING STANDARD #10):
the raw level I_t (how many names are breaking down right now) is already
a fairly direct breadth-deterioration signal, similar in spirit to what
stress_index.py's crude "SPY vs MA50/MA200" breadth component captures.
The genuinely non-obvious claim is whether R_t (the SECOND-ORDER,
acceleration view) adds INDEPENDENT lead-time value on top of the level.
If R_t's correlation with forward returns is no stronger than I_t's own,
this import adds nothing new — a valid negative result to record, not a
reason to hide the entry (same discipline as the CSD probe's own outcome).

LADDER PATH: gate-2 SIGNAL test (statistical predictive power, no trading)
built on daily bars already gate-1-verified elsewhere in this codebase
(backtest_v2.fetch_bars, Alpaca-first/Yahoo-fallback) — no new gate-1
needed, same precedent as critical_slowing_down_probe.py.

UNIVERSE: bot_engine.SECTOR_MAP's existing "Technology" tickers (the
live bot's own sector classification — reused, not re-derived) as the
cross-sectional case-counting basket, QQQ (tech-heavy, already used
elsewhere in this codebase as a sector-adjacent index) as the forward-
return target.

HONEST LIMITATIONS (state up front):
  - The window-ratio R_t is Cori's method with a UNIFORM, unweighted
    serial interval over `tau` days, not the full gamma/discretized
    serial-interval convolution the epidemiology literature uses — no
    market-specific "generation interval" is established, so a uniform
    window is the honest, assumption-minimal choice, not a claim of
    reproducing the full method.
  - "New N-day low" is a single, simple event definition. A different
    threshold (volatility-normalized shock, % drawdown, etc.) could show
    different results — this probe tests ONE reasonable operationalization,
    not the whole design space (REASONING STANDARD #4 discipline: state
    what was tried, discount accordingly).
  - Single sector (Technology), single target (QQQ), single archive
    window. A clean result here is a first data point, not a general law.

USAGE:
    python3 scripts/contagion_reproduction_probe.py [--days 2520]
        [--window 20] [--tau 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_N_FOR_STATS = 30  # matches regime_detector_compare.py's discipline of an honest floor
DEFAULT_HORIZONS = (5, 10, 20)


def new_low_flags(closes: Sequence[float], window: int) -> list:
    """1 if closes[i] is a new `window`-day trailing low (its own close is
    the minimum of the trailing `window` closes including itself), else 0.
    None for indices before `window` trading days of history exist — an
    undefined event is reported as None, never fabricated as 0."""
    out = []
    for i in range(len(closes)):
        if i < window - 1:
            out.append(None)
            continue
        trailing = closes[i - window + 1:i + 1]
        out.append(1 if closes[i] <= min(trailing) else 0)
    return out


def align_common_dates(bars_by_ticker: dict) -> tuple:
    """Intersect every ticker's date set, return (sorted common dates,
    {ticker: closes reindexed to those dates}). Handles tickers with
    slightly different trading-calendar gaps (halts, late IPOs, etc.)
    without assuming every input series is already date-aligned."""
    date_sets = [set(b["date"]) for b in bars_by_ticker.values() if b and b.get("date")]
    if not date_sets:
        return [], {}
    common = sorted(set.intersection(*date_sets))
    aligned = {}
    for ticker, bars in bars_by_ticker.items():
        if not bars or not bars.get("date"):
            continue
        close_by_date = dict(zip(bars["date"], bars["close"]))
        aligned[ticker] = [close_by_date[d] for d in common]
    return common, aligned


def cross_sectional_event_counts(aligned_closes_by_ticker: dict, window: int) -> list:
    """Per-date sum of new_low_flags across every ticker. None on any date
    where at least one ticker's flag is still undefined (insufficient
    trailing history) — never silently treats "unknown" as "zero cases"."""
    if not aligned_closes_by_ticker:
        return []
    per_ticker_flags = {t: new_low_flags(c, window) for t, c in aligned_closes_by_ticker.items()}
    # Callers are expected to pre-align every ticker to the same date grid
    # (align_common_dates does this); defensively use the shortest series
    # rather than assuming equal lengths, so a caller mistake degrades to a
    # trimmed result instead of an IndexError.
    n_dates = min(len(v) for v in per_ticker_flags.values())
    out = []
    for i in range(n_dates):
        flags = [per_ticker_flags[t][i] for t in per_ticker_flags]
        if any(f is None for f in flags):
            out.append(None)
        else:
            out.append(sum(flags))
    return out


def windowed_reproduction_number(event_counts: Sequence, tau: int) -> list:
    """Cori-style simplified R_t: sum(I over the last tau days, ending at
    and including t) / sum(I over the tau days immediately before that).
    None whenever either window contains an undefined (None) count, or the
    prior window's sum is 0 (growth off a zero base is undefined — not
    reported as +inf or silently clamped)."""
    out = []
    for i in range(len(event_counts)):
        if i < 2 * tau - 1:
            out.append(None)
            continue
        cur_window = event_counts[i - tau + 1:i + 1]
        prior_window = event_counts[i - 2 * tau + 1:i - tau + 1]
        if any(v is None for v in cur_window) or any(v is None for v in prior_window):
            out.append(None)
            continue
        prior_sum = sum(prior_window)
        if prior_sum == 0:
            out.append(None)
            continue
        out.append(sum(cur_window) / prior_sum)
    return out


def forward_return_pct(closes: Sequence[float], i: int, n: int) -> Optional[float]:
    if i + n >= len(closes) or closes[i] is None or closes[i] <= 0:
        return None
    fwd = closes[i + n]
    if fwd is None:
        return None
    return (fwd / closes[i] - 1) * 100


def _clean_pairs(xs: list, ys: list) -> tuple:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if not pairs:
        return [], []
    xa, ya = zip(*pairs)
    return list(xa), list(ya)


def compute_lead_signal(event_counts: Sequence, rt_series: Sequence,
                         target_closes: Sequence[float],
                         horizons: Sequence[int] = DEFAULT_HORIZONS) -> dict:
    """For each horizon, Spearman rank correlation of (a) R_t and (b) the
    raw case-count level against the target's forward return, over every
    date where both the signal and the forward return are defined. Honest
    insufficient_n gate: below MIN_N_FOR_STATS, reports n and no fabricated
    statistic (same discipline as regime_detector_compare.py's own floor)."""
    from scipy.stats import spearmanr

    out: dict = {}
    n = len(event_counts)
    for h in horizons:
        rt_vals, level_vals, fwd_vals = [], [], []
        for i in range(n):
            fr = forward_return_pct(target_closes, i, h)
            if fr is None:
                continue
            rt_vals.append(rt_series[i] if i < len(rt_series) else None)
            level_vals.append(event_counts[i])
            fwd_vals.append(fr)

        rt_x, rt_y = _clean_pairs(rt_vals, fwd_vals)
        lvl_x, lvl_y = _clean_pairs(level_vals, fwd_vals)

        entry: dict = {"n_rt": len(rt_x), "n_level": len(lvl_x)}
        if len(rt_x) < MIN_N_FOR_STATS:
            entry["rt"] = {"insufficient_n": True, "n": len(rt_x)}
        else:
            rho, p = spearmanr(rt_x, rt_y)
            entry["rt"] = {"rho": round(float(rho), 4), "p": round(float(p), 6), "n": len(rt_x)}
        if len(lvl_x) < MIN_N_FOR_STATS:
            entry["level"] = {"insufficient_n": True, "n": len(lvl_x)}
        else:
            rho, p = spearmanr(lvl_x, lvl_y)
            entry["level"] = {"rho": round(float(rho), 4), "p": round(float(p), 6), "n": len(lvl_x)}
        out[f"h{h}"] = entry
    return out


def _tech_universe() -> list:
    """Reuses the live bot's own sector classification (bot_engine.SECTOR_MAP)
    instead of hand-picking a fresh ticker list — EDGE DOCTRINE #3, don't
    re-derive what the codebase already has."""
    import bot_engine
    seen, out = set(), []
    for ticker, sector in bot_engine.SECTOR_MAP.items():
        if sector == "Technology" and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def run_probe(tickers: Optional[list] = None, target_ticker: str = "QQQ",
              days: int = 2520, window: int = 20, tau: int = 5,
              horizons: Sequence[int] = DEFAULT_HORIZONS) -> dict:
    import backtest_v2

    if tickers is None:
        tickers = _tech_universe()

    bars_by_ticker = {}
    for t in tickers:
        b = backtest_v2.fetch_bars(t, days)
        if b and b.get("date"):
            bars_by_ticker[t] = b
    target_bars = backtest_v2.fetch_bars(target_ticker, days)

    missing = [t for t in tickers if t not in bars_by_ticker]
    if not bars_by_ticker or not target_bars or not target_bars.get("date"):
        return {"error": "no data available", "missing": missing}

    common_dates, aligned = align_common_dates(bars_by_ticker)
    target_close_by_date = dict(zip(target_bars["date"], target_bars["close"]))
    target_closes = [target_close_by_date.get(d) for d in common_dates]

    event_counts = cross_sectional_event_counts(aligned, window)
    rt_series = windowed_reproduction_number(event_counts, tau)
    signal = compute_lead_signal(event_counts, rt_series, target_closes, horizons)

    return {
        "universe": sorted(bars_by_ticker.keys()),
        "missing_tickers": missing,
        "target": target_ticker,
        "window": window,
        "tau": tau,
        "date_range": [common_dates[0], common_dates[-1]] if common_dates else None,
        "n_dates": len(common_dates),
        "signal": signal,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2520)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--tau", type=int, default=5)
    ap.add_argument("--target", default="QQQ")
    args = ap.parse_args()
    result = run_probe(target_ticker=args.target, days=args.days,
                        window=args.window, tau=args.tau)
    print(json.dumps(result, indent=2))
