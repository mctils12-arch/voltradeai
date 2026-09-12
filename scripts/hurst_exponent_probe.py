#!/usr/bin/env python3
"""
Rolling Hurst exponent probe: does the market's own trend-persistence
statistic predict whether the CURRENT trend continues or reverses?

FOREIGN-FIELD IMPORT (EDGE DOCTRINE #4, CLAUDE.md) from hydrology /
geophysics — a sixth field, not yet used by any prior import in this file
(ecology/critical-slowing-down 2026-08-18, epidemiology/R_t 2026-08-26,
reliability-engineering/hazard-rate 2026-08-29, seismology/Omori-Utsu
2026-08-30, information-theory/permutation-entropy 2026-09-05). Harold
Edwin Hurst (1951, "Long-Term Storage Capacity of Reservoirs", Trans.
ASCE 116) developed rescaled-range (R/S) analysis studying Nile River
flood-height persistence for reservoir sizing — a purely hydrological
question about whether high-flow years cluster or alternate randomly.
Mandelbrot (1963, 1971) and later Peters ("Fractal Market Analysis", 1994)
imported the same statistic to price series as the "Fractal Market
Hypothesis": H=0.5 is a memoryless random walk, H>0.5 means increments are
POSITIVELY correlated (a trending/persistent series — a big up-move tends
to be followed by another up-move), H<0.5 means increments are NEGATIVELY
correlated (a mean-reverting/anti-persistent series).

WHY THIS IS A GENUINELY DIFFERENT DESIGN, not a sixth variant of the same
fished idea (REASONING STANDARD #4): the 2026-09-05 permutation-entropy
session's own STRUCTURAL META-FINDING (research/open_questions.md, same
date) diagnosed that every prior import in this family used an
ONSET-COUNTING design (rare discrete regime-transition events vs a control
sample) and hit the same statistical-power ceiling five times running
(n~10-15 onsets, however the universe was widened). It explicitly
recommended, as the alternative NOT yet tried, "a continuous rolling
statistic scored against forward returns directly" — the same shape as
this file's own already-passed GATE 2 designs (e.g.
wikimedia_pageviews_attention's z-score vs forward-metric). This probe is
exactly that: it never counts an "onset" or builds a control sample of
non-events. Every trading day gets its own Hurst estimate, and every day
with a valid estimate is one data point in one continuous correlation
test against forward returns. This also sidesteps the n~10-15 sample-size
ceiling entirely — with ~2500 trading days over 10y, a valid H estimate on
even a small majority of them gives an n in the hundreds to thousands.

HYPOTHESIS (pre-registered BEFORE running against real data, REASONING
STANDARD #10): define trailing_trend_t = sign(sum of daily log returns
over the L days ending at t), and continuation_score_t = trailing_trend_t
* (sum of daily log returns over the H days starting at t+1) — positive
when the forward move continues the prior trend's direction, negative
when it reverses. PREDICTION: rolling_hurst_t (computed over a window
ending at t, using only information available at t) is POSITIVELY
correlated with continuation_score_t — days classified as "trending"
(H>0.5) are followed by more trend-continuation than days classified as
"mean-reverting" (H<0.5).

PRIOR, stated honestly before computing anything (REASONING STANDARD #10
and #5 — second-order thinking): this prediction is close to definitional
— H itself is DEFINED from the autocorrelation structure of the very same
return series being tested, so finding SOME positive relationship is the
weak-form expected result, not a discovery. The genuinely open empirical
questions are (1) whether the effect is large enough to matter net of the
noise in a real 2500-point sample, and (2) whether it would still matter
after costs/regime-conditioning if it did clear GATE 2. Per REASONING
STANDARD #5 (who is on the other side, why hasn't this been arbitraged):
Hurst/R-S trading signals are a 40+-year-old, heavily studied idea in the
academic literature, and the dominant finding since Lo (1991, "Long-Term
Memory in Stock Market Prices", Econometrica) is that naive R/S estimates
on short financial samples are severely biased by short-range
autocorrelation and heteroskedasticity and OVERSTATE apparent long memory
— i.e. the textbook R/S statistic this probe implements is specifically
the version the literature says is least trustworthy. Combined with five
prior foreign-field imports in this exact file already GATE-2-killed or
unresolved, my prior is that this most likely fails GATE 2 too, or passes
only as a restatement of ordinary short-lag autocorrelation (the same
"is this independent information, or AR1 in a new unit" question the
permutation-entropy entry raised) — but the continuous design is worth
running once cleanly rather than assumed to fail, since it was never
tried in this family and is cheap to compute from data already fetched.

LADDER PATH: this is a GATE 2 (SIGNAL) test only — pure statistical
predictive power on daily bars, no trading logic, no sizing. A future
GATE 3 (backtested entry/exit ablation) would only be warranted if GATE 2
clears with an effect size + significance a human would find persuasive
net of REASONING STANDARD #4's multiple-testing discount (this is the
SIXTH foreign-field import attempted in this file).

MEASUREMENT INTEGRITY note (this is a research probe, not measurement
code CLAUDE.md's own section governs, but its spirit applies): reports
the null result as plainly as a positive one; does not retry with
different window/lag parameters after seeing an unfavorable result
(REASONING STANDARD #4 — one theory-motivated spec, run once, reported
honestly).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest_v2  # noqa: E402

MIN_PAIRS_FOR_STATS = 5


def log_returns(closes: Sequence[float]) -> list[float]:
    """Daily log returns from a closing-price series. Requires closes[i] > 0."""
    out = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            out.append(None)
        else:
            out.append(math.log(closes[i] / closes[i - 1]))
    return out


def _rs_stat(chunk: Sequence[float]) -> float | None:
    """Classic rescaled-range statistic for one contiguous chunk of returns:
    R (range of the mean-adjusted cumulative sum) / S (sample std dev).
    None if the chunk is degenerate (zero variance, or too short)."""
    n = len(chunk)
    if n < 2:
        return None
    mean = sum(chunk) / n
    dev = [x - mean for x in chunk]
    cum = 0.0
    cum_series = []
    for d in dev:
        cum += d
        cum_series.append(cum)
    r = max(cum_series) - min(cum_series)
    var = sum(d * d for d in dev) / n
    if var <= 0:
        return None
    s = math.sqrt(var)
    if s == 0:
        return None
    return r / s


def hurst_rs(returns: Sequence[float], min_chunk: int = 8, max_chunk: int | None = None
             ) -> float | None:
    """Estimate the Hurst exponent from one window of returns via chunked
    rescaled-range analysis: split the window into non-overlapping chunks
    of size n for several n in [min_chunk, max_chunk], average R/S across
    chunks of the same size, then regress log(mean R/S) on log(n) — the
    slope is H. Returns None if fewer than 3 distinct chunk sizes produce
    a valid (finite, positive) R/S value, since a 2-point "regression"
    is not a meaningful estimate."""
    clean = [x for x in returns if x is not None]
    if len(clean) < min_chunk * 2:
        return None
    n_total = len(clean)
    if max_chunk is None:
        max_chunk = n_total // 2
    max_chunk = min(max_chunk, n_total // 2)
    if max_chunk < min_chunk:
        return None

    sizes = sorted(set(int(round(s)) for s in
                        _log_spaced(min_chunk, max_chunk, 8) if s >= min_chunk))
    xs, ys = [], []
    for n in sizes:
        n_chunks = n_total // n
        if n_chunks < 1:
            continue
        rs_vals = []
        for c in range(n_chunks):
            chunk = clean[c * n:(c + 1) * n]
            rs = _rs_stat(chunk)
            if rs is not None and rs > 0:
                rs_vals.append(rs)
        if rs_vals:
            xs.append(math.log(n))
            ys.append(math.log(sum(rs_vals) / len(rs_vals)))

    if len(xs) < 3:
        return None
    slope = _ols_slope(xs, ys)
    return slope


def _log_spaced(lo: int, hi: int, count: int) -> list[float]:
    if lo <= 0 or hi <= 0 or hi < lo:
        return []
    log_lo, log_hi = math.log(lo), math.log(hi)
    if count <= 1:
        return [lo]
    step = (log_hi - log_lo) / (count - 1)
    return [math.exp(log_lo + i * step) for i in range(count)]


def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def rolling_hurst(returns: Sequence[float], window: int = 252, min_chunk: int = 8
                   ) -> list[float | None]:
    """H_t for each index t in `returns`, estimated ONLY from returns[t-window+1..t]
    (never future data — no lookahead). None for the first `window`-1 entries
    and any window where hurst_rs itself returns None."""
    out: list[float | None] = [None] * len(returns)
    for t in range(window - 1, len(returns)):
        window_slice = returns[t - window + 1:t + 1]
        out[t] = hurst_rs(window_slice, min_chunk=min_chunk)
    return out


def continuation_scores(returns: Sequence[float], hurst: Sequence[float | None],
                         lookback: int = 20, horizon: int = 20
                         ) -> list[tuple[float, float]]:
    """For each valid index t (hurst[t] is not None, and both a full
    lookback window before t and a full horizon window after t exist),
    returns (hurst[t], continuation_score) pairs. continuation_score =
    sign(trailing L-day return) * (forward H-day return) — positive means
    the forward move continued the trend direction that was already
    established as of day t using only information available at t."""
    out = []
    n = len(returns)
    for t in range(lookback, n - horizon):
        h = hurst[t]
        if h is None:
            continue
        trailing = sum(r for r in returns[t - lookback:t] if r is not None)
        if trailing == 0:
            continue
        forward = sum(r for r in returns[t + 1:t + 1 + horizon] if r is not None)
        score = (1.0 if trailing > 0 else -1.0) * forward
        out.append((h, score))
    return out


def spearman(pairs: Sequence[tuple[float, float]]) -> dict | None:
    if len(pairs) < MIN_PAIRS_FOR_STATS:
        return None
    from scipy.stats import spearmanr
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rho, p_value = spearmanr(xs, ys)
    return {"rho": round(float(rho), 4), "p_value": round(float(p_value), 5), "n": len(pairs)}


def destrided_spearman(pairs: Sequence[tuple[float, float]], stride: int) -> dict | None:
    """`spearman()` computed on every `stride`-th pair instead of all of
    them. `continuation_scores()` produces one (H, score) pair PER DAY,
    but consecutive days' `horizon`-day forward windows overlap almost
    entirely (day t and day t+1's forward windows share horizon-1 of
    their horizon days) — so the naive n from spearman() on the full
    daily series is not the effective (independent) sample size, and its
    p-value is systematically too small. Calling this with
    stride=horizon selects one pair per non-overlapping forward window,
    which is the standard fix for this exact autocorrelated-overlapping-
    sample problem in return-horizon studies. This does not change the
    estimated effect size (rho) much if the relationship is real, but it
    is often the difference between a p-value that looks significant and
    one that does not — always report both, never only the naive one."""
    return spearman(list(pairs[::stride])) if stride > 0 else None


def tertile_welch(pairs: Sequence[tuple[float, float]]) -> dict | None:
    """Top-tertile-H vs bottom-tertile-H mean continuation_score, Welch
    t-test. Reuses permutation_entropy_probe.welch_vs_control rather than
    reimplementing a second significance-test helper (EDGE DOCTRINE #3)."""
    if len(pairs) < MIN_PAIRS_FOR_STATS * 2:
        return None
    sorted_pairs = sorted(pairs, key=lambda p: p[0])
    n = len(sorted_pairs)
    cut = n // 3
    if cut < MIN_PAIRS_FOR_STATS:
        return None
    low = [p[1] for p in sorted_pairs[:cut]]
    high = [p[1] for p in sorted_pairs[-cut:]]

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "permutation_entropy_probe",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "permutation_entropy_probe.py"))
    pep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pep)
    result = pep.welch_vs_control(high, low)
    if result is None:
        return None
    return {
        "high_h_mean_continuation": round(sum(high) / len(high), 6),
        "low_h_mean_continuation": round(sum(low) / len(low), 6),
        "high_h_n": len(high),
        "low_h_n": len(low),
        **result,
    }


def run_probe(ticker: str = "SPY", days: int = 2520, window: int = 252,
              min_chunk: int = 8, lookback: int = 20, horizon: int = 20) -> dict:
    bars = backtest_v2.fetch_bars(ticker, days)
    closes = bars.get("close", []) if bars else []
    dates = bars.get("date", []) if bars else []
    if len(closes) < window + lookback + horizon + 10:
        return {"error": "insufficient bars", "ticker": ticker, "n_bars": len(closes)}

    rets = log_returns(closes)
    hurst = rolling_hurst(rets, window=window, min_chunk=min_chunk)
    pairs = continuation_scores(rets, hurst, lookback=lookback, horizon=horizon)

    valid_h = [h for h in hurst if h is not None]
    return {
        "ticker": ticker,
        "n_bars": len(closes),
        "date_range": [dates[0], dates[-1]] if dates else None,
        "window": window,
        "lookback": lookback,
        "horizon": horizon,
        "n_valid_hurst_days": len(valid_h),
        "mean_hurst": round(sum(valid_h) / len(valid_h), 4) if valid_h else None,
        "n_scored_pairs": len(pairs),
        "spearman_naive_daily": spearman(pairs),
        "spearman_destrided": destrided_spearman(pairs, stride=horizon),
        "tertile_welch_naive_daily": tertile_welch(pairs),
        "tertile_welch_destrided": tertile_welch(list(pairs[::horizon])),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--days", type=int, default=2520, help="~10y of trading days")
    ap.add_argument("--window", type=int, default=252, help="rolling Hurst estimation window")
    ap.add_argument("--min-chunk", type=int, default=8, dest="min_chunk")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=20)
    args = ap.parse_args()
    print(json.dumps(run_probe(args.ticker, args.days, args.window,
                                args.min_chunk, args.lookback, args.horizon), indent=2))
