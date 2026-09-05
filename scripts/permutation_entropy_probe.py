#!/usr/bin/env python3
"""
Permutation-entropy early-warning-signal probe.

FOREIGN-FIELD IMPORT (EDGE DOCTRINE #4, CLAUDE.md) from information
theory / nonlinear dynamics, a field not yet used by any prior import in
this file (ecology/critical-slowing-down 2026-08-18, epidemiology/R_t
2026-08-26, reliability-engineering/hazard-rate 2026-08-29, seismology/
Omori-Utsu 2026-08-30 — all four already spent; ACTIVE ANGLE-HUNTING's own
standing-behaviors list names "signal processing" as a fifth candidate
field never yet tried). Bandt & Pompe 2002 ("Permutation Entropy: A
Natural Complexity Measure for Time Series", Phys. Rev. Lett. 88, 174102)
define a complexity measure over a time series' ORDINAL PATTERNS (the
relative rank order of m consecutive values) rather than their raw
magnitudes — robust to noise/scale, and cheap to compute from bars this
codebase already has. A finance-specific literature (Zunino et al. 2009,
"Permutation entropy of fractional Brownian motion and fractional
Gaussian noise", Phys. Lett. A; Bariviera 2021 survey) has used rolling
permutation entropy of returns as a market-(in)efficiency / regime
diagnostic: a fully random (i.i.d.) return series visits all m! ordinal
patterns with equal frequency (entropy -> 1, normalized); a strongly
trending or herding series concentrates on a few patterns (entropy -> 0).

HYPOTHESIS (pre-registered, full ladder entry in research/open_questions.md):
rolling normalized permutation entropy of SPY daily log returns, computed
over a trailing window, falls (the return sequence becomes MORE ordered)
in the days before a regime transition into a MORE SEVERE regime
(BULL/CAUTION -> NEUTRAL/BEAR/PANIC, same severity order and same onset
definition as the CSD entry), versus a regime-matched control sample of
non-transition days drawn from the same "from" regimes (not an
unconditional archive-wide pool — the CSD entry's own MEASUREMENT
INTEGRITY finding about regime-duration-weighted control pools applies
identically here, so this probe reuses that exact mechanism rather than
re-deriving a weaker one).

PRIOR (stated before this was ever run against real data, REASONING
STANDARD #10): I expect entropy to FALL before a severity-increasing
transition — panic/selloff price action is more directional/herding
(one-sided flow, momentum, correlated selling) than calm-regime chop,
and a more directional return sequence visits fewer distinct ordinal
patterns. This is the SAME underlying intuition as the CSD entry's
"rising AR1" finding (both describe returns becoming less
"random-walk-like"), which is exactly why the honest, useful question is
NOT "does entropy move at all" but "does entropy's move carry
INDEPENDENT information beyond AR1, or is it just AR1 restated in
different units" — permutation entropy captures ordinal structure across
an m-length window (order, not magnitude), which is a genuinely different
statistic from a single lag-1 linear correlation coefficient, but the two
could still covary strongly in practice. This probe reports the
onset-vs-control comparison for entropy exactly like the CSD probe does
for AR1/variance, AND a plain descriptive Pearson correlation between the
two series' valid overlapping values, so a future reader can see whether
this import adds anything beyond what CSD already found (REASONING
STANDARD #4's "discount by number of variants tried" applies here with a
concrete instrument, not just a verbal caveat) — four prior foreign-field
imports in this file were all killed or left underpowered (CSD, R_t,
Omori-Utsu all GATE 2 killed; hazard-rate GATE 2 not passed even pooled),
so the prior on this one clearing the ladder is not high either.

LADDER PATH: gate-2 SIGNAL test (statistical predictive power, no
trading) built directly on top of SPY/VXX daily bars already gate-1
verified elsewhere in this codebase (backtest_v2.fetch_bars, the same
Alpaca-first/Yahoo-fallback path every prior probe in this file reuses)
— no new gate-1 needed, same precedent as CSD/hazard-rate/Omori.

WHY THIS SESSION DID NOT RUN IT AGAINST REAL DATA: this sandbox has
neither ALPACA_KEY/ALPACA_SECRET in its environment (confirmed via `env`)
nor working network access to Yahoo Finance this session (`yfinance.
Ticker('SPY').history(period='5d')` hard-timed-out at the egress proxy —
`ws_closed_mid_exchange` on query2.finance.yahoo.com/guce.yahoo.com/
fc.yahoo.com, the identical failure class every prior blocked session in
this file logged, confirmed live this session rather than assumed from a
prior session's note). The statistical core below is pure and
unit-tested against synthetic data in test_permutation_entropy_probe.py,
so a future session with real data access can call run_probe() directly
with zero additional design work (EDGE DOCTRINE #3: compile the reasoning
now, don't re-derive it later).

USAGE (future session, once ALPACA_KEY/SECRET or working Yahoo access
exists):
    python3 scripts/permutation_entropy_probe.py [--days 2520]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Sequence

MIN_ONSETS_FOR_STATS = 5  # same n>=5 reporting floor used elsewhere in this repo


def log_returns(closes: Sequence[float]) -> list[float]:
    """Daily log returns — identical definition to the CSD probe's own
    log_returns, duplicated rather than imported because this module
    otherwise has zero dependency on critical_slowing_down_probe.py except
    via the lazy _load_csd_module() path (matches hazard_rate_probe.py's
    and omori_aftershock_probe.py's own established precedent of not
    creating a hard top-level import between sibling probe scripts)."""
    out = []
    for i in range(1, len(closes)):
        prev, cur = closes[i - 1], closes[i]
        if prev is None or cur is None or prev <= 0 or cur <= 0:
            out.append(0.0)
            continue
        out.append(math.log(cur / prev))
    return out


def ordinal_pattern(window: Sequence[float]) -> tuple[int, ...]:
    """Bandt-Pompe ordinal pattern of a length-m window: the permutation
    of {0, ..., m-1} obtained by ranking window's values ascending, ties
    broken by original position (stable sort) so a flat/constant window
    always maps to the identity pattern (0, 1, ..., m-1) rather than an
    arbitrary one — ties are common in return series only via literal
    zero-return days, which do occur (illiquid names, exact-price-repeat
    days), so this must be deterministic, not implementation-defined."""
    order = sorted(range(len(window)), key=lambda i: (window[i], i))
    rank = [0] * len(window)
    for r, i in enumerate(order):
        rank[i] = r
    return tuple(rank)


def permutation_entropy(values: Sequence[float], m: int = 3, tau: int = 1
                         ) -> float | None:
    """Normalized Shannon entropy (base e, divided by ln(m!)) of the
    distribution of ordinal patterns found in `values` using embedding
    dimension `m` and delay `tau`. Returns a value in [0, 1]: 0 = every
    extracted pattern identical (maximally ordered), 1 = all m! patterns
    equally frequent (maximally disordered — the i.i.d./white-noise
    limit). Returns None if fewer than m! patterns' worth of samples are
    available (same "insufficient data, don't fabricate a number" stance
    every other probe in this file takes) — concretely, requires at least
    m! + 1 extractable windows so the distribution isn't a single trivial
    observation."""
    if m < 2 or tau < 1:
        raise ValueError("m must be >= 2 and tau must be >= 1")
    span = (m - 1) * tau
    n_windows = len(values) - span
    if n_windows < 1:
        return None
    import math as _math
    n_patterns_possible = _math.factorial(m)
    if n_windows < n_patterns_possible + 1:
        return None

    counts: dict[tuple[int, ...], int] = {}
    for start in range(n_windows):
        window = [values[start + k * tau] for k in range(m)]
        pat = ordinal_pattern(window)
        counts[pat] = counts.get(pat, 0) + 1

    total = sum(counts.values())
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p)
    return h / math.log(n_patterns_possible)


def rolling_permutation_entropy(returns: Sequence[float], window: int,
                                 m: int = 3, tau: int = 1
                                 ) -> list[float | None]:
    """Trailing-window normalized permutation entropy at every index of
    `returns`, None until index `window - 1` (matches rolling_ar1/
    rolling_variance's own left-alignment convention in the CSD probe —
    entry i uses returns[i-window+1 .. i], never future data, so this is
    safe to read at decision time with no lookahead)."""
    out: list[float | None] = [None] * len(returns)
    for i in range(window - 1, len(returns)):
        out[i] = permutation_entropy(returns[i - window + 1: i + 1], m=m, tau=tau)
    return out


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def pearson_correlation(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Plain Pearson r over paired (a[i], b[i]) with neither None — used
    only as a descriptive independence check against AR1 (see module
    docstring), never as the gate-2 statistic itself. None if fewer than
    3 paired points or either series has zero variance."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def compute_entropy_lead_signal(returns: Sequence[float], onsets: list[dict],
                                 window: int = 60, m: int = 3, tau: int = 1,
                                 lead_offsets: Sequence[int] = (20, 10, 5, 1),
                                 rng_seed: int = 1337,
                                 regime_at: Sequence[str] | None = None
                                 ) -> dict:
    """Onset-vs-regime-matched-control comparison for rolling permutation
    entropy, structurally identical to critical_slowing_down_probe.py's
    own compute_lead_signal (same MIN_ONSETS_FOR_STATS floor, same
    regime-matched-control-with-fallback mechanism and the same
    control_regime_matched honesty flag, same fixed-seed reproducible
    control sampling) — deliberately mirrored rather than re-designed, so
    this entry's result is comparable apples-to-apples against CSD's
    already-published one instead of differing by incidental methodology
    choices."""
    ent = rolling_permutation_entropy(returns, window, m=m, tau=tau)
    onset_idx = {o["index"] for o in onsets}
    valid_pool = [i for i in range(len(returns))
                  if ent[i] is not None and i not in onset_idx]

    rng = random.Random(rng_seed)
    result: dict = {"n_onsets": len(onsets), "window": window, "m": m, "tau": tau,
                     "by_lead_days": {},
                     "regime_matched_control_requested": regime_at is not None}
    if len(onsets) < MIN_ONSETS_FOR_STATS:
        result["insufficient_n"] = True
        result["note"] = (f"only {len(onsets)} qualifying onsets found "
                           f"(<{MIN_ONSETS_FOR_STATS}) — no comparison "
                           "computed; per this repo's n>=5 reporting floor "
                           "this is reported as insufficient, not as a "
                           "null/negative result")
        return result

    for lead in lead_offsets:
        onset_ent, from_regimes = [], set()
        for o in onsets:
            j = o["index"] - lead
            if 0 <= j < len(returns) and ent[j] is not None:
                onset_ent.append(ent[j])
                from_regimes.add(o.get("from"))
        if len(onset_ent) < MIN_ONSETS_FOR_STATS:
            result["by_lead_days"][lead] = {
                "insufficient_n": True, "n": len(onset_ent)}
            continue

        pool = valid_pool
        matched = False
        if regime_at is not None:
            regime_pool = [i for i in valid_pool
                           if i < len(regime_at) and regime_at[i] in from_regimes]
            if len(regime_pool) >= max(MIN_ONSETS_FOR_STATS, len(onset_ent)) * 5:
                pool = regime_pool
                matched = True

        control_idx = rng.sample(pool, min(len(pool), len(onset_ent) * 20))
        control_ent = [ent[i] for i in control_idx]
        result["by_lead_days"][lead] = {
            "n_onset": len(onset_ent),
            "n_control": len(control_ent),
            "control_regime_matched": matched,
            "control_from_regimes": sorted(from_regimes),
            "onset_mean_entropy": _mean(onset_ent),
            "control_mean_entropy": _mean(control_ent),
        }
    return result


def _load_csd_module():
    """Loads critical_slowing_down_probe.py by path for find_transition_
    onsets and rolling_ar1 reuse (EDGE DOCTRINE #3 — don't re-derive
    already-built onset detection; also needed for this probe's own
    AR1-independence correlation check, see module docstring). Same
    boilerplate hazard_rate_probe.py's and omori_aftershock_probe.py's own
    _load_csd_module() already established — not re-derived here."""
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "critical_slowing_down_probe",
        os.path.join(os.path.dirname(__file__), "critical_slowing_down_probe.py"))
    csd = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(csd)
    return csd


def run_probe(days: int = 2520, window: int = 60, m: int = 3, tau: int = 1,
              lead_offsets: Sequence[int] = (20, 10, 5, 1),
              ticker: str = "SPY") -> dict:
    """Orchestrates the full gate-2 probe against REAL data. Requires
    network/Alpaca access this sandbox did not have — see module
    docstring. Reuses backtest_v2.fetch_bars/regime_series and the CSD
    entry's find_transition_onsets/rolling_ar1 verbatim (EDGE DOCTRINE #3:
    don't re-derive already-verified data plumbing or onset detection).

    `window` defaults to 60 trading days (a trading quarter), not CSD's
    20-day AR1/variance window: permutation entropy with m=3 has 6
    possible ordinal patterns, and a 20-day window yields only ~18
    extractable length-3 subsequences (~3 per pattern on average) — too
    thin to estimate a 6-outcome distribution with any stability. A
    60-day window yields ~58 subsequences (~9-10 per pattern), still not
    generous but the smallest window this probe considers honest; see
    permutation_entropy()'s own `n_windows < n_patterns_possible + 1`
    floor for the hard cutoff below which this returns None rather than a
    number computed from too few samples."""
    import backtest_v2 as bt

    csd = _load_csd_module()

    primary = bt.fetch_bars(ticker, days)
    vxx = bt.fetch_bars("VXX", days)
    labels, quality = bt.regime_series(primary, vxx)
    returns = log_returns(primary["close"])
    aligned_labels = labels[1:]
    onsets = csd.find_transition_onsets(aligned_labels)
    signal = compute_entropy_lead_signal(returns, onsets, window=window, m=m,
                                          tau=tau, lead_offsets=lead_offsets,
                                          regime_at=aligned_labels)

    ar1 = csd.rolling_ar1(returns, 20)
    entropy_series = rolling_permutation_entropy(returns, window, m=m, tau=tau)
    ar1_entropy_correlation = pearson_correlation(ar1, entropy_series)

    return {
        "ticker": ticker,
        "vxx_data_quality": quality,
        "n_days": len(primary["date"]),
        "date_range": [primary["date"][0], primary["date"][-1]] if primary["date"] else [],
        "onsets": onsets,
        "signal": signal,
        "ar1_entropy_correlation": ar1_entropy_correlation,
        "ar1_entropy_correlation_note": (
            "descriptive Pearson r between rolling_ar1(returns, 20) and "
            "rolling_permutation_entropy(returns, window) over their full "
            "overlapping valid range — a strong |r| would mean this "
            "import's signal, even if it clears gate 2 on its own, may not "
            "carry information beyond what the already-tested CSD AR1 "
            "statistic already provides; this is NOT itself a gate-2 test, "
            "just an honesty check against re-discovering the same effect "
            "under a new name (module docstring PRIOR section)"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=2520,
                     help="calendar days of SPY/VXX history to pull (default ~10y)")
    ap.add_argument("--ticker", default="SPY",
                     help="primary ticker whose own regime onsets are measured")
    ap.add_argument("--window", type=int, default=60,
                     help="rolling entropy window in trading days (default 60)")
    ap.add_argument("--m", type=int, default=3,
                     help="ordinal pattern embedding dimension (default 3, i.e. 6 patterns)")
    args = ap.parse_args()
    try:
        out = run_probe(days=args.days, ticker=args.ticker, window=args.window, m=args.m)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(out, indent=2, default=str))
