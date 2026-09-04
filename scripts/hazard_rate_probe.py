#!/usr/bin/env python3
"""
Renewal-process hazard-rate probe: does "time since the last regime-severity
failure" carry predictive information, the way it does for a mechanical
component's maintenance schedule?

FOREIGN-FIELD IMPORT (EDGE DOCTRINE #4, CLAUDE.md) from reliability
engineering / aviation maintenance: Barlow & Proschan, "Mathematical Theory
of Reliability" (1965) — the standard "bathtub curve" hazard-rate model used
to schedule aircraft-component overhauls. A renewal process's failure
history is memoryless (constant hazard, exponential inter-failure gaps) only
if nothing about elapsed time-since-last-failure predicts the next one. Real
mechanical systems deviate from that in two directions: DFR (decreasing
failure rate — infant mortality, failures cluster right after the last one)
and IFR (increasing failure rate — wear-out, a component overdue for
maintenance is MORE likely to fail soon, not equally likely). The standard
first diagnostic for which regime a renewal process is in is the coefficient
of variation (CV = std/mean) of its inter-failure gaps: CV ~= 1 is
consistent with memoryless/exponential, CV < 1 with more-regular-than-random
spacing (IFR/wear-out — an "overdue" signature), CV > 1 with burstier-than-
random spacing (DFR/clustering).

This is a DIFFERENT statistical technique from the two prior foreign-field
imports on file: the 2026-08-18/21 ecology entry (critical slowing down)
measured a single index's own rolling autocorrelation/variance ahead of a
transition (a within-series diagnostic on the approach to failure); the
2026-08-26 epidemiology entry (R_t) measured cross-sectional breakdown
counts across a peer basket (a contagion-rate diagnostic). This one treats
the sequence of failures ITSELF as a point process and asks whether its
timing is memoryless, aging, or clustering — a question neither prior import
asked.

"FAILURE" here reuses the exact onset definition the 2026-08-18/21 CSD entry
already built and unit-tested (`find_transition_onsets` in
critical_slowing_down_probe.py: a regime-severity jump, per
regime_util.classify_regime_5level, that persists >=3 days after >=20 days
of prior stability) — deliberately not re-derived, per EDGE DOCTRINE #3.

HYPOTHESIS (pre-registered, see research/open_questions.md for the full
ladder entry): the coefficient of variation of SPY regime-severity onset
gaps (trading days between consecutive onsets) is measurably below 1 —
i.e. onsets are spaced MORE REGULARLY than a memoryless process would
predict, an aviation-maintenance "overdue for an incident" signature that
would make elapsed calm-duration itself a usable early-warning input,
independent of any volatility/autocorrelation-based signal.

PRIOR (stated before ever running this against real data): known volatility
clustering already means failures often follow failures in quick
succession — that is a DFR (bursty, CV > 1) signature, the OPPOSITE of this
hypothesis. My prior is CV > 1 (bursty), not < 1. Finding CV < 1 would be
the genuinely novel, actionable result (an "unusually long quiet stretch is
itself elevated risk" signal, distinct from anything else in this
codebase). Finding CV >= 1 is a clean negative for the specific "the market
is overdue for a correction because it's been calm a long time" heuristic —
a common trader belief worth explicitly testing and, if false, recording as
false (REASONING STANDARD #10), not silently assumed either way.

Given the small number of onset events over any single-ticker archive (the
CSD entry found n=7 for SPY 2019-2026), REASONING STANDARD #4 applies: a
6-gap sample is underpowered for a point estimate alone, so this probe also
reports a seeded bootstrap range on the CV rather than a bare number, and a
secondary day-level "hazard by duration-since-last-onset bucket" breakdown
is reported as descriptive only, explicitly flagged where any bucket falls
below a minimum day-count floor.

LADDER PATH: gate-2 SIGNAL test (statistical predictive power, no trading),
built directly on regime labels that are already gate-1-verified elsewhere
in this codebase (backtest_v2.fetch_bars/regime_series, the same plumbing
the CSD entry reused) — no new gate-1 needed.

WHY THIS SESSION DID NOT RUN IT AGAINST REAL DATA: this sandbox has neither
ALPACA_KEY/ALPACA_SECRET in its environment nor working network access to
Yahoo Finance (query1.finance.yahoo.com returned HTTP 429 this session; no
pre-existing .bt_cache/bt2_SPY_*.json either) — the identical constraint the
CSD entry hit at first pass. The statistical core below is pure and unit-
tested against synthetic data in test_hazard_rate_probe.py, so a future
session with real data access can call run_probe() directly.

USAGE (future session, once ALPACA_KEY/SECRET or working Yahoo access
exists):
    python3 scripts/hazard_rate_probe.py [--days 2520] [--horizon 10]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Sequence

MIN_GAPS_FOR_STATS = 5  # same n>=5 reporting floor used elsewhere in this repo
DEFAULT_BUCKET_EDGES = (5, 10, 20, 40, 80)
DEFAULT_MIN_DAYS_PER_BUCKET = 30


def inter_onset_gaps(onset_indices: Sequence[int]) -> list[int]:
    """Trading-day gaps between consecutive onsets (sorted). [] if fewer
    than 2 onsets exist — no gap can be measured yet."""
    idx = sorted(onset_indices)
    return [idx[i] - idx[i - 1] for i in range(1, len(idx))]


def gap_cv(gaps: Sequence[float]) -> float | None:
    """Coefficient of variation (population std / mean) of renewal gaps —
    the standard reliability-engineering diagnostic for whether a failure
    process is memoryless (CV ~= 1), aging/wear-out (CV < 1), or clustering
    (CV > 1). None if fewer than 2 gaps or the mean is zero (degenerate,
    can't estimate a meaningful ratio)."""
    n = len(gaps)
    if n < 2:
        return None
    m = sum(gaps) / n
    if m == 0:
        return None
    var = sum((g - m) ** 2 for g in gaps) / n
    return math.sqrt(var) / m


def bootstrap_cv_range(gaps: Sequence[float], n_boot: int = 2000,
                        rng_seed: int = 1337,
                        pct: tuple[float, float] = (0.1, 0.9)) -> dict | None:
    """Seeded bootstrap resample of `gaps` to give an honest sense of
    estimation uncertainty on gap_cv() at small n (REASONING STANDARD #4).
    None if fewer than 2 gaps (gap_cv itself would already be None)."""
    if len(gaps) < 2:
        return None
    rng = random.Random(rng_seed)
    vals = []
    for _ in range(n_boot):
        sample = [gaps[rng.randrange(len(gaps))] for _ in range(len(gaps))]
        cv = gap_cv(sample)
        if cv is not None:
            vals.append(cv)
    if not vals:
        return None
    vals.sort()
    lo_i = int(pct[0] * len(vals))
    hi_i = min(int(pct[1] * len(vals)), len(vals) - 1)
    return {"lo": vals[lo_i], "hi": vals[hi_i], "n_boot_valid": len(vals)}


def duration_since_last_onset(onset_indices: Sequence[int],
                               n: int) -> list[int | None]:
    """Trading days elapsed since the most recent onset at or before index
    t (0 if t is itself an onset day). None before the first onset in the
    series — there is no "last failure" to measure a duration from yet."""
    idx = sorted(onset_indices)
    out: list[int | None] = [None] * n
    last = None
    j = 0
    for t in range(n):
        while j < len(idx) and idx[j] <= t:
            last = idx[j]
            j += 1
        out[t] = None if last is None else t - last
    return out


def forward_onset_within(onset_indices: Sequence[int], n: int,
                          horizon: int) -> list[bool]:
    """out[t] = True iff an onset occurs strictly after t, within the next
    `horizon` trading days. Backward-looking duration paired with this
    forward-looking flag is the same no-lookahead shape as the CSD entry's
    lead-offset comparison: duration(t) uses only data up to and including
    t, forward_onset_within(t) is the (out-of-sample-at-t) label."""
    idx = set(onset_indices)
    out = [False] * n
    for t in range(n):
        out[t] = any((t + k) in idx for k in range(1, horizon + 1))
    return out


def bucket_hazard(durations: Sequence[int | None], forward_flags: Sequence[bool],
                   bucket_edges: Sequence[int] = DEFAULT_BUCKET_EDGES,
                   min_days: int = DEFAULT_MIN_DAYS_PER_BUCKET) -> list[dict]:
    """Buckets days by duration-since-last-onset into half-open bins
    [0,e0),[e0,e1),...,[e_last,inf) and reports the empirical fraction of
    days in each bucket for which forward_onset_within is True (a discrete-
    time hazard estimate). Days with duration=None (before the first onset)
    are excluded. Buckets under `min_days` are flagged insufficient rather
    than reporting a noisy ratio as if it were reliable."""
    edges = list(bucket_edges) + [None]
    bounds = []
    lo = 0
    for hi in edges:
        bounds.append((lo, hi))
        lo = hi

    out = []
    for lo, hi in bounds:
        flags = [f for d, f in zip(durations, forward_flags)
                  if d is not None and d >= lo and (hi is None or d < hi)]
        n = len(flags)
        entry: dict = {"range": [lo, hi], "n_days": n}
        if n < min_days:
            entry["insufficient_n"] = True
        else:
            entry["hazard"] = sum(flags) / n
        out.append(entry)
    return out


def _load_csd_module():
    """Loads critical_slowing_down_probe.py by path for find_transition_
    onsets reuse (EDGE DOCTRINE #3 — don't re-derive already-built onset
    detection). Factored out of run_probe so run_pooled_probe below can
    share the same import-by-path boilerplate instead of duplicating it."""
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "critical_slowing_down_probe",
        os.path.join(os.path.dirname(__file__), "critical_slowing_down_probe.py"))
    csd = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(csd)
    return csd


def run_probe(days: int = 2520, horizon: int = 10, ticker: str = "SPY",
              bucket_edges: Sequence[int] = DEFAULT_BUCKET_EDGES) -> dict:
    """Orchestrates the full gate-2 probe against REAL data. Requires
    network/Alpaca access this sandbox did not have — see module
    docstring. Reuses backtest_v2.fetch_bars/regime_series and the CSD
    entry's find_transition_onsets verbatim (EDGE DOCTRINE #3: don't
    re-derive already-verified data plumbing or onset detection).

    `ticker` is the primary equity series whose own regime-severity onsets
    are measured (spy_vs_ma50/ma200 computed from ITS OWN close series,
    same as the original SPY-only run); VXX stays the shared market-wide
    volatility backdrop regardless of `ticker`, since VXX is a broad-market
    vol index, not SPY-specific — this is the additive broader-universe
    follow-up the 2026-08-31 GATE 2 provisional-positive entry named
    (`research/open_questions.md`), not a re-derivation: every downstream
    function here (`inter_onset_gaps`/`gap_cv`/the bucket machinery) already
    took an onset-index list, not a hardcoded ticker."""
    import backtest_v2 as bt

    csd = _load_csd_module()

    primary = bt.fetch_bars(ticker, days)
    vxx = bt.fetch_bars("VXX", days)
    labels, quality = bt.regime_series(primary, vxx)
    n = len(labels)

    onsets = csd.find_transition_onsets(labels)
    onset_idx = [o["index"] for o in onsets]

    gaps = inter_onset_gaps(onset_idx)
    cv = gap_cv(gaps)
    cv_range = bootstrap_cv_range(gaps)

    durations = duration_since_last_onset(onset_idx, n)
    forward = forward_onset_within(onset_idx, n, horizon)
    buckets = bucket_hazard(durations, forward, bucket_edges)
    base_rate = (sum(forward) / n) if n else None

    return {
        "ticker": ticker,
        "vxx_data_quality": quality,
        "n_days": n,
        "date_range": [primary["date"][0], primary["date"][-1]] if primary["date"] else [],
        "n_onsets": len(onsets),
        "gaps_trading_days": gaps,
        "gap_cv": cv,
        "gap_cv_insufficient_n": len(gaps) < MIN_GAPS_FOR_STATS,
        "gap_cv_bootstrap_range": cv_range,
        "base_rate_onset_within_horizon": base_rate,
        "hazard_by_duration_bucket": buckets,
        "horizon": horizon,
    }


# ---------------------------------------------------------------------------
# POOLED IDIOSYNCRATIC-ONSET FOLLOW-UP (2026-08-31 GATE 2 entry's own named
# NEXT (ii), open_questions.md — queued and unclaimed until this session).
#
# The 2026-08-31 broader-universe run found SPY/QQQ/AAPL/MSFT's onset DATES
# were mostly byte-identical: classify_regime_5level's PANIC/BEAR/CAUTION
# branches fire on VXX alone, before any ticker-specific term, so testing
# five tickers through that classifier was much closer to re-running the
# same VXX series five times than to five independent trials. Only AAPL and
# MSFT contributed one genuinely idiosyncratic onset each (their own price
# series broke a boundary VXX alone did not trigger) — real signal, but at
# n<=2 per ticker, far too underpowered on its own (MIN_GAPS_FOR_STATS=5).
#
# This section implements that entry's own follow-up (ii): strip each
# ticker's SPY-shared onsets, then POOL the idiosyncratic remainder across
# many tickers into one combined renewal-process gap series, the only way
# to reach a usable n without re-coupling through the shared VXX backdrop
# the filter exists to remove.
# ---------------------------------------------------------------------------

def onset_dates(onsets: Sequence[dict], dates: Sequence[str]) -> list[str]:
    """Maps onset bar-indices (from find_transition_onsets) to calendar-date
    strings via THAT ticker's own date array. Indices are bar positions
    into one ticker's own bars and are never assumed comparable across
    tickers with a potentially different trading-day array (a listing gap,
    a different fetch window) — this always looks the date up rather than
    reusing another ticker's index against a different array."""
    return [dates[o["index"]] for o in onsets if 0 <= o["index"] < len(dates)]


def idiosyncratic_onset_dates(ticker_dates: Sequence[str],
                               reference_dates: Sequence[str]) -> list[str]:
    """Keeps only the onset dates NOT also present in the reference
    ticker's (SPY's) own onset-date set — see the module-level note above
    for why: a shared VXX-driven onset date, counted once per ticker,
    would silently multiply one market-wide event into N 'independent'
    onsets. What survives is the part of each ticker's onset history
    attributable to its OWN price series, not the shared vol backdrop."""
    ref = set(reference_dates)
    return [d for d in ticker_dates if d not in ref]


def pool_idiosyncratic_onsets(per_ticker_dates: dict[str, Sequence[str]],
                               min_days_apart: int = 5) -> dict:
    """Pools idiosyncratic onset dates from MANY tickers into one combined
    chronological renewal-process gap series.

    UNRESOLVED JUDGMENT CALL, stated rather than hidden (REASONING
    STANDARD #7/#10): two DIFFERENT tickers going idiosyncratic within a
    few days of each other could be two genuinely independent renewals, or
    one correlated shock landing on both (e.g. two names in the same
    sector). `min_days_apart` drops any pooled date within that many
    calendar days of an already-kept date (earliest kept, later dropped)
    as a likely-correlated duplicate rather than counting it as a second
    independent renewal — a real methodology choice that trades away some
    genuine independent signal to avoid inflating n with correlated
    near-duplicates. `n_dropped_as_correlated_duplicate` is always
    reported so a future reader can see how much this cost;
    `min_days_apart=0` recovers the undeduplicated pool for comparison.

    Gaps here are CALENDAR days between pooled dates, not trading-day bar
    indices like inter_onset_gaps() elsewhere in this module — pooled
    dates come from different tickers' own bar arrays, not guaranteed to
    share one common index space, so calendar-day subtraction on the date
    strings themselves is the only sound common ground. A real (if usually
    0-3-day) numeric difference from the trading-day convention used
    elsewhere in this file, stated explicitly rather than silently mixed
    with it."""
    from datetime import date as _date

    all_dated = sorted(
        (d, tkr) for tkr, dates in per_ticker_dates.items() for d in dates
    )
    kept: list[tuple[str, str]] = []
    dropped_dup = 0
    last_kept: "_date | None" = None
    for d, tkr in all_dated:
        dt = _date.fromisoformat(d)
        if last_kept is not None and (dt - last_kept).days < min_days_apart:
            dropped_dup += 1
            continue
        kept.append((d, tkr))
        last_kept = dt

    gaps = [
        (_date.fromisoformat(kept[i][0]) - _date.fromisoformat(kept[i - 1][0])).days
        for i in range(1, len(kept))
    ]
    return {
        "pooled_dates": [d for d, _ in kept],
        "pooled_tickers": [t for _, t in kept],
        "n_pooled_onsets": len(kept),
        "n_dropped_as_correlated_duplicate": dropped_dup,
        "min_days_apart": min_days_apart,
        "gaps_calendar_days": gaps,
        "gap_cv": gap_cv(gaps),
        "gap_cv_insufficient_n": len(gaps) < MIN_GAPS_FOR_STATS,
        "gap_cv_bootstrap_range": bootstrap_cv_range(gaps),
    }


def run_pooled_probe(tickers: Sequence[str] = ("QQQ", "AAPL", "MSFT", "IWM"),
                      reference_ticker: str = "SPY", days: int = 2520,
                      min_days_apart: int = 5) -> dict:
    """Orchestrates the 2026-08-31 entry's own named follow-up (ii) against
    REAL data. Requires network/Alpaca access this sandbox did not have at
    build time — see module docstring; a future session with real data
    access can call this directly or via `--pool`. Reuses fetch_bars/
    regime_series/find_transition_onsets verbatim, same as run_probe
    (EDGE DOCTRINE #3)."""
    import backtest_v2 as bt
    csd = _load_csd_module()

    vxx = bt.fetch_bars("VXX", days)

    def _onset_dates_for(tkr: str) -> tuple[list[str], dict]:
        primary = bt.fetch_bars(tkr, days)
        labels, quality = bt.regime_series(primary, vxx)
        onsets = csd.find_transition_onsets(labels)
        meta = {
            "ticker": tkr, "vxx_data_quality": quality,
            "n_days": len(labels),
            "date_range": [primary["date"][0], primary["date"][-1]] if primary["date"] else [],
        }
        return onset_dates(onsets, primary["date"]), meta

    ref_dates, ref_meta = _onset_dates_for(reference_ticker)
    ref_meta["n_onsets_total"] = len(ref_dates)
    ref_meta["n_onsets_idiosyncratic"] = None  # the reference DEFINES "shared"; it has no idiosyncratic count of its own

    per_ticker_idio: dict[str, list[str]] = {}
    per_ticker_meta: dict[str, dict] = {reference_ticker: ref_meta}
    for tkr in tickers:
        dates, meta = _onset_dates_for(tkr)
        idio = idiosyncratic_onset_dates(dates, ref_dates)
        meta["n_onsets_total"] = len(dates)
        meta["n_onsets_idiosyncratic"] = len(idio)
        per_ticker_idio[tkr] = idio
        per_ticker_meta[tkr] = meta

    pooled = pool_idiosyncratic_onsets(per_ticker_idio, min_days_apart=min_days_apart)

    return {
        "reference_ticker": reference_ticker,
        "tickers": list(tickers),
        "per_ticker": per_ticker_meta,
        "pooled": pooled,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=2520,
                     help="calendar days of history to pull (default ~10y)")
    ap.add_argument("--horizon", type=int, default=10,
                     help="trading-day forward window defining 'a new onset soon' (single-ticker mode only)")
    ap.add_argument("--ticker", type=str, default="SPY",
                     help="primary equity series to measure onsets on in single-ticker mode; the REFERENCE ticker in --pool mode (VXX stays the shared vol backdrop either way)")
    ap.add_argument("--pool", action="store_true",
                     help="run the pooled-idiosyncratic-onset follow-up (2026-08-31 NEXT (ii)) across --tickers instead of the single-ticker probe")
    ap.add_argument("--tickers", type=str, default="QQQ,AAPL,MSFT,IWM",
                     help="comma-separated tickers pooled against --ticker as the reference (--pool mode only)")
    ap.add_argument("--min-days-apart", type=int, default=5,
                     help="drop a pooled onset within this many calendar days of an already-kept one, as a likely-correlated duplicate (--pool mode only)")
    args = ap.parse_args()
    try:
        if args.pool:
            out = run_pooled_probe(
                tickers=[t.strip() for t in args.tickers.split(",") if t.strip()],
                reference_ticker=args.ticker, days=args.days,
                min_days_apart=args.min_days_apart)
        else:
            out = run_probe(days=args.days, horizon=args.horizon, ticker=args.ticker)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(out, indent=2, default=str))
