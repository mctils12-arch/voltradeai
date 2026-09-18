#!/usr/bin/env python3
"""
Insider-flow CUSUM probe: does a formally-detected, sustained shift in
MARKET-WIDE aggregate insider buying/selling sentiment lead a corresponding
shift in forward broad-market (SPY) returns?

FOREIGN-FIELD IMPORT (EDGE DOCTRINE #4, CLAUDE.md) from statistical process
control / industrial quality engineering — a SEVENTH field, not yet used by
any prior import in this file (ecology/critical-slowing-down 2026-08-18,
epidemiology/R_t 2026-08-26, reliability-engineering/hazard-rate 2026-08-29,
seismology/Omori-Utsu 2026-08-30, information-theory/permutation-entropy
2026-09-05, hydrology/Hurst-exponent 2026-09-12). Walter Shewhart (1924) and
E.S. Page ("Continuous Inspection Schemes", Biometrika 1954) developed
control-chart theory to answer a factory-floor question: given a stream of
noisy measurements off a production line, what is the EARLIEST moment you
can be confident the process mean has genuinely shifted, without either
false-alarming on ordinary noise or waiting so long that defective units
pile up? Page's CUSUM (cumulative sum) chart is the classical answer for
detecting SMALL, SUSTAINED shifts specifically — it is provably faster than
a simple moving-average or single-sample threshold test at that job (its
whole design point is an average-run-length guarantee), which is why it is
standard in semiconductor fabs and pharma manufacturing to this day.

WHY THIS IS A GENUINELY DIFFERENT DESIGN, not an eighth variant of the same
fished idea (REASONING STANDARD #4): every prior import in this family
either (a) counted rare discrete "onset" events against a control sample
(critical-slowing-down, R_t, hazard-rate, Omori-Utsu — all hit the same
n~10-15 statistical-power ceiling), or (b) computed a windowed/rolling
descriptive statistic and correlated it against forward returns
(permutation-entropy, Hurst — both landed on "not significant once
correctly de-strided"). CUSUM is neither: it is a SEQUENTIAL, path-dependent
accumulator with a formal decision boundary derived from control-chart
theory, not a windowed re-estimate of a single-sample statistic. This probe
uses it in the continuous-score shape the 2026-09-05/09-12 sessions'
own STRUCTURAL META-FINDING recommended (score the accumulator itself
against forward returns every day, never reduce to a handful of onset
events), so it inherits neither failure mode by construction — though that
alone is no guarantee of a real effect; it only removes two specific,
already-diagnosed reasons for a false negative.

NON-PRICE DATA SOURCE (per the 2026-09-12/09-13 Hurst sessions' own filed
NEXT — the two REMAINING untried axes after six price-only imports were
"(a) intraday structure and (b) a non-price data source"; this picks (b)):
`sec_form4_bulk.py`'s already-archived, already-gate-1-passed SEC Form 4
bulk dataset (officer/director open-market P/S transactions, FILING_DATE-
keyed, no lookahead — the exact fields and no-lookahead convention
`form4_gate2_test.py` already established, reused here via `filing_date`
rather than `trans_date`, EDGE DOCTRINE #3). IMPORTANT DISTINCTION from
the existing Form-4 work in this file: `form4_gate2_test.py` tests a
PER-TICKER, PER-EVENT question ("does THIS insider's buy predict THIS
stock's forward return?") and that exact question was GATE 2 KILLED
2026-07-22 (no 20d separation, significant negative 60d — a momentum/
timing-at-extremes confound). This probe asks a DIFFERENT, market-wide
BREADTH question instead ("does the ADD-UP of every officer/director's
trades, market-wide, shifting from net-selling to net-buying or vice
versa, lead the BROAD market?") — the same relationship type
`wikimedia_pageviews_attention`'s already-gate-2-passed design uses
(an aggregate flow/attention statistic vs. a broad forward-return target,
not a single-name event study). The 2026-07-22 kill is real evidence this
probe's prior should be skeptical of insider trades carrying individual
alpha, but it does not settle the aggregate-breadth question, which is a
structurally distinct claim (a market-wide sentiment/positioning signal,
not an individual-stock-picking signal) — REASONING STANDARD #4 still
applies: this is not a free pass because of the different framing, only a
non-duplicate one.

HYPOTHESIS (pre-registered BEFORE running against real data, REASONING
STANDARD #10): let flow_t = market-wide net officer/director dollar flow
(sum of P transaction dollar_value minus sum of S transaction dollar_value)
on trading day t, keyed by FILING_DATE only (never trans_date — insiders
have up to 2 business days to file, so trans_date-keying would leak
information the market could not yet have acted on). Standardize into
z_t (rolling z-score over a trailing window) and run a two-sided Page CUSUM
over z_t to get cusum_t = C+_t - C-_t (signed cumulative deviation).
PREDICTION: cusum_t is POSITIVELY correlated with continuation_score_t =
forward H-day SPY log return starting at t+1 — a sustained run of
above-average net insider BUYING (cusum_t > 0 and growing) precedes
better-than-average forward market returns, and a sustained run of net
insider SELLING (cusum_t < 0) precedes worse-than-average forward returns.

PRIOR, stated honestly before computing anything (REASONING STANDARD #10
and #5): weak-to-skeptical. Three separate reasons to expect a small or
null effect, not zero but discounted: (1) this file's own 2026-07-22 kill
of the per-ticker version is evidence AGAINST insider transactions
carrying much exploitable information content at all in this dataset,
even though the aggregate framing is a different claim; (2) REASONING
STANDARD #5 (second-order thinking) — aggregate insider sentiment as a
market-timing signal is itself an old, well-studied idea in the academic
literature (Seyhun 1986, 1988 finds SOME aggregate predictive content for
insider purchases specifically, but the effect there is measured in
MONTHS, not the 5-60 trading-day horizons this codebase's other GATE 2
tests use, and Seyhun's own later work finds the edge has decayed since
wider disclosure/EDGAR availability made the data cheap to watch — exactly
the "faster/bigger players arbitrage it away once it's cheap to see"
mechanism this standard asks about); (3) six prior foreign-field imports
in this exact file are already GATE-2-killed or unresolved, and REASONING
STANDARD #4 says each additional untested idea should be discounted
further by that track record. Genuinely open, and the reason this is
still worth one clean run: the CUSUM design and the aggregate-breadth
framing are both new (never tried in this family), and a market-wide
sentiment signal tested against SPY itself (not a single stock) sidesteps
the specific momentum/timing-at-extremes confound the 2026-07-22 kill
diagnosed for individual names.

LADDER PATH: this is a GATE 2 (SIGNAL) test only — pure statistical
predictive power, no trading logic, no sizing, and nothing wired into
`deep_score`/tier decisions regardless of outcome. A future GATE 3
(backtested entry/exit ablation) would only be warranted if GATE 2 clears
with an effect size + significance a human would find persuasive net of
REASONING STANDARD #4's now-substantial multiple-testing discount (this is
the SEVENTH foreign-field import attempted in this file).

MEASUREMENT INTEGRITY note (this is a research probe, not measurement code
CLAUDE.md's own section governs, but its spirit applies): reports the null
result as plainly as a positive one; does not retry with different
k/h/window parameters after seeing an unfavorable result (REASONING
STANDARD #4 — one theory-motivated spec, run once, reported honestly). The
CUSUM slack (k) and decision threshold (h) below are the textbook default
choices for detecting a ~1-sigma sustained shift at a conventional false-
alarm rate (Montgomery, "Introduction to Statistical Quality Control"),
chosen BEFORE seeing any real output, not tuned to it.

SANDBOX LIMITATION, stated honestly: this sandbox has no archived Form 4
quarters on disk (`sec_form4_bulk.archived_quarters()` returns `[]` here —
the real archive lives on the Railway volume in production, built
incrementally by the live bot's Tier 3 hourly call). `run_probe()` detects
this and returns a plain `{"error": ...}` rather than fabricating a result;
this session ships the probe built and unit-tested on synthetic data only,
matching the established precedent of several prior foreign-field imports
in this file (permutation-entropy, hazard-rate) that were "script built,
not yet run against real data" on their filing session. A FUTURE session
with production archive access should run this against the real data next.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest_v2  # noqa: E402
import sec_form4_bulk  # noqa: E402

MIN_PAIRS_FOR_STATS = 5

# Page's CUSUM defaults: k = half the shift size to detect, in standard-
# deviation units (the textbook choice for a ~1-sigma sustained shift);
# h = decision threshold, in standard-deviation units (h=5 gives a
# conventional in-control average run length in the hundreds — the classic
# Montgomery/Page recommendation, not tuned against this dataset).
DEFAULT_K = 0.5
DEFAULT_H = 5.0


def net_flow_by_filing_date(records: Sequence[dict]) -> dict[str, float]:
    """Market-wide net officer/director dollar flow per FILING_DATE (never
    trans_date — filing is the earliest date the market could have known,
    matching form4_gate2_test.py's own no-lookahead convention). P
    (open-market purchase) adds dollar_value, S (open-market sale)
    subtracts it. Multiple tickers/owners on the same filing date are
    summed into one market-wide number."""
    out: dict[str, float] = {}
    for r in records:
        code = r.get("trans_code")
        date = r.get("filing_date")
        value = r.get("dollar_value")
        if code not in ("P", "S") or not date or value is None:
            continue
        signed = value if code == "P" else -value
        out[date] = out.get(date, 0.0) + signed
    return out


def align_to_trading_days(flow_by_date: dict[str, float], trading_dates: Sequence[str]
                           ) -> list[float]:
    """Map a {calendar_date: net_flow} dict onto the SPY trading-day
    calendar (`trading_dates`, ascending ISO strings). A filing disclosed on
    a non-trading day (weekend, holiday, after-hours EDGAR processing) rolls
    FORWARD to the next available trading day — the market cannot act on it
    any earlier, and rolling forward (never backward) preserves no-lookahead.
    Any flow whose filing_date is after the last trading date in the
    calendar is dropped (nothing to roll onto). Returns one float per
    trading day, 0.0 for days with no filings."""
    if not trading_dates:
        return []
    out = [0.0] * len(trading_dates)
    n = len(trading_dates)
    for date, value in flow_by_date.items():
        idx = _first_index_on_or_after(trading_dates, date)
        if idx is None or idx >= n:
            continue
        out[idx] += value
    return out


def _first_index_on_or_after(sorted_dates: Sequence[str], target: str) -> int | None:
    lo, hi = 0, len(sorted_dates)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_dates[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(sorted_dates) else None


def rolling_zscore(values: Sequence[float], window: int = 60) -> list[float | None]:
    """z_t computed ONLY from values[t-window+1..t] (never future data — no
    lookahead). None for the first `window`-1 entries and any window with
    zero variance."""
    out: list[float | None] = [None] * len(values)
    for t in range(window - 1, len(values)):
        chunk = values[t - window + 1:t + 1]
        mean = sum(chunk) / len(chunk)
        var = sum((v - mean) ** 2 for v in chunk) / len(chunk)
        if var <= 0:
            continue
        std = math.sqrt(var)
        out[t] = (values[t] - mean) / std
    return out


def cusum(z: Sequence[float | None], k: float = DEFAULT_K) -> list[float | None]:
    """Two-sided Page's CUSUM, computed ONLY from z[0..t] at each step t (no
    lookahead by construction — a strictly sequential recurrence). Returns
    cusum_t = c_plus_t - c_minus_t (signed net cumulative deviation) for
    every index with a defined z_t, None otherwise. c_plus and c_minus reset
    to 0 whenever z_t is None (a gap in the standardized series), since the
    accumulator has no valid input to carry forward across it."""
    out: list[float | None] = [None] * len(z)
    c_plus = c_minus = 0.0
    for t, zt in enumerate(z):
        if zt is None:
            c_plus = c_minus = 0.0
            continue
        c_plus = max(0.0, c_plus + zt - k)
        c_minus = min(0.0, c_minus + zt + k)
        out[t] = c_plus + c_minus
    return out


def cusum_alarms(cusum_series: Sequence[float | None], h: float = DEFAULT_H) -> list[int]:
    """Classic textbook CUSUM alarm onsets (|cusum_t| crosses h from below),
    provided for completeness/comparison against the continuous design —
    per the 2026-09-05/09-12 sessions' own finding, this onset-counting view
    is expected to be underpowered (n~10-15 at most) and is NOT the primary
    analysis; see continuation_scores()/destrided_spearman() below for that."""
    alarms = []
    armed = True
    for t, c in enumerate(cusum_series):
        if c is None:
            armed = True
            continue
        if armed and abs(c) >= h:
            alarms.append(t)
            armed = False
        elif abs(c) < h:
            armed = True
    return alarms


def continuation_scores(returns: Sequence[float], cusum_series: Sequence[float | None],
                         horizon: int = 20) -> list[tuple[float, float]]:
    """For each valid index t (cusum_series[t] is not None, and a full
    horizon window after t exists), returns (cusum_t, forward_return) pairs.
    forward_return = sum of daily log returns over the horizon days starting
    at t+1 — no sign transform (unlike the Hurst probe's continuation_score,
    which multiplies by a trailing-trend sign): the pre-registered hypothesis
    here is a direct level prediction (more net buying -> higher forward
    return), not a trend-continuation claim."""
    out = []
    n = len(returns)
    for t in range(0, n - horizon):
        c = cusum_series[t] if t < len(cusum_series) else None
        if c is None:
            continue
        forward = sum(r for r in returns[t + 1:t + 1 + horizon] if r is not None)
        out.append((c, forward))
    return out


def _load_hurst_probe():
    """Reuses destrided_spearman()/spearman() from hurst_exponent_probe.py
    rather than reimplementing a third copy of the same significance-testing
    helpers (EDGE DOCTRINE #3)."""
    spec = importlib.util.spec_from_file_location(
        "hurst_exponent_probe",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hurst_exponent_probe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_probe(ticker: str = "SPY", days: int = 2520, zscore_window: int = 60,
              horizon: int = 20, k: float = DEFAULT_K, h: float = DEFAULT_H) -> dict:
    quarters = sec_form4_bulk.archived_quarters()
    if not quarters:
        return {"error": "no archived Form 4 quarters available in this sandbox",
                "note": "run against production's /data/voltrade archive instead"}

    records = sec_form4_bulk.load_all_records(quarters)
    if not records:
        return {"error": "archived Form 4 quarters present but contained zero P/S records",
                "quarters": quarters}

    bars = backtest_v2.fetch_bars(ticker, days)
    closes = bars.get("close", []) if bars else []
    dates = bars.get("date", []) if bars else []
    if len(closes) < zscore_window + horizon + 10:
        return {"error": "insufficient bars", "ticker": ticker, "n_bars": len(closes)}

    hurst_mod = _load_hurst_probe()
    rets = hurst_mod.log_returns(closes)

    flow_by_date = net_flow_by_filing_date(records)
    flow = align_to_trading_days(flow_by_date, dates)
    z = rolling_zscore(flow, window=zscore_window)
    cs = cusum(z, k=k)
    alarms = cusum_alarms(cs, h=h)
    pairs = continuation_scores(rets, cs, horizon=horizon)

    return {
        "ticker": ticker,
        "n_bars": len(closes),
        "date_range": [dates[0], dates[-1]] if dates else None,
        "quarters_used": quarters,
        "n_form4_records": len(records),
        "n_trading_days_with_flow": sum(1 for v in flow if v != 0.0),
        "zscore_window": zscore_window,
        "horizon": horizon,
        "cusum_k": k,
        "cusum_h": h,
        "n_cusum_alarms": len(alarms),
        "n_scored_pairs": len(pairs),
        "spearman_naive_daily": hurst_mod.spearman(pairs),
        "spearman_destrided": hurst_mod.destrided_spearman(pairs, stride=horizon),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--days", type=int, default=2520, help="~10y of trading days")
    ap.add_argument("--zscore-window", type=int, default=60, dest="zscore_window")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--h", type=float, default=DEFAULT_H)
    args = ap.parse_args()
    print(json.dumps(run_probe(args.ticker, args.days, args.zscore_window,
                                args.horizon, args.k, args.h), indent=2))
