#!/usr/bin/env python3
"""
Critical-slowing-down (CSD) early-warning-signal probe.

FOREIGN-FIELD IMPORT (EDGE DOCTRINE #4, CLAUDE.md) from ecology:
Scheffer et al. 2009, "Early-warning signals for critical transitions"
(Nature 461:53-59). As a dynamical system nears a tipping point, its
recovery rate from small perturbations slows down. Two statistics make
this visible BEFORE the transition itself, without knowing when it will
happen: rising lag-1 autocorrelation (AR1) and rising variance of the
system's fluctuations around its current state. A smaller finance
literature has applied the same diagnostic ahead of market regime shifts.

HYPOTHESIS (pre-registered, see research/open_questions.md for the full
ladder entry): rolling AR1 and rolling variance of SPY daily log returns,
each computed over a trailing 20-trading-day window, rise in the 10
trading days before a regime transition into a MORE SEVERE regime
(BULL/CAUTION -> NEUTRAL/BEAR/PANIC, severity order per
regime_util.classify_regime_5level) versus a control sample of
non-transition windows drawn from the same archive.

PRIOR (stated before ever running this against real data): expect
variance to rise before transitions — that's just volatility clustering,
already partially captured by the existing vxx_ratio classifier input,
not a novel finding on its own. The genuinely testable claim is whether
AR1 gives INDEPENDENT lead-time value beyond variance/vxx_ratio — i.e.
does it turn upward measurably earlier or more consistently than the
existing threshold classifier reacts? If AR1's rise is not measurably
earlier/stronger than variance's own rise, this import adds nothing
beyond what the classifier already captures — that is a valid, useful
negative result and must be recorded as such, not silently dropped
(REASONING STANDARD #10).

LADDER PATH: gate-2 SIGNAL test (statistical predictive power, no
trading) built directly on top of SPY/VXX daily bars that are already
gate-1-verified elsewhere in this codebase (backtest_v2.fetch_bars, the
existing Alpaca-first/Yahoo-fallback path) — no new gate-1 needed, same
precedent as other signals derived from already-verified price data.

WHY THIS SESSION DID NOT RUN IT AGAINST REAL DATA: this sandbox has
neither ALPACA_KEY/ALPACA_SECRET in its environment nor working network
access to Yahoo Finance (query1.finance.yahoo.com returned HTTP 429 on
repeated attempts this session; no pre-existing .bt_cache/bt2_SPY_*.json
either). The statistical core below (log_returns/rolling_ar1/
rolling_variance/find_transition_onsets) is pure and unit-tested against
synthetic data in test_critical_slowing_down_probe.py, so a future
session with real data access can call run_probe() directly with zero
additional design work — compile the reasoning now, don't re-derive it
later (EDGE DOCTRINE #3).

USAGE (future session, once ALPACA_KEY/SECRET or working Yahoo access
exists):
    python3 scripts/critical_slowing_down_probe.py [--days 2520]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Sequence

SEVERITY = {"BULL": 0, "NEUTRAL": 1, "CAUTION": 2, "BEAR": 3, "PANIC": 4}

MIN_ONSETS_FOR_STATS = 5  # same n>=5 reporting floor used elsewhere in this repo


def log_returns(closes: Sequence[float]) -> list[float]:
    """Daily log returns. First element of `closes` has no prior day, so
    the output is one shorter than the input (index i corresponds to the
    return realized ON `closes` index i+1)."""
    out = []
    for i in range(1, len(closes)):
        prev, cur = closes[i - 1], closes[i]
        if prev is None or cur is None or prev <= 0 or cur <= 0:
            out.append(0.0)
            continue
        out.append(math.log(cur / prev))
    return out


def rolling_ar1(returns: Sequence[float], window: int) -> list[float | None]:
    """Trailing lag-1 autocorrelation of `returns` over `window`-day windows.
    None until enough history exists. Standard Pearson correlation between
    r[t] and r[t-1] within the window; None (not 0.0) when the window's
    variance is degenerate (constant returns) so a flat synthetic series
    never masquerades as "no autocorrelation"."""
    out: list[float | None] = [None] * len(returns)
    for i in range(len(returns)):
        if i + 1 < window + 1:
            continue
        w = returns[i - window + 1: i + 1]
        x = w[:-1]
        y = w[1:]
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        cov = sum((x[j] - mx) * (y[j] - my) for j in range(n)) / n
        vx = sum((v - mx) ** 2 for v in x) / n
        vy = sum((v - my) ** 2 for v in y) / n
        if vx <= 0 or vy <= 0:
            continue
        out[i] = cov / math.sqrt(vx * vy)
    return out


def rolling_variance(returns: Sequence[float], window: int) -> list[float | None]:
    out: list[float | None] = [None] * len(returns)
    for i in range(len(returns)):
        if i + 1 < window:
            continue
        w = returns[i - window + 1: i + 1]
        m = sum(w) / len(w)
        out[i] = sum((v - m) ** 2 for v in w) / len(w)
    return out


def find_transition_onsets(labels: Sequence[str], persist_days: int = 3,
                            stable_days_before: int = 20) -> list[dict]:
    """Onset = first day severity strictly increases AND the new, more
    severe regime holds for at least `persist_days` (filters single-day
    label flapping, not a real transition) AND the PRIOR regime had
    already been stable for at least `stable_days_before` days (so there
    is real pre-onset history to measure a rolling window against, and
    the "onset" isn't itself the tail of a previous transition)."""
    sev = [SEVERITY.get(l, 1) for l in labels]
    onsets = []
    stable_run = 0
    for i in range(1, len(labels)):
        if sev[i] == sev[i - 1]:
            stable_run += 1
            continue
        jump = sev[i] - sev[i - 1]
        prior_stable = stable_run >= stable_days_before
        stable_run = 0
        if jump <= 0:
            continue
        if i + persist_days > len(labels):
            continue
        held = all(sev[j] >= sev[i] for j in range(i, i + persist_days))
        if held and prior_stable:
            onsets.append({"index": i, "from": labels[i - 1], "to": labels[i],
                            "severity_jump": jump})
    return onsets


def _mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_lead_signal(returns: Sequence[float], onsets: list[dict],
                         window: int = 20,
                         lead_offsets: Sequence[int] = (20, 10, 5, 1),
                         rng_seed: int = 1337) -> dict:
    """For each onset, read rolling AR1/variance at `lead_offsets` days
    before onset. Compare the onset-window mean against a control sample
    of the same size drawn from non-onset days (base rate per REASONING
    STANDARD #3), using a fixed seed so the comparison is reproducible."""
    import random

    ar1 = rolling_ar1(returns, window)
    var = rolling_variance(returns, window)
    onset_idx = {o["index"] for o in onsets}
    valid_pool = [i for i in range(len(returns))
                  if ar1[i] is not None and var[i] is not None
                  and i not in onset_idx]

    rng = random.Random(rng_seed)
    result: dict = {"n_onsets": len(onsets), "window": window,
                     "by_lead_days": {}}
    if len(onsets) < MIN_ONSETS_FOR_STATS:
        result["insufficient_n"] = True
        result["note"] = (f"only {len(onsets)} qualifying onsets found "
                           f"(<{MIN_ONSETS_FOR_STATS}) — no comparison "
                           "computed; per this repo's n>=5 reporting floor "
                           "this is reported as insufficient, not as a "
                           "null/negative result")
        return result

    for lead in lead_offsets:
        onset_ar1, onset_var = [], []
        for o in onsets:
            j = o["index"] - lead
            if 0 <= j < len(returns) and ar1[j] is not None and var[j] is not None:
                onset_ar1.append(ar1[j])
                onset_var.append(var[j])
        if len(onset_ar1) < MIN_ONSETS_FOR_STATS:
            result["by_lead_days"][lead] = {
                "insufficient_n": True, "n": len(onset_ar1)}
            continue
        control_idx = rng.sample(valid_pool, min(len(valid_pool),
                                                   len(onset_ar1) * 20))
        control_ar1 = [ar1[i] for i in control_idx]
        control_var = [var[i] for i in control_idx]
        result["by_lead_days"][lead] = {
            "n_onset": len(onset_ar1),
            "n_control": len(control_ar1),
            "onset_mean_ar1": _mean(onset_ar1),
            "control_mean_ar1": _mean(control_ar1),
            "onset_mean_var": _mean(onset_var),
            "control_mean_var": _mean(control_var),
        }
    return result


def run_probe(days: int = 2520, window: int = 20,
              lead_offsets: Sequence[int] = (20, 10, 5, 1)) -> dict:
    """Orchestrates the full gate-2 probe against REAL data. Requires
    network/Alpaca access this sandbox did not have — see module
    docstring. Reuses backtest_v2.fetch_bars/regime_series verbatim
    (EDGE DOCTRINE #3: don't re-derive already-verified data plumbing)."""
    import backtest_v2 as bt

    spy = bt.fetch_bars("SPY", days)
    vxx = bt.fetch_bars("VXX", days)
    labels, quality = bt.regime_series(spy, vxx)
    returns = log_returns(spy["close"])
    # labels[0] has no return; align labels to the returns index (labels[i+1] <-> returns[i])
    aligned_labels = labels[1:]
    onsets = find_transition_onsets(aligned_labels)
    signal = compute_lead_signal(returns, onsets, window=window,
                                  lead_offsets=lead_offsets)
    return {
        "vxx_data_quality": quality,
        "n_days": len(spy["date"]),
        "date_range": [spy["date"][0], spy["date"][-1]] if spy["date"] else [],
        "onsets": onsets,
        "signal": signal,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=2520,
                     help="calendar days of SPY/VXX history to pull (default ~10y)")
    args = ap.parse_args()
    try:
        out = run_probe(days=args.days)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
    print(json.dumps(out, indent=2, default=str))
