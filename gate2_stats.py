#!/usr/bin/env python3
"""
gate2_stats.py — shared ROOT VALIDATION LADDER gate 2 (SIGNAL) statistical
core: the no-lookahead entry-index rule and the Newey-West (HAC,
Bartlett-kernel) forward-return significance test every weekly/periodic
gate-2 screen in this repo needs.

EDGE DOCTRINE #3 (compile knowledge into code, never analyze the same
thing twice with reasoning): `_newey_west_diff_test` was independently
hand-written three times before this file existed — cot_gate2_test.py
(2026-07-08, the origin), cftc_tff_gate2_test.py, and
scripts/eia930_gate2.py, whose own docstring recorded manually
re-deriving and re-verifying it "identical to cftc_tff_gate2_test.py's
_newey_west_diff_test, EXCEPT the lag ... explicitly checked before
writing this script, not assumed" rather than importing a shared
function. `find_entry_index` was duplicated the same way. Compiled here
once so the next gate-2 screen calls a library function instead of
retyping statistics research has already reasoned through.

Pure statistical measurement only — no trading, no bot_engine/deep_score/
system_config import, no network/file I/O. Any change to this file is
MEASUREMENT INTEGRITY code per CLAUDE.md: its own PR, never bundled with
a strategy change, and the PR must state what each metric reported before
vs. after on identical historical inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def find_entry_index(bar_dates: list, publish_date: str):
    """First bar strictly after `publish_date` (no lookahead)."""
    for i, d in enumerate(bar_dates):
        if d > publish_date:
            return i
    return None


def newey_west_diff_test(rows: list, horizon: int, bucket: str,
                          lag: int | None = None) -> dict | None:
    """HAC (Newey-West, Bartlett-kernel) test of whether `bucket` rows'
    forward return differs from the COMPLEMENT (non-bucket rows) at this
    horizon. Implemented as OLS of forward return on a 0/1 bucket-dummy
    over ALL rows in chronological order (`rows` must already be
    time-ordered — every caller's compute_forward_returns preserves the
    source archive's oldest->newest order): the dummy coefficient is
    algebraically exactly the conditional mean difference, and its
    Newey-West sandwich variance corrects for the autocorrelation that
    periodically-sampled, horizon-day-wide overlapping windows create
    between neighboring observations.

    This "baseline" (the complement) is NOT the same as a screen's own
    pooled-all-rows baseline (e.g. summarize()'s "baseline" bucket in the
    callers below), which dilutes the comparison group with the very
    observations being tested. Comparing against the complement instead
    is the standard two-sample setup and the more conservative one (a raw
    pooled-baseline gap is typically an underestimate of the true
    bucket-vs-rest gap for this reason).

    Truncation lag defaults to round(horizon / 5) — the number of
    periodic observations a `horizon`-trading-day forward window overlaps
    with its neighbors when sampled roughly every 5 trading days (weekly).
    A caller sampling at a different cadence should pass `lag` explicitly
    (scripts/eia930_gate2.py does, for its own cadence) rather than rely
    on this default. Returns None (never a fabricated number) if there
    are too few observations to estimate the lagged autocovariance terms,
    or if `bucket` matches none/all of the rows (an undefined contrast)."""
    y, x = [], []
    for r in rows:
        fr = r["forward_returns"].get(horizon)
        if fr is None:
            continue
        y.append(fr)
        x.append(1.0 if r["bucket"] == bucket else 0.0)
    n = len(y)
    if lag is None:
        lag = max(1, round(horizon / 5))
    if n < 2 * lag + 4 or sum(x) == 0 or sum(x) == n:
        return None

    y_arr = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(n), np.asarray(x, dtype=float)])
    beta, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
    resid = y_arr - X @ beta

    xu = X * resid[:, None]
    S = xu.T @ xu
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1)
        cross = xu[l:].T @ xu[:-l]
        S += w * (cross + cross.T)

    xtx_inv = np.linalg.inv(X.T @ X)
    cov = xtx_inv @ S @ xtx_inv
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    beta1 = float(beta[1])
    t_stat = beta1 / se if se > 0 else 0.0
    p_value = float(2 * (1 - norm.cdf(abs(t_stat))))
    return {
        "n": n,
        "lag_weeks": lag,
        "mean_diff_pct": round(beta1 * 100, 3),
        "hac_se_pct": round(se * 100, 3),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_value, 4),
    }
