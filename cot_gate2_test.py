#!/usr/bin/env python3
"""
cot_gate2_test.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for the CFTC
Commitments of Traders positioning archive (cftc_cot.py, gate 1 DATA passed
2026-07-05, see research/open_questions.md).

Question: does an extreme non-commercial (speculator) COT index — the
standard 0-100 trailing-156-week percentile already computed by
cftc_cot.attach_cot_index — predict a forward mean-reversion move in the
underlying ETF, beating a same-symbol random-entry (all-weeks) baseline?
REASONING STANDARD #3 (demand base rates) and #4 (discount for the number
of symbol/bucket/horizon combinations tried — 7 symbols x 2 buckets x 2
horizons = 28 comparisons here; this is a screen, not a conclusion).

No lookahead (REASONING STANDARD #7): a report "as of" a Tuesday is not
public until CFTC's Friday ~15:30 ET release, so the entry anchor for every
week is the first trading day STRICTLY AFTER report_date + 3 calendar days
(i.e. the Friday publish), never the as-of Tuesday itself.

METHODOLOGICAL FIX (2026-07-08, see open_questions.md's "METHODOLOGICAL
FINDING" on the original SLV/USO screen): weekly-sampled forward returns at
a 20d/60d horizon overlap ~4/~12 weeks between consecutive observations —
raw means (summarize() below) cannot distinguish a real effect from a
handful of autocorrelated episodes. `hac_significance()` adds a
Newey-West (HAC, Bartlett-kernel) significance test on top of the existing
raw-means screen so a large raw gap can be told apart from noise before
anything is promoted toward LOGIC gate 3. It still cannot single-handedly
PASS a symbol — REASONING STANDARD #4 discounting and out-of-sample
confirmation still apply — but it replaces "eyeball the gap" with an actual
test statistic.

Pure statistical measurement only — SIGNAL gate, no trading involved. Does
not import or touch bot_engine.py / deep_score / system_config.py.

Usage: python3 cot_gate2_test.py [--out cot_gate2_results.json]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import norm

from cftc_cot import SYMBOLS, LOOKBACK_WEEKS, attach_cot_index, fetch_symbol_history

HORIZONS = (20, 60)  # trading days
EXTREME_HIGH = 80.0
EXTREME_LOW = 20.0
PUBLISH_LAG_DAYS = 3  # Tuesday as-of -> Friday publish


def find_entry_index(bar_dates: list[str], publish_date: str) -> int | None:
    """First bar strictly after `publish_date` (no lookahead)."""
    for i, d in enumerate(bar_dates):
        if d > publish_date:
            return i
    return None


def bucket_for(index_value: float | None) -> str | None:
    if index_value is None:
        return None
    if index_value >= EXTREME_HIGH:
        return "extreme_high"
    if index_value <= EXTREME_LOW:
        return "extreme_low"
    return "mid"


def compute_forward_returns(cot_records: list[dict], bars: dict) -> list[dict]:
    """Pure function: for each COT week, find its no-lookahead entry anchor
    in `bars` and compute forward N-day returns. Weeks too close to the end
    of `bars` for a given horizon are dropped for that horizon (right-
    censoring honesty), never zero-filled."""
    bar_dates = bars["date"]
    bar_closes = bars["close"]
    out = []
    for rec in cot_records:
        report_date = rec["report_date"]
        publish_date = (datetime.strptime(report_date, "%Y-%m-%d") +
                         timedelta(days=PUBLISH_LAG_DAYS)).strftime("%Y-%m-%d")
        entry_idx = find_entry_index(bar_dates, publish_date)
        row = {
            "report_date": report_date,
            "cot_index_noncomm": rec.get("cot_index_noncomm"),
            "bucket": bucket_for(rec.get("cot_index_noncomm")),
            "entry_date": bar_dates[entry_idx] if entry_idx is not None else None,
            "forward_returns": {},
        }
        if entry_idx is not None:
            entry_price = bar_closes[entry_idx]
            for h in HORIZONS:
                exit_idx = entry_idx + h
                if exit_idx < len(bar_closes) and entry_price:
                    row["forward_returns"][h] = bar_closes[exit_idx] / entry_price - 1
        out.append(row)
    return out


def summarize(rows: list[dict]) -> dict:
    """Per-horizon: baseline (all weeks with a valid forward return) vs each
    extreme bucket's mean forward return, with sample counts so a reader can
    apply REASONING STANDARD #4 discounting themselves."""
    summary = {}
    for h in HORIZONS:
        vals = {"baseline": [], "extreme_high": [], "extreme_low": []}
        for r in rows:
            fr = r["forward_returns"].get(h)
            if fr is None:
                continue
            vals["baseline"].append(fr)
            if r["bucket"] in ("extreme_high", "extreme_low"):
                vals[r["bucket"]].append(fr)
        summary[str(h)] = {
            bucket: {
                "n": len(v),
                "mean_pct": round(sum(v) / len(v) * 100, 3) if v else None,
            }
            for bucket, v in vals.items()
        }
    return summary


def _newey_west_diff_test(rows: list[dict], horizon: int, bucket: str,
                           lag: int | None = None) -> dict | None:
    """HAC (Newey-West, Bartlett-kernel) test of whether `bucket` weeks'
    forward return differs from the COMPLEMENT (non-bucket weeks) at this
    horizon. Implemented as OLS of forward return on a 0/1 bucket-dummy
    over ALL weeks in chronological order (`rows` must already be
    time-ordered — compute_forward_returns preserves cot_records'
    oldest->newest order): the dummy coefficient is algebraically exactly
    the conditional mean difference, and its Newey-West sandwich variance
    corrects for the autocorrelation that weekly-sampled, horizon-day-wide
    overlapping windows create between neighboring observations.

    NOTE this baseline is NOT the same "baseline" as summarize() above,
    which pools ALL weeks (including the bucket's own) — that dilutes the
    comparison group with the very observations being tested. Comparing
    against the complement instead is the standard two-sample setup and
    the more conservative one (the raw screen's gap is typically an
    underestimate of the true bucket-vs-rest gap for this reason).

    Truncation lag defaults to round(horizon / 5) weeks — the number of
    weekly observations a `horizon`-trading-day forward window overlaps
    with its neighbors, i.e. exactly the span REASONING STANDARD #4
    flagged as inflating the naive sample size. Returns None (never a
    fabricated number) if there are too few observations to estimate the
    lagged autocovariance terms, or if `bucket` matches none/all of the
    weeks (an undefined contrast)."""
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


def hac_significance(rows: list[dict]) -> dict:
    """Per-horizon, per-extreme-bucket Newey-West test vs the complement —
    see _newey_west_diff_test's docstring for the baseline-definition
    caveat vs summarize()."""
    out = {}
    for h in HORIZONS:
        out[str(h)] = {
            bucket: _newey_west_diff_test(rows, h, bucket)
            for bucket in ("extreme_high", "extreme_low")
        }
    return out


def run(symbol: str, fetch_bars_fn) -> dict:
    meta = SYMBOLS[symbol]
    cot_records, rejected = fetch_symbol_history(meta["code"], limit=LOOKBACK_WEEKS)
    cot_records = attach_cot_index(cot_records)
    if not cot_records:
        return {"symbol": symbol, "status": "no_cot_data"}

    earliest = datetime.strptime(cot_records[0]["report_date"], "%Y-%m-%d")
    days_needed = (datetime.utcnow() - earliest).days + 30
    bars = fetch_bars_fn(symbol, days_needed)
    if not bars or not bars.get("date"):
        return {"symbol": symbol, "status": "no_price_data"}

    rows = compute_forward_returns(cot_records, bars)
    return {
        "symbol": symbol,
        "status": "ok",
        "weeks": len(cot_records),
        "rejected_cot_records": rejected,
        "summary": summarize(rows),
        "significance": hac_significance(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cot_gate2_results.json")
    args = ap.parse_args()

    from backtest_v2 import fetch_bars  # local import: keeps this script's
    # unit tests free of backtest_v2's network/cache side effects

    results = {}
    for symbol in SYMBOLS:
        print(f"--- {symbol} ---")
        try:
            results[symbol] = run(symbol, fetch_bars)
        except Exception as e:  # one symbol's failure must not abort the rest
            results[symbol] = {"symbol": symbol, "status": "error", "reason": str(e)[:200]}
        print(json.dumps(results[symbol], indent=2, default=str))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
