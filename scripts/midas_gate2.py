#!/usr/bin/env python3
"""
scripts/midas_gate2.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for the
MIDAS HFT-COLONIZATION FILTER hypothesis (research/open_questions.md,
"MIDAS HFT-COLONIZATION FILTER HYPOTHESIS" entry). Gate 1 (DATA) passed
2026-07-10 for the raw feed (server/secMidas.ts). Gate 2 was blocked twice
over (multi-quarter archive depth; a missing multi-day read surface) until
the 2026-08-25 PRODUCT session shipped `aggregateMidasQuarterByTicker` +
the `midas_quarter` diag probe specifically to unblock this script.

PRE-REGISTERED HYPOTHESIS (written before running, REASONING STANDARD #10):
among McapRank<=2 small caps, a HIGH cancel-to-trade / hidden-rate /
odd-lot-rate (within-mcapRank-stratum tercile, computed from a QUARTER-SUMMED
numerator/denominator so one thin day can't swing it) predicts WORSE forward
returns than the LOW tercile and worse than the full-sample (random-entry,
same universe, same holding period — REASONING STANDARD #3) baseline, i.e.
high < baseline < low. Tested independently per metric (3 candidate filters,
not one) — REASONING STANDARD #4 discounts each accordingly, not combined
into an unregistered composite score.

NO-LOOKAHEAD / PUBLISH-LAG: SEC MIDAS publishes on its own lag (secMidas.ts's
header comment: "multi-quarter publish lag"). Nothing in this repo stores a
per-quarter first-archived timestamp, so this script uses a conservative,
EVIDENCE-DERIVED upper bound instead of guessing: on 2026-08-25 the live
`midas_quarter` probe already had 2026q2 (quarter end 2026-06-30) archived —
only possible if that quarter's real lag was <= 56 calendar days. Applying
that same 56-day bound to every quarter is a safe (never-before-actual-
availability) assumption, not a most-likely one; a quarter whose +lag entry
date has too little forward history left is simply excluded (see
`quarter_is_ready`), not forced.

Sampling: the full McapRank<=2 universe is 750-800 tickers/quarter — not
feasible to price-fetch in one run against this repo's Yahoo-only fallback
(no ALPACA_KEY/SECRET in most sandboxes). Draws a fixed-seed stratified
random sample of `n_sample` tickers per quarter (proportional to that
quarter's own McapRank 1-vs-2 split) instead — same order of magnitude
(n=100-130/side) as this repo's other gate-2 screens.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from datetime import datetime, timedelta

METRICS = ["cancelToTrade", "hiddenRatePct", "oddLotRatePct"]
DEFAULT_HORIZONS = (5, 20, 60)
DEFAULT_PUBLISH_LAG_DAYS = 56
DEFAULT_N_SAMPLE = 120
DEFAULT_BASE_URL = "https://voltradeai-production.up.railway.app"


def _http_get_json(url: str, timeout: int = 30, retries: int = 3) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "voltradeai-midas-gate2/1.0"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:  # transient network
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def fetch_midas_quarter(period: str, base_url: str, token: str) -> dict:
    """Live GET against the `midas_quarter` diag probe (server/diag.ts +
    server/bot.ts's `case "midas_quarter"`). Returns the raw probe payload
    ({probe, period, min_days_floor, n_tickers, rows})."""
    url = f"{base_url}/api/diag/midas_quarter?period={period}&token={token}"
    return _http_get_json(url)


def entry_date_for(quarter_end: str, lag_days: int = DEFAULT_PUBLISH_LAG_DAYS) -> str:
    dt = datetime.strptime(quarter_end, "%Y-%m-%d") + timedelta(days=lag_days)
    return dt.strftime("%Y-%m-%d")


def quarter_is_ready(quarter_end: str, as_of: str, lag_days: int = DEFAULT_PUBLISH_LAG_DAYS,
                      min_forward_days: int = 1) -> bool:
    """False when this quarter's own +lag entry date leaves fewer than
    `min_forward_days` calendar days of history before `as_of` — the
    no-lookahead guard that excludes a too-recent quarter instead of
    silently testing a horizon that hasn't happened yet."""
    entry = datetime.strptime(entry_date_for(quarter_end, lag_days), "%Y-%m-%d")
    return (datetime.strptime(as_of, "%Y-%m-%d") - entry).days >= min_forward_days


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Fixed-seed random draw of `n` rows from `rows`, stratified by
    mcapRank proportional to that stratum's share of the population."""
    rng = random.Random(seed)
    by_stratum: dict[int, list[dict]] = {}
    for r in rows:
        by_stratum.setdefault(r["mcapRank"], []).append(r)
    total = sum(len(v) for v in by_stratum.values())
    out: list[dict] = []
    for members in by_stratum.values():
        take = min(len(members), max(1, round(n * len(members) / total))) if total else 0
        out.extend(rng.sample(members, take))
    return out


def tercile_bucket(rows: list[dict], metric: str) -> dict[str, str]:
    """Within each mcapRank stratum, sorts by `metric` and labels the
    bottom/middle/top third low/mid/high. Returns {ticker: bucket}."""
    out: dict[str, str] = {}
    by_stratum: dict[int, list[dict]] = {}
    for r in rows:
        by_stratum.setdefault(r["mcapRank"], []).append(r)
    for members in by_stratum.values():
        ordered = sorted(members, key=lambda r: r[metric])
        n = len(ordered)
        t1, t2 = n // 3, 2 * n // 3
        for i, r in enumerate(ordered):
            out[r["ticker"]] = "low" if i < t1 else ("mid" if i < t2 else "high")
    return out


def forward_returns_from_bars(bars: dict, entry_date: str, horizons: tuple[int, ...],
                               find_entry_index_fn) -> dict[int, float]:
    """Pure: given an already-fetched `bars` dict ({date, close, ...}, oldest
    -> newest) and a no-lookahead `find_entry_index_fn` (gate2_stats.py's
    `find_entry_index`), returns {horizon: forward_return}. A horizon whose
    exit bar doesn't exist yet is omitted (right-censoring honesty), never
    zero-filled."""
    dates, closes = bars.get("date") or [], bars.get("close") or []
    if not dates:
        return {}
    idx = find_entry_index_fn(dates, entry_date)
    if idx is None:
        return {}
    entry_price = closes[idx]
    if not entry_price:
        return {}
    out = {}
    for h in horizons:
        exit_idx = idx + h
        if exit_idx < len(closes):
            out[h] = closes[exit_idx] / entry_price - 1
    return out


def welch_vs_baseline(sample: list[float], baseline: list[float]) -> dict | None:
    """Welch two-sample t-test, `sample` vs. the pooled full-universe-sample
    baseline (same design as usaspending_gate2.py's significance_by_bucket,
    adapted from a repeated-event archive to a single-snapshot cross
    section). None (never fabricated) below n=5 on either side."""
    if len(sample) < 5 or len(baseline) < 5:
        return None
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(sample, baseline, equal_var=False)
    return {
        "n": len(sample),
        "n_baseline": len(baseline),
        "mean_pct": round(sum(sample) / len(sample) * 100, 3),
        "baseline_mean_pct": round(sum(baseline) / len(baseline) * 100, 3),
        "mean_diff_pct": round((sum(sample) / len(sample) - sum(baseline) / len(baseline)) * 100, 3),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_value), 4),
    }


def run_quarter(period: str, quarter_end: str, rows: list[dict], fetch_bars_fn,
                 find_entry_index_fn, n_sample: int = DEFAULT_N_SAMPLE,
                 seed: int = 0, lag_days: int = DEFAULT_PUBLISH_LAG_DAYS,
                 horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                 days_back_buffer: int = 130, as_of: str | None = None) -> dict:
    """Orchestrates one quarter's test. `fetch_bars_fn(ticker, days) -> bars`
    and `find_entry_index_fn(dates, publish_date) -> idx|None` are injected
    so this function has no hardcoded network/import dependency and can be
    unit-tested with fakes."""
    entry_date = entry_date_for(quarter_end, lag_days)
    sample = stratified_sample(rows, n_sample, seed)

    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d") if as_of else datetime.utcnow()
    days_back = (as_of_dt - datetime.strptime(entry_date, "%Y-%m-%d")).days + days_back_buffer

    fwd: dict[str, dict[int, float]] = {}
    fetch_failures = 0
    for r in sample:
        t = r["ticker"]
        try:
            bars = fetch_bars_fn(t, days_back)
        except Exception:
            fetch_failures += 1
            fwd[t] = {}
            continue
        fwd[t] = forward_returns_from_bars(bars, entry_date, horizons, find_entry_index_fn)

    baseline_by_h: dict[int, list[float]] = {h: [] for h in horizons}
    for t, fr in fwd.items():
        for h, v in fr.items():
            baseline_by_h[h].append(v)

    metrics_out: dict[str, dict] = {}
    for metric in METRICS:
        buckets = tercile_bucket(sample, metric)
        m_result = {}
        for h in horizons:
            base = baseline_by_h[h]
            row = {}
            for bucket in ("high", "low"):
                vals = [fwd[t][h] for t in fwd if buckets.get(t) == bucket and h in fwd[t]]
                row[bucket] = welch_vs_baseline(vals, base)
            m_result[str(h)] = row
        metrics_out[metric] = m_result

    return {
        "period": period,
        "quarter_end": quarter_end,
        "entry_date": entry_date,
        "n_universe": len(rows),
        "n_sampled": len(sample),
        "fetch_failures": fetch_failures,
        "baseline_n": {str(h): len(v) for h, v in baseline_by_h.items()},
        "baseline_mean_pct": {str(h): (round(sum(v) / len(v) * 100, 3) if v else None)
                               for h, v in baseline_by_h.items()},
        "metrics": metrics_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", nargs="+", default=["2025q4", "2026q1"],
                     help="quarters to test, e.g. 2025q4 2026q1")
    ap.add_argument("--quarter-ends", nargs="+", default=["2025-12-31", "2026-03-31"],
                     help="matching quarter-end dates for --periods, same order")
    ap.add_argument("--n-sample", type=int, default=DEFAULT_N_SAMPLE)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--lag-days", type=int, default=DEFAULT_PUBLISH_LAG_DAYS)
    ap.add_argument("--base-url", default=os.environ.get("VOLTRADE_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--out", default="midas_gate2_results.json")
    args = ap.parse_args()

    token = os.environ.get("DIAG_TOKEN")
    if not token:
        raise SystemExit("DIAG_TOKEN not set in environment")
    if len(args.periods) != len(args.quarter_ends):
        raise SystemExit("--periods and --quarter-ends must have the same length")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backtest_v2 import fetch_bars  # local import: keeps unit tests network-free
    from gate2_stats import find_entry_index

    today = datetime.utcnow().strftime("%Y-%m-%d")
    out = {"generated": today, "n_sample": args.n_sample, "seed": args.seed,
           "lag_days": args.lag_days, "quarters": {}}

    for period, qend in zip(args.periods, args.quarter_ends):
        if not quarter_is_ready(qend, today, args.lag_days):
            out["quarters"][period] = {"status": "not_ready", "quarter_end": qend,
                                        "entry_date": entry_date_for(qend, args.lag_days)}
            print(f"{period}: not ready yet (entry date {entry_date_for(qend, args.lag_days)})")
            continue
        probe = fetch_midas_quarter(period, args.base_url, token)
        rows = probe.get("rows") or []
        print(f"{period}: {len(rows)} tickers archived, running test...")
        result = run_quarter(period, qend, rows, fetch_bars, find_entry_index,
                              n_sample=args.n_sample, seed=args.seed, lag_days=args.lag_days,
                              as_of=today)
        out["quarters"][period] = result
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  done: n_sampled={result['n_sampled']} fetch_failures={result['fetch_failures']}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
