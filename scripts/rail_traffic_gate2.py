#!/usr/bin/env python3
"""
scripts/rail_traffic_gate2.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for
the STB EP724 rail carload archive (datacore/rail/ep724_carloads.json;
GATE 1 (DATA) PASSED 2026-09-27, scripts/rail_traffic_gate1.py). Precedent
named as the ready-now NEXT item in this same file's own gate-1 session and
in research/open_questions.md's 2026-09-27 "CROSS-CONNECTION (ACTIVE
ANGLE-HUNTING #1)" entry — reproduced here, not re-derived.

QUESTION (PRIOR restated, written before running — REASONING STANDARD #10):
does a SURPRISE in system-wide intermodal (Containers + Trailers) rail
carload growth — how the current 4-week block compares to its own trailing
history, not the raw level — lead or coincide with forward returns of a
transportation-sector benchmark? MECHANISM (REASONING STANDARD #5):
intermodal volume is a real-time proxy for consumer-goods throughput
(import containers + domestic trailer freight) that reaches the tape before
it is digested into railroads' or retailers' own quarterly earnings.
PRIOR: a genuine but small and UNCROWDED edge, because AAR's own weekly
rail traffic press release already makes the raw headline level public
same-day (base rate is NOT "nobody sees this") — any edge left has to come
from either (a) a SURPRISE framing relative to this archive's own 9-year
seasonal history (not a headline AAR publishes) or (b) a genuinely
directional bet (extreme_high -> positive, extreme_low -> negative), not
just "growth is up". Kill if the primary pre-registered test below shows no
separation, or the wrong sign, at the pre-registered N and horizon.

WHY "SURPRISE" NOT RAW LEVEL (the open_questions.md entry's own stated
reason for not testing raw week-over-week growth directly): intermodal
carloads carry strong within-year seasonality (holiday weeks, etc.) that a
raw growth number would conflate with real news. This script instead
computes a Larry-Williams-style 0-100 PERCENTILE INDEX of each week's
growth rate against its own trailing 104-week (2-year) history — the SAME
formula cftc_cot.py's `_cot_index` already uses for COT positioning
(cot_gate2_test.py's own precedent for exactly this bucketing shape) —
independently reimplemented here rather than importing a private helper
across an unrelated module; EDGE DOCTRINE #3 follow-up: extracting a shared
percentile-index helper into gate2_stats.py is a valid future MEASUREMENT
INTEGRITY PR (its own docstring requires that class of change be its own
PR), not bundled into this one.

TWO GROWTH MEASURES tested (per the open_questions.md NEXT note's own
wording, "week-over-week/4-week growth"): `rolling4_growth` (this week's
trailing 4-week sum vs. the PRIOR non-overlapping trailing 4-week sum, log
scale — smooths single-week reporting noise) is the PRE-REGISTERED PRIMARY
measure; `wow_growth` (plain week-over-week log growth) is a secondary
robustness check only, not a bar to pass. PRIMARY is designated because the
form the open_questions.md hypothesis singles out for the mechanism
(consumer-goods throughput trend) is a multi-week block comparison, and a
single week is dominated by day-count/holiday noise the 4-week measure
already partly cancels.

PRE-REGISTERED PRIMARY TEST (stated before any real number is computed):
growth4-based percentile index, EXTREME_HIGH bucket (>=80), benchmark IYT
(iShares Transportation Average ETF — the single most representative
liquid transportation-sector instrument, chosen ex ante, not picked after
seeing which benchmark works), horizon = 5 trading days. Bar: bucket
N >= MIN_BUCKET_N (20, so a Newey-West estimate over the bucket is not
built from a handful of weeks), Newey-West HAC p < 0.05, and the mean
forward-return DIFFERENCE positive (matches the mechanism: better-than-
usual intermodal growth -> better-than-usual transport-sector forward
return). Same test at EXTREME_LOW (expected sign: negative) is reported
as a SECONDARY directional-consistency check, not required for PASS on its
own (REASONING STANDARD #4: prefer fewer, theory-motivated tests) — but a
clean PASS with a secondary sign that flatly CONTRADICTS the mechanism
(e.g. extreme_low also positive) would itself be reported as a reason for
caution, not silently dropped.

SECONDARY / EXPLORATORY (REASONING STANDARD #4 — discount by the number of
things tried, reported for context, never itself a PASS/FAIL bar): horizon
20d for both buckets/both benchmarks; wow_growth-based buckets at both
horizons/both benchmarks; the RAIL_BASKET benchmark (equal-weighted daily
returns of UNP/CSX/NSC/CP, the archive's own US-listed Class I railroad
set, synthesized into one price series — see `build_basket_bars`) at both
growth measures/both horizons. 2 growth measures x 2 benchmarks x 2
horizons x 2 buckets = 16 total comparisons across primary+secondary;
anything beyond the single pre-registered primary test needs a
Bonferroni-adjusted bar of 0.05/15 ~= 0.0033 (15 = 16 minus the one
primary) to be treated as more than descriptive.

NO LOOKAHEAD (REASONING STANDARD #7): the archive's own "week ending" date
is not the date the number becomes public. STB/AAR's weekly rail-traffic
report is not confirmed against a stated publish calendar in this session
(unlike cot_gate2_test.py's confirmed Tuesday-as-of/Friday-publish rule) —
PUBLISH_LAG_DAYS=5 calendar days is used as a deliberately CONSERVATIVE
(upper-bound) assumption, stated honestly rather than guessed optimistic:
extra slack in the entry timing only adds noise to this screen, it can
never manufacture a lookahead edge the way an over-eager (too-early) lag
assumption could.

CONFOUND SCREEN: none built — per the open_questions.md entry's own prior
assessment, no domestic/liquidity confound is obviously expected here
(unlike settlement_stress_composite's foreign-ADR confound), since both
sides of this test are purely domestic (US Class I railroads, a US
transportation-sector ETF).

Pure statistical measurement only — SIGNAL gate, no trading involved. Does
not import or touch bot_engine.py / deep_score / system_config.py.

Usage: python3 scripts/rail_traffic_gate2.py [--archive PATH] [--out rail_traffic_gate2_results.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from gate2_stats import find_entry_index, newey_west_diff_test  # noqa: E402

ARCHIVE = os.path.join(REPO_ROOT, "datacore", "rail", "ep724_carloads.json")

CARLOAD_MEASURE = "Weekly Carloads By 22 Commodity Categories"
INTERMODAL_VARIABLES = {"Containers", "Trailers"}

ROLL_WEEKS = 4
PCT_LOOKBACK_WEEKS = 104
MIN_WINDOW_FOR_INDEX = 52
EXTREME_HIGH = 80.0
EXTREME_LOW = 20.0
PUBLISH_LAG_DAYS = 5
HORIZONS = (5, 20)
MIN_BUCKET_N = 20
ALPHA_PRIMARY = 0.05

PRIMARY_BENCHMARK = "IYT"
RAIL_BASKET = ("UNP", "CSX", "NSC", "CP")
PRIMARY_MEASURE = "growth4"


# ── Archive aggregation ──────────────────────────────────────────────────────
def system_weekly_intermodal(doc: dict) -> list:
    """System-wide (all reporting railroads summed) weekly intermodal
    (Containers + Trailers) carload total, aligned to doc["weeks"]. A week
    with zero reporting railroads for these two variables is None, never
    zero-filled — same convention as rail_traffic_gate1.py's monthly_totals
    and server/railTraffic.ts."""
    weeks = doc["weeks"]
    n = len(weeks)
    totals = [0.0] * n
    reported = [False] * n
    for key, vals in doc["series"].items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        _railroad, measure, variable = parts
        if measure != CARLOAD_MEASURE or variable not in INTERMODAL_VARIABLES:
            continue
        for i, v in enumerate(vals):
            if v is None:
                continue
            totals[i] += v
            reported[i] = True
    return [totals[i] if reported[i] else None for i in range(n)]


def rolling4_growth(series: list) -> list:
    """Log growth of the trailing ROLL_WEEKS-week sum vs. the PRIOR
    non-overlapping ROLL_WEEKS-week sum. None wherever either window
    contains a missing week or a non-positive sum (never fabricated)."""
    out = [None] * len(series)
    for i in range(len(series)):
        if i < 2 * ROLL_WEEKS - 1:
            continue
        cur = series[i - ROLL_WEEKS + 1: i + 1]
        prev = series[i - 2 * ROLL_WEEKS + 1: i - ROLL_WEEKS + 1]
        if any(v is None for v in cur) or any(v is None for v in prev):
            continue
        cur_sum, prev_sum = sum(cur), sum(prev)
        if cur_sum <= 0 or prev_sum <= 0:
            continue
        out[i] = math.log(cur_sum / prev_sum)
    return out


def wow_growth(series: list) -> list:
    """Plain week-over-week log growth. Secondary robustness measure
    only — see module docstring on why rolling4_growth is primary."""
    out = [None] * len(series)
    for i in range(1, len(series)):
        a, b = series[i - 1], series[i]
        if a is None or b is None or a <= 0 or b <= 0:
            continue
        out[i] = math.log(b / a)
    return out


def trailing_percentile_index(values: list, lookback: int,
                               min_window: int = MIN_WINDOW_FOR_INDEX) -> list:
    """Larry-Williams-style 0-100 percentile of the current value within
    its own trailing `lookback` window of non-None values (causal — the
    window always includes only values up to and including the current
    index, matching cftc_cot.py's `_cot_index`). None until at least
    `min_window` values have accumulated (avoids a degenerate percentile
    from a tiny early window) and None wherever the input itself is None."""
    out = [None] * len(values)
    window: list = []
    for i, v in enumerate(values):
        if v is not None:
            window.append(v)
            if len(window) > lookback:
                window.pop(0)
        if v is None or len(window) < min_window:
            continue
        lo, hi = min(window), max(window)
        out[i] = 50.0 if hi == lo else round((v - lo) / (hi - lo) * 100, 1)
    return out


def bucket_for(index_value) -> str | None:
    if index_value is None:
        return None
    if index_value >= EXTREME_HIGH:
        return "extreme_high"
    if index_value <= EXTREME_LOW:
        return "extreme_low"
    return "mid"


def build_events(weeks: list, growth: list, index: list, measure: str) -> list:
    """One event per week with a valid percentile index, carrying the week,
    the raw growth value, the percentile index, and the resulting bucket."""
    out = []
    for i, idx in enumerate(index):
        if idx is None:
            continue
        out.append({
            "week": weeks[i],
            "measure": measure,
            "value": growth[i],
            "index": idx,
            "bucket": bucket_for(idx),
        })
    return out


# ── No-lookahead forward returns (shared shape with cot_gate2_test.py) ──────
def compute_forward_returns(events: list, bars: dict) -> list:
    """Pure function: for each event, finds its no-lookahead entry (first
    bar strictly after week + PUBLISH_LAG_DAYS) and computes forward N-day
    returns. Events too close to the end of `bars` for a horizon are
    dropped for that horizon (right-censoring honesty, never zero-filled)."""
    bar_dates = bars["date"]
    bar_closes = bars["close"]
    out = []
    for ev in events:
        publish_date = (datetime.strptime(ev["week"], "%Y-%m-%d") +
                         timedelta(days=PUBLISH_LAG_DAYS)).strftime("%Y-%m-%d")
        entry_idx = find_entry_index(bar_dates, publish_date)
        row = dict(ev)
        row["entry_date"] = bar_dates[entry_idx] if entry_idx is not None else None
        row["forward_returns"] = {}
        if entry_idx is not None:
            entry_price = bar_closes[entry_idx]
            for h in HORIZONS:
                exit_idx = entry_idx + h
                if exit_idx < len(bar_closes) and entry_price:
                    row["forward_returns"][h] = bar_closes[exit_idx] / entry_price - 1
        out.append(row)
    return out


def bucket_count(rows: list, horizon: int, bucket: str) -> int:
    return sum(1 for r in rows if r["bucket"] == bucket and horizon in r["forward_returns"])


def run_test(rows: list, horizon: int, bucket: str, min_n: int = MIN_BUCKET_N,
             alpha: float = ALPHA_PRIMARY, expected_sign: int = 1) -> dict:
    """Wraps newey_west_diff_test with the pre-registered N floor and a
    verdict against the given alpha and expected sign (+1 = expect a
    positive mean difference, -1 = expect negative). Never fabricates a
    verdict when there isn't enough data."""
    n_bucket = bucket_count(rows, horizon, bucket)
    if n_bucket < min_n:
        return {"n_bucket": n_bucket, "verdict": "WAITING",
                "reason": f"n_bucket {n_bucket} < MIN_BUCKET_N {min_n}"}
    hac = newey_west_diff_test(rows, horizon, bucket)
    if hac is None:
        return {"n_bucket": n_bucket, "verdict": "WAITING",
                "reason": "newey_west_diff_test returned None (degenerate bucket)"}
    sign_ok = (hac["mean_diff_pct"] > 0) if expected_sign > 0 else (hac["mean_diff_pct"] < 0)
    passed = hac["p_value"] < alpha and sign_ok
    reason = "OK" if passed else (
        f"p={hac['p_value']} >= alpha {alpha}" if hac["p_value"] >= alpha
        else f"sign mismatch (mean_diff_pct={hac['mean_diff_pct']}, expected {'positive' if expected_sign > 0 else 'negative'})"
    )
    return {"n_bucket": n_bucket, "hac": hac, "verdict": "PASS" if passed else "FAIL",
            "reason": reason}


# ── Rail-basket synthetic benchmark ─────────────────────────────────────────
def build_basket_bars(all_bars: dict) -> dict:
    """Equal-weighted synthetic daily-close series over the tickers in
    `all_bars` ({ticker: bars_dict}), built from each ticker's own daily
    return rather than raw price levels (so tickers at very different
    price scales combine fairly). Only dates present in EVERY ticker's
    bars are used (inner join) — a date only one ticker has (a listing gap,
    a fetch hole) is silently excluded rather than guessed. Starts the
    synthetic index at 100.0; the level itself is never meaningful, only
    its forward returns are used downstream."""
    tickers = list(all_bars.keys())
    if not tickers:
        return {"date": [], "close": []}
    common = None
    for t in tickers:
        s = set(all_bars[t]["date"])
        common = s if common is None else (common & s)
    common_dates = sorted(common or set())
    closes_by_ticker = {t: dict(zip(all_bars[t]["date"], all_bars[t]["close"])) for t in tickers}

    out_dates = [common_dates[0]] if common_dates else []
    out_closes = [100.0] if common_dates else []
    for i in range(1, len(common_dates)):
        rets = []
        for t in tickers:
            c0 = closes_by_ticker[t][common_dates[i - 1]]
            c1 = closes_by_ticker[t][common_dates[i]]
            if c0:
                rets.append(c1 / c0 - 1)
        avg = sum(rets) / len(rets) if rets else 0.0
        out_dates.append(common_dates[i])
        out_closes.append(out_closes[-1] * (1 + avg))
    return {"date": out_dates, "close": out_closes}


# ── Orchestration ────────────────────────────────────────────────────────────
def run(doc: dict, benchmark_bars: dict) -> dict:
    weeks = doc["weeks"]
    system_series = system_weekly_intermodal(doc)
    measures = {
        "growth4": rolling4_growth(system_series),
        "wow": wow_growth(system_series),
    }

    results = {}
    for measure_name, growth in measures.items():
        index = trailing_percentile_index(growth, PCT_LOOKBACK_WEEKS)
        events = build_events(weeks, growth, index, measure_name)
        for bench_name, bars in benchmark_bars.items():
            rows = compute_forward_returns(events, bars)
            for h in HORIZONS:
                for bucket, sign in (("extreme_high", 1), ("extreme_low", -1)):
                    is_primary = (measure_name == PRIMARY_MEASURE and
                                  bench_name == PRIMARY_BENCHMARK and
                                  h == HORIZONS[0] and bucket == "extreme_high")
                    alpha = ALPHA_PRIMARY if is_primary else ALPHA_PRIMARY / 15
                    key = f"{measure_name}|{bench_name}|h{h}|{bucket}"
                    r = run_test(rows, h, bucket, alpha=alpha, expected_sign=sign)
                    r["is_primary"] = is_primary
                    r["alpha_used"] = alpha
                    results[key] = r

    primary_key = f"{PRIMARY_MEASURE}|{PRIMARY_BENCHMARK}|h{HORIZONS[0]}|extreme_high"
    secondary_key = f"{PRIMARY_MEASURE}|{PRIMARY_BENCHMARK}|h{HORIZONS[0]}|extreme_low"
    primary = results[primary_key]
    secondary = results.get(secondary_key)
    if primary["verdict"] == "PASS":
        secondary_sign_ok = bool(secondary and secondary.get("hac") and secondary["hac"]["mean_diff_pct"] < 0)
        overall = "GATE2_PASS (primary confirmed, secondary directional check consistent)" if secondary_sign_ok \
            else "GATE2_PASS (primary only — secondary extreme_low sign not consistent, flagged not hidden)"
    elif primary["verdict"] == "WAITING":
        overall = "WAITING"
    else:
        overall = "GATE2_FAIL (primary pre-registered test did not clear its bar)"

    return {
        "primary_key": primary_key,
        "overall_verdict": overall,
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--out", default="rail_traffic_gate2_results.json")
    args = ap.parse_args()

    from backtest_v2 import fetch_bars  # local import: keeps this script's
    # unit tests free of backtest_v2's network/cache side effects

    with open(args.archive) as f:
        doc = json.load(f)

    weeks = doc["weeks"]
    earliest = datetime.strptime(weeks[0], "%Y-%m-%d")
    days_needed = (datetime.utcnow() - earliest).days + 30

    print(f"fetching {PRIMARY_BENCHMARK} ...", file=sys.stderr, flush=True)
    iyt_bars = fetch_bars(PRIMARY_BENCHMARK, days_needed)
    basket_raw = {}
    for t in RAIL_BASKET:
        print(f"fetching {t} ...", file=sys.stderr, flush=True)
        basket_raw[t] = fetch_bars(t, days_needed)
    basket_bars = build_basket_bars(basket_raw)

    benchmark_bars = {PRIMARY_BENCHMARK: iyt_bars, "rail_basket": basket_bars}
    result = run(doc, benchmark_bars)

    print(json.dumps(result, indent=2, default=str))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {args.out}")
    print(f"\nVERDICT: {result['overall_verdict']}")
    return 0 if result["overall_verdict"].startswith("GATE2_PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
