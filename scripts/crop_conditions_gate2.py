#!/usr/bin/env python3
"""
scripts/crop_conditions_gate2.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for
the crop_conditions_usda_nass root. This is the exact NEXT item queued by the
2026-09-06 fourth session this UTC day (research/experiments.md, PR #1015):
"GATE 2 (condition-delta vs forward grain futures returns ... ) is now
directly runnable against real archived history without further infra work;
the exact analogous design used for wikimedia_pageviews_attention/
nhtsa_vehicle_complaints ... is the template to reuse rather than re-derive."

DATA SOURCE: the production `/api/data/crop-conditions/history` route that
same session shipped (server/cropConditions.ts's readConditionsAggregateHistory,
served over HTTP) — this sandbox has no NASS_API_KEY (server-side key-gated,
per crop_conditions_gate1.ts's own note), so reading what production already
computed is the honest substitute, same convention gate 1 used. Live-checked
this session: 14 contiguous weekly rows, 2026-05-31 -> 2026-08-30, both
commodities present every week (no gaps).

METRIC: week-over-week change in the GOOD+EXCELLENT ("G/E") percentage —
the standard ag-industry condition summary statistic (the two best classes
of the five-class VERY POOR/POOR/FAIR/GOOD/EXCELLENT scale) — computed
per commodity from the archived weekly rows.

PROXY INSTRUMENTS: NASS reports farmland condition, not a price; there is no
way to trade "corn condition" directly. This repo's own established pattern
for testing a fundamental-data hypothesis is a NAMED, literal sector/asset
proxy chosen from the hypothesis itself, not selected after looking at
candidates (see scripts/eia930_gate2.py's XLI choice for "industrial
activity"). The literal proxies here are Teucrium's physically-backed grain
ETFs: CORN (Teucrium Corn Fund, near-term corn futures) and SOYB (Teucrium
Soybean Fund, near-term soybean futures) — both regular NYSE Arca-listed
tickers, fetchable through backtest_v2.fetch_bars (Alpaca-first/Yahoo
fallback) like any other equity/ETF symbol in this codebase; no raw futures
access exists in this stack (Alpaca paper is equities/options only), and an
ETF proxy is the same kind of honest substitution EIA930 gate 2 already
established as this repo's convention, not a new pattern invented here.
Live-checked this session: both tickers return 275 days of bars with full
coverage of the archive window.

PRE-REGISTRATION (REASONING STANDARD #10, written before any forward return
was computed — the delta/bucket counts below WERE computed first, during
design, to size the test honestly; no forward return or p-value was):
HYPOTHESIS: an improving G/E percentage (larger expected harvest, more
supply) precedes NEGATIVE forward returns in the corresponding grain ETF; a
worsening G/E percentage (crop stress, smaller expected harvest, less
supply) precedes POSITIVE forward returns. This is the standard agricultural-
economics supply/price relationship, not a novel claim.
PRIOR (informal, ~10%, stated before running — LOWER than this repo's other
recent gate-2 priors, e.g. nhtsa_vehicle_complaints' ~20%): REASONING
STANDARD #5, second-order — unlike NHTSA complaints (an obscure, uncurated
government feed) or a single Wikipedia page, USDA's weekly Crop Progress
report is one of THE most closely watched releases in commodities trading;
CBOT corn/soybean futures are professionally covered by grain-elevator
desks, ag hedge funds, and algorithmic readers of the report within seconds
of its Monday 4pm ET release. EDGE DOCTRINE #2 ("fish where whales can't")
cuts AGAINST this root, not for it — this is the opposite of an
under-covered niche. Any same-day reaction is almost certainly already
priced by the time this script's entry rule allows a trade (see ENTRY RULE
below); what is being tested is whether a *multi-day* drift beyond the
immediate reaction exists, which is a materially harder bar.
LOW-POWER CAVEAT (stated before running, not discovered after): this
session's own live count of the 14-week archive found only 13 week-over-week
deltas per commodity, split roughly 2-3 "improving" (delta>0) vs. 6-8
"worsening" (delta<0) vs. 2-5 zero-delta (excluded, no signal either way) —
see module-level DELTA_COUNTS_LIVE_CHECK_2026_09_06 comment below for the
exact live counts. The "improving" bucket in particular is too small (2-3
observations) for a meaningful significance test on its own; this is
disclosed here, before running, not used to justify discarding the bucket —
both buckets are tested and reported exactly as gate2_stats.newey_west_diff_test
returns them (which is itself a None, not a fabricated number, for buckets
below its own minimum-n floor).
VERDICT RULE (stated before running): GATE 2 passes only if at least one
(commodity, horizon, bucket) comparison clears the Bonferroni bar for this
2-commodity x 3-horizon x 2-bucket family (alpha/12 ~= 0.004167) AND is in
the hypothesized direction (worsening bucket: mean_diff_pct > 0; improving
bucket: mean_diff_pct < 0). A significant result in the wrong direction, or
nothing clearing the bar, is NOT a pass.
NOT ATTEMPTED HERE (left for gate 3, if gate 2 passes): any entry/exit rule,
cost/slippage deduction, ablation, or claim this is tradeable.

# Live counts recorded during design, 2026-09-06 (this session), BEFORE any
# forward return was computed — corn deltas: [0,1,0,-1,0,1,-1,-4,-2,0,-1,-3,0]
# (2 positive, 6 negative, 5 zero); soybean deltas:
# [-1,1,0,-1,-1,1,1,-3,0,-1,-1,-1,-2] (3 positive, 8 negative, 2 zero).
DELTA_COUNTS_LIVE_CHECK_2026_09_06 = True

ENTRY RULE (no lookahead): USDA's Crop Progress report releases the Monday
following each week_ending (a Sunday) at 4pm ET — crop_conditions_gate1.ts's
own hand-verified case (week_ending 2026-08-02 -> released 2026-08-03)
confirms week_ending+1 calendar day for that instance. This script does NOT
assume every week follows that exact +1 pattern (a Monday federal holiday
shifts the real release to Tuesday, and the archive carries no per-week
release-date field to check against) — it instead uses week_ending+2
calendar days as a deliberately conservative publish-date floor, then
gate2_stats.find_entry_index's own "first bar STRICTLY AFTER publish_date"
rule for the actual entry day. This trades a small amount of entry-timing
precision for a guarantee against ever entering before the true public
release, which matters more here (REASONING STANDARD #7) than shaving one
day off the horizon.

Pure statistical measurement only -- SIGNAL gate, no trading involved. Does
not import or touch bot_engine.py / deep_score / system_config.py.

Usage: python3 scripts/crop_conditions_gate2.py [--url https://voltradeai.com]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import date, datetime, timedelta
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from gate2_stats import find_entry_index, newey_west_diff_test as _newey_west_diff_test  # noqa: E402

HISTORY_URL_PATH = "/api/data/crop-conditions/history?weeks=104"

# commodity key (as served by the history route) -> proxy ETF ticker.
COMMODITY_TICKERS = {"corn": "CORN", "soybeans": "SOYB"}

HORIZONS = (5, 10, 20)  # trading days -- same convention as the repo's other weekly-cadence gate-2 scripts
PUBLISH_BUFFER_DAYS = 2  # conservative floor; see module docstring ENTRY RULE


# ── Pure functions (unit-tested, no network) ────────────────────────────────

def good_excellent_pct(commodity_row: dict) -> Optional[float]:
    """GOOD + EXCELLENT percentage from one week's {CLASS: pct} row. None if
    either class is missing (never silently treated as 0 — a missing class
    is a data gap, not a true zero)."""
    if "GOOD" not in commodity_row or "EXCELLENT" not in commodity_row:
        return None
    return commodity_row["GOOD"] + commodity_row["EXCELLENT"]


def compute_weekly_deltas(trend: Sequence[dict], commodity: str) -> list[dict]:
    """{week_ending, ge_pct, delta_ge_pct} rows, ascending by week_ending
    (trend is assumed already ascending, matching the history route's own
    order). The first week in `trend` never produces a row (no prior week to
    diff against) -- N rows in, at most N-1 deltas out."""
    out: list[dict] = []
    prev: Optional[float] = None
    for row in trend:
        ge = good_excellent_pct(row.get(commodity, {}))
        if ge is not None and prev is not None:
            out.append({
                "week_ending": row["week_ending"],
                "ge_pct": ge,
                "delta_ge_pct": ge - prev,
            })
        if ge is not None:
            prev = ge
    return out


def release_date_for(week_ending: str, buffer_days: int = PUBLISH_BUFFER_DAYS) -> str:
    d = datetime.strptime(week_ending, "%Y-%m-%d").date() + timedelta(days=buffer_days)
    return d.isoformat()


def bucket_for(delta_ge_pct: Optional[float]) -> Optional[str]:
    if delta_ge_pct is None or delta_ge_pct == 0:
        return None
    return "improving" if delta_ge_pct > 0 else "worsening"


def compute_forward_returns(deltas: Sequence[dict], bars: dict, horizons: Sequence[int] = HORIZONS) -> list[dict]:
    bar_dates, bar_closes = bars["date"], bars["close"]
    out = []
    for row in deltas:
        publish_date = release_date_for(row["week_ending"])
        entry_idx = find_entry_index(bar_dates, publish_date)
        out_row = {
            "week_ending": row["week_ending"],
            "delta_ge_pct": row["delta_ge_pct"],
            "bucket": bucket_for(row["delta_ge_pct"]),
            "publish_date": publish_date,
            "entry_date": bar_dates[entry_idx] if entry_idx is not None else None,
            "forward_returns": {},
        }
        if entry_idx is not None:
            entry_price = bar_closes[entry_idx]
            for h in horizons:
                exit_idx = entry_idx + h
                if exit_idx < len(bar_closes) and entry_price:
                    out_row["forward_returns"][h] = bar_closes[exit_idx] / entry_price - 1
        out.append(out_row)
    return out


def summarize(rows: Sequence[dict], horizons: Sequence[int] = HORIZONS) -> dict:
    summary = {}
    for h in horizons:
        vals = {"baseline": [], "improving": [], "worsening": []}
        for r in rows:
            fr = r["forward_returns"].get(h)
            if fr is None:
                continue
            vals["baseline"].append(fr)
            if r["bucket"] in ("improving", "worsening"):
                vals[r["bucket"]].append(fr)
        summary[str(h)] = {
            bucket: {"n": len(v), "mean_pct": round(sum(v) / len(v) * 100, 3) if v else None}
            for bucket, v in vals.items()
        }
    return summary


def hac_significance(rows: Sequence[dict], horizons: Sequence[int] = HORIZONS) -> dict:
    return {
        str(h): {bucket: _newey_west_diff_test(rows, h, bucket) for bucket in ("improving", "worsening")}
        for h in horizons
    }


BONFERRONI_N = len(COMMODITY_TICKERS) * len(HORIZONS) * 2  # commodities x horizons x buckets
ALPHA = 0.05


def evaluate_pass_bar(significance_by_commodity: dict, horizons: Sequence[int] = HORIZONS) -> dict:
    bar = ALPHA / BONFERRONI_N
    hits = []
    for commodity, significance in significance_by_commodity.items():
        for h in horizons:
            for bucket, want_sign in (("worsening", 1.0), ("improving", -1.0)):
                test = significance[str(h)][bucket]
                if test and test["p_value"] < bar and (test["mean_diff_pct"] * want_sign) > 0:
                    hits.append({"commodity": commodity, "horizon": h, "bucket": bucket, **test})
    return {"bonferroni_bar": round(bar, 6), "passing_comparisons": hits, "PASSED": len(hits) > 0}


# ── Network ──────────────────────────────────────────────────────────────────

def fetch_trend(base_url: str, timeout: int = 30) -> list[dict]:
    url = base_url.rstrip("/") + HISTORY_URL_PATH
    req = urllib.request.Request(url, headers={"User-Agent": "voltradeai-datacore-gate2/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    trend = body.get("trend") or []
    if not trend:
        raise SystemExit(f"{url} returned no trend rows (body: {json.dumps(body)[:300]})")
    return trend


def run_gate2(base_url: str, horizons: Sequence[int] = HORIZONS) -> dict:
    from backtest_v2 import fetch_bars  # local import: keeps unit tests free of network/cache side effects

    trend = fetch_trend(base_url)
    earliest = datetime.strptime(trend[0]["week_ending"], "%Y-%m-%d")
    days_needed = (datetime.utcnow() - earliest).days + 120

    per_commodity = {}
    significance_by_commodity = {}
    for commodity, ticker in COMMODITY_TICKERS.items():
        deltas = compute_weekly_deltas(trend, commodity)
        bars = fetch_bars(ticker, days_needed)
        if not bars or not bars.get("date"):
            per_commodity[commodity] = {"ticker": ticker, "error": "no price data"}
            continue
        rows = compute_forward_returns(deltas, bars, horizons)
        summary = summarize(rows, horizons)
        significance = hac_significance(rows, horizons)
        significance_by_commodity[commodity] = significance
        per_commodity[commodity] = {
            "ticker": ticker,
            "n_weeks_in_archive": len(trend),
            "n_deltas": len(deltas),
            "n_improving": sum(1 for d in deltas if bucket_for(d["delta_ge_pct"]) == "improving"),
            "n_worsening": sum(1 for d in deltas if bucket_for(d["delta_ge_pct"]) == "worsening"),
            "n_flat_excluded": sum(1 for d in deltas if bucket_for(d["delta_ge_pct"]) is None),
            "summary": summary,
            "significance": significance,
            "rows": rows,
        }

    pass_bar = evaluate_pass_bar(significance_by_commodity, horizons)
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "base_url": base_url,
        "horizons": list(horizons),
        "publish_buffer_days": PUBLISH_BUFFER_DAYS,
        "per_commodity": per_commodity,
        "pass_bar": pass_bar,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="https://voltradeai.com")
    ap.add_argument("--out", default="crop_conditions_gate2_results.json")
    args = ap.parse_args()

    result = run_gate2(args.url)
    print(json.dumps(result, indent=2, default=str))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {args.out}")
    print(f"\n== GATE 2 VERDICT: {'PASS' if result['pass_bar']['PASSED'] else 'NOT PASSED'} "
          f"(bar={result['pass_bar']['bonferroni_bar']}, "
          f"{len(result['pass_bar']['passing_comparisons'])} comparison(s) hit) ==")
    return 0 if result["pass_bar"]["PASSED"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
