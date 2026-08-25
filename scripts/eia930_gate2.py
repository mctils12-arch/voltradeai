#!/usr/bin/env python3
"""eia930_gate2.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for the EIA-930
grid demand archive (server/gridDemand.ts, gate 1 DATA passed 2026-07-06 —
see research/experiments.md and datacore/signal_ladder.json's
eia930_grid_demand entry).

PRE-REGISTERED HYPOTHESIS (stated before running, REASONING STANDARD #10;
copied verbatim from gridDemand.ts's own docstring, which names this exact
test as its gate-2 plan): "weather-adjusted regional demand residuals ...
nowcast industrial activity ... gate 2 = demand residual vs industrial-
sector returns." Operationalized here as: US48 daily electricity demand,
after removing the portion explained by weather (degree days) and ordinary
calendar seasonality, predicts forward returns in XLI (Industrial Select
Sector SPDR — the direct, literal "industrial sector" ETF, chosen because
it is the one proxy the hypothesis names, not selected after looking at
candidates). Extra demand beyond what weather explains is read as a signal
of extra industrial/economic activity (bullish for industrials); a
weather-adjusted demand shortfall reads bearish.

UNIVERSE: single symbol (XLI) — deliberately not a multi-ETF screen. The
hypothesis as stated in the source module is about ONE named sector, so a
single test is the correct pre-registration, not an underpowered N=1
convenience sample; testing multiple unrelated sector ETFs here would be
exactly the kind of after-the-fact fishing REASONING STANDARD #4 warns
against.

DATA:
  - EIA-930 US48 total demand, daily, Eastern-timezone convention (EIA's
    own "Data Date" boundary; the API also serves Arizona/Central/Mountain/
    Pacific variants of the same physical demand — Eastern picked once,
    up front, as the canonical convention; the four alternates were not
    tried and compared). api.eia.gov v2 electricity/rto/daily-region-data,
    verified live this session: 2019-01-01 -> 2026-08-07, 2776 daily rows,
    zero gaps, one HTTP call (under the 5000-row JSON cap).
  - NOAA CPC population-weighted degree days (datacore/cpc/degree_days.json,
    scripts/cpc_degree_days.py, refreshed this session to 2026-08-06).
    National-level HDD/CDD = an EQUAL-weighted mean across the 48 CONUS
    state series. HONESTY CAVEAT: each state series is itself population-
    weighted internally (NOAA's own methodology), but this script does not
    further population-weight ACROSS states for the national aggregate
    (e.g. California and Wyoming count the same) — a documented
    approximation, not a claim of a true national population-weighted
    figure. If gate 2 passes, a future session should test whether a
    correctly state-population-weighted national series changes the
    result before this moves toward LOGIC (gate 3).

METHOD (no lookahead at any step):
  1. Regress log(demand) on HDD, CDD, 6 weekday dummies, and 11 month
     dummies over a TRAINING window (2019-01-01 .. 2021-12-31, 3 full
     years, ~1096 obs for 19 predictors + intercept) via OLS. Coefficients
     are frozen after this step and never refit on later data.
  2. Apply the frozen coefficients to EVERY day 2019-01-01 -> present to
     get a residual = actual log(demand) - predicted log(demand). Days
     before the VALIDATION window only exist to seed the trailing
     percentile lookback in step 3 (same role the pre-existing COT-index
     trailing window plays elsewhere in this codebase) — they are never
     used to test forward returns.
  3. Smooth: 10-trading-day trailing mean of the residual (index reads
     high when demand has run persistently above what weather+calendar
     predicts, not on a single noisy day). Trailing PERCENTILE: for each
     day in the VALIDATION window, rank that day's smoothed residual
     against the trailing 504 calendar days (~2yr) of smoothed residual
     history — a causal, expanding/rolling percentile exactly analogous
     to _cot_index's trailing-window transform used in the CFTC gate-2
     scripts, never a look-ahead full-sample percentile.
  4. VALIDATION window: 2022-01-01 -> latest day with both demand and
     weather data (2026-08-06) — fully out-of-sample vs the regression
     fit in step 1.
  5. ENTRY CADENCE: weekly, not daily — the last trading day of each ISO
     week's residual index value (matches the weekly cadence of every
     prior gate-2 script in this repo, and daily entries at a 20-60
     trading-day horizon would overlap almost completely, making the
     effective sample size far smaller than the raw day count suggests).
  6. BUCKETS: extreme_high >= 90th trailing percentile, extreme_low <=
     10th (top/bottom decile, matching grid_stress_gate2.py's decile
     framing). Entry = first XLI trading day strictly after the index
     date (no lookahead). HORIZONS: 20 and 60 trading days (same
     convention as every other gate-2 script here).
  7. SIGNIFICANCE: Newey-West HAC t-test of each bucket's forward return
     vs the complement, OLS-with-dummy construction identical to
     cftc_tff_gate2_test.py's _newey_west_diff_test, EXCEPT the lag: prior
     gate-2 scripts use weekly-cadence CFTC data with lag = round(h/5)
     (approximating the number of *weekly* reports spanned by a horizon
     of h *trading* days). This script's residual index is also sampled
     weekly (step 5), so the same lag = round(h/5) convention applies
     unchanged -- explicitly checked before writing this script, not
     assumed.
  8. PRE-STATED PASS BAR: both horizons must show the predicted-direction
     sign (extreme_high mean forward return > extreme_low mean forward
     return) AND at least one horizon must clear a Bonferroni-corrected
     p < 0.0125 (0.05 / 4 comparisons: 2 buckets x 2 horizons, same
     correction style as cftc_tff_gate2_test.py). FAIL: sign disagreement
     between horizons, or no comparison clears the bar -> layer of death
     logged as SIGNAL (gate 2), RAW /api/data/grid-demand display
     unaffected (raw overlays carry no predictive claim).

Pure statistical measurement only -- SIGNAL gate, no trading involved. Does
not import or touch bot_engine.py / deep_score / system_config.py.

Usage: python3 scripts/eia930_gate2.py [--out eia930_gate2_results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import date, datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from gate2_stats import find_entry_index, newey_west_diff_test as _newey_west_diff_test  # noqa: E402

EIA_URL = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
DEGREE_DAYS_PATH = os.path.join(os.path.dirname(__file__), "..", "datacore", "cpc", "degree_days.json")

TRAIN_START = "2019-01-01"
TRAIN_END = "2021-12-31"
VALIDATION_START = "2022-01-01"

HORIZONS = (20, 60)  # trading days
EXTREME_HIGH = 90.0
EXTREME_LOW = 10.0
SMOOTH_WINDOW = 10       # trailing days averaged into the residual index
PERCENTILE_LOOKBACK = 504  # ~2 trading years
SYMBOL = "XLI"

# Sanity bound for a US48 DAILY demand total (sum of 24 hourly readings).
# Real US48 peak is ~700-750GW hourly -> a physically impossible daily
# total is orders of magnitude outside this; matches the spirit of
# gridDemand.ts's own DEMAND_BOUNDS but at the daily-aggregate grain
# (that check is hourly and does not apply directly to this series).
DAILY_DEMAND_BOUNDS_MWH = (3_000_000, 30_000_000)


def fetch_eia930_daily(start: str, end: str) -> dict[str, float]:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise SystemExit("EIA_API_KEY not set in environment")
    url = (
        f"{EIA_URL}?api_key={key}&frequency=daily&data[0]=value"
        "&facets[respondent][]=US48&facets[type][]=D&facets[timezone][]=Eastern"
        f"&start={start}&end={end}"
        "&sort[0][column]=period&sort[0][direction]=asc&length=5000"
    )
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.load(r)
    if payload.get("warnings"):
        raise SystemExit(f"EIA API returned warnings (row cap likely exceeded): {payload['warnings']}")
    out: dict[str, float] = {}
    rejected = 0
    for row in payload["response"]["data"]:
        try:
            v = float(row["value"])
        except (TypeError, ValueError, KeyError):
            rejected += 1
            continue
        if not (DAILY_DEMAND_BOUNDS_MWH[0] <= v <= DAILY_DEMAND_BOUNDS_MWH[1]):
            rejected += 1
            continue
        out[row["period"]] = v
    print(f"EIA-930 US48 daily: {len(out)} valid days, {rejected} rejected (bounds/parse)")
    return out


def load_national_degree_days() -> dict[str, tuple[float, float]]:
    """Returns {date: (HDD, CDD)} — equal-weighted mean across all 48 CONUS
    state series in datacore/cpc/degree_days.json. See module docstring for
    the "not further population-weighted across states" honesty caveat."""
    with open(DEGREE_DAYS_PATH) as f:
        raw = json.load(f)
    h_dates = raw["axes"]["H"]
    c_dates = raw["axes"]["C"]
    state_h = [k for k in raw["series"] if k.endswith("|H")]
    state_c = [k for k in raw["series"] if k.endswith("|C")]
    if not state_h or not state_c:
        raise SystemExit("degree_days.json has no state H/C series — run scripts/cpc_degree_days.py first")

    def _mean_by_date(dates, keys):
        n = len(keys)
        acc: dict[str, float] = {}
        for d_idx, d in enumerate(dates):
            total = 0.0
            count = 0
            for k in keys:
                vals = raw["series"][k]
                if d_idx < len(vals) and vals[d_idx] is not None:
                    total += vals[d_idx]
                    count += 1
            if count >= n * 0.9:  # require near-complete cross-section for that day
                acc[d] = total / count
        return acc

    h_by_date = _mean_by_date(h_dates, state_h)
    c_by_date = _mean_by_date(c_dates, state_c)
    common = set(h_by_date) & set(c_by_date)
    return {d: (h_by_date[d], c_by_date[d]) for d in common}


def build_design_matrix(dates: list[str], dd: dict[str, tuple[float, float]]):
    """[intercept, HDD, CDD, weekday(6 dummies, Mon baseline), month(11
    dummies, Jan baseline)] per date. Returns (X, valid_dates) — dates
    missing degree-day coverage are dropped, never zero-filled."""
    rows = []
    valid_dates = []
    for d in dates:
        if d not in dd:
            continue
        hdd, cdd = dd[d]
        dt = datetime.strptime(d, "%Y-%m-%d")
        weekday = dt.weekday()  # 0=Mon
        month = dt.month  # 1-12
        row = [1.0, hdd, cdd]
        row += [1.0 if weekday == i else 0.0 for i in range(1, 7)]
        row += [1.0 if month == m else 0.0 for m in range(2, 13)]
        rows.append(row)
        valid_dates.append(d)
    return np.array(rows, dtype=float), valid_dates


def fit_and_residualize(demand: dict[str, float], dd: dict[str, tuple[float, float]]):
    all_dates = sorted(set(demand) & set(dd))
    X_all, dates_all = build_design_matrix(all_dates, dd)
    y_all = np.array([np.log(demand[d]) for d in dates_all])

    train_mask = np.array([TRAIN_START <= d <= TRAIN_END for d in dates_all])
    n_train = int(train_mask.sum())
    if n_train < 200:
        raise SystemExit(f"training window too small ({n_train} days) — check date coverage")

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

    resid_all = y_all - X_all @ beta
    resid_by_date = {d: float(r) for d, r in zip(dates_all, resid_all)}
    return resid_by_date, n_train, beta.tolist()


def smooth_and_percentile(resid_by_date: dict[str, float]):
    dates = sorted(resid_by_date)
    vals = [resid_by_date[d] for d in dates]
    smoothed = []
    for i in range(len(vals)):
        lo = max(0, i - SMOOTH_WINDOW + 1)
        smoothed.append(sum(vals[lo:i + 1]) / (i - lo + 1))

    pct = [None] * len(vals)
    for i in range(len(vals)):
        lo = max(0, i - PERCENTILE_LOOKBACK + 1)
        window = smoothed[lo:i + 1]
        if len(window) < 60:  # need a meaningful trailing sample before scoring
            continue
        rank = sum(1 for w in window if w <= smoothed[i])
        pct[i] = 100.0 * rank / len(window)

    return {d: {"resid": vals[i], "smoothed": smoothed[i], "pctile": pct[i]} for i, d in enumerate(dates)}


def weekly_entries(index_by_date: dict[str, dict], validation_start: str):
    """Last available index date of each ISO week within the validation
    window -- the weekly entry cadence stated in the pre-registration."""
    dates = sorted(d for d in index_by_date if d >= validation_start and index_by_date[d]["pctile"] is not None)
    by_week: dict[tuple[int, int], str] = {}
    for d in dates:
        iso = datetime.strptime(d, "%Y-%m-%d").isocalendar()
        by_week[(iso[0], iso[1])] = d  # later date in the same week overwrites -> last
    return [by_week[k] for k in sorted(by_week)]


def bucket_for(pctile):
    if pctile is None:
        return None
    if pctile >= EXTREME_HIGH:
        return "extreme_high"
    if pctile <= EXTREME_LOW:
        return "extreme_low"
    return "mid"


def compute_forward_returns(entry_dates: list[str], index_by_date: dict, bars: dict):
    bar_dates, bar_closes = bars["date"], bars["close"]
    out = []
    for d in entry_dates:
        pctile = index_by_date[d]["pctile"]
        entry_idx = find_entry_index(bar_dates, d)
        row = {
            "index_date": d,
            "pctile": pctile,
            "bucket": bucket_for(pctile),
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


def summarize(rows):
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
            bucket: {"n": len(v), "mean_pct": round(sum(v) / len(v) * 100, 3) if v else None}
            for bucket, v in vals.items()
        }
    return summary


def hac_significance(rows):
    return {
        str(h): {bucket: _newey_west_diff_test(rows, h, bucket) for bucket in ("extreme_high", "extreme_low")}
        for h in HORIZONS
    }


def evaluate_pass_bar(summary, significance):
    """Applies the pre-stated PASS bar (see module docstring step 8)."""
    sign_ok = {}
    for h in HORIZONS:
        hi = summary[str(h)]["extreme_high"]["mean_pct"]
        lo = summary[str(h)]["extreme_low"]["mean_pct"]
        sign_ok[str(h)] = (hi is not None and lo is not None and hi > lo)
    any_bonferroni = False
    for h in HORIZONS:
        for bucket in ("extreme_high", "extreme_low"):
            test = significance[str(h)][bucket]
            if test and test["p_value"] < 0.0125:
                any_bonferroni = True
    passed = all(sign_ok.values()) and any_bonferroni
    return {
        "sign_agrees_both_horizons": sign_ok,
        "any_comparison_clears_bonferroni_0.0125": any_bonferroni,
        "PASSED": passed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eia930_gate2_results.json")
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args()

    from backtest_v2 import fetch_bars  # local import: keeps this script's
    # unit tests free of backtest_v2's network/cache side effects

    demand = fetch_eia930_daily(TRAIN_START, args.end)
    dd = load_national_degree_days()
    resid_by_date, n_train, beta = fit_and_residualize(demand, dd)
    index_by_date = smooth_and_percentile(resid_by_date)
    entries = weekly_entries(index_by_date, VALIDATION_START)
    print(f"training days: {n_train}, validation weekly entries: {len(entries)}")

    earliest = datetime.strptime(entries[0], "%Y-%m-%d") if entries else datetime.strptime(VALIDATION_START, "%Y-%m-%d")
    days_needed = (datetime.utcnow() - earliest).days + 120
    bars = fetch_bars(SYMBOL, days_needed)
    if not bars or not bars.get("date"):
        raise SystemExit(f"no price data for {SYMBOL}")

    rows = compute_forward_returns(entries, index_by_date, bars)
    summary = summarize(rows)
    significance = hac_significance(rows)
    pass_bar = evaluate_pass_bar(summary, significance)

    result = {
        "symbol": SYMBOL,
        "training_window": [TRAIN_START, TRAIN_END],
        "validation_window": [VALIDATION_START, args.end],
        "n_training_days": n_train,
        "n_weekly_entries": len(entries),
        "regression_beta": beta,
        "summary": summary,
        "significance": significance,
        "pass_bar": pass_bar,
    }
    print(json.dumps(result, indent=2, default=str))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
