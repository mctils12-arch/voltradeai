#!/usr/bin/env python3
"""
critical_slowing_down_probe.py — FOREIGN-FIELD IMPORT (ecology / resilience
theory) tested as a backtestable hypothesis per CLAUDE.md's EDGE DOCTRINE #4
and the ACTIVE ANGLE-HUNTING standing behavior. Pure [RESEARCH] — no trading
path touched, no parameter changed, no production import.

THE IMPORTED TECHNIQUE: "critical slowing down" (Scheffer et al. 2009,
Nature, "Early-warning signals for critical transitions") is ecology's
standard early-warning toolkit for detecting an approaching regime shift
(lake eutrophication, ecosystem collapse) BEFORE it happens, without
knowing the underlying mechanism. As a dynamical system nears a tipping
point, its recovery rate from small perturbations slows — which shows up
statistically as RISING LAG-1 AUTOCORRELATION and RISING VARIANCE in the
system's own fluctuations, ahead of the transition itself.

WHY THIS QUESTION, WHY THIS METHOD (REASONING STANDARD #10 — prior stated
before running): this system's OWN regime-detector research
(open_questions.md, 2026-07-09, regime_detector_compare.py) already found
that the VXX-ratio-style signal (a LEVEL measure of current implied vol)
does almost all of the forward-volatility-prediction work, and the Markov
chain component (a return-DIRECTION classifier) adds essentially nothing.
Neither existing signal measures the system's APPROACH to instability —
they measure where it already IS. Critical-slowing-down indicators are
explicitly a LEADING statistic (rising autocorrelation should show up
BEFORE the vol spike that eventually moves VIX), so the pre-registered
PRIOR is: trailing-window return autocorrelation should (a) correlate with
FORWARD realized volatility, similarly to how vix_ratio does, and (b) lead
vix_ratio's own rise in a lead-lag cross-correlation — i.e. autocorrelation
today should correlate more strongly with vix_ratio some days IN THE
FUTURE than with vix_ratio today. If (b) fails (correlation peaks at lag
0, or autocorrelation only ever lags vix_ratio), the "early warning"
framing is not supported by this data and the honest conclusion is that
autocorrelation is at best a coincident restatement of existing volatility
info, not a genuinely leading signal.

SECOND-ORDER TEST (REASONING STANDARD #5): critical slowing down is a
PUBLISHED, well-known technique — including prior academic attempts to
apply it to financial markets (e.g. interbank/LIBOR stress, FX). This
script does not claim to have found something nobody has ever tried. The
residual, non-arbitraged angle this session is actually testing is
narrower and stated honestly: does it add INCREMENTAL predictive power on
top of the volatility-LEVEL signal this system already computes and
trades on (vix_ratio), for THIS system's own signal-gating purposes? A
large fund publishing an index-level critical-slowing-down paper is not
the same claim as "this specific small system's regime gate should also
consume it" — the bar cleared here is architecture quality (a better
regime input), not novel macro alpha. Discounted accordingly: this is
ONE variant tested (20-day window, lag-1 autocorrelation, SP500-proxy),
not a swept parameter grid, so a positive result here is a candidate for
a SECOND out-of-sample confirmation before being trusted, per REASONING
STANDARD #4 — not a ship-it verdict on one pass.

HONEST DATA SUBSTITUTION (same as regime_detector_compare.py, its direct
sibling and dependency): live SPY/VXX are unreachable from this sandbox;
FRED's public, keyless fredgraph.csv serves SP500 (index proxy for SPY)
and VIXCLS (spot-VIX proxy for VXX) instead. This script imports
`parse_fred_csv_text`/`load_fred_csv_file`/`align_series` from
regime_detector_compare.py rather than re-implementing the fetch/parse
path, so both scripts share one source of truth for "how FRED data enters
this repo's research tooling."

Usage:
    curl -sS "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500" -o sp500.csv
    curl -sS "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS" -o vixcls.csv
    python3 critical_slowing_down_probe.py --sp500-csv sp500.csv --vix-csv vixcls.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_detector_compare import (  # noqa: E402  (reuse the one fetch/parse/align path)
    align_series,
    load_fred_csv_file,
    fetch_fred_series,
)

AUTOCORR_WINDOW = 20  # trailing days of returns used to compute rolling lag-1 autocorrelation / variance
LEAD_LAG_HORIZONS = (0, 5, 10, 20)  # trading days ahead, for the "does autocorr lead vix_ratio" test


def rolling_autocorr_lag1(returns: np.ndarray, window: int) -> np.ndarray:
    """Pure function: for each index i, lag-1 autocorrelation of
    returns[i-window+1 .. i] (needs window+1 real values, since lag-1
    pairing consumes one). NaN where there isn't enough trailing history
    or the window is constant (zero variance -> undefined correlation)."""
    n = len(returns)
    out = np.full(n, np.nan)
    for i in range(n):
        start = i - window + 1
        if start < 1:  # need one extra point before `start` for the lag-1 pair
            continue
        seg = returns[start - 1:i + 1]
        if len(seg) < window + 1:
            continue
        x, y = seg[:-1], seg[1:]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        out[i] = float(np.corrcoef(x, y)[0, 1])
    return out


def rolling_std(returns: np.ndarray, window: int) -> np.ndarray:
    """Pure function: trailing sample stdev of returns[i-window+1 .. i]."""
    n = len(returns)
    out = np.full(n, np.nan)
    for i in range(n):
        start = i - window + 1
        if start < 0:
            continue
        seg = returns[start:i + 1]
        if len(seg) < window:
            continue
        out[i] = float(np.std(seg, ddof=1))
    return out


def compute_features(dates: list[str], spy: list[float], vix: list[float],
                      window: int = AUTOCORR_WINDOW) -> list[dict]:
    """Pure function: given aligned daily closes, compute per-day features.
    No lookahead — every trailing feature at index i uses only data up to
    and including i; every forward feature uses only i+1 onward."""
    n = len(dates)
    spy_arr = np.array(spy, dtype=float)
    vix_arr = np.array(vix, dtype=float)

    spy_ret = np.full(n, np.nan)
    spy_ret[1:] = (spy_arr[1:] / spy_arr[:-1] - 1.0) * 100.0

    trailing_autocorr = rolling_autocorr_lag1(spy_ret, window)
    trailing_vol = rolling_std(spy_ret, window)

    warmup = window + 30  # window+1 for autocorr's own lag pair, 30 for vix_ratio's trailing avg (round up)
    rows = []
    for i in range(n):
        if i < warmup or i >= n - 20:
            continue  # need trailing history for autocorr/vol/vix_ratio, 20d forward for fwd_vol_20

        vix_avg30 = float(np.mean(vix_arr[i - 29:i + 1]))
        vix_ratio = float(vix_arr[i]) / vix_avg30 if vix_avg30 else np.nan

        fwd5 = spy_ret[i + 1:i + 6]
        fwd20 = spy_ret[i + 1:i + 21]
        fwd_vol_5 = float(np.std(fwd5, ddof=1)) if np.all(~np.isnan(fwd5)) and len(fwd5) == 5 else None
        fwd_vol_20 = float(np.std(fwd20, ddof=1)) if np.all(~np.isnan(fwd20)) and len(fwd20) == 20 else None

        ac = trailing_autocorr[i]
        tv = trailing_vol[i]
        rows.append({
            "date": dates[i],
            "trailing_autocorr": float(ac) if not np.isnan(ac) else None,
            "trailing_vol": float(tv) if not np.isnan(tv) else None,
            "vix_ratio": vix_ratio,
            "fwd_vol_5": fwd_vol_5,
            "fwd_vol_20": fwd_vol_20,
        })
    return rows


def _clean_pairs(xs: list, ys: list) -> tuple[list[float], list[float]]:
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None and not (isinstance(x, float) and np.isnan(x))]
    if not pairs:
        return [], []
    xa, ya = zip(*pairs)
    return list(xa), list(ya)


def _spearman(xs: list, ys: list) -> dict:
    xa, ya = _clean_pairs(xs, ys)
    if len(xa) < 10:
        return {"n": len(xa), "rho": None, "p": None}
    rho, p = spearmanr(xa, ya)
    return {"n": len(xa), "rho": round(float(rho), 4), "p": round(float(p), 6)}


def compare_signals(rows: list[dict]) -> dict:
    """Pure function: three tests against the pre-registered PRIOR.
    (1) does trailing_autocorr correlate with forward realized vol, same
        shape as regime_detector_compare.py's existing vix_ratio test, so
        the two are directly comparable;
    (2) lead-lag: does trailing_autocorr(t) correlate with vix_ratio(t+k)
        more strongly at some k>0 than at k=0 — the actual "early warning"
        claim, not just "these two co-move";
    (3) conditional: does trailing_autocorr still correlate with forward
        vol WITHIN each vix_ratio tercile — i.e. does it add information
        beyond current vol LEVEL, or is it just a restatement of it."""
    out: dict = {"n": len(rows), "window": AUTOCORR_WINDOW}

    predictive = {}
    for sig_name in ("trailing_autocorr", "trailing_vol", "vix_ratio"):
        sig_vals = [r[sig_name] for r in rows]
        predictive[sig_name] = {
            "fwd_vol_5": _spearman(sig_vals, [r["fwd_vol_5"] for r in rows]),
            "fwd_vol_20": _spearman(sig_vals, [r["fwd_vol_20"] for r in rows]),
        }
    out["predictive_power"] = predictive

    n = len(rows)
    autocorr_vals = [r["trailing_autocorr"] for r in rows]
    lead_lag = {}
    for k in LEAD_LAG_HORIZONS:
        if k == 0:
            vix_shifted = [r["vix_ratio"] for r in rows]
        else:
            vix_shifted = [rows[i + k]["vix_ratio"] if i + k < n else None for i in range(n)]
        lead_lag[f"lag_{k}"] = _spearman(autocorr_vals, vix_shifted)
    out["lead_lag_autocorr_vs_future_vix_ratio"] = lead_lag

    vix_vals = [r["vix_ratio"] for r in rows if r["vix_ratio"] is not None and not np.isnan(r["vix_ratio"])]
    conditional = {}
    if len(vix_vals) >= 30:
        terciles = np.percentile(vix_vals, [33.33, 66.67])
        buckets = {"low_vix_ratio": [], "mid_vix_ratio": [], "high_vix_ratio": []}
        for r in rows:
            v = r["vix_ratio"]
            if v is None or np.isnan(v):
                continue
            key = "low_vix_ratio" if v <= terciles[0] else ("high_vix_ratio" if v > terciles[1] else "mid_vix_ratio")
            buckets[key].append(r)
        for name, bucket_rows in buckets.items():
            conditional[name] = {
                "n": len(bucket_rows),
                "fwd_vol_5": _spearman([r["trailing_autocorr"] for r in bucket_rows],
                                        [r["fwd_vol_5"] for r in bucket_rows]),
                "fwd_vol_20": _spearman([r["trailing_autocorr"] for r in bucket_rows],
                                         [r["fwd_vol_20"] for r in bucket_rows]),
            }
    out["conditional_on_vix_ratio_tercile"] = conditional

    for hz_name in ("fwd_vol_5", "fwd_vol_20"):
        clean = [r[hz_name] for r in rows if r[hz_name] is not None]
        out[f"{hz_name}_base_mean"] = round(float(np.mean(clean)), 4) if clean else None
        out[f"{hz_name}_base_std"] = round(float(np.std(clean)), 4) if clean else None
    return out


def run(sp500_csv: Optional[str] = None, vix_csv: Optional[str] = None,
        offline_json: Optional[str] = None) -> tuple[dict, list[dict]]:
    if offline_json:
        with open(offline_json) as f:
            cached = json.load(f)
        dates, spy, vix = cached["dates"], cached["spy"], cached["vix"]
    elif sp500_csv and vix_csv:
        spy_d = load_fred_csv_file(sp500_csv)
        vix_d = load_fred_csv_file(vix_csv)
        dates, spy, vix = align_series(spy_d, vix_d)
    else:
        spy_d = fetch_fred_series("SP500")
        vix_d = fetch_fred_series("VIXCLS")
        dates, spy, vix = align_series(spy_d, vix_d)
    rows = compute_features(dates, spy, vix)
    result = compare_signals(rows)
    result["date_range"] = [rows[0]["date"], rows[-1]["date"]] if rows else None
    result["fetched_at"] = datetime.utcnow().isoformat() + "Z"
    return result, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp500-csv", help="path to a pre-fetched fredgraph.csv?id=SP500 file")
    ap.add_argument("--vix-csv", help="path to a pre-fetched fredgraph.csv?id=VIXCLS file")
    ap.add_argument("--offline", help="path to a cached {dates,spy,vix} JSON instead of CSVs or a live fetch")
    ap.add_argument("--save-raw", help="optional path to dump the aligned {dates,spy,vix} JSON (for --offline reruns)")
    args = ap.parse_args()

    if args.save_raw and not args.offline:
        spy_d = load_fred_csv_file(args.sp500_csv) if args.sp500_csv else fetch_fred_series("SP500")
        vix_d = load_fred_csv_file(args.vix_csv) if args.vix_csv else fetch_fred_series("VIXCLS")
        dates, spy, vix = align_series(spy_d, vix_d)
        with open(args.save_raw, "w") as f:
            json.dump({"dates": dates, "spy": spy, "vix": vix}, f)
        print(f"Saved aligned raw series to {args.save_raw}", file=sys.stderr)

    result, _ = run(sp500_csv=args.sp500_csv, vix_csv=args.vix_csv, offline_json=args.offline)
    print(json.dumps(result, indent=2))
