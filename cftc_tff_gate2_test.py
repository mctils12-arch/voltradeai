#!/usr/bin/env python3
"""
cftc_tff_gate2_test.py — ROOT VALIDATION LADDER gate 2 (SIGNAL) for the CFTC
Traders in Financial Futures archive (server/cftcTff.ts, gate 1 DATA passed
2026-07-31 — see research/experiments.md and datacore/signal_ladder.json's
cftc_tff_positioning entry, which names this exact test as its queued NEXT
STEP).

PRE-REGISTERED HYPOTHESIS (stated before running, REASONING STANDARD #10;
copied verbatim from the gate-1 entry's own NEXT note so this run cannot
silently drift into testing something else): leveraged-money net-positioning
extremes (the same 0-100 trailing-156-week COT-index transform already
proven on the Legacy-COT archive) predict a forward MEAN-REVERSION move in
the underlying financial future, beating a same-symbol random-entry
baseline. Dealer positioning ("the informed side" per cftcTff.ts's own
docstring) is a SEPARATE, un-pre-registered hypothesis and is deliberately
NOT tested here — running both in one pass would double the comparison
count (14 vs 7 symbols) without either being pre-registered ahead of time;
it is left as an explicit follow-up (see bottom of this docstring and the
experiments.md entry for this run).

WHY TFF NEEDS ITS OWN GATE-2 SCRIPT, NOT A cot_gate2_test.py RE-RUN: TFF
covers financial futures (equity index, US rates, FX, USD index) with a
FINANCIAL trader taxonomy (dealer/asset-manager/leveraged-money/other) —
structurally different markets and categories from Legacy COT's commodity-
heavy hedger/speculator split (cftc_cot.py's own docstring flags this exact
distinction as unresolved: "worth an A/B before believing legacy-report
results" on equity index/treasury contracts). The two archives share zero
overlapping-in-substance test results; this is a fresh gate-2 screen, not a
duplicate of the Legacy-COT one.

UNIVERSE: 7 liquid ETF proxies with a directly corresponding TFF-reported
futures contract (codes verified live against the real dataset this
session, not assumed from memory):
  SPY->E-MINI S&P 500 (13874A), QQQ->NASDAQ MINI (209742),
  TLT->UST BOND (020601), IEF->UST 10Y NOTE (043602),
  SHY->UST 2Y NOTE (042601), UUP->USD INDEX (098662, ICE),
  FXE->EURO FX (099741). Same symbol COUNT as the passed Legacy-COT screen
  (7) — chosen for direct futures<->ETF correspondence, not cherry-picked
  for a result.

METHODOLOGY mirrors cot_gate2_test.py exactly (same publish-lag/no-lookahead
entry rule, same COT-index transform, same extreme_high/low >=80/<=20
buckets, same forward 20d/60d horizons) with ONE deliberate improvement
(REASONING STANDARD #4): the Newey-West HAC significance test is applied
FROM THE START, not iterated to after an initial raw-means pass — the
Legacy-COT screen only added HAC after its first pass looked promising on
raw means, which is exactly the kind of look-then-decide sequencing this
run avoids by going straight to the corrected test.

Pure statistical measurement only — SIGNAL gate, no trading involved. Does
not import or touch bot_engine.py / deep_score / system_config.py.

Usage: python3 cftc_tff_gate2_test.py [--out cftc_tff_gate2_results.json]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta

import numpy as np
import requests
from scipy.stats import norm

DATASET_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
USER_AGENT = "VolTradeAI research@voltradeai.com"

# Verified live against the real dataset 2026-08-03 (this session) — exact
# cftc_contract_market_code + market_and_exchange_names cross-checked
# before writing this file, per READ BEFORE WRITE.
SYMBOLS = {
    "SPY": {"code": "13874A", "name": "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"},
    "QQQ": {"code": "209742", "name": "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE"},
    "TLT": {"code": "020601", "name": "UST BOND - CHICAGO BOARD OF TRADE"},
    "IEF": {"code": "043602", "name": "UST 10Y NOTE - CHICAGO BOARD OF TRADE"},
    "SHY": {"code": "042601", "name": "UST 2Y NOTE - CHICAGO BOARD OF TRADE"},
    "UUP": {"code": "098662", "name": "USD INDEX - ICE FUTURES U.S."},
    "FXE": {"code": "099741", "name": "EURO FX - CHICAGO MERCANTILE EXCHANGE"},
}

LOOKBACK_WEEKS = 156  # same 3-year convention as cftc_cot.py
HORIZONS = (20, 60)  # trading days, matches cot_gate2_test.py
EXTREME_HIGH = 80.0
EXTREME_LOW = 20.0
PUBLISH_LAG_DAYS = 3  # Tuesday as-of -> Friday publish, same as Legacy COT
_TOLERANCE = 5  # rare report-revision rounding artifact, mirrors cftc_cot.py


def _to_int(row, key):
    try:
        return int(float(row.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def validate_record(row):
    """GATE 1 (DATA), Python mirror of server/cftcTff.ts's tffAccountingIssues
    (already the live gate-1 validator gating the production archiver) —
    duplicated here so this standalone research script has no runtime
    dependency on the Node server. Verifies CFTC's own accounting identity
    for TFF's four financial trader categories (each with its own spread
    field, counted on BOTH the long and short leg):

      dealer_l+dealer_sp + asset_mgr_l+asset_mgr_sp + lev_money_l+lev_money_sp
        + other_rept_l+other_rept_sp        == tot_rept_positions_long_all
      (mirror on the short side, same spread values reused per leg)
      tot_rept_long + nonrept_long          == open_interest_all
      tot_rept_short + nonrept_short        == open_interest_all

    Returns (is_valid, reason)."""
    oi = _to_int(row, "open_interest_all")
    if oi <= 0:
        return False, "open_interest_all is zero/missing"

    dealer_long = _to_int(row, "dealer_positions_long_all")
    dealer_short = _to_int(row, "dealer_positions_short_all")
    dealer_spread = _to_int(row, "dealer_positions_spread_all")
    am_long = _to_int(row, "asset_mgr_positions_long")
    am_short = _to_int(row, "asset_mgr_positions_short")
    am_spread = _to_int(row, "asset_mgr_positions_spread")
    lev_long = _to_int(row, "lev_money_positions_long")
    lev_short = _to_int(row, "lev_money_positions_short")
    lev_spread = _to_int(row, "lev_money_positions_spread")
    other_long = _to_int(row, "other_rept_positions_long")
    other_short = _to_int(row, "other_rept_positions_short")
    other_spread = _to_int(row, "other_rept_positions_spread")
    tot_long = _to_int(row, "tot_rept_positions_long_all")
    tot_short = _to_int(row, "tot_rept_positions_short")
    nonrept_long = _to_int(row, "nonrept_positions_long_all")
    nonrept_short = _to_int(row, "nonrept_positions_short_all")

    calc_long = dealer_long + dealer_spread + am_long + am_spread + lev_long + lev_spread + other_long + other_spread
    calc_short = dealer_short + dealer_spread + am_short + am_spread + lev_short + lev_spread + other_short + other_spread

    checks = [
        (abs(calc_long - tot_long), "long-side reported total"),
        (abs(calc_short - tot_short), "short-side reported total"),
        (abs((tot_long + nonrept_long) - oi), "long-side open interest"),
        (abs((tot_short + nonrept_short) - oi), "short-side open interest"),
    ]
    for delta, label in checks:
        if delta > _TOLERANCE:
            return False, f"{label} mismatch by {delta}"
    return True, "ok"


def _derive_fields(row):
    lev_long = _to_int(row, "lev_money_positions_long")
    lev_short = _to_int(row, "lev_money_positions_short")
    oi = _to_int(row, "open_interest_all")
    net_lev_money = lev_long - lev_short
    return {
        "report_date": row.get("report_date_as_yyyy_mm_dd", "")[:10],
        "open_interest": oi,
        "lev_money_long": lev_long,
        "lev_money_short": lev_short,
        "net_lev_money": net_lev_money,
        "net_lev_money_pct_oi": round(net_lev_money / oi * 100, 2) if oi else 0.0,
    }


def fetch_symbol_history(contract_code, limit=LOOKBACK_WEEKS):
    """Fetch up to `limit` most recent weekly TFF reports for one CFTC
    contract code, directly from CFTC's Socrata API (same pattern as
    cftc_cot.py's fetch_symbol_history — no dependency on our own polling
    archive, which per the 2026-07-31 gate-1 entry has only ~4 weeks of
    history so far). Returns validated, derived records sorted oldest ->
    newest, plus a rejected-record count."""
    params = {
        "$where": f"cftc_contract_market_code='{contract_code}'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": limit,
    }
    resp = requests.get(
        DATASET_URL, params=params,
        headers={"User-Agent": USER_AGENT}, timeout=20,
    )
    resp.raise_for_status()
    rows = resp.json()

    out = []
    rejected = 0
    for row in rows:
        ok, reason = validate_record(row)
        if not ok:
            rejected += 1
            continue
        out.append(_derive_fields(row))
    out.sort(key=lambda r: r["report_date"])
    return out, rejected


def _cot_index(values, lookback):
    """Identical transform to cftc_cot.py's _cot_index: where the current
    reading sits (0-100) between the min/max of the trailing `lookback`
    window. Descriptive only until run() below tests it."""
    if not values:
        return None
    window = values[-lookback:]
    lo, hi = min(window), max(window)
    if hi == lo:
        return 50.0
    return round((values[-1] - lo) / (hi - lo) * 100, 1)


def attach_lev_money_index(records, lookback=LOOKBACK_WEEKS):
    series = []
    for rec in records:
        series.append(rec["net_lev_money_pct_oi"])
        rec["lev_money_index"] = _cot_index(series, lookback)
    return records


def find_entry_index(bar_dates: list, publish_date: str):
    """First bar strictly after `publish_date` (no lookahead) — byte-for-
    byte the same rule as cot_gate2_test.py's find_entry_index."""
    for i, d in enumerate(bar_dates):
        if d > publish_date:
            return i
    return None


def bucket_for(index_value):
    if index_value is None:
        return None
    if index_value >= EXTREME_HIGH:
        return "extreme_high"
    if index_value <= EXTREME_LOW:
        return "extreme_low"
    return "mid"


def compute_forward_returns(tff_records, bars):
    """Pure function: for each TFF week, find its no-lookahead entry anchor
    in `bars` and compute forward N-day returns. Weeks too close to the end
    of `bars` for a given horizon are dropped for that horizon (right-
    censoring honesty), never zero-filled."""
    bar_dates = bars["date"]
    bar_closes = bars["close"]
    out = []
    for rec in tff_records:
        report_date = rec["report_date"]
        publish_date = (datetime.strptime(report_date, "%Y-%m-%d") +
                         timedelta(days=PUBLISH_LAG_DAYS)).strftime("%Y-%m-%d")
        entry_idx = find_entry_index(bar_dates, publish_date)
        row = {
            "report_date": report_date,
            "lev_money_index": rec.get("lev_money_index"),
            "bucket": bucket_for(rec.get("lev_money_index")),
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


def _newey_west_diff_test(rows, horizon, bucket, lag=None):
    """HAC (Newey-West, Bartlett-kernel) test of whether `bucket` weeks'
    forward return differs from the COMPLEMENT (non-bucket weeks) at this
    horizon. Byte-for-byte the same construction as cot_gate2_test.py's
    _newey_west_diff_test (OLS of forward return on a 0/1 bucket-dummy,
    Newey-West sandwich variance with lag = round(horizon / 5) weeks) —
    see that function's docstring for the full derivation and the
    baseline-definition caveat vs summarize(). Returns None (never a
    fabricated number) if there are too few observations or the bucket
    dummy is degenerate."""
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


def hac_significance(rows):
    """Per-horizon, per-extreme-bucket Newey-West test vs the complement."""
    out = {}
    for h in HORIZONS:
        out[str(h)] = {
            bucket: _newey_west_diff_test(rows, h, bucket)
            for bucket in ("extreme_high", "extreme_low")
        }
    return out


def run(symbol, fetch_bars_fn):
    meta = SYMBOLS[symbol]
    tff_records, rejected = fetch_symbol_history(meta["code"], limit=LOOKBACK_WEEKS)
    tff_records = attach_lev_money_index(tff_records)
    if not tff_records:
        return {"symbol": symbol, "status": "no_tff_data"}

    earliest = datetime.strptime(tff_records[0]["report_date"], "%Y-%m-%d")
    days_needed = (datetime.utcnow() - earliest).days + 30
    bars = fetch_bars_fn(symbol, days_needed)
    if not bars or not bars.get("date"):
        return {"symbol": symbol, "status": "no_price_data"}

    rows = compute_forward_returns(tff_records, bars)
    return {
        "symbol": symbol,
        "status": "ok",
        "weeks": len(tff_records),
        "rejected_tff_records": rejected,
        "summary": summarize(rows),
        "significance": hac_significance(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cftc_tff_gate2_results.json")
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
