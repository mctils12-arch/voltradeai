#!/usr/bin/env python3
"""
cftc_tff_tlt_disjoint_replication.py — LADDER PATH step (1) for the TLT
leveraged-money-positioning MOMENTUM candidate (open_questions.md, filed
2026-08-03, "TLT (UST BOND) leveraged-money net-positioning extremes show
MOMENTUM continuation at 20d, not mean-reversion").

cftc_tff_gate2_test.py's 7-symbol, pre-registered mean-reversion screen was
REJECTED at gate 2 overall, but TLT alone showed a strong, Bonferroni-
surviving effect in the OPPOSITE (continuation) direction on the single
2023-08..2026-07 sample window (156 weeks). REASONING STANDARD #2/#4: a
one-window result over one rate cycle is unconfirmed until it survives a
window it was not found on. This script re-runs the IDENTICAL TLT-only
construction — same COT-index transform, same extreme_high/low >=80/<=20
buckets, same 20d/60d horizons, same Newey-West HAC significance test,
same LOOKBACK_WEEKS=156 — on a DISJOINT window that shares zero weeks with
the original. The 2023-08..2026-07 window is never reused as its own
confirmation.

WINDOW CHOICE: the 156 weeks immediately PRECEDING the original window's
start (cutoff 2023-08-01, exclusive), i.e. roughly 2020-08..2023-08. Chosen
over reaching back to TFF's 2009 inception because it is directly adjacent
and spans a genuinely different rate regime (COVID zero-rate policy through
the fastest Fed hiking cycle in decades) from the original window's
post-hike/cutting regime — a real out-of-sample test of REASONING STANDARD
#2's "regime that dominated the sample" concern, not just more of the same
tape.

Pure statistical measurement only — SIGNAL gate, no trading involved. Does
not import or touch bot_engine.py / deep_score / system_config.py.

Usage: python3 cftc_tff_tlt_disjoint_replication.py [--out FILE] [--window-end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

from cftc_tff_gate2_test import (
    SYMBOLS,
    attach_lev_money_index,
    compute_forward_returns,
    fetch_symbol_history_range,
    hac_significance,
    summarize,
)

SYMBOL = "TLT"
WINDOW_END = "2023-08-01"  # exclusive cutoff = the original window's start


def run(fetch_bars_fn, window_end=WINDOW_END):
    meta = SYMBOLS[SYMBOL]
    tff_records, rejected = fetch_symbol_history_range(meta["code"], window_end)
    tff_records = attach_lev_money_index(tff_records)
    if not tff_records:
        return {"symbol": SYMBOL, "status": "no_tff_data"}

    earliest = datetime.strptime(tff_records[0]["report_date"], "%Y-%m-%d")
    days_needed = (datetime.utcnow() - earliest).days + 30
    bars = fetch_bars_fn(SYMBOL, days_needed)
    if not bars or not bars.get("date"):
        return {"symbol": SYMBOL, "status": "no_price_data"}

    rows = compute_forward_returns(tff_records, bars)
    return {
        "symbol": SYMBOL,
        "status": "ok",
        "window": {
            "start": tff_records[0]["report_date"],
            "end": tff_records[-1]["report_date"],
        },
        "weeks": len(tff_records),
        "rejected_tff_records": rejected,
        "summary": summarize(rows),
        "significance": hac_significance(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cftc_tff_tlt_disjoint_gate2_results.json")
    ap.add_argument("--window-end", default=WINDOW_END)
    args = ap.parse_args()

    from backtest_v2 import fetch_bars  # local import: keeps this script's
    # unit tests free of backtest_v2's network/cache side effects

    result = run(fetch_bars, args.window_end)
    print(json.dumps(result, indent=2, default=str))

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
