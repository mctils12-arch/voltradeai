#!/usr/bin/env python3
"""
cot_gate2_oos.py — out-of-sample POWER check for the cot_gate2_test.py carry.

cot_gate2_test.py's only surviving carry is USO extreme_high -> lower forward
return (HAC p=0.0003 @20d, 0.0107 @60d, in-sample, fails Bonferroni). The
signal_ladder readiness trigger waited for a COUNT OF WEEKLY REPORTS after
2026-07-08, but a report only tests the carry if the index sits in the
extreme bucket that week. This script counts those weeks (and the independent
EPISODES they form — consecutive extreme weeks share overlapping forward
windows, so they are one observation) and refuses to call a verdict below
MIN_EPISODES. Compiled from the 2026-10-21 session's manual run so no future
session re-derives it. Pure measurement; no trading logic.

Usage: python3 cot_gate2_oos.py [--since 2026-07-08] [--symbol USO] [--bucket extreme_high]
"""
from __future__ import annotations

import argparse
from datetime import datetime

MIN_EPISODES = 5  # independent extreme episodes needed before a verdict means anything


def oos_power(rows: list[dict], since: str, bucket: str, horizon: int = 20) -> dict:
    """rows = cot_gate2_test.compute_forward_returns output. Counts OOS
    (report_date strictly after `since`) weeks, extreme weeks, matured
    forward returns, and independent episodes (a run of consecutive weeks
    in `bucket` = one episode)."""
    oos = sorted((r for r in rows if r["report_date"] > since), key=lambda r: r["report_date"])
    episodes, prev = 0, None
    for r in oos:
        hit = r["bucket"] == bucket
        if hit and not prev:
            episodes += 1
        prev = hit
    ext = [r for r in oos if r["bucket"] == bucket]
    matured = [r["forward_returns"][horizon] for r in ext if horizon in r["forward_returns"]]
    allm = [r["forward_returns"][horizon] for r in oos if horizon in r["forward_returns"]]
    verdict = "UNTESTABLE" if episodes < MIN_EPISODES else "TESTABLE"
    return {
        "since": since, "bucket": bucket, "horizon": horizon,
        "oos_weeks": len(oos), "extreme_weeks": len(ext), "episodes": episodes,
        "matured_extreme": len(matured),
        "extreme_mean": sum(matured) / len(matured) if matured else None,
        "baseline_mean": sum(allm) / len(allm) if allm else None,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-07-08")
    ap.add_argument("--symbol", default="USO")
    ap.add_argument("--bucket", default="extreme_high")
    a = ap.parse_args()
    from backtest_v2 import fetch_bars
    from cftc_cot import LOOKBACK_WEEKS, SYMBOLS, attach_cot_index, fetch_symbol_history
    from cot_gate2_test import compute_forward_returns
    recs = attach_cot_index(fetch_symbol_history(SYMBOLS[a.symbol]["code"], limit=LOOKBACK_WEEKS)[0])
    days = (datetime.utcnow() - datetime.strptime(recs[0]["report_date"], "%Y-%m-%d")).days + 30
    rows = compute_forward_returns(recs, fetch_bars(a.symbol, days))
    print(oos_power(rows, a.since, a.bucket))


if __name__ == "__main__":
    main()
