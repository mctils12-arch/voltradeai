#!/usr/bin/env python3
"""
illiquid_universe_probe_threshold_ablation_redraw.py — INDEPENDENT
RE-DRAW check on the STRICT_RSI mean_reversion threshold variant, for
the open_questions.md entry filed 2026-07-24 ("Does mean_reversion have
a real, exploitable edge specifically in illiquid small-caps...").

CONTEXT: the 2026-08-08 threshold-ablation session (LADDER PATH step 4,
illiquid_universe_probe_threshold_ablation.py) found STRICT_RSI (tighter
RSI oversold bands) was the one variant that improved BOTH the illiquid
group's own mean_reversion Sharpe (+0.269 -> +0.308) AND the
illiquid-vs-moderate spread (+0.492 -> +0.548) on the ORIGINAL pinned
2026-07-24 sample — but flagged it as unconfirmed: "ONE pinned sample,
no independent re-draw... a +0.039 Sharpe delta on n=10 tickers... could
partly be a smaller-sample artifact." That entry's own filed NEXT step:
"run STRICT_RSI specifically through steps 1-3's own discipline
(independent re-draw, train/test split, bootstrap/permutation
significance test on the improvement) before it would be eligible for a
dedicated RULE-REVIEW PR." This script is that discipline's STEP 1
analog (INDEPENDENT RE-DRAW), applied to the STRICT_RSI variant rather
than to the base unmodified-threshold logic (which already cleared its
own step 1 via illiquid_universe_probe_redraw.py on 2026-07-28).

PRIOR, stated before running (Reasoning Standard #10): if STRICT_RSI's
improvement reflects a real selectivity gain (tighter oversold bands
filter noise-driven false signals in a noisier microcap price series,
per the original threshold-ablation entry's own theory) rather than a
small-sample artifact of the one pinned draw, a fresh, non-overlapping
illiquid/moderate sample should show the SAME DIRECTION: STRICT_RSI's
illiquid mean_sharpe >= baseline's, and the illiquid-vs-moderate spread
widens under STRICT_RSI vs baseline. Counter-prior (Reasoning Standard
#4): n=10/n=7 with ~15% fewer trades under the tightened variant is a
small-sample regime already flagged as capable of producing a
Sharpe-estimate-variance artifact — a second draw showing a different or
reversed pattern is an equally honest, expected-possible outcome, not a
surprise to explain away.

METHOD: draws a FRESH illiquid/moderate NASDAQ Capital Market sample via
sample_stride=43 — distinct from both the 2026-07-24 original draw's
stride (40) and the 2026-07-28 redraw's stride (41), so this is a third,
genuinely independent draw rather than a coincidental repeat of either
prior one. Screening reuses illiquid_universe_probe_redraw.py's
screen_and_bucket() unchanged (same >=500-bar history requirement, same
trailing-252d volume tiering via classify_liquidity_tier). Runs BASELINE
and STRICT_RSI ONLY via illiquid_universe_probe_threshold_ablation.py's
run_variant/summarize_variant/mean_reversion_spread, reused unchanged
with the fresh-draw groups substituted for the pinned ones — DEEP_DROP
and VOLUME_WEIGHTED already refuted their own priors on the original
sample (both underperformed baseline on illiquid's own Sharpe), so
re-testing them here would not change any live decision and is out of
scope (Reasoning Standard #4: don't multiply comparisons beyond the one
surviving candidate).

NO STRATEGY/THRESHOLD/CONFIG CHANGE SHIPS FROM THIS SCRIPT. Even a clean
reproduction here is only one of the three named ladder-discipline steps
(train/test split and a formal significance test on the improvement
remain open) before STRICT_RSI would be eligible for its own dedicated
RULE-REVIEW PR.

Run with: python3 scripts/illiquid_universe_probe_threshold_ablation_redraw.py
(network calls: 1 to nasdaqtrader.com, then 1 Yahoo fetch per screened
candidate + 2 backtest runs [baseline, STRICT_RSI] per kept candidate —
no API key required, same as the prior probes in this family.)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import illiquid_universe_probe as orig  # noqa: E402
from illiquid_universe_probe_redraw import screen_and_bucket  # noqa: E402
from illiquid_universe_probe_threshold_ablation import (  # noqa: E402
    VARIANTS, run_variant, summarize_variant, mean_reversion_spread,
)

SAMPLE_STRIDE = 43


def draw_groups(sample_stride: int = SAMPLE_STRIDE) -> dict:
    """Fetches a fresh NASDAQ Capital Market candidate pool and screens it
    into illiquid/moderate buckets, mirroring illiquid_universe_probe_
    redraw.py's __main__ but returned as a pure dict (no printing) so
    this is separately callable/testable."""
    candidates = orig.fetch_nasdaq_capital_market_candidates(sample_stride=sample_stride)
    overlap_0724 = sorted((set(orig.ILLIQUID) | set(orig.MODERATE)) & set(candidates))
    buckets = screen_and_bucket(candidates)
    return {
        "illiquid": buckets["illiquid"],
        "moderate": buckets["moderate"],
        "screened_out": buckets["screened_out"],
        "overlap_with_2026_07_24_draw": overlap_0724,
    }


if __name__ == "__main__":
    print(f"=== drawing fresh illiquid/moderate sample (stride={SAMPLE_STRIDE}) ===", file=sys.stderr)
    draw = draw_groups()
    print(f"illiquid ({len(draw['illiquid'])}): {draw['illiquid']}", file=sys.stderr)
    print(f"moderate ({len(draw['moderate'])}): {draw['moderate']}", file=sys.stderr)
    print(f"overlap with 2026-07-24 pinned lists: {draw['overlap_with_2026_07_24_draw']}", file=sys.stderr)

    groups = {"illiquid": draw["illiquid"], "moderate": draw["moderate"]}
    if len(draw["illiquid"]) < 2 or len(draw["moderate"]) < 2:
        print("insufficient draw size for a group comparison, aborting", file=sys.stderr)
        sys.exit(1)

    report = {
        "sample_stride": SAMPLE_STRIDE,
        "candidate_tickers": groups,
        "screened_out": draw["screened_out"],
        "overlap_with_2026_07_24_draw": draw["overlap_with_2026_07_24_draw"],
    }

    print("=== BASELINE (unmodified thresholds) ===", file=sys.stderr)
    baseline_rows = run_variant(None, groups=groups)
    baseline_summary = summarize_variant(baseline_rows)
    report["baseline"] = {
        "summary": baseline_summary,
        "illiquid_minus_moderate_mean_reversion_sharpe": mean_reversion_spread(baseline_summary),
    }
    print(json.dumps(report["baseline"], indent=2), file=sys.stderr)

    print("=== VARIANT: STRICT_RSI ===", file=sys.stderr)
    strict_thresholds = VARIANTS["STRICT_RSI"]["thresholds"]
    strict_rows = run_variant(strict_thresholds, groups=groups)
    strict_summary = summarize_variant(strict_rows)
    report["STRICT_RSI"] = {
        "thresholds": strict_thresholds,
        "summary": strict_summary,
        "illiquid_minus_moderate_mean_reversion_sharpe": mean_reversion_spread(strict_summary),
    }
    print(json.dumps(report["STRICT_RSI"], indent=2), file=sys.stderr)

    print(json.dumps(report, indent=2))
