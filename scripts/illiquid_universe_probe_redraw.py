#!/usr/bin/env python3
"""
illiquid_universe_probe_redraw.py — LADDER PATH step 1 ("INDEPENDENT
RE-DRAW") for the open_questions.md entry filed 2026-07-24
("[2026-07-24 — ACTIVE ANGLE-HUNTING / EDGE DOCTRINE #2] Does
mean_reversion have a real, exploitable edge specifically in illiquid
small-caps, or is the 2026-07-24 probe result a single-sample
artifact?"). That entry's own LADDER PATH names this as the required
next step before the finding can be trusted: "different stride/seed on
the NASDAQ symbol directory ... to check this isn't an artifact of the
specific 10 names drawn 2026-07-24."

PRIOR, stated 2026-07-28 BEFORE running (Reasoning Standard #10): the
2026-07-24 result showed mean_reversion mean_sharpe +0.246 (8/10
positive) in illiquid vs. -0.222 in moderate, with momentum showing the
opposite pattern. If that finding reflects a real structural effect
(thin order books mechanically exhausting oversold pressure without
sustained institutional selling flow — Reasoning Standard #5), an
independent, non-overlapping sample of illiquid NASDAQ Capital Market
names should reproduce the SAME DIRECTION (mean_reversion doing
relatively better in illiquid than moderate, momentum doing relatively
worse in illiquid than moderate) even if the exact magnitude differs.
Counter-prior: this could easily be a single-sample artifact — 10 and 7
name groups have very little statistical power, and a second draw could
just as easily show a different or even opposite pattern. Expected
(stated before running): direction reproduces, magnitude differs.

METHOD: reuses illiquid_universe_probe.py's own classify_liquidity_tier
(mirrors backtest_v2.py's liquidity_cost_pct() tier boundaries),
run_group, and summarize functions unchanged — same backtest engine,
same v1.0.480 liquidity-tiered fill cost, same YEARS=4 window. The ONLY
thing that changes is the candidate draw: sample_stride=41 (vs. the
original script's 40) on nasdaqtrader.com's live symbol directory,
fetched 2026-07-28 (4 days after the original 2026-07-24 draw; NASDAQ
listings churn daily so this is a directory snapshot, not a re-read of
the same file) — zero ticker overlap with the original ILLIQUID/
MODERATE lists (verified: intersection is empty). LIQUID comparison
group is intentionally UNCHANGED (same 7 T1_TICKERS_FALLBACK names) —
the re-draw question is about the illiquid/moderate candidate
selection, not about re-drawing the live bot's own anchor universe.

Candidates are screened the same way as the original: >=500 trading
days of history required (via backtest_v2.fetch_bars), then bucketed by
trailing-252d average share volume using classify_liquidity_tier's
>5M / 1-5M / <=1M boundaries. No re-run needed for full-liquid (>5M)
names here — they are excluded, since the LIQUID group is fixed and
this probe only needs a fresh illiquid+moderate draw.

Run with: python3 scripts/illiquid_universe_probe_redraw.py
(network calls: 1 to nasdaqtrader.com, then 1 Yahoo fetch per screened
candidate + 1 backtest run per kept candidate — no API key required,
same as the original probe.)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import illiquid_universe_probe as orig  # noqa: E402

SAMPLE_STRIDE = 41
YEARS = orig.YEARS
MIN_BARS = 500
TARGET_ILLIQUID = 10
TARGET_MODERATE = 7


def screen_and_bucket(candidates, target_illiquid=TARGET_ILLIQUID, target_moderate=TARGET_MODERATE):
    """Fetches each candidate once, classifies by trailing-252d volume,
    and stops once both target counts are met (or the candidate list is
    exhausted) — mirrors how the original draw naturally screened 34
    candidates down to 10 illiquid / 7 moderate."""
    from backtest_v2 import fetch_bars

    buckets = {"illiquid": [], "moderate": [], "liquid_5m_plus": [], "screened_out": []}
    days = int(YEARS * 365) + 400
    for t in candidates:
        if len(buckets["illiquid"]) >= target_illiquid and len(buckets["moderate"]) >= target_moderate:
            break
        try:
            bars = fetch_bars(t, days)
        except Exception as e:
            buckets["screened_out"].append({"ticker": t, "reason": f"fetch_error: {str(e)[:120]}"})
            continue
        if len(bars.get("close", [])) < MIN_BARS:
            buckets["screened_out"].append({"ticker": t, "reason": f"insufficient_history ({len(bars.get('close', []))} bars)"})
            continue
        avg_vol = sum(bars["volume"][-252:]) / min(252, len(bars["volume"]))
        tier = orig.classify_liquidity_tier(avg_vol)
        if tier == "illiquid" and len(buckets["illiquid"]) < target_illiquid:
            buckets["illiquid"].append(t)
        elif tier == "moderate" and len(buckets["moderate"]) < target_moderate:
            buckets["moderate"].append(t)
        else:
            buckets["liquid_5m_plus" if tier == "liquid_5m_plus" else "screened_out"].append(
                {"ticker": t, "reason": f"tier={tier} (bucket full or >5M)"})
    return buckets


if __name__ == "__main__":
    print(f"=== fetching NASDAQ Capital Market candidates (stride={SAMPLE_STRIDE}) ===", file=sys.stderr)
    candidates = orig.fetch_nasdaq_capital_market_candidates(sample_stride=SAMPLE_STRIDE)
    overlap = (set(orig.ILLIQUID) | set(orig.MODERATE)) & set(candidates)
    print(f"{len(candidates)} candidates, overlap with 2026-07-24 draw: {sorted(overlap)}", file=sys.stderr)

    buckets = screen_and_bucket(candidates)
    print(f"illiquid ({len(buckets['illiquid'])}): {buckets['illiquid']}", file=sys.stderr)
    print(f"moderate ({len(buckets['moderate'])}): {buckets['moderate']}", file=sys.stderr)
    print(f"screened out: {buckets['screened_out']}", file=sys.stderr)

    results = {}
    for name, tickers in [("illiquid", buckets["illiquid"]), ("moderate", buckets["moderate"]),
                           ("liquid", orig.LIQUID)]:
        print(f"=== backtesting {name} (n={len(tickers)}) ===", file=sys.stderr)
        results[name] = orig.run_group(tickers)
        print(json.dumps(orig.summarize(results[name]), indent=2), file=sys.stderr)

    out = {
        "sample_stride": SAMPLE_STRIDE,
        "candidate_tickers": {"illiquid": buckets["illiquid"], "moderate": buckets["moderate"]},
        "screened_out": buckets["screened_out"],
        "summary": {g: orig.summarize(r) for g, r in results.items()},
    }
    print(json.dumps(out, indent=2))
