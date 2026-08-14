#!/usr/bin/env python3
"""
microcap_cost_floor_check.py — EDGE DOCTRINE axis (b) COSTS-AND-FRICTIONS-
FIRST workup (REASONING STANDARD #6) for the "loosen MIN_PRICE/MIN_VOLUME
to admit sub-$5 microcaps" question the 2026-08-11 illiquid_universe_probe
ladder-step-5 entry (open_questions.md) surfaced and explicitly deferred:
"start by pricing the SPECIFIC new risks named above (spread/fill realism,
halt/manipulation exposure, data-source staleness at sub-$5 sizes) before
any backtest."

This script prices two of those three risks with STRUCTURAL facts that do
not require live market data (no Alpaca/market-data credentials are
available in this sandbox — see the third risk, data-source staleness,
which this script explicitly does NOT attempt and flags as still needing
a live-market-hours session):

1. TICK-SIZE SPREAD FLOOR. Reg NMS Rule 612 sets the minimum quoted price
   variation at $0.01 for any NMS stock priced >= $1.00 (sub-$1 stocks may
   quote in sub-penny increments, so this floor does NOT apply below
   $1.00 — flagged separately, not silently computed). The tightest
   possible quoted spread for a price >= $1.00 stock is exactly one tick;
   a marketable order crossing that spread pays roughly half a tick versus
   the midpoint, per side. This is a LOWER BOUND on real-world spread cost
   (actual quoted spreads for thin names run several ticks wide, not one),
   so a stock whose tick-floor ALONE exceeds the backtest's modeled
   per-side cost proves the model under-charges that name, not just "might".

2. LULD VOLATILITY-HALT BAND WIDTH. FINRA/Nasdaq's published Limit
   Up-Limit Down table (verified via WebSearch this session against
   finra.org/investors/insights/guardrails-market-volatility and the
   Nasdaq LULD FAQ, since this is a static regulatory constant worth
   getting right rather than trusting training-data recall — re-verify
   against the live NMS Plan text if this analysis becomes decision-
   relevant, as neither source was fetched via an authenticated primary
   filing this session): Tier 1 (S&P 500/Russell 1000/certain ETPs) gets
   5% intraday bands above $3, Tier 2 (everything else) gets 10% above
   $3; BOTH tiers widen to 20% for $0.75-$3 and to "lesser of $0.15 or
   75%" below $0.75; bands double in the last 25 minutes of the session
   for Tier 1 and for sub-$3 Tier 2 names. None of this repo's pinned
   ILLIQUID/MODERATE microcap tickers are S&P 500/Russell 1000 members,
   so all are Tier 2. This is a REGULATORY PERMISSION for a much larger
   intrabar move before any circuit-breaker pause protects a resting
   order — the current cost/backtest model has no representation of this
   at all (liquidity_cost_pct is a single flat per-side percentage; nothing
   in backtest_v2.py or bot.ts's fill simulation varies with halt/gap risk).

NO STRATEGY, THRESHOLD, OR CONFIG CHANGE SHIPS FROM THIS SCRIPT. It is a
pure informational report, same class as ladder_readiness_check.py — exit
code is always 0.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_v2  # noqa: E402 — live cost constant, not a copy
import illiquid_universe_probe as orig  # noqa: E402 — reuse pinned ticker lists verbatim

ILLIQUID = orig.ILLIQUID
MODERATE = orig.MODERATE

# Point-in-time snapshot, NOT re-fetched live (no market-data credentials in
# this sandbox). Prices are the exact last-close values recorded by the
# 2026-08-11 illiquid_universe_probe_universe_gate.py live run (open_questions.md
# item, "RESULT (live run, 2026-08-11)"), transcribed verbatim from that
# session's filed result, not re-derived from memory. A future session with
# market-data access should re-fetch fresh closes before treating this as
# more than a directional check — prices move, the STRUCTURAL facts below
# (tick floor formula, LULD table) do not.
PRICE_SNAPSHOT_DATE = "2026-08-11"
PRICE_SNAPSHOT = {
    # ILLIQUID group (n=10)
    "AXG": 2.57, "CISO": 0.26, "DYAI": 1.00, "EPOW": 0.55, "GALT": 3.12,
    "NRXP": 3.35, "PROF": 7.34, "SNOA": 1.29, "TRAW": 0.54, "WNW": 2.68,
    # MODERATE group (n=7) — IMNM already clears MIN_PRICE and is excluded
    # here since it's not part of the "sub-$5 microcap" question this
    # script prices; the other 6 are the relevant sub-$5 names.
    "CRDF": 1.03, "KTTA": 0.47, "ONCY": 0.80, "SXTP": 1.13, "VIVO": 3.94,
    "ZCMD": 1.14,
}

MIN_TICK = 0.01  # Reg NMS Rule 612 minimum price variation, price >= $1.00


def tick_floor_pct(price):
    """Theoretical minimum per-side spread-crossing cost (%) from Reg NMS
    Rule 612's $0.01 minimum tick, for price >= $1.00 only. Returns None
    below $1.00 (sub-penny-eligible; the tick floor does not constrain
    those names, and this function does not guess a substitute floor —
    that would need live quote data, not structural regulation)."""
    if price < 1.00:
        return None
    half_tick = MIN_TICK / 2.0
    return (half_tick / price) * 100.0


def luld_band_pct(price, tier="tier2", near_close=False):
    """FINRA/Nasdaq published LULD price-band percentage for `price`,
    assuming Tier 2 (the correct tier for every pinned ticker here — none
    are S&P 500/Russell 1000 members). Verified against finra.org and the
    Nasdaq LULD FAQ this session (see module docstring); this is a fixed
    regulatory table, not live market data."""
    if price < 0.75:
        base = min(0.15 / price, 0.75) * 100.0
    elif price <= 3.00:
        base = 20.0
    else:
        base = 10.0 if tier == "tier2" else 5.0
    if near_close and (tier == "tier1" or price <= 3.00):
        base *= 2.0
    return base


def build_report(prices=None):
    prices = prices or PRICE_SNAPSHOT
    model_cost_pct = backtest_v2._ILLIQUID_COST_PCT * 100.0  # live constant, as a %
    rows = []
    for ticker, price in sorted(prices.items()):
        floor = tick_floor_pct(price)
        rows.append({
            "ticker": ticker,
            "price": price,
            "price_snapshot_date": PRICE_SNAPSHOT_DATE,
            "tick_floor_pct": floor,
            "tick_floor_applies": floor is not None,
            "model_cost_pct": model_cost_pct,
            "model_undercharges": (floor is not None and floor > model_cost_pct),
            "luld_band_pct_midday": luld_band_pct(price),
            "luld_band_pct_near_close": luld_band_pct(price, near_close=True),
        })
    undercharged = [r for r in rows if r["model_undercharges"]]
    unpriceable = [r for r in rows if not r["tick_floor_applies"]]
    return {
        "model_cost_pct_per_side": model_cost_pct,
        "rows": rows,
        "n_priced": len(rows) - len(unpriceable),
        "n_undercharged_by_model": len(undercharged),
        "n_sub_penny_unpriceable": len(unpriceable),
        "undercharged_tickers": [r["ticker"] for r in undercharged],
        "sub_penny_tickers": [r["ticker"] for r in unpriceable],
    }


def format_report(report):
    lines = []
    lines.append(
        f"backtest_v2._ILLIQUID_COST_PCT (live, sub-1M-share/day bucket): "
        f"{report['model_cost_pct_per_side']:.3f}% per side"
    )
    lines.append("")
    lines.append(f"{'ticker':<7}{'price':>8}{'tick-floor%':>13}{'LULD mid%':>11}{'LULD close%':>13}  flag")
    for r in report["rows"]:
        floor_s = f"{r['tick_floor_pct']:.3f}" if r["tick_floor_pct"] is not None else "n/a<$1"
        flag = "UNDERCHARGED" if r["model_undercharges"] else ("sub-penny" if not r["tick_floor_applies"] else "ok")
        lines.append(
            f"{r['ticker']:<7}{r['price']:>8.2f}{floor_s:>13}"
            f"{r['luld_band_pct_midday']:>11.1f}{r['luld_band_pct_near_close']:>13.1f}  {flag}"
        )
    lines.append("")
    lines.append(
        f"{report['n_undercharged_by_model']}/{report['n_priced']} priceable names have a tick-size "
        f"spread floor ALONE exceeding the model's flat {report['model_cost_pct_per_side']:.3f}% "
        f"per-side cost: {', '.join(report['undercharged_tickers']) or 'none'}"
    )
    lines.append(
        f"{report['n_sub_penny_unpriceable']} names trade below $1.00, where Reg NMS Rule 612's "
        f"penny tick floor does not apply (sub-penny quoting permitted) — this script does NOT "
        f"estimate a substitute floor for them, that needs live quote data: "
        f"{', '.join(report['sub_penny_tickers']) or 'none'}"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    args = parser.parse_args()
    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
