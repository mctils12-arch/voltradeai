#!/usr/bin/env python3
"""
Cross-sectional follow-up to scripts/hurst_exponent_probe.py: does the
rolling-Hurst trend-continuation signal that failed GATE 2 on SPY show up
more strongly in a genuinely illiquid/small-cap corner of the market than
in a liquid mega-cap comparison group?

WHY THIS IS NOT A SEVENTH COLD-START IMPORT (REASONING STANDARD #4):
`hurst_exponent_probe.py`'s own 2026-09-12 entry (research/open_questions.md,
search "SIXTH foreign field") is the sixth foreign-field import in this
file's family (ecology/CSD, epidemiology/R_t, reliability/hazard-rate,
seismology/Omori-Utsu, information-theory/permutation-entropy, hydrology/
Hurst) to fail GATE 2 on liquid index/mega-cap tickers, and it explicitly
named, as its own unaddressed NEXT item, the structural question this
script answers: is the market efficient at these horizons for THIS
universe (liquid names), or would "a genuinely different data axis
(cross-sectional breadth, intraday structure, or a non-price data
source)" show something six single-ticker/mega-cap tests could not? This
is that cross-sectional-breadth follow-up — same Hurst machinery, same
parameters, a different axis (many tickers across a liquidity spectrum,
pooled), per EDGE DOCTRINE #2 ("fish where whales can't": capacity-
constrained players cannot compete away structure in illiquid names the
way they do in SPY/mega-caps, so genuine persistence might survive there
even where it has been arbitraged out of the liquid comparison group).

REUSE, NOT REIMPLEMENTATION (EDGE DOCTRINE #3, and this repo's own
established convention — hurst_exponent_probe.py's own tertile_welch()
already imports permutation_entropy_probe.welch_vs_control via
importlib rather than reimplementing a second significance-test helper):
this file imports hurst_exponent_probe's rolling_hurst/continuation_scores/
destrided_spearman/tertile_welch/log_returns unchanged, backtest_v2.fetch_bars
unchanged (same Alpaca-first/Yahoo-fallback, no-lookahead data path), and
illiquid_universe_probe's ILLIQUID/MODERATE/LIQUID ticker lists unchanged
(SOURCE: scripts/illiquid_universe_probe.py, screened and documented there
2026-07-24 — a systematic sample of NASDAQ Capital Market names bucketed
by backtest_v2's own liquidity-cost tiers, plus a 7-name liquid mega-cap
comparison group drawn from tiered_strategy.T1_TICKERS_FALLBACK). Nothing
here re-derives or re-screens those lists; they are pinned for
reproducibility exactly as illiquid_universe_probe.py states.

PRIOR, stated BEFORE running anything against real data (REASONING
STANDARD #10): genuinely uncertain, with a mild directional lean toward a
STRONGER Hurst/continuation signal in the illiquid group than the liquid
group, on EDGE DOCTRINE #2's structural logic (momentum/persistence in
liquid mega-caps gets arbitraged away fast by capacity-unconstrained
players; illiquid small-caps are too small for that capital to bother
with, so genuine autocorrelation structure might survive there and
actually be large enough to matter). That lean is tempered by two things
I am watching for and will report honestly regardless of outcome:
  (a) REASONING STANDARD #4 — six prior same-family imports have already
      failed GATE 2 on liquid tickers using this exact statistic; a
      seventh application of the same math (even on a new universe)
      should be discounted for repeated fishing, not treated as a clean
      new theory-motivated test.
  (b) illiquid_universe_probe.py's OWN measured result: this ILLIQUID
      group's mean buy-and-hold return was -74.7% and MODERATE's was
      -74.6% over the same ~4y window (near-identical, severe secular
      decline across both non-liquid groups, unlike a random illiquid
      sample). A Hurst/continuation "signal" computed on a near-monotonic
      decline may be measuring something structurally different from a
      tradable persistence effect — e.g. sustained one-directional drift
      (dilution, going-concern doubt, delisting-risk grind) that a
      trend-following statistic will always call "persistent" almost by
      construction, not a genuine regime-dependent alternation between
      trending and mean-reverting days the way SPY's ~50/50 up/down tape
      lets the test discriminate. If the pooled illiquid result is
      dominated by one or two of the ten names (see the leave-one-out
      check below), or if the effect looks identical in sign/magnitude to
      just "buy-and-hold always continues down", that is reported as a
      confound, not a discovery.
Honest expectation given (a) and (b): most likely another clean negative,
but if EDGE DOCTRINE #2's structural story is ever going to show up
anywhere in this probe family, this is the shape it should take (a
cross-sectional pool, not one more single ticker), so it is worth the one
run.

WHAT THIS SCRIPT DOES: for each of the three pinned groups (ILLIQUID,
MODERATE, LIQUID), fetch up to `days` daily bars per ticker (default 2520,
~10y, matching the SPY run — some illiquid names will have far less
history; this is reported per-ticker, not hidden), compute rolling Hurst
+ continuation scores with the SAME parameters the SPY run used (252-day
rolling window, 8-day min chunk, 20-day lookback/horizon — the defaults
in hurst_exponent_probe.run_probe), per-ticker de-stride (stride=horizon,
the same overlapping-forward-window fix the SPY entry built), then POOL
each group's de-strided pairs and run the same Spearman/tertile-Welch
tests on the pooled sample — the actual GATE-2-relevant number, since it
has a larger n than any single ticker.

LIMITATION, stated honestly and not to be glossed over when reading the
pooled n: pooling across TICKERS is not the same statistical guarantee as
pooling across TIME for one ticker. De-striding within a ticker removes
the OVERLAPPING-FORWARD-WINDOW autocorrelation (consecutive days sharing
19 of their 20 forward-return days). It does NOT remove CROSS-SECTIONAL
correlation between tickers — if all ten illiquid names get pushed down
together on the same macro-selloff days (a shared market/sector factor),
their supposedly-independent pairs on those days are not actually
independent draws, and the true effective n is smaller than the raw
pooled count. The pooled n reported below should be read as an upper
bound on statistical power, not a confirmed independent sample size.

LADDER PATH: GATE 2 (SIGNAL) only, exactly like the SPY run — pure
statistical predictive power on daily bars, no entry/exit logic, no
sizing, no strategy code touched. GATE 2 bar (this repo's established
threshold, per the DTS/BLS and original Hurst entries): |rho| >= 0.30 AND
p < 0.05 on the de-strided/pooled sample. A pass here would still need
out-of-sample confirmation and a GATE 3 (LOGIC) backtest ablation before
it could ever inform real strategy code — this script and its results do
not touch datacore/signal_ladder.json or any strategy/scoring/sizing
code regardless of outcome.

MEASUREMENT INTEGRITY note (mirrors hurst_exponent_probe.py's own):
reports the null result as plainly as a positive one; does not retry with
different window/lookback/horizon parameters after seeing an unfavorable
result (REASONING STANDARD #4 — one theory-motivated spec, run once,
reported honestly, on both groups equally).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backtest_v2  # noqa: E402

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_sibling_module(name: str, filename: str):
    """Load a sibling scripts/ module by file path, same pattern
    hurst_exponent_probe.py's own tertile_welch() uses to reach
    permutation_entropy_probe.py — avoids a package-relative import that
    would break when these scripts are run standalone."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_SCRIPTS_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hep = _load_sibling_module("hurst_exponent_probe", "hurst_exponent_probe.py")
iup = _load_sibling_module("illiquid_universe_probe", "illiquid_universe_probe.py")

# Pinned ticker lists. SOURCE OF TRUTH: scripts/illiquid_universe_probe.py
# (screened/documented there 2026-07-24; NOT re-derived or re-screened
# here — imported from that module, not retyped, so this file cannot
# silently drift from the pinned source).
ILLIQUID: list[str] = list(iup.ILLIQUID)
MODERATE: list[str] = list(iup.MODERATE)
LIQUID: list[str] = list(iup.LIQUID)

# Same defaults as hurst_exponent_probe.run_probe's own SPY run.
DEFAULT_DAYS = 2520
DEFAULT_WINDOW = 252
DEFAULT_MIN_CHUNK = 8
DEFAULT_LOOKBACK = 20
DEFAULT_HORIZON = 20

MIN_BARS = DEFAULT_WINDOW + DEFAULT_LOOKBACK + DEFAULT_HORIZON + 10

# This repo's established GATE 2 bar (DTS/BLS entry, and the original
# SPY Hurst entry that used the identical threshold).
GATE2_RHO_FLOOR = 0.30
GATE2_P_CEILING = 0.05


def ticker_pairs(ticker: str, days: int = DEFAULT_DAYS, window: int = DEFAULT_WINDOW,
                  min_chunk: int = DEFAULT_MIN_CHUNK, lookback: int = DEFAULT_LOOKBACK,
                  horizon: int = DEFAULT_HORIZON) -> dict:
    """Fetch bars for one ticker and compute its (hurst, continuation_score)
    pairs via hurst_exponent_probe's own unmodified functions. Always
    returns a dict with 'ticker'/'n_bars'/'pairs'/'error' — never raises —
    so one obscure illiquid ticker failing to fetch (real network error,
    or genuinely too little trading history) cannot kill the whole
    cross-sectional run; it is reported per-ticker instead."""
    try:
        bars = backtest_v2.fetch_bars(ticker, days)
    except Exception as e:  # noqa: BLE001 - report, never crash the whole probe
        return {"ticker": ticker, "n_bars": 0, "pairs": [], "error": f"fetch failed: {e}"[:200]}
    closes = bars.get("close", []) if bars else []
    if len(closes) < MIN_BARS:
        return {"ticker": ticker, "n_bars": len(closes), "pairs": [],
                "error": f"insufficient bars ({len(closes)} < {MIN_BARS} required)"}
    rets = hep.log_returns(closes)
    hurst = hep.rolling_hurst(rets, window=window, min_chunk=min_chunk)
    pairs = hep.continuation_scores(rets, hurst, lookback=lookback, horizon=horizon)
    return {"ticker": ticker, "n_bars": len(closes), "pairs": pairs, "error": None}


def destride(pairs: Sequence[tuple], horizon: int) -> list:
    """Every horizon-th pair for ONE ticker's own daily series — the exact
    slicing hurst_exponent_probe.destrided_spearman() applies internally
    (`pairs[::stride]`), reused verbatim rather than reimplemented (this
    helper exists only so the raw de-strided pairs can be pooled across
    tickers before the significance test runs, which destrided_spearman()
    itself does not expose — it computes spearman() directly on the slice
    instead of returning it)."""
    return list(pairs[::horizon]) if horizon > 0 else []


def pool_group(per_ticker_rows: Sequence[dict]) -> list:
    """Concatenate each ticker's own already-de-strided pairs
    (`destrided_pairs` key) into one pooled cross-sectional sample for a
    group. See the module docstring LIMITATION section: this is bigger-n
    than any single ticker, but pooling across tickers does not remove
    cross-sectional correlation between them the way de-striding removes
    within-ticker overlap — report both facts together, never the n alone."""
    out: list = []
    for row in per_ticker_rows:
        out.extend(row.get("destrided_pairs", []))
    return out


def _passes_gate2(stat: dict | None) -> bool:
    if not stat:
        return False
    return abs(stat["rho"]) >= GATE2_RHO_FLOOR and stat["p_value"] < GATE2_P_CEILING


def leave_one_out(per_ticker_rows: Sequence[dict]) -> dict:
    """Robustness check the task calls for: does any single ticker
    dominate the pooled result? Recomputes the pooled de-strided Spearman
    with each ticker excluded in turn, ranks exclusions by |change in
    pooled rho|, and flags whether excluding the single largest-
    contribution ticker flips the GATE 2 pass/fail verdict."""
    usable = [r for r in per_ticker_rows if r.get("destrided_pairs")]
    if len(usable) < 2:
        return {"note": "fewer than 2 tickers with usable pairs; leave-one-out skipped",
                "per_exclusion": []}

    full_pairs = pool_group(usable)
    full_stat = hep.spearman(full_pairs)
    full_pass = _passes_gate2(full_stat)

    rows = []
    for excluded in usable:
        remaining = [r for r in usable if r["ticker"] != excluded["ticker"]]
        stat = hep.spearman(pool_group(remaining))
        delta = None
        if stat and full_stat:
            delta = round(stat["rho"] - full_stat["rho"], 4)
        rows.append({
            "excluded_ticker": excluded["ticker"],
            "excluded_n_pairs": len(excluded["destrided_pairs"]),
            "pooled_rho_without": stat["rho"] if stat else None,
            "pooled_p_without": stat["p_value"] if stat else None,
            "gate2_pass_without": _passes_gate2(stat),
            "delta_rho_from_full": delta,
        })
    rows.sort(key=lambda r: abs(r["delta_rho_from_full"] or 0.0), reverse=True)
    conclusion_flips = any(r["gate2_pass_without"] != full_pass for r in rows)

    return {
        "full_pooled_rho": full_stat["rho"] if full_stat else None,
        "full_pooled_p": full_stat["p_value"] if full_stat else None,
        "full_pooled_n": full_stat["n"] if full_stat else None,
        "full_pooled_gate2_pass": full_pass,
        "largest_single_ticker_contribution": rows[0]["excluded_ticker"] if rows else None,
        "conclusion_flips_on_any_single_exclusion": conclusion_flips,
        "per_exclusion": rows,
    }


def run_group(name: str, tickers: Sequence[str], days: int = DEFAULT_DAYS,
              window: int = DEFAULT_WINDOW, min_chunk: int = DEFAULT_MIN_CHUNK,
              lookback: int = DEFAULT_LOOKBACK, horizon: int = DEFAULT_HORIZON) -> dict:
    per_ticker_rows = []
    for t in tickers:
        row = ticker_pairs(t, days=days, window=window, min_chunk=min_chunk,
                            lookback=lookback, horizon=horizon)
        row["destrided_pairs"] = destride(row["pairs"], horizon)
        row["ticker_stat"] = hep.destrided_spearman(row["pairs"], horizon) if row["pairs"] else None
        per_ticker_rows.append(row)

    pooled_pairs = pool_group(per_ticker_rows)
    pooled_spearman = hep.spearman(pooled_pairs)
    pooled_tertile = hep.tertile_welch(pooled_pairs)

    return {
        "group": name,
        "tickers": list(tickers),
        "per_ticker": [
            {
                "ticker": r["ticker"],
                "n_bars": r["n_bars"],
                "error": r["error"],
                "n_destrided_pairs": len(r["destrided_pairs"]),
                "rho": r["ticker_stat"]["rho"] if r["ticker_stat"] else None,
                "p_value": r["ticker_stat"]["p_value"] if r["ticker_stat"] else None,
                "n": r["ticker_stat"]["n"] if r["ticker_stat"] else None,
            }
            for r in per_ticker_rows
        ],
        "pooled_n_pairs": len(pooled_pairs),
        "pooled_spearman": pooled_spearman,
        "pooled_tertile_welch": pooled_tertile,
        "pooled_gate2_pass": _passes_gate2(pooled_spearman),
        "leave_one_out": leave_one_out(per_ticker_rows),
    }


def run_probe(days: int = DEFAULT_DAYS, window: int = DEFAULT_WINDOW,
              min_chunk: int = DEFAULT_MIN_CHUNK, lookback: int = DEFAULT_LOOKBACK,
              horizon: int = DEFAULT_HORIZON) -> dict:
    kwargs = dict(days=days, window=window, min_chunk=min_chunk, lookback=lookback,
                  horizon=horizon)
    illiquid = run_group("illiquid", ILLIQUID, **kwargs)
    moderate = run_group("moderate", MODERATE, **kwargs)
    liquid = run_group("liquid", LIQUID, **kwargs)

    def _row(g):
        s = g["pooled_spearman"]
        return {
            "group": g["group"],
            "pooled_n_pairs": g["pooled_n_pairs"],
            "rho": s["rho"] if s else None,
            "p_value": s["p_value"] if s else None,
            "gate2_pass": g["pooled_gate2_pass"],
        }

    return {
        "illiquid": illiquid,
        "moderate": moderate,
        "liquid": liquid,
        "comparison": [_row(illiquid), _row(moderate), _row(liquid)],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS, help="~10y of trading days")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="rolling Hurst window")
    ap.add_argument("--min-chunk", type=int, default=DEFAULT_MIN_CHUNK, dest="min_chunk")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = ap.parse_args()
    print(json.dumps(run_probe(args.days, args.window, args.min_chunk, args.lookback,
                                args.horizon), indent=2))
