#!/usr/bin/env python3
"""
illiquid_universe_probe_threshold_ablation.py — LADDER PATH step 4
("illiquid-tuned RE-THRESHOLDING") for the open_questions.md entry filed
2026-07-24 ("Does mean_reversion have a real, exploitable edge specifically
in illiquid small-caps, or is the 2026-07-24 probe result a single-sample
artifact?").

Steps so far, all on the UNMODIFIED (liquid-tuned) strategies/mean_reversion.py
thresholds: 1 (INDEPENDENT RE-DRAW) CLOSED — reproduces almost exactly on a
disjoint sample. 2 (TRAIN/TEST SPLIT) CLOSED as "checked, not confirmed" —
illiquid > moderate holds in both halves, but illiquid's OWN Sharpe
sign-flips between halves. 3 (SIGNIFICANCE TEST) — illiquid vs moderate is
Bonferroni-clear (CI [0.145, 0.837], p=0.0195); illiquid vs liquid is not
(underpowered). A 2026-08-01 REGIME-CONDITIONED check found the
illiquid > moderate spread is positive in every regime bucket, ruling out
regime-mix as the explanation for step 2's within-illiquid sign-flip (cause
still unexplained). This script is the entry's own filed step 4, quoted
verbatim: "the natural next step is NOT porting the liquid-tuned
mean_reversion thresholds unchanged into a live illiquid-universe strategy
... Re-tuning would be its own RULE-REVIEW-gated effort with counterfactual
evidence" — this script IS that counterfactual evidence, not a threshold
change. `strategies/mean_reversion.py` gained an optional `thresholds=`
override this session (DEFAULT_THRESHOLDS dict, zero behavior change for any
existing caller — see test_mean_reversion_thresholds.py) purely so this
research harness can monkeypatch variant thresholds without touching
backtest_v2.py (the MEASUREMENT-INTEGRITY-named backtest engine) at all.

NO STRATEGY/THRESHOLD/CONFIG CHANGE SHIPS FROM THIS SCRIPT. Even a variant
that clearly beats the baseline here only produces evidence for a FUTURE,
separate RULE-REVIEW session to act on (one logical change per PR; a
threshold change is its own PR with its own logged rollback trigger, per
CLAUDE.md's RULE REVIEW section) — this session's own primary action is the
research artifact, not the ship.

PRIOR, stated before running (Reasoning Standard #10): three theory-motivated
variants, each tightening/loosening exactly ONE of the score's three input
axes (RSI / 5d-drop / volume-ratio) — not a combinatorial grid, per
Reasoning Standard #4's "prefer fewer, theory-motivated tests; discount by
the number of variants tried." All three variants share one underlying
theory: the CURRENT thresholds were tuned against the bot's actual live
(liquid, mega-cap) universe (see the original 2026-07-24 entry's own
counter-prior), and microcap price/volume series are structurally noisier
around a much smaller base — so a raw RSI<20 read or a -3% 5d move, which
is a real (if modest) signal for AAPL, may be routine noise for a
$50M-market-cap illiquid name. Expected direction: STRICT_RSI and
DEEP_DROP (both requiring a MORE extreme condition before scoring high)
should raise the illiquid group's mean_reversion Sharpe by filtering out
noise-driven false signals, at the cost of fewer trades (fewer, higher-
conviction entries). VOLUME_WEIGHTED is the more speculative variant — it
LOWERS the volume-ratio bar (a smaller absolute spike is already
meaningful against a thin illiquid baseline) while INCREASING its score
weight, betting the volume dimension is the load-bearing "sellers
exhausted" signal in illiquid names specifically (the mechanistic story
the original entry's WHY-THIS-MIGHT-BE-REAL section proposed). Counter-
prior (Reasoning Standard #4): thinner trade counts from any tightened
variant could also just be a small-sample noise reduction that HAPPENS to
land on a higher Sharpe by chance (fewer trades <=> higher per-trade
variance in the resulting Sharpe estimate) — n_trades is reported
alongside every Sharpe below specifically so this isn't hidden.

MOMENTUM is out of scope for this step (matching the significance-test
step's own scoping note: momentum's illiquid-worst/liquid-best pattern
already reproduced cleanly with no sign instability — it doesn't need
re-thresholding scrutiny the way mean_reversion's sign-flipping does).

Reuses ILLIQUID/MODERATE (the ORIGINAL 2026-07-24 pinned lists, not the
2026-07-28 redraw — matching every prior ladder-path step's precedent of
keeping this one pinned sample as the through-line) and run_group()/
summarize() unchanged from illiquid_universe_probe.py. Baseline is
re-run fresh in THIS script execution (not the cached 07-24 numbers) so
baseline and variants share identical bar-fetch timing.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import illiquid_universe_probe as orig  # noqa: E402

GROUPS = {"illiquid": orig.ILLIQUID, "moderate": orig.MODERATE}

# Each variant touches exactly ONE input axis relative to
# strategies.mean_reversion.DEFAULT_THRESHOLDS (see module docstring for
# the theory behind each).
VARIANTS = {
    "STRICT_RSI": {
        "rationale": "tighten RSI oversold bands (20/30/40 -> 15/25/35): "
                      "require a more extreme RSI read before crediting "
                      "'oversold' on a noisier microcap price series",
        "thresholds": {"rsi_extreme": 15, "rsi_oversold": 25, "rsi_mild": 35},
    },
    "DEEP_DROP": {
        "rationale": "widen 5d-drop bands (-10/-5/-3 -> -18/-10/-6): a "
                      "routine microcap wiggle should not count as the "
                      "same 'panic exhaustion' condition a real -10%+ "
                      "mega-cap move represents",
        "thresholds": {"chg_big": -18, "chg_med": -10, "chg_small": -6},
    },
    "VOLUME_WEIGHTED": {
        "rationale": "lower the volume-ratio bar (2/1.5 -> 1.5/1.2) and "
                      "raise its point weight (20/10 -> 30/15): a smaller "
                      "absolute spike already stands out against a thin "
                      "illiquid baseline, and volume is this hypothesis's "
                      "proposed mechanistic 'sellers exhausted' marker",
        "thresholds": {"vr_high": 1.5, "vr_high_pts": 30,
                        "vr_med": 1.2, "vr_med_pts": 15},
    },
}


def make_variant_score(original_score, variant_thresholds):
    """Wraps strategies.mean_reversion.score so backtest_v2's
    `from strategies import mean_reversion; mean_reversion.score(...)`
    call sites transparently pick up the variant thresholds — no change
    to backtest_v2.py itself. Preserves the override contract (an
    explicit caller-supplied `thresholds` kwarg still wins over the
    variant, matching score()'s own {**DEFAULTS, **override} merge)."""
    def variant_score(rsi, change_pct_5d, volume_ratio, thresholds=None):
        merged = variant_thresholds if thresholds is None else {**variant_thresholds, **thresholds}
        return original_score(rsi, change_pct_5d, volume_ratio, thresholds=merged)
    return variant_score


def run_variant(thresholds: dict | None, groups=GROUPS) -> dict:
    """Runs run_group() over every named group under the given threshold
    override (None = baseline/unmodified). Monkeypatches
    strategies.mean_reversion.score for the duration only, restored in
    finally regardless of outcome."""
    from strategies import mean_reversion
    original_score = mean_reversion.score
    if thresholds:
        mean_reversion.score = make_variant_score(original_score, thresholds)
    try:
        return {name: orig.run_group(tickers) for name, tickers in groups.items()}
    finally:
        mean_reversion.score = original_score


def summarize_variant(group_rows: dict) -> dict:
    return {name: orig.summarize(rows) for name, rows in group_rows.items()}


def mean_reversion_spread(summary: dict) -> float | None:
    """illiquid mean_sharpe - moderate mean_sharpe, the same spread step 3
    significance-tested — the single comparison this step tracks per
    variant."""
    illiquid = summary.get("illiquid", {}).get("mean_reversion")
    moderate = summary.get("moderate", {}).get("mean_reversion")
    if not illiquid or not moderate:
        return None
    return round(illiquid["mean_sharpe"] - moderate["mean_sharpe"], 4)


if __name__ == "__main__":
    report = {}

    print("=== BASELINE (unmodified thresholds) ===", file=sys.stderr)
    baseline_rows = run_variant(None)
    baseline_summary = summarize_variant(baseline_rows)
    report["baseline"] = {
        "summary": baseline_summary,
        "illiquid_minus_moderate_mean_reversion_sharpe": mean_reversion_spread(baseline_summary),
    }
    print(json.dumps(report["baseline"], indent=2), file=sys.stderr)

    for vname, vdef in VARIANTS.items():
        print(f"=== VARIANT: {vname} ===", file=sys.stderr)
        rows = run_variant(vdef["thresholds"])
        summary = summarize_variant(rows)
        report[vname] = {
            "rationale": vdef["rationale"],
            "thresholds": vdef["thresholds"],
            "summary": summary,
            "illiquid_minus_moderate_mean_reversion_sharpe": mean_reversion_spread(summary),
        }
        print(json.dumps(report[vname], indent=2), file=sys.stderr)

    print(json.dumps(report, indent=2))
