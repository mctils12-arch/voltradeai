#!/usr/bin/env python3
"""
illiquid_universe_probe_significance.py — LADDER PATH step 3
("SIGNIFICANCE TEST") for the open_questions.md entry filed 2026-07-24
("Does mean_reversion have a real, exploitable edge specifically in
illiquid small-caps, or is the 2026-07-24 probe result a single-sample
artifact?").

Steps so far: 1 (INDEPENDENT RE-DRAW, illiquid_universe_probe_redraw.py)
CLOSED — direction reproduces almost exactly in magnitude on a disjoint
sample. 2 (TRAIN/TEST SPLIT, illiquid_universe_probe_traintest.py)
CLOSED as "checked, not confirmed" — illiquid > moderate holds in both
halves, but illiquid's own mean_reversion Sharpe sign-flips between the
early and late half (negative early, strongly positive late), which
this step's own entry flagged as raising the bar for step 3.

PRIOR, stated 2026-07-28 BEFORE running (Reasoning Standard #10): given
step 2's own finding that the illiquid-vs-moderate spread is directionally
stable across both time halves while illiquid's OWN Sharpe is not
stable in sign, the prior here is that illiquid vs MODERATE clears a
bootstrap/permutation significance check more comfortably than illiquid
vs LIQUID — the liquid mean_reversion reading has near-zero statistical
power (11 total trades across 7 tickers, explicitly flagged as "not a
fair mean_reversion baseline" in the original 2026-07-24 entry), so a
wide CI / large p-value there would say "underpowered comparison," not
"no effect." COUNTER-PRIOR (Reasoning Standard #4): n=10 and n=7 are
both small samples for a between-group bootstrap on a noisy per-ticker
Sharpe statistic — even the illiquid-vs-moderate comparison may not
clear a strict significance bar, and that would be an equally honest
result, not a null one to argue around.

METHOD: this step fits nothing new and re-runs no strategy logic beyond
what illiquid_universe_probe.py's run_group() already computes — it
takes the resulting PER-TICKER mean_reversion Sharpe ratios (one number
per ticker; each ticker's own Sharpe is itself computed from that
ticker's own trade history, so treating "one Sharpe per ticker" as the
unit of resampling is a cluster bootstrap over tickers, not over
trades — the right unit here since trades within one ticker are not
independent draws) and runs two classical small-sample tests on the
GROUP-MEAN spread:
  1. PERCENTILE BOOTSTRAP CI on (mean(group_a) - mean(group_b)):
     resample each group's ticker list independently, WITH replacement,
     n_boot times; the observed spread is "significant" in the loose
     bootstrap sense if the resulting CI excludes 0.
  2. PERMUTATION TEST (exact-in-spirit, randomized in practice): pools
     both groups' Sharpes, repeatedly reassigns them into two groups of
     the ORIGINAL sizes at random (a null world where group membership
     doesn't matter), and reports the two-sided fraction of permuted
     spreads at least as extreme as the one actually observed — the
     more standard formal significance test for a small-sample group
     difference, and the second option this open_questions.md entry's
     own ladder-path step 3 named ("...or a paired per-ticker test").
Both are pure, seedable, and unit-tested against synthetic data with a
known separated/non-separated relationship before being run against the
real per-ticker Sharpes below (CLAUDE.md READ BEFORE WRITE: verify the
statistical machinery on cases with a known right answer before trusting
its verdict on the one real case that matters).

SCOPE: mean_reversion ONLY, matching this specific ladder-path entry
("bootstrap CI on the mean_reversion Sharpe spread"). momentum's
illiquid-worst/liquid-best pattern already reproduced cleanly across
both the redraw (step 1) and the train/test split (step 2) with no
sign instability, so it does not need this step's scrutiny the way
mean_reversion's sign-flipping does.

NOT ACTIONABLE REGARDLESS OF THIS STEP'S OUTCOME: even a clean
significance pass here would only close the SIGNAL-adjacent statistical
question for the UNMODIFIED strategy logic on this ONE pinned sample —
ladder-path steps 4-5 (illiquid-tuned re-thresholding, then a LOGIC-gate
ablation against the live bot's actual deep_score/tier1_csp_core
candidate path, not this ETF-rotation-style backtest engine) remain
fully open regardless of this result. No strategy, threshold, or config
change ships from this script.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import illiquid_universe_probe as orig  # noqa: E402


def _resample_mean(rng: random.Random, sample: list) -> float:
    n = len(sample)
    return sum(sample[rng.randrange(n)] for _ in range(n)) / n


def bootstrap_ci_mean_diff(a: list, b: list, n_boot: int = 10000,
                            ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap CI on mean(a) - mean(b). Resamples each group
    independently (unequal n is fine — a and b need not match in size).
    Raises ValueError on a group with fewer than 2 members (a single
    observation can't be meaningfully resampled)."""
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"need >=2 observations per group, got {len(a)}/{len(b)}")
    rng = random.Random(seed)
    diffs = sorted(
        _resample_mean(rng, a) - _resample_mean(rng, b) for _ in range(n_boot)
    )
    tail = (1.0 - ci) / 2.0
    lo_idx = int(n_boot * tail)
    hi_idx = int(n_boot * (1.0 - tail)) - 1
    return {
        "observed_diff": round(sum(a) / len(a) - sum(b) / len(b), 4),
        "ci_level": ci,
        "ci_low": round(diffs[lo_idx], 4),
        "ci_high": round(diffs[hi_idx], 4),
        "excludes_zero": diffs[lo_idx] > 0 or diffs[hi_idx] < 0,
        "n_boot": n_boot,
        "n_a": len(a),
        "n_b": len(b),
    }


def permutation_test_mean_diff(a: list, b: list, n_perm: int = 10000,
                                seed: int = 0) -> dict:
    """Two-sided permutation test on mean(a) - mean(b): pools both groups,
    repeatedly reassigns membership at random (preserving the original
    group sizes), and reports the fraction of permuted |spread| at least
    as large as the observed |spread| — the null hypothesis is that group
    membership carries no information about the Sharpe value."""
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"need >=2 observations per group, got {len(a)}/{len(b)}")
    rng = random.Random(seed)
    observed = sum(a) / len(a) - sum(b) / len(b)
    pooled = list(a) + list(b)
    n_a = len(a)
    at_least_as_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm_diff = sum(pooled[:n_a]) / n_a - sum(pooled[n_a:]) / len(b)
        if abs(perm_diff) >= abs(observed):
            at_least_as_extreme += 1
    return {
        "observed_diff": round(observed, 4),
        "p_value_two_sided": round(at_least_as_extreme / n_perm, 4),
        "n_perm": n_perm,
        "n_a": len(a),
        "n_b": len(b),
    }


def _mean_reversion_sharpes(rows: list) -> list:
    """Extracts one mean_reversion Sharpe per ticker, skipping fetch-error
    rows and any row where mean_reversion never populated (e.g. a ticker
    with zero trades for that strategy)."""
    return [r["mean_reversion"]["sharpe"] for r in rows
            if "error" not in r and "mean_reversion" in r]


def run_significance(group_rows: dict, n_boot: int = 10000, n_perm: int = 10000,
                      seed: int = 0) -> dict:
    """Pure aggregation over already-computed per-group per-ticker rows
    (same shape as illiquid_universe_probe.run_group's output) — no
    network, safe to unit test. group_rows: {"illiquid": [...],
    "moderate": [...], "liquid": [...]}."""
    sharpes = {name: _mean_reversion_sharpes(rows) for name, rows in group_rows.items()}
    out = {"n_mean_reversion_tickers": {name: len(s) for name, s in sharpes.items()}}
    comparisons = [("illiquid", "moderate"), ("illiquid", "liquid")]
    out["comparisons"] = {}
    for group_a, group_b in comparisons:
        a, b = sharpes.get(group_a, []), sharpes.get(group_b, [])
        key = f"{group_a}_vs_{group_b}"
        if len(a) < 2 or len(b) < 2:
            out["comparisons"][key] = {
                "skipped": True,
                "reason": f"insufficient tickers ({len(a)}/{len(b)}) for a group bootstrap/permutation",
            }
            continue
        out["comparisons"][key] = {
            "bootstrap_ci": bootstrap_ci_mean_diff(a, b, n_boot=n_boot, seed=seed),
            "permutation_test": permutation_test_mean_diff(a, b, n_perm=n_perm, seed=seed),
        }
    return out


if __name__ == "__main__":
    results = {}
    for name, tickers in [("illiquid", orig.ILLIQUID), ("moderate", orig.MODERATE),
                          ("liquid", orig.LIQUID)]:
        print(f"=== fetching {name} ===", file=sys.stderr)
        results[name] = orig.run_group(tickers)
    sig = run_significance(results)
    print(json.dumps(sig, indent=2))
