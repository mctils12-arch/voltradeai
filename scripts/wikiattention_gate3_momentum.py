#!/usr/bin/env python3
"""
scripts/wikiattention_gate3_momentum.py — a SEPARATE, freshly pre-registered
GATE 3 (LOGIC) experiment for wikimedia_pageviews_attention, distinct from
scripts/wikiattention_gate3.py's already-CLOSED long-only spec (2026-09-05
session, NOT PASSED, sign-unstable across 4 live draws — see that file's
own docstring and datacore/signal_ladder.json's UPDATE paragraph).

That session's own NEXT(2) named, but deliberately did NOT test (REASONING
STANDARD #4 — one diagnostic design per pass), a genuinely different
question: "does the spike day's OWN same-day return predict continuation
or reversal over the next 1-5 days, independent of the (now-closed)
unconditional long-only question?" This script is that experiment. It does
not re-test the closed spec (RECURRENCE ESCALATES-style discipline: this
is a fresh hypothesis, not a variant chase of the killed one).

PRE-REGISTRATION (REASONING STANDARD #10 — written before any statistic
below is computed by a real run):

  CONDITIONING VARIABLE: for each news-free attention-spike day i (the
  EXACT validated signal from gate 2 / gate 2-newsfree — same z>=2.0/
  90-day trailing window/no-8-K-on-day-or-day-before filter, reused by
  import, EDGE DOCTRINE #3, never re-derived), compute
  same_day_return[i] = closes[i] / closes[i-1] - 1 — the spike day's own
  close-to-close move. This is known at the close of day i, the same
  entry timestamp the (closed) long-only spec already used — no lookahead.

  BUCKETS: UP = same_day_return > 0, DOWN = same_day_return < 0 (exact
  zero excluded — expected to be negligible/zero in practice for
  continuously-priced equities).

  METRIC: forward_return(closes, i, h) for h in {1, 3, 5} — the IDENTICAL
  function and entry/exit convention wikiattention_gate3.py already uses
  (buy close[i], sell close[i+h]), reused by import rather than
  re-derived. This asks: after an attention spike that ALREADY moved up
  (or down) that same day, does the subsequent 1-5 day return continue in
  that direction (MOMENTUM) or move opposite to it (REVERSAL)?

  PRIMARY TEST (one, chosen before seeing any result): a Welch two-sample
  t-test of the UP bucket's forward returns vs the DOWN bucket's forward
  returns (wikiattention_gate2.welch_vs_baseline, reused as a generic
  two-sample test — it makes no assumption about which argument is a
  "baseline"). PRE-DECLARED PRIMARY HORIZON: h=5 — the multi-day horizon
  the literature's reversal effect (Barber & Odean 2008, cited in the
  closed long-only spec's own prior) operates over, chosen before running
  anything, not selected after seeing which horizon looked best. h=1 and
  h=3 are secondary/confirmatory, Bonferroni-corrected across the
  3-horizon family (alpha/3 ~= 0.0167), same family-size convention as
  wikiattention_gate3.py's own VERDICT RULE.

  MINIMUM VIABLE BUCKET SIZE: 15 observations per bucket per horizon
  (stricter than welch_vs_baseline's own hard n>=5/side floor, which only
  gates whether the test computes at all — this is the separate,
  pre-registered bar for trusting the RESULT, matching this codebase's
  general practice of a stated minimum-N bar before rendering PASS/FAIL,
  e.g. rail_traffic_gate2.py's n_bucket>=20, settlement_stress_gate2.py's
  MIN_DOMESTIC_EPISODES=20). Below this bar: INSUFFICIENT_DATA, no
  PASS/FAIL rendered (REASONING STANDARD #4 — distrust proportional to
  how little was tested).

  CLASSIFICATION: at the primary horizon, if the Welch test on UP vs DOWN
  clears the Bonferroni bar, classify as:
    MOMENTUM  if mean(UP) > mean(DOWN) (spike continues in its own
              same-day direction);
    REVERSAL  if mean(UP) < mean(DOWN) (spike direction reverses).
  A significant spread with an ambiguous sign pattern (e.g. both bucket
  means positive, or both negative) is still classified by the SPREAD's
  sign alone — the spread, not the individual bucket signs, is what a
  long-the-favorable/short-the-unfavorable pair trade actually captures.

  TRADEABLE-RULE CHECK (per ROOT VALIDATION LADDER gate 3's own
  definition — "entry/exit rules are backtested... against the validated
  signal"): the implied rule is a PAIR position, long the bucket whose
  mean return is higher and short the bucket whose mean return is lower,
  entered simultaneously at the spike-day close. COSTS: system_config's
  own SLIPPAGE_ILLIQUID (this root's primary group is small/mid-cap,
  same EDGE DOCTRINE #2 reasoning as wikiattention_gate3.py) applied as a
  full 2x round-trip cost to EACH leg independently (two separate
  single-name positions, not a netted spread order) — GATE 3 PASSES only
  if the combined pair (long leg's mean return - round_trip_cost) +
  (-1 x short leg's mean return - round_trip_cost) is itself positive,
  in addition to the primary Welch test clearing its Bonferroni bar.

  PRIOR (stated before running, not fit after): LOW-TO-MODERATE, ~20-25%
  — slightly above the closed long-only spec's own 15-20% prior, because
  a conditional (direction-aware) rule has more room to find structure
  than an unconditional one, but still discounted for the same
  SECOND-ORDER reasons that spec's prior already established (Barber &
  Odean's own finding is about RETAIL FLOW pressure broadly, not
  specifically conditioned on the spike day's own realized return sign —
  applying it here is an extrapolation, not a direct citation) and for
  REASONING STANDARD #4 (this is the second diagnostic design tried on
  this same root's price-direction question; a family-wise view, not
  just this test's own p-value, should discount any marginal finding).

REUSES (EDGE DOCTRINE #3): wikiattention_gate3.py's forward_return/
_slippage_costs/run-time network plumbing (which itself reuses gate 2 /
gate 2-newsfree), imported by path — this script adds only the
same-day-return conditioning and bucket split, nothing else is
re-derived.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wag2 = _load("wikiattention_gate2", "wikiattention_gate2.py")
wagnf = _load("wikiattention_gate2_newsfree", "wikiattention_gate2_newsfree.py")
wag3 = _load("wikiattention_gate3", "wikiattention_gate3.py")

HORIZONS = (1, 3, 5)
PRIMARY_HORIZON = 5
BONFERRONI_FAMILY_SIZE = 3
MIN_N_PER_BUCKET = 15


def same_day_return(closes: Sequence[float], idx: int) -> Optional[float]:
    """The spike day's own close-to-close move. None at idx==0 (no prior
    close to compare) or a zero prior close (never fabricated)."""
    if idx <= 0 or closes[idx - 1] == 0:
        return None
    return closes[idx] / closes[idx - 1] - 1


def evaluate_ticker_momentum(dates: Sequence[str], views: Sequence[Optional[float]],
                              closes: Sequence[float], filing_dates: set,
                              threshold: float = wag2.DEFAULT_Z_THRESHOLD,
                              window: int = wag2.DEFAULT_TRAILING_WINDOW,
                              horizons: Sequence[int] = HORIZONS) -> dict:
    """Splits this ticker's news-free spike days into UP/DOWN buckets by
    same_day_return sign, then computes forward_return (wag3's own
    buy-at-spike-close/sell-at-close+h convention) for each bucket at
    every horizon. Pure given its inputs — no network."""
    n = len(dates)
    z = wag2.zscore_series(views, window)
    all_spikes = set(wag2.spike_day_indices(z, threshold))
    newsfree_spikes = [i for i in all_spikes if wagnf.is_newsfree_spike_idx(i, dates, filing_dates)]

    up_idx, down_idx = [], []
    for i in newsfree_spikes:
        sdr = same_day_return(closes, i)
        if sdr is None or sdr == 0:
            continue
        (up_idx if sdr > 0 else down_idx).append(i)

    out = {
        "n_days": n,
        "n_spike_days_newsfree": len(newsfree_spikes),
        "n_up": len(up_idx),
        "n_down": len(down_idx),
        "horizons": {},
    }
    for h in horizons:
        ret_up = [r for i in up_idx if (r := wag3.forward_return(closes, i, h)) is not None]
        ret_down = [r for i in down_idx if (r := wag3.forward_return(closes, i, h)) is not None]
        out["horizons"][h] = {
            "up_vs_down": wag2.welch_vs_baseline(ret_up, ret_down),
            "_raw": {"ret_up": ret_up, "ret_down": ret_down},
        }
    return out


def pool_buckets(per_ticker: dict, tickers: Sequence[str], horizons: Sequence[int]) -> dict:
    out = {}
    for h in horizons:
        ru, rd = [], []
        n_pooled = 0
        for t in tickers:
            row = per_ticker.get(t, {}).get("horizons", {}).get(h)
            if not row or "_raw" not in row:
                continue
            ru.extend(row["_raw"]["ret_up"])
            rd.extend(row["_raw"]["ret_down"])
            n_pooled += 1
        out[h] = {
            "up_vs_down": wag2.welch_vs_baseline(ru, rd),
            "n_up": len(ru),
            "n_down": len(rd),
            "n_tickers_pooled": n_pooled,
        }
    return out


def apply_momentum_verdict(pooled: dict, round_trip_cost: float,
                            primary_horizon: int = PRIMARY_HORIZON,
                            min_n_per_bucket: int = MIN_N_PER_BUCKET,
                            family_size: int = BONFERRONI_FAMILY_SIZE) -> dict:
    """Pure function applying the pre-registered verdict rule (module
    docstring) to a pooled {horizon: {"up_vs_down": welch_result|None,
    "n_up": int, "n_down": int}} dict. Unit-testable against synthetic
    pooled dicts, no network dependency."""
    alpha = 0.05 / family_size
    per_horizon = {}
    for h, row in pooled.items():
        if row["n_up"] < min_n_per_bucket or row["n_down"] < min_n_per_bucket or not row.get("up_vs_down"):
            per_horizon[h] = {
                "status": "insufficient_data",
                "n_up": row["n_up"], "n_down": row["n_down"],
                "min_required": min_n_per_bucket,
            }
            continue
        wr = row["up_vs_down"]
        significant = wr["p_value"] < alpha
        classification = "momentum" if wr["mean_diff"] > 0 else "reversal"
        # long the higher-mean bucket, short the lower-mean bucket
        long_mean, short_mean = (wr["mean"], wr["baseline_mean"]) if wr["mean_diff"] > 0 else (wr["baseline_mean"], wr["mean"])
        pair_return_net = (long_mean - round_trip_cost) + (-short_mean - round_trip_cost)
        per_horizon[h] = {
            "n_up": row["n_up"], "n_down": row["n_down"],
            "p_value": wr["p_value"],
            "alpha_bar": round(alpha, 5),
            "significant": significant,
            "classification": classification,
            "mean_up": wr["mean"], "mean_down": wr["baseline_mean"],
            "pair_return_net_of_cost": round(pair_return_net, 4),
            "pair_profitable_net_of_cost": bool(pair_return_net > 0),
            "horizon_pass": bool(significant and pair_return_net > 0),
        }
    primary = per_horizon.get(primary_horizon, {"status": "insufficient_data"})
    gate3_pass = bool(primary.get("horizon_pass"))
    return {
        "gate3_pass": gate3_pass,
        "primary_horizon": primary_horizon,
        "primary_result": primary,
        "alpha_bar": round(alpha, 5),
        "per_horizon": per_horizon,
    }


# ── Network orchestration ───────────────────────────────────────────────────

def run_momentum_gate3(tickers: Sequence[str], days: int = wag2.DEFAULT_MIN_TRADING_DAYS + 250,
                        threshold: float = wag2.DEFAULT_Z_THRESHOLD,
                        window: int = wag2.DEFAULT_TRAILING_WINDOW,
                        horizons: Sequence[int] = HORIZONS,
                        wiki_spacing_s: float = 0.6, sec_spacing_s: float = 0.3) -> dict:
    articles = wag2._wiki_articles()
    end = datetime.utcnow() - timedelta(days=1)
    start = end - timedelta(days=days)
    end_s, start_s = end.strftime("%Y%m%d"), start.strftime("%Y%m%d")
    cutoff_iso = start.strftime("%Y-%m-%d")

    sys.path.insert(0, os.path.dirname(_HERE))
    import backtest_v2 as bt

    cik_map = wagnf.fetch_cik_map()
    costs = wag3._slippage_costs()

    per_ticker = {}
    for t in tickers:
        article = articles.get(t)
        cik10 = cik_map.get(t)
        if not article or not cik10:
            per_ticker[t] = {"error": f"missing {'article' if not article else 'CIK'} for ticker"}
            continue
        try:
            views_by_date = wag2.fetch_wiki_daily_views(article, start_s, end_s)
        except Exception as e:
            per_ticker[t] = {"error": f"wiki fetch failed: {e}"}
            continue
        finally:
            time.sleep(wiki_spacing_s)
        try:
            filing_dates, fully_covered = wagnf.fetch_8k_dates_for_cik(cik10, cutoff_iso)
        except Exception as e:
            per_ticker[t] = {"error": f"EDGAR submissions fetch failed: {e}"}
            continue
        finally:
            time.sleep(sec_spacing_s)
        try:
            bars = bt.fetch_bars(t, days)
            if not bars or not bars.get("date"):
                raise RuntimeError("empty bars")
        except Exception as e:
            per_ticker[t] = {"error": f"price fetch failed: {e}"}
            continue
        trading_dates = bars["date"]
        views = wag2.align_views_to_trading_days(views_by_date, trading_dates)
        row = evaluate_ticker_momentum(trading_dates, views, bars["close"], filing_dates, threshold, window, horizons)
        row["cap_tier"] = "mega" if t in wag2.MEGA_CAP_TICKERS else "small_mid"
        per_ticker[t] = row

    small_mid = [t for t in tickers if t not in wag2.MEGA_CAP_TICKERS]
    pooled_small_mid = pool_buckets(per_ticker, small_mid, horizons)
    verdict = apply_momentum_verdict(pooled_small_mid, costs["small_mid_round_trip"])

    per_ticker_summary = {}
    for t, row in per_ticker.items():
        if "horizons" not in row:
            per_ticker_summary[t] = row
            continue
        trimmed = dict(row)
        trimmed["horizons"] = {h: {k: v for k, v in hrow.items() if k != "_raw"} for h, hrow in row["horizons"].items()}
        per_ticker_summary[t] = trimmed

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window_days": days,
        "threshold": threshold,
        "trailing_window": window,
        "horizons": list(horizons),
        "tickers_requested": list(tickers),
        "costs": costs,
        "pooled_small_mid_cap": pooled_small_mid,
        "verdict": verdict,
        "per_ticker": per_ticker_summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default=None, help="comma-separated; default = full seed set")
    ap.add_argument("--days", type=int, default=wag2.DEFAULT_MIN_TRADING_DAYS + 250)
    ap.add_argument("--threshold", type=float, default=wag2.DEFAULT_Z_THRESHOLD)
    ap.add_argument("--window", type=int, default=wag2.DEFAULT_TRAILING_WINDOW)
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--wiki-spacing", type=float, default=0.6)
    args = ap.parse_args()

    tickers = args.tickers.split(",") if args.tickers else [t for t in wag2._wiki_articles().keys() if t not in wag2.MEGA_CAP_TICKERS]
    horizons = [int(h) for h in args.horizons.split(",")]
    result = run_momentum_gate3(tickers, args.days, args.threshold, args.window, horizons, wiki_spacing_s=args.wiki_spacing)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
