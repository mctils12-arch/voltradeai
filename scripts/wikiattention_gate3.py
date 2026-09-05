#!/usr/bin/env python3
"""
scripts/wikiattention_gate3.py — ROOT VALIDATION LADDER GATE 3 (LOGIC) for
wikimedia_pageviews_attention. GATE 2 (news-free control, 2026-09-04 session,
PR #1004) validated exactly one thing: forward TRADING VOLUME is elevated
after a news-free attention spike in small/mid-cap names. It explicitly made
NO price-direction claim and found NO realized-volatility effect (that
channel stayed unproven). This script asks the question GATE 2 left open:
can a simple, honest entry/exit rule built on the validated (news-free
spike) signal actually beat a same-universe/same-horizon random-entry base
rate, net of costs? Per the ladder's own definition (CLAUDE.md ROOT
VALIDATION LADDER, gate 5): "entry/exit rules are backtested by ablation
against the validated signal."

PRE-REGISTERED HYPOTHESIS AND RULE (REASONING STANDARD #10 — written before
any statistic below is computed by a real run):

  RULE UNDER TEST (exactly one, chosen before seeing any result, per
  REASONING STANDARD #4 — one diagnostic design per pass, no variant
  chasing): LONG-ONLY. Buy at the close of a news-free attention-spike day
  (the exact signal gate 2 validated — same z>=2.0 threshold, same 90-day
  trailing window, same "no 8-K on the spike day or the trading day before"
  filter as scripts/wikiattention_gate2_newsfree.py, reused by import rather
  than re-derived, EDGE DOCTRINE #3). Sell at the close of trading day
  spike_idx + h, h in {1, 3, 5} (the same three horizons gate 2 already
  tested). No shorts, no scaling, no stop — the simplest rule the validated
  signal could support, since gate 2 gives no basis for anything more
  elaborate (no direction, no magnitude, no volatility edge was validated).

  BASELINE (same convention as every prior gate-2/newsfree script): every
  trading day that is NOT a spike day (news-free or not), same forward-
  return computation, pooled the same way. This is the "same-universe,
  same-holding-period random entry" REASONING STANDARD #3 demands.

  COSTS: system_config.py's own BASE_CONFIG (imported, not re-derived per
  READ-BEFORE-WRITE #3) already carries SLIPPAGE_PCT=0.05%/side (liquid
  large caps) and SLIPPAGE_ILLIQUID=0.2%/side (names under 1M daily
  volume). This root's own primary group is explicitly the small/mid-cap,
  more-retail, less-liquid names (EDGE DOCTRINE #2 — "fish where whales
  can't"); applying SLIPPAGE_PCT there would understate real cost, so the
  small/mid-cap group's round-trip cost is 2 x SLIPPAGE_ILLIQUID = 0.40%,
  the mega-cap comparison group's is 2 x SLIPPAGE_PCT = 0.10%. Any
  ticker-specific ADV/spread data would sharpen this; none is wired into
  this repo today (same honest limitation the gate-2 cap-tier split itself
  already carries).

  PRIOR (stated before running, not fit after): LOW, ~15-20%. Three
  reasons, stated before any number below is computed: (1) gate 2 validated
  a VOLUME effect, not a DIRECTION or MAGNITUDE effect -- there is no
  gate-2 finding that forward price moves up, only that more shares trade;
  a naive long-only rule has no direct evidential support from gate 2
  itself, it is a genuinely separate empirical question. (2) SECOND-ORDER
  THINKING (REASONING STANDARD #5): the classic retail-attention finding in
  the literature (Barber & Odean 2008, "All That Glitters") is that
  attention-driven retail buying creates temporary price PRESSURE that
  partially reverses over the following days -- the opposite of what a
  long-only continuation rule needs, and exactly the kind of structural
  reason (who's on the other side, and why hasn't it been arbitraged) this
  file demands before crediting an edge. (3) COSTS FIRST (REASONING
  STANDARD #6): a 0.40% round-trip cost on small/mid-cap names is large
  relative to plausible short-horizon mean effects; even a real but modest
  edge could be cost-negative.

  VERDICT RULE (stated in advance): GATE 3 PASSES only if BOTH hold for the
  PRIMARY small/mid-cap group: (a) mean forward return on spike days
  significantly exceeds the baseline mean at the Bonferroni bar for this
  test family (3 horizons tested, alpha/3 ~= 0.0167 -- narrower family than
  gate 2's 10-cell one since this is a single metric x single group x 3
  horizons); AND (b) the spike sample's mean return net of the round-trip
  cost above is itself positive (beating the base rate on a pre-cost basis
  while being unprofitable net of costs is not a tradeable rule, it is a
  restated base-rate comparison). If either fails, GATE 3 is NOT PASSED and
  the root's LOGIC layer is the diagnosed fault layer (ROOT VALIDATION
  LADDER: "a failure at layer N with 1..N-1 verified is a fault AT layer N"
  -- DATA and SIGNAL both passed here, so a LOGIC-layer failure is
  localized, not a re-litigation of gate 1/2). The mega-cap comparison
  group is informational only, same secondary-group convention as gate 2.

REUSES (EDGE DOCTRINE #3 — compile once): wikiattention_gate2.py's
zscore_series/spike_day_indices/pool machinery and
wikiattention_gate2_newsfree.py's is_newsfree_spike_idx/fetch_cik_map/
fetch_8k_dates_for_cik/fetch_wiki_daily_views plumbing, both imported by
path rather than re-derived.
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

HORIZONS = (1, 3, 5)
BONFERRONI_FAMILY_SIZE = 3  # 3 horizons, 1 metric, 1 primary group -- see docstring VERDICT RULE


def _slippage_costs() -> dict:
    """Reads system_config.cfg directly rather than hardcoding a number
    (READ BEFORE WRITE #3) -- round-trip = 2x the per-side constant already
    used elsewhere in this codebase for exactly this liquidity split."""
    sys.path.insert(0, os.path.dirname(_HERE))
    from system_config import cfg
    return {
        "small_mid_round_trip": 2 * cfg["SLIPPAGE_ILLIQUID"],
        "mega_round_trip": 2 * cfg["SLIPPAGE_PCT"],
    }


# ── Pure functions (unit-tested, no network) ────────────────────────────────

def forward_return(closes: Sequence[float], idx: int, horizon: int) -> Optional[float]:
    """Buy-at-close[idx], sell-at-close[idx+horizon] simple return. None if
    the horizon runs past the end of the series (no fabricated fill)."""
    j = idx + horizon
    if j >= len(closes) or closes[idx] == 0:
        return None
    return closes[j] / closes[idx] - 1


def evaluate_ticker_gate3(dates: Sequence[str], views: Sequence[Optional[float]],
                           closes: Sequence[float], filing_dates: set,
                           threshold: float = wag2.DEFAULT_Z_THRESHOLD,
                           window: int = wag2.DEFAULT_TRAILING_WINDOW,
                           horizons: Sequence[int] = HORIZONS) -> dict:
    """Mirrors wikiattention_gate2_newsfree.evaluate_ticker_newsfree's
    sample/baseline split exactly (news-free spikes = sample, every
    non-spike day = baseline), but computes forward RETURN instead of
    forward volume ratio / realized vol -- the LOGIC-layer question gate 2
    never asked."""
    n = len(dates)
    z = wag2.zscore_series(views, window)
    all_spikes = set(wag2.spike_day_indices(z, threshold))
    newsfree_spikes = {i for i in all_spikes if wagnf.is_newsfree_spike_idx(i, dates, filing_dates)}
    out = {
        "n_days": n,
        "n_spike_days_total": len(all_spikes),
        "n_spike_days_newsfree": len(newsfree_spikes),
        "horizons": {},
    }
    for h in horizons:
        ret_spike, ret_base = [], []
        for i in range(n):
            r = forward_return(closes, i, h)
            if r is None:
                continue
            if i in newsfree_spikes:
                ret_spike.append(r)
            elif i not in all_spikes:  # baseline excludes ALL spike days, news-free or not
                ret_base.append(r)
        out["horizons"][h] = {
            "forward_return": wag2.welch_vs_baseline(ret_spike, ret_base),
            "_raw": {"ret_spike": ret_spike, "ret_base": ret_base},
        }
    return out


def pool_returns(per_ticker: dict, tickers: Sequence[str], horizons: Sequence[int]) -> dict:
    out = {}
    for h in horizons:
        rs, rb = [], []
        for t in tickers:
            row = per_ticker.get(t, {}).get("horizons", {}).get(h)
            if not row or "_raw" not in row:
                continue
            rs.extend(row["_raw"]["ret_spike"])
            rb.extend(row["_raw"]["ret_base"])
        welch = wag2.welch_vs_baseline(rs, rb)
        out[h] = {
            "forward_return": welch,
            "n_tickers_pooled": sum(1 for t in tickers if t in per_ticker and "horizons" in per_ticker[t]),
        }
    return out


def apply_verdict(pooled_small_mid: dict, round_trip_cost: float,
                   family_size: int = BONFERRONI_FAMILY_SIZE) -> dict:
    """Applies the pre-registered verdict rule (see module docstring) to
    the primary small/mid-cap pooled result. Pure function, unit-testable
    against synthetic pooled dicts without any network/run dependency."""
    alpha = 0.05 / family_size
    per_horizon = {}
    all_pass = True
    any_data = False
    for h, row in pooled_small_mid.items():
        wr = row.get("forward_return")
        if wr is None:
            per_horizon[h] = {"status": "insufficient_data"}
            all_pass = False
            continue
        any_data = True
        beats_baseline = wr["p_value"] < alpha and wr["mean_diff"] > 0
        net_of_cost = wr["mean"] - round_trip_cost
        profitable_net = net_of_cost > 0
        per_horizon[h] = {
            "p_value": wr["p_value"],
            "alpha_bar": round(alpha, 5),
            "beats_baseline_significant": beats_baseline,
            "mean_spike_return": wr["mean"],
            "mean_baseline_return": wr["baseline_mean"],
            "round_trip_cost": round(round_trip_cost, 4),
            "net_of_cost_return": round(net_of_cost, 4),
            "profitable_net_of_cost": profitable_net,
            "horizon_pass": bool(beats_baseline and profitable_net),
        }
        if not (beats_baseline and profitable_net):
            all_pass = False
    return {
        "gate3_pass": bool(all_pass and any_data),
        "alpha_bar": round(alpha, 5),
        "per_horizon": per_horizon,
    }


# ── Network orchestration ───────────────────────────────────────────────────

def run_gate3(tickers: Sequence[str], days: int = wag2.DEFAULT_MIN_TRADING_DAYS + 250,
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
    costs = _slippage_costs()

    per_ticker = {}
    coverage_flags = {}
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
            coverage_flags[t] = fully_covered
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
        row = evaluate_ticker_gate3(trading_dates, views, bars["close"], filing_dates, threshold, window, horizons)
        row["cap_tier"] = "mega" if t in wag2.MEGA_CAP_TICKERS else "small_mid"
        row["edgar_coverage_full"] = fully_covered
        per_ticker[t] = row

    small_mid = [t for t in tickers if t not in wag2.MEGA_CAP_TICKERS]
    mega = [t for t in tickers if t in wag2.MEGA_CAP_TICKERS]
    pooled_small_mid = pool_returns(per_ticker, small_mid, horizons)
    pooled_mega = pool_returns(per_ticker, mega, horizons)
    verdict = apply_verdict(pooled_small_mid, costs["small_mid_round_trip"])

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
        "edgar_coverage_incomplete_tickers": [t for t, ok in coverage_flags.items() if not ok],
        "costs": costs,
        "pooled": {"small_mid_cap": pooled_small_mid, "mega_cap_comparison": pooled_mega},
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
    args = ap.parse_args()

    tickers = args.tickers.split(",") if args.tickers else list(wag2._wiki_articles().keys())
    horizons = [int(h) for h in args.horizons.split(",")]
    result = run_gate3(tickers, args.days, args.threshold, args.window, horizons)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
