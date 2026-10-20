#!/usr/bin/env python3
"""scripts/exit_model_reconciliation.py — research test 1 ([RULE-REVIEW],
2026-10-20): which of the bot's exit models should be THE model?

research/bot_logic_research_2026-10-20.md Q1 / KNOWN BROKEN #48: the configured
exits (read by nothing), the backtest's legacy exits and the live R-ladder are
three different models that were never compared on the same inputs. This runs
backtest_v2 over a DECLARED universe with every exit model — legacy, live and
the two control arms (time_only, wide_stop) — for momentum and mean-reversion
entries, and reports, per arm, strategy and regime AT ENTRY:

  * net expectancy per trade (mean / median net %), win rate, holding days,
    the exit-reason mix;
  * the RANDOM-ENTRY BASE RATE for the same arm (REASONING STANDARD #3): the
    score gate replaced by a coin calibrated to the strategy's own entry
    frequency on that ticker, averaged over --seeds seeds, same regime gate,
    sizing and costs — and the strategy's ALPHA = expectancy minus that;
  * equity-curve Sharpe (mean over tickers, with Lo 2002's SE), CAGR, max DD;
  * the deflated-Sharpe threshold for the number of arms declared (Bailey &
    Lopez de Prado 2014, eq. 1): a winner is believed only above it.

Nothing live is touched: this is measurement. The report goes to research/
(--out), with the raw aggregates as JSON beside it. Caveats are printed in
the report itself (survivorship of the declared universe; the control arms
are close-checked and therefore slightly optimistic on stops vs the live
intrabar walk; single-position-per-ticker simulation; Yahoo bars when Alpaca
keys are absent).

Usage:
  python3 scripts/exit_model_reconciliation.py            # full run (~minutes)
  python3 scripts/exit_model_reconciliation.py --quick    # 4 tickers, 3 y, 3 seeds
  --years 10 --seeds 10 --workers 4 --out research/exit_model_reconciliation_<date>.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import date, datetime, timezone
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import backtest_v2 as bt  # noqa: E402

# DECLARED IN ADVANCE (REASONING STANDARD #4 — the variant count is part of
# the test): 30 liquid names with >= 10 y of daily history, index/sector ETFs
# plus large caps across sectors. SURVIVORSHIP: every name is a survivor and
# large; the live scanner's 12k-name universe (small caps included) is not
# represented — stated in the report, not hidden.
UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLY",
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "BAC", "XOM", "CVX",
    "JNJ", "PFE", "UNH", "HD", "WMT", "COST", "CAT", "BA", "DIS", "INTC",
]
STRATEGIES = ("momentum", "mean_reversion")
ARMS = tuple(bt.EXIT_MODELS)          # legacy, live, time_only, wide_stop — N=4 declared
REGIMES = ("BULL", "CAUTION", "NEUTRAL", "BEAR", "PANIC")

# PRIOR, restated from the research file BEFORE the run (REASONING STANDARD #10)
PRIOR = (
    "60% that the live R-ladder has LOWER net expectancy than the wide_stop "
    "control on momentum entries; 75% that any stop tighter than 2xATR hurts "
    "mean-reversion entries (time_only / wide_stop beat legacy and live there); "
    "50% on squeeze (not simulated: no short-interest history). Exploratory "
    "arms die 2026-12-31 if nothing clears the DSR threshold."
)

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """Bailey & Lopez de Prado (2014) eq. 1: E[max SR] of n zero-skill trials
    with Sharpe dispersion var_sharpe. The bar a chosen arm must clear."""
    if n_trials < 2 or var_sharpe <= 0:
        return 0.0
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    return math.sqrt(var_sharpe) * ((1 - EULER_GAMMA) * z(1 - 1 / n_trials)
                                    + EULER_GAMMA * z(1 - 1 / (n_trials * math.e)))


def lo_se_annual(sharpe_annual: float, t_days: int) -> float:
    """Lo (2002) eq. 9 under IID: SE(SR_daily) = sqrt((1 + SR^2/2)/T), annualized."""
    if t_days <= 1:
        return float("nan")
    sr_d = sharpe_annual / math.sqrt(bt.TRADING_DAYS)
    return math.sqrt((1 + sr_d * sr_d / 2) / t_days) * math.sqrt(bt.TRADING_DAYS)


def _days(years: float) -> int:
    return int(max(0.5, min(float(years), 10)) * 365) + 400


def _series(ticker: str, days: int):
    bars = bt.fetch_bars(ticker, days)
    spy = bars if ticker == "SPY" else bt.fetch_bars("SPY", days)
    try:
        vxx = bt.fetch_bars("VXX", days)
    except Exception:
        vxx = None
    return bars, spy, vxx


def eligible_bars(bars: dict, spy: dict, vxx) -> int:
    """Bars on which an entry is allowed by the regime gate (max_pos > 0)."""
    regimes, _ = bt.regime_series(spy, vxx)
    rbd = dict(zip(spy["date"], regimes))
    return sum(1 for d in bars["date"][:-1]
               if bt.get_adaptive_params(rbd.get(d, "NEUTRAL"))["max_pos"] > 0)


def run_job(job: dict) -> dict:
    """One simulate(): a strategy run (seed None) or a random-entry run."""
    ticker, strategy, arm, seed, rate, days = (job["ticker"], job["strategy"], job["arm"],
                                               job.get("seed"), job.get("rate"), job["days"])
    bars, spy, vxx = _series(ticker, days)
    out = bt.simulate(ticker, strategy, bars, spy, vxx, exit_model=arm,
                      random_entry=(rate, seed) if seed is not None else None)
    regimes, quality = bt.regime_series(spy, vxx)
    rbd = dict(zip(spy["date"], regimes))
    idx = {d: i for i, d in enumerate(bars["date"])}
    trades = []
    for t in out["trades"]:
        i = idx.get(t["date_entry"])
        # the gating regime is the SIGNAL bar's (entry_i - 1), never the fill bar's
        reg = rbd.get(bars["date"][i - 1], "NEUTRAL") if i else "NEUTRAL"
        trades.append({"net_pct": t["net_pct"], "pnl": t["pnl"], "hold": t["holding_days"],
                       "reason": t["exit_reason"], "regime": reg})
    return {"ticker": ticker, "strategy": strategy, "arm": arm, "seed": seed, "rate": rate,
            "n_trades": out["n_trades"], "sharpe": out["sharpe"], "cagr": out["cagr_pct"],
            "maxdd": out["max_drawdown_pct"], "t_days": len(bars["date"]),
            "data_quality": quality, "trades": trades}


def _mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def aggregate(results: list) -> dict:
    """Pure: job results -> the report's numbers."""
    strat = [r for r in results if r["seed"] is None]
    rnd = [r for r in results if r["seed"] is not None]
    by = {}
    for r in strat:
        key = (r["strategy"], r["arm"])
        b = by.setdefault(key, {"trades": [], "sharpes": [], "ses": [], "cagrs": [], "dds": [],
                                "tickers": set(), "random": []})
        b["trades"].extend(r["trades"])
        b["sharpes"].append(r["sharpe"])
        b["ses"].append(lo_se_annual(r["sharpe"], r["t_days"]))
        b["cagrs"].append(r["cagr"]); b["dds"].append(r["maxdd"]); b["tickers"].add(r["ticker"])
    for r in rnd:
        by.setdefault((r["strategy"], r["arm"]), {"trades": [], "sharpes": [], "ses": [], "cagrs": [],
                                                   "dds": [], "tickers": set(), "random": []})["random"].extend(r["trades"])
    out = {}
    for (strategy, arm), b in by.items():
        rows = {}
        for reg in ("ALL",) + REGIMES:
            ts = b["trades"] if reg == "ALL" else [t for t in b["trades"] if t["regime"] == reg]
            rs = b["random"] if reg == "ALL" else [t for t in b["random"] if t["regime"] == reg]
            if not ts and not rs:
                continue
            nets = [t["net_pct"] for t in ts]
            reasons = {}
            for t in ts:
                reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
            rand_mean = _mean([t["net_pct"] for t in rs])
            se = (statistics.pstdev(nets) / math.sqrt(len(nets))) if len(nets) > 1 else None
            rows[reg] = {
                "n": len(ts), "mean_net_pct": round(_mean(nets), 3) if ts else None,
                "se_net_pct": round(se, 3) if se is not None else None,
                "median_net_pct": round(statistics.median(nets), 3) if ts else None,
                "win_rate_pct": round(100 * sum(1 for x in nets if x > 0) / len(nets), 1) if ts else None,
                "hold_days": round(_mean([t["hold"] for t in ts]), 1) if ts else None,
                "exit_mix": {k: round(v / len(ts), 3) for k, v in sorted(reasons.items())} if ts else {},
                "random_n": len(rs), "random_mean_net_pct": round(rand_mean, 3) if rs else None,
                "alpha_pct_per_trade": round(_mean(nets) - rand_mean, 3) if ts and rs else None,
            }
        out[f"{strategy}|{arm}"] = {
            "strategy": strategy, "arm": arm, "tickers": len(b["tickers"]),
            "sharpe_mean": round(_mean(b["sharpes"]), 3) if b["sharpes"] else None,
            "sharpe_lo_se_mean": round(_mean([x for x in b["ses"] if x == x]), 3) if b["ses"] else None,
            "sharpe_sd_across_tickers": round(statistics.pstdev(b["sharpes"]), 3) if len(b["sharpes"]) > 1 else None,
            "cagr_mean": round(_mean(b["cagrs"]), 2) if b["cagrs"] else None,
            "maxdd_mean": round(_mean(b["dds"]), 2) if b["dds"] else None,
            "by_regime": rows,
        }
    # DSR bar per strategy: the dispersion of the N declared arms' mean Sharpes
    dsr = {}
    for strategy in STRATEGIES:
        srs = [v["sharpe_mean"] for k, v in out.items() if v["strategy"] == strategy and v["sharpe_mean"] is not None]
        var = statistics.pvariance(srs) if len(srs) > 1 else 0.0
        dsr[strategy] = {"n_arms": len(srs), "var_sharpe": round(var, 5),
                         "expected_max_sharpe_zero_skill": round(expected_max_sharpe(len(srs), var), 3)}
    # per-ticker preference: which arm has the best Sharpe per ticker/strategy
    pref = {}
    per = {}
    for r in strat:
        per.setdefault((r["strategy"], r["ticker"]), {})[r["arm"]] = r["sharpe"]
    for (strategy, _t), arms in per.items():
        best = max(arms, key=arms.get)
        pref.setdefault(strategy, {}).setdefault(best, 0)
        pref[strategy][best] += 1
    return {"cells": out, "dsr": dsr, "best_arm_count_by_ticker": pref}


def render(agg: dict, meta: dict) -> str:
    L = []
    L.append(f"# Exit-model reconciliation — research test 1 ({meta['date']})")
    L.append("")
    L.append("[RULE-REVIEW] measurement run of scripts/exit_model_reconciliation.py. "
             "Nothing live changed. research/bot_logic_research_2026-10-20.md Q1; KNOWN BROKEN #48.")
    L.append("")
    L.append(f"- Universe (declared, N={meta['n_universe']}): {', '.join(meta['universe'])}")
    L.append(f"- Years: {meta['years']} · arms declared: {', '.join(ARMS)} (N={len(ARMS)}) · random seeds per cell: {meta['seeds']}")
    L.append(f"- Data: {meta['data_source']} · tickers with degraded VXX coverage: {meta['degraded']}")
    L.append(f"- Runtime: {meta['runtime_s']} s on {meta['workers']} workers · generated {meta['generated_at']}")
    L.append("")
    L.append(f"PRIOR (stated before the run): {PRIOR}")
    L.append("")
    L.append("## Equity-curve view (mean over tickers; single position per ticker)")
    L.append("")
    L.append("| strategy | arm | tickers | Sharpe (mean) | Lo SE (mean) | SD across tickers | CAGR % | max DD % |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key in sorted(agg["cells"]):
        c = agg["cells"][key]
        L.append(f"| {c['strategy']} | {c['arm']} | {c['tickers']} | {c['sharpe_mean']} | {c['sharpe_lo_se_mean']} | {c['sharpe_sd_across_tickers']} | {c['cagr_mean']} | {c['maxdd_mean']} |")
    L.append("")
    for strategy, d in agg["dsr"].items():
        L.append(f"- DSR bar, {strategy}: with N={d['n_arms']} arms and Sharpe dispersion V={d['var_sharpe']}, "
                 f"a zero-skill best-of-N would show E[max SR] ≈ {d['expected_max_sharpe_zero_skill']}; "
                 f"an arm is believed only if its mean Sharpe clears that AND its Lo SE.")
    L.append(f"- Best arm by ticker (count of tickers each arm wins on Sharpe): {json.dumps(agg['best_arm_count_by_ticker'])}")
    L.append("")
    L.append("## Per-trade view by regime at entry (pooled across tickers)")
    L.append("")
    L.append("`alpha` = strategy expectancy − random-entry expectancy for the SAME arm, regime gate, sizing and costs.")
    L.append("")
    for strategy in STRATEGIES:
        L.append(f"### {strategy}")
        L.append("")
        L.append("| arm | regime | n | mean net % | ± SE | median % | win % | hold d | exit mix | random n | random mean % | alpha %/trade |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            c = agg["cells"].get(f"{strategy}|{arm}")
            if not c:
                continue
            for reg, r in c["by_regime"].items():
                mix = " ".join(f"{k}:{v:.0%}" for k, v in r["exit_mix"].items())
                L.append(f"| {arm} | {reg} | {r['n']} | {r['mean_net_pct']} | {r['se_net_pct']} | {r['median_net_pct']} | {r['win_rate_pct']} | {r['hold_days']} | {mix} | {r['random_n']} | {r['random_mean_net_pct']} | {r['alpha_pct_per_trade']} |")
        L.append("")
    L.append("## Caveats (bias direction stated)")
    L.append("")
    L.append("- SURVIVORSHIP: every universe name is a large, surviving ticker; the live scanner's small caps are absent → OPTIMISTIC for every arm alike, and silent on where the bot actually trades.")
    L.append("- The control arms are close-checked (legacy convention); the live arm walks bars intrabar → the controls are slightly OPTIMISTIC on stops relative to live.")
    L.append("- One position per ticker, regime-gated entries, backtest_v2's cost model; squeeze is not simulated (no short-interest history).")
    L.append("- The random baseline's rate is calibrated per (ticker, strategy, arm) to the strategy's own trade count over the regime-eligible bars.")
    L.append("- Multiple testing: 4 arms × 2 strategies × regimes; read every cell against the DSR bar and the Lo SE, not against zero. Per-trade `± SE` is the pooled standard error (trades treated as independent; overlapping regimes and shared market days make the true SE larger).")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="4 tickers, 3 years, 3 seeds (smoke)")
    ap.add_argument("--years", type=float, default=10)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tickers", default=None, help="comma list overriding the declared universe (smoke only)")
    a = ap.parse_args(argv)
    universe = list(UNIVERSE)
    years, seeds = a.years, a.seeds
    if a.quick:
        universe, years, seeds = universe[:4], 3, 3
    if a.tickers:
        universe = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    days = _days(years)
    t0 = time.time()
    # prefetch (and cache) every series in the parent so workers only read
    degraded = 0
    elig = {}
    for tk in universe:
        bars, spy, vxx = _series(tk, days)
        _, q = bt.regime_series(spy, vxx)
        degraded += q != "ok"
        elig[tk] = eligible_bars(bars, spy, vxx)
    jobs = [{"ticker": tk, "strategy": s, "arm": arm, "days": days}
            for tk in universe for s in STRATEGIES for arm in ARMS]
    with Pool(a.workers) as pool:
        strat_results = pool.map(run_job, jobs)
        rjobs = []
        for r in strat_results:
            n_elig = max(1, elig[r["ticker"]])
            rate = min(1.0, max(r["n_trades"], 1) / n_elig)
            for seed in range(1, seeds + 1):
                rjobs.append({"ticker": r["ticker"], "strategy": r["strategy"], "arm": r["arm"],
                              "days": days, "seed": seed, "rate": rate})
        rand_results = pool.map(run_job, rjobs)
    results = strat_results + rand_results
    agg = aggregate(results)
    meta = {"date": date.today().isoformat(), "universe": universe, "n_universe": len(universe),
            "years": years, "seeds": seeds, "workers": a.workers, "degraded": degraded,
            "data_source": "Alpaca" if (os.environ.get("ALPACA_KEY") and os.environ.get("ALPACA_SECRET")) else "Yahoo (no Alpaca keys)",
            "runtime_s": round(time.time() - t0, 1), "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = a.out or os.path.join(ROOT, "research", f"exit_model_reconciliation_{meta['date']}{'_quick' if a.quick else ''}.md")
    with open(out, "w") as f:
        f.write(render(agg, meta))
    with open(out[:-3] + ".json", "w") as f:
        json.dump({"meta": meta, "prior": PRIOR, **agg}, f, indent=1, default=list)
    print(f"wrote {out} (+ .json) in {meta['runtime_s']} s; jobs {len(jobs)} strategy + {len(rjobs)} random")
    return 0


if __name__ == "__main__":
    sys.exit(main())
