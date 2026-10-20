#!/usr/bin/env python3
"""scripts/floor_basket_vs_index.py — research test 4 (gate 3 → 4): the passive
ETF FLOOR's composition vs SPY under the LIVE regime-driven exposure schedule
([RULE-REVIEW] measurement, 2026-10-20; research/bot_logic_research_2026-10-20.md
Q4). The composition is the human's decision — this is the evidence request.

PRIOR (stated before the run; the 2026-10-14 entry, unchanged): 55% that
SPY-core has the better benchmark-relative Sharpe; 35% that it has the
higher CAGR.

ARMS (declared in advance, N=4 — share of the floor TARGET):
  basket       QQQ .50 / SMH .15 / KWEB .10 / VXUS .15 / CASH .10 (as configured)
  spy_core     SPY .65 (QQQ+SMH → SPY) / KWEB .10 / VXUS .15 / CASH .10
  fifty_fifty  QQQ .325 / SPY .325 / KWEB .10 / VXUS .15 / CASH .10
  qqq_only     QQQ 1.00 (the v1.0.29 original)
REFERENCES: spy_same_exposure (SPY 1.00 of the floor target — the beta-matched
benchmark the research names) and spy_buy_hold (100% SPY always, no GLD —
the GOAL's yardstick).

POLICY, from system_config.py BASE_CONFIG and bot_engine._manage_spy_floor /
_manage_defensive_floor, with the approximations STATED:
  * floor target by regime: SPY_FLOOR_BULL .70 / NEUTRAL .90 / CAUTION .35 /
    BEAR 0 / PANIC 0; GLD defensive floor DEFENSIVE_FLOOR_BULL .08 / NEUTRAL
    .08 / CAUTION .15 / BEAR .30 / PANIC .40 of equity — identical across arms;
  * the QQQ trend override caps the floor target at .50 when QQQ closes below
    its 200-day MA and at .20 under a death cross (50d < 200d and price <
    200d) — applied on QQQ's trend for EVERY arm (live it keys on the floor
    ticker, QQQ);
  * regime labels from backtest_v2.regime_series (VXX ratio + SPY moving
    averages); VXX's series-B history starts 2018-01 — earlier days classify
    with a neutral VXX ratio (the classifier's own "degraded" path, reported);
  * decide at the close, hold through the next day (one-day lag); rebalance
    when the regime or the trend cap changes, or when any member drifts more
    than FLOOR_BASKET_DRIFT_THRESHOLD (3% of equity) from target; cost
    COST_BPS per side on traded value (spread under the live limit-order
    regime; stated, not measured);
  * the rest of equity is cash at 0% — identical across arms, so arm
    differences come from the floor's composition alone. The ACTIVE sleeves
    are out of scope here (they are the other sleeves' own tests).

METRICS per arm (daily returns of the whole equity): CAGR, max drawdown,
Sharpe ± Lo (2002) SE, beta and Jensen alpha vs SPY (rf 0), tracking error
and information ratio vs spy_same_exposure, the Ledoit–Wolf (2008) 95%
interval of (Sharpe_arm − Sharpe_spy_same_exposure) under the iid
approximation with the realized correlation (stated), the DSR bar with N=4
(Bailey–López de Prado E[max SR] over the arms' Sharpe dispersion), and per
regime: annualized mean, Sharpe, days.

Run: python3 scripts/floor_basket_vs_index.py [--years 10.8] [--out research/]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

PRIOR = ("55% that SPY-core has the better benchmark-relative Sharpe than the configured basket; "
         "35% that it has the higher CAGR (the 2026-10-14 entry, unchanged).")
TRADING_DAYS = 252
COST_BPS = 5.0
DRIFT_THRESHOLD = 0.03
REGIMES = ("BULL", "NEUTRAL", "CAUTION", "BEAR", "PANIC")
FLOOR_TARGET = {"BULL": 0.70, "NEUTRAL": 0.90, "CAUTION": 0.35, "BEAR": 0.0, "PANIC": 0.0}
GLD_TARGET = {"BULL": 0.08, "NEUTRAL": 0.08, "CAUTION": 0.15, "BEAR": 0.30, "PANIC": 0.40}
TREND_CAP = {"early": 0.50, "death": 0.20}
ARMS: Dict[str, Dict[str, float]] = {
    "basket": {"QQQ": 0.50, "SMH": 0.15, "KWEB": 0.10, "VXUS": 0.15},
    "spy_core": {"SPY": 0.65, "KWEB": 0.10, "VXUS": 0.15},
    "fifty_fifty": {"QQQ": 0.325, "SPY": 0.325, "KWEB": 0.10, "VXUS": 0.15},
    "qqq_only": {"QQQ": 1.0},
}
REFERENCES: Dict[str, Dict[str, float]] = {"spy_same_exposure": {"SPY": 1.0}}
BUY_HOLD = "spy_buy_hold"
SYMBOLS = ("SPY", "QQQ", "SMH", "KWEB", "VXUS", "GLD")


def _load_emr():
    spec = importlib.util.spec_from_file_location("emr", os.path.join(HERE, "exit_model_reconciliation.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── policy ─────────────────────────────────────────────────────────────────

def sma(xs: Sequence[float], i: int, n: int) -> Optional[float]:
    if i + 1 < n:
        return None
    return sum(xs[i + 1 - n:i + 1]) / n


def qqq_trend(closes: Sequence[float], i: int) -> Optional[str]:
    """None (normal), "early" (price < 200d MA) or "death" (50d < 200d and price < 200d);
    None before 200 bars exist (no override possible, as live)."""
    ma200 = sma(closes, i, 200)
    if ma200 is None:
        return None
    ma50 = sma(closes, i, 50)
    if closes[i] < ma200:
        return "death" if (ma50 is not None and ma50 < ma200) else "early"
    return None


def targets(regime: str, trend: Optional[str], weights: Dict[str, float], buy_hold: bool = False) -> Dict[str, float]:
    """Equity-share targets for one day: the floor target (regime, capped by the
    QQQ trend) split by `weights`, plus the GLD defensive floor; the rest is cash."""
    if buy_hold:
        return {"SPY": 1.0}
    f = FLOOR_TARGET[regime]
    if trend in TREND_CAP:
        f = min(f, TREND_CAP[trend])
    out = {sym: round(w * f, 10) for sym, w in weights.items() if w > 0 and f > 0}
    g = GLD_TARGET[regime]
    if g > 0:
        out["GLD"] = round(out.get("GLD", 0.0) + g, 10)
    return out


# ── simulation ─────────────────────────────────────────────────────────────

def simulate(prices: Dict[str, Sequence[float]], regimes: Sequence[str], weights: Dict[str, float],
             buy_hold: bool = False, cost_bps: float = COST_BPS, drift: float = DRIFT_THRESHOLD) -> dict:
    """Daily equity path (start 1.0) for one arm. prices: aligned close lists per
    symbol (same length as regimes). Decide at the close of day t from regime[t]
    and QQQ's trend at t; the allocation earns day t+1's returns."""
    n = len(regimes)
    qqq = prices["QQQ"]
    syms = sorted(set(list(weights) + ["GLD", "SPY"]))
    hold = {s: 0.0 for s in syms}      # value held per symbol
    cash = 1.0
    equity = [1.0]
    rebalances = 0
    turnover = 0.0
    costs = 0.0
    last_key: Optional[Tuple[str, Optional[str]]] = None
    alloc_regime: List[str] = []       # the regime that set each day's allocation (for the per-regime split)
    for t in range(n):
        if t > 0:
            for s in syms:
                if hold[s]:
                    hold[s] *= prices[s][t] / prices[s][t - 1]
        eq = cash + sum(hold.values())
        if t > 0:
            equity.append(eq)
        trend = qqq_trend(qqq, t)
        tgt = targets(regimes[t], trend, weights, buy_hold)
        key = (regimes[t], trend)
        need = last_key != key or any(abs(hold.get(s, 0.0) / eq - tgt.get(s, 0.0)) > drift for s in set(syms) | set(tgt))
        if need:
            traded = 0.0
            for s in syms:
                want = tgt.get(s, 0.0) * eq
                have = hold.get(s, 0.0)
                if abs(want - have) > 1e-12:
                    traded += abs(want - have)
                    cash -= (want - have)
                    hold[s] = want
            if traded > 1e-12:   # a regime change with identical targets (buy-and-hold) trades nothing
                fee = traded * cost_bps / 10_000.0
                cash -= fee
                costs += fee
                turnover += traded / eq if eq else 0.0
                rebalances += 1
            last_key = key
        alloc_regime.append(regimes[t])
    return {"equity": equity, "rebalances": rebalances, "turnover": round(turnover, 3), "costs": round(costs, 6),
            "alloc_regime": alloc_regime}


def daily_returns(equity: Sequence[float]) -> List[float]:
    return [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity))]


# ── statistics ─────────────────────────────────────────────────────────────

def _mean(x: Sequence[float]) -> float:
    return sum(x) / len(x) if x else float("nan")


def _std(x: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))


def sharpe_annual(r: Sequence[float]) -> float:
    """Annualized Sharpe (rf 0); NaN for a (numerically) zero-variance series —
    ten identical floats can leave a 1e-18 sample deviation, not an edge."""
    s = _std(r)
    if not (s == s) or s < 1e-12:
        return float("nan")
    return (_mean(r) / s) * math.sqrt(TRADING_DAYS)


def cagr(equity: Sequence[float]) -> float:
    n = len(equity) - 1
    if n <= 0 or equity[0] <= 0 or equity[-1] <= 0:
        return float("nan")
    return (equity[-1] / equity[0]) ** (TRADING_DAYS / n) - 1.0


def max_drawdown(equity: Sequence[float]) -> float:
    peak, mdd = -math.inf, 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1.0)
    return mdd


def beta_alpha(r: Sequence[float], rb: Sequence[float]) -> Tuple[float, float]:
    """OLS beta and annualized Jensen alpha vs the benchmark (rf 0)."""
    n = min(len(r), len(rb))
    if n < 3:
        return float("nan"), float("nan")
    mr, mb = _mean(r[:n]), _mean(rb[:n])
    cov = sum((r[i] - mr) * (rb[i] - mb) for i in range(n)) / (n - 1)
    var = sum((rb[i] - mb) ** 2 for i in range(n)) / (n - 1)
    beta = cov / var if var > 0 else float("nan")
    alpha = (mr - beta * mb) * TRADING_DAYS if var > 0 else float("nan")
    return beta, alpha


def te_ir(r: Sequence[float], rb: Sequence[float]) -> Tuple[float, float]:
    n = min(len(r), len(rb))
    d = [r[i] - rb[i] for i in range(n)]
    te = _std(d) * math.sqrt(TRADING_DAYS)
    ir = (_mean(d) * TRADING_DAYS) / te if te and te > 0 else float("nan")
    return te, ir


def corr(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    ma, mb = _mean(a[:n]), _mean(b[:n])
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)) * sum((b[i] - mb) ** 2 for i in range(n)))
    return num / den if den > 0 else float("nan")


def lw_sharpe_diff_ci(r1: Sequence[float], r2: Sequence[float]) -> Tuple[float, float, float]:
    """Ledoit–Wolf (2008) interval for SR1 − SR2 (annualized), iid approximation:
    Var(ŜR1 − ŜR2) ≈ [(1 + SR1²/2) + (1 + SR2²/2) − 2(ρ + SR1·SR2·ρ²/2)] / T on daily SRs."""
    n = min(len(r1), len(r2))
    s1, s2 = sharpe_annual(r1[:n]) / math.sqrt(TRADING_DAYS), sharpe_annual(r2[:n]) / math.sqrt(TRADING_DAYS)
    rho = corr(r1, r2)
    if any(map(math.isnan, (s1, s2, rho))) or n < 3:
        return float("nan"), float("nan"), float("nan")
    var = ((1 + s1 * s1 / 2) + (1 + s2 * s2 / 2) - 2 * (rho + s1 * s2 * rho * rho / 2)) / n
    se = math.sqrt(max(var, 0.0)) * math.sqrt(TRADING_DAYS)
    diff = (s1 - s2) * math.sqrt(TRADING_DAYS)
    return diff, diff - 1.96 * se, diff + 1.96 * se


def arm_metrics(equity: Sequence[float], spy_r: Sequence[float], ref_r: Sequence[float], emr) -> dict:
    r = daily_returns(equity)
    sr = sharpe_annual(r)
    beta, alpha = beta_alpha(r, spy_r)
    te, ir = te_ir(r, ref_r)
    diff, lo, hi = lw_sharpe_diff_ci(r, ref_r)
    return {
        "days": len(r), "cagr_pct": round(cagr(equity) * 100, 2), "max_dd_pct": round(max_drawdown(equity) * 100, 2),
        "sharpe": round(sr, 3), "sharpe_se": round(emr.lo_se_annual(sr, len(r)), 3),
        "beta_spy": round(beta, 3), "alpha_pct": round(alpha * 100, 2),
        "te_pct": round(te * 100, 2), "ir": round(ir, 3),
        "sharpe_minus_ref": round(diff, 3), "lw95_lo": round(lo, 3), "lw95_hi": round(hi, 3),
        "vol_pct": round(_std(r) * math.sqrt(TRADING_DAYS) * 100, 2),
    }


def by_regime(equity: Sequence[float], alloc_regime: Sequence[str]) -> Dict[str, dict]:
    """Daily returns grouped by the regime that set the allocation earning them."""
    r = daily_returns(equity)
    groups: Dict[str, List[float]] = defaultdict(list)
    for i, ret in enumerate(r):
        groups[alloc_regime[i]].append(ret)   # allocation decided at close i earns return i (index i+1 of equity)
    out = {}
    for reg in REGIMES:
        x = groups.get(reg, [])
        if not x:
            continue
        out[reg] = {"days": len(x), "ann_mean_pct": round(_mean(x) * TRADING_DAYS * 100, 2), "sharpe": round(sharpe_annual(x), 3) if len(x) > 2 else None}
    return out


# ── data ───────────────────────────────────────────────────────────────────

def align(bars_by_sym: Dict[str, dict], spine: str = "SPY") -> Tuple[List[str], Dict[str, List[float]], Dict[str, int]]:
    """Align every symbol's closes on the spine's dates; a missing date carries the
    last close forward (counted). Days before a symbol's first close are dropped
    from the spine so every arm starts on the same day."""
    dates = list(bars_by_sym[spine]["date"])
    maps = {s: dict(zip(b["date"], b["close"])) for s, b in bars_by_sym.items()}
    first = max(min(m) for m in maps.values())
    dates = [d for d in dates if d >= first]
    out: Dict[str, List[float]] = {}
    carried: Dict[str, int] = {}
    for s, m in maps.items():
        last = None
        xs = []
        c = 0
        for d in dates:
            if d in m and m[d]:
                last = float(m[d])
            else:
                c += 1
            xs.append(last)
        out[s] = xs
        carried[s] = c
    return dates, out, carried


def run(bars_by_sym: Dict[str, dict], regimes_by_date: Dict[str, str], emr, cost_bps: float = COST_BPS) -> dict:
    dates, prices, carried = align(bars_by_sym)
    regimes = [regimes_by_date.get(d, "NEUTRAL") for d in dates]
    sims = {name: simulate(prices, regimes, w, cost_bps=cost_bps) for name, w in {**ARMS, **REFERENCES}.items()}
    sims[BUY_HOLD] = simulate(prices, regimes, {"SPY": 1.0}, buy_hold=True, cost_bps=cost_bps)
    spy_r = daily_returns(sims[BUY_HOLD]["equity"])
    ref_r = daily_returns(sims["spy_same_exposure"]["equity"])
    arms = {}
    for name, s in sims.items():
        m = arm_metrics(s["equity"], spy_r, ref_r, emr)
        m.update({"rebalances": s["rebalances"], "turnover": s["turnover"], "costs_pct": round(s["costs"] * 100, 3),
                  "by_regime": by_regime(s["equity"], s["alloc_regime"]), "final_equity": round(s["equity"][-1], 4)})
        arms[name] = m
    srs = [arms[a]["sharpe"] for a in ARMS if not math.isnan(arms[a]["sharpe"])]
    var = (_std(srs) ** 2) if len(srs) > 1 else 0.0
    bar = emr.expected_max_sharpe(len(ARMS), var)
    best = max(ARMS, key=lambda a: arms[a]["sharpe"])
    regime_days = {reg: regimes.count(reg) for reg in REGIMES}
    return {"from": dates[0], "to": dates[-1], "days": len(dates), "carried_closes": carried, "regime_days": regime_days,
            "arms": arms, "dsr": {"n_arms": len(ARMS), "var_sharpe": round(var, 5), "expected_max_sharpe_bar": round(bar, 3), "best_arm": best,
                                  "best_clears_bar": bool(arms[best]["sharpe"] > bar)}}


# ── report ─────────────────────────────────────────────────────────────────

def verdict(res: dict) -> str:
    a = res["arms"]
    b, s = a["basket"], a["spy_core"]
    parts = []
    parts.append(f"benchmark-relative Sharpe: spy_core {s['sharpe_minus_ref']:+.3f} vs basket {b['sharpe_minus_ref']:+.3f} — "
                 + ("SPY-core better" if s["sharpe_minus_ref"] > b["sharpe_minus_ref"] else "basket better")
                 + f" (basket's LW 95% interval vs the same-exposure SPY: [{b['lw95_lo']:+.3f}, {b['lw95_hi']:+.3f}]"
                 + (" — includes 0" if b["lw95_lo"] <= 0 <= b["lw95_hi"] else " — excludes 0") + ")")
    parts.append(f"CAGR: spy_core {s['cagr_pct']:.2f}% vs basket {b['cagr_pct']:.2f}% — " + ("SPY-core higher" if s["cagr_pct"] > b["cagr_pct"] else "basket higher"))
    d = res["dsr"]
    parts.append(f"DSR bar (N=4) {d['expected_max_sharpe_bar']:.2f}: best arm {d['best_arm']} " + ("clears it" if d["best_clears_bar"] else "does NOT clear it"))
    return "; ".join(parts) + "."


def _f(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render(res: dict, meta: dict) -> str:
    L = [f"# Research test 4: the ETF floor's composition vs SPY under the live exposure schedule — {meta['date']}", "",
         f"Generated {meta['generated_at']} by scripts/floor_basket_vs_index.py — {res['from']} → {res['to']} ({res['days']} trading days), "
         f"regime days {res['regime_days']}, carried-forward closes {res['carried_closes']}, cost {meta['cost_bps']} bp/side, data {meta['data_source']}.", "",
         "## PRIOR (stated before the run)", "", PRIOR, "",
         "## VERDICT", "", verdict(res), "",
         "## Arms (whole-equity daily returns; the floor target, GLD leg, trend caps and cash are identical across arms)", "",
         "| arm | CAGR % | max DD % | vol % | Sharpe ± SE | beta SPY | alpha %/y | TE % | IR | Sharpe − same-exposure SPY [LW 95%] | rebalances | turnover | costs % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in list(ARMS) + list(REFERENCES) + [BUY_HOLD]:
        m = res["arms"][name]
        L.append(f"| {name} | {_f(m['cagr_pct'])} | {_f(m['max_dd_pct'])} | {_f(m['vol_pct'])} | {_f(m['sharpe'], 3)} ± {_f(m['sharpe_se'], 3)} | {_f(m['beta_spy'], 3)} | {_f(m['alpha_pct'])} | {_f(m['te_pct'])} | {_f(m['ir'], 3)} | {_f(m['sharpe_minus_ref'], 3)} [{_f(m['lw95_lo'], 3)}, {_f(m['lw95_hi'], 3)}] | {m['rebalances']} | {_f(m['turnover'], 1)} | {_f(m['costs_pct'], 3)} |")
    L += ["", f"DSR: N = {res['dsr']['n_arms']} arms, Sharpe variance across arms {res['dsr']['var_sharpe']}, E[max SR] bar {res['dsr']['expected_max_sharpe_bar']}, best arm {res['dsr']['best_arm']} "
          + ("clears the bar." if res["dsr"]["best_clears_bar"] else "does not clear the bar — the arms are statistically indistinguishable at this horizon."), "",
          "## By regime (the regime that set the allocation earning each day's return)", "",
          "| arm | " + " | ".join(f"{r} (days / ann. mean % / Sharpe)" for r in REGIMES) + " |", "|---|" + "---|" * len(REGIMES)]
    for name in list(ARMS) + list(REFERENCES) + [BUY_HOLD]:
        br = res["arms"][name]["by_regime"]
        cells = []
        for r in REGIMES:
            x = br.get(r)
            cells.append("—" if not x else f"{x['days']} / {_f(x['ann_mean_pct'])} / {_f(x['sharpe'], 2)}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## HONESTY", "",
          "- The regime classifier runs on VXX's series-B history (2018-01 →); earlier days carry a neutral VXX ratio, so pre-2018 labels lean on SPY's moving averages alone (the classifier's own degraded path).",
          "- Costs are a stated 5 bp per side on traded value, not the measured fills; the live floor pays real spreads on regime flips (2,755 fills since May) that this does not model beyond that charge.",
          "- The active sleeves are not simulated: every arm holds the same cash remainder, so the comparison isolates the floor's composition under the live exposure schedule, not the account.",
          "- Ledoit–Wolf intervals use the iid approximation with the realized correlation; the DSR bar is the E[max SR] of N=4 arms over their Sharpe dispersion — four correlated arms, so the bar is conservative in one direction and the overlap in the other.",
          "- One history, one draw: ten years contain two bear phases; per-regime rows are the regime-conditioning the REASONING STANDARD demands, and the small-day regimes are noise.",
          ""]
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--years", type=float, default=10.8)
    ap.add_argument("--cost-bps", type=float, default=COST_BPS)
    ap.add_argument("--out", default=os.path.join(ROOT, "research"))
    a = ap.parse_args(argv)
    t0 = time.time()
    import backtest_v2 as bt  # noqa: E402
    emr = _load_emr()
    days = int(a.years * 365.25) + 400   # 200 extra bars for the MAs + the classifier's warm-up
    bars = {s: bt.fetch_bars(s, days) for s in SYMBOLS}
    vxx = bt.fetch_bars("VXX", days)
    missing = [s for s, b in bars.items() if not b or not b.get("date")]
    if missing:
        raise SystemExit(f"no bars for {missing}")
    labels, quality = bt.regime_series(bars["SPY"], vxx)
    regimes_by_date = dict(zip(bars["SPY"]["date"], labels))
    res = run(bars, regimes_by_date, emr, a.cost_bps)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    meta = {"date": date, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "cost_bps": a.cost_bps,
            "data_source": f"backtest_v2.fetch_bars (regime quality: {quality})", "runtime_s": round(time.time() - t0, 1), "prior": PRIOR}
    os.makedirs(a.out, exist_ok=True)
    md = os.path.join(a.out, f"floor_basket_vs_index_{date}.md")
    with open(md, "w") as f:
        f.write(render(res, meta))
    with open(os.path.join(a.out, f"floor_basket_vs_index_{date}.json"), "w") as f:
        json.dump({"meta": meta, "result": {k: v for k, v in res.items()}}, f)
    print(f"wrote {md}\n{verdict(res)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
