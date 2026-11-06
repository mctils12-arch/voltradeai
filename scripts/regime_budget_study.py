#!/usr/bin/env python3
"""
regime_budget_study.py — gate-2 check of the regime risk budget (2026-11-06,
human refocus spec section 3 step 1). [RESEARCH], index level, no trading.

QUESTION. Does a LOW risk budget (regime_engine.risk_budget_series, the exact
code the bot runs) precede higher realized volatility and deeper drawdowns,
and what does it say about returns? Pre-registered (CLAUDE.md reasoning
standard 10, priors in the experiments entry before the first run):
  outcomes over the next H=21 sessions after the close of day t (the budget is
  decided at that close from data up to it):
    fwd_rv   annualized stdev of SPY's daily log returns t+1..t+H
    fwd_mdd  worst peak-to-trough drawdown a holder from the close of t sees
             inside t..t+H (<= 0)
    fwd_ret  P[t+H] / P[t] - 1
  IC = Spearman rank correlation of the day's value with each outcome; the
  t-stat is computed on NON-overlapping samples (every H-th day) because the
  daily windows overlap (a daily t-stat would be inflated ~sqrt(H) times).
  A table by the UI's budget bands; the same IC for VIX alone, the engine's
  existing exposure_continuous and the raw (un-glided) budget, so the
  composite is judged against the obvious single input; first half vs second
  half of the sample for stability.
  An INDICATIVE overlay: SPY exposure = budget decided at close t, applied from
  close t+1 (lag 2, as scripts/regime_backtest.py), 5 bp per unit turnover,
  cash at 0%, against buy-and-hold AND against a constant exposure equal to the
  budget's own mean (the base rate: how much of any drawdown cut is just
  holding less on average).

DATA. Unadjusted daily closes for the engine (what it reads live) and adjusted
closes for outcomes, via --closes-json (a pre-fetched {symbol: {"close":
{date: px}, "adjclose": {date: px}}} file) or regime_engine.fetch_closes
(Alpaca first, Yahoo fallback; adjusted closes then equal the closes, so
outcomes ignore dividends). Nothing is cached or written except --out / --json.

NOT. A strategy backtest of any consumer: the live consumer is the tier
engine's satellite headroom, whose P&L this file does not model.

Run: python3 scripts/regime_budget_study.py [--closes-json PATH] [--days 7700]
     [--horizon 21] [--out PATH.md] [--json PATH.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import regime_engine as re_  # noqa: E402  (path set above)


# ── pure statistics ──────────────────────────────────────────────────────────
def ranks(xs):
    """Average ranks (1-based), ties shared."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def t_of_r(r, n):
    if r is None or n < 3 or abs(r) >= 1:
        return None
    return r * math.sqrt((n - 2) / (1 - r * r))


def forward_outcomes(px, i, h):
    """(fwd_rv, fwd_mdd, fwd_ret) over sessions i+1..i+h of price list px, or None."""
    if i + h >= len(px) or any(p is None or p <= 0 for p in px[i:i + h + 1]):
        return None
    rets = [math.log(px[k] / px[k - 1]) for k in range(i + 1, i + h + 1)]
    rv = st.pstdev(rets) * math.sqrt(252)
    peak, mdd = px[i], 0.0
    for k in range(i + 1, i + h + 1):
        peak = max(peak, px[k])
        mdd = min(mdd, px[k] / peak - 1.0)
    return rv, mdd, px[i + h] / px[i] - 1.0


def overlay(px, expo, lag=2, cost_bp=5.0):
    """Daily returns of exposure path `expo` (decided at close t, held from close t+lag-1)."""
    out = []
    prev_e = 0.0
    for t in range(lag, len(px)):
        e = expo[t - lag] if expo[t - lag] is not None else prev_e
        r = px[t] / px[t - 1] - 1.0
        out.append(e * r - abs(e - prev_e) * cost_bp / 1e4)
        prev_e = e
    return out


def perf(daily, avg_expo=None):
    if not daily:
        return {}
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in daily:
        eq *= 1 + r
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
    yrs = len(daily) / 252.0
    mu, sd = st.mean(daily), st.pstdev(daily)
    return {"cagr": eq ** (1 / yrs) - 1 if yrs > 0 else None, "vol": sd * math.sqrt(252),
            "sharpe": (mu / sd * math.sqrt(252)) if sd > 0 else None, "max_dd": mdd, "avg_exposure": avg_expo}


# ── study ────────────────────────────────────────────────────────────────────
def load(args):
    syms = ["SPY"] + [s for s in re_.BREADTH_BASKET if s != "SPY"] + ["^VIX", "^VIX3M"]
    if args.closes_json:
        with open(args.closes_json) as f:
            raw = json.load(f)
        closes = {s: raw[s]["close"] for s in syms if s in raw}
        adj = raw.get("SPY", {}).get("adjclose") or closes["SPY"]
        return closes, adj, {s: "file" for s in closes}
    closes, sources = re_.fetch_closes(syms, args.days)
    return closes, closes["SPY"], sources


def study(closes, adj_spy, h=21):
    dates, al = re_.align(closes, "SPY")
    basket = {s: al[s] for s in re_.BREADTH_BASKET if s in al and s != "SPY"}
    sig = re_.signal_series(dates, al["SPY"], basket, al.get("^VIX"), al.get("^VIX3M"))
    states = re_.run_states(sig)
    bud = re_.risk_budget_series(sig)
    px = [adj_spy.get(d) for d in dates]
    for k in range(1, len(px)):          # forward-fill a missing adjusted close
        if px[k] is None:
            px[k] = px[k - 1]
    p = bud["params"]
    drv = [re_.budget_scores(s, a, p)[1] for s, a in zip(sig, bud["anchor"])]
    series = {
        "budget": bud["budget"],
        "budget_raw": bud["raw"],
        "neg_vix": [(-s["vix"]) if s["vix"] is not None else None for s in sig],
        "exposure_continuous": [r["exposure_continuous"] for r in states],
        **{f"score_{k}": [d[k] for d in drv] for k in re_.BUDGET_DRIVERS},
    }
    # start where EVERY driver has a reading (the VIX anchor needs vix_anchor_min_obs prior sessions)
    start = 0
    while start < len(dates) and any(drv[start][k] is None for k in re_.BUDGET_DRIVERS):
        start += 1
    rows = []
    for i in range(start, len(dates) - h):
        fo = forward_outcomes(px, i, h)
        if fo is None:
            continue
        rows.append((i, fo))
    out = {"dates": [dates[start], dates[rows[-1][0]] if rows else None], "n_days": len(rows), "horizon": h, "ic": {}}
    names = ("fwd_rv", "fwd_mdd", "fwd_ret")
    for key, vals in series.items():
        ic = {}
        for j, nm in enumerate(names):
            x = [vals[i] for i, _ in rows]
            y = [fo[j] for _, fo in rows]
            r_all = spearman(x, y)
            xs = x[::h]
            ys = y[::h]
            r_nov = spearman(xs, ys)
            ic[nm] = {"ic_daily": r_all, "ic_nonoverlap": r_nov, "n_nonoverlap": len(xs), "t_nonoverlap": t_of_r(r_nov, len(xs))}
        out["ic"][key] = ic
    # halves (stability)
    mid = len(rows) // 2
    out["halves"] = {}
    for tag, part in (("first", rows[:mid]), ("second", rows[mid:])):
        x = [series["budget"][i] for i, _ in part]
        out["halves"][tag] = {"from": dates[part[0][0]], "to": dates[part[-1][0]],
                              **{nm: spearman(x[::h], [fo[j] for _, fo in part][::h]) for j, nm in enumerate(names)}}
    # fixed bands (quantiles would tie: about 40% of days sit at 1.0) (the UI's): full >= 0.80 > reduced >= 0.50 > defensive >= 0.25 > minimal
    out["bands"] = []
    for name, lo_b, hi_b in (("minimal", 0.0, 0.25), ("defensive", 0.25, 0.50), ("reduced", 0.50, 0.80), ("full", 0.80, 1.01)):
        bk = [fo for i, fo in rows if lo_b <= series["budget"][i] < hi_b]
        if bk:
            out["bands"].append(bucket_row(name, bk, len(rows)))
    # indicative overlay
    lo = start
    sub_px = px[lo:]
    expo = [series["budget"][i] for i in range(lo, len(px))]
    mean_e = st.mean(e for e in expo if e is not None)
    vix_only = [series["score_vix_level"][i] for i in range(lo, len(px))]
    eng = [series["exposure_continuous"][i] for i in range(lo, len(px))]
    out["overlay"] = {
        "buy_and_hold": perf(overlay(sub_px, [1.0] * len(sub_px)), 1.0),
        "budget_scaled": perf(overlay(sub_px, expo), mean_e),
        "constant_at_budget_mean": perf(overlay(sub_px, [mean_e] * len(sub_px)), mean_e),
        "vix_score_only (diagnostic)": perf(overlay(sub_px, vix_only), st.mean(v for v in vix_only if v is not None)),
        "engine_exposure_continuous (diagnostic)": perf(overlay(sub_px, eng), st.mean(eng)),
    }
    return out


def bucket_row(name, bk, total):
    return {"bucket": name, "n": len(bk), "share": len(bk) / total,
            "mean_fwd_rv": st.mean(fo[0] for fo in bk), "mean_fwd_mdd": st.mean(fo[1] for fo in bk),
            "p_mdd_worse_5": sum(1 for fo in bk if fo[1] < -0.05) / len(bk),
            "p_mdd_worse_10": sum(1 for fo in bk if fo[1] < -0.10) / len(bk),
            "mean_fwd_ret": st.mean(fo[2] for fo in bk), "median_fwd_ret": st.median(fo[2] for fo in bk)}


def fmt(x, p=3):
    return "n/a" if x is None else f"{x:.{p}f}"


def render(res):
    lines = [f"# Regime risk budget, gate-2 study ({res['dates'][0]} to {res['dates'][1]}, {res['n_days']} days, H={res['horizon']})", ""]
    lines.append("| series | outcome | IC daily | IC non-overlap | n | t (non-overlap) |")
    lines.append("|---|---|---|---|---|---|")
    for key, ic in res["ic"].items():
        for nm, v in ic.items():
            lines.append(f"| {key} | {nm} | {fmt(v['ic_daily'])} | {fmt(v['ic_nonoverlap'])} | {v['n_nonoverlap']} | {fmt(v['t_nonoverlap'], 2)} |")
    lines += ["", "| budget band | n | share of days | mean fwd RV | mean fwd max DD | P(DD<-5%) | P(DD<-10%) | mean fwd ret | median fwd ret |",
              "|---|---|---|---|---|---|---|---|---|"]
    for q in res["bands"]:
        lines.append(f"| {q['bucket']} | {q['n']} | {q['share'] * 100:.1f}% | {q['mean_fwd_rv'] * 100:.1f}% | "
                     f"{q['mean_fwd_mdd'] * 100:.2f}% | {q['p_mdd_worse_5'] * 100:.1f}% | {q['p_mdd_worse_10'] * 100:.1f}% | "
                     f"{q['mean_fwd_ret'] * 100:.2f}% | {q['median_fwd_ret'] * 100:.2f}% |")
    lines += ["", "| halves (budget, non-overlap Spearman) | from | to | fwd_rv | fwd_mdd | fwd_ret |", "|---|---|---|---|---|---|"]
    for tag, h in res["halves"].items():
        lines.append(f"| {tag} | {h['from']} | {h['to']} | {fmt(h['fwd_rv'])} | {fmt(h['fwd_mdd'])} | {fmt(h['fwd_ret'])} |")
    lines += ["", "| overlay (indicative, SPY, lag 2, 5 bp) | CAGR | vol | Sharpe | max DD | avg exposure |", "|---|---|---|---|---|---|"]
    for k, v in res["overlay"].items():
        lines.append(f"| {k} | {v['cagr'] * 100:.2f}% | {v['vol'] * 100:.1f}% | {fmt(v['sharpe'], 2)} | {v['max_dd'] * 100:.1f}% | {fmt(v['avg_exposure'], 2)} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--closes-json", default=None)
    ap.add_argument("--days", type=int, default=7700)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    closes, adj, sources = load(args)
    res = study(closes, adj, args.horizon)
    res["sources"] = sources
    text = render(res)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
