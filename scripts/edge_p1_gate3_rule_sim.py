#!/usr/bin/env python3
"""scripts/edge_p1_gate3_rule_sim.py — Edge P1, ladder gate 3 (LOGIC): the
pre-registered rule "no new short premium when R_{t-1} < 0.2"
(research/edge_preregistration.md § P1, "Ladder path") simulated on SPY
short-premium entries over the gate-2 sample (2012-05 ->).

WHY A MODEL REPLAY. The registered gate-3 harness, scripts/csp_manager_rule_sim.py,
marks the puts the bot would have sold from the options-chain archive, which
starts in 2026-08: a few months, far too short for a 252-day-rank state that
persists for weeks. backtest_v2.py has no options replay and the repo holds no
historical option quotes. So this replays the SYSTEM'S CSP RULES on SPY with
Black-Scholes MODEL prices, on the same 14-year sample gate 2 used. Exactly:

  Entries   one per SPY session t, at the close of t, from 2012-05-01 to the
            last session whose position resolves before --end (positions
            still open at --end are censored and excluded, counted).
  Books     (1) "csp" (PRIMARY): sell one put at BS delta -0.30
                (tiered_strategy.T1_TARGET_DELTA), strike on SPY's $1 grid.
            (2) "strangle" (secondary): sell a 1-SD strangle, put strike
                S*exp(-s*sqrt(T)), call strike S*exp(+s*sqrt(T)), $1 grid.
            Expiry for both: the first Friday >= t + 35 calendar days
            (DTE 35..41, inside T1's 30..45 window).
  Pricing   Black-Scholes with sigma = VIX/100 of the SAME day at entry and at
            every later mark (vega P&L is in the path), both legs, no skew, no
            term structure, r = csp_manager_rule_sim.R_RISK_FREE, no dividend
            yield. These are MODEL marks, never quotes.
  Path      SPY RAW (dividend-unadjusted) closes: the price a SPY option
            settles against. The rank and the regime labels come from the
            gate-2 panel (adjusted closes), unchanged.
  Exits     the live options_manager order, constants from
            csp_manager_rule_sim.RULES (pinned to options_manager.py by
            test_csp_manager_rule_sim.py): DTE <= 5 critical close; profit
            >= 50% of the credit; loss <= -2x the credit; DTE <= 21 close;
            expiry settles at intrinsic. Daily closes only: the live stop
            fires intraday, so a gap through the stop is booked at the close.
  Costs     each leg sold at mid - h and bought back at mark + h,
            h = max($0.02, 1% of that leg's mid) per share; expiry settles
            with no spread.
  Rule      skip the day-t entry when R_{t-1} < 0.2, where R is gate 2's
            trailing-252 percentile rank of SqueezeMetrics GEX and R_{t-1} is
            the last value published before session t (GEX for t-1 posts after
            t-1's close). R_{t-1} missing -> entry kept (the live filter is
            inactive without data, the same posture).
  P&L unit  per trade: P&L / collateral, collateral = put strike x 100 (the
            put side's cash-secured collateral; the strangle's call side is
            NOT separately collateralised here — stated, not modelled).
  Portfolio a ladder: every entry gets 1/L of the capital, L = the no-rule
            book's average number of open positions, so the no-rule book is
            ~100% deployed on average. Daily return = sum over open positions
            of (that day's mark-to-market dollars / collateral) / L. Additive,
            no compounding. The rule-on book uses the same L, so its skipped
            days simply leave capital idle (idle cash earns 0 here — stated).
  Inference circular moving-block bootstrap, 21-session blocks, 2000 draws,
            seed 7: of the per-trade mean difference (skipped - kept) over
            entry sessions, and of the portfolio Sharpe / max-drawdown
            difference (rule on - rule off) over daily returns.

DATA: identical inputs to scripts/edge_p1_dealer_gamma.py (SqueezeMetrics
DIX.csv — published ESTIMATE, research use only, never committed; Cboe VIX;
Yahoo SPY daily chart). --data-dir reads DIX.csv / VIX_History.csv / spy.json
from a directory instead of downloading.

Run:  python3 scripts/edge_p1_gate3_rule_sim.py [--data-dir DIR] [--end YYYY-MM-DD]
      [--json-out FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from datetime import date, timedelta
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):   # appended, never prepended: a scripts/ module must not shadow a root one
    if _p not in sys.path:
        sys.path.append(_p)

import edge_p1_dealer_gamma as p1  # noqa: E402  gate-2 loaders, rank and panel (same definitions)
from csp_manager_rule_sim import RULES, R_RISK_FREE, _ncdf, bs_put  # noqa: E402  live manager rules

PRIOR = (
    "Stated before the run (2026-11-06), on the PRIMARY csp book, 2019+ holdout: "
    "80% that skipped trades earn less per trade than kept ones (mean skipped - kept < 0); "
    "55% that the block-bootstrap 95% CI of that difference excludes 0 (overlapping trades); "
    "60% that the skipped trades' mean is still POSITIVE (gate 2: realized vol averages 82% "
    "of VIX even in the bottom gamma quintile); 70% that the skipped trades' worst-5% quantile "
    "is worse than the kept trades'; 60% that the rule-on portfolio's max drawdown is <= the "
    "rule-off one; 45% that its Sharpe is higher. Verdict prior: HELPS 45%, MIXED 40%, HURTS 15%."
)

VERDICT_RULE = (
    "Pre-stated (2026-11-06, before the run). Decision sample: the csp book, entries 2019-01-01 ->. "
    "HURTS if (mean skipped - kept >= 0 AND the skipped trades' worst-5% quantile >= the kept "
    "trades') — the rule discards average-or-better trades without cutting their tail — OR the "
    "rule-on portfolio has BOTH a lower Sharpe AND a deeper max drawdown than rule-off. "
    "HELPS if mean skipped - kept < 0 with the bootstrap 95% CI upper bound < 0 AND the rule-on "
    "max drawdown <= rule-off. Otherwise MIXED (a trade-off: some forgone premium for smaller "
    "tails, or not resolvable at this sample size). Human directive 2026-11-06: ship ON in paper "
    "unless the evidence says it hurts — so only HURTS argues for shipping it OFF."
)

# ── constants (stated in the docstring; the test pins the shared ones) ─────
LOW_RANK = 0.2              # the registered rule: R_{t-1} < 0.2
TARGET_PUT_DELTA = 0.30     # tiered_strategy.T1_TARGET_DELTA
MIN_DTE_CAL = 35            # expiry = first Friday >= t + 35 calendar days
HALF_SPREAD_MIN = 0.02      # $/share, per leg, each side
HALF_SPREAD_PCT = 0.01      # of the leg's mid
FULL_START = p1.FULL_START          # 2012-05-01
HOLDOUT_START = p1.HOLDOUT_START    # 2019-01-01
DEFAULT_END = "2026-11-04"
BOOT_BLOCK = 21
BOOT_N = 2000
BOOT_SEED = 7
BOOKS = ("csp", "strangle")


# ── pricing ────────────────────────────────────────────────────────────────
def bs_call(S: float, K: float, T: float, sigma: float, r: float = R_RISK_FREE) -> float:
    """Black-Scholes European call; intrinsic at T <= 0."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d1 - sq)


def put_delta(S: float, K: float, T: float, sigma: float, r: float = R_RISK_FREE) -> float:
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    return _ncdf(d1) - 1.0


def strike_for_put_delta(S: float, T: float, sigma: float, delta: float = TARGET_PUT_DELTA,
                         r: float = R_RISK_FREE) -> float:
    """The strike whose BS put delta is -delta, rounded to SPY's $1 grid."""
    d1 = NormalDist().inv_cdf(1.0 - delta)
    k = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r + 0.5 * sigma * sigma) * T))
    return float(round(k))


def strangle_strikes(S: float, T: float, sigma: float) -> Tuple[float, float]:
    """(put strike, call strike) one standard deviation out, $1 grid."""
    m = sigma * math.sqrt(T)
    return float(round(S * math.exp(-m))), float(round(S * math.exp(m)))


def expiry_for(entry: date, min_dte: int = MIN_DTE_CAL) -> date:
    """First Friday on or after entry + min_dte calendar days."""
    d = entry + timedelta(days=min_dte)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def half_spread(mid: float) -> float:
    return max(HALF_SPREAD_MIN, HALF_SPREAD_PCT * mid)


def _leg_value(typ: str, K: float, S: float, T: float, sigma: float) -> float:
    return bs_put(S, K, T, sigma) if typ == "P" else bs_call(S, K, T, sigma)


# ── one position ───────────────────────────────────────────────────────────
def simulate_position(kind: str, i0: int, dates: Sequence[str], closes: Sequence[float],
                      vix: Sequence[Optional[float]]) -> Optional[dict]:
    """Sell `kind` at the close of session i0 and mark it at every later close
    under the live exit order. Returns the trade with its per-session P&L path
    in dollars (`path`: {session index: $}), whose sum is exactly the trade's
    P&L. None when the entry is unusable or the data ends before an exit
    (censored)."""
    if kind not in BOOKS:
        raise ValueError(f"unknown book {kind}")
    S0, v0 = closes[i0], vix[i0]
    if not (S0 and S0 > 0 and v0 and v0 > 0):
        return None
    t0 = date.fromisoformat(dates[i0])
    exp = expiry_for(t0)
    sig0 = v0 / 100.0
    T0 = (exp - t0).days / 365.0
    if kind == "csp":
        legs = [("P", strike_for_put_delta(S0, T0, sig0))]
    else:
        kp, kc = strangle_strikes(S0, T0, sig0)
        legs = [("P", kp), ("C", kc)]
    mids0 = [_leg_value(typ, K, S0, T0, sig0) for typ, K in legs]
    if any(m <= 0 for m in mids0):
        return None
    hs0 = [half_spread(m) for m in mids0]
    credit = sum(m - h for m, h in zip(mids0, hs0)) * 100.0
    collateral = legs[0][1] * 100.0
    path: Dict[int, float] = {i0: -sum(hs0) * 100.0}   # the entry half-spread, paid at once
    prev = sum(mids0)
    last_sig = sig0
    for i in range(i0 + 1, len(dates)):
        S = closes[i]
        if not (S and S > 0):
            continue
        t = date.fromisoformat(dates[i])
        dte = (exp - t).days
        if vix[i] and vix[i] > 0:
            last_sig = vix[i] / 100.0
        if dte <= 0:
            intrinsic = sum(max(K - S, 0.0) if typ == "P" else max(S - K, 0.0) for typ, K in legs)
            path[i] = (prev - intrinsic) * 100.0
            return _trade(kind, i0, i, dates, legs, credit, collateral, credit - intrinsic * 100.0,
                          "expiry", path, S0, v0, exp)
        marks = [_leg_value(typ, K, S, dte / 365.0, last_sig) for typ, K in legs]
        mark = sum(marks)
        exit_hs = sum(half_spread(m) for m in marks)
        pnl = credit - (mark + exit_hs) * 100.0
        reason = None
        if dte <= RULES["dte_critical"]:
            reason = "dte_critical"
        elif pnl >= RULES["profit_target_pct"] * credit:
            reason = "profit_target"
        elif pnl <= -RULES["loss_limit_mult"] * credit:
            reason = "loss_limit"
        elif dte <= RULES["dte_exit"]:
            reason = "dte_exit"
        if reason:
            path[i] = (prev - mark - exit_hs) * 100.0
            return _trade(kind, i0, i, dates, legs, credit, collateral, pnl, reason, path, S0, v0, exp)
        path[i] = (prev - mark) * 100.0
        prev = mark
    return None


def _trade(kind, i0, i, dates, legs, credit, collateral, pnl, reason, path, S0, v0, exp) -> dict:
    return {"book": kind, "i0": i0, "exit_i": i, "day": dates[i0], "exit_day": dates[i],
            "exp": exp.isoformat(), "legs": [[typ, K] for typ, K in legs], "S0": S0, "vix0": v0,
            "credit": credit, "collateral": collateral, "pnl": pnl,
            "pnl_pct": pnl / collateral * 100.0, "exit_reason": reason, "path": path}


# ── books, stats, portfolio ────────────────────────────────────────────────
def run_book(kind: str, dates: Sequence[str], closes: Sequence[float],
             vix: Sequence[Optional[float]], entry_meta: Dict[str, dict],
             start: str = FULL_START) -> Tuple[List[dict], dict]:
    """Every session in entry_meta (date -> {R_lag1, regime}) on/after `start`
    is an entry. Adds `R_lag1`, `regime` and `skipped` (the rule fires) to each
    trade. Returns (trades, counts)."""
    trades: List[dict] = []
    counts = {"entries": 0, "censored_or_unusable": 0}
    for i0, d in enumerate(dates):
        if d < start or d not in entry_meta:
            continue
        counts["entries"] += 1
        tr = simulate_position(kind, i0, dates, closes, vix)
        if tr is None:
            counts["censored_or_unusable"] += 1
            continue
        m = entry_meta[d]
        tr["R_lag1"] = m.get("R_lag1")
        tr["regime"] = m.get("regime")
        tr["skipped"] = rule_skips(m.get("R_lag1"))
        trades.append(tr)
    return trades, counts


def rule_skips(r_lag1: Optional[float], threshold: float = LOW_RANK) -> bool:
    """The registered rule: no new short premium when R_{t-1} < threshold.
    Missing rank -> not skipped (inactive without data)."""
    return r_lag1 is not None and r_lag1 < threshold


def trade_stats(pnls: Sequence[float]) -> dict:
    if not pnls:
        return {"n": 0}
    a = np.asarray(pnls, dtype=float)
    q01, q05 = np.percentile(a, [1, 5])
    tail = np.sort(a)[:max(1, int(math.ceil(0.05 * len(a))))]
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "win_rate": float((a > 0).mean()), "q01": float(q01), "q05": float(q05),
            "cvar05": float(tail.mean()), "worst": float(a.min())}


def ladder_depth(trades: Sequence[dict]) -> float:
    """Average open positions per session of a book (no rule)."""
    if not trades:
        return 1.0
    first = min(t["i0"] for t in trades)
    last = max(t["exit_i"] for t in trades)
    held = sum(t["exit_i"] - t["i0"] for t in trades)
    return max(1.0, held / max(1, last - first))


def ladder_returns(trades: Sequence[dict], n_sessions: int, depth: float,
                   include: Callable[[dict], bool]) -> List[float]:
    r = [0.0] * n_sessions
    for tr in trades:
        if not include(tr):
            continue
        for i, dollars in tr["path"].items():
            r[i] += dollars / tr["collateral"] / depth
    return r


def deployment(trades: Sequence[dict], n_sessions: int, depth: float,
               include: Callable[[dict], bool]) -> List[float]:
    """Fraction of capital committed as collateral at each session's close."""
    dep = [0.0] * n_sessions
    for tr in trades:
        if not include(tr):
            continue
        for i in range(tr["i0"], tr["exit_i"]):
            dep[i] += 1.0 / depth
    return dep


def portfolio_metrics(r: Sequence[float]) -> dict:
    a = np.asarray(r, dtype=float)
    if len(a) < 2:
        return {"n": int(len(a))}
    cum = np.cumsum(a)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    dd = peak - cum
    sd = float(a.std(ddof=1))
    roll = np.convolve(a, np.ones(21), mode="valid") if len(a) >= 21 else a
    return {"n": int(len(a)), "total_pct": float(cum[-1] * 100.0),
            "ann_ret_pct": float(a.mean() * 252 * 100.0),
            "ann_vol_pct": float(sd * math.sqrt(252) * 100.0),
            "sharpe": float(a.mean() / sd * math.sqrt(252)) if sd > 0 else None,
            "max_dd_pct": float(dd.max() * 100.0),
            "worst_day_pct": float(a.min() * 100.0),
            "worst_21d_pct": float(roll.min() * 100.0)}


def _block_indices(n: int, block: int, rng: random.Random) -> List[int]:
    out: List[int] = []
    while len(out) < n:
        s = rng.randrange(n)
        out.extend((s + k) % n for k in range(block))
    return out[:n]


def boot_mean_diff(trades: Sequence[dict], block: int = BOOT_BLOCK, n_boot: int = BOOT_N,
                   seed: int = BOOT_SEED) -> Optional[dict]:
    """mean(skipped) - mean(kept), resampling ENTRY SESSIONS in circular blocks
    (consecutive entries overlap in time, so trades are not independent)."""
    ts = sorted(trades, key=lambda t: t["i0"])
    sk = [t for t in ts if t["skipped"]]
    kp = [t for t in ts if not t["skipped"]]
    if len(sk) < 5 or len(kp) < 5:
        return None
    point = float(np.mean([t["pnl_pct"] for t in sk]) - np.mean([t["pnl_pct"] for t in kp]))
    rng = random.Random(seed)
    pn = [t["pnl_pct"] for t in ts]
    fl = [t["skipped"] for t in ts]
    diffs = []
    for _ in range(n_boot):
        idx = _block_indices(len(ts), block, rng)
        s = [pn[i] for i in idx if fl[i]]
        k = [pn[i] for i in idx if not fl[i]]
        if s and k:
            diffs.append(sum(s) / len(s) - sum(k) / len(k))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": point, "ci95": [float(lo), float(hi)], "p_lt0": float(np.mean(np.asarray(diffs) < 0)),
            "draws": len(diffs)}


def boot_portfolio_diff(r_on: Sequence[float], r_off: Sequence[float], block: int = BOOT_BLOCK,
                        n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> Optional[dict]:
    """Joint block bootstrap of the paired daily series: Sharpe(on) - Sharpe(off)
    and maxDD(on) - maxDD(off) (percentage points; negative = shallower with the rule)."""
    n = len(r_on)
    if n < 2 * block:
        return None
    rng = random.Random(seed)
    a_on, a_off = np.asarray(r_on, dtype=float), np.asarray(r_off, dtype=float)
    ds, dd = [], []
    for _ in range(n_boot):
        idx = _block_indices(n, block, rng)
        m_on, m_off = portfolio_metrics(a_on[idx]), portfolio_metrics(a_off[idx])
        if m_on.get("sharpe") is None or m_off.get("sharpe") is None:
            continue
        ds.append(m_on["sharpe"] - m_off["sharpe"])
        dd.append(m_on["max_dd_pct"] - m_off["max_dd_pct"])
    if not ds:
        return None
    return {"sharpe_diff_ci95": [float(x) for x in np.percentile(ds, [2.5, 97.5])],
            "maxdd_diff_ci95": [float(x) for x in np.percentile(dd, [2.5, 97.5])],
            "p_sharpe_on_higher": float(np.mean(np.asarray(ds) > 0)),
            "p_dd_on_shallower": float(np.mean(np.asarray(dd) < 0)), "draws": len(ds)}


def evaluate(trades: Sequence[dict], n_sessions: int, depth: float) -> dict:
    """All / kept / skipped trade stats, both portfolios, both bootstraps,
    over one set of trades (one book, one period)."""
    allp = [t["pnl_pct"] for t in trades]
    kept = [t["pnl_pct"] for t in trades if not t["skipped"]]
    skip = [t["pnl_pct"] for t in trades if t["skipped"]]
    out = {"trades_all": trade_stats(allp), "trades_kept": trade_stats(kept),
           "trades_skipped": trade_stats(skip), "n_skipped": len(skip),
           "share_skipped": (len(skip) / len(allp)) if allp else None,
           "exit_reasons": _reasons(trades)}
    if not trades:
        return out
    lo = min(t["i0"] for t in trades)
    hi = max(t["exit_i"] for t in trades)
    r_off = ladder_returns(trades, n_sessions, depth, lambda t: True)[lo:hi + 1]
    r_on = ladder_returns(trades, n_sessions, depth, lambda t: not t["skipped"])[lo:hi + 1]
    dep_off = deployment(trades, n_sessions, depth, lambda t: True)[lo:hi + 1]
    dep_on = deployment(trades, n_sessions, depth, lambda t: not t["skipped"])[lo:hi + 1]
    out["portfolio_off"] = {**portfolio_metrics(r_off), "avg_deployment": float(np.mean(dep_off))}
    out["portfolio_on"] = {**portfolio_metrics(r_on), "avg_deployment": float(np.mean(dep_on))}
    out["boot_trade_diff"] = boot_mean_diff(trades)
    out["boot_portfolio_diff"] = boot_portfolio_diff(r_on, r_off)
    return out


def _reasons(trades: Sequence[dict]) -> dict:
    out: Dict[str, Dict[str, int]] = {}
    for t in trades:
        g = "skipped" if t["skipped"] else "kept"
        out.setdefault(g, {})
        out[g][t["exit_reason"]] = out[g].get(t["exit_reason"], 0) + 1
    return out


def by_regime(trades: Sequence[dict]) -> dict:
    out = {}
    for g in sorted({t["regime"] for t in trades if t.get("regime")}):
        sub = [t for t in trades if t.get("regime") == g]
        kept = [t["pnl_pct"] for t in sub if not t["skipped"]]
        skip = [t["pnl_pct"] for t in sub if t["skipped"]]
        out[g] = {"n": len(sub), "n_skipped": len(skip), "kept": trade_stats(kept),
                  "skipped": trade_stats(skip),
                  "diff_mean": (float(np.mean(skip) - np.mean(kept)) if skip and kept else None)}
    return out


def verdict(ev: dict) -> Tuple[str, List[str]]:
    """Map the decision sample's evaluation onto VERDICT_RULE."""
    k, s = ev["trades_kept"], ev["trades_skipped"]
    if not k.get("n") or not s.get("n"):
        return "MIXED", ["no skipped or no kept trades in the decision sample"]
    diff = s["mean"] - k["mean"]
    on, off = ev["portfolio_on"], ev["portfolio_off"]
    why = [f"mean skipped - kept = {diff:+.3f} pts of collateral; worst-5% skipped {s['q05']:+.2f} "
           f"vs kept {k['q05']:+.2f}; Sharpe on {on['sharpe']:.2f} vs off {off['sharpe']:.2f}; "
           f"max DD on {on['max_dd_pct']:.2f}% vs off {off['max_dd_pct']:.2f}%"]
    if (diff >= 0 and s["q05"] >= k["q05"]) or (on["sharpe"] < off["sharpe"] and on["max_dd_pct"] > off["max_dd_pct"]):
        return "HURTS", why
    b = ev.get("boot_trade_diff") or {}
    if diff < 0 and b.get("ci95") and b["ci95"][1] < 0 and on["max_dd_pct"] <= off["max_dd_pct"]:
        return "HELPS", why
    return "MIXED", why


# ── loading ────────────────────────────────────────────────────────────────
def load_spy_raw(path: str) -> Dict[str, list]:
    """Dates + RAW (dividend-unadjusted) closes, with the same row filter as
    edge_p1_dealer_gamma.load_spy so the two calendars are identical."""
    from datetime import datetime, timezone
    with open(path) as fh:
        res = json.load(fh)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    out = {"date": [], "close": []}
    for i, ts in enumerate(res["timestamp"]):
        if None in (q["close"][i], q["high"][i], q["low"][i], adj[i]):
            continue
        out["date"].append(datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d"))
        out["close"].append(float(q["close"][i]))
    return out


def entry_meta_from_panel(rows: Sequence[dict]) -> Dict[str, dict]:
    """date -> {R_lag1, regime} from the gate-2 panel rows (rank uses GEX <= t-1)."""
    return {r["date"]: {"R_lag1": r.get("R_lag1"), "regime": r.get("regime")} for r in rows}


def run(data_dir: str, end: str) -> dict:
    gex_rows, dix_price = p1.load_dix(os.path.join(data_dir, "DIX.csv"))
    vix_by_date = p1.load_vix(os.path.join(data_dir, "VIX_History.csv"))
    spy_adj = p1.load_spy(os.path.join(data_dir, "spy.json"))
    spy_raw = load_spy_raw(os.path.join(data_dir, "spy.json"))
    if spy_adj["date"] != spy_raw["date"]:
        raise SystemExit("adjusted and raw SPY calendars differ")
    keep = [k for k, d in enumerate(spy_raw["date"]) if d <= end]
    dates = [spy_raw["date"][k] for k in keep]
    closes = [spy_raw["close"][k] for k in keep]
    vix = [vix_by_date.get(d) for d in dates]
    rows, counts = p1.build_panel(spy_adj, gex_rows, vix_by_date, end, p1._classifier())
    meta = entry_meta_from_panel(rows)
    res = {"inputs": {f: p1._sha(os.path.join(data_dir, f)) for f in ("DIX.csv", "VIX_History.csv", "spy.json")},
           "range": {"dix": [gex_rows[0][0], gex_rows[-1][0]], "spy": [dates[0], dates[-1]], "end": end},
           "panel_counts": counts, "alignment": p1.date_alignment_check(spy_adj, dix_price),
           "rules": dict(RULES), "r": R_RISK_FREE, "books": {}}
    for kind in BOOKS:
        trades, bc = run_book(kind, dates, closes, vix, meta, FULL_START)
        depth = ladder_depth(trades)
        book = {"counts": bc, "depth": depth}
        for name, sub in (("full", trades),
                          ("pre2019", [t for t in trades if t["day"] < HOLDOUT_START]),
                          ("2019+", [t for t in trades if t["day"] >= HOLDOUT_START])):
            book[name] = evaluate(sub, len(dates), depth)
            book[name]["regimes"] = by_regime(sub)
            book[name]["entry_range"] = [sub[0]["day"], sub[-1]["day"]] if sub else None
        res["books"][kind] = book
    res["verdict"], res["verdict_why"] = verdict(res["books"]["csp"]["2019+"])
    res["verdict_full_sample_csp"] = verdict(res["books"]["csp"]["full"])
    res["verdict_strangle_2019"] = verdict(res["books"]["strangle"]["2019+"])
    return res


# ── report ─────────────────────────────────────────────────────────────────
def _ts(name: str, s: dict) -> str:
    if not s.get("n"):
        return f"| {name} | 0 | | | | | | |"
    return (f"| {name} | {s['n']} | {s['mean']:+.3f} | {s['median']:+.3f} | {s['win_rate']:.1%} | "
            f"{s['q05']:+.2f} | {s['q01']:+.2f} | {s['cvar05']:+.2f} |")


def _pf(name: str, p: dict) -> str:
    return (f"| {name} | {p['total_pct']:+.1f} | {p['ann_ret_pct']:+.2f} | {p['ann_vol_pct']:.2f} | "
            f"{p['sharpe']:.2f} | {p['max_dd_pct']:.2f} | {p['worst_day_pct']:+.2f} | "
            f"{p['worst_21d_pct']:+.2f} | {p['avg_deployment']:.0%} |")


def render(res: dict) -> str:
    L = ["Edge P1 gate 3: 'no new short premium when R_{t-1} < 0.2' on SPY model replays",
         f"PRIOR: {PRIOR}", f"VERDICT RULE: {VERDICT_RULE}",
         f"inputs sha256[:16]: {res['inputs']}", f"ranges: {res['range']}",
         f"rules: {res['rules']}  r = {res['r']}",
         f"date alignment corr k=-1/0/+1: {res['alignment']}", ""]
    for kind, book in res["books"].items():
        L.append(f"## book {kind}  (ladder depth L = {book['depth']:.2f}; counts {book['counts']})")
        for per in ("full", "pre2019", "2019+"):
            ev = book[per]
            L.append(f"### {kind} {per}: entries {ev.get('entry_range')}; skipped {ev['n_skipped']} "
                     f"({(ev['share_skipped'] or 0):.1%})")
            L.append("| trades (P&L % of collateral) | n | mean | median | win | worst 5% | worst 1% | CVaR 5% |")
            L.append("|---|---|---|---|---|---|---|---|")
            L.append(_ts("all (rule off)", ev["trades_all"]))
            L.append(_ts("kept (rule on)", ev["trades_kept"]))
            L.append(_ts("skipped by the rule", ev["trades_skipped"]))
            if "portfolio_off" in ev:
                L.append("| portfolio | total % | ann ret % | ann vol % | Sharpe | max DD % | worst day % | worst 21d % | avg deployed |")
                L.append("|---|---|---|---|---|---|---|---|---|")
                L.append(_pf("rule off", ev["portfolio_off"]))
                L.append(_pf("rule on", ev["portfolio_on"]))
            b = ev.get("boot_trade_diff")
            if b:
                L.append(f"bootstrap mean skipped - kept: {b['point']:+.3f} pts, 95% CI [{b['ci95'][0]:+.3f}, "
                         f"{b['ci95'][1]:+.3f}], P(<0) {b['p_lt0']:.3f} ({b['draws']} draws)")
            bp = ev.get("boot_portfolio_diff")
            if bp:
                L.append(f"bootstrap Sharpe on - off 95% CI {[round(x, 3) for x in bp['sharpe_diff_ci95']]} "
                         f"(P on higher {bp['p_sharpe_on_higher']:.3f}); max DD on - off 95% CI "
                         f"{[round(x, 2) for x in bp['maxdd_diff_ci95']]} pts (P shallower {bp['p_dd_on_shallower']:.3f})")
            L.append(f"exit reasons: {ev['exit_reasons']}")
            L.append("by regime (n, skipped, kept mean, skipped mean, skipped - kept, kept worst5, skipped worst5):")
            for g, v in ev["regimes"].items():
                km, sm = v["kept"].get("mean"), v["skipped"].get("mean")
                L.append(f"  {g}: {v['n']}  {v['n_skipped']}  "
                         f"{'' if km is None else f'{km:+.3f}'}  {'' if sm is None else f'{sm:+.3f}'}  "
                         f"{'' if v['diff_mean'] is None else format(v['diff_mean'], '+.3f')}  "
                         f"{'' if km is None else format(v['kept']['q05'], '+.2f')}  "
                         f"{'' if sm is None else format(v['skipped']['q05'], '+.2f')}")
            L.append("")
    L.append(f"VERDICT (csp, 2019+): {res['verdict']} — {'; '.join(res['verdict_why'])}")
    L.append(f"same rule on csp full sample: {res['verdict_full_sample_csp'][0]}; "
             f"strangle 2019+: {res['verdict_strangle_2019'][0]} (reported, not the decision)")
    return "\n".join(L)


def _strip_paths(res: dict) -> dict:
    return json.loads(json.dumps(res, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", help="read DIX.csv / VIX_History.csv / spy.json from this dir")
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--json-out", help="write the derived summary JSON here (no raw rows)")
    args = ap.parse_args(argv)
    if args.data_dir:
        res = run(args.data_dir, args.end)
    else:
        with tempfile.TemporaryDirectory(prefix="edge_p1_g3_") as tmp:
            p1.fetch_all(tmp, args.end)
            res = run(tmp, args.end)
    print(render(res))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(_strip_paths(res), fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
