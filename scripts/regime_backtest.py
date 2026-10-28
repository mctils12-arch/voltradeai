#!/usr/bin/env python3
"""
Regime-layer backtest tool — index level, isolated. [RESEARCH] 2026-10-28,
Stage B step 1 of docs/regime/FINDINGS.md (promotes the Stage A harness that
produced docs/regime/RESEARCH.md §5).

WHAT IT IS. An exposure path e[t] in [0, 1], decided at the close of day t,
applied to the index return from close t+lag-1 to close t+lag (lag 2 = trade
at the next session, hold from its close), charged `cost_bp` per unit of
turnover, leverage capped at 1, cash earning nothing. Reports CAGR, max
drawdown, Sharpe, exposure-level flips per year and time in market, for the
full window and for the five test episodes the program names (2008,
2015–16, late 2018, 2020, 2022) plus 2023–26.

WHAT IT IS NOT. A strategy backtest. There is no point-in-time universe, no
delisted name, no single-name fill, no option in it — docs/regime/FINDINGS.md
explains why Section 10's conditions cannot be met with the repo's data and
why the stock and options sleeves are measured forward instead. Any number
from this file describes the REGIME LAYER on an index, nothing else.

ARMS (reference, from the literature — docs/regime/RESEARCH.md §2.1, §3.1):
  buy_and_hold      e = 1
  sma200_daily      e = 1 while the close is above its 200-day SMA (daily)
  faber_monthly     e decided only on the last trading day of each month:
                    1 if the close is above its 10-month (210-day) SMA
  vol_target_<t>    e = min(1, t / realized vol over the trailing 20 sessions)
  faber_x_vt15_<f>  Faber (never below floor f) × the 15% vol target
  current_proxy     the live classifier (regime_util.classify_regime_5level)
                    with ^VIX / its 30-day mean standing in for the VXX ratio,
                    the floor schedule SPY_FLOOR_* (.70/.90/.35/0/0) normalised
                    to BULL = 1, and the 3-reading dwell approximated as three
                    trading days for calming moves — the SCHEDULE's effect,
                    not the floor's level.
The live regime engine's own exposure path is added as an arm by
regime_engine.py (Stage B step 2) so the engine and this tool share one
implementation.

SENSITIVITY. `--sweep` re-runs every parametric arm at ±20% of each of its
parameters and reports whether any ORDERING of the arms (by CAGR, by max DD,
by Sharpe) reverses — the program's robustness test, applied as a range.

DATA. backtest_v2.fetch_bars (Alpaca-first, Yahoo fallback, 24-h disk cache
under .bt_cache/). Index closes only; nothing is written outside the cache
and the --out/--json paths.

Run: python3 scripts/regime_backtest.py [--symbols SPY,IWM] [--start 2007-08-28]
     [--end 2026-10-26] [--lag 2] [--cost-bp 5] [--sweep]
     [--out research/regime_backtest_YYYY-MM-DD.md] [--json <path>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PERIODS = [
    ("GFC 2007-08..2009-06", "2007-08-28", "2009-06-30"),
    ("2015-16", "2015-06-01", "2016-03-31"),
    ("late 2018", "2018-09-01", "2018-12-31"),
    ("2020", "2020-01-01", "2020-06-30"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023-26", "2023-01-01", "2026-12-31"),
]
DEFAULT_START = "2007-08-28"
DEFAULT_SYMBOLS = ("SPY", "IWM")
# The live floor schedule, system_config.py SPY_FLOOR_* (BULL/NEUTRAL/CAUTION/BEAR/PANIC).
FLOOR_SCHEDULE = {"BULL": 0.70, "NEUTRAL": 0.90, "CAUTION": 0.35, "BEAR": 0.0, "PANIC": 0.0}
FLOOR_RISK_RANK = {"BULL": 0, "NEUTRAL": 1, "CAUTION": 2, "BEAR": 3, "PANIC": 4}
FLOOR_CONFIRM_READINGS = 3   # system_config FLOOR_REGIME_CONFIRM_CYCLES


# ── pure helpers ─────────────────────────────────────────────────────────────
def sma(values, i, n):
    """Simple average of values[i-n+1 .. i]; None before enough history."""
    if i < n - 1:
        return None
    window = values[i - n + 1:i + 1]
    return sum(window) / n


def realized_vol(values, i, window=20):
    """Annualized close-to-close log-return vol over the trailing `window` sessions ending at i."""
    if i < window:
        return None
    rets = [math.log(values[t] / values[t - 1]) for t in range(i - window + 1, i + 1)]
    return st.pstdev(rets) * math.sqrt(252)


def month_end_flags(dates):
    """True on the last trading day of each calendar month (and on the final date)."""
    out = []
    for i, d in enumerate(dates):
        last = (i + 1 == len(dates)) or (dates[i + 1][:7] != d[:7])
        out.append(last)
    return out


def evaluate(dates, closes, expo, lag=2, cost_bp=5.0, start=None, end=None):
    """Evaluate an exposure path. expo[i] is decided at close i and applies to the
    return closes[i+lag-1] -> closes[i+lag]. Turnover is charged at cost_bp per
    unit of |Δe|. Returns CAGR, max drawdown, Sharpe (daily, annualized),
    exposure-level flips per year, time in market and the day count."""
    n = len(closes)
    if len(dates) != n or len(expo) != n:
        raise ValueError("dates, closes and expo must have the same length")
    if lag < 1:
        raise ValueError("lag must be >= 1")
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    rets = []
    turn = 0.0
    prev = 0.0
    flips = 0
    in_market = 0
    days = 0
    prev_level = None
    for i in range(n - lag):
        d = dates[i + lag]
        if start and d < start:
            prev = expo[i]
            continue
        if end and d > end:
            break
        e = max(0.0, min(1.0, float(expo[i])))
        r = closes[i + lag] / closes[i + lag - 1] - 1.0
        tc = abs(e - prev) * cost_bp / 1e4
        turn += abs(e - prev)
        ret = e * r - tc
        eq *= (1.0 + ret)
        rets.append(ret)
        prev = e
        days += 1
        in_market += 1 if e > 0 else 0
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
        level = round(e, 2)
        if prev_level is not None and level != prev_level:
            flips += 1
        prev_level = level
    years = days / 252.0 if days else 1.0
    cagr = eq ** (1.0 / years) - 1.0 if days else 0.0
    sd = st.pstdev(rets) if len(rets) > 2 else 0.0
    sharpe = (st.mean(rets) / sd * math.sqrt(252)) if sd > 0 else 0.0
    return {
        "cagr": cagr, "mdd": mdd, "sharpe": sharpe,
        "flips_yr": flips / years, "tim": (in_market / days) if days else 0.0,
        "turnover_yr": turn / years, "days": days,
    }


# ── reference arms (each returns an exposure path aligned to `closes`) ───────
def arm_buy_and_hold(dates, closes):
    return [1.0] * len(closes)


def arm_sma200_daily(dates, closes, window=200):
    out = []
    for i in range(len(closes)):
        m = sma(closes, i, window)
        out.append(1.0 if (m is None or closes[i] > m) else 0.0)
    return out


def arm_faber_monthly(dates, closes, months=10):
    """Faber (2007): on the last trading day of each month, 1 if the close is above
    its `months`-month SMA (21 sessions per month), else 0; held until next month end."""
    window = int(round(months * 21))
    flags = month_end_flags(dates)
    state = 1.0
    out = []
    for i in range(len(closes)):
        if flags[i]:
            m = sma(closes, i, window)
            if m is not None:
                state = 1.0 if closes[i] > m else 0.0
        out.append(state)
    return out


def arm_vol_target(dates, closes, target=0.15, window=20, cap=1.0, floor=0.0):
    out = []
    for i in range(len(closes)):
        rv = realized_vol(closes, i, window)
        e = cap if rv is None or rv <= 0 else min(cap, target / rv)
        out.append(max(floor, e))
    return out


def arm_faber_x_vol_target(dates, closes, floor=0.5, target=0.15, months=10, window=20):
    fab = arm_faber_monthly(dates, closes, months)
    vt = arm_vol_target(dates, closes, target, window)
    return [max(floor, fab[i]) * vt[i] for i in range(len(closes))]


def classify_live(vxx_ratio, spy_vs_ma50, spy_below_200_days, spy_above_200d):
    """The live classifier (regime_util.classify_regime_5level). Imported lazily so the
    pure helpers above stay importable without the repo on sys.path."""
    from regime_util import classify_regime_5level
    return classify_regime_5level(vxx_ratio, spy_vs_ma50,
                                  spy_below_200_days=spy_below_200_days,
                                  spy_above_200d=spy_above_200d)


def current_schedule_proxy(dates, closes, vix_by_date, classify=classify_live,
                           schedule=FLOOR_SCHEDULE, confirm=FLOOR_CONFIRM_READINGS):
    """The live exposure SCHEDULE replayed on an index: the live classifier with
    vix / its 30-day mean standing in for the VXX ratio (VXX history is not
    reliable before 2018 — AUDIT.md §5), the floor percentages normalised so
    BULL = 1, and the floor dwell approximated as `confirm` trading days for
    calming / lateral moves (escalations immediate, as in _resolve_floor_regime).
    Returns (exposure path, list of confirmed labels)."""
    norm = schedule["BULL"]
    vix_series = [vix_by_date.get(d) for d in dates]
    last_vix = None
    confirmed = None
    pending = None
    pend_n = 0
    expo = []
    labels = []
    below_streak = 0
    for i in range(len(closes)):
        v = vix_series[i] if vix_series[i] is not None else last_vix
        last_vix = v
        window = [x for x in vix_series[max(0, i - 29):i + 1] if x is not None]
        vr = (v / (sum(window) / len(window))) if (v and window) else 1.0
        ma50 = sma(closes, i, 50) or closes[i]
        ma200 = sma(closes, i, 200)
        above = (closes[i] > ma200) if ma200 is not None else True
        below_streak = 0 if above else below_streak + 1
        raw = classify(vr, closes[i] / ma50, below_streak, above)
        if confirmed is None:
            confirmed = raw
        elif raw == confirmed:
            pending, pend_n = None, 0
        elif FLOOR_RISK_RANK[raw] > FLOOR_RISK_RANK[confirmed] and raw in ("CAUTION", "BEAR", "PANIC"):
            confirmed, pending, pend_n = raw, None, 0
        else:
            if pending == raw:
                pend_n += 1
            else:
                pending, pend_n = raw, 1
            if pend_n >= confirm:
                confirmed, pending, pend_n = raw, None, 0
        expo.append(schedule[confirmed] / norm)
        labels.append(confirmed)
    return expo, labels


# ── the live engine as an arm (same implementation the bot runs) ─────────────
def engine_arms():
    """regime_engine.exposure_paths on the gating index: the quantized multiplier
    consumers read, the continuous value, and the never-fully-out core multiplier.
    Context arms receive the whole closes dict (basket + index family)."""
    try:
        import regime_engine as re_
    except ImportError:
        return {}

    def _paths(dates, closes_by_symbol, gating_symbol, extra, which, **params):
        basket = {s: closes_by_symbol[s] for s in re_.BREADTH_BASKET if s in closes_by_symbol and s != gating_symbol}
        vix = extra.get("^VIX")
        vix3m = extra.get("^VIX3M")
        cont, quant, core, _comps = re_.exposure_paths(dates, closes_by_symbol[gating_symbol], basket, vix, vix3m, params or None)
        return {"continuous": cont, "quantized": quant, "core": core}[which]

    return {
        "engine_quantized": (lambda d, c, g, x, **p: _paths(d, c, g, x, "quantized", **p), {"band": 0.01, "filter_days": 6, "vol_halflife": 20}, "context"),
        "engine_continuous": (lambda d, c, g, x, **p: _paths(d, c, g, x, "continuous", **p), {}, "context"),
        "engine_core": (lambda d, c, g, x, **p: _paths(d, c, g, x, "core", **p), {}, "context"),
    }


def engine_symbols():
    try:
        import regime_engine as re_
        return list(re_.BREADTH_BASKET) + ["^VIX3M"]
    except ImportError:
        return []


# ── arm registry (name -> (builder, parameters[, "context"])) ────────────────
def reference_arms():
    """Parametric arm table. Parameters listed here are the ones --sweep perturbs."""
    return {
        "buy_and_hold": (arm_buy_and_hold, {}),
        "sma200_daily": (arm_sma200_daily, {"window": 200}),
        "faber_monthly": (arm_faber_monthly, {"months": 10}),
        "vol_target_10": (arm_vol_target, {"target": 0.10, "window": 20}),
        "vol_target_15": (arm_vol_target, {"target": 0.15, "window": 20}),
        "vol_target_20": (arm_vol_target, {"target": 0.20, "window": 20}),
        "faber_x_vt15_floor0": (arm_faber_x_vol_target, {"floor": 0.0, "target": 0.15, "months": 10, "window": 20}),
        "faber_x_vt15_floor50": (arm_faber_x_vol_target, {"floor": 0.5, "target": 0.15, "months": 10, "window": 20}),
    }


def perturbations(params, pct=0.20):
    """Every single-parameter ±pct variant of a parameter dict (floors/caps stay in [0,1])."""
    out = []
    for k, v in params.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        for sign in (-1, 1):
            nv = v * (1 + sign * pct)
            if isinstance(v, int):
                nv = max(1, int(round(nv)))
            if k in ("floor", "cap"):
                nv = max(0.0, min(1.0, nv))
            p = dict(params)
            p[k] = nv
            out.append((f"{k}{'+' if sign > 0 else '-'}{int(pct * 100)}%", p))
    return out


def claims_hold(full):
    """The three PRE-REGISTERED orderings RESEARCH.md §5.4 rests on, checked on a
    {arm: FULL-window metrics} dict for the gating index. Returns {claim: bool}.
    Arms missing from `full` make their claim vacuously True (so a partial table
    never fails a claim it cannot test)."""
    bh = full.get("buy_and_hold")
    gated = {n: r for n, r in full.items() if n != "buy_and_hold"}
    gated = {n: r for n, r in gated.items() if not n.startswith("engine_")}   # the engine is measured, not claimed
    trend_gated = {n: r for n, r in gated.items() if n.startswith(("sma200", "faber", "current_proxy"))}
    out = {}
    # Trend gates (binary or floored) give up CAGR with leverage capped at 1 (Zakamulin, RESEARCH §2.1, §5.5).
    out["trend_gated_arms_trail_buy_and_hold_cagr"] = (bh is None) or all(r["cagr"] <= bh["cagr"] + 1e-12 for r in trend_gated.values())
    # The loose vol dial (target ≈ the index's long-run vol) is CAGR-neutral within 1 pp while cutting the
    # drawdown — found by the first committed run (2026-10-28) against the prior that it would trail; recorded as such.
    vt20 = full.get("vol_target_20")
    out["loose_vol_dial_is_cagr_neutral_within_1pp"] = (bh is None or vt20 is None) or abs(vt20["cagr"] - bh["cagr"]) <= 0.01
    out["gated_arms_cut_max_drawdown"] = (bh is None) or all(r["mdd"] >= bh["mdd"] for r in gated.values())
    f0, f50 = full.get("faber_x_vt15_floor0"), full.get("faber_x_vt15_floor50")
    out["never_fully_out_beats_fully_out_cagr"] = (f0 is None or f50 is None) or f50["cagr"] > f0["cagr"]
    cur = full.get("current_proxy")
    out["current_schedule_proxy_has_lowest_cagr"] = (cur is None) or all(cur["cagr"] <= r["cagr"] + 1e-12 for n, r in full.items() if n != "current_proxy")
    return out


def rank_order(results, key, reverse=True):
    """Arm names ordered by a metric (ties broken by name for determinism)."""
    return [name for name, _ in sorted(results.items(), key=lambda kv: (-kv[1][key] if reverse else kv[1][key], kv[0]))]


def ordering_reversals(base_results, variant_results, keys=("cagr", "mdd", "sharpe")):
    """Pairs of arms whose order under `key` differs between base and variant results.
    `mdd` is compared as a drawdown (less negative = better)."""
    reversals = []
    for key in keys:
        base_rank = rank_order(base_results, key)
        var_rank = rank_order(variant_results, key)
        pos_b = {n: i for i, n in enumerate(base_rank)}
        pos_v = {n: i for i, n in enumerate(var_rank)}
        names = [n for n in base_rank if n in pos_v]
        for a_i, a in enumerate(names):
            for b in names[a_i + 1:]:
                if (pos_b[a] < pos_b[b]) != (pos_v[a] < pos_v[b]):
                    reversals.append((key, a, b))
    return reversals


# ── data ─────────────────────────────────────────────────────────────────────
def load_closes(symbol, days=7000):
    import backtest_v2 as bt
    bars = bt.fetch_bars(symbol, days)
    if not bars or not bars.get("date"):
        raise RuntimeError(f"no bars for {symbol}")
    return dict(zip(bars["date"], bars["close"]))


def aligned(series_by_symbol, anchor):
    """Dates present in the anchor series; other series forward-filled onto them."""
    dates = sorted(series_by_symbol[anchor])
    out = {}
    for sym, s in series_by_symbol.items():
        vals = []
        last = None
        for d in dates:
            v = s.get(d)
            if v is not None:
                last = v
            vals.append(last)
        out[sym] = vals
    return dates, out


# ── run ──────────────────────────────────────────────────────────────────────
def run_arms(dates, closes_by_symbol, arms, symbols, lag, cost_bp, start, end, vix_by_date=None):
    """Evaluate every arm on every symbol over the full window and the episodes.
    Returns {arm: {symbol: {"FULL": metrics, <episode>: metrics}}, ...} plus
    per-arm level counts. The exposure path is always decided on the FIRST
    symbol's closes (the gating index) and applied to each symbol."""
    gating = closes_by_symbol[symbols[0]]
    out = {}
    levels = {}
    for name, spec in arms.items():
        builder, params = spec[0], spec[1]
        if name == "current_proxy":
            expo, labels = current_schedule_proxy(dates, gating, vix_by_date or {})
        elif len(spec) > 2 and spec[2] == "context":
            extra_series = {k: v for k, v in closes_by_symbol.items() if k.startswith("^")}
            expo = builder(dates, closes_by_symbol, symbols[0], extra_series, **params)
        else:
            expo = builder(dates, gating, **params)
        out[name] = {}
        for sym in symbols:
            c = closes_by_symbol[sym]
            res = {"FULL": evaluate(dates, c, expo, lag, cost_bp, start, end)}
            for label, a, b in PERIODS:
                if b < start or a > end:
                    continue
                res[label] = evaluate(dates, c, expo, lag, cost_bp, max(a, start), min(b, end))
            out[name][sym] = res
        counts = {}
        for i, e in enumerate(expo):
            if start and dates[i] < start:
                continue
            if end and dates[i] > end:
                continue
            k = f"{round(e, 2):.2f}"
            counts[k] = counts.get(k, 0) + 1
        levels[name] = counts
    return out, levels


def sweep(dates, closes_by_symbol, arms, symbols, lag, cost_bp, start, end, pct=0.20, vix_by_date=None):
    """±pct on every numeric parameter of every parametric arm, one at a time,
    plus lag 1 and doubled cost for the whole table. For each variant: the
    FULL-window metrics on the gating index, the number of pairwise ordering
    swaps vs the base run (a diagnostic — near-tied arms swap on tiny changes),
    and whether the PRE-REGISTERED claims (claims_hold) still hold — the
    robustness statement the program asks for is the claims one."""
    base, _ = run_arms(dates, closes_by_symbol, arms, symbols[:1], lag, cost_bp, start, end, vix_by_date=vix_by_date)
    base_full = {n: r[symbols[0]]["FULL"] for n, r in base.items()}
    rows = []
    all_reversals = []
    broken_claims = []

    def record(arm, tag, params, full):
        rev = ordering_reversals(base_full, full)
        claims = claims_hold(full)
        failed = [c for c, ok in claims.items() if not ok]
        metrics = full.get(arm, {"cagr": None, "mdd": None, "sharpe": None})
        rows.append({"arm": arm, "variant": tag, "params": params,
                     "cagr": metrics["cagr"], "mdd": metrics["mdd"], "sharpe": metrics["sharpe"],
                     "reversals": len(rev), "claims_ok": not failed, "failed_claims": failed})
        all_reversals.extend((arm, tag, k, a, b) for k, a, b in rev)
        broken_claims.extend((arm, tag, c) for c in failed)

    for name, spec in arms.items():
        builder, params = spec[0], spec[1]
        if name in ("buy_and_hold", "current_proxy"):
            continue
        for tag, p in perturbations(params, pct):
            variant_arms = dict(arms)
            variant_arms[name] = (builder, p) + tuple(spec[2:])
            res, _ = run_arms(dates, closes_by_symbol, variant_arms, symbols[:1], lag, cost_bp, start, end, vix_by_date=vix_by_date)
            record(name, tag, p, {n: r[symbols[0]]["FULL"] for n, r in res.items()})
    # execution assumptions: lag 1 and doubled cost as structural variants of the whole table
    for lag_v, cost_v in ((1, cost_bp), (lag, max(cost_bp * 2, 10.0))):
        res, _ = run_arms(dates, closes_by_symbol, arms, symbols[:1], lag_v, cost_v, start, end, vix_by_date=vix_by_date)
        record("ALL", f"lag {lag_v}, cost {cost_v:g} bp", {}, {n: r[symbols[0]]["FULL"] for n, r in res.items()})
    return rows, all_reversals, broken_claims


def render(results, levels, meta, sweep_rows=None, reversals=None):
    sym = meta["symbols"]
    lines = [f"# Regime-layer backtest — {meta['run_date']}", "",
             f"Index-level, isolated (scripts/regime_backtest.py). Window {meta['start']} → {meta['end']}, lag {meta['lag']} session(s), "
             f"{meta['cost_bp']:g} bp per unit of turnover, leverage capped at 1, cash earns 0. Exposure decided on {sym[0]} closes and applied to each column's index. "
             "NOT a strategy backtest (docs/regime/FINDINGS.md): no universe, no fills, no options.", ""]
    lines.append("## Full window")
    lines.append("")
    hdr = "| Arm | " + " | ".join(f"{s} CAGR / max DD / Sharpe" for s in sym) + " | flips/yr | time in mkt |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(sym) + 3))
    for name, per_sym in results.items():
        cells = []
        for s in sym:
            r = per_sym[s]["FULL"]
            cells.append(f"{r['cagr']*100:.2f}% / {r['mdd']*100:.1f}% / {r['sharpe']:.2f}")
        r0 = per_sym[sym[0]]["FULL"]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {r0['flips_yr']:.1f} | {r0['tim']*100:.0f}% |")
    lines.append("")
    lines.append(f"## Episodes ({sym[0]})")
    lines.append("")
    ep_names = [label for label, _, _ in PERIODS if label in next(iter(results.values()))[sym[0]]]
    lines.append("| Arm | " + " | ".join(f"{e} CAGR / DD" for e in ep_names) + " |")
    lines.append("|" + "---|" * (len(ep_names) + 1))
    for name, per_sym in results.items():
        cells = []
        for e in ep_names:
            r = per_sym[sym[0]][e]
            cells.append(f"{r['cagr']*100:.1f}% / {r['mdd']*100:.1f}%")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Exposure levels (days at each level, gating index)")
    lines.append("")
    for name, counts in levels.items():
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
        lines.append(f"- {name}: " + ", ".join(f"{k} → {v}" for k, v in top))
    base_full = {n: per_sym[sym[0]]["FULL"] for n, per_sym in results.items()}
    claims = claims_hold(base_full)
    lines.append("")
    lines.append("## Pre-registered claims (gating index, full window)")
    lines.append("")
    for c, ok in claims.items():
        lines.append(f"- {c.replace('_', ' ')}: **{'holds' if ok else 'FAILS'}**")
    if sweep_rows is not None:
        lines.append("")
        lines.append("## ±20% sensitivity (gating index, full window)")
        lines.append("")
        lines.append("| Arm | Variant | CAGR | max DD | Sharpe | pairwise swaps vs base | claims |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in sweep_rows:
            c = f"{row['cagr']*100:.2f}%" if row["cagr"] is not None else "—"
            d = f"{row['mdd']*100:.1f}%" if row["mdd"] is not None else "—"
            s = f"{row['sharpe']:.2f}" if row["sharpe"] is not None else "—"
            ok = "hold" if row.get("claims_ok", True) else "FAIL: " + ", ".join(row.get("failed_claims", []))
            lines.append(f"| {row['arm']} | {row['variant']} | {c} | {d} | {s} | {row['reversals']} | {ok} |")
        lines.append("")
        n_var = len(sweep_rows)
        n_bad = sum(1 for r in sweep_rows if not r.get("claims_ok", True))
        if n_bad == 0:
            lines.append(f"**Every pre-registered claim holds under all {n_var} variants** (±20% on each parameter, lag 1, doubled cost).")
        else:
            lines.append(f"**{n_bad} of {n_var} variants break a pre-registered claim** — see the claims column; the conclusion that depends on it is NOT robust.")
        if reversals:
            lines.append(f"Pairwise swaps between near-tied reference arms: {len(reversals)} (diagnostic only; the arms are not the claim).")
    lines.append("")
    lines.append("Verification: `python3 -m pytest -q test_regime_backtest.py`; rerun with the same flags to reproduce (cached bars refresh after 24 h, so the end date may advance).")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="comma list; the first is the gating index")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--lag", type=int, default=2)
    ap.add_argument("--cost-bp", type=float, default=5.0)
    ap.add_argument("--sweep", action="store_true", help="±20%% single-parameter sensitivity + lag/cost variants")
    ap.add_argument("--no-engine", action="store_true", help="skip the live regime_engine arms")
    ap.add_argument("--out", default=None, help="markdown path (default: print)")
    ap.add_argument("--json", default=None, help="JSON path for the raw results")
    args = ap.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    arms = reference_arms()
    arms["current_proxy"] = (None, {})
    eng = {} if args.no_engine else engine_arms()
    extra_symbols = engine_symbols() if eng else []
    series = {s: load_closes(s) for s in symbols}
    for s in extra_symbols:
        if s not in series:
            try:
                series[s] = load_closes(s)
            except Exception as e:  # a basket member without history only weakens breadth; say so
                print(f"warning: no bars for {s}: {e}")
    vix = load_closes("^VIX")
    series["^VIX"] = vix
    dates, closes = aligned(series, symbols[0])
    arms.update(eng)
    results, levels = run_arms(dates, closes, arms, symbols, args.lag, args.cost_bp, args.start, args.end, vix_by_date=vix)
    sweep_rows = reversals = broken = None
    if args.sweep:
        sweep_rows, reversals, broken = sweep(dates, closes, arms, symbols, args.lag, args.cost_bp, args.start, args.end, vix_by_date=vix)
    meta = {"run_date": date.today().isoformat(), "symbols": symbols, "start": args.start,
            "end": min(args.end, dates[-1]), "lag": args.lag, "cost_bp": args.cost_bp,
            "first_date": dates[0], "last_date": dates[-1], "sessions": len(dates)}
    text = render(results, levels, meta, sweep_rows, reversals)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "results": results, "levels": levels,
                       "claims": claims_hold({n: r[symbols[0]]["FULL"] for n, r in results.items()}),
                       "sweep": sweep_rows, "reversals": reversals, "broken_claims": broken}, f, indent=1, default=str)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
