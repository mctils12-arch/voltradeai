#!/usr/bin/env python3
"""
Research test 2, gate 3 (LOGIC): the options manager's rules simulated on the
cash-secured puts the bot would have sold — what the CSP sleeve earns when it
is allowed to hold. [RULE-REVIEW] measurement, 2026-10-26.

INPUT: the `chain_csp_entry` diag probe (server/chainCspEntry.ts) — per
underlying per archived chain day, the put options_execution._select_sell_put
would have sold, with its archived bid/ask/iv/delta (indicative feed).
The archive cannot mark the same contract on later days for single names
(rotating candidate list), so each entry is marked FORWARD from the
underlying's daily closes (backtest_v2.fetch_bars: Alpaca-first, Yahoo
fallback) with a Black–Scholes MODEL MARK at the entry's implied vol. That is
inference standing in for ground truth, labelled as such on every number.

RULES (the live order in options_manager.manage, each constant pinned to that
file's source by test_csp_manager_rule_sim.py): critical close at DTE <= 5;
profit target at 50% of the credit; loss limit at -2x the credit; 21-DTE
close. The 60-minute minimum hold is sub-daily and cannot bind here; the
VRP-evaporation exit needs the live vol surface and is NOT replicated.

VARIANTS (four, each theory-motivated — discount any winner by four):
  live          the rules above
  no_dte_exit   hold to expiry: the 21-DTE close and the 5-DTE critical close off
  no_stop       the -2x loss limit off (the left-tail hypothesis: the stop
                crystallizes earnings jumps at the worst point)
  earnings_gate live rules, but an entry whose holding window contains an
                earnings date is never sold (yfinance earnings dates; names
                without data are kept and counted)

COSTS: sold at the archived BID (the spread vs mid is paid at entry); bought
back at the model mark plus half the archived entry spread (a modelled exit
spread — stated). `pnl_gross` is mid-to-model with no spread, for the
decomposition.

PAIRED READ: a rule-change variant is also reported as the same entry's P&L
under the variant minus under live (both closed) — an unpaired difference of
means mixes the rule with the censoring it changes. earnings_gate, a subset,
is read as live-rules P&L on kept vs dropped entries, and dropped once more
without its single worst entry.

OVERLAP: the same name is an entry on every archived day, so consecutive
entries overlap almost entirely. Statistics run on a DEDUPED set — one
entry per name per `dedupe_days` — and the raw count is reported beside it.

HONESTY: a model mark at constant IV smooths the intraday path the -2x stop
actually fires on, so the stop's measured cost is a LOWER bound; indicative
quotes; the probe's liquidity test lacks the volume branch (a subset of the
live universe); r = R_RISK_FREE, a stated constant; calendar DTE as the
manager uses it. Nothing here trades or changes a rule.

Run: python3 scripts/csp_manager_rule_sim.py --probe https://voltradeai.com
     (DIAG_TOKEN in the environment, never printed) [--earnings]
     [--sleeves-json <dump of the sleeves probe>] [--from-json <saved rows>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PRIOR = (
    "Stated before the run: 60% that the live rules show NEGATIVE expectancy per "
    "contract net of the archived spread on the deduped single-name set (the sleeve "
    "lost when it held); 70% that `no_stop` beats `live` (the -2x stop crystallizes "
    "the left tail); 55% that `no_dte_exit` beats `live` by less than the spread it "
    "saves; 65% that `earnings_gate` removes most of the worst-10 entries."
)

# Pinned to options_manager.py by the tests.
RULES = {"profit_target_pct": 0.50, "loss_limit_mult": 2.0, "dte_exit": 21, "dte_critical": 5}
R_RISK_FREE = 0.04
VARIANTS = ("live", "no_dte_exit", "no_stop", "earnings_gate")
DTE_BUCKETS = ((30, 35, "30-35"), (36, 42, "36-42"), (43, 50, "43-50"))


# ── pricing ────────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put(S: float, K: float, T: float, sigma: float, r: float = R_RISK_FREE) -> float:
    """Black–Scholes European put. At T <= 0 the intrinsic value."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0.0)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def _d(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


def _day_or_none(s: object) -> Optional[date]:
    try:
        return _d(str(s))
    except ValueError:
        return None


# ── one entry, one variant ─────────────────────────────────────────────────

def simulate_entry(entry: dict, bars: dict, variant: str = "live",
                   earnings: Optional[Iterable[str]] = None) -> Optional[dict]:
    """Mark one chain_csp_entry row forward through `bars` under `variant`.
    None when the entry is unusable (no quote/iv/strike, no bar on the entry
    day). A gated-out entry under earnings_gate returns {"skipped": "earnings"}."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    try:
        K = float(entry["strike"]); bid = float(entry["bid"]); ask = float(entry["ask"]); iv = float(entry["iv"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (K > 0 and bid > 0 and ask >= bid and iv > 0) or not entry.get("exp"):
        return None
    entry_day, exp = _d(entry["day"]), _d(entry["exp"])
    dates: List[str] = list(bars.get("date") or [])
    closes: List[float] = list(bars.get("close") or [])
    try:
        i0 = dates.index(entry_day.isoformat())
    except ValueError:
        return None
    if variant == "earnings_gate" and earnings:
        for e in earnings:
            ed = _day_or_none(e)
            if ed is None:  # a malformed provider date cannot gate anything; coverage is reported, not the row
                continue
            if entry_day <= ed <= exp:
                return {"skipped": "earnings", "u": entry["u"], "day": entry["day"], "earnings": ed.isoformat()}
    mid = (bid + ask) / 2.0
    half_spread = (ask - bid) / 2.0
    credit = bid * 100.0
    base = {"u": entry["u"], "day": entry["day"], "exp": exp.isoformat(), "strike": K, "dte_entry": (exp - entry_day).days,
            "credit": round(credit, 2), "iv": iv, "delta": entry.get("delta"), "spread_pct": entry.get("spreadPct")}
    use_dte_exits = variant != "no_dte_exit"
    use_stop = variant != "no_stop"
    for i in range(i0 + 1, len(dates)):
        t = _d(dates[i]); S = closes[i]
        if not (S and S > 0):
            continue
        dte = (exp - t).days
        if dte <= 0:
            exit_mark = max(K - S, 0.0)
            return {**base, "exit_day": dates[i], "exit_mark": round(exit_mark, 4), "exit_reason": "expiry",
                    "hold_days": (t - entry_day).days, "S_exit": S,
                    "pnl": round(credit - exit_mark * 100.0, 2), "pnl_gross": round((mid - exit_mark) * 100.0, 2),
                    "pnl_pct_collateral": round((credit - exit_mark * 100.0) / (K * 100.0) * 100.0, 3), "censored": False}
        mark = bs_put(S, K, dte / 365.0, iv)
        exit_cost = mark + half_spread
        pnl = credit - exit_cost * 100.0
        reason = None
        if use_dte_exits and dte <= RULES["dte_critical"]:
            reason = "dte_critical"
        elif pnl >= RULES["profit_target_pct"] * credit:
            reason = "profit_target"
        elif use_stop and pnl <= -RULES["loss_limit_mult"] * credit:
            reason = "loss_limit"
        elif use_dte_exits and dte <= RULES["dte_exit"]:
            reason = "dte_exit"
        if reason:
            return {**base, "exit_day": dates[i], "exit_mark": round(mark, 4), "exit_reason": reason,
                    "hold_days": (t - entry_day).days, "S_exit": S,
                    "pnl": round(pnl, 2), "pnl_gross": round((mid - mark) * 100.0, 2),
                    "pnl_pct_collateral": round(pnl / (K * 100.0) * 100.0, 3), "censored": False}
    # bars ran out before any exit
    if len(dates) > i0 + 1:
        S = closes[-1]; t = _d(dates[-1]); dte = max((exp - t).days, 0)
        mark = bs_put(S, K, dte / 365.0, iv)
        pnl = credit - (mark + half_spread) * 100.0
        return {**base, "exit_day": dates[-1], "exit_mark": round(mark, 4), "exit_reason": "censored",
                "hold_days": (t - entry_day).days, "S_exit": S, "pnl": round(pnl, 2),
                "pnl_gross": round((mid - mark) * 100.0, 2), "pnl_pct_collateral": round(pnl / (K * 100.0) * 100.0, 3),
                "censored": True}
    return None


# ── sets of entries ────────────────────────────────────────────────────────

def dedupe(entries: Iterable[dict], days: int = 5) -> List[dict]:
    """One entry per underlying per `days`-day window (the first), in day order."""
    out: List[dict] = []
    last: Dict[str, date] = {}
    for e in sorted(entries, key=lambda r: (str(r.get("day")), str(r.get("u")))):
        if e.get("reason", "ok") != "ok" or not e.get("occ"):
            continue
        u, d = e["u"], _d(e["day"])
        prev = last.get(u)
        if prev is not None and (d - prev).days < days:
            continue
        last[u] = d
        out.append(e)
    return out


def filter_entries(rows: Iterable[dict], max_abs_delta: Optional[float] = None,
                   asset_names: Optional[Dict[str, Optional[str]]] = None) -> dict:
    """The selectable rows the bot would actually have traded: optionally only
    picks within `max_abs_delta` of the money (the archive carries no open
    interest, so the probe's size-only liquidity test skews picks toward the
    money — the live bot targets −0.30), and optionally minus the leveraged /
    inverse funds the live bot excludes (leveraged_etf_filter, by the asset's
    NAME when one is given, else its seed list). Counts what each cut removed."""
    out: List[dict] = []
    cut_delta = 0; cut_lev = 0
    excluded: set = set()
    if asset_names is not None:
        from leveraged_etf_filter import is_leveraged_or_inverse
        for r in rows:
            u = str(r.get("u") or "").upper()
            if u in excluded:
                cut_lev += 1
                continue
            if is_leveraged_or_inverse(u, asset_names.get(u)):
                excluded.add(u); cut_lev += 1
                continue
            d = r.get("delta")
            if max_abs_delta is not None and (d is None or abs(float(d)) > max_abs_delta):
                cut_delta += 1
                continue
            out.append(r)
    else:
        for r in rows:
            d = r.get("delta")
            if max_abs_delta is not None and (d is None or abs(float(d)) > max_abs_delta):
                cut_delta += 1
                continue
            out.append(r)
    return {"rows": out, "cut_delta": cut_delta, "cut_leveraged": cut_lev, "excluded_names": sorted(excluded)}


def asset_names(symbols: Iterable[str], cache_path: str) -> Dict[str, Optional[str]]:
    """{SYMBOL: long name or None} via yfinance, cached to a JSON file — the
    offline stand-in for the live /v2/assets names the exclusion reads."""
    out: Dict[str, Optional[str]] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            out = json.load(f)
    try:
        import yfinance as yf
    except ImportError:
        return out
    for s in symbols:
        if s in out:
            continue
        try:
            info = yf.Ticker(s).info or {}
            out[s] = info.get("longName") or info.get("shortName")
        except Exception as e:  # a name without a name stays on the seed-only rule, and is counted
            print(f"[csp-sim] asset name unavailable for {s}: {type(e).__name__}", file=sys.stderr)
            out[s] = None
    with open(cache_path, "w") as f:
        json.dump(out, f)
    return out


def simulate_all(entries: Iterable[dict], bars_by_u: Dict[str, dict], variant: str,
                 regime_by_day: Optional[Dict[str, str]] = None,
                 earnings_by_u: Optional[Dict[str, Sequence[str]]] = None) -> dict:
    results: List[dict] = []
    skipped = 0; unusable = 0
    for e in entries:
        bars = bars_by_u.get(e["u"])
        if not bars:
            unusable += 1
            continue
        r = simulate_entry(e, bars, variant, (earnings_by_u or {}).get(e["u"]))
        if r is None:
            unusable += 1
            continue
        if r.get("skipped"):
            skipped += 1
            continue
        r["regime"] = (regime_by_day or {}).get(e["day"])
        results.append(r)
    return {"variant": variant, "results": results, "skipped_earnings": skipped, "unusable": unusable}


def _pct(vals: Sequence[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals); k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


def _stats(rs: Sequence[dict]) -> dict:
    closed = [r for r in rs if not r["censored"]]
    pnl = [r["pnl"] for r in closed]
    n = len(pnl)
    mean = sum(pnl) / n if n else None
    sd = math.sqrt(sum((x - mean) ** 2 for x in pnl) / (n - 1)) if n > 1 else None
    return {
        "n": len(rs), "n_closed": n, "n_censored": len(rs) - n,
        "pnl_mean": round(mean, 2) if n else None, "pnl_se": round(sd / math.sqrt(n), 2) if sd is not None else None,
        "pnl_median": _pct(pnl, 0.5), "pnl_p5": _pct(pnl, 0.05), "pnl_p95": _pct(pnl, 0.95), "pnl_sum": round(sum(pnl), 2),
        "win_rate": round(sum(1 for x in pnl if x > 0) / n, 3) if n else None,
        "expectancy_pct_collateral": round(sum(r["pnl_pct_collateral"] for r in closed) / n, 3) if n else None,
        "spread_cost_mean": round(sum(r["pnl_gross"] - r["pnl"] for r in closed) / n, 2) if n else None,
        "hold_days_mean": round(sum(r["hold_days"] for r in closed) / n, 1) if n else None,
    }


def aggregate(sim: dict) -> dict:
    rs: List[dict] = sim["results"]
    by_reason: Dict[str, int] = defaultdict(int)
    for r in rs:
        by_reason[r["exit_reason"]] += 1
    by_regime = {}
    for reg in sorted({r.get("regime") for r in rs if r.get("regime")}):
        by_regime[reg] = _stats([r for r in rs if r.get("regime") == reg])
    by_dte = {}
    for lo, hi, label in DTE_BUCKETS:
        sub = [r for r in rs if lo <= r["dte_entry"] <= hi]
        if sub:
            by_dte[label] = _stats(sub)
    per_u: Dict[str, dict] = {}
    for r in rs:
        if r["censored"]:
            continue
        p = per_u.setdefault(r["u"], {"n": 0, "pnl_sum": 0.0})
        p["n"] += 1; p["pnl_sum"] = round(p["pnl_sum"] + r["pnl"], 2)
    worst = sorted((r for r in rs if not r["censored"]), key=lambda r: r["pnl"])[:10]
    return {"variant": sim["variant"], "skipped_earnings": sim["skipped_earnings"], "unusable": sim["unusable"],
            "overall": _stats(rs), "by_reason": dict(by_reason), "by_regime": by_regime, "by_dte_entry": by_dte,
            "per_underlying": per_u,
            "worst": [{k: r[k] for k in ("u", "day", "exp", "strike", "credit", "exit_day", "exit_reason", "pnl", "S_exit")} for r in worst]}


def live_comparison(aggs: Dict[str, dict], sleeves: Optional[dict]) -> List[dict]:
    """Per underlying: the live sleeve's realized (sleeves probe, OCC root) beside the live-rules simulation."""
    if not sleeves:
        return []
    live_by_root = {r.get("root"): r for r in sleeves.get("optionsByUnderlying") or [] if r.get("root")}
    sim = aggs.get("live", {}).get("per_underlying", {})
    rows = []
    for u in sorted(set(live_by_root) & set(sim)):
        rows.append({"u": u, "live_realized": live_by_root[u].get("realized"), "live_fills": live_by_root[u].get("fills"),
                     "sim_n": sim[u]["n"], "sim_pnl_sum": sim[u]["pnl_sum"]})
    return rows


def _paired_stats(diffs: Sequence[float]) -> dict:
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean_diff": round(diffs[0], 2) if n else None, "se": None, "t": None,
                "changed": sum(1 for x in diffs if abs(x) > 0.005)}
    m = sum(diffs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    return {"n": n, "mean_diff": round(m, 2), "se": round(se, 2), "t": round(m / se, 2) if se > 0 else None,
            "changed": sum(1 for x in diffs if abs(x) > 0.005)}


def _mean(vals: Sequence[float]) -> Optional[float]:
    return round(sum(vals) / len(vals), 2) if vals else None


def paired_vs_live(sims: Dict[str, dict]) -> dict:
    """The same entry under two rule sets. For a rule change (no_stop, no_dte_exit)
    the read is the variant's P&L minus live's on entries closed under BOTH, overall
    and by entry-day regime — an unpaired difference of means mixes the rule's effect
    with the censoring the rule changes (a longer hold censors more recent entries),
    and the pair removes that; `changed` counts entries whose exit the rule moved.
    earnings_gate is a subset, not a rule change: its read is live-rules P&L on the
    entries it keeps vs the ones it drops, and the dropped set once more without its
    single worst entry — one gap can carry a gate, and that must be visible."""
    live = {(r["u"], r["day"]): r for r in sims.get("live", {}).get("results", [])}
    out: Dict[str, dict] = {}
    for v in ("no_stop", "no_dte_exit"):
        if v not in sims:
            continue
        var = {(r["u"], r["day"]): r for r in sims[v]["results"]}
        pairs = [(k, var[k]["pnl"] - live[k]["pnl"]) for k in live
                 if k in var and not live[k]["censored"] and not var[k]["censored"]]
        by_regime = {}
        for reg in sorted({live[k].get("regime") for k, _ in pairs} - {None}):
            by_regime[reg] = _paired_stats([d for k, d in pairs if live[k].get("regime") == reg])
        out[v] = {"overall": _paired_stats([d for _, d in pairs]), "by_regime": by_regime}
    if "earnings_gate" in sims:
        kept_keys = {(r["u"], r["day"]) for r in sims["earnings_gate"]["results"]}
        kept = [live[k]["pnl"] for k in live if k in kept_keys and not live[k]["censored"]]
        dropped_rs = sorted((live[k] for k in live if k not in kept_keys and not live[k]["censored"]), key=lambda r: r["pnl"])
        dropped = [r["pnl"] for r in dropped_rs]
        worst = dropped_rs[0] if dropped_rs else None
        out["earnings_gate"] = {
            "kept": {"n": len(kept), "mean": _mean(kept)},
            "dropped": {"n": len(dropped), "mean": _mean(dropped)},
            "dropped_ex_worst": {"n": max(len(dropped) - 1, 0), "mean": _mean(dropped[1:]),
                                 "worst_u": worst["u"] if worst else None, "worst_day": worst["day"] if worst else None,
                                 "worst_pnl": worst["pnl"] if worst else None}}
    return out


# ── report ─────────────────────────────────────────────────────────────────

def verdict(aggs: Dict[str, dict], paired: Optional[dict] = None) -> str:
    live = aggs["live"]["overall"]
    parts = []
    if live["n_closed"]:
        parts.append(f"live rules: expectancy {live['expectancy_pct_collateral']:+.2f}% of collateral per entry "
                     f"(mean ${live['pnl_mean']:+,.0f} ± {live['pnl_se']:,.0f} SE per contract, win rate {live['win_rate']:.0%}, n {live['n_closed']})")
        for v in ("no_stop", "no_dte_exit", "earnings_gate"):
            o = aggs.get(v, {}).get("overall") or {}
            if o.get("n_closed"):
                d = o["expectancy_pct_collateral"] - live["expectancy_pct_collateral"]
                tail = ""
                pv = (paired or {}).get(v) or {}
                if v != "earnings_gate" and (pv.get("overall") or {}).get("n"):
                    po = pv["overall"]
                    tail = (f"; paired ${po['mean_diff']:+,.0f} ± {po['se'] if po['se'] is not None else 0:,.0f} per contract, "
                            f"{po['changed']} of {po['n']} entries changed")
                elif v == "earnings_gate" and pv.get("dropped", {}).get("n"):
                    tail = (f"; dropped entries averaged ${pv['dropped']['mean']:+,.0f} under live rules (n {pv['dropped']['n']}), "
                            f"${pv['dropped_ex_worst']['mean'] if pv['dropped_ex_worst']['mean'] is not None else 0:+,.0f} without {pv['dropped_ex_worst']['worst_u']}")
                parts.append(f"{v}: {d:+.2f} pts vs live (n {o['n_closed']}{tail})")
    else:
        parts.append("no closed entries — nothing to judge")
    return "; ".join(parts) + ". Four variants were tried: discount the best by four. Model marks, not fills."


def _row(label: str, s: dict) -> str:
    f = lambda v, fmt: (fmt.format(v) if v is not None else "—")
    return (f"| {label} | {s['n']} | {s['n_closed']} | {s['n_censored']} | {f(s['pnl_mean'], '{:+,.0f}')} | {f(s['pnl_se'], '{:,.0f}')} | "
            f"{f(s['pnl_median'], '{:+,.0f}')} | {f(s['pnl_p5'], '{:+,.0f}')} | {f(s['pnl_p95'], '{:+,.0f}')} | {f(s['win_rate'], '{:.0%}')} | "
            f"{f(s['expectancy_pct_collateral'], '{:+.2f}')} | {f(s['spread_cost_mean'], '{:,.0f}')} | {f(s['hold_days_mean'], '{:.1f}')} |")


HEADER = ("| set | n | closed | censored | mean $ | SE | median $ | p5 | p95 | win | exp % coll | spread $ | hold d |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def render(aggs: Dict[str, dict], meta: dict, comparison: Sequence[dict] = (), paired: Optional[dict] = None) -> str:
    L = [f"# Research test 2 (gate 3): the manager's rules on the CSPs the bot would have sold — {meta['date']}", "",
         f"Generated {meta['generated_at']} by scripts/csp_manager_rule_sim.py — entry rows {meta['entry_rows']} "
         f"({meta['entry_ok']} with a selectable contract) over {meta['entry_days']} archived days and {meta['entry_underlyings']} underlyings; "
         f"deduped to {meta['deduped']} entries ({meta['dedupe_days']}-day window); bars {meta['bars_names']} names; source {meta['source']}.", "",
         "## PRIOR (stated before the run)", "", PRIOR, "", "## VERDICT", "", verdict(aggs, paired), "",
         "## Variants (per contract; sold at the archived bid, bought back at the MODEL mark + half the entry spread)", "", HEADER]
    for v in VARIANTS:
        if v in aggs:
            L.append(_row(v, aggs[v]["overall"]))
    L += ["", "## Exit reasons", "", "| variant | " + " | ".join(("profit_target", "loss_limit", "dte_exit", "dte_critical", "expiry", "censored")) + " | skipped (earnings) |",
          "|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        if v in aggs:
            br = aggs[v]["by_reason"]
            L.append(f"| {v} | " + " | ".join(str(br.get(k, 0)) for k in ("profit_target", "loss_limit", "dte_exit", "dte_critical", "expiry", "censored")) + f" | {aggs[v]['skipped_earnings']} |")
    if paired:
        L += ["", "## Paired vs live (same entry, both closed; $ per contract = variant minus live; unpaired means above mix the rule with the censoring it changes)", "",
              "| variant | set | n | mean diff $ | SE | t | entries changed |", "|---|---|---|---|---|---|---|"]
        for v in ("no_stop", "no_dte_exit"):
            pv = paired.get(v)
            if not pv:
                continue
            for label, st in [("all", pv["overall"])] + list(pv["by_regime"].items()):
                if st["n"]:
                    L.append(f"| {v} | {label} | {st['n']} | {st['mean_diff']:+,.1f} | {st['se'] if st['se'] is not None else '—'} | "
                             f"{st['t'] if st['t'] is not None else '—'} | {st['changed']} |")
        eg = paired.get("earnings_gate")
        if eg and eg["dropped"]["n"]:
            exw = eg["dropped_ex_worst"]
            L += ["", f"earnings_gate is a subset, not a rule change — under LIVE rules the entries it keeps averaged "
                      f"${eg['kept']['mean']:+,.0f} (n {eg['kept']['n']}) and the entries it drops ${eg['dropped']['mean']:+,.0f} (n {eg['dropped']['n']}); "
                      f"without the single worst dropped entry ({exw['worst_u']} {exw['worst_day']}, ${exw['worst_pnl']:+,.0f}) the dropped set averaged "
                      f"${exw['mean'] if exw['mean'] is not None else 0:+,.0f}."]
    if aggs["live"]["by_regime"]:
        L += ["", "## Live rules by entry-day regime", "", HEADER]
        for reg, s in aggs["live"]["by_regime"].items():
            L.append(_row(reg, s))
    if aggs["live"]["by_dte_entry"]:
        L += ["", "## Live rules by entry DTE", "", HEADER]
        for b, s in aggs["live"]["by_dte_entry"].items():
            L.append(_row(b, s))
    L += ["", "## Worst 10 entries, live rules", "", "| u | entry | exp | strike | credit | exit | reason | pnl | S at exit |", "|---|---|---|---|---|---|---|---|---|"]
    for w in aggs["live"]["worst"]:
        L.append(f"| {w['u']} | {w['day']} | {w['exp']} | {w['strike']} | {w['credit']:.0f} | {w['exit_day']} | {w['exit_reason']} | {w['pnl']:+,.0f} | {w['S_exit']:.2f} |")
    if comparison:
        L += ["", "## Live sleeve vs simulation, by underlying (names in both; live = realized from fills, sim = live-rules sum)", "",
              "| u | live realized $ | live fills | sim n | sim pnl sum $ |", "|---|---|---|---|---|"]
        for c in comparison:
            L.append(f"| {c['u']} | {c['live_realized']:+,.0f} | {c['live_fills']} | {c['sim_n']} | {c['sim_pnl_sum']:+,.0f} |")
    L += ["", "## Honesty", "",
          "- Every exit is a Black–Scholes MODEL MARK at the entry's implied vol (constant), not a fill: it smooths the intraday path the −2× stop fires on, so the stop's cost reads as a LOWER bound, and it ignores the vol change a crash brings (an under-mark on the way down).",
          f"- Costs: sold at the archived bid; bought back at the model mark plus half the entry spread (modelled). r = {R_RISK_FREE:.2%}, a stated constant. Calendar DTE as the manager uses it.",
          "- The probe's liquidity test lacks the volume branch the live `_is_liquid` has: this universe is a SUBSET of the live one. Indicative quotes throughout.",
          "- The 60-minute minimum hold cannot bind at daily resolution; the VRP-evaporation exit is not replicated (no historical vol surface).",
          f"- Overlap: one entry per name per {meta['dedupe_days']} days was kept ({meta['deduped']} of {meta['entry_ok']}); SEs are plain, not HAC — treat them as optimistic.",
          ("- Selection cuts: " + (f"picks with |delta| > {meta['max_abs_delta']} dropped ({meta.get('cut_delta', 0)}) — the archive carries no open interest, so the probe's size-only liquidity test skews picks toward the money against the live −0.30 target; " if meta.get("max_abs_delta") is not None else "NO delta cut — picks skew toward the money (archive has no open interest) and this set is riskier than the live −0.30 target; ")
           + (f"leveraged / inverse funds dropped as the live bot does ({meta.get('cut_leveraged', 0)} rows, {len(meta.get('excluded_names') or [])} names, by asset name where yfinance had one, else the seed list)." if meta.get("exclude_leveraged") else "leveraged / inverse funds NOT excluded — the pre-2026-10-20 live universe included them, and they dominate the selectable rows.")),
          "- Four variants were tried; discount the best by four (REASONING STANDARD #4). Nothing here trades or changes a rule."]
    return "\n".join(L) + "\n"


# ── IO ─────────────────────────────────────────────────────────────────────

def fetch_probe(base_url: str, token: str, max_calls: int = 30, sleep_s: float = 1.0) -> List[dict]:
    rows: List[dict] = []
    for _ in range(max_calls):
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/diag/chain_csp_entry?max=15", headers={"x-diag-token": token})
        with urllib.request.urlopen(req, timeout=120) as resp:
            d = json.load(resp)
        rows = list(d.get("rows") or [])
        if not d.get("pending"):
            return rows
        time.sleep(sleep_s)
    return rows


def load_entries(from_json: Optional[str], probe: Optional[str]) -> List[dict]:
    if from_json:
        with open(from_json) as f:
            d = json.load(f)
        return list(d.get("rows") if isinstance(d, dict) else d)
    token = os.environ.get("DIAG_TOKEN", "")
    if not probe or not token:
        raise SystemExit("need --from-json, or --probe with DIAG_TOKEN in the environment")
    return fetch_probe(probe, token)


def earnings_dates(symbols: Iterable[str], cache_path: str) -> Dict[str, List[str]]:
    """yfinance earnings dates per symbol (past and upcoming), cached to a JSON file; a name without data maps to []."""
    out: Dict[str, List[str]] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            out = json.load(f)
    try:
        import yfinance as yf
    except ImportError:
        return out
    for s in symbols:
        if s in out:
            continue
        try:
            df = yf.Ticker(s).get_earnings_dates(limit=16)
            out[s] = sorted({str(ts)[:10] for ts in (df.index if df is not None else [])})
        except Exception as e:  # labelled in the report via coverage, never fatal
            print(f"[csp-sim] earnings dates unavailable for {s}: {type(e).__name__}", file=sys.stderr)
            out[s] = []
    with open(cache_path, "w") as f:
        json.dump(out, f)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--from-json", help="a saved probe dump ({rows: [...]}) instead of the live probe")
    ap.add_argument("--probe", help="base URL of the site (DIAG_TOKEN from the environment)")
    ap.add_argument("--bars-days", type=int, default=400)
    ap.add_argument("--dedupe-days", type=int, default=5)
    ap.add_argument("--max-abs-delta", type=float, default=None,
                    help="keep only picks with |delta| <= this (the live target is 0.30; the archive has no OI, so picks skew to the money)")
    ap.add_argument("--exclude-leveraged", action="store_true",
                    help="drop the leveraged / inverse funds the live bot excludes (asset names via yfinance, cached)")
    ap.add_argument("--earnings", action="store_true", help="enable the earnings_gate variant (yfinance earnings dates)")
    ap.add_argument("--sleeves-json", help="a saved dump of the sleeves probe, for the live-vs-sim comparison")
    ap.add_argument("--no-regime", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "research"))
    ap.add_argument("--save-dump", help="write the probe rows to this JSON path (for reproducibility)")
    a = ap.parse_args(argv)
    t0 = time.time()
    rows = load_entries(a.from_json, a.probe)
    if a.save_dump:
        with open(a.save_dump, "w") as f:
            json.dump({"rows": rows}, f)
    ok = [r for r in rows if r.get("reason") == "ok"]
    os.makedirs(a.out, exist_ok=True)
    names = asset_names(sorted({r["u"] for r in ok}), os.path.join(a.out, "csp_sim_asset_names.json")) if a.exclude_leveraged else None
    flt = filter_entries(ok, a.max_abs_delta, names)
    entries = dedupe(flt["rows"], a.dedupe_days)
    import backtest_v2 as bt  # noqa: E402 (sys.path above)
    bars_by_u: Dict[str, dict] = {}
    for s in sorted({e["u"] for e in entries}):
        try:
            b = bt.fetch_bars(s, a.bars_days)
            if b and b.get("date"):
                bars_by_u[s] = b
        except Exception as e:
            print(f"[csp-sim] no bars for {s}: {type(e).__name__}: {e}", file=sys.stderr)
    regime_by_day = None
    if not a.no_regime:
        try:
            spy = bars_by_u.get("SPY") or bt.fetch_bars("SPY", a.bars_days)
            labels, _q = bt.regime_series(spy, bt.fetch_bars("VXX", a.bars_days))
            regime_by_day = dict(zip(spy["date"], labels))
        except Exception as e:
            print(f"[csp-sim] regime labels unavailable: {type(e).__name__}: {e}", file=sys.stderr)
    earn = earnings_dates(sorted(bars_by_u), os.path.join(a.out, "csp_sim_earnings_cache.json")) if a.earnings else None
    variants = [v for v in VARIANTS if v != "earnings_gate" or a.earnings]
    sims = {v: simulate_all(entries, bars_by_u, v, regime_by_day, earn) for v in variants}
    aggs = {v: aggregate(sims[v]) for v in variants}
    paired = paired_vs_live(sims)
    sleeves = None
    if a.sleeves_json:
        with open(a.sleeves_json) as f:
            sleeves = json.load(f)
    comparison = live_comparison(aggs, sleeves)
    date_s = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    meta = {"date": date_s, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "entry_rows": len(rows), "entry_ok": len(ok), "entry_days": len({r["day"] for r in rows}),
            "entry_underlyings": len({r["u"] for r in rows}), "deduped": len(entries), "dedupe_days": a.dedupe_days,
            "max_abs_delta": a.max_abs_delta, "cut_delta": flt["cut_delta"], "exclude_leveraged": a.exclude_leveraged,
            "cut_leveraged": flt["cut_leveraged"], "excluded_names": flt["excluded_names"],
            "bars_names": len(bars_by_u), "source": a.from_json or a.probe, "runtime_s": round(time.time() - t0, 1), "prior": PRIOR}
    md = os.path.join(a.out, f"csp_manager_rule_sim_{date_s}.md")
    with open(md, "w") as f:
        f.write(render(aggs, meta, comparison, paired))
    with open(os.path.join(a.out, f"csp_manager_rule_sim_{date_s}.json"), "w") as f:
        json.dump({"meta": meta, "aggregates": aggs, "paired": paired, "comparison": comparison}, f)
    print(f"wrote {md} — {verdict(aggs, paired)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
