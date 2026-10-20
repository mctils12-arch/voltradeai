#!/usr/bin/env python3
"""scripts/vrp_single_name_vs_index.py — research test 2, gate 2 (SIGNAL):
is there a variance risk premium (VRP) in the single names the bot sells
premium on, and in the indexes? ([RULE-REVIEW] measurement, 2026-10-20;
research/bot_logic_research_2026-10-20.md Q2.)

PRIOR (stated before the run): 85% that the index VRP (SPY, QQQ) is
positive — IV30² above the subsequently realized variance, i.e. premium
sellers were paid — and 65% that the pooled single-name VRP is
indistinguishable from zero (Driessen–Maenhout–Vilkov: zero individual VRP
not rejected for most S&P 500 names). If the prior holds, the single-name
short-premium sleeve has no gate-2 signal and may not be sized by a
VRP-harvest prior.

DATA: IV30 per underlying-day from the chain_iv30 diag probe
(server/chainIv.ts — derived from the options chain archive, indicative
feed, since 2026-07-04; the probe computes ≤15 uncached days per call and
reports `pending`, so the loader calls until none remain). Realized
variance from the NEXT `n` (21) trading days of daily closes
(backtest_v2.fetch_bars: Alpaca-first, Yahoo fallback, disk-cached).
VRP := IV30² − RV² in annualized variance units (+ = sellers paid), also as
vol points (IV30 − RV). Overlapping windows → Newey–West (Bartlett, lag
n−1) SE on every mean. Regime per day from backtest_v2.regime_series.

HONESTY: ~3.5 months of archive, mostly one regime, ≈ 3 independent
21-day windows per name — the TIME axis is the binding constraint and
pooling across names does not fix it (cross-sectional VRPs are correlated
through the market). Indicative quotes, not NBBO. No trading simulated.
The archive is forward-only and grows daily: rerun, never back-fill.

Run (pure): python3 scripts/vrp_single_name_vs_index.py --from-json dump.json
Live: DIAG_TOKEN in the env (never printed), --probe https://voltradeai.com
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
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

PRIOR = ("85% that the index VRP (SPY, QQQ) is positive (IV30² > subsequently realized "
         "variance: premium sellers paid); 65% that the pooled single-name VRP is "
         "indistinguishable from zero.")
INDEX_SYMS = ("SPY", "QQQ")
DEFAULT_N = 21
MIN_NAME_PAIRS = 5   # a per-name t-stat below this is noise (NW over 1–4 overlapping days)
TRADING_DAYS = 252


# ── loading ────────────────────────────────────────────────────────────────

def fetch_probe(base_url: str, token: str, max_calls: int = 20, sleep_s: float = 1.0) -> List[dict]:
    """All IV30 rows from the chain_iv30 probe, calling until nothing is pending.
    The token goes in a header only and is never logged."""
    rows: Dict[tuple, dict] = {}
    for _ in range(max_calls):
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/diag/chain_iv30?max=15", headers={"x-diag-token": token})
        with urllib.request.urlopen(req, timeout=180) as r:
            body = json.loads(r.read().decode("utf-8"))
        for row in body.get("rows", []):
            rows[(row["u"], row["day"])] = row
        if not body.get("pending"):
            break
        time.sleep(sleep_s)
    return sorted(rows.values(), key=lambda r: (r["u"], r["day"]))


def load_iv30(from_json: Optional[str], probe: Optional[str]) -> List[dict]:
    if from_json:
        with open(from_json) as f:
            d = json.load(f)
        return d["rows"] if isinstance(d, dict) else d
    token = os.environ.get("DIAG_TOKEN", "")
    if not probe or not token:
        raise SystemExit("need --from-json, or --probe with DIAG_TOKEN in the environment")
    return fetch_probe(probe, token)


# ── statistics ─────────────────────────────────────────────────────────────

def realized_var_ahead(closes: Sequence[float], i: int, n: int = DEFAULT_N) -> Optional[float]:
    """Annualized realized variance of the n daily log returns AFTER index i
    (closes[i] → closes[i+n]); zero-mean convention. None without n bars."""
    if i < 0 or i + n >= len(closes):
        return None
    s = 0.0
    for k in range(i, i + n):
        a, b = closes[k], closes[k + 1]
        if not (a and b) or a <= 0 or b <= 0:
            return None
        r = math.log(b / a)
        s += r * r
    return s * TRADING_DAYS / n


def newey_west_se(x: Sequence[float], lag: int) -> float:
    """HAC (Bartlett kernel) standard error of the mean of an overlapping series."""
    n = len(x)
    if n < 2:
        return float("nan")
    m = sum(x) / n
    d = [v - m for v in x]
    g0 = sum(v * v for v in d) / n
    var = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = sum(d[t] * d[t - k] for t in range(k, n)) / n
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return math.sqrt(max(var, 0.0) / n)


def pair(iv_rows: Iterable[dict], bars_by_u: Dict[str, dict], n: int = DEFAULT_N,
         regime_by_day: Optional[Dict[str, str]] = None) -> List[dict]:
    """Join each (u, day) IV30 with the variance realized over the next n
    trading days. Days without n forward bars (the last n archived days) are
    dropped — that is the unavoidable forward-window loss, stated in the report."""
    out: List[dict] = []
    idx_cache: Dict[str, Dict[str, int]] = {}
    for row in iv_rows:
        iv = row.get("iv30")
        if iv is None or not (iv > 0):
            continue
        u = row["u"]
        bars = bars_by_u.get(u)
        if not bars or not bars.get("date"):
            continue
        idx = idx_cache.setdefault(u, {d: i for i, d in enumerate(bars["date"])})
        i = idx.get(row["day"])
        if i is None:
            continue
        rv2 = realized_var_ahead(bars["close"], i, n)
        if rv2 is None:
            continue
        rv = math.sqrt(rv2)
        out.append({"u": u, "day": row["day"], "iv30": iv, "rv": round(rv, 4), "method": row.get("method"),
                    "vrp_var": round(iv * iv - rv2, 5), "vrp_vol": round(iv - rv, 4),
                    "regime": (regime_by_day or {}).get(row["day"])})
    return out


def _stats(vals: Sequence[float], lag: int, min_n: int = 2) -> dict:
    n = len(vals)
    if not n:
        return {"n": 0, "mean": None, "se": None, "t": None}
    m = sum(vals) / n
    se = newey_west_se(vals, lag) if n >= min_n else float("nan")
    return {"n": n, "mean": round(m, 5), "se": (round(se, 5) if se == se else None),
            "t": (round(m / se, 2) if se and se == se and se > 0 else None)}


def _dist(vals: Sequence[float]) -> dict:
    xs = sorted(vals)
    if not xs:
        return {"median": None, "p5": None, "p95": None}
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"median": round(xs[len(xs) // 2], 4), "p5": round(q(0.05), 4), "p95": round(q(0.95), 4)}


def aggregate(pairs: Sequence[dict], n: int = DEFAULT_N, index_syms: Sequence[str] = INDEX_SYMS) -> dict:
    """Per underlying, index vs pooled single names, and by regime."""
    lag = max(n - 1, 0)
    by_u: Dict[str, List[dict]] = defaultdict(list)
    for p in pairs:
        by_u[p["u"]].append(p)
    per_u = {}
    for u, ps in sorted(by_u.items()):
        ps.sort(key=lambda p: p["day"])
        per_u[u] = {
            "group": "index" if u in index_syms else "single",
            "vrp_var": _stats([p["vrp_var"] for p in ps], lag, MIN_NAME_PAIRS),
            "vrp_vol": _stats([p["vrp_vol"] for p in ps], lag, MIN_NAME_PAIRS),
            "share_iv_above_rv": round(sum(1 for p in ps if p["vrp_var"] > 0) / len(ps), 3),
            "mean_iv30": round(sum(p["iv30"] for p in ps) / len(ps), 4),
            "mean_rv": round(sum(p["rv"] for p in ps) / len(ps), 4),
            "first": ps[0]["day"], "last": ps[-1]["day"],
        }

    def pooled(ps: Sequence[dict]) -> dict:
        # pooled by DAY first (the cross-section is one observation per day), then NW over days
        by_day: Dict[str, List[dict]] = defaultdict(list)
        for p in ps:
            by_day[p["day"]].append(p)
        days = sorted(by_day)
        dv = [sum(x["vrp_var"] for x in by_day[d]) / len(by_day[d]) for d in days]
        dl = [sum(x["vrp_vol"] for x in by_day[d]) / len(by_day[d]) for d in days]
        return {"names": len({p["u"] for p in ps}), "pairs": len(ps), "days": len(days),
                "vrp_var": _stats(dv, lag), "vrp_vol": _stats(dl, lag),
                "dist_vol": _dist([p["vrp_vol"] for p in ps]), "dist_var": _dist([p["vrp_var"] for p in ps]),
                "share_iv_above_rv": (round(sum(1 for p in ps if p["vrp_var"] > 0) / len(ps), 3) if ps else None)}

    singles = [p for p in pairs if p["u"] not in index_syms]
    groups = {"single_pooled": pooled(singles)}
    for s in index_syms:
        groups[s] = pooled([p for p in pairs if p["u"] == s])
    by_regime = {}
    for reg in sorted({p["regime"] for p in pairs if p.get("regime")}):
        by_regime[reg] = {
            "single_pooled": pooled([p for p in singles if p.get("regime") == reg]),
            **{s: pooled([p for p in pairs if p["u"] == s and p.get("regime") == reg]) for s in index_syms},
        }
    return {"n_forward_days": n, "lag": lag, "pairs": len(pairs), "per_underlying": per_u, "groups": groups, "by_regime": by_regime}


# ── report ─────────────────────────────────────────────────────────────────

def verdict(agg: dict, index_syms: Sequence[str] = INDEX_SYMS) -> str:
    g = agg["groups"]
    idx_t = [g[s]["vrp_var"]["t"] for s in index_syms if g.get(s) and g[s]["vrp_var"]["t"] is not None]
    sgl_t = g["single_pooled"]["vrp_var"]["t"]
    parts = []
    if idx_t:
        parts.append("index VRP " + ("POSITIVE at |t|>2" if all(t is not None and t > 2 for t in idx_t) else "NOT significant at |t|>2") + f" (t = {idx_t})")
    else:
        parts.append("index VRP: NOT MEASURABLE — the chain archive holds no SPY/QQQ days (the daily underlying list is the CSP candidates, which rotate; the index anchors fall outside its cap — a pipeline gap, filed)")
    if sgl_t is None:
        parts.append("single-name VRP: no pairs")
    else:
        parts.append("single-name pooled VRP " + ("indistinguishable from zero (|t|<2)" if abs(sgl_t) < 2 else ("POSITIVE" if sgl_t > 0 else "NEGATIVE") + " at |t|>2") + f" (t = {sgl_t})")
    return "; ".join(parts) + ". Discount: ~3.5 months, overlapping windows, one regime dominant — gate-2 evidence accrues as the archive grows."


def _fmt(s: dict) -> str:
    if not s or s.get("n") in (None, 0):
        return "n 0"
    return f"n {s['n']} · mean {s['mean']:+.4f} · NW-SE {s['se'] if s['se'] is not None else 'nan'} · t {s['t'] if s['t'] is not None else '—'}"


def render(agg: dict, meta: dict) -> str:
    g = agg["groups"]
    L = [f"# Research test 2 (gate 2): single-name vs index variance risk premium — {meta['date']}", "",
         f"Generated {meta['generated_at']} by scripts/vrp_single_name_vs_index.py — IV30 rows {meta['iv_rows']} "
         f"({meta['iv_days']} archived days, {meta['iv_underlyings']} underlyings; source {meta['source']}), "
         f"forward window {agg['n_forward_days']} trading days, NW lag {agg['lag']}, pairs {agg['pairs']}.", "",
         "## PRIOR (stated before the run)", "", PRIOR, "",
         "## VERDICT", "", verdict(agg), "",
         "## Groups (VRP in annualized variance units, + = premium sellers paid; pooled per day, then Newey–West over days)", "",
         "| group | names | pairs | days | VRP (variance) | VRP (vol pts) | vol pts median [p5, p95] | variance median [p5, p95] | share IV30 > RV |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in g.items():
        dv, da = s.get("dist_vol", {}), s.get("dist_var", {})
        L.append(f"| {name} | {s['names']} | {s['pairs']} | {s['days']} | {_fmt(s['vrp_var'])} | {_fmt(s['vrp_vol'])} | {dv.get('median')} [{dv.get('p5')}, {dv.get('p95')}] | {da.get('median')} [{da.get('p5')}, {da.get('p95')}] | {s['share_iv_above_rv']} |")
    if agg["by_regime"]:
        L += ["", "## By regime", "", "| regime | group | pairs | days | VRP (variance) | VRP (vol pts) |", "|---|---|---|---|---|---|"]
        for reg, groups in agg["by_regime"].items():
            for name, s in groups.items():
                L.append(f"| {reg} | {name} | {s['pairs']} | {s['days']} | {_fmt(s['vrp_var'])} | {_fmt(s['vrp_vol'])} |")
    n_all = len(agg["per_underlying"])
    per = [(u, s) for u, s in agg["per_underlying"].items() if s["vrp_var"]["n"] >= MIN_NAME_PAIRS or s["group"] == "index"]
    L += ["", f"## Per underlying — the {len(per)} names with ≥ {MIN_NAME_PAIRS} pairs (of {n_all}; the archived universe rotates daily, so most names have 1–4 observations), sorted by VRP t-stat", "",
          "| u | group | n | mean IV30 | mean RV | VRP (vol pts) | t | share IV30 > RV | first → last |", "|---|---|---|---|---|---|---|---|---|"]
    per.sort(key=lambda kv: (kv[1]["vrp_var"]["t"] if kv[1]["vrp_var"]["t"] is not None else -99))
    for u, s in per:
        L.append(f"| {u} | {s['group']} | {s['vrp_var']['n']} | {s['mean_iv30']:.3f} | {s['mean_rv']:.3f} | {s['vrp_vol']['mean']:+.4f} | {s['vrp_var']['t'] if s['vrp_var']['t'] is not None else '—'} | {s['share_iv_above_rv']} | {s['first']} → {s['last']} |")
    L += ["", "## HONESTY", "",
          "- The archive began 2026-07-04 (indicative feed, not NBBO); the last 21 archived days have no forward window yet and are excluded — rerun as the archive grows.",
          "- Overlapping 21-day windows: every SE is Newey–West (Bartlett, lag 20); pooled single names are averaged per DAY first because cross-sectional VRPs share the market's variance shock.",
          "- IV30 is delta-ATM, total-variance interpolated (server/chainIv.ts); rows labeled `nearest` used one expiry inside 20–45 DTE.",
          "- Realized variance uses the zero-mean convention over close-to-close log returns (no overnight/intraday split).",
          "- SURVIVORSHIP / SELECTION: the underlyings are the CSP universe + positions on each archive day — names the scanner selected, typically on a high-IV day; a positive VRP here is partly the selection itself (IV mean-reverts after the day it is high). That is the population the bot sells premium on, so it is the right denominator for the bot, not for the market.",
          "- The archive holds NO index chains (SPY / QQQ / IWM never made the daily cap): the index control cannot be measured until the archiver pins them (filed).",
          ""]
    return "\n".join(L)


# ── main ───────────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-json", help="a saved probe dump ({rows: [...]}) instead of the live probe")
    ap.add_argument("--probe", help="base URL of the site (DIAG_TOKEN from the environment)")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="forward trading days for realized variance")
    ap.add_argument("--bars-days", type=int, default=400)
    ap.add_argument("--no-regime", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "research"))
    ap.add_argument("--save-dump", help="write the probe rows to this JSON path (for reproducibility)")
    a = ap.parse_args(argv)
    t0 = time.time()
    rows = load_iv30(a.from_json, a.probe)
    if a.save_dump:
        with open(a.save_dump, "w") as f:
            json.dump({"rows": rows}, f)
    import backtest_v2 as bt  # noqa: E402 (sys.path above)
    syms = sorted({r["u"] for r in rows})
    bars_by_u = {}
    for s in syms:
        try:
            b = bt.fetch_bars(s, a.bars_days)
            if b and b.get("date"):
                bars_by_u[s] = b
        except Exception as e:  # a name without bars is dropped, and said so
            print(f"[vrp] no bars for {s}: {type(e).__name__}: {e}", file=sys.stderr)
    regime_by_day = None
    if not a.no_regime:
        try:
            spy = bars_by_u.get("SPY") or bt.fetch_bars("SPY", a.bars_days)
            vxx = bt.fetch_bars("VXX", a.bars_days)
            labels, _q = bt.regime_series(spy, vxx)
            regime_by_day = dict(zip(spy["date"], labels))
        except Exception as e:
            print(f"[vrp] regime labels unavailable: {type(e).__name__}: {e}", file=sys.stderr)
    pairs = pair(rows, bars_by_u, a.n, regime_by_day)
    agg = aggregate(pairs, a.n)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    meta = {"date": date, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "iv_rows": len(rows), "iv_days": len({r["day"] for r in rows}), "iv_underlyings": len(syms),
            "source": a.from_json or a.probe, "runtime_s": round(time.time() - t0, 1), "prior": PRIOR}
    os.makedirs(a.out, exist_ok=True)
    md = os.path.join(a.out, f"vrp_single_name_vs_index_{date}.md")
    with open(md, "w") as f:
        f.write(render(agg, meta))
    with open(os.path.join(a.out, f"vrp_single_name_vs_index_{date}.json"), "w") as f:
        json.dump({"meta": meta, "aggregate": agg, "pairs": pairs}, f)
    print(f"wrote {md} — pairs {len(pairs)}; {verdict(agg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
