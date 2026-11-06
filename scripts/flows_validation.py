#!/usr/bin/env python3
"""
flows_validation.py — gate-2-style checks of the institutional flow model
(flows_model.py), on daily history. [RESEARCH], measurement only: no trading
imports beyond the model's own pure functions, no network.

Inputs: Yahoo chart-API JSON files (h_<SYM>.json, adjusted closes = total
return) in one directory: SPY, IEF, VFITX (bond proxy back to 1991), the
sector SPDRs and SMH. Usage:
    python3 scripts/flows_validation.py <dir> <out.json>

PRE-REGISTRATION (written 2026-11-06 before any test was run; the signs are the
papers' predictions, the samples and splits are fixed here):

T1a  Calendar rebalancing, month-end block. For each month end T: signal
     S = monthly drift + (quarterly drift at quarter ends) of a 60/40 SPY/bond
     portfolio reset at each month (quarter) end, measured at the close of T-4.
     Outcome: SPY log return close T-4 -> close T+1 (the sessions the model
     marks: T-3 .. T+1). Prediction: corr < 0 (Harvey, Mazzoleni & Melone).
     Bond = IEF (2002-08 on; the production proxy) and VFITX (1991-10 on).
     Splits: in-sample <= 2013-12, out-of-sample >= 2014-01, and the
     post-paper sample >= 2023-04 (their data end 2023-03).
T1b  Same signal, daily, their specification: every session t among the last
     5 of its month, next-session SPY return on S_t. Prediction: slope < 0.
     Newey-West (lag 5) t.
T2   Threshold signal (delta 0..2% averaged), every session, next-session SPY
     return. Prediction: slope < 0.
T3   Month-end cash windows (Etula et al. 2020), settlement-adjusted: per
     month, mean daily SPY return in the cash-raise window minus the mean in
     the reversal + new-month window. Prediction: < 0. Splits: the paper's
     1995-07..2013-12, then 2014-01 on (out of sample), and the T+1 era
     (2024-06 on).
T4   Tax-loss, sector ETFs: losers = the production rule (the three YTD-worst
     sectors with a YTD loss, weighted 1, 2/3, 1/3). Excess return vs the
     equal-weight sector average. Oct (ranked at the September close):
     prediction < 0; Nov (ranked at the October close): > 0. December
     (ranked at the November close, Dec 1-23): < 0; turn of year (ranked at
     Dec 23, Dec 24-Jan 7): > 0.
T5   Sector rotation, monthly: production rotation score (momentum z126/2
     minus 0.5 x crowding) at each month end vs the next month's sector log
     return minus the cross-sectional mean. Per-month Spearman IC, mean and t;
     and a tilted portfolio (weights 1 + 0.5 x score, normalized) minus equal
     weight. Prediction: IC > 0, tilt > 0. Components (momentum alone,
     crowding alone) reported too. Splits: <= 2012-12 / >= 2013-01.
T6   The shipped equity estimate (production weights): pressure at close t vs
     the SPY return over the next 5 sessions, sampled every 5th session
     (non-overlapping). Prediction: corr > 0 (negative pressure, lower
     returns). And the floor hook's effect on a buy-and-hold SPY position:
     exposure x the floor multiplier (cash otherwise, at 0%), 1 bp per unit
     of exposure traded, vs buy-and-hold.

Multiple testing: 6 families, several splits each. Treat any single |t| < 3
as weak; read the out-of-sample columns first.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import flows_model as fm  # noqa: E402

SECTOR_ETFS = [e for _, e in fm.SECTORS]

# The weights T6 was pre-registered with (before T1-T5 were run). The shipped
# defaults (flows_model.DEFAULTS) were set after reading T1-T5, so T6 under
# them is post hoc and reported as such.
PREREGISTERED_WEIGHTS = {"FLOWS_W_REBALANCE": 0.5, "FLOWS_W_THRESHOLD": 0.2, "FLOWS_W_MONTH_END": 0.3,
                         "FLOWS_REBAL_SCALE": 0.02, "FLOWS_THRESHOLD_SCALE": 0.004}


def load(path: str) -> Dict[str, float]:
    with open(path) as f:
        res = json.load(f)["chart"]["result"][0]
    ts = res["timestamp"]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    out = {}
    for t, v in zip(ts, adj):
        if v is None or not math.isfinite(v) or v <= 0:
            continue
        out[datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")] = float(v)
    return out


# ── statistics ────────────────────────────────────────────────────────────────

def mean(x):
    return sum(x) / len(x) if x else float("nan")


def sd(x):
    if len(x) < 2:
        return float("nan")
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def tstat_mean(x):
    s = sd(x)
    return mean(x) / (s / math.sqrt(len(x))) if x and s and s > 0 else float("nan")


def ols(x: Sequence[float], y: Sequence[float], nw_lag: int = 0) -> Dict:
    n = len(x)
    if n < 10:
        return {"n": n}
    mx, my = mean(x), mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx <= 0:
        return {"n": n}
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    a0 = my - b * mx
    e = [c - a0 - b * a for a, c in zip(x, y)]
    xd = [a - mx for a in x]
    s = sum((xd[i] * e[i]) ** 2 for i in range(n))
    for L in range(1, nw_lag + 1):
        w = 1 - L / (nw_lag + 1)
        s += 2 * w * sum(xd[i] * e[i] * xd[i - L] * e[i - L] for i in range(L, n))
    se = math.sqrt(s) / sxx
    sy = sd(y)
    r = b * math.sqrt(sxx / (n - 1)) / sy if sy and sy > 0 else float("nan")
    return {"n": n, "slope": b, "t": b / se if se > 0 else float("nan"), "corr": r}


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    if len(x) < 3:
        return float("nan")
    return pearson(ranks(list(x)), ranks(list(y)))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Plain correlation for any n >= 3 (ols() refuses n < 10, and a sector
    cross-section can have 9 names)."""
    if len(x) < 3:
        return float("nan")
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def hit_rate(signal, outcome):
    """Share of observations where the outcome's sign is the one predicted
    (outcome < 0 when signal > 0), next to the base rate of each sign."""
    pairs = [(s, y) for s, y in zip(signal, outcome) if s != 0]
    if not pairs:
        return {}
    hits = sum(1 for s, y in pairs if (s > 0 and y < 0) or (s < 0 and y > 0))
    neg = sum(1 for _, y in pairs if y < 0) / len(pairs)
    sig_pos = sum(1 for s, _ in pairs if s > 0) / len(pairs)
    # what a signal with no information would score: P(s>0)P(y<0) + P(s<0)P(y>0)
    base = sig_pos * neg + (1 - sig_pos) * (1 - neg)
    return {"hit": round(hits / len(pairs), 3), "no_skill": round(base, 3), "n": len(pairs)}


def split(rows: List[Tuple[str, float, float]], start: str, end: str):
    sel = [(x, y) for d, x, y in rows if start <= d <= end]
    return [a for a, _ in sel], [b for _, b in sel]


def summarize(rows, splits, nw_lag=0, predict_sign=-1):
    out = {}
    for name, (a, b) in splits.items():
        x, y = split(rows, a, b)
        o = ols(x, y, nw_lag)
        o.update(hit_rate(x, y) if predict_sign < 0 else hit_rate([-v for v in x], y))
        o["spearman"] = spearman(x, y) if len(x) >= 3 else float("nan")
        o["window"] = [a, b]
        out[name] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in o.items()}
    return out


# ── tests ─────────────────────────────────────────────────────────────────────

def aligned(spy: Dict[str, float], bond: Dict[str, float]):
    days = sorted(d for d in spy if d in bond)
    return days, [spy[d] for d in days], [bond[d] for d in days]


def month_end_index(days: List[str]) -> List[bool]:
    return [i + 1 < len(days) and days[i + 1][:7] != days[i][:7] for i in range(len(days))]


def k_to_month_end(days: List[str]) -> List[Optional[int]]:
    """sessions to the month end for every session (None for the last month, incomplete)."""
    out: List[Optional[int]] = [None] * len(days)
    me = month_end_index(days)
    nxt = None
    for i in range(len(days) - 1, -1, -1):
        if me[i]:
            nxt = i
        out[i] = None if nxt is None else nxt - i
    return out


def t1(spy, bond, warm=300):
    days, eq, bd = aligned(spy, bond)
    me = month_end_index(days)
    qe = [me[i] and int(days[i][5:7]) in (3, 6, 9, 12) for i in range(len(days))]
    m_path = fm.drift_path(eq, bd, me)
    q_path = fm.drift_path(eq, bd, qe)
    k = k_to_month_end(days)
    block, daily = [], []
    for i in range(warm, len(days) - 1):
        if k[i] is None:
            continue
        end = i + k[i]
        qflag = int(days[end][5:7]) in (3, 6, 9, 12)
        s = fm.rebalance_signal(m_path[i], q_path[i], qflag)
        if k[i] <= 4:
            daily.append((days[i + 1], s, eq[i + 1] / eq[i] - 1))
        if k[i] == 4 and end + 1 < len(days):
            block.append((days[end], s, math.log(eq[end + 1] / eq[i])))
    sig = [s for _, s, _ in block]
    return block, daily, {"signal_sd_pts": round(sd(sig) * 100, 3), "signal_mean_pts": round(mean(sig) * 100, 3),
                          "n_months": len(block), "start": days[warm]}


def t2(spy, bond, warm=300):
    days, eq, bd = aligned(spy, bond)
    th = fm.threshold_path(eq, bd)
    rows = [(days[i + 1], th[i], eq[i + 1] / eq[i] - 1) for i in range(warm, len(days) - 1)]
    return rows, {"signal_sd_pts": round(sd([r[1] for r in rows]) * 100, 4)}


def phase_by_index(days: List[str], i: int, k: Optional[int], j_month: int) -> Optional[str]:
    """month_end_phase on data-derived sessions (same rule as calendar_effects)."""
    s = fm.ce.settlement_days(date.fromisoformat(days[i]))
    last_sell = s + 1
    if k is not None and last_sell <= k < last_sell + fm.ce.CASH_SELL_SESSIONS:
        return "cash_sell"
    if k is not None and 0 <= k < last_sell:
        return "reversal"
    if j_month < fm.ce.NEW_MONTH_BUY_SESSIONS:
        return "new_month_buy"
    return None


def session_of_month(days):
    out, j = [], 0
    for i, d in enumerate(days):
        j = 0 if i == 0 or d[:7] != days[i - 1][:7] else j + 1
        out.append(j)
    return out


def t3(spy):
    days = sorted(spy)
    k = k_to_month_end(days)
    jm = session_of_month(days)
    per_month: Dict[str, Dict[str, List[float]]] = {}
    for i in range(1, len(days)):
        ph = phase_by_index(days, i, k[i], jm[i])
        if not ph:
            continue
        r = spy[days[i]] / spy[days[i - 1]] - 1
        # new-month sessions belong to the PREVIOUS month's cycle
        key = days[i][:7]
        if ph == "new_month_buy":
            y, m = int(key[:4]), int(key[5:7])
            key = f"{y - (m == 1)}-{(m - 2) % 12 + 1:02d}"
        per_month.setdefault(key, {"cash_sell": [], "after": []})["cash_sell" if ph == "cash_sell" else "after"].append(r)
    rows = []
    for key, v in sorted(per_month.items()):
        if v["cash_sell"] and v["after"]:
            rows.append((key, mean(v["cash_sell"]) - mean(v["after"])))
    out = {}
    for name, a, b in (("paper_1995_2013", "1995-07", "2013-12"), ("oos_2014_on", "2014-01", "2099-12"),
                       ("t_plus_1_era", "2024-06", "2099-12"), ("all", "1993-01", "2099-12")):
        x = [d for m, d in rows if a <= m <= b]
        out[name] = {"n": len(x), "mean_diff_bp_per_day": round(mean(x) * 1e4, 2) if x else None,
                     "t": round(tstat_mean(x), 2) if len(x) > 2 else None,
                     "share_negative": round(sum(1 for v in x if v < 0) / len(x), 3) if x else None}
    return out


def t4(series: Dict[str, Dict[str, float]]):
    spy_days = sorted(series["SPY"])
    years = sorted({d[:4] for d in spy_days})

    def last_on_or_before(day):
        cands = [d for d in spy_days if d <= day]
        return cands[-1] if cands else None

    def excess(rank_day, start_day, end_day, base_day, selling):
        avail = [e for e in SECTOR_ETFS if all(series.get(e, {}).get(x) for x in (base_day, rank_day, start_day, end_day))]
        if len(avail) < 6:
            return None
        ytd = {e: series[e][rank_day] / series[e][base_day] - 1 for e in avail}
        scores = fm.tax_loss_scores(ytd, selling=True)
        losers = {e: -v for e, v in scores.items() if v < 0}
        if not losers:
            return None
        ret = {e: series[e][end_day] / series[e][start_day] - 1 for e in avail}
        ew = mean(list(ret.values()))
        wsum = sum(losers.values())
        lr = sum(ret[e] * w for e, w in losers.items()) / wsum
        return lr - ew

    res = {"oct": [], "nov": [], "dec": [], "jan": []}
    for y in years:
        yi = int(y)
        base = last_on_or_before(f"{yi - 1}-12-31")
        sep = last_on_or_before(f"{y}-09-30")
        octe = last_on_or_before(f"{y}-10-31")
        nove = last_on_or_before(f"{y}-11-30")
        dec23 = last_on_or_before(f"{y}-12-23")
        jan7 = last_on_or_before(f"{yi + 1}-01-07")
        if not base or base[:4] != str(yi - 1):
            continue
        if sep and octe and octe > sep:
            v = excess(sep, sep, octe, base, True)
            if v is not None:
                res["oct"].append((y, v))
        if octe and nove and nove > octe:
            v = excess(octe, octe, nove, base, False)
            if v is not None:
                res["nov"].append((y, v))
        if nove and dec23 and dec23 > nove:
            v = excess(nove, nove, dec23, base, True)
            if v is not None:
                res["dec"].append((y, v))
        if dec23 and jan7 and jan7 > dec23 and jan7[:4] == str(yi + 1):
            v = excess(dec23, dec23, jan7, base, False)
            if v is not None:
                res["jan"].append((y, v))
    out = {}
    for k, rows in res.items():
        x = [v for _, v in rows]
        out[k] = {"n": len(x), "mean_excess_pct": round(mean(x) * 100, 3) if x else None,
                  "t": round(tstat_mean(x), 2) if len(x) > 2 else None,
                  "share_positive": round(sum(1 for v in x if v > 0) / len(x), 3) if x else None,
                  "prediction": "< 0" if k in ("oct", "dec") else "> 0",
                  "first_year": rows[0][0] if rows else None}
    return out


def t5(series: Dict[str, Dict[str, float]]):
    days = sorted(series["SPY"])
    me = month_end_index(days)
    ends = [i for i in range(len(days)) if me[i]]
    rows = {"rotation": [], "momentum": [], "crowd_flag": [], "tilt": []}
    for a, b in zip(ends, ends[1:]):
        if a < 260:
            continue
        sc, fut = {}, {}
        for e in SECTOR_ETFS:
            s = series.get(e) or {}
            st = fm.relative_stats(s, series["SPY"], days, a)
            rs = fm.rotation_score(st)
            if not rs or not s.get(days[b]) or not s.get(days[a]):
                continue
            sc[e] = rs
            fut[e] = math.log(s[days[b]] / s[days[a]])
        if len(sc) < 6:
            continue
        cm = mean(list(fut.values()))
        ex = {e: fut[e] - cm for e in fut}
        names = list(sc)
        rows["rotation"].append((days[a], spearman([sc[e]["rotation"] for e in names], [ex[e] for e in names])))
        rows["momentum"].append((days[a], spearman([sc[e]["momentum"] for e in names], [ex[e] for e in names])))
        crowded = [ex[e] for e in names if sc[e]["crowding"] > 0]
        if crowded:
            rows["crowd_flag"].append((days[a], mean(crowded)))
        w = {e: max(0.0, 1 + 0.5 * sc[e]["rotation"]) for e in names}
        ws = sum(w.values())
        tilt = sum(w[e] / ws * (math.exp(fut[e]) - 1) for e in names) - mean([math.exp(fut[e]) - 1 for e in names])
        rows["tilt"].append((days[a], tilt))
    out = {}
    for k, r in rows.items():
        for name, a, b in (("is_to_2012", "1999-01-01", "2012-12-31"), ("oos_2013_on", "2013-01-01", "2099-12-31"),
                           ("all", "1999-01-01", "2099-12-31")):
            x = [v for d, v in r if a <= d <= b and not math.isnan(v)]
            out[f"{k}.{name}"] = {"n": len(x), "mean": round(mean(x), 5) if x else None,
                                  "t": round(tstat_mean(x), 2) if len(x) > 2 else None}
    for name in ("is_to_2012", "oos_2013_on", "all"):
        c = out[f"tilt.{name}"]
        if c["mean"] is not None:
            c["annualized_pct"] = round(c["mean"] * 12 * 100, 3)
        cf = out[f"crowd_flag.{name}"]
        if cf["mean"] is not None:
            cf["note"] = "mean next-month excess log return of crowded sectors (prediction < 0)"
    return out


def t6(spy, bond, cfg=None, warm=300, horizon=5):
    """The shipped equity estimate on data-derived sessions, and the floor hook."""
    days, eq, bd = aligned(spy, bond)
    me = month_end_index(days)
    qe = [me[i] and int(days[i][5:7]) in (3, 6, 9, 12) for i in range(len(days))]
    m_path, q_path, th = fm.drift_path(eq, bd, me), fm.drift_path(eq, bd, qe), fm.threshold_path(eq, bd)
    k = k_to_month_end(days)
    jm = session_of_month(days)
    w_r, w_t, w_m = (float(fm.cfg_value(cfg, x)) for x in ("FLOWS_W_REBALANCE", "FLOWS_W_THRESHOLD", "FLOWS_W_MONTH_END"))
    rs, ts = float(fm.cfg_value(cfg, "FLOWS_REBAL_SCALE")), float(fm.cfg_value(cfg, "FLOWS_THRESHOLD_SCALE"))
    phase_val = {"cash_sell": -1.0, "reversal": 1.0, "new_month_buy": 1.0}
    press = [None] * len(days)
    for i in range(warm, len(days) - horizon):
        if k[i] is None or any(k[j] is None for j in range(i, i + horizon + 1)):
            continue
        reb = 0.0
        for j in range(horizon):
            t = i + j
            if k[t] <= 4:
                qflag = int(days[t + k[t]][5:7]) in (3, 6, 9, 12)
                reb += -fm._clip(fm.rebalance_signal(m_path[i], q_path[i], qflag) / rs)
        reb /= horizon
        thr = -fm._clip(th[i] / ts) / horizon
        mev = sum(phase_val.get(phase_by_index(days, i + 1 + j, k[i + 1 + j], jm[i + 1 + j]), 0.0)
                  for j in range(horizon)) / horizon
        press[i] = fm._clip(w_r * reb + w_t * thr + w_m * mev)
    rows = [(days[i + 1], press[i], math.log(eq[i + horizon] / eq[i]))
            for i in range(warm, len(days) - horizon, horizon) if press[i] is not None]
    corr = {}
    for name, a, b in (("is_to_2013", "1990-01-01", "2013-12-31"), ("oos_2014_on", "2014-01-01", "2099-12-31"),
                       ("post_paper_2023_04_on", "2023-04-01", "2099-12-31"), ("all", "1990-01-01", "2099-12-31")):
        x, y = split(rows, a, b)
        o = ols(x, y)
        o["spearman"] = spearman(x, y)
        o["window"] = [a, b]
        corr[name] = {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in o.items()}
    # the floor hook on a 1x SPY position
    fcfg = dict(cfg or {})
    overlay = {}
    for name, a, b in (("all", "1990-01-01", "2099-12-31"), ("oos_2014_on", "2014-01-01", "2099-12-31")):
        lw_bh = lw_ov = 0.0
        peak_bh = peak_ov = 0.0
        mdd_bh = mdd_ov = 0.0
        held, traded, n, cut_days = 1.0, 0.0, 0, 0
        for i in range(warm, len(days) - 1):
            if not (a <= days[i + 1] <= b):
                continue
            p = press[i]
            doc = {"ok": True, "asOf": days[i], "equity": {"pressure": p if p is not None else 0.0, "drivers": []}}
            m = fm.floor_multiplier(doc, fcfg, today=date.fromisoformat(days[i]))["multiplier"]
            cost = abs(m - held) * 1e-4
            traded += abs(m - held)
            held = m
            cut_days += m < 1.0
            r = eq[i + 1] / eq[i] - 1
            lw_bh += math.log(1 + r)
            lw_ov += math.log(1 + held * r - cost)
            peak_bh, peak_ov = max(peak_bh, lw_bh), max(peak_ov, lw_ov)
            mdd_bh, mdd_ov = min(mdd_bh, lw_bh - peak_bh), min(mdd_ov, lw_ov - peak_ov)
            n += 1
        if n:
            yrs = n / 252.0
            overlay[name] = {"sessions": n, "cut_share": round(cut_days / n, 3),
                             "bh_log_growth_pct_yr": round(lw_bh / yrs * 100, 3),
                             "overlay_log_growth_pct_yr": round(lw_ov / yrs * 100, 3),
                             "bh_max_dd_pct": round((math.exp(mdd_bh) - 1) * 100, 2),
                             "overlay_max_dd_pct": round((math.exp(mdd_ov) - 1) * 100, 2),
                             "turnover_per_yr": round(traded / yrs, 2),
                             "note": "cash earns 0 here (conservative); the live floor's cut goes to cash too"}
    pv = [p for p in press if p is not None]
    return {"corr_next5": corr, "floor_overlay": overlay,
            "pressure_distribution": {"n": len(pv), "share_sell": round(sum(1 for p in pv if p <= -0.15) / len(pv), 3),
                                      "share_buy": round(sum(1 for p in pv if p >= 0.15) / len(pv), 3),
                                      "p05": round(sorted(pv)[int(0.05 * len(pv))], 3),
                                      "p95": round(sorted(pv)[int(0.95 * len(pv))], 3)}}


def main(argv):
    src, out_path = argv[1], argv[2]
    series = {}
    for f in os.listdir(src):
        if f.startswith("h_") and f.endswith(".json"):
            series[f[2:-5]] = load(os.path.join(src, f))
    res = {"generated": datetime.now(timezone.utc).isoformat(), "inputs": {s: [min(v), max(v), len(v)] for s, v in series.items()}}
    splits_t1 = {"is_to_2013": ("1990-01-01", "2013-12-31"), "oos_2014_on": ("2014-01-01", "2099-12-31"),
                 "post_paper_2023_04_on": ("2023-04-01", "2099-12-31"), "all": ("1990-01-01", "2099-12-31")}
    for bond in ("IEF", "VFITX"):
        if bond not in series:
            continue
        block, daily, info = t1(series["SPY"], series[bond])
        res[f"T1a_block_{bond}"] = {"info": info, **summarize(block, splits_t1)}
        res[f"T1b_daily_{bond}"] = summarize(daily, splits_t1, nw_lag=5)
        rows, info2 = t2(series["SPY"], series[bond])
        res[f"T2_threshold_{bond}"] = {"info": info2, **summarize(rows, splits_t1, nw_lag=5)}
    res["T3_month_end_cash"] = t3(series["SPY"])
    res["T4_tax_loss"] = t4(series)
    res["T5_rotation"] = t5(series)
    for bond in ("IEF", "VFITX"):
        if bond in series:
            res[f"T6_preregistered_weights_{bond}"] = t6(series["SPY"], series[bond], PREREGISTERED_WEIGHTS)
            # the shipped defaults were set AFTER T1-T5 were read: post hoc, label it so
            res[f"T6_shipped_defaults_{bond}_posthoc"] = t6(series["SPY"], series[bond], None)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv)
