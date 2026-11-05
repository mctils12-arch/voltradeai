#!/usr/bin/env python3
"""
floor_study.py — the pre-registered floor study (2026-11-05).

Which passive core maximises long-run log wealth vs buy-and-hold SPY, after
financing and trading costs, with no lookahead? Variants, metrics and the ship
rule are fixed in the pre-registration (research/floor_study_2026-11-05.md,
§ Pre-registration); this script implements exactly those and nothing else.

Inputs: Yahoo chart-API JSON files (daily, adjusted close = total return) for
SPY QQQ SMH KWEB VXUS GLD and ^IRX (13-week T-bill yield, %). Usage:
    python3 floor_study.py <dir with h_SPY.json ...> <out.json>
Timing: every signal uses closes <= t and is held from close t to close t+1.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

TRADING_DAYS = 252
COST_BP = 2.0                 # per unit of exposure traded
FIN_SPREAD = 0.01             # financing above the T-bill rate, per year
LEV_ETF_FEE = 0.009           # 2x ETF expense ratio, per year (V2)
V1_TARGET, V1_CAP, V1_BAND, V1_WIN = 0.16, 1.5, 0.10, 20
V2_LEV, V2_SMA = 2.0, 200
V3_LEV = 1.25
C0_WEIGHTS = {"QQQ": 0.45, "SMH": 0.135, "KWEB": 0.09, "VXUS": 0.135, "GLD": 0.08}   # + 0.11 cash
SUBPERIODS = (("1993-01-01", "2007-12-31"), ("2008-01-01", "2016-12-31"), ("2017-01-01", "2026-12-31"))


def load(path: str, adjusted: bool = True) -> Dict[str, float]:
    d = json.load(open(path))["chart"]["result"][0]
    ts = d["timestamp"]
    ind = d["indicators"]
    vals = ind["adjclose"][0]["adjclose"] if adjusted and "adjclose" in ind else ind["quote"][0]["close"]
    out: Dict[str, float] = {}
    for t, v in zip(ts, vals):
        if v is None or not math.isfinite(v):
            continue
        day = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        out[day] = float(v)
    return out


def daily_rf(irx: Dict[str, float], days: Sequence[str]) -> List[float]:
    """T-bill yield (annual %) -> a daily simple rate, forward-filled onto `days`."""
    out, last = [], 0.0
    for d in days:
        if d in irx and irx[d] >= 0:
            last = irx[d]
        out.append(last / 100.0 / TRADING_DAYS)
    return out


def returns(px: Sequence[float]) -> List[float]:
    return [px[i] / px[i - 1] - 1.0 for i in range(1, len(px))]


def lever_ret(w: float, r: float, rf: float) -> float:
    """One day's return at exposure w to an asset returning r: cash earns rf, borrowing pays rf + spread."""
    if w <= 1.0:
        return w * r + (1.0 - w) * rf
    return w * r - (w - 1.0) * (rf + FIN_SPREAD / TRADING_DAYS)


def run_b0(r: Sequence[float]) -> List[float]:
    return list(r)


def run_v1(r: Sequence[float], rf: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Vol-managed: w = min(cap, target / sigma_20d), decided at close t, held t -> t+1; band rebalancing."""
    out, ws = [], []
    held = 1.0
    for i in range(len(r)):
        # decision at close i-1 uses returns up to index i-1 (r[i-20..i-1]); held over r[i]
        if i >= V1_WIN:
            window = r[i - V1_WIN:i]
            mu = sum(window) / V1_WIN
            sd = math.sqrt(sum((x - mu) ** 2 for x in window) / (V1_WIN - 1)) * math.sqrt(TRADING_DAYS)
            target = min(V1_CAP, V1_TARGET / sd) if sd > 0 else V1_CAP
        else:
            target = 1.0
        cost = 0.0
        if abs(target - held) > V1_BAND:
            cost = abs(target - held) * COST_BP / 1e4
            held = target
        out.append(lever_ret(held, r[i], rf[i]) - cost)
        ws.append(held)
    return out, ws


def run_v2(px: Sequence[float], r: Sequence[float], rf: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Leverage rotation: 2x (synthetic, fee + financing) when close_t > SMA200_t, else T-bills."""
    out, ws = [], []
    held = 0.0
    for i in range(len(r)):
        # r[i] is px[i] -> px[i+1]; the signal is at close i (px[0..i])
        on = False
        if i + 1 > V2_SMA:
            sma = sum(px[i + 1 - V2_SMA:i + 1]) / V2_SMA
            on = px[i] > sma
        target = V2_LEV if on else 0.0
        cost = abs(target - held) * COST_BP / 1e4
        held = target
        if held > 0:
            day = held * r[i] - (held - 1.0) * (rf[i] + FIN_SPREAD / TRADING_DAYS) - LEV_ETF_FEE / TRADING_DAYS
        else:
            day = rf[i]
        out.append(day - cost)
        ws.append(held)
    return out, ws


def run_v3(r: Sequence[float], rf: Sequence[float]) -> List[float]:
    out = []
    for i in range(len(r)):
        drift = abs(V3_LEV * (V3_LEV - 1.0) * r[i])       # rebalance back to 1.25x after the day's move
        out.append(lever_ret(V3_LEV, r[i], rf[i]) - drift * COST_BP / 1e4)
    return out


def run_c0(days: Sequence[str], series: Dict[str, Dict[str, float]], rf: Sequence[float]) -> List[float]:
    """Static NEUTRAL-regime basket, monthly rebalanced (first session of each month)."""
    syms = list(C0_WEIGHTS)
    w = dict(C0_WEIGHTS)
    cash_w = 1.0 - sum(w.values())
    out = []
    for i in range(1, len(days)):
        d0, d1 = days[i - 1], days[i]
        rets = {s: series[s][d1] / series[s][d0] - 1.0 for s in syms}
        port = sum(w[s] * rets[s] for s in syms) + cash_w * rf[i]
        cost = 0.0
        # drift the weights
        tot = 1.0 + port
        w = {s: w[s] * (1.0 + rets[s]) / tot for s in syms}
        cash_w = cash_w * (1.0 + rf[i]) / tot
        if i + 1 < len(days) and days[i + 1][:7] != d1[:7]:      # last session of the month -> rebalance at its close
            turnover = sum(abs(w[s] - C0_WEIGHTS[s]) for s in syms) + abs(cash_w - (1.0 - sum(C0_WEIGHTS.values())))
            cost = turnover * COST_BP / 1e4
            w = dict(C0_WEIGHTS)
            cash_w = 1.0 - sum(w.values())
        out.append(port - cost)
    return out


def metrics(rets: Sequence[float], rf: Sequence[float]) -> Dict[str, float]:
    n = len(rets)
    if n < 2:
        return {}
    logs = [math.log1p(x) for x in rets]
    wealth, peak, mdd = 1.0, 1.0, 0.0
    for x in rets:
        wealth *= 1.0 + x
        peak = max(peak, wealth)
        mdd = min(mdd, wealth / peak - 1.0)
    ex = [rets[i] - rf[i] for i in range(n)]
    mu = sum(ex) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in ex) / (n - 1))
    vol = math.sqrt(sum((x - sum(rets) / n) ** 2 for x in rets) / (n - 1)) * math.sqrt(TRADING_DAYS)
    years = n / TRADING_DAYS
    return {
        "cagr_pct": round((wealth ** (1 / years) - 1) * 100, 2),
        "vol_pct": round(vol * 100, 2),
        "sharpe": round(mu / sd * math.sqrt(TRADING_DAYS), 3) if sd > 0 else 0.0,
        "max_dd_pct": round(mdd * 100, 2),
        "log_growth_pct": round(sum(logs) / n * TRADING_DAYS * 100, 2),
        "years": round(years, 2),
    }


def worst_year(days: Sequence[str], rets: Sequence[float]) -> Tuple[str, float]:
    by: Dict[str, float] = {}
    for d, x in zip(days, rets):
        by[d[:4]] = by.get(d[:4], 1.0) * (1.0 + x)
    y = min(by, key=lambda k: by[k])
    return y, round((by[y] - 1) * 100, 2)


def rolling_beat(rets: Sequence[float], bench: Sequence[float], win: int = 3 * TRADING_DAYS) -> Optional[float]:
    la = [math.log1p(x) for x in rets]
    lb = [math.log1p(x) for x in bench]
    if len(la) <= win:
        return None
    ca, cb = [0.0], [0.0]
    for a, b in zip(la, lb):
        ca.append(ca[-1] + a)
        cb.append(cb[-1] + b)
    beats = sum(1 for i in range(win, len(la) + 1) if ca[i] - ca[i - win] > cb[i] - cb[i - win])
    return round(beats / (len(la) - win + 1) * 100, 1)


def sliced(days, rets, rf, lo, hi):
    idx = [i for i, d in enumerate(days) if lo <= d <= hi]
    return [rets[i] for i in idx], [rf[i] for i in idx]


def main(src: str, out_path: str) -> None:
    spy_px = load(f"{src}/h_SPY.json")
    irx = load(f"{src}/h_IRX.json", adjusted=False)
    days = sorted(d for d in spy_px if d <= "2026-11-04")
    px = [spy_px[d] for d in days]
    r = returns(px)
    rdays = days[1:]                      # r[i] is the return INTO rdays[i]
    rf = daily_rf(irx, rdays)

    results: Dict[str, Dict] = {}
    b0 = run_b0(r)
    v1, w1 = run_v1(r, rf)
    v2, w2 = run_v2(px, r, rf)
    v3 = run_v3(r, rf)
    series = {"B0_SPY": b0, "V1_volmanaged": v1, "V2_leverage_rotation": v2, "V3_constant_1_25x": v3}

    qqq_px = load(f"{src}/h_QQQ.json")
    qdays = [d for d in rdays if d in qqq_px and days[days.index(d) - 1] in qqq_px]

    def full(name, rets_):
        m = metrics(rets_, rf)
        m["worst_year"] = worst_year(rdays, rets_)
        m["rolling3y_beat_spy_pct"] = rolling_beat(rets_, b0)
        m["subperiods"] = {}
        for lo, hi in SUBPERIODS:
            a, f = sliced(rdays, rets_, rf, lo, hi)
            b, _ = sliced(rdays, b0, rf, lo, hi)
            ma, mb = metrics(a, f), metrics(b, f)
            m["subperiods"][f"{lo[:4]}-{hi[:4]}"] = {"self": ma, "spy": mb,
                                                     "beats_spy_log_growth": ma.get("log_growth_pct", 0) > mb.get("log_growth_pct", 0)}
        # trend-state split: SPY above/below its 200d SMA at the prior close
        up, dn = [], []
        for i in range(len(r)):
            if i + 1 > V2_SMA:
                sma = sum(px[i + 1 - V2_SMA:i + 1]) / V2_SMA
                (up if px[i] > sma else dn).append(i)
        for label, idx in (("spy_above_200d", up), ("spy_below_200d", dn)):
            m[label] = metrics([rets_[i] for i in idx], [rf[i] for i in idx])
        results[name] = m

    for name, rets_ in series.items():
        full(name, rets_)

    # QQQ buy-and-hold on its own window, with SPY over the SAME days
    q_idx = [i for i, d in enumerate(rdays) if d in qqq_px and i > 0 and rdays[i - 1] in qqq_px]
    q_r = [qqq_px[rdays[i]] / qqq_px[rdays[i - 1]] - 1.0 for i in q_idx]
    q_rf = [rf[i] for i in q_idx]
    q_spy = [b0[i] for i in q_idx]
    results["B1_QQQ"] = {"self": metrics(q_r, q_rf), "spy_same_window": metrics(q_spy, q_rf),
                         "from": rdays[q_idx[0]], "rolling3y_beat_spy_pct": rolling_beat(q_r, q_spy)}

    # current-floor basket, static, on the window where every member exists
    basket = {s: load(f"{src}/h_{s}.json") for s in C0_WEIGHTS}
    cdays = [d for d in days if all(d in basket[s] for s in basket)]
    c_rets = run_c0(cdays, basket, daily_rf(irx, cdays))
    c_rf = daily_rf(irx, cdays)[1:]
    spy_c = [spy_px[cdays[i]] / spy_px[cdays[i - 1]] - 1.0 for i in range(1, len(cdays))]
    v1_c = [v1[rdays.index(d)] for d in cdays[1:]]
    v2_c = [v2[rdays.index(d)] for d in cdays[1:]]
    v3_c = [v3[rdays.index(d)] for d in cdays[1:]]
    q_c = [qqq_px[cdays[i]] / qqq_px[cdays[i - 1]] - 1.0 for i in range(1, len(cdays))]
    results["C0_current_basket_static"] = {
        "from": cdays[0],
        "self": metrics(c_rets, c_rf), "spy_same_window": metrics(spy_c, c_rf), "qqq_same_window": metrics(q_c, c_rf),
        "v1_same_window": metrics(v1_c, c_rf), "v2_same_window": metrics(v2_c, c_rf), "v3_same_window": metrics(v3_c, c_rf),
        "rolling3y_beat_spy_pct": rolling_beat(c_rets, spy_c),
    }

    # exposure stats
    results["exposure"] = {
        "V1_mean_w": round(sum(w1) / len(w1), 3), "V1_share_levered_pct": round(sum(1 for w in w1 if w > 1.0) / len(w1) * 100, 1),
        "V1_rebalances": sum(1 for i in range(1, len(w1)) if w1[i] != w1[i - 1]),
        "V2_share_on_pct": round(sum(1 for w in w2 if w > 0) / len(w2) * 100, 1),
        "V2_switches": sum(1 for i in range(1, len(w2)) if w2[i] != w2[i - 1]),
    }

    # pre-registered ship rule
    spy_m = results["B0_SPY"]
    verdict = {}
    for name in ("V1_volmanaged", "V2_leverage_rotation", "V3_constant_1_25x"):
        m = results[name]
        wins = sum(1 for v in m["subperiods"].values() if v["beats_spy_log_growth"])
        checks = {
            "log_growth_above_spy": m["log_growth_pct"] > spy_m["log_growth_pct"],
            "sharpe_ge_spy": m["sharpe"] >= spy_m["sharpe"],
            "max_dd_within_10pp": m["max_dd_pct"] >= spy_m["max_dd_pct"] - 10.0,
            "beats_spy_2_of_3_subperiods": wins >= 2,
        }
        verdict[name] = {"checks": checks, "passes": all(checks.values())}
    c = results["C0_current_basket_static"]
    verdict["C0_current_basket_static"] = {"checks": {
        "log_growth_above_spy": c["self"]["log_growth_pct"] > c["spy_same_window"]["log_growth_pct"],
        "sharpe_ge_spy": c["self"]["sharpe"] >= c["spy_same_window"]["sharpe"],
        "max_dd_within_10pp": c["self"]["max_dd_pct"] >= c["spy_same_window"]["max_dd_pct"] - 10.0,
    }, "note": "2013+ window only; no subperiod test possible"}
    passing = [k for k, v in verdict.items() if v.get("passes")]
    results["verdict"] = verdict
    results["chosen"] = max(passing, key=lambda k: results[k]["log_growth_pct"]) if passing else None
    results["window"] = {"from": rdays[0], "to": rdays[-1], "sessions": len(rdays)}
    json.dump(results, open(out_path, "w"), indent=1)
    print(json.dumps({k: (v if k in ("verdict", "chosen", "window", "exposure") else
                          {kk: vv for kk, vv in (v.items() if isinstance(v, dict) else []) if kk not in ("subperiods",)})
                      for k, v in results.items()}, indent=1)[:9000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
