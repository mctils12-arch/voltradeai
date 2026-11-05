#!/usr/bin/env python3
"""scripts/forecast_direction_study.py — Part C (human-approved batch,
2026-11-04): does any candidate feature beat the point-in-time BASE RATE for
the daily forecast's direction call, on a purged walk-forward holdout, after
a multiple-testing correction? Research only; nothing here runs in production.

PRE-REGISTERED before any result (copied verbatim into
research/forecast_direction_study_2026-11-04.md); summary:

  targets  PRIMARY y = 1[C_{t+1} > C_t] (decision at the t close — the scored
           daily call; flat = not up, as the weekly scorer counts it).
           INTRADAY y = 1[C_{t+1} > O_{t+1}] (rest of day from the open) and
           y = 1[C_D > P_10:30] (rest of day from the first hour's close).
  features F1 gap_t = log(O_t/C_{t-1}) / sd20      F2 imom_t = log(C_t/O_t) / sd20
           F3 dvix_t = log(VIX_t/VIX_{t-1}) / its trailing-252 std
           F4 breadth = 2 * (share of regime_engine.BREADTH_BASKET above its
              200-day SMA - 0.5)            F5 gex = 2 * (SqueezeMetrics GEX
              trailing-252 percentile rank - 0.5), P1's definition
           G1 ovn gap log(O_{t+1}/C_t)/sd20 -> rest of day from the open
           G2 first hour log(P_10:30/O_D)/sd20 -> rest of day from 10:30
           (sd20 = the name's 20-session daily vol at t; all clipped at +/-4)
  model    ridge logistic (forecast.fit_logistic, l2 = 1) on ONE feature vs
           the intercept-only model (= the training up-frequency, the
           point-in-time base rate); walk-forward, expanding, refit every 21
           sessions, 5-session embargo, >= 504 training sessions (G2: the
           60-minute sample is ~725 sessions, so G2 trains on its first half
           and is scored on the second — a deviation forced by data length,
           logged before G2 was run)
  segments SELECTION origins .. 2018-12-31, HOLDOUT 2019-01-02 .. 2026-11-02
  universe SPY alone; pooled panel = 15 TRAIN_BASKET ETFs + 20 names
  tests    m = 14 (T1-T5 F1..F5 SPY; T6-T10 panel; T11 combination of the
           individual passes, counted whether or not it runs; T12 G1 SPY;
           T13 G1 panel; T14 G2)
  PASS     holdout dBrier < 0, one-sided day-block-bootstrap p < 0.05/14
           (20-session blocks of whole days, 10,000 resamples), and dBrier < 0
           on SELECTION too

DATA: the range study's folders (scripts/forecast_daily_study.py fetch) plus
SqueezeMetrics DIX.csv for F5 (research use only; licence unverified —
NEVER commit or redistribute it; same posture as scripts/edge_p1_dealer_gamma.py).

Run:  python3 scripts/forecast_direction_study.py --data-dir DIR --dix PATH/DIX.csv \
          [--md research/forecast_direction_study_2026-11-04.md] [--json ...json]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.append(HERE)   # appended: scripts/source_validation.py must never shadow the root module

import forecast_daily_study as RS  # noqa: E402  (shared loader + bootstrap)
from forecast import fit_logistic  # noqa: E402

# ── pre-registered constants (do not tune) ──────────────────────────────────
SELECTION_END = "2018-12-31"
HOLDOUT = ("2019-01-02", "2026-11-02")
REFIT_EVERY = 21
EMBARGO = 5
MIN_TRAIN = 504
CLIP = 4.0
BLOCK = 20
N_BOOT = 10_000
SEED = 20261104
M_TESTS = 14
ALPHA_FAMILY = 0.05
P_BAR = ALPHA_FAMILY / M_TESTS
PANEL = RS.TRAIN_BASKET + RS.STOCKS
BREADTH_BASKET = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLC", "XLRE", "IWM", "MDY")
BREADTH_MIN_MEMBERS = 6
RANK_WINDOW = 252


# ── features (pure; point-in-time) ──────────────────────────────────────────
def sd20(close: np.ndarray) -> np.ndarray:
    """Sample std of the 20 daily log returns ending at t (closes t-20..t)."""
    n = len(close)
    r = np.full(n, np.nan)
    r[1:] = np.log(close[1:] / close[:-1])
    out = np.full(n, np.nan)
    for t in range(20, n):
        w = r[t - 19:t + 1]
        if np.all(np.isfinite(w)):
            s = float(np.std(w, ddof=1))
            out[t] = s if s > 0 else np.nan
    return out


def clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -CLIP, CLIP)


def name_features(O: np.ndarray, C: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-name arrays over origins t. F1/F2 known at the t close; G1's
    feature (the NEXT session's gap) is stored at t but is only known at the
    t+1 open — it is used only against the rest-of-day target."""
    n = len(C)
    s = sd20(C)
    gap = np.full(n, np.nan)
    gap[1:] = np.log(O[1:] / C[:-1])
    imom = np.log(C / O)
    nxt_gap = np.full(n, np.nan)
    nxt_gap[:-1] = np.log(O[1:] / C[:-1])
    y = np.full(n, np.nan)
    y[:-1] = (C[1:] > C[:-1]).astype(float)
    y[:-1][~(np.isfinite(C[1:]) & np.isfinite(C[:-1]))] = np.nan
    y_rod = np.full(n, np.nan)            # rest of day from the t+1 open
    y_rod[:-1] = (C[1:] > O[1:]).astype(float)
    y_rod[:-1][~(np.isfinite(C[1:]) & np.isfinite(O[1:]))] = np.nan
    return {"F1": clip(gap / s), "F2": clip(imom / s), "G1": clip(nxt_gap / s), "y": y, "y_rod": y_rod, "sd20": s}


def dvix_feature(vix: np.ndarray) -> np.ndarray:
    n = len(vix)
    d = np.full(n, np.nan)
    d[1:] = np.log(vix[1:] / vix[:-1])
    out = np.full(n, np.nan)
    for t in range(253, n):
        w = d[t - 251:t + 1]
        w = w[np.isfinite(w)]
        if len(w) >= 200 and np.isfinite(d[t]):
            sd = float(np.std(w, ddof=1))
            if sd > 0:
                out[t] = d[t] / sd
    return clip(out)


def breadth_feature(closes: Dict[str, np.ndarray]) -> np.ndarray:
    """2 * (share of the basket above its 200-day SMA - 0.5): regime_engine's
    breadth (members with a full 200-session window; None under 6 members)."""
    n = len(next(iter(closes.values())))
    out = np.full(n, np.nan)
    for t in range(199, n):
        up = tot = 0
        for c in closes.values():
            w = c[t - 199:t + 1]
            if not np.all(np.isfinite(w)):
                continue
            tot += 1
            up += int(c[t] > float(np.mean(w)))
        if tot >= BREADTH_MIN_MEMBERS:
            out[t] = 2.0 * (up / tot - 0.5)
    return out


def trailing_pct_rank(values: Sequence[float], window: int = RANK_WINDOW) -> List[Optional[float]]:
    """Identical to scripts/edge_p1_dealer_gamma.trailing_pct_rank (P1)."""
    out: List[Optional[float]] = []
    for t in range(len(values)):
        if t < window - 1:
            out.append(None)
            continue
        w = values[t - window + 1:t + 1]
        v = values[t]
        below = sum(1 for x in w if x < v)
        ties = sum(1 for x in w if x == v) - 1
        out.append((below + 0.5 * ties) / (window - 1))
    return out


def gex_feature(dix_path: Optional[str], dates: Sequence[str]) -> np.ndarray:
    out = np.full(len(dates), np.nan)
    if not dix_path or not os.path.exists(dix_path):
        return out
    rows = []
    with open(dix_path, newline="") as fh:
        for r in csv.DictReader(fh):
            g = RS._num(r.get("gex") or "")
            if g is not None and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(r.get("date") or "")):
                rows.append((r["date"], g))
    rows.sort()
    ranks = trailing_pct_rank([g for _, g in rows])
    by = {d: rk for (d, _), rk in zip(rows, ranks) if rk is not None}
    for k, d in enumerate(dates):
        if d in by:
            out[k] = 2.0 * (by[d] - 0.5)
    return out


# ── walk-forward logistic vs base rate ──────────────────────────────────────
def walk_forward(X: Dict[str, np.ndarray], Y: Dict[str, np.ndarray], names: Sequence[str],
                 first_eval: int, last_eval: int, n_dates: int, cols: Sequence[str],
                 min_train: int = MIN_TRAIN, embargo: int = EMBARGO) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[dict]]:
    """Pooled fit over `names`. Returns (p_model, p_base, fits) as arrays over
    origins. A refit at master index `at` trains on origins t <= at - 1 -
    embargo (labels t+1 <= at - embargo) and forecasts origins [at, at+21)."""
    rows_t, rows_s, rows_x, rows_y = [], [], [], []
    for k, s in enumerate(names):
        feats = [X[c][s] for c in cols]
        ok = np.isfinite(Y[s])
        for f in feats:
            ok &= np.isfinite(f)
        t = np.nonzero(ok)[0]
        rows_t.append(t)
        rows_s.append(np.full(len(t), k))
        rows_x.append(np.column_stack([f[t] for f in feats]) if feats else np.zeros((len(t), 0)))
        rows_y.append(Y[s][t])
    T = np.concatenate(rows_t)
    Xa = np.vstack(rows_x)
    Ya = np.concatenate(rows_y)
    pm = {s: np.full(n_dates, np.nan) for s in names}
    pb = {s: np.full(n_dates, np.nan) for s in names}
    fits = []
    # training sessions: the number of distinct origin dates before the cutoff
    for at in range(first_eval, last_eval + 1, REFIT_EVERY):
        upto = min(at + REFIT_EVERY, last_eval + 1)
        tr = T <= at - 1 - embargo
        if len(np.unique(T[tr])) < min_train:
            continue
        base = float(np.mean(Ya[tr]))
        coef = fit_logistic(Xa[tr].tolist(), Ya[tr].astype(int).tolist()) if cols else [math.log(base / (1 - base))]
        fits.append({"refit_at": at, "coef": coef, "base": base, "n_train": int(tr.sum())})
        for k, s in enumerate(names):
            seg = np.arange(at, upto)
            pb[s][seg] = base
            if cols:
                F = np.column_stack([X[c][s][seg] for c in cols])
                good = np.all(np.isfinite(F), axis=1)
                z = coef[0] + F @ np.asarray(coef[1:])
                p = np.where(good, 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))), np.nan)
                pm[s][seg] = p
            else:
                pm[s][seg] = base
    return pm, pb, fits


def per_day_loss(pm: Dict[str, np.ndarray], pb: Dict[str, np.ndarray], Y: Dict[str, np.ndarray],
                 names: Sequence[str], lo: int, hi: int, mask: Optional[np.ndarray] = None) -> dict:
    D = hi - lo + 1
    dsum = np.zeros(D)
    bm = np.zeros(D)
    bb = np.zeros(D)
    hits = np.zeros(D)
    ups = np.zeros(D)
    cnt = np.zeros(D)
    for s in names:
        p, b, y = pm[s][lo:hi + 1], pb[s][lo:hi + 1], Y[s][lo:hi + 1]
        ok = np.isfinite(p) & np.isfinite(b) & np.isfinite(y)
        if mask is not None:
            ok &= mask
        lm = np.where(ok, (p - y) ** 2, 0.0)
        lb = np.where(ok, (b - y) ** 2, 0.0)
        dsum += lm - lb
        bm += lm
        bb += lb
        hits += np.where(ok, ((p >= 0.5) == (y == 1)).astype(float), 0.0)
        ups += np.where(ok, y, 0.0)
        cnt += ok.astype(float)
    n = cnt.sum()
    return {"n": int(n), "days": int((cnt > 0).sum()), "d_sum": dsum, "cnt": cnt,
            "brier_model": float(bm.sum() / n) if n else float("nan"),
            "brier_base": float(bb.sum() / n) if n else float("nan"),
            "d_brier": float(dsum.sum() / n) if n else float("nan"),
            "hit_rate": float(hits.sum() / n) if n else float("nan"),
            "up_rate": float(ups.sum() / n) if n else float("nan")}


def bootstrap_mean(d_sum: np.ndarray, cnt: np.ndarray, block: int = BLOCK, n_boot: int = N_BOOT,
                   seed: int = SEED) -> Tuple[float, float, float]:
    """(lo95, hi95, one-sided p = P*(mean >= 0)) of sum(d)/sum(cnt) under the
    moving day-block bootstrap."""
    keep = cnt > 0
    d, c = d_sum[keep], cnt[keep]
    D = len(d)
    if D == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(D / block))
    smax = max(1, D - block + 1)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, smax, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:D]
        stats[b] = d[ix].sum() / c[ix].sum()
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)), float(np.mean(stats >= 0))


def verdict(hold: dict, sel: Optional[dict], p: float) -> str:
    if not np.isfinite(hold["d_brier"]):
        return "NO DATA"
    if hold["d_brier"] < 0 and p < P_BAR and sel is not None and sel["d_brier"] < 0:
        return "PASS"
    return "FAIL"


# ── 60-minute first-hour momentum (G2) ──────────────────────────────────────
def first_hour_rows(P: "RS.Panel", data_dir: str, names: Sequence[str], sd: Dict[str, np.ndarray]):
    """Per name, arrays over master dates: feature = log(P_10:30/O_D)/sd20_{D-1}
    and target 1[C_D > P_10:30] (official daily close)."""
    n = len(P.dates)
    X, Y = {}, {}
    for s in names:
        x, y = np.full(n, np.nan), np.full(n, np.nan)
        for o in RS.intraday_obs(P, data_dir, s):
            k = o["k"]
            sdv = sd[s][k - 1] if s in sd else np.nan
            if not np.isfinite(sdv):
                continue
            p_open, p_1030 = o["s"][0], o["s"][1]
            x[k] = float(np.clip(math.log(p_1030 / p_open) / sdv, -CLIP, CLIP))
            y[k] = 1.0 if o["c"] > p_1030 else 0.0
        X[s], Y[s] = x, y
    return X, Y


# ── run ─────────────────────────────────────────────────────────────────────
def run(data_dir: str, dix: Optional[str]) -> dict:
    t0 = time.time()
    P = RS.Panel(data_dir)
    n = len(P.dates)
    hold_lo = next(k for k, d in enumerate(P.dates) if d >= HOLDOUT[0])
    hold_hi = max(k for k, d in enumerate(P.dates) if d <= HOLDOUT[1])
    sel_hi = max(k for k, d in enumerate(P.dates) if d <= SELECTION_END)
    names_panel = tuple(s for s in PANEL if s in P.C)
    feats = {s: name_features(P.O[s], P.C[s]) for s in names_panel}
    dvix = dvix_feature(P.iv["VIX"])
    breadth = breadth_feature({s: P.C[s] for s in BREADTH_BASKET if s in P.C})
    gex = gex_feature(dix, P.dates)
    X = {"F1": {s: feats[s]["F1"] for s in names_panel}, "F2": {s: feats[s]["F2"] for s in names_panel},
         "F3": {s: dvix for s in names_panel}, "F4": {s: breadth for s in names_panel},
         "F5": {s: gex for s in names_panel}, "G1": {s: feats[s]["G1"] for s in names_panel}}
    Y = {s: feats[s]["y"] for s in names_panel}
    Yrod = {s: feats[s]["y_rod"] for s in names_panel}
    # regimes for descriptive splits
    labels = RS.regime_labels(P)
    vix = P.iv["VIX"]
    first_any = 0
    sel_v = vix[:sel_hi + 1]
    cuts = tuple(float(x) for x in np.quantile(sel_v[np.isfinite(sel_v)], [1 / 3, 2 / 3]))
    D = hold_hi - hold_lo + 1
    vh = vix[hold_lo:hold_hi + 1]
    lab = labels[hold_lo:hold_hi + 1]
    splits = {"VIX low": vh < cuts[0], "VIX mid": (vh >= cuts[0]) & (vh < cuts[1]), "VIX high": vh >= cuts[1]}
    for L in ("BULL", "NEUTRAL", "CAUTION", "BEAR", "PANIC"):
        splits[L] = np.array([x == L for x in lab])
    res = {"spec": {"selection_end": SELECTION_END, "holdout": HOLDOUT, "refit_every": REFIT_EVERY, "embargo": EMBARGO,
                    "min_train": MIN_TRAIN, "block": BLOCK, "n_boot": N_BOOT, "seed": SEED, "m_tests": M_TESTS,
                    "p_bar": P_BAR}, "inputs": RS.input_hashes(data_dir), "vix_tercile_cuts": cuts, "tests": {}, "base": {}}
    first_eval_all = 21  # the walk-forward starts as soon as MIN_TRAIN sessions exist
    tests = [("T1", "F1", ("SPY",), "y"), ("T2", "F2", ("SPY",), "y"), ("T3", "F3", ("SPY",), "y"),
             ("T4", "F4", ("SPY",), "y"), ("T5", "F5", ("SPY",), "y"),
             ("T6", "F1", names_panel, "y"), ("T7", "F2", names_panel, "y"), ("T8", "F3", names_panel, "y"),
             ("T9", "F4", names_panel, "y"), ("T10", "F5", names_panel, "y"),
             ("T12", "G1", ("SPY",), "y_rod"), ("T13", "G1", names_panel, "y_rod")]
    # base rates (intercept only) — the reference every test is judged against
    for uni, names in (("SPY", ("SPY",)), ("panel", names_panel)):
        for tgt, YY in (("y", Y), ("y_rod", Yrod)):
            pm, pb, _ = walk_forward({}, YY, names, first_eval_all, hold_hi, n, [])
            h = per_day_loss(pm, pb, YY, names, hold_lo, hold_hi)
            res["base"][f"{uni}_{tgt}"] = {"n": h["n"], "days": h["days"], "up_rate": h["up_rate"],
                                           "brier_base": h["brier_base"], "brier_coinflip": 0.25,
                                           "skill_vs_coinflip": 1 - h["brier_base"] / 0.25}
    passed = []
    for tid, f, names, tgt in tests:
        YY = Y if tgt == "y" else Yrod
        pm, pb, fits = walk_forward(X, YY, names, first_eval_all, hold_hi, n, [f])
        hold = per_day_loss(pm, pb, YY, names, hold_lo, hold_hi)
        sel = per_day_loss(pm, pb, YY, names, 0, sel_hi)
        lo, hi, p = bootstrap_mean(hold["d_sum"], hold["cnt"])
        v = verdict(hold, sel if sel["n"] else None, p)
        if v == "PASS":
            passed.append((f, names, tgt))
        sp = {}
        for rn, m in splits.items():
            r = per_day_loss(pm, pb, YY, names, hold_lo, hold_hi, m)
            sp[rn] = {"n": r["n"], "d_brier": r["d_brier"], "brier_base": r["brier_base"]}
        last = fits[-1] if fits else None
        res["tests"][tid] = {"feature": f, "universe": "SPY" if names == ("SPY",) else "panel", "target": tgt,
                             "holdout": {k: v2 for k, v2 in hold.items() if k not in ("d_sum", "cnt")},
                             "selection": {k: v2 for k, v2 in sel.items() if k not in ("d_sum", "cnt")},
                             "d_brier_ci95": (lo, hi), "p_one_sided": p, "verdict": v,
                             "last_fit": {"coef": [round(c, 6) for c in last["coef"]], "base": round(last["base"], 4),
                                          "n_train": last["n_train"]} if last else None,
                             "first_fit_at": P.dates[fits[0]["refit_at"]] if fits else None,
                             "splits_holdout": sp}
        print(f"[{time.time() - t0:6.1f}s] {tid} {f} {res['tests'][tid]['universe']} {tgt}: dBrier "
              f"{hold['d_brier'] * 1e4:+.2f}e-4 p {p:.4f} sel {sel['d_brier'] * 1e4 if sel['n'] else float('nan'):+.2f}e-4 -> {v}", flush=True)
    # T11: combination of the individual passes (primary target), only if >= 2 passed
    prim = [x for x in passed if x[2] == "y"]
    if len(prim) >= 2:
        names = prim[0][1]
        cols = [x[0] for x in prim if x[1] == names]
        pm, pb, fits = walk_forward(X, Y, names, first_eval_all, hold_hi, n, cols)
        hold = per_day_loss(pm, pb, Y, names, hold_lo, hold_hi)
        sel = per_day_loss(pm, pb, Y, names, 0, sel_hi)
        lo, hi, p = bootstrap_mean(hold["d_sum"], hold["cnt"])
        res["tests"]["T11"] = {"feature": "+".join(cols), "verdict": verdict(hold, sel, p), "p_one_sided": p,
                               "holdout": {k: v2 for k, v2 in hold.items() if k not in ("d_sum", "cnt")},
                               "d_brier_ci95": (lo, hi)}
    else:
        res["tests"]["T11"] = {"feature": "combination of passes", "verdict": "NOT RUN (fewer than 2 individual passes)"}
    # T14: first-hour momentum (60-minute bars)
    names14 = tuple(s for s in ("SPY", "QQQ", "IWM", "DIA") + RS.STOCKS if s in P.C)
    sd = {s: feats[s]["sd20"] if s in feats else sd20(P.C[s]) for s in names14}
    X14, Y14 = first_hour_rows(P, data_dir, names14, sd)
    ks = sorted({int(k) for s in names14 for k in np.nonzero(np.isfinite(Y14[s]))[0]})
    if len(ks) > 100:
        mid = ks[len(ks) // 2]
        q1 = ks[len(ks) // 4]
        pm, pb, fits = walk_forward({"G2": X14}, Y14, names14, mid, ks[-1], n, ["G2"], min_train=len([k for k in ks if k < mid]) - EMBARGO - 2)
        hold = per_day_loss(pm, pb, Y14, names14, mid, ks[-1])
        pm1, pb1, _ = walk_forward({"G2": X14}, Y14, names14, q1, mid - 1, n, ["G2"], min_train=len([k for k in ks if k < q1]) - EMBARGO - 2)
        sel = per_day_loss(pm1, pb1, Y14, names14, q1, mid - 1)
        lo, hi, p = bootstrap_mean(hold["d_sum"], hold["cnt"])
        res["tests"]["T14"] = {"feature": "G2", "universe": "SPY+QQQ+IWM+DIA+20 names (60m)", "target": "y_from_1030",
                               "holdout": {k: v2 for k, v2 in hold.items() if k not in ("d_sum", "cnt")},
                               "selection": {k: v2 for k, v2 in sel.items() if k not in ("d_sum", "cnt")},
                               "d_brier_ci95": (lo, hi), "p_one_sided": p, "verdict": verdict(hold, sel, p),
                               "sessions": len(ks), "split_date": P.dates[mid], "first": P.dates[ks[0]], "last": P.dates[ks[-1]],
                               "last_fit": {"coef": [round(c, 6) for c in fits[-1]["coef"]]} if fits else None}
        spy = per_day_loss(pm, pb, Y14, ("SPY",), mid, ks[-1])
        res["tests"]["T14"]["spy_only_holdout"] = {k: v2 for k, v2 in spy.items() if k not in ("d_sum", "cnt")}
        print(f"[{time.time() - t0:6.1f}s] T14 G2: dBrier {hold['d_brier'] * 1e4:+.2f}e-4 p {p:.4f} -> {res['tests']['T14']['verdict']}", flush=True)
    res["shipped_features"] = sorted({t["feature"] for t in res["tests"].values() if t.get("verdict") == "PASS"
                                      and t.get("target") == "y"})
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


def render_md(res: dict) -> str:
    L: List[str] = []
    A = L.append
    A(f"Inputs (sha256/16): {res['inputs']}. VIX tercile cut points (pre-2019): "
      f"{res['vix_tercile_cuts'][0]:.2f} / {res['vix_tercile_cuts'][1]:.2f}. Bonferroni bar: one-sided p < {res['spec']['p_bar']:.5f}.")
    A("")
    A("### Base rates (HOLDOUT 2019-01-02 .. 2026-11-02, walk-forward point-in-time)")
    A("")
    A("| universe / target | rows | days | up-frequency | base-rate Brier | coin flip | base-rate skill vs coin flip |")
    A("|---|---|---|---|---|---|---|")
    for k, b in res["base"].items():
        A(f"| {k} | {b['n']} | {b['days']} | {b['up_rate'] * 100:.2f}% | {b['brier_base']:.5f} | 0.25000 | {b['skill_vs_coinflip'] * 100:+.2f}% |")
    A("")
    A("### Tests")
    A("")
    A("| test | feature | universe | target | holdout rows | Brier base | Brier model | dBrier x1e4 [95% CI] | one-sided p | selection dBrier x1e4 | holdout hit rate | verdict |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tid in sorted(res["tests"], key=lambda x: int(x[1:])):
        t = res["tests"][tid]
        if "holdout" not in t:
            A(f"| {tid} | {t['feature']} | — | — | — | — | — | — | — | — | — | {t['verdict']} |")
            continue
        h = t["holdout"]
        s = t.get("selection") or {}
        ci = t.get("d_brier_ci95") or (float("nan"), float("nan"))
        sd = s.get("d_brier")
        A(f"| {tid} | {t['feature']} | {t.get('universe', '')} | {t.get('target', '')} | {h['n']} | {h['brier_base']:.5f} | {h['brier_model']:.5f} | "
          f"{h['d_brier'] * 1e4:+.2f} [{ci[0] * 1e4:+.2f}, {ci[1] * 1e4:+.2f}] | {t['p_one_sided']:.4f} | "
          f"{(sd * 1e4) if sd is not None and np.isfinite(sd) else float('nan'):+.2f} | {h['hit_rate'] * 100:.2f}% | **{t['verdict']}** |")
    A("")
    A("### Holdout dBrier x1e4 by regime (descriptive; negative = the feature helped)")
    A("")
    keys = None
    for tid in sorted(res["tests"], key=lambda x: int(x[1:])):
        t = res["tests"][tid]
        if "splits_holdout" not in t:
            continue
        if keys is None:
            keys = list(t["splits_holdout"])
            A("| test | " + " | ".join(keys) + " |")
            A("|---|" + "---|" * len(keys))
        A(f"| {tid} {t['feature']} {t['universe']} | " + " | ".join(
            (f"{v['d_brier'] * 1e4:+.1f} (n {v['n']})" if v["n"] else "—") for v in t["splits_holdout"].values()) + " |")
    A("")
    A(f"Shipped features (PASS on the primary target): {res['shipped_features'] or 'none'}.")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dix", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = run(a.data_dir, a.dix)
    md = render_md(res)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(md)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(RS._json_safe(res), fh, indent=1, sort_keys=True)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
