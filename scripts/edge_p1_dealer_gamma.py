#!/usr/bin/env python3
"""scripts/edge_p1_dealer_gamma.py — pre-registered test P1, ladder gate 2
(SIGNAL): does dealer gamma (SqueezeMetrics GEX) predict SPY's next-5-day
realized variance relative to VIX-implied variance, incremental to VIX?

Spec is FIXED by research/edge_preregistration.md § "P1 — Dealer gamma
exposure → realized-vs-implied volatility" (registered 2026-10-27, before any
outcome was seen). Nothing here is tuned; every constant below is copied from
that section.

  G_t  = SqueezeMetrics `gex` on day t (published after the close of t).
  R_t  = percentile rank of G_t within the trailing 252 GEX rows ending at t
         (inclusive; first 252 rows are burn-in -> test starts 2012-05).
  y_t  = log( RV_{t+1..t+5} / IV_t ),
         RV = sum of squared SPY close-to-close log returns over t+1..t+5
              x 252/5 (annualized variance),
         IV = (VIX_t / 100)^2.
  OLS  y_t = a + b*R_t + c*log(IV_t) + e_t, Newey-West (Bartlett) lag 5.
  PASS : b < 0 with t <= -2.5 on 2012-05 -> 2026-10 AND b < 0 with
         t <= -1.5 on the holdout 2019-01 -> 2026-10.
  KILL : b >= 0, or t > -2.5 on the full window, or the holdout sign flips.
  (Full passes but holdout same sign with t > -1.5 is neither: INCONCLUSIVE.)

Secondaries (reported, never used to pass): (i) Parkinson range RV;
(iii) split by VIX tercile and by the bot's regime labels. (ii) — the 30-min
intraday-momentum split on 1-minute bars — needs 1-minute SPY history this
sandbox cannot get free (no Alpaca keys); NOT run, stated in the output.
Robustness (declared): R lagged one extra day.

DATA (free, keyless; downloaded to a temp dir at runtime, never committed):
  DIX.csv   https://squeezemetrics.com/monitor/static/DIX.csv  (research use
            only, licence unverified — NEVER commit or redistribute it)
  VIX       https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
  SPY       Yahoo v8 chart daily bars (same endpoint as backtest_v2._yahoo_bars,
            the repo's keyless fallback); dividend/split-adjusted close.
Determinism: --end caps the sample (default 2026-10-28) so a rerun on a
later vintage reproduces the same rows unless upstream revises history; the
sha256 of every input file is printed so a vintage change is visible.

Run:  python3 scripts/edge_p1_dealer_gamma.py [--data-dir DIR] [--end YYYY-MM-DD]
      (--data-dir reads DIX.csv / VIX_History.csv / spy.json from DIR
       instead of downloading.)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PRIOR = ("70% that b < 0 with t <= -2.5 in-sample; 45% that it also clears the "
         "holdout bar (t <= -1.5, same sign); 35% that the effect is >= 0.19 log "
         "points (large enough to matter for the sleeve).")

# ── pre-registered constants (edge_preregistration.md P1) — do not tune ─────
HORIZON = 5            # trading days t+1..t+5
RANK_WINDOW = 252      # trailing GEX rows for the percentile rank
NW_LAG = 5             # Newey-West Bartlett lag
TRADING_DAYS = 252
FULL_START = "2012-05-01"
HOLDOUT_START = "2019-01-01"
DEFAULT_END = "2026-10-28"
T_FULL = -2.5
T_HOLDOUT = -1.5
MDE = 0.19             # minimum detectable slope (log points, rank 0 -> 1)
GATE3_LOW_RANK = 0.2   # the gate-3 candidate state "R < 0.2" (descriptive only)

DIX_URL = "https://squeezemetrics.com/monitor/static/DIX.csv"
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
SPY_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/SPY"
           "?period1=1293840000&period2={end}&interval=1d&events=div%2Csplit")


# ── pure computation ────────────────────────────────────────────────────────
def log_returns(closes: Sequence[float]) -> List[Optional[float]]:
    """r[j] = ln(C_j / C_{j-1}); r[0] is None (no prior close)."""
    out: List[Optional[float]] = [None]
    for j in range(1, len(closes)):
        out.append(math.log(closes[j] / closes[j - 1]))
    return out


def forward_realized_var(logret: Sequence[Optional[float]], i: int,
                         h: int = HORIZON) -> Optional[float]:
    """Annualized realized variance over days i+1..i+h ONLY (never day i or
    earlier, never beyond i+h): sum(r^2) * 252 / h. None if the window runs
    past the data."""
    if i + h >= len(logret):
        return None
    rs = logret[i + 1:i + h + 1]
    if any(r is None for r in rs):
        return None
    return sum(r * r for r in rs) * TRADING_DAYS / h


def forward_parkinson_var(highs: Sequence[float], lows: Sequence[float], i: int,
                          h: int = HORIZON) -> Optional[float]:
    """Parkinson range variance over days i+1..i+h, annualized:
    mean_j[(ln H_j/L_j)^2 / (4 ln 2)] * 252. Ignores overnight gaps, so its
    level sits below close-to-close; the regression intercept absorbs that."""
    if i + h >= len(highs):
        return None
    tot = 0.0
    for j in range(i + 1, i + h + 1):
        if not highs[j] or not lows[j] or highs[j] < lows[j]:
            return None
        tot += math.log(highs[j] / lows[j]) ** 2 / (4.0 * math.log(2.0))
    return tot / h * TRADING_DAYS


def implied_var_from_vix(vix: float) -> float:
    """VIX (vol points, annualized) -> annualized implied variance."""
    return (vix / 100.0) ** 2


def trailing_pct_rank(values: Sequence[float], window: int = RANK_WINDOW) -> List[Optional[float]]:
    """Percentile rank of values[t] within values[t-window+1..t] (inclusive),
    scaled to [0, 1]: (count strictly below + 0.5 * ties excluding self) /
    (window - 1). Uses ONLY rows <= t. None for the first window-1 rows."""
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


def vix_ratio_30(vix_series: Sequence[Optional[float]], i: int) -> float:
    """VIX_i / mean(VIX_{i-29..i}) — the VXX-ratio proxy regime_backtest.py
    uses before reliable VXX history (data through i only)."""
    win = [x for x in vix_series[max(0, i - 29):i + 1] if x is not None]
    v = vix_series[i]
    if v is None or not win:
        return 1.0
    return v / (sum(win) / len(win))


def trailing_realized_var(logret: Sequence[Optional[float]], i: int,
                          h: int = HORIZON) -> Optional[float]:
    """Annualized realized variance over days i-h+1..i (data through t only)
    — the post-hoc vol-clustering control."""
    if i - h + 1 < 1:
        return None
    rs = logret[i - h + 1:i + 1]
    if any(r is None for r in rs):
        return None
    return sum(r * r for r in rs) * TRADING_DAYS / h


def build_panel(spy: Dict[str, list], gex_rows: Sequence[Tuple[str, float]],
                vix_by_date: Dict[str, float], end: str = DEFAULT_END,
                classify=None, vix9d_by_date: Optional[Dict[str, float]] = None
                ) -> Tuple[List[dict], dict]:
    """One row per SPY trading day t with a GEX rank, a VIX close and a full
    t+1..t+5 forward window inside [.., end]. SPY's own calendar is the clock
    for the forward window. Returns (rows, counts). `vix9d_by_date` and the
    trailing-RV field feed the POST-HOC diagnostics only."""
    keep = [k for k, d in enumerate(spy["date"]) if d <= end]
    dates = [spy["date"][k] for k in keep]
    closes = [spy["close"][k] for k in keep]
    highs = [spy["high"][k] for k in keep]
    lows = [spy["low"][k] for k in keep]
    gex_rows = sorted((d, g) for d, g in gex_rows if d <= end)
    ranks = trailing_pct_rank([g for _, g in gex_rows])
    rank_by_date = {d: r for (d, _), r in zip(gex_rows, ranks) if r is not None}
    gex_by_date = {d: g for d, g in gex_rows}

    lr = log_returns(closes)
    vix_series = [vix_by_date.get(d) for d in dates]
    rows: List[dict] = []
    counts = {"spy_days": len(dates), "gex_rows": len(gex_rows),
              "skip_no_rank": 0, "skip_no_vix": 0, "skip_no_forward": 0}
    below200 = 0
    for i, d in enumerate(dates):
        regime = None
        if classify is not None:
            ma50 = sum(closes[max(0, i - 49):i + 1]) / len(closes[max(0, i - 49):i + 1])
            ma200 = (sum(closes[i - 199:i + 1]) / 200.0) if i >= 199 else None
            above = (closes[i] > ma200) if ma200 is not None else True
            below200 = 0 if above else below200 + 1
            regime = classify(vix_ratio_30(vix_series, i), closes[i] / ma50, below200, above)
        if d not in rank_by_date:
            counts["skip_no_rank"] += 1
            continue
        if vix_series[i] is None:
            counts["skip_no_vix"] += 1
            continue
        rv = forward_realized_var(lr, i)
        pk = forward_parkinson_var(highs, lows, i)
        if rv is None or rv <= 0:
            counts["skip_no_forward"] += 1
            continue
        iv = implied_var_from_vix(vix_series[i])
        prev = dates[i - 1] if i > 0 else None
        trv = trailing_realized_var(lr, i)
        v9 = (vix9d_by_date or {}).get(d)
        iv9 = implied_var_from_vix(v9) if v9 else None
        rows.append({
            "date": d, "idx": i, "gex": gex_by_date[d], "R": rank_by_date[d],
            "R_lag1": rank_by_date.get(prev) if prev else None,
            "vix": vix_series[i], "log_iv": math.log(iv),
            "y": math.log(rv / iv),
            "y_park": math.log(pk / iv) if pk and pk > 0 else None,
            "regime": regime,
            "log_trv": math.log(trv) if trv and trv > 0 else None,
            "log_iv9": math.log(iv9) if iv9 else None,
            "y9": math.log(rv / iv9) if iv9 else None,
        })
    return rows, counts


def ols_nw(y: Sequence[float], X: Sequence[Sequence[float]], lag: int = NW_LAG,
           time_index: Optional[Sequence[int]] = None) -> dict:
    """OLS with Newey-West (Bartlett) HAC covariance, no small-sample
    correction. Cross-products are taken between observations whose
    `time_index` (SPY trading-day index) differs by 1..lag, so a gappy
    subsample never pairs rows that are not actually within `lag` days.
    X must already include the constant column."""
    Y = np.asarray(y, dtype=float)
    Xm = np.asarray(X, dtype=float)
    n, k = Xm.shape
    idx = list(range(n)) if time_index is None else list(time_index)
    xtx_inv = np.linalg.inv(Xm.T @ Xm)
    beta = xtx_inv @ Xm.T @ Y
    e = Y - Xm @ beta
    u = Xm * e[:, None]
    S = u.T @ u
    pos = {t: r for r, t in enumerate(idx)}
    for ell in range(1, lag + 1):
        w = 1.0 - ell / (lag + 1.0)
        G = np.zeros((k, k))
        for r, t in enumerate(idx):
            s = pos.get(t - ell)
            if s is not None:
                G += np.outer(u[r], u[s])
        S += w * (G + G.T)
    V = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.diag(V))
    resid_sd = float(np.sqrt(e @ e / max(1, n - k)))
    return {"beta": beta.tolist(), "se": se.tolist(),
            "t": (beta / se).tolist(), "n": n, "resid_sd": resid_sd}


def ols_iid(y: Sequence[float], X: Sequence[Sequence[float]]) -> dict:
    """Plain OLS with classical SE (for the non-overlapping diagnostic)."""
    Y = np.asarray(y, dtype=float)
    Xm = np.asarray(X, dtype=float)
    n, k = Xm.shape
    xtx_inv = np.linalg.inv(Xm.T @ Xm)
    beta = xtx_inv @ Xm.T @ Y
    e = Y - Xm @ beta
    s2 = float(e @ e) / (n - k)
    se = np.sqrt(np.diag(xtx_inv) * s2)
    return {"beta": beta.tolist(), "se": se.tolist(), "t": (beta / se).tolist(), "n": n}


def non_overlapping(rows: Sequence[dict], h: int = HORIZON, offset: int = 0) -> List[dict]:
    """Rows whose SPY-day index is offset mod h, and whose forward windows
    therefore never share a day with each other."""
    out, last = [], None
    for r in rows:
        if (r["idx"] - offset) % h != 0:
            continue
        if last is not None and r["idx"] - last < h:
            continue
        out.append(r)
        last = r["idx"]
    return out


def regress(rows: Sequence[dict], rkey: str = "R", ykey: str = "y",
            controls: Sequence[str] = ("log_iv",)) -> Optional[dict]:
    """y ~ 1 + R + controls, NW lag 5. The pre-registered spec is the default
    (controls = log IV only); anything else is a labelled post-hoc diagnostic."""
    need = (rkey, ykey) + tuple(controls)
    rs = [r for r in rows if all(r.get(k) is not None for k in need)]
    if len(rs) < 30:
        return None
    X = [[1.0, r[rkey]] + [r[c] for c in controls] for r in rs]
    fit = ols_nw([r[ykey] for r in rs], X, NW_LAG, [r["idx"] for r in rs])
    fit["b"], fit["b_se"], fit["b_t"] = fit["beta"][1], fit["se"][1], fit["t"][1]
    fit["c"], fit["c_t"] = fit["beta"][2], fit["t"][2]
    return fit


def verdict(full: dict, hold: dict) -> Tuple[str, List[str]]:
    """Map (full, holdout) fits onto the pre-registered pass/kill rule."""
    why = []
    if full["b"] >= 0:
        why.append(f"full-window b = {full['b']:+.4f} >= 0")
    if full["b_t"] > T_FULL:
        why.append(f"full-window t = {full['b_t']:+.2f} > {T_FULL}")
    if (hold["b"] < 0) != (full["b"] < 0):
        why.append(f"holdout sign flips (b = {hold['b']:+.4f})")
    if why:
        return "FAIL", why
    if hold["b_t"] <= T_HOLDOUT:
        return "PASS", [f"full t = {full['b_t']:+.2f} <= {T_FULL}; holdout t = "
                        f"{hold['b_t']:+.2f} <= {T_HOLDOUT}"]
    return "INCONCLUSIVE", [f"full passes but holdout t = {hold['b_t']:+.2f} > {T_HOLDOUT} "
                            "(same sign, below bar)"]


def base_rate(rows: Sequence[dict], ykey: str = "y") -> dict:
    ys = [r[ykey] for r in rows if r.get(ykey) is not None]
    if not ys:
        return {"n": 0}
    return {"n": len(ys), "mean_y": sum(ys) / len(ys),
            "share_rv_gt_iv": sum(1 for v in ys if v > 0) / len(ys),
            "vol_ratio": math.exp(sum(ys) / len(ys) / 2.0)}


def quintile_table(rows: Sequence[dict]) -> List[dict]:
    out = []
    for q in range(5):
        lo, hi = q / 5.0, (q + 1) / 5.0
        sub = [r for r in rows if (lo <= r["R"] < hi) or (q == 4 and r["R"] == 1.0)]
        br = base_rate(sub)
        br["bucket"] = f"[{lo:.1f},{hi:.1f}{']' if q == 4 else ')'}"
        out.append(br)
    return out


def date_alignment_check(spy: Dict[str, list], dix_price: Dict[str, float]) -> Dict[int, float]:
    """Correlation of SPY daily log returns with DIX `price` (SPX) log returns
    at shifts -1/0/+1 on common dates. Lag 0 should dominate (~0.99) if
    DIX dates are the same-session dates the spec assumes."""
    dates = [d for d in spy["date"] if d in dix_price]
    sd = dict(zip(spy["date"], spy["close"]))
    s = [sd[d] for d in dates]
    p = [dix_price[d] for d in dates]
    rs = np.diff(np.log(s))
    rp = np.diff(np.log(p))
    out = {}
    for k in (-1, 0, 1):
        if k < 0:
            a, b = rs[:k], rp[-k:]
        elif k > 0:
            a, b = rs[k:], rp[:-k]
        else:
            a, b = rs, rp
        out[k] = float(np.corrcoef(a, b)[0, 1])
    return out


# ── loading (network / disk) ────────────────────────────────────────────────
VIX9D_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv"


def _get(url: str, path: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(path, "wb") as fh:
        fh.write(resp.read())


def fetch_all(dest: str, end: str) -> None:
    end_ts = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 4 * 86400
    _get(DIX_URL, os.path.join(dest, "DIX.csv"))
    _get(VIX_URL, os.path.join(dest, "VIX_History.csv"))
    _get(VIX9D_URL, os.path.join(dest, "VIX9D_History.csv"))
    _get(SPY_URL.format(end=end_ts), os.path.join(dest, "spy.json"))


def load_dix(path: str) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    gex, price = [], {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("gex") or not row.get("date"):
                continue
            gex.append((row["date"], float(row["gex"])))
            price[row["date"]] = float(row["price"])
    return gex, price


def load_vix(path: str) -> Dict[str, float]:
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            m, d, y = row["DATE"].split("/")
            out[f"{y}-{m}-{d}"] = float(row["CLOSE"])
    return out


def load_spy(path: str) -> Dict[str, list]:
    with open(path) as fh:
        res = json.load(fh)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    out = {"date": [], "close": [], "high": [], "low": []}
    for i, ts in enumerate(res["timestamp"]):
        if None in (q["close"][i], q["high"][i], q["low"][i], adj[i]):
            continue
        out["date"].append(datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d"))
        out["close"].append(float(adj[i]))
        out["high"].append(float(q["high"][i]))
        out["low"].append(float(q["low"][i]))
    return out


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


# ── report ──────────────────────────────────────────────────────────────────
def _fmt_fit(name: str, f: Optional[dict]) -> str:
    if f is None:
        return f"| {name} | n<30 | | | | |"
    return (f"| {name} | {f['n']} | {f['b']:+.4f} | {f['b_se']:.4f} | {f['b_t']:+.2f} | "
            f"{f['c']:+.3f} (t {f['c_t']:+.1f}) |")


def _classifier():
    sys.path.insert(0, ROOT)
    try:
        from regime_util import classify_regime_5level
    except ImportError as exc:
        print(f"WARNING: regime_util unavailable ({exc}); regime split skipped")
        return None
    return lambda vr, s50, b200, above: classify_regime_5level(
        vr, s50, spy_below_200_days=b200, spy_above_200d=above)


def run(data_dir: str, end: str) -> dict:
    gex_rows, dix_price = load_dix(os.path.join(data_dir, "DIX.csv"))
    vix = load_vix(os.path.join(data_dir, "VIX_History.csv"))
    spy = load_spy(os.path.join(data_dir, "spy.json"))
    vix9d = load_vix(os.path.join(data_dir, "VIX9D_History.csv"))
    rows, counts = build_panel(spy, gex_rows, vix, end, _classifier(), vix9d)
    full = [r for r in rows if r["date"] >= FULL_START]
    hold = [r for r in full if r["date"] >= HOLDOUT_START]
    insample = [r for r in full if r["date"] < HOLDOUT_START]
    res = {
        "inputs": {f: _sha(os.path.join(data_dir, f)) for f in ("DIX.csv", "VIX_History.csv", "VIX9D_History.csv", "spy.json")},
        "counts": counts,
        "range": {"dix": [gex_rows[0][0], gex_rows[-1][0]],
                  "spy": [spy["date"][0], spy["date"][-1]],
                  "vix": [min(vix), max(vix)],
                  "panel": [full[0]["date"], full[-1]["date"]]},
        "alignment": date_alignment_check(spy, dix_price),
        "gex_negative_share": sum(1 for r in full if r["gex"] < 0) / len(full),
        "full": regress(full), "holdout": regress(hold), "pre2019": regress(insample),
        "lag1_full": regress(full, "R_lag1"), "lag1_holdout": regress(hold, "R_lag1"),
        "park_full": regress(full, ykey="y_park"), "park_holdout": regress(hold, ykey="y_park"),
        "base_full": base_rate(full), "base_holdout": base_rate(hold),
        "quintiles_full": quintile_table(full), "quintiles_holdout": quintile_table(hold),
    }
    # non-overlapping diagnostic (inference check on the SAME spec, not a variant)
    nov = []
    for off in range(HORIZON):
        sub = non_overlapping(full, HORIZON, off)
        f = ols_iid([r["y"] for r in sub], [[1.0, r["R"], r["log_iv"]] for r in sub])
        nov.append({"offset": off, "n": f["n"], "b": f["beta"][1], "se": f["se"][1], "t": f["t"][1]})
    res["nonoverlap"] = nov
    # VIX terciles (cut points from the full test sample — a descriptive split)
    vs = sorted(r["vix"] for r in full)
    c1, c2 = vs[len(vs) // 3], vs[2 * len(vs) // 3]
    res["vix_cuts"] = [c1, c2]
    res["vix_terciles"] = {
        f"low (<{c1:.2f})": regress([r for r in full if r["vix"] < c1]),
        f"mid [{c1:.2f},{c2:.2f})": regress([r for r in full if c1 <= r["vix"] < c2]),
        f"high (>={c2:.2f})": regress([r for r in full if r["vix"] >= c2]),
    }
    regimes = sorted({r["regime"] for r in full if r["regime"]})
    res["regimes"] = {g: {"fit": regress([r for r in full if r["regime"] == g]),
                          "base": base_rate([r for r in full if r["regime"] == g])}
                      for g in regimes}
    risk_on = [r for r in full if r["regime"] in ("BULL", "NEUTRAL")]
    risk_off = [r for r in full if r["regime"] in ("CAUTION", "BEAR", "PANIC")]
    res["regime_pooled"] = {"BULL+NEUTRAL": regress(risk_on), "CAUTION+BEAR+PANIC": regress(risk_off)}
    res["verdict"], res["verdict_why"] = verdict(res["full"], res["holdout"])
    # ── POST-HOC diagnostics (NOT pre-registered, never used for the verdict):
    # they bound the INTERPRETATION of b. (a) vol clustering: low GEX follows
    # selloffs, so R may proxy recent realized vol that VIX under-prices;
    # (b) horizon mismatch: VIX is 30-day, the outcome 5-day — in calm/contango
    # tapes VIX overstates 5-day implied. VIX9D is the closer-horizon implied.
    post = {}
    for nm, sub in (("full", full), ("holdout", hold)):
        post[f"+trailRV {nm}"] = regress(sub, controls=("log_iv", "log_trv"))
        post[f"VIX9D {nm}"] = regress(sub, ykey="y9", controls=("log_iv9",))
        post[f"VIX9D+trailRV {nm}"] = regress(sub, ykey="y9", controls=("log_iv9", "log_trv"))
    res["posthoc"] = post
    pr = [(r["R"], r["log_trv"]) for r in full if r["log_trv"] is not None]
    res["corr_R_logtrv"] = float(np.corrcoef([a for a, _ in pr], [b for _, b in pr])[0, 1])
    res["base_gex_negative"] = base_rate([r for r in full if r["gex"] < 0])
    return res


def render(res: dict) -> str:
    L = []
    L.append("P1 dealer gamma -> log(RV_5d / VIX^2): pre-registered gate-2 test")
    L.append(f"PRIOR (stated before the run): {PRIOR}")
    L.append(f"inputs sha256[:16]: {res['inputs']}")
    L.append(f"ranges: {res['range']}")
    L.append(f"counts: {res['counts']}")
    a = res["alignment"]
    L.append(f"date alignment corr(SPY r_t, DIX price r_t+k): k=-1 {a[-1]:+.3f}  k=0 {a[0]:+.3f}  k=+1 {a[1]:+.3f}")
    L.append(f"share of test days with GEX < 0: {res['gex_negative_share']:.3f}")
    L.append("")
    L.append("| spec | n | b (R) | NW se | t | c (log IV) |")
    L.append("|---|---|---|---|---|---|")
    L.append(_fmt_fit("PRIMARY full 2012-05 ->", res["full"]))
    L.append(_fmt_fit("PRIMARY holdout 2019-01 ->", res["holdout"]))
    L.append(_fmt_fit("pre-2019 half (info)", res["pre2019"]))
    L.append(_fmt_fit("robust: R lagged 1d, full", res["lag1_full"]))
    L.append(_fmt_fit("robust: R lagged 1d, holdout", res["lag1_holdout"]))
    L.append(_fmt_fit("sec (i) Parkinson, full", res["park_full"]))
    L.append(_fmt_fit("sec (i) Parkinson, holdout", res["park_holdout"]))
    for k, f in res["vix_terciles"].items():
        L.append(_fmt_fit(f"sec (iii) VIX {k}", f))
    for k, v in res["regimes"].items():
        L.append(_fmt_fit(f"sec (iii) regime {k}", v["fit"]))
    for k, f in res["regime_pooled"].items():
        L.append(_fmt_fit(f"sec (iii) regime {k}", f))
    L.append("sec (ii) intraday-momentum split on 1-min bars: NOT RUN (no free 1-min SPY history; Alpaca keys absent)")
    L.append("")
    L.append(f"primary residual sd: {res['full']['resid_sd']:.3f} (pre-reg power assumed 0.6)")
    L.append("non-overlapping diagnostic (classical SE, every 5th day):")
    for d in res["nonoverlap"]:
        L.append(f"  offset {d['offset']}: n {d['n']}  b {d['b']:+.4f}  se {d['se']:.4f}  t {d['t']:+.2f}")
    L.append("")
    for nm in ("base_full", "base_holdout"):
        b = res[nm]
        L.append(f"base rate {nm}: n {b['n']}  mean y {b['mean_y']:+.3f}  share RV>IV {b['share_rv_gt_iv']:.3f}  "
                 f"RV/IV vol ratio {b['vol_ratio']:.3f}")
    for nm in ("quintiles_full", "quintiles_holdout"):
        L.append(f"{nm} (R bucket: n, mean y, share RV>IV, vol ratio):")
        for q in res[nm]:
            L.append(f"  {q['bucket']}: {q['n']}  {q['mean_y']:+.3f}  {q['share_rv_gt_iv']:.3f}  {q['vol_ratio']:.3f}")
    L.append("regime base rates (n, mean y, share RV>IV):")
    for k, v in res["regimes"].items():
        b = v["base"]
        L.append(f"  {k}: {b['n']}  {b['mean_y']:+.3f}  {b['share_rv_gt_iv']:.3f}")
    L.append("")
    L.append("POST-HOC DIAGNOSTICS (not pre-registered; never used for the verdict):")
    L.append(f"corr(R_t, log trailing 5d RV_t) = {res['corr_R_logtrv']:+.3f}")
    b = res["base_gex_negative"]
    L.append(f"GEX < 0 days: n {b['n']}  mean y {b['mean_y']:+.3f}  share RV>IV {b['share_rv_gt_iv']:.3f}  "
             f"vol ratio {b['vol_ratio']:.3f}")
    L.append("| diagnostic | n | b (R) | NW se | t | first control |")
    L.append("|---|---|---|---|---|---|")
    for k, f in res["posthoc"].items():
        L.append(_fmt_fit(k, f))
    L.append("")
    L.append(f"VERDICT: {res['verdict']} — {'; '.join(res['verdict_why'])}")
    f = res["full"]
    L.append(f"effect vs MDE: |b| = {abs(f['b']):.3f} vs {MDE} log pts; 95% CI on b "
             f"[{f['b'] - 1.96 * f['b_se']:+.3f}, {f['b'] + 1.96 * f['b_se']:+.3f}]")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", help="read inputs from this dir instead of downloading")
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--json-out", help="write derived summary JSON here (no raw data)")
    args = ap.parse_args(argv)
    if args.data_dir:
        res = run(args.data_dir, args.end)
    else:
        with tempfile.TemporaryDirectory(prefix="edge_p1_") as tmp:
            fetch_all(tmp, args.end)
            res = run(tmp, args.end)
    print(render(res))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
