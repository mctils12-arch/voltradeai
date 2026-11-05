#!/usr/bin/env python3
"""scripts/forecast_daily_study.py — Part C (human-approved batch, 2026-11-04):
the walk-forward study behind the daily forecast's 68% RANGE model
(forecast_range.py, fc-range-har-v1). Research only: nothing here runs in
production, and production never reads Yahoo — it keeps Alpaca bars, the Cboe
VIX archive and the chain IV30 archive (read forecast_range.py's header).

The spec was PRE-REGISTERED before any result was computed; it is copied into
research/forecast_daily_study_2026-11-04.md verbatim. Nothing below is tuned
after seeing an outcome. Summary of the spec:

  origin t   features use closes <= t and the implied-vol index close of t
  target     |log(C_{t+h}/C_t)|, h = 1 (the daily call) and h = 5 (weekly check)
  regressors v_d = |r_t|, v_w = sqrt(mean r^2, t-4..t), v_m = sqrt(mean r^2,
             t-21..t), v_iv = IV_t / 100 / sqrt(252)
  variants   V_clim constant band (the name's expanding 68.27% quantile of |y|)
             V0     the current path (forecast.sigma_for: IV * sqrt(cal/365), else
                    20-session realized vol * sqrt(h)), z = 1
             V1     the dossier HAR (dossier_ranges.sigma_forecast), z = 1
             V2/V3/V4 linear quantile regression of |y| at tau = 0.6827, no
                    intercept, theta >= 0, on [v_iv] / [v_d v_w v_m] / both
  fit groups index (SPY QQQ IWM DIA + VIX VXN RVX VXD), stock-IV (AAPL AMZN
             GOOG IBM GS + Cboe single-stock vol indices), HAR-only (every
             name, no IV); other ETFs with Cboe vol indices = transfer check
  walk-fwd   expanding window, refit every 21 sessions, training labels end
             >= h sessions before the first origin a refit forecasts, >= 504
             training sessions
  segments   SELECTION 2013-01-02..2018-12-31 (model choice), HOLDOUT
             2019-01-02..2026-11-02 (reported for every variant)
  metrics    coverage of the band (block-bootstrap 95% CI, 20-session blocks
             of whole days), interval score IS = 2q + (2/alpha)(|y|-q)+,
             width 2q, IS skill vs V_clim; splits by VIX tercile (cut points
             from SELECTION), the bot's 5-level regime label, weekday
  intraday   60-minute bars: remaining-variance share f(tau) by ET mark,
             estimated on the first half of the sessions, coverage of the live
             band S_tau * exp(+/- q sqrt(f)) on the second half

DATA (free, keyless; downloaded to --data-dir, never committed):
  Yahoo v8 chart API daily bars (the keyless endpoint backtest_v2._yahoo_bars
  already uses) — split-adjusted, NOT dividend-adjusted closes, the same
  convention as Alpaca `adjustment=split`; Yahoo 60-minute bars (~730 days);
  Cboe daily_prices <INDEX>_History.csv (VIX family + single-stock and ETF vol
  indices, the same host server/cboeVix.ts archives from).

Run:  python3 scripts/forecast_daily_study.py fetch --data-dir DIR
      python3 scripts/forecast_daily_study.py range --data-dir DIR \
          [--md research/forecast_daily_study_2026-11-04.md] \
          [--json research/forecast_daily_study_2026-11-04.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import forecast_range as FR  # noqa: E402  (the production model module: one source of truth)

# ── pre-registered constants (do not tune) ──────────────────────────────────
END = "2026-11-03"                  # last completed session at registration
FETCH_START = "1999-01-01"          # warm-up for 200-day averages / the direction study
DATA_START = "2009-09-18"           # range study: first day every Cboe index-group CSV exists
SELECTION = ("2013-01-02", "2018-12-31")
HOLDOUT = ("2019-01-02", "2026-11-02")
REFIT_EVERY = 21
MIN_TRAIN = 504
BLOCK = 20
N_BOOT = 2000
SEED = 20261104
TAU = FR.RANGE_TAU
ALPHA = 1.0 - TAU
HORIZONS = (1, 5)

INDEX_IV = {"SPY": "VIX", "QQQ": "VXN", "IWM": "RVX", "DIA": "VXD"}
STOCK_IV = {"AAPL": "VXAPL", "AMZN": "VXAZN", "GOOG": "VXGOG", "IBM": "VXIBM", "GS": "VXGS"}
OTHER_IV = {"GLD": "GVZ", "USO": "OVX", "EEM": "VXEEM", "SLV": "VXSLV", "EWZ": "VXEWZ", "FXI": "VXFXI"}
TRAIN_BASKET = ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV",
                "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC")   # = forecast.TRAIN_BASKET
STOCKS = ("AAPL", "AMZN", "GOOG", "IBM", "GS", "MSFT", "JPM", "XOM", "PFE", "KO",
          "WMT", "NVDA", "AMD", "TSLA", "NFLX", "BA", "CAT", "F", "INTC", "DIS")
BREADTH_EXTRA = ("MDY",)            # regime_engine.BREADTH_BASKET = 11 sector SPDRs + IWM + MDY
HAR_PANEL = tuple(dict.fromkeys(TRAIN_BASKET + STOCKS + tuple(OTHER_IV)))
DAILY_SYMBOLS = tuple(dict.fromkeys(HAR_PANEL + BREADTH_EXTRA))
INTRADAY_SYMBOLS = tuple(dict.fromkeys(tuple(INDEX_IV) + STOCKS + tuple(OTHER_IV)))
INDEX_GROUP_INTRADAY = tuple(INDEX_IV)
CBOE = tuple(dict.fromkeys(tuple(INDEX_IV.values()) + tuple(STOCK_IV.values()) + tuple(OTHER_IV.values()) + ("VIX9D",)))
MARKS = ((9, 30), (10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30))

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval={iv}&includePrePost=false&events=div%2Csplit"
YAHOO_RANGE = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval={iv}&includePrePost=false"
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{t}_History.csv"


# ── fetch ───────────────────────────────────────────────────────────────────
def _get(url: str, path: str, tries: int = 4) -> None:
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                blob = resp.read()
            with open(path, "wb") as fh:
                fh.write(blob)
            return
        except Exception as e:  # transient network / rate limit — retried, then raised
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"fetch failed: {url}: {last}")


def fetch_all(dest: str, end: str = END) -> None:
    os.makedirs(os.path.join(dest, "daily"), exist_ok=True)
    os.makedirs(os.path.join(dest, "h1"), exist_ok=True)
    os.makedirs(os.path.join(dest, "cboe"), exist_ok=True)
    p1 = int(datetime.strptime(FETCH_START, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    p2 = int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 86400
    for s in DAILY_SYMBOLS:
        _get(YAHOO.format(sym=s, p1=p1, p2=p2, iv="1d"), os.path.join(dest, "daily", f"{s}.json"))
    for s in INTRADAY_SYMBOLS:
        _get(YAHOO_RANGE.format(sym=s, rng="730d", iv="60m"), os.path.join(dest, "h1", f"{s}.json"))
    for t in CBOE:
        _get(CBOE_URL.format(t=t), os.path.join(dest, "cboe", f"{t}.csv"))


# ── load ────────────────────────────────────────────────────────────────────
def _et_day(ts: int, gmtoffset: int) -> str:
    return datetime.fromtimestamp(ts + gmtoffset, timezone.utc).strftime("%Y-%m-%d")


def load_daily(path: str) -> Dict[str, dict]:
    """{date: {o,h,l,c}} from a Yahoo v8 chart JSON. `close` is Yahoo's quote
    close: split-adjusted, not dividend-adjusted (Alpaca adjustment=split)."""
    with open(path) as fh:
        res = json.load(fh)["chart"]["result"][0]
    off = int(res.get("meta", {}).get("gmtoffset") or 0)
    q = res["indicators"]["quote"][0]
    out: Dict[str, dict] = {}
    for i, ts in enumerate(res.get("timestamp") or []):
        o, h, l, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, l, c) or c <= 0 or o <= 0:
            continue
        out[_et_day(ts, off)] = {"o": float(o), "h": float(h), "l": float(l), "c": float(c)}
    return out


def load_hourly(path: str) -> Dict[str, List[Tuple[int, float, float]]]:
    """{ET date: [(ET minute-of-day of the bar start, open, close), ...]} from
    a Yahoo 60-minute chart JSON (regular session only)."""
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    with open(path) as fh:
        res = json.load(fh)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    out: Dict[str, List[Tuple[int, float, float]]] = {}
    for i, ts in enumerate(res.get("timestamp") or []):
        o, c = q["open"][i], q["close"][i]
        if o is None or c is None or o <= 0 or c <= 0:
            continue
        t = datetime.fromtimestamp(ts, et)
        out.setdefault(t.strftime("%Y-%m-%d"), []).append((t.hour * 60 + t.minute, float(o), float(c)))
    for v in out.values():
        v.sort()
    return out


def load_cboe(path: str) -> Dict[str, float]:
    """{date: close} from a Cboe daily_prices CSV (DATE,OPEN,HIGH,LOW,CLOSE or
    DATE,<NAME> close-only)."""
    out: Dict[str, float] = {}
    with open(path) as fh:
        head = fh.readline().strip().upper().split(",")
        if not head or head[0] != "DATE":
            return out
        col = head.index("CLOSE") if "CLOSE" in head else len(head) - 1
        for line in fh:
            row = _cboe_row(line.strip().split(","), col)
            if row is not None:
                out[row[0]] = row[1]
    return out


def _cboe_row(c: List[str], col: int) -> Optional[Tuple[str, float]]:
    """(YYYY-MM-DD, close) of one Cboe CSV row, or None for a malformed or
    non-positive row (skipped, never filled)."""
    if len(c) <= col or not re.fullmatch(r"\d{2}/\d{2}/\d{4}", c[0].strip()):
        return None
    m, d, y = c[0].strip().split("/")
    v = _num(c[col])
    return (f"{y}-{m}-{d}", v) if v is not None and v > 0 else None


def _num(x: str) -> Optional[float]:
    s = str(x).strip()
    return float(s) if re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", s) else None


def input_hashes(data_dir: str) -> Dict[str, str]:
    out = {}
    for sub in ("daily", "cboe", "h1"):
        d = os.path.join(data_dir, sub)
        if not os.path.isdir(d):
            continue
        h = hashlib.sha256()
        for f in sorted(os.listdir(d)):
            with open(os.path.join(d, f), "rb") as fh:
                h.update(f.encode() + fh.read())
        out[sub] = h.hexdigest()[:16]
    return out


class Panel:
    """Every daily series aligned to SPY's calendar (dates <= END)."""

    def __init__(self, data_dir: str, start: str = FETCH_START, end: str = END):
        raw = {s: load_daily(os.path.join(data_dir, "daily", f"{s}.json"))
               for s in DAILY_SYMBOLS if os.path.exists(os.path.join(data_dir, "daily", f"{s}.json"))}
        self.dates = [d for d in sorted(raw["SPY"]) if start <= d <= end]
        self.idx = {d: k for k, d in enumerate(self.dates)}
        n = len(self.dates)
        self.sym = sorted(raw)
        self.O: Dict[str, np.ndarray] = {}
        self.C: Dict[str, np.ndarray] = {}
        for s, rows in raw.items():
            o, c = np.full(n, np.nan), np.full(n, np.nan)
            for d, b in rows.items():
                k = self.idx.get(d)
                if k is not None:
                    o[k], c[k] = b["o"], b["c"]
            self.O[s], self.C[s] = o, c
        self.iv: Dict[str, np.ndarray] = {}
        for t in CBOE:
            p = os.path.join(data_dir, "cboe", f"{t}.csv")
            if not os.path.exists(p):
                continue
            v = np.full(n, np.nan)
            for d, x in load_cboe(p).items():
                k = self.idx.get(d)
                if k is not None:
                    v[k] = x
            self.iv[t] = v
        self.day = np.array([date.fromisoformat(d) for d in self.dates])

    def returns(self, s: str) -> np.ndarray:
        c = self.C[s]
        r = np.full(len(c), np.nan)
        r[1:] = np.log(c[1:] / c[:-1])
        return r


# ── per-origin rows (pure; no lookahead) ────────────────────────────────────
def har_rows(close: np.ndarray, h: int) -> dict:
    """Arrays over origins t (master-calendar index) of the HAR regressors and
    the target |log(C_{t+h}/C_t)|. A regressor at t uses returns r[t-21..t]
    only (closes C[t-22..t]); NaN where any input is missing. The target
    NaN where C[t+h] is missing or t+h is past the panel."""
    n = len(close)
    r = np.full(n, np.nan)
    r[1:] = np.log(close[1:] / close[:-1])
    r2 = r * r
    vd = np.abs(r)
    vw = np.full(n, np.nan)
    vm = np.full(n, np.nan)
    rv20 = np.full(n, np.nan)
    for t in range(22, n):
        w = r2[t - 4:t + 1]
        m = r2[t - 21:t + 1]
        if np.all(np.isfinite(m)):
            vw[t] = math.sqrt(float(np.mean(w)))
            vm[t] = math.sqrt(float(np.mean(m)))
            rv20[t] = float(np.std(r[t - 19:t + 1], ddof=1))
    y = np.full(n, np.nan)
    if h < n:
        y[:n - h] = np.abs(np.log(close[h:] / close[:n - h]))
    ok = np.isfinite(vd) & np.isfinite(vw) & np.isfinite(vm)
    vd[~ok] = np.nan
    return {"v_d": vd, "v_w": vw, "v_m": vm, "rv20": rv20, "y": y, "r": r}


def label_index(t: int, h: int) -> int:
    return t + h


def training_mask(origins: np.ndarray, refit_at: int, h: int) -> np.ndarray:
    """Rows a refit at master index `refit_at` may train on: label index
    t + h <= refit_at - h (the label ends >= h sessions before the first
    origin the refit forecasts — purge + an embargo equal to the horizon)."""
    return origins + h <= refit_at - h


def interval_scores(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    return FR.interval_score(q, y, ALPHA)


# ── walk-forward fitting ────────────────────────────────────────────────────
def refit_points(first_eval: int, last_eval: int, every: int = REFIT_EVERY) -> List[int]:
    return list(range(first_eval, last_eval + 1, every))


def walk_forward_q(rows: Dict[str, dict], names: Sequence[str], cols: Sequence[str], h: int,
                   first_eval: int, last_eval: int, iv: Optional[Dict[str, np.ndarray]] = None,
                   need_iv: bool = False) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    """Pooled quantile fit over `names`, refit every REFIT_EVERY sessions.
    Returns ({name: q array over origins}, [{refit_at, theta, n_train}])."""
    n = len(next(iter(rows.values()))["y"])
    X_all, y_all, t_all, s_all = [], [], [], []
    for k, s in enumerate(names):
        R = rows[s]
        feats = [R[c] if c != "v_iv" else iv[s] for c in cols]
        ok = np.isfinite(R["y"]) & np.all([np.isfinite(f) for f in feats], axis=0)
        if need_iv:
            ok &= np.isfinite(iv[s])
        t = np.nonzero(ok)[0]
        t = t[t >= 0]
        X_all.append(np.column_stack([f[t] for f in feats]))
        y_all.append(R["y"][t])
        t_all.append(t)
        s_all.append(np.full(len(t), k))
    X = np.vstack(X_all) if X_all else np.zeros((0, len(cols)))
    Y = np.concatenate(y_all) if y_all else np.zeros(0)
    T = np.concatenate(t_all) if t_all else np.zeros(0, int)
    out = {s: np.full(n, np.nan) for s in names}
    fits = []
    theta = None
    pts = refit_points(first_eval, last_eval)
    for j, at in enumerate(pts):
        upto = pts[j + 1] if j + 1 < len(pts) else last_eval + 1
        tr = training_mask(T, at, h)
        if int(tr.sum()) < MIN_TRAIN:
            continue
        theta = FR.fit_quantile_nonneg(X[tr], Y[tr], TAU, theta0=theta)
        fits.append({"refit_at": at, "theta": [float(v) for v in theta], "n_train": int(tr.sum())})
        for k, s in enumerate(names):
            R = rows[s]
            feats = [R[c] if c != "v_iv" else iv[s] for c in cols]
            seg = np.arange(at, min(upto, n))
            Xs = np.column_stack([f[seg] for f in feats])
            out[s][seg] = Xs @ theta
    return out, fits


def walk_forward_clim(rows: Dict[str, dict], names: Sequence[str], h: int, first_eval: int,
                      last_eval: int) -> Dict[str, np.ndarray]:
    """V_clim: each name's own expanding 68.27% quantile of |y| (same refit
    schedule and embargo as the fitted variants)."""
    n = len(next(iter(rows.values()))["y"])
    out = {s: np.full(n, np.nan) for s in names}
    pts = refit_points(first_eval, last_eval)
    for s in names:
        y = rows[s]["y"]
        t = np.nonzero(np.isfinite(y) & np.isfinite(rows[s]["v_m"]))[0]
        for j, at in enumerate(pts):
            upto = pts[j + 1] if j + 1 < len(pts) else last_eval + 1
            tr = t[training_mask(t, at, h)]
            if len(tr) < MIN_TRAIN // 4:
                continue
            out[s][at:min(upto, n)] = float(np.quantile(y[tr], TAU))
    return out


def v0_q(panel: Panel, rows: Dict[str, dict], names: Sequence[str], h: int,
         ivmap: Optional[Dict[str, str]]) -> Dict[str, np.ndarray]:
    """The current forecast.sigma_for path at horizon h, z = 1: IV * sqrt(cal
    days to the target close / 365) when the name has IV that day, else the
    20-session realized vol * sqrt(h)."""
    n = len(panel.dates)
    out = {}
    cal = np.full(n, np.nan)
    for t in range(n - h):
        cal[t] = max(1, (panel.day[t + h] - panel.day[t]).days)
    for s in names:
        q = rows[s]["rv20"] * math.sqrt(h)
        if ivmap and s in ivmap and ivmap[s] in panel.iv:
            v = panel.iv[ivmap[s]] / 100.0 * np.sqrt(cal / 365.0)
            q = np.where(np.isfinite(v), v, np.nan)
        out[s] = q
    return out


def v1_q(panel: Panel, names: Sequence[str], h: int, first_eval: int, last_eval: int) -> Dict[str, np.ndarray]:
    """The dossier HAR (dossier_ranges, dr-har-ewma-v1) exactly as served: at
    each origin, fit on the name's own last <= 750 sessions, EWMA fallback
    and sanity bounds, z = 1. Uses closes <= t only."""
    import dossier_ranges as DR
    n = len(panel.dates)
    out = {}
    for s in names:
        c = panel.C[s]
        q = np.full(n, np.nan)
        ok = np.nonzero(np.isfinite(c))[0]
        if len(ok) < 300:
            out[s] = q
            continue
        k0 = ok[0]
        cl = c[k0:]
        if not np.all(np.isfinite(cl)):
            # a gap inside the series: use the longest finite tail only
            bad = np.nonzero(~np.isfinite(cl))[0]
            k0 = k0 + bad[-1] + 1
            cl = c[k0:]
        r = np.diff(np.log(cl))
        lam = DR.EWMA_LAMBDA
        ew = np.full(len(r), np.nan)       # ew[k] = DR.ewma_var(r[:k+1])
        if len(r) >= DR.EWMA_SEED_N + 5:
            var = float(np.mean(r[:DR.EWMA_SEED_N] ** 2))
            for k in range(DR.EWMA_SEED_N, len(r)):
                var = lam * var + (1.0 - lam) * float(r[k]) ** 2
                if k >= DR.EWMA_SEED_N + 4:
                    ew[k] = var
        for t in range(max(first_eval, k0 + 30), min(last_eval + 1, n)):
            c_loc = t - k0                  # close index in the local series
            end = c_loc - 1                 # last known return index
            if end < 25:
                continue
            coef = DR.fit_har(r, h, end=end)
            hv = DR.har_var(coef, r, end)
            e = float(ew[end]) if np.isfinite(ew[end]) else None
            v, _, _ = DR.choose_var(hv, e)
            if v:
                q[t] = math.sqrt(v * h)
        out[s] = q
    return out


# ── regimes ─────────────────────────────────────────────────────────────────
def regime_labels(panel: Panel) -> List[Optional[str]]:
    """The bot's 5-level label per day (regime_util.classify_regime_5level),
    fed exactly as P1 fed it: VIX / its 30-day mean for the VXX ratio, SPY /
    its 50-day mean, consecutive days below the 200-day mean."""
    try:
        from regime_util import classify_regime_5level
    except ImportError:
        return [None] * len(panel.dates)
    spy = panel.C["SPY"]
    vix = panel.iv.get("VIX")
    out: List[Optional[str]] = []
    below = 0
    for i in range(len(spy)):
        win = spy[max(0, i - 49):i + 1]
        ma50 = float(np.mean(win))
        ma200 = float(np.mean(spy[i - 199:i + 1])) if i >= 199 else None
        above = (spy[i] > ma200) if ma200 is not None else True
        below = 0 if above else below + 1
        vw = vix[max(0, i - 29):i + 1] if vix is not None else np.array([])
        vw = vw[np.isfinite(vw)]
        vr = (float(vix[i]) / float(np.mean(vw))) if (vix is not None and np.isfinite(vix[i]) and len(vw)) else 1.0
        out.append(classify_regime_5level(vr, float(spy[i]) / ma50, spy_below_200_days=below, spy_above_200d=above))
    return out


# ── bootstrap ───────────────────────────────────────────────────────────────
def block_bootstrap_ratio(num: np.ndarray, den: np.ndarray, block: int = BLOCK, n_boot: int = N_BOOT,
                          seed: int = SEED) -> Tuple[float, float]:
    """95% CI of sum(num)/sum(den) under a moving-block bootstrap over DAYS
    (num/den are per-day sums, so a day's names move together)."""
    D = len(num)
    if D == 0 or den.sum() <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(D / block))
    starts_max = max(1, D - block + 1)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, starts_max, nb)
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:D]
        stats[b] = num[ix].sum() / max(1e-12, den[ix].sum())
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def block_bootstrap_mean_diff(diff_sum: np.ndarray, cnt: np.ndarray, block: int = BLOCK,
                              n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[float, float, float]:
    """(mean, lo95, hi95) of a per-row difference averaged over rows, with the
    same day-block bootstrap; also usable for one-sided p (see p_value_le0)."""
    lo, hi = block_bootstrap_ratio(diff_sum, cnt, block, n_boot, seed)
    return float(diff_sum.sum() / max(1e-12, cnt.sum())), lo, hi


# ── evaluation tables ───────────────────────────────────────────────────────
def segment_range(panel: Panel, seg: Tuple[str, str]) -> Tuple[int, int]:
    lo = next(k for k, d in enumerate(panel.dates) if d >= seg[0])
    hi = max(k for k, d in enumerate(panel.dates) if d <= seg[1])
    return lo, hi


def evaluate(q_by_var: Dict[str, Dict[str, np.ndarray]], rows: Dict[str, dict], names: Sequence[str],
             lo: int, hi: int, stride: int = 1, day_filter: Optional[np.ndarray] = None) -> Dict[str, dict]:
    """Per variant: n, coverage (+CI), mean IS, mean width, on the COMMON rows
    (every variant has a finite q and the target is finite)."""
    D = hi - lo + 1
    common = np.ones((len(names), D), bool)
    for v, qs in q_by_var.items():
        for k, s in enumerate(names):
            common[k] &= np.isfinite(qs[s][lo:hi + 1]) & (qs[s][lo:hi + 1] > 0)
    for k, s in enumerate(names):
        common[k] &= np.isfinite(rows[s]["y"][lo:hi + 1])
    if stride > 1:
        keep = np.zeros(D, bool)
        keep[::stride] = True
        common &= keep[None, :]
    if day_filter is not None:
        common &= day_filter[None, :]
    out = {}
    for v, qs in q_by_var.items():
        cov_d = np.zeros(D)
        is_d = np.zeros(D)
        w_d = np.zeros(D)
        n_d = common.sum(0).astype(float)
        for k, s in enumerate(names):
            m = common[k]
            q = np.where(m, qs[s][lo:hi + 1], 0.0)
            y = np.where(m, rows[s]["y"][lo:hi + 1], 0.0)
            cov_d += np.where(m, (y <= q).astype(float), 0.0)
            is_d += np.where(m, interval_scores(q, y), 0.0)
            w_d += np.where(m, 2.0 * q, 0.0)
        n = n_d.sum()
        ci = block_bootstrap_ratio(cov_d, n_d)
        out[v] = {"n": int(n), "coverage": float(cov_d.sum() / n) if n else float("nan"),
                  "coverage_ci": ci, "is": float(is_d.sum() / n) * 100 if n else float("nan"),
                  "width": float(w_d.sum() / n) * 100 if n else float("nan"),
                  "_is_d": is_d, "_n_d": n_d}
    return out


def diff_vs(res: Dict[str, dict], a: str, b: str) -> Tuple[float, float, float]:
    """mean IS(a) - IS(b) in percent, day-block bootstrap 95% CI."""
    d = (res[a]["_is_d"] - res[b]["_is_d"]) * 100
    return block_bootstrap_mean_diff(d, res[a]["_n_d"])


# ── intraday (60-minute) sub-study ──────────────────────────────────────────
def intraday_obs(panel: Panel, data_dir: str, sym: str) -> List[dict]:
    """Per complete regular session D of `sym` (7 hourly bars): the official
    daily closes C_{D-1}, C_D and the price at each mark (09:30 = the first
    bar's open; 10:30.. 15:30 = the previous bar's close)."""
    p = os.path.join(data_dir, "h1", f"{sym}.json")
    if not os.path.exists(p):
        return []
    hb = load_hourly(p)
    out = []
    c = panel.C[sym]
    for d, bars in hb.items():
        k = panel.idx.get(d)
        if k is None or k == 0 or not np.isfinite(c[k]) or not np.isfinite(c[k - 1]):
            continue
        starts = [b[0] for b in bars]
        if starts[:7] != [m[0] * 60 + m[1] for m in MARKS]:
            continue
        prices = [bars[0][1]] + [bars[j][2] for j in range(6)]
        out.append({"k": k, "date": d, "c0": float(c[k - 1]), "c": float(c[k]), "s": prices})
    out.sort(key=lambda o: o["k"])
    return out


def remaining_fraction(obs_by_sym: Dict[str, List[dict]], syms: Sequence[str]) -> List[float]:
    """f(mark) = mean over names of [sum_D (log C_D/S_mark)^2 / sum_D (log C_D/C_{D-1})^2]."""
    per = []
    for s in syms:
        ob = obs_by_sym.get(s) or []
        if len(ob) < 50:
            continue
        den = sum(math.log(o["c"] / o["c0"]) ** 2 for o in ob)
        if den <= 0:
            continue
        per.append([sum(math.log(o["c"] / o["s"][j]) ** 2 for o in ob) / den for j in range(len(MARKS))])
    if not per:
        return [float("nan")] * len(MARKS)
    return [float(np.mean([p[j] for p in per])) for j in range(len(MARKS))]


# ── main study ──────────────────────────────────────────────────────────────
def _fmt(x: float, nd: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _cov(r: dict) -> str:
    lo, hi = r["coverage_ci"]
    return f"{r['coverage'] * 100:.1f}% [{_fmt(lo * 100, 1)}, {_fmt(hi * 100, 1)}]"


def run_range(data_dir: str) -> dict:
    t_start = time.time()
    P = Panel(data_dir)
    n = len(P.dates)
    first = next(k for k, d in enumerate(P.dates) if d >= DATA_START)
    sel = segment_range(P, SELECTION)
    hold = segment_range(P, HOLDOUT)
    res: dict = {"spec": {"end": END, "data_start": DATA_START, "selection": SELECTION, "holdout": HOLDOUT,
                          "refit_every": REFIT_EVERY, "min_train": MIN_TRAIN, "tau": TAU, "block": BLOCK,
                          "n_boot": N_BOOT, "seed": SEED},
                 "inputs": input_hashes(data_dir), "groups": {}, "coef_final": {}, "intraday": {}}
    ivmap_all = {**INDEX_IV, **STOCK_IV, **OTHER_IV}
    # restrict every series to >= DATA_START for training/evaluation (features may look back)
    for s in list(P.C):
        P.C[s][:max(0, first - 22)] = np.nan   # first valid origin (22 returns) is DATA_START
    labels = regime_labels(P)
    vix = P.iv["VIX"]
    v_sel = vix[sel[0]:sel[1] + 1]
    cuts = tuple(float(x) for x in np.quantile(v_sel[np.isfinite(v_sel)], [1 / 3, 2 / 3]))
    res["vix_tercile_cuts"] = cuts
    iv_daily = {s: P.iv[t] / 100.0 / math.sqrt(252.0) for s, t in ivmap_all.items() if t in P.iv}
    chosen_q1: Dict[str, np.ndarray] = {}
    for h in HORIZONS:
        rows = {s: har_rows(P.C[s], h) for s in HAR_PANEL if s in P.C}
        groups = {
            "index": (tuple(INDEX_IV), INDEX_IV),
            "stock_iv": (tuple(STOCK_IV), STOCK_IV),
            "har": (tuple(s for s in HAR_PANEL if s in rows), None),
        }
        for g, (names, ivm) in groups.items():
            last_eval = hold[1]
            # first evaluated origin: SELECTION start, or later until MIN_TRAIN rows exist
            q_by: Dict[str, Dict[str, np.ndarray]] = {}
            fits = {}
            first_eval = sel[0]
            if ivm:
                q_by["V2"], fits["V2"] = walk_forward_q(rows, names, ["v_iv"], h, first_eval, last_eval, iv_daily, True)
                q_by["V3"], fits["V3"] = walk_forward_q(rows, names, ["v_d", "v_w", "v_m"], h, first_eval, last_eval, iv_daily, True)
                q_by["V4"], fits["V4"] = walk_forward_q(rows, names, ["v_d", "v_w", "v_m", "v_iv"], h, first_eval, last_eval, iv_daily, True)
            else:
                q_by["V3"], fits["V3"] = walk_forward_q(rows, names, ["v_d", "v_w", "v_m"], h, first_eval, last_eval)
            for v, f in fits.items():
                _THETA_CACHE[f"{g}_h{h}_{v}"] = f
            q_by["V0"] = v0_q(P, rows, names, h, ivm)
            q_by["V1"] = v1_q(P, names, h, first_eval, last_eval)
            q_by["V_clim"] = walk_forward_clim(rows, names, h, first_eval, last_eval)
            stride = 5 if h == 5 else 1
            ev_sel = evaluate(q_by, rows, names, sel[0], sel[1], stride)
            ev_hold = evaluate(q_by, rows, names, hold[0], hold[1], stride)
            cands = [v for v in ("V2", "V3", "V4") if v in ev_sel]
            ranked = sorted(cands, key=lambda v: ev_sel[v]["is"])
            chosen = next((v for v in ranked if abs(ev_sel[v]["coverage"] - TAU) <= 0.03), ranked[0])
            d_v0 = diff_vs(ev_hold, chosen, "V0")
            d_clim = diff_vs(ev_hold, chosen, "V_clim")
            ship = (d_v0[2] < 0) and abs(ev_hold[chosen]["coverage"] - TAU) <= 0.05
            key = f"{g}_h{h}"
            gres = {"names": list(names), "h": h, "chosen": chosen, "ship_rule_R2": bool(ship),
                    "d_is_vs_v0_holdout": d_v0, "d_is_vs_clim_holdout": d_clim,
                    "selection": {v: {k: x for k, x in r.items() if not k.startswith("_")} for v, r in ev_sel.items()},
                    "holdout": {v: {k: x for k, x in r.items() if not k.startswith("_")} for v, r in ev_hold.items()},
                    "fits_last": {v: f[-1] for v, f in fits.items() if f},
                    "fits_first": {v: f[0] for v, f in fits.items() if f},
                    "n_refits": {v: len(f) for v, f in fits.items()},
                    "theta_path": {v: [x["theta"] for x in f[::12]] for v, f in fits.items()}}
            # pairwise IS differences on the holdout (all vs V0 / vs chosen)
            gres["d_is_holdout"] = {v: diff_vs(ev_hold, v, "V0") for v in ev_hold if v != "V0"}
            gres["d_is_selection_vs_v3"] = {v: diff_vs(ev_sel, v, "V3") for v in ("V2", "V4") if v in ev_sel}
            # regime splits (holdout, chosen + V0 + V_clim)
            D = hold[1] - hold[0] + 1
            vix_h = vix[hold[0]:hold[1] + 1]
            terc = {"low": vix_h < cuts[0], "mid": (vix_h >= cuts[0]) & (vix_h < cuts[1]), "high": vix_h >= cuts[1]}
            lab = labels[hold[0]:hold[1] + 1]
            regs = {f"VIX {k}": v for k, v in terc.items()}
            for L in ("BULL", "NEUTRAL", "CAUTION", "BEAR", "PANIC"):
                regs[L] = np.array([x == L for x in lab])
            wd = np.array([P.day[hold[0] + j].weekday() for j in range(D)])
            for k2, nm in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri")):
                regs[nm] = wd == k2
            sub = {v: q_by[v] for v in (chosen, "V0", "V_clim", "V1")}
            gres["splits_holdout"] = {}
            for rname, mask in regs.items():
                ev = evaluate(sub, rows, names, hold[0], hold[1], stride, mask)
                gres["splits_holdout"][rname] = {v: {"n": r["n"], "coverage": r["coverage"], "coverage_ci": r["coverage_ci"],
                                                     "is": r["is"], "width": r["width"]} for v, r in ev.items()}
            # per-name holdout (chosen vs V0)
            gres["per_name_holdout"] = {}
            for s in names:
                ev = evaluate({v: {s: q_by[v][s]} for v in (chosen, "V0", "V_clim")}, rows, [s], hold[0], hold[1], stride)
                gres["per_name_holdout"][s] = {v: {"n": r["n"], "coverage": r["coverage"], "is": r["is"], "width": r["width"]}
                                               for v, r in ev.items()}
            res["groups"][key] = gres
            if fits.get(chosen):
                res["coef_final"][key] = {"variant": chosen, **fits[chosen][-1]}
            for v in ("V3", "V4"):
                if fits.get(v):
                    res["coef_final"][f"{key}_{v}"] = {"variant": v, **fits[v][-1]}
            if h == 1:
                for s in names:
                    if g == "har" and s in chosen_q1:
                        continue
                    chosen_q1.setdefault(s, q_by[chosen][s])
            print(f"[{time.time() - t_start:6.1f}s] {key}: chosen {chosen}, holdout cov "
                  f"{ev_hold[chosen]['coverage']:.4f}, IS {ev_hold[chosen]['is']:.4f} vs V0 {ev_hold['V0']['is']:.4f}", flush=True)
        # transfer check: other ETFs with their own IV, index/stock-IV thetas
        if h == 1:
            names = tuple(s for s in OTHER_IV if s in rows)
            q_by = {"V0": v0_q(P, rows, names, h, OTHER_IV), "V_clim": walk_forward_clim(rows, names, h, sel[0], hold[1])}
            for src in ("index", "stock_iv"):
                q_by[f"V4[{src}]"] = _apply_theta_path(rows, names, iv_daily, src, P, h, hold[1])
            q_by["V3[har]"] = _apply_har_theta(rows, names, h, hold[1])
            ev = evaluate(q_by, rows, names, hold[0], hold[1])
            res["transfer_other_etf_h1"] = {v: {k: x for k, x in r.items() if not k.startswith("_")} for v, r in ev.items()}
            res["transfer_other_etf_h1_d_vs_v0"] = {v: diff_vs(ev, v, "V0") for v in ev if v != "V0"}
    # intraday sub-study
    res["intraday"] = run_intraday(P, data_dir, chosen_q1)
    res["elapsed_s"] = round(time.time() - t_start, 1)
    return res


_THETA_CACHE: Dict[str, List[dict]] = {}


def _apply_theta_path(rows, names, iv_daily, src, P, h, last_eval):
    """Apply a source group's walk-forward V4 thetas (same refit dates) to
    other names — the transfer check. Uses the stored per-refit thetas."""
    path = _THETA_CACHE.get(f"{src}_h{h}_V4")
    n = len(P.dates)
    out = {s: np.full(n, np.nan) for s in names}
    if not path:
        return out
    for j, f in enumerate(path):
        at = f["refit_at"]
        upto = path[j + 1]["refit_at"] if j + 1 < len(path) else last_eval + 1
        th = np.array(f["theta"])
        for s in names:
            seg = np.arange(at, min(upto, n))
            X = np.column_stack([rows[s]["v_d"][seg], rows[s]["v_w"][seg], rows[s]["v_m"][seg], iv_daily[s][seg]])
            out[s][seg] = X @ th
    return out


def _apply_har_theta(rows, names, h, last_eval):
    path = _THETA_CACHE.get(f"har_h{h}_V3")
    n = len(next(iter(rows.values()))["y"])
    out = {s: np.full(n, np.nan) for s in names}
    if not path:
        return out
    for j, f in enumerate(path):
        at = f["refit_at"]
        upto = path[j + 1]["refit_at"] if j + 1 < len(path) else last_eval + 1
        th = np.array(f["theta"])
        for s in names:
            seg = np.arange(at, min(upto, n))
            X = np.column_stack([rows[s]["v_d"][seg], rows[s]["v_w"][seg], rows[s]["v_m"][seg]])
            out[s][seg] = X @ th
    return out


def _std_moves(obs_by_sym: Dict[str, List[dict]], syms: Sequence[str], q1: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Per (name, session) with a pre-open band q made at the D-1 close: the
    full-day move |log C_D/C_{D-1}| / q and, per mark, |log C_D/S_mark| / q."""
    full, marks = [], []
    for s in syms:
        q = q1.get(s)
        if q is None:
            continue
        for o in obs_by_sym.get(s, []):
            qd = q[o["k"] - 1]
            if not np.isfinite(qd) or qd <= 0:
                continue
            full.append(abs(math.log(o["c"] / o["c0"])) / qd)
            marks.append([abs(math.log(o["c"] / o["s"][j])) / qd for j in range(len(MARKS))])
    return np.asarray(full), np.asarray(marks).reshape(-1, len(MARKS))


def quantile_fraction(full: np.ndarray, marks: np.ndarray) -> List[float]:
    """POST-HOC variant (declared in the doc): f(mark) = [Q68(|C/S_mark| / q) /
    Q68(|C/C_prev| / q)]^2 — the shrinkage of the band's own quantile, not of
    the variance (a variance ratio is driven by a few event hours; the band is
    a quantile). A running minimum keeps it non-increasing through the day."""
    if len(full) < 50:
        return [float("nan")] * len(MARKS)
    base = float(np.quantile(full, TAU))
    g = [float(np.quantile(marks[:, j], TAU)) / base for j in range(marks.shape[1])]
    out, cur = [], float("inf")
    for v in g:
        cur = min(cur, v * v)
        out.append(cur)
    return out


def _live_coverage(obs_by_sym: Dict[str, List[dict]], syms: Sequence[str], q1: Dict[str, np.ndarray],
                   f: Sequence[float]) -> List[dict]:
    rows = []
    for j in range(len(MARKS)):
        hit = tot = 0
        for s in syms:
            q = q1.get(s)
            if q is None:
                continue
            for o in obs_by_sym.get(s, []):
                qd = q[o["k"] - 1]          # the band made at the D-1 close
                if not np.isfinite(qd) or qd <= 0 or not np.isfinite(f[j]):
                    continue
                tot += 1
                hit += int(abs(math.log(o["c"] / o["s"][j])) <= qd * math.sqrt(max(0.0, f[j])))
        rows.append({"mark": f"{MARKS[j][0]:02d}:{MARKS[j][1]:02d}", "n": tot, "coverage": hit / tot if tot else None})
    return rows


def run_intraday(P: Panel, data_dir: str, q1: Dict[str, np.ndarray]) -> dict:
    obs = {s: intraday_obs(P, data_dir, s) for s in INTRADAY_SYMBOLS}
    all_k = sorted({o["k"] for v in obs.values() for o in v})
    if not all_k:
        return {"error": "no 60-minute data"}
    mid = all_k[len(all_k) // 2]
    first = {s: [o for o in v if o["k"] < mid] for s, v in obs.items()}
    second = {s: [o for o in v if o["k"] >= mid] for s, v in obs.items()}
    groups = {"index": INDEX_GROUP_INTRADAY, "other": tuple(s for s in INTRADAY_SYMBOLS if s not in INDEX_GROUP_INTRADAY)}
    out: dict = {"sessions": len(all_k), "split_date": P.dates[mid], "first_date": P.dates[all_k[0]],
                 "last_date": P.dates[all_k[-1]], "marks": [f"{h:02d}:{m:02d}" for h, m in MARKS], "groups": {}}
    for g, syms in groups.items():
        f1 = remaining_fraction(first, syms)                       # PRE-REGISTERED: variance ratio
        f_full = remaining_fraction(obs, syms)
        q_first = quantile_fraction(*_std_moves(first, syms, q1))  # POST-HOC: quantile ratio
        q_full = quantile_fraction(*_std_moves(obs, syms, q1))
        full2, _ = _std_moves(second, syms, q1)
        out["groups"][g] = {
            "names": [s for s in syms if obs.get(s)],
            "f_first_half": f1, "f_full": f_full,
            "coverage_second_half": _live_coverage(second, syms, q1, f1),
            "fq_first_half": q_first, "fq_full": q_full,
            "coverage_second_half_q": _live_coverage(second, syms, q1, q_first),
            "preopen_coverage_second_half": {"n": int(len(full2)),
                                             "coverage": float(np.mean(full2 <= 1.0)) if len(full2) else None},
        }
    return out


def render(res: dict) -> str:  # filled by render_md below (kept separate for tests)
    return render_md(res)


def render_md(res: dict) -> str:
    L: List[str] = []
    A = L.append
    A(f"Inputs (sha256/16 of each data folder): {res['inputs']}. Elapsed {res.get('elapsed_s')} s.")
    A("")
    A(f"VIX tercile cut points (SELECTION segment): {res['vix_tercile_cuts'][0]:.2f} / {res['vix_tercile_cuts'][1]:.2f}.")
    for key, g in res["groups"].items():
        A("")
        A(f"### {key} — names: {', '.join(g['names'])}")
        A("")
        A(f"Chosen on SELECTION (rule R1): **{g['chosen']}**. Ship rule R2 on HOLDOUT: **{'PASS' if g['ship_rule_R2'] else 'FAIL'}** "
          f"(dIS vs V0 {g['d_is_vs_v0_holdout'][0]:+.4f} [{g['d_is_vs_v0_holdout'][1]:+.4f}, {g['d_is_vs_v0_holdout'][2]:+.4f}] pct-pts; "
          f"IS skill vs V_clim {(-g['d_is_vs_clim_holdout'][0] / g['holdout']['V_clim']['is'] * 100):+.1f}%).")
        for segname in ("selection", "holdout"):
            A("")
            A(f"{segname.upper()} | n | coverage [95% CI] | mean IS (%) | mean width (%)")
            A("|---|---|---|---|---|")
            for v, r in g[segname].items():
                A(f"| {v} | {r['n']} | {_cov(r)} | {_fmt(r['is'], 4)} | {_fmt(r['width'], 3)} |")
        A("")
        A("dIS vs V0 on HOLDOUT (negative = better than V0): " +
          "; ".join(f"{v} {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" for v, d in g["d_is_holdout"].items()))
        if g.get("d_is_selection_vs_v3"):
            A("")
            A("dIS vs V3 on SELECTION: " + "; ".join(f"{v} {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" for v, d in g["d_is_selection_vs_v3"].items()))
        A("")
        A("Final thetas (last refit): " + "; ".join(f"{v}: {', '.join(f'{x:.4f}' for x in f['theta'])} (n_train {f['n_train']})" for v, f in g["fits_last"].items()))
        A("")
        A(f"HOLDOUT splits | {g['chosen']} cov | {g['chosen']} IS | V0 cov | V0 IS | V_clim cov | V1 cov | n")
        A("|---|---|---|---|---|---|---|---|")
        for rn, sp in g["splits_holdout"].items():
            c = sp[g["chosen"]]
            A(f"| {rn} | {c['coverage'] * 100:.1f}% | {_fmt(c['is'], 4)} | {sp['V0']['coverage'] * 100:.1f}% | {_fmt(sp['V0']['is'], 4)} | "
              f"{sp['V_clim']['coverage'] * 100:.1f}% | {sp['V1']['coverage'] * 100:.1f}% | {c['n']} |" if c['n'] else f"| {rn} | — | — | — | — | — | — | 0 |")
        A("")
        A(f"HOLDOUT per name | {g['chosen']} cov | {g['chosen']} IS | V0 cov | V0 IS | V_clim IS | n")
        A("|---|---|---|---|---|---|---|")
        for s, pr in g["per_name_holdout"].items():
            c = pr[g["chosen"]]
            if not c["n"]:
                continue
            A(f"| {s} | {c['coverage'] * 100:.1f}% | {_fmt(c['is'], 4)} | {pr['V0']['coverage'] * 100:.1f}% | {_fmt(pr['V0']['is'], 4)} | {_fmt(pr['V_clim']['is'], 4)} | {c['n']} |")
    if res.get("transfer_other_etf_h1"):
        A("")
        A("### Transfer check (h = 1, HOLDOUT): GLD USO EEM SLV EWZ FXI with their own Cboe vol index")
        A("")
        A("| variant | n | coverage [95% CI] | mean IS (%) | width (%) | dIS vs V0 |")
        A("|---|---|---|---|---|---|")
        for v, r in res["transfer_other_etf_h1"].items():
            d = res["transfer_other_etf_h1_d_vs_v0"].get(v)
            A(f"| {v} | {r['n']} | {_cov(r)} | {_fmt(r['is'], 4)} | {_fmt(r['width'], 3)} | {('%+.4f [%+.4f, %+.4f]' % d) if d else '—'} |")
    it = res.get("intraday") or {}
    if it.get("groups"):
        A("")
        A(f"### Intraday live band (60-minute bars, {it['sessions']} sessions {it['first_date']} → {it['last_date']}; f fit before {it['split_date']}, coverage after)")
        for g, v in it["groups"].items():
            A("")
            A(f"{g} ({', '.join(v['names'])}): pre-open band coverage on the same sessions "
              f"{(v['preopen_coverage_second_half']['coverage'] or float('nan')) * 100:.1f}% (n {v['preopen_coverage_second_half']['n']})")
            A("")
            A("| mark (ET) | PRE-REG f (variance ratio) 1st half | f full | live coverage 2nd half | POST-HOC f (quantile ratio) 1st half | f full | live coverage 2nd half | n |")
            A("|---|---|---|---|---|---|---|---|")
            for j, c in enumerate(v["coverage_second_half"]):
                cq = v["coverage_second_half_q"][j]
                A(f"| {c['mark']} | {v['f_first_half'][j]:.3f} | {v['f_full'][j]:.3f} | "
                  f"{(c['coverage'] * 100) if c['coverage'] is not None else float('nan'):.1f}% | "
                  f"{v['fq_first_half'][j]:.3f} | {v['fq_full'][j]:.3f} | "
                  f"{(cq['coverage'] * 100) if cq['coverage'] is not None else float('nan'):.1f}% | {c['n']} |")
    return "\n".join(L) + "\n"


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [_json_safe(v) for v in o.tolist()]
    return o


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("fetch", "range"))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch_all(a.data_dir)
        print(json.dumps(input_hashes(a.data_dir)))
        return 0
    res = run_range(a.data_dir)
    md = render_md(res)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(md)
    if a.json:
        slim = {k: v for k, v in res.items() if k != "groups"}
        slim["groups"] = {k: {kk: vv for kk, vv in g.items() if kk not in ("theta_path",)} for k, g in res["groups"].items()}
        with open(a.json, "w") as fh:
            json.dump(_json_safe(slim), fh, indent=1, sort_keys=True)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
