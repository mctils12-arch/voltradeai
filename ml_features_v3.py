"""ml_features_v3.py — ONE point-in-time feature definition for the v3
CANDIDATE stock model, used by training AND serving.

[REPAIR] 2026-10-27 (audit 2026-10-27 §4.1; human Phase 4 "Fix the model").

WHAT WAS BROKEN in the live v2 vector (ml_model_v2.FEATURE_COLS):
  * 9 of 34 features are constant or dead at serve time (price_vs_52w_high
    and range_pct always 0, ewma_vol always 2, earnings_surprise/intel_score/
    news_sentiment/insider_signal/days_to_earnings constant, cross_sec_rank
    probably 0.5) and trend_strength is distorted by the constant ewma_vol.
  * Training (_compute_features) and serving (bot_engine.deep_score's
    ml_features dict) compute the SAME NAME differently: rsi_14, adx,
    change_pct_today (full day vs intraday partial), volume_ratio (today/20d
    vs 5d/90d), idiosyncratic_ret (21d beta-adjusted vs today minus a key
    macro never emits), sector_momentum (stock-vs-SPY 5d vs sector ETF 1d),
    vrp / iv_rank_proxy / put_call_proxy (market-wide VXX vs per-stock IV),
    iv_rank_stock (duplicate), market_breadth (universe breadth vs SPY proxy).

THE FIX: a single pure function computes every feature from completed daily
bars (and completed reference bars) — training calls it at every historical
index, serving calls it at the last completed bar after stripping the
in-progress session. Train/serve parity holds by construction and is pinned
by test_ml_features_v3.py (same fixture through both paths -> identical
rows). Every feature has a FIXED lookback (no infinite-memory smoothing), so
a serve-time fetch of >= MIN_BARS bars reproduces the training value exactly.

DECISION TIME: after the close of the last completed bar i (serve: the
session before today). Nothing in bar i+1 or later is read. The v3 label
(ml_model_v3) enters at the NEXT session's open, which is the honest
analogue of the bot entering intraday the following session.

Features the v3 model DROPS (no point-in-time source on both paths) are in
V2_FEATURE_DISPOSITION with the reason. Live v2 serving is NOT changed by
this module — the v2 model keeps its inputs until a promotion decision.

Pure: numpy + stdlib + live_exit_policy (pure). No I/O, no network.
"""
from __future__ import annotations

import bisect
import logging
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from live_exit_policy import wilder_atr

logger = logging.getLogger("ml_features_v3")

FEATURE_SET_VERSION = "v3.0"

V3_FEATURE_COLS: List[str] = [
    # stock, price/technical
    "mom_1m",             # 21-bar return, %
    "mom_3m",             # 63-bar return, %
    "mom_12_1",           # Jegadeesh-Titman: return from i-252 to i-21, %
    "rsi_14",             # Wilder RSI(14) over a fixed 100-change window
    "price_vs_52w_high",  # close vs max HIGH of the last 252 bars, % (<= 0)
    "range_pct",          # (high-low)/close of bar i, %
    "change_pct_last",    # close-to-close change of bar i (a COMPLETED day), %
    "close_above_open",   # 1 if bar i closed above its open
    "above_ma10",         # 1 if close_i > mean of the 10 prior closes
    "adx",                # Wilder ADX(14) over a fixed 100-bar window
    "frac_diff_price",    # FFD log price, d=0.4, last 20 closes (ml_model_v2._frac_diff)
    # stock, volume / volatility
    "volume_ratio",       # volume_i / mean of the 20 prior volumes, capped 10
    "ewma_vol",           # EWMA (lambda 0.94) std of daily log returns, fixed 100-return window, %
    "atr_pct",            # Wilder ATR(14) on the last 42 bars (bot_engine._get_atr's window) / close, %
    "hv_rank",            # percentile of today's 20d realized vol within its last 200 values, 0-100
    "trend_strength",     # min(|change_pct_last| / ewma_vol, 5)
    # stock vs market
    "idiosyncratic_ret",  # 21d return minus OLS-beta(60d) x SPY 21d return, %
    "rel_strength_5d",    # 5d return minus SPY 5d return, %
    # market (reference series as of the stock's bar date)
    "vxx_ratio",          # VXX close / mean of 30 prior closes
    "spy_vs_ma50",        # SPY close / mean of 50 prior closes
    "spy_ret_5d",         # mean of the last 5 SPY daily returns, %
    "vxx_rank_252",       # VXX close rank within its prior 252 closes, 0-100
    "vol_of_vol",         # std of the last 10 VXX daily returns, %
    "credit_spread",      # TLT 21d return minus HYG 21d return
]

REF_SYMBOLS = ("SPY", "VXX", "TLT", "HYG")

# Longest lookback any feature reads (mom_12_1 reads close[i-252]); a row is
# computed only when i >= MAX_LOOKBACK, so every feature uses a full window.
MAX_LOOKBACK = 252
MIN_BARS = MAX_LOOKBACK + 1
SMOOTH_WINDOW = 100          # RSI / ADX / EWMA fixed windows
EWMA_LAMBDA = 0.94           # RiskMetrics daily decay
HV_WINDOW, HV_RANK_WINDOW = 20, 200
ATR_BARS = 42                # bot_engine._get_atr: limit = period * 3
BETA_WINDOW = 60
FRAC_D, FRAC_WIDTH = 0.4, 20
MIN_PRICE = 2.0              # training-row eligibility (illiquid/penny rows excluded)
MIN_AVG_VOLUME = 300_000     # same floor as ml_model_v2._compute_features

# Every v2 feature, what v3 does with it, and why (audit §4.1). "fixed" =
# kept with ONE definition both paths share; "replaced" = a v3 feature with
# a different (honest) name carries the information; "dropped" = no PIT
# source on both paths.
V2_FEATURE_DISPOSITION: Dict[str, str] = {
    "momentum_1m": "fixed -> mom_1m (one 21-bar definition)",
    "momentum_3m": "fixed -> mom_3m (train used 63 bars, serve 62)",
    "rsi_14": "fixed -> rsi_14 (train: simple average excluding today; serve: analyze.compute_rsi)",
    "price_vs_52w_high": "fixed (serve was always 0: quick_result never sets high_52w)",
    "volume_ratio": "fixed (train today/20d; serve analyze 5d/90d)",
    "ewma_vol": "fixed (train was 20d std, serve always 2: analyze emits ewma_rv not ewma_vol)",
    "atr_pct": "fixed (one Wilder ATR on bot_engine._get_atr's 42-bar window)",
    "vrp": "dropped (train: market-wide f(vxx_ratio), serve: per-stock IV-RV; no PIT per-stock IV history)",
    "iv_rank_proxy": "replaced -> vxx_rank_252 (it was always the VXX rank in training)",
    "vwap_position": "replaced -> close_above_open (it was close>open, never VWAP)",
    "adx": "fixed (train: unsmoothed DM ratio; serve: Wilder DX, not ADX)",
    "range_pct": "fixed (serve was hard-coded 0 in quick_result)",
    "vxx_ratio": "fixed (reference bars, same formula)",
    "spy_vs_ma50": "fixed (reference bars, same formula)",
    "markov_state": "replaced -> spy_ret_5d (it was a 3-level bucketing of this mean)",
    "regime_score": "dropped (closed-form function of spy_vs_ma50 and vxx_ratio: redundant)",
    "sector_momentum": "replaced -> rel_strength_5d (train: stock-vs-SPY 5d; serve: sector ETF 1d)",
    "cross_sec_rank": "dropped (serve has no universe cross-section: most-actives carries no 'change')",
    "earnings_surprise": "dropped (look-ahead: latest surprise applied to all history; no PIT report dates)",
    "put_call_proxy": "dropped (bucketing of the dropped vrp)",
    "vol_of_vol": "fixed (reference bars, same formula)",
    "frac_diff_price": "fixed (train excluded today's close, serve included it)",
    "idiosyncratic_ret": "fixed (real 60d OLS beta; serve used a macro key that is never emitted)",
    "change_pct_today": "replaced -> change_pct_last (last COMPLETED day; serve was an intraday partial)",
    "above_ma10": "fixed (train: prior 10 closes; serve: last 10 incl. today)",
    "trend_strength": "fixed (serve divided by the constant ewma_vol=2)",
    "intel_score": "dropped (constant 0 in training and at serve)",
    "news_sentiment": "dropped (constant 0 in training and at serve; no PIT news history built)",
    "insider_signal": "dropped (constant 0 in training and at serve)",
    "iv_rank_stock": "replaced -> hv_rank (serve was per-stock HV rank, train VXX rank x volume)",
    "days_to_earnings": "dropped (constant 0.5 in training and at serve; no PIT earnings calendar)",
    "credit_spread": "fixed (reference bars, same formula)",
    "market_breadth": "dropped (serve has no universe MA50 breadth; its fallback is f(spy_vs_ma50))",
    "put_call_ratio": "dropped (VXX price level / 20: non-stationary level of a decaying ETN)",
}


# ── Bar arrays ───────────────────────────────────────────────────────────────

def bar_date(bar: dict) -> str:
    return str(bar.get("t") or "")[:10]


def to_arrays(bars: Sequence[dict]) -> Dict[str, np.ndarray]:
    """Sorted, de-duplicated (last wins), positive-price bars as arrays."""
    by_date: Dict[str, dict] = {}
    for b in bars or []:
        d = bar_date(b)
        if len(d) != 10:
            continue
        try:
            o, h, lo, c = (float(b.get(k) or 0) for k in ("o", "h", "l", "c"))
            v = float(b.get("v") or 0)
        except (TypeError, ValueError) as e:
            logger.debug(f"skipping malformed bar {d}: {e}")
            continue
        if min(o, h, lo, c) <= 0:
            continue
        by_date[d] = {"o": o, "h": max(h, o, c, lo), "l": min(lo, o, c, h), "c": c, "v": max(v, 0.0)}
    dates = sorted(by_date)
    return {
        "t": np.array(dates, dtype=object),
        "o": np.array([by_date[d]["o"] for d in dates], dtype=np.float64),
        "h": np.array([by_date[d]["h"] for d in dates], dtype=np.float64),
        "l": np.array([by_date[d]["l"] for d in dates], dtype=np.float64),
        "c": np.array([by_date[d]["c"] for d in dates], dtype=np.float64),
        "v": np.array([by_date[d]["v"] for d in dates], dtype=np.float64),
    }


def strip_incomplete(bars: Sequence[dict], today_et: str) -> List[dict]:
    """Serve-time: drop the in-progress session (and anything dated later), so
    the last bar is the last COMPLETED session — the training decision point."""
    return [b for b in (bars or []) if bar_date(b) and bar_date(b) < today_et]


def asof_closes(ref_dates: List[str], ref_closes: np.ndarray, dates: Sequence[str]) -> np.ndarray:
    """Reference close as of each date (last ref bar dated <= it); NaN before
    the reference series starts."""
    out = np.full(len(dates), np.nan)
    for k, d in enumerate(dates):
        j = bisect.bisect_right(ref_dates, d) - 1
        if j >= 0:
            out[k] = ref_closes[j]
    return out


# ── Primitive indicators (fixed windows) ────────────────────────────────────

def _pct(a: float, b: float) -> float:
    return (a / b - 1.0) * 100.0 if b > 0 else math.nan


def _wilder_rsi(closes: np.ndarray, period: int = 14) -> float:
    d = np.diff(closes)
    if len(d) < period + 1:
        return math.nan
    gains = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    ag = float(gains[:period].mean())
    al = float(losses[:period].mean())
    for g, l in zip(gains[period:], losses[period:]):
        ag = (ag * (period - 1) + g) / period
        al = (al * (period - 1) + l) / period
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def _wilder_adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> float:
    n = len(c)
    if n < 2 * period + 2:
        return math.nan
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    atr = float(tr[:period].sum())
    p = float(pdm[:period].sum())
    m = float(mdm[:period].sum())
    dxs = []
    for k in range(period, len(tr) + 1):
        if k > period:
            atr = atr - atr / period + tr[k - 1]
            p = p - p / period + pdm[k - 1]
            m = m - m / period + mdm[k - 1]
        pdi = 100.0 * p / atr if atr > 0 else 0.0
        mdi = 100.0 * m / atr if atr > 0 else 0.0
        dxs.append(100.0 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) > 0 else 0.0)
    if len(dxs) < period:
        return math.nan
    adx = float(np.mean(dxs[:period]))
    for dx in dxs[period:]:
        adx = (adx * (period - 1) + dx) / period
    return adx


def _ewma_vol(closes: np.ndarray) -> float:
    r = np.diff(np.log(closes))
    if len(r) < 21:
        return math.nan
    var = float(np.var(r[:20]))
    for x in r[20:]:
        var = EWMA_LAMBDA * var + (1.0 - EWMA_LAMBDA) * float(x) * float(x)
    return math.sqrt(var) * 100.0


def _hv_rank(closes: np.ndarray) -> float:
    r = np.diff(np.log(closes))
    if len(r) < HV_WINDOW + HV_RANK_WINDOW - 1:
        return math.nan
    r = r[-(HV_WINDOW + HV_RANK_WINDOW - 1):]
    win = np.lib.stride_tricks.sliding_window_view(r, HV_WINDOW)
    rv = win.std(axis=1, ddof=1)
    return float((rv <= rv[-1]).mean() * 100.0)


def frac_diff(closes: Sequence[float], d: float = FRAC_D) -> float:
    """Same weights/clip as ml_model_v2._frac_diff (drift-pinned by test)."""
    if len(closes) < FRAC_WIDTH:
        return 0.0
    w = [1.0]
    for k in range(1, FRAC_WIDTH):
        w.append(-w[-1] * (d - k + 1) / k)
    w = np.array(w[::-1])
    price_slice = np.array(closes[-FRAC_WIDTH:], dtype=np.float64)
    val = float(np.dot(w, np.log(price_slice + 1e-8)))
    return max(-3.0, min(3.0, val))


# ── Feature functions (the single definition) ───────────────────────────────

def row_eligible(a: Dict[str, np.ndarray], i: int) -> bool:
    """Training-row filter (liquidity floor); never applied at serve time."""
    if i < MAX_LOOKBACK or i >= len(a["c"]):
        return False
    if a["c"][i] < MIN_PRICE:
        return False
    return float(a["v"][i - 20:i].mean()) >= MIN_AVG_VOLUME


def stock_features_at(a: Dict[str, np.ndarray], i: int, spy_asof: np.ndarray) -> Optional[dict]:
    """Stock-specific features at bar i from bars [i-MAX_LOOKBACK .. i] only.
    spy_asof: SPY close as of each of the stock's bar dates (asof_closes)."""
    if i < MAX_LOOKBACK or i >= len(a["c"]):
        return None
    c, h, l, o, v = a["c"], a["h"], a["l"], a["o"], a["v"]
    ci = float(c[i])
    vol_prior = float(v[i - 20:i].mean())
    ewma = _ewma_vol(c[i - SMOOTH_WINDOW:i + 1])
    chg = _pct(ci, float(c[i - 1]))
    atr = wilder_atr(list(h[i - ATR_BARS + 1:i + 1]), list(l[i - ATR_BARS + 1:i + 1]),
                     list(c[i - ATR_BARS + 1:i + 1]))
    s = spy_asof
    spy_ok = not np.isnan(s[i - BETA_WINDOW:i + 1]).any() and not np.isnan(s[i - 21])
    if spy_ok:
        rs = np.diff(np.log(c[i - BETA_WINDOW:i + 1]))
        rm = np.diff(np.log(s[i - BETA_WINDOW:i + 1]))
        var_m = float(np.var(rm))
        beta = float(np.cov(rs, rm, bias=True)[0, 1] / var_m) if var_m > 0 else 1.0
        idio = _pct(ci, float(c[i - 21])) - beta * _pct(float(s[i]), float(s[i - 21]))
        rel5 = _pct(ci, float(c[i - 5])) - _pct(float(s[i]), float(s[i - 5]))
    else:
        idio = rel5 = math.nan
    return {
        "mom_1m": _pct(ci, float(c[i - 21])),
        "mom_3m": _pct(ci, float(c[i - 63])),
        "mom_12_1": _pct(float(c[i - 21]), float(c[i - 252])),
        "rsi_14": _wilder_rsi(c[i - SMOOTH_WINDOW:i + 1]),
        "price_vs_52w_high": _pct(ci, float(h[i - 251:i + 1].max())),
        "range_pct": (float(h[i]) - float(l[i])) / ci * 100.0,
        "change_pct_last": chg,
        "close_above_open": 1.0 if ci > float(o[i]) else 0.0,
        "above_ma10": 1.0 if ci > float(c[i - 10:i].mean()) else 0.0,
        "adx": _wilder_adx(h[i - SMOOTH_WINDOW + 1:i + 1], l[i - SMOOTH_WINDOW + 1:i + 1],
                           c[i - SMOOTH_WINDOW + 1:i + 1]),
        "frac_diff_price": frac_diff(c[i - FRAC_WIDTH + 1:i + 1]),
        "volume_ratio": min(float(v[i]) / vol_prior, 10.0) if vol_prior > 0 else math.nan,
        "ewma_vol": ewma,
        "atr_pct": (atr / ci * 100.0) if atr else math.nan,
        "hv_rank": _hv_rank(c[i - (HV_WINDOW + HV_RANK_WINDOW - 1):i + 1]),
        "trend_strength": (min(abs(chg) / max(ewma, 0.1), 5.0)
                           if not (math.isnan(chg) or math.isnan(ewma)) else math.nan),
        "idiosyncratic_ret": idio,
        "rel_strength_5d": rel5,
    }


class RefSeries:
    """The four reference series (completed bars), searchable by date."""

    def __init__(self, ref_bars: Dict[str, Sequence[dict]]):
        self.arr = {s: to_arrays(ref_bars.get(s) or []) for s in REF_SYMBOLS}
        self.dates = {s: list(self.arr[s]["t"]) for s in REF_SYMBOLS}
        self._cache: Dict[str, dict] = {}

    def index_asof(self, sym: str, d: str) -> int:
        return bisect.bisect_right(self.dates[sym], d) - 1

    def spy_asof(self, dates: Sequence[str]) -> np.ndarray:
        return asof_closes(self.dates["SPY"], self.arr["SPY"]["c"], dates)

    def market_features(self, d: str) -> dict:
        """Market features as of date d (each series at its last bar <= d)."""
        if d in self._cache:
            return self._cache[d]
        out = {k: math.nan for k in ("vxx_ratio", "spy_vs_ma50", "spy_ret_5d",
                                     "vxx_rank_252", "vol_of_vol", "credit_spread")}
        vx = self.arr["VXX"]["c"]
        j = self.index_asof("VXX", d)
        if j >= 30:
            out["vxx_ratio"] = float(vx[j] / vx[j - 30:j].mean())
        if j >= 10:
            out["vol_of_vol"] = float(np.std(np.diff(vx[j - 10:j + 1]) / vx[j - 10:j]) * 100.0)
        if j >= 252:
            lo, hi = float(vx[j - 252:j].min()), float(vx[j - 252:j].max())
            out["vxx_rank_252"] = (float(vx[j]) - lo) / (hi - lo) * 100.0 if hi > lo else 50.0
        sp = self.arr["SPY"]["c"]
        j = self.index_asof("SPY", d)
        if j >= 50:
            out["spy_vs_ma50"] = float(sp[j] / sp[j - 50:j].mean())
        if j >= 5:
            out["spy_ret_5d"] = float(np.mean(np.diff(sp[j - 5:j + 1]) / sp[j - 5:j]) * 100.0)
        jt, jh = self.index_asof("TLT", d), self.index_asof("HYG", d)
        if jt >= 21 and jh >= 21:
            t, hy = self.arr["TLT"]["c"], self.arr["HYG"]["c"]
            out["credit_spread"] = float((t[jt] / t[jt - 21] - 1.0) - (hy[jh] / hy[jh - 21] - 1.0))
        self._cache[d] = out
        return out


def features_at(a: Dict[str, np.ndarray], i: int, refs: RefSeries,
                spy_asof: Optional[np.ndarray] = None) -> Optional[dict]:
    """THE v3 feature row at bar i (training and serving both call this)."""
    if spy_asof is None:
        spy_asof = refs.spy_asof(list(a["t"]))
    st = stock_features_at(a, i, spy_asof)
    if st is None:
        return None
    st.update(refs.market_features(str(a["t"][i])))
    return st


def row_vector(feats: dict) -> List[float]:
    return [float(feats.get(k, math.nan)) for k in V3_FEATURE_COLS]


# ── The two paths ───────────────────────────────────────────────────────────

def training_rows(stock_bars: Sequence[dict], refs: RefSeries,
                  eligible_only: bool = True) -> List[tuple]:
    """TRAINING PATH: [(i, date, feature_dict)] for every computable bar."""
    a = to_arrays(stock_bars)
    if len(a["c"]) < MIN_BARS:
        return []
    spy = refs.spy_asof(list(a["t"]))
    out = []
    for i in range(MAX_LOOKBACK, len(a["c"])):
        if eligible_only and not row_eligible(a, i):
            continue
        f = features_at(a, i, refs, spy)
        if f is not None:
            out.append((i, str(a["t"][i]), f))
    return out


def serve_features(stock_bars: Sequence[dict], ref_bars: Dict[str, Sequence[dict]],
                   today_et: str) -> Optional[dict]:
    """SERVING PATH: the feature row for a decision taken during session
    `today_et` — computed at the last COMPLETED bar (the in-progress session
    and anything later is stripped from the stock AND reference bars).
    None when fewer than MIN_BARS completed bars are available."""
    a = to_arrays(strip_incomplete(stock_bars, today_et))
    if len(a["c"]) < MIN_BARS:
        return None
    refs = RefSeries({s: strip_incomplete(ref_bars.get(s) or [], today_et) for s in REF_SYMBOLS})
    f = features_at(a, len(a["c"]) - 1, refs)
    if f is None:
        return None
    f["_asof_date"] = str(a["t"][-1])
    return f
