#!/usr/bin/env python3
"""walkforward.py — long-term, point-in-time, regime-split test of the stock
score and its price-derivable inputs over the 2016+ price panel.

[RULE-REVIEW] MEASUREMENT code (CLAUDE.md MEASUREMENT INTEGRITY). It measures;
it never tunes, never trades, never changes a live threshold. Its output
(DATA_DIR/walkforward/latest.json) is the Brain page's LONG-TERM column for the
stock score (contract: research/walkforward.md).

WHAT IS MEASURED. Every week (the last trading day of each ISO week) the live
scanner's price-derivable score inputs are recomputed for every name in the
point-in-time universe using ONLY bars up to that day's close, exactly as the
live code computes them (bot_engine.deep_score, analyze.py, strategies/*,
bot_engine quick score — each mirror is pinned to the live function or live
source lines by test_walkforward.py):

  quick_score     bot_engine quick score: |daily change| capped, volume, direction, extreme penalty
  momentum        strategies.momentum.score(12-1 momentum, 1-month return, day volume)
  mean_reversion  strategies.mean_reversion.score(RSI-14, 5-day change, 5d/90d volume ratio)
  volume          deep_score volume sub-score (5d/90d volume ratio buckets)
  vol_premium     deep_score EWMA-vs-RV20 bonus (+8 / +5 / 0)
  adx             deep_score ADX trend adjustment (+6 / +3 / -4)
  obv             deep_score OBV accumulation adjustment (+5 / -5)
  rel_strength    deep_score relative-strength bonus (+6 when 19-day return beats SPY by > 5 pts)
  price_composite the live rules composite with every NON-reconstructable term held
                  at its neutral value (VRP 50, squeeze 50, macro/news/intel/alt/
                  social/fundamental/recommendation/contrarian 0), clamped 0..100
                  — its ranking is the ranking of the measurable terms only.

NOT MEASURABLE LONG-TERM (no point-in-time history exists in the repo; they
stay "not measurable long-term" honestly): VRP (implied vol), squeeze (short
interest, sentiment, Reddit), news, intel/insider (Form 4, MSPR, analyst),
alt data, social, Google Trends, fundamentals, the AI recommendation and
contrarian boosts, the macro adjustments (VIX regime, sector momentum, risk
on/off), the ML blend. See NOT_MEASURABLE.

HOW. Outcome = enter at the NEXT bar's open, exit at the close h trading days
after the decision (h = 5, 20), net of the cost model below; plus, for each
input's top-N, the LIVE exit model (live_exit_policy via backtest_v2's
live_atr_pct, walked O->L->H->C, the same ladder backtest_v2 exit_model="live"
uses). Per input: rank IC (Spearman, per date, then averaged; Newey-West t),
quintile means and the Q5-Q1 long-short spread net of both legs' costs, a
top-N portfolio vs SPY and vs the RANDOM-ENTRY BASE RATE (same universe, same
dates, same holding: the equal-weight universe mean is the exact expectation of
random entry; for live exits, seeded random draws), hit rates, by regime
(regime_util.classify_regime_5level on panel SPY + VXX, the backtest_v2 rules)
and by SPY-vs-200d proxy, by year, by liquidity tercile. Multiple testing:
Bonferroni over every IC test; Deflated Sharpe (Bailey & Lopez de Prado 2014)
over every top-N Sharpe estimate.

COST MODEL (per side, backtest_v2.liquidity_cost_pct exactly): the trailing
20-bar mean share-volume tier (>20M 0.035%, >5M 0.075%, >1M 0.115%, else
0.185%) or the Reg NMS half-tick floor at the raw price, whichever is larger;
entry cost keyed on the decision bar, exit cost on the exit bar. Share volume
is the RAW-share estimate close x volume / raw close (back-adjusted volume
would put old, later-split names in cheaper tiers: optimistic). Bias: the
tiers are mid-points of the live fill haircut; no market impact at size (the
top-N is equal-weight, notional-free) -> costs are OPTIMISTIC for the
thinnest names.

DELISTING ASSUMPTION (research/price_panel_coverage.md caveat 3): a series that
stops trading before an exit and before the panel's as_of exits at its last
close times (1 + r): r = 0 when it ends at a dated merger / rename (cash or
stock at the deal price), r = -30% otherwise (Shumway 1997, performance
delistings, including worthless removals and undated inactive names). Bias:
unknown-reason ends at -30% may be too harsh for quiet mergers the
corporate-actions feed missed (PESSIMISTIC on those) and too mild for true
bankruptcies (OPTIMISTIC on those).

BIAS SUMMARY (stated per MEASUREMENT INTEGRITY): survivorship-REDUCED, not
free (a delisted name in neither the asset list nor the corporate-actions feed
is missing -> OPTIMISTIC); prices are back-adjusted (returns fine, price floor
uses raw close); deep_score's intraday partial bar and analyze.py's
split-only adjustment are not reproducible (decisions here are at the close on
dividend-adjusted bars — small, direction mixed); universe floors approximate
the live snapshot floors (raw close >= $5, est. raw shares >= 500k that day,
20-day dollar volume >= $2.5M, symbol without '.' and <= 5 chars).

MEMORY: the panel streams one shard at a time (price_panel.iter_series); only
per-(name, rebalance date) rows are kept (~100 bytes each). The run aborts
cleanly (status error, latest.json untouched) above --max-rss-mb.

CLI: python3 walkforward.py --run [--panel-root DIR] [--out-dir DIR] [--max-rss-mb N]
     prints one JSON summary line; writes <out-dir>/latest.json (success only,
     atomic) and <out-dir>/last_run.json (every run).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import sys
import time
from datetime import date, datetime, timezone
from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import price_panel as pp

SCHEMA = "walkforward/v1"

# ── configuration (measurement settings, not trading parameters) ────────────
START = "2017-01-01"            # first rebalance: one year of warm-up for 12-1 momentum
HORIZONS = (5, 20)              # trading days
PRIMARY_HORIZON = 5             # the live exits' ~7-calendar-day time stop
TOP_N = 10
MIN_NAMES_PER_DATE = 30         # fewer -> no IC that date
MIN_DATES_FOR_VERDICT = 52      # one year of weekly dates
ALPHA = 0.05
# Live scanner floors (bot_engine.py MIN_PRICE / MIN_VOLUME, pinned by test_walkforward.py)
MIN_PRICE = 5
MIN_VOLUME = 500000
MIN_DOLLAR_VOLUME = 2.5e6       # trailing 20-bar mean; = MIN_VOLUME x MIN_PRICE
ADV_WINDOW = 20
LIVE_EXIT_MAX_BARS = 60         # a live-exit position still open after this is closed at the close ("max_hold")
RANDOM_LIVE_DRAWS = 3
SEED = 20261029
DELIST_RETURN_MERGER = 0.0
DELIST_RETURN_OTHER = -0.30     # Shumway (1997) performance delistings
DEFAULT_MAX_RSS_MB = 900
RSS_CHECK_EVERY = 200          # series between peak-RSS checks

# deep_score windows (bot_engine.py: start_d = now - 400 days, limit=300) and
# analyze.py's history (_fetch_alpaca_bars(days=365)) and SPY fetch (days=30)
DEEP_DAYS = 400
DEEP_LIMIT = 300
HIST_DAYS = 365
SPY_RS_DAYS = 30

INPUTS = ("quick_score", "momentum", "mean_reversion", "volume", "vol_premium", "adx", "obv",
          "rel_strength", "price_composite")
INPUT_INFO = {
    "quick_score": ("Quick score (pre-rank)", "bot_engine quick score: min(|chg|,15)*3 + min(vol/1M,5)*2 + direction bonus + extreme-move penalty"),
    "momentum": ("Momentum sub-score (25%)", "strategies.momentum.score(12-1 momentum, 1-month return, day share volume) on deep_score's 400-day bars"),
    "mean_reversion": ("Mean-reversion sub-score (20%)", "strategies.mean_reversion.score(analyze RSI-14, deep 5-day change, analyze 5d/90d volume ratio); 50 when RSI is missing"),
    "volume": ("Volume sub-score (15%)", "deep_score: 90/75/60/50/35 by 5d/90d volume ratio >3/>2/>1.5/>1/else"),
    "vol_premium": ("EWMA vol-premium bonus", "deep_score: +8 if EWMA(0.94,60) vol - RV20 > 5 pts, +5 if < -5 pts, else 0"),
    "adx": ("ADX trend bonus", "deep_score: +6/+3/-4 when the Wilder DX of the 400-day bars is >40/>25/<15"),
    "obv": ("OBV bonus", "deep_score: +5/-5 when the last-5 OBV mean is >1.05x/<0.95x the previous 5"),
    "rel_strength": ("Relative-strength bonus", "deep_score: +6 when analyze's 19-day return minus SPY's exceeds 5 pts"),
    "price_composite": ("Rules composite, price-measurable part", "0.25 momentum + 0.20 mean reversion + 0.15 volume + 0.25x50 (VRP) + 0.15x50 (squeeze) + vol-premium + ADX + OBV + rel-strength, clamped 0..100"),
}
NOT_MEASURABLE = {
    "vrp": "Needs implied volatility (options chains): no point-in-time history before the forward-only chain archive.",
    "squeeze": "Needs short interest, sentiment and Reddit buzz history: none point-in-time.",
    "news": "Headline sentiment: no archived point-in-time news for 2016+.",
    "intel": "Form 4 / MSPR / analyst / earnings-pattern intel: not reconstructed point-in-time here (Form 4 has its own gate-2 test).",
    "insider_mspr": "Finnhub MSPR: no point-in-time history.",
    "social": "Reddit / RSS buzz: no point-in-time history.",
    "google_trends": "Off in live (constant 0).",
    "alt": "Wikipedia / short-volume / FTD / GDELT / FRED composite: not reconstructed point-in-time.",
    "fundamentals": "yfinance ratings, targets, P/E: today's values only (lookahead if used historically).",
    "contrarian": "Contrarian flags come from sentiment + short interest: no history.",
    "macro_adj": "VIX regime, sector momentum (needs a point-in-time sector map) and risk on/off: not reconstructed; regime is a SPLIT here, not an input.",
    "ml_v2": "The ML blend's 34 features include non-price inputs; it has its own honest CV (research/ml_v3_eval.md).",
}

PRIOR = ("Stated 2026-10-29, before any real-panel run: most price-based inputs' rank IC is near 0 net of costs "
         "(|mean IC| < 0.02, none surviving Bonferroni); momentum is slightly positive in the least liquid tercile "
         "outside crash regimes; short-horizon inputs that reward large recent moves (quick_score, volume, "
         "rel_strength) are flat to negative at 5 days (short-term reversal); the price composite's top-N does "
         "not beat the random-entry base rate net of costs.")

CAVEATS = [
    "Survivorship-REDUCED, not free: the panel universe is Alpaca assets + corporate actions from 2016; delisted names in neither are missing (optimistic).",
    "Delisting returns assumed: 0 at a dated merger/rename, -30% otherwise (Shumway 1997).",
    "Prices back-adjusted for splits and dividends (returns are total-return-like); price floors use the raw close.",
    "Live deep_score uses an intraday partial bar and analyze.py split-only bars; here every input is computed at the close on dividend-adjusted bars.",
    "Costs are the backtest_v2 liquidity tiers + half-tick floor per side; no market impact at size (optimistic for the thinnest names).",
    "Only the price-derivable parts of the stock score are measurable; see not_measurable.",
    "Alpaca market data: derived statistics only — no price leaves this file.",
]


# ── small helpers ────────────────────────────────────────────────────────────
def _round(x: Optional[float], nd: int = 4) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


def _pct(x: Optional[float]) -> Optional[float]:
    return None if x is None or not math.isfinite(float(x)) else round(float(x) * 100.0, 4)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0   # Linux: KB


class MemoryBudgetExceeded(RuntimeError):
    pass


def _window_sum(x: np.ndarray, w: int) -> np.ndarray:
    """out[i] = sum(x[i-w+1..i]) for i >= w-1, NaN before."""
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = sliding_window_view(x, w).sum(axis=1)
    return out


# ── the live score's price inputs, vectorized over one series ───────────────
def wilder_filter(x: np.ndarray, a: float) -> np.ndarray:
    """G[k] = a*G[k-1] + x[k], G[-1] = 0 (C-speed via scipy when present)."""
    try:
        from scipy.signal import lfilter
        return lfilter([1.0], [1.0, -a], x)
    except ImportError:   # pragma: no cover - scipy is in requirements.txt
        g = np.empty(len(x))
        acc = 0.0
        for i, v in enumerate(x):
            acc = a * acc + v
            g[i] = acc
        return g


def raw_share_volume(s: pp.Series) -> np.ndarray:
    """Split-invariant share volume: dollar volume / raw close (served volume
    where no raw close was recorded). Back-adjusted volume would inflate the
    share counts of names that split later."""
    v = s.v.astype(np.float64)
    rc_ok = ~np.isnan(s.raw_close) & (s.raw_close > 0)
    return np.where(rc_ok, np.round(s.c * v / np.where(rc_ok, s.raw_close, 1.0)), v)


def _pyround(x: np.ndarray, nd: int) -> np.ndarray:
    """Python's round() elementwise (the live code rounds with it; np.round
    can differ on the last digit), NaN preserved."""
    return np.array([round(float(v), nd) if math.isfinite(v) else float("nan") for v in x], dtype=np.float64)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """out[i] = x[i-k] (NaN for i < k)."""
    out = np.full(len(x), np.nan)
    if len(x) > k:
        out[k:] = x[:-k] if k else x
    return out


def series_features(s: pp.Series, spy_days: np.ndarray, spy_c: np.ndarray,
                    at: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Every price input the live score uses, at bars `at` (default: all),
    each computed from bars <= that bar only (every window ends at the bar;
    nothing indexes a later bar). Mirrors, pinned in test_walkforward.py:
      analyze.compute_rsi / compute_volume_metrics / realized_vol / ewma_vol /
      compute_relative_strength on the 365-day history (analyze_ticker);
      bot_engine.deep_score ADX / OBV / 12-1 momentum / 5-day change on the
      400-day bars; the bot_engine quick score on the day bar; the strategy
      modules themselves (momentum.score, mean_reversion.score)."""
    from strategies import mean_reversion, momentum

    c, h, lo = s.c, s.h, s.l
    v = s.v.astype(np.float64)
    days = s.days
    n = len(c)
    sel = np.arange(n) if at is None else np.asarray(at, dtype=np.int64)
    idx = sel
    w_deep = np.searchsorted(days, days[sel] - DEEP_DAYS, side="left")
    n_deep = np.minimum(idx - w_deep + 1, DEEP_LIMIT)
    w_hist = np.searchsorted(days, days[sel] - HIST_DAYS, side="left")
    n_hist = idx - w_hist + 1

    v_raw_all = raw_share_volume(s)
    v_raw = v_raw_all[sel]
    prev_c = np.where(idx >= 1, c[np.maximum(idx - 1, 0)], c[idx])   # snapshot prevDailyBar (missing -> c)
    with np.errstate(divide="ignore", invalid="ignore"):
        change_pct = np.where(prev_c > 0, (c[idx] - prev_c) / prev_c * 100.0, 0.0)

    # bot_engine quick score
    ach = np.abs(change_pct)
    quick = (np.minimum(ach, 15.0) * 3 + np.minimum(v_raw / 1_000_000, 5.0) * 2
             + np.where(ach > 10, 5, np.where(ach > 5, 2, 0))
             + np.where(ach > 50, -30, np.where(ach > 30, -15, 0)))

    # analyze.compute_rsi: the last 14 changes, simple averages, round(.., 2)
    d = np.diff(c)
    g14 = _window_sum(np.concatenate([[0.0], np.where(d > 0, d, 0.0)]), 14)[sel]
    l14 = _window_sum(np.concatenate([[0.0], np.where(d > 0, 0.0, -d)]), 14)[sel]
    ok = (n_hist >= 15) & np.isfinite(g14)
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_g, avg_l = g14 / 14, l14 / 14
        rsi_raw = 100 - (100 / (1 + avg_g / avg_l))
    rsi = np.where(ok, np.where(avg_l == 0, 100.0, _pyround(np.where(ok & (avg_l != 0), rsi_raw, np.nan), 2)), np.nan)

    # analyze.compute_volume_metrics volume_ratio_5d: mean(last 5) / mean(last min(90, history)), round 2
    cs = np.concatenate([[0.0], np.cumsum(v)])
    m90 = np.minimum(90, n_hist)
    avg90 = (cs[idx + 1] - cs[idx + 1 - m90]) / m90
    rec5 = (cs[idx + 1] - cs[np.maximum(idx + 1 - 5, 0)]) / 5
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = np.where(avg90 > 0, _pyround(rec5 / np.where(avg90 > 0, avg90, 1.0), 2), 1.0)
    vol_ratio = np.where(n_hist >= 30, vr, 1.0)
    vol_ratio = np.where(vol_ratio == 0, 1.0, vol_ratio)              # `... or 1.0`

    # analyze.realized_vol(hist, 20) / ewma_vol(hist, 0.94, 60), round 2
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(c[1:] / c[:-1])                                    # lr[k-1] = return INTO bar k
    rv20 = np.full(len(sel), np.nan)
    ewma = np.full(len(sel), np.nan)
    if n >= 21:
        var20 = np.full(n, np.nan)
        var20[20:] = sliding_window_view(lr, 20).var(axis=1, ddof=1)
        rv20 = np.where(n_hist >= 21, _pyround(np.sqrt(var20[sel] * 252) * 100, 2), np.nan)
    if n >= 61:
        lam = 0.94
        win = sliding_window_view(lr, 60)                              # row j = returns into bars j+1 .. j+60
        var = np.full(n, np.nan)
        var[60:] = lam ** 60 * win.var(axis=1, ddof=1) + (win ** 2) @ ((1 - lam) * lam ** np.arange(59, -1, -1))
        ewma = _pyround(np.sqrt(var[sel] * 252) * 100, 2)
    ewma = np.where(n_hist - 1 >= 60, ewma, rv20)                      # < 60 returns -> realized_vol(hist, 20)
    rv20_live = np.where(np.isnan(rv20) | (rv20 == 0), 20.0, rv20)     # `realized_vol(...) or 20.0`
    ewma_live = np.where(np.isnan(ewma) | (ewma == 0), rv20_live, ewma)  # `ewma_vol(...) or rv20`
    prem = ewma_live - rv20_live
    vol_premium = np.where(prem > 5, 8.0, np.where(prem < -5, 5.0, 0.0))

    # bot_engine.deep_score ADX: Wilder, seeded with the mean of the window's
    # first 14 TRs, then iterated to the bar — closed form over a global filter:
    # sum_{k=s}^{i} a^(i-k) x_k = G[i] - a^(i-s+1) G[s-1]
    tr = np.zeros(n)
    pdm = np.zeros(n)
    mdm = np.zeros(n)
    if n > 1:
        up = h[1:] - h[:-1]
        dn = lo[:-1] - lo[1:]
        pdm[1:] = np.where((up > dn) & (up > 0), up, 0.0)
        mdm[1:] = np.where((dn > up) & (dn > 0), dn, 0.0)
        tr[1:] = np.maximum(h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))
    a = 13.0 / 14.0
    ok = n_deep >= 15
    dx = np.full(len(sel), np.nan)
    if ok.any():
        w0 = np.where(ok, w_deep, 0)
        s14 = np.where(ok, w_deep + 14, 0)                             # last seed TR index (seed = TRs w0+1 .. w0+14)
        ii = np.where(ok, idx, 0)
        steps = (ii - s14).astype(np.float64)

        def smooth(x):
            csx = np.concatenate([[0.0], np.cumsum(x)])
            seed = (csx[s14 + 1] - csx[w0 + 1]) / 14
            g = wilder_filter(x, a)
            return a ** steps * seed + (g[ii] - a ** steps * g[s14]) / 14

        atr, p14, m14 = smooth(tr), smooth(pdm), smooth(mdm)
        with np.errstate(divide="ignore", invalid="ignore"):
            pdi = np.where(atr > 0, p14 / atr * 100, 0.0)
            mdi = np.where(atr > 0, m14 / atr * 100, 0.0)
            dxv = np.where(pdi + mdi > 0, np.abs(pdi - mdi) / (pdi + mdi) * 100, 0.0)
        dx = np.where(ok, _pyround(np.where(ok, dxv, np.nan), 1), np.nan)
    adx_adj = np.where(np.isnan(dx), 0.0, np.where(dx > 40, 6.0, np.where(dx > 25, 3.0, np.where(dx < 15, -4.0, 0.0))))

    # bot_engine.deep_score OBV, relative to the window start (obv_values[0] = 0)
    sv = np.zeros(n)
    if n > 1:
        sv[1:] = np.where(c[1:] > c[:-1], v[1:], np.where(c[1:] < c[:-1], -v[1:], 0.0))
    S = np.cumsum(sv)
    cS = np.concatenate([[0.0], np.cumsum(S)])
    ok = (n_deep >= 14) & (n_deep >= 10)
    sw = S[w_deep]
    recent = (cS[idx + 1] - cS[np.maximum(idx - 4, 0)] - 5 * sw) / 5
    prev = (cS[np.maximum(idx - 4, 0)] - cS[np.maximum(idx - 9, 0)] - 5 * sw) / 5
    obv_adj = np.where(ok, np.where(recent > prev * 1.05, 5.0, np.where(recent < prev * 0.95, -5.0, 0.0)), 0.0)

    # deep_score 5-day change and Jegadeesh-Titman 12-1 momentum on the 400-day bars
    captured = n_deep >= 14                                            # `if len(bars_data) >= 14:` captures the closes
    c5, c20, c251 = _shift(c, 5)[sel], _shift(c, 20)[sel], _shift(c, 251)[sel]
    with np.errstate(divide="ignore", invalid="ignore"):
        chg5 = np.where(captured & (n_deep >= 6) & (c5 > 0), (c[idx] / c5 - 1.0) * 100, _pyround(change_pct, 2))
        ret1 = (c[idx] / c20 - 1.0) * 100
        ret12 = (c[idx] / c251 - 1.0) * 100
    full = captured & (n_deep >= 252)
    part = captured & ~full & (n_deep >= 21)
    mom_12_1 = np.where(full, ret12 - ret1, np.where(part, ret1 * 0.5, 0.0))
    mom_1m = np.where(full | part, ret1, 0.0)

    # analyze.compute_relative_strength: 19-bar return (iloc[-20]) minus SPY's over
    # its 30-day fetch; with < 20 SPY bars the absolute return stands in
    k = np.searchsorted(spy_days, days[sel])
    kk = np.minimum(k, len(spy_days) - 1)
    k_ok = (k < len(spy_days)) & (spy_days[kk] == days[sel])
    c19 = _shift(c, 19)[sel]
    with np.errstate(divide="ignore", invalid="ignore"):
        stock_ret = (c[idx] / c19 - 1) * 100
        spy_m = kk - np.searchsorted(spy_days, spy_days[kk] - SPY_RS_DAYS, side="left") + 1
        spy_ret = np.where(kk >= 19, (spy_c[kk] / spy_c[np.maximum(kk - 19, 0)] - 1) * 100, np.nan)
        rs = np.where(spy_m >= 20, _pyround(stock_ret - spy_ret, 1), _pyround(stock_ret, 1))
    rs_bonus = np.where((n_hist >= 20) & k_ok & (rs > 5), 6.0, 0.0)

    # the strategy modules themselves, then deep_score's composite arithmetic
    mom_score = np.array([momentum.score(float(m), float(m1), int(av))["score"]
                          for m, m1, av in zip(mom_12_1, mom_1m, v_raw)], dtype=np.float64)
    rsi_mr = np.where(rsi == 0, np.nan, rsi)                           # `vol_metrics.get("rsi_14") or None`
    mr_score = np.array([50.0 if not math.isfinite(r_) else mean_reversion.score(float(r_), float(ch), float(vr_))["score"]
                         for r_, ch, vr_ in zip(rsi_mr, chg5, vol_ratio)], dtype=np.float64)
    vol_score = np.where(vol_ratio > 3, 90.0, np.where(vol_ratio > 2, 75.0, np.where(vol_ratio > 1.5, 60.0,
                                                                              np.where(vol_ratio > 1, 50.0, 35.0))))
    composite = (mom_score * 0.25 + mr_score * 0.20 + 50 * 0.25 + 50 * 0.15 + vol_score * 0.15) + vol_premium
    composite = np.clip(composite + adx_adj + obv_adj + rs_bonus, 0, 100)

    return {
        "quick_score": quick, "momentum": mom_score, "mean_reversion": mr_score, "volume": vol_score,
        "vol_premium": vol_premium, "adx": adx_adj, "obv": obv_adj, "rel_strength": rs_bonus,
        "price_composite": composite,
        # intermediates, pinned against the live functions by the tests
        "_rsi": rsi, "_vol_ratio": vol_ratio, "_rv20": rv20, "_ewma": ewma, "_dx": dx, "_rs": rs,
        "_mom_12_1": mom_12_1, "_mom_1m": mom_1m, "_chg5": chg5, "_v_raw": v_raw, "_change_pct": change_pct,
        "_obv_recent": recent, "_obv_prev": prev,
    }


# ── cost model (backtest_v2.liquidity_cost_pct, vectorized) ─────────────────
def cost_per_side(v_raw: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Per-side cost fraction at every bar: max(volume tier on the trailing
    20-bar mean raw-share volume incl. the bar, half-tick floor at the raw price)."""
    import backtest_v2 as bt
    avg = pp.trailing_mean(v_raw, 20)
    tier = np.full(len(avg), bt._ILLIQUID_COST_PCT)
    for threshold, cost in reversed(bt._LIQUIDITY_TIERS):
        tier = np.where(avg > threshold, cost, tier)
    with np.errstate(divide="ignore", invalid="ignore"):
        tick = np.where(price >= 1.0, (bt.MIN_TICK / 2.0) / price, 0.0)
    return np.maximum(tier, tick)


def delisting_return(s: pp.Series) -> float:
    if s.event_ends_series and s.event and s.event.get("kind") in ("merger", "name_change"):
        return DELIST_RETURN_MERGER
    return DELIST_RETURN_OTHER


# ── regime labels (backtest_v2.regime_series rules, multi-series VXX) ──────
def vxx_ratio_by_day(vxx_series: Sequence[pp.Series]) -> Dict[int, float]:
    """VXX close / its own 30-bar SMA, per series (never across two VXX
    products); on overlapping days the later-listed series wins."""
    out: Dict[int, float] = {}
    for s in sorted(vxx_series, key=lambda x: x.first_day):
        cs = np.concatenate([[0.0], np.cumsum(s.c)])
        for i, d in enumerate(s.days):
            if i + 1 >= 30:
                avg = (cs[i + 1] - cs[i + 1 - 30]) / 30
                out[int(d)] = float(s.c[i] / avg) if avg else 1.0
    return out


def regime_labels(spy_days: np.ndarray, spy_c: np.ndarray, vxx: Dict[int, float]) -> Tuple[List[str], float]:
    """Per SPY day, the backtest_v2.regime_series label (classify_regime_5level
    on VXX ratio, SPY/SMA50, consecutive days below SMA200). Past-only."""
    from regime_util import classify_regime_5level
    labels, run, missing = [], 0, 0
    cs = np.concatenate([[0.0], np.cumsum(spy_c)])
    for i in range(len(spy_c)):
        ma50 = (cs[i + 1] - cs[i + 1 - 50]) / 50 if i + 1 >= 50 else None
        ma200 = (cs[i + 1] - cs[i + 1 - 200]) / 200 if i + 1 >= 200 else None
        vs50 = (spy_c[i] / ma50) if ma50 else 1.0
        above = bool(ma200 and spy_c[i] > ma200)
        run = 0 if above else run + 1
        d = int(spy_days[i])
        if d in vxx:
            vr = vxx[d]
        else:
            vr = 1.0
            missing += 1
        labels.append(classify_regime_5level(vr, vs50, spy_below_200_days=run, spy_above_200d=above))
    return labels, (missing / len(spy_c) if len(spy_c) else 1.0)


def spy_200d_labels(spy_c: np.ndarray) -> List[Optional[str]]:
    cs = np.concatenate([[0.0], np.cumsum(spy_c)])
    out: List[Optional[str]] = []
    for i in range(len(spy_c)):
        if i + 1 < 200:
            out.append(None)
        else:
            out.append("above_200d" if spy_c[i] > (cs[i + 1] - cs[i + 1 - 200]) / 200 else "below_200d")
    return out


def weekly_rebalance_indices(spy_days: np.ndarray, start_day: int) -> np.ndarray:
    """SPY index of the last trading day of each ISO week, from start_day."""
    keys = [date.fromordinal(int(d) + pp._EPOCH_ORD).isocalendar()[:2] for d in spy_days]
    out = [i for i in range(len(spy_days)) if spy_days[i] >= start_day and (i + 1 == len(spy_days) or keys[i + 1] != keys[i])]
    return np.asarray(out, dtype=np.int64)


# ── metrics ──────────────────────────────────────────────────────────────────
def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share their mean rank."""
    from scipy.stats import rankdata as _rd
    return _rd(x, method="average")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def newey_west_t(x: Sequence[float], lags: int) -> Tuple[Optional[float], Optional[float]]:
    """(mean, t) of a series with a Bartlett-kernel HAC variance of the mean."""
    a = np.asarray([v for v in x if v is not None and math.isfinite(v)], dtype=np.float64)
    T = len(a)
    if T < 3:
        return (float(a.mean()) if T else None), None
    m = a.mean()
    e = a - m
    var = float(e @ e) / T
    for lag in range(1, min(lags, T - 1) + 1):
        w = 1 - lag / (lags + 1)
        var += 2 * w * float(e[lag:] @ e[:-lag]) / T
    if var <= 0:
        return float(m), None
    return float(m), float(m / math.sqrt(var / T))


def two_sided_p(t: Optional[float]) -> Optional[float]:
    if t is None or not math.isfinite(t):
        return None
    return 2 * (1 - NormalDist().cdf(abs(t)))


def bonferroni(p: Optional[float], m: int) -> Optional[float]:
    return None if p is None else min(1.0, p * m)


def quintile_of(x: np.ndarray) -> np.ndarray:
    """0..4 by average rank percentile; tied values always share a bucket."""
    r = rankdata(x)
    return np.minimum(4, ((r - 0.5) / len(x) * 5).astype(int))


def sharpe_stats(x: Sequence[float]) -> Optional[dict]:
    a = np.asarray([v for v in x if v is not None and math.isfinite(v)], dtype=np.float64)
    if len(a) < 8 or a.std(ddof=1) == 0:
        return None
    sd = a.std(ddof=1)
    z = (a - a.mean()) / a.std(ddof=0)
    return {"sr": float(a.mean() / sd), "n": len(a), "skew": float((z ** 3).mean()), "kurt": float((z ** 4).mean())}


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Bailey & Lopez de Prado (2014) eq. for E[max SR] under the null across N trials."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    g = 0.5772156649015329
    nd = NormalDist()
    return math.sqrt(var_sr) * ((1 - g) * nd.inv_cdf(1 - 1 / n_trials) + g * nd.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, sr0: float) -> Optional[float]:
    """P(true SR > sr0) — the Probabilistic Sharpe Ratio at the deflated
    benchmark sr0 (per-period SRs; kurt is raw, normal = 3)."""
    den = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if n_obs < 2 or den <= 0:
        return None
    return NormalDist().cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(den))


# ── pass 1: rows ─────────────────────────────────────────────────────────────
FEAT_COLS = INPUTS


def _symbol_ok(sym: str) -> bool:
    return "." not in sym and len(sym) <= 5


def collect_rows(build_dir: str, spy_days: np.ndarray, spy_c: np.ndarray, rebal: np.ndarray,
                 as_of: str, max_rss_mb: float, series_keys: List[str], series_meta: List[dict]) -> Tuple[Dict[str, np.ndarray], dict]:
    """Stream every series; keep one row per (eligible name, rebalance date)."""
    rebal_days = spy_days[rebal]
    pos_of_day = {int(d): j for j, d in enumerate(rebal_days)}
    chunks: Dict[str, List[np.ndarray]] = {k: [] for k in ("sid", "j", "k", *FEAT_COLS, "dv", "c", "v", "delisted", "no_raw",
                                                             *[f"net{h}" for h in HORIZONS], *[f"gross{h}" for h in HORIZONS],
                                                             *[f"dl{h}" for h in HORIZONS], "unenterable")}
    stats = {"series_seen": 0, "series_used": 0, "symbols_used": set(), "delisted_symbols_used": set()}
    nspy = len(spy_days)
    for s in pp.iter_series(build_dir):
        stats["series_seen"] += 1
        sid = len(series_keys)
        series_keys.append(s.key)
        series_meta.append({"symbol": s.symbol, "delisted": s.delisted})
        if not _symbol_ok(s.symbol) or len(s.c) < 2:
            continue
        on = np.isin(s.days, rebal_days)
        if not on.any():
            continue
        price = pp.pit_price(s)
        v_raw = raw_share_volume(s)
        dv = pp.trailing_mean(s.c * s.v.astype(np.float64), ADV_WINDOW)
        elig = on & (price >= MIN_PRICE) & (v_raw >= MIN_VOLUME) & (dv >= MIN_DOLLAR_VOLUME)
        ii = np.nonzero(elig)[0]
        if len(ii) == 0:
            continue
        feats = series_features(s, spy_days, spy_c, at=ii)
        cost = cost_per_side(v_raw, price)
        jj = np.array([pos_of_day[int(d)] for d in s.days[ii]], dtype=np.int64)
        kk = rebal[jj]
        n = len(s.c)
        terminated = s.ends_before(as_of)
        dr = delisting_return(s)
        unenter = (ii + 1 >= n)
        entry_i = np.minimum(ii + 1, n - 1)
        for hz in HORIZONS:
            net = np.full(len(ii), np.nan)
            gross = np.full(len(ii), np.nan)
            dl = np.zeros(len(ii), dtype=bool)
            mature = kk + hz < nspy
            tgt = np.where(mature, spy_days[np.minimum(kk + hz, nspy - 1)], 0)
            ex = np.searchsorted(s.days, tgt, side="right") - 1
            ok = mature & ~unenter & (s.days[entry_i] <= tgt) & (ex >= entry_i)
            ended = ok & (ex == n - 1) & (s.days[n - 1] < tgt) & terminated
            mult = np.where(ended, 1.0 + dr, 1.0)
            ent = s.o[entry_i]
            exc = s.c[np.maximum(ex, 0)]
            with np.errstate(divide="ignore", invalid="ignore"):
                g = exc * mult / ent - 1
                nt = exc * (1 - cost[np.maximum(ex, 0)]) * mult / (ent * (1 + cost[ii])) - 1
            gross[ok] = g[ok]
            net[ok] = nt[ok]
            dl[ended] = True
            chunks[f"net{hz}"].append(net.astype(np.float32))
            chunks[f"gross{hz}"].append(gross.astype(np.float32))
            chunks[f"dl{hz}"].append(dl)
        for f in FEAT_COLS:
            chunks[f].append(feats[f].astype(np.float32))
        chunks["sid"].append(np.full(len(ii), sid, dtype=np.int32))
        chunks["j"].append(jj.astype(np.int32))
        chunks["k"].append(kk.astype(np.int32))
        chunks["dv"].append(dv[ii].astype(np.float32))
        chunks["c"].append(s.c[ii].astype(np.float64))
        chunks["v"].append(s.v[ii].astype(np.int64))
        chunks["delisted"].append(np.full(len(ii), s.delisted))
        chunks["no_raw"].append(np.isnan(s.raw_close[ii]))
        chunks["unenterable"].append(unenter)
        stats["series_used"] += 1
        stats["symbols_used"].add(s.symbol)
        if s.delisted:
            stats["delisted_symbols_used"].add(s.symbol)
        if stats["series_seen"] % RSS_CHECK_EVERY == 0 and rss_mb() > max_rss_mb:
            raise MemoryBudgetExceeded(f"RSS {rss_mb():.0f} MB > budget {max_rss_mb:.0f} MB after {stats['series_seen']} series")
    rows = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in chunks.items()}
    return rows, stats


def dedupe_rows(rows: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], int]:
    """One row per (date, close, volume): the same security under two
    tickers (rename continuations) or two mappings counts once. Keeps the
    first in stream order (deterministic)."""
    if len(rows["sid"]) == 0:
        return rows, 0
    key = np.stack([rows["j"].astype(np.float64), np.round(rows["c"], 4), rows["v"].astype(np.float64)], axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    keep = np.sort(first)
    removed = len(rows["sid"]) - len(keep)
    return {k: v[keep] for k, v in rows.items()}, int(removed)


# ── pass 2: live exits for the picked (series, date) pairs ──────────────────
def tie_break(sid: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-random secondary key: among tied scores the top-N
    is a seeded random draw (live breaks ties by scan order, which a history
    cannot reproduce)."""
    x = (sid.astype(np.uint64) * np.uint64(2654435761) + j.astype(np.uint64) * np.uint64(40503) + np.uint64(SEED)) % np.uint64(4294967291)
    return x.astype(np.float64)


def pick_top(rows: Dict[str, np.ndarray], f: str, sel: np.ndarray, n: int) -> np.ndarray:
    """Row indices of the top-n by input f among `sel` (row indices of one date)."""
    if len(sel) == 0:
        return sel
    order = np.lexsort((-tie_break(rows["sid"][sel], rows["j"][sel]), -rows[f][sel].astype(np.float64)))
    return sel[order[:n]]


def simulate_live_exit(s: pp.Series, i: int, regime_by_day: Dict[int, str], as_of: str,
                       spy_days: np.ndarray, spy_o: np.ndarray, spy_c: np.ndarray, cost: np.ndarray) -> Optional[dict]:
    """Enter at bar i+1's open (+ cost at bar i), walk the live exit ladder
    (live_exit_policy, as backtest_v2 exit_model="live": ATR% and regime from
    bars strictly before each bar), every fill net of that bar's cost."""
    import backtest_v2 as bt
    import live_exit_policy as lep
    n = len(s.c)
    if i + 1 >= n:
        return None
    bars = {"date": s.dates, "high": s.h, "low": s.l, "close": s.c}
    entry = float(s.o[i + 1] * (1 + cost[i]))
    lp = lep.LivePosition(entry, bt.live_atr_pct(bars, i + 1), lep.regime_class(regime_by_day.get(int(s.days[i]), "NEUTRAL")))
    entry_date = s.dates[i + 1]
    proceeds, n_fills, reason, k = 0.0, 0, None, i + 1
    last = min(n - 1, i + LIVE_EXIT_MAX_BARS)
    for k in range(i + 1, last + 1):
        lp.refresh(bt.live_atr_pct(bars, k), lep.regime_class(regime_by_day.get(int(s.days[k - 1]), "NEUTRAL")))
        closed = lep.walk_bar(lp, float(s.o[k]), float(s.h[k]), float(s.l[k]), float(s.c[k]), lep.time_eligible(entry_date, s.dates[k]))
        for qty, px in lp.fills[n_fills:]:
            proceeds += qty * px * (1 - cost[k])
        n_fills = len(lp.fills)
        if closed:
            reason = lep.REASON_NAMES.get(lp.exit_reason, str(lp.exit_reason))
            break
    if reason is None:
        rem = 1.0 - sum(q for q, _ in lp.fills)
        if k == n - 1 and s.ends_before(as_of):
            proceeds += rem * float(s.c[k]) * (1 - cost[k]) * (1 + delisting_return(s))
            reason = "delisted"
        elif k == i + LIVE_EXIT_MAX_BARS:
            proceeds += rem * float(s.c[k]) * (1 - cost[k])
            reason = "max_hold"
        else:
            return None                                   # still open at the panel's end: not matured
    ke = int(np.searchsorted(spy_days, s.days[i + 1]))
    kx = int(np.searchsorted(spy_days, s.days[k], side="right")) - 1
    spy = float(spy_c[kx] / spy_o[ke] - 1) if 0 <= ke < len(spy_days) and kx >= ke else float("nan")
    return {"net": proceeds / entry - 1, "spy": spy, "bars": k - i, "reason": reason}


def live_exit_pass(build_dir: str, picks: Dict[int, set], regime_by_day: Dict[int, str], as_of: str,
                   spy_days: np.ndarray, spy_o: np.ndarray, spy_c: np.ndarray, rebal: np.ndarray,
                   series_keys: List[str], max_rss_mb: float) -> Dict[Tuple[int, int], dict]:
    """picks: sid -> {rebalance position j}. Re-streams the panel; decodes
    only what it needs; the series order is pass 1's (sid = stream position)."""
    out: Dict[Tuple[int, int], dict] = {}
    want = {series_keys[sid]: sid for sid in picks}
    rebal_days = spy_days[rebal]
    seen = 0
    for s in pp.iter_series(build_dir):
        seen += 1
        sid = want.get(s.key)
        if sid is None:
            continue
        cost = cost_per_side(raw_share_volume(s), pp.pit_price(s))
        for j in sorted(picks[sid]):
            i = s.index_on(int(rebal_days[j]))
            if i < 0:
                continue
            r = simulate_live_exit(s, i, regime_by_day, as_of, spy_days, spy_o, spy_c, cost)
            if r is not None:
                out[(sid, j)] = r
        if seen % RSS_CHECK_EVERY == 0 and rss_mb() > max_rss_mb:
            raise MemoryBudgetExceeded(f"RSS {rss_mb():.0f} MB > budget {max_rss_mb:.0f} MB in the live-exit pass")
    return out


# ── evaluation ───────────────────────────────────────────────────────────────
def _nw_lags(h: int) -> int:
    return max(1, math.ceil(h / 5))


def spy_forward(spy_o: np.ndarray, spy_c: np.ndarray, rebal: np.ndarray, hz: int) -> np.ndarray:
    """SPY over the same holding: next open -> close hz days after the decision (gross)."""
    out = np.full(len(rebal), np.nan)
    ok = rebal + hz < len(spy_c)
    out[ok] = spy_c[rebal[ok] + hz] / spy_o[rebal[ok] + 1] - 1
    return out


def evaluate(rows: Dict[str, np.ndarray], rebal: np.ndarray, spy_days: np.ndarray, spy_o: np.ndarray, spy_c: np.ndarray,
             labels: Dict[str, List[Optional[str]]], live: Dict[Tuple[int, int], dict],
             picks_by: Dict[str, Dict[int, np.ndarray]], random_live: Dict[int, List[Tuple[int, int]]]) -> Tuple[dict, dict]:
    """Per input x horizon: IC, quintiles, top-N vs SPY and random, splits."""
    rebal_days = spy_days[rebal]
    order = np.argsort(rows["j"], kind="stable")
    j_sorted = rows["j"][order]
    n_dates = len(rebal_days)
    bounds = np.searchsorted(j_sorted, np.arange(n_dates + 1))
    by_date = [order[bounds[d]:bounds[d + 1]] for d in range(n_dates)]
    years = [str(date.fromordinal(int(d) + pp._EPOCH_ORD).year) for d in rebal_days]
    m_ic_tests = len(INPUTS) * len(HORIZONS)

    # liquidity terciles per date (trailing dollar volume), a size proxy
    liq = np.empty(len(rows["sid"]), dtype=np.int8)
    for sel in by_date:
        if len(sel):
            r = rankdata(rows["dv"][sel])
            liq[sel] = np.minimum(2, ((r - 0.5) / len(sel) * 3).astype(int))
    liq_names = ("low", "mid", "high")

    results: Dict[str, dict] = {}
    sharpe_trials: List[Tuple[str, int, dict]] = []
    baselines = {}
    for hz in HORIZONS:
        net, gross = rows[f"net{hz}"].astype(np.float64), rows[f"gross{hz}"].astype(np.float64)
        # SPY and the random-entry base rate per date
        uni = np.full(n_dates, np.nan)
        uni_hit = np.full(n_dates, np.nan)
        for d, sel in enumerate(by_date):
            y = net[sel]
            y = y[np.isfinite(y)]
            if len(y) >= MIN_NAMES_PER_DATE:
                uni[d] = y.mean()
                uni_hit[d] = (y > 0).mean()
        spy_f = spy_forward(spy_o, spy_c, rebal, hz)
        has = np.isfinite(uni)
        baselines[str(hz)] = {"random_entry_mean_net_pct": _pct(np.nanmean(uni)) if has.any() else None,
                              "random_entry_hit_rate": _round(np.nanmean(uni_hit)) if has.any() else None,
                              "spy_mean_pct": _pct(np.nanmean(spy_f[has])) if has.any() else None,
                              "random_entry_minus_spy_pct": _pct(newey_west_t(uni - spy_f, _nw_lags(hz))[0]) if has.any() else None}
        for f in INPUTS:
            x_all = rows[f].astype(np.float64)
            ic = np.full(n_dates, np.nan)
            ic_g = np.full(n_dates, np.nan)
            qmeans = np.full((n_dates, 5), np.nan)
            qcount = np.zeros((n_dates, 5))
            ls_net = np.full(n_dates, np.nan)
            ls_gross = np.full(n_dates, np.nan)
            top = np.full(n_dates, np.nan)
            top_hit = []
            ic_liq = {b: np.full(n_dates, np.nan) for b in range(3)}
            for d, sel in enumerate(by_date):
                y = net[sel]
                ok = np.isfinite(y)
                sel_ok = sel[ok]
                if len(sel_ok) < MIN_NAMES_PER_DATE:
                    continue
                x, y, yg = x_all[sel_ok], net[sel_ok], gross[sel_ok]
                ic[d] = spearman(x, y)
                ic_g[d] = spearman(x, yg)
                q = quintile_of(x)
                for b in range(5):
                    mb = q == b
                    if mb.any():
                        qmeans[d, b] = y[mb].mean()
                        qcount[d, b] = mb.sum()
                if (q == 4).any() and (q == 0).any():
                    c5 = (yg[q == 4] - y[q == 4]).mean()          # round-trip cost drag of the long leg
                    c1 = (yg[q == 0] - y[q == 0]).mean()
                    ls_gross[d] = yg[q == 4].mean() - yg[q == 0].mean()
                    ls_net[d] = ls_gross[d] - c5 - c1
                tp = pick_top(rows, f, sel_ok, TOP_N)
                top[d] = net[tp].mean()
                top_hit.extend((net[tp] > 0).tolist())
                for b in range(3):
                    mb = liq[sel_ok] == b
                    if mb.sum() >= MIN_NAMES_PER_DATE // 3:
                        ic_liq[b][d] = spearman(x[mb], y[mb])
            lags = _nw_lags(hz)
            m_ic, t_ic = newey_west_t(ic, lags)
            p_ic = two_sided_p(t_ic)
            m_ls, t_ls = newey_west_t(ls_net, lags)
            ex_rand = top - uni
            m_ex, t_ex = newey_west_t(ex_rand, lags)
            m_spy, t_spy = newey_west_t(top - spy_f, lags)
            step = _nw_lags(hz) if hz > 5 else 1
            nonover = ex_rand[::step]
            st = sharpe_stats(nonover)
            if st:
                sharpe_trials.append((f, hz, st))
            n_ok = int(np.isfinite(ic).sum())
            split = {}
            for name, lab in labels.items():
                groups: Dict[str, List[int]] = {}
                for d in range(n_dates):
                    if lab[d] is not None and np.isfinite(ic[d]):
                        groups.setdefault(lab[d], []).append(d)
                split[name] = {g: _split_block(ic, ls_net, ex_rand, ds, lags) for g, ds in sorted(groups.items())}
            by_year: Dict[str, List[int]] = {}
            for d in range(n_dates):
                if np.isfinite(ic[d]):
                    by_year.setdefault(years[d], []).append(d)
            results.setdefault(f, {"horizons": {}})["horizons"][str(hz)] = {
                "ic": {"mean": _round(m_ic), "t_nw": _round(t_ic, 3), "p": _round(p_ic, 6),
                       "p_bonferroni": _round(bonferroni(p_ic, m_ic_tests), 6), "hit_rate": _round(np.nanmean(ic > 0) if n_ok else None),
                       "std": _round(np.nanstd(ic)) if n_ok else None, "n_dates": n_ok,
                       "mean_on_gross": _round(np.nanmean(ic_g)) if np.isfinite(ic_g).any() else None,
                       "nw_lags": lags},
                "quintiles": {"mean_net_pct": [_pct(np.nanmean(qmeans[:, b])) if np.isfinite(qmeans[:, b]).any() else None for b in range(5)],
                              "mean_names": [_round(qcount[:, b][qcount[:, b] > 0].mean(), 1) if (qcount[:, b] > 0).any() else None for b in range(5)],
                              "spread_gross_pct": _pct(np.nanmean(ls_gross)) if np.isfinite(ls_gross).any() else None,
                              "spread_net_long_short_pct": _pct(m_ls), "t_nw_long_short": _round(t_ls, 3)},
                "top_n": {"n": TOP_N, "mean_net_pct": _pct(np.nanmean(top)) if np.isfinite(top).any() else None,
                          "random_entry_net_pct": _pct(np.nanmean(uni[np.isfinite(top)])) if np.isfinite(top).any() else None,
                          "spy_pct": _pct(np.nanmean(spy_f[np.isfinite(top)])) if np.isfinite(top).any() else None,
                          "excess_vs_spy_pct": _pct(m_spy), "t_nw_vs_spy": _round(t_spy, 3),
                          "excess_vs_random_pct": _pct(m_ex), "t_nw_vs_random": _round(t_ex, 3),
                          "hit_rate": _round(float(np.mean(top_hit))) if top_hit else None,
                          "random_hit_rate": _round(np.nanmean(uni_hit[np.isfinite(top)])) if np.isfinite(top).any() else None,
                          "sharpe_excess_vs_random_ann": _round(st["sr"] * math.sqrt(252 / hz), 3) if st else None,
                          "sharpe_obs": st["n"] if st else 0},
                "by_regime": split.get("regime", {}),
                "by_spy_200d": split.get("spy_200d", {}),
                "by_year": {y: _split_block(ic, ls_net, ex_rand, ds, lags) for y, ds in sorted(by_year.items())},
                "by_liquidity": {liq_names[b]: {"n_dates": int(np.isfinite(ic_liq[b]).sum()),
                                                "mean_ic": _round(newey_west_t(ic_liq[b], lags)[0]),
                                                "t_nw": _round(newey_west_t(ic_liq[b], lags)[1], 3)} for b in range(3)},
            }
    # Deflated Sharpe over every top-N Sharpe estimate. The trials have
    # different period lengths (5d weekly, 20d every 4th week), so the
    # cross-trial variance is taken on ANNUALIZED Sharpes and the expected
    # maximum is converted back to each trial's own period before the PSR.
    ann = {hz: math.sqrt(252 / hz) for hz in HORIZONS}
    srs_ann = [t[2]["sr"] * ann[t[1]] for t in sharpe_trials]
    var_sr = float(np.var(srs_ann, ddof=1)) if len(srs_ann) > 1 else 0.0
    sr0_ann = expected_max_sharpe(len(srs_ann), var_sr)
    for f, hz, st in sharpe_trials:
        results[f]["horizons"][str(hz)]["top_n"]["dsr"] = _round(deflated_sharpe(st["sr"], st["n"], st["skew"], st["kurt"], sr0_ann / ann[hz]), 4)
    for f in INPUTS:
        results[f]["live_exit"] = _live_block(f, picks_by.get(f, {}), live, random_live, rows)
    mt = {"ic_tests": m_ic_tests, "bonferroni_alpha": round(ALPHA / m_ic_tests, 6), "sharpe_trials": len(srs_ann),
          "sharpe_trials_var_annualized": _round(var_sr, 6), "expected_max_sharpe_annualized": _round(sr0_ann, 4),
          "note": "IC p-values are Bonferroni-adjusted over every input x horizon; DSR deflates each top-N excess-vs-random Sharpe by the expected maximum of the trials run."}
    return results, {"multiple_testing": mt, "baselines": baselines}


def _split_block(ic, ls_net, ex_rand, ds, lags) -> dict:
    ds = list(ds)
    m_ic, t_ic = newey_west_t(ic[ds], lags)
    m_ls, _ = newey_west_t(ls_net[ds], lags)
    m_ex, _ = newey_west_t(ex_rand[ds], lags)
    return {"n_dates": len(ds), "mean_ic": _round(m_ic), "t_nw": _round(t_ic, 3),
            "spread_net_long_short_pct": _pct(m_ls), "top_n_excess_vs_random_pct": _pct(m_ex)}


def _live_block(f: str, picks: Dict[int, np.ndarray], live: Dict[Tuple[int, int], dict],
                random_live: Dict[int, List[Tuple[int, int]]], rows: Dict[str, np.ndarray]) -> dict:
    per_date, rnd_date, nets, spys, holds, rnd_nets, reasons = [], [], [], [], [], [], {}
    for j, ridx in picks.items():
        rs = [live.get((int(rows["sid"][r]), int(j))) for r in ridx]
        rs = [r for r in rs if r]
        rr = [live.get(p) for p in random_live.get(j, [])]
        rr = [r for r in rr if r]
        if not rs or not rr:
            continue
        per_date.append(np.mean([r["net"] for r in rs]))
        rnd_date.append(np.mean([r["net"] for r in rr]))
        for r in rs:
            nets.append(r["net"])
            if math.isfinite(r["spy"]):
                spys.append(r["spy"])
            holds.append(r["bars"])
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        rnd_nets.extend(r["net"] for r in rr)
    if not per_date:
        return {"n_picks": 0, "reason": "no matured live-exit positions"}
    ex = np.asarray(per_date) - np.asarray(rnd_date)
    m, t = newey_west_t(ex, 2)
    return {"n_picks": len(nets), "n_dates": len(per_date), "mean_net_pct": _pct(np.mean(nets)),
            "spy_same_window_pct": _pct(np.mean(spys)) if spys else None,
            "random_entry_net_pct": _pct(np.mean(rnd_nets)), "excess_vs_random_pct": _pct(m), "t_nw_vs_random": _round(t, 3),
            "hit_rate": _round(float(np.mean(np.asarray(nets) > 0))), "random_hit_rate": _round(float(np.mean(np.asarray(rnd_nets) > 0))),
            "mean_hold_bars": _round(float(np.mean(holds)), 2), "exit_reasons": dict(sorted(reasons.items())),
            "model": "live_exit_policy (backtest_v2 exit_model='live' walk), fractional thirds, max hold %d bars" % LIVE_EXIT_MAX_BARS}


def verdict(res: dict) -> Tuple[str, str]:
    hz = res["horizons"].get(str(PRIMARY_HORIZON), {})
    ic = hz.get("ic", {})
    if not ic or (ic.get("n_dates") or 0) < MIN_DATES_FOR_VERDICT:
        return "insufficient", f"fewer than {MIN_DATES_FOR_VERDICT} weekly dates with an IC"
    pb, m = ic.get("p_bonferroni"), ic.get("mean") or 0.0
    top = hz.get("top_n", {})
    tail = f"top-{TOP_N} net vs random entry {top.get('excess_vs_random_pct')} pts/period (t {top.get('t_nw_vs_random')})"
    if pb is not None and pb < ALPHA:
        return ("positive" if m > 0 else "negative"), f"{PRIMARY_HORIZON}d IC {m:+.4f} survives Bonferroni (p_adj {pb:.4g}); {tail}"
    return "null", f"{PRIMARY_HORIZON}d IC {m:+.4f} does not survive Bonferroni (p_adj {pb}); {tail}"


def brain_items(inputs: dict, window: dict, as_of: str) -> dict:
    """The Brain page's LongItem shape, keyed by Brain component id."""
    out = {}
    for f in INPUTS:
        r = inputs[f]
        hz = r["horizons"].get(str(PRIMARY_HORIZON), {})
        ic, top = hz.get("ic", {}), hz.get("top_n", {})
        out[f] = {
            "id": f"walkforward_{f}", "live": True, "title": f"{INPUT_INFO[f][0]} — 2016+ point-in-time walk-forward",
            "source_file": "DATA_DIR/walkforward/latest.json", "date": as_of,
            "horizon": f"{window.get('first_rebalance')} → {window.get('last_rebalance')}, weekly, {PRIMARY_HORIZON}d and 20d holds",
            "finding": (f"{PRIMARY_HORIZON}d rank IC {ic.get('mean')} (t {ic.get('t_nw')}, Bonferroni p {ic.get('p_bonferroni')}, "
                        f"{ic.get('n_dates')} dates); top-{TOP_N} net {top.get('mean_net_pct')}%/period vs random entry "
                        f"{top.get('random_entry_net_pct')}% (DSR {top.get('dsr')})."),
            "compared_to": "random entry, same universe, dates and holding (and SPY)",
            "sign": r["verdict"], "variants_tried": len(INPUTS) * len(HORIZONS), "quotes": [],
        }
    return out


# ── run ──────────────────────────────────────────────────────────────────────
def _write_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)


def default_out_dir() -> str:
    import storage_config
    return storage_config.WALKFORWARD_DIR


def _find(build_dir: str, symbols: Sequence[str]) -> Dict[str, List[pp.Series]]:
    """One streaming pass collecting the benchmark series (SPY, VXX)."""
    out: Dict[str, List[pp.Series]] = {sym: [] for sym in symbols}
    for s in pp.iter_series(build_dir):
        if s.symbol in out:
            out[s.symbol].append(s)
    return out


def run(panel_root: Optional[str] = None, out_dir: Optional[str] = None, max_rss_mb: float = DEFAULT_MAX_RSS_MB,
        start: str = START, now: Optional[datetime] = None, code_version: Optional[str] = None) -> dict:
    t0 = time.time()
    out_dir = out_dir or default_out_dir()
    now = now or datetime.now(timezone.utc)
    build = pp.current_build(panel_root)
    if not build:
        res = {"status": "skipped", "reason": "no complete price-panel build (current.json absent)"}
        _write_atomic(os.path.join(out_dir, "last_run.json"), {**res, "finished_at": now.isoformat()})
        return res
    as_of = str(build["manifest"].get("as_of") or "")
    bench = _find(build["dir"], ("SPY", "VXX"))
    spy_list = bench["SPY"]
    if not spy_list or not as_of:
        res = {"status": "error", "reason": "panel has no SPY series or no as_of", "build_id": build["build_id"]}
        _write_atomic(os.path.join(out_dir, "last_run.json"), {**res, "finished_at": now.isoformat()})
        return res
    spy = max(spy_list, key=lambda s: len(s.c))
    spy_days, spy_c, spy_o = spy.days, spy.c, spy.o
    rebal = weekly_rebalance_indices(spy_days, pp.day_num(start))
    rebal = rebal[rebal >= 1]
    vxx = vxx_ratio_by_day(bench["VXX"])
    reg, vxx_missing = regime_labels(spy_days, spy_c, vxx)
    p200 = spy_200d_labels(spy_c)
    import backtest_v2 as bt
    reg_quality = "degraded" if vxx_missing > bt._VXX_COVERAGE_DEGRADED_THRESHOLD else "ok"

    series_keys: List[str] = []
    series_meta: List[dict] = []
    rows, st = collect_rows(build["dir"], spy_days, spy_c, rebal, as_of, max_rss_mb, series_keys, series_meta)
    rows, dups = dedupe_rows(rows)
    n_rows = len(rows["sid"])
    if n_rows == 0:
        res = {"status": "insufficient", "reason": "no eligible (name, date) rows", "build_id": build["build_id"]}
        _write_atomic(os.path.join(out_dir, "last_run.json"), {**res, "finished_at": now.isoformat()})
        return res

    # picks for the live-exit pass: every input's top-N + seeded random draws
    order = np.argsort(rows["j"], kind="stable")
    bounds = np.searchsorted(rows["j"][order], np.arange(len(rebal) + 1))
    picks_by: Dict[str, Dict[int, np.ndarray]] = {f: {} for f in INPUTS}
    random_live: Dict[int, List[Tuple[int, int]]] = {}
    need: Dict[int, set] = {}
    rng = random.Random(SEED)
    net_p = rows[f"net{PRIMARY_HORIZON}"]
    for j in range(len(rebal)):
        sel = order[bounds[j]:bounds[j + 1]]
        sel = sel[np.isfinite(net_p[sel])] if len(sel) else sel
        if len(sel) < MIN_NAMES_PER_DATE:
            continue
        for f in INPUTS:
            tp = pick_top(rows, f, sel, TOP_N)
            picks_by[f][j] = tp
            for r in tp:
                need.setdefault(int(rows["sid"][r]), set()).add(j)
        draws = []
        for _ in range(RANDOM_LIVE_DRAWS):
            for r in rng.sample(list(sel), min(TOP_N, len(sel))):
                draws.append((int(rows["sid"][r]), j))
                need.setdefault(int(rows["sid"][r]), set()).add(j)
        random_live[j] = draws
    regime_by_day = {int(d): reg[i] for i, d in enumerate(spy_days)}
    live = live_exit_pass(build["dir"], need, regime_by_day, as_of, spy_days, spy_o, spy_c, rebal, series_keys, max_rss_mb)

    labels = {"regime": [reg[int(k)] for k in rebal], "spy_200d": [p200[int(k)] for k in rebal]}
    inputs, extra = evaluate(rows, rebal, spy_days, spy_o, spy_c, labels, live, picks_by, random_live)
    for f in INPUTS:
        inputs[f]["label"], inputs[f]["live_definition"] = INPUT_INFO[f]
        inputs[f]["measurable"] = True
        inputs[f]["verdict"], inputs[f]["verdict_note"] = verdict(inputs[f])

    dates_with_ic = [int(rebal[j]) for j in range(len(rebal)) if bounds[j + 1] - bounds[j] >= MIN_NAMES_PER_DATE]
    per_date = np.diff(bounds)
    n_sym = len(st["symbols_used"])
    window = {"first_rebalance": pp.day_str(spy_days[rebal[0]]) if len(rebal) else None,
              "last_rebalance": pp.day_str(spy_days[rebal[-1]]) if len(rebal) else None,
              "n_rebalances": int(len(rebal)), "n_rebalances_scored": len(dates_with_ic),
              "rebalance": "weekly: the last trading day of each ISO week (SPY calendar)",
              "horizons_days": list(HORIZONS), "primary_horizon_days": PRIMARY_HORIZON,
              "entry": "next bar's open after the decision close", "exit": "close h trading days after the decision"}
    payload = {
        "schema": SCHEMA, "status": "ok", "generated_at": now.isoformat(), "as_of": as_of, "code_version": code_version,
        "panel": {"build_id": build["build_id"], "as_of": as_of, "completed_at": build.get("completed_at")},
        "window": window,
        "universe": {
            "n_series_in_panel": st["series_seen"], "n_series_used": st["series_used"], "n_symbols": n_sym,
            "delisted_symbol_share": _round(len(st["delisted_symbols_used"]) / n_sym) if n_sym else None,
            "delisted_row_share": _round(float(rows["delisted"].mean())),
            "rows": n_rows, "names_per_date": {"mean": _round(float(per_date.mean()), 1), "min": int(per_date.min()), "max": int(per_date.max())},
            "duplicates_removed": dups, "rows_without_raw_close": int(rows["no_raw"].sum()),
            "unenterable_rows": int(rows["unenterable"].sum()),
            "delisting_return_applied": {str(h): int(rows[f"dl{h}"].sum()) for h in HORIZONS},
            "filters": {"min_raw_close": MIN_PRICE, "min_day_raw_shares": MIN_VOLUME, "min_dollar_volume_20d": MIN_DOLLAR_VOLUME,
                        "symbol_rule": "no '.', <= 5 characters (live scanner)", "membership": "a bar ON the decision date; trailing windows only"},
        },
        "cost_model": {"per_side": "max(backtest_v2 trailing-20-bar raw-share-volume tier, half-tick / raw price)",
                       "tiers": [[t, c] for t, c in bt._LIQUIDITY_TIERS] + [["else", bt._ILLIQUID_COST_PCT]],
                       "entry_keyed_on": "decision bar", "exit_keyed_on": "exit bar", "bias": "optimistic for the thinnest names (no impact)"},
        "delisting_assumption": {"merger_or_rename": DELIST_RETURN_MERGER, "other": DELIST_RETURN_OTHER, "source": "Shumway (1997)"},
        "regime": {"labels": "regime_util.classify_regime_5level, backtest_v2.regime_series rules, on panel SPY + VXX (per-series 30-bar ratio)",
                   "vxx_missing_share": _round(vxx_missing), "quality": reg_quality, "proxy": "SPY close vs its 200-day SMA",
                   "note": "regime is a SPLIT of the results, never an input; labels use data up to each date only"},
        "multiple_testing": extra["multiple_testing"],
        "baselines": extra["baselines"],
        "prior": PRIOR,
        "inputs": inputs,
        "composite": "price_composite",
        "not_measurable": NOT_MEASURABLE,
        "brain_long_term": brain_items(inputs, window, as_of),
        "caveats": CAVEATS,
        "elapsed_s": round(time.time() - t0, 1),
        "peak_rss_mb": round(rss_mb(), 1),
    }
    _write_atomic(os.path.join(out_dir, "latest.json"), payload)
    summary = {"status": "ok", "build_id": build["build_id"], "as_of": as_of, "code_version": code_version, "rows": n_rows, "symbols": n_sym,
               "elapsed_s": payload["elapsed_s"], "peak_rss_mb": payload["peak_rss_mb"],
               "verdicts": {f: inputs[f]["verdict"] for f in INPUTS}}
    _write_atomic(os.path.join(out_dir, "last_run.json"), {**summary, "finished_at": datetime.now(timezone.utc).isoformat()})
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Point-in-time walk-forward test of the live stock score's price inputs.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--panel-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-rss-mb", type=float, default=DEFAULT_MAX_RSS_MB)
    ap.add_argument("--start", default=START)
    ap.add_argument("--code-version", default=None, help="package.json version of the code that ran (attribution)")
    ap.add_argument("--nice", type=int, default=0, help="raise this process's niceness first (the weekly job passes 19)")
    a = ap.parse_args(argv)
    if not a.run:
        ap.print_help()
        return 2
    if a.nice > 0:
        try:
            os.nice(a.nice)       # never compete with the trading loop / daemon for CPU
        except OSError as e:
            print(f"[walkforward] could not lower priority: {e}", file=sys.stderr)
    out_dir = a.out_dir or default_out_dir()
    try:
        res = run(a.panel_root, out_dir, a.max_rss_mb, a.start, code_version=a.code_version)
    except MemoryBudgetExceeded as e:
        res = {"status": "error", "reason": f"memory budget: {e}"}
        _write_atomic(os.path.join(out_dir, "last_run.json"), {**res, "finished_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps(res, separators=(",", ":")))
    return 0 if res.get("status") in ("ok", "skipped", "insufficient") else 1


if __name__ == "__main__":
    sys.exit(main())
