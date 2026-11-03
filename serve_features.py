"""
serve_features — serve-time ML feature construction that matches training.

[REPAIR] 2026-11-03 (approved item B3, T-BOT): the bot page's Issues panel
reported "12 / 34 model features never move at serve time" (ml_insight.
recent_scoring over the newest shadow records). Read against
ml_model_v2._compute_features (the training formulas) and bot_engine.
deep_score (the serve path), the 12 split into three kinds — and only one
kind is a serve-path defect:

  PER-TICKER FEATURES WRONGLY CONSTANT AT SERVE (bugs — fixed here):
    price_vs_52w_high  quick_result never carries high_52w, so deep_score
                       computed (price - price) / price = 0 for every name.
                       Training: (c - max(prior 252 highs)) / hi52 * 100.
    ewma_vol           deep_score read vol_metrics["ewma_vol"]; analyze.py
                       emits "ewma_rv" (and in ANNUALISED % — a different
                       unit), so the default 2 was served every time.
                       Training: std of the prior 20 daily returns * 100.
    range_pct          quick_scan hard-codes "range_pct": 0 on every quick
                       result. Training: (high - low) / close * 100 of the
                       decision bar.
    cross_sec_rank     deep_score ranked today's move against the `change`
                       field of Alpaca's most-actives screener; that payload
                       has no `change` field (symbol / volume / trade_count
                       only), so the population was always empty and the 0.5
                       default was served. Training: rank of today's % change
                       within the universe's moves that day.
    sector_momentum    deep_score served the sector ETF's daily % move for the
                       41 tickers in SECTOR_MAP and 0 for everyone else.
                       Training: the stock's 5-day return minus SPY's.
    trend_strength     |change| / ewma_vol, capped at 5 — with ewma_vol pinned
                       at 2 every candidate the scan selects (big movers)
                       saturated at the cap. Follows the ewma_vol fix.
  MARKET-WIDE FEATURE WRONGLY CONSTANT ACROSS DAYS (bug — fixed here):
    credit_spread      the TLT/HYG fetch asked the MULTI-symbol bars endpoint
                       for `limit: 30` — Alpaca applies that limit to the
                       TOTAL across symbols (shadow_portfolio's XA_FETCH_LIMIT
                       note), so one symbol came back short of the 22 bars
                       the 21-day spread needs and the 0.0 default was cached
                       for every scan. Live: one distinct value over 38 days.
  TRAINING-CONSTANT FEATURES (not serve bugs — the model never saw variance):
    intel_score, news_sentiment, insider_signal, days_to_earnings,
    earnings_surprise are 0.0 / 0.5 on EVERY training row by construction
    (_build_training_data never overrides _compute_features' defaults; the
    EPS-surprise look-ahead was removed 2026-10-27). Feeding real values at
    serve would put the model in leaves it never trained on; the honest fix
    is dropping them from FEATURE_COLS at a retrain (a model change, not a
    serve change). ml_insight classifies them, it does not count them.
  MARKET-WIDE FEATURES CONSTANT WITHIN A SCAN BY NATURE (not bugs):
    vxx_ratio, spy_vs_ma50, markov_state, regime_score, vol_of_vol,
    market_breadth, put_call_ratio take one value per macro snapshot, so a
    per-record mode fraction over a few hours of scans measures the scan
    cadence, not the serve path. ml_insight judges them ACROSS DAYS instead.

SHADOW-FIRST. VOLTRADE_SERVE_FEATURE_FIX selects the mode:
  off     today's exact behaviour — one scoring, no extra fetch, no record field
  shadow  (DEFAULT) both vectors are built and scored; the CURRENT score is
          served; the corrected score and the per-feature diff ride the shadow
          record (`serve_fix`) so the counterfactual is measurable before any
          promotion
  on      the corrected vector is served (and the current one is scored and
          recorded, so the comparison stays symmetric)

Everything here is pure (no I/O); bot_engine supplies the bars, closes, the
scan's change distribution and the corrected credit spread. The formulas are
pinned against ml_model_v2._compute_features by test_serve_features.py.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from source_validation import MARKET_WIDE_FEATURES  # one source of truth for "market-wide"

ENV_VAR = "VOLTRADE_SERVE_FEATURE_FIX"
MODES = ("off", "shadow", "on")
DEFAULT_MODE = "shadow"
_OFF_VALUES = frozenset({"off", "0", "false", "none"})
_ON_VALUES = frozenset({"on", "1", "true", "live"})

# The seven features this repair corrects, in FEATURE_COLS order.
FIXED_FEATURES = (
    "price_vs_52w_high", "ewma_vol", "range_pct", "sector_momentum",
    "cross_sec_rank", "trend_strength", "credit_spread",
)

# Constant on every training row by construction (see module docstring).
# test_serve_features.py asserts this set against ml_model_v2._build_training_data
# itself, so it cannot drift from the training pipeline unnoticed.
TRAINING_CONSTANT_FEATURES = frozenset({
    "intel_score", "news_sentiment", "insider_signal", "days_to_earnings", "earnings_surprise",
})

# Training's own liquidity floor for a row (ml_model_v2._compute_features:
# `volumes[idx] < 300_000 -> None`); the scan's change distribution for
# cross_sec_rank applies the same floor so the population matches.
SCAN_CHANGES_MIN_VOLUME = 300_000
# Training needs more than 10 names on the date before it ranks; below that
# the row carries 0.5 (_build_training_data). Same rule here.
CROSS_SEC_MIN_POPULATION = 11
# A scan's change distribution older than this is not "today's scan" any more
# (deep_score is also reachable from the look-up path outside a scan).
SCAN_CHANGES_MAX_AGE_S = 900.0

# Rounding, exactly as _compute_features rounds the same keys.
_ROUND = {
    "price_vs_52w_high": 2, "ewma_vol": 3, "range_pct": 3, "sector_momentum": 3,
    "cross_sec_rank": 3, "trend_strength": 3, "credit_spread": 4,
}


def serve_fix_mode(env=None) -> str:
    """off | shadow | on from VOLTRADE_SERVE_FEATURE_FIX (case-insensitive).
    Absent or unrecognised -> the default, shadow: the served score stays
    today's and the corrected one is only recorded."""
    source = env if env is not None else os.environ
    raw = str(source.get(ENV_VAR, "") or "").strip().lower()
    if raw in _OFF_VALUES:
        return "off"
    if raw in _ON_VALUES:
        return "on"
    return DEFAULT_MODE


def feature_kind(name: str) -> str:
    """per_ticker | market_wide | training_constant — how ml_insight must judge
    constancy for this feature."""
    if name in TRAINING_CONSTANT_FEATURES:
        return "training_constant"
    if name in MARKET_WIDE_FEATURES:
        return "market_wide"
    return "per_ticker"


# ── training formulas (ml_model_v2._compute_features), serve-shaped ──────────
# Each takes the series up to and including the decision bar (training's idx
# is the LAST element) and returns an unrounded float, or None when the
# inputs cannot support the formula — None means "keep the current value",
# never a fabricated default.

def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def price_vs_52w_high(highs: Sequence[float], c: float) -> Optional[float]:
    """(c - hi52) / hi52 * 100, hi52 = max of the 252 highs BEFORE the decision
    bar (training: `max(highs[max(0, idx-252):idx])`, needs idx >= 5)."""
    idx = len(highs) - 1
    c = _f(c)
    if idx < 5 or c is None or c <= 0:
        return None
    prior = [h for h in (_f(x) for x in highs[max(0, idx - 252):idx]) if h is not None and h > 0]
    if not prior:
        return None
    hi52 = max(prior)
    return (c - hi52) / hi52 * 100


def ewma_vol_20(closes: Sequence[float]) -> Optional[float]:
    """std of the 20 daily returns before the decision bar, x100 (training:
    `rets` over i in [max(1, idx-20), idx); the return INTO the decision bar
    is excluded; population std via np.std)."""
    idx = len(closes) - 1
    if idx < 2:
        return None
    cl = [_f(x) for x in closes]
    rets = [(cl[i] - cl[i - 1]) / cl[i - 1]
            for i in range(max(1, idx - 20), idx)
            if cl[i] is not None and cl[i - 1] is not None and cl[i - 1] > 0]
    if not rets:
        return None
    return float(np.std(rets) * 100)


def range_pct(high, low, c) -> Optional[float]:
    """(high - low) / c * 100 of the decision bar."""
    h, l, c = _f(high), _f(low), _f(c)
    if h is None or l is None or c is None or c <= 0 or h < l:
        return None
    return (h - l) / c * 100


def cross_sec_rank(change_pct, universe_changes: Iterable[float]) -> Optional[float]:
    """Share of the universe's moves today at or below this one (training:
    `sum(x <= stock_chg) / len(day_changes)`, only when more than 10 names)."""
    chg = _f(change_pct)
    if chg is None:
        return None
    xs = [x for x in (_f(v) for v in universe_changes) if x is not None]
    if len(xs) < CROSS_SEC_MIN_POPULATION:
        return None
    return sum(1 for x in xs if x <= chg) / len(xs)


def sector_momentum_5d(closes: Sequence[float], spy_closes: Sequence[float]) -> Optional[float]:
    """stock 5-day return minus SPY 5-day return, in % (training's
    "sector momentum: simplified")."""
    if len(closes) < 6 or len(spy_closes) < 6:
        return None
    c0, c5 = _f(closes[-1]), _f(closes[-6])
    s0, s5 = _f(spy_closes[-1]), _f(spy_closes[-6])
    if None in (c0, c5, s0, s5) or c5 <= 0 or s5 <= 0:
        return None
    return (c0 - c5) / c5 * 100 - (s0 - s5) / s5 * 100


def trend_strength(change_pct, ewma_vol) -> Optional[float]:
    """min(|change_pct| / max(ewma_vol, 0.1), 5.0)."""
    chg, ev = _f(change_pct), _f(ewma_vol)
    if chg is None or ev is None:
        return None
    return min(abs(chg) / max(ev, 0.1), 5.0)


def credit_spread_21d(tlt_closes: Sequence[float], hyg_closes: Sequence[float]) -> Optional[float]:
    """(TLT 21-bar return) - (HYG 21-bar return); needs 22 closes of each
    (training: bars idx and idx-21)."""
    if len(tlt_closes) < 22 or len(hyg_closes) < 22:
        return None
    t0, t21 = _f(tlt_closes[-1]), _f(tlt_closes[-22])
    h0, h21 = _f(hyg_closes[-1]), _f(hyg_closes[-22])
    if None in (t0, t21, h0, h21) or t21 <= 0 or h21 <= 0:
        return None
    return (t0 / t21 - 1) - (h0 / h21 - 1)


# ── the corrected vector ─────────────────────────────────────────────────────

def cross_sec_population(change_pct, scan_changes: Optional[Dict[str, float]],
                         universe_symbols: Optional[Iterable[str]] = None,
                         symbol: Optional[str] = None) -> Optional[list]:
    """The moves to rank against. Training's universe is the most-active names
    plus the core list, so that subset of the scan's distribution comes first;
    the whole (volume-floored) scan distribution is the fallback; fewer than
    CROSS_SEC_MIN_POPULATION names -> None (rank stays at today's value). The
    candidate's own move is in the population exactly once, as in training
    (its scan entry, if any, is replaced by `change_pct`)."""
    if not isinstance(scan_changes, dict) or not scan_changes:
        return None
    own = _f(change_pct)
    if own is None:
        return None
    sym = str(symbol).upper() if symbol else None
    if universe_symbols:
        subset = [scan_changes[s] for s in {str(u).upper() for u in universe_symbols}
                  if s in scan_changes and s != sym]
        if len(subset) + 1 >= CROSS_SEC_MIN_POPULATION:
            return subset + [own]
    pop = [v for s, v in scan_changes.items() if s != sym] + [own]
    return pop if len(pop) >= CROSS_SEC_MIN_POPULATION else None


def fixed_feature_vector(current: dict, *, symbol: Optional[str] = None,
                         bars: Optional[Sequence[dict]] = None,
                         closes: Optional[Sequence[float]] = None, change_pct=None, price=None,
                         spy_closes: Optional[Sequence[float]] = None,
                         scan_changes: Optional[Dict[str, float]] = None,
                         universe_symbols: Optional[Iterable[str]] = None,
                         credit_spread_fixed=None) -> Tuple[dict, dict]:
    """Return (fixed_vector, diff). fixed_vector is `current` with every
    FIXED_FEATURES value replaced by the training formula wherever the inputs
    allow it; diff = {name: {"current": old, "fixed": new}} for the values that
    actually changed. A feature whose inputs are missing keeps the current
    value and is absent from diff — never a fabricated default."""
    cur = dict(current or {})
    fixed = dict(cur)
    diff: dict = {}

    def _set(name: str, val: Optional[float]) -> None:
        if val is None:
            return
        val = round(float(val), _ROUND[name])
        old = _f(cur.get(name))
        if old is None or abs(old - val) > 1e-9:
            fixed[name] = val
            diff[name] = {"current": cur.get(name), "fixed": val}

    bars = [b for b in (bars or []) if isinstance(b, dict)]
    if closes is None and bars:
        closes = [b.get("c") for b in bars]
    closes = [x for x in (_f(v) for v in (closes or [])) if x is not None]
    last = bars[-1] if bars else {}
    px = _f(price)
    if px is None or px <= 0:
        px = _f(last.get("c")) if last else (closes[-1] if closes else None)
    chg = _f(change_pct)
    if chg is None:
        chg = _f(cur.get("change_pct_today"))

    if bars and px:
        highs = [b.get("h", b.get("c")) for b in bars]
        _set("price_vs_52w_high", price_vs_52w_high(highs, px))
        _set("range_pct", range_pct(last.get("h"), last.get("l"), px))
    ev = ewma_vol_20(closes) if closes else None
    _set("ewma_vol", ev)
    if spy_closes:
        _set("sector_momentum", sector_momentum_5d(closes, [x for x in (_f(v) for v in spy_closes) if x is not None]))
    pop = cross_sec_population(chg, scan_changes, universe_symbols, symbol=symbol)
    if pop is not None:
        _set("cross_sec_rank", cross_sec_rank(chg, pop))
    # trend_strength follows the ewma_vol the model will actually see
    ev_for_trend = ev if ev is not None else _f(cur.get("ewma_vol"))
    _set("trend_strength", trend_strength(chg, ev_for_trend))
    _set("credit_spread", _f(credit_spread_fixed))
    return fixed, diff
