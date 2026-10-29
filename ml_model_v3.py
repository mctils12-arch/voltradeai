"""ml_model_v3.py — the v3 CANDIDATE stock model (SHADOW ONLY).

Human Phase 4 "Fix the model" (2026-10-27). The live v2 model
(ml_model_v2.py, voltrade_ml_v2.pkl) keeps its influence on trading; this
module trains and evaluates a candidate SIDE BY SIDE and logs its score next
to v2's on every shadow record. Promotion is a later, separate decision.

The candidate differs from v2 in exactly the four audited ways:
  1. honest CV (ml_cv.py): folds by decision date, embargo >= label horizon,
     early stopping on an inner slice of the training window;
  2. no earnings-surprise look-ahead (dropped: no point-in-time source);
  3. one feature definition for train and serve (ml_features_v3.py);
  4. ONE label that matches the real exit rules (this file, below).

LABEL (exit_aligned_v1_next_open). Decision after the close of bar i
(features read bars <= i). Entry at the OPEN of bar i+1 — the honest
analogue of the bot buying intraday in the session after the scan's last
completed bar. The position is then walked bar by bar under the LIVE exit
policy, live_exit_policy.LivePosition (the pure mirror of server/bot.ts
checkPositionOnTick: 2xATR R unit, scale-outs in thirds at 1R and the
regime's 2nd R, breakeven, phase trails, conditional TP, 7-calendar-day
flat time stop), the same policy shadow_portfolio's exit-aligned ruler uses.
A path still open after LABEL_CAP_BARS bars is marked to that close
(censored). return_pct is the tranche-weighted gross return; the label is
1 iff the return NET of the round-trip cost is > 0. Rows whose path is cut
off by the end of the data are dropped, never labeled from a stub.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

import ml_features_v3 as f3
from live_exit_policy import LivePosition, regime_class, time_eligible, walk_bar

LABEL_METHOD = "exit_aligned_v1_next_open"
# == shadow_portfolio.XA_CAP_BARS (censoring cap of the exit-aligned ruler;
# live has no horizon). Drift-pinned by test_ml_model_v3.py.
LABEL_CAP_BARS = 20
# == shadow_portfolio.XA_ROUND_TRIP_COST_PCT (system_config SLIPPAGE_PCT
# 0.05% per side x 2). Drift-pinned by test_ml_model_v3.py.
ROUND_TRIP_COST_PCT = 0.10
# CV embargo in trading dates: the label reads at most LABEL_CAP_BARS bars
# after the decision bar (entry bar + 19 more), so one more than the cap
# guarantees no training label window touches a test decision date.
EMBARGO_DATES = LABEL_CAP_BARS + 1

logger = logging.getLogger("ml_model_v3")


def regime_class_at(feats: dict) -> str:
    """bot.ts isBullish/isBearish class at decision time, from the row's own
    vxx_ratio / spy_vs_ma50 through system_config.get_market_regime (the
    5-level regime manage_positions writes into stop_state)."""
    vx, sp = feats.get("vxx_ratio"), feats.get("spy_vs_ma50")
    if vx is None or sp is None or math.isnan(vx) or math.isnan(sp):
        return "neutral"
    try:
        from system_config import get_market_regime
        return regime_class(get_market_regime(float(vx), float(sp)))
    except ImportError as e:
        logger.warning(f"system_config unavailable, regime class neutral: {e}")
        return "neutral"


def label_exit_aligned(entry: float, fwd: Dict[str, np.ndarray], start: int,
                       atr_pct: float, regime_cls: str,
                       cap: int = LABEL_CAP_BARS) -> Optional[dict]:
    """Walk bars fwd[start : start+cap] (arrays from f3.to_arrays) from an
    entry at `entry` dated fwd['t'][start]. None when the data ends before
    the position exits or reaches the cap."""
    if entry <= 0 or not atr_pct or atr_pct <= 0 or math.isnan(atr_pct):
        return None
    n = len(fwd["c"])
    entry_date = str(fwd["t"][start])
    pos = LivePosition(entry, atr_pct, regime_cls)
    used = 0
    for k in range(start, min(start + cap, n)):
        used += 1
        te = time_eligible(entry_date, str(fwd["t"][k]))
        if walk_bar(pos, float(fwd["o"][k]), float(fwd["h"][k]), float(fwd["l"][k]),
                    float(fwd["c"][k]), te):
            return _result(pos, str(fwd["t"][k]), used, censored=False)
    if used < cap:
        return None
    last = start + cap - 1
    px = float(fwd["c"][last])
    pos.max_px, pos.min_px = max(pos.max_px, px), min(pos.min_px, px)
    pos._close(px, "open_at_cap")
    return _result(pos, str(fwd["t"][last]), used, censored=True)


def _result(pos: LivePosition, exit_date: str, bars_held: int, censored: bool) -> dict:
    gross = sum(w * (px - pos.entry) / pos.entry * 100.0 for w, px in pos.fills)
    net = gross - ROUND_TRIP_COST_PCT
    return {"label": 1 if net > 0 else 0, "return_pct": gross, "net_return_pct": net,
            "exit_reason": pos.exit_reason, "exit_date": exit_date,
            "bars_held": bars_held, "censored": censored}


def build_dataset(all_bars: Dict[str, Sequence[dict]],
                  tickers: Optional[Sequence[str]] = None) -> dict:
    """Features (ml_features_v3 training path) + exit-aligned labels for every
    eligible (ticker, bar) whose full LABEL_CAP_BARS forward window exists
    (pass COMPLETED bars only — strip today's session first). Rows are returned in
    chronological order (decision date, then ticker)."""
    refs = f3.RefSeries({s: all_bars.get(s) or [] for s in f3.REF_SYMBOLS})
    names = sorted(t for t in (tickers or all_bars) if t not in f3.REF_SYMBOLS and all_bars.get(t))
    X, y, dates, ends, tick, gross, net, reg, why = [], [], [], [], [], [], [], [], []
    for t in names:
        a = f3.to_arrays(all_bars[t])
        n = len(a["c"])
        if n < f3.MIN_BARS + 1:
            continue
        spy = refs.spy_asof(list(a["t"]))
        # Only decisions whose WHOLE cap window exists: keeping late rows
        # that happened to exit early (and dropping the still-open ones)
        # would bias the sample's tail toward fast stops/time stops.
        for i in range(f3.MAX_LOOKBACK, n - LABEL_CAP_BARS):
            if not f3.row_eligible(a, i):
                continue
            feats = f3.features_at(a, i, refs, spy)
            if feats is None:
                continue
            rc = regime_class_at(feats)
            lab = label_exit_aligned(float(a["o"][i + 1]), a, i + 1, feats["atr_pct"], rc)
            if lab is None:
                continue
            X.append(f3.row_vector(feats))
            y.append(lab["label"])
            dates.append(str(a["t"][i]))
            ends.append(lab["exit_date"])
            tick.append(t)
            gross.append(lab["return_pct"])
            net.append(lab["net_return_pct"])
            reg.append(rc)
            why.append(lab["exit_reason"])
    order = sorted(range(len(y)), key=lambda k: (dates[k], tick[k]))
    pick = lambda seq, dt=None: np.array([seq[k] for k in order], dtype=dt)  # noqa: E731
    return {
        "X": pick(X, np.float32).reshape(len(order), len(f3.V3_FEATURE_COLS)),
        "y": pick(y, np.int8),
        "dates": pick(dates, object), "label_end_dates": pick(ends, object),
        "tickers": pick(tick, object), "regime_class": pick(reg, object),
        "return_pct": pick(gross, np.float64), "net_return_pct": pick(net, np.float64),
        "exit_reason": pick(why, object),
        "feature_names": list(f3.V3_FEATURE_COLS), "label_method": LABEL_METHOD,
    }
