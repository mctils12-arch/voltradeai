"""
Mean Reversion Strategy — scores oversold stocks 0-100 for bounce potential.
"""

# Live thresholds — unchanged values, now named so a research script can
# override a subset without touching this file (see DEFAULT_THRESHOLDS
# usage in score() below). Added for the illiquid-universe re-thresholding
# ablation (research/open_questions.md, 2026-07-24 entry, LADDER PATH step
# 4) — a pure parametrization, zero behavior change for any existing
# caller (bot_engine.py, backtest_v2.py) since none pass `thresholds`.
DEFAULT_THRESHOLDS = {
    "rsi_extreme": 20, "rsi_extreme_pts": 40,
    "rsi_oversold": 30, "rsi_oversold_pts": 30,
    "rsi_mild": 40, "rsi_mild_pts": 15,
    "rsi_overbought": 70, "rsi_overbought_pts": -15,
    "chg_big": -10, "chg_big_pts": 30,
    "chg_med": -5, "chg_med_pts": 20,
    "chg_small": -3, "chg_small_pts": 10,
    "vr_high": 2, "vr_high_pts": 20,
    "vr_med": 1.5, "vr_med_pts": 10,
}


def score(rsi, change_pct_5d, volume_ratio, thresholds=None):
    if rsi is None: return {"score": 0, "signal": "NO DATA", "reason": "No data"}
    t = DEFAULT_THRESHOLDS if not thresholds else {**DEFAULT_THRESHOLDS, **thresholds}

    s = 0
    if rsi < t["rsi_extreme"]: s += t["rsi_extreme_pts"]
    elif rsi < t["rsi_oversold"]: s += t["rsi_oversold_pts"]
    elif rsi < t["rsi_mild"]: s += t["rsi_mild_pts"]
    elif rsi > t["rsi_overbought"]: s += t["rsi_overbought_pts"]

    if change_pct_5d and change_pct_5d < t["chg_big"]: s += t["chg_big_pts"]
    elif change_pct_5d and change_pct_5d < t["chg_med"]: s += t["chg_med_pts"]
    elif change_pct_5d and change_pct_5d < t["chg_small"]: s += t["chg_small_pts"]

    if volume_ratio and volume_ratio > t["vr_high"]: s += t["vr_high_pts"]
    elif volume_ratio and volume_ratio > t["vr_med"]: s += t["vr_med_pts"]

    s = max(0, min(100, s))
    sig = "STRONG BUY" if s >= 65 else "BUY" if s >= 45 else "WATCH" if s >= 25 else "NO EDGE"
    return {"score": s, "signal": sig, "reason": f"RSI: {rsi:.0f}, 5d drop: {(change_pct_5d or 0):.1f}%, vol: {(volume_ratio or 1):.1f}x"}
