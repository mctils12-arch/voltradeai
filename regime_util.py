#!/usr/bin/env python3
"""
VolTradeAI — Canonical Regime Classification
==============================================
Single source of truth for classifying market state as bear/neutral/bull.

PROBLEM THIS SOLVES
  Three files had their own regime logic with different VXX thresholds:
    - system_config.py:278   get_market_regime()      — 5 levels (PANIC..BULL)
    - ml_model_v2.py:83       _classify_regime()       — 3 levels (bear/neutral/bull)
    - markov_regime.py         MarkovRegime             — continuous from SPY returns

  Training and inference used DIFFERENT classifications, so the ML model
  sometimes saw labels at inference time that it had never been trained on.
  Additionally, a small change in threshold in one file wouldn't propagate
  to the others — you'd get silent mismatches.

FIX
  All three modules now import `classify_regime()` from this file as the
  canonical source. The existing functions in each module become thin
  wrappers that delegate here. This preserves backward compatibility while
  ensuring consistency.

DEFAULT THRESHOLDS
  Based on the existing ml_model_v2 values, which are calibrated to
  historical VXX distributions over the training period:

  bear:    vxx_ratio >= 1.15  OR  spy_vs_ma50 < 0.94
  bull:    vxx_ratio <= 0.95  AND spy_vs_ma50 >= 0.98
  else:    neutral

  system_config's 5-level scheme (PANIC/BEAR/CAUTION/NEUTRAL/BULL) maps on
  top of this: PANIC extends the bear zone for extreme VXX spikes, BULL
  is a stricter subset of the "bull" category.
"""

from typing import Tuple


# ── Thresholds — edit HERE to change regime behavior everywhere ────────────
BEAR_VXX_THRESHOLD = 1.15     # VXX/VIX ratio >= this → bear
BEAR_SPY_THRESHOLD = 0.94     # SPY/MA50 < this → bear (even if VXX calm)
BULL_VXX_THRESHOLD = 0.95     # VXX ratio <= this AND
BULL_SPY_THRESHOLD = 0.98     #   SPY/MA50 >= this → bull

# 5-level extensions (for system_config compatibility)
PANIC_VXX_THRESHOLD = 1.40    # VXX ratio >= this → panic
CAUTION_VXX_THRESHOLD = 1.05  # VXX ratio >= this → caution (sub-bear)


def classify_regime(vxx_ratio: float, spy_vs_ma50: float) -> str:
    """
    Canonical 3-level regime classification.

    Used directly by ml_model_v2._classify_regime for training/inference.

    Args:
        vxx_ratio:    current VXX / 30-day avg VXX  (1.0 = normal)
        spy_vs_ma50:  SPY / SPY 50-day MA  (1.0 = at MA, >1.0 = above)

    Returns:
        "bear" | "neutral" | "bull"
    """
    vxx = float(vxx_ratio) if vxx_ratio is not None else 1.0
    spy = float(spy_vs_ma50) if spy_vs_ma50 is not None else 1.0

    if vxx >= BEAR_VXX_THRESHOLD or spy < BEAR_SPY_THRESHOLD:
        return "bear"
    if vxx <= BULL_VXX_THRESHOLD and spy >= BULL_SPY_THRESHOLD:
        return "bull"
    return "neutral"


def classify_regime_5level(vxx_ratio: float, spy_vs_ma50: float,
                            spy_below_200_days: int = 0,
                            spy_above_200d: bool = True) -> str:
    """
    5-level regime classification for system_config compatibility.

    PANIC → BEAR → CAUTION → NEUTRAL → BULL

    Extends the 3-level classification with finer grain at both tails.
    PANIC carves out extreme VXX spikes; CAUTION carves out elevated-VXX
    sub-bear states; BULL requires full confirmation.

    Args:
        vxx_ratio:             VXX / 30-day avg
        spy_vs_ma50:           SPY / 50-day MA
        spy_below_200_days:    consecutive days SPY closed below 200-day MA
        spy_above_200d:        SPY currently above 200-day MA

    Returns:
        "PANIC" | "BEAR" | "CAUTION" | "NEUTRAL" | "BULL"
    """
    vxx = float(vxx_ratio) if vxx_ratio is not None else 1.0
    spy = float(spy_vs_ma50) if spy_vs_ma50 is not None else 1.0

    # PANIC: extreme VXX spike
    if vxx >= PANIC_VXX_THRESHOLD:
        return "PANIC"

    # BEAR: confirmed trend break
    # spy_below_200_days >= 5 means persistent breakdown, not noise
    if vxx >= BEAR_VXX_THRESHOLD or spy < BEAR_SPY_THRESHOLD:
        return "BEAR"
    if not spy_above_200d and spy_below_200_days >= 5:
        return "BEAR"

    # CAUTION: elevated vol but not confirmed bear
    if vxx >= CAUTION_VXX_THRESHOLD:
        return "CAUTION"

    # BULL: full confirmation
    if vxx <= BULL_VXX_THRESHOLD and spy >= BULL_SPY_THRESHOLD and spy_above_200d:
        return "BULL"

    return "NEUTRAL"


def regime_probability(vxx_ratio: float, spy_vs_ma50: float) -> Tuple[float, float, float]:
    """
    Continuous regime signal [p_bear, p_neutral, p_bull].
    Useful as a feature for the ML model rather than a discrete label.

    Soft thresholds using logistic-like transitions around the hard cutoffs.
    """
    import math
    vxx = float(vxx_ratio) if vxx_ratio is not None else 1.0
    spy = float(spy_vs_ma50) if spy_vs_ma50 is not None else 1.0

    # Bearish signal: VXX above 1.0 or SPY below 0.97 → probability mass
    bear_vxx = 1.0 / (1.0 + math.exp(-(vxx - 1.10) * 8))
    bear_spy = 1.0 / (1.0 + math.exp((spy - 0.96) * 15))
    p_bear = max(bear_vxx, bear_spy)

    # Bullish signal: VXX below 1.0 AND SPY above 1.0
    bull_vxx = 1.0 / (1.0 + math.exp((vxx - 0.95) * 15))
    bull_spy = 1.0 / (1.0 + math.exp(-(spy - 1.00) * 10))
    p_bull = min(bull_vxx, bull_spy)

    # Normalize so they sum to 1
    p_neutral = max(0.0, 1.0 - p_bear - p_bull)
    total = p_bear + p_neutral + p_bull
    if total > 0:
        p_bear /= total
        p_neutral /= total
        p_bull /= total

    return (p_bear, p_neutral, p_bull)


# ── Self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Regression test: verify common cases classify correctly
    tests = [
        # (vxx, spy, expected_3level, expected_5level)
        (1.50, 0.92, "bear",    "PANIC"),
        (1.25, 0.95, "bear",    "BEAR"),
        (1.10, 0.99, "neutral", "CAUTION"),
        (1.00, 1.00, "neutral", "NEUTRAL"),
        (0.90, 1.02, "bull",    "BULL"),
        (0.85, 1.05, "bull",    "BULL"),
    ]

    print("Regime classification tests:")
    for vxx, spy, exp3, exp5 in tests:
        got3 = classify_regime(vxx, spy)
        got5 = classify_regime_5level(vxx, spy)
        p = regime_probability(vxx, spy)
        status = "✓" if (got3 == exp3 and got5 == exp5) else "✗"
        print(f"  {status} VXX={vxx:.2f} SPY={spy:.2f}: "
              f"3-level={got3:<8} 5-level={got5:<8} "
              f"probs=[bear={p[0]:.2f} neut={p[1]:.2f} bull={p[2]:.2f}]")
