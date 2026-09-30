# Regression test for the 2026-09-30 fabricated-IV repair (analyze.py).
#
# Break pinned: analyze_ticker() set atm_iv = rv20 * 1.1 when no option chain
# yielded a plausible ATM IV, so VRP = 0.1 * rv20 — for rv20 > 50 that printed
# "Sell vol — implied vol is overpriced" and fed get_recommendation, from no
# implied-vol observation at all. Offline-safe: pure functions.
from analyze import compute_vrp, get_recommendation


def test_missing_iv_is_unknown_not_a_signal():
    vrp, regime, signal = compute_vrp(None, 80.0)   # old code: vrp=+8 -> "Sell vol"
    assert vrp is None
    assert regime == "unknown"
    assert "Sell vol" not in signal and "Buy vol" not in signal


def test_real_iv_regimes_unchanged():
    assert compute_vrp(70.0, 60.0) == (10.0, "high", "Sell vol — implied vol is overpriced vs realized")
    assert compute_vrp(50.0, 60.0)[:2] == (-10.0, "low")
    assert compute_vrp(61.0, 60.0)[:2] == (1.0, "neutral")


def test_recommendation_accepts_missing_vrp():
    # Must not raise TypeError on `None < -2`, and must not claim IV is rich.
    rec = get_recommendation("XYZ", 100.0, None, None, None, {"score": 80}, {}, {}, {})
    assert isinstance(rec, dict)
    assert rec.get("action") != "SELL PREMIUM"
