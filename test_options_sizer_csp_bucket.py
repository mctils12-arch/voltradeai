# Regression test for the 2026-07-29 zero-CSP-fills root cause.
#
# Production trail: /api/diag/audit?type=T2-FAIL showed 100/100 recent CSP
# candidates rejected as "exceed position budget" / "Not enough capital",
# every single one carrying an identical "size_pct 1.00%" — the absolute
# floor, not a value that varies with score/vol/liquidity. /api/diag/orders
# confirmed zero us_option fills across a 17-day window (2026-07-10 through
# 2026-07-27) despite 200 us_equity fills in the same span.
#
# Root cause: both _dynamic_options_size() copies (options_execution.py and
# instrument_selector.py — the latter's own docstring says "Mirrors the
# logic in options_execution.py") sized every options trade off
# stats["overall"] — the blended win-rate across ALL trade types, including
# the documented -EV "stocks" bucket. calculate_position() already does the
# correct per-bucket lookup (position_sizing.py BUG #12 FIX 2026-05-03) for
# stock/ETF sizing, but that alignment never reached the options-only
# sizers. When the blended "overall" bucket computes non-positive Kelly,
# _kelly_fraction returns 0.0 and dynamic_pct collapses to the outer
# max(0.01, ...) floor regardless of any other scalar — exactly the
# observed "always 1.00%" pattern.
#
# Fix: both functions now read stats["by_strategy"]["csp_options"] (71.6%
# WR, +EV per position_sizing.py's own documented defaults) instead of
# "overall". Every call into either function is an options trade, so the
# bucket is known statically — no _infer_strategy() call needed.
import json

import options_execution
import instrument_selector
import position_sizing


def _neutral_scalars(module, monkeypatch):
    """Pin every non-Kelly scalar to 1.0 so only the bucket-selection logic
    under test can move the result."""
    monkeypatch.setattr(module, "_volatility_scalar", lambda *a, **kw: 1.0)
    monkeypatch.setattr(module, "_regime_scalar", lambda *a, **kw: 1.0)
    monkeypatch.setattr(module, "_earnings_scalar", lambda *a, **kw: 1.0)
    monkeypatch.setattr(module, "_time_scalar", lambda *a, **kw: 1.0)
    monkeypatch.setattr(module, "_portfolio_heat_scalar", lambda *a, **kw: 1.0)
    monkeypatch.setattr(module, "_liquidity_scalar", lambda *a, **kw: 1.0)


def _fake_stats(overall_negative_ev=True):
    """overall (blended) is negative-EV; csp_options bucket is clearly +EV —
    mirrors the documented real-world shape (stocks bucket drags "overall"
    down while the options bucket alone is the system's proven edge)."""
    overall = (
        {"win_rate": 0.40, "avg_win": 1.0, "avg_loss": 2.0, "total_trades": 500}
        if overall_negative_ev else
        {"win_rate": 0.716, "avg_win": 0.69, "avg_loss": 0.46, "total_trades": 500}
    )
    return {
        "overall": overall,
        "by_strategy": {
            "csp_options": {"win_rate": 0.716, "avg_win": 0.69, "avg_loss": 0.46, "trades": 500},
            "stocks": {"win_rate": 0.40, "avg_win": 1.0, "avg_loss": 2.0, "trades": 100},
        },
    }


def test_options_execution_uses_csp_options_bucket_not_overall(monkeypatch):
    monkeypatch.setattr(options_execution, "_get_historical_stats", lambda: _fake_stats())
    monkeypatch.setattr(options_execution, "_confidence_scalar", lambda *a, **kw: 1.0)
    _neutral_scalars(options_execution, monkeypatch)

    size = options_execution._dynamic_options_size(
        {"ticker": "XYZ", "price": 50, "score": 70}, equity=100_000,
        existing_positions=[], macro=None,
    )
    # Pre-fix: overall is negative-EV -> _kelly_fraction returns 0.0 -> the
    # outer clamp floors dynamic_pct at exactly 0.01 regardless of scalars.
    assert size > 0.01, (
        "still using the negative-EV 'overall' bucket instead of the "
        "positive-EV csp_options bucket"
    )
    # csp_options: kelly = (.716*.69 - .284*.46)/.69 -> third-Kelly ~0.176,
    # clamped to ABSOLUTE_MAX_POSITION_PCT (0.08) inside _kelly_fraction,
    # then halved by OPTIONS_LEVERAGE_DISCOUNT (0.50) with all other
    # scalars pinned to 1.0 -> 0.04.
    assert size == 0.04


def test_instrument_selector_uses_csp_options_bucket_not_overall(monkeypatch):
    monkeypatch.setattr(instrument_selector, "_get_historical_stats", lambda: _fake_stats())
    monkeypatch.setattr(instrument_selector, "_confidence_scalar_safe", lambda *a, **kw: 1.0)
    _neutral_scalars(instrument_selector, monkeypatch)

    size = instrument_selector._dynamic_options_size(
        {"ticker": "XYZ", "price": 50, "score": 70}, equity=100_000,
        existing_positions=[], macro=None,
    )
    assert size > 0.01
    assert size == 0.04  # same derivation as the options_execution.py test above


def test_options_execution_falls_back_to_overall_if_bucket_missing(monkeypatch):
    # Defensive: stats["by_strategy"] should always carry csp_options per
    # position_sizing.py's own defaults, but the .get(..., stats["overall"])
    # fallback must not crash if it's ever absent.
    stats = _fake_stats()
    del stats["by_strategy"]["csp_options"]
    monkeypatch.setattr(options_execution, "_get_historical_stats", lambda: stats)
    monkeypatch.setattr(options_execution, "_confidence_scalar", lambda *a, **kw: 1.0)
    _neutral_scalars(options_execution, monkeypatch)

    size = options_execution._dynamic_options_size(
        {"ticker": "XYZ", "price": 50, "score": 70}, equity=100_000,
        existing_positions=[], macro=None,
    )
    assert size == 0.01  # overall bucket here is negative-EV -> floors at 0.01


def test_real_defaults_csp_options_bucket_is_positive_ev():
    # Pin against position_sizing.py's actual shipped defaults (not a mock)
    # so a future edit to those defaults that breaks the +EV assumption
    # this fix relies on fails loudly here instead of silently reappearing
    # as a live zero-fill outage.
    stats = position_sizing._get_historical_stats()
    b = stats["by_strategy"]["csp_options"]
    kelly = position_sizing._kelly_fraction(b["win_rate"], b["avg_win"], b["avg_loss"])
    assert kelly > 0.01
