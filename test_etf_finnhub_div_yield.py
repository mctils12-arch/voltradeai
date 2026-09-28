"""Regression: Finnhub dividend yield units on the ETF builder holding card
(2026-09-28). dividendYieldIndicatedAnnual is a percent; the old code only
divided by 100 when the value was > 0.2, so a 0.15% yield displayed as 15%."""
import inspect

import etf_data_sources as eds


def test_small_yield_is_converted_not_left_as_decimal():
    assert eds.finnhub_pct_to_decimal(0.15) == 0.0015   # was left as 0.15 -> "15.00%"


def test_normal_and_zero_and_missing():
    assert eds.finnhub_pct_to_decimal(1.5) == 0.015
    assert eds.finnhub_pct_to_decimal(0.0) == 0.0
    assert eds.finnhub_pct_to_decimal(None) is None


def test_magnitude_guess_is_gone():
    src = inspect.getsource(eds.fetch_metrics_finnhub)
    assert '> 0.2' not in src
    assert "finnhub_pct_to_decimal(" in src
