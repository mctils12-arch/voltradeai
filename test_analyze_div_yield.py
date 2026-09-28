# Regression: AAPL showed "Div. Yield 32.00%" (rate $1.08, price ~$340) because
# Yahoo's dividendYield became percent-form (0.32) and analyze.py multiplied by 100.
from analyze import derive_div_yield


def test_aapl_observed_case_is_about_a_third_of_a_percent():
    assert derive_div_yield(1.08, 340.0) == 0.32


def test_normal_yield():
    assert derive_div_yield(3.0, 100.0) == 3.0


def test_unknowable_returns_none():
    assert derive_div_yield(None, 100.0) is None
    assert derive_div_yield(0, 100.0) is None
    assert derive_div_yield(1.0, 0) is None
    assert derive_div_yield("x", 10) is None
