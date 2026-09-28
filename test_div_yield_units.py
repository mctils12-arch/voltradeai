"""Regression: dividend yield units (2026-09-28). The live site showed AAPL
"Div. Yield 32.00%" beside "Div. Rate $1.08" on a ~$341 stock: Yahoo now
sends dividendYield already in percent (0.32) and the old "< 1 means decimal,
x100" rule multiplied it again. normalize_div_yield must never guess units
from magnitude."""
import analyze


def test_aapl_live_case_percent_units():
    assert analyze.normalize_div_yield(0.32, 1.08, 341.07) == 0.32


def test_same_yield_in_legacy_decimal_units():
    assert analyze.normalize_div_yield(0.0032, 1.08, 341.07) == 0.32


def test_high_yield_above_one_percent_unchanged():
    assert analyze.normalize_div_yield(3.1, 1.2, 38.0) == 3.1


def test_raw_value_inconsistent_with_rate_over_price_trusts_rate():
    # neither 5.0 nor 500 is near 1.08/341 -> use the independent rate/price
    assert analyze.normalize_div_yield(5.0, 1.08, 341.07) == 0.32


def test_fallbacks_without_rate_or_price():
    assert analyze.normalize_div_yield(None, None, 100.0, trailing_decimal=0.0031) == 0.31
    assert analyze.normalize_div_yield(0.45, None, None) == 0.45   # current upstream convention
    assert analyze.normalize_div_yield(None, None, None) is None
    assert analyze.normalize_div_yield("bad", None, None) is None


def test_the_old_magnitude_rule_is_gone():
    import inspect
    src = inspect.getsource(analyze.analyze_ticker)
    assert "_dy * 100 if _dy < 1" not in src
    assert "normalize_div_yield(" in src
