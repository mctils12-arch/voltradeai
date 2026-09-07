"""
test_un_comtrade_gate1.py — pure-function battery for
scripts/un_comtrade_gate1.py (parse_fred_csv/compute_ratios/evaluate_partner).
No network — fetch_fred_series (the one networked function) is exercised
live only by manually running the script, same convention as
test_jodi_eia_reconcile.py leaving its own live fetch untested in pytest.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "un_comtrade_gate1", os.path.join(os.path.dirname(__file__), "scripts", "un_comtrade_gate1.py"))
gate1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate1)


def test_parse_fred_csv_converts_millions_to_dollars():
    csv_text = "observation_date,IMPCH\n2025-01-01,41770.432020\n2025-02-01,31654.580012\n"
    out = gate1.parse_fred_csv(csv_text)
    assert out == {"202501": 41770432020.0, "202502": 31654580012.0}


def test_parse_fred_csv_skips_missing_marker_never_coerces_to_zero():
    csv_text = "observation_date,IMPCH\n2025-01-01,.\n2025-02-01,100.0\n"
    out = gate1.parse_fred_csv(csv_text)
    assert out == {"202502": 100000000.0}


def test_parse_fred_csv_raises_on_unexpected_header():
    import pytest
    with pytest.raises(ValueError, match="unexpected FRED CSV header"):
        gate1.parse_fred_csv("not,the,right,header\n1,2,3,4\n")


def test_compute_ratios_only_uses_overlapping_periods_with_nonnull_cif():
    points = [["202501", 100.0, 90.0], ["202502", None, 80.0], ["202503", 120.0, 110.0]]
    fred = {"202501": 50.0, "202503": 60.0}  # 202502 missing from FRED entirely
    ratios = gate1.compute_ratios(points, fred)
    assert ratios == [("202501", 2.0), ("202503", 2.0)]


def test_compute_ratios_skips_zero_fred_value_to_avoid_div_by_zero():
    points = [["202501", 100.0, 90.0]]
    fred = {"202501": 0.0}
    assert gate1.compute_ratios(points, fred) == []


def test_evaluate_partner_no_overlap_fails_with_reason():
    ev = gate1.evaluate_partner([])
    assert ev == {"n": 0, "mean": None, "stdev": None, "pass": False, "reason": "no overlapping periods"}


def test_evaluate_partner_passes_a_stable_offset_in_band():
    # mean 1.06, stdev ~0.0141 -- inside [1.00,1.20] and well under the 0.05 stability band
    ratios = [("202501", 1.05), ("202502", 1.07), ("202503", 1.06)]
    ev = gate1.evaluate_partner(ratios)
    assert ev["pass"] is True
    assert ev["n"] == 3
    assert round(ev["mean"], 4) == 1.06
    assert ev["reason"] == "OK"


def test_evaluate_partner_fails_on_mean_outside_band():
    ratios = [("202501", 1.5), ("202502", 1.5)]
    ev = gate1.evaluate_partner(ratios)
    assert ev["pass"] is False
    assert "mean" in ev["reason"]


def test_evaluate_partner_fails_on_unstable_ratio_even_if_mean_in_band():
    # mean is a fine 1.05 but the individual values swing wildly -- not a
    # real structural offset, should FAIL on stability even though the
    # mean alone would pass.
    ratios = [("202501", 0.80), ("202502", 1.30)]
    ev = gate1.evaluate_partner(ratios)
    assert 1.00 <= ev["mean"] <= 1.20
    assert ev["pass"] is False
    assert "unstable" in ev["reason"]


def test_evaluate_partner_fails_on_ratio_below_one():
    """A ratio < 1.00 (CIF cheaper than the customs-basis value) is itself
    a red flag per the module's own pre-registered prior — must fail, not
    just log a note."""
    ratios = [("202501", 0.95), ("202502", 0.94)]
    ev = gate1.evaluate_partner(ratios)
    assert ev["pass"] is False
