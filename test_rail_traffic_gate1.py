"""
test_rail_traffic_gate1.py — pure-function battery for
scripts/rail_traffic_gate1.py (parse_fred_csv/monthly_totals/
compute_ratios/evaluate). No network — fetch_fred_series (the one
networked function) is exercised live only by manually running the
script, same convention as test_un_comtrade_gate1.py leaving
fetch_fred_series untested in pytest.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "rail_traffic_gate1", os.path.join(os.path.dirname(__file__), "scripts", "rail_traffic_gate1.py"))
gate1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate1)


def test_parse_fred_csv_reads_raw_counts_no_unit_conversion():
    csv_text = "observation_date,RAILFRTCARLOADS\n2025-01-01,1064273.0\n2025-02-01,952181.0\n"
    out = gate1.parse_fred_csv(csv_text)
    assert out == {"2025-01": 1064273.0, "2025-02": 952181.0}


def test_parse_fred_csv_skips_missing_marker_never_coerces_to_zero():
    csv_text = "observation_date,RAILFRTCARLOADS\n2025-01-01,.\n2025-02-01,100.0\n"
    out = gate1.parse_fred_csv(csv_text)
    assert out == {"2025-02": 100.0}


def test_parse_fred_csv_raises_on_unexpected_header():
    import pytest
    with pytest.raises(ValueError, match="unexpected FRED CSV header"):
        gate1.parse_fred_csv("not,the,right,header\n1,2,3,4\n")


def _fixture_doc():
    # 5 weeks spanning two calendar months: 2025-01 gets 4 weeks (full),
    # 2025-02 gets 1 week (partial -- must be excluded by MIN_WEEKS_PER_MONTH).
    weeks = ["2025-01-03", "2025-01-10", "2025-01-17", "2025-01-24", "2025-02-07"]
    return {
        "weeks": weeks,
        "series": {
            # non-intermodal commodity, two railroads
            "BNSF|Weekly Carloads By 22 Commodity Categories|Grain": [100.0, 110.0, 90.0, 105.0, 50.0],
            "UP|Weekly Carloads By 22 Commodity Categories|Grain": [80.0, None, 70.0, 75.0, 40.0],
            # intermodal commodities
            "BNSF|Weekly Carloads By 22 Commodity Categories|Containers": [200.0, 210.0, 190.0, 205.0, 60.0],
            "UP|Weekly Carloads By 22 Commodity Categories|Trailers": [10.0, 12.0, 11.0, 9.0, 5.0],
            # a different measure entirely -- must never be summed in
            "BNSF|Average Train Speed  (MPH)|System": [22.1, 21.9, 22.4, 22.0, 21.8],
        },
    }


def test_monthly_totals_sums_only_the_carload_measure_and_skips_nulls():
    doc = _fixture_doc()
    totals = gate1.monthly_totals(doc, want_intermodal=False)
    # 2025-01 Grain: BNSF 100+110+90+105=405, UP 80+70+75=225 (None skipped) -> 630
    assert totals == {"2025-01": 630.0}


def test_monthly_totals_partial_month_excluded_by_min_weeks():
    doc = _fixture_doc()
    totals = gate1.monthly_totals(doc, want_intermodal=False)
    assert "2025-02" not in totals, "2025-02 has only 1 archived week, below MIN_WEEKS_PER_MONTH"


def test_monthly_totals_intermodal_split_is_disjoint_from_non_intermodal():
    doc = _fixture_doc()
    inter = gate1.monthly_totals(doc, want_intermodal=True)
    non_inter = gate1.monthly_totals(doc, want_intermodal=False)
    # 2025-01 Containers: 200+210+190+205=805, Trailers: 10+12+11+9=42 -> 847
    assert inter == {"2025-01": 847.0}
    assert set(inter) == set(non_inter)  # both halves cover the same archived months
    assert inter["2025-01"] != non_inter["2025-01"]  # but never double-count the same carloads


def test_compute_ratios_only_uses_overlapping_months_with_nonzero_fred():
    ours = {"2025-01": 630.0, "2025-02": 500.0, "2025-03": 400.0}
    fred = {"2025-01": 300.0, "2025-03": 0.0}  # 2025-02 missing from FRED, 2025-03 zero
    ratios = gate1.compute_ratios(ours, fred)
    assert ratios == [("2025-01", 2.1)]


def test_evaluate_no_overlap_fails_with_reason():
    ev = gate1.evaluate([])
    assert ev == {"n": 0, "mean": None, "cv": None, "pass": False, "reason": "no overlapping months"}


def test_evaluate_passes_a_stable_offset_in_band():
    # mean 1.5, low cv -- inside [1.0, 2.5] and well under the 0.20 cv bar
    ratios = [("2025-01", 1.45), ("2025-02", 1.52), ("2025-03", 1.53)]
    ev = gate1.evaluate(ratios)
    assert ev["pass"] is True
    assert ev["n"] == 3
    assert ev["reason"] == "OK"


def test_evaluate_fails_on_mean_outside_band():
    ratios = [("2025-01", 3.5), ("2025-02", 3.6)]
    ev = gate1.evaluate(ratios)
    assert ev["pass"] is False
    assert "mean" in ev["reason"]


def test_evaluate_fails_on_ratio_below_one():
    """A ratio < 1.00 (our multi-railroad sum smaller than the once-counted
    national total) is itself a red flag per the module's own pre-
    registered prior -- must fail, not just log a note."""
    ratios = [("2025-01", 0.92), ("2025-02", 0.95)]
    ev = gate1.evaluate(ratios)
    assert ev["pass"] is False
    assert "mean" in ev["reason"]


def test_evaluate_fails_on_unstable_ratio_even_if_mean_in_band():
    # mean is a fine 1.5 but swings wildly between months -- not a real
    # structural offset, should fail on cv even though the mean passes.
    ratios = [("2025-01", 1.0), ("2025-02", 2.0)]
    ev = gate1.evaluate(ratios)
    assert 1.0 <= ev["mean"] <= 2.5
    assert ev["pass"] is False
    assert "cv" in ev["reason"]


def test_live_archive_produces_two_disjoint_nonempty_monthly_series():
    """Coherence check against the real committed archive (no network) --
    both halves should have real, disjoint, non-trivial monthly volume."""
    import json
    archive_path = os.path.join(os.path.dirname(__file__), "datacore", "rail", "ep724_carloads.json")
    with open(archive_path) as f:
        doc = json.load(f)
    non_inter = gate1.monthly_totals(doc, want_intermodal=False)
    inter = gate1.monthly_totals(doc, want_intermodal=True)
    assert len(non_inter) > 12, "expects over a year of full months in the committed archive"
    assert len(inter) > 12
    for month in set(non_inter) & set(inter):
        assert non_inter[month] > 0 and inter[month] > 0
