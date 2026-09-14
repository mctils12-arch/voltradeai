"""test_eia930_solar_exceedance_pattern.py — pure-function tests for
scripts/eia930_solar_exceedance_pattern.py. No network; fetch_fueltype_window
(the only I/O function) is exercised only by running the script live, same
convention as every sibling eia930/eia860 gate-1 script's test file.
"""
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "eia930_solar_exceedance_pattern",
    os.path.join(os.path.dirname(__file__), "scripts", "eia930_solar_exceedance_pattern.py"))
sep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sep)


# ---- parse_period_date_hour ----

def test_parse_period_date_hour_splits_date_and_hour():
    assert sep.parse_period_date_hour("2026-09-13T04") == ("2026-09-13", 4)


def test_parse_period_date_hour_raises_on_unexpected_format():
    with pytest.raises(ValueError):
        sep.parse_period_date_hour("2026-09-13")


# ---- parse_rows ----

def test_parse_rows_parses_numeric_strings():
    rows, unparseable = sep.parse_rows([{"period": "2026-09-13T04", "value": "123.5"}])
    assert rows == [("2026-09-13T04", 123.5)]
    assert unparseable == 0


def test_parse_rows_treats_missing_value_as_none_not_a_drop():
    rows, unparseable = sep.parse_rows([{"period": "2026-09-13T04", "value": None},
                                         {"period": "2026-09-13T05", "value": ""}])
    assert rows == [("2026-09-13T04", None), ("2026-09-13T05", None)]
    assert unparseable == 0


def test_parse_rows_counts_non_numeric_value_as_unparseable_and_keeps_period():
    rows, unparseable = sep.parse_rows([{"period": "2026-09-13T04", "value": "not-a-number"}])
    assert rows == [("2026-09-13T04", None)]
    assert unparseable == 1


def test_parse_rows_empty_input():
    assert sep.parse_rows([]) == ([], 0)


# ---- exceedance_stats ----

def test_exceedance_stats_no_hours_exceed():
    rows = [("2026-09-01T10", 50.0), ("2026-09-01T11", 80.0)]
    stats = sep.exceedance_stats(rows, capacity_mw=100.0)
    assert stats["hours_total"] == 2
    assert stats["hours_exceeding"] == 0
    assert stats["exceeding_fraction"] == 0.0
    assert stats["distinct_days_with_exceedance"] == 0
    assert stats["max_ratio_of_capacity"] is None
    assert stats["exceeding_hour_of_day_histogram"] == {}
    assert stats["top_5_exceeding_hours"] == []


def test_exceedance_stats_sustained_diurnal_shape_many_days_same_hour():
    rows = [
        ("2026-09-01T18", 130.0), ("2026-09-02T18", 125.0), ("2026-09-03T18", 132.0),
        ("2026-09-01T10", 40.0),
    ]
    stats = sep.exceedance_stats(rows, capacity_mw=100.0)
    assert stats["hours_total"] == 4
    assert stats["hours_exceeding"] == 3
    assert stats["exceeding_fraction"] == 0.75
    assert stats["distinct_days_with_exceedance"] == 3
    assert stats["exceeding_hour_of_day_histogram"] == {18: 3}
    assert stats["max_ratio_of_capacity"] == 1.32


def test_exceedance_stats_isolated_spike_shape_one_day_only():
    rows = [("2026-09-01T14", 200.0), ("2026-09-02T14", 90.0), ("2026-09-03T14", 95.0)]
    stats = sep.exceedance_stats(rows, capacity_mw=100.0)
    assert stats["hours_exceeding"] == 1
    assert stats["distinct_days_with_exceedance"] == 1
    assert stats["exceeding_hour_of_day_histogram"] == {14: 1}


def test_exceedance_stats_respects_tolerance():
    rows = [("2026-09-01T14", 104.0)]
    assert sep.exceedance_stats(rows, capacity_mw=100.0, tolerance=0.0)["hours_exceeding"] == 1
    assert sep.exceedance_stats(rows, capacity_mw=100.0, tolerance=0.05)["hours_exceeding"] == 0


def test_exceedance_stats_none_values_excluded_from_totals_and_exceedance():
    rows = [("2026-09-01T14", None), ("2026-09-01T15", 50.0)]
    stats = sep.exceedance_stats(rows, capacity_mw=100.0)
    assert stats["hours_total"] == 1
    assert stats["hours_exceeding"] == 0


def test_exceedance_stats_zero_capacity_yields_none_ceiling_and_no_exceedance():
    rows = [("2026-09-01T14", 50.0)]
    stats = sep.exceedance_stats(rows, capacity_mw=0.0)
    assert stats["ceiling_mwh"] is None
    assert stats["hours_exceeding"] == 0
    assert stats["exceeding_fraction"] == 0.0


def test_exceedance_stats_empty_rows():
    stats = sep.exceedance_stats([], capacity_mw=100.0)
    assert stats["hours_total"] == 0
    assert stats["exceeding_fraction"] is None
    assert stats["max_ratio_of_capacity"] is None


def test_exceedance_stats_top_5_exceeding_hours_sorted_descending_and_capped():
    rows = [(f"2026-09-0{i}T12", 100.0 + i) for i in range(1, 8)]
    stats = sep.exceedance_stats(rows, capacity_mw=50.0)
    top5 = stats["top_5_exceeding_hours"]
    assert len(top5) == 5
    assert [t["value_mwh"] for t in top5] == [107.0, 106.0, 105.0, 104.0, 103.0]
    assert top5[0]["ratio_of_capacity"] == 2.14
