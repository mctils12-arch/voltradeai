"""
test_eia_weekly.py — EIA weekly stream battery (scripts/eia_weekly.py).
Pure-function tests on synthetic Data-1 rows shaped like the real sheets
(layout verified live 2026-07-05); no network.
"""
import importlib.util
import os

import pytest

xlrd = pytest.importorskip("xlrd")  # session dep, same as the tank-fill comparator

_spec = importlib.util.spec_from_file_location(
    "eia_weekly", os.path.join(os.path.dirname(__file__), "scripts", "eia_weekly.py"))
eia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eia)

DATEMODE = 0  # 1900-based
D_2026_06_26 = 46199.0  # xldate for 2026-06-26 under datemode 0


def test_parse_extracts_title_and_points():
    rows = [
        ["Back to Contents", "Data 1: Weekly U.S. Ending Stocks excluding SPR of Crude Oil", ""],
        ["Sourcekey", "WCESTUS1", ""],
        ["Date", "Weekly U.S. Ending Stocks excluding SPR of Crude Oil (Thousand Barrels)", ""],
        [D_2026_06_26 - 7, 415000.0],
        [D_2026_06_26, 416500.0],
    ]
    title, points = eia.parse_data_sheet(rows, DATEMODE)
    assert title == "Data 1: Weekly U.S. Ending Stocks excluding SPR of Crude Oil"
    assert points == [["2026-06-19", 415000.0], ["2026-06-26", 416500.0]]


def test_parse_skips_gaps_never_zero():
    rows = [
        ["x", "Data 1: Weekly something", ""],
        [D_2026_06_26 - 7, ""],            # missing value -> skipped
        [D_2026_06_26, 100.0],
        ["not a date", 5.0],               # malformed date cell -> skipped
    ]
    _, points = eia.parse_data_sheet(rows, DATEMODE)
    assert points == [["2026-06-26", 100.0]]


def test_series_table_keys_stable():
    keys = [s["key"] for s in eia.SERIES]
    assert keys == ["crude_stocks_us", "gasoline_stocks_us", "distillate_stocks_us",
                    "spr_crude_stocks", "natgas_working_l48"]
    for s in eia.SERIES:
        assert s["url"].startswith("https://www.eia.gov/dnav/")
