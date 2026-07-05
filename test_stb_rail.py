"""
test_stb_rail.py — STB EP724 battery (scripts/stb_rail.py). Pure-function
tests on synthetic row tuples shaped exactly like the real workbook
(header verified live 2026-07-05); no network, no xlsx dependency.
"""
import importlib.util
import os
from datetime import datetime

import pytest

_spec = importlib.util.spec_from_file_location(
    "stb_rail", os.path.join(os.path.dirname(__file__), "scripts", "stb_rail.py"))
stb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stb)

HEADER = ("Railroad/\nRegion", "Category No.", "Sub-Category", "Measure", "Variable", "Sub-Variable",
          datetime(2026, 6, 24), datetime(2026, 7, 1))


def rows(*data):
    return [HEADER, *data]


def test_parse_selects_carloads_speed_col_only():
    parsed = stb.parse_workbook(rows(
        ("BNSF", "11", "", "Weekly Carloads By 22 Commodity Categories", "Coal", "", 4000, 4100),
        ("BNSF", "1", "", "Average Train Speed  (MPH)", "System", "", 24.8, 25.0),
        ("BNSF", "1", "", "Average Train Speed  (MPH)", "Coal unit", "", 24.1, 24.1),   # not System -> dropped
        ("BNSF", "3", "", "Cars On Line (Count)", "Covered Hopper", "", 30000, 30100),
        ("BNSF", "2", "", "Average Terminal Dwell Time ", "ATLANTA", "", 20.0, 21.0),   # unselected measure
        ("UP", "11", "", "Weekly Carloads By 22 Commodity Categories", "Grain", "", 8000, 7900),
    ))
    assert parsed["weeks"] == ["2026-06-24", "2026-07-01"]
    keys = sorted(parsed["series"])
    assert keys == [
        "BNSF|Average Train Speed  (MPH)|System",
        "BNSF|Cars On Line (Count)|Covered Hopper",
        "BNSF|Weekly Carloads By 22 Commodity Categories|Coal",
        "UP|Weekly Carloads By 22 Commodity Categories|Grain",
    ]
    assert parsed["series"]["UP|Weekly Carloads By 22 Commodity Categories|Grain"] == [8000.0, 7900.0]


def test_parse_nonnumeric_is_null_never_zero():
    parsed = stb.parse_workbook(rows(
        ("CN", "11", "", "Weekly Carloads By 22 Commodity Categories", "Coke", "", "*", "1,234"),
        ("CP", "11", "", "Weekly Carloads By 22 Commodity Categories", "Coke", "", None, 55),
    ))
    assert parsed["series"]["CN|Weekly Carloads By 22 Commodity Categories|Coke"] == [None, 1234.0]
    assert parsed["series"]["CP|Weekly Carloads By 22 Commodity Categories|Coke"] == [None, 55.0]


def test_parse_short_rows_pad_with_null():
    parsed = stb.parse_workbook(rows(
        ("NS", "11", "", "Weekly Carloads By 22 Commodity Categories", "Chemicals", "", 900),
    ))
    assert parsed["series"]["NS|Weekly Carloads By 22 Commodity Categories|Chemicals"] == [900.0, None]


def test_parse_refuses_changed_week_axis():
    bad_header = HEADER[:6] + (datetime(2026, 6, 24), "not a date", datetime(2026, 7, 8))
    with pytest.raises(SystemExit):
        stb.parse_workbook([bad_header,
                            ("BNSF", "11", "", "Weekly Carloads By 22 Commodity Categories", "Coal", "", 1, 2, 3)])


def test_find_consolidated_url_newest_wins():
    html = (
        '<a href="https://www.stb.gov/wp-content/uploads/files/rsir/All%20Class%201%20Railroads/'
        'EP724%20Consolidated%20Data%20through%202026-06-24.xlsx">old</a>'
        '<a href="https://www.stb.gov/wp-content/uploads/files/rsir/All%20Class%201%20Railroads/'
        'EP724%20Consolidated%20Data%20through%202026-07-01.xlsx">new</a>'
    )
    assert stb.find_consolidated_url(html).endswith("2026-07-01.xlsx")
    with pytest.raises(SystemExit):
        stb.find_consolidated_url("<html>no links</html>")
