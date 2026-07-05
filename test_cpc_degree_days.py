"""
test_cpc_degree_days.py — CPC degree-days battery (pure parse functions;
fixture lines shaped exactly like the live files probed 2026-07-05).
"""
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "cpc_degree_days", os.path.join(os.path.dirname(__file__), "scripts", "cpc_degree_days.py"))
cpc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpc)

FILE = """Product: Daily Cooling Degree Days
Regions: CPC::Regions::StatesCONUS
Weights: Population
Region|20260101|20260102|20260103
AL|0|1|2
AR|0|0|M
TX|5|6
"""


def test_parse_states_file_real_shape():
    dates, states = cpc.parse_states_file(FILE)
    assert dates == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert states["AL"] == [0, 1, 2]
    assert states["AR"] == [0, 0, None], "non-numeric 'M' becomes None, never zero"
    assert states["TX"] == [5, 6, None], "short rows pad with None"


def test_parse_refuses_changed_format():
    with pytest.raises(ValueError):
        cpc.parse_states_file("Product: x\nno region header\n")
    with pytest.raises(ValueError):
        cpc.parse_states_file("Region|2026-01-01\nAL|1\n")  # dashed date = changed format


def test_parse_ignores_non_state_rows():
    dates, states = cpc.parse_states_file("Region|20260101\nAL|1\nCONUS_TOTAL|99\n")
    assert "CONUS_TOTAL" not in states, "only 2-char state codes are states"
    assert states == {"AL": [1]}
