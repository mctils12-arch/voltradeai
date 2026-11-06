#!/usr/bin/env python3
"""
scripts/flows_validation.py ([RESEARCH] measurement, 2026-11-06): its
statistics are right, and its data-derived month-end windows are the SAME
windows calendar_effects.py gives the live model — otherwise the validation
would measure something other than what ships.

Run: python3 -m pytest -q test_flows_validation.py
"""
import math
import os
import sys
from datetime import date, timedelta

_ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import calendar_effects as ce
import flows_validation as fv


def _sessions(start, end):
    out, d = [], start
    while d <= end:
        if ce.is_trading_day(d):
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def test_ols_spearman_hit_rate():
    x = [float(i) for i in range(50)]
    y = [2.0 * v + 1.0 + (0.01 if i % 2 else -0.01) for i, v in enumerate(x)]
    o = fv.ols(x, y)
    assert math.isclose(o["slope"], 2.0, rel_tol=1e-3) and o["t"] > 100 and o["corr"] > 0.999
    assert fv.ols(x, y, nw_lag=5)["t"] > 10
    assert math.isclose(fv.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert math.isclose(fv.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
    h = fv.hit_rate([1, 1, -1, -1], [-0.1, 0.2, 0.3, -0.4])
    assert h["hit"] == 0.5 and h["n"] == 4 and h["no_skill"] == 0.5
    assert math.isclose(fv.tstat_mean([1.0, 2.0, 3.0]), 2.0 / (1.0 / math.sqrt(3)))


def test_data_windows_match_the_live_calendar():
    days = _sessions(date(2026, 1, 2), date(2026, 12, 31))
    k = fv.k_to_month_end(days)
    jm = fv.session_of_month(days)
    for i, d in enumerate(days):
        if k[i] is None:
            continue
        dd = date.fromisoformat(d)
        assert k[i] == ce.sessions_to_month_end(dd), d
        assert jm[i] == ce.session_of_month(dd), d
        assert fv.phase_by_index(days, i, k[i], jm[i]) == ce.month_end_phase(dd), d


def test_month_end_index_and_last_month_open():
    days = ["2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03"]
    assert fv.month_end_index(days) == [False, True, False, False]
    assert fv.k_to_month_end(days) == [1, 0, None, None], "the open month is not scored"
