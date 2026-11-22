"""Tests for scripts/floor_drawdown_limit.py (2026-11-22): the limit of the floor's
second rollback leg, computed from the pre-registered study's own V1 series.

They pin the definition against arithmetic written out here, the committed
result against the study it was computed from, and the server's constant
against the committed result."""
import importlib.util
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("floor_drawdown_limit", os.path.join(HERE, "scripts", "floor_drawdown_limit.py"))
fdl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fdl)

RESULT = os.path.join(HERE, "research", "floor_study_drawdown_limit_2026-11-22.json")
STUDY = os.path.join(HERE, "research", "floor_study_2026-11-05.json")


def test_max_drawdown_is_the_largest_fall_from_a_high_to_a_later_low():
    assert fdl.max_drawdown([100, 110, 99, 105, 120, 108]) == 99 / 110 - 1          # -10%, not the later -10% from 120
    assert fdl.max_drawdown([100, 101, 102]) == 0.0                                  # never falls
    assert math.isclose(fdl.max_drawdown([100, 50, 200, 150]), -0.5)                 # a low BEFORE the high does not count against it
    assert fdl.max_drawdown([]) == 0.0


def test_a_window_is_63_sessions_and_sees_only_its_own_closes():
    # 200 flat sessions with one -20% day at index 100
    r = [0.0] * 200
    r[100] = -0.20
    dd = fdl.window_drawdowns(r, 63)
    assert len(dd) == 200 - 63 + 1
    # window k covers returns k .. k+62: the crash is inside windows 38 .. 100
    for k, v in enumerate(dd):
        want = -0.20 if 38 <= k <= 100 else 0.0
        assert math.isclose(v, want, abs_tol=1e-12), (k, v)


def test_the_three_readings_differ_as_documented():
    # a fall and a full recovery inside one window: the window HAD a drawdown, is not IN one, and returned zero
    r = [0.0] * 10 + [-0.10] + [0.0] * 20 + [1 / 0.9 - 1] + [0.0] * 31
    assert len(r) == 63
    assert math.isclose(fdl.window_drawdowns(r, 63)[0], -0.10)
    assert math.isclose(fdl.from_window_high(r, 63)[0], 0.0, abs_tol=1e-12)
    assert math.isclose(fdl.window_returns(r, 63)[0], 0.0, abs_tol=1e-12)


def test_severity_percentile_is_the_level_95_percent_of_windows_stayed_above():
    xs = [-i / 100 for i in range(101)]            # 0, -1%, ... -100%
    assert math.isclose(fdl.severity_percentile(xs, 0.95), -0.95)
    assert math.isclose(fdl.severity_percentile(xs, 0.99), -0.99)
    assert sum(1 for x in xs if x > fdl.severity_percentile(xs, 0.95)) == 95
    assert fdl.severity_percentile([-0.3], 0.95) == -0.3


def test_the_committed_result_is_the_studys_v1_and_reproduced_it():
    out = json.load(open(RESULT))
    study = json.load(open(STUDY))
    assert out["window"] == study["window"]
    rep = out["reproduction"]
    assert rep["reproduces_study"] is True
    for k in ("cagr_pct", "sharpe", "max_dd_pct", "log_growth_pct"):
        assert rep["study"][k] == study["V1_volmanaged"][k]
        assert abs(rep["recomputed"][k] - study["V1_volmanaged"][k]) <= fdl.REPRODUCTION_TOL
    v = out["V1_volmanaged"]
    assert v["window_sessions"] == fdl.WINDOW_SESSIONS == 63
    assert v["windows"] == study["window"]["sessions"] - 63 + 1
    # ordering every such result must satisfy: worst <= 99th <= 95th < 0, and a 3-month window cannot fall further than the whole history
    assert v["worst_pct"] <= v["p99_pct"] <= v["p95_pct"] < 0
    assert v["worst_pct"] >= study["V1_volmanaged"]["max_dd_pct"]
    # the floor's 95th-percentile window is shallower than buy-and-hold SPY's (what the study chose it for)
    assert v["p95_pct"] > out["B0_SPY"]["p95_pct"]
    # the pre-registered study file is NOT where this number lives: it stays as published
    assert not re.search(r"percentile|p95|drawdown_3m|3m_dd", json.dumps(study), re.I)


def test_the_servers_limit_is_the_committed_number():
    out = json.load(open(RESULT))
    src = open(os.path.join(HERE, "server", "floorTracking.ts")).read()
    limit = float(re.search(r"^export const FLOOR_DRAWDOWN_LIMIT_PCT = (-?[\d.]+);", src, re.M).group(1))
    window = int(re.search(r"^export const FLOOR_DRAWDOWN_WINDOW_SESSIONS = (\d+);", src, re.M).group(1))
    assert limit == out["V1_volmanaged"]["p95_pct"]
    assert window == out["V1_volmanaged"]["window_sessions"]
