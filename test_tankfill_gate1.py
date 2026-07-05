"""
test_tankfill_gate1.py — tank-fill v2 PR-4 battery for the gate-1 scorer's
pure pieces. The matching/delta/verdict rules were stated in the script
header BEFORE the gate ran; these tests pin them so the ruler cannot drift
after results exist (MEASUREMENT INTEGRITY).
"""
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "tankfill_gate1", os.path.join(os.path.dirname(__file__), "scripts", "tankfill_gate1.py"))
g1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g1)


def reading(d, fills_dm, scene="s"):
    return {"date": d, "scene": f"{scene}_{d}",
            "tanks": [{"fill": f, "d_m": dm} for f, dm in fills_dm]}


def test_scene_composite_d2_weighted():
    r = reading("2026-01-02", [(1.0, 100.0), (0.0, 50.0)])
    # weights 10000 vs 2500 -> 0.8
    assert g1.scene_composite(r) == pytest.approx(0.8)
    assert g1.scene_composite({"date": "x", "skipped": "cloud"}) is None


def test_match_weeks_window_and_lowest_cloud():
    eia = {"2026-01-02": 30000.0, "2026-01-09": 31000.0}
    rs = [
        dict(reading("2026-01-01", [(0.5, 60)]), cloud_pct=20.0),   # matches 01-02 (1d)
        dict(reading("2026-01-03", [(0.6, 60)]), cloud_pct=5.0),    # same week, less cloud -> wins
        dict(reading("2026-01-06", [(0.7, 60)]), cloud_pct=1.0),    # 3d from BOTH — ties to nearer? 3 vs 3: first seen kept
        dict(reading("2026-01-20", [(0.9, 60)]), cloud_pct=1.0),    # no EIA week within 3d
    ]
    m = g1.match_weeks(rs, eia)
    assert [x["eia_date"] for x in m] == ["2026-01-02", "2026-01-09"]
    assert m[0]["composite"] == pytest.approx(0.6)  # lowest-cloud scene won the week


def test_paired_deltas_gap_cap():
    matched = [
        {"eia_date": "2026-01-02", "kbbl": 30000.0, "composite": 0.50},
        {"eia_date": "2026-01-09", "kbbl": 31000.0, "composite": 0.55},   # 7d -> pair
        {"eia_date": "2026-02-13", "kbbl": 29000.0, "composite": 0.40},   # 35d -> dropped
        {"eia_date": "2026-02-20", "kbbl": 28000.0, "composite": 0.35},   # 7d -> pair
    ]
    d = g1.paired_deltas(matched)
    assert len(d) == 2
    assert d[0] == (pytest.approx(0.05), pytest.approx(1000.0))
    assert d[1] == (pytest.approx(-0.05), pytest.approx(-1000.0))


def test_sign_hit_rate_excludes_zeros():
    hit, n = g1.sign_hit_rate([(0.1, 500.0), (-0.1, -200.0), (0.0, 300.0), (0.1, -100.0)])
    assert n == 3
    assert hit == pytest.approx(2 / 3)


def test_verdict_pass_and_fail_paths():
    # 21 weekly matched rows, perfectly correlated deltas with a reversal
    matched = []
    kbbl, comp = 30000.0, 0.5
    for i in range(21):
        sign = 1 if (i // 5) % 2 == 0 else -1  # reversals every 5 weeks
        kbbl += sign * 400.0
        comp += sign * 0.01
        matched.append({"eia_date": f"2026-{1 + i // 4:02d}-{2 + (i % 4) * 7:02d}",
                        "kbbl": kbbl, "composite": comp})
    v = g1.verdict(matched)
    assert v["n_matched_ok"] and v["reversal_ok"]
    assert v["delta_r"] == pytest.approx(1.0)
    assert v["sign_hit"] == pytest.approx(1.0)
    assert v["PASS"] is True
    # anti-correlated composite must FAIL on both delta checks
    flipped = [dict(m, composite=1.0 - m["composite"]) for m in matched]
    v2 = g1.verdict(flipped)
    assert v2["PASS"] is False and v2["delta_r"] == pytest.approx(-1.0)
    # thin sample fails regardless of correlation
    v3 = g1.verdict(matched[:6])
    assert v3["PASS"] is False and not v3["n_matched_ok"]


def test_pearson_degenerate():
    assert g1.pearson([1, 2], [1, 2]) is None          # n < 3
    assert g1.pearson([1, 1, 1], [1, 2, 3]) is None    # zero variance
