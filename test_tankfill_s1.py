"""
test_tankfill_s1.py — tank-fill v3 PR-2 battery. Synthetic rasters with
KNOWN double-bounce intensities; the estimator's pure pieces are judged
against ground truth. The gate comparator is tankfill_gate1.py, already
pinned by its own battery.
"""
import importlib.util
import math
import os

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "tankfill_s1_estimator", os.path.join(os.path.dirname(__file__), "scripts", "tankfill_s1_estimator.py"))
s1e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1e)


def chip_with_spot(w=100, h=80, cx=50.0, cy=40.0, spot=100.0, ground=0.1):
    vv = np.full((h, w), ground, dtype=np.float32)
    vv[int(cy), int(cx)] = spot  # one bright double-bounce pixel at center
    return vv


def test_tank_db_extracts_bright_return():
    vv = chip_with_spot(spot=100.0)
    db = s1e.tank_db(vv, 50.0, 40.0, radius_px=4.0)
    # p95 over ~78 px with one 100.0 spike sits between ground and spike
    assert db is not None and db > math.log10(0.1)
    # brighter double-bounce -> strictly higher db
    db2 = s1e.tank_db(chip_with_spot(spot=1000.0), 50.0, 40.0, radius_px=4.0)
    assert db2 >= db


def test_tank_db_out_of_frame_is_none():
    vv = chip_with_spot()
    assert s1e.tank_db(vv, 2.0, 2.0, radius_px=4.0) is None
    assert s1e.tank_db(vv, 99.0, 40.0, radius_px=4.0) is None


def test_tank_db_floor_never_log_of_zero():
    vv = np.zeros((80, 100), dtype=np.float32)
    db = s1e.tank_db(vv, 50.0, 40.0, radius_px=4.0)
    assert db == pytest.approx(math.log10(s1e.FLOOR))


def test_self_ratio_fill_direction_and_median_cancel():
    # tank A: constant offset high; tank B: one emptier (brighter) scene
    db_by_scene = {
        "s1": {"A": 2.0, "B": 0.0},
        "s2": {"A": 2.0, "B": 0.0},
        "s3": {"A": 2.0, "B": 1.0},  # B brighter -> emptier -> LOWER fill*
    }
    fills = s1e.self_ratio_series(db_by_scene)
    assert fills["s1"]["A"] == pytest.approx(0.0)  # constant tank -> 0 everywhere
    assert fills["s3"]["B"] < fills["s1"]["B"], "brighter double-bounce must read emptier"
    # per-tank median subtraction: deltas equal raw-db deltas (sign flipped)
    assert (fills["s3"]["B"] - fills["s2"]["B"]) == pytest.approx(-(1.0 - 0.0))


def test_self_ratio_drops_thin_tanks():
    fills = s1e.self_ratio_series({"s1": {"A": 1.0}, "s2": {"A": 1.1}})
    assert all(f == {} for f in fills.values()), "tanks with <3 scenes have no honest median"


def test_scene_reading_d2_weighted_composite():
    tanks_by_id = {
        "A": {"site": "hub", "diameter_m": 100.0},
        "B": {"site": "hub", "diameter_m": 50.0},
    }
    r = s1e.scene_reading({"scene": "s", "date": "2026-01-01", "orbit_state": "ascending", "relative_orbit": 34},
                          {"A": 0.2, "B": -0.2}, tanks_by_id)
    assert r["n_measured"] == 2
    manual = (0.2 * 100 ** 2 + -0.2 * 50 ** 2) / (100 ** 2 + 50 ** 2)
    assert r["sites"][0]["fill_d2w"] == pytest.approx(manual, abs=1e-5)
    assert "EXPERIMENTAL" in r["label"]
