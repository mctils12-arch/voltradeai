"""
test_cdse_s1_chips.py — tank-fill v3 PR-1 battery (scripts/cdse_s1_chips.py).
No network: the STAC fixture is a RECORDED earth-search sentinel-1-grd
response (values verbatim from the 2026-07-05 live probe). Coverage of the
bbox itself is pinned by test_cdse_chips.py (v2 shares CHIP_BBOX; this
battery pins the two stacks staying aligned).
"""
import importlib.util
import json
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "cdse_s1_chips", os.path.join(os.path.dirname(__file__), "scripts", "cdse_s1_chips.py"))
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)

_spec2 = importlib.util.spec_from_file_location(
    "cdse_chips", os.path.join(os.path.dirname(__file__), "scripts", "cdse_chips.py"))
s2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(s2)

RECORDED_STAC = {
    "type": "FeatureCollection",
    "features": [
        {"id": "S1A_IW_GRDH_1SDV_20260607T002833_20260607T002858_064856_082C22", "properties": {
            "datetime": "2026-06-07T00:28:33.000000Z", "sat:orbit_state": "ascending",
            "sat:relative_orbit": 34, "s1:polarizations": ["VV", "VH"], "sar:instrument_mode": "IW"}},
        {"id": "S1D_IW_GRDH_1SDV_20260626T002751_20260626T002816_003400_005FC2", "properties": {
            "datetime": "2026-06-26T00:27:51.000000Z", "sat:orbit_state": "ascending",
            "sat:relative_orbit": 34, "s1:polarizations": ["VV", "VH"], "sar:instrument_mode": "IW"}},
    ],
}


def test_bbox_matches_v2_stack():
    """The S1 and S2 chips must stay pixel-aligned for fusion — same bbox,
    same resolution, same raster size."""
    assert s1.CHIP_BBOX == s2.CHIP_BBOX
    assert s1.chip_px() == s2.chip_px()


def test_pu_carries_float32_multiplier():
    w, h = s1.chip_px()
    pu = s1.pu_estimate(w, h)
    # 2 bands FLOAT32 = 2/3 x 2 = 4/3 of the area term
    assert pu == pytest.approx((w * h) / (512 * 512) * (2 / 3) * 2)
    assert 0.8 <= pu <= 2.0
    # 24-month backfill (~60 scenes single-orbit) well under 5% of free tier
    assert pu * 100 < s1.FREE_TIER_PU_MONTH * 0.05


def test_parse_stac_recorded_orbit_metadata():
    scenes = s1.parse_stac(RECORDED_STAC)
    assert len(scenes) == 2
    a = scenes[0]
    assert a["orbit_state"] == "ascending" and a["relative_orbit"] == 34
    assert a["polarizations"] == ["VV", "VH"] and a["mode"] == "IW"
    assert a["date"] == "2026-06-07"
    # missing orbit metadata stays null, never guessed
    bare = s1.parse_stac({"features": [{"id": "x", "properties": {"datetime": "2026-01-01T00:00:00Z"}}]})
    assert bare[0]["orbit_state"] is None and bare[0]["relative_orbit"] is None


def test_usable_requires_iw_dual_pol():
    ok = s1.parse_stac(RECORDED_STAC)[0]
    assert s1.usable(ok)
    assert not s1.usable(dict(ok, mode="EW"))
    assert not s1.usable(dict(ok, polarizations=["VV"]))
    assert not s1.usable(dict(ok, scene=""))


def test_pick_scene_per_date_deterministic():
    scenes = [
        {"scene": "B_SLICE", "date": "2026-06-07", "datetime": "", "orbit_state": "ascending",
         "relative_orbit": 34, "polarizations": ["VV", "VH"], "mode": "IW"},
        {"scene": "A_SLICE", "date": "2026-06-07", "datetime": "", "orbit_state": "ascending",
         "relative_orbit": 34, "polarizations": ["VV", "VH"], "mode": "IW"},
    ]
    picked = s1.pick_scene_per_date(scenes)
    assert [s["scene"] for s in picked] == ["A_SLICE"]


def test_build_process_request_shape():
    req = s1.build_process_request("2026-06-26")
    data = req["input"]["data"][0]
    assert data["type"] == "sentinel-1-grd"
    assert data["dataFilter"]["timeRange"] == {"from": "2026-06-26T00:00:00Z", "to": "2026-06-26T23:59:59Z"}
    assert data["processing"] == {"orthorectify": True, "backCoeff": "SIGMA0_ELLIPSOID"}
    assert req["input"]["bounds"]["bbox"] == s1.CHIP_BBOX
    assert 'sampleType: "FLOAT32"' in req["evalscript"]
    for b in s1.BANDS:
        assert b in req["evalscript"]
    # credentials can never leak into request payloads
    blob = json.dumps(req) + json.dumps(s1.build_stac_request("2026-01-01", "2026-01-31"))
    for env in ("CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET"):
        val = os.environ.get(env)
        if val:
            assert val not in blob


def test_index_dedup_and_monthly_pu(tmp_path):
    idx = str(tmp_path / "s1_chips_index.jsonl")
    assert s1.seen_scenes(idx) == set()
    s1.append_index({"scene": "s1a", "est_pu": 1.12, "pulled_at": "2026-07-05T09:00:00+00:00"}, idx)
    assert s1.seen_scenes(idx) == {"s1a"}
    from datetime import datetime, timezone
    july = datetime(2026, 7, 15, tzinfo=timezone.utc)
    assert s1.month_pu_spent(idx, july) == pytest.approx(1.12)
