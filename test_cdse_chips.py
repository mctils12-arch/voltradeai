"""
test_cdse_chips.py — tank-fill v2 PR-2 battery (scripts/cdse_chips.py).
No network: parsers run against a RECORDED earth-search response (values
copied verbatim from the live probe of 2026-07-05); geometry/PU/index tests
are pure. The coverage test is the ratchet for the workup-bbox defect found
building this PR: the filed workup bbox missed 20 measurable tanks.
"""
import importlib.util
import json
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "cdse_chips", os.path.join(os.path.dirname(__file__), "scripts", "cdse_chips.py"))
cdse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdse)

# Recorded earth-search /v1/search response over CHIP_BBOX (trimmed to the
# fields parse_stac reads; values verbatim from the 2026-07-05 live probe).
RECORDED_STAC = {
    "type": "FeatureCollection",
    "features": [
        {"id": "S2B_14SQE_20260617_0_L2A", "properties": {
            "datetime": "2026-06-17T17:23:51.610000Z", "eo:cloud_cover": 2.90599,
            "view:sun_elevation": 71.8628957968826, "view:sun_azimuth": 128.177178324603}},
        {"id": "S2B_14SPE_20260617_0_L2A", "properties": {
            "datetime": "2026-06-17T17:23:55.342000Z", "eo:cloud_cover": 0.075478,
            "view:sun_elevation": 71.1422817915377, "view:sun_azimuth": 125.990302408067}},
        {"id": "S2C_14SQE_20260622_0_L2A", "properties": {
            "datetime": "2026-06-22T17:23:52.802000Z", "eo:cloud_cover": 38.544112,
            "view:sun_elevation": 71.7180617815304, "view:sun_azimuth": 127.527502385648}},
        {"id": "S2C_14SPE_20260622_0_L2A", "properties": {
            "datetime": "2026-06-22T17:23:56.551000Z", "eo:cloud_cover": 15.414776,
            "view:sun_elevation": 70.9918076061155, "view:sun_azimuth": 125.377533605787}},
    ],
}


def test_chip_bbox_covers_every_measurable_tank():
    """RATCHET (2026-07-05): the filed workup bbox [-96.80,35.90,-96.72,35.98]
    missed 20 of 234 measurable tanks. CHIP_BBOX must cover every measurable
    registry tank with >=100 m margin — registry-driven, so adding tanks to
    the registry re-tests coverage automatically."""
    with open(cdse.REGISTRY) as fh:
        reg = json.load(fh)
    margin_lon = 100.0 / 90_000.0  # ~100 m in degrees at 36N
    margin_lat = 100.0 / 110_540.0
    missed = []
    for f in reg["features"]:
        if not f["properties"]["measurable_10m"]:
            continue
        lon, lat = f["geometry"]["coordinates"]
        if not (cdse.CHIP_BBOX[0] + margin_lon <= lon <= cdse.CHIP_BBOX[2] - margin_lon
                and cdse.CHIP_BBOX[1] + margin_lat <= lat <= cdse.CHIP_BBOX[3] - margin_lat):
            missed.append(f["properties"]["id"])
    assert missed == [], f"measurable tanks outside CHIP_BBOX(+100m margin): {missed}"


def test_chip_size_and_pu_within_free_tier():
    w, h = cdse.chip_px()
    assert 400 <= w <= 700 and 300 <= h <= 600, f"chip {w}x{h} out of expected range"
    pu = cdse.pu_estimate(w, h)
    assert 0.8 <= pu <= 2.5, f"PU/scene {pu} outside sanity band"
    # 24-month backfill (~2 usable scenes/wk pre-cloud -> ~100 after) plus a
    # month's weekly cadence must stay under 5% of the 10,000 PU free tier.
    assert pu * (100 + 5) < cdse.FREE_TIER_PU_MONTH * 0.05


def test_parse_stac_recorded_response():
    scenes = cdse.parse_stac(RECORDED_STAC)
    assert len(scenes) == 4
    s = scenes[1]
    assert s["scene"] == "S2B_14SPE_20260617_0_L2A"
    assert s["date"] == "2026-06-17"
    assert s["datetime"].startswith("2026-06-17T17:23:55")
    assert abs(s["cloud_pct"] - 0.075478) < 1e-9
    assert 70 < s["sun_elev"] < 72 and 125 < s["sun_azim"] < 127
    # missing angles stay null, never guessed
    bare = cdse.parse_stac({"features": [{"id": "x", "properties": {"datetime": "2026-01-01T00:00:00Z"}}]})
    assert bare[0]["sun_elev"] is None and bare[0]["sun_azim"] is None


def test_pick_scene_per_date_lowest_cloud():
    picked = cdse.pick_scene_per_date(cdse.parse_stac(RECORDED_STAC))
    assert [s["scene"] for s in picked] == ["S2B_14SPE_20260617_0_L2A", "S2C_14SPE_20260622_0_L2A"]
    # null cloud loses to a real reading; empty ids dropped
    scenes = [
        {"scene": "a", "date": "2026-01-01", "datetime": "", "cloud_pct": None, "sun_elev": 1, "sun_azim": 1},
        {"scene": "b", "date": "2026-01-01", "datetime": "", "cloud_pct": 50.0, "sun_elev": 1, "sun_azim": 1},
        {"scene": "", "date": "2026-01-02", "datetime": "", "cloud_pct": 0.0, "sun_elev": 1, "sun_azim": 1},
    ]
    assert [s["scene"] for s in cdse.pick_scene_per_date(scenes)] == ["b"]


def test_build_process_request_shape():
    req = cdse.build_process_request("2026-06-17")
    data = req["input"]["data"][0]
    assert data["type"] == "sentinel-2-l2a"
    assert data["dataFilter"]["timeRange"] == {"from": "2026-06-17T00:00:00Z", "to": "2026-06-17T23:59:59Z"}
    assert data["dataFilter"]["mosaickingOrder"] == "leastCC"
    assert req["input"]["bounds"]["bbox"] == cdse.CHIP_BBOX
    w, h = cdse.chip_px()
    assert req["output"]["width"] == w and req["output"]["height"] == h
    assert req["output"]["responses"][0]["format"]["type"] == "image/tiff"
    for b in cdse.BANDS:
        assert b in req["evalscript"]
    assert 'sampleType: "UINT16"' in req["evalscript"]
    assert 'units: "DN"' in req["evalscript"]
    # credentials can never leak into request payloads
    blob = json.dumps(req) + json.dumps(cdse.build_stac_request("2026-01-01", "2026-01-31"))
    for env in ("CDSE_CLIENT_ID", "CDSE_CLIENT_SECRET"):
        val = os.environ.get(env)
        if val:
            assert val not in blob


def test_index_dedup_and_monthly_pu_accounting(tmp_path):
    idx = str(tmp_path / "chips_index.jsonl")
    assert cdse.seen_scenes(idx) == set()
    cdse.append_index({"scene": "s1", "est_pu": 1.4, "pulled_at": "2026-07-05T01:00:00+00:00"}, idx)
    cdse.append_index({"scene": "s2", "est_pu": 1.4, "pulled_at": "2026-06-30T23:00:00+00:00"}, idx)
    assert cdse.seen_scenes(idx) == {"s1", "s2"}
    from datetime import datetime, timezone
    july = datetime(2026, 7, 15, tzinfo=timezone.utc)
    june = datetime(2026, 6, 15, tzinfo=timezone.utc)
    assert cdse.month_pu_spent(idx, july) == pytest.approx(1.4)
    assert cdse.month_pu_spent(idx, june) == pytest.approx(1.4)
    # corrupt lines are skipped, never fatal
    with open(idx, "a") as fh:
        fh.write("not json\n")
    assert cdse.seen_scenes(idx) == {"s1", "s2"}


def test_get_token_requires_credentials(monkeypatch):
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("CDSE_CLIENT_SECRET", raising=False)
    with pytest.raises(SystemExit):
        cdse.get_token()
