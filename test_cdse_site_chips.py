"""
test_cdse_site_chips.py — battery for scripts/cdse_site_chips.py (DATACORE
MAXIMUS Phase 3b: latest-Sentinel-2-per-site RAW imagery chips). No network:
STAC parsing runs against a recorded earth-search shape; geometry/PU/index
tests are pure. Mirrors test_cdse_chips.py's style for the same family of
script.
"""
import importlib.util
import json
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "cdse_site_chips", os.path.join(os.path.dirname(__file__), "scripts", "cdse_site_chips.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_registry_loads_all_16_sites():
    sites = mod.load_registry()
    assert len(sites) == 16
    ids = {s["id"] for s in sites}
    assert "cushing_hub" in ids and "port_la" in ids and "stld_butler" in ids


def test_site_bbox_is_centered_square_km():
    bbox = mod.site_bbox(35.9487, -96.7587, 1.0)
    assert bbox[0] < -96.7587 < bbox[2]
    assert bbox[1] < 35.9487 < bbox[3]
    # ~1km half-width at this latitude: lon span noticeably wider in degrees
    # than lat span is WRONG (cos(lat) shrinks lon degrees) -- confirm the
    # projection correction actually narrows the lon span vs a naive
    # equirectangular equal-degree box.
    lon_span_deg = bbox[2] - bbox[0]
    lat_span_deg = bbox[3] - bbox[1]
    assert lon_span_deg > lat_span_deg  # cos(36N) < 1 widens degrees-per-km east-west


def test_half_km_for_category_matches_facility_scale():
    assert mod.half_km_for("tank_farm") < mod.half_km_for("steel_mill") < mod.half_km_for("port")
    assert mod.half_km_for("unknown_category") == mod.DEFAULT_HALF_KM


def test_chip_px_and_pu_estimate_stay_cheap():
    bbox = mod.site_bbox(35.9487, -96.7587, mod.half_km_for("tank_farm"))
    w, h = mod.chip_px(bbox)
    assert 100 <= w <= 400 and 100 <= h <= 400
    pu = mod.pu_estimate(w, h)
    assert pu < 1.0  # a single facility chip must stay well under 1 PU
    # a full 16-site refresh (largest category = port) must stay a small
    # fraction of the monthly free tier even before the script's own guard
    port_bbox = mod.site_bbox(33.74, -118.272, mod.half_km_for("port"))
    pw, ph = mod.chip_px(port_bbox)
    port_pu = mod.pu_estimate(pw, ph)
    assert 16 * port_pu < mod.FREE_TIER_PU_MONTH * 0.05


def test_parse_stac_and_pick_best_scene_prefers_most_recent():
    recorded = {
        "type": "FeatureCollection",
        "features": [
            {"id": "newer", "properties": {"datetime": "2026-07-10T17:00:00Z", "eo:cloud_cover": 12.0}},
            {"id": "older", "properties": {"datetime": "2026-06-20T17:00:00Z", "eo:cloud_cover": 1.0}},
        ],
    }
    scenes = mod.parse_stac(recorded)
    assert len(scenes) == 2
    best = mod.pick_best_scene(scenes)
    # STAC request itself sorts desc-by-datetime and prefilters cloud<ceiling;
    # pick_best_scene trusts that ordering (most recent under the ceiling),
    # it does not re-sort by cloud.
    assert best["scene"] == "newer"
    assert mod.pick_best_scene([]) is None


def test_build_stac_request_shape():
    bbox = [-96.77, 35.92, -96.71, 35.96]
    req = mod.build_stac_request(bbox, "2026-01-01T00:00:00Z", "2026-04-01T23:59:59Z", max_cloud=35.0)
    assert req["collections"] == ["sentinel-2-l2a"]
    assert req["bbox"] == bbox
    assert req["query"]["eo:cloud_cover"]["lt"] == 35.0
    assert req["sortby"] == [{"field": "properties.datetime", "direction": "desc"}]


def test_build_process_request_shape_and_no_credential_leak(monkeypatch):
    monkeypatch.setenv("CDSE_CLIENT_ID", "test-id-should-never-leak")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "test-secret-should-never-leak")
    bbox = [-96.77, 35.92, -96.71, 35.96]
    req = mod.build_process_request("2026-06-17", bbox, 200, 150)
    data = req["input"]["data"][0]
    assert data["dataFilter"]["timeRange"] == {"from": "2026-06-17T00:00:00Z", "to": "2026-06-17T23:59:59Z"}
    assert data["dataFilter"]["mosaickingOrder"] == "leastCC"
    assert req["input"]["bounds"]["bbox"] == bbox
    assert req["output"]["width"] == 200 and req["output"]["height"] == 150
    assert req["output"]["responses"][0]["format"]["type"] == "image/jpeg"
    for b in ("B04", "B03", "B02"):
        assert b in req["evalscript"]
    blob = json.dumps(req)
    assert "test-id-should-never-leak" not in blob
    assert "test-secret-should-never-leak" not in blob


def test_index_read_write_roundtrip_and_log_pu_accounting(tmp_path):
    index_path = str(tmp_path / "site_latest_index.json")
    assert mod.read_index(index_path) == {}
    idx = {"cushing_hub": {"file": "/imagery/sites/cushing_hub.jpg", "date": "2026-07-20"}}
    mod.write_index(idx, index_path)
    assert mod.read_index(index_path) == idx

    log_path = str(tmp_path / "site_chips_log.jsonl")
    mod.append_log({"site": "cushing_hub", "ok": True, "est_pu": 0.3, "pulled_at": "2026-07-05T01:00:00+00:00"}, log_path)
    mod.append_log({"site": "port_la", "ok": True, "est_pu": 0.9, "pulled_at": "2026-06-30T23:00:00+00:00"}, log_path)
    mod.append_log({"site": "port_lb", "ok": False, "reason": "no clean scene", "checked_at": "2026-07-05T01:00:00+00:00"}, log_path)
    from datetime import datetime, timezone
    july = datetime(2026, 7, 15, tzinfo=timezone.utc)
    june = datetime(2026, 6, 15, tzinfo=timezone.utc)
    # failed pulls (ok:False, no est_pu) never count toward spend
    assert mod.month_pu_spent(log_path, july) == pytest.approx(0.3)
    assert mod.month_pu_spent(log_path, june) == pytest.approx(0.9)
    # corrupt lines are skipped, never fatal
    with open(log_path, "a") as fh:
        fh.write("not json\n")
    assert mod.month_pu_spent(log_path, july) == pytest.approx(0.3)


def test_days_since_handles_missing_and_malformed_dates():
    from datetime import datetime, timezone
    now = datetime(2026, 7, 22, tzinfo=timezone.utc)
    assert mod.days_since("2026-07-15", now) == 7
    assert mod.days_since("", now) > 365
    assert mod.days_since("not-a-date", now) > 365


def test_get_token_requires_credentials(monkeypatch):
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("CDSE_CLIENT_SECRET", raising=False)
    with pytest.raises(SystemExit):
        mod.get_token()
