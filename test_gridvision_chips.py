"""test_gridvision_chips.py — GRID VISION Phase B chip-INDEX builder battery
(scripts/gridvision_chips.py). No network: pure geometry + OSM parsing + chip
footprint math on small inline fixtures. Deterministic (no random/now-dependent
assertions).
"""
import importlib.util
import json
import math
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "gridvision_chips", os.path.join(os.path.dirname(__file__), "scripts", "gridvision_chips.py"))
chips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chips)


def test_chip_bbox_geometry():
    # 512 px * 0.6 m = 307.2 m edge; half = 153.6 m. Center recovered exactly.
    lon, lat = -97.75, 30.25
    bb = chips.chip_bbox_for_seed(lon, lat, chip_px=512, gsd_m=0.6)
    assert (bb[0] + bb[2]) / 2 == pytest.approx(lon)
    assert (bb[1] + bb[3]) / 2 == pytest.approx(lat)
    # north-south extent in meters ~= edge
    ns_m = (bb[3] - bb[1]) * 110_540.0
    assert ns_m == pytest.approx(307.2, abs=0.5)
    ew_m = (bb[2] - bb[0]) * 111_320.0 * math.cos(math.radians(lat))
    assert ew_m == pytest.approx(307.2, abs=0.5)
    # finer GSD -> smaller chip footprint
    fine = chips.chip_bbox_for_seed(lon, lat, chip_px=512, gsd_m=0.3)
    assert (fine[2] - fine[0]) < (bb[2] - bb[0])


def test_in_bbox():
    b = [-98.1, 30.0, -97.5, 30.5]
    assert chips.in_bbox(-97.75, 30.25, b)
    assert not chips.in_bbox(-99.0, 30.25, b)
    assert not chips.in_bbox(-97.75, 31.0, b)


def test_seed_from_feature():
    tower = {"geometry": {"type": "Point", "coordinates": [-97.75, 30.25]},
             "properties": {"power": "tower", "id": "node/1"}}
    s = chips.seed_from_feature(tower)
    assert s["cls"] == "tower" and s["lon"] == -97.75 and s["osm_id"] == "node/1"
    sub = {"geometry": {"type": "Polygon", "coordinates": [[
        [-97.80, 30.20], [-97.78, 30.20], [-97.78, 30.22], [-97.80, 30.22], [-97.80, 30.20]]]},
        "properties": {"power": "substation", "id": "way/2"}}
    s2 = chips.seed_from_feature(sub)
    assert s2["cls"] == "substation"
    assert s2["lon"] == pytest.approx(-97.79) and s2["lat"] == pytest.approx(30.21)
    # a power=line is NOT an in-scope seed
    assert chips.seed_from_feature(
        {"geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
         "properties": {"power": "line"}}) is None
    # non-power feature
    assert chips.seed_from_feature({"geometry": {"type": "Point", "coordinates": [0, 0]},
                                    "properties": {"building": "yes"}}) is None


def test_parse_geojsonseq_tolerates_rs_prefix():
    a = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-97.75, 30.25]},
         "properties": {"power": "tower"}}
    text = "\x1e" + json.dumps(a) + "\n\n" + json.dumps(a) + "\nnot json\n"
    feats = chips.parse_geojsonseq(text)
    assert len(feats) == 2  # blank + garbage lines skipped


def test_parse_overpass_node_way_center():
    payload = {"elements": [
        {"type": "node", "id": 1, "lat": 30.25, "lon": -97.75, "tags": {"power": "tower"}},
        {"type": "way", "id": 2, "center": {"lat": 30.21, "lon": -97.79},
         "tags": {"power": "substation"}},
        {"type": "way", "id": 3, "geometry": [
            {"lat": 30.20, "lon": -97.80}, {"lat": 30.20, "lon": -97.78},
            {"lat": 30.22, "lon": -97.78}, {"lat": 30.20, "lon": -97.80}],
         "tags": {"power": "substation"}},
        {"type": "node", "id": 4, "lat": 30.0, "lon": -97.0, "tags": {"power": "pole"}},
    ]}
    feats = chips.parse_overpass(payload)
    assert len(feats) == 3  # pole dropped (out of scope)
    powers = sorted(f["properties"]["power"] for f in feats)
    assert powers == ["substation", "substation", "tower"]


def test_build_chip_index_restricts_to_corridor_and_carries_stac_query():
    bbox = [-98.1, 30.0, -97.5, 30.5]
    inside = {"geometry": {"type": "Point", "coordinates": [-97.75, 30.25]},
              "properties": {"power": "tower", "id": "node/1"}}
    outside = {"geometry": {"type": "Point", "coordinates": [-99.0, 30.25]},
               "properties": {"power": "tower", "id": "node/9"}}
    sub = {"geometry": {"type": "Point", "coordinates": [-97.70, 30.30]},
           "properties": {"power": "substation", "id": "node/2"}}
    idx = chips.build_chip_index([inside, outside, sub], bbox=bbox, gsd_m=0.6)
    assert idx["meta"]["n_chips"] == 2
    assert idx["meta"]["by_class"] == {"substation": 1, "tower": 1}
    assert idx["meta"]["seeds_skipped_outside_bbox"] == 1
    assert idx["meta"]["chip_edge_m"] == pytest.approx(307.2)
    c0 = idx["chips"][0]
    assert c0["provenance"] == "osm-weak-label"
    # the stored chip_bbox and the STAC query's bbox are byte-identical (no drift)
    assert c0["stac_query"]["bbox"] == c0["chip_bbox"]
    # and the query is exactly what gridvision_naip_stac.build_search_body emits
    import gridvision_naip_stac as naip
    assert c0["stac_query"] == naip.build_search_body(
        c0["chip_bbox"], chips.DATE_FROM, chips.DATE_TO, limit=10)
    assert c0["stac_query"]["collections"] == ["naip"]


def test_build_chip_index_dedup_grid():
    # grid-snap thinning: two coincident tower nodes (a common OSM artifact)
    # collapse to ONE chip; a far-apart tower stays separate. Deterministic
    # because coincident points always share a grid cell (no boundary ambiguity).
    a = {"geometry": {"type": "Point", "coordinates": [-97.7500, 30.2500]},
         "properties": {"power": "tower", "id": "node/1"}}
    dup = {"geometry": {"type": "Point", "coordinates": [-97.7500, 30.2500]},
           "properties": {"power": "tower", "id": "node/1b"}}
    far = {"geometry": {"type": "Point", "coordinates": [-97.6000, 30.4000]},
           "properties": {"power": "tower", "id": "node/2"}}
    assert chips.build_chip_index([a, dup, far], gsd_m=0.6, dedup_grid_m=0.0)["meta"]["n_chips"] == 3
    dedup = chips.build_chip_index([a, dup, far], gsd_m=0.6, dedup_grid_m=300.0)
    assert dedup["meta"]["n_chips"] == 2  # coincident pair merged; far one kept


def test_default_corridor_is_a_window_not_whole_state():
    # sanity: the pilot bbox is a small Austin-area window, not all of Texas
    w, s, e, n = chips.TX_PILOT_BBOX
    assert (e - w) < 1.0 and (n - s) < 1.0
