"""test_submarine_cables_build.py — submarine telecom cables layer builder
(scripts/submarine_cables_build.py). No network: pure tag-classification +
geometry math on small inline Overpass-shaped fixtures. Deterministic.
"""
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "submarine_cables_build", os.path.join(os.path.dirname(__file__), "scripts", "submarine_cables_build.py"))
cables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cables)


def _way(way_id=1, tags=None, coords=None):
    tags = tags or {}
    coords = coords or [(-10.0, 50.0), (-9.5, 50.2)]
    return {
        "type": "way",
        "id": way_id,
        "tags": tags,
        "geometry": [{"lon": lon, "lat": lat} for lon, lat in coords],
    }


def test_telecom_cable_accepted():
    el = _way(way_id=101, tags={
        "seamark:type": "cable_submarine",
        "name": "TAT-8",
        "operator": "AT&T",
        "communication": "line",
    })
    feat, reason = cables.way_to_feature(el)
    assert reason is None
    assert feat["type"] == "Feature"
    assert feat["properties"]["name"] == "TAT-8"
    assert feat["properties"]["operator"] == "AT&T"
    assert feat["properties"]["osm_id"] == 101
    assert feat["properties"]["source_url"] == "https://www.openstreetmap.org/way/101"
    assert feat["geometry"]["type"] == "LineString"


def test_power_tagged_cable_excluded_even_with_primary_tag():
    # a way that matched the primary telecom query but ALSO carries a power
    # tag (e.g. a submarine interconnector like NorNed) must be dropped —
    # this is the exact exclusion the mission spec called out by name.
    el = _way(way_id=202, tags={
        "seamark:type": "cable_submarine",
        "power": "cable",
        "voltage": "450000",
    })
    feat, reason = cables.way_to_feature(el)
    assert feat is None
    assert reason == "power_tagged"


def test_disused_flag_carried_as_boolean():
    el = _way(way_id=303, tags={"seamark:type": "cable_submarine", "disused": "yes", "name": "STRATOS 1"})
    feat, _ = cables.way_to_feature(el)
    assert feat["properties"]["disused"] is True

    el2 = _way(way_id=304, tags={"seamark:type": "cable_submarine", "name": "MAREA"})
    feat2, _ = cables.way_to_feature(el2)
    assert feat2["properties"]["disused"] is False


def test_missing_optional_tags_stay_none_never_fabricated():
    el = _way(way_id=404, tags={"seamark:type": "cable_submarine"})
    feat, _ = cables.way_to_feature(el)
    p = feat["properties"]
    assert p["name"] is None
    assert p["operator"] is None
    assert p["category"] is None
    assert p["start_date"] is None
    assert p["wikipedia"] is None


def test_start_date_prefers_start_date_over_opening_date():
    el = _way(way_id=505, tags={"start_date": "2018", "opening_date": "2017"})
    feat, _ = cables.way_to_feature(el)
    assert feat["properties"]["start_date"] == "2018"
    el2 = _way(way_id=506, tags={"opening_date": "2017"})
    feat2, _ = cables.way_to_feature(el2)
    assert feat2["properties"]["start_date"] == "2017"


def test_non_way_element_rejected():
    node = {"type": "node", "id": 1, "tags": {}, "lat": 0, "lon": 0}
    feat, reason = cables.way_to_feature(node)
    assert feat is None
    assert reason == "not_a_way"


def test_way_without_geometry_rejected():
    el = {"type": "way", "id": 606, "tags": {"seamark:type": "cable_submarine"}}
    feat, reason = cables.way_to_feature(el)
    assert feat is None
    assert reason == "no_geometry"


def test_way_with_single_point_geometry_rejected():
    # a LineString needs >=2 points — a degenerate 1-point "way" must not
    # silently become a zero-length line
    el = {"type": "way", "id": 607, "tags": {"seamark:type": "cable_submarine"},
          "geometry": [{"lon": 1.0, "lat": 2.0}]}
    feat, reason = cables.way_to_feature(el)
    assert feat is None
    assert reason == "no_geometry"


def test_coords_rounded_to_4dp():
    el = _way(way_id=707, coords=[(-10.123456789, 50.987654321), (-9.111111, 50.222222)])
    feat, _ = cables.way_to_feature(el)
    coords = feat["geometry"]["coordinates"]
    assert coords[0] == [-10.1235, 50.9877]
    assert coords[1] == [-9.1111, 50.2222]


def test_build_secondary_stays_unactivated_and_honest():
    # v1 ships primary-tag-only; build_secondary must return empty and say
    # why (never silently return nothing with no explanation, and never
    # fabricate ways to pad the count).
    feats, meta = cables.build_secondary()
    assert feats == []
    assert meta["activated"] is False
    assert meta["count"] == 0
    assert "reason" in meta and len(meta["reason"]) > 20
