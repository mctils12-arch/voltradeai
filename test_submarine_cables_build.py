"""
test_submarine_cables_build.py — submarine cable layer builder
(scripts/submarine_cables_build.py). Pure-function tests: haversine length,
Douglas-Peucker simplification, and the OSM-tag category classifier. No
network access — Overpass fetch/tiling is exercised live at build time
(manual quarterly refresh, same pattern as military_installations_build.py),
not in CI.
"""
import importlib.util
import math
import os

_spec = importlib.util.spec_from_file_location(
    "submarine_cables_build", os.path.join(os.path.dirname(__file__), "scripts", "submarine_cables_build.py"))
cables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cables)


def test_haversine_km_known_distance():
    # New York (-74.0060, 40.7128) to London (-0.1276, 51.5074) — real great-circle
    # distance is ~5,570 km; allow a generous tolerance for a hand-rolled formula check.
    ny = (-74.0060, 40.7128)
    london = (-0.1276, 51.5074)
    d = cables.haversine_km(ny, london)
    assert 5500 < d < 5650


def test_haversine_km_zero_for_identical_point():
    assert cables.haversine_km((10.0, 20.0), (10.0, 20.0)) == 0.0


def test_line_length_km_sums_segments():
    # a simple 3-point path: length must equal the sum of its two segments,
    # and must be >= the direct straight-line distance between the endpoints.
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    total = cables.line_length_km(coords)
    seg1 = cables.haversine_km(coords[0], coords[1])
    seg2 = cables.haversine_km(coords[1], coords[2])
    assert math.isclose(total, seg1 + seg2, rel_tol=1e-9)
    direct = cables.haversine_km(coords[0], coords[-1])
    assert total >= direct


def test_douglas_peucker_keeps_endpoints():
    coords = [[0.0, 0.0], [1.0, 0.01], [2.0, -0.01], [3.0, 0.0]]
    simplified = cables.douglas_peucker(coords, epsilon_deg=0.5)
    assert simplified[0] == coords[0]
    assert simplified[-1] == coords[-1]
    # a large epsilon collapses a near-straight line to just the endpoints
    assert len(simplified) == 2


def test_douglas_peucker_preserves_a_real_corner():
    # a sharp right-angle bend must survive even a moderate epsilon
    coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    simplified = cables.douglas_peucker(coords, epsilon_deg=0.01)
    assert len(simplified) == 3
    assert simplified[1] == [1.0, 0.0]


def test_douglas_peucker_short_input_passthrough():
    assert cables.douglas_peucker([[0.0, 0.0]], 0.01) == [[0.0, 0.0]]
    two = [[0.0, 0.0], [1.0, 1.0]]
    assert cables.douglas_peucker(two, 0.01) == two


def test_classify_category_telecom_fiber_optic():
    kind, disused = cables.classify_category({"seamark:cable_submarine:category": "fiber_optic"})
    assert kind == "telecom"
    assert disused is False


def test_classify_category_telecom_via_communication_tag_no_explicit_category():
    kind, disused = cables.classify_category({"communication": "line"})
    assert kind == "telecom"


def test_classify_category_power():
    kind, disused = cables.classify_category({"seamark:cable_submarine:category": "power"})
    assert kind == "power"


def test_classify_category_power_via_transmission():
    kind, _ = cables.classify_category({"seamark:cable_submarine:category": "transmission"})
    assert kind == "power"


def test_classify_category_mixed_power_and_optical():
    kind, disused = cables.classify_category({"seamark:cable_submarine:category": "power;optical"})
    assert kind == "mixed"
    assert disused is False


def test_classify_category_disused_flag_set():
    kind, disused = cables.classify_category({"seamark:cable_submarine:category": "disused"})
    assert disused is True
    assert kind == "unclassified"


def test_classify_category_mooring_is_other_not_telecom():
    kind, _ = cables.classify_category({"seamark:cable_submarine:category": "mooring"})
    assert kind == "other"


def test_classify_category_no_tags_is_unclassified():
    kind, disused = cables.classify_category({})
    assert kind == "unclassified"
    assert disused is False


def test_classify_category_telegraph_and_telephone_are_telecom():
    assert cables.classify_category({"seamark:cable_submarine:category": "telegraph"})[0] == "telecom"
    assert cables.classify_category({"seamark:cable_submarine:category": "telephone"})[0] == "telecom"


# ── fetch_region: cache + auto-split-on-failure (network mocked out) ──

def test_fetch_region_caches_successful_tile(tmp_path, monkeypatch):
    monkeypatch.setattr(cables, "CACHE_DIR", str(tmp_path))
    calls = []

    def fake_fetch_tile(bbox):
        calls.append(bbox)
        return [{"type": "way", "id": 1}], 1.0, "fake-mirror"

    monkeypatch.setattr(cables, "fetch_tile", fake_fetch_tile)
    bbox = (0.0, 0.0, 10.0, 10.0)
    els1, meta1 = cables.fetch_region(bbox)
    els2, meta2 = cables.fetch_region(bbox)  # second call must hit the disk cache
    assert len(calls) == 1
    assert els1 == els2 == [{"type": "way", "id": 1}]
    assert meta2[0]["mirror"] == "cache"


def test_fetch_region_splits_on_failure_and_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(cables, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cables, "MAX_SPLIT_DEPTH", 3)
    monkeypatch.setattr(cables, "MIN_SPLIT_DEG", 1.0)

    bbox = (0.0, 0.0, 10.0, 20.0)  # lon span (20) > lat span (10) — splits on longitude

    def fake_fetch_tile(b):
        # only the exact original bbox fails — forces exactly one split,
        # both resulting halves succeed.
        if b == bbox:
            raise RuntimeError("simulated Overpass failure")
        return [{"type": "way", "id": f"{b[0]}-{b[1]}"}], 0.5, "fake-mirror"

    monkeypatch.setattr(cables, "fetch_tile", fake_fetch_tile)
    elements, leaf_meta = cables.fetch_region(bbox)
    assert len(elements) == 2  # both halves recovered
    assert all("error" not in m for m in leaf_meta)


def test_fetch_region_gives_up_below_min_split_size():
    def fake_fetch_tile(bbox):
        raise RuntimeError("simulated permanent failure")

    import unittest.mock
    with unittest.mock.patch.object(cables, "fetch_tile", fake_fetch_tile), \
         unittest.mock.patch.object(cables, "CACHE_DIR", "/tmp/voltradeai_test_cache_giveup"), \
         unittest.mock.patch.object(cables, "MAX_SPLIT_DEPTH", 2), \
         unittest.mock.patch.object(cables, "MIN_SPLIT_DEG", 4.0):
        elements, leaf_meta = cables.fetch_region((0.0, 0.0, 2.0, 2.0))  # already below MIN_SPLIT_DEG
        assert elements == []
        assert len(leaf_meta) == 1
        assert "error" in leaf_meta[0]


def test_classify_category_returns_tuple_of_str_and_bool():
    kind, disused = cables.classify_category({"seamark:cable_submarine:category": "fibre_optic"})
    assert isinstance(kind, str)
    assert isinstance(disused, bool)
