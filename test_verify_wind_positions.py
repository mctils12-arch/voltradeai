"""test_verify_wind_positions.py — pure geo/classification core for the
wind-position gate-1 (DATA) validator. No network — Overpass fetch and
CLI plumbing are exercised by running the script directly (see the
module docstring); this suite pins the logic that decides verified vs.
approximate vs. unverified from an already-fetched point set."""
import importlib.util
import math
import os

_spec = importlib.util.spec_from_file_location(
    "vwp", os.path.join(os.path.dirname(__file__), "scripts", "verify_wind_positions.py"))
vwp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vwp)


def test_haversine_known_distance():
    # 1 deg latitude ~ 111 km
    d = vwp.haversine_km(38.0, -95.0, 39.0, -95.0)
    assert 110 < d < 112


def test_haversine_zero_for_same_point():
    assert vwp.haversine_km(38.0, -95.0, 38.0, -95.0) == 0.0


def test_classify_no_points_is_unverified():
    status, nearest, count, centroid = vwp.classify(38.0, -95.0, [])
    assert status == "unverified" and nearest is None and count == 0 and centroid is None


def test_classify_close_turbine_is_verified():
    # ~1km north of the plant point
    lat, lon = 38.0, -95.0
    near_lat = lat + (1.0 / vwp.KM_PER_DEG_LAT)
    status, nearest, count, centroid = vwp.classify(lat, lon, [(near_lat, lon)])
    assert status == "verified"
    assert nearest is not None and nearest < vwp.VERIFIED_KM
    assert count == 1


def test_classify_far_but_in_range_turbine_is_approximate():
    lat, lon = 38.0, -95.0
    far_lat = lat + (9.0 / vwp.KM_PER_DEG_LAT)  # ~9km, between VERIFIED_KM and SEARCH_RADIUS_KM
    status, nearest, count, centroid = vwp.classify(lat, lon, [(far_lat, lon)])
    assert status == "approximate"
    assert vwp.VERIFIED_KM < nearest < vwp.SEARCH_RADIUS_KM


def test_classify_beyond_search_radius_is_unverified():
    lat, lon = 38.0, -95.0
    too_far_lat = lat + (40.0 / vwp.KM_PER_DEG_LAT)  # well beyond SEARCH_RADIUS_KM
    status, nearest, count, centroid = vwp.classify(lat, lon, [(too_far_lat, lon)])
    assert status == "unverified" and nearest is None


def test_classify_centroid_averages_only_in_range_points():
    lat, lon = 38.0, -95.0
    near = (lat + 1.0 / vwp.KM_PER_DEG_LAT, lon)
    also_near = (lat - 1.0 / vwp.KM_PER_DEG_LAT, lon)
    too_far = (lat + 100.0 / vwp.KM_PER_DEG_LAT, lon)
    status, nearest, count, centroid = vwp.classify(lat, lon, [near, also_near, too_far])
    assert status == "verified"
    assert count == 2  # too_far excluded from the in-range centroid/count
    assert abs(centroid[0] - lat) < 1e-6


def test_classify_waverly_reference_case_shape():
    """Reference case named in the human directive: Waverly Wind Farm LLC
    (KS), GPPD point 38.2569,-95.8142. Live Overpass query this session
    (2026-07-16) found the nearest real turbine cluster ~9km away — this
    pins the SHAPE of that finding (approximate, not verified/unverified)
    against a synthetic point set at the same real distance, since the
    live OSM response itself is not deterministic to commit as a fixture."""
    plant_lat, plant_lon = 38.2569, -95.8142
    turbines = [(38.3889046, -95.6604884), (38.2389222, -95.6894317)]
    status, nearest, count, centroid = vwp.classify(plant_lat, plant_lon, turbines)
    assert status == "approximate"
    assert 5 < nearest < 15


def test_bin_key_floors_not_rounds():
    assert vwp.bin_key(38.9, -95.1, bin_deg=2.0) == (38.0, -96.0)
    assert vwp.bin_key(-1.0, -1.0, bin_deg=2.0) == (-2.0, -2.0)


def test_pad_deg_exceeds_search_radius_with_margin():
    pad_lat, pad_lon = vwp.pad_deg(38.0, 40.0)
    assert pad_lat * vwp.KM_PER_DEG_LAT > vwp.SEARCH_RADIUS_KM
    # longitude padding must be wider than latitude padding away from the equator
    # (fewer km per degree of longitude at higher latitude)
    assert pad_lon > pad_lat


def test_pad_deg_widens_toward_poles():
    _, pad_lon_low = vwp.pad_deg(10.0, 12.0)
    _, pad_lon_high = vwp.pad_deg(48.0, 50.0)
    assert pad_lon_high > pad_lon_low


def test_turbine_points_reads_node_and_center_shapes():
    payload = {
        "elements": [
            {"type": "node", "lat": 38.1, "lon": -95.2},
            {"type": "way", "center": {"lat": 38.3, "lon": -95.4}},
            {"type": "way"},  # no geometry at all -> skipped, not a crash
        ]
    }
    pts = vwp.turbine_points(payload)
    assert pts == [(38.1, -95.2), (38.3, -95.4)]


def test_group_bins_partitions_by_floor_bin():
    plants = [
        ["A", 10.0, "wind", "op", 38.1, -95.2, 0],
        ["B", 5.0, "wind", "op", 38.9, -95.9, 0],
        ["C", 3.0, "wind", "op", 30.0, -102.0, 0],
    ]
    bins = vwp.group_bins(plants)
    assert len(bins) == 2
    assert len(bins[(38.0, -96.0)]) == 2
    assert len(bins[(30.0, -102.0)]) == 1


def test_bin_is_resolved_true_when_all_keys_present_and_no_errors():
    results = {"a": {"status": "verified"}, "b": {"status": "unverified"}}
    assert vwp.bin_is_resolved(["a", "b"], results) is True


def test_bin_is_resolved_false_when_key_missing():
    results = {"a": {"status": "verified"}}
    assert vwp.bin_is_resolved(["a", "b"], results) is False


def test_bin_is_resolved_false_when_any_member_errored():
    """A prior Overpass failure must force a retry on the next resume run —
    it is never treated as a completed (if negative) finding."""
    results = {"a": {"status": "verified"}, "b": {"status": "error"}}
    assert vwp.bin_is_resolved(["a", "b"], results) is False
