"""test_gridvision_geo.py — pure geo-referencing + OSM-recall core (gate-1 c /
rollout). No network. Honest OSM semantics: recall is a lower bound."""
import importlib.util
import math
import os

_spec = importlib.util.spec_from_file_location(
    "gv_geo", os.path.join(os.path.dirname(__file__), "scripts", "gridvision_geo.py"))
geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geo)


def test_pixel_to_lonlat_corners_and_center():
    bbox = [-110.0, 32.0, -109.9, 32.1]   # W,S,E,N  (0.1 deg square)
    # top-left pixel -> NW corner (W, N)
    assert geo.pixel_to_lonlat(0, 0, bbox, 100, 100) == (-110.0, 32.1)
    # bottom-right pixel -> SE corner (E, S)
    lon, lat = geo.pixel_to_lonlat(100, 100, bbox, 100, 100)
    assert abs(lon - (-109.9)) < 1e-9 and abs(lat - 32.0) < 1e-9
    # center
    lon, lat = geo.pixel_to_lonlat(50, 50, bbox, 100, 100)
    assert abs(lon - (-109.95)) < 1e-9 and abs(lat - 32.05) < 1e-9


def test_box_center_lonlat():
    bbox = [-110.0, 32.0, -109.0, 33.0]
    lon, lat = geo.box_center_lonlat([0, 0, 100, 100], bbox, 100, 100)
    assert abs(lon - (-109.5)) < 1e-9 and abs(lat - 32.5) < 1e-9


def test_haversine_known_distance():
    # ~0.001 deg latitude ~ 111 m
    d = geo.haversine_m(-110.0, 32.0, -110.0, 32.001)
    assert 100 < d < 120


def test_dedup_points_collapses_seam_dupes():
    pts = [(-110.0, 32.0, 0.9), (-110.00001, 32.00001, 0.7), (-110.01, 32.01, 0.8)]
    kept = geo.dedup_points(pts, min_sep_m=30.0)
    assert len(kept) == 2 and kept[0][2] == 0.9   # nearer-dup dropped, top score kept


def test_recall_vs_osm_lower_bound():
    osm = [(-110.0, 32.0), (-110.01, 32.01), (-110.02, 32.02)]
    det = [(-110.0, 32.00005, 0.9), (-110.01, 32.01, 0.8)]   # 2 of 3 OSM covered
    r = geo.recall_vs_osm(det, osm, match_m=30.0)
    assert r["n_osm"] == 3 and r["n_matched"] == 2
    assert abs(r["recall_lower_bound"] - 2 / 3) < 1e-4
    # no OSM -> recall undefined, not a crash or a fake 0
    assert geo.recall_vs_osm(det, [], 30.0)["recall_lower_bound"] is None


def test_osm_corroborated_gates_conf_and_proximity():
    osm = [(-110.0, 32.0)]
    dets = [(-110.0, 32.00005, 0.9),   # high conf + near OSM -> corroborated
            (-110.0, 32.00005, 0.3),   # near OSM but low conf -> excluded
            (-120.0, 40.0, 0.95)]      # high conf but far from any OSM -> excluded
    out = geo.osm_corroborated(dets, osm, match_m=30.0, min_conf=0.5)
    assert len(out) == 1 and out[0][2] == 0.9
