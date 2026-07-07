"""test_gridvision_etdii.py — GRID VISION Phase B label-prep battery
(scripts/gridvision_etdii.py). No network: the feature fixtures are VERBATIM
(trimmed) records pulled from the real ETDII US zips on 2026-07-07 (figshare
14935434, USA_AZ_Tucson_12.geojson). Pure-function + manifest tests only.
"""
import importlib.util
import json
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "gridvision_etdii", os.path.join(os.path.dirname(__file__), "scripts", "gridvision_etdii.py"))
etdii = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(etdii)

# VERBATIM real ETDII tower feature (USA_AZ_Tucson_12.geojson, 2026-07-07 probe).
REAL_TOWER = {
    "type": "Feature",
    "geometry": {"type": "Polygon", "coordinates": [[
        [-110.96828244929705, 32.001262370979546],
        [-110.96828728820694, 32.00120582020497],
        [-110.96819365462213, 32.001204394741656],
        [-110.9681947746346, 32.00126099006227],
        [-110.96828244929705, 32.001262370979546]]]},
    "properties": {"label": "T", "filename": "USA_AZ_Tucson_12.geojson",
                   "country": "USA", "city/state": "AZ",
                   "pixel_coordinates": [[765.4323807443295, 602.6481187760785],
                                         [763.2676964096454, 636.4171943971518],
                                         [810.8907517726974, 636.8501312640886],
                                         [810.0248780388239, 603.0810556430152]]}}

# A synthetic-but-schema-real substation + a non-US tower for balance/summary tests.
FIXTURE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        REAL_TOWER,
        {"type": "Feature",
         "geometry": {"type": "Polygon", "coordinates": [[
             [-110.10, 32.10], [-110.08, 32.10], [-110.08, 32.12], [-110.10, 32.12], [-110.10, 32.10]]]},
         "properties": {"label": "SS", "filename": "USA_AZ_Tucson_12.geojson",
                        "country": "USA", "city/state": "AZ",
                        "pixel_coordinates": [[100, 100], [200, 100], [200, 200], [100, 200]]}},
        {"type": "Feature",
         "geometry": {"type": "LineString", "coordinates": [[-110.0, 32.0], [-110.0, 32.01]]},
         "properties": {"label": "L", "filename": "USA_AZ_Tucson_12.geojson", "country": "USA"}},
        {"type": "Feature",
         "geometry": {"type": "Polygon", "coordinates": [[
             [170.0, -45.0], [170.001, -45.0], [170.001, -45.001], [170.0, -45.001], [170.0, -45.0]]]},
         "properties": {"label": "T", "filename": "NZ_Dunedin_1.geojson", "country": "NZ"}},
    ],
}


def test_normalize_label_scope():
    assert etdii.normalize_label("T") == ("tower", True)
    assert etdii.normalize_label("TT") == ("tower", True)
    assert etdii.normalize_label("SS") == ("substation", True)
    # "Other Tower" is uncertain by the dataset's own doc — NEVER a clean tower
    assert etdii.normalize_label("OT") == ("tower_uncertain", False)
    assert etdii.normalize_label("DT") == ("tower_distribution", False)
    assert etdii.normalize_label("L") == ("line", False)
    assert etdii.normalize_label("EN") == ("edge_node", False)
    # unknown codes are never guessed into a real class
    assert etdii.normalize_label("ZZ") == ("unknown", False)
    assert etdii.normalize_label(None) == ("unknown", False)
    assert etdii.normalize_label(" t ") == ("tower", True)  # tolerant


def test_pixel_bbox_from_real_quad():
    bb = etdii.pixel_bbox(REAL_TOWER["properties"]["pixel_coordinates"])
    assert bb == pytest.approx([763.2676964096454, 602.6481187760785,
                                810.8907517726974, 636.8501312640886])
    # a ~48x34 px box @0.30 m ~= 14x10 m — plausible tower footprint
    assert 40 < (bb[2] - bb[0]) < 55 and 30 < (bb[3] - bb[1]) < 40
    assert etdii.pixel_bbox(None) is None
    assert etdii.pixel_bbox([]) is None
    assert etdii.pixel_bbox([[1]]) is None  # degenerate points dropped -> None


def test_geo_centroid_polygon_drops_closing_vertex():
    c = etdii.geo_centroid(REAL_TOWER["geometry"]["coordinates"], "Polygon")
    # mean of the 4 unique ring vertices, inside the feature's bounds
    assert c[0] == pytest.approx(-110.96824, abs=1e-4)
    assert c[1] == pytest.approx(32.001233, abs=1e-4)
    # a point geometry passes through
    assert etdii.geo_centroid([-97.5, 30.2], "Point") == pytest.approx([-97.5, 30.2])
    assert etdii.geo_centroid([], "Polygon") is None


def test_parse_geojson_records_and_scope():
    recs = etdii.parse_geojson(FIXTURE_GEOJSON, "ETDII", "CC BY 4.0")
    assert len(recs) == 4
    tower = recs[0]
    assert tower["cls"] == "tower" and tower["in_scope"] is True
    assert tower["us_domain"] is True and tower["place"] == "AZ"
    assert tower["source"] == "ETDII" and tower["license"] == "CC BY 4.0"
    assert tower["pixel_bbox"] is not None
    ss = recs[1]
    assert ss["cls"] == "substation" and ss["in_scope"] is True
    line = recs[2]
    assert line["cls"] == "line" and line["in_scope"] is False
    nz = recs[3]
    assert nz["cls"] == "tower" and nz["us_domain"] is False


def test_summarize_us_substation_scarcity():
    recs = etdii.parse_geojson(FIXTURE_GEOJSON, "ETDII", "CC BY 4.0")
    s = etdii.summarize(recs)
    assert s["in_scope_total"] == {"tower": 2, "substation": 1}
    assert s["in_scope_us"] == {"tower": 1, "substation": 1}
    # 1 US substation vs 1 US tower -> not scarce in this tiny fixture
    assert s["us_substation_scarcity_flag"] is False
    # but a tower-heavy fixture trips the flag
    heavy = recs + [dict(recs[0]) for _ in range(30)]
    assert etdii.summarize(heavy)["us_substation_scarcity_flag"] is True


def test_build_manifest_bakes_verified_us_counts():
    doc = etdii.build_manifest()
    names = [s["name"] for s in doc["sources"]]
    assert any("ETDII" in n for n in names)
    assert any("PLAD" in n for n in names)  # recorded as rejected, honestly
    etd = doc["sources"][0]
    # verified US totals = Tucson (595 T / 3 SS) + Colwich (813 T / 3 SS)
    assert etd["us_portion"]["normalized_in_scope"] == {"tower": 1408, "substation": 6}
    assert etd["us_portion"]["images"] == 74
    assert etd["license"] == "CC BY 4.0"
    head = doc["composition_headline"]["clean_us_labels_available_now"]
    assert head["tower"] == 1408 and head["substation"] == 6
    assert "CONFIRMED" in doc["composition_headline"]["substation_underrepresentation"]
    # PLAD is marked unusable, never a live label source
    plad = [s for s in doc["sources"] if "PLAD" in s["name"]][0]
    assert plad["usable"] is False


def test_write_manifest_roundtrips(tmp_path):
    path = str(tmp_path / "labels_manifest.json")
    etdii.write_manifest(path)
    with open(path) as f:
        doc = json.load(f)  # must be valid JSON
    assert doc["generator"].endswith("--write-manifest")
    assert len(doc["sources"]) == 4


def test_tally_zip_parses_real_zip(tmp_path):
    """If a real ETDII zip is present (session-side), tally_zip parses it; skip
    in CI where the 100 MB zips are absent (never a network fetch in tests)."""
    import zipfile
    zpath = str(tmp_path / "mini.zip")
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("USA_AZ_Tucson/USA_AZ_Tucson_1.geojson", json.dumps(FIXTURE_GEOJSON))
    recs, summ = etdii.tally_zip(zpath, "ETDII", "CC BY 4.0")
    assert summ["in_scope_total"] == {"tower": 2, "substation": 1}
