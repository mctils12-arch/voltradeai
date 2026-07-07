"""
test_gem_ingest.py — GEM asset-registry ingest battery (scripts/gem_ingest.py).
Pure-function tests on synthetic structures shaped like the real release
files (probed 2026-07-07: master FeatureCollection with GEM Mine ID
properties; xlsx sheets with trailing all-None rows — verified 729 of
them on the LNG sheet). Plus committed-artifact coherence.
"""
import importlib.util
import json
import os
from datetime import datetime

import pytest

_spec = importlib.util.spec_from_file_location(
    "gem_ingest", os.path.join(os.path.dirname(__file__), "scripts", "gem_ingest.py"))
gem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gem)


def _feature(mid, cat, sub, coords, geom_type="Polygon", **props):
    return {
        "type": "Feature",
        "geometry": {"type": geom_type, "coordinates": coords},
        "properties": {
            "GEM Mine ID": mid, "Mine Name": f"Mine {mid}", "Country / Area": "Testland",
            "Coal Grade": "thermal", "Owners": "Owner Co [100%]", "Parent Company": "Parent Co",
            "mine feature category": cat, "mine feature subcategory": sub, **props,
        },
    }


def test_summarize_coal_mines_bbox_centroid_categories():
    master = {"type": "FeatureCollection", "features": [
        _feature("M1", "mine boundary", "surface", [[[10.0, 50.0], [10.2, 50.0], [10.2, 50.2], [10.0, 50.2], [10.0, 50.0]]]),
        _feature("M1", "methane source", "vent", [10.1, 50.1], geom_type="Point"),
        _feature("M2", "mine boundary", None, [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]),
    ]}
    mines = gem.summarize_coal_mines(master)
    assert set(mines) == {"M1", "M2"}
    m1 = mines["M1"]
    assert m1["n_features"] == 2
    assert m1["categories"] == {"mine boundary/surface": 1, "methane source/vent": 1}
    assert m1["bbox"] == [10.0, 50.0, 10.2, 50.2]
    assert m1["centroid"] == [10.1, 50.1]
    assert "not a surveyed point" in m1["centroid_note"], "centroid honesty label required"
    assert mines["M2"]["categories"] == {"mine boundary": 1}
    assert "_coords" not in m1, "internal accumulator must not leak into the artifact"


def test_summarize_skips_featureless_ids_and_handles_multipolygon():
    master = {"features": [
        _feature("M3", "mine boundary", "underground",
                 [[[[0, 0], [2, 0], [2, 2], [0, 0]]], [[[5, 5], [6, 5], [6, 6], [5, 5]]]],
                 geom_type="MultiPolygon"),
        {"type": "Feature", "geometry": None, "properties": {}},  # no GEM Mine ID → skipped
    ]}
    mines = gem.summarize_coal_mines(master)
    assert set(mines) == {"M3"}
    assert mines["M3"]["bbox"] == [0, 0, 6, 6]


class _FakeWs:
    def __init__(self, rows): self._rows = rows
    def iter_rows(self, values_only=True):
        return iter(self._rows)


def test_sheet_rows_drops_all_none_rows_nulls_and_isoifies_dates():
    ws = _FakeWs([
        ("Country", "ID", "FID Date", None),
        ("US", "T1", datetime(2025, 3, 4), None),
        (None, None, None, None),          # trailing formatted-empty (verified 729 on LNG)
        ("FR", "T2", None, "extra"),
        (None, None, None, None),
    ])
    rows = gem.sheet_rows(ws)
    assert len(rows) == 2
    assert rows[0] == {"Country": "US", "ID": "T1", "FID Date": "2025-03-04"}
    assert "FID Date" not in rows[1], "nulls dropped per row, never zero-filled"
    assert rows[1]["col3"] == "extra", "unnamed columns keep positional names"


def test_committed_artifacts_coherent():
    base = os.path.join(os.path.dirname(__file__), "datacore", "gem")
    mines = json.load(open(os.path.join(base, "coal_mines.json")))
    assert mines["mine_count"] == len(mines["mines"]) >= 250
    assert mines["feature_count"] >= 2000
    assert mines["provenance"]["attribution"] == "Global Energy Monitor"
    assert "CC BY 4.0" in mines["provenance"]["license"]
    appin = mines["mines"]["M0005"]
    assert appin["country"] == "Australia" and appin["centroid"][0] > 150, "spot-check: Appin is in NSW"

    pipes = json.load(open(os.path.join(base, "gas_pipelines.json")))
    assert pipes["row_count"] == len(pipes["pipelines"]) == 4246
    assert "NO route coordinates" in pipes["provenance"]["geometry_note"], "geometry-absence honesty required"
    statuses = {p.get("Status") for p in pipes["pipelines"][:200]}
    assert "operating" in statuses

    fin = json.load(open(os.path.join(base, "gas_finance.json")))
    assert fin["lng_terminal_count"] == len(fin["lng_terminals"]) == 243
    assert fin["gas_plant_count"] == len(fin["gas_power_plants"]) == 531


def test_suite_artifacts_coherent():
    """Part-2 suite artifacts (gem_suite_ingest.py) — committed-file
    coherence: counts self-consistent, join keys present, projections
    and skips stated."""
    import gzip
    base = os.path.join(os.path.dirname(__file__), "datacore", "gem")

    def load(name):
        gz = os.path.join(base, name + ".json.gz")
        if os.path.exists(gz):
            with gzip.open(gz, "rt") as f:
                return json.load(f)
        return json.load(open(os.path.join(base, name + ".json")))

    own = load("ownership")
    assert own["counts"] == {"entities": 26250, "entity_edges": 24351, "asset_edges": 49391}
    ent0 = own["entities"][0]
    assert "Entity ID" in ent0
    ciks = [e for e in own["entities"][:2000] if e.get("US SEC Central Index Key")]
    assert len(ciks) > 0, "SEC CIK crosswalk must survive ingest (the EDGAR join spine)"
    edge = own["entity_edges"][0]
    assert "Subject Entity ID" in edge and "Interested Party ID" in edge

    power = load("power_units")
    assert power["unit_count"] == len(power["units"]) == 182428
    assert power["fields"][0] == "Type" and "Latitude" in power["fields"]
    assert "projection" in power["provenance_note" if "provenance_note" in power else "projection_note"].lower()
    types = {u[0] for u in power["units"]}  # full pass — the sheet is type-ordered
    assert types == {"bioenergy", "coal", "geothermal", "hydropower", "nuclear",
                     "oil/gas", "utility-scale solar", "wind"}, f"unexpected type census: {sorted(types)}"

    mines = load("coal_mine_tracker")
    assert mines["counts"]["non_closed"] == 5382

    fin = load("coal_finance")
    assert fin["counts"]["financings"] == 6556
    assert "Financier" in fin["financings"][0], "header-row-2 contract (banner row skipped)"

    carriers = load("lng_carriers")
    assert any("IMO number" in c for c in carriers["carriers"][:5]), "IMO join key to the AIS archive"

    gmet = load("methane_emitters")
    assert gmet["counts"]["plumes"] == 3474

    skips = json.load(open(os.path.join(base, "suite_skips.json")))
    assert "Portal_Energetico_tracker_2026-06-24.xlsx" in skips["skipped"], "skips must be stated, never silent"
    assert "geot_rollups" in skips
