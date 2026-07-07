"""test_grid_ba_capacity.py — GRID VISION A1 step-3 battery
(scripts/grid_ba_capacity.py). Pure geometry tests (ray casting with
holes/multipart, last-hit cache) plus committed-artifact coherence:
the conservation invariant is RECOMPUTED here so it cannot drift from
the committed registry."""
import importlib.util
import json
import os

_spec = importlib.util.spec_from_file_location(
    "grid_ba_capacity",
    os.path.join(os.path.dirname(__file__), "scripts", "grid_ba_capacity.py"))
gb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gb)

SQUARE = [[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]]
SQUARE_WITH_HOLE = SQUARE + [[(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]]
TWO_ISLANDS = [[(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)],
               [(5, 5), (7, 5), (7, 7), (5, 7), (5, 5)]]


def test_point_in_rings_basic_hole_multipart():
    assert gb.point_in_rings(5, 5, SQUARE)
    assert not gb.point_in_rings(11, 5, SQUARE)
    assert not gb.point_in_rings(5, 5, SQUARE_WITH_HOLE), "point in hole is outside"
    assert gb.point_in_rings(2, 2, SQUARE_WITH_HOLE)
    assert gb.point_in_rings(1, 1, TWO_ISLANDS)
    assert gb.point_in_rings(6, 6, TWO_ISLANDS)
    assert not gb.point_in_rings(3.5, 3.5, TWO_ISLANDS)


def test_county_index_last_hit_cache_stays_correct():
    counties = [("A", "001", (0, 0, 10, 10), SQUARE),
                ("B", "002", (20, 0, 30, 10), [[(20, 0), (30, 0), (30, 10), (20, 10), (20, 0)]])]
    idx = gb.CountyIndex(counties)
    assert idx.locate(5, 5) == "A"
    assert idx.locate(6, 5) == "A"      # cache hit
    assert idx.locate(25, 5) == "B"     # cache miss -> correct re-scan
    assert idx.locate(24, 5) == "B"
    assert idx.locate(15, 5) is None    # between counties -> None, never guessed


def test_committed_artifact_conservation_recomputed():
    base = os.path.join(os.path.dirname(__file__), "datacore", "gridvision")
    a = json.load(open(os.path.join(base, "tx_ba_capacity.json")))
    reg = json.load(open(os.path.join(base, "tx_grid_registry.json")))

    # recompute conservation from committed pieces — same-input invariant
    dist = {}
    for cls_km in list(a["per_county_circuit_km"].values()) + [a["out_of_state_circuit_km"]]:
        for cls, km in cls_km.items():
            dist[cls] = dist.get(cls, 0.0) + km
    reg_total = sum(reg["lines"]["circuit_km_by_class"].values())
    dist_total = sum(dist.values())
    assert abs(reg_total - dist_total) <= max(0.005 * reg_total, 1.0), \
        f"conservation broke: registry {reg_total} vs distributed {dist_total}"
    assert a["conservation"]["conserved_within_0p5pct"] is True

    # 2026-07-07 run facts (pin the shipped state; re-runs update deliberately)
    assert a["ways_processed"] == 31220
    assert len(a["per_county_circuit_km"]) == 254, "every TX county carries capacity"
    assert abs(a["ba_exclusive_circuit_km"]["ERCO"]["345kV+"] - 19206.8) < 0.2
    assert 0.15 <= a["kv345_ambiguous_share"] <= 0.35, \
        "pre-stated CREZ-seam band (products plan); outside it = investigate"
    # ambiguous pools are BA SETS, never invented splits
    for key in a["ba_ambiguous_circuit_km"]:
        assert "|" in key, f"ambiguous pool key must be a multi-BA set: {key}"
    assert a["unknown_county_circuit_km"] == {}, \
        "every county name from the shapefile must resolve in the crosswalk"
    assert "never split by invented ratio" in a["provenance"]["method"]
