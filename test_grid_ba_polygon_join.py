"""Pure-function tests for scripts/grid_ba_polygon_join.py — no network,
no real datacore files touched (synthetic fixtures only)."""
import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "grid_ba_polygon_join", os.path.join(_HERE, "scripts", "grid_ba_polygon_join.py"))
gbpj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gbpj)


# ---- identity: reuses grid_ba_capacity.py's point_in_rings, does not
# re-derive its own ray-casting implementation (EDGE DOCTRINE #3) ----

def test_reuses_grid_ba_capacity_point_in_rings():
    # gbpj's own module-level `point_in_rings = _gbc.point_in_rings` (its
    # already-imported copy of grid_ba_capacity.py) must be the exact same
    # function object the module calls — proves the join reuses it rather
    # than re-deriving its own ray-casting implementation (EDGE DOCTRINE #3).
    assert gbpj.point_in_rings is gbpj._gbc.point_in_rings


# ---- normalize_entity_name ----

def test_normalize_strips_corporate_suffixes_and_punctuation():
    assert gbpj.normalize_entity_name("PJM Interconnection, LLC") == "PJM INTERCONNECTION"
    assert gbpj.normalize_entity_name("Idaho Power Company") == "IDAHO POWER"


def test_normalize_collapses_whitespace():
    assert gbpj.normalize_entity_name("  Duke   Energy  Carolinas ") == "DUKE ENERGY CAROLINAS"


# ---- match_ba_crosswalk ----

def test_crosswalk_exact_match():
    hifld = [{"id": "h1", "name": "Southwest Power Pool"}]
    eia = [{"id": "SWPP", "name": "Southwest Power Pool"}]
    matched, unmatched = gbpj.match_ba_crosswalk(hifld, eia, overrides={}, excluded={})
    assert unmatched == []
    assert matched == [{"hifld_id": "h1", "hifld_name": "Southwest Power Pool",
                         "ba_code": "SWPP", "ba_name": "Southwest Power Pool", "method": "exact"}]


def test_crosswalk_substring_match():
    hifld = [{"id": "h1", "name": "Salt River Project"}]
    eia = [{"id": "SRP", "name": "Salt River Project Agricultural Improvement and Power District"}]
    matched, unmatched = gbpj.match_ba_crosswalk(hifld, eia, overrides={}, excluded={})
    assert matched[0]["ba_code"] == "SRP"
    assert matched[0]["method"] == "substr"


def test_crosswalk_override_wins_even_if_a_string_match_would_also_exist():
    hifld = [{"id": "h1", "name": "Totally Different Wording"}]
    eia = [{"id": "OTHER", "name": "Totally Different Wording"}]
    overrides = {"h1": ("REAL", "Real Entity", "manual override reason")}
    matched, unmatched = gbpj.match_ba_crosswalk(hifld, eia, overrides=overrides, excluded={})
    assert matched[0]["ba_code"] == "REAL"
    assert matched[0]["method"].startswith("override:")


def test_crosswalk_no_match_falls_back_to_excluded_reason():
    hifld = [{"id": "h1", "name": "Nonexistent Utility"}]
    eia = [{"id": "SWPP", "name": "Southwest Power Pool"}]
    matched, unmatched = gbpj.match_ba_crosswalk(
        hifld, eia, overrides={}, excluded={"h1": "known non-EIA930 entity"})
    assert matched == []
    assert unmatched == [{"hifld_id": "h1", "hifld_name": "Nonexistent Utility",
                           "reason": "known non-EIA930 entity"}]


def test_crosswalk_no_match_and_not_excluded_reports_honest_default_reason():
    hifld = [{"id": "h9", "name": "Brand New Utility Nobody Has Seen"}]
    eia = [{"id": "SWPP", "name": "Southwest Power Pool"}]
    matched, unmatched = gbpj.match_ba_crosswalk(hifld, eia, overrides={}, excluded={})
    assert unmatched[0]["reason"] == "no match found"


def test_the_71_hardcoded_overrides_and_exclusions_are_disjoint_and_have_no_duplicate_hifld_ids():
    # regression guard: an id can't be both a confirmed match and a
    # confirmed non-match, and the two hardcoded tables can't silently
    # grow a duplicate key
    assert set(gbpj.BA_CROSSWALK_OVERRIDES) & set(gbpj.BA_EXCLUDED) == set()
    assert len(gbpj.BA_CROSSWALK_OVERRIDES) == 6
    assert len(gbpj.BA_EXCLUDED) == 7


# ---- geometry_rings / bbox_of_rings ----

def test_geometry_rings_polygon():
    geom = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}
    rings = gbpj.geometry_rings(geom)
    assert rings == geom["coordinates"]


def test_geometry_rings_multipolygon_flattens_all_parts():
    geom = {"type": "MultiPolygon", "coordinates": [
        [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
        [[[5, 5], [5, 6], [6, 6], [6, 5], [5, 5]]],
    ]}
    rings = gbpj.geometry_rings(geom)
    assert len(rings) == 2


def test_geometry_rings_none_geometry_is_empty_not_a_crash():
    assert gbpj.geometry_rings(None) == []


def test_bbox_of_rings():
    rings = [[[0, 0], [0, 1], [2, 1], [2, 0], [0, 0]]]
    assert gbpj.bbox_of_rings(rings) == (0, 0, 2, 1)


def test_bbox_of_rings_empty_is_none():
    assert gbpj.bbox_of_rings([]) is None


# ---- build_ba_index ----

_SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]]}


def test_build_ba_index_skips_features_with_no_resolved_code():
    features = [
        {"properties": {"ba_id": "h1", "name": "A"}, "geometry": _SQUARE},
        {"properties": {"ba_id": "h2", "name": "B"}, "geometry": _SQUARE},
    ]
    index = gbpj.build_ba_index(features, {"h1": "AAA"})  # h2 unresolved
    assert len(index) == 1
    assert index[0][0] == "AAA"


# ---- assign_plant_ba / summarize_assignment ----

def test_assign_plant_inside_one_polygon():
    index = [("AAA", (0, 0, 10, 10), _SQUARE["coordinates"])]
    plants = [["Plant A", 100.0, "gas", "Owner", 5.0, 5.0, 1]]
    out = gbpj.assign_plant_ba(plants, index)
    assert out == [{"name": "Plant A", "fuel": "gas", "capacity_mw": 100.0, "ba_codes": ["AAA"]}]


def test_assign_plant_outside_every_polygon_is_honestly_unmatched():
    index = [("AAA", (0, 0, 10, 10), _SQUARE["coordinates"])]
    plants = [["Plant Z", 50.0, "wind", "Owner", 99.0, 99.0, 1]]
    out = gbpj.assign_plant_ba(plants, index)
    assert out[0]["ba_codes"] == []


def test_assign_plant_with_missing_coordinates_is_unmatched_not_a_crash():
    index = [("AAA", (0, 0, 10, 10), _SQUARE["coordinates"])]
    plants = [["Plant NoCoords", 10.0, "solar", "Owner", None, None, 0]]
    out = gbpj.assign_plant_ba(plants, index)
    assert out[0]["ba_codes"] == []


def test_assign_plant_in_two_overlapping_polygons_reports_both_never_guesses():
    overlap = {"type": "Polygon", "coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]]}
    index = [
        ("AAA", (0, 0, 10, 10), _SQUARE["coordinates"]),
        ("BBB", (0, 0, 10, 10), overlap["coordinates"]),
    ]
    plants = [["Plant Overlap", 20.0, "hydro", "Owner", 5.0, 5.0, 1]]
    out = gbpj.assign_plant_ba(plants, index)
    assert set(out[0]["ba_codes"]) == {"AAA", "BBB"}


def test_summarize_assignment_counts_match_unmatch_ambiguous_and_sums_capacity_by_ba():
    assignments = [
        {"name": "P1", "fuel": "gas", "capacity_mw": 10.0, "ba_codes": ["AAA"]},
        {"name": "P2", "fuel": "gas", "capacity_mw": 20.0, "ba_codes": ["AAA"]},
        {"name": "P3", "fuel": "wind", "capacity_mw": 5.0, "ba_codes": []},
        {"name": "P4", "fuel": "solar", "capacity_mw": 1.0, "ba_codes": ["AAA", "BBB"]},
    ]
    summary = gbpj.summarize_assignment(assignments)
    assert summary == {
        "total": 4, "matched": 2, "unmatched": 1, "ambiguous": 1,
        "by_ba": {"AAA": {"count": 2, "capacity_mw": 30.0}},
    }


def test_summarize_assignment_treats_missing_capacity_as_zero_not_a_crash():
    assignments = [{"name": "P1", "fuel": "gas", "capacity_mw": None, "ba_codes": ["AAA"]}]
    summary = gbpj.summarize_assignment(assignments)
    assert summary["by_ba"]["AAA"]["capacity_mw"] == 0.0
