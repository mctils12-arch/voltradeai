"""Regression tests for the nuclear-tests site-consistency gate.

The original import gate caught impossible coordinates only. EGMONT (UK,
site NTS) was plotted at 36.0,-112.0 beside the Grand Canyon and MADISON
(USA, site NTS) at Novaya Zemlya — valid coordinates contradicting the
record's own site. These tests would have failed against that data.
"""
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "nuclear_tests_site_check", ROOT / "scripts" / "nuclear_tests_site_check.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _site(site, lat, lon, n, jitter=0.02):
    return [{"n": f"{site}-{i}", "c": "X", "d": "1970-01-01", "r": site,
             "lat": lat + (i % 5) * jitter, "lon": lon - (i % 3) * jitter} for i in range(n)]


def test_flags_a_valid_coordinate_that_contradicts_its_compact_site():
    tests = _site("NTS", 37.0, -116.0, 40)
    tests.append({"n": "EGMONT", "c": "UK", "d": "1984-12-09", "r": "NTS", "lat": 36.0, "lon": -112.0})
    found = gate.find_contradictions(tests)
    assert [f["name"] for f in found] == ["EGMONT"]
    assert found[0]["dist_km"] > 300


def test_regional_labels_are_never_judged():
    # Soviet peaceful-explosion programs are labelled by REGION and really do
    # spread over hundreds of km — they must not be "corrected" onto a centroid
    tests = []
    for i, (lat, lon) in enumerate([(60.8, 97.6), (69.2, 81.6), (69.6, 90.5), (64.3, 91.8),
                                    (61.0, 95.0), (66.0, 88.0), (68.0, 93.0)]):
        tests.append({"n": f"PNE-{i}", "c": "USSR", "d": "1978-01-01",
                      "r": "KRASNO RUSS", "lat": lat, "lon": lon})
    assert gate.find_contradictions(tests) == []


def test_normal_spread_within_a_large_range_is_not_flagged():
    # Novaya Zemlya's tests legitimately span ~260 km of islands
    tests = _site("NZ RUSS", 73.4, 54.8, 30, jitter=0.5)
    tests.append({"n": "FAR-BUT-REAL", "c": "USSR", "d": "1961-10-30", "r": "NZ RUSS",
                  "lat": 75.3, "lon": 55.5})
    assert gate.find_contradictions(tests) == []


def test_apply_replots_at_site_and_keeps_catalog_coordinates():
    tests = _site("NTS", 37.0, -116.0, 40)
    tests.append({"n": "MADISON", "c": "USA", "d": "1962-12-12", "r": "NTS", "lat": 74.3, "lon": 52.4})
    found = gate.find_contradictions(tests)
    gate.apply_fixes(tests, found)
    rec = tests[-1]
    assert rec["loc"] == "site"
    assert (rec["src_lat"], rec["src_lon"]) == (74.3, 52.4), "catalog coordinates must be preserved, not discarded"
    assert abs(rec["lat"] - 37.0) < 0.2 and abs(rec["lon"] + 116.0) < 0.2
    # idempotent: a resolved record is not found again
    assert gate.find_contradictions(tests) == []


def test_committed_dataset_has_no_unresolved_contradictions():
    doc = json.loads((ROOT / "datacore" / "nuclear_tests.json").read_text())
    assert gate.find_contradictions(copy.deepcopy(doc["tests"])) == []
    fixed = {r["n"] for r in doc["tests"] if r.get("loc") == "site"}
    assert {"EGMONT", "MADISON", "LEDA", "TELKEM-2", "ACHILLE"} <= fixed
