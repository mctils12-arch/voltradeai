"""Tests for scripts/geo_consistency.py — the reusable "point contradicts its
own place fields" gate (generalized from the 2026-09-28 EGMONT bug). Offline:
tiny synthetic boundaries, no Natural Earth download."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("geo_consistency", ROOT / "scripts" / "geo_consistency.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)


def _sq(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


ADMIN0 = {"features": [
    {"properties": {"ISO_A2": "US", "ADM0_A3": "USA", "NAME": "United States of America"},
     "geometry": _sq(-120, 30, -100, 45)},
    {"properties": {"ISO_A2": "SA", "ADM0_A3": "SAU", "NAME": "Saudi Arabia"},
     "geometry": _sq(40, 20, 50, 30)},
]}
ADMIN1 = {"features": [
    {"properties": {"iso_a2": "US", "name": "Nevada", "postal": "NV"}, "geometry": _sq(-120, 35, -114, 42)},
    {"properties": {"iso_a2": "US", "name": "Arizona", "postal": "AZ"}, "geometry": _sq(-114, 31, -109, 37)},
]}
B = g.Boundaries(admin0=ADMIN0, admin1=ADMIN1)


def test_inside_claim_is_not_flagged_and_aliases_resolve():
    recs = [{"name": "a", "lat": 37, "lon": -116, "country": "USA"},
            {"name": "b", "lat": 37, "lon": -116, "country": "United States"},
            {"name": "c", "lat": 25, "lon": 45, "country": "Saudi Arabia"}]
    found, stats = g.check_country(recs, B)
    assert found == [] and stats["judged"] == 3


def test_contradiction_is_diagnosed():
    recs = [{"name": "swapped", "lat": 45, "lon": 25, "country": "SA"},        # valid but swapped
            {"name": "swapped-oor", "lat": -116, "lon": 37, "country": "US"},  # lat out of range
            {"name": "lonflip", "lat": 37, "lon": 116, "country": "US"},
            {"name": "plain", "lat": 25, "lon": 45, "country": "US"}]           # in SA, claims US
    found, _ = g.check_country(recs, B)
    diag = {f["name"]: f["diagnosis"] for f in found}
    assert diag == {"swapped": "swapped_latlon", "swapped-oor": "swapped_latlon",
                    "lonflip": "lon_sign_flipped", "plain": "mismatch"}
    assert next(f for f in found if f["name"] == "plain")["actually_in"] == "SA"


def test_zero_distance_counts_as_inside():
    # regression: `d or 1e9` turned 0.0 km (inside) into "infinitely far"
    assert g._within(0.0, 25) is True
    assert g._within(None, 25) is False


def test_tolerance_spares_coastal_points_and_unknown_claims_are_counted():
    recs = [{"name": "offshore", "lat": 37, "lon": -120.1, "country": "US"},   # ~9 km off the edge
            {"name": "x", "lat": 1, "lon": 1, "country": "Atlantis"}]
    found, stats = g.check_country(recs, B, tol_km=25)
    assert found == [] and stats["unknown_claim"] == 1


def test_admin1_wrong_state_like_egmont():
    recs = [{"name": "ok", "lat": 37, "lon": -116, "state": "NV"},
            {"name": "EGMONT-like", "lat": 36, "lon": -112, "state": "Nevada"}]
    found, _ = g.check_admin1(recs, B)
    assert [f["name"] for f in found] == ["EGMONT-like"]
    assert found[0]["km_outside_claim"] > 100


def test_site_outliers_generalizes_the_nuclear_gate():
    recs = [{"name": f"t{i}", "site": "NTS", "lat": 37 + (i % 5) * 0.02, "lon": -116 - (i % 3) * 0.02}
            for i in range(30)]
    recs.append({"name": "EGMONT", "site": "NTS", "lat": 36.0, "lon": -112.0})
    regional = [{"name": f"r{i}", "site": "REGION", "lat": 60 + i, "lon": 90 + 2 * i} for i in range(8)]
    out = g.site_outliers(recs + regional)
    assert [o["name"] for o in out] == ["EGMONT"]


def test_fallback_distance_is_to_the_edge_not_the_nearest_vertex():
    # a point 0.1 deg west of a 15-deg-long straight border: ~9 km, not ~800
    d = g._seg_km(37.0, -120.1, [-120, 30], [-120, 45])
    assert 8 < d < 10
