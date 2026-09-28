#!/usr/bin/env python3
"""Geo-consistency checks: does a record's POINT agree with the record's OWN
place fields (country / state / named site)?

Born from the 2026-09-28 EGMONT bug: a UK nuclear test whose catalog site
said "Nevada Test Site" but whose coordinates sat by the Grand Canyon. The
old import gates only rejected IMPOSSIBLE coordinates (out of range, null
island); a VALID coordinate contradicting the record's own metadata passed
silently. This module is the reusable version of that gate, for every layer.

Checks (all pure functions over plain dict records):
  * check_country(records, ...)  point vs claimed country (ISO2/ISO3/name)
  * check_admin1(records, ...)   point vs claimed state/province
  * site_outliers(records, ...)  point far from the other records sharing its
                                 named site (the generalized nuclear gate)
Each contradiction is DIAGNOSED: swapped lat/lon, flipped sign, or plain
mismatch — a swap/sign fix is mechanical; a mismatch needs a source.

Boundaries: Natural Earth 1:10m admin-0 + admin-1 (public domain), fetched
once into a local cache (not committed — ~54 MB). Coastal/offshore records
(ports, platforms, buoys) sit OUTSIDE land polygons legitimately, so a point
only contradicts its claim when it is more than `tol_km` from the claimed
polygon. Uses shapely when installed (fast, exact nearest-point distance);
otherwise a pure-Python even-odd ray cast with a vertex-distance fallback.

CLI (quick look at one JSON file):
  python3 scripts/geo_consistency.py FILE --path tests --lat lat --lon lon \
      [--country c] [--state st] [--site r] [--name n] [--tol-km 25]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

CACHE = Path(os.environ.get("VOLTRADE_GEO_CACHE", Path.home() / ".cache" / "voltrade_geo"))
NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
ADMIN0 = "ne_10m_admin_0_countries.geojson"
ADMIN1 = "ne_10m_admin_1_states_provinces.geojson"

try:  # optional accelerator
    from shapely.geometry import Point, shape
    from shapely.ops import nearest_points
    from shapely.strtree import STRtree
    HAVE_SHAPELY = True
except Exception:  # pragma: no cover - exercised only without shapely
    HAVE_SHAPELY = False


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p = math.radians
    h = (math.sin(p(lat2 - lat1) / 2) ** 2
         + math.cos(p(lat1)) * math.cos(p(lat2)) * math.sin(p(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(h)))


def _fetch(name: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / name
    if not fp.exists() or fp.stat().st_size == 0:
        with urllib.request.urlopen(NE_BASE + name, timeout=180) as r:
            fp.write_bytes(r.read())
    return json.loads(fp.read_text())


# ---------------------------------------------------------------- polygons
class _Poly:
    """One admin polygon: key + geometry, point-in and distance-to."""

    def __init__(self, key: str, label: str, geom: dict):
        self.key, self.label = key, label
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        self.outer = [p[0] for p in polys if p]
        xs = [x for ring in self.outer for x, _ in ring]
        ys = [y for ring in self.outer for _, y in ring]
        self.bbox = (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)
        self.shp = shape(geom) if HAVE_SHAPELY else None

    def contains(self, lat: float, lon: float) -> bool:
        if self.shp is not None:
            return self.shp.covers(Point(lon, lat))
        x0, y0, x1, y1 = self.bbox
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            return False
        return any(_ray(ring, lon, lat) for ring in self.outer)

    def distance_km(self, lat: float, lon: float) -> float:
        if self.contains(lat, lon):
            return 0.0
        if self.shp is not None:
            q = nearest_points(self.shp, Point(lon, lat))[0]
            return haversine_km(lat, lon, q.y, q.x)
        return min(_seg_km(lat, lon, ring[k], ring[k + 1])
                   for ring in self.outer for k in range(len(ring) - 1))


def _seg_km(lat, lon, a, b) -> float:
    """Point-to-edge distance (not point-to-vertex: a long straight border
    has no vertices near most of its length). Local equirectangular
    projection around the point, then haversine to the closest point."""
    k = math.cos(math.radians(lat))
    ax, ay, bx, by = (a[0] - lon) * k, a[1] - lat, (b[0] - lon) * k, b[1] - lat
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / (dx * dx + dy * dy)))
    return haversine_km(lat, lon, lat + ay + t * dy, lon + (ax + t * dx) / (k or 1e-9))


def _ray(ring, x, y) -> bool:
    inside, j = False, len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


class Boundaries:
    """Country (admin-0) and state/province (admin-1) lookup + name aliases."""

    def __init__(self, admin0: dict | None = None, admin1: dict | None = None):
        a0 = admin0 if admin0 is not None else _fetch(ADMIN0)
        self.countries: dict[str, _Poly] = {}
        self.alias: dict[str, str] = {}
        for f in a0["features"]:
            p = f["properties"]
            iso2 = p.get("ISO_A2_EH") if p.get("ISO_A2") in (None, "-99") else p.get("ISO_A2")
            iso3 = p.get("ADM0_A3") or p.get("ISO_A3")
            key = (iso2 if iso2 and iso2 != "-99" else iso3 or p.get("NAME", "?")).upper()
            if key in self.countries:  # rare duplicates: merge by keeping both shapes
                key = f"{key}#{len(self.countries)}"
            self.countries[key] = _Poly(key, p.get("NAME") or key, f["geometry"])
            for a in (iso2, iso3, p.get("ISO_A3"), p.get("NAME"), p.get("NAME_LONG"), p.get("ADMIN"),
                      p.get("FORMAL_EN"), p.get("NAME_EN"), p.get("SOVEREIGNT"), p.get("ABBREV")):
                if a and a != "-99":
                    self.alias.setdefault(_norm(a), key.split("#")[0])
        for a, k in _EXTRA_COUNTRY_ALIASES.items():
            self.alias.setdefault(_norm(a), k)
        self._admin1_src = admin1
        self._admin1: dict[str, list[_Poly]] | None = None
        self._tree = None

    # ---- countries
    def country_key(self, name) -> str | None:
        return self.alias.get(_norm(name)) if name not in (None, "") else None

    def _polys_for(self, ckey: str) -> list[_Poly]:
        return [p for k, p in self.countries.items() if k.split("#")[0] == ckey]

    def country_at(self, lat: float, lon: float) -> str | None:
        if HAVE_SHAPELY:
            if self._tree is None:
                self._plist = list(self.countries.values())
                self._tree = STRtree([p.shp for p in self._plist])
            for i in self._tree.query(Point(lon, lat)):
                if self._plist[int(i)].contains(lat, lon):
                    return self._plist[int(i)].key.split("#")[0]
            return None
        for p in self.countries.values():
            if p.contains(lat, lon):
                return p.key.split("#")[0]
        return None

    def distance_to_country_km(self, lat: float, lon: float, ckey: str) -> float | None:
        polys = self._polys_for(ckey)
        return min(p.distance_km(lat, lon) for p in polys) if polys else None

    # ---- admin-1 (states / provinces), loaded lazily (40 MB)
    def _load_admin1(self):
        if self._admin1 is None:
            a1 = self._admin1_src if self._admin1_src is not None else _fetch(ADMIN1)
            self._admin1 = defaultdict(list)
            for f in a1["features"]:
                p = f["properties"]
                if not f.get("geometry"):
                    continue
                ckey = (p.get("iso_a2") or "").upper()
                poly = _Poly(f"{ckey}:{p.get('name')}", p.get("name") or "?", f["geometry"])
                for a in (p.get("name"), p.get("name_en"), p.get("postal"), p.get("iso_3166_2"),
                          p.get("name_alt"), p.get("abbrev"), p.get("gn_name"), p.get("woe_name")):
                    for part in str(a or "").split("|"):
                        if part.strip():
                            self._admin1[f"{ckey}|{_norm(part)}"].append(poly)
        return self._admin1

    def distance_to_admin1_km(self, lat, lon, country_key: str, state) -> float | None:
        polys = self._load_admin1().get(f"{country_key.upper()}|{_norm(state)}")
        if not polys:
            return None
        return min(p.distance_km(lat, lon) for p in {id(p): p for p in polys}.values())


_EXTRA_COUNTRY_ALIASES = {
    "US": "US", "United States": "US", "U.S.": "US", "U.S.A.": "US", "USA": "US", "United States of America": "US",
    "UK": "GB", "U.K.": "GB", "Great Britain": "GB", "Britain": "GB", "England": "GB", "Scotland": "GB",
    "Wales": "GB", "Northern Ireland": "GB", "Russia": "RU", "Russian Federation": "RU",
    "South Korea": "KR", "Korea, South": "KR", "Republic of Korea": "KR", "North Korea": "KP",
    "Iran": "IR", "Syria": "SY", "Vietnam": "VN", "Viet Nam": "VN", "Laos": "LA", "Czechia": "CZ",
    "Czech Republic": "CZ", "Turkey": "TR", "Türkiye": "TR", "Ivory Coast": "CI", "Côte d'Ivoire": "CI",
    "Taiwan": "TW", "Bolivia": "BO", "Venezuela": "VE", "Tanzania": "TZ", "Moldova": "MD",
    "Macedonia": "MK", "North Macedonia": "MK", "Burma": "MM", "Myanmar": "MM", "Brunei": "BN",
    "DR Congo": "CD", "Democratic Republic of the Congo": "CD", "Congo (Kinshasa)": "CD",
    "Republic of the Congo": "CG", "Congo (Brazzaville)": "CG", "Eswatini": "SZ", "Swaziland": "SZ",
    "The Netherlands": "NL", "Holland": "NL", "UAE": "AE", "United Arab Emirates": "AE",
}


def _norm(s) -> str:
    return " ".join(str(s).strip().lower().replace(".", " ").replace("_", " ").split())


def _within(d, tol_km: float) -> bool:
    # a distance of 0.0 means INSIDE — it is falsy, so never default it with `or`
    return d is not None and d <= tol_km


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- checks
def _diagnose(bnd: Boundaries, lat, lon, ok) -> str:
    """Name the mechanical error when one explains the contradiction."""
    for label, la, lo in (("swapped_latlon", lon, lat), ("lon_sign_flipped", lat, -lon),
                          ("lat_sign_flipped", -lat, lon), ("both_signs_flipped", -lat, -lon)):
        if -90 <= la <= 90 and -180 <= lo <= 180 and ok(la, lo):
            return label
    return "mismatch"


def check_country(records, bnd: Boundaries, lat="lat", lon="lon", country="country",
                  name="name", tol_km: float = 25.0):
    """Records whose point lies > tol_km outside their claimed country.
    Returns (contradictions, stats). Unknown country names are counted, not judged."""
    out, stats = [], defaultdict(int)
    for i, r in enumerate(records):
        la, lo = _num(r.get(lat)), _num(r.get(lon))
        claim = r.get(country)
        if la is None or lo is None:
            stats["no_coords"] += 1
            continue
        if claim in (None, ""):
            stats["no_claim"] += 1
            continue
        ck = bnd.country_key(claim)
        if ck is None:
            stats["unknown_claim"] += 1
            stats[f"unknown:{claim}"] += 1
            continue
        stats["judged"] += 1
        if not (-90 <= la <= 90 and -180 <= lo <= 180):
            fits = -90 <= lo <= 90 and _within(bnd.distance_to_country_km(lo, la, ck), tol_km)
            out.append({"index": i, "name": r.get(name), "claim": claim, "lat": la, "lon": lo,
                        "km_outside_claim": None, "actually_in": None,
                        "diagnosis": "swapped_latlon" if fits else "out_of_range"})
            continue
        d = bnd.distance_to_country_km(la, lo, ck)
        if d is None or d <= tol_km:
            continue
        diag = _diagnose(bnd, la, lo, lambda a, b: _within(bnd.distance_to_country_km(a, b, ck), tol_km))
        out.append({"index": i, "name": r.get(name), "claim": claim, "lat": la, "lon": lo,
                    "km_outside_claim": round(d, 1), "actually_in": bnd.country_at(la, lo),
                    "diagnosis": diag})
    return out, dict(stats)


def check_admin1(records, bnd: Boundaries, lat="lat", lon="lon", state="state",
                 country: str | None = None, default_country: str = "US",
                 name="name", tol_km: float = 15.0):
    """Records whose point lies > tol_km outside their claimed state/province."""
    out, stats = [], defaultdict(int)
    for i, r in enumerate(records):
        la, lo = _num(r.get(lat)), _num(r.get(lon))
        claim = r.get(state)
        if la is None or lo is None or claim in (None, ""):
            stats["skipped"] += 1
            continue
        ck = bnd.country_key(r.get(country)) if country else default_country
        if not (-90 <= la <= 90 and -180 <= lo <= 180):
            out.append({"index": i, "name": r.get(name), "claim": claim, "lat": la, "lon": lo,
                        "km_outside_claim": None, "actually_in_country": None, "diagnosis": "out_of_range"})
            continue
        d = bnd.distance_to_admin1_km(la, lo, ck or default_country, claim)
        if d is None:
            stats["unknown_claim"] += 1
            stats[f"unknown:{claim}"] += 1
            continue
        stats["judged"] += 1
        if d <= tol_km:
            continue
        diag = _diagnose(bnd, la, lo,
                         lambda a, b: _within(bnd.distance_to_admin1_km(a, b, ck or default_country, claim), tol_km))
        out.append({"index": i, "name": r.get(name), "claim": claim, "lat": la, "lon": lo,
                    "km_outside_claim": round(d, 1), "actually_in_country": bnd.country_at(la, lo),
                    "diagnosis": diag})
    return out, dict(stats)


def site_outliers(records, site="site", lat="lat", lon="lon", name="name",
                  min_records: int = 5, compact_median_km: float = 40.0,
                  floor_km: float = 300.0, p90_multiple: float = 10.0):
    """The generalized nuclear-tests gate: on a COMPACT named site (median
    spread <= compact_median_km, >= min_records), flag records farther than
    max(floor_km, p90_multiple x site p90) from the site's median point.
    Diffuse labels (regions, 'Pacific') are never judged."""
    by = defaultdict(list)
    for i, r in enumerate(records):
        la, lo, s = _num(r.get(lat)), _num(r.get(lon)), r.get(site)
        if la is not None and lo is not None and s not in (None, ""):
            by[s].append((i, la, lo))
    out = []
    for s, pts in by.items():
        if len(pts) < min_records:
            continue
        mlat = statistics.median(p[1] for p in pts)
        mlon = statistics.median(p[2] for p in pts)
        devs = sorted(haversine_km(a, b, mlat, mlon) for _, a, b in pts)
        if statistics.median(devs) > compact_median_km:
            continue
        limit = max(floor_km, p90_multiple * devs[int(0.9 * (len(devs) - 1))])
        for i, a, b in pts:
            d = haversine_km(a, b, mlat, mlon)
            if d > limit:
                out.append({"index": i, "name": records[i].get(name), "site": s, "lat": a, "lon": b,
                            "km_from_site": round(d), "site_center": [round(mlat, 3), round(mlon, 3)]})
    return out


def _dig(doc, path: str):
    for part in [p for p in path.split(".") if p]:
        doc = doc[int(part)] if isinstance(doc, list) else doc[part]
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("--path", default="", help="dotted path to the record list inside the JSON")
    ap.add_argument("--lat", default="lat")
    ap.add_argument("--lon", default="lon")
    ap.add_argument("--country")
    ap.add_argument("--state")
    ap.add_argument("--site")
    ap.add_argument("--name", default="name")
    ap.add_argument("--tol-km", type=float, default=25.0)
    a = ap.parse_args()
    recs = _dig(json.loads(Path(a.file).read_text()), a.path)
    if isinstance(recs, dict) and recs.get("type") == "FeatureCollection":
        recs = [{**(f.get("properties") or {}), "lat": f["geometry"]["coordinates"][1],
                 "lon": f["geometry"]["coordinates"][0]} for f in recs["features"]
                if f.get("geometry", {}).get("type") == "Point"]
    report = {"records": len(recs), "shapely": HAVE_SHAPELY}
    if a.country or a.state:
        bnd = Boundaries()
        if a.country:
            report["country"], report["country_stats"] = check_country(
                recs, bnd, a.lat, a.lon, a.country, a.name, a.tol_km)
        if a.state:
            report["admin1"], report["admin1_stats"] = check_admin1(
                recs, bnd, a.lat, a.lon, a.state, a.country, "US", a.name, min(a.tol_km, 15.0))
    if a.site:
        report["site_outliers"] = site_outliers(recs, a.site, a.lat, a.lon, a.name)
    print(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
