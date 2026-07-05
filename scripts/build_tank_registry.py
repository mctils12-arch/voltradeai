#!/usr/bin/env python3
"""
build_tank_registry.py — one-time (re-runnable) builder for the Cushing
tank registry: tank-fill v2 workup PR-1 (research/open_questions.md,
TANK-FILL % v2 section).

Pulls man_made=storage_tank polygons around Cushing OK from OpenStreetMap
via Overpass, computes per-tank center + equivalent-circle diameter, and
assigns each tank to the nearest strategic tank_farm site (or "ring").
Output: datacore/sentinel2/cushing_tanks.geojson — git-versioned, the
fixed geometry the crescent-shadow estimator (PR-3) samples against.

Heights: OSM has NO height tags here (0 of 333 checked 2026-07-05) — the
registry carries the API-650 default 14.6 m (48 ft) with an explicit
provenance flag; the workup's calibration options (winter shadows, 3DEP
lidar) can overwrite per-tank later. License: ODbL — attribution
"© OpenStreetMap contributors" carried in the file and the manifest.

Usage: python3 scripts/build_tank_registry.py [--out path]
"""

import argparse
import json
import math
import urllib.request
from datetime import datetime, timezone

BBOX = (35.86, -96.84, 36.00, -96.68)  # s, w, n, e — the Cushing ring
OVERPASS = "https://overpass-api.de/api/interpreter"
DEFAULT_HEIGHT_M = 14.6  # API-650 standard 48 ft shell; flagged assumption

SITES = [  # datacore/sites/strategic_sites.json tank_farm entries
    ("cushing_hub", 35.9487, -96.7587),
    ("cushing_enbridge", 35.93, -96.766),
    ("cushing_plains", 35.955, -96.755),
]
SITE_RADIUS_DEG = 0.02  # ~2.2 km assignment radius; beyond -> "ring"


def fetch_tanks():
    q = f"""[out:json][timeout:60];
(way["man_made"="storage_tank"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}););
out geom;"""
    req = urllib.request.Request(OVERPASS, data=q.encode(),
                                 headers={"User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def ring_area_m2(coords):
    """Planar shoelace on locally-projected lon/lat (fine at tank scale)."""
    lat0 = math.radians(sum(c[1] for c in coords) / len(coords))
    mx = 111320.0 * math.cos(lat0)
    my = 110540.0
    pts = [((c[0]) * mx, (c[1]) * my) for c in coords]
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datacore/sentinel2/cushing_tanks.geojson")
    args = ap.parse_args()

    data = fetch_tanks()
    feats = []
    for el in data.get("elements", []):
        geom = el.get("geometry") or []
        if len(geom) < 4:
            continue
        coords = [(g["lon"], g["lat"]) for g in geom]
        area = ring_area_m2(coords)
        if area < 20:  # sub-5m blobs are mapping noise
            continue
        d_m = 2.0 * math.sqrt(area / math.pi)  # equivalent-circle diameter
        clon = sum(c[0] for c in coords) / len(coords)
        clat = sum(c[1] for c in coords) / len(coords)
        site = "ring"
        best = SITE_RADIUS_DEG
        for sid, slat, slon in SITES:
            dd = max(abs(clat - slat), abs(clon - slon))
            if dd <= best:
                best, site = dd, sid
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(clon, 6), round(clat, 6)]},
            "properties": {
                "id": f"osm_way_{el['id']}",
                "diameter_m": round(d_m, 1),
                "area_m2": round(area, 1),
                "site": site,
                "height_m": DEFAULT_HEIGHT_M,
                "height_src": "assumed_api650_48ft",  # provenance flag, per workup
                "measurable_10m": d_m >= 40.0,        # crescent readable at S2 resolution
            },
        })

    feats.sort(key=lambda f: -f["properties"]["diameter_m"])
    out = {
        "type": "FeatureCollection",
        "license": "ODbL — © OpenStreetMap contributors (openstreetmap.org/copyright)",
        "built": datetime.now(timezone.utc).isoformat(),
        "bbox": list(BBOX),
        "source": "Overpass man_made=storage_tank; equivalent-circle diameters from polygon area",
        "height_note": "height_m is the API-650 48ft DEFAULT (no OSM height tags exist here); height_src flags it — calibration (winter shadows / 3DEP lidar) may overwrite per-tank",
        "count": len(feats),
        "measurable_count": sum(1 for f in feats if f["properties"]["measurable_10m"]),
        "features": feats,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    dias = [f["properties"]["diameter_m"] for f in feats]
    dias_sorted = sorted(dias)
    print(f"tanks: {len(feats)} | measurable(>=40m): {out['measurable_count']} | "
          f"median diameter: {dias_sorted[len(dias_sorted)//2] if dias else 0}m | "
          f"max: {max(dias) if dias else 0}m -> {args.out}")


if __name__ == "__main__":
    main()
