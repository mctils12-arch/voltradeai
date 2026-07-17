#!/usr/bin/env python3
"""gridvision_chips.py — GRID VISION Phase B: OSM-seeded NAIP chip-INDEX builder.

Given OSM power features (power=tower nodes, power=substation nodes/polygons)
inside a corridor/bbox, compute the fixed-size NAIP chip FOOTPRINT centered on
each seed and emit a CHIP INDEX: a list of
  {osm_id, lat, lon, cls, chip_bbox:[W,S,E,N], gsd_m, chip_px, stac_query}.
This produces FOOTPRINTS + STAC query params, NOT pixels — streaming the actual
NAIP COG windows is the documented downstream step, done by
scripts/gridvision_naip_stac.py (verified MPC STAC/SAS endpoints).

Charter: research/grid_vision.md; Phase B plan: research/grid_vision_phaseb.md;
compute basis: grid_vision_research.md §4.6-4.7 (TX corridor-verify @0.6 m NAIP
is CPU-feasible, ~380k chips / ~7 h — NOT the whole state).

CORRIDOR-RESTRICTED, per the charter: chips are cut ONLY around seeds inside the
supplied bbox window (a pilot corridor), never the whole state. Because seeds
ARE the towers/substations, the index is inherently corridor-local; a tighter
200-m line-buffer restriction (grid_vision_research.md §4.6) is a downstream
refinement over the same seeds.

OSM-AS-WEAK-LABEL — HONESTY (grid_vision_research.md §2.8; NOT ground truth):
  - RECALL only. OSM presence ~= a reliable positive; OSM ABSENCE proves
    nothing. Precision over OSM-empty land needs a human-sampled pass.
  - power=tower NODES are mapped far less completely than the lines they carry —
    many real corridors have the line way but no tower nodes (recall gaps).
  - Positional offset of meters (imagery-vintage digitizing) — match/score with
    a buffer, never point-IoU.
  - Temporal mismatch: OSM edit date vs NAIP capture date differ by years; new
    lines and demolished corridors both occur. The chip carries the NAIP date
    (via the STAC step) so the mismatch is visible, never hidden.
Seeds are labeled provenance = 'osm-weak-label' so nothing downstream mistakes
them for verified ground truth.

INPUT: either the GeoJSONSeq from scripts/build_power_tiles.sh (RS-delimited,
same as scripts/grid_capacity_tx.py) — its osmium tags-filter includes
n/power=tower as of 2026-07-17, so tower seeds fall out of the normal PBF
pipeline — or a raw Overpass API JSON response (still useful for a quick
small-bbox freshness diff without a full regional PBF download).

  Overpass (session-side): POST https://overpass-api.de/api/interpreter with
    [out:json];(node["power"="tower"]({S},{W},{N},{E});
                node["power"="substation"]({S},{W},{N},{E});
                way["power"="substation"]({S},{W},{N},{E}););out center;

CLI (session-side; stdlib only):
  python3 scripts/gridvision_chips.py power.geojsonseq --gsd 0.6 --out chip_index.json
  python3 scripts/gridvision_chips.py overpass.json --format overpass --bbox -98.1 30.0 -97.5 30.5
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone

# Reuse the verified STAC query builder so chip stac_query params match exactly
# what gridvision_naip_stac.py will POST downstream.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gridvision_naip_stac as naip  # noqa: E402

CHIP_PX = 512               # detector tile edge in pixels (grid_vision_research §4.6)
GSD_DEFAULT = 0.6           # NAIP standard GSD; corridor-verify is CPU-feasible here
DATE_FROM = "2018-01-01"    # NAIP recency window for the STAC query per chip
DATE_TO = naip.NAIP_TEMPORAL_END

# Documented ERCOT/TX pilot corridor window (Austin metro) — VERIFIED to return
# 2022 TX NAIP @0.6 m from MPC this session. A pilot window, NOT the whole state.
TX_PILOT_BBOX = [-98.10, 30.00, -97.50, 30.50]

BASE = os.path.join(os.path.dirname(__file__), "..")

# OSM power tag -> normalized seed class (only in-scope targets seed chips).
_SEED_CLASS = {"tower": "tower", "substation": "substation"}


# ── geometry (pure, unit-tested, no network) ────────────────────────────────

def chip_bbox_for_seed(lon, lat, chip_px=CHIP_PX, gsd_m=GSD_DEFAULT):
    """Fixed-size chip footprint [W,S,E,N] centered on (lon,lat). Edge =
    chip_px * gsd_m meters; degrees via the local meridian/parallel scale
    (same constants as scripts/cdse_chips.py, inverted)."""
    edge_m = chip_px * gsd_m
    half = edge_m / 2.0
    dlat = half / 110_540.0
    dlon = half / (111_320.0 * math.cos(math.radians(lat)))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def in_bbox(lon, lat, bbox):
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def _poly_center(coords):
    """Vertex-mean (lon,lat) of a polygon's outer ring (closed ring aware)."""
    ring = coords[0] if coords and isinstance(coords[0][0], (list, tuple)) else coords
    body = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    n = len(body)
    if not n:
        return None
    return [sum(p[0] for p in body) / n, sum(p[1] for p in body) / n]


def seed_from_feature(feat):
    """A GeoJSON power feature -> {osm_id, lon, lat, cls} or None if it is not
    an in-scope tower/substation or has no usable location."""
    props = feat.get("properties", {}) or {}
    cls = _SEED_CLASS.get(props.get("power"))
    if cls is None:
        return None
    geom = feat.get("geometry", {}) or {}
    gt = (geom.get("type") or "").lower()
    coords = geom.get("coordinates")
    if gt == "point" and coords and len(coords) >= 2:
        lon, lat = float(coords[0]), float(coords[1])
    elif gt in ("polygon", "multipolygon") and coords:
        c = _poly_center(coords[0] if gt == "multipolygon" else coords)
        if not c:
            return None
        lon, lat = c
    else:
        return None
    return {"osm_id": props.get("id") or props.get("@id") or feat.get("id"),
            "lon": lon, "lat": lat, "cls": cls}


# ── parsers (pure, unit-tested) ─────────────────────────────────────────────

def parse_geojsonseq(text):
    """RS/newline-delimited GeoJSON features -> [feature dicts]. Tolerates the
    \\x1e record-separator prefix osmium emits (as scripts/grid_capacity_tx.py)."""
    feats = []
    for row in text.splitlines():
        row = row.lstrip("\x1e").strip()
        if not row:
            continue
        try:
            feats.append(json.loads(row))
        except ValueError:
            continue
    return feats


def parse_overpass(payload):
    """Overpass API JSON ('out center;') -> GeoJSON-like feature dicts. node ->
    Point; way/relation with a 'center' -> Point at the center; way with
    'geometry' -> Polygon. Only power=tower/substation kept."""
    feats = []
    for el in payload.get("elements", []):
        tags = el.get("tags", {}) or {}
        if tags.get("power") not in _SEED_CLASS:
            continue
        et = el.get("type")
        if et == "node" and "lat" in el and "lon" in el:
            geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif "center" in el:
            c = el["center"]
            geom = {"type": "Point", "coordinates": [c["lon"], c["lat"]]}
        elif et == "way" and el.get("geometry"):
            ring = [[p["lon"], p["lat"]] for p in el["geometry"]]
            geom = {"type": "Polygon", "coordinates": [ring]}
        else:
            continue
        feats.append({"type": "Feature", "geometry": geom,
                      "properties": {"power": tags.get("power"),
                                     "id": f"{et}/{el.get('id')}"}})
    return feats


# ── chip index (pure, unit-tested) ──────────────────────────────────────────

def build_chip_index(features, bbox=TX_PILOT_BBOX, chip_px=CHIP_PX, gsd_m=GSD_DEFAULT,
                     date_from=DATE_FROM, date_to=DATE_TO, dedup_grid_m=0.0):
    """features -> chip index dict {meta, chips:[...]} restricted to seeds inside
    `bbox`. Each chip carries the exact STAC query params gridvision_naip_stac
    will POST. Optional dedup_grid_m is GRID-SNAP THINNING: keep one seed per
    occupied grid cell of that size, per class (dense substation yards have many
    OSM nodes). It is snap-to-grid, NOT radius clustering — two seeds closer than
    the grid can still straddle a cell boundary and both survive; that is
    acceptable thinning, not exact dedup. 0 disables (one chip per seed)."""
    chips = []
    seen = set()
    skipped_outside = 0
    by_cls = Counter()
    for feat in features:
        seed = seed_from_feature(feat)
        if seed is None:
            continue
        if not in_bbox(seed["lon"], seed["lat"], bbox):
            skipped_outside += 1
            continue
        lon, lat = seed["lon"], seed["lat"]
        if dedup_grid_m > 0:
            dlat = dedup_grid_m / 110_540.0
            dlon = dedup_grid_m / (111_320.0 * math.cos(math.radians(lat)))
            key = (seed["cls"], round(lon / dlon), round(lat / dlat))
            if key in seen:
                continue
            seen.add(key)
        # round the footprint FIRST, then derive the STAC query from the same
        # rounded bbox so a downstream reader can trust chip_bbox == the query's
        # bbox exactly (no silent precision drift between the two).
        cbbox = [round(v, 7) for v in chip_bbox_for_seed(lon, lat, chip_px, gsd_m)]
        by_cls[seed["cls"]] += 1
        chips.append({
            "osm_id": seed["osm_id"],
            "cls": seed["cls"],
            "provenance": "osm-weak-label",
            "lat": round(lat, 7),
            "lon": round(lon, 7),
            "chip_bbox": cbbox,
            "gsd_m": gsd_m,
            "chip_px": chip_px,
            "stac_query": naip.build_search_body(cbbox, date_from, date_to, limit=10),
        })
    return {
        "meta": {
            "built": datetime.now(timezone.utc).isoformat(),
            "generator": "scripts/gridvision_chips.py",
            "corridor_bbox": bbox,
            "chip_px": chip_px,
            "gsd_m": gsd_m,
            "chip_edge_m": round(chip_px * gsd_m, 1),
            "naip_date_window": [date_from, date_to],
            "n_chips": len(chips),
            "by_class": dict(sorted(by_cls.items())),
            "seeds_skipped_outside_bbox": skipped_outside,
            "provenance": "OSM power=tower/substation seeds (ODbL, weak labels — "
                          "RECALL only; see script header). NAIP imagery public "
                          "domain, streamed per chip via MPC STAC downstream.",
            "license": "OSM ODbL 1.0 (seeds) + NAIP USDA public domain (imagery)",
        },
        "chips": chips,
    }


def load_features(path, fmt):
    with open(path, encoding="utf-8") as f:
        if fmt == "overpass":
            return parse_overpass(json.load(f))
        return parse_geojsonseq(f.read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help="OSM power features (geojsonseq or overpass json)")
    ap.add_argument("--format", choices=["geojsonseq", "overpass"], default="geojsonseq")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                    default=TX_PILOT_BBOX)
    ap.add_argument("--gsd", type=float, default=GSD_DEFAULT, help="NAIP GSD (0.3/0.6/1.0)")
    ap.add_argument("--chip-px", type=int, default=CHIP_PX)
    ap.add_argument("--dedup-grid-m", type=float, default=0.0,
                    help="merge same-class seeds within this grid (m); 0=off")
    ap.add_argument("--from", dest="date_from", default=DATE_FROM)
    ap.add_argument("--to", dest="date_to", default=DATE_TO)
    ap.add_argument("--out", default=None, help="write chip index JSON here")
    args = ap.parse_args(argv)

    feats = load_features(args.input, args.format)
    idx = build_chip_index(feats, args.bbox, args.chip_px, args.gsd,
                           args.date_from, args.date_to, args.dedup_grid_m)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(idx, f, indent=1)
        print(f"wrote {args.out}: {idx['meta']['n_chips']} chips "
              f"{idx['meta']['by_class']} (edge {idx['meta']['chip_edge_m']} m)")
    else:
        print(json.dumps(idx["meta"], indent=1))


if __name__ == "__main__":
    main()
