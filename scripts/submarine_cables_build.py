#!/usr/bin/env python3
"""
Submarine cables — static reference-geography layer builder.

Filed 2026-08-11 (research/open_questions.md, "Bilawal-derived build
candidates", item 4): OSM's `seamark:type=cable_submarine` tag is the
complete, ODbL-licensed, commercially-clean set. A prior probe undercounted
it with a malformed query and claimed 169,074 km; the corrected live figure
(this build) supersedes that number every time it is re-run — do not
hardcode a coverage percentage anywhere downstream, read it from this
artifact's provenance block.

NOAA MarineCadastre was evaluated and REJECTED: its InPort lineage traces to
NASCA (a private trade association, "(c) 2009 NASCA") plus cable-industry
vendors and state contributions not covered by 17 USC 105 — not the clean
"US-federal public domain" precedent our SEC/EIA/USGS layers rely on.

MANUAL quarterly refresh (no scheduler, no GitHub Actions), same pattern as
scripts/military_installations_build.py. Re-run this script to refresh.

SOURCE: OpenStreetMap `way["seamark:type"="cable_submarine"]`, global, via
Overpass. A single global query reliably times out / connection-resets on
both public mirrors (verified live, this build) — so this script tiles the
world into a lat/lon grid, queries each tile independently with retries
across mirrors, and DEDUPES by OSM way id (a cable spanning a tile boundary
is returned by every tile it touches).

HONESTY ON GEOMETRY: OSM's raw node-by-node geometry is kept for LENGTH
(haversine over the full-resolution polyline — the number the provenance
block reports) but a simplified polyline (Douglas-Peucker, ~1.5 km
tolerance) is what ships in the artifact for rendering. This is a
disclosed simplification for map display, not a claim of higher precision
than the source; length is always computed before simplification.

SCHEMA per feature: id, name, category(telecom|power|mixed|other|
unclassified), disused(bool), length_km, geometry(GeoJSON LineString,
simplified), source_url, tags_source(dict of the raw cable tags kept for
the detail panel: operator/landing names where OSM records them).
"""
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "datacore", "submarine_cables.json")

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
RETRIEVED = date.today().isoformat()
USER_AGENT = "voltradeai/submarine-cables (reference geography build)"

# TeleGeography's publicly cited "~1.5 million km of active submarine cable"
# figure is the industry benchmark used in the filed research to express our
# OSM coverage as a percentage. It is NOT a dataset we ingest — a single
# widely-cited number, restated here so the coverage % in the provenance
# block has a documented denominator.
TELEGEOGRAPHY_BENCHMARK_KM = 1_500_000

# World grid: tiled to keep each Overpass call small enough to avoid the
# connection resets / timeouts a single global query produces (verified
# live, 2026-08-12). Bands are NOT uniform — denser in the Atlantic/
# Mediterranean where research/open_questions.md's scan found ~83% of the
# geometry concentrates, so a tile there doesn't try to carry too much.
TILES = [
    # (lamin, lomin, lamax, lomax)
    (24, -85, 72, -30),   # N Atlantic west (US/Canada east coast -> mid-Atlantic)
    (24, -30, 72, 15),    # N Atlantic east (mid-Atlantic -> W Europe)
    (30, 15, 72, 45),     # N Europe / Baltic / Black Sea
    (-10, -30, 30, 15),   # tropical Atlantic / W Africa
    (-40, -75, -10, -30), # S Atlantic west (S America east coast)
    (-40, -30, -10, 20),  # S Atlantic east (S Africa west coast)
    (-40, 20, 5, 55),     # Indian Ocean west / E Africa / Red Sea
    (5, 40, 40, 80),      # Persian Gulf / Arabian Sea / India
    (-40, 80, 5, 145),    # Indian Ocean east / SE Asia west
    (5, 90, 45, 145),     # East/SE Asia mainland + coast
    (-50, 110, 5, 180),   # Australia / Indonesia / Pacific west
    (5, 100, 55, 180),    # China / Japan / Korea / N Pacific west
    (-60, -180, 5, -110), # S Pacific / W South America
    (5, -180, 65, -110),  # N Pacific east (US/Canada west coast, Alaska)
    (-60, -110, 5, -30),  # S Pacific east / Antarctic-adjacent Atlantic gap
    (-90, -180, -40, 180),  # Southern Ocean / Antarctica margins (sparse)
    (65, -180, 90, 180),     # Arctic (sparse)
]


def overpass(query, mirrors=OVERPASS_MIRRORS, tries_per_mirror=2):
    last = None
    for url in mirrors:
        for attempt in range(tries_per_mirror):
            try:
                data = urllib.parse.urlencode({"data": query}).encode()
                req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read().decode()
                return json.loads(body), time.time() - t0, url
            except Exception as e:  # noqa: BLE001
                last = f"{url} (attempt {attempt + 1}): {e}"
                time.sleep(6)
    raise RuntimeError(f"all Overpass mirrors failed: {last}")


def fetch_tile(bbox):
    lamin, lomin, lamax, lomax = bbox
    query = (
        f"[out:json][timeout:90];"
        f'way["seamark:type"="cable_submarine"]({lamin},{lomin},{lamax},{lomax});'
        f"out tags geom;"
    )
    result, secs, mirror = overpass(query)
    return result.get("elements", []), secs, mirror


# ── per-tile disk cache + auto-split-on-failure ──
#
# A single global Overpass query reliably connection-resets/times out
# (verified live 2026-08-12), and even a moderate tile can fail on a busy
# mirror. Two resilience layers, both aimed at making the quarterly manual
# re-run NOT require babysitting or a from-scratch retry after one bad tile:
#   1. Every successfully-fetched tile is cached to disk by its bbox, so a
#      killed/interrupted run resumes instantly past tiles already done.
#   2. A tile that fails outright is SPLIT in half along its longer
#      dimension and each half retried independently (recursively, up to
#      MAX_SPLIT_DEPTH) rather than the whole region being silently
#      dropped — a smaller box is a materially easier Overpass query, not
#      just a retry of the same one.
CACHE_DIR = os.path.join("/tmp", "voltradeai_submarine_cables_cache")
MAX_SPLIT_DEPTH = 3
MIN_SPLIT_DEG = 4.0  # stop splitting once a side is already this small


def _cache_path(bbox):
    import hashlib
    key = hashlib.sha1(",".join(f"{v:.4f}" for v in bbox).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{key}.json")


def _fetch_tile_cached(bbox):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(bbox)
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
        return cached["elements"], 0.0, "cache"
    elements, secs, mirror = fetch_tile(bbox)
    with open(path, "w") as f:
        json.dump({"bbox": bbox, "elements": elements}, f)
    return elements, secs, mirror


def fetch_region(bbox, depth=0):
    """Fetch one region, auto-splitting on failure. Returns
    (elements, meta_list) — meta_list has one entry per leaf tile actually
    queried (post-split), each recording success/failure/cache-hit."""
    try:
        elements, secs, mirror = _fetch_tile_cached(bbox)
        return elements, [{"bbox": bbox, "depth": depth, "elements": len(elements),
                           "seconds": round(secs, 1), "mirror": mirror}]
    except Exception as e:  # noqa: BLE001
        lamin, lomin, lamax, lomax = bbox
        lat_span, lon_span = lamax - lamin, lomax - lomin
        can_split = depth < MAX_SPLIT_DEPTH and max(lat_span, lon_span) > MIN_SPLIT_DEG
        if not can_split:
            return [], [{"bbox": bbox, "depth": depth, "error": str(e)[:200]}]
        print(f"  split {bbox} (depth {depth}) after failure: {str(e)[:120]}", file=sys.stderr)
        if lat_span >= lon_span:
            mid = (lamin + lamax) / 2
            a, b = (lamin, lomin, mid, lomax), (mid, lomin, lamax, lomax)
        else:
            mid = (lomin + lomax) / 2
            a, b = (lamin, lomin, lamax, mid), (lamin, mid, lamax, lomax)
        els_a, meta_a = fetch_region(a, depth + 1)
        time.sleep(2)
        els_b, meta_b = fetch_region(b, depth + 1)
        return els_a + els_b, meta_a + meta_b


# ── haversine length (full-resolution, computed BEFORE any simplification) ──

def haversine_km(a, b):
    R = 6371.0088
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))


def line_length_km(coords):
    return sum(haversine_km(coords[i], coords[i + 1]) for i in range(len(coords) - 1))


# ── Douglas-Peucker simplification (render-only; length uses raw coords) ──

def _perp_dist(pt, a, b):
    # planar approximation (degrees) — fine at simplification tolerances of
    # ~0.01-0.03 deg (~1-3 km); this is a display simplification, not a
    # navigational figure.
    if a == b:
        return math.hypot(pt[0] - a[0], pt[1] - a[1])
    x, y = pt
    x1, y1 = a
    x2, y2 = b
    num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / den if den else math.hypot(pt[0] - a[0], pt[1] - a[1])


def douglas_peucker(coords, epsilon_deg):
    if len(coords) < 3:
        return coords[:]
    dmax, idx = 0.0, 0
    for i in range(1, len(coords) - 1):
        d = _perp_dist(coords[i], coords[0], coords[-1])
        if d > dmax:
            dmax, idx = d, i
    if dmax > epsilon_deg:
        left = douglas_peucker(coords[: idx + 1], epsilon_deg)
        right = douglas_peucker(coords[idx:], epsilon_deg)
        return left[:-1] + right
    return [coords[0], coords[-1]]


SIMPLIFY_EPSILON_DEG = 0.015  # ~1.5 km at the equator


def classify_category(tags):
    """Telecom vs power vs mixed vs other, from documented OSM tags only."""
    cat = str(tags.get("seamark:cable_submarine:category") or "").lower()
    disused = cat == "disused" or tags.get("disused") == "yes" or tags.get("seamark:type") == "cable_submarine:disused"
    has_power = "power" in cat or "transmission" in cat or tags.get("power") is not None
    has_telecom = any(k in cat for k in ("fib", "optical", "communication", "telephone", "telefon", "telegraph")) \
        or tags.get("communication") is not None
    if has_power and has_telecom:
        kind = "mixed"
    elif has_power:
        kind = "power"
    elif has_telecom:
        kind = "telecom"
    elif cat == "mooring":
        kind = "other"
    else:
        kind = "unclassified"
    return kind, disused


def build():
    all_by_id = {}
    tile_meta = []
    failed_leaves = []
    for i, bbox in enumerate(TILES):
        elements, leaf_meta = fetch_region(bbox)
        new = 0
        for el in elements:
            if el.get("type") != "way" or el.get("id") in all_by_id:
                continue
            all_by_id[el["id"]] = el
            new += 1
        leaves_failed = [m for m in leaf_meta if "error" in m]
        failed_leaves.extend(leaves_failed)
        tile_meta.append({"bbox": bbox, "elements": len(elements), "new": new,
                          "leaves": leaf_meta, "leaves_failed": len(leaves_failed)})
        status = "OK" if not leaves_failed else f"{len(leaves_failed)} leaf sub-tile(s) still failed after splitting"
        print(f"tile {i + 1}/{len(TILES)} {bbox}: {len(elements)} ways ({new} new) — {status}", file=sys.stderr)
        time.sleep(3)  # be polite between calls — shared public mirrors

    features = []
    total_km = 0.0
    counts = {"telecom": 0, "power": 0, "mixed": 0, "other": 0, "unclassified": 0}
    disused_count = 0
    dropped_no_geom = 0
    for el in all_by_id.values():
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            dropped_no_geom += 1
            continue
        coords = [[p["lon"], p["lat"]] for p in geom]
        length_km = line_length_km(coords)
        total_km += length_km
        simplified = douglas_peucker(coords, SIMPLIFY_EPSILON_DEG)
        simplified = [[round(c[0], 4), round(c[1], 4)] for c in simplified]
        tags = el.get("tags", {})
        kind, disused = classify_category(tags)
        counts[kind] += 1
        if disused:
            disused_count += 1
        name = tags.get("seamark:cable_submarine:name") or tags.get("name")
        features.append({
            "id": el["id"],
            "name": name,
            "category": kind,
            "disused": disused,
            "length_km": round(length_km, 1),
            "geometry": {"type": "LineString", "coordinates": simplified},
            "source_url": f"https://www.openstreetmap.org/way/{el['id']}",
            "wikidata": tags.get("wikidata"),
            "operator_source": tags.get("source"),
        })

    coverage_pct = round(100 * total_km / TELEGEOGRAPHY_BENCHMARK_KM, 1)
    unclassified_pct = round(100 * counts["unclassified"] / len(features), 1) if features else 0.0
    doc = {
        "provenance": {
            "banner": (
                f"OpenStreetMap-catalogued submarine cable routes ({len(features):,} way segments, "
                f"~{round(total_km):,} km union). Cable digitization on OSM is heavily concentrated in "
                f"Europe / the NE Atlantic (~83% of geometry per the filed research scan) — Asia, "
                f"Africa, South America and the Pacific will look sparse here relative to their real "
                f"cable count; that is a mapping-coverage gap in the source data, not a claim those "
                f"regions have few cables. {counts['unclassified']:,} segments ({unclassified_pct}%) carry "
                f"no cable-type tag at all in the source (the largest single block is Finnish Väylä-sourced "
                f"Baltic geometry, tagged submarine=yes but with no telecom/power category) — shown as "
                f"'unclassified', never guessed into a category. Route lines are as digitized in OSM "
                f"(waypoints, not exact landing-to-landing survey paths); geometry shown here is simplified "
                f"for rendering (Douglas-Peucker, ~1.5 km tolerance) — length_km per feature is computed "
                f"from the full-resolution source before simplification."
            ),
            "attribution": "© OpenStreetMap contributors (ODbL)",
            "license": "OpenStreetMap ODbL — attribution required, commercial use permitted",
            "retrieved_date": RETRIEVED,
            "source_tag": 'way["seamark:type"="cable_submarine"]',
            "total_length_km": round(total_km, 1),
            "telegeography_benchmark_km": TELEGEOGRAPHY_BENCHMARK_KM,
            "coverage_pct_of_telegeography_benchmark": coverage_pct,
            "category_counts": counts,
            "disused_count": disused_count,
            "dropped_no_geometry": dropped_no_geom,
            "simplification": {"method": "douglas_peucker", "epsilon_deg": SIMPLIFY_EPSILON_DEG,
                               "note": "render-only; length_km uses full-resolution source coordinates"},
            "tiles": tile_meta,
            "fetch_gaps": {
                "failed_leaf_regions": len(failed_leaves),
                "note": "regions where every Overpass mirror failed even after auto-splitting the query "
                        "smaller — cables in these boxes are ABSENT from this artifact, not counted as zero. "
                        "Re-run this script to retry (cached tiles are skipped; only failed regions refetch).",
                "regions": failed_leaves,
            } if failed_leaves else {"failed_leaf_regions": 0},
            "rejected_source": {
                "name": "NOAA MarineCadastre",
                "reason": "InPort lineage traces to NASCA (private trade association, © 2009 NASCA) "
                          "plus cable-industry vendors and state contributions not covered by 17 USC 105 "
                          "— not clean public domain like our SEC/EIA/USGS federal layers.",
            },
        },
        "count": len(features),
        "cables": features,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f)
    print(json.dumps({
        "written": OUT,
        "count": len(features),
        "total_length_km": round(total_km, 1),
        "coverage_pct": coverage_pct,
        "category_counts": counts,
        "disused_count": disused_count,
        "dropped_no_geometry": dropped_no_geom,
        "failed_leaf_regions": len(failed_leaves),
        "artifact_bytes": os.path.getsize(OUT),
    }, indent=2))


if __name__ == "__main__":
    build()
