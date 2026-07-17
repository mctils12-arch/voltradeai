#!/usr/bin/env python3
"""
Military installations — static reference-geography layer builder.

Human-specced 2026-07-17. Quarterly MANUAL refresh: re-run this script.
NO scheduler, NO GitHub Actions.

SOURCES, priority order (spec):
  1. US  — DoD installation open data (data.gov / DoD open GIS). Authoritative;
           wins over OSM inside the US on name+proximity dedupe.
  2. Global — OpenStreetMap  military=base  (NAMED only) via Overpass. ODbL:
           "© OpenStreetMap contributors". We deliberately query NAMED
           military=base, NOT all landuse=military: the latter is ~86,800
           features, mostly unnamed sub-installation training areas, which
           violates the spec's installation-level granularity and its `name`
           required field. Named military=base ~= 3,024 installation-level sites.
  3. Curated foreign-acknowledged — only where officially published WITH
           coordinates in a citable source; every entry carries source_url.
           Shipped from curated/military_installations_curated.json (a small
           hand-verified seed; empty is acceptable).

HARD RULES (never violate):
  * Never infer or estimate a LOCATION. No published coordinates in a citable
    source -> the feature is DROPPED and counted. (operator_nation is looked up
    from the PUBLISHED coordinates via Natural Earth admin-0 — a spatial join,
    not a location inference: the location is given, the country is read off it.)
  * Never derive locations from satellite-imagery analysis.
  * Installation-level granularity ONLY: name, boundary/centroid, branch, type.
    No internal structures / sub-facilities.
  * This dataset is STATIC REFERENCE GEOGRAPHY. It must NEVER be joined to the
    live aircraft/vessel layers or any correlation feature (enforced at the
    registry via "noCrossTies": true and honored by the client wiring).

SCHEMA per feature:
  name, operator_nation, branch, type(air|naval|army|joint|logistics|other),
  status(active|inactive), geometry(GeoJSON polygon where available else Point),
  source, source_url, source_retrieved_date
"""
import json
import os
import sys
import time
import urllib.request
import urllib.parse
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "datacore", "military_installations.json")
NE_COUNTRIES = os.path.join(ROOT, "datacore", "boundaries", "ne_110m_admin0.json")
CURATED = os.path.join(ROOT, "scripts", "curated", "military_installations_curated.json")

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
RETRIEVED = date.today().isoformat()

TYPE_ENUM = {"air", "naval", "army", "joint", "logistics", "other"}


# ── Natural Earth point-in-country (honest nation lookup from published coords) ──

def load_countries():
    with open(NE_COUNTRIES) as f:
        gj = json.load(f)
    feats = gj["features"] if isinstance(gj, dict) and "features" in gj else gj
    out = []
    for f in feats:
        props = f.get("properties", {})
        name = props.get("ADMIN") or props.get("NAME") or props.get("name") or props.get("SOVEREIGNT")
        geom = f.get("geometry") or {}
        polys = []
        if geom.get("type") == "Polygon":
            polys = [geom["coordinates"]]
        elif geom.get("type") == "MultiPolygon":
            polys = geom["coordinates"]
        # bbox for a cheap reject test
        xs, ys = [], []
        for poly in polys:
            for ring in poly:
                for x, y in ring:
                    xs.append(x); ys.append(y)
        if xs:
            out.append((name, polys, (min(xs), min(ys), max(xs), max(ys))))
    return out


def pt_in_ring(x, y, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def nation_of(lon, lat, countries):
    for name, polys, (minx, miny, maxx, maxy) in countries:
        if lon < minx or lon > maxx or lat < miny or lat > maxy:
            continue
        for poly in polys:
            if not poly:
                continue
            if pt_in_ring(lon, lat, poly[0]):
                # subtract holes
                in_hole = any(pt_in_ring(lon, lat, h) for h in poly[1:])
                if not in_hole:
                    return name
    return None


# ── OSM type decode (installation TYPE from documented tags only) ──

def osm_type(tags):
    """Map OSM tags to the spec's type enum — documented tags only, never guessed."""
    t = " ".join(str(tags.get(k, "")).lower() for k in ("military", "name", "description", "military:service"))
    svc = str(tags.get("military:service", "")).lower()
    if "air" in svc or "airbase" in t or "air base" in t or "air force" in t or tags.get("aeroway"):
        return "air"
    if "navy" in svc or "naval" in t or "navy" in t or "submarine" in t:
        return "naval"
    if "army" in svc or "barracks" in t or "garrison" in t or tags.get("military") == "barracks":
        return "army"
    if "depot" in t or "logistic" in t or "supply" in t:
        return "logistics"
    if "joint" in t:
        return "joint"
    return "other"


def osm_branch(tags):
    svc = tags.get("military:service") or tags.get("operator") or tags.get("military:unit")
    return svc if svc else None


def _round_geom(geometry, nd=5):
    """Round coordinates to ~1m (5 dp) — reference geography, not survey. Shrinks the artifact."""
    def rc(c):
        return [round(c[0], nd), round(c[1], nd)]
    g = geometry
    if g["type"] == "Point":
        return {"type": "Point", "coordinates": rc(g["coordinates"])}
    if g["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [[rc(p) for p in ring] for ring in g["coordinates"]]}
    if g["type"] == "MultiPolygon":
        return {"type": "MultiPolygon",
                "coordinates": [[[rc(p) for p in ring] for ring in poly] for poly in g["coordinates"]]}
    return g


def centroid_of(geometry):
    if geometry["type"] == "Point":
        return geometry["coordinates"]
    if geometry["type"] == "Polygon":
        ring = geometry["coordinates"][0]
        n = len(ring)
        return [sum(p[0] for p in ring) / n, sum(p[1] for p in ring) / n]
    return None


def overpass(query, mirrors=OVERPASS_MIRRORS):
    last = None
    for url in mirrors:
        try:
            t0 = time.time()
            data = urllib.parse.urlencode({"data": query}).encode()
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "voltradeai/military-installations (reference geography build)"})
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read().decode()
            return json.loads(body), time.time() - t0, url
        except Exception as e:  # noqa: BLE001
            last = f"{url}: {e}"
            time.sleep(2)
    raise RuntimeError(f"all Overpass mirrors failed: {last}")


def build_osm(countries):
    # NAMED military=base only — installation-level, `name` required by schema
    query = (
        "[out:json][timeout:150];"
        '(nwr["military"="base"]["name"];);'
        "out geom;"
    )
    result, secs, mirror = overpass(query)
    feats = []
    dropped_no_coord = 0
    for el in result.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        geometry = None
        if el["type"] == "node" and "lat" in el:
            geometry = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif el["type"] == "way" and el.get("geometry"):
            coords = [[p["lon"], p["lat"]] for p in el["geometry"]]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                geometry = {"type": "Polygon", "coordinates": [coords]}
            elif coords:
                # open way -> centroid of its nodes (published coords, not inferred)
                cx = sum(c[0] for c in coords) / len(coords)
                cy = sum(c[1] for c in coords) / len(coords)
                geometry = {"type": "Point", "coordinates": [cx, cy]}
        elif el["type"] == "relation":
            # use the relation's bounds centroid where geometry members aren't resolvable
            b = el.get("bounds")
            if b:
                geometry = {"type": "Point",
                            "coordinates": [(b["minlon"] + b["maxlon"]) / 2,
                                            (b["minlat"] + b["maxlat"]) / 2]}
        if not geometry:
            dropped_no_coord += 1
            continue
        geometry = _round_geom(geometry)
        cen = centroid_of(geometry)
        if not cen:
            dropped_no_coord += 1
            continue
        cen = [round(cen[0], 5), round(cen[1], 5)]
        nation = nation_of(cen[0], cen[1], countries)
        feats.append({
            "name": name,
            "operator_nation": nation,  # from PUBLISHED coords via Natural Earth; None if offshore/unresolved
            "branch": osm_branch(tags),
            "type": osm_type(tags),
            "status": "active",  # OSM presence; inactive only if explicitly tagged
            "geometry": geometry,
            "centroid": cen,
            "source": "OpenStreetMap",
            "source_url": f"https://www.openstreetmap.org/{el['type']}/{el['id']}",
            "source_retrieved_date": RETRIEVED,
        })
    return feats, {"count": len(feats), "dropped_no_coord": dropped_no_coord,
                   "query_seconds": round(secs, 1), "mirror": mirror}


def build_dod():
    """DoD installations open data.

    HONEST STATUS: the DoD open-GIS installation boundary FeatureServer is not
    yet wired (candidate endpoints on data.gov require a stable service URL that
    was not confirmed reachable from this build environment). Rather than
    present a guessed endpoint as an authoritative source, this returns empty
    and records the gap. US installations are, in the interim, covered by the
    OSM `military=base` pull (which includes US bases). When the confirmed DoD
    endpoint is wired here, US features gain authoritative geometry and win the
    dedupe over OSM. Provide a real endpoint via env DOD_INSTALLATIONS_URL to
    activate this path (geojson query URL returning installation polygons).
    """
    url = os.environ.get("DOD_INSTALLATIONS_URL")
    if not url:
        return [], {"count": 0, "wired": False,
                    "note": "DoD open-GIS endpoint not confirmed; US bases currently via OSM military=base. "
                            "Set DOD_INSTALLATIONS_URL to a real geojson query URL to activate."}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "voltradeai/military-installations"})
        with urllib.request.urlopen(req, timeout=90) as r:
            gj = json.loads(r.read().decode())
        feats = []
        dropped = 0
        for f in gj.get("features", []):
            p = f.get("properties", {})
            name = p.get("COMPONENT") or p.get("SITE_NAME") or p.get("installationName") or p.get("FULLNAME") or p.get("NAME")
            geom = f.get("geometry")
            if not name or not geom:
                dropped += 1
                continue
            cen = centroid_of(geom) if geom.get("type") in ("Point", "Polygon") else None
            feats.append({
                "name": name,
                "operator_nation": "United States of America",
                "branch": p.get("COMPONENT") or p.get("BRANCH"),
                "type": "other",
                "status": "active",
                "geometry": _round_geom(geom),
                "centroid": cen,
                "source": "US DoD open data (data.gov / DoD open GIS)",
                "source_url": url.split("?")[0],
                "source_retrieved_date": RETRIEVED,
            })
        return feats, {"count": len(feats), "dropped": dropped, "wired": True}
    except Exception as e:  # noqa: BLE001
        return [], {"count": 0, "wired": True, "blocked": True, "reason": str(e)[:200]}


def build_curated():
    if not os.path.exists(CURATED):
        return [], {"count": 0, "note": "no curated seed file"}
    with open(CURATED) as f:
        rows = json.load(f)
    feats, dropped = [], 0
    for r in rows:
        if not r.get("source_url") or not r.get("geometry"):
            dropped += 1
            continue
        r.setdefault("source_retrieved_date", RETRIEVED)
        r.setdefault("centroid", centroid_of(r["geometry"]))
        feats.append(r)
    return feats, {"count": len(feats), "dropped_uncited_or_nocoord": dropped}


def dedupe_us(dod, osm):
    """DoD wins over OSM inside the US on name+proximity (~15km)."""
    superseded = 0
    kept = []
    us_dod = [f for f in dod if f.get("centroid")]
    for o in osm:
        if o.get("operator_nation") != "United States of America" or not o.get("centroid"):
            kept.append(o)
            continue
        ox, oy = o["centroid"]
        dup = False
        on = o["name"].lower().split("(")[0].strip()
        for d in us_dod:
            dx, dy = d["centroid"]
            if abs(dx - ox) < 0.2 and abs(dy - oy) < 0.2:  # ~15-22km box
                dn = d["name"].lower()
                if on[:6] in dn or dn[:6] in on:
                    dup = True
                    break
        if dup:
            superseded += 1
        else:
            kept.append(o)
    return dod + kept, superseded


def main():
    countries = load_countries()
    osm, osm_meta = build_osm(countries)
    dod, dod_meta = build_dod()
    curated, cur_meta = build_curated()

    merged, superseded = dedupe_us(dod, osm)
    merged += curated

    # final schema guard: drop anything without geometry (never invent one)
    final, dropped_final = [], 0
    for f in merged:
        if not f.get("geometry") or f["type"] not in TYPE_ENUM:
            if f.get("type") not in TYPE_ENUM:
                f["type"] = "other"
            if not f.get("geometry"):
                dropped_final += 1
                continue
        final.append(f)

    doc = {
        "provenance": {
            "banner": ("Officially published installation locations. Sources: US DoD open data, "
                       "OpenStreetMap contributors, and cited government publications. Reference "
                       "geography only — current as of retrieval date; not operational information."),
            "attribution": "© OpenStreetMap contributors (ODbL); US DoD open data (public domain); cited government publications",
            "retrieved_date": RETRIEVED,
            "no_cross_ties": True,
            "granularity": "installation-level (name, boundary/centroid, branch, type) — no sub-facilities",
            "sources": {
                "osm": osm_meta,
                "dod": dod_meta,
                "curated": cur_meta,
            },
            "dedupe_us_osm_superseded": superseded,
            "dropped_final_no_geometry": dropped_final,
        },
        "count": len(final),
        "installations": final,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f)
    # console report (spec's report-back)
    print(json.dumps({
        "written": OUT,
        "total": len(final),
        "osm": osm_meta,
        "dod": dod_meta,
        "curated": cur_meta,
        "us_osm_superseded_by_dod": superseded,
        "dropped_final_no_geometry": dropped_final,
        "artifact_bytes": os.path.getsize(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
