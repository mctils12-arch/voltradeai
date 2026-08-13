#!/usr/bin/env python3
"""
Submarine telecommunications cables — static reference-geography layer
builder (RAW overlay, T-CLIENT + datacore PRODUCT session, 2026-08-13).

Session-refresh cadence (no scheduler, no GitHub Actions — same pattern as
scripts/military_installations_build.py): re-run this script by hand
whenever the layer needs an OSM refresh.

SOURCE: OpenStreetMap via the public Overpass API, ODbL. NOT NOAA
MarineCadastre — its InPort lineage carries non-federal, copyrighted
contributions (a private trade association, cable-industry vendors, ICPC,
Packet Clearing House) that are NOT public domain, so the "US-federal
therefore public domain" analogy that clears our SEC/EIA/USGS layers does
NOT apply there (research/open_questions.md, 2026-08-11 "SUBMARINE CABLES"
finding). OSM-only is the correct, commercially-clean call.

TAG STRATEGY (verified live against a real Overpass snapshot this session,
per CLAUDE.md READ BEFORE WRITE — do not trust the prior research's exact
counts, they drift as OSM is edited, AND its "+42% from location=underwater"
claim below turned out to be overstated once checked against the right
baseline):
  PRIMARY:  way[seamark:type=cable_submarine] — the S-57/OpenSeaMap
            "submarine cable" object class. Confirmed live 2026-08-13:
            8,722 ways globally (matches the 2026-08-11 probe's count,
            same order of magnitude — OSM is edited continuously so exact
            parity across two days is not expected or required).
            NOT telecom-exclusive: this tag also covers submarine POWER
            interconnectors (e.g. NorNed, Baltic Cable). A way in this set
            that also carries a `power` tag is a power cable, not telecom
            — EXCLUDED (counted as dropped_power, 1,251 ways).
  SECONDARY, checked and NOT activated (`location=underwater` ways lacking
            the primary tag): `location=underwater` ALONE is not cable-
            specific — verified live this session via a real tag sample:
            it is heavily used for submarine PIPELINES (`substance` tag)
            and power cables (`power`/`voltage`), not just telecom. Adding
            the telecom-specific co-tag narrows this correctly but the
            YIELD IS TINY: live-queried this session,
            way[location=underwater][communication][!power] MINUS the
            primary set = **18 ways** globally (0.24% of the primary
            7,471) — confirmed by an exact Overpass count query, not
            estimated. NOT WORTH the added query complexity for v1; the
            prior research's claim of "+42% / 888 ways" from this specific
            combination does not hold up against a same-day live check —
            it likely compared against `submarine=yes`-only as the
            baseline (which itself undercounts: verified live at 7,265
            ways, LESS than the primary 8,722) rather than against the
            primary tag's own already-large coverage. See
            research/open_questions.md for the full correction.
  A SEPARATE, LARGER candidate FOUND BUT NOT ACTIVATED this session:
            way[submarine=yes] ways that do NOT carry seamark:type=
            cable_submarine and are not power-tagged = 1,297 ways
            (confirmed live count). This is not negligible, but this
            session did not get a clean tag sample of that set before
            Overpass access degraded again — shipping it unverified risks
            admitting non-telecom submarine features (the `submarine=yes`
            key is used outside the seamark vocabulary and its uniform
            telecom-specificity was not established here). Filed as a
            scoped, quantified follow-up (not a vague "check it out"):
            confirm ~10 real tag sets from that 1,297-way set, then wire
            it into build_secondary() below.
  EXCLUSION: any matched way carrying a `power` tag is dropped as a power
            cable regardless of which tag matched it.

HONESTY: coverage is real (OSM cable geometry exists on every ocean) but
OSM mapping completeness is heavily skewed to Europe/NE-Atlantic — most
of the world's oceans will render sparse. This is a MAPPING COMPLETENESS
gap, not an assertion that cables don't exist there, and the registry
description says so explicitly (CLAUDE.md PREMIUM EXPERIENCE STANDARD:
never let sparse coverage read as "nothing here").

SCHEMA per feature: GeoJSON LineString/MultiLineString,
  properties: {name?, operator?, osm_id, source_url}
"""
import json
import math
import os
import time
import urllib.request
import urllib.parse
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "client", "public", "cables", "submarine_cables.json")

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
RETRIEVED = date.today().isoformat()

# TeleGeography publicly cites "over 1.5 million km of submarine cable in
# service" as of early 2026 (verified via web search this session — see
# PR description / experiments.md; NOT independently re-derived here, this
# module does not fetch TeleGeography). Kept as a constant so the client
# copy and this script's console report never drift apart.
TELEGEOGRAPHY_TOTAL_KM_CITATION = (
    "TeleGeography's public submarine cable map (resources.telegeography.com), "
    "publicly cited as over 1.5 million km of cable in service, early 2026"
)


def overpass(query, mirrors=OVERPASS_MIRRORS, attempts_per_mirror=3):
    last = None
    for attempt in range(attempts_per_mirror):
        for url in mirrors:
            try:
                data = urllib.parse.urlencode({"data": query}).encode()
                req = urllib.request.Request(
                    url, data=data,
                    headers={"User-Agent": "voltradeai/submarine-cables (reference geography build)"})
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=180) as r:
                    body = r.read().decode()
                return json.loads(body), time.time() - t0, url
            except Exception as e:  # noqa: BLE001
                last = f"{url}: {e}"
                time.sleep(5)
    raise RuntimeError(f"all Overpass mirrors failed after {attempts_per_mirror} rounds: {last}")


def _round_coords(coords, nd=4):
    """~11m precision at the equator — reference geography, not survey
    (matches datacore/boundaries/ne_50m_admin1_lines.json precedent)."""
    return [[round(c[0], nd), round(c[1], nd)] for c in coords]


def way_to_geometry(el):
    geom = el.get("geometry")
    if not geom:
        return None
    coords = [[p["lon"], p["lat"]] for p in geom if "lon" in p and "lat" in p]
    if len(coords) < 2:
        return None
    return {"type": "LineString", "coordinates": _round_coords(coords)}


def way_to_feature(el):
    """Classify one Overpass `way` element into a GeoJSON Feature, or a
    (None, reason) drop — pure, no network, unit-testable (test_submarine_
    cables_build.py). EXCLUSION rule lives here: any `power` tag drops the
    way as a power cable regardless of which query matched it (the seamark
    cable_submarine tag is not telecom-exclusive)."""
    if el.get("type") != "way":
        return None, "not_a_way"
    tags = el.get("tags", {})
    if "power" in tags:
        return None, "power_tagged"
    geometry = way_to_geometry(el)
    if not geometry:
        return None, "no_geometry"
    return {
        "type": "Feature",
        "properties": {
            "name": tags.get("name"),
            "operator": tags.get("operator"),
            "category": tags.get("seamark:cable_submarine:category"),
            "disused": tags.get("disused") == "yes",
            "start_date": tags.get("start_date") or tags.get("opening_date"),
            "wikipedia": tags.get("wikipedia"),
            "osm_id": el["id"],
            "source_url": f"https://www.openstreetmap.org/way/{el['id']}",
        },
        "geometry": geometry,
    }, None


def build_primary():
    """way[seamark:type=cable_submarine], minus anything carrying a power tag."""
    query = (
        "[out:json][timeout:180];"
        'way["seamark:type"="cable_submarine"];'
        "out geom;"
    )
    result, secs, mirror = overpass(query)
    feats = []
    dropped_power = 0
    dropped_no_geom = 0
    for el in result.get("elements", []):
        feat, reason = way_to_feature(el)
        if feat is None:
            if reason == "power_tagged":
                dropped_power += 1
            elif reason == "no_geometry":
                dropped_no_geom += 1
            continue
        feats.append(feat)
    return feats, {
        "count": len(feats),
        "dropped_power_tagged": dropped_power,
        "dropped_no_geometry": dropped_no_geom,
        "query_seconds": round(secs, 1),
        "mirror": mirror,
        "tag": "seamark:type=cable_submarine",
    }


def build_secondary():
    """location=underwater + communication, minus the primary tag and power.

    HONEST STATUS: NOT ACTIVATED. Confirmed live this session via an exact
    Overpass count query (see module docstring): this combination adds only
    18 ways beyond the primary set (0.24% of 7,471) — real, but too small
    to justify the extra query and dedupe complexity for v1. NOT a guess:
    the count is verified, the decision not to wire it is a judgment call
    on yield vs. complexity, logged here instead of silently doing nothing.
    A separate, larger, NOT-YET-TAG-VERIFIED candidate (way[submarine=yes]
    minus primary minus power = 1,297 ways) is a scoped follow-up in
    research/open_questions.md — do not activate it here without first
    confirming a live tag sample is telecom-specific.
    """
    return [], {
        "count": 0,
        "activated": False,
        "reason": ("Verified live this session: location=underwater+communication minus the "
                   "primary tag and power-tagged ways = 18 ways globally (0.24% of primary) — "
                   "confirmed by an exact Overpass count, not estimated, but too small to justify "
                   "wiring in for v1. A larger unverified candidate (submarine=yes minus primary, "
                   "1,297 ways) is filed as a follow-up in research/open_questions.md, not shipped "
                   "without a live tag-quality check."),
    }


def main():
    primary, primary_meta = build_primary()
    secondary, secondary_meta = build_secondary()

    # dedupe by osm_id (secondary, once activated, must not double-count ways
    # the primary query already matched)
    seen = {f["properties"]["osm_id"] for f in primary}
    secondary_unique = [f for f in secondary if f["properties"]["osm_id"] not in seen]

    features = primary + secondary_unique
    total_km = 0.0
    for f in features:
        coords = f["geometry"]["coordinates"]
        for i in range(1, len(coords)):
            lon1, lat1 = coords[i - 1]
            lon2, lat2 = coords[i]
            # equirectangular approximation — fine for a coverage estimate,
            # not for navigation
            dx = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
            dy = math.radians(lat2 - lat1)
            total_km += math.hypot(dx, dy) * 6371.0

    doc = {
        "type": "FeatureCollection",
        "_doc": (
            "Submarine telecommunications cables, OpenStreetMap seamark:type=cable_submarine "
            "(© OpenStreetMap contributors, ODbL) — power-tagged ways excluded. RAW overlay, "
            "no ladder gating. Coverage is real everywhere cables exist but OSM mapping "
            "completeness is heavily skewed to Europe/NE-Atlantic; sparse rendering elsewhere "
            "reflects mapping completeness, not an absence of cables. Compiled by "
            "scripts/submarine_cables_build.py. Global reference for scale: "
            f"{TELEGEOGRAPHY_TOTAL_KM_CITATION}."
        ),
        "retrieved_date": RETRIEVED,
        "attribution": "© OpenStreetMap contributors, ODbL",
        "count": len(features),
        "approx_total_km": round(total_km, 0),
        "sources": {"primary": primary_meta, "secondary": secondary_meta},
        "features": features,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f)
    print(json.dumps({
        "written": OUT,
        "total_features": len(features),
        "approx_total_km": doc["approx_total_km"],
        "primary": primary_meta,
        "secondary": secondary_meta,
        "artifact_bytes": os.path.getsize(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
