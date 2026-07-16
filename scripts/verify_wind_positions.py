#!/usr/bin/env python3
"""verify_wind_positions.py — ROOT VALIDATION LADDER gate 1 (DATA):
cross-check GPPD/EIA-reported US wind-plant coordinates against
OpenStreetMap wind-turbine ground truth.

TRIGGER: human directive (2026-07-15/16, EARTH TWIN round 10 feedback) —
verify wind-farm map positions against a real ground-truth source;
Waverly Wind Farm LLC (KS) named as the reference case. A prior session's
background worktree agent for this task was reclaimed before landing any
artifact (see research/earth_twin_program.md RESUME STATE, 2026-07-16) —
this is a fresh build, not a resume.

SOURCE OF TRUTH: OpenStreetMap via Overpass API (ODbL) — wind turbines
tagged `power=generator` + `generator:source=wind` (nodes), or
`power=plant` + `plant:source=wind` (ways/relations, farm boundary with a
centroid). Compared against datacore/powerplants/us_power_plants.json
(WRI GPPD + EIA-860), filtered to fuel == "wind".

METHOD (bbox-binned, not per-plant, to respect Overpass fair use and cut
~1100 plants down to ~130-280 requests): plants are grouped into a
coarse lat/lon grid; one Overpass bbox query per POPULATED bin fetches
every wind turbine in that bin PLUS a padding margin wider than
SEARCH_RADIUS_KM, so every plant's true nearest-turbine distance is
computable from its own bin's single query result — the padding, not the
bin size, is what guarantees correctness (see pad_deg()).

CLASSIFICATION per plant (pure, no network — see classify()):
  verified    nearest OSM turbine <= VERIFIED_KM (point sits inside the
              farm's own turbine footprint)
  approximate turbine(s) exist within SEARCH_RADIUS_KM but farther than
              VERIFIED_KM (farm is real and roughly located, registry
              point likely an admin/substation offset or coordinate drift)
  unverified  no OSM turbine within SEARCH_RADIUS_KM
  error       the bin's Overpass query failed (network) — NOT a finding;
              kept in its own bucket so it can never be misread as a
              real negative result

HONESTY: OSM coverage is incomplete, especially for smaller/older farms —
"unverified" means "OSM has no turbine near this point today", never
"this plant doesn't exist" or "the registry coordinate is wrong". This
script produces a RAW cross-reference finding, not a corrected coordinate;
nothing here overwrites us_power_plants.json.

Re-run (session-side; ~1 Overpass request per populated bin, rate-limited):
    python3 scripts/verify_wind_positions.py [--out PATH] [--fresh]
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "datacore", "powerplants", "us_power_plants.json")
DST = os.path.join(HERE, "..", "datacore", "powerplants", "wind_position_verification.json")
OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "voltradeai-datacore/1.0 (research@voltradeai.com)"

BIN_DEG = 2.0             # coarse grid cell (deg); populated bins only get queried
SEARCH_RADIUS_KM = 15.0   # beyond this, treat as "no OSM turbine nearby"
VERIFIED_KM = 3.0         # nearest turbine within this -> point is inside the farm footprint
REQUEST_SLEEP_S = 3.0     # 2026-07-16 session run hit HTTP 429 on ~10% of bins at 1.2s; widened
MAX_RETRIES = 4
KM_PER_DEG_LAT = 111.0


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bin_key(lat, lon, bin_deg=BIN_DEG):
    return (math.floor(lat / bin_deg) * bin_deg, math.floor(lon / bin_deg) * bin_deg)


def pad_deg(bin_lat0, bin_lat1, radius_km=SEARCH_RADIUS_KM, margin=1.3):
    """Degrees of padding beyond a bin's own edges that guarantee any point
    within the bin has its full radius_km neighborhood covered. margin>1
    is slack against the flat-latitude longitude approximation below."""
    pad_lat = (radius_km * margin) / KM_PER_DEG_LAT
    center_lat = (bin_lat0 + bin_lat1) / 2.0
    cos_lat = max(math.cos(math.radians(center_lat)), 0.15)  # guard near-pole degenerate case
    pad_lon = (radius_km * margin) / (KM_PER_DEG_LAT * cos_lat)
    return pad_lat, pad_lon


def build_bbox_query(south, west, north, east):
    return (
        f"[out:json][timeout:60];"
        f"(node[\"power\"=\"generator\"][\"generator:source\"=\"wind\"]({south},{west},{north},{east});"
        f"way[\"power\"=\"plant\"][\"plant:source\"=\"wind\"]({south},{west},{north},{east});"
        f"relation[\"power\"=\"plant\"][\"plant:source\"=\"wind\"]({south},{west},{north},{east});"
        f");out center;"
    )


def query_overpass(query_text):
    req = urllib.request.Request(OVERPASS, data=query_text.encode(),
                                  headers={"User-Agent": USER_AGENT})
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            time.sleep(REQUEST_SLEEP_S * (attempt + 1) * 2)
    raise RuntimeError(f"overpass failed after {MAX_RETRIES} attempts: {last_err}")


def turbine_points(payload):
    """Overpass JSON -> [(lat, lon), ...]. Pure, no network."""
    pts = []
    for el in payload.get("elements", []):
        lat = el.get("lat", (el.get("center") or {}).get("lat"))
        lon = el.get("lon", (el.get("center") or {}).get("lon"))
        if lat is not None and lon is not None:
            pts.append((lat, lon))
    return pts


def classify(plant_lat, plant_lon, points):
    """Pure, no network. points = [(lat, lon), ...] already filtered to
    the relevant search neighborhood (the caller guarantees this via
    bbox padding — this function does not itself enforce SEARCH_RADIUS_KM,
    so callers must not pass points from outside the padded query)."""
    if not points:
        return "unverified", None, 0, None
    dists = [haversine_km(plant_lat, plant_lon, lat, lon) for lat, lon in points]
    nearest = min(dists)
    if nearest > SEARCH_RADIUS_KM:
        return "unverified", None, 0, None
    in_range = [(lat, lon) for (lat, lon), d in zip(points, dists) if d <= SEARCH_RADIUS_KM]
    centroid = [sum(p[0] for p in in_range) / len(in_range),
                sum(p[1] for p in in_range) / len(in_range)]
    status = "verified" if nearest <= VERIFIED_KM else "approximate"
    return status, round(nearest, 3), len(in_range), [round(c, 4) for c in centroid]


def load_wind_plants():
    data = json.load(open(SRC, encoding="utf-8"))
    return [p for p in data["plants"] if p[2] == "wind"]


def group_bins(plants):
    bins = {}
    for p in plants:
        lat, lon = p[4], p[5]
        bins.setdefault(bin_key(lat, lon), []).append(p)
    return bins


def bin_is_resolved(keys, results):
    """A bin only counts as done on resume if every member key is present
    AND none of them are stuck in "error" — a prior Overpass failure is
    not a finding and must be retried, never silently left unresolved."""
    return all(k in results and results[k]["status"] != "error" for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DST)
    ap.add_argument("--fresh", action="store_true", help="ignore any existing --out, start clean")
    ap.add_argument("--max-bins", type=int, default=None, help="cap populated bins processed (debug only)")
    args = ap.parse_args()

    plants = load_wind_plants()
    bins = group_bins(plants)
    total_wind = len(plants)
    total_bins = len(bins)

    done = {}
    if not args.fresh and os.path.exists(args.out):
        prior = json.load(open(args.out, encoding="utf-8"))
        for r in prior.get("results", []):
            done[r["key"]] = r
    results = dict(done)

    bin_items = list(bins.items())
    if args.max_bins:
        bin_items = bin_items[: args.max_bins]

    for bi, ((blat, blon), members) in enumerate(bin_items):
        keys = [f"{m[0]}|{m[4]}|{m[5]}" for m in members]
        if bin_is_resolved(keys, results):
            continue  # whole bin already resolved (resume path); "error" bins retry
        pad_lat, pad_lon = pad_deg(blat, blat + BIN_DEG)
        south, north = blat - pad_lat, blat + BIN_DEG + pad_lat
        west, east = blon - pad_lon, blon + BIN_DEG + pad_lon
        try:
            payload = query_overpass(build_bbox_query(south, west, north, east))
            pts = turbine_points(payload)
            err = None
        except Exception as e:
            pts, err = None, str(e)

        for m, key in zip(members, keys):
            name, mw, fuel, owner, lat, lon, _reg_verified = m
            if err is not None:
                status, nearest_km, count, centroid = "error", None, 0, None
            else:
                status, nearest_km, count, centroid = classify(lat, lon, pts)
            results[key] = {
                "key": key, "name": name, "mw": mw, "owner": owner,
                "lat": lat, "lon": lon, "status": status,
                "nearest_turbine_km": nearest_km,
                "osm_turbines_within_radius": count,
                "osm_centroid": centroid,
            }
        print(f"[bin {bi+1}/{len(bin_items)}] ({blat},{blon}) {len(members)} plant(s) "
              f"-> {'ERROR: ' + err if err else str(len(pts)) + ' turbines'}", file=sys.stderr)

        # checkpoint every bin — a long run must never lose completed work to a crash
        _write(args.out, results, total_wind, total_bins)
        time.sleep(REQUEST_SLEEP_S)

    _write(args.out, results, total_wind, total_bins)
    counts = {}
    for r in results.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\n{len(results)}/{total_wind} wind plants processed across {total_bins} bins -> {args.out}")
    print(counts)


def _write(out_path, results_by_key, total_wind, total_bins):
    ordered = sorted(results_by_key.values(), key=lambda r: -r["mw"])
    counts = {}
    for r in ordered:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    out = {
        "_doc": (
            "Gate-1 DATA validation (ROOT VALIDATION LADDER): US wind-plant "
            "coordinates from datacore/powerplants/us_power_plants.json (WRI GPPD "
            "+ EIA-860) cross-checked against OpenStreetMap wind-turbine ground "
            "truth (power=generator[generator:source=wind] nodes + "
            "power=plant[plant:source=wind] ways/relations) via Overpass API. "
            f"status: verified (nearest OSM turbine <= {VERIFIED_KM}km) / "
            f"approximate (turbine(s) found within {SEARCH_RADIUS_KM}km but "
            "farther) / unverified (no OSM turbine within the search radius — "
            "NOT proof the plant is missing or mis-registered; OSM coverage is "
            "incomplete, especially for smaller/older farms) / error (the bin's "
            "Overpass query failed — not a finding). Raw cross-reference only; "
            "does not modify us_power_plants.json. Built by "
            "scripts/verify_wind_positions.py."
        ),
        "method": {
            "source_turbines": "OpenStreetMap via Overpass API (ODbL)",
            "search_radius_km": SEARCH_RADIUS_KM,
            "verified_threshold_km": VERIFIED_KM,
            "query_strategy": "bbox-binned (2deg grid cells, padded beyond search_radius_km)",
        },
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_wind_plants_in_source": total_wind,
        "total_bins": total_bins,
        "processed": len(ordered),
        "coverage_note": (
            f"{len(ordered)}/{total_wind} wind plants processed"
            + ("" if len(ordered) >= total_wind else " — INCOMPLETE RUN, remaining "
               "plants absent from results (not counted in any status); re-run "
               "without --fresh to resume")
        ),
        "counts": counts,
        "results": ordered,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
