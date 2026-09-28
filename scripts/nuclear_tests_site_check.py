#!/usr/bin/env python3
"""Site-consistency gate for datacore/nuclear_tests.json.

The original import gate (2026-07-11) only rejected IMPOSSIBLE coordinates
(out of range, or null island). It could not catch coordinates that are
valid but contradict the record's OWN site field — and the upstream catalog
mirror has several: EGMONT (UK, site NTS) sat at 36.0,-112.0 beside the
Grand Canyon; MADISON (USA, site NTS) sat at Novaya Zemlya, 7,600 km away.
A human spotted EGMONT on the map on 2026-09-28.

Rule, calibrated on the dataset itself:
  * a site is a COMPACT test range when the median distance of its records
    from the site's median point is <= COMPACT_MEDIAN_KM. Regional labels
    (the Soviet peaceful-explosion programs: KRASNO RUSS, TYUMEN RUSS,
    KAZAKH, ...) genuinely spread over hundreds of km and are never judged.
  * on a compact site, a record is a contradiction when it lies more than
    max(FLOOR_KM, P90_MULTIPLE x the site's 90th-percentile spread) from
    the site's median point.

Nothing is invented. A contradicted record is re-plotted at its recorded
site's median point, marked loc="site", and keeps the catalog's original
coordinates in src_lat/src_lon so the card can say exactly what happened.
The site field is corroborated by the rest of the record (country, name,
date) and by every other test at that site; the precise shot point is
unknown and is labelled as such.

Usage:
  python3 scripts/nuclear_tests_site_check.py            # report only
  python3 scripts/nuclear_tests_site_check.py --apply    # rewrite the JSON
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "datacore" / "nuclear_tests.json"

MIN_SITE_RECORDS = 5
COMPACT_MEDIAN_KM = 40.0
FLOOR_KM = 300.0
P90_MULTIPLE = 10.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p = math.radians
    h = (math.sin(p(lat2 - lat1) / 2) ** 2
         + math.cos(p(lat1)) * math.cos(p(lat2)) * math.sin(p(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def _coords(rec: dict) -> tuple[float, float]:
    # a record already re-plotted keeps its catalog coordinates in src_*;
    # judging must always use those, so re-running is idempotent
    return (rec.get("src_lat", rec["lat"]), rec.get("src_lon", rec["lon"]))


def site_profiles(tests: list[dict]) -> dict[str, dict]:
    """Per-site median point and spread, for sites with enough records."""
    by_site: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rec in tests:
        by_site[rec["r"]].append(_coords(rec))
    out = {}
    for site, pts in by_site.items():
        if len(pts) < MIN_SITE_RECORDS:
            continue
        mlat = statistics.median(p[0] for p in pts)
        mlon = statistics.median(p[1] for p in pts)
        devs = sorted(haversine_km(a, b, mlat, mlon) for a, b in pts)
        p90 = devs[int(0.9 * (len(devs) - 1))]
        median_dev = statistics.median(devs)
        out[site] = {
            "n": len(pts), "lat": mlat, "lon": mlon,
            "median_dev_km": median_dev, "p90_km": p90,
            "compact": median_dev <= COMPACT_MEDIAN_KM,
            "limit_km": max(FLOOR_KM, P90_MULTIPLE * p90),
        }
    return out


def find_contradictions(tests: list[dict]) -> list[dict]:
    """Records on a compact site whose coordinates contradict that site."""
    profiles = site_profiles(tests)
    found = []
    for i, rec in enumerate(tests):
        if rec.get("loc") == "site":
            continue  # already resolved by a previous --apply
        prof = profiles.get(rec["r"])
        if not prof or not prof["compact"]:
            continue
        lat, lon = _coords(rec)
        dist = haversine_km(lat, lon, prof["lat"], prof["lon"])
        if dist > prof["limit_km"]:
            found.append({"index": i, "name": rec.get("n"), "country": rec.get("c"),
                          "date": rec.get("d"), "site": rec["r"],
                          "src_lat": lat, "src_lon": lon, "dist_km": round(dist),
                          "site_lat": round(prof["lat"], 3), "site_lon": round(prof["lon"], 3)})
    return found


def apply_fixes(tests: list[dict], found: list[dict]) -> None:
    for f in found:
        rec = tests[f["index"]]
        rec["src_lat"], rec["src_lon"] = f["src_lat"], f["src_lon"]
        rec["lat"], rec["lon"] = f["site_lat"], f["site_lon"]
        rec["loc"] = "site"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="rewrite the dataset in place")
    args = ap.parse_args()
    doc = json.loads(DATA.read_text())
    tests = doc["tests"]
    found = find_contradictions(tests)
    for f in found:
        print(f"{f['name']:<14} {f['country']:<7} {f['date']}  site={f['site']:<12} "
              f"catalog=({f['src_lat']}, {f['src_lon']})  {f['dist_km']:,} km from site "
              f"-> ({f['site_lat']}, {f['site_lon']})")
    print(f"{len(found)} site-contradicting record(s)")
    if args.apply and found:
        apply_fixes(tests, found)
        doc["_site_gate"] = {
            "rule": (f"compact site = median spread <= {COMPACT_MEDIAN_KM:g} km (>= {MIN_SITE_RECORDS} records); "
                     f"contradiction = > max({FLOOR_KM:g} km, {P90_MULTIPLE:g} x site p90) from the site median"),
            "action": "re-plotted at the recorded site's median point, loc='site'; catalog coordinates kept in src_lat/src_lon",
            "script": "scripts/nuclear_tests_site_check.py",
            "records": [f["name"] for f in found],
        }
        # compact, no trailing newline: byte-identical to the committed file
        # for every untouched record, so the diff shows only the fixes
        DATA.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
        print(f"applied to {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
