#!/usr/bin/env python3
"""Compile the slim admin-1 (state/province) boundary-lines artifact.

Natural Earth 1:50m admin_1_states_provinces_lines (PUBLIC DOMAIN) →
datacore/boundaries/ne_50m_admin1_lines.json, mirroring the admin-0
compile: properties stripped to the fields the map/popup actually uses,
coordinates rounded to 4dp (~11 m — far inside 1:50m generalization
error), so the artifact stays small enough to import into the server
bundle like ne_110m_admin0.json.

The 2026-07-04 admin-0 compile was ad-hoc (no script survived); this one
is kept per the compile-knowledge rule so refreshes are a re-run, not a
re-derivation.

Usage: python3 scripts/boundaries_admin1_build.py [source.geojson]
  (downloads from the nvkelso/natural-earth-vector mirror if no path given)
"""
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

SRC_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_50m_admin_1_states_provinces_lines.geojson")
OUT = Path(__file__).resolve().parent.parent / "datacore" / "boundaries" / "ne_50m_admin1_lines.json"


def round_coords(c, dp=4):
    if isinstance(c, (int, float)):
        return round(c, dp)
    return [round_coords(x, dp) for x in c]


def main() -> None:
    if len(sys.argv) > 1:
        raw = json.loads(Path(sys.argv[1]).read_text())
    else:
        with urllib.request.urlopen(SRC_URL, timeout=120) as r:
            raw = json.load(r)
    feats = []
    for f in raw["features"]:
        p = f.get("properties") or {}
        g = f.get("geometry")
        if not g:  # drop-not-infer: a boundary line with no geometry is nothing
            continue
        feats.append({
            "type": "Feature",
            "properties": {
                # NAME is usually null on line segments; adm0 is what the
                # status note / any popup would say. min_zoom kept so the
                # client can honor Natural Earth's own declutter ranking.
                "adm0": p.get("ADM0_NAME"),
                "iso3": p.get("ADM0_A3"),
                "min_zoom": p.get("MIN_ZOOM"),
            },
            "geometry": {"type": g["type"], "coordinates": round_coords(g["coordinates"])},
        })
    doc = {
        "type": "FeatureCollection",
        "_doc": (f"Natural Earth 1:50m admin-1 state/province boundary lines (PUBLIC DOMAIN). "
                 f"Compiled {date.today().isoformat()} from nvkelso/natural-earth-vector; properties "
                 "stripped to adm0+iso3+min_zoom, coords rounded 4dp. Generalized — reference layer, "
                 "not survey-grade; disputed-boundary depictions follow Natural Earth's de-facto policy."),
        "features": feats,
    }
    OUT.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"{len(feats)} features -> {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
