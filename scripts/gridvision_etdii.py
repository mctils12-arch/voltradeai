#!/usr/bin/env python3
"""gridvision_etdii.py — GRID VISION Phase B training-label prep: download +
parse the CC-BY overhead tower/substation label datasets into ONE normalized
record schema, and emit datacore/gridvision/labels_manifest.json.

Charter: research/grid_vision.md; Phase B plan: research/grid_vision_phaseb.md;
research inputs: research/grid_vision_research.md Items 1, 2, 3.1. Phase B
starts with VERIFY, so this script only PREPARES labels — no GPU, no training.

WHAT IS AND ISN'T A USABLE LABEL (honesty, per grid_vision_research.md 2.8/3.5):
the license-clean detector stack is Duke-US (CC BY 4.0) + OSM (ODbL, weak) +
NAIP (public domain). Honest Phase B scope = transmission TOWERS + SUBSTATIONS
only; distribution poles/towers are <=10 px at NAIP GSD and are NOT in scope.

SOURCES (all probed live 2026-07-07 via the figshare v2 API through the agent
proxy — file lists, sizes, licenses, and the ETDII annotation schema VERIFIED;
per-class US counts VERIFIED by parsing the two US zips this session):

  ETDII (figshare 14935434, "Electric Transmission Infrastructure Satellite
      Imagery Dataset for Computer Vision", DOI 10.6084/m9.figshare.14935434.v2,
      CC BY 4.0, published 2021-07-21). 264 km2, 113 images @0.30 m, 7 cities /
      2 countries. US portion (our target): USA_AZ_Tucson (26 img, USGS ortho)
      + USA_KS_Colwich_Maize (48 img, USGS ortho). Labels (Documentation.pdf,
      read this session): Tower (T), Other Tower (OT, uncertain), Line (L),
      Edge Node (EN), Substation (SS). Geometry in WGS84; every feature also
      carries pixel_coordinates (a pixel-space quad) usable directly as a
      detection box. This is the GridTracer deposit (Huang et al. 2022).

  Duke E T&D (figshare 6931088, "Electric Transmission and Distribution
      Infrastructure Imagery Dataset", CC BY 4.0). 28.1 GB, 16 zips, 511
      images, 6 countries. Richer 7-class schema (DT/DL/TT/TL/OT/OL/SS) that
      SPLITS transmission vs distribution. US zips add Hartford CT, Clyde NC,
      Wilmington NC (0.15 m) beyond ETDII's two. NOT downloaded here (28 GB) —
      file list + license VERIFIED via API; per-class counts REPORTED from its
      Documentation, never tallied. CAVEAT: its 3 SpaceNet cities rest on
      CC BY-SA imagery — use US zips only for a clean-license set.

  PLAD (github.com/andreluizbvs/PLAD) — named in the task; EVALUATED + REJECTED.
      grid_vision_research.md 2.4 VERIFIED LICENSE = GPL-3.0 (copyleft, unusable
      for a commercial model) AND the domain is wrong (133 drone close-ups of
      tower hardware — insulators/dampers — no transfer to nadir satellite).
      Re-probe of the GitHub API this session returned empty (the proxy blocks
      unauthenticated github.com API/asset reads — flagged, not papered over),
      so PLAD's license here is REPORTED from prior verified research, not
      re-verified today. It is recorded as rejected, never entered as a label.

NORMALIZED RECORD (one per annotation):
  {source, license, image_ref, raw_label, cls, in_scope, geom_type,
   pixel_bbox:[xmin,ymin,xmax,ymax] | None, geo_centroid:[lon,lat] | None,
   country, place, us_domain}
`cls` in {tower, substation, tower_uncertain, tower_distribution, line,
edge_node, unknown}; `in_scope` True ONLY for tower + substation (the honest
NAIP-GSD target). OT is kept SEPARATE (the dataset itself flags it uncertain) —
never silently folded into clean towers.

CLI (session-side; stdlib only; NO network needed for --write-manifest):
  python3 scripts/gridvision_etdii.py --sources          # documented table
  python3 scripts/gridvision_etdii.py --write-manifest    # emit labels_manifest.json
  python3 scripts/gridvision_etdii.py --tally-zip USA_AZ_Tucson.zip --source ETDII
  # FULL FETCH (documented, run session-side, ~220 MB for the two US zips):
  #   for fid in 28887792 28887795; do \
  #     curl -L "https://ndownloader.figshare.com/files/$fid" -o etdii_$fid.zip; done
  #   python3 scripts/gridvision_etdii.py --tally-zip etdii_28887792.zip --source ETDII
"""
import argparse
import io
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

BASE = os.path.join(os.path.dirname(__file__), "..")
MANIFEST = os.path.join(BASE, "datacore", "gridvision", "labels_manifest.json")
FIGSHARE_API = "https://api.figshare.com/v2/articles"

# ── normalization (pure, unit-tested) ───────────────────────────────────────
# raw label (ETDII 5-class AND Duke 7-class schemas) -> (cls, in_scope)
# in_scope = usable as a transmission-tower-or-substation label at NAIP GSD.
LABEL_MAP = {
    # transmission towers — the primary in-scope target
    "T": ("tower", True),          # ETDII "Tower"
    "TT": ("tower", True),         # Duke "Transmission Tower"
    "SS": ("substation", True),    # both schemas "Substation"
    # uncertain / out-of-scope, recorded honestly, never counted as clean towers
    "OT": ("tower_uncertain", False),   # "Other Tower" — dataset flags uncertainty
    "DT": ("tower_distribution", False),  # distribution tower — <=10 px at NAIP GSD
    "L": ("line", False),
    "TL": ("line", False),
    "DL": ("line", False),
    "OL": ("line", False),
    "EN": ("edge_node", False),    # where a line exits the image edge
}


def normalize_label(raw):
    """raw ETDII/Duke label code -> (cls, in_scope). Unknown codes map to
    ('unknown', False) — never guessed into a real class."""
    key = ("" if raw is None else str(raw)).strip().upper()
    return LABEL_MAP.get(key, ("unknown", False))


def pixel_bbox(pixel_coordinates):
    """A dataset pixel-quad -> [xmin, ymin, xmax, ymax] (floats). Returns None
    for missing/empty/degenerate input — a chip builder must skip, not guess."""
    if not pixel_coordinates:
        return None
    xs, ys = [], []
    for pt in pixel_coordinates:
        if not pt or len(pt) < 2:
            continue
        try:
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
        except (TypeError, ValueError):
            continue
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _iter_vertices(coords, geom_type):
    """Yield (lon, lat) vertices from a GeoJSON geometry's coordinates."""
    gt = (geom_type or "").lower()
    if gt == "point":
        if coords and len(coords) >= 2:
            yield float(coords[0]), float(coords[1])
    elif gt == "linestring":
        for pt in coords or []:
            if pt and len(pt) >= 2:
                yield float(pt[0]), float(pt[1])
    elif gt == "polygon":
        for ring in coords or []:
            # ring is closed (last == first): drop the duplicate closing vertex
            body = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
            for pt in body:
                if pt and len(pt) >= 2:
                    yield float(pt[0]), float(pt[1])
    elif gt == "multipolygon":
        for poly in coords or []:
            for pt in _iter_vertices(poly, "polygon"):
                yield pt


def geo_centroid(coords, geom_type):
    """Vertex-mean [lon, lat] of a geometry (a stable seed point for chipping).
    Deliberately a simple vertex mean, not a true area centroid — the seed only
    needs to land inside the feature, and it must be deterministic. None if the
    geometry has no usable vertices."""
    pts = list(_iter_vertices(coords, geom_type))
    if not pts:
        return None
    n = len(pts)
    return [sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n]


def parse_geojson(gj, source, license_id):
    """One ETDII/Duke per-image GeoJSON dict -> list of normalized records."""
    out = []
    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}
        raw = props.get("label")
        cls, in_scope = normalize_label(raw)
        country = props.get("country")
        out.append({
            "source": source,
            "license": license_id,
            "image_ref": props.get("filename"),
            "raw_label": raw,
            "cls": cls,
            "in_scope": in_scope,
            "geom_type": geom.get("type"),
            "pixel_bbox": pixel_bbox(props.get("pixel_coordinates")),
            "geo_centroid": geo_centroid(geom.get("coordinates"), geom.get("type")),
            "country": country,
            # ETDII uses the key "city/state"; Duke uses "city"
            "place": props.get("city/state") or props.get("city"),
            "us_domain": (str(country).strip().upper() == "USA"),
        })
    return out


def summarize(records):
    """records -> composition summary: per-class counts split US vs non-US, and
    the in-scope tower/substation balance the charter tracks."""
    by_cls = Counter()
    by_cls_us = Counter()
    imgs = set()
    for r in records:
        by_cls[r["cls"]] += 1
        if r["us_domain"]:
            by_cls_us[r["cls"]] += 1
        if r.get("image_ref"):
            imgs.add((r["source"], r["image_ref"]))
    us_tower = by_cls_us.get("tower", 0)
    us_sub = by_cls_us.get("substation", 0)
    return {
        "n_records": len(records),
        "n_images": len(imgs),
        "by_class": dict(sorted(by_cls.items())),
        "by_class_us_only": dict(sorted(by_cls_us.items())),
        "in_scope_total": {
            "tower": by_cls.get("tower", 0),
            "substation": by_cls.get("substation", 0),
        },
        "in_scope_us": {"tower": us_tower, "substation": us_sub},
        "us_substation_scarcity_flag": us_sub < 0.05 * max(us_tower, 1),
    }


# ── source descriptors + this-session VERIFIED counts (baked, reproducible) ──
# Counts below were COMPUTED this session by parsing the actual figshare zips
# (--tally-zip); they are real, not estimates. Re-derive any of them with the
# documented FULL FETCH + --tally-zip. Numbers NOT computed this session are
# marked "reported": true and cite their documentation, never invented.
VERIFIED_ETDII_US = {  # tallied 2026-07-07 from the two US zips (all geojsons)
    "USA_AZ_Tucson":       {"images": 26, "T": 595, "OT": 23, "L": 251, "EN": 145, "SS": 3},
    "USA_KS_Colwich_Maize": {"images": 48, "T": 813, "OT": 102, "L": 533, "EN": 126, "SS": 3},
}
VERIFIED_ETDII_NZ_SAMPLE = {  # one NZ city tallied as an annotation-format probe
    "NZ_Dunedin": {"images": 6, "T": 334, "OT": 28, "L": 95, "EN": 60, "SS": 2},
}


def _sum_counts(d, keys):
    return {k: sum(city.get(k, 0) for city in d.values()) for k in keys}


def build_manifest():
    """Assemble labels_manifest.json from the documented source descriptors +
    the baked this-session verified counts. Pure (no network)."""
    us = _sum_counts(VERIFIED_ETDII_US, ["images", "T", "OT", "L", "EN", "SS"])
    etdii = {
        "name": "ETDII — Electric Transmission Infrastructure Satellite Imagery Dataset for Computer Vision",
        "figshare_article": 14935434,
        "doi": "10.6084/m9.figshare.14935434.v2",
        "url": "https://doi.org/10.6084/m9.figshare.14935434",
        "figshare_api": f"{FIGSHARE_API}/14935434",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Nair, Pathirathna, You, Han, Huang, Yang, Streltsov, "
                       "Bradbury, Collins, Malof, Hu (Duke Energy Initiative). "
                       "Imagery: USGS (public domain) / LINZ (CC BY 4.0).",
        "gsd_m": 0.30,
        "verification": "VERIFIED 2026-07-07: figshare API file list + license "
                        "+ Documentation.pdf legend read; US per-class counts "
                        "tallied by parsing both US zips this session.",
        "us_portion": {
            "cities": list(VERIFIED_ETDII_US.keys()),
            "images": us["images"],
            "class_counts_raw": {"T": us["T"], "OT": us["OT"], "L": us["L"],
                                 "EN": us["EN"], "SS": us["SS"]},
            "normalized_in_scope": {"tower": us["T"], "substation": us["SS"]},
            "per_city": VERIFIED_ETDII_US,
            "reported": False,
        },
        "nonus_portion": {
            "cities_documented": ["NZ_Dunedin", "NZ_Gisborne", "NZ_Palmerston_North",
                                  "NZ_Rotorua", "NZ_Tauranga"],
            "images_documented": 39,  # Documentation.pdf: 6+6+14+7+6
            "sample_tallied": VERIFIED_ETDII_NZ_SAMPLE,
            "note": "only NZ_Dunedin tallied this session (format probe); other "
                    "NZ cities' per-class counts REPORTED from Documentation, "
                    "not tallied.",
            "reported": True,
        },
        "annotation_format": "per-image .geojson (WGS84 geom + pixel_coordinates "
                             "quad) + .csv (per-vertex) + multiclass masks "
                             "(.npz/.png). Labels: T/OT/L/EN/SS.",
        "fitness": "PRIMARY in-scope label set for the first fine-tune. US "
                   "portion is @0.30 m and license-clean (USGS PD imagery).",
    }
    duke = {
        "name": "Duke Electric Transmission and Distribution Infrastructure Imagery Dataset",
        "figshare_article": 6931088,
        "doi": "10.6084/m9.figshare.6931088",
        "url": "https://doi.org/10.6084/m9.figshare.6931088",
        "figshare_api": f"{FIGSHARE_API}/6931088",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Bradbury, Han, Nair, Pathirathna, You et al. (Duke "
                       "Data+). US imagery USGS/CT-ECO public domain.",
        "size_gb": 28.1,
        "verification": "VERIFIED 2026-07-07: figshare API file list (16 zips) "
                        "+ license. NOT downloaded (28 GB); per-class counts "
                        "REPORTED from research/grid_vision_research.md 2.1.",
        "classes_reported": ["DT", "DL", "TT", "TL", "OT", "OL", "SS"],
        "us_cities_reported": ["Hartford CT (0.15m)", "Clyde NC (0.15m)",
                               "Wilmington NC (0.15m)", "Colwich+Maize KS (0.15m)",
                               "Tucson AZ (0.30m)"],
        "spacetime_caveat": "3 SpaceNet cities (Khartoum/Shanghai/Rio) rest on "
                            "CC BY-SA imagery — use US zips only for clean license.",
        "reported": True,
        "fitness": "SUPERSET with transmission/distribution split (TT vs DT). "
                   "Adds 0.15 m US cities beyond ETDII. Pull the US zips "
                   "(Hartford/Clyde/Wilmington) to grow the US set; downsample "
                   "0.15 m -> 0.30/0.60 m to match NAIP GSD.",
    }
    plad = {
        "name": "PLAD / STN PLAD",
        "url": "https://github.com/andreluizbvs/PLAD",
        "license": "GPL-3.0 (REPORTED from grid_vision_research.md 2.4 — the "
                   "GitHub API was proxy-blocked this session, not re-verified)",
        "verification": "EVALUATED + REJECTED. Domain wrong (133 drone close-ups "
                        "of tower hardware, not nadir satellite) AND copyleft "
                        "license. Recorded to close the question; NOT a label "
                        "source.",
        "usable": False,
    }
    osm = {
        "name": "OSM power features (weak labels / corridor seeds)",
        "url": "https://www.openstreetmap.org (Overpass power=tower/substation)",
        "license": "ODbL 1.0 — attribution + database share-alike",
        "role": "NOT a ground-truth label set. Seeds chip locations (see "
                "scripts/gridvision_chips.py) and measures RECALL on corridors "
                "(precision needs a human-sampled pass). Weak: tower nodes "
                "mapped far less completely than the lines; meters of positional "
                "offset; temporal mismatch vs NAIP capture date.",
        "usable": "weak-label / seed only",
    }
    us = VERIFIED_ETDII_US
    total_us_tower = sum(c["T"] for c in us.values())
    total_us_sub = sum(c["SS"] for c in us.values())
    return {
        "built": datetime.now(timezone.utc).isoformat(),
        "purpose": "GRID VISION Phase B training-label manifest. IP-clean stack "
                   "only (Duke-US CC BY, NAIP public domain, OSM ODbL-attributed). "
                   "Honest scope: transmission towers + substations.",
        "generator": "scripts/gridvision_etdii.py --write-manifest",
        "sources": [etdii, duke, plad, osm],
        "composition_headline": {
            "clean_us_labels_available_now": {
                "source": "ETDII US (verified this session)",
                "tower": total_us_tower,
                "substation": total_us_sub,
            },
            "substation_underrepresentation": (
                f"CONFIRMED: only {total_us_sub} substation polygons across all "
                f"{sum(c['images'] for c in us.values())} US ETDII images vs "
                f"{total_us_tower} towers — substations are ~{total_us_sub/total_us_tower*100:.1f}% "
                "of the in-scope US labels. The charter's underrepresentation "
                "flag is real; substations need Duke-US zips + OSM-seeded "
                "self-bootstrapping to reach a trainable count."
            ),
            "us_naip_domain_note": "ETDII US imagery is USGS ortho @0.30 m, not "
                                   "NAIP; treat as US-domain-adjacent and augment "
                                   "with OSM-over-NAIP weak labels for true NAIP "
                                   "radiometry (grid_vision_research.md 2.10).",
        },
    }


# ── network (thin; everything above is testable without it) ─────────────────

def http_json(url, timeout=60):
    import urllib.request
    req = urllib.request.Request(
        url, headers={"User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def figshare_files(article_id):
    """Live figshare metadata -> (license_name, [file dicts])."""
    d = http_json(f"{FIGSHARE_API}/{article_id}")
    lic = (d.get("license") or {}).get("name")
    return lic, d.get("files", [])


def tally_zip(zip_path, source, license_id):
    """Parse every per-image .geojson inside a downloaded dataset zip ->
    (records, summary). Session-side only (zips are big binaries)."""
    records = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".geojson"):
                try:
                    gj = json.loads(z.read(name))
                except (ValueError, KeyError):
                    continue
                records.extend(parse_geojson(gj, source, license_id))
    return records, summarize(records)


def write_manifest(path=MANIFEST):
    doc = build_manifest()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sources", action="store_true",
                    help="print the documented source table and exit")
    ap.add_argument("--write-manifest", action="store_true",
                    help="write datacore/gridvision/labels_manifest.json (no network)")
    ap.add_argument("--tally-zip", metavar="ZIP",
                    help="parse a downloaded dataset zip and print composition")
    ap.add_argument("--source", default="ETDII", help="source tag for --tally-zip")
    ap.add_argument("--license", default="CC BY 4.0", help="license for --tally-zip")
    args = ap.parse_args(argv)

    if args.tally_zip:
        records, summ = tally_zip(args.tally_zip, args.source, args.license)
        print(json.dumps(summ, indent=1))
        return
    if args.write_manifest:
        doc = write_manifest()
        h = doc["composition_headline"]["clean_us_labels_available_now"]
        print(f"wrote {os.path.relpath(MANIFEST)}: {len(doc['sources'])} sources; "
              f"clean US in-scope labels tower={h['tower']} substation={h['substation']}")
        return
    # default: documented table
    for s in build_manifest()["sources"]:
        print(f"- {s['name']}\n    license: {s.get('license')}\n    "
              f"{s.get('verification', s.get('role', ''))}")


if __name__ == "__main__":
    main()
