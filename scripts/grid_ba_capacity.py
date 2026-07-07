#!/usr/bin/env python3
"""grid_ba_capacity.py — GRID VISION A1 step 3: distribute the TX
capacity registry geographically — per-county and per-BA circuit-km —
by assigning each OSM line SEGMENT to a county (point-in-polygon of
the segment midpoint over Census cartographic county boundaries) and
rolling counties up to balancing authorities via tx_county_ba.json.

HONESTY DESIGN: multi-BA counties (80 of 254 at EIA-861 granularity)
contribute their km to an AMBIGUOUS pool keyed by the BA set
("ERCO|SWPP"), never split by an invented ratio. Per-BA exclusive km
comes only from single-BA counties. Segments falling outside every TX
county (border-crossing lines in the extract) land in out_of_state,
counted not dropped.

CONSERVATION INVARIANT (pre-stated): same GeoJSONSeq input + the same
circuit-km convention as scripts/grid_capacity_tx.py means
  sum(per-county) + out_of_state == tx_grid_registry.json lines totals
to rounding. The build FAILS if conservation breaks — segments may
never silently vanish.

PRE-STATED EXPECTATIONS (2026-07-07, before first run; REASONING
STANDARD #10): (1) conservation holds within 0.5%; (2) the ambiguous
share of 345kV+ circuit-km lands roughly 15-35% (CREZ corridors run
through the Panhandle ERCO/SWPP seam); (3) out_of_state well under 5%
(Geofabrik clips near the border). Divergence is investigated, not
accepted.

Deps: pyshp (session-run only — imported inside load_counties so CI
tests never need it; declared in requirements-dev.txt).

Usage: python3 scripts/grid_ba_capacity.py <power.geojsonseq> \
    <county_shp_basename> [--registry datacore/gridvision/tx_grid_registry.json] \
    [--crosswalk datacore/gridvision/tx_county_ba.json] \
    [--out datacore/gridvision/tx_ba_capacity.json]
"""
import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "grid_capacity_tx", os.path.join(_HERE, "grid_capacity_tx.py"))
_gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gc)
parse_voltages = _gc.parse_voltages
circuit_km_for = _gc.circuit_km_for
haversine_km = _gc.haversine_km


def point_in_rings(lon, lat, rings):
    """Even-odd ray casting across all rings (handles holes and
    multipart islands without orientation analysis)."""
    inside = False
    for ring in rings:
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i]
            xj, yj = ring[j]
            if (yi > lat) != (yj > lat) and \
               lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
    return inside


class CountyIndex:
    """Bbox-filtered point-in-polygon with last-hit caching — line
    segments are spatially contiguous, so most lookups resolve on the
    previous county without a single ray cast."""

    def __init__(self, counties):
        # counties: list of (name, geoid, bbox, rings)
        self.counties = counties
        self._last = None

    def locate(self, lon, lat):
        if self._last is not None:
            name, geoid, bbox, rings = self._last
            if bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3] \
                    and point_in_rings(lon, lat, rings):
                return name
        for c in self.counties:
            name, geoid, bbox, rings = c
            if bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3] \
                    and point_in_rings(lon, lat, rings):
                self._last = c
                return name
        return None


def load_counties(shp_basename, statefp="48"):
    import shapefile  # pyshp — session-run dependency only
    sf = shapefile.Reader(shp_basename)
    out = []
    for rec, shape in zip(sf.records(), sf.shapes()):
        if rec["STATEFP"] != statefp:
            continue
        pts = shape.points
        parts = list(shape.parts) + [len(pts)]
        rings = [pts[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
        out.append((rec["NAME"], rec["GEOID"], tuple(shape.bbox), rings))
    return out


def run(seq_path, shp_basename, registry_path, crosswalk_path, out_path):
    idx = CountyIndex(load_counties(shp_basename))
    per_county = defaultdict(lambda: defaultdict(float))
    out_of_state = defaultdict(float)
    ways = 0
    with open(seq_path, "r", encoding="utf-8") as f:
        for row in f:
            row = row.lstrip("\x1e").strip()
            if not row:
                continue
            feat = json.loads(row)
            props = feat.get("properties") or {}
            if props.get("power") != "line":
                continue
            geom = feat.get("geometry") or {}
            if geom.get("type") != "LineString":
                continue
            ways += 1
            volts, _bad = parse_voltages(props.get("voltage"))
            circuits = props.get("circuits")
            coords = geom["coordinates"]
            for i in range(len(coords) - 1):
                (x1, y1), (x2, y2) = coords[i][:2], coords[i + 1][:2]
                seg_km = haversine_km(x1, y1, x2, y2)
                if seg_km == 0:
                    continue
                county = idx.locate((x1 + x2) / 2, (y1 + y2) / 2)
                sink = per_county[county] if county else out_of_state
                for cls, ckm in circuit_km_for(seg_km, volts, circuits).items():
                    sink[cls] += ckm

    # conservation check vs the committed registry (same input+convention)
    registry = json.load(open(registry_path))
    reg_ckm = registry["lines"]["circuit_km_by_class"]
    dist_ckm = defaultdict(float)
    for cls_km in per_county.values():
        for cls, km in cls_km.items():
            dist_ckm[cls] += km
    for cls, km in out_of_state.items():
        dist_ckm[cls] += km
    conservation = {}
    for cls in sorted(set(reg_ckm) | set(dist_ckm)):
        a, b = reg_ckm.get(cls, 0.0), round(dist_ckm.get(cls, 0.0), 1)
        conservation[cls] = {"registry": a, "distributed": b,
                             "abs_diff": round(abs(a - b), 1)}
    total_reg = sum(reg_ckm.values())
    total_dist = sum(v["distributed"] for v in conservation.values())
    conserved = abs(total_reg - total_dist) <= max(0.005 * total_reg, 1.0)

    # BA rollup via the crosswalk — exclusive vs ambiguous, never split
    crosswalk = json.load(open(crosswalk_path))["counties"]
    ba_exclusive = defaultdict(lambda: defaultdict(float))
    ba_ambiguous = defaultdict(lambda: defaultdict(float))
    unknown_county = defaultdict(float)
    for county, cls_km in per_county.items():
        cw = crosswalk.get(county)
        if cw is None:
            for cls, km in cls_km.items():
                unknown_county[cls] += km
            continue
        bas = cw["bas"]
        sink = ba_exclusive[bas[0]] if len(bas) == 1 else ba_ambiguous["|".join(bas)]
        for cls, km in cls_km.items():
            sink[cls] += km

    def r(d):
        return {k: round(v, 1) for k, v in sorted(d.items())}

    kv345_excl = sum(v.get("345kV+", 0.0) for v in ba_exclusive.values())
    kv345_amb = sum(v.get("345kV+", 0.0) for v in ba_ambiguous.values())
    kv345_share_amb = round(kv345_amb / (kv345_excl + kv345_amb), 4) \
        if (kv345_excl + kv345_amb) else None

    artifact = {
        "provenance": {
            "source": "OSM power=line segments (same extract as "
                      "tx_grid_registry.json) x Census cartographic county "
                      "boundaries (cb_2025_us_county_500k, public domain, "
                      "cite Census) x EIA-861 county retail-BA crosswalk "
                      "(tx_county_ba.json)",
            "license": "ODbL 1.0 (OSM-derived); (c) OpenStreetMap "
                       "contributors; county boundaries US Census (public "
                       "domain); crosswalk EIA (public domain)",
            "attribution": "OpenStreetMap contributors; U.S. Census Bureau; "
                           "U.S. Energy Information Administration",
            "method": "segment-midpoint point-in-polygon; multi-BA county km "
                      "pooled AMBIGUOUS by BA set, never split by invented "
                      "ratio; county retail BA is NOT grid topology (stated "
                      "in tx_county_ba.json); out-of-state segments counted, "
                      "not dropped",
        },
        "ways_processed": ways,
        "per_county_circuit_km": {k: r(v) for k, v in sorted(per_county.items(), key=lambda kv: kv[0] or "")},
        "ba_exclusive_circuit_km": {k: r(v) for k, v in sorted(ba_exclusive.items())},
        "ba_ambiguous_circuit_km": {k: r(v) for k, v in sorted(ba_ambiguous.items())},
        "out_of_state_circuit_km": r(out_of_state),
        "unknown_county_circuit_km": r(unknown_county),
        "conservation": {
            "by_class": conservation,
            "total_registry": round(total_reg, 1),
            "total_distributed": round(total_dist, 1),
            "conserved_within_0p5pct": conserved,
        },
        "kv345_ambiguous_share": kv345_share_amb,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=1)
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_path")
    ap.add_argument("shp_basename")
    root = os.path.dirname(_HERE)
    ap.add_argument("--registry", default=os.path.join(root, "datacore", "gridvision", "tx_grid_registry.json"))
    ap.add_argument("--crosswalk", default=os.path.join(root, "datacore", "gridvision", "tx_county_ba.json"))
    ap.add_argument("--out", default=os.path.join(root, "datacore", "gridvision", "tx_ba_capacity.json"))
    args = ap.parse_args()
    art = run(args.seq_path, args.shp_basename, args.registry, args.crosswalk, args.out)
    print(json.dumps({
        "ways": art["ways_processed"],
        "conservation": art["conservation"]["conserved_within_0p5pct"],
        "totals": {"registry": art["conservation"]["total_registry"],
                   "distributed": art["conservation"]["total_distributed"]},
        "ba_exclusive_345kV": {k: v.get("345kV+") for k, v in art["ba_exclusive_circuit_km"].items() if v.get("345kV+")},
        "kv345_ambiguous_share": art["kv345_ambiguous_share"],
        "out_of_state": art["out_of_state_circuit_km"],
    }, indent=1))
    sys.exit(0 if art["conservation"]["conserved_within_0p5pct"] else 3)
