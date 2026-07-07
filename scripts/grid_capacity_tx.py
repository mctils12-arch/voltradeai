#!/usr/bin/env python3
"""grid_capacity_tx.py — GRID VISION A1 gate-1: Texas grid capacity
registry from the OSM power extract (research/grid_vision_products.md
build-order item 1; ladder DATA gate for the grid-stress signal).

Input: power.geojsonseq produced by the scripts/build_power_tiles.sh
filter steps (osmium tags-filter w/power=line,minor_line,cable +
nwr/power=substation,plant on the Geofabrik texas-latest extract).
Output: datacore/gridvision/tx_grid_registry.json — aggregate line-km
and circuit-km per voltage class, substation/plant counts, and the
gate-1 comparison. Feature-level data is NOT committed (session-run
intermediate); the aggregate is the artifact.

ODbL: this artifact derives from OpenStreetMap — attribution
"(c) OpenStreetMap contributors, ODbL 1.0" carried in provenance; a
published database derived from OSM stays ODbL (see
research/grid_vision_products.md, license boundary).

GATE-1 CRITERIA — PRE-STATED 2026-07-07 BEFORE the comparison ran
(REASONING STANDARD #10; do not edit after the fact):
  Reference: ERCOT Fact Sheet "55,000+ miles of high-voltage
  transmission lines" = >=88,514 km (ERCOT region only — the state
  extract also covers non-ERCOT El Paso/Panhandle/East TX, so the
  reference is a LOWER-BOUND anchor, not an exact target).
  PRIOR: OSM US coverage is good >=345kV, uneven 69-161kV
  (grid_vision_research.md Item 2.8) -> expected state >=69kV
  circuit-km lands in 60-110% of 88,500 km, with 345kV closest to
  complete.
  PASS (registry usable as the stress-index capacity input):
    (a) >=69kV circuit-km in [44_250, 115_000] km  (50%-130% of ref);
    (b) unknown-voltage share of power=line km < 35%;
    (c) 345kV-class circuit-km >= 8_000 km (CREZ alone added ~3,600
        circuit-miles of 345kV; materially less mapped than 8,000 km
        means coverage too thin to trust).
  FAIL handling: log layer-of-death in experiments.md; if only (a)
  fails while (c) passes, the stress index may proceed on the
  >=345kV subset with the restriction stated.

Usage: python3 scripts/grid_capacity_tx.py <power.geojsonseq> \
    [--extract-date YYYY-MM-DD] [--out datacore/gridvision/tx_grid_registry.json]
"""
import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

# volts -> class bucket (labels chosen to match US nominal families)
def voltage_class(volts):
    if volts is None:
        return "unknown"
    if volts >= 345_000:
        return "345kV+"
    if volts >= 230_000:
        return "230kV"
    if volts >= 100_000:
        return "100-229kV"
    if volts >= 69_000:
        return "69-99kV"
    return "<69kV"


_NUM = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(kv)?\s*$", re.IGNORECASE)


def parse_voltages(raw):
    """OSM voltage tag -> (list of volts as int, unparseable_count).
    Convention: volts as integers, ';'-separated for multi-circuit
    ways carrying different voltages ("138000;69000"). Tolerates
    'kV' suffixes; anything else (e.g. 'medium', '?') is counted
    unparseable, never guessed."""
    if raw is None or str(raw).strip() == "":
        return [], 0
    volts, bad = [], 0
    for part in str(raw).split(";"):
        m = _NUM.match(part)
        if not m:
            bad += 1
            continue
        v = float(m.group(1))
        if m.group(2):  # explicit kV suffix
            v *= 1000.0
        elif v < 1000:  # bare small number: OSM sometimes tags kV without unit
            v *= 1000.0
        volts.append(int(round(v)))
    return volts, bad


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def linestring_km(coords):
    return sum(
        haversine_km(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
        for i in range(len(coords) - 1)
    )


def circuit_km_for(way_km, volts, circuits_tag):
    """Circuit-km accounting, honesty-first:
    - multi-voltage way: each voltage = one circuit of that voltage
      (the OSM convention for shared towers) -> way_km per voltage;
    - single voltage + circuits=N tag: N x way_km on that voltage;
    - single voltage, no circuits tag: 1 x way_km;
    - no parseable voltage: 1 x way_km into 'unknown'."""
    out = defaultdict(float)
    if not volts:
        out["unknown"] += way_km
        return out
    if len(volts) > 1:
        for v in volts:
            out[voltage_class(v)] += way_km
        return out
    n = 1
    if circuits_tag:
        try:
            n = max(1, int(str(circuits_tag).split(";")[0]))
        except ValueError:
            n = 1
    out[voltage_class(volts[0])] += way_km * n
    return out


def new_bucket():
    return {"ways": 0, "km": 0.0, "circuit_km": defaultdict(float),
            "voltage_census_km": defaultdict(float),
            "unknown_voltage_ways": 0, "unknown_voltage_km": 0.0,
            "multi_voltage_ways": 0, "ways_with_circuits_tag": 0,
            "unparseable_voltage_parts": 0}


def fold_line(bucket, props, km):
    volts, bad = parse_voltages(props.get("voltage"))
    bucket["ways"] += 1
    bucket["km"] += km
    bucket["unparseable_voltage_parts"] += bad
    if not volts:
        bucket["unknown_voltage_ways"] += 1
        bucket["unknown_voltage_km"] += km
    if len(volts) > 1:
        bucket["multi_voltage_ways"] += 1
    if props.get("circuits"):
        bucket["ways_with_circuits_tag"] += 1
    for v in volts:
        bucket["voltage_census_km"][str(v)] += km
    for cls, ckm in circuit_km_for(km, volts, props.get("circuits")).items():
        bucket["circuit_km"][cls] += ckm


def finalize(bucket):
    census = sorted(bucket["voltage_census_km"].items(),
                    key=lambda kv: kv[1], reverse=True)[:20]
    return {
        "ways": bucket["ways"],
        "km": round(bucket["km"], 1),
        "circuit_km_by_class": {k: round(v, 1) for k, v in sorted(bucket["circuit_km"].items())},
        "voltage_census_top20_km": {k: round(v, 1) for k, v in census},
        "unknown_voltage": {
            "ways": bucket["unknown_voltage_ways"],
            "km": round(bucket["unknown_voltage_km"], 1),
            "share_of_km": round(bucket["unknown_voltage_km"] / bucket["km"], 4) if bucket["km"] else None,
        },
        "multi_voltage_ways": bucket["multi_voltage_ways"],
        "ways_with_circuits_tag": bucket["ways_with_circuits_tag"],
        "unparseable_voltage_parts": bucket["unparseable_voltage_parts"],
    }


def run(seq_path, extract_date, out_path):
    line_b, minor_b, cable_b = new_bucket(), new_bucket(), new_bucket()
    subs = {"total": 0, "points": 0, "areas": 0}
    plants = {"total": 0}
    skipped_geoms = 0
    with open(seq_path, "r", encoding="utf-8") as f:
        for row in f:
            row = row.lstrip("\x1e").strip()
            if not row:
                continue
            feat = json.loads(row)
            props = feat.get("properties") or {}
            geom = feat.get("geometry") or {}
            p = props.get("power")
            if p in ("line", "minor_line", "cable"):
                if geom.get("type") != "LineString":
                    skipped_geoms += 1
                    continue
                km = linestring_km(geom["coordinates"])
                fold_line({"line": line_b, "minor_line": minor_b,
                           "cable": cable_b}[p], props, km)
            elif p == "substation":
                subs["total"] += 1
                subs["points" if geom.get("type") == "Point" else "areas"] += 1
            elif p == "plant":
                plants["total"] += 1

    lines = finalize(line_b)
    ckm = lines["circuit_km_by_class"]
    hv_circuit_km = sum(v for k, v in ckm.items()
                        if k in ("69-99kV", "100-229kV", "230kV", "345kV+"))
    kv345 = ckm.get("345kV+", 0.0)
    unknown_share = lines["unknown_voltage"]["share_of_km"] or 0.0
    gate1 = {
        "reference": "ERCOT Fact Sheet: 55,000+ miles high-voltage transmission "
                     "= >=88,514 km (ERCOT region; lower-bound anchor for the "
                     "whole-state extract)",
        "criteria_prestated": "2026-07-07, before comparison (script header): "
                              "(a) >=69kV circuit-km in [44250, 115000]; "
                              "(b) unknown-voltage share of power=line km < 0.35; "
                              "(c) 345kV-class circuit-km >= 8000",
        "computed": {
            "hv_circuit_km_gte69kV": round(hv_circuit_km, 1),
            "pct_of_reference": round(hv_circuit_km / 88514.0, 4),
            "unknown_voltage_share": unknown_share,
            "kv345_circuit_km": kv345,
        },
        "checks": {
            "a_hv_circuit_km_in_band": 44_250 <= hv_circuit_km <= 115_000,
            "b_unknown_share_lt_35pct": unknown_share < 0.35,
            "c_345kV_gte_8000km": kv345 >= 8_000,
        },
    }
    gate1["verdict"] = "PASS" if all(gate1["checks"].values()) else "FAIL"

    artifact = {
        "provenance": {
            "source": f"OpenStreetMap via Geofabrik texas-latest extract "
                      f"(downloaded {extract_date}); filtered per "
                      "scripts/build_power_tiles.sh (osmium power= tags)",
            "license": "ODbL 1.0 — derivative database of OSM; "
                       "(c) OpenStreetMap contributors",
            "attribution": "OpenStreetMap contributors",
            "method": "scripts/grid_capacity_tx.py — aggregate registry; "
                      "feature-level data intentionally not committed",
            "circuit_km_convention": "multi-voltage way = one circuit per "
                                     "voltage; circuits=N multiplies "
                                     "single-voltage ways; untagged = 1 "
                                     "circuit into 'unknown' — never guessed",
        },
        "lines": lines,
        "minor_lines": finalize(minor_b),
        "cables": finalize(cable_b),
        "substations": subs,
        "plants": plants,
        "skipped_non_linestring_geoms": skipped_geoms,
        "gate1": gate1,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=1)
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_path")
    ap.add_argument("--extract-date", default="unknown")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datacore", "gridvision", "tx_grid_registry.json"))
    args = ap.parse_args()
    art = run(args.seq_path, args.extract_date, args.out)
    g = art["gate1"]
    print(json.dumps({"gate1": g, "lines_km": art["lines"]["km"],
                      "substations": art["substations"]["total"]}, indent=1))
    sys.exit(0 if g["verdict"] == "PASS" else 3)
