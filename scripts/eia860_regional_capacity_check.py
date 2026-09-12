#!/usr/bin/env python3
"""eia860_regional_capacity_check.py — the REGIONAL follow-up
scripts/grid_generation_gate1_ba.py's own 2026-09-11 NEXT(1) filed: for the
two remaining non-confounded per-region gate-1 FAILs (ERCO solar 1.076x,
SWPP solar 2.013x, --source eia860), is the registry's shortfall a
RESIDUAL REGIONAL gap the 2026-09-11 national solar-undercount fix
(scripts/eia860_add_missing_plants.py) didn't fully close for these two
specific regions, or is the registry ALREADY close to EIA-860's own
BA-level total for these regions (in which case the gate-1 overshoot is
NOT a registry-completeness problem and needs a different explanation)?

THE NATIONAL FIX WAS ONLY EVER VALIDATED NATIONALLY: it compared US-wide
registry solar/wind capacity against US-wide EIA-860 solar/wind capacity.
It was never independently checked per-BA. This script closes exactly
that gap for the two regions in question — no more, no less (the prior
NEXT item explicitly asked for "each is its own small, focused check, not
a repeat of the national one").

METHOD: EIA-860's OWN reported "Balancing Authority Code" column
(Schedule 2, the Plant file — the SAME ground-truth column
scripts/grid_ba_eia860_join.py already joins the registry against) is
used to build a plant_code -> ba_code map DIRECTLY, independent of any
coordinate-rounding join (no coordinate matching needed here since both
sides of THIS comparison are keyed by EIA Plant Code natively: Schedule 3
solar/wind generator rows carry Plant Code, and Schedule 2 carries
Plant Code -> BA Code). This yields EIA-860's own regional solar/wind
capacity total for ERCO and SWPP, independent of the registry entirely —
a genuine second, independent ground-truth read, not a re-derivation of
the same registry number.

That EIA-860-own-BA-total is then compared against the REGISTRY's own
matched capacity for the same (ba, fuel) pairs, using the ALREADY-BUILT,
ALREADY-COMMITTED datacore/powerplants/plant_balancing_authority_eia860.json
(scripts/grid_ba_eia860_join.py's own output — reused via importlib, not
recomputed, EDGE DOCTRINE #3) so this check adds no new coordinate-join
logic and cannot silently drift from the join grid_generation_gate1_ba.py
itself already trusts.

PRIOR, stated before running (REASONING STANDARD #10): the national fix
added plants EIA-860 carries that GPPD/registry had NO row for AT ALL
(scripts/eia860_add_missing_plants.py, 4,053 solar / 286 wind rows,
2026-09-11). Missing-plant coverage gaps of that kind are not guaranteed
to be geographically uniform — solar buildout has been regionally
lumpy (West Texas/ERCOT and the SPP footprint are both real recent
utility-scale solar growth corridors) — so the PRIOR expectation is that
SOME residual regional shortfall plausibly remains for ERCO/SWPP
specifically even after a nationally-balanced fix, but the prior gate-1
ratios (1.076x/2.013x) are ALSO consistent with zero residual registry
gap if EIA-930's own regional generation attribution includes generation
categories the registry-nameplate side structurally cannot (behind-the-
meter/distributed solar attributed to a BA's reported total, or
cross-BA transmission double-counting) — this script's job is to tell
those two apart, not to assume the registry-gap story is correct because
it would be a tidier narrative.

Usage (files not fetched automatically, same manual-download precedent
every sibling eia860_*.py script documents):
    curl -L -o /tmp/eia860.zip https://www.eia.gov/electricity/data/eia860/xls/eia8602025.zip
    unzip /tmp/eia860.zip -d /tmp/eia860
    python3 scripts/eia860_regional_capacity_check.py \
        --plants /tmp/eia860/2___Plant_Y2025.xlsx \
        --solar /tmp/eia860/3_3_Solar_Y2025.xlsx --wind /tmp/eia860/3_2_Wind_Y2025.xlsx \
        --bas ERCO,SWPP
"""
import argparse
import importlib.util
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
BA_JOIN_PATH = os.path.join(REPO_ROOT, "datacore", "powerplants", "plant_balancing_authority_eia860.json")

_mpc_spec = importlib.util.spec_from_file_location(
    "eia860_missing_plants_check", os.path.join(HERE, "eia860_missing_plants_check.py"))
_mpc = importlib.util.module_from_spec(_mpc_spec)
_mpc_spec.loader.exec_module(_mpc)
eia860_capacity_by_code = _mpc.eia860_capacity_by_code
load_eia860_generator_rows = _mpc.load_eia860_generator_rows

_join_spec = importlib.util.spec_from_file_location(
    "grid_ba_eia860_join", os.path.join(HERE, "grid_ba_eia860_join.py"))
_join = importlib.util.module_from_spec(_join_spec)
_join_spec.loader.exec_module(_join)
registry_capacity_by_ba = _join.registry_capacity_by_ba

DEFAULT_BAS = ("ERCO", "SWPP")
DEFAULT_FUELS = ("solar", "wind")


def eia860_plant_ba_map(rows):
    """rows: iterable of (plant_code, ba_code) tuples straight off EIA-860
    Schedule 2's own "Plant Code" / "Balancing Authority Code" columns —
    no coordinate matching, since this is a native Plant-Code-to-Plant-Code
    join (Schedule 3 generator rows already carry Plant Code directly).
    Returns (plant_to_ba: {plant_code: ba_code}, skipped_no_ba: count) —
    rows with no reported BA code are counted, never guessed."""
    out = {}
    skipped_no_ba = 0
    for code, ba in rows:
        if not ba or not str(ba).strip():
            skipped_no_ba += 1
            continue
        out[code] = str(ba).strip()
    return out, skipped_no_ba


def eia860_capacity_by_ba(capacity_by_code, plant_to_ba):
    """capacity_by_code: {plant_code: mw} for ONE fuel (e.g.
    eia860_capacity_by_code()'s output on a Schedule 3 solar or wind file).
    plant_to_ba: {plant_code: ba_code} (eia860_plant_ba_map()'s output).
    Returns (capacity_by_ba: {ba_code: mw}, unmapped_mw, unmapped_plants) —
    a plant code EIA-860 reports capacity for but no BA code for (or a code
    Schedule 2 has no row for at all) contributes to unmapped_mw/
    unmapped_plants instead of being silently dropped or guessed at."""
    cap = defaultdict(float)
    unmapped_mw = 0.0
    unmapped_plants = 0
    for code, mw in capacity_by_code.items():
        ba = plant_to_ba.get(code)
        if ba is None:
            unmapped_mw += mw
            unmapped_plants += 1
            continue
        cap[ba] += mw
    return dict(cap), round(unmapped_mw, 1), unmapped_plants


def compare_regional_capacity(eia860_ba_capacity, registry_ba_capacity, ba, fuel):
    """eia860_ba_capacity: {ba_code: mw} for this fuel — this script's own
    independent EIA-860-BA-level ground truth. registry_ba_capacity:
    {ba_code: {fuel: mw}} — grid_ba_eia860_join.registry_capacity_by_ba's
    per_ba_capacity output (the SAME source grid_generation_gate1_ba.py's
    --source eia860 already trusts). ratio < 1 means the registry's
    matched capacity for this (ba, fuel) is BELOW EIA-860's own reported
    total for that region — a residual regional registry gap. ratio close
    to 1 means the registry is already essentially complete for this
    (ba, fuel) pair, so a gate-1 overshoot there is NOT a registry-
    completeness problem."""
    eia860_mw = eia860_ba_capacity.get(ba, 0.0)
    registry_mw = registry_ba_capacity.get(ba, {}).get(fuel, 0.0)
    ratio = round(registry_mw / eia860_mw, 3) if eia860_mw else None
    return {
        "ba": ba,
        "fuel": fuel,
        "eia860_reported_mw": round(eia860_mw, 1),
        "registry_matched_mw": round(registry_mw, 1),
        "registry_to_eia860_ratio": ratio,
        "gap_mw": round(eia860_mw - registry_mw, 1),
    }


def load_eia860_plant_ba_rows(xlsx_path):
    """EIA-860 Schedule 2 (Plant file) -> iterable of (plant_code, ba_code)
    tuples. Same header-lookup-by-name convention as every sibling
    eia860_*.py script; no coordinate columns needed here, unlike
    grid_ba_eia860_join.py's load_eia860_ba_directory (that function
    builds a coordinate index for a DIFFERENT join — the registry has no
    EIA Plant Code to join on directly, so it needs coordinates; this
    script's other side, EIA-860's own Schedule 3, already carries Plant
    Code natively, so no coordinate is needed on this path)."""
    import openpyxl  # session-run only, same convention as sibling scripts
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    next(rows)  # title row
    hdr = next(rows)
    i_code = hdr.index("Plant Code")
    i_ba = hdr.index("Balancing Authority Code")
    for row in rows:
        if row is None or row[i_code] is None:
            continue
        yield (row[i_code], row[i_ba])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants", required=True, help="path to EIA-860 2___Plant_Y<year>.xlsx")
    ap.add_argument("--solar", required=True, help="path to EIA-860 3_3_Solar_Y<year>.xlsx")
    ap.add_argument("--wind", required=True, help="path to EIA-860 3_2_Wind_Y<year>.xlsx")
    ap.add_argument("--bas", default=",".join(DEFAULT_BAS))
    ap.add_argument("--ba-join", default=BA_JOIN_PATH,
                     help="grid_ba_eia860_join.py's committed output (registry side ground truth)")
    args = ap.parse_args()

    target_bas = [b.strip() for b in args.bas.split(",") if b.strip()]

    plant_to_ba, skipped_no_ba = eia860_plant_ba_map(load_eia860_plant_ba_rows(args.plants))

    with open(args.ba_join) as f:
        ba_join = json.load(f)
    registry_ba_capacity, _ = registry_capacity_by_ba(ba_join["assignments"])

    results = []
    for fuel, path in (("solar", args.solar), ("wind", args.wind)):
        capacity_by_code = eia860_capacity_by_code(load_eia860_generator_rows(path))
        eia860_ba_cap, unmapped_mw, unmapped_plants = eia860_capacity_by_ba(capacity_by_code, plant_to_ba)
        for ba in target_bas:
            row = compare_regional_capacity(eia860_ba_cap, registry_ba_capacity, ba, fuel)
            row["eia860_unmapped_mw_this_fuel_nationally"] = unmapped_mw
            row["eia860_unmapped_plants_this_fuel_nationally"] = unmapped_plants
            results.append(row)

    report = {
        "check": "eia860_regional_capacity_check",
        "note": "compares EIA-860's OWN BA-level reported capacity (ground truth, "
                "independent of the registry) against the registry's matched capacity "
                "for the same (ba, fuel) pairs — isolates whether a gate-1 regional "
                "overshoot is a residual registry-completeness gap or something else",
        "eia860_plant_rows_skipped_no_ba_code": skipped_no_ba,
        "ba_join_source": os.path.relpath(args.ba_join, REPO_ROOT),
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
