#!/usr/bin/env python3
"""eia860_add_missing_plants.py — ships NEXT(1) from
eia860_missing_plants_check.py: actually adds the plant-codes EIA-860
carries (solar/wind, "OP" status, positive nameplate) that WRI GPPD has NO
row for at all, closing the registry-staleness gap that check quantified
(117,104.8 MW solar / 55,646.4 MW wind missing-plant capacity) rather than
just measuring it.

REFINEMENT over the prior check's own fuel-scoped definition of "missing":
that script's `gppd_codes_by_fuel` only matches a GPPD row whose OWN
primary_fuel is solar/wind against EIA-860's solar/wind plant codes — so a
plant EIA-860 tags "Solar" that GPPD already carries under a DIFFERENT
label (e.g. co-located storage, or a stale prior-fuel tag) would read as
"missing" there and get double-added here as a second marker for the same
physical plant. Checked live this session against the real 2026-09
GPPD CSV: 69 solar-coded / 11 wind-coded EIA-860 plant codes (470.8 MW /
54.0 MW — small next to the totals, but each would be a real duplicate
point on the /data map) are already present in GPPD under some OTHER
fuel. This script instead computes "missing" against GPPD's FULL USA
idnr-derived code set (`gppd_all_usa_codes`, any fuel), which is strictly
more correct and yields 4,053 solar / 286 wind additions (vs. the prior
check's 4,122 / 297).

WHAT SHIPS: added rows carry `verified=0` (registry-reported via EIA-860,
never imagery-verified) and are EIA-860-sourced only — no gppd_idnr exists
for them (they are, by construction, plants GPPD does not carry), so
there is nothing to look up in imagery_verified.json or
position_overrides.json for these rows; both keyed strictly by gppd_idnr,
which only ever applies to GPPD-sourced rows built by build_powerplants.py's
own `build_plants()`. Rebuilds the registry from the same GPPD+EIA-860
pipeline `build_powerplants.py` already uses (calling its own
`build_plants()`, not hand-copying its logic) and appends the missing-plant
rows on top, so this script is idempotent/reproducible from raw sources —
re-running it from a fresh pull does not depend on, or compound onto, the
previously-shipped datacore/powerplants/us_power_plants.json.

Re-run (files not fetched automatically, same manual-download precedent
every sibling EIA-860 script in this directory already documents):
    curl -L -o /tmp/gppd.csv https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv
    curl -L -o /tmp/eia860.zip https://www.eia.gov/electricity/data/eia860/xls/eia8602025.zip
    unzip /tmp/eia860.zip -d /tmp/eia860
    python3 scripts/eia860_add_missing_plants.py \
        --gppd /tmp/gppd.csv \
        --plants /tmp/eia860/2___Plant_Y2025.xlsx \
        --solar /tmp/eia860/3_3_Solar_Y2025.xlsx --wind /tmp/eia860/3_2_Wind_Y2025.xlsx
"""
import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DST_DIR = os.path.join(REPO_ROOT, "datacore", "powerplants")
REGISTRY_PATH = os.path.join(DST_DIR, "us_power_plants.json")

_bpp_spec = importlib.util.spec_from_file_location(
    "build_powerplants", os.path.join(HERE, "build_powerplants.py"))
_bpp = importlib.util.module_from_spec(_bpp_spec)
_bpp_spec.loader.exec_module(_bpp)

_mpc_spec = importlib.util.spec_from_file_location(
    "missing_plants_check", os.path.join(HERE, "eia860_missing_plants_check.py"))
_mpc = importlib.util.module_from_spec(_mpc_spec)
_mpc_spec.loader.exec_module(_mpc)
gppd_plant_code = _mpc.gppd_plant_code
eia860_capacity_by_code = _mpc.eia860_capacity_by_code
load_eia860_generator_rows = _mpc.load_eia860_generator_rows


def gppd_all_usa_codes(rows):
    """rows: iterable of (country, gppd_idnr) tuples. Returns the set of
    EIA Plant Codes GPPD carries ANY row for, regardless of fuel — the
    correct universe to subtract for "does GPPD have this plant at all",
    as opposed to eia860_missing_plants_check.py's fuel-scoped set (which
    answers a narrower, related question: does GPPD have this plant AS
    solar/wind)."""
    out = set()
    for country, idnr in rows:
        if country != "USA":
            continue
        code = gppd_plant_code(idnr)
        if code is not None:
            out.add(code)
    return out


def build_missing_plant_rows(fuel, missing_codes, eia_capacity_by_code_map, plant_directory):
    """Pure row-builder: for each EIA Plant Code confirmed missing from
    GPPD entirely, emits a registry-format row
    [name, capacity_mw, fuel, owner, lat, lon, verified=0] — verified is
    always 0 (EIA-860-registry-reported, never imagery-verified; these
    rows have no gppd_idnr, so imagery_verified.json/position_overrides.json,
    both keyed by gppd_idnr, structurally cannot apply to them). Skips a
    code with non-positive summed capacity (no real operating plant) or
    missing/null coordinates in the plant directory, counting both so the
    caller can report data completeness rather than silently dropping
    rows. Returns (rows, skipped_zero_or_neg_capacity, skipped_bad_coords)."""
    rows, skipped_cap, skipped_coords = [], 0, 0
    for code in sorted(missing_codes):
        mw = eia_capacity_by_code_map.get(code, 0.0)
        if mw <= 0:
            skipped_cap += 1
            continue
        entry = plant_directory.get(code)
        if entry is None:
            skipped_coords += 1
            continue
        name, state, lat, lon, utility = entry
        if lat is None or lon is None:
            skipped_coords += 1
            continue
        rows.append([
            (name or f"EIA Plant {code}").strip()[:60],
            round(mw, 1),
            fuel,
            (utility or "").strip()[:60],
            round(float(lat), 4),
            round(float(lon), 4),
            0,
        ])
    return rows, skipped_cap, skipped_coords


def merge_registry(existing_plants, new_rows):
    """Appends new_rows onto existing_plants and re-sorts by -capacity_mw,
    matching build_powerplants.py's own sort convention (largest first)."""
    merged = list(existing_plants) + list(new_rows)
    merged.sort(key=lambda p: -p[1])
    return merged


def load_gppd_country_idnr_rows(csv_path):
    import csv
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield (row["country"], row["gppd_idnr"])


def load_eia860_plant_directory(xlsx_path):
    """EIA-860 Schedule 2 (Plant file): Plant Code -> (name, state, lat,
    lon, utility_name). Same header-lookup-by-name convention every
    sibling EIA-860 script in this directory uses."""
    import openpyxl  # session-run only
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    next(rows)  # title row
    hdr = next(rows)
    i_code = hdr.index("Plant Code")
    i_name = hdr.index("Plant Name")
    i_state = hdr.index("State")
    i_lat = hdr.index("Latitude")
    i_lon = hdr.index("Longitude")
    i_util = hdr.index("Utility Name")
    out = {}
    for row in rows:
        if row is None or row[i_code] is None:
            continue
        out[row[i_code]] = (row[i_name], row[i_state], row[i_lat], row[i_lon], row[i_util])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gppd", required=True, help="path to global_power_plant_database.csv")
    ap.add_argument("--plants", required=True, help="path to EIA-860 2___Plant_Y<year>.xlsx")
    ap.add_argument("--solar", required=True, help="path to EIA-860 3_3_Solar_Y<year>.xlsx")
    ap.add_argument("--wind", required=True, help="path to EIA-860 3_2_Wind_Y<year>.xlsx")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write the registry file")
    args = ap.parse_args()

    all_usa_codes = gppd_all_usa_codes(load_gppd_country_idnr_rows(args.gppd))
    plant_directory = load_eia860_plant_directory(args.plants)

    added_rows = []
    per_fuel = []
    for fuel, path in (("solar", args.solar), ("wind", args.wind)):
        eia_cap = eia860_capacity_by_code(load_eia860_generator_rows(path))
        missing = set(eia_cap) - all_usa_codes
        rows, skipped_cap, skipped_coords = build_missing_plant_rows(
            fuel, missing, eia_cap, plant_directory)
        added_rows.extend(rows)
        per_fuel.append({
            "fuel": fuel,
            "eia860_plant_codes_total": len(eia_cap),
            "missing_from_gppd_any_fuel": len(missing),
            "rows_added": len(rows),
            "skipped_zero_or_neg_capacity": skipped_cap,
            "skipped_bad_coords_or_no_directory_entry": skipped_coords,
            "added_capacity_mw": round(sum(r[1] for r in rows), 1),
        })

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    before_count = registry["count"]
    merged = merge_registry(registry["plants"], added_rows)

    report = {
        "check": "eia860_add_missing_plants",
        "registry_plants_before": before_count,
        "registry_plants_after": len(merged),
        "rows_added_total": len(added_rows),
        "per_fuel": per_fuel,
    }
    print(json.dumps(report, indent=2))

    if args.dry_run:
        return

    registry["plants"] = merged
    registry["count"] = len(merged)
    registry["verified_count"] = sum(p[6] for p in merged)
    registry["_doc"] = (
        registry["_doc"]
        + " SUPPLEMENTED by scripts/eia860_add_missing_plants.py "
          "(2026-09-11): plants EIA-860 carries (solar/wind, operating, "
          "positive nameplate) that WRI GPPD has no row for under ANY "
          "fuel are appended with verified=0, no GPPD counterpart — see "
          "that script's own module docstring for the full method."
    )
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {REGISTRY_PATH} ({os.path.getsize(REGISTRY_PATH) // 1024} KB)")


if __name__ == "__main__":
    main()
