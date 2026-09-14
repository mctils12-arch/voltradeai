#!/usr/bin/env python3
"""eia860m_add_brand_new_plants.py — ships the NEW HYPOTHESIS filed by the
2026-09-13 eia860m_refresh_registry.py capacity-override session: plants
commissioned AFTER EIA-860 ANNUAL's 2024-12-31 snapshot can never be added
by eia860_add_missing_plants.py or eia860m_refresh_registry.py, because
BOTH derive their "missing" candidate set by starting from EIA-860 ANNUAL's
own Schedule 3 solar/wind plant codes — a plant absent from that set can
never appear in a subtraction that starts from it, regardless of what
capacity source is later applied. That 2026-09-13 session precisely
localized this residual for ERCO/SWPP solar (17/2,353.1 MW and 9/715.3 MW
respectively) and confirmed live that EIA-860M's own "Operating" sheet
carries Plant ID/Plant Name/Entity Name/Latitude/Longitude natively on
every generator row — this population can be sourced from EIA-860M ALONE,
with no EIA-860 ANNUAL dependency for membership, capacity, OR coordinates.

MEMBERSHIP: an EIA-860M solar/wind (OP-status) plant code is "brand new"
here iff it is absent from BOTH (a) EIA-860 ANNUAL's own Schedule 3
solar/wind code set (reuses eia860_missing_plants_check.py's
eia860_capacity_by_code/load_eia860_generator_rows unchanged, EDGE DOCTRINE
#3) AND (b) WRI GPPD's full USA code set (reuses
eia860_add_missing_plants.py's gppd_all_usa_codes unchanged) — the same
any-fuel dedupe that script's own docstring already established is needed
to avoid double-marking a plant GPPD already carries under a different
label. (a) and (b) are two INDEPENDENT sources; a plant could in principle
clear one filter and not the other, so both are always checked rather than
assuming EIA-860-ANNUAL-absence implies GPPD-absence.

This population is mutually exclusive BY CONSTRUCTION with
eia860_add_missing_plants.py's own added rows: that script's candidate set
requires a code to be a MEMBER of EIA-860 ANNUAL's Schedule 3 (then
subtracts GPPD); this script's candidate set requires the code to be
ABSENT from EIA-860 ANNUAL's Schedule 3 entirely. No plant code can satisfy
both, so no additional cross-check against that script's already-added
rows is needed before appending.

LIVE RESULT (verified this session, 2026-09-14, fresh downloads — EIA-860M
"as of July 2026", EIA-860 2025 ANNUAL, GPPD CSV): 246 solar plant codes
(13,379.0 MW) / 11 wind plant codes (6,417.2 MW) nationally are absent from
EIA-860 ANNUAL entirely, and ZERO of either population is already in GPPD
under any fuel — the full 257-plant, 19,796.2 MW population is genuinely
new to this registry. ERCO gets 2,353.1 MW solar and SWPP gets 715.3 MW,
matching the 2026-09-13 session's own partition numbers exactly (a
same-vintage-data cross-check, not a coincidence). Missing-coordinate rate:
0/257 — EIA-860M's own Latitude/Longitude columns are fully populated for
this population, unlike EIA-860 ANNUAL's Schedule 2 plant directory (which
structurally cannot have a row for a plant it doesn't know about at all).

TOP-100-VERIFIED INVARIANT: server/powerplants.test.ts hard-asserts every
top-100-by-capacity_mw plant is imagery-verified (verified_count == 100,
exact). Every row this script adds carries verified=0 (no gppd_idnr to add
to imagery_verified.json's id-keyed audit list — see
split_top_n_unverifiable's own docstring). ONE new row from this session's
live run, "SunZia Wind South" (2,561.2 MW, CISO — not ERCO/SWPP, so this
holdback does not touch this session's own solar/ERCO/SWPP result), is
large enough to rank inside the merged registry's top-100 and is therefore
HELD BACK rather than shipped unverified; the other 256 rows all ship.
This is a general, reusable safeguard, not a one-off name check — any
future run holds back whatever newly crosses the same line, reported in
this script's own JSON output under `rows_held_back_top100_unverifiable`.

Re-run (files not fetched automatically, same manual-download precedent
every sibling EIA-860 script in this directory documents):
    curl -L -o /tmp/gppd.csv https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv
    curl -L -o /tmp/eia860.zip https://www.eia.gov/electricity/data/eia860/xls/eia8602025.zip
    unzip /tmp/eia860.zip -d /tmp/eia860
    curl -L -A "Mozilla/5.0" -o /tmp/eia860m/july_generator2026.xlsx \
        https://www.eia.gov/electricity/data/eia860m/xls/july_generator2026.xlsx
    python3 scripts/eia860m_add_brand_new_plants.py \
        --gppd /tmp/gppd.csv \
        --solar /tmp/eia860/3_3_Solar_Y2025.xlsx --wind /tmp/eia860/3_2_Wind_Y2025.xlsx \
        --generators /tmp/eia860m/july_generator2026.xlsx
"""
import argparse
import importlib.util
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DST_DIR = os.path.join(REPO_ROOT, "datacore", "powerplants")
REGISTRY_PATH = os.path.join(DST_DIR, "us_power_plants.json")


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mpc = _load("eia860_missing_plants_check", "eia860_missing_plants_check.py")
eia860_capacity_by_code = _mpc.eia860_capacity_by_code
load_eia860_generator_rows = _mpc.load_eia860_generator_rows

_amp = _load("eia860_add_missing_plants", "eia860_add_missing_plants.py")
gppd_all_usa_codes = _amp.gppd_all_usa_codes
load_gppd_country_idnr_rows = _amp.load_gppd_country_idnr_rows
merge_registry = _amp.merge_registry

FUEL_CODES = {"SUN": "solar", "WND": "wind"}
OPERATING_STATUS_PREFIX = "(OP)"


def eia860m_plants_by_fuel(rows, fuel_codes=FUEL_CODES):
    """rows: iterable of (plant_id, plant_name, entity_name, ba_code,
    energy_source_code, status, nameplate_mw, lat, lon) straight off
    EIA-860M's own "Operating" sheet (one row per generator — a plant with
    several generators of the same fuel yields several rows sharing the
    same plant_id). Returns {fuel: {plant_id: (name, entity, ba, lat, lon,
    total_mw)}} with capacity summed across every generator row at that
    plant_id, name/entity/ba/lat/lon taken from the FIRST row seen for that
    plant_id (EIA-860M repeats these per-generator, not per-plant, but they
    are plant-level facts and are identical across a plant's own rows in
    practice). A row with an unmapped Energy Source Code, a Status not
    starting "(OP)", or a missing plant_id is excluded (never guessed)."""
    out = defaultdict(dict)
    for plant_id, name, entity, ba, esc, status, mw, lat, lon in rows:
        fuel = fuel_codes.get(esc)
        if fuel is None:
            continue
        if not status or not str(status).startswith(OPERATING_STATUS_PREFIX):
            continue
        if plant_id is None:
            continue
        bucket = out[fuel]
        if plant_id not in bucket:
            bucket[plant_id] = [name, entity, ba, lat, lon, 0.0]
        bucket[plant_id][5] += (mw or 0.0)
    return {fuel: {pid: tuple(v) for pid, v in plants.items()} for fuel, plants in out.items()}


def build_brand_new_plant_rows(fuel, eia860m_plants, annual_codes, gppd_codes):
    """Pure row-builder. eia860m_plants: {plant_id: (name, entity, ba, lat,
    lon, mw)} for one fuel, from eia860m_plants_by_fuel. annual_codes: set
    of EIA-860 ANNUAL plant codes for this fuel (eia860_capacity_by_code's
    keys). gppd_codes: WRI GPPD's full USA code set, any fuel
    (gppd_all_usa_codes's return value). Emits a registry-format row
    [name, capacity_mw, fuel, owner, lat, lon, verified=0] for every
    plant_id present in EIA-860M but absent from BOTH annual_codes and
    gppd_codes. Skips non-positive capacity or missing coordinates,
    counting both so the caller can report data completeness rather than
    silently dropping rows. Returns (rows, skipped_cap, skipped_coords)."""
    rows, skipped_cap, skipped_coords = [], 0, 0
    for plant_id in sorted(eia860m_plants):
        if plant_id in annual_codes or plant_id in gppd_codes:
            continue
        name, entity, ba, lat, lon, mw = eia860m_plants[plant_id]
        if mw <= 0:
            skipped_cap += 1
            continue
        if lat is None or lon is None:
            skipped_coords += 1
            continue
        rows.append([
            (name or f"EIA Plant {plant_id}").strip()[:60],
            round(mw, 1),
            fuel,
            (entity or "").strip()[:60],
            round(float(lat), 4),
            round(float(lon), 4),
            0,
        ])
    return rows, skipped_cap, skipped_coords


def split_top_n_unverifiable(existing_plants, new_rows, top_n=100):
    """Every row this script adds carries verified=0 by construction (see
    module docstring) — it is EIA-860M-sourced, has no gppd_idnr, and so
    structurally cannot be added to imagery_verified.json's id-keyed audit
    list the way a GPPD-sourced plant can (build_powerplants.py's own
    build_plants() sets row[6] from `idnr in verified`; this script's rows
    never pass through that loop at all). server/powerplants.test.ts
    enforces "every one of the top-100-by-MW plants is imagery-verified"
    as a hard invariant (verified_count == 100, exact). Appending an
    unverified row large enough to rank inside the MERGED registry's own
    top-N would silently break that invariant. A new row's capacity need
    only be compared against the PRE-merge top-N cutoff (the N-th largest
    EXISTING plant): if it exceeds that, at most N-1 existing plants can
    out-rank it, so it is guaranteed to land inside the merged top-N
    regardless of how many other new rows are added alongside it — a
    correct, if conservative for very large batches, one-pass check with
    no need to re-sort after every hypothetical insertion. Returns
    (safe_rows, held_back_rows) — held_back rows are NOT dropped; the
    caller reports them so a future session can imagery-verify (or
    formally extend the verification mechanism to non-GPPD rows) before
    they ship."""
    if len(existing_plants) < top_n:
        return list(new_rows), []
    cutoff = sorted((p[1] for p in existing_plants), reverse=True)[top_n - 1]
    safe, held_back = [], []
    for row in new_rows:
        (held_back if row[1] > cutoff else safe).append(row)
    return safe, held_back


def load_eia860m_operating_rows(xlsx_path):
    """EIA-860M "Operating" sheet -> generator yielding (plant_id,
    plant_name, entity_name, ba_code, energy_source_code, status,
    nameplate_mw, lat, lon). Header-lookup-by-name, same convention as
    every sibling eia860_*.py script; a superset of
    eia860m_recent_capacity_check.py's own loader (that script only needs
    ba/esc/status/mw — this one additionally needs plant identity and
    coordinates to build new registry rows, not just sum capacity)."""
    import openpyxl  # session-run only, same convention as sibling scripts
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Operating"]
    rows_iter = ws.iter_rows(values_only=True)
    next(rows_iter)  # title row
    next(rows_iter)  # blank row
    hdr = next(rows_iter)
    i_pid = hdr.index("Plant ID")
    i_pname = hdr.index("Plant Name")
    i_entity = hdr.index("Entity Name")
    i_ba = hdr.index("Balancing Authority Code")
    i_esc = hdr.index("Energy Source Code")
    i_status = hdr.index("Status")
    i_cap = hdr.index("Nameplate Capacity (MW)")
    i_lat = hdr.index("Latitude")
    i_lon = hdr.index("Longitude")

    def _rows():
        for row in rows_iter:
            if row is None:
                continue
            yield (row[i_pid], row[i_pname], row[i_entity], row[i_ba],
                   row[i_esc], row[i_status], row[i_cap], row[i_lat], row[i_lon])

    return _rows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gppd", required=True, help="path to global_power_plant_database.csv")
    ap.add_argument("--solar", required=True, help="path to EIA-860 ANNUAL 3_3_Solar_Y<year>.xlsx")
    ap.add_argument("--wind", required=True, help="path to EIA-860 ANNUAL 3_2_Wind_Y<year>.xlsx")
    ap.add_argument("--generators", required=True, help="path to EIA-860M generatorYYYY.xlsx")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write the registry file")
    args = ap.parse_args()

    gppd_codes = gppd_all_usa_codes(load_gppd_country_idnr_rows(args.gppd))
    annual_codes = {
        "solar": set(eia860_capacity_by_code(load_eia860_generator_rows(args.solar))),
        "wind": set(eia860_capacity_by_code(load_eia860_generator_rows(args.wind))),
    }
    eia860m_plants = eia860m_plants_by_fuel(load_eia860m_operating_rows(args.generators))

    added_rows = []
    per_fuel = []
    for fuel in ("solar", "wind"):
        plants = eia860m_plants.get(fuel, {})
        rows, skipped_cap, skipped_coords = build_brand_new_plant_rows(
            fuel, plants, annual_codes.get(fuel, set()), gppd_codes)
        added_rows.extend(rows)
        per_fuel.append({
            "fuel": fuel,
            "eia860m_plant_codes_total": len(plants),
            "absent_from_eia860_annual_and_gppd": len(rows) + skipped_cap + skipped_coords,
            "rows_added": len(rows),
            "skipped_zero_or_neg_capacity": skipped_cap,
            "skipped_missing_coords": skipped_coords,
            "added_capacity_mw": round(sum(r[1] for r in rows), 1),
        })

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    before_count = registry["count"]
    safe_rows, held_back_rows = split_top_n_unverifiable(registry["plants"], added_rows)
    merged = merge_registry(registry["plants"], safe_rows)

    report = {
        "check": "eia860m_add_brand_new_plants",
        "registry_plants_before": before_count,
        "registry_plants_after": len(merged),
        "rows_added_total": len(added_rows),
        "rows_shipped": len(safe_rows),
        "rows_held_back_top100_unverifiable": [
            {"name": r[0], "capacity_mw": r[1], "fuel": r[2]} for r in held_back_rows
        ],
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
        + " SUPPLEMENTED AGAIN by scripts/eia860m_add_brand_new_plants.py "
          "(2026-09-14): plants EIA-860M's Operating sheet carries (solar/"
          "wind, in-service, positive nameplate) that are absent from BOTH "
          "EIA-860 ANNUAL and GPPD entirely — commissioned after the "
          "ANNUAL snapshot's cutoff — are appended with verified=0, "
          "sourced (membership, capacity, coordinates) from EIA-860M "
          "alone; a row large enough to rank inside the top-100-by-MW is "
          "held back rather than shipped unverified (no gppd_idnr to "
          "imagery-verify against) — see that script's own module "
          "docstring for the full method and this run's held-back list."
    )
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {REGISTRY_PATH} ({os.path.getsize(REGISTRY_PATH) // 1024} KB)")


if __name__ == "__main__":
    main()
