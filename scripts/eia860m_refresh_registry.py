#!/usr/bin/env python3
"""eia860m_refresh_registry.py — ships the FUSION HYPOTHESIS (b) NEXT
filed by the 2026-09-12 session (research/open_questions.md, "registry
EIA-860M freshness refresh"): the registry
(datacore/powerplants/us_power_plants.json) is built once from EIA-860
ANNUAL and never updated between annual releases, so a fast-growing fuel
(solar being the clearest case) accumulates a growing capacity gap that
eia860m_recent_capacity_check.py (2026-09-12) measured but did not fix —
that script found ERCO solar capacity grew 30,050.7 -> 32,269.4 MW
(+2,218.7 MW) and SWPP solar grew 1,345.0 -> 2,078.1 MW (+733.1 MW)
between the EIA-860 ANNUAL snapshot (as-of 2024-12-31) and EIA-860M's
July-2026 vintage, real capacity growth the registry cannot see, causing
false "exceeds capacity" gate-1 alarms in grid_generation_gate1_ba.py.

THIS IS DIFFERENT FROM eia860_add_missing_plants.py (2026-09-11), which
added whole plants GPPD has no row for at all. This script instead
PATCHES CAPACITY on plants the registry already carries, using the join
key build_powerplants.py's own build_plants() already establishes:
GPPD's gppd_idnr for a USA row is "USA"+<EIA Plant Code>, a clean 1:1 key
for ~99.6% of USA GPPD rows (see eia860_missing_plants_check.py's
gppd_plant_code(), the same logic factored out). There is no safe way to
patch capacity on the already-built JSON in place (the registry row
format [name, capacity_mw, fuel, owner, lat, lon, verified] never stores
the plant code — see build_powerplants.py's own header comment), so this
script REBUILDS the registry via build_powerplants.build_plants()
(reusing it, not re-implementing GPPD CSV parsing) with a new
capacity_override parameter applied at construction time, exactly the
same way that function's existing EIA-coordinate override already works.
NEVER attempt a name/lat-lon fuzzy match to reconnect an existing JSON
row to a plant code instead — research/position_audit_2026-07-18.md's
"Hardeeville lesson" is this codebase's standing warning against exactly
that class of mistake. For the SAME reason, the missing-plants
supplement (below) is also fully RECOMPUTED from raw sources each run,
never spliced out of the prior registry file by inference — the current
registry row format carries no marker distinguishing a GPPD-sourced row
from an EIA-860-added one, so there is no safe way to identify the
existing supplement rows without a join key either.

SCOPE, deliberately narrow (matches this repo's existing EIA-860M
precedent script and the filed hypothesis): solar + wind only, the two
fuels EIA-860M's "Operating" sheet lets us query natively via Energy
Source Code (SUN/WND) without a further code-to-fuel lookup. Not
generalized to other fuels in this PR.

UPDATE (2026-09-13, second same-date session — resolves the NEXT this
module docstring filed above): the missing-plant rows
eia860_add_missing_plants.py appends (plants GPPD has no row for under
any fuel, ~4,339 solar/wind rows) are now ALSO refreshed from EIA-860M
where it covers them, closing exactly the gap the paragraph below
originally left open. `build_missing_plants_supplement` now takes an
optional `capacity_override_by_fuel_code` ({fuel: {plant_code: mw}}) —
built here in `main()` from the SAME `capacity_override` map already
fetched for the GPPD-matched base (no second EIA-860M fetch, EDGE
DOCTRINE #3) — and forwards the per-fuel slice straight through to
`eia860_add_missing_plants.build_missing_plant_rows`'s own new
`capacity_override_by_code` parameter (that function's own docstring
has the full contract: EIA-860 ANNUAL still decides WHICH codes are
missing-from-GPPD; EIA-860M, when it covers a code, only replaces the
CAPACITY VALUE applied to it — a code EIA-860M's Operating sheet does
not carry still falls back to EIA-860 ANNUAL's own figure, never drops
to zero). This was analyzed, not assumed, to be the dominant residual
lever for ERCO/SWPP per the live cross-check filed in
research/open_questions.md 2026-09-13 (25%/35% of the plant IDs driving
ERCO's/SWPP's EIA-860M growth exist in GPPD at all — the rest are
exactly this missing-plants population).

ORIGINAL SCOPE NOTE (2026-09-13, first session, kept for the record —
now superseded by the UPDATE above): this script recomputed and
re-applied that exact supplement unchanged (via that script's own
functions, reused through the same importlib pattern that script
itself already uses for ITS dependencies) rather than refreshing those
rows from EIA-860M too. That residual staleness was smaller (those
plants are new-to-GPPD, not necessarily new-to-the-grid) and was filed
as a NEXT item rather than fixed in that first PR — one logical change
per PR.

WHAT SHIPS: rebuilds the GPPD-sourced base plant list via
build_powerplants.build_plants(), with capacity_override drawn from
EIA-860M's July-2026 "Operating" sheet for the (plant_code, fuel) pairs
it covers, then re-applies the missing-plants supplement (recomputed
fresh from EIA-860 ANNUAL's Schedule 2/3 files, exactly as
eia860_add_missing_plants.py's own main() does) so the registry does not
regress from 14,172 rows back to the ~9,833-row GPPD-only base.
Idempotent/reproducible from raw sources — re-running it from a fresh
pull does not depend on, or compound onto, the previously shipped
datacore/powerplants/us_power_plants.json.

ARGS NOTE: the filed hypothesis's own re-run sketch listed only
--gppd/--plants/--generators; recomputing the missing-plants supplement
from EIA-860 ANNUAL (as this module docstring above states, deliberately
not re-deriving it by inference from the undifferentiated prior registry
file, per the Hardeeville-lesson reasoning) additionally requires that
same annual release's Schedule 3 solar/wind generator files — the same
--solar/--wind eia860_add_missing_plants.py already takes. Documented
here rather than silently added, since it deviates from that sketch.

Re-run (files not fetched automatically, same manual-download precedent
every sibling EIA-860 script in this directory already documents):
    curl -L -o /tmp/gppd.csv https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv
    curl -L -o /tmp/eia860.zip https://www.eia.gov/electricity/data/eia860/xls/eia8602025.zip
    unzip /tmp/eia860.zip -d /tmp/eia860
    curl -L -A "Mozilla/5.0" -o /tmp/eia860m/july_generator2026.xlsx https://www.eia.gov/electricity/data/eia860m/xls/july_generator2026.xlsx
    python3 scripts/eia860m_refresh_registry.py \
        --gppd /tmp/gppd.csv --plants /tmp/eia860/2___Plant_Y2025.xlsx \
        --generators /tmp/eia860m/july_generator2026.xlsx \
        --solar /tmp/eia860/3_3_Solar_Y2025.xlsx --wind /tmp/eia860/3_2_Wind_Y2025.xlsx

NOTE (verified live 2026-09-13): august_generator2026.xlsx and
september_generator2026.xlsx currently return HTTP 200 but are soft-404
HTML (EIA has not published them yet as of this session) —
july_generator2026.xlsx (13.9MB, real xlsx, confirmed this session) is
the actual latest live file. Re-verify this when re-running; if a newer
month is now live, use it and note the vintage in the report instead of
assuming July.
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


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bpp = _load_sibling("build_powerplants")
_amp = _load_sibling("eia860_add_missing_plants")
_mrc = _load_sibling("eia860m_recent_capacity_check")  # reuses parse_as_of_period only

ENERGY_SOURCE_TO_FUEL = {"SUN": "solar", "WND": "wind"}
OPERATING_STATUS_PREFIX = "(OP)"


def eia860m_capacity_by_plant_fuel(rows, fuel_codes=ENERGY_SOURCE_TO_FUEL):
    """rows: iterable of (plant_id, energy_source_code, status,
    nameplate_mw) straight off EIA-860M's own "Operating" sheet. Returns
    ({(plant_id, fuel): total_mw}, skipped_bad_plant_id), summed across
    every generator row at that plant+fuel (a plant can have multiple
    generators of the same fuel). Excludes (never guesses at): a row
    with an unmapped Energy Source Code, a missing/blank Plant ID, or a
    Status not starting "(OP)" (Operating) — the same
    OPERATING_STATUS_PREFIX convention eia860m_recent_capacity_check.py
    already uses, so "(OA)"/"(OS)" (temporarily/indefinitely out of
    service) rows are excluded here too. A Plant ID that survives the
    blank check but still isn't int-coercible (a genuine data anomaly,
    not the expected missing/blank case) is COUNTED rather than
    silently dropped — a swallowed exception is how a broken pipeline
    keeps reporting success (CLAUDE.md)."""
    out = defaultdict(float)
    skipped_bad_plant_id = 0
    for plant_id, esc, status, mw in rows:
        fuel = fuel_codes.get(esc)
        if fuel is None:
            continue
        if not status or not str(status).startswith(OPERATING_STATUS_PREFIX):
            continue
        if plant_id is None or (isinstance(plant_id, str) and not plant_id.strip()):
            continue
        try:
            code = int(plant_id)
        except (TypeError, ValueError):
            skipped_bad_plant_id += 1
            continue
        out[(code, fuel)] += (mw or 0.0)
    return dict(out), skipped_bad_plant_id


def load_eia860m_operating_plant_rows(xlsx_path):
    """EIA-860M "Operating" sheet -> (as_of_period, rows) where rows is a
    generator that yields (plant_id, energy_source_code, status,
    nameplate_mw) tuples. Header-lookup-by-name, same convention as
    every sibling eia860_*.py script; reuses
    eia860m_recent_capacity_check.parse_as_of_period for the as-of-period
    parse rather than re-implementing it — the sheet-navigation
    boilerplate (skip title row, skip blank row, header lookup) is
    necessarily duplicated from that sibling script's own
    load_eia860m_operating_rows since the column SET differs (this one
    also needs Plant ID, that one needs Balancing Authority Code)."""
    import openpyxl  # session-run only, same convention as sibling scripts
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Operating"]
    rows_iter = ws.iter_rows(values_only=True)
    title_row = next(rows_iter)
    as_of = _mrc.parse_as_of_period(title_row[0] if title_row else None)
    next(rows_iter)  # blank row
    hdr = next(rows_iter)
    i_pid = hdr.index("Plant ID")
    i_esc = hdr.index("Energy Source Code")
    i_status = hdr.index("Status")
    i_cap = hdr.index("Nameplate Capacity (MW)")

    def _rows():
        for row in rows_iter:
            if row is None:
                continue
            yield (row[i_pid], row[i_esc], row[i_status], row[i_cap])

    return as_of, _rows()


def capacity_delta_report(before_plants, after_plants, fuels=("solar", "wind")):
    """Pure summarizer: national totals + row counts + MW delta by fuel,
    comparing two registry-format plant lists (same row schema). Used to
    self-verify the refresh against the numbers already on record in
    open_questions.md (ERCO/SWPP growth is a strict subset of the
    national solar delta) without re-deriving them."""
    out = []
    for fuel in fuels:
        before = [p for p in before_plants if p[2] == fuel]
        after = [p for p in after_plants if p[2] == fuel]
        before_mw = round(sum(p[1] for p in before), 1)
        after_mw = round(sum(p[1] for p in after), 1)
        out.append({
            "fuel": fuel,
            "rows_before": len(before),
            "rows_after": len(after),
            "mw_before": before_mw,
            "mw_after": after_mw,
            "mw_delta": round(after_mw - before_mw, 1),
        })
    return out


def build_missing_plants_supplement(gppd_path, plants_xlsx, solar_xlsx, wind_xlsx,
                                     capacity_override_by_fuel_code=None):
    """Recomputes eia860_add_missing_plants.py's own supplement fresh
    from raw sources (never spliced out of the prior registry file — see
    module docstring). Returns (rows, per_fuel_report).

    capacity_override_by_fuel_code (2026-09-13, second same-date update):
    optional {fuel: {plant_code: mw}} — when a missing plant's code has a
    positive entry here, its row is built with THIS capacity (a more
    current EIA-860M reading) instead of EIA-860 ANNUAL's own figure;
    membership (which codes are missing-from-GPPD) is unaffected, exactly
    per eia860_add_missing_plants.build_missing_plant_rows's own contract.
    Default None reproduces the original EIA-860-ANNUAL-only behavior
    exactly. The per-fuel report also now counts how many of this fuel's
    added rows actually got an EIA-860M-refreshed capacity, so the
    refresh's real coverage is visible rather than assumed."""
    all_usa_codes = _amp.gppd_all_usa_codes(_amp.load_gppd_country_idnr_rows(gppd_path))
    plant_directory = _amp.load_eia860_plant_directory(plants_xlsx)
    rows, report = [], []
    for fuel, path in (("solar", solar_xlsx), ("wind", wind_xlsx)):
        eia_cap = _amp.eia860_capacity_by_code(_amp.load_eia860_generator_rows(path))
        missing = set(eia_cap) - all_usa_codes
        override_this_fuel = (capacity_override_by_fuel_code or {}).get(fuel, {})
        fuel_rows, skipped_cap, skipped_coords = _amp.build_missing_plant_rows(
            fuel, missing, eia_cap, plant_directory,
            capacity_override_by_code=override_this_fuel)
        # Candidate count, not a promise every one became a row (a code
        # can still be skipped for bad coords after its capacity source
        # changes) — named accordingly so the report never overstates
        # how many rows actually shipped with a refreshed figure.
        missing_codes_matched_in_eia860m = (
            len(missing & set(override_this_fuel)) if override_this_fuel else 0)
        rows.extend(fuel_rows)
        report.append({
            "fuel": fuel, "rows_added": len(fuel_rows),
            "skipped_zero_or_neg_capacity": skipped_cap,
            "skipped_bad_coords_or_no_directory_entry": skipped_coords,
            "missing_codes_matched_in_eia860m": missing_codes_matched_in_eia860m,
        })
    return rows, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gppd", required=True, help="path to global_power_plant_database.csv")
    ap.add_argument("--plants", required=True, help="path to EIA-860 ANNUAL 2___Plant_Y<year>.xlsx (coordinates + missing-plant directory)")
    ap.add_argument("--generators", required=True, help="path to EIA-860M generatorYYYY.xlsx (the capacity refresh source)")
    ap.add_argument("--solar", required=True, help="path to EIA-860 ANNUAL 3_3_Solar_Y<year>.xlsx (missing-plants supplement)")
    ap.add_argument("--wind", required=True, help="path to EIA-860 ANNUAL 3_2_Wind_Y<year>.xlsx (missing-plants supplement)")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write the registry file")
    args = ap.parse_args()

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        prior_registry = json.load(f)
    before_plants = prior_registry["plants"]

    # EIA-860 ANNUAL coordinates (reuse build_powerplants.load_eia()) +
    # verified/overrides side-files, same as build_powerplants.main().
    eia_coords = _bpp.load_eia(args.plants)
    verified_path = os.path.join(DST_DIR, "imagery_verified.json")
    verified = set(json.load(open(verified_path))["ids"]) if os.path.exists(verified_path) else set()
    overrides_path = os.path.join(DST_DIR, "position_overrides.json")
    overrides = {}
    if os.path.exists(overrides_path):
        overrides = {o["gppd_idnr"]: o for o in json.load(open(overrides_path))["overrides"]}

    # EIA-860M capacity override, solar+wind only.
    as_of, m_rows = load_eia860m_operating_plant_rows(args.generators)
    capacity_override, capacity_override_skipped = eia860m_capacity_by_plant_fuel(m_rows)

    # Rebuild the GPPD-sourced BASE via build_powerplants.build_plants(),
    # with the EIA-860M capacity override applied at construction time.
    base_plants, eia_used, overrides_used = _bpp.build_plants(
        args.gppd, eia_coords, verified, overrides, capacity_override=capacity_override)

    # Re-apply the missing-plants supplement, NOW ALSO capacity-refreshed
    # from EIA-860M where it covers a missing code (2026-09-13 second
    # same-date update — see module docstring). Reuses the SAME
    # capacity_override map already fetched above for the GPPD-matched
    # base, reshaped from {(code, fuel): mw} to {fuel: {code: mw}} — no
    # second EIA-860M fetch (EDGE DOCTRINE #3).
    capacity_override_by_fuel_code = defaultdict(dict)
    for (code, fuel), mw in capacity_override.items():
        capacity_override_by_fuel_code[fuel][code] = mw
    missing_rows, missing_report = build_missing_plants_supplement(
        args.gppd, args.plants, args.solar, args.wind,
        capacity_override_by_fuel_code=capacity_override_by_fuel_code)

    merged = _amp.merge_registry(base_plants, missing_rows)

    report = {
        "check": "eia860m_refresh_registry",
        "eia860m_as_of": as_of,
        "registry_plants_before": prior_registry["count"],
        "registry_plants_after": len(merged),
        "eia_coords_used": eia_used,
        "position_overrides_used": overrides_used,
        "capacity_override_pairs_from_eia860m": len(capacity_override),
        "capacity_override_skipped_bad_plant_id": capacity_override_skipped,
        "missing_plants_supplement": missing_report,
        "capacity_by_fuel": capacity_delta_report(before_plants, merged),
    }
    print(json.dumps(report, indent=2))

    if args.dry_run:
        return

    registry = dict(prior_registry)
    registry["plants"] = merged
    registry["count"] = len(merged)
    registry["verified_count"] = sum(p[6] for p in merged)
    total_matched = sum(r["missing_codes_matched_in_eia860m"] for r in missing_report)
    total_missing_rows = sum(r["rows_added"] for r in missing_report)
    registry["_doc"] = (
        registry["_doc"]
        + f" RE-REFRESHED by scripts/eia860m_refresh_registry.py (2026-09-13, "
          f"second same-date run, EIA-860M as-of {as_of}): extends the "
          "first run's GPPD-matched-plant capacity refresh to ALSO cover "
          "the missing-plants supplement (plants GPPD has no row for at "
          f"all) — {total_matched} of {total_missing_rows} solar/wind "
          "missing-plant rows now carry an EIA-860M-current capacity "
          "instead of the ~20-month-lagged EIA-860 ANNUAL figure. See "
          "this script's own module docstring and "
          "research/open_questions.md's FUSION HYPOTHESIS (b) thread for "
          "the full method and the live ERCO/SWPP gate-1 re-check this "
          "was built to close."
    )
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {REGISTRY_PATH} ({os.path.getsize(REGISTRY_PATH) // 1024} KB)")


if __name__ == "__main__":
    main()
