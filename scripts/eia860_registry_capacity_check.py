#!/usr/bin/env python3
"""eia860_registry_capacity_check.py — quantifies the registry-staleness
finding from FUSION HYPOTHESIS (b) GATE 1 (grid_generation_gate1.py,
2026-09-11): that run found `datacore/powerplants/us_power_plants.json`'s
"solar" fuel bucket FAILING its own exceeds-capacity check (observed EIA-930
generation 3.085x the registry's stated solar capacity), read as a stale
registry snapshot (WRI GPPD, per build_powerplants.py's own header, is only
coordinate-refreshed from EIA-860, not capacity-refreshed) rather than an
EIA-930 defect, since every other fuel bucket passed comfortably. This
script is that read's own NEXT step, made concrete: instead of inferring
staleness indirectly from an hourly-generation ceiling breach, it compares
the registry's bucket total DIRECTLY against EIA-860's own authoritative
per-fuel nameplate capacity — the same source `build_powerplants.py` already
calls "authoritative" for plant coordinates but has never used for capacity.

PRIOR (stated before running against real EIA-860 2025 data, REASONING
STANDARD #10): given the FUSION (b) run's 3.085x generation/capacity ratio
for solar and a plausible ~75-85% peak capacity factor for a fleet at solar
noon, expect EIA-860's true national solar nameplate total to land in the
neighborhood of 3.5-4.5x the registry's 37,468 MW figure (i.e. roughly
130,000-170,000 MW) — enough to explain the gate-1 breach as a registry gap,
not a further data anomaly. Wind is a SPOT-CHECK only (this session's own
NEXT(2) instruction) — no comparable breach was observed for wind in the
gate-1 run (0.706 ratio, PASS), so the prior for wind is that EIA-860 and
the registry are already reasonably close (no more than ~10-20% apart).

SCOPE, stated honestly: this compares NATIONAL TOTALS only (EIA-860 fleet
nameplate vs. registry bucket sum) — it does NOT attempt a per-plant join
between WRI GPPD's `gppd_idnr`-keyed registry entries and EIA-860's
Plant-Code-keyed generator rows. That join (needed to actually REFRESH each
solar plant's individual capacity value in the registry, the literal ask in
NEXT(2)) is deliberately NOT attempted here: GPPD does not carry an EIA
Plant Code field for a safe join key, and a fuzzy name/lat-lon match risks
exactly the kind of silent misattribution research/position_audit_2026-07-18.md's
"Hardeeville lesson" already warned this codebase about once. This script
answers "how stale, nationally, and is it plausible as the sole explanation"
— a real, defensible, well-posed question — and leaves the riskier per-plant
rebuild as its own scoped follow-up (filed in this session's log entry).

Data source: EIA-860 Schedule 3 "Wind Technology Data" / "Solar Technology
Data" generator-level files ("Operable Units Only" per their own sheet
title), status filtered to 'OP' (Operating) only — EIA-860 sometimes retains
a handful of 'OS'/'OA' (out-of-service) rows even on this "operable" sheet
(confirmed live in the 2025 file: 14 OS + 2 OA among 1,573 wind rows), which
should not count toward currently-installed nameplate capacity for an
exceeds-capacity comparison; solar's rows in the same live file were 100%
'OP' at check time, but the filter is applied identically to both fuels
rather than assumed.

Re-run (files not fetched automatically — same manual-download precedent
scripts/build_powerplants.py already documents for this exact zip):
    curl -L -o /tmp/eia860.zip https://www.eia.gov/electricity/data/eia860/xls/eia8602025.zip
    unzip /tmp/eia860.zip -d /tmp/eia860
    python3 scripts/eia860_registry_capacity_check.py \
        --solar /tmp/eia860/3_3_Solar_Y2025.xlsx --wind /tmp/eia860/3_2_Wind_Y2025.xlsx
(Swap in whatever year's zip is current; the EIA-860 landing page lists the
latest filename — this script does not hardcode a year.)
"""
import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
REGISTRY_PATH = os.path.join(REPO_ROOT, "datacore", "powerplants", "us_power_plants.json")

_spec = importlib.util.spec_from_file_location(
    "grid_generation_gate1", os.path.join(HERE, "grid_generation_gate1.py"))
_gg1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gg1)
registry_capacity_by_fuel = _gg1.registry_capacity_by_fuel

OPERATING_STATUS = "OP"


def sum_operable_nameplate(rows):
    """rows: iterable of (status, nameplate_mw) tuples, one per generator
    row. Returns (total_mw, rows_counted, rows_skipped_not_operating).
    A None/blank nameplate value counts as 0.0, never skipped silently —
    a generator that exists but reports no capacity is still evidence the
    plant exists, just not double-counted as capacity."""
    total = 0.0
    counted = 0
    skipped = 0
    for status, nameplate_mw in rows:
        if status != OPERATING_STATUS:
            skipped += 1
            continue
        total += nameplate_mw or 0.0
        counted += 1
    return total, counted, skipped


def load_eia860_nameplate_rows(xlsx_path):
    """Reads an EIA-860 Schedule 3 generator-level xlsx (Wind or Solar
    Technology Data) and yields (status, nameplate_mw) tuples. Skips the
    title row; the header row (row 2) is used to locate columns by name
    rather than assumed position, so a column reorder in a future EIA-860
    vintage fails loudly (KeyError) instead of silently misreading."""
    import openpyxl  # session-run only, same convention as build_powerplants.py
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    next(rows)  # title row
    hdr = next(rows)
    i_status = hdr.index("Status")
    i_cap = hdr.index("Nameplate Capacity (MW)")
    for row in rows:
        if row is None or row[i_status] is None:
            continue
        yield (row[i_status], row[i_cap])


def compare_fuel(fuel, registry_mw, eia860_mw):
    """Pure comparison — no verdict threshold imposed here (unlike
    grid_generation_gate1.py's exceeds-capacity check, this is a
    magnitude-of-staleness report, not a pass/fail gate); ratio > 1 means
    EIA-860 reports MORE capacity than the registry currently carries."""
    ratio = (eia860_mw / registry_mw) if registry_mw > 0 else None
    return {
        "fuel": fuel,
        "registry_capacity_mw": round(registry_mw, 1),
        "eia860_nameplate_mw": round(eia860_mw, 1),
        "ratio_eia860_over_registry": round(ratio, 3) if ratio is not None else None,
        "gap_mw": round(eia860_mw - registry_mw, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solar", required=True, help="path to EIA-860 3_3_Solar_Y<year>.xlsx")
    ap.add_argument("--wind", required=True, help="path to EIA-860 3_2_Wind_Y<year>.xlsx")
    args = ap.parse_args()

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    registry_cap, excluded = registry_capacity_by_fuel(registry["plants"])

    results = []
    for fuel, path in (("solar", args.solar), ("wind", args.wind)):
        total_mw, counted, skipped = sum_operable_nameplate(load_eia860_nameplate_rows(path))
        cmp = compare_fuel(fuel, registry_cap.get(fuel, 0.0), total_mw)
        cmp["eia860_generator_rows_counted"] = counted
        cmp["eia860_generator_rows_skipped_not_operating"] = skipped
        results.append(cmp)

    report = {
        "check": "eia860_registry_capacity_check",
        "registry_source": os.path.relpath(REGISTRY_PATH, REPO_ROOT),
        "registry_plants_total": registry.get("count"),
        "registry_plants_excluded_noncontiguous": excluded,
        "note": "national totals only — not a per-plant registry refresh; see module docstring SCOPE",
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
