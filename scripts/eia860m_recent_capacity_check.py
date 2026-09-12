#!/usr/bin/env python3
"""eia860m_recent_capacity_check.py — the REGISTRY-STALENESS follow-up to
scripts/eia860_regional_capacity_check.py's own NEXT(1), which ruled OUT a
residual registry-completeness gap for ERCO/SWPP solar (registry already
matches EIA-860 2025 ANNUAL's own regional totals within noise) but left
open what IS producing the gate-1 overshoots (ERCO 1.076x, SWPP ~2x,
scripts/grid_generation_gate1_ba.py --source eia860).

HYPOTHESIS TESTED: EIA-860 ANNUAL is not stale in the sense of "missing
plants" (that was the 2026-09-11 national fix's own finding, and this
session confirmed it holds regionally too) — it is stale in the sense of
DATE. The "2025" annual release reports capacity operating as of
2024-12-31; anything commissioned since then is invisible to it and to
this repo's registry (which is built from that same annual file). EIA
separately publishes Form EIA-860M, a MONTHLY update to the same generator
inventory, roughly 2 months behind real time. If ERCO/SWPP solar buildout
since 2024-12-31 accounts for most or all of the gate-1 overshoot, that is
a currency problem, not a completeness or EIA-930-measurement problem —
a structurally different, and more actionable, finding.

METHOD: EIA-860M's single "Operating" sheet carries "Balancing Authority
Code" AND "Energy Source Code" natively on every generator row — unlike
the ANNUAL EIA-860's Schedule 2 (Plant) / Schedule 3 (fuel-specific
generator) split, no Plant-Code join is needed here at all; this is a
straight groupby. Only rows whose Status starts "(OP)" (Operating) count
— EIA-860M's Operating sheet is already filtered to in-service units by
its own title, but a small number of "(OA)"/"(OS)" (temporarily/
indefinitely out of service) rows are also present and are excluded here
to match gate-1's own "installed, in-service capacity" semantics.

Usage (file not fetched automatically, same manual-download precedent
every sibling eia860_*.py script documents):
    curl -L -A "Mozilla/5.0" \
        -o /tmp/eia860m/july_generator2026.xlsx \
        https://www.eia.gov/electricity/data/eia860m/xls/july_generator2026.xlsx
    python3 scripts/eia860m_recent_capacity_check.py \
        --generators /tmp/eia860m/july_generator2026.xlsx \
        --bas ERCO,SWPP \
        --eia-max solar:ERCO=32327.0,SWPP=2650.0

The --eia-max values come from grid_generation_gate1_ba.py's own live
"max_generation_mwh" output (this script does not re-fetch EIA-930 itself
— that is that script's job, EDGE DOCTRINE #3) and are optional: without
them this script still reports the registry-vs-EIA-860M staleness gap on
its own, just without recomputing what fraction of the gate-1 overshoot it
explains.
"""
import argparse
import importlib.util
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
BA_JOIN_PATH = os.path.join(REPO_ROOT, "datacore", "powerplants", "plant_balancing_authority_eia860.json")

_join_spec = importlib.util.spec_from_file_location(
    "grid_ba_eia860_join", os.path.join(HERE, "grid_ba_eia860_join.py"))
_join = importlib.util.module_from_spec(_join_spec)
_join_spec.loader.exec_module(_join)
registry_capacity_by_ba = _join.registry_capacity_by_ba

DEFAULT_BAS = ("ERCO", "SWPP")
ENERGY_SOURCE_TO_FUEL = {"SUN": "solar", "WND": "wind"}
OPERATING_STATUS_PREFIX = "(OP)"


def eia860m_capacity_by_ba(rows, fuel_codes=ENERGY_SOURCE_TO_FUEL):
    """rows: iterable of (ba_code, energy_source_code, status, nameplate_mw)
    straight off EIA-860M's own "Operating" sheet. Returns
    {fuel: {ba_code: mw}}. A row with an unmapped Energy Source Code, a
    missing/empty BA code, or a Status not starting "(OP)" is excluded
    (never guessed) — see module docstring for why."""
    cap = defaultdict(lambda: defaultdict(float))
    for ba, esc, status, mw in rows:
        fuel = fuel_codes.get(esc)
        if fuel is None:
            continue
        if not status or not str(status).startswith(OPERATING_STATUS_PREFIX):
            continue
        if not ba or not str(ba).strip():
            continue
        cap[fuel][str(ba).strip()] += (mw or 0.0)
    return {fuel: dict(bas) for fuel, bas in cap.items()}


def parse_as_of_period(title):
    """title: the sheet's own first-row label, e.g. "Inventory of
    Operating Generators as of July 2026". Returns the trailing
    "Month YYYY" substring, or None if the title doesn't match the
    expected EIA-860M format — surfaced as provenance, never guessed at."""
    if not title:
        return None
    m = re.search(r"as of\s+(\w+\s+\d{4})\s*$", str(title).strip())
    return m.group(1) if m else None


def compare_staleness(eia860m_mw, registry_mw, ba, fuel, eia930_max_mwh=None):
    """registry_mw: this repo's registry capacity for (ba, fuel), sourced
    from EIA-860 ANNUAL (2025 vintage, as-of 2024-12-31). eia860m_mw:
    EIA-860M's own more-current reading for the same (ba, fuel).
    growth_ratio > 1 means real capacity has grown since the annual
    snapshot the registry was built from. When eia930_max_mwh is supplied
    (grid_generation_gate1_ba.py's own live gate-1 reading), also computes
    what the gate-1 overshoot ratio WOULD be against the more-current
    EIA-860M capacity instead of the stale registry figure — this is the
    number that answers "does registry staleness explain the gate-1 FAIL,
    fully, partially, or not at all"."""
    growth_ratio = round(eia860m_mw / registry_mw, 3) if registry_mw else None
    out = {
        "ba": ba,
        "fuel": fuel,
        "registry_mw_stale_eia860_annual": round(registry_mw, 1),
        "eia860m_mw_current": round(eia860m_mw, 1),
        "capacity_growth_since_annual_snapshot_mw": round(eia860m_mw - registry_mw, 1),
        "capacity_growth_ratio": growth_ratio,
    }
    if eia930_max_mwh is not None:
        ratio_vs_stale = round(eia930_max_mwh / registry_mw, 3) if registry_mw else None
        ratio_vs_current = round(eia930_max_mwh / eia860m_mw, 3) if eia860m_mw else None
        out["eia930_max_mwh"] = round(eia930_max_mwh, 1)
        out["gate1_ratio_vs_stale_registry"] = ratio_vs_stale
        out["gate1_ratio_vs_current_eia860m"] = ratio_vs_current
    return out


def load_eia860m_operating_rows(xlsx_path):
    """EIA-860M "Operating" sheet -> (as_of_period, rows) where rows is a
    generator that yields (ba_code, energy_source_code, status,
    nameplate_mw) tuples. Header-lookup-by-name, same convention as every
    sibling eia860_*.py script; unlike those, no Plant-Code join is needed
    (see module docstring)."""
    import openpyxl  # session-run only, same convention as sibling scripts
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Operating"]
    rows_iter = ws.iter_rows(values_only=True)
    title_row = next(rows_iter)
    as_of = parse_as_of_period(title_row[0] if title_row else None)
    next(rows_iter)  # blank row
    hdr = next(rows_iter)
    i_ba = hdr.index("Balancing Authority Code")
    i_esc = hdr.index("Energy Source Code")
    i_status = hdr.index("Status")
    i_cap = hdr.index("Nameplate Capacity (MW)")

    def _rows():
        for row in rows_iter:
            if row is None:
                continue
            yield (row[i_ba], row[i_esc], row[i_status], row[i_cap])

    return as_of, _rows()


def _parse_eia_max_arg(spec):
    """"solar:ERCO=32327.0,SWPP=2650.0" -> {"solar": {"ERCO": 32327.0, "SWPP": 2650.0}}.
    Accepts multiple fuels separated by ";", e.g.
    "solar:ERCO=1,SWPP=2;wind:ISNE=3". Malformed input raises rather than
    silently dropping a value a report would then present as "not measured"."""
    out = {}
    if not spec:
        return out
    for fuel_block in spec.split(";"):
        fuel_block = fuel_block.strip()
        if not fuel_block:
            continue
        fuel, _, pairs = fuel_block.partition(":")
        fuel = fuel.strip()
        d = {}
        for pair in pairs.split(","):
            pair = pair.strip()
            if not pair:
                continue
            ba, _, val = pair.partition("=")
            d[ba.strip()] = float(val)
        out[fuel] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", required=True, help="path to EIA-860M generatorYYYY.xlsx")
    ap.add_argument("--bas", default=",".join(DEFAULT_BAS))
    ap.add_argument("--ba-join", default=BA_JOIN_PATH,
                     help="grid_ba_eia860_join.py's committed output (registry side ground truth)")
    ap.add_argument("--eia-max", default=None,
                     help='optional live gate-1 max_generation_mwh readings, e.g. '
                          '"solar:ERCO=32327.0,SWPP=2650.0" (see module docstring)')
    args = ap.parse_args()

    target_bas = [b.strip() for b in args.bas.split(",") if b.strip()]
    eia_max_by_fuel = _parse_eia_max_arg(args.eia_max)

    as_of, rows = load_eia860m_operating_rows(args.generators)
    eia860m_cap = eia860m_capacity_by_ba(rows)

    with open(args.ba_join) as f:
        ba_join = json.load(f)
    registry_ba_capacity, _ = registry_capacity_by_ba(ba_join["assignments"])

    results = []
    for fuel in sorted(eia860m_cap.keys()):
        eia_max_this_fuel = eia_max_by_fuel.get(fuel, {})
        for ba in target_bas:
            eia860m_mw = eia860m_cap.get(fuel, {}).get(ba, 0.0)
            registry_mw = registry_ba_capacity.get(ba, {}).get(fuel, 0.0)
            results.append(compare_staleness(
                eia860m_mw, registry_mw, ba, fuel,
                eia930_max_mwh=eia_max_this_fuel.get(ba)))

    report = {
        "check": "eia860m_recent_capacity_check",
        "note": "compares this repo's registry capacity (built from EIA-860 ANNUAL, "
                "as-of 2024-12-31 for the 2025 release) against EIA-860M's own more-"
                "current reading for the same (ba, fuel) pairs, to isolate how much of "
                "a gate-1 overshoot is explained by registry DATE staleness rather than "
                "a completeness gap or an EIA-930 measurement issue",
        "eia860m_as_of": as_of,
        "ba_join_source": os.path.relpath(args.ba_join, REPO_ROOT),
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
