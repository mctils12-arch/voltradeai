#!/usr/bin/env python3
"""grid_county_ba.py — GRID VISION A1 step 2: Texas county -> balancing
authority crosswalk from EIA-861 (2024 final), the join key between the
OSM capacity registry and per-BA EIA-930 demand.

Source chain (workup GV-A5, grid_vision_research.md Item 5, all
public-domain US-gov works, verified live 2026-07-07):
  utility -> BA: 'BA Code' over THREE files (Sales_Ult_Cust sheet
    'States' header row 3; Short_Form sheet '861S' header row 1;
    Delivery_Companies sheet 'Delivery Companies' header row 3 — the
    big ERCOT TDUs live ONLY here); drop utility 99999 ('Adjustment')
    and literal 'NA' BA codes. STATE-PREFERRED: for TX counties use
    the utility's TX-row BA codes; fall back to its other-state codes
    ONLY when it has no TX row (the Farmers-NM case: files under NM,
    serves TX counties). Verified 2026-07-07 first run: an any-state
    union smeared Rio Grande EC's NM-row EPE across its 18 TX
    counties whose TX row says ERCO — state-preferred restores the
    workup's independently computed EPE=3.
  county -> utilities: Service_Territory sheet 'Counties_States'
    (header row 1), State == TX.
  BA codes validated against EIA-930 reference tables sheet 'BAs' —
  same code space, no crosswalk needed.

HONESTY: EIA-861 measures RETAIL service territory, not grid topology
(a transmission asset in an SWPP-retail Panhandle county can be an
ERCOT asset). The artifact labels every assignment 'county retail BA';
NO primary BA is designated for multi-BA counties — 861 alone cannot
weight them (needs a polygon arbiter; EIA Atlas BA layer when its hub
migration finishes). Never guessed.

EXPECTED (stated before this script first ran, from the workup's
independently computed join): 254/254 TX counties covered; county
membership ERCO 218, SWPP 78, MISO 36, EPE 3, PNM 2; 80 multi-BA
counties. A mismatch is investigated, not accepted.

Usage: python3 scripts/grid_county_ba.py <dir-with-861-xlsx> \
    <EIA930_Reference_Tables.xlsx> [--out datacore/gridvision/tx_county_ba.json]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import openpyxl

TX_COUNTY_COUNT = 254


def sheet_rows(path, sheet, header_row):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[sheet]
    rows = ws.iter_rows(values_only=True)
    header = None
    for i, r in enumerate(rows, start=1):
        if i < header_row:
            continue
        if i == header_row:
            header = [str(c).split("\n")[0].strip() if c is not None else f"col{j}"
                      for j, c in enumerate(r)]
            continue
        if all(c is None for c in r):
            continue
        yield dict(zip(header, r))
    wb.close()


def utility_ba_map(dir861):
    """Utility Number -> {state: set(BA codes)} across the three files."""
    specs = [
        ("Sales_Ult_Cust_2024.xlsx", "States", 3),
        ("Short_Form_2024.xlsx", "861S", 1),
        ("Delivery_Companies_2024.xlsx", "Delivery Companies", 3),
    ]
    out = defaultdict(lambda: defaultdict(set))
    for fn, sheet, hrow in specs:
        for row in sheet_rows(os.path.join(dir861, fn), sheet, hrow):
            uid = str(row.get("Utility Number") or "").strip()
            ba = str(row.get("BA Code") or "").strip()
            state = str(row.get("State") or "").strip()
            if not uid or uid == "99999" or ba in ("", "NA", "None"):
                continue
            out[uid][state].add(ba)
    return out


def bas_for_state(u2ba, uid, state):
    """State-preferred lookup: the utility's codes filed FOR this state;
    any-state fallback only when it filed no row for this state."""
    per_state = u2ba.get(uid)
    if not per_state:
        return set()
    if per_state.get(state):
        return set(per_state[state])
    return set().union(*per_state.values())


def tx_county_utilities(dir861):
    out = defaultdict(set)
    path = os.path.join(dir861, "Service_Territory_2024.xlsx")
    for row in sheet_rows(path, "Counties_States", 1):
        if str(row.get("State") or "").strip() != "TX":
            continue
        county = str(row.get("County") or "").strip()
        uid = str(row.get("Utility Number") or "").strip()
        if county and uid:
            out[county].add(uid)
    return out


def eia930_ba_codes(ref_path):
    codes = set()
    for row in sheet_rows(ref_path, "BAs", 1):
        c = str(row.get("BA Code") or "").strip()
        if c and c != "None":
            codes.add(c)
    return codes


def run(dir861, ref930, out_path):
    u2ba = utility_ba_map(dir861)
    c2u = tx_county_utilities(dir861)
    codes930 = eia930_ba_codes(ref930)

    counties = {}
    unmatched_utilities = set()
    for county, uids in sorted(c2u.items()):
        bas = set()
        for uid in uids:
            got = bas_for_state(u2ba, uid, "TX")
            if got:
                bas |= got
            else:
                unmatched_utilities.add(uid)
        counties[county] = {
            "bas": sorted(bas),
            "multi_ba": len(bas) > 1,
            "serving_utilities": len(uids),
        }

    ba_counts = defaultdict(int)
    for c in counties.values():
        for b in c["bas"]:
            ba_counts[b] += 1
    not_in_930 = sorted({b for c in counties.values() for b in c["bas"]} - codes930)
    no_ba = sorted(k for k, v in counties.items() if not v["bas"])

    artifact = {
        "provenance": {
            "source": "EIA-861 2024 final (f8612024.zip; Sales_Ult_Cust + "
                      "Short_Form + Delivery_Companies + Service_Territory) "
                      "+ EIA-930 reference tables; downloaded 2026-07-07",
            "license": "U.S. government publications, public domain; "
                       "Source: U.S. Energy Information Administration",
            "attribution": "U.S. Energy Information Administration",
            "method": "scripts/grid_county_ba.py — county retail-BA union "
                      "via serving utilities; NO primary BA designated for "
                      "multi-BA counties (861 cannot weight them; polygon "
                      "arbiter queued). RETAIL service, not grid topology — "
                      "stated on every consumer surface.",
        },
        "counties": counties,
        "summary": {
            "county_count": len(counties),
            "ba_county_counts": dict(sorted(ba_counts.items())),
            "multi_ba_counties": sum(1 for c in counties.values() if c["multi_ba"]),
            "counties_with_no_ba": no_ba,
            "ba_codes_not_in_eia930": not_in_930,
            "unmatched_serving_utilities": sorted(unmatched_utilities),
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=1, sort_keys=False)
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir861")
    ap.add_argument("ref930")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datacore", "gridvision", "tx_county_ba.json"))
    args = ap.parse_args()
    art = run(args.dir861, args.ref930, args.out)
    s = art["summary"]
    print(json.dumps(s, indent=1))
    ok = (s["county_count"] == TX_COUNTY_COUNT and not s["counties_with_no_ba"]
          and not s["ba_codes_not_in_eia930"])
    sys.exit(0 if ok else 3)
