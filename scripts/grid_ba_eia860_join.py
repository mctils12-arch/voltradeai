#!/usr/bin/env python3
"""grid_ba_eia860_join.py — a REAL, non-guessed plant -> balancing-authority
attribution for FUSION HYPOTHESIS (b)'s per-region gate-1, replacing the
lat/lon-vs-HIFLD-polygon inference scripts/grid_ba_polygon_join.py used.

WHY THIS EXISTS: grid_ba_polygon_join.py's own two-prior-session NEXT (the
2026-09-11 "sixth session this UTC day" entry in research/experiments.md)
left the ambiguous-plant question explicitly undecided between (a) build a
real ownership/interconnection-share attribution rule, or (b) formally scope
the check to low-exclusion-fraction regions only — stating (a) "needs data
this repo doesn't have." That premise was wrong: EIA-860's own Plant
schedule (Schedule 2, the SAME file scripts/eia860_add_missing_plants.py
already downloads and uses this same week) carries a `Balancing Authority
Code` column DIRECTLY REPORTED per plant by the utility that owns it — this
is ground truth, not an inferred geometric overlap, and it assigns exactly
ONE code per plant by construction (no polygon-overlap ambiguity is even
possible). This module joins the registry to that column by coordinate
match instead of point-in-polygon.

WHY COORDINATE MATCH IS A SAFE JOIN KEY HERE (verified this session, not
assumed): scripts/build_powerplants.py's registry rows do not retain an EIA
Plant Code (only [name, capacity_mw, fuel, owner, lat, lon, verified]), so
there is no direct code-to-code join available. But every registry
coordinate checked this session against EIA-860's own Plant-schedule
Latitude/Longitude for the same plant matched EXACTLY after rounding to 4
decimal places (~11m) — both GPPD-sourced rows (e.g. West County Energy
Center: registry 26.6986/-80.3747, EIA-860 26.6986/-80.3747) and EIA-860-
added rows (Grand Coulee: registry 47.9575/-118.9773, EIA-860
47.957511/-118.977323, identical after rounding) — because GPPD's own US
plant coordinates are themselves sourced from EIA-860/EIA-923 in the first
place. A 4-decimal-place exact match is therefore the join key, not a fuzzy
nearest-neighbor tolerance that would reintroduce guessing.

EMPIRICAL RESULT this session (2026-09-11, live EIA-860 2025 Schedule 2,
17,349 plant rows, 248 with no reported BA code, 28 with no coordinate):
of the CURRENT registry's 14,172 plants, 13,609 (96.0%) get a single,
confident, ground-truth BA code this way; 548 (3.9%) have no EIA-860
coordinate match (a genuine coverage gap, honestly reported, not guessed);
only 15 (0.1%) land on a coordinate shared by EIA-860 records reporting
CONFLICTING BA codes (a real coordinate collision — e.g. adjacent plants
rounding to the same 4-decimal point — reported as ambiguous, never picked
arbitrarily). Cross-checked directly against grid_ba_polygon_join.py's own
3,420 polygon-ambiguous (multi-BA) plants: this join resolves 3,335 of them
(97.5%) to one confident code, of which 3,086 (92.5% of resolved) fall
inside the polygon join's own candidate set (mutual corroboration between
the two independent methods) and 249 resolve to a DIFFERENT code than any
polygon candidate offered (EIA-860's authoritative report overriding a
HIFLD geometry quirk — plausible for the federal-PMA-embedded-in-a-host-
utility case the polygon join's own docstring already named). This SETTLES
the two-session-old ambiguous-plant question as option (a): a real
attribution rule was buildable from data already in this repo's own
pipeline, and now exists.

SCOPE: this module builds the join and its comparison-to-polygon summary.
It does NOT re-run grid_generation_gate1_ba.py's per-region reconciliation
itself (see that script's own `--source eia860` flag, added the same PR) —
one logical change per component, EDGE DOCTRINE #3 (reuse, not reimplement).

Usage:
    python3 scripts/grid_ba_eia860_join.py \
        --eia860-plants /tmp/eia860/2___Plant_Y2025.xlsx \
        [--registry datacore/powerplants/us_power_plants.json] \
        [--polygon-join datacore/powerplants/plant_balancing_authority.json] \
        [--out datacore/powerplants/plant_balancing_authority_eia860.json]

(the raw EIA-860 xlsx is not committed to this repo — same manual-download
precedent every sibling EIA-860 script in this directory already documents:
    curl -L -o /tmp/eia860.zip https://www.eia.gov/electricity/data/eia860/xls/eia8602025.zip
    unzip /tmp/eia860.zip -d /tmp/eia860
)
"""
import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)


def load_eia860_ba_directory(xlsx_path):
    """EIA-860 Schedule 2 (Plant file) -> list of {plant_code, name, lat,
    lon, ba_code}. Rows with no usable coordinate or no reported BA code
    are skipped (counted by the caller via len(directory) vs raw row
    count), never defaulted to a guessed value."""
    import openpyxl  # session-run only, same convention as sibling scripts
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    next(rows)  # title row
    hdr = next(rows)
    idx = {h: i for i, h in enumerate(hdr)}
    out = []
    for r in rows:
        lat, lon = r[idx["Latitude"]], r[idx["Longitude"]]
        ba = r[idx["Balancing Authority Code"]]
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            continue
        if not ba or not str(ba).strip():
            continue
        out.append({
            "plant_code": r[idx["Plant Code"]],
            "name": r[idx["Plant Name"]],
            "lat": lat,
            "lon": lon,
            "ba_code": str(ba).strip(),
        })
    return out


def build_coord_index(directory, precision=4):
    """directory: load_eia860_ba_directory() output. Returns
    {(round(lat,precision), round(lon,precision)): [ba_code, ...]} — a
    coordinate key maps to every BA code any EIA-860 plant at that rounded
    point reports, so the caller can detect a genuine collision (multiple
    DISTINCT codes) rather than silently picking one."""
    index = defaultdict(list)
    for rec in directory:
        key = (round(rec["lat"], precision), round(rec["lon"], precision))
        index[key].append(rec["ba_code"])
    return dict(index)


def assign_registry_ba(plants, coord_index, precision=4):
    """plants: build_powerplants.py row format
    [name, capacity_mw, fuel, owner, lat, lon, verified]. Returns list of
    {name, fuel, capacity_mw, ba_code, status} where status is exactly one
    of "matched" (single confident code), "unmatched" (no EIA-860
    coordinate hit), or "ambiguous" (coordinate hit, but the EIA-860
    records there report conflicting codes — a real collision, not
    resolved by guessing). ba_code is None unless status == "matched"."""
    out = []
    for p in plants:
        name, capacity_mw, fuel, _owner, lat, lon, _verified = p
        status, ba_code = "unmatched", None
        if lat is not None and lon is not None:
            key = (round(float(lat), precision), round(float(lon), precision))
            codes = coord_index.get(key)
            if codes:
                distinct = set(codes)
                if len(distinct) == 1:
                    status, ba_code = "matched", next(iter(distinct))
                else:
                    status = "ambiguous"
        out.append({"name": name, "fuel": fuel, "capacity_mw": capacity_mw,
                     "ba_code": ba_code, "status": status})
    return out


def registry_capacity_by_ba(assignments):
    """Same two-value return shape as grid_ba_polygon_join.py's function of
    the same name, so grid_generation_gate1_ba.py can select either source
    behind one interface: (per_ba_capacity: {ba_code: {fuel: mw}},
    excluded_mw: {} always — this join has no per-region excluded bucket,
    since an unmatched/ambiguous plant here isn't attributable to ANY
    region even provisionally, unlike the polygon join's real geometric
    overlap. Unmatched/ambiguous totals are reported separately via
    summarize(), not folded into a per-BA figure that would misrepresent
    which region lost the capacity)."""
    cap = defaultdict(lambda: defaultdict(float))
    for a in assignments:
        if a["status"] == "matched":
            cap[a["ba_code"]][a["fuel"]] += a["capacity_mw"] or 0.0
    return {ba: dict(fuels) for ba, fuels in cap.items()}, {}


def summarize(assignments):
    matched = sum(1 for a in assignments if a["status"] == "matched")
    unmatched = sum(1 for a in assignments if a["status"] == "unmatched")
    ambiguous = sum(1 for a in assignments if a["status"] == "ambiguous")
    unmatched_mw = sum(a["capacity_mw"] or 0.0 for a in assignments if a["status"] == "unmatched")
    ambiguous_mw = sum(a["capacity_mw"] or 0.0 for a in assignments if a["status"] == "ambiguous")
    return {
        "total": len(assignments), "matched": matched, "unmatched": unmatched,
        "ambiguous": ambiguous, "unmatched_capacity_mw": round(unmatched_mw, 1),
        "ambiguous_capacity_mw": round(ambiguous_mw, 1),
    }


def compare_to_polygon_join(eia860_assignments, polygon_assignments):
    """Both lists must be positional-parallel to the SAME registry plants
    list (both this module's run() and grid_ba_polygon_join.py's run()
    iterate `for p in plants` over the same us_power_plants.json in order —
    verified this session via len() equality before trusting the zip).
    Reports, for the polygon join's own ambiguous (multi-BA) subset only:
    how many this join resolves to one code, how many of those land inside
    the polygon join's own candidate set (corroboration) vs a code the
    polygon join never offered (an EIA-860 override of a geometry quirk)."""
    poly_ambiguous = [(e, p) for e, p in zip(eia860_assignments, polygon_assignments)
                       if len(p.get("ba_codes") or []) > 1]
    resolved = 0
    corroborated = 0
    overridden = 0
    for e, p in poly_ambiguous:
        if e["status"] != "matched":
            continue
        resolved += 1
        if e["ba_code"] in p["ba_codes"]:
            corroborated += 1
        else:
            overridden += 1
    return {
        "polygon_ambiguous_plants": len(poly_ambiguous),
        "resolved_by_eia860_join": resolved,
        "resolved_matches_a_polygon_candidate": corroborated,
        "resolved_to_a_different_ba_than_any_polygon_candidate": overridden,
        "still_unresolved": len(poly_ambiguous) - resolved,
    }


def run(eia860_plants_path, registry_path, polygon_join_path, out_path):
    directory = load_eia860_ba_directory(eia860_plants_path)
    coord_index = build_coord_index(directory)

    registry = json.load(open(registry_path))
    assignments = assign_registry_ba(registry["plants"], coord_index)
    summary = summarize(assignments)

    comparison = None
    if polygon_join_path and os.path.exists(polygon_join_path):
        polygon_join = json.load(open(polygon_join_path))
        polygon_assignments = polygon_join["assignments"]
        if len(polygon_assignments) == len(assignments):
            comparison = compare_to_polygon_join(assignments, polygon_assignments)

    out = {
        "_doc": ("Plant -> EIA-930 balancing-authority assignment via direct join "
                 "against EIA-860 Schedule 2's own reported Balancing Authority Code "
                 "(ground truth, not geometric inference) — see module docstring for "
                 "the coordinate-match rationale and this session's empirical resolution "
                 "rate. Built by scripts/grid_ba_eia860_join.py. Recomputed fresh from "
                 "the registry + a fresh EIA-860 download each run, like "
                 "grid_ba_polygon_join.py — not a stable cross-rebuild join key."),
        "source": "scripts/grid_ba_eia860_join.py",
        "eia860_source": eia860_plants_path,
        "registry_source": registry_path,
        "eia860_plant_rows_with_coord_and_ba": len(directory),
        "summary": summary,
        "comparison_to_polygon_join": comparison,
        "assignments": assignments,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"eia860 directory: {len(directory)} usable plant rows")
    print(f"registry plants: {summary['matched']} matched / {summary['unmatched']} unmatched / "
          f"{summary['ambiguous']} ambiguous (of {summary['total']})")
    if comparison:
        print(f"polygon-ambiguous cross-check: {comparison['resolved_by_eia860_join']}/"
              f"{comparison['polygon_ambiguous_plants']} resolved "
              f"({comparison['resolved_matches_a_polygon_candidate']} corroborated, "
              f"{comparison['resolved_to_a_different_ba_than_any_polygon_candidate']} overridden)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eia860-plants", required=True,
                     help="path to EIA-860 Schedule 2 (2___Plant_YYYY.xlsx), manually downloaded")
    ap.add_argument("--registry", default=os.path.join(
        REPO_ROOT, "datacore", "powerplants", "us_power_plants.json"))
    ap.add_argument("--polygon-join", default=os.path.join(
        REPO_ROOT, "datacore", "powerplants", "plant_balancing_authority.json"),
        help="grid_ba_polygon_join.py's output, for the cross-check comparison; pass '' to skip")
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "datacore", "powerplants", "plant_balancing_authority_eia860.json"))
    args = ap.parse_args()
    run(args.eia860_plants, args.registry, args.polygon_join, args.out)
