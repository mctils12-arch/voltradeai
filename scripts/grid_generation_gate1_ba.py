#!/usr/bin/env python3
"""grid_generation_gate1_ba.py — the PER-REGION half of FUSION HYPOTHESIS
(b)'s ROOT VALIDATION LADDER gate 1 (DATA) that scripts/grid_generation_gate1.py
deliberately left unbuilt at US48-only scope. Stated ground truth (CLAUDE.md /
research/open_questions.md): "EIA-930 totals reconciling to registry capacity
... within ~5% per region", for the literal region list CISO/ERCO/MISO/PJM/
NYIS/ISNE/SWPP/FPL.

TWO SELECTABLE ATTRIBUTION SOURCES (--source), same registry, same per-fuel
reconcile logic — only the plant -> balancing-authority assignment differs:

- "eia860" (DEFAULT since 2026-09-11): scripts/grid_ba_eia860_join.py's
  ground-truth join against EIA-860's own reported Balancing Authority Code
  per plant. Resolves 96.0% of all registry plants to a single confident
  code (vs "polygon"'s 73.7%), including 3,335 of "polygon"'s own 3,420
  ambiguous plants (97.5% of them) — see that module's docstring for the
  full empirical comparison. This SETTLES the ambiguous-plant question the
  2026-09-11 fifth-session entry filed as NEXT, left undecided by the sixth
  session's own run below (kept as "polygon" for cross-reference, not
  deleted — the two methods corroborate each other on 92.5% of what
  "eia860" resolves, which is itself evidence neither is a fluke).
- "polygon": scripts/grid_ba_polygon_join.py's point-in-polygon join against
  the HIFLD Control Areas layer. datacore/powerplants/plant_balancing_
  authority.json's own summary: 10,440/14,172 plants land in exactly ONE
  balancing authority, 312 in none, and 3,420 (24.1%) in MORE than one — a
  real, independently-verified finding (federal Power Marketing
  Administrations like WALC/BPAT/SPA sell wholesale power to customers
  embedded inside a host utility's own footprint, so their HIFLD polygon
  genuinely overlaps the host's rather than partitioning space with it),
  not a join defect. Ambiguous (multi-BA) plants under this source are
  EXCLUDED from every single region's registry-capacity sum, never
  attributed to one BA by guessing — the conservative choice for an
  "exceeds capacity" ceiling check specifically (omitting real capacity can
  only make a region's ceiling LOWER, which can only make a FAIL verdict
  MORE likely, never manufacture a false PASS). The excluded capacity, and
  what FRACTION of the region's true total capacity it represents, are
  reported per region under this source only (the eia860 source has no
  per-region excluded bucket — see its own module docstring for why).

METHOD: identical per-fuel-bucket max-hourly-generation-vs-capacity-ceiling
check as grid_generation_gate1.py (reused via importlib, not reimplemented —
EDGE DOCTRINE #3), run once per named respondent, with the selected source's
matched registry capacity substituted for that respondent's CONUS-wide
capacity.

Usage: python3 scripts/grid_generation_gate1_ba.py [--days N] [--tolerance F]
       [--source eia860|polygon]
       [--respondents CISO,ERCO,MISO,PJM,NYIS,ISNE,SWPP,FPL]
"""
import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

_spec = importlib.util.spec_from_file_location(
    "grid_generation_gate1", os.path.join(_HERE, "grid_generation_gate1.py"))
_gate1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate1)

_eia860_spec = importlib.util.spec_from_file_location(
    "grid_ba_eia860_join", os.path.join(_HERE, "grid_ba_eia860_join.py"))
_eia860_join = importlib.util.module_from_spec(_eia860_spec)
_eia860_spec.loader.exec_module(_eia860_join)

POLYGON_ASSIGNMENTS_PATH = os.path.join(_REPO_ROOT, "datacore", "powerplants", "plant_balancing_authority.json")
EIA860_ASSIGNMENTS_PATH = os.path.join(_REPO_ROOT, "datacore", "powerplants", "plant_balancing_authority_eia860.json")

# The literal per-region list the FUSION (b) ground-truth statement names,
# restricted to entries that are real HIFLD-crosswalked EIA-930 respondent
# codes (SE/NW/SW in the original wording are EIA-930 ROLLUP regions, not
# HIFLD control areas — no polygon exists for them in this repo, so they are
# not attempted here; a rollup would need its own aggregation rule, not a
# guessed polygon).
DEFAULT_RESPONDENTS = ("CISO", "ERCO", "MISO", "PJM", "NYIS", "ISNE", "SWPP", "FPL")


def registry_capacity_by_ba(assignments):
    """assignments: plant_balancing_authority.json's `assignments` list,
    each {name, fuel, capacity_mw, ba_codes}. Returns
    (per_ba_capacity: {ba_code: {fuel_bucket: capacity_mw}},
     excluded_ambiguous_mw_by_ba: {ba_code: mw}).
    A plant with ba_codes of length != 1 (zero = unmatched, 2+ = ambiguous
    overlap) contributes to NEITHER a region's capacity sum NOR the
    per-BA excluded total for a region it doesn't list — only the regions
    it actually overlaps get their excluded-mw counter incremented, so the
    number reported alongside a verdict reflects real capacity omitted from
    THAT region specifically, not the whole ambiguous pool."""
    cap = defaultdict(lambda: defaultdict(float))
    excluded_mw = defaultdict(float)
    for p in assignments:
        ba_codes = p.get("ba_codes") or []
        fuel = p.get("fuel")
        mw = p.get("capacity_mw") or 0.0
        if len(ba_codes) == 1:
            cap[ba_codes[0]][fuel] += mw
        else:
            for ba in ba_codes:
                excluded_mw[ba] += mw
    return {ba: dict(fuels) for ba, fuels in cap.items()}, dict(excluded_mw)


def excluded_capacity_fraction(counted_mw, excluded_mw):
    """What fraction of a region's TRUE total capacity (counted + the
    ambiguous capacity this policy omits) never entered the capacity side of
    the check at all. None when the region has zero known capacity either
    way (nothing to take a fraction of)."""
    total = counted_mw + excluded_mw
    if not total:
        return None
    return round(excluded_mw / total, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="trailing window size in days")
    ap.add_argument("--tolerance", type=float, default=0.05, help="fractional headroom above capacity before FAIL")
    ap.add_argument("--respondents", default=",".join(DEFAULT_RESPONDENTS))
    ap.add_argument("--source", choices=("eia860", "polygon"), default="eia860",
                     help="plant->balancing-authority attribution source (see module docstring)")
    args = ap.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("EIA_API_KEY not set — cannot run live gate-1 fetch", file=sys.stderr)
        sys.exit(1)

    respondents = [r.strip() for r in args.respondents.split(",") if r.strip()]

    if args.source == "eia860":
        assignments_path = EIA860_ASSIGNMENTS_PATH
        capacity_fn = _eia860_join.registry_capacity_by_ba
        ambiguous_plant_policy = ("EIA-860 ground-truth Balancing Authority Code join — plants with "
                                   "no coordinate match or a genuine coordinate collision are excluded "
                                   "from every region's capacity sum (see module summary.unmatched/"
                                   "ambiguous below), never attributed by guess")
    else:
        assignments_path = POLYGON_ASSIGNMENTS_PATH
        capacity_fn = registry_capacity_by_ba
        ambiguous_plant_policy = "excluded from every single region's capacity sum (never attributed by guess)"

    with open(assignments_path) as f:
        ba_join = json.load(f)
    per_ba_cap, excluded_mw = capacity_fn(ba_join["assignments"])

    now = datetime.now(timezone.utc)
    end = now.strftime("%Y-%m-%dT%H")
    start = (now - timedelta(days=args.days)).strftime("%Y-%m-%dT%H")

    regions = {}
    for ba in respondents:
        cap = per_ba_cap.get(ba, {})
        counted_mw = sum(cap.values())
        excluded = excluded_mw.get(ba, 0.0)
        # Only meaningful for --source polygon (see module docstring) — the
        # eia860 source has no per-region excluded bucket, so this is always
        # 0.0/None there and the global summary.unmatched/ambiguous figures
        # in the top-level report are the honest substitute.
        excluded_fraction = excluded_capacity_fraction(counted_mw, excluded)
        rows = _gate1.fetch_window(ba, start, end, api_key)
        eia_max = _gate1.aggregate_max_by_fueltype(rows)
        gen_bucket_max = _gate1.bucket_generation_max(eia_max)
        verdicts = _gate1.reconcile(cap, gen_bucket_max, tolerance=args.tolerance)
        regions[ba] = {
            "eia_rows_fetched": len(rows),
            "matched_registry_plants_capacity_mw": round(counted_mw, 1),
            "ambiguous_capacity_excluded_mw": round(excluded, 1),
            "ambiguous_capacity_excluded_fraction": excluded_fraction,
            "verdicts": verdicts,
            "other_bucket_reported_not_verdicted": {
                "registry_capacity_mw": round(cap.get("other", 0.0), 1),
                "eia_max_mwh": round(gen_bucket_max.get("other", 0.0), 1),
            },
        }

    report = {
        "root": "grid_generation_fuel_mix",
        "gate": 1,
        "scope": "per_region",
        "source": args.source,
        "ambiguous_plant_policy": ambiguous_plant_policy,
        "window": {"start": start, "end": end, "days": args.days},
        "ba_join_source": os.path.relpath(assignments_path, _REPO_ROOT),
        "ba_join_summary": ba_join.get("summary", {}),
        "regions": regions,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
