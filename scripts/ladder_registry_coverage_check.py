#!/usr/bin/env python3
"""
ladder_registry_coverage_check.py -- cross-checks every "built" EDGE
DOCTRINE candidate (scripts/data_stream_registry_check.py's CANDIDATES)
against datacore/signal_ladder.json's roots list, so a built pipeline can
no longer silently have zero ladder-bookkeeping entry.

WHY THIS EXISTS: data_stream_registry_check.py's own module docstring is
explicit about its scope -- it answers "is candidate X built", not "is
candidate X tracked in the ladder file that governs RAW-vs-SIGNAL
labeling and spinout status" (CLAUDE.md's RAW OVERLAYS vs SIGNALS rule +
SPINOUT-READY DATA LAYER STANDING BEHAVIOR). Those are different
questions, and this session found they can disagree: this check's first
live run found epa_camd_cems (server/epaCamd.ts) -- a pipeline shipped as
a live /data map layer (plant_operations, v1.0.385) and named in its own
header as "the ladder gate-1 truth source for the whole power vertical"
-- with NO entry anywhere in signal_ladder.json's 47 roots. Fixed in the
same PR that added this check (now aliased below, see ALIASES). Seven
more built candidates came back with the same gap; each needs its own
module read to assign an honest status (raw_only vs a real gate number)
rather than a guess made from this script alone, so they are filed as a
queued NEXT in research/open_questions.md and pinned here as the CURRENT
count -- the same "no big-bang backfill" discipline
research/PROGRAM_STATE.md's Q11 already established for a structurally
identical gap (layersRegistry.test.ts's renderKind/lod backfill).

ALIASES: data_stream_registry_check.py and signal_ladder.json were built
independently, by different sessions, and use different id conventions
for the SAME underlying pipeline (e.g. "cftc_cot" here vs
"cftc_cot_positioning" + "cftc_tff_positioning" there -- one registry
candidate can legitimately map to 1+ ladder roots, or to zero when the
gap is real). ALIASES records that mapping explicitly, by hand, so a
missing mapping reads as a genuine gap rather than a naming mismatch --
the same shape as this repo's other hand-verified alias tables (D12's
allowlist in scripts/program_status.sh, CANDIDATES itself).

Usage:
  python3 scripts/ladder_registry_coverage_check.py            # human report
  python3 scripts/ladder_registry_coverage_check.py --json      # machine JSON
Exit code 1 if a "built" candidate has no ALIASES entry at all (the table
itself is incomplete) or an ALIASES entry points at a ladder id that no
longer exists (real drift). A nonzero but STABLE "uncovered" count is not
itself a failure here -- see test_ladder_registry_coverage_check.py, which
pins the exact set so it can only change on a conscious edit.
"""

import argparse
import importlib.util
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER_PATH = os.path.join(REPO_ROOT, "datacore", "signal_ladder.json")

# registry candidate id -> the ladder root id(s) that ARE that same
# underlying pipeline, hand-verified this session by reading both the
# candidate's own note and the target root's own note/module header (see
# module docstring). An id mapped to an EMPTY list was specifically
# checked and confirmed to have no ladder root at all -- not an oversight
# in this table, a genuine gap tracked by the pinned test instead.
ALIASES = {
    "sentinel2_tank_shadows": ["sentinel_tank_fill_cushing"],
    "edgar_form4": ["sec_form4_bulk_archive", "sec_form4_insider_clustering"],
    "usaspending": ["usaspending_contracts"],
    "cftc_cot": ["cftc_cot_positioning", "cftc_tff_positioning"],
    "fda_calendar": ["fda_calendar"],
    "wikimedia_pageviews": ["wikimedia_pageviews_attention"],
    "faa_airport_status": ["faa_airport_status"],
    "cbp_border_wait": ["cbp_border_wait_times"],
    "noaa_swpc_space_weather": ["space_weather_swpc"],
    "so2_column_gibs": ["so2_column_gibs"],
    "usgs_volcano_alerts": ["usgs_volcano_alerts"],
    "epa_camd_cems": ["epa_camd_cems"],
    "global_energy_monitor": ["global_energy_monitor"],
    "entsoe_eu_power": ["entsoe_eu_power"],
    "usgs_earthquakes": ["usgs_earthquakes"],
    "ndbc_buoys": ["ndbc_buoys"],
    "sec_ftd": ["sec_ftd"],
    "sec_midas": ["sec_midas"],
    "occ_options_volume": ["occ_options_volume"],
    "finra_query_cluster": ["finra_short_volume"],
    "jodi_oil_gas": ["jodi_oil_stocks"],
    "eu_macro_ecb_bbk_eurostat": ["eu_macro_ecb_eurostat_bundesbank"],
    # Same underlying finrathreshold manifest as finra_query_cluster above
    # (both CANDIDATES entries share that manifest_key) -- one pipeline,
    # two registry rows, one ladder root.
    "nasdaq_finra_threshold_list": ["finra_short_volume"],
    "cboe_vix_term_structure": ["cboe_vix_term_structure"],
    "dtcc_sbsdr": ["dtcc_sbsdr_equity_swaps"],
    "un_comtrade": ["un_comtrade_bilateral_trade"],
}


def _load_registry_module():
    spec = importlib.util.spec_from_file_location(
        "data_stream_registry_check",
        os.path.join(REPO_ROOT, "scripts", "data_stream_registry_check.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_ladder_ids():
    with open(LADDER_PATH) as f:
        data = json.load(f)
    return {r["id"] for r in data["roots"]}


def audit(registry_module=None):
    registry = registry_module or _load_registry_module()
    ladder_ids = _load_ladder_ids()
    built = [c for c in registry.CANDIDATES if c["status"] == "built"]

    unaliased = sorted(c["id"] for c in built if c["id"] not in ALIASES)

    uncovered = []
    for c in built:
        mapped = ALIASES.get(c["id"], [])
        matched = [m for m in mapped if m in ladder_ids]
        if not matched:
            uncovered.append({"id": c["id"], "name": c["name"], "mapped_to": mapped})

    # An alias pointing at a ladder id that no longer exists is itself
    # drift (the ladder root was renamed/removed and this table was not
    # updated) -- distinct from "never had a root".
    stale_alias_targets = sorted(
        {target for targets in ALIASES.values() for target in targets if target not in ladder_ids}
    )

    return {
        "built_count": len(built),
        "unaliased_built_candidates": unaliased,
        "uncovered": uncovered,
        "stale_alias_targets": stale_alias_targets,
    }


def _print_human_report(result):
    print(f"{result['built_count']} built candidates checked against datacore/signal_ladder.json")
    if result["unaliased_built_candidates"]:
        print(f"\nUNALIASED (ALIASES table is incomplete): {result['unaliased_built_candidates']}")
    if result["uncovered"]:
        print(f"\nUNCOVERED ({len(result['uncovered'])}) -- built pipelines with no matching ladder root:")
        for u in result["uncovered"]:
            print(f"  {u['id']}: {u['name']}")
    if result["stale_alias_targets"]:
        print(f"\nSTALE ALIAS TARGETS (ladder id no longer exists): {result['stale_alias_targets']}")
    if not (result["unaliased_built_candidates"] or result["uncovered"] or result["stale_alias_targets"]):
        print("\nFull coverage: every built candidate maps to a live ladder root.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a report")
    args = ap.parse_args()

    result = audit()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human_report(result)

    return 1 if (result["unaliased_built_candidates"] or result["stale_alias_targets"]) else 0


if __name__ == "__main__":
    sys.exit(main())
