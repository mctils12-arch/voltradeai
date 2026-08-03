#!/usr/bin/env python3
"""
data_stream_registry_check.py — EDGE DOCTRINE compiled-knowledge registry.

WHY THIS EXISTS (EDGE DOCTRINE #3, COMPILE KNOWLEDGE INTO CODE): the same
question — "is free-data candidate X already built?" — has been answered by
hand, via grep + reading research/experiments.md, in well over a dozen
separate sessions (JODI/EDGAR-Form4/USAspending/CFTC-COT/FDA-calendar/
Google-Trends alone recur verbatim ~15+ times in experiments.md as of
2026-08-03). Each re-derivation burns tokens reasoning about something
already known. This script makes the answer a runtime fact instead of a
prose memory: it cross-checks a hand-curated CANDIDATES table against the
actual repo state (datacore/manifests/*.json, datacore/layers.json) so the
table cannot silently drift from reality without a test failing.

SCOPE: this is NOT a replacement for research/data_census.md (which is the
full prose census of every probed source, built or not). It covers the
EDGE-DOCTRINE-named candidates (CLAUDE.md) plus the set that has been
repeatedly re-litigated across sessions. Extend CANDIDATES as new
candidates get probed/built/declined — the audit() self-check keeps the
table honest against manifests/layers that already exist.

Usage:
  python3 scripts/data_stream_registry_check.py            # human report
  python3 scripts/data_stream_registry_check.py --json      # machine JSON
  python3 scripts/data_stream_registry_check.py --unbuilt   # candidates NOT built, for "what's next"
Exit code 1 if any DRIFT is found (a "built" candidate whose manifest/layer
evidence no longer exists on disk, or vice versa).
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFESTS_DIR = os.path.join(REPO_ROOT, "datacore", "manifests")
LAYERS_PATH = os.path.join(REPO_ROOT, "datacore", "layers.json")

# status values:
#   built                — archived (manifest) and/or shown as a live /data layer
#   declined_gate1_fail  — probed, entered the ladder, failed gate 1 (DATA)
#   declined_dead_source — probed, source is unreachable/bot-blocked/retired
#   blocked_free_key     — buildable today, waiting on a human-registered free API key
#   blocked_registration — needs a human registration/form-fill beyond a simple key
#   candidate_unbuilt    — probed viable, not yet prioritized/built
CANDIDATES = [
    # --- CLAUDE.md EDGE DOCTRINE axis (a), named list ---
    {"id": "sentinel2_tank_shadows", "name": "Sentinel-2/Landsat tank-shadow crude inventory",
     "edge_doctrine_named": True, "status": "built",
     "manifest_keys": ["sentinel2", "sentinel2v2", "sentinel2chips", "sentinel2sitechips",
                        "sentinel1chips", "sentinel1readings"],
     "layer_ids": [], "note": "scripts/sentinel2_tankfill.py + tankfill_estimator.py family"},
    {"id": "edgar_form4", "name": "SEC EDGAR Form 4 insider buys",
     "edge_doctrine_named": True, "status": "built",
     "manifest_keys": ["filings"], "layer_ids": [], "note": "server/edgarForm4.ts"},
    {"id": "usaspending", "name": "USAspending.gov contracts",
     "edge_doctrine_named": True, "status": "built",
     "manifest_keys": ["usaspending"], "layer_ids": [], "note": "server/usaSpending.ts"},
    {"id": "cftc_cot", "name": "CFTC COT disaggregated positioning",
     "edge_doctrine_named": True, "status": "built",
     "manifest_keys": ["cftccot", "cftctff", "cot"], "layer_ids": [],
     "note": "server/cftcCot.ts + server/cftcTff.ts + cftc_cot.py"},
    {"id": "fda_calendar", "name": "FDA approval/advisory-committee calendar",
     "edge_doctrine_named": True, "status": "built",
     "manifest_keys": ["fda"], "layer_ids": [], "note": "server/fdaEvents.ts"},
    {"id": "google_trends_pytrends", "name": "Google Trends via pytrends",
     "edge_doctrine_named": True, "status": "declined_gate1_fail",
     "manifest_keys": [], "layer_ids": [],
     "note": "upstream pytrends archived read-only 2025-04-17; gate-1 stability probe "
             "(2026-07-05) failed median cross-pull r bar. Replaced by wikiattention."},

    # --- Repeatedly-re-litigated extended candidates (post-doctrine-list build orders) ---
    {"id": "wikimedia_pageviews", "name": "Wikimedia pageviews attention proxy (pytrends replacement)",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["wikiattention"], "layer_ids": [], "note": "server/wikiAttention.ts"},
    {"id": "faa_airport_status", "name": "FAA airport status / delay programs",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["faastatus"], "layer_ids": ["faa_airports"], "note": "server/faaStatus.ts"},
    {"id": "cbp_border_wait", "name": "CBP border-crossing wait times",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["cbpborderwait"], "layer_ids": ["border_waits"], "note": "server/cbpBorderWait.ts"},
    {"id": "noaa_swpc_space_weather", "name": "NOAA SWPC space weather (Kp/X-ray/aurora)",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["spaceweather"], "layer_ids": [], "note": "server/spaceWeather.ts"},
    {"id": "so2_column_gibs", "name": "SO2 column density (OMPS/GIBS)",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": [], "layer_ids": ["so2"],
     "note": "tile pass-through (client/src/lib/gibs.ts) — RAW overlay, no archive manifest by design"},
    {"id": "usgs_volcano_alerts", "name": "USGS volcano alert levels + GVP coordinates",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["volcanoes"], "layer_ids": ["volcanoes"], "note": "server/usgsVolcanoes.ts"},
    {"id": "epa_camd_cems", "name": "EPA CAMD CEMS unit-level plant operations",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["epacamd"], "layer_ids": [], "note": "server/epaCamd.ts"},
    {"id": "global_energy_monitor", "name": "Global Energy Monitor asset registry",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["gem"], "layer_ids": [], "note": "scripts/gem_ingest.py (session-run, Mike Drive delivery)"},
    {"id": "entsoe_eu_power", "name": "ENTSO-E EU load/generation/day-ahead prices",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["euload", "eugenmix", "eudayaheadprices"], "layer_ids": [],
     "note": "server/euLoad.ts + euGenerationMix.ts + euDayAheadPrices.ts"},
    {"id": "usgs_earthquakes", "name": "USGS earthquake feed",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["earthquakes"], "layer_ids": [], "note": "server/usgsQuakes.ts"},
    {"id": "ndbc_buoys", "name": "NOAA NDBC ocean buoys",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["buoys"], "layer_ids": [], "note": "server/ndbcBuoys.ts"},
    {"id": "sec_ftd", "name": "SEC fails-to-deliver",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["secftd"], "layer_ids": [], "note": "server/secFtd.ts"},
    {"id": "sec_midas", "name": "SEC MIDAS market-structure metrics",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["secmidas"], "layer_ids": [], "note": "server/secMidas.ts"},
    {"id": "occ_options_volume", "name": "OCC total options volume",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["occvolume"], "layer_ids": [], "note": "server/occVolume.ts, GATE 1 PASSED 2026-08-03"},
    {"id": "finra_query_cluster", "name": "FINRA short volume / threshold / ATS / short-interest cluster",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["finrashortvol", "finrashortvolotc", "finrathreshold", "finrablocks",
                        "finramonthly", "finraweekly", "finrashortinterest"],
     "layer_ids": [], "note": "server/finraQuery.ts + server/finraShortVolume.ts"},
    {"id": "jodi_oil_gas", "name": "JODI world oil/gas stocks",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["jodi"], "layer_ids": [], "note": "scripts/jodi_oil.py (session-run monthly)"},
    {"id": "eu_macro_ecb_bbk_eurostat", "name": "ECB + Bundesbank + Eurostat macro cluster",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["eumacro"], "layer_ids": [], "note": "server/euMacro.ts refreshEuMacro"},
    {"id": "nasdaq_finra_threshold_list", "name": "Nasdaq/FINRA consolidated threshold securities list",
     "edge_doctrine_named": False, "status": "built",
     "manifest_keys": ["finrathreshold"], "layer_ids": [],
     "note": "NYSE's own threshold API is bot-blocked (declined_dead_source); FINRA's covers both venues"},

    # --- Declined (dead / bot-blocked sources) ---
    {"id": "cme_datamine", "name": "CME futures data",
     "edge_doctrine_named": False, "status": "declined_dead_source",
     "manifest_keys": [], "layer_ids": [],
     "note": "Akamai 403 on everything, FTP TLS reset. Covered by CFTC COT + CBOE VX curve."},
    {"id": "nyse_threshold_api", "name": "NYSE threshold securities API",
     "edge_doctrine_named": False, "status": "declined_dead_source",
     "manifest_keys": [], "layer_ids": [], "note": "headless 500s + bot check; FINRA thresholdList covers both venues"},
    {"id": "pboc", "name": "PBOC (People's Bank of China) data",
     "edge_doctrine_named": False, "status": "declined_dead_source",
     "manifest_keys": [], "layer_ids": [], "note": "scrape-hostile (WAF, HTML/xlsx only); high build cost, no China thesis demands it"},

    # --- Blocked on a human action (free key or registration) ---
    {"id": "openaq_v3", "name": "OpenAQ v3 air quality (SO2/NO2 downwind proxy)",
     "edge_doctrine_named": False, "status": "blocked_free_key",
     "manifest_keys": [], "layer_ids": [],
     "note": "401 without free key (S3 bulk archive path exists keyless as a lower-priority fallback)"},
    {"id": "uspto_patents", "name": "USPTO patent filing velocity (ODP API)",
     "edge_doctrine_named": False, "status": "blocked_registration",
     "manifest_keys": [], "layer_ids": [],
     "note": "needs human ID.me signup for the ODP key; Google Patents BigQuery is a keyless backfill-only fallback"},
    {"id": "cloudflare_radar", "name": "Cloudflare Radar internet outages",
     "edge_doctrine_named": False, "status": "blocked_free_key",
     "manifest_keys": [], "layer_ids": [], "note": "needs a free API token, not yet requested via wishlist.md"},
    {"id": "viirs_nightfire", "name": "VIIRS Nightfire gas flaring (EOG/Colorado Mines)",
     "edge_doctrine_named": False, "status": "blocked_registration",
     "manifest_keys": [], "layer_ids": [],
     "note": "eogdata.mines.edu 302s to a registration flow; strongest pure EDGE-DOCTRINE candidate on the "
             "board but needs a BUILD-FIRST free-alternative writeup before it may even enter wishlist.md"},

    # --- Probed, viable, simply not yet built (lower rank / needs a build decision) ---
    {"id": "cboe_daily_stats", "name": "CBOE daily put/call ratios + VX curve",
     "edge_doctrine_named": False, "status": "candidate_unbuilt",
     "manifest_keys": [], "layer_ids": [],
     "note": "probed 200 keyless; total P/C is crowded, equity-vs-index P/C spread x small-cap short volume "
             "is the residual claim; resale of raw series needs a Cboe license"},
    {"id": "dtcc_sbsdr", "name": "DTCC swap data repository",
     "edge_doctrine_named": False, "status": "candidate_unbuilt",
     "manifest_keys": [], "layer_ids": [],
     "note": "keyless but 147MB/day — needs a volume-budget decision before build, never started by default"},
    {"id": "un_comtrade", "name": "UN Comtrade bilateral trade flows",
     "edge_doctrine_named": False, "status": "candidate_unbuilt",
     "manifest_keys": [], "layer_ids": [],
     "note": "keyless preview tier probed 200; too lagged (1-6mo) for direct alpha, structural-thesis input only"},
]


def _load_manifest_keys_on_disk():
    if not os.path.isdir(MANIFESTS_DIR):
        return set()
    return {fn[:-5] for fn in os.listdir(MANIFESTS_DIR) if fn.endswith(".json")}


def _load_layer_ids_on_disk():
    if not os.path.exists(LAYERS_PATH):
        return set()
    with open(LAYERS_PATH) as f:
        data = json.load(f)
    return {layer.get("id") for layer in data.get("layers", []) if layer.get("id")}


def audit():
    """Cross-checks CANDIDATES against actual repo state. Returns (report, drift)."""
    manifest_keys_on_disk = _load_manifest_keys_on_disk()
    layer_ids_on_disk = _load_layer_ids_on_disk()

    report = []
    drift = []
    claimed_manifest_keys = set()
    claimed_layer_ids = set()

    for c in CANDIDATES:
        manifest_hits = [k for k in c["manifest_keys"] if k in manifest_keys_on_disk]
        manifest_misses = [k for k in c["manifest_keys"] if k not in manifest_keys_on_disk]
        layer_hits = [lid for lid in c["layer_ids"] if lid in layer_ids_on_disk]
        layer_misses = [lid for lid in c["layer_ids"] if lid not in layer_ids_on_disk]
        claimed_manifest_keys.update(c["manifest_keys"])
        claimed_layer_ids.update(c["layer_ids"])

        has_evidence = bool(manifest_hits or layer_hits)
        entry = {
            "id": c["id"], "name": c["name"], "status": c["status"],
            "manifest_hits": manifest_hits, "layer_hits": layer_hits,
        }

        if c["status"] == "built" and not has_evidence:
            entry["drift"] = "claims built but no manifest/layer evidence found on disk"
            drift.append(entry)
        elif c["status"] == "built" and (manifest_misses or layer_misses):
            entry["drift"] = f"partially missing: manifest_misses={manifest_misses} layer_misses={layer_misses}"
            drift.append(entry)
        elif c["status"] != "built" and has_evidence:
            entry["drift"] = "table says NOT built but manifest/layer evidence exists — table is stale"
            drift.append(entry)

        report.append(entry)

    uncatalogued_manifests = sorted(manifest_keys_on_disk - claimed_manifest_keys)
    uncatalogued_layers = sorted((layer_ids_on_disk - claimed_layer_ids))

    return {
        "candidates": report,
        "drift": drift,
        "uncatalogued_manifests": uncatalogued_manifests,
        "uncatalogued_layers_sample": uncatalogued_layers[:10],
        "uncatalogued_layers_total": len(uncatalogued_layers),
    }


def _print_human_report(result, unbuilt_only):
    built = [c for c in CANDIDATES if c["status"] == "built"]
    not_built = [c for c in CANDIDATES if c["status"] != "built"]

    if not unbuilt_only:
        print(f"BUILT: {len(built)}/{len(CANDIDATES)} tracked candidates")
        for c in built:
            print(f"  [OK] {c['id']}: {c['name']}")
        print()

    print(f"NOT BUILT: {len(not_built)}/{len(CANDIDATES)}")
    for c in not_built:
        print(f"  [{c['status']}] {c['id']}: {c['name']}")
        print(f"        {c['note']}")

    if result["drift"]:
        print("\n*** DRIFT DETECTED (table vs. repo state disagree) ***")
        for d in result["drift"]:
            print(f"  {d['id']}: {d['drift']}")

    if result["uncatalogued_manifests"]:
        print(f"\nUncatalogued manifests on disk (not in this table, informational only): "
              f"{', '.join(result['uncatalogued_manifests'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a report")
    ap.add_argument("--unbuilt", action="store_true", help="print only NOT-built candidates")
    args = ap.parse_args()

    result = audit()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human_report(result, args.unbuilt)

    return 1 if result["drift"] else 0


if __name__ == "__main__":
    sys.exit(main())
