#!/usr/bin/env python3
"""grid_generation_gate1.py — ROOT VALIDATION LADDER gate 1 (DATA) for
FUSION HYPOTHESIS (b) ("generation shifts x utility tickers",
CLAUDE.md/research/open_questions.md). Stated ground truth: "EIA-930
totals reconciling to registry capacity (datacore/powerplants/
us_power_plants.json) within ~5% per region."

SCOPE OF THIS RUN, stated up front (REASONING STANDARD #10): the
signal_ladder.json readiness_trigger fired 2026-09-11 (>=2 archive days
since gridGeneration.ts shipped 2026-09-09), but the archived JSONL itself
lives only on the Railway persistent volume, which this sandbox cannot
read, and production (voltradeai.com) is separately down (KNOWN BROKEN
#41 in open_questions.md) as of this same session. Rather than block on
either, this script queries the EIA v2 API directly (the same free,
keyless-adjacent origin server/gridGeneration.ts itself reads, confirmed
reachable from this sandbox live) for the equivalent historical window
via start/end params — this is querying the SAME ground truth our own
archive would hold, not a substitute or invented data source; EIA is the
authority, our archive is only ever a copy of it.

REGION SCOPE, honestly cut down from the full RESPONDENTS list: this run
covers only US48 (the national aggregate), not the finer per-BA regions
(CISO/ERCO/MISO/PJM/NYIS/ISNE/SWPP/FPL/SE/NW/SW) the literal ground-truth
statement also names. A per-BA reconciliation needs a plant -> balancing-
authority join; datacore/powerplants/us_power_plants.json carries lat/lon
but no BA/respondent field, and no BA territory polygon dataset (e.g. the
HIFLD "Control Areas" layer) exists yet anywhere in this repo — state
boundaries are NOT a safe stand-in (ERCOT excludes El Paso and the
Panhandle; CAISO excludes LADWP/SMUD/other California munis; PJM/MISO/SWPP
span many states with no clean line) and building an honest one is its own
gate-1-scale task, filed as NEXT rather than faked here with a bounding
box. US48 is still a real, meaningful DATA check: it validates units,
scale, and gross plausibility (does generation-by-source ever exceed its
own physical ceiling, installed capacity) before any regional fusion
claim is attempted.

METHOD: for each EIA-930 fuel-type code, take the MAX hourly generation
(MWh in one hour is numerically bounded by MW of installed capacity) over
the observed window and compare it against the summed registry capacity
for the corresponding fuel bucket, contiguous-US plants only (AK/HI/PR/
Guam excluded by bounding box — EIA's own US48 series excludes them too,
confirmed live: 252 of 9833 registry plants fall outside the contiguous-US
bbox, every one an AK/HI/PR/Guam utility). VERDICT is one-sided: generation
may legitimately run well BELOW registry capacity (normal capacity-factor
headroom, seasonal outages) without that being a data problem — only
generation EXCEEDING capacity by more than TOLERANCE is a red flag (either
the registry undercounts real installed capacity, or the fuel mapping is
wrong). The "other" bucket (EIA storage/geothermal/unknown codes vs the
registry's catch-all "other" fuel string) is reported but NOT verdicted —
the two category definitions are too heterogeneous (battery MWh discharge
bounded by battery POWER capacity, which the plant registry does not
track as a distinct quantity) for an exceeds-capacity comparison to be a
valid test either way; forcing a verdict on a badly-posed comparison is
not evidence in either direction.

Usage: python3 scripts/grid_generation_gate1.py [--days N] [--tolerance F]
"""
import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
REGISTRY_PATH = os.path.join(REPO_ROOT, "datacore", "powerplants", "us_power_plants.json")

# Contiguous-US bounding box (matches EIA-930's own US48 definition: no
# Alaska, Hawaii, Puerto Rico, Guam). A loose box, not a precise coastline
# — fine for excluding entire non-CONUS territories, not for a fine join.
CONUS_BBOX = (24.0, 50.0, -125.0, -66.0)  # (min_lat, max_lat, min_lon, max_lon)

# EIA-930 fuel-type code -> registry fuel bucket (registry only has 8:
# nuclear/coal/gas/hydro/wind/solar/oil/other). PS (hydro pumped storage)
# folds into hydro alongside WAT — our registry does not carry a separate
# "pumped storage" fuel string, it counts pumped-hydro plants under
# "hydro" (spot-checked: e.g. Grand Coulee-scale entries). Every other
# EIA storage/unknown/geothermal code folds into "other" — reported, not
# verdicted, per the module docstring above.
EIA_TO_REGISTRY_FUEL = {
    "NUC": "nuclear",
    "COL": "coal",
    "NG": "gas",
    "OIL": "oil",
    "WAT": "hydro",
    "PS": "hydro",
    "WND": "wind",
    "SUN": "solar",
    "GEO": "other",
    "BAT": "other",
    "OES": "other",
    "UES": "other",
    "UNK": "other",
    "SNB": "other",
    "WNB": "other",
    "OTH": "other",
}

VERDICTED_BUCKETS = ("nuclear", "coal", "gas", "oil", "hydro", "wind", "solar")


def registry_capacity_by_fuel(plants, bbox=CONUS_BBOX):
    """plants: list of [name, capacity_mw, fuel, operator, lat, lon, verified].
    Returns {fuel_bucket: total_capacity_mw}, contiguous-US only."""
    min_lat, max_lat, min_lon, max_lon = bbox
    cap = defaultdict(float)
    excluded = 0
    for p in plants:
        _name, capacity_mw, fuel, _op, lat, lon, _verified = p
        if lat is None or lon is None or not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            excluded += 1
            continue
        cap[fuel] += capacity_mw or 0.0
    return dict(cap), excluded


def aggregate_max_by_fueltype(rows):
    """rows: EIA v2 response.data entries {fueltype, value, ...}.
    Returns {eia_fueltype: max_mwh_in_one_hour} over whatever window rows span.
    A row whose value fails to parse as a float is skipped (can't contribute
    to a max) but the count is surfaced via stderr, not silently dropped —
    a wrong-type value from a live API is itself worth knowing about."""
    out = defaultdict(float)
    seen = set()
    unparseable = 0
    for row in rows:
        v = row.get("value")
        if v in (None, ""):
            continue
        parsed = None
        if isinstance(v, (int, float)):
            parsed = float(v)
        else:
            try:
                parsed = float(v)
            except (TypeError, ValueError):
                unparseable += 1
        if parsed is None:
            continue
        v = parsed
        ft = row.get("fueltype")
        if not ft:
            continue
        seen.add(ft)
        if ft not in out or v > out[ft]:
            out[ft] = v
    if unparseable:
        print(f"[grid_generation_gate1] {unparseable} row(s) had a non-numeric value, skipped", file=sys.stderr)
    # ensure every seen fueltype has an entry even if its max was <= 0
    return {ft: out.get(ft, 0.0) for ft in seen}


def bucket_generation_max(eia_max_by_fueltype):
    """Roll EIA fuel-type maxima up into registry buckets. Conservatively
    SUMS the per-fueltype maxima within a bucket (e.g. hydro = max(WAT) +
    max(PS)) rather than taking a single hour's combined value — this is
    an upper bound on true same-hour combined generation, which only makes
    the exceeds-capacity check MORE conservative (harder to false-negative
    a real registry gap), never less."""
    bucket = defaultdict(float)
    for eia_ft, mwh_max in eia_max_by_fueltype.items():
        registry_fuel = EIA_TO_REGISTRY_FUEL.get(eia_ft)
        if registry_fuel is None:
            continue
        bucket[registry_fuel] += max(0.0, mwh_max)  # storage charging can be negative; floor at 0 for a capacity ceiling check
    return dict(bucket)


def reconcile(registry_cap_mw, generation_max_mwh, tolerance=0.05):
    """Per verdicted fuel bucket: PASS if max hourly generation <=
    capacity * (1 + tolerance); FAIL if it exceeds that ceiling (registry
    likely undercounts real capacity for that fuel). Buckets missing from
    either side are INCONCLUSIVE (no data to compare), never silently
    skipped."""
    out = {}
    for fuel in VERDICTED_BUCKETS:
        cap = registry_cap_mw.get(fuel)
        gen = generation_max_mwh.get(fuel)
        if cap is None or gen is None:
            out[fuel] = {"verdict": "INCONCLUSIVE", "reason": "missing on one side", "capacity_mw": cap, "max_generation_mwh": gen}
            continue
        ceiling = cap * (1 + tolerance)
        ratio = (gen / cap) if cap else float("inf")
        verdict = "FAIL" if gen > ceiling else "PASS"
        out[fuel] = {
            "verdict": verdict,
            "capacity_mw": round(cap, 1),
            "max_generation_mwh": round(gen, 1),
            "ratio_of_capacity": round(ratio, 3),
            "tolerance": tolerance,
        }
    return out


# ── Network (excluded from unit tests, exercised only by running this file directly) ──

def fetch_window(respondent, start_iso_hour, end_iso_hour, api_key, timeout=30):
    url = (
        "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
        f"?api_key={api_key}&frequency=hourly&data%5B0%5D=value"
        f"&facets%5Brespondent%5D%5B%5D={respondent}"
        f"&start={start_iso_hour}&end={end_iso_hour}"
        "&sort%5B0%5D%5Bcolumn%5D=period&sort%5B0%5D%5Bdirection%5D=desc"
        "&length=5000"
    )
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    data = (body.get("response") or {}).get("data") or []
    total = (body.get("response") or {}).get("total")
    if total is not None and int(total) > len(data):
        raise RuntimeError(f"fetch_window: response truncated ({len(data)} of {total} rows) — narrow the window")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="trailing window size in days")
    ap.add_argument("--tolerance", type=float, default=0.05, help="fractional headroom above capacity before FAIL")
    ap.add_argument("--respondent", default="US48")
    args = ap.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("EIA_API_KEY not set — cannot run live gate-1 fetch", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(timezone.utc)
    end = now.strftime("%Y-%m-%dT%H")
    start = (now - timedelta(days=args.days)).strftime("%Y-%m-%dT%H")

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    registry_cap, excluded = registry_capacity_by_fuel(registry["plants"])

    rows = fetch_window(args.respondent, start, end, api_key)
    eia_max = aggregate_max_by_fueltype(rows)
    gen_bucket_max = bucket_generation_max(eia_max)

    verdicts = reconcile(registry_cap, gen_bucket_max, tolerance=args.tolerance)

    report = {
        "root": "grid_generation_fuel_mix",
        "gate": 1,
        "respondent": args.respondent,
        "window": {"start": start, "end": end, "days": args.days},
        "registry_source": os.path.relpath(REGISTRY_PATH, REPO_ROOT),
        "registry_plants_total": registry.get("count"),
        "registry_plants_excluded_noncontiguous": excluded,
        "eia_rows_fetched": len(rows),
        "verdicts": verdicts,
        "other_bucket_reported_not_verdicted": {
            "registry_capacity_mw": round(registry_cap.get("other", 0.0), 1),
            "eia_max_mwh": round(gen_bucket_max.get("other", 0.0), 1),
            "note": "heterogeneous category definitions (battery/geothermal/unknown vs registry catch-all) — not a valid exceeds-capacity test either way",
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
