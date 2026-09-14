#!/usr/bin/env python3
"""eia930_solar_exceedance_pattern.py — FUSION HYPOTHESIS (b)'s gate-1 (DATA)
follow-up to SWPP solar's residual overshoot (research/open_questions.md,
FUSION HYPOTHESES (b), NEXT(2) as of the 2026-09-14 eia860m_add_brand_new_plants
session: "SWPP solar's residual ~28.6% overshoot — still open ... post-
July-2026 commissioning EIA-860M itself misses, or an SWPP-specific EIA-930
respondent quirk, neither checked here").

WHAT THIS RULES OUT, not confirms: every prior session's registry-staleness
investigation (completeness, then currency via EIA-860M) implicitly treated
scripts/grid_generation_gate1.py's own statistic — the SINGLE max hourly
reading over a trailing window — as if any value above the tolerance ceiling
were equally likely to be either (a) a rare, one-off EIA-930 data-quality
spike (a bad revision, a momentary telemetry glitch) or (b) a sustained
physical/methodological pattern. Those two have very different implications
and neither has been distinguished before this script: (a) would mean the
single MAX statistic itself is fragile (one bad hour manufactures a FAIL);
(b) would mean registry-side fixes (completeness, currency — already tried
and only partially effective for SWPP) are chasing the wrong mechanism
entirely, because no plausible registry gap would make generation exceed
true installed capacity on a near-daily, same-hour-of-day basis.

METHOD: pull every hourly reading for one (respondent, EIA fueltype) pair
over a trailing window (default 30 days — 7 is what gate-1 itself checks,
but is too short a baseline to tell "one-off" from "sustained" apart) and
describe, without forcing a verdict, how often and when generation exceeds
a given capacity ceiling: total hours exceeding, the fraction of all hours,
how many DISTINCT CALENDAR DAYS have at least one exceeding hour, and the
hour-of-day histogram of the exceeding hours. A one-off spike shows up as
1-2 days and a scattered/no hour-of-day pattern; a sustained diurnal effect
(inverter AC-clipping headroom, BTM solar embedded in the EIA-930 series,
or anything else that recurs whenever midday irradiance is good) shows up
as many distinct days clustered tightly around the same few hours. This
script deliberately does NOT compute a classifier or verdict from those
numbers (REASONING STANDARD #4 — a threshold tuned by the same session that
already suspects the answer is not evidence) — it reports the distribution;
interpretation is written into research/open_questions.md by a human-
readable session entry, same as every other gate-1 diagnostic in this repo.

NOT a MEASUREMENT INTEGRITY change: this script does not alter
grid_generation_gate1(.py|_ba.py)'s own PASS/FAIL statistic or tolerance —
it is a separate, read-only diagnostic over the same public EIA-930 series,
gate-1 DATA layer only. Any future change to the gate-1 statistic itself
(e.g. away from a bare single-hour max) is its own PR per CLAUDE.md's
MEASUREMENT INTEGRITY section and is NOT proposed or shipped here.

Usage:
    python3 scripts/eia930_solar_exceedance_pattern.py \
        --respondent SWPP --fueltype SUN --capacity-mw 2060.3 --days 30
"""
import argparse
import json
import os
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone

EIA_FUEL_TYPE_DATA_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"


def parse_period_date_hour(period):
    """"2026-09-13T04" -> ("2026-09-13", 4). Raises on a period string that
    doesn't match this shape rather than guessing — every period this API
    has ever returned to this repo's sibling scripts follows it exactly."""
    date_part, sep, hour_part = str(period).partition("T")
    if not sep:
        raise ValueError(f"unexpected period format: {period!r}")
    return date_part, int(hour_part)


def parse_rows(api_rows):
    """api_rows: EIA v2 response.data entries {period, value, ...}. Returns
    (parsed: [(period, value_or_None)], unparseable_count). A non-numeric
    value becomes None (excluded from the exceedance count downstream) but
    its period is kept and the count is returned — surfaced, never silently
    dropped, same convention as grid_generation_gate1.aggregate_max_by_fueltype."""
    out = []
    unparseable = 0
    for row in api_rows:
        period = row.get("period")
        v = row.get("value")
        if v in (None, ""):
            out.append((period, None))
            continue
        try:
            out.append((period, float(v)))
        except (TypeError, ValueError):
            unparseable += 1
            out.append((period, None))
    return out, unparseable


def exceedance_stats(rows, capacity_mw, tolerance=0.0):
    """rows: [(period, value_or_None)]. capacity_mw: the ceiling to compare
    against (the same registry/EIA-860M figure gate-1 itself would use).
    tolerance: identical semantics to grid_generation_gate1.reconcile's own
    tolerance param — an hour counts as "exceeding" only above
    capacity_mw * (1 + tolerance). Returns a pure descriptive report (no
    verdict — see module docstring)."""
    ceiling = capacity_mw * (1 + tolerance) if capacity_mw else None
    hours_total = sum(1 for _, v in rows if v is not None)
    exceeding = []
    if ceiling is not None:
        for period, value in rows:
            if value is not None and value > ceiling:
                ratio = (value / capacity_mw) if capacity_mw else None
                exceeding.append((period, value, ratio))
    hour_hist = Counter()
    days = set()
    for period, _value, _ratio in exceeding:
        date_part, hour = parse_period_date_hour(period)
        hour_hist[hour] += 1
        days.add(date_part)
    exceeding_sorted = sorted(exceeding, key=lambda t: -t[1])
    ratios = [r for _, _, r in exceeding if r is not None]
    return {
        "capacity_mw": capacity_mw,
        "tolerance": tolerance,
        "ceiling_mwh": round(ceiling, 1) if ceiling is not None else None,
        "hours_total": hours_total,
        "hours_exceeding": len(exceeding),
        "exceeding_fraction": round(len(exceeding) / hours_total, 4) if hours_total else None,
        "distinct_days_with_exceedance": len(days),
        "max_ratio_of_capacity": round(max(ratios), 3) if ratios else None,
        "exceeding_hour_of_day_histogram": dict(sorted(hour_hist.items())),
        "top_5_exceeding_hours": [
            {"period": p, "value_mwh": round(v, 1), "ratio_of_capacity": round(r, 3) if r is not None else None}
            for p, v, r in exceeding_sorted[:5]
        ],
    }


# ── Network (excluded from unit tests, exercised only by running this file directly) ──

def fetch_fueltype_window(respondent, fueltype, start_iso_hour, end_iso_hour, api_key, timeout=30):
    """Single-(respondent, fueltype) hourly pull. Deliberately NOT reusing
    grid_generation_gate1.fetch_window (which has no fueltype facet and
    pulls every fuel type at once) — this script's default 30-day window
    is exactly the range where an all-fueltypes pull would risk crossing
    the API's 5000-row page size (720 hours x ~10 fueltypes ~= 7200) and
    hitting that function's own truncation guard; scoping to one fueltype
    keeps a single page correct up to roughly 200 days."""
    url = (
        f"{EIA_FUEL_TYPE_DATA_URL}?api_key={api_key}&frequency=hourly&data%5B0%5D=value"
        f"&facets%5Brespondent%5D%5B%5D={respondent}"
        f"&facets%5Bfueltype%5D%5B%5D={fueltype}"
        f"&start={start_iso_hour}&end={end_iso_hour}"
        "&sort%5B0%5D%5Bcolumn%5D=period&sort%5B0%5D%5Bdirection%5D=desc"
        "&length=5000"
    )
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    resp = body.get("response") or {}
    data = resp.get("data") or []
    total = resp.get("total")
    if total is not None and int(total) > len(data):
        raise RuntimeError(f"fetch_fueltype_window: response truncated ({len(data)} of {total} rows) — narrow the window")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--respondent", required=True)
    ap.add_argument("--fueltype", required=True, help="raw EIA-930 fuel-type code, e.g. SUN")
    ap.add_argument("--capacity-mw", type=float, required=True,
                     help="the ceiling to compare against (registry or EIA-860M capacity for this respondent/fuel)")
    ap.add_argument("--days", type=int, default=30,
                     help="trailing window size in days (default 30 — long enough to tell a one-off spike from a sustained pattern; gate-1 itself checks 7)")
    ap.add_argument("--tolerance", type=float, default=0.0,
                     help="same semantics as grid_generation_gate1's own tolerance (default 0.0 here — this script counts every hour strictly above nameplate, not just ones past gate-1's 5% headroom, since the question is the shape of the whole exceeding distribution, not a pass/fail line)")
    args = ap.parse_args()

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        print("EIA_API_KEY not set — cannot run live fetch", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(timezone.utc)
    end = now.strftime("%Y-%m-%dT%H")
    start = (now - timedelta(days=args.days)).strftime("%Y-%m-%dT%H")

    api_rows = fetch_fueltype_window(args.respondent, args.fueltype, start, end, api_key)
    rows, unparseable = parse_rows(api_rows)
    if unparseable:
        print(f"[eia930_solar_exceedance_pattern] {unparseable} row(s) had a non-numeric value, excluded", file=sys.stderr)

    stats = exceedance_stats(rows, args.capacity_mw, tolerance=args.tolerance)

    report = {
        "check": "eia930_solar_exceedance_pattern",
        "respondent": args.respondent,
        "fueltype": args.fueltype,
        "window": {"start": start, "end": end, "days": args.days},
        "eia_rows_fetched": len(api_rows),
        **stats,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
