#!/usr/bin/env python3
"""
nightlights_archiver.py — daily VIIRS/SNPP corrected night-lights brightness
archive (datacore/signal_ladder.json id: nasa_gibs_nightlights). Builds the
NEXT item that same-day session's own gate-1 fix (scripts/
nasa_gibs_nightlights_gate1.py, v1.0.864) named as the real prerequisite for
the "metro-radiance-delta-as-GDP-proxy" hypothesis: "build a daily archiver
on the corrected layer... before attempting that hypothesis's own gate 1."
This session builds GATE 0 (RAW) storage only — no signal claim, no gate-1
attempt against ground truth (that needs weeks of accumulated points and its
own independent external comparison, e.g. a national statistics office
activity index; not attempted here).

WHY THESE 3 LOCATIONS, NOT NEW ONES: reuses gate1's own BRIGHT_LOCATIONS
(vegas/tokyo/london) and DARK_LOCATIONS (4 open-ocean controls) verbatim via
sibling-module import rather than inventing new tile coordinates this
session has not verified. Gate1 already proved CANDIDATE_LAYER clears the
bright/dark discrimination bar at these exact tiles on every one of its 3
successfully-fetched sampled dates — building on unverified new coordinates
here would be a new, unvalidated claim, not a use of gate1's result.

LAYER: CANDIDATE_LAYER only (VIIRS_SNPP_GapFilled_BRDF_Corrected_
DayNightBand_Radiance) — the layer gate1 found passes the bar and that
datamap.tsx now ships. SHIPPED_LAYER (the failing uncorrected product) is
never archived; there is nothing worth accumulating from a product gate1
showed is dominated by lunar phase/cloud reflectance.

DATE LAG: GIBS's own GetCapabilities `<Default>` for this layer read
2026-09-06 when queried 2026-09-07 (recorded in KNOWN STATE) — a ~1-day
publish lag, not same-day. This script tries "yesterday" UTC first, then
walks further back through MAX_LAG_DAYS_TRIED-1 more days, on EACH run —
skipping any date already fully captured and stopping at the first
not-yet-captured date that fetches cleanly. A run the day after a gap
(missed session, or a real GIBS 404 per gate1's own 2026-07-15 whole-globe-
gap finding) opportunistically backfills the most recent hole instead of
only ever reaching for a fixed single date; a fully caught-up run is a
no-op (every candidate date already captured).

QUALITY FLAG: gate1's own module docstring says an uncorrected-for-day
metric "is not usable as a daily series without per-day quality filtering
nobody has built yet." This script builds that filter: every captured day
also stores its own bright/dark ratio and whether it clears gate1's
BRIGHT_DARK_RATIO_MIN, computed from the SAME dark-location tiles gate1
validated, so a future consumer can honestly exclude a low-quality day
(cloud cover, a transient GIBS gap) instead of trusting every point equally.

STORAGE: datacore/nightlights_metro_brightness.json — APPEND-ONLY per
location (one point per location per UTC date, never re-fetched or
overwritten once captured — same "captured once" discipline as
scripts/un_comtrade_ingest.py's merge_into_series / server/
portDwellCapture.ts's captureIfDue()).

Run (session-side; keyless, no Railway wiring — same seeded pattern as
scripts/un_comtrade_ingest.py/scripts/jodi_oil.py):
  python3 scripts/nightlights_archiver.py [--date YYYY-MM-DD] [--out PATH]
"""
import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone

_gate1_spec = importlib.util.spec_from_file_location(
    "nasa_gibs_nightlights_gate1",
    os.path.join(os.path.dirname(__file__), "nasa_gibs_nightlights_gate1.py"))
gate1 = importlib.util.module_from_spec(_gate1_spec)
_gate1_spec.loader.exec_module(gate1)

OUT = os.path.join(os.path.dirname(__file__), "..", "datacore", "nightlights_metro_brightness.json")
ATTRIBUTION = "NASA GIBS / VIIRS SNPP (GapFilled BRDF-Corrected Day/Night Band Radiance)"
LICENSE = "public domain (NASA)"
MAX_LAG_DAYS_TRIED = 3


def candidate_dates(as_of: datetime, max_tries: int) -> list:
    """Pure. 'Yesterday' UTC first (GIBS's observed ~1-day publish lag),
    then progressively further back — a caller stops at the first date that
    fetches cleanly, so this just enumerates fallback candidates in order."""
    return [(as_of - timedelta(days=n)).strftime("%Y-%m-%d") for n in range(1, max_tries + 1)]


def already_captured(series: dict, date: str) -> bool:
    """True only if EVERY tracked location (bright + dark) already has a
    point for this date — a partial prior capture (e.g. an earlier run that
    fetched some locations before failing) is not treated as done."""
    all_locations = list(gate1.BRIGHT_LOCATIONS) + list(gate1.DARK_LOCATIONS)
    return all(
        any(p[0] == date for p in series.get(name, {}).get("points", []))
        for name in all_locations
    )


def capture_date(date: str, fetch_fn=None) -> dict:
    """Networked by default (calls gate1.fetch_tile for every location via
    the injectable fetch_fn, same convention as gate1.evaluate_date).
    Returns per-location brightness plus the bright/dark quality ratio.
    Raises if ANY location fails to fetch — a partial day is not archived
    (never mixes real and missing readings under one date)."""
    fetch_fn = fetch_fn or gate1.fetch_tile
    bright = {name: gate1.mean_brightness(fetch_fn(gate1.CANDIDATE_LAYER, date, y, x))
              for name, (y, x) in gate1.BRIGHT_LOCATIONS.items()}
    dark = {name: gate1.mean_brightness(fetch_fn(gate1.CANDIDATE_LAYER, date, y, x))
            for name, (y, x) in gate1.DARK_LOCATIONS.items()}
    bright_mean = sum(bright.values()) / len(bright)
    dark_mean = sum(dark.values()) / len(dark)
    ratio = (bright_mean / dark_mean) if dark_mean else float("inf")
    return {
        "date": date, "bright": bright, "dark": dark,
        "ratio": ratio, "quality_pass": ratio >= gate1.BRIGHT_DARK_RATIO_MIN,
    }


def merge_capture(series: dict, capture: dict) -> int:
    """Mutates `series` ({location: {"kind": "bright"|"dark", "points":
    [[date, brightness, ratio, quality_pass], ...]}}) in place. Skips any
    location that already has a point for this date (append-only, never
    overwrite — mirrors un_comtrade_ingest.merge_into_series). Returns the
    count of NEW points actually added."""
    added = 0
    date = capture["date"]
    for kind, readings in (("bright", capture["bright"]), ("dark", capture["dark"])):
        for name, brightness in readings.items():
            s = series.setdefault(name, {"kind": kind, "points": []})
            if any(p[0] == date for p in s["points"]):
                continue
            s["points"].append([date, brightness, capture["ratio"], capture["quality_pass"]])
            added += 1
    for s in series.values():
        s["points"].sort(key=lambda p: p[0])
    return added


def build_artifact(series: dict, now_iso: str) -> dict:
    for s in series.values():
        pts = s["points"]
        s["n"] = len(pts)
        s["first"] = pts[0][0] if pts else None
        s["last"] = pts[-1][0] if pts else None
    latest = max((s["last"] for s in series.values() if s["last"]), default=None)
    quality_pass_days = sorted({p[0] for s in series.values() for p in s["points"] if p[3]})
    return {
        "source": gate1.GIBS_BASE,
        "layer": gate1.CANDIDATE_LAYER,
        "tile_matrix_set": gate1.TILE_MATRIX_SET,
        "zoom": gate1.ZOOM,
        "attribution": ATTRIBUTION,
        "license": LICENSE,
        "built_at": now_iso,
        "point_fields": ["date", "brightness_0_255", "bright_dark_ratio_that_day", "quality_pass"],
        "quality_bar": f">= {gate1.BRIGHT_DARK_RATIO_MIN}x bright/dark ratio (gate1's own pre-registered bar)",
        "latest_date": latest,
        "quality_pass_dates": quality_pass_days,
        "series_count": len(series),
        "series": series,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD; default tries yesterday UTC then falls back")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        series = existing.get("series", {})
        print(f"loaded existing archive: {existing.get('series_count', 0)} series, latest {existing.get('latest_date')}")
    else:
        series = {}
        print("no existing archive — starting fresh")

    now = datetime.now(timezone.utc)
    dates_to_try = [args.date] if args.date else candidate_dates(now, MAX_LAG_DAYS_TRIED)

    capture = None
    tried = []
    for date in dates_to_try:
        if already_captured(series, date):
            print(f"  {date}: already fully captured, skipping")
            continue
        try:
            capture = capture_date(date)
            break
        except Exception as e:  # noqa: BLE001 — recorded, tried next candidate
            tried.append((date, str(e)))
            print(f"  {date}: fetch failed ({e})")

    if capture is None:
        if not tried and not dates_to_try:
            print("no candidate dates to try")
        elif not tried:
            print("nothing to do — every candidate date already captured")
            return 0
        else:
            print(f"FAILED: no candidate date fetched cleanly after {len(tried)} attempt(s)")
            return 1
    else:
        added = merge_capture(series, capture)
        print(f"  {capture['date']}: ratio={capture['ratio']:.2f} quality_pass={capture['quality_pass']} "
              f"({added} new points)")

    artifact = build_artifact(series, now.isoformat())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=1, sort_keys=True)
        f.write("\n")

    if args.json:
        print(json.dumps(artifact, indent=2))
    print(f"WROTE {args.out}: {artifact['series_count']} series, latest {artifact['latest_date']}, "
          f"{len(artifact['quality_pass_dates'])} quality-pass day(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
