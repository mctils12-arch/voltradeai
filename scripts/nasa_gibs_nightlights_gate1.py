#!/usr/bin/env python3
"""
nasa_gibs_nightlights_gate1.py — ROOT VALIDATION LADDER GATE 1 (DATA)
mechanical tile/date-alignment spot-check for the nasa_gibs_nightlights
root (datacore/signal_ladder.json). Closes the specific gap that root's
own note names: "even the mechanical gate-1 tile/date-alignment spot-check
was not yet done."

WHAT THIS CHECKS: does the GIBS layer actually wired into the client's
"nightlights" data toggle (client/src/pages/datamap.tsx,
VIIRS_SNPP_DayNightBand_At_Sensor_Radiance) show elevated brightness over
known BRIGHT locations (major metros) vs known DARK locations (open
ocean, far from any coast) — the most basic possible external-truth check
for a layer whose whole point is "shows city lights." This is NOT the
full metro-radiance-delta-as-GDP-proxy gate-1/gate-2 test (that needs an
archived daily series this root does not have yet) — it is the cheaper,
prerequisite mechanical check: is the raw material even measuring the
right thing before any archiver gets built on top of it.

PRE-REGISTERED PRIOR (REASONING STANDARD #10, stated before the run this
module's docstring describes): the "_At_Sensor_Radiance" product name
signals no lunar/BRDF correction has been applied. Published VIIRS-DNB
literature (Elvidge et al.'s "VIIRS Nighttime Lights" method papers) is
explicit that uncorrected at-sensor radiance is dominated by lunar phase
and cloud-top reflectance, not surface lighting — correction (moon-angle/
BRDF, as GIBS's own "GapFilled_BRDF_Corrected" and the now-discontinued
"ENCC" products name) is what makes a DNB product usable for lights-only
analysis. Prior: SHIPPED_LAYER is EXPECTED to fail a bright-vs-dark
discrimination bar that CANDIDATE_LAYER passes.

PRE-REGISTERED BAR: for a layer to be gate-1-viable for archiving as a
"night lights" series, on EVERY sampled date the mean brightness of
BRIGHT_LOCATIONS (known major-metro tiles) must be at least
BRIGHT_DARK_RATIO_MIN times the mean brightness of DARK_LOCATIONS (known
open-ocean tiles, far from any coast or shipping lane) — "every date", not
"on average", because an archiver that is only sometimes right is not
usable as a daily series without per-day quality filtering nobody has
built yet.

Usage:
  python3 scripts/nasa_gibs_nightlights_gate1.py [--dates D1,D2,...] [--json]
Exit code: 0 = CANDIDATE_LAYER passes the bar on every sampled date,
1 = it does not (or every fetch failed).
"""
import argparse
import io
import json
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

GIBS_BASE = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"
TILE_MATRIX_SET = "GoogleMapsCompatible_Level8"
ZOOM = 6  # coarse tiles (~2500km at the equator) average out sub-tile noise cheaply

# Layer currently wired into client/src/pages/datamap.tsx's "nightlights"
# toggle — what this script is actually gate-checking.
SHIPPED_LAYER = "VIIRS_SNPP_DayNightBand_At_Sensor_Radiance"
# GIBS's live (still-updating as of this session's own GetCapabilities probe,
# 2026-09-07 — ENCC, the other corrected product GIBS offers, stopped
# publishing 2023-07-07) moon/BRDF-corrected alternative. Same
# TileMatrixSet and daily cadence as SHIPPED_LAYER — a drop-in swap.
CANDIDATE_LAYER = "VIIRS_SNPP_GapFilled_BRDF_Corrected_DayNightBand_Radiance"

# z=6 tile (y, x) for each named location — hand-picked, well-known
# bright/dark ground truth, independent of anything this script measures.
# Tile math: standard Web Mercator slippy-tile formula (see tile_xy below).
BRIGHT_LOCATIONS = {
    "vegas": (25, 11),    # Las Vegas / LA / Phoenix basin
    "tokyo": (25, 56),    # Tokyo / Osaka / Seoul corridor
    "london": (21, 31),   # London / Paris / Benelux
}
DARK_LOCATIONS = {
    "ocean_pacific": (32, 5),
    "ocean_atlantic": (40, 30),
    "ocean_indian": (37, 46),
    "ocean_south_pacific": (35, 7),
}
BRIGHT_DARK_RATIO_MIN = 2.0
DEFAULT_DATES = ["2026-06-15", "2026-07-15", "2026-08-15", "2026-09-01"]
FETCH_RETRIES = 3


def tile_xy(lat: float, lon: float, z: int) -> tuple:
    """Pure. Standard Web Mercator slippy-tile (x, y) for a lat/lon at zoom
    z — used only to document/regenerate the hand-picked coordinates above,
    not called at runtime (BRIGHT_LOCATIONS/DARK_LOCATIONS are already
    tile coords, computed with this exact formula, so a location can be
    re-derived or a new one added without re-deriving the math)."""
    import math
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_url(layer: str, date: str, y: int, x: int) -> str:
    return f"{GIBS_BASE}/{layer}/default/{date}/{TILE_MATRIX_SET}/{ZOOM}/{y}/{x}.png"


def fetch_tile(layer: str, date: str, y: int, x: int) -> bytes:
    """Networked. Retries on any error — GIBS occasionally 404s a real
    date/tile combo transiently (observed live this session, not
    hypothetical); a failure after all retries propagates rather than
    being silently treated as a data point."""
    url = tile_url(layer, date, y, x)
    req = urllib.request.Request(url, headers={"User-Agent": "voltradeai-datacore/1.0"})
    last_err = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001 — retried and re-raised, never swallowed
            last_err = e
            if attempt < FETCH_RETRIES:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"GIBS fetch failed after {FETCH_RETRIES} attempts ({layer} {date} {y}/{x}): {last_err}")


def mean_brightness(png_bytes: bytes) -> float:
    """Pure (given bytes already in hand). Mean of the RGB channels over
    every pixel in the tile, 0-255. GIBS DNB tiles are colormapped PNGs
    (not raw radiance values), so this is a brightness PROXY, not a
    physical unit — fine for a discrimination check (is location A
    visibly brighter than location B), wrong for anything claiming an
    absolute radiance number."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return float(np.asarray(img, dtype=np.float64).mean())


def evaluate_date(layer: str, date: str, fetch_fn=None) -> dict:
    """Networked by default (calls fetch_tile for every location);
    `fetch_fn(layer, date, y, x) -> bytes` is injectable for tests. Returns
    per-location brightness plus the bright/dark ratio and whether it
    clears BRIGHT_DARK_RATIO_MIN."""
    fetch_fn = fetch_fn or fetch_tile
    bright = {name: mean_brightness(fetch_fn(layer, date, y, x)) for name, (y, x) in BRIGHT_LOCATIONS.items()}
    dark = {name: mean_brightness(fetch_fn(layer, date, y, x)) for name, (y, x) in DARK_LOCATIONS.items()}
    bright_mean = sum(bright.values()) / len(bright)
    dark_mean = sum(dark.values()) / len(dark)
    ratio = (bright_mean / dark_mean) if dark_mean else float("inf")
    return {
        "date": date, "bright": bright, "dark": dark,
        "bright_mean": bright_mean, "dark_mean": dark_mean,
        "ratio": ratio, "pass": ratio >= BRIGHT_DARK_RATIO_MIN,
    }


def evaluate_layer(layer: str, dates: list, fetch_fn=None) -> dict:
    """Networked by default (see evaluate_date). One entry per date that
    fetched successfully; a date where every retry failed is recorded
    under 'errors', not silently dropped from the pass/fail count."""
    results, errors = [], []
    for date in dates:
        try:
            results.append(evaluate_date(layer, date, fetch_fn=fetch_fn))
        except Exception as e:  # noqa: BLE001 — recorded, not swallowed
            errors.append({"date": date, "error": str(e)})
    verdict = bool(results) and all(r["pass"] for r in results)
    return {"layer": layer, "results": results, "errors": errors, "pass": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default=",".join(DEFAULT_DATES))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]

    shipped = evaluate_layer(SHIPPED_LAYER, dates)
    candidate = evaluate_layer(CANDIDATE_LAYER, dates)

    if args.json:
        print(json.dumps({"shipped": shipped, "candidate": candidate}, indent=2))
    else:
        for label, verdict in (("SHIPPED  " + SHIPPED_LAYER, shipped), ("CANDIDATE " + CANDIDATE_LAYER, candidate)):
            print(f"=== {label} ===  overall {'PASS' if verdict['pass'] else 'FAIL'}")
            for r in verdict["results"]:
                print(f"  {r['date']}: bright_mean={r['bright_mean']:.1f} dark_mean={r['dark_mean']:.1f} "
                      f"ratio={r['ratio']:.2f} bar>={BRIGHT_DARK_RATIO_MIN} -> {'PASS' if r['pass'] else 'FAIL'}")
            for e in verdict["errors"]:
                print(f"  {e['date']}: ERROR {e['error']}")

    sys.exit(0 if candidate["pass"] else 1)


if __name__ == "__main__":
    main()
