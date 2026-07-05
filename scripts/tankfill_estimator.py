#!/usr/bin/env python3
"""
tankfill_estimator.py — tank-fill v2 PR-3: per-tank crescent-shadow fill
estimator over the PR-2 master chips (workup: research/open_questions.md
"TANK-FILL % v2"; build plan PR-3 of 7). EXPERIMENTAL — ladder gate 1 not
attempted yet (that is PR-4); nothing here is a SIGNAL.

Physics (known art — floating-roof rim shadow): the shell rim casts a
crescent onto the floating roof on the UP-SUN side of the interior. Shadow
reach s = roof_depth / tan(sun_elev); fill = 1 - depth / shell_height. At
10 m the crescent is SUB-PIXEL most of the year, so the estimator never
looks for the crescent shape — it measures the up-sun vs down-sun half-disk
reflectance ratio of each tank interior (self-normalizing per tank: paint,
band, atmosphere all cancel) and inverts the darkening through circle-lens
geometry to a depth.

STATED ASSUMPTIONS (carried on every reading, per the workup honesty rules):
  - shell height: registry default 14.6 m API-650 (height_src flags it)
  - shadow albedo ratio SHADOW_K: shadowed roof reflects ~35% of lit roof
    (typical cast-shadow ratio in red band; calibration constant, may be
    re-fit in PR-4 against EIA once matched weeks exist)
  - summer readings carry little signal: expected reach < 0.5 px flags the
    per-tank reading quality "subpixel" (the exact artifact that poisoned
    v1 — never hidden, always labeled)

Per-tank values are archived LOW-CONFIDENCE; SITE-level D^2-weighted
aggregates are the candidate signal. Output: one JSON line per scene in
datacore/sentinel2/readings_v2.jsonl (dedup by scene).

Run (session-side; deps numpy + tifffile, never runtime):
  python3 scripts/tankfill_estimator.py            # process all new chips
  python3 scripts/tankfill_estimator.py --scene S2B_14SPE_20260617_0_L2A
"""
import argparse
import json
import math
import os

BASE = os.path.join(os.path.dirname(__file__), "..")
REGISTRY = os.path.join(BASE, "datacore", "sentinel2", "cushing_tanks.geojson")
CHIP_DIR = os.path.join(BASE, "datacore", "sentinel2", "chips")
CHIP_INDEX = os.path.join(BASE, "datacore", "sentinel2", "chips_index.jsonl")
OUT_JSONL = os.path.join(BASE, "datacore", "sentinel2", "readings_v2.jsonl")

RES_M = 10.0
BAND_IDX = {"B02": 0, "B03": 1, "B04": 2, "B08": 3, "SCL": 4}
# SCL classes that invalidate a pixel: 3 cloud shadow, 8/9 cloud med/high,
# 10 cirrus, 0 nodata, 1 saturated
SCL_BAD = {0, 1, 3, 8, 9, 10}
SHADOW_K = 0.35            # shadowed/lit reflectance ratio (stated assumption)
# NO edge margin: the crescent hugs the rim, so excluding edge pixels
# discards exactly the pixels carrying the signal (verified on the synthetic
# battery — a 1 px margin zeroed a 0.5 px-reach crescent entirely). Edge
# handling is done by coverage WEIGHTS in half_weights instead.
MIN_INTERIOR_PX = 12       # below this the halves are too thin to compare
TANK_CLOUD_MAX = 0.10      # >10% masked interior px -> tank skipped that scene
SCENE_CLOUD_MAX = 0.40     # >40% masked chip -> scene skipped entirely
REG_SEARCH_PX = 3          # registration search window (+/- px)
SUBPIXEL_REACH_PX = 0.5    # expected full-depth reach below this -> "subpixel"


# ── geometry (pure, unit-tested) ────────────────────────────────────────────

def geo_to_px(lon, lat, bbox, width, height):
    """WGS84 -> fractional pixel (x right, y down) for a north-up chip."""
    x = (lon - bbox[0]) / (bbox[2] - bbox[0]) * width
    y = (bbox[3] - lat) / (bbox[3] - bbox[1]) * height
    return x, y


def lens_darkening_to_depth(ratio, radius_px, sun_elev_deg, shadow_k=SHADOW_K):
    """Invert the up-sun/down-sun ratio to a roof depth (meters).

    For shadow reach s (px), the crescent area ~ 2*R*s (small-s lens
    complement), all of it in the up-sun half (area pi*R^2/2), so the
    shadowed fraction of that half f = 4*s / (pi*R). With shadowed pixels
    at shadow_k times lit reflectance:
        ratio = 1 - f*(1 - shadow_k)  =>  s = (1 - ratio) * pi * R / (4*(1-shadow_k))
    depth = s * tan(sun_elev), converted px -> meters.
    Negative darkening (up-sun BRIGHTER) clamps to depth 0."""
    dark = max(0.0, 1.0 - ratio)
    s_px = dark * math.pi * radius_px / (4.0 * (1.0 - shadow_k))
    return s_px * RES_M * math.tan(math.radians(sun_elev_deg))


def expected_reach_px(height_m, sun_elev_deg):
    """Shadow reach at FULL depth (empty tank) — the reading's sensitivity."""
    return height_m / math.tan(math.radians(sun_elev_deg)) / RES_M


# ── raster ops (numpy imported lazily like the sibling scripts) ─────────────

def registration_offset(band, ref_band, search=REG_SEARCH_PX):
    """Integer (dx, dy) CONTENT shift of `band` relative to `ref_band`
    (positive dx = the scene's content sits dx px further right than the
    reference), found by maximizing Pearson r over the common window. ~1 px
    orthorectification jitter is the workup's stated error source; adding
    (dx, dy) to reference-frame tank positions keeps masks on the tanks."""
    import numpy as np
    best, best_r = (0, 0), -2.0
    h, w = band.shape
    c = band[search:h - search, search:w - search].astype(np.float64)
    c = c - c.mean()
    for sy in range(-search, search + 1):
        for sx in range(-search, search + 1):
            # content shifted by (sx, sy): band[y, x] == ref[y - sy, x - sx]
            r_win = ref_band[search - sy:h - search - sy, search - sx:w - search - sx].astype(np.float64)
            r_win = r_win - r_win.mean()
            denom = math.sqrt(float((c * c).sum()) * float((r_win * r_win).sum()))
            if denom == 0:
                continue
            r = float((c * r_win).sum()) / denom
            if r > best_r:
                best_r, best = r, (sx, sy)
    return best, best_r


def half_weights(shape, cx, cy, radius_px, sun_azim_deg, ss=4):
    """Per-pixel COVERAGE weights (w_up, w_dn) for the up-sun / down-sun
    halves of the tank disk, computed by ss x ss supersampling, plus the
    window (y0, y1, x0, x1) they apply to.

    Why weights instead of boolean halves: assigning whole pixels to a half
    by their center puts the entire proj==0 boundary line (mostly bright
    interior pixels) in one half — measured ~5% false darkening on a
    synthetic FULL tank. Fractional membership makes the two halves exactly
    symmetric, and mixed tank/ground edge pixels then bias both halves
    equally and cancel in the ratio.

    Coordinate convention everywhere: continuous, pixel i spans [i, i+1)
    (center i + 0.5) — matches geo_to_px. Azimuth is compass deg; image y
    grows southward; the up-sun half faces the sun, where the crescent falls."""
    import numpy as np
    x0 = max(0, int(math.floor(cx - radius_px - 1)))
    x1 = min(shape[1], int(math.ceil(cx + radius_px + 1)))
    y0 = max(0, int(math.floor(cy - radius_px - 1)))
    y1 = min(shape[0], int(math.ceil(cy + radius_px + 1)))
    if x1 <= x0 or y1 <= y0:
        return None
    nh, nw = y1 - y0, x1 - x0
    yy, xx = np.mgrid[0:nh * ss, 0:nw * ss]
    sx = x0 + (xx + 0.5) / ss
    sy = y0 + (yy + 0.5) / ss
    dx = sx - cx
    dy = sy - cy
    inside = (dx * dx + dy * dy) < radius_px ** 2
    az = math.radians(sun_azim_deg)
    ux, uy = math.sin(az), -math.cos(az)
    proj = dx * ux + dy * uy
    w_up = (inside & (proj > 0)).reshape(nh, ss, nw, ss).mean(axis=(1, 3))
    w_dn = (inside & (proj <= 0)).reshape(nh, ss, nw, ss).mean(axis=(1, 3))
    return (y0, y1, x0, x1), w_up, w_dn


def measure_tank(chip, cx, cy, radius_px, sun_azim_deg):
    """Coverage-weighted up-sun/down-sun B04 ratio for one tank, SCL-masked.
    Returns (ratio, n_px, masked_frac) or None if unusable."""
    import numpy as np
    hw = half_weights(chip.shape[:2], cx, cy, radius_px, sun_azim_deg)
    if hw is None:
        return None
    (y0, y1, x0, x1), w_up, w_dn = hw
    n_int = float(w_up.sum() + w_dn.sum())
    if n_int < MIN_INTERIOR_PX:
        return None
    scl = chip[y0:y1, x0:x1, BAND_IDX["SCL"]]
    good = ~np.isin(scl, list(SCL_BAD))
    masked_frac = 1.0 - float((w_up + w_dn)[good].sum()) / n_int
    if masked_frac > TANK_CLOUD_MAX:
        return None
    b04 = chip[y0:y1, x0:x1, BAND_IDX["B04"]].astype(np.float64)
    wu = w_up * good
    wd = w_dn * good
    if wu.sum() < MIN_INTERIOR_PX / 3.0 or wd.sum() < MIN_INTERIOR_PX / 3.0:
        return None
    dn_mean = float((wd * b04).sum() / wd.sum())
    if dn_mean <= 0:
        return None
    up_mean = float((wu * b04).sum() / wu.sum())
    return up_mean / dn_mean, int(round(n_int)), masked_frac


# ── scene processing ────────────────────────────────────────────────────────

def load_registry(path=REGISTRY):
    with open(path) as fh:
        reg = json.load(fh)
    return [
        {
            "id": f["properties"]["id"],
            "lon": f["geometry"]["coordinates"][0],
            "lat": f["geometry"]["coordinates"][1],
            "diameter_m": f["properties"]["diameter_m"],
            "site": f["properties"]["site"],
            "height_m": f["properties"]["height_m"],
            "height_src": f["properties"]["height_src"],
        }
        for f in reg["features"]
        if f["properties"]["measurable_10m"]
    ]


def process_scene(chip, meta, tanks, ref_band=None):
    """One scene -> reading dict (one committed JSONL line). `meta` is the
    chips_index record (bbox, sun angles); `ref_band` enables registration."""
    import numpy as np
    scl = chip[..., BAND_IDX["SCL"]]
    scene_masked = float(np.isin(scl, list(SCL_BAD)).mean())
    if scene_masked > SCENE_CLOUD_MAX:
        return {"scene": meta["scene"], "date": meta["date"], "skipped": "cloud",
                "scene_masked_frac": round(scene_masked, 3)}
    if meta.get("sun_elev") is None or meta.get("sun_azim") is None:
        return {"scene": meta["scene"], "date": meta["date"], "skipped": "no_sun_angles"}

    dx = dy = 0
    reg_r = None
    if ref_band is not None:
        (dx, dy), reg_r = registration_offset(chip[..., BAND_IDX["B04"]], ref_band)

    h, w = chip.shape[:2]
    bbox = meta["bbox"]
    per_tank = []
    # NOTE: key is skipped_tanks, not "skipped" — the scene-level skip marker
    # uses "skipped" and a collision made every good reading print as skipped
    skipped = {"cloud": 0, "edge": 0, "small": 0}
    for t in tanks:
        x, y = geo_to_px(t["lon"], t["lat"], bbox, w, h)
        x, y = x + dx, y + dy
        r_px = t["diameter_m"] / 2.0 / RES_M
        if not (r_px + 1 <= x <= w - r_px - 1 and r_px + 1 <= y <= h - r_px - 1):
            skipped["edge"] += 1
            continue
        m = measure_tank(chip, x, y, r_px, meta["sun_azim"])
        if m is None:
            skipped["cloud"] += 1
            continue
        ratio, n_px, masked_frac = m
        depth = lens_darkening_to_depth(ratio, r_px, meta["sun_elev"])
        fill = max(0.0, min(1.0, 1.0 - depth / t["height_m"]))
        reach = expected_reach_px(t["height_m"], meta["sun_elev"])
        per_tank.append({
            "id": t["id"], "site": t["site"], "d_m": t["diameter_m"],
            "ratio": round(ratio, 5), "depth_m": round(depth, 2),
            "fill": round(fill, 4), "px": n_px,
            "q": "subpixel" if reach < SUBPIXEL_REACH_PX else "ok",
            "height_src": t["height_src"],
        })

    sites = {}
    for r in per_tank:
        sites.setdefault(r["site"], []).append(r)
    site_aggs = []
    for site, rows in sorted(sites.items()):
        wsum = sum(x["d_m"] ** 2 for x in rows)
        fill_w = sum(x["fill"] * x["d_m"] ** 2 for x in rows) / wsum
        fills = sorted(x["fill"] for x in rows)
        site_aggs.append({
            "site": site, "n_tanks": len(rows),
            "fill_d2w": round(fill_w, 4),
            "fill_median": round(fills[len(fills) // 2], 4),
        })

    return {
        "scene": meta["scene"], "date": meta["date"],
        "sun_elev": round(meta["sun_elev"], 2), "sun_azim": round(meta["sun_azim"], 2),
        "registration": {"dx": dx, "dy": dy, "r": round(reg_r, 4) if reg_r is not None else None},
        "scene_masked_frac": round(scene_masked, 3),
        "n_measured": len(per_tank), "skipped_tanks": skipped,
        "sites": site_aggs,
        "tanks": per_tank,
        "assumptions": {"shadow_k": SHADOW_K, "height_default_m": 14.6},
        "label": "EXPERIMENTAL — per-tank crescent estimator, ladder gate 1 not attempted",
    }


# ── archive plumbing (same pattern as sibling scripts) ─────────────────────

def seen_scenes(path=OUT_JSONL):
    seen = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line)["scene"])
                except Exception:
                    pass
    return seen


def chip_meta(index_path=CHIP_INDEX):
    out = []
    if os.path.exists(index_path):
        with open(index_path) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def main():
    import numpy as np
    import tifffile
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="process one scene id only")
    args = ap.parse_args()

    tanks = load_registry()
    seen = seen_scenes()
    metas = [m for m in chip_meta() if m["scene"] not in seen]
    if args.scene:
        metas = [m for m in metas if m["scene"] == args.scene]
    metas.sort(key=lambda m: m["date"])
    if not metas:
        print("no new chips to process")
        return

    # reference for registration: the earliest already-processed low-cloud
    # chip, else the first of this batch (which then defines the frame)
    ref_band = None
    processed = []
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    for meta in metas:
        fp = os.path.join(CHIP_DIR, meta["file"])
        if not os.path.exists(fp):
            print(f"  ! chip missing on disk (gitignored — re-pull via cdse_chips.py): {meta['file']}")
            continue
        chip = tifffile.imread(fp)
        reading = process_scene(chip, meta, tanks, ref_band)
        if ref_band is None and "skipped" not in reading:
            ref_band = chip[..., BAND_IDX["B04"]].copy()
        with open(OUT_JSONL, "a") as fh:
            fh.write(json.dumps(reading) + "\n")
        processed.append(reading)
        if "skipped" in reading:
            print(f"  {meta['date']} {meta['scene']}: SKIPPED ({reading['skipped']})")
        else:
            aggs = ", ".join(f"{s['site']}={s['fill_d2w']:.3f}(n={s['n_tanks']})" for s in reading["sites"])
            print(f"  {meta['date']} {meta['scene']}: {reading['n_measured']} tanks, reg=({reading['registration']['dx']},{reading['registration']['dy']}) | {aggs}")
    print(f"appended {len(processed)} readings -> {os.path.relpath(OUT_JSONL)}")


if __name__ == "__main__":
    main()
