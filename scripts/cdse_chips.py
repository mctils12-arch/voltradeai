#!/usr/bin/env python3
"""
cdse_chips.py — tank-fill v2 PR-2: master-chip acquisition for the Cushing OK
tank registry (workup: research/open_questions.md "TANK-FILL % v2"; build plan
PR-2 of 7). Pulls ONE 5-band UINT16 GeoTIFF chip per usable Sentinel-2 scene.

Two sources, each the proven path for its half (probed live 2026-07-05):
  discovery: Element84 earth-search STAC (ANONYMOUS — the v1 pipeline's
             source) — dates, cloud %, AND per-scene sun elevation/azimuth.
             The credentialed CDSE SH Catalog was probed first and does NOT
             expose sun angles at all (default property set is constellation/
             datetime/eo:cloud_cover/gsd/proj:*; fields.include of view:*
             returns nothing) — so discovery stays keyless and consistent
             with sentinel2_tankfill.py.
  chips:     Sentinel Hub Process API on CDSE (credentialed; proven with a
             real S1 chip 2026-07-05, wishlist.md) — server-side AOI window,
             5 bands, UINT16, free-tier processing units.

Session-run, never on Railway. Credentials: CDSE_CLIENT_ID / CDSE_CLIENT_SECRET
in the session environment (free tier; values are never printed or archived).

BBOX CORRECTION vs the filed workup (found 2026-07-05 building this): the
workup bbox [-96.80, 35.90, -96.72, 35.98] MISSES 20 of the registry's 234
measurable tanks (registry extends east to lon -96.7149) while wasting area
west/north. CHIP_BBOX below is derived from the registry extent plus ~250 m
margin, covers ALL measurable tanks, and is ~1/3 the pixel area —
~1.3 PU/scene instead of the workup's ~4.1 (free tier 10,000 PU/month).
test_cdse_chips.py pins coverage against the live registry file so the bbox
can never silently miss tanks again.

Outputs:
  datacore/sentinel2/chips/<date>_<sceneid>.tif   (GITIGNORED — binary, ~2 MB)
  datacore/sentinel2/chips_index.jsonl            (committed metadata: scene,
      date, sun angles, cloud %, bytes, est. processing units; dedup by scene)

DECODE NOTE for PR-3 (verified on a real chip 2026-07-05): the workup said
"pillow decodes the UINT16 TIFF" — WRONG: pillow cannot read 5-sample/pixel
TIFFs. tifffile reads it directly ((H, W, 5) uint16). tifffile is a
session-local dep like rasterio/xlrd for the v1 script — never a runtime dep.

Run:
  python3 scripts/cdse_chips.py --since 2026-06-01 --dry-run   # list + PU cost
  python3 scripts/cdse_chips.py --since 2026-06-01 --max-scenes 5
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# lon/lat WGS84 [W, S, E, N]; derived from cushing_tanks.geojson measurable
# extent (lon -96.7666..-96.7149, lat 35.9246..35.9573) + ~250 m margin.
CHIP_BBOX = [-96.770, 35.922, -96.712, 35.960]
RES_M = 10
BANDS = ["B02", "B03", "B04", "B08", "SCL"]
MAX_CLOUD = 40.0  # discovery prefilter; per-pixel SCL mask does the real work in PR-3
FREE_TIER_PU_MONTH = 10_000

BASE = os.path.join(os.path.dirname(__file__), "..")
REGISTRY = os.path.join(BASE, "datacore", "sentinel2", "cushing_tanks.geojson")
CHIP_DIR = os.path.join(BASE, "datacore", "sentinel2", "chips")
INDEX = os.path.join(BASE, "datacore", "sentinel2", "chips_index.jsonl")

# ── pure geometry / cost (unit-tested, no network) ──────────────────────────

def chip_px(bbox=CHIP_BBOX, res_m=RES_M):
    """Output raster size for a WGS84 bbox at res_m meters/pixel."""
    lat_mid = (bbox[1] + bbox[3]) / 2.0
    w = (bbox[2] - bbox[0]) * 111_320.0 * math.cos(math.radians(lat_mid)) / res_m
    h = (bbox[3] - bbox[1]) * 110_540.0 / res_m
    return int(round(w)), int(round(h))


def pu_estimate(width, height, n_bands=len(BANDS)):
    """Sentinel Hub processing-unit estimate: area/(512^2) x bands/3, floor
    0.005. UINT16 output carries no extra multiplier (only FLOAT32 does)."""
    return max(0.005, (width * height) / (512.0 * 512.0) * (n_bands / 3.0))


def bbox_covers(bbox, lon, lat, margin_deg=0.0):
    return (bbox[0] + margin_deg <= lon <= bbox[2] - margin_deg
            and bbox[1] + margin_deg <= lat <= bbox[3] - margin_deg)


# ── request builders + parsers (unit-tested, no network) ────────────────────

EVALSCRIPT = """//VERSION=3
// 5-band UINT16 master chip: reflectance bands as DN (x10000), SCL raw.
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "B08", "SCL"], units: "DN" }],
    output: { bands: 5, sampleType: "UINT16" }
  };
}
function evaluatePixel(s) {
  return [s.B02, s.B03, s.B04, s.B08, s.SCL];
}
"""


def build_stac_request(since, until, bbox=CHIP_BBOX, max_cloud=MAX_CLOUD):
    return {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{since}T00:00:00Z/{until}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.datetime", "direction": "asc"}],
        "limit": 200,
    }


def parse_stac(payload):
    """Normalize earth-search features -> scene dicts. Scenes missing sun
    angles are kept with nulls — PR-3 skips those, never guesses."""
    out = []
    for f in payload.get("features", []):
        p = f.get("properties", {})
        dt = p.get("datetime", "")
        out.append({
            "scene": f.get("id", ""),
            "date": dt[:10],
            "datetime": dt,
            "cloud_pct": p.get("eo:cloud_cover"),
            "sun_elev": p.get("view:sun_elevation"),
            "sun_azim": p.get("view:sun_azimuth"),
        })
    return out


def pick_scene_per_date(scenes):
    """One scene per calendar date: lowest cloud (both UTM tiles 14SPE/14SQE
    can cover Cushing — same rule as sentinel2_tankfill.py v1)."""
    by_date = {}
    for s in scenes:
        if not s["date"] or not s["scene"]:
            continue
        prev = by_date.get(s["date"])
        if prev is None or (s["cloud_pct"] or 1e9) < (prev["cloud_pct"] or 1e9):
            by_date[s["date"]] = s
    return [by_date[d] for d in sorted(by_date)]


def build_process_request(scene_date, bbox=CHIP_BBOX):
    """Process API payload for one scene-date's master chip. timeRange is the
    scene's calendar day with mosaickingOrder leastCC: the AOI sits fully
    inside both covering tiles, so the day's least-cloudy scene is selected —
    the same rule pick_scene_per_date applied at discovery."""
    w, h = chip_px(bbox)
    return {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{scene_date}T00:00:00Z",
                        "to": f"{scene_date}T23:59:59Z",
                    },
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width": w,
            "height": h,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }


# ── index (dedup + PU accounting; unit-tested via tmp paths) ────────────────

def seen_scenes(index_path=INDEX):
    seen = set()
    if os.path.exists(index_path):
        with open(index_path) as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line)["scene"])
                except Exception:
                    pass
    return seen


def month_pu_spent(index_path=INDEX, now=None):
    """Estimated PU already spent this calendar month (UTC), from the index."""
    now = now or datetime.now(timezone.utc)
    ym = now.strftime("%Y-%m")
    total = 0.0
    if os.path.exists(index_path):
        with open(index_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if str(r.get("pulled_at", "")).startswith(ym):
                        total += float(r.get("est_pu", 0))
                except Exception:
                    pass
    return total


def append_index(rec, index_path=INDEX):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


# ── network (thin; everything above stays testable without it) ─────────────

def http_json(url, body, headers, timeout=60):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get_token():
    cid = os.environ.get("CDSE_CLIENT_ID")
    sec = os.environ.get("CDSE_CLIENT_SECRET")
    if not cid or not sec:
        sys.exit("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set (session env; see wishlist.md CDSE entry)")
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": cid,
        "client_secret": sec,
    }).encode()
    tok = http_json(TOKEN_URL, body, {"content-type": "application/x-www-form-urlencoded"})
    return tok["access_token"]


def fetch_chip(scene, token, chip_dir=CHIP_DIR):
    payload = build_process_request(scene["date"])
    req = urllib.request.Request(
        PROCESS_URL,
        data=json.dumps(payload).encode(),
        headers={
            "content-type": "application/json",
            "accept": "image/tiff",
            "authorization": f"Bearer {token}",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        tif = r.read()
    os.makedirs(chip_dir, exist_ok=True)
    fname = f"{scene['date']}_{scene['scene']}.tif"
    with open(os.path.join(chip_dir, fname), "wb") as fh:
        fh.write(tif)
    return fname, len(tif)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-06-01")
    ap.add_argument("--until", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--max-scenes", type=int, default=0, help="0 = no cap")
    ap.add_argument("--dry-run", action="store_true", help="list scenes + PU cost, pull nothing")
    args = ap.parse_args()

    w, h = chip_px()
    per_scene = pu_estimate(w, h)
    cat = http_json(STAC_URL, build_stac_request(args.since, args.until),
                    {"content-type": "application/json"})
    scenes = pick_scene_per_date(parse_stac(cat))
    seen = seen_scenes()
    todo = [s for s in scenes if s["scene"] not in seen]
    if args.max_scenes:
        todo = todo[: args.max_scenes]
    spent = month_pu_spent()
    print(f"chip {w}x{h}px, {len(BANDS)} bands -> {per_scene:.2f} PU/scene; "
          f"{len(scenes)} usable scenes, {len(todo)} new; "
          f"month PU: {spent:.1f} spent + {per_scene * len(todo):.1f} planned "
          f"of {FREE_TIER_PU_MONTH} free tier")
    if spent + per_scene * len(todo) > FREE_TIER_PU_MONTH * 0.5:
        sys.exit("refusing: planned pull would exceed 50% of the monthly free tier — split the backfill")
    if args.dry_run:
        for s in todo:
            print(f"  {s['date']} {s['scene']} cloud={s['cloud_pct']} sun_elev={s['sun_elev']}")
        return

    token = get_token()
    for s in todo:
        try:
            fname, nbytes = fetch_chip(s, token)
        except Exception as e:
            print(f"  ! {s['scene']}: {e}", file=sys.stderr)
            continue
        rec = dict(s)
        rec.update({
            "file": fname,
            "bytes": nbytes,
            "est_pu": round(per_scene, 3),
            "bbox": CHIP_BBOX,
            "bands": BANDS,
            "pulled_at": datetime.now(timezone.utc).isoformat(),
        })
        append_index(rec)
        print(f"  {s['date']} {s['scene']} -> {fname} ({nbytes/1e6:.1f} MB)")
        time.sleep(1)  # polite pacing; free tier also rate-limits by requests/min


if __name__ == "__main__":
    main()
