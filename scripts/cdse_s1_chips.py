#!/usr/bin/env python3
"""
cdse_s1_chips.py — tank-fill v3 PR-1: Sentinel-1 GRD master-chip acquisition
for the Cushing tank registry (successor root to the gate-1-failed optical
v2; workup: research/open_questions.md "TANK-FILL v3"). One 2-band FLOAT32
GeoTIFF (VV, VH linear sigma0) per scene — radar is cloud-immune and
sun-independent, killing the v2 confound class entirely.

Same proven split as the v2 client (cdse_chips.py):
  discovery: Element84 earth-search STAC (ANONYMOUS) — collection
             sentinel-1-grd carries sat:orbit_state + sat:relative_orbit
             (probed 2026-07-05: Cushing is covered by ASCENDING relative
             orbit 34, S1A + S1D, ~4-6 scenes/month).
  chips:     Sentinel Hub Process API on CDSE (CDSE_CLIENT_ID/SECRET),
             orthorectified SIGMA0_ELLIPSOID (Cushing is flat — terrain
             flattening unnecessary), FLOAT32 so bright double-bounce
             returns (probed up to sigma0 ~2500) are never clamped.

PU: 523x420 px x 2/3 bands x2 (FLOAT32) ~ 1.12 PU/scene; a 24-month
backfill is ~100 PU of the 10,000/month free tier. Same 50% refusal guard.

Outputs:
  datacore/sentinel2/chips_s1/<date>_<sceneid>.tif  (GITIGNORED, ~1.6 MB)
  datacore/sentinel2/s1_chips_index.jsonl           (committed metadata;
      dedup by scene; carries orbit_state + relative_orbit — the estimator
      conditions on orbit, never mixes geometries silently)

Run:
  python3 scripts/cdse_s1_chips.py --since 2024-07-01 --dry-run
  python3 scripts/cdse_s1_chips.py --since 2024-07-01
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

# Same registry-derived bbox as v2 (test-pinned there against all 234
# measurable tanks); the two stacks stay pixel-aligned for fusion.
CHIP_BBOX = [-96.770, 35.922, -96.712, 35.960]
RES_M = 10
BANDS = ["VV", "VH"]
FLOAT32_PU_MULT = 2.0
FREE_TIER_PU_MONTH = 10_000

BASE = os.path.join(os.path.dirname(__file__), "..")
CHIP_DIR = os.path.join(BASE, "datacore", "sentinel2", "chips_s1")
INDEX = os.path.join(BASE, "datacore", "sentinel2", "s1_chips_index.jsonl")

# ── pure geometry / cost (unit-tested, no network) ──────────────────────────

def chip_px(bbox=CHIP_BBOX, res_m=RES_M):
    lat_mid = (bbox[1] + bbox[3]) / 2.0
    w = (bbox[2] - bbox[0]) * 111_320.0 * math.cos(math.radians(lat_mid)) / res_m
    h = (bbox[3] - bbox[1]) * 110_540.0 / res_m
    return int(round(w)), int(round(h))


def pu_estimate(width, height, n_bands=len(BANDS)):
    """Sentinel Hub PU: area/(512^2) x bands/3 x2 for FLOAT32 output."""
    return max(0.005, (width * height) / (512.0 * 512.0) * (n_bands / 3.0) * FLOAT32_PU_MULT)


# ── request builders + parsers (unit-tested, no network) ────────────────────

EVALSCRIPT = """//VERSION=3
// 2-band FLOAT32: linear sigma0 — double-bounce returns reach sigma0 in the
// thousands (probed), so integer sample types would clamp the signal.
function setup() {
  return { input: [{ bands: ["VV", "VH"] }], output: { bands: 2, sampleType: "FLOAT32" } };
}
function evaluatePixel(s) {
  return [s.VV, s.VH];
}
"""


def build_stac_request(since, until, bbox=CHIP_BBOX):
    return {
        "collections": ["sentinel-1-grd"],
        "bbox": bbox,
        "datetime": f"{since}T00:00:00Z/{until}T23:59:59Z",
        "sortby": [{"field": "properties.datetime", "direction": "asc"}],
        "limit": 200,
    }


def parse_stac(payload):
    """Normalize earth-search S1 features. Orbit metadata is REQUIRED for the
    estimator's per-orbit conditioning — scenes missing it are kept with
    nulls and excluded from gate-1 scoring, never guessed."""
    out = []
    for f in payload.get("features", []):
        p = f.get("properties", {})
        dt = p.get("datetime", "")
        pol = p.get("s1:polarizations") or p.get("sar:polarizations") or []
        out.append({
            "scene": f.get("id", ""),
            "date": dt[:10],
            "datetime": dt,
            "orbit_state": p.get("sat:orbit_state"),
            "relative_orbit": p.get("sat:relative_orbit"),
            "polarizations": pol,
            "mode": p.get("sar:instrument_mode"),
        })
    return out


def usable(scene):
    """IW mode with both VV+VH — the registry-wide estimator's contract."""
    return (scene["mode"] == "IW" and "VV" in scene["polarizations"]
            and "VH" in scene["polarizations"] and bool(scene["scene"]))


def pick_scene_per_date(scenes):
    """One scene per date. S1 has no cloud metric; ties (rare — adjacent
    slices of one pass) resolve to the first by id for determinism."""
    by_date = {}
    for s in scenes:
        if not s["date"]:
            continue
        prev = by_date.get(s["date"])
        if prev is None or s["scene"] < prev["scene"]:
            by_date[s["date"]] = s
    return [by_date[d] for d in sorted(by_date)]


def build_process_request(scene_date, bbox=CHIP_BBOX):
    """Day-window timeRange like the v2 client: one S1 pass covers the AOI
    fully, and pick_scene_per_date already reduced to one scene per day."""
    w, h = chip_px(bbox)
    return {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{scene_date}T00:00:00Z",
                        "to": f"{scene_date}T23:59:59Z",
                    },
                },
                "processing": {"orthorectify": True, "backCoeff": "SIGMA0_ELLIPSOID"},
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


# ── network (thin) ──────────────────────────────────────────────────────────

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
    ap.add_argument("--since", default="2024-07-01")
    ap.add_argument("--until", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--max-scenes", type=int, default=0, help="0 = no cap")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    w, h = chip_px()
    per_scene = pu_estimate(w, h)
    cat = http_json(STAC_URL, build_stac_request(args.since, args.until),
                    {"content-type": "application/json"})
    scenes = pick_scene_per_date([s for s in parse_stac(cat) if usable(s)])
    seen = seen_scenes()
    todo = [s for s in scenes if s["scene"] not in seen]
    if args.max_scenes:
        todo = todo[: args.max_scenes]
    spent = month_pu_spent()
    print(f"chip {w}x{h}px FLOAT32 {'+'.join(BANDS)} -> {per_scene:.2f} PU/scene; "
          f"{len(scenes)} usable scenes, {len(todo)} new; "
          f"month PU: {spent:.1f} spent + {per_scene * len(todo):.1f} planned "
          f"of {FREE_TIER_PU_MONTH} free tier")
    if spent + per_scene * len(todo) > FREE_TIER_PU_MONTH * 0.5:
        sys.exit("refusing: planned pull would exceed 50% of the monthly free tier — split the backfill")
    if args.dry_run:
        for s in todo:
            print(f"  {s['date']} {s['scene']} {s['orbit_state']}/{s['relative_orbit']}")
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
            "backscatter": "SIGMA0_ELLIPSOID, orthorectified",
            "pulled_at": datetime.now(timezone.utc).isoformat(),
        })
        append_index(rec)
        print(f"  {s['date']} {s['scene']} ({s['orbit_state']}/{s['relative_orbit']}) -> {fname} ({nbytes/1e6:.1f} MB)")
        time.sleep(1)


if __name__ == "__main__":
    main()
