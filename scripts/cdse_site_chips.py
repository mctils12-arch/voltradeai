#!/usr/bin/env python3
"""
cdse_site_chips.py — Latest Sentinel-2 (cloud-free) facility chip, per
strategic site (DATACORE MAXIMUS Phase 3b, filed in wishlist.md; the
generalization of the Cushing tank-shadow chip pattern per
datacore/SENTINEL2_CHANGE_SPEC.md's "RAW imagery surfaces before any
index does" ladder step). This is a RAW OVERLAY, not a signal: a real,
recent, cloud-filtered true-color photo of each of the 16 imagery-
verified strategic sites (datacore/sites/strategic_sites.json), refreshed
on demand. No index/measurement is computed here — that stays the
separate, still-gate-1-failed tank-shadow line (sentinel2/sentinel2v2/
sentinel1readings manifests).

Two sources, same split as scripts/cdse_chips.py (proven pattern, reused
not re-derived): discovery via Element84 earth-search STAC (keyless,
gives per-scene whole-scene cloud_cover); the chip itself via the
Sentinel Hub Process API on CDSE (CDSE_CLIENT_ID/CDSE_CLIENT_SECRET,
credentialed, free tier 10,000 PU/month) rendering a true-color PNG
directly (no TIFF decode step needed — this is DISPLAY output, not a
band-math input like the tank-fill chips).

LIVE-VERIFIED 2026-07-22 (this session): STAC discovery + Process API
true-color render both worked end-to-end against the Cushing hub bbox
(a real 300x200 PNG showing the tank farm, sun-lit, cloud-free).

Cloud handling is scene-level (whole-scene eo:cloud_cover from STAC),
NOT a per-pixel SCL mask over the AOI — a real, honestly-reported number
next to the image, not a guaranteed-clean crop. This is RAW display
(no predictive claim), so an honest cloud_pct alongside the photo is the
correct bar, not the pixel-perfect masking a SIGNAL would need.

Session-run only, never on Railway (matches every other CDSE script in
this repo) — the free-tier credential lives in the session environment,
not Railway; the site expects committed chip files + a committed index,
refreshed by whichever session runs this.

Outputs:
  client/public/imagery/sites/<site_id>.jpg   (committed — small, ~30-90KB
      true-color JPEG; these ARE meant to be served, unlike the tank-fill
      TIFF chips which stay gitignored working files)
  datacore/sentinel2/site_chips_log.jsonl     (committed append-only log:
      every pull attempt, success or honest skip, for PU audit trail)
  datacore/sentinel2/site_latest_index.json  (committed — OVERWRITTEN per
      site on each successful pull: the CURRENT latest chip per site;
      server/routes.ts reads this to attach imagery metadata to
      /api/data/sites)

Run:
  python3 scripts/cdse_site_chips.py --dry-run         # list candidates + PU cost, pull nothing
  python3 scripts/cdse_site_chips.py --site cushing_hub # single site (testing)
  python3 scripts/cdse_site_chips.py                    # all sites, skips any pulled within --min-age-days
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

RES_M = 8  # output pixel size; mild oversample vs the 10m optical bands (smoother on-screen, no extra cost driver — PU scales with output px, not input resolution)
MAX_CLOUD = 35.0  # whole-scene eo:cloud_cover ceiling accepted
LOOKBACK_DAYS = 90
FREE_TIER_PU_MONTH = 10_000
SELF_BUDGET_FRACTION = 0.5  # refuse a run that would push THIS script's own month spend past this share

# half-width in km per category — ports are physically larger facilities
# than a single tank farm or mill, so their AOI is wider (datacore/sites
# categories, not a per-site override — extend here if a future category
# needs its own width).
CATEGORY_HALF_KM = {"tank_farm": 0.9, "steel_mill": 1.2, "port": 2.0}
DEFAULT_HALF_KM = 1.2

BASE = os.path.join(os.path.dirname(__file__), "..")
REGISTRY = os.path.join(BASE, "datacore", "sites", "strategic_sites.json")
CHIP_DIR = os.path.join(BASE, "client", "public", "imagery", "sites")
LOG = os.path.join(BASE, "datacore", "sentinel2", "site_chips_log.jsonl")
INDEX = os.path.join(BASE, "datacore", "sentinel2", "site_latest_index.json")

ATTRIBUTION = "Contains modified Copernicus Sentinel data"

# ── pure geometry / cost (unit-tested, no network) ──────────────────────────

def site_bbox(lat, lon, half_km):
    """WGS84 bbox [W,S,E,N] for a square half_km around (lat, lon)."""
    half_lat_deg = half_km / 110.54
    half_lon_deg = half_km / (111.32 * math.cos(math.radians(lat)))
    return [lon - half_lon_deg, lat - half_lat_deg, lon + half_lon_deg, lat + half_lat_deg]


def half_km_for(category):
    return CATEGORY_HALF_KM.get(category, DEFAULT_HALF_KM)


def chip_px(bbox, res_m=RES_M):
    """Output raster size for a WGS84 bbox at res_m meters/pixel (same
    formula as scripts/cdse_chips.py's chip_px — meters-per-degree at the
    bbox's own latitude, not re-imported to keep this script standalone)."""
    lat_mid = (bbox[1] + bbox[3]) / 2.0
    w = (bbox[2] - bbox[0]) * 111_320.0 * math.cos(math.radians(lat_mid)) / res_m
    h = (bbox[3] - bbox[1]) * 110_540.0 / res_m
    return max(32, int(round(w))), max(32, int(round(h)))


def pu_estimate(width, height, n_bands=3):
    """Sentinel Hub PU estimate: area/(512^2) x bands/3, floor 0.005."""
    return max(0.005, (width * height) / (512.0 * 512.0) * (n_bands / 3.0))


TRUE_COLOR_EVALSCRIPT = """//VERSION=3
// True-color display PNG, mild gain for a viewable (not radiometrically
// precise) photo — this is a RAW display chip, not a measurement input.
function setup() {
  return { input: ["B04", "B03", "B02"], output: { bands: 3 } };
}
function evaluatePixel(s) {
  return [2.5 * s.B04, 2.5 * s.B03, 2.5 * s.B02];
}
"""


def load_registry(path=REGISTRY):
    with open(path) as fh:
        return json.load(fh)["sites"]


def build_stac_request(bbox, since_iso, until_iso, max_cloud=MAX_CLOUD):
    return {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{since_iso}/{until_iso}",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "limit": 20,
    }


def parse_stac(payload):
    out = []
    for f in payload.get("features", []):
        p = f.get("properties", {})
        dt = p.get("datetime", "")
        out.append({"scene": f.get("id", ""), "date": dt[:10], "datetime": dt,
                     "cloud_pct": p.get("eo:cloud_cover")})
    return out


def pick_best_scene(scenes):
    """Most recent scene under the cloud ceiling — STAC already sorted
    desc by datetime and prefiltered by cloud_cover, so the first result
    IS the answer; this stays a separate step so tests can pin the
    tie-break rule (most recent, not least-cloudy) independent of the
    request's own sort."""
    return scenes[0] if scenes else None


def build_process_request(scene_date, bbox, width, height):
    return {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": f"{scene_date}T00:00:00Z", "to": f"{scene_date}T23:59:59Z"},
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {"width": width, "height": height,
                    "responses": [{"identifier": "default", "format": {"type": "image/jpeg", "quality": 85}}]},
        "evalscript": TRUE_COLOR_EVALSCRIPT,
    }


# ── index / log (unit-tested via tmp paths) ─────────────────────────────────

def read_index(index_path=INDEX):
    if not os.path.exists(index_path):
        return {}
    with open(index_path) as fh:
        return json.load(fh)


def write_index(idx, index_path=INDEX):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w") as fh:
        json.dump(idx, fh, indent=1, sort_keys=True)
        fh.write("\n")


def append_log(rec, log_path=LOG):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def month_pu_spent(log_path=LOG, now=None):
    now = now or datetime.now(timezone.utc)
    ym = now.strftime("%Y-%m")
    total = 0.0
    if os.path.exists(log_path):
        with open(log_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("ok") and str(r.get("pulled_at", "")).startswith(ym):
                        total += float(r.get("est_pu", 0))
                except Exception:
                    pass
    return total


def days_since(date_str, now=None):
    now = now or datetime.now(timezone.utc)
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return 10 ** 6
    return (now - d).days


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
        "grant_type": "client_credentials", "client_id": cid, "client_secret": sec,
    }).encode()
    tok = http_json(TOKEN_URL, body, {"content-type": "application/x-www-form-urlencoded"})
    return tok["access_token"]


def fetch_chip_bytes(scene_date, bbox, width, height, token):
    payload = build_process_request(scene_date, bbox, width, height)
    req = urllib.request.Request(
        PROCESS_URL, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "accept": "image/jpeg",
                  "authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def discover(bbox, since_iso, until_iso):
    cat = http_json(STAC_URL, build_stac_request(bbox, since_iso, until_iso),
                     {"content-type": "application/json"})
    return pick_best_scene(parse_stac(cat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None, help="single site id (default: all)")
    ap.add_argument("--min-age-days", type=int, default=7,
                     help="skip a site whose current chip is already this fresh")
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sites = load_registry()
    if args.site:
        sites = [s for s in sites if s["id"] == args.site]
        if not sites:
            sys.exit(f"no such site id: {args.site}")

    idx = read_index()
    now = datetime.now(timezone.utc)
    until_iso = now.strftime("%Y-%m-%dT23:59:59Z")
    since_iso = (now.replace(hour=0, minute=0, second=0, microsecond=0)
                 - timedelta(days=args.lookback_days)).strftime("%Y-%m-%dT00:00:00Z")

    plan = []
    for s in sites:
        cur = idx.get(s["id"])
        if cur and days_since(cur.get("date", "")) < args.min_age_days:
            print(f"skip {s['id']}: current chip is {days_since(cur['date'])}d old (< {args.min_age_days}d)")
            continue
        bbox = site_bbox(s["lat"], s["lon"], half_km_for(s["category"]))
        w, h = chip_px(bbox)
        plan.append((s, bbox, w, h))

    total_pu = sum(pu_estimate(w, h) for _, _, w, h in plan)
    spent = month_pu_spent()
    print(f"{len(plan)} site(s) planned, {total_pu:.2f} PU estimated; "
          f"month PU (this script): {spent:.1f} spent + {total_pu:.1f} planned "
          f"of {FREE_TIER_PU_MONTH} free tier")
    if spent + total_pu > FREE_TIER_PU_MONTH * SELF_BUDGET_FRACTION:
        sys.exit(f"refusing: would exceed {SELF_BUDGET_FRACTION*100:.0f}% of the monthly free tier")

    if args.dry_run:
        for s, bbox, w, h in plan:
            print(f"  {s['id']} ({s['category']}) bbox={bbox} {w}x{h}px ~{pu_estimate(w,h):.2f}PU")
        return

    if not plan:
        return

    token = get_token()
    os.makedirs(CHIP_DIR, exist_ok=True)
    for s, bbox, w, h in plan:
        rec = {"site": s["id"], "checked_at": now.isoformat()}
        try:
            best = discover(bbox, since_iso, until_iso)
            if not best:
                rec.update({"ok": False, "reason": f"no scene under {MAX_CLOUD}% cloud in {args.lookback_days}d"})
                append_log(rec)
                print(f"  ! {s['id']}: {rec['reason']}")
                continue
            img = fetch_chip_bytes(best["date"], bbox, w, h, token)
            fname = f"{s['id']}.jpg"
            with open(os.path.join(CHIP_DIR, fname), "wb") as fh:
                fh.write(img)
            est_pu = round(pu_estimate(w, h), 3)
            rec.update({
                "ok": True, "scene": best["scene"], "date": best["date"],
                "cloud_pct": best["cloud_pct"], "bbox": bbox, "width": w, "height": h,
                "bytes": len(img), "est_pu": est_pu, "pulled_at": now.isoformat(),
            })
            append_log(rec)
            idx[s["id"]] = {
                "file": f"/imagery/sites/{fname}", "scene": best["scene"], "date": best["date"],
                "cloud_pct": best["cloud_pct"], "bbox": bbox, "attribution": ATTRIBUTION,
                "pulled_at": now.isoformat(),
            }
            write_index(idx)
            print(f"  {s['id']}: {best['date']} cloud={best['cloud_pct']:.1f}% -> {fname} ({len(img)/1e3:.0f}KB, {est_pu}PU)")
        except Exception as e:
            rec.update({"ok": False, "reason": str(e)})
            append_log(rec)
            print(f"  ! {s['id']}: {e}", file=sys.stderr)
        time.sleep(1)  # polite pacing, mirrors cdse_chips.py


if __name__ == "__main__":
    main()
