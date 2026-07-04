#!/usr/bin/env python3
"""
sentinel2_tankfill.py — Sentinel-2 tank-shadow index for the Cushing OK tank
farms (gate-1 kickoff, datacore/SENTINEL2_CHANGE_SPEC.md; directive
2026-07-04). EXPERIMENTAL readout — NOT a SIGNAL (ladder gate 2 unpassed).

Zero credentials required (probed live 2026-07-04):
  - scenes:  Element84 earth-search STAC (anonymous) over AWS Open Data
             sentinel-cogs public S3 — windowed COG range reads, ~KB/site
  - truth:   EIA weekly Cushing crude stocks XLS (keyless public history)

v1 index (facility-scale honesty — never per-tank, never object counts):
  raw_dark_frac = share of AOI pixels with B04 reflectance < 0.35 x AOI median
                  (floating-roof shadows read dark in red; threshold is
                  scene-relative so atmospheric brightness cancels)
  shadow_index  = raw_dark_frac / tan(solar_zenith)
                  (shadow AREA scales ~ tan(zenith); normalizing makes weeks
                  with different sun angles comparable — without this the
                  index is mostly a sun-angle plot)

Readings append to datacore/sentinel2/readings.jsonl (git-versioned archive;
dedup by site+scene). Run:
  python3 scripts/sentinel2_tankfill.py --since 2026-03-15 [--compare]

Session-local deps: rasterio, xlrd (like pillow for site_verify.py — not
runtime dependencies; Railway never runs this).
"""
import argparse, json, math, os, sys, urllib.request, tempfile
from datetime import datetime, timezone

STAC = "https://earth-search.aws.element84.com/v1/search"
EIA_XLS = "https://www.eia.gov/dnav/pet/hist_xls/W_EPC0_SAX_YCUOK_MBBLw.xls"
SITES_JSON = os.path.join(os.path.dirname(__file__), "..", "datacore", "sites", "strategic_sites.json")
OUT_JSONL = os.path.join(os.path.dirname(__file__), "..", "datacore", "sentinel2", "readings.jsonl")
HALF_DEG = 0.012          # ~1.3 km half-window around each verified coordinate
DARK_REL = 0.35           # dark threshold as a fraction of AOI median
MAX_CLOUD = 30.0


def stac_scenes(since: str, until: str):
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [-96.80, 35.90, -96.72, 35.98],
        "datetime": f"{since}T00:00:00Z/{until}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": MAX_CLOUD}},
        "sortby": [{"field": "properties.datetime", "direction": "asc"}],
        "limit": 200,
    }
    req = urllib.request.Request(STAC, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        feats = json.load(r)["features"]
    # one scene per date: lowest cloud (both UTM tiles 14SPE/14SQE can cover Cushing)
    by_date = {}
    for f in feats:
        d = f["properties"]["datetime"][:10]
        if d not in by_date or f["properties"]["eo:cloud_cover"] < by_date[d]["properties"]["eo:cloud_cover"]:
            by_date[d] = f
    return [by_date[d] for d in sorted(by_date)]


def tank_sites():
    with open(SITES_JSON) as fh:
        d = json.load(fh)
    return [s for s in d["sites"] if s["category"] == "tank_farm"]


def shadow_reading(scene, site):
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds
    href = scene["assets"]["red"]["href"]
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs,
                             site["lon"] - HALF_DEG, site["lat"] - HALF_DEG,
                             site["lon"] + HALF_DEG, site["lat"] + HALF_DEG)
        arr = src.read(1, window=from_bounds(*b, src.transform))
    valid = arr[arr > 0]
    if valid.size < 5000:
        return None
    med = float(np.median(valid))
    raw = float((valid < DARK_REL * med).mean())
    sun_elev = float(scene["properties"].get("view:sun_elevation") or 0)
    if sun_elev <= 5:
        return None
    zenith = math.radians(90.0 - sun_elev)
    return {
        "site": site["id"],
        "scene": scene["id"],
        "date": scene["properties"]["datetime"][:10],
        "cloud_pct": round(scene["properties"]["eo:cloud_cover"], 2),
        "sun_elev": round(sun_elev, 2),
        "px": int(valid.size),
        "median_reflectance": int(med),
        "raw_dark_frac": round(raw, 5),
        "shadow_index": round(raw / math.tan(zenith), 5),
        "label": "EXPERIMENTAL — facility-scale tank-shadow index, gate 2 unpassed",
    }


def load_existing():
    seen = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    seen.add((r["site"], r["scene"]))
                except Exception:
                    pass
    return seen


def eia_weekly():
    import xlrd
    with tempfile.NamedTemporaryFile(suffix=".xls", delete=False) as tf:
        with urllib.request.urlopen(EIA_XLS, timeout=60) as r:
            tf.write(r.read())
        p = tf.name
    wb = xlrd.open_workbook(p)
    sh = wb.sheet_by_index(1)
    out = {}
    for i in range(sh.nrows):
        try:
            d = xlrd.xldate_as_datetime(sh.cell_value(i, 0), wb.datemode).date()
            v = float(sh.cell_value(i, 1))
            out[d.isoformat()] = v
        except Exception:
            continue
    os.unlink(p)
    return out  # {friday_date: kbbl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-03-15")
    ap.add_argument("--until", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--compare", action="store_true", help="correlate weekly index vs EIA Cushing stocks")
    args = ap.parse_args()

    scenes = stac_scenes(args.since, args.until)
    sites = tank_sites()
    print(f"{len(scenes)} usable scenes (cloud<{MAX_CLOUD}%), {len(sites)} verified tank-farm AOIs")
    seen = load_existing()
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    new = []
    for sc in scenes:
        for site in sites:
            if (site["id"], sc["id"]) in seen:
                continue
            try:
                r = shadow_reading(sc, site)
            except Exception as e:
                print(f"  ! {sc['id']} {site['id']}: {e}", file=sys.stderr)
                continue
            if r:
                new.append(r)
                print(f"  {r['date']} {r['site']:18s} dark={r['raw_dark_frac']:.4f} idx={r['shadow_index']:.4f} cloud={r['cloud_pct']}%")
    if new:
        with open(OUT_JSONL, "a") as fh:
            for r in new:
                fh.write(json.dumps(r) + "\n")
    print(f"appended {len(new)} readings -> {os.path.relpath(OUT_JSONL)}")

    if args.compare:
        # weekly mean index (all sites) matched to the EIA Friday within +/-3 days
        from datetime import date, timedelta
        rows = []
        with open(OUT_JSONL) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        by_date = {}
        for r in rows:
            by_date.setdefault(r["date"], []).append(r["shadow_index"])
        weekly = {d: sum(v) / len(v) for d, v in by_date.items()}
        eia = eia_weekly()
        pairs = []
        for d, idx in sorted(weekly.items()):
            dd = date.fromisoformat(d)
            best = None
            for k in eia:
                kd = date.fromisoformat(k)
                if abs((kd - dd).days) <= 3:
                    best = (k, eia[k])
            if best:
                pairs.append((d, idx, best[0], best[1]))
        print(f"\n{len(pairs)} scene-weeks matched to EIA weeks:")
        for d, idx, ed, v in pairs:
            print(f"  {d} idx={idx:.4f}  |  EIA {ed} {int(v)} kbbl")
        if len(pairs) >= 6:
            import numpy as np
            a = np.array([p[1] for p in pairs]); b = np.array([p[3] for p in pairs])
            r_lvl = float(np.corrcoef(a, b)[0, 1])
            da, db = np.diff(a), np.diff(b)
            r_dlt = float(np.corrcoef(da, db)[0, 1]) if len(da) > 2 else float("nan")
            print(f"\nPearson r (levels): {r_lvl:+.3f}   Pearson r (week-over-week deltas): {r_dlt:+.3f}   n={len(pairs)}")
        else:
            print("(<6 matched weeks — correlation withheld, sample too thin)")


if __name__ == "__main__":
    main()
