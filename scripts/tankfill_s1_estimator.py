#!/usr/bin/env python3
"""
tankfill_s1_estimator.py — tank-fill v3 PR-2: per-tank double-bounce
estimator over the S1 chips + the gate-1 attempt. EXPERIMENTAL — design and
criteria were pre-stated in the workup (open_questions.md TANK-FILL v3
STATUS, 2026-07-05) BEFORE this ran; discounted prior P(pass) ~25%.

Physics: floating roof + shell wall form a corner reflector toward the
radar; the double-bounce VV return scales with EXPOSED WALL HEIGHT (roof
depth). So per tank:
    db      = log10(p95 VV within the tank disk + 1 px halo)
    fill*   = -(db - median_t(db))       # per-tank self-ratio, log-domain
Higher return = more exposed wall = emptier. The per-tank median across the
scene series removes tank-specific geometry/calibration. NOTE (stated
before scoring): the median uses the full series, but a constant per-tank
offset CANCELS EXACTLY in the delta test — the binding gate-1 criteria are
immune to it; only the supporting levels correlation sees the whole-series
normalization.

Scoring uses ASCENDING/relative-orbit-34 scenes only (v1 contract — one
viewing geometry, no incidence normalization); other geometries are
archived but excluded. Output readings_s1.jsonl is a WHOLE-FILE REBUILD
each run (normalization depends on the full series; deterministic from the
chips), NOT append-only.

Gate 1 (comparator reused from tankfill_gate1.py, unchanged): >=20 matched
scene-weeks spanning >=1 EIA reversal; PASS = delta r >= +0.3 AND
delta-sign >= 65%. Matched weeks < 20 -> INSUFFICIENT-SAMPLE, not FAIL.

Run (session-side; deps numpy + tifffile + xlrd):
  python3 scripts/tankfill_s1_estimator.py            # estimate + gate-1
  python3 scripts/tankfill_s1_estimator.py --no-gate  # readings only
"""
import argparse
import importlib.util
import json
import math
import os

BASE = os.path.join(os.path.dirname(__file__), "..")
CHIP_DIR = os.path.join(BASE, "datacore", "sentinel2", "chips_s1")
CHIP_INDEX = os.path.join(BASE, "datacore", "sentinel2", "s1_chips_index.jsonl")
OUT_JSONL = os.path.join(BASE, "datacore", "sentinel2", "readings_s1.jsonl")

RES_M = 10.0
HALO_PX = 1.0            # disk + halo: the double-bounce line hugs the wall
P95 = 95.0
SCORE_ORBIT = ("ascending", 34)  # v1 scoring contract (workup)
FLOOR = 1e-4             # sigma0 floor before log (nodata/shadow pixels)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── pure pieces (unit-tested) ───────────────────────────────────────────────

def tank_db(chip_vv, cx, cy, radius_px, halo=HALO_PX, p=P95):
    """log10(p95 VV) within the tank disk + halo, or None if out of frame."""
    import numpy as np
    h, w = chip_vv.shape
    r = radius_px + halo
    x0, x1 = int(math.floor(cx - r)), int(math.ceil(cx + r)) + 1
    y0, y1 = int(math.floor(cy - r)), int(math.ceil(cy + r)) + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = ((xx + 0.5) - cx) ** 2 + ((yy + 0.5) - cy) ** 2 < r * r
    vals = chip_vv[y0:y1, x0:x1][m]
    if vals.size < 4:
        return None
    return float(np.log10(max(float(np.percentile(vals, p)), FLOOR)))


def self_ratio_series(db_by_scene):
    """{scene: {tank_id: db}} -> per-scene fill*-direction values:
    fill* = -(db - per-tank median across scenes). Tanks seen in fewer than
    3 scenes are dropped (a median of 1-2 points normalizes nothing)."""
    per_tank = {}
    for vals in db_by_scene.values():
        for tid, v in vals.items():
            per_tank.setdefault(tid, []).append(v)
    medians = {}
    for tid, xs in per_tank.items():
        if len(xs) >= 3:
            s = sorted(xs)
            n = len(s)
            medians[tid] = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    out = {}
    for scene, vals in db_by_scene.items():
        out[scene] = {tid: -(v - medians[tid]) for tid, v in vals.items() if tid in medians}
    return out


def scene_reading(meta, fills, tanks_by_id):
    """One readings_s1 line: per-tank fill* + D^2-weighted composites."""
    rows = []
    for tid, f in sorted(fills.items()):
        t = tanks_by_id[tid]
        rows.append({"id": tid, "site": t["site"], "d_m": t["diameter_m"], "fill": round(f, 5)})
    sites = {}
    for r in rows:
        sites.setdefault(r["site"], []).append(r)
    site_aggs = []
    for site, rs in sorted(sites.items()):
        wsum = sum(x["d_m"] ** 2 for x in rs)
        site_aggs.append({"site": site, "n_tanks": len(rs),
                          "fill_d2w": round(sum(x["fill"] * x["d_m"] ** 2 for x in rs) / wsum, 5)})
    return {
        "scene": meta["scene"], "date": meta["date"],
        "orbit_state": meta.get("orbit_state"), "relative_orbit": meta.get("relative_orbit"),
        "n_measured": len(rows), "sites": site_aggs, "tanks": rows,
        "label": "EXPERIMENTAL — S1 double-bounce self-ratio (fill-direction, unitless); gate 1 per workup",
    }


# ── pipeline ────────────────────────────────────────────────────────────────

def main():
    import numpy as np
    import tifffile
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    est2 = _load("tankfill_estimator", "tankfill_estimator.py")
    g1 = _load("tankfill_gate1", "tankfill_gate1.py")
    tanks = est2.load_registry()
    tanks_by_id = {t["id"]: t for t in tanks}

    metas = []
    with open(CHIP_INDEX) as fh:
        for line in fh:
            try:
                metas.append(json.loads(line))
            except Exception:
                pass
    metas.sort(key=lambda m: m["date"])
    scoring = [m for m in metas if (m.get("orbit_state"), m.get("relative_orbit")) == SCORE_ORBIT]
    skipped_orbit = len(metas) - len(scoring)
    if skipped_orbit:
        print(f"{skipped_orbit} scenes on other geometries archived but EXCLUDED from scoring (v1 contract)")

    db_by_scene = {}
    meta_by_scene = {}
    for meta in scoring:
        fp = os.path.join(CHIP_DIR, meta["file"])
        if not os.path.exists(fp):
            print(f"  ! chip missing (re-pull via cdse_s1_chips.py): {meta['file']}")
            continue
        chip = tifffile.imread(fp)
        vv = chip[..., 0]
        w, h = vv.shape[1], vv.shape[0]
        bbox = meta["bbox"]
        vals = {}
        for t in tanks:
            x, y = est2.geo_to_px(t["lon"], t["lat"], bbox, w, h)
            db = tank_db(vv, x, y, t["diameter_m"] / 2.0 / RES_M)
            if db is not None:
                vals[t["id"]] = db
        db_by_scene[meta["scene"]] = vals
        meta_by_scene[meta["scene"]] = meta

    fills_by_scene = self_ratio_series(db_by_scene)
    readings = [scene_reading(meta_by_scene[s], f, tanks_by_id)
                for s, f in sorted(fills_by_scene.items(), key=lambda kv: meta_by_scene[kv[0]]["date"])]
    with open(OUT_JSONL, "w") as fh:  # whole-file rebuild (see module docstring)
        for r in readings:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(readings)} readings -> {os.path.relpath(OUT_JSONL)} "
          f"({len(tanks)} registry tanks; per-scene n_measured median "
          f"{sorted(r['n_measured'] for r in readings)[len(readings)//2] if readings else 0})")

    if args.no_gate:
        return
    eia = g1.eia_weekly()
    matched = g1.match_weeks(readings, eia)
    print(f"\n{len(matched)} matched scene-weeks")
    v = g1.verdict(matched)
    if v["n_matched"] < 20:
        v["VERDICT"] = "INSUFFICIENT-SAMPLE (pre-stated: <20 matched weeks is not a FAIL — wait for cadence)"
    else:
        v["VERDICT"] = "PASS" if v["PASS"] else "FAIL"
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
