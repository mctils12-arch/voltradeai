#!/usr/bin/env python3
"""GRID VISION — 3DEP LiDAR tower-extraction feasibility probe.

Tests whether USGS 3DEP Entwine (EPT) point clouds can yield transmission-tower
locations, per the grid-vision data-modality research (LiDAR was ranked #2
after HIFLD). Two questions, answered per collect:

  1. CLASSIFICATION: does the collect carry ASPRS wire/tower classes
     (13 wire-guard / 14 wire-conductor / 15 transmission tower / 16 connector)?
  2. GEOMETRY: at known tower sites (OSM power=tower ground truth), do
     non-noise returns exist in the structure band (8-60 m above ground)?

FINDINGS 2026-07-12 (see research/experiments.md): both tested collects —
NV_ClarkCo_1_B22 and AZ_BrawleyRillito_TL_2018 (a dedicated transmission-line
survey) — carry NO wire/tower classes (min-spec: ground/unassigned/noise), and
NV showed ZERO structure-band returns at three OSM-confirmed tower sites
(only 1-2 point specks at 60-106 m HAG = air noise). Tower extraction from
3DEP EPT is therefore UNPROVEN; this script exists so any future session can
survey more collects with one command instead of re-deriving the walker.

Usage:
  python3 scripts/gridvision_lidar_probe.py classify <resource_name> [n_nodes]
      Count classification codes in the N largest available nodes.
  python3 scripts/gridvision_lidar_probe.py hag <resource_name> <x3857> <y3857> [half_m]
      Height-above-ground histogram in a box around a point (e.g. an OSM tower).

Requires: laspy + lazrs (pip install laspy lazrs). Network: usgs-lidar-public
S3 bucket (public, no auth).
"""
import io
import json
import sys
import time
import urllib.request

EPT_BASE = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"
WIRE_CLASSES = {13: "wire-guard", 14: "wire-conductor", 15: "transmission-tower", 16: "wire-connector"}
NOISE_CLASSES = (7, 18)
STRUCTURE_BAND = (8.0, 60.0)  # meters HAG where tower lattice/wire returns live


def fetch(url, tries=4, timeout=120):
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def load_ept(resource):
    return json.loads(fetch(f"{EPT_BASE}/{resource}/ept.json"))


def load_hierarchy(resource):
    return json.loads(fetch(f"{EPT_BASE}/{resource}/ept-hierarchy/0-0-0-0.json"))


def read_node(resource, key):
    import laspy
    return laspy.read(io.BytesIO(fetch(f"{EPT_BASE}/{resource}/ept-data/{key}.laz")))


def covering_nodes(resource, ept, hier, x, y, max_depth=12):
    """All EPT node keys whose cell contains (x, y) in EPSG:3857, expanding
    sub-hierarchy pages (value -1) as needed. Points for one location are
    spread across EVERY tree level, so all ancestors matter."""
    b = ept["bounds"]
    hier = dict(hier)
    nodes = []
    for d in range(0, max_depth):
        n = 2 ** d
        s = (b[3] - b[0]) / n
        ix, iy = int((x - b[0]) / s), int((y - b[1]) / s)
        prefix = f"{d}-{ix}-{iy}-"
        keys = [k for k in list(hier) if k.startswith(prefix)]
        if not keys:
            break
        for k in keys:
            if hier[k] == -1:
                sub = json.loads(fetch(f"{EPT_BASE}/{resource}/ept-hierarchy/{k}.json"))
                hier.update(sub)
                if sub.get(k, 0) > 0:
                    nodes.append(k)
            elif hier[k] > 0:
                nodes.append(k)
    return sorted(set(nodes))


def cmd_classify(resource, n_nodes=6):
    import collections
    hier = load_hierarchy(resource)
    real = sorted(((k, v) for k, v in hier.items() if v > 0), key=lambda kv: -kv[1])[:n_nodes]
    total = collections.Counter()
    for key, _ in real:
        las = read_node(resource, key)
        counts = collections.Counter(int(c) for c in las.classification)
        total.update(counts)
        print(f"  {key}: {sum(counts.values())} pts -> {dict(sorted(counts.items()))}")
    wire = {k: v for k, v in total.items() if k in WIRE_CLASSES}
    print(f"\n{resource}: wire/tower classes present: {wire or 'NONE (min-spec classification)'}")
    return wire


def cmd_hag(resource, x, y, half=120.0):
    import numpy as np
    ept = load_ept(resource)
    hier = load_hierarchy(resource)
    nodes = covering_nodes(resource, ept, hier, x, y)
    print(f"covering nodes: {nodes}")
    xs, ys, zs, cs = [], [], [], []
    for key in nodes:
        las = read_node(resource, key)
        xa, ya = np.asarray(las.x), np.asarray(las.y)
        m = (xa > x - half) & (xa < x + half) & (ya > y - half) & (ya < y + half)
        xs.append(xa[m]); ys.append(ya[m])
        zs.append(np.asarray(las.z)[m]); cs.append(np.asarray(las.classification)[m])
    px = np.concatenate(xs); py = np.concatenate(ys)
    pz = np.concatenate(zs); pc = np.concatenate(cs)
    ok = ~np.isin(pc, NOISE_CLASSES)
    px, py, pz = px[ok], py[ok], pz[ok]
    print(f"non-noise pts in ±{half:.0f}m box: {len(px)}")
    if not len(px):
        print("NO COVERAGE at this point (footprint miss — check resources.geojson polygon)")
        return None
    cell = 10.0
    x0, y0 = px.min(), py.min()
    gi = ((px - x0) // cell).astype(int)
    gj = ((py - y0) // cell).astype(int)
    height = int(gj.max()) + 2
    gid = gi * height + gj
    order = np.argsort(gid)
    uniq, starts = np.unique(gid[order], return_index=True)
    gmin = dict(zip(uniq.tolist(), np.minimum.reduceat(pz[order], starts).tolist()))
    hag = np.array([z - gmin[g] for z, g in zip(pz, gid)])
    print("HAG histogram (non-noise):")
    for lo, hi in [(2, 5), (5, 8), (8, 12), (12, 20), (20, 35), (35, 60), (60, 999)]:
        print(f"  {lo:3d}-{hi:3d} m: {int(((hag >= lo) & (hag < hi)).sum())}")
    band = int(((hag >= STRUCTURE_BAND[0]) & (hag < STRUCTURE_BAND[1])).sum())
    print(f"structure band {STRUCTURE_BAND[0]:.0f}-{STRUCTURE_BAND[1]:.0f} m: {band} pts "
          f"({'possible structure' if band >= 30 else 'no tower signature'})")
    return band


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    cmd, resource = argv[1], argv[2]
    if cmd == "classify":
        cmd_classify(resource, int(argv[3]) if len(argv) > 3 else 6)
    elif cmd == "hag":
        cmd_hag(resource, float(argv[3]), float(argv[4]),
                float(argv[5]) if len(argv) > 5 else 120.0)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
