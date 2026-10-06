"""Measure where each FAA chart's MAP ends and its border (legend, title
block, collar) begins — from the FAA's own clipped tile-service mosaic.

Every FAA GeoTIFF ships with its collar, and the FAA publishes no clipping
outlines. But the FAA's tile service IS a clipped, seamless mosaic of the
same charts. So: render each chart into map tiles, compare with the mosaic,
and the region where the mosaic shows THIS chart is the chart's map area as
the FAA itself cut it. Nothing is typed in by hand.

1. COARSE OWNERSHIP at `measure_zoom` (16x16-px blocks): which chart does
   the mosaic show in each block? Structural comparison (text, linework)
   near native scale — robust to the one-edition difference between the
   GeoTIFF and the service; a collar never matches at all.
2. Each chart's blocks -> cleaned mask -> polygon -> simplified.
3. EDGE REFINEMENT at `refine_zoom`: every long polygon edge is re-measured
   at ~20-40 points with a 1-D profile across it (where does the mosaic stop
   matching this chart?) and a line is fitted through them. Corners become
   the intersections of neighbouring fitted edges.

Edges belong to a chart, not an edition (the Seattle sectional covers the
same area every cycle), so measurement runs once per chart shape. Each
result stores the GeoTIFF's georeferencing fingerprint; a new edition whose
fingerprint drifts is re-measured (and the bake refuses to use a stale edge
silently — see bake.py).
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import (TILE, ChartSource, block_mean, fetch_faa_tile, m_to_lonlat, m_to_tile, tile_bounds_m,
                    tiles_for_bounds_m, ORIGIN)

BLOCK = 16
# STRUCTURAL difference: per-pixel |RGB| (sum over channels, 0..765) after a
# 1-px Gaussian blur of both images, averaged over a block/window. Colour
# means alone cannot tell two overlapping charts apart (both draw the same
# terrain tints); text and linework can. Measured 2026-10-06 at the
# Seattle/Klamath Falls seam (44.5N, 121W), 2026-09-03 GeoTIFFs vs the
# service's 2026-07-09 mosaic:
#   zoom 11 (near native): the chart the FAA shows ~2-7, the overlapping
#     neighbour ~45-150, a collar ~120-165  -> REFINE_THRESHOLD
#   zoom 9 (coarse): shown ~25-60, neighbour ~30-60 (not separable), collar
#     ~180-230 -> COARSE_THRESHOLD only rejects collars; the zoom-11 edge
#     refinement is what places the seam.
COARSE_THRESHOLD = 100.0
REFINE_THRESHOLD = 30.0
MIN_PIECE_BLOCKS = 40
# Majority smoothing of ownership + a looser simplification were tried
# 2026-10-06 (radius 3, DP 4 blocks): straighter outlines, more edges refined,
# but the full-sectional gate got WORSE on 1500 sampled tiles — coverage
# 0.99670 vs 0.99814, collar spill 0.00306 vs 0.00038 — because real notches
# and insets were smoothed away. Kept switchable, off by default.
SMOOTH_RADIUS = 0  # majority window radius in blocks (0 = off)
DP_TOLERANCE_BLOCKS = 2.5  # coarse outline simplification before refinement


# ── pure geometry ───────────────────────────────────────────────────────────

Pt = Tuple[float, float]


def dp_simplify(pts: Sequence[Pt], tol: float) -> List[Pt]:
    """Douglas-Peucker on an open polyline."""
    if len(pts) < 3:
        return list(pts)
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    best, bi = -1.0, -1
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        d = abs(dy * (px - x0) - dx * (py - y0)) / L if L > 0 else math.hypot(px - x0, py - y0)
        if d > best:
            best, bi = d, i
    if best <= tol:
        return [pts[0], pts[-1]]
    return dp_simplify(pts[: bi + 1], tol)[:-1] + dp_simplify(pts[bi:], tol)


def simplify_ring(ring: Sequence[Pt], tol: float) -> List[Pt]:
    """Closed ring (first != last) -> simplified closed ring, split at the two
    mutually farthest-ish vertices so DP has stable anchors."""
    r = list(ring)
    if r and r[0] == r[-1]:
        r = r[:-1]
    if len(r) < 4:
        return r
    a = 0
    b = max(range(len(r)), key=lambda i: (r[i][0] - r[a][0]) ** 2 + (r[i][1] - r[a][1]) ** 2)
    first = dp_simplify(r[a: b + 1], tol)
    second = dp_simplify(r[b:] + [r[a]], tol)
    out = first[:-1] + second[:-1]
    return out


def fit_line(points: Sequence[Pt]) -> Tuple[Pt, Pt]:
    """Total-least-squares line: (centroid, unit direction)."""
    P = np.asarray(points, float)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    d = vt[0]
    return (float(c[0]), float(c[1])), (float(d[0]), float(d[1]))


def intersect(l1: Tuple[Pt, Pt], l2: Tuple[Pt, Pt]) -> Optional[Pt]:
    (p, d), (q, e) = l1, l2
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-6:  # near-parallel: no stable corner
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return (p[0] + t * d[0], p[1] + t * d[1])


def step_location(match: Sequence[bool]) -> int:
    """Index t (0..n) best splitting a profile into matching-before /
    not-matching-after: maximises #match[:t] + #nonmatch[t:]."""
    m = np.asarray(match, bool).astype(int)
    n = len(m)
    before = np.concatenate([[0], np.cumsum(m)])
    after_non = np.concatenate([np.cumsum((1 - m)[::-1])[::-1], [0]])
    score = before + after_non
    return int(np.argmax(score))


# ── measurement ─────────────────────────────────────────────────────────────

class TileCache:
    """Small LRU of rendered chart tiles + FAA mosaic tiles."""

    def __init__(self, faa_service: str, faa_cache_dir: str, cap: int = 256):
        self.service, self.dir, self.cap = faa_service, faa_cache_dir, cap
        self.faa: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self.chart: "OrderedDict[tuple, tuple]" = OrderedDict()

    def _lru(self, d, key, make):
        v = d.get(key)
        if v is None:
            v = make()
            d[key] = v
            if len(d) > self.cap:
                d.popitem(last=False)
        else:
            d.move_to_end(key)
        return v

    def prefetch(self, keys, workers: int = 16) -> None:
        """Populate the FAA tile disk cache concurrently (network-bound)."""
        from concurrent.futures import ThreadPoolExecutor

        keys = list(dict.fromkeys(keys))
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(lambda k: fetch_faa_tile(self.service, *k, self.dir), keys))

    def faa_tile(self, z, x, y):
        return self._lru(self.faa, (z, x, y), lambda: fetch_faa_tile(self.service, z, x, y, self.dir))

    def chart_tile(self, src: ChartSource, z, x, y):
        return self._lru(self.chart, (src.name, z, x, y), lambda: src.render(z, x, y))

    def diff_tile(self, src: ChartSource, z, x, y):
        """(structural diff (256,256) float32, comparable (256,256) bool)."""
        def make():
            rgb, valid = src.render(z, x, y)
            faa = self.faa_tile(z, x, y)
            return structural_diff(rgb, faa[..., :3].astype(np.float32)), valid & (faa[..., 3] == 255)
        return self._lru(self.chart, ("diff", src.name, z, x, y), make)


def structural_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel sum |a-b| over RGB after a 1-px Gaussian blur of each."""
    from scipy import ndimage as nd

    return np.abs(nd.gaussian_filter(a, (1, 1, 0)) - nd.gaussian_filter(b, (1, 1, 0))).sum(-1)


def _intersects(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


_CW: Dict[str, object] = {}


def _coarse_init(specs, service, cache_dir, z):
    _CW["charts"] = [ChartSource.open(n, p) for n, p in specs]
    _CW["service"], _CW["dir"], _CW["z"] = service, cache_dir, z


def _coarse_tile(xy):
    """Owner per block of one tile (worker process)."""
    x, y = xy
    z = _CW["z"]
    charts = _CW["charts"]
    tb = tile_bounds_m(z, x, y)
    faa = fetch_faa_tile(_CW["service"], z, x, y, _CW["dir"])  # disk cache, prefetched
    fa = block_mean(faa[..., 3].astype(np.float32), BLOCK)
    if fa.max() <= 0:
        return xy, None
    fr = faa[..., :3].astype(np.float32)
    nb = TILE // BLOCK
    best = np.full((nb, nb), np.inf, np.float32)
    owner = np.full((nb, nb), -1, np.int32)
    for ci, c in enumerate(charts):
        if not _intersects(c.bounds_m, tb):
            continue
        rgb, valid = c.render(z, x, y)
        d = block_mean(structural_diff(rgb, fr), BLOCK)
        ok = (block_mean(valid.astype(np.float32), BLOCK) >= 0.999) & (fa >= 254.5) & (d < COARSE_THRESHOLD) & (d < best)
        best[ok] = d[ok]
        owner[ok] = ci
    return xy, (owner if (owner >= 0).any() else None)


def coarse_matches(charts: Sequence[ChartSource], z: int, cache: TileCache, log=print,
                   workers: int = 0) -> Dict[tuple, Dict[int, np.ndarray]]:
    """{(x, y): {chart index: bool per BLOCK-px block}} — which chart the FAA
    mosaic shows in each block (the best structural match, if it is a match
    at all), over every z tile any chart touches. Neighbouring charts overlap
    and draw the same ground, so this needs a zoom near the charts' native
    scale (z10 for sectionals) where text and linework tell them apart.
    Network first (prefetch, threads), then CPU (one process per core)."""
    import os
    from multiprocessing import Pool

    tiles = set()
    for c in charts:
        tiles.update(tiles_for_bounds_m(c.bounds_m, z))
    tiles = sorted(tiles, key=lambda t: (t[1], t[0]))
    cache.prefetch([(z, x, y) for x, y in tiles])
    out: Dict[tuple, Dict[int, np.ndarray]] = {}
    specs = [(c.name, c.path) for c in charts]
    with Pool(workers or os.cpu_count() or 2, initializer=_coarse_init, initargs=(specs, cache.service, cache.dir, z)) as pool:
        for i, ((x, y), owner) in enumerate(pool.imap_unordered(_coarse_tile, tiles, chunksize=16)):
            if owner is not None:
                out[(x, y)] = {int(ci): owner == ci for ci in np.unique(owner) if ci >= 0}
            if log and i % 2000 == 0:
                log(f"  coarse z{z}: {i + 1}/{len(tiles)} tiles")
    return out


def chart_masks(own: Dict[tuple, Dict[int, np.ndarray]], n_charts: int, z: int):
    """Per chart: (bool block mask, (x0, y0) tile origin) cleaned up —
    closed, holes filled, opened, small pieces dropped."""
    from scipy import ndimage as nd

    nb = TILE // BLOCK
    res = {}
    for ci in range(n_charts):
        keys = [k for k, v in own.items() if ci in v]
        if not keys:
            continue
        x0 = min(k[0] for k in keys); x1 = max(k[0] for k in keys)
        y0 = min(k[1] for k in keys); y1 = max(k[1] for k in keys)
        # one tile of margin so morphology never touches the array edge
        x0 -= 1; y0 -= 1; x1 += 1; y1 += 1
        m = np.zeros(((y1 - y0 + 1) * nb, (x1 - x0 + 1) * nb), bool)
        anyown = np.zeros_like(m)
        for (x, y), v in own.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                sl = (slice((y - y0) * nb, (y - y0 + 1) * nb), slice((x - x0) * nb, (x - x0 + 1) * nb))
                for oc, om in v.items():
                    anyown[sl] |= om
                if ci in v:
                    m[sl] = v[ci]
        # optional MAJORITY SMOOTHING (see SMOOTH_RADIUS): a block belongs to
        # this chart if it owns most of the owned blocks around it
        if SMOOTH_RADIUS > 0:
            k = 2 * SMOOTH_RADIUS + 1
            mine = nd.uniform_filter(m.astype(np.float32), k)
            owned = nd.uniform_filter(anyown.astype(np.float32), k)
            m = (mine > 0.5 * owned) & (owned > 0)
        m = nd.binary_closing(m, iterations=3)
        m = nd.binary_fill_holes(m)
        m = nd.binary_opening(m, iterations=2)
        lab, n = nd.label(m)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        keep = [i for i in range(1, n + 1) if sizes[i] >= MIN_PIECE_BLOCKS]
        if not keep:
            continue
        res[ci] = (np.isin(lab, keep), (x0, y0))
    return res


def mask_rings_m(mask: np.ndarray, origin: Tuple[int, int], z: int) -> List[List[Pt]]:
    """Exterior rings of a block mask, in 3857 metres."""
    from rasterio import features
    from rasterio.transform import Affine

    x0, y0 = origin
    tb = tile_bounds_m(z, x0, y0)
    bsz = (tb[2] - tb[0]) / (TILE // BLOCK)
    tr = Affine(bsz, 0, tb[0], 0, -bsz, tb[3])
    rings = []
    for geom, val in features.shapes(mask.astype(np.uint8), mask=mask, transform=tr):
        if val != 1:
            continue
        rings.append([tuple(p) for p in geom["coordinates"][0]])
    return rings


@dataclass
class EdgeResult:
    """A chart's map area as rings in the chart's own PIXEL space (col, row).
    Chart edges are straight lines on the paper, i.e. straight in the
    chart's own projection — NOT on the Web-Mercator map (a straight Lambert
    edge bows by ~0.04 deg of latitude across a sectional, ~4 km). So lines
    are fitted, and corners intersected, in pixel space."""
    chart: str
    rings_px: List[List[Pt]]
    refined_edges: int = 0
    total_edges: int = 0
    residual_px: List[float] = field(default_factory=list)


def px_to_m(src: ChartSource, pts: Sequence[Pt]) -> List[Pt]:
    from rasterio.warp import transform

    if not pts:
        return []
    xs, ys = zip(*[src.ds.transform * (c, r) for c, r in pts])
    mx, my = transform(src.ds.crs, "EPSG:3857", list(xs), list(ys))
    return list(zip(mx, my))


def m_to_px(src: ChartSource, pts: Sequence[Pt]) -> List[Pt]:
    from rasterio.warp import transform

    if not pts:
        return []
    xs, ys = zip(*pts)
    sx, sy = transform("EPSG:3857", src.ds.crs, list(xs), list(ys))
    inv = ~src.ds.transform
    return [inv * (x, y) for x, y in zip(sx, sy)]


def point_in_ring(p: Pt, ring: Sequence[Pt]) -> bool:
    x, y = p
    inside = False
    n = len(ring)
    for i in range(n):
        (x1, y1), (x2, y2) = ring[i], ring[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def densify(ring: Sequence[Pt], step: float) -> List[Pt]:
    """Insert points every `step` along each edge (closed ring, no repeat)."""
    out: List[Pt] = []
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        k = max(1, int(math.ceil(L / step)))
        out += [(a[0] + (b[0] - a[0]) * j / k, a[1] + (b[1] - a[1]) * j / k) for j in range(k)]
    return out


def refine_ring(src: ChartSource, ring: List[Pt], zr: int, cache: TileCache, block_m: float,
                block_px: float) -> Tuple[List[Pt], int, List[float]]:
    """Re-measure each long edge (pixel-space ring) at zoom `zr`."""
    n = len(ring)
    if n < 3:
        return ring, 0, []
    px_m = 2 * ORIGIN / (2 ** zr) / TILE
    half = int(6 * block_m / px_m)  # profile half-length: +/- 6 coarse blocks
    lines: List[Optional[Tuple[Pt, Pt]]] = []
    resid: List[float] = []
    refined = 0
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 6 * block_px:
            lines.append(None)
            continue
        d = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        nrm = (d[1], -d[0])
        if point_in_ring((mid[0] + nrm[0] * block_px * 0.5, mid[1] + nrm[1] * block_px * 0.5), ring):
            nrm = (-nrm[0], -nrm[1])  # make it point OUT of the map area
        k = int(min(40, max(8, L / (2 * block_px))))
        samples = []
        for j in range(k):
            t = (j + 0.5) / k
            # stay clear of corners (another edge's transition lives there)
            if t * L < 3 * block_px or (1 - t) * L < 3 * block_px:
                continue
            c = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            samples.append(c)
        if len(samples) < 5:
            lines.append(None)
            continue
        cm = px_to_m(src, samples)
        cn = px_to_m(src, [(c[0] + nrm[0] * 10, c[1] + nrm[1] * 10) for c in samples])
        nrms_m = []
        for (x0, y0), (x1, y1) in zip(cm, cn):
            ll = math.hypot(x1 - x0, y1 - y0) or 1.0
            nrms_m.append(((x1 - x0) / ll, (y1 - y0) / ll))
        cache.prefetch(_profile_tiles(cm, nrms_m, half, px_m, zr))
        hits_m = []
        for c, nm in zip(cm, nrms_m):
            off = _profile_step(src, c, nm, half, px_m, zr, cache)
            if off is not None:
                hits_m.append((c[0] + off * nm[0], c[1] + off * nm[1]))
        pts = m_to_px(src, hits_m)
        if len(pts) < 5:
            lines.append(None)
            continue
        line = fit_line(pts)
        for _ in range(2):  # drop outliers (edition changes, labels) and refit
            r = [abs((p[0] - line[0][0]) * line[1][1] - (p[1] - line[0][1]) * line[1][0]) for p in pts]
            keep = [p for p, rr in zip(pts, r) if rr <= max(2.0, 2.5 * float(np.median(r)))]
            if len(keep) < 5:
                break
            pts = keep
            line = fit_line(pts)
        r = [abs((p[0] - line[0][0]) * line[1][1] - (p[1] - line[0][1]) * line[1][0]) for p in pts]
        resid.append(float(np.median(r)))
        lines.append(line)
        refined += 1
    out: List[Pt] = []
    for i in range(n):
        prev_l, cur_l = lines[i - 1], lines[i]
        v = ring[i]
        if prev_l and cur_l:
            p = intersect(prev_l, cur_l)
            # a corner may only move a little; a far intersection means the
            # two edges are nearly collinear
            if p and math.hypot(p[0] - v[0], p[1] - v[1]) < 4 * block_px:
                v = p
        elif cur_l or prev_l:
            l = cur_l or prev_l
            (cx, cy), (dx, dy) = l
            t = (v[0] - cx) * dx + (v[1] - cy) * dy
            q = (cx + t * dx, cy + t * dy)
            if math.hypot(q[0] - v[0], q[1] - v[1]) < 4 * block_px:
                v = q
        out.append(v)
    return out, refined, resid


def _profile_tiles(centres_m, nrms_m, half, px_m, z):
    """Every refine-zoom tile the profiles will touch (for prefetch)."""
    keys = []
    for c, nrm in zip(centres_m, nrms_m):
        for s in (-half, -half // 2, 0, half // 2, half):
            tx, ty = m_to_tile(c[0] + s * px_m * nrm[0], c[1] + s * px_m * nrm[1], z)
            keys.append((z, tx, ty))
    return keys


def _sample(img: np.ndarray, fx: float, fy: float, r: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    x, y = int(fx), int(fy)
    return img[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]


def _profile_step(src: ChartSource, c: Pt, nrm: Pt, half: int, px_m: float, z: int, cache: TileCache) -> Optional[float]:
    """Offset (m, along the outward normal) where the FAA mosaic stops
    showing this chart, or None when the profile is inconclusive."""
    match = []
    for s in range(-half, half + 1, 2):
        mx, my = c[0] + s * px_m * nrm[0], c[1] + s * px_m * nrm[1]
        tx, ty = m_to_tile(mx, my, z)
        tb = tile_bounds_m(z, tx, ty)
        fx = (mx - tb[0]) / px_m
        fy = (tb[3] - my) / px_m
        d, comparable = cache.diff_tile(src, z, tx, ty)
        ok = _sample(comparable, fx, fy)
        if ok.size == 0 or not ok.all():
            match.append(False)
            continue
        match.append(float(_sample(d, fx, fy).mean()) < REFINE_THRESHOLD)
    m = np.asarray(match)
    n = len(m)
    inner, outer = m[: n // 4], m[-(n // 4):]
    # inconclusive unless the inside clearly matches and the outside clearly does not
    if inner.mean() < 0.7 or outer.mean() > 0.3:
        return None
    t = step_location(m)
    return (-half + 2 * t - 1) * px_m


def _refine_chart(args):
    """Polygon + edge refinement for one chart (worker process)."""
    name, path, service, cache_dir, mask, origin, zc, zr = args
    src = ChartSource.open(name, path)
    cache = TileCache(service, cache_dir)
    block_m = 2 * ORIGIN / (2 ** zc) / TILE * BLOCK
    er = EdgeResult(name, [])
    for ring_m in mask_rings_m(mask, origin, zc):
        ring_px = m_to_px(src, ring_m[:-1] if ring_m[0] == ring_m[-1] else ring_m)
        # source pixels per coarse block, measured on this ring
        seg_m = math.hypot(ring_m[1][0] - ring_m[0][0], ring_m[1][1] - ring_m[0][1]) or block_m
        seg_px = math.hypot(ring_px[1][0] - ring_px[0][0], ring_px[1][1] - ring_px[0][1]) or 1.0
        block_px = block_m * seg_px / seg_m
        ring = simplify_ring(ring_px, DP_TOLERANCE_BLOCKS * block_px)
        rr, nref, res = refine_ring(src, ring, zr, cache, block_m, block_px)
        er.rings_px.append(rr)
        er.refined_edges += nref
        er.total_edges += len(ring)
        er.residual_px += res
    src.ds.close()
    return er


def measure_family(fam, charts: Sequence[ChartSource], faa_cache_dir: str, log=print, workers: int = 0,
                   coarse_cache: Optional[str] = None) -> List[EdgeResult]:
    """`coarse_cache`: optional pickle of the coarse ownership, keyed by the
    charts' names + fingerprints (the slow, CPU-bound half; edge refinement
    can be re-run against it)."""
    import os
    import pickle
    from multiprocessing import Pool

    cache = TileCache(fam.service, faa_cache_dir)
    zc = fam.measure_zoom
    key = [(c.name, fingerprint(c)) for c in charts]
    own = None
    if coarse_cache and os.path.exists(coarse_cache):
        with open(coarse_cache, "rb") as f:
            saved = pickle.load(f)
        if saved.get("key") == key and saved.get("z") == zc:
            own = saved["own"]
            log(f"[{fam.id}] coarse ownership at z{zc}: reused {coarse_cache}")
    if own is None:
        log(f"[{fam.id}] coarse ownership at z{zc} over {len(charts)} charts")
        own = coarse_matches(charts, zc, cache, log, workers)
        if coarse_cache:
            with open(coarse_cache, "wb") as f:
                pickle.dump({"key": key, "z": zc, "own": own}, f)
    masks = chart_masks(own, len(charts), zc)
    jobs = [(charts[ci].name, charts[ci].path, fam.service, faa_cache_dir, mask, origin, zc, fam.refine_zoom)
            for ci, (mask, origin) in sorted(masks.items())]
    log(f"[{fam.id}] refining edges of {len(jobs)} charts at z{fam.refine_zoom}")
    results = []
    with Pool(workers or os.cpu_count() or 2) as pool:
        for er in pool.imap_unordered(_refine_chart, jobs):
            log(f"  {er.chart}: {len(er.rings_px)} piece(s), {er.refined_edges}/{er.total_edges} edges refined, "
                f"median fit residual {np.median(er.residual_px) if er.residual_px else float('nan'):.1f} px")
            results.append(er)
    results.sort(key=lambda r: r.chart)
    unowned = [c.name for i, c in enumerate(charts) if i not in masks]
    if unowned:
        log(f"  not in the FAA mosaic (excluded): {', '.join(unowned)}")
    return results


def fingerprint(src: ChartSource) -> dict:
    """What must stay the same for a stored edge to be reused."""
    t = src.ds.transform
    return {"width": src.ds.width, "height": src.ds.height, "crs": src.ds.crs.to_wkt()[:4000],
            "transform": [round(v, 6) for v in (t.a, t.b, t.c, t.d, t.e, t.f)]}


def fingerprint_matches(a: dict, b: dict, tol_m: float = 50.0) -> bool:
    if not a or not b or a.get("crs") != b.get("crs"):
        return False
    ta, tb = a["transform"], b["transform"]
    # origin within tol, pixel size within 0.1%
    if abs(ta[2] - tb[2]) > tol_m or abs(ta[5] - tb[5]) > tol_m:
        return False
    if abs(ta[0] - tb[0]) > abs(tb[0]) * 1e-3 or abs(ta[4] - tb[4]) > abs(tb[4]) * 1e-3:
        return False
    return True


def to_json(fam_id: str, edition: str, results: List[EdgeResult], srcs: Dict[str, ChartSource]) -> dict:
    """Stored edges: authoritative rings in each GeoTIFF's pixel space + the
    georeferencing fingerprint they are valid for, plus a lon/lat preview
    (densified) for humans and map overlays."""
    charts = {}
    for er in results:
        src = srcs[er.chart]
        preview = []
        for r in er.rings_px:
            if len(r) < 3:
                continue
            ll = [m_to_lonlat(*p) for p in px_to_m(src, densify(r, 256))]
            preview.append([[round(a, 5), round(b, 5)] for a, b in ll])
        charts[er.chart] = {
            "rings_px": [[[round(c, 2), round(rw, 2)] for c, rw in r] for r in er.rings_px if len(r) >= 3],
            "fingerprint": fingerprint(src),
            "refined_edges": er.refined_edges, "total_edges": er.total_edges,
            "median_residual_px": round(float(np.median(er.residual_px)), 2) if er.residual_px else None,
            "preview_lonlat": preview,
        }
    return {"family": fam_id, "measured_from_edition": edition,
            "method": "ownership vs the FAA tile-service mosaic (structural match); edges refined by 1-D profiles and fitted as straight lines in each chart's pixel space",
            "coarse_threshold": COARSE_THRESHOLD, "refine_threshold": REFINE_THRESHOLD, "block_px": BLOCK,
            "charts": charts}
