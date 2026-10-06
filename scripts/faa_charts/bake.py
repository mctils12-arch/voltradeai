"""Bake one chart family for one FAA edition into a single PMTiles file.

  chart GeoTIFFs + measured map edges (edges.py)  ->  seamless mosaic tiles
  at the family's native zoom  ->  2x2-shrunk levels down to zoom 2  ->
  one .pmtiles (the format the site already serves from R2 for the power
  grid; range-readable, so the server reads one tile per request).

Each max-zoom tile is rendered at 2x (512 px, nearest on palette indices),
clipped to every contributing chart's measured map polygon, composited, and
averaged down to 256 px with premultiplied alpha — anti-aliased and never
darkened at chart edges. Tiles are WebP (lossy colour, lossless alpha) —
a third smaller than JPEG/PNG at the same legibility.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import TILE, ChartSource, tile_bounds_m, tiles_for_bounds_m

# WebP q80: measured 2026-10-06 on 300 sampled sectional z11 tiles — 15.2 KB
# vs 23.1 KB for JPEG q85 / PNG (-34%), chart text still crisp at z11. Every
# browser MapLibre supports decodes WebP; the server passes bytes through.
WEBP_QUALITY = 80
SUPERSAMPLE = 2
OVERVIEW_MIN_ZOOM = 2  # server/aeroCharts.ts AERO_OVERVIEW_MIN_ZOOM
# Seam fill reach: 1.5 coarse measurement blocks at z10 (16 px x 153 m), the
# largest offset an unrefined shared edge can carry (edges.py).
SEAM_FILL_M = 1.5 * 16 * (2 * 20037508.342789244 / 2 ** 10 / 256)
ANTIMERIDIAN_SNAP_M = 3 * 16 * (2 * 20037508.342789244 / 2 ** 10 / 256)

Ring = List[Tuple[float, float]]


# ── pure raster helpers (unit-tested) ───────────────────────────────────────

def downsample_premultiplied(rgba: np.ndarray, f: int) -> np.ndarray:
    """(H, W, 4) float RGBA (alpha 0..1) -> (H/f, W/f, 4), averaging colour
    weighted by alpha so transparent pixels never pull colour toward black."""
    h, w = rgba.shape[0] // f, rgba.shape[1] // f
    a = rgba[: h * f, : w * f, 3:4]
    pm = rgba[: h * f, : w * f, :3] * a
    pm = pm.reshape(h, f, w, f, 3).mean((1, 3))
    am = a.reshape(h, f, w, f, 1).mean((1, 3))
    rgb = np.where(am > 0, pm / np.maximum(am, 1e-9), 0)
    return np.concatenate([rgb, am], -1)


def compose_children(children: Sequence[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """Four child tiles (256x256x4 float, alpha 0..1, or None) in quadtree
    order [NW, NE, SW, SE] -> their parent, each child shrunk to a quadrant."""
    if len(children) != 4:
        raise ValueError("compose_children needs exactly 4 children")
    if all(c is None for c in children):
        return None
    out = np.zeros((TILE, TILE, 4), np.float32)
    h = TILE // 2
    for q, c in enumerate(children):
        if c is None:
            continue
        out[(q // 2) * h:(q // 2 + 1) * h, (q % 2) * h:(q % 2 + 1) * h] = downsample_premultiplied(c, 2)
    return out


def encode(rgba: np.ndarray) -> Optional[bytes]:
    """float RGBA (alpha 0..1) -> WebP (alpha kept only when some pixel is
    not opaque), None when nothing is visible."""
    from PIL import Image

    a = rgba[..., 3]
    if a.max() <= 0:
        return None
    u8 = np.clip(np.rint(np.concatenate([rgba[..., :3], a[..., None] * 255], -1)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    if (u8[..., 3] == 255).all():
        Image.fromarray(u8[..., :3]).save(buf, "WEBP", quality=WEBP_QUALITY, method=4)
    else:
        Image.fromarray(u8).save(buf, "WEBP", quality=WEBP_QUALITY, method=4, alpha_quality=100)
    return buf.getvalue()


def decode(data: bytes) -> np.ndarray:
    from PIL import Image

    a = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"), np.float32)
    a[..., 3] /= 255.0
    return a


def parent_tiles(tiles, z: int):
    return sorted({(x // 2, y // 2) for (x, y) in tiles})


# ── per-tile rendering (worker processes) ───────────────────────────────────

_W: Dict[str, object] = {}


def _worker_init(chart_specs, polys, z):
    _W["charts"] = [ChartSource.open(n, p) for n, p in chart_specs]
    _W["polys"] = polys  # chart index -> [(bbox, ring)]
    _W["z"] = z


def _bbox_hit(b, t) -> bool:
    return b[0] < t[2] and t[0] < b[2] and b[1] < t[3] and t[1] < b[3]


def render_tile(xy) -> Tuple[Tuple[int, int], Optional[bytes]]:
    """One max-zoom tile: every chart clipped to its measured map polygon,
    first chart wins in overlaps; then SEAM FILL — a pixel no polygon covers
    but that lies within SEAM_FILL_M of TWO OR MORE charts' polygons is a
    sliver between neighbouring charts (an edge not refined to the exact
    line), filled from the nearest chart that has map there. A gap next to
    only one chart is an outer border and stays empty (no collar spill)."""
    from rasterio import features
    from rasterio.transform import from_bounds
    from scipy import ndimage as nd

    x, y = xy
    z = _W["z"]
    tb = tile_bounds_m(z, x, y)
    size = TILE * SUPERSAMPLE
    px_m = (tb[2] - tb[0]) / size
    k = int(np.ceil(SEAM_FILL_M / px_m))
    ext = (tb[0] - k * px_m, tb[1] - k * px_m, tb[2] + k * px_m, tb[3] + k * px_m)
    big = size + 2 * k
    acc = np.zeros((size, size, 4), np.float32)
    rendered: Dict[int, tuple] = {}
    near: Dict[int, np.ndarray] = {}  # chart -> polygon mask over the extended tile
    for ci, parts in _W["polys"].items():
        rings = [r for bbox, r in parts if _bbox_hit(bbox, ext)]
        if not rings:
            continue
        clip_ext = features.rasterize([({"type": "Polygon", "coordinates": [r]}, 1) for r in rings],
                                      out_shape=(big, big), transform=from_bounds(*ext, big, big),
                                      fill=0, dtype="uint8").astype(bool)
        if not clip_ext.any():
            continue
        near[ci] = clip_ext
        clip = clip_ext[k:k + size, k:k + size]
        if not clip.any():
            continue
        rgb, valid = _W["charts"][ci].render(z, x, y, size)
        rendered[ci] = (rgb, valid)
        m = clip & valid & (acc[..., 3] == 0)  # first chart wins inside overlap slivers
        acc[m, :3] = rgb[m]
        acc[m, 3] = 1.0
    gap = acc[..., 3] == 0
    if len(near) >= 2 and gap.any():
        dists = {}
        for ci, clip_ext in near.items():
            d = nd.distance_transform_edt(~clip_ext)[k:k + size, k:k + size]
            if (d[gap] <= k).any():
                dists[ci] = d
        if len(dists) >= 2:
            stack = np.stack([dists[ci] for ci in dists])
            seam = gap & ((stack <= k).sum(0) >= 2)
            order = np.argsort(stack, axis=0)  # nearest chart first
            ids = list(dists)
            for rank in range(len(ids)):
                if not seam.any():
                    break
                for j, ci in enumerate(ids):
                    pick = seam & (order[rank] == j) & (stack[j] <= k)
                    if not pick.any():
                        continue
                    if ci not in rendered:
                        rendered[ci] = _W["charts"][ci].render(z, x, y, size)
                    rgb, valid = rendered[ci]
                    m = pick & valid
                    acc[m, :3] = rgb[m]
                    acc[m, 3] = 1.0
                    seam &= ~m
    if acc[..., 3].max() == 0:
        return (x, y), None
    return (x, y), encode(downsample_premultiplied(acc, SUPERSAMPLE))


def _compose_worker(args):
    (px, py), kids = args
    return (px, py), encode_or_none(compose_children([decode(k) if k else None for k in kids]))


def encode_or_none(rgba):
    return None if rgba is None else encode(rgba)


# ── the bake ────────────────────────────────────────────────────────────────

def unwrap_ring(ring: Ring) -> List[Ring]:
    """A 3857 ring that jumps across the antimeridian (x flips sign by ~2x
    ORIGIN) -> the continuous ring east of it plus a copy shifted one world
    west, so rasterizing a tile on either side sees the polygon."""
    from common import ORIGIN

    xs = [p[0] for p in ring]
    if max(xs) - min(xs) <= ORIGIN:
        return [ring]
    east = [(x + 2 * ORIGIN if x < 0 else x, y) for x, y in ring]
    return [east, [(x - 2 * ORIGIN, y) for x, y in east]]


def snap_antimeridian(ring: Ring) -> Ring:
    """A chart crossing the antimeridian is measured as two pieces cut at
    +-180 by the tile grid; the coarse mask clean-up (opening) can leave
    their edges up to ~3 measurement blocks short of the line (5.2 km on the
    Western Aleutian Islands East sectional, 2026-09-03). Points within
    ANTIMERIDIAN_SNAP_M of it are put ON it, so the two halves meet exactly."""
    from common import ORIGIN

    t = ANTIMERIDIAN_SNAP_M
    return [(ORIGIN if x > ORIGIN - t else -ORIGIN if x < -ORIGIN + t else x, y) for x, y in ring]


def polys_from_edges(edges_json: dict, charts: Sequence[ChartSource]) -> Dict[int, List[Tuple[tuple, Ring]]]:
    """Stored edges (rings in each GeoTIFF's pixel space, edges.py) ->
    {chart index: [(3857 bbox, ring in 3857), ...]}. Edges are straight in
    pixel space, so they are densified every 64 source pixels (~2.7 km on a
    sectional) before projecting — the bowing between points is < 1 m. A
    ring crossing the antimeridian is kept continuous and duplicated one
    world west (unwrap_ring)."""
    from edges import densify, px_to_m

    out = {}
    for i, c in enumerate(charts):
        e = edges_json["charts"].get(c.name)
        if not e:
            continue
        parts = []
        for r in e["rings_px"]:
            ring = snap_antimeridian(px_to_m(c, densify([tuple(p) for p in r], 64)))
            if len(ring) < 3:
                continue
            for u in unwrap_ring(ring + [ring[0]]):
                xs = [p[0] for p in u]
                ys = [p[1] for p in u]
                parts.append(((min(xs), min(ys), max(xs), max(ys)), u))
        if parts:
            out[i] = parts
    return out


def write_pmtiles(out_path: str, entries, header: dict, metadata: dict) -> None:
    """entries: [(tile id, path-or-bytes)] sorted by tile id; WebP tiles.
    Written to a .part file and renamed, so a crash never leaves a truncated
    archive under the final name."""
    import pmtiles.tile as pt
    from pmtiles.writer import Writer

    tmp_out = out_path + ".part"
    with open(tmp_out, "wb") as f:
        wr = Writer(f)
        for tid, src in entries:
            if isinstance(src, (bytes, bytearray)):
                wr.write_tile(tid, bytes(src))
            else:
                with open(src, "rb") as fh:
                    wr.write_tile(tid, fh.read())
        wr.finalize({"tile_type": pt.TileType.WEBP, "tile_compression": pt.Compression.NONE, **header}, metadata)
    os.replace(tmp_out, out_path)


def bake_family(fam, chart_specs: Sequence[Tuple[str, str]], edges_json: dict, out_path: str, work_dir: str,
                edition: str, workers: int = os.cpu_count() or 2, log=print, max_zoom: Optional[int] = None) -> dict:
    """Render, pyramid and pack. Returns the bake report (tile counts, bytes,
    timing) that is also written into the PMTiles metadata."""
    import pmtiles.tile as pt

    t0 = time.time()
    names = [n for n, _ in chart_specs]
    polys = polys_from_edges(edges_json, [ChartSource.open(n, p) for n, p in chart_specs])
    missing = [n for i, n in enumerate(names) if i not in polys]
    if missing:
        log(f"[{fam.id}] no measured edge (not baked; excluded as in the FAA mosaic): {', '.join(missing)}")
    zmax = max_zoom or fam.max_bake_zoom
    tiles = set()
    from common import ORIGIN

    for parts in polys.values():
        for (w, s_, e, n), _ in parts:
            box = (max(w, -ORIGIN), s_, min(e, ORIGIN), n)
            if box[0] < box[2]:
                tiles.update(tiles_for_bounds_m(box, zmax))
    tiles = sorted(tiles, key=lambda t: (t[1], t[0]))
    log(f"[{fam.id}] z{zmax}: {len(tiles)} candidate tiles from {len(polys)} charts")

    os.makedirs(work_dir, exist_ok=True)
    store = {}  # z -> {(x,y): path}
    # max zoom: render (encoded tiles kept on disk, not in RAM)
    zdir = os.path.join(work_dir, str(zmax))
    os.makedirs(zdir, exist_ok=True)
    store[zmax] = {}
    with Pool(workers, initializer=_worker_init, initargs=(list(chart_specs), polys, zmax)) as pool:
        for i, ((x, y), data) in enumerate(pool.imap_unordered(render_tile, tiles, chunksize=8)):
            if data:
                p = os.path.join(zdir, f"{x}_{y}")
                with open(p, "wb") as f:
                    f.write(data)
                store[zmax][(x, y)] = p
            if i % 5000 == 0:
                log(f"  z{zmax}: {i}/{len(tiles)} ({time.time() - t0:.0f}s)")
    # pyramid down to the overview floor
    with Pool(workers) as pool:
        for z in range(zmax - 1, OVERVIEW_MIN_ZOOM - 1, -1):
            child = store[z + 1]
            parents = parent_tiles(child.keys(), z + 1)
            jobs = []
            for (px, py) in parents:
                kids = []
                for (cx, cy) in ((2 * px, 2 * py), (2 * px + 1, 2 * py), (2 * px, 2 * py + 1), (2 * px + 1, 2 * py + 1)):
                    p = child.get((cx, cy))
                    kids.append(open(p, "rb").read() if p else None)
                jobs.append(((px, py), kids))
            zdir = os.path.join(work_dir, str(z))
            os.makedirs(zdir, exist_ok=True)
            store[z] = {}
            for (px, py), data in pool.imap_unordered(_compose_worker, jobs, chunksize=16):
                if data:
                    p = os.path.join(zdir, f"{px}_{py}")
                    with open(p, "wb") as f:
                        f.write(data)
                    store[z][(px, py)] = p
            log(f"  z{z}: {len(store[z])} tiles ({time.time() - t0:.0f}s)")

    # pack
    entries = []
    for z, d in store.items():
        for (x, y), p in d.items():
            entries.append((pt.zxy_to_tileid(z, x, y), p))
    entries.sort()
    counts = {str(z): len(d) for z, d in sorted(store.items())}
    total_bytes = sum(os.path.getsize(p) for _, p in entries)
    from common import m_to_lonlat

    bx = [b for parts in polys.values() for b, _ in parts]
    w, s = m_to_lonlat(max(-ORIGIN, min(b[0] for b in bx)), min(b[1] for b in bx))
    e, n = m_to_lonlat(min(ORIGIN, max(b[2] for b in bx)), max(b[3] for b in bx))
    report = {
        "family": fam.id, "edition": edition, "charts": len(polys), "excluded_charts": missing,
        "min_zoom": OVERVIEW_MIN_ZOOM, "max_zoom": zmax, "faa_min_zoom": fam.min_zoom,
        "tiles_per_zoom": counts, "tile_bytes": total_bytes, "seconds": round(time.time() - t0),
        "source": "FAA AIS GeoTIFFs (aeronav.faa.gov), clipped to edges measured against the FAA tile-service mosaic",
    }
    write_pmtiles(out_path, entries, {
        "min_zoom": OVERVIEW_MIN_ZOOM, "max_zoom": zmax,
        "min_lon_e7": int(w * 1e7), "min_lat_e7": int(s * 1e7), "max_lon_e7": int(e * 1e7), "max_lat_e7": int(n * 1e7),
        "center_zoom": max(OVERVIEW_MIN_ZOOM, fam.min_zoom), "center_lon_e7": int((w + e) / 2 * 1e7),
        "center_lat_e7": int((s + n) / 2 * 1e7)}, report)
    shutil.rmtree(work_dir, ignore_errors=True)
    report["pmtiles_bytes"] = os.path.getsize(out_path)
    log(f"[{fam.id}] baked {sum(counts.values())} tiles, {report['pmtiles_bytes'] / 1e9:.2f} GB in {report['seconds']}s")
    return report
