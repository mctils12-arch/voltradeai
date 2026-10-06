"""Shared helpers for the FAA chart bake: Web-Mercator tile math, rendering a
FAA GeoTIFF into a map tile, and fetching the FAA's own tile-service mosaic
(the ground truth the chart edges are measured against).

Pure functions (tile math, palette LUT) are unit-tested in
tests/test_faa_charts.py; rendering/fetching are exercised by the bake run.
"""
from __future__ import annotations

import io
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

R = 6378137.0
ORIGIN = math.pi * R  # half the Web-Mercator world width, metres
TILE = 256

FAA_TILE_BASE = "https://tiles.arcgis.com/tiles/ssFJjBXIUyZDrSYZ/arcgis/rest/services"
USER_AGENT = "VolTradeAI-chartbake/1.0 (+FAA chart edge measurement; cached, one fetch per tile)"


# ── tile math (pure) ────────────────────────────────────────────────────────

def tile_bounds_m(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """(west, south, east, north) of tile z/x/y in EPSG:3857 metres."""
    s = 2 * ORIGIN / (2 ** z)
    return (-ORIGIN + x * s, ORIGIN - (y + 1) * s, -ORIGIN + (x + 1) * s, ORIGIN - y * s)


def lonlat_to_m(lon: float, lat: float) -> Tuple[float, float]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    return (math.radians(lon) * R, R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def m_to_lonlat(x: float, y: float) -> Tuple[float, float]:
    return (math.degrees(x / R), math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2))


def m_to_tile(x: float, y: float, z: int) -> Tuple[int, int]:
    """Tile containing the 3857 point, clamped to the world."""
    n = 2 ** z
    tx = int((x + ORIGIN) / (2 * ORIGIN) * n)
    ty = int((ORIGIN - y) / (2 * ORIGIN) * n)
    return (min(max(tx, 0), n - 1), min(max(ty, 0), n - 1))


def tiles_for_bounds_m(bounds: Tuple[float, float, float, float], z: int):
    """Every tile at z intersecting a 3857 bbox, row-major."""
    w, s, e, n = bounds
    x0, y0 = m_to_tile(w, n, z)
    x1, y1 = m_to_tile(e, s, z)
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            yield (tx, ty)


def palette_lut(colormap: dict) -> np.ndarray:
    """rasterio colormap {index: (r,g,b,a)} -> (256, 4) uint8 lookup table.
    Indices the palette does not define stay fully transparent."""
    lut = np.zeros((256, 4), np.uint8)
    for k, v in colormap.items():
        if 0 <= int(k) < 256:
            lut[int(k)] = tuple(v)[:4] if len(v) >= 4 else (*tuple(v)[:3], 255)
    return lut


def block_mean(a: np.ndarray, b: int) -> np.ndarray:
    """Mean over non-overlapping b x b blocks of an (H, W[, C]) array (H, W
    multiples of b)."""
    h, w = a.shape[0] // b, a.shape[1] // b
    a = a[: h * b, : w * b]
    if a.ndim == 2:
        return a.reshape(h, b, w, b).mean((1, 3))
    return a.reshape(h, b, w, b, a.shape[2]).mean((1, 3))


# ── rendering a chart GeoTIFF into a tile ───────────────────────────────────

@dataclass
class ChartSource:
    """An open FAA chart GeoTIFF: palette (paletted uint8, the FAA's format)
    or RGB(A)."""
    name: str
    path: str
    ds: object  # rasterio dataset
    lut: Optional[np.ndarray]
    bounds_m: Tuple[float, float, float, float]

    @staticmethod
    def open(name: str, path: str) -> "ChartSource":
        import rasterio
        from rasterio.warp import transform_bounds

        ds = rasterio.open(path)
        lut = None
        if ds.count == 1:
            try:
                lut = palette_lut(ds.colormap(1))
            except ValueError:
                lut = None  # greyscale single band
        b = transform_bounds(ds.crs, "EPSG:3857", *ds.bounds, densify_pts=64)
        return ChartSource(name, path, ds, lut, b)

    def render(self, z: int, x: int, y: int, size: int = TILE) -> Tuple[np.ndarray, np.ndarray]:
        """(rgb float32 (size,size,3), valid bool (size,size)) for tile z/x/y.
        Nearest-neighbour on palette indices (palette colours must never be
        blended as numbers); callers supersample and average for anti-aliasing."""
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject

        dst_t = from_bounds(*tile_bounds_m(z, x, y), size, size)
        # Warp into uint16 initialised to 65535: pixels the source does not
        # cover keep the sentinel, so one warp yields colour AND footprint
        # (every uint8 palette index, 0 included, is a real colour).
        bands = 1 if self.ds.count == 1 else min(self.ds.count, 3)
        out = np.zeros((bands, size, size), np.uint16)
        for i in range(bands):
            reproject(rasterio.band(self.ds, i + 1), out[i], dst_transform=dst_t, dst_crs="EPSG:3857",
                      resampling=Resampling.nearest, src_nodata=None, dst_nodata=NO_SOURCE)
        valid = out[0] != NO_SOURCE
        if bands == 1:
            idx = np.where(valid, out[0], 0).astype(np.uint8)
            if self.lut is not None:
                return self.lut[idx][..., :3].astype(np.float32), valid
            return np.repeat(idx[..., None], 3, -1).astype(np.float32), valid
        rgb = np.where(valid[..., None], np.moveaxis(out, 0, -1), 0)
        return rgb.astype(np.float32), valid


NO_SOURCE = 65535


# ── the FAA tile-service mosaic (edge ground truth) ─────────────────────────

def fetch_faa_tile(service: str, z: int, x: int, y: int, cache_dir: str,
                   retries: int = 3, timeout: float = 20.0) -> np.ndarray:
    """RGBA uint8 (256,256,4) of the FAA's own clipped mosaic tile; fully
    transparent where the service has nothing (404). Disk-cached: chart edges
    are measured once per chart shape, not once per edition."""
    from PIL import Image

    path = os.path.join(cache_dir, service, str(z), str(y), f"{x}.bin")
    if os.path.exists(path):
        data = open(path, "rb").read()
    else:
        url = f"{FAA_TILE_BASE}/{service}/MapServer/tile/{z}/{y}/{x}"
        data = b""
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"user-agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break
                if attempt == retries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == retries - 1:
                    raise
            time.sleep(1.5 * (attempt + 1))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    if not data:
        return np.zeros((TILE, TILE, 4), np.uint8)
    try:
        return np.array(Image.open(io.BytesIO(data)).convert("RGBA"))
    except Exception:
        return np.zeros((TILE, TILE, 4), np.uint8)
