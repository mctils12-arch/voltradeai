#!/usr/bin/env python3
"""cog_reader.py — GRID VISION Phase B: NAIP Cloud-Optimized-GeoTIFF WINDOW reader.

Given a signed NAIP COG href (from gridvision_naip_stac.sign_href) + a geographic
chip bbox (the chip_bbox emitted by gridvision_chips.build_chip_index) + a target
GSD, read JUST that window out of the remote COG (HTTP range reads, no full
download) and return an image array downsampled to the target GSD.

WHY THIS FILE EXISTS: gridvision_naip_stac.py returns a signed COG href only —
there was NO pixel reader. This is that missing reader. It is the (A) piece of
the on-pod pipeline: the reader that turns the chip INDEX into actual NAIP pixels
for the detector.

STRICT PURITY SPLIT (the whole reason this file is testable offline):
  - The bbox->pixel-window math and the GSD downsample math are PURE functions
    (pixel_window_from_bounds, downsample_factor, out_shape_for_window,
    north_up_transform). They take plain numbers, return plain numbers, import
    nothing heavy, and are exercised by test_gridvision_train.py with NO rasterio.
  - read_cog_window() is the ONLY function that touches rasterio. rasterio is an
    on-pod pip dependency (NOT a session/CI dependency), so it is imported INSIDE
    the function. Offline tests never call it and therefore never import rasterio.

CRS CAVEAT (honest, verified against the STAC item shape): the chip_bbox is in
geographic degrees (EPSG:4326, lon/lat) but a NAIP COG's native CRS is a UTM zone
(projected metres). A windowed read therefore reprojects the request bbox into
the raster CRS first (rasterio.warp.transform_bounds) — done inside
read_cog_window on-pod. The pure window math below operates in the raster's OWN
CRS/affine and is unit-tested there; the reprojection is the only part that needs
rasterio/pyproj and is exercised on-pod, not in CI.
"""
import math

# Affine transform convention (matches rasterio.transform.Affine.to_gdal-ish
# order but as a plain 6-tuple): (a, b, c, d, e, f) where
#     x = a*col + b*row + c      (map X / easting of a pixel corner)
#     y = d*col + e*row + f       (map Y / northing)
# For a standard north-up raster: b == d == 0, a = +pixel_size, e = -pixel_size,
# (c, f) = the map coords of the top-left corner (col=0, row=0).


def north_up_transform(origin_x, origin_y, pixel_size):
    """Build a north-up affine (a,b,c,d,e,f) for a raster whose top-left corner
    is (origin_x, origin_y) with square pixels of `pixel_size` map units. Pure."""
    px = float(pixel_size)
    if px <= 0:
        raise ValueError("pixel_size must be > 0")
    return (px, 0.0, float(origin_x), 0.0, -px, float(origin_y))


def pixel_window_from_bounds(west, south, east, north, transform, raster_size=None):
    """Geographic/projected bounds (in the RASTER's own CRS) -> integer pixel
    window (col_off, row_off, width, height), mirroring rasterio.windows.
    from_bounds. Assumes a north-up transform (b == d == 0).

    Offsets floor toward the top-left; width/height ceil so the window fully
    covers the requested bounds (never clips a partial edge pixel). Offsets are
    clamped to >= 0. If `raster_size` = (n_cols, n_rows) is given, the window is
    additionally clamped to the raster extent so a read never runs off the COG.
    Pure — no rasterio."""
    a, b, c, d, e, f = transform
    if b != 0 or d != 0:
        raise ValueError("pixel_window_from_bounds assumes a north-up transform (b==d==0)")
    if a == 0 or e == 0:
        raise ValueError("degenerate transform (a or e is zero)")
    # column runs with +x; row runs with -y (e is negative for north-up).
    cols = [(west - c) / a, (east - c) / a]
    rows = [(north - f) / e, (south - f) / e]
    col_off = int(math.floor(min(cols)))
    row_off = int(math.floor(min(rows)))
    col_far = int(math.ceil(max(cols)))
    row_far = int(math.ceil(max(rows)))
    col_off = max(0, col_off)
    row_off = max(0, row_off)
    width = max(0, col_far - col_off)
    height = max(0, row_far - row_off)
    if raster_size is not None:
        n_cols, n_rows = int(raster_size[0]), int(raster_size[1])
        col_off = min(col_off, n_cols)
        row_off = min(row_off, n_rows)
        width = max(0, min(width, n_cols - col_off))
        height = max(0, min(height, n_rows - row_off))
    return (col_off, row_off, width, height)


def downsample_factor(native_gsd, target_gsd):
    """How many native pixels map to one output pixel = target_gsd / native_gsd.
    >1 => genuine DOWNSAMPLE (target coarser than native — e.g. 0.30 m NAIP read
    at 0.60 m => 2.0). ==1 => native. <1 => the target is FINER than the source;
    reading cannot invent detail, so callers should treat <1 as 'read native'
    (factor is still returned honestly, never silently clamped). Pure."""
    native = float(native_gsd)
    target = float(target_gsd)
    if native <= 0 or target <= 0:
        raise ValueError("gsd values must be > 0")
    return target / native


def out_shape_for_window(win_width, win_height, factor):
    """Output (out_width, out_height) in pixels for a native window of
    (win_width, win_height) resampled by `factor` native-px-per-out-px. A window
    of zero size stays zero; otherwise each dimension is at least 1 px. Pure —
    this is exactly the out_shape rasterio's windowed read is asked for."""
    if factor <= 0:
        raise ValueError("factor must be > 0")
    ow = 0 if win_width <= 0 else max(1, int(round(win_width / factor)))
    oh = 0 if win_height <= 0 else max(1, int(round(win_height / factor)))
    return (ow, oh)


# ── the ONLY rasterio-touching function (on-pod; guarded import) ─────────────

def read_cog_window(signed_href, chip_bbox_lonlat, target_gsd=0.6,
                    bbox_crs="EPSG:4326", bands=(1, 2, 3)):
    """Read one chip window out of a remote NAIP COG at `target_gsd`.

    signed_href      : signed COG URL from gridvision_naip_stac.sign_href()
    chip_bbox_lonlat : [W, S, E, N] in `bbox_crs` (the chip index is EPSG:4326)
    target_gsd       : output ground sample distance in metres (0.6 to match the
                       detector's training GSD)
    returns          : (array, meta) where array is (n_bands, h, w) uint8 and meta
                       carries native_gsd, factor, window, out_shape, crs.

    ON-POD ONLY. rasterio + rasterio.warp are imported HERE so offline tests and
    --dry-run never import them. Uses HTTP range reads via GDAL's /vsicurl (no
    full-file download). The bbox is reprojected from bbox_crs into the COG's UTM
    CRS before the pure window math is applied."""
    import rasterio  # on-pod pip dep — intentionally a local import
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window
    from rasterio.enums import Resampling

    w, s, e, n = [float(v) for v in chip_bbox_lonlat]
    with rasterio.open(signed_href) as ds:
        # native GSD from the raster's own affine (|a| == pixel size in map units)
        native_gsd = abs(ds.transform.a)
        # reproject the request bbox from its CRS into the raster CRS
        rw, rs, re_, rn = transform_bounds(bbox_crs, ds.crs, w, s, e, n)
        transform = (ds.transform.a, ds.transform.b, ds.transform.c,
                     ds.transform.d, ds.transform.e, ds.transform.f)
        col_off, row_off, width, height = pixel_window_from_bounds(
            rw, rs, re_, rn, transform, raster_size=(ds.width, ds.height))
        factor = downsample_factor(native_gsd, target_gsd)
        out_w, out_h = out_shape_for_window(width, height, factor)
        if out_w == 0 or out_h == 0:
            return None, {"native_gsd": native_gsd, "factor": factor,
                          "window": (col_off, row_off, width, height),
                          "out_shape": (out_w, out_h), "crs": str(ds.crs),
                          "note": "empty window (chip outside raster or zero size)"}
        window = Window(col_off, row_off, width, height)
        arr = ds.read(list(bands), window=window,
                      out_shape=(len(bands), out_h, out_w),
                      resampling=Resampling.average if factor > 1 else Resampling.nearest)
        return arr, {"native_gsd": native_gsd, "factor": factor,
                     "window": (col_off, row_off, width, height),
                     "out_shape": (out_w, out_h), "crs": str(ds.crs)}
