#!/usr/bin/env python3
"""gebco_seafloor_tiles.py — GEBCO bathymetry + TID confidence PMTiles pipeline
(EARTH TWIN E2 v2, research/earth_twin_program.md V4: "self-tiled GEBCO pmtiles
via a scripts/ pipeline … for real 15-arcsec fidelity" + "TID grid … ship it as
the seafloor_confidence overlay").

WHAT IT DOES (BUILD-DON'T-BUY: we own every step, zero paid tooling):
  1. FETCH  — submits a basket to GEBCO's own grid-subsetting service
              (https://download.gebco.net, VERIFIED LIVE 2026-07-16 through the
              agent proxy: POST /api/queue -> {basketId}; GET
              /api/queue/status/{id} until "finished"; GET
              /api/queue/download/{id} -> zip of Esri ASCII grids + the official
              GEBCO_Grid_documentation.pdf + terms PDF). Requests BOTH data
              sources of the current global grid: Bathymetry (elevation, m) and
              the Type Identifier (TID) grid, as Esri ASCII (stdlib-parseable —
              the global-run NetCDF path is documented at the bottom, NOT built).
  2. BUILD  — renders web-mercator 256px raster tiles from the subset:
              - depth tiles: TERRARIUM-encoded PNGs (same encoding the live
                seafloor v1 layer consumes from AWS Terrain Tiles), pixel =
                area-mean of covered 15-arcsec cells (degenerates to the
                containing cell at native zoom). Out-of-coverage/nodata pixels
                encode elevation 0 -> the client depth ramp
                (client/src/lib/bathymetry.ts) renders them fully TRANSPARENT —
                missing data disappears, it is never painted as fake depth.
              - TID tiles: the RAW TID CODE terrarium-encoded as "elevation"
                (code c -> elevation c), NEAREST-sampled at every zoom (source
                codes are categorical: never averaged, never majority-voted —
                a coarser pixel shows one real cell's code, not an aggregate).
                Out-of-coverage encodes -1 (not a TID code) -> transparent under
                the client's step expression. Styling stays CLIENT-SIDE from
                datacore/gebco/tid_decode.json — these tiles carry data, not
                colors.
              Both are packed into PMTiles v3 archives (pure-python writer
              below; verified against the real `pmtiles` JS reader in
              server/seafloorTiles.test.ts) plus ONE provenance sidecar JSON
              carrying the verbatim GEBCO attribution, the true extent/
              resolution/zoom range, the sampling methods above, and the
              MEASURED per-group TID cell shares for the covered region (the
              honest "X% of this region is direct-measured" number — computed,
              never quoted).

HONESTY RAILS:
  - Every TID cell is validated against datacore/gebco/tid_decode.json (the
    verbatim official decode table). An unrecognized code ABORTS the build —
    a new GEBCO grid revision must update the decode table first, never get
    silently mis-tiled.
  - Coverage is whatever was actually fetched, recorded in the provenance
    extent. Partial coverage renders transparent, never interpolated shut.
  - maxzoom vs native resolution is recorded (15 arc-sec ≈ z9 at slight
    oversample; z8 is ~19.8 arc-sec/px). The provenance states which.
  - GEBCO terms: public domain, commercial use allowed, attribution REQUIRED,
    "should NOT be used for navigation" — attribution + disclaimer are written
    into the provenance and must be surfaced by any client wiring.

USAGE (session-side dev tool; Railway never runs this; stdlib only):
  python3 scripts/gebco_seafloor_tiles.py fetch --bbox 138 7 146 15 \
      --out /tmp/gebco_work
  python3 scripts/gebco_seafloor_tiles.py build --zip /tmp/gebco_work/basket.zip \
      --name mariana --maxzoom 9 --tilesdir client/public/tiles
  python3 scripts/gebco_seafloor_tiles.py run --bbox 138 7 146 15 \
      --name mariana --maxzoom 9 --tilesdir client/public/tiles \
      --workdir /tmp/gebco_work

FULL-RUN NOTES (documented, deliberately NOT built until needed):
  - The download app caps one basket item at 14400 deg^2 (a 120x120 deg box);
    global coverage = multiple fetch bboxes mosaicked per zoom, or the global
    NetCDF from CEDA (bathymetry 7.0 GB + TID 3.5 GB; netCDF4+numpy readers —
    session-local deps per the scripts/ convention).
  - A full-planet z0-9 terrarium set is tens of GB of PNG: ship as a
    boot-fetched volume asset (the power_us.pmtiles precedent), never a repo
    commit. Regional subsets (this pipeline's default) commit fine.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from array import array
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TID_DECODE_PATH = REPO_ROOT / "datacore" / "gebco" / "tid_decode.json"

DOWNLOAD_APP = "https://download.gebco.net"
TILE_SIZE = 256
# Terrarium: elevation = (R*256 + G + B/256) - 32768  (AWS Terrain Tiles docs)
TERRARIUM_OFFSET = 32768
# Out-of-coverage marker for TID tiles: NOT a TID code, decodes to -1 which the
# client step expression renders transparent (below every real code).
TID_NO_DATA = -1


# ── decode table ─────────────────────────────────────────────────────────────

def load_tid_decode(path: Path = TID_DECODE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    # sanity: the documented shape this pipeline depends on
    assert set(table["groups"].keys()) == {"land", "direct", "indirect", "unknown"}
    assert all(v["group"] in table["groups"] for v in table["codes"].values())
    return table


# ── Esri ASCII grid ──────────────────────────────────────────────────────────

@dataclass
class Grid:
    """Esri ASCII raster, row-major from the NORTH (first data row = top)."""
    ncols: int
    nrows: int
    xll: float
    yll: float
    cellsize: float
    nodata: float
    values: array  # array('d'), nrows*ncols, row-major from top
    _psum: array | None = field(default=None, repr=False)
    _pcnt: array | None = field(default=None, repr=False)

    @property
    def xmax(self) -> float:
        return self.xll + self.ncols * self.cellsize

    @property
    def ytop(self) -> float:
        return self.yll + self.nrows * self.cellsize

    def cell(self, row: int, col: int) -> float:
        return self.values[row * self.ncols + col]

    def col_of(self, lon: float) -> int:
        return int(math.floor((lon - self.xll) / self.cellsize))

    def row_of(self, lat: float) -> int:
        return int(math.floor((self.ytop - lat) / self.cellsize))

    def _build_prefix(self) -> None:
        """Summed-area tables (value + valid-count) for O(1) box means."""
        w, h = self.ncols, self.nrows
        psum = array("d", bytes(8 * (w + 1) * (h + 1)))
        pcnt = array("l", bytes(pcnt_itemsize() * (w + 1) * (h + 1)))
        stride = w + 1
        v = self.values
        nod = self.nodata
        for r in range(h):
            row_sum = 0.0
            row_cnt = 0
            base = r * w
            for c in range(w):
                x = v[base + c]
                if x != nod:
                    row_sum += x
                    row_cnt += 1
                idx = (r + 1) * stride + (c + 1)
                psum[idx] = psum[idx - stride] + row_sum
                pcnt[idx] = pcnt[idx - stride] + row_cnt
        self._psum, self._pcnt = psum, pcnt

    def box_mean(self, r0: int, r1: int, c0: int, c1: int) -> float | None:
        """Mean of valid cells in rows r0..r1, cols c0..c1 inclusive (clamped);
        None when the box holds no valid cell."""
        if self._psum is None:
            self._build_prefix()
        r0 = max(r0, 0)
        c0 = max(c0, 0)
        r1 = min(r1, self.nrows - 1)
        c1 = min(c1, self.ncols - 1)
        if r0 > r1 or c0 > c1:
            return None
        stride = self.ncols + 1
        ps, pc = self._psum, self._pcnt
        a = (r1 + 1) * stride + (c1 + 1)
        b = r0 * stride + (c1 + 1)
        c = (r1 + 1) * stride + c0
        d = r0 * stride + c0
        cnt = pc[a] - pc[b] - pc[c] + pc[d]
        if cnt <= 0:
            return None
        return (ps[a] - ps[b] - ps[c] + ps[d]) / cnt


def pcnt_itemsize() -> int:
    return array("l").itemsize


def parse_esri_ascii(text: str) -> Grid:
    """Parse an Esri ASCII raster (the GEBCO download-app subset format)."""
    lines = text.splitlines()
    hdr: dict[str, float] = {}
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 2 and parts[0].lower() in (
            "ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata_value"
        ):
            hdr[parts[0].lower()] = float(parts[1])
            i += 1
        else:
            break
    for req in ("ncols", "nrows", "xllcorner", "yllcorner", "cellsize"):
        if req not in hdr:
            raise ValueError(f"Esri ASCII header missing {req}")
    ncols, nrows = int(hdr["ncols"]), int(hdr["nrows"])
    values = array("d")
    for line in lines[i:]:
        if line.strip():
            values.extend(float(t) for t in line.split())
    if len(values) != ncols * nrows:
        raise ValueError(f"expected {ncols * nrows} cells, got {len(values)}")
    return Grid(
        ncols=ncols, nrows=nrows, xll=hdr["xllcorner"], yll=hdr["yllcorner"],
        cellsize=hdr["cellsize"], nodata=hdr.get("nodata_value", -32767.0),
        values=values,
    )


# ── web mercator ─────────────────────────────────────────────────────────────

MERC_MAX_LAT = 85.05112878


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(lon_w, lat_s, lon_e, lat_n) of web-mercator tile z/x/y."""
    n = 2 ** z
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon_w, lat_s, lon_e, lat_n


def lat_of_pixel_edge(z: int, y: int, j: float) -> float:
    """Latitude of the pixel edge j (0..TILE_SIZE) from the tile's top."""
    n = 2 ** z
    yy = (y + j / TILE_SIZE) / n
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy))))


def lon_of_pixel_edge(z: int, x: int, i: float) -> float:
    n = 2 ** z
    return (x + i / TILE_SIZE) / n * 360.0 - 180.0


def tiles_for_bbox(z: int, lon_w: float, lat_s: float, lon_e: float, lat_n: float):
    """All (x, y) tiles at zoom z intersecting the bbox."""
    n = 2 ** z
    lat_n = min(lat_n, MERC_MAX_LAT)
    lat_s = max(lat_s, -MERC_MAX_LAT)
    x0 = max(0, int(math.floor((lon_w + 180.0) / 360.0 * n)))
    x1 = min(n - 1, int(math.floor((lon_e + 180.0) / 360.0 * n - 1e-12)))

    def y_of(lat: float) -> float:
        return (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n

    y0 = max(0, int(math.floor(y_of(lat_n))))
    y1 = min(n - 1, int(math.floor(y_of(lat_s) - 1e-12)))
    return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


# ── terrarium encoding ───────────────────────────────────────────────────────

def terrarium_encode(elev_m: float) -> tuple[int, int, int]:
    """Terrarium RGB for an elevation in meters (1/256 m quantization)."""
    vi = int(round((elev_m + TERRARIUM_OFFSET) * 256.0))
    vi = max(0, min(vi, (1 << 24) - 1))
    return (vi >> 16) & 255, (vi >> 8) & 255, vi & 255


def terrarium_decode(r: int, g: int, b: int) -> float:
    return (r * 256 + g + b / 256.0) - TERRARIUM_OFFSET


# ── minimal PNG writer (stdlib) ──────────────────────────────────────────────

def png_encode_rgb(width: int, height: int, rgb_rows: list[bytes]) -> bytes:
    """8-bit RGB PNG, filter 0 on every scanline (kept simple + testable)."""
    if len(rgb_rows) != height or any(len(r) != width * 3 for r in rgb_rows):
        raise ValueError("bad scanline shape")
    raw = b"".join(b"\x00" + r for r in rgb_rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


# ── PMTiles v3 writer (pure python; spec: protomaps/PMTiles spec/v3) ─────────

def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def zxy_to_tileid(z: int, x: int, y: int) -> int:
    """PMTiles tile id: tiles-before-zoom count + Hilbert index at zoom z.
    Pinned against the real pmtiles JS lib in server/seafloorTiles.test.ts."""
    acc = (4 ** z - 1) // 3
    n = 2 ** z
    rx = ry = 0
    d = 0
    s = n // 2
    tx, ty = x, y
    while s > 0:
        rx = 1 if (tx & s) > 0 else 0
        ry = 1 if (ty & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # rotate
        if ry == 0:
            if rx == 1:
                tx = s - 1 - tx
                ty = s - 1 - ty
            tx, ty = ty, tx
        s //= 2
    return acc + d


def serialize_directory(entries: list[tuple[int, int, int, int]]) -> bytes:
    """entries: (tile_id, offset, length, run_length), sorted by tile_id."""
    out = bytearray(varint(len(entries)))
    last_id = 0
    for i, (tid, _off, _ln, _rl) in enumerate(entries):
        out += varint(tid - last_id if i else tid)
        last_id = tid
    for _tid, _off, _ln, rl in entries:
        out += varint(rl)
    for _tid, _off, ln, _rl in entries:
        out += varint(ln)
    for _tid, off, _ln, _rl in entries:
        out += varint(off + 1)  # spec allows offset+1 always
    return bytes(out)


def build_pmtiles(
    tiles: dict[int, bytes],
    metadata: dict,
    min_zoom: int,
    max_zoom: int,
    bounds: tuple[float, float, float, float],
    tile_type: int = 2,  # png
) -> bytes:
    """Single-root-directory PMTiles v3 archive (no leaves; fine at our scale —
    the root must stay within the spec's 16KB prefetch guidance, asserted)."""
    ids = sorted(tiles.keys())
    data = bytearray()
    entries = []
    for tid in ids:
        blob = tiles[tid]
        entries.append((tid, len(data), len(blob), 1))
        data += blob
    root = gzip.compress(serialize_directory(entries), mtime=0)
    if len(root) > 16384 - 127:
        raise ValueError(
            f"root directory {len(root)}B exceeds the 16KB prefetch budget — "
            "add leaf-directory support before tiling regions this large"
        )
    meta = gzip.compress(json.dumps(metadata).encode("utf-8"), mtime=0)

    root_off = 127
    meta_off = root_off + len(root)
    leaf_off = meta_off + len(meta)
    tile_off = leaf_off  # no leaves
    lon_w, lat_s, lon_e, lat_n = bounds

    def e7(v: float) -> int:
        return int(round(v * 1e7))

    hdr = bytearray()
    hdr += b"PMTiles"
    hdr += bytes([3])
    hdr += struct.pack("<Q", root_off)
    hdr += struct.pack("<Q", len(root))
    hdr += struct.pack("<Q", meta_off)
    hdr += struct.pack("<Q", len(meta))
    hdr += struct.pack("<Q", leaf_off)
    hdr += struct.pack("<Q", 0)
    hdr += struct.pack("<Q", tile_off)
    hdr += struct.pack("<Q", len(data))
    hdr += struct.pack("<Q", len(ids))  # addressed tiles
    hdr += struct.pack("<Q", len(ids))  # tile entries
    hdr += struct.pack("<Q", len(ids))  # tile contents
    hdr += bytes([1])                   # clustered
    hdr += bytes([2])                   # internal compression: gzip
    hdr += bytes([1])                   # tile compression: none (PNG)
    hdr += bytes([tile_type])
    hdr += bytes([min_zoom])
    hdr += bytes([max_zoom])
    hdr += struct.pack("<i", e7(lon_w))
    hdr += struct.pack("<i", e7(lat_s))
    hdr += struct.pack("<i", e7(lon_e))
    hdr += struct.pack("<i", e7(lat_n))
    hdr += bytes([(min_zoom + max_zoom) // 2])
    hdr += struct.pack("<i", e7((lon_w + lon_e) / 2))
    hdr += struct.pack("<i", e7((lat_s + lat_n) / 2))
    assert len(hdr) == 127
    return bytes(hdr) + root + meta + bytes(data)


# ── tile rendering ───────────────────────────────────────────────────────────

def render_depth_tile(grid: Grid, z: int, x: int, y: int) -> bytes | None:
    """Terrarium PNG for one tile; None when no pixel overlaps coverage.
    Pixel = area-mean of covered cells (nearest at native zoom); pixels with
    no valid cell encode elevation 0 -> transparent under the depth ramp."""
    rows: list[bytes] = []
    any_data = False
    zero = terrarium_encode(0.0)
    for j in range(TILE_SIZE):
        lat_hi = lat_of_pixel_edge(z, y, j)
        lat_lo = lat_of_pixel_edge(z, y, j + 1)
        r0 = grid.row_of(lat_hi)
        r1 = grid.row_of(lat_lo)
        if r1 < r0:
            r0, r1 = r1, r0
        line = bytearray()
        for i in range(TILE_SIZE):
            lon_w = lon_of_pixel_edge(z, x, i)
            lon_e = lon_of_pixel_edge(z, x, i + 1)
            c0 = grid.col_of(lon_w)
            c1 = grid.col_of(lon_e - 1e-12)
            m = grid.box_mean(r0, r1, c0, c1)
            if m is None:
                line += bytes(zero)
            else:
                any_data = True
                line += bytes(terrarium_encode(m))
        rows.append(bytes(line))
    if not any_data:
        return None
    return png_encode_rgb(TILE_SIZE, TILE_SIZE, rows)


def render_tid_tile(
    grid: Grid, z: int, x: int, y: int, valid_codes: set[int], nodata_code: int
) -> bytes | None:
    """Terrarium PNG whose 'elevation' is the RAW TID code, NEAREST-sampled
    (categorical: never averaged). Out-of-coverage -> TID_NO_DATA (-1).
    Any cell code outside the decode table aborts (honesty ratchet)."""
    rows: list[bytes] = []
    any_data = False
    nod = terrarium_encode(TID_NO_DATA)
    for j in range(TILE_SIZE):
        lat_c = lat_of_pixel_edge(z, y, j + 0.5)
        r = grid.row_of(lat_c)
        line = bytearray()
        in_row = 0 <= r < grid.nrows
        for i in range(TILE_SIZE):
            lon_c = lon_of_pixel_edge(z, x, i + 0.5)
            c = grid.col_of(lon_c)
            if not in_row or not (0 <= c < grid.ncols):
                line += bytes(nod)
                continue
            code = int(grid.cell(r, c))
            if code == nodata_code:
                line += bytes(nod)
                continue
            if code not in valid_codes:
                raise ValueError(
                    f"TID code {code} at cell ({r},{c}) is not in "
                    f"datacore/gebco/tid_decode.json — update the decode table "
                    f"from the official GEBCO documentation before tiling"
                )
            any_data = True
            line += bytes(terrarium_encode(float(code)))
        rows.append(bytes(line))
    if not any_data:
        return None
    return png_encode_rgb(TILE_SIZE, TILE_SIZE, rows)


# ── download-app client ──────────────────────────────────────────────────────

def basket_body(
    lon_w: float, lat_s: float, lon_e: float, lat_n: float,
    grid_id: int = 1, data_source_ids: tuple[int, ...] = (1, 3),
    format_ids: tuple[int, ...] = (3,),
) -> dict:
    """POST /api/queue body (contract read from the app's own JS 2026-07-16).
    Defaults: current global grid (grid_id 1), Bathymetry (1) + TID (3),
    Esri ASCII (format 3)."""
    return {
        "id": "0",
        "email": None,
        "submission_date": time.strftime("%Y-%m-%dT%H:%M:%S.000000", time.gmtime()),
        "processing_status": "new",
        "items": [{
            "id": 0,
            "grid_id": grid_id,
            "data_source_ids": list(data_source_ids),
            "formats": list(format_ids),
            "left": lon_w, "right": lon_e, "top": lat_n, "bottom": lat_s,
        }],
    }


def fetch_basket(bbox: tuple[float, float, float, float], out_zip: Path,
                 poll_s: float = 5.0, timeout_s: float = 900.0) -> Path:
    lon_w, lat_s, lon_e, lat_n = bbox
    body = json.dumps(basket_body(lon_w, lat_s, lon_e, lat_n)).encode()
    req = urllib.request.Request(
        f"{DOWNLOAD_APP}/api/queue", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        basket_id = json.loads(r.read())["basketId"]
    print(f"basket {basket_id} queued; polling…")
    t0 = time.time()
    while True:
        with urllib.request.urlopen(
            f"{DOWNLOAD_APP}/api/queue/status/{basket_id}", timeout=60
        ) as r:
            status = json.loads(r.read()).get("status")
        if status == "finished":
            break
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"basket {basket_id} still '{status}' after {timeout_s}s")
        time.sleep(poll_s)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        f"{DOWNLOAD_APP}/api/queue/download/{basket_id}", timeout=600
    ) as r, open(out_zip, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"downloaded {out_zip} ({out_zip.stat().st_size} bytes)")
    return out_zip


# ── build orchestration ──────────────────────────────────────────────────────

def grids_from_zip(zip_path: Path) -> tuple[Grid, Grid]:
    """(bathymetry, tid) grids from a download-app basket zip."""
    z = zipfile.ZipFile(zip_path)
    bathy_name = tid_name = None
    for n in z.namelist():
        if not n.endswith(".asc"):
            continue
        if "_tid_" in n:
            tid_name = n
        else:
            bathy_name = n
    if not bathy_name or not tid_name:
        raise ValueError(
            f"basket zip must hold one bathymetry and one TID .asc "
            f"(got {z.namelist()})"
        )
    bathy = parse_esri_ascii(z.read(bathy_name).decode("ascii"))
    tid = parse_esri_ascii(z.read(tid_name).decode("ascii"))
    if (bathy.ncols, bathy.nrows) != (tid.ncols, tid.nrows):
        raise ValueError("bathymetry and TID subsets have different shapes")
    return bathy, tid


def tid_histogram(grid: Grid, valid_codes: set[int], nodata_code: int) -> dict[int, int]:
    hist: dict[int, int] = {}
    for v in grid.values:
        code = int(v)
        if code == nodata_code:
            continue
        if code not in valid_codes:
            raise ValueError(f"TID code {code} not in the decode table")
        hist[code] = hist.get(code, 0) + 1
    return hist


def build_assets(zip_path: Path, name: str, maxzoom: int, tiles_dir: Path) -> dict:
    decode = load_tid_decode()
    valid_codes = {int(k) for k in decode["codes"]}
    nodata_code = int(decode["nodataValue"])
    bathy, tid = grids_from_zip(zip_path)
    bounds = (bathy.xll, bathy.yll, bathy.xmax, bathy.ytop)
    native_arcsec = bathy.cellsize * 3600.0
    print(f"coverage {bounds}, {bathy.ncols}x{bathy.nrows} cells, "
          f"{native_arcsec:.1f} arc-sec")

    hist = tid_histogram(tid, valid_codes, nodata_code)
    total = sum(hist.values())
    group_of = {int(k): v["group"] for k, v in decode["codes"].items()}
    group_cells: dict[str, int] = {}
    for code, cnt in hist.items():
        g = group_of[code]
        group_cells[g] = group_cells.get(g, 0) + cnt
    group_share = {g: round(c / total, 4) for g, c in sorted(group_cells.items())}
    print(f"TID groups (measured shares of covered cells): {group_share}")

    depth_tiles: dict[int, bytes] = {}
    tid_tiles: dict[int, bytes] = {}
    for z in range(0, maxzoom + 1):
        n_z = 0
        for x, y in tiles_for_bbox(z, *bounds):
            png = render_depth_tile(bathy, z, x, y)
            if png is not None:
                depth_tiles[zxy_to_tileid(z, x, y)] = png
            png = render_tid_tile(tid, z, x, y, valid_codes, nodata_code)
            if png is not None:
                tid_tiles[zxy_to_tileid(z, x, y)] = png
                n_z += 1
        print(f"  z{z}: {n_z} tiles")

    attribution = decode["source"]["attribution"]
    common_meta = {
        "attribution": attribution,
        "bounds": list(bounds),
        "native_resolution_arcsec": round(native_arcsec, 3),
        "disclaimer": "Indicative only — NOT for navigation (GEBCO terms of use).",
    }
    dem_pm = build_pmtiles(
        depth_tiles,
        {**common_meta, "name": f"seafloor_gebco_{name}",
         "description": "GEBCO bathymetry, terrarium-encoded elevation (m)",
         "encoding": "terrarium"},
        0, maxzoom, bounds,
    )
    tid_pm = build_pmtiles(
        tid_tiles,
        {**common_meta, "name": f"seafloor_tid_{name}",
         "description": "GEBCO TID source-type code terrarium-encoded as "
                        "elevation; decode via datacore/gebco/tid_decode.json",
         "encoding": "terrarium-tid-code"},
        0, maxzoom, bounds,
    )

    tiles_dir.mkdir(parents=True, exist_ok=True)
    dem_path = tiles_dir / f"seafloor_gebco_{name}.pmtiles"
    tid_path = tiles_dir / f"seafloor_tid_{name}.pmtiles"
    dem_path.write_bytes(dem_pm)
    tid_path.write_bytes(tid_pm)

    provenance = {
        "_doc": "Provenance for the GEBCO v2 seafloor assets (EARTH TWIN E2 v2)."
                " Written by scripts/gebco_seafloor_tiles.py — regenerate, never"
                " hand-edit. Any client wiring MUST surface attribution and the"
                " not-for-navigation disclaimer.",
        "dataset": decode["dataset"],
        "attribution": attribution,
        "license": decode["source"]["termsOfUse"],
        "documentationUrl": decode["source"]["documentationUrl"],
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_service": f"{DOWNLOAD_APP} grid-subsetting API (Esri ASCII subset)",
        "coverage": {
            "bbox": {"west": bounds[0], "south": bounds[1],
                     "east": bounds[2], "north": bounds[3]},
            "cells": [bathy.ncols, bathy.nrows],
            "native_resolution_arcsec": round(native_arcsec, 3),
            "note": "REGIONAL subset — everywhere outside this bbox has NO v2 "
                    "data and renders transparent. Honest partial coverage, "
                    "never fabricated global coverage.",
        },
        "tiling": {
            "scheme": "web-mercator 256px PMTiles v3, zooms 0..%d" % maxzoom,
            "maxzoom_vs_native": (
                f"z{maxzoom} ≈ {360 / (TILE_SIZE * 2 ** maxzoom) * 3600:.1f} "
                f"arc-sec/px vs {native_arcsec:.1f} arc-sec native"
            ),
            "depth_sampling": "area-mean of covered cells per pixel "
                              "(degenerates to containing cell at native zoom); "
                              "no-coverage pixels encode elevation 0 (renders "
                              "transparent under the depth ramp)",
            "tid_sampling": "nearest cell (categorical — never averaged); "
                            "no-coverage pixels encode -1 (renders transparent)",
        },
        "files": {
            "bathymetry": dem_path.name,
            "tid": tid_path.name,
        },
        "tid": {
            "decode_table": "datacore/gebco/tid_decode.json (verbatim official)",
            "histogram_by_code": {str(k): v for k, v in sorted(hist.items())},
            "group_share_of_covered_cells": group_share,
        },
        "notForNavigation": True,
    }
    prov_path = tiles_dir / f"seafloor_gebco_{name}.provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dem_path} ({dem_path.stat().st_size:,}B), "
          f"{tid_path} ({tid_path.stat().st_size:,}B), {prov_path.name}")
    return provenance


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_bbox(p):
        p.add_argument("--bbox", nargs=4, type=float, required=True,
                       metavar=("W", "S", "E", "N"))

    f = sub.add_parser("fetch", help="submit + download a GEBCO basket zip")
    add_bbox(f)
    f.add_argument("--out", type=Path, required=True, help="work directory")

    b = sub.add_parser("build", help="tile a downloaded basket zip into PMTiles")
    b.add_argument("--zip", type=Path, required=True)
    b.add_argument("--name", required=True, help="asset suffix, e.g. mariana")
    b.add_argument("--maxzoom", type=int, default=9)
    b.add_argument("--tilesdir", type=Path,
                   default=REPO_ROOT / "client" / "public" / "tiles")

    r = sub.add_parser("run", help="fetch + build in one go")
    add_bbox(r)
    r.add_argument("--name", required=True)
    r.add_argument("--maxzoom", type=int, default=9)
    r.add_argument("--tilesdir", type=Path,
                   default=REPO_ROOT / "client" / "public" / "tiles")
    r.add_argument("--workdir", type=Path, required=True)

    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch_basket(tuple(a.bbox), a.out / "basket.zip")
    elif a.cmd == "build":
        build_assets(a.zip, a.name, a.maxzoom, a.tilesdir)
    elif a.cmd == "run":
        zip_path = fetch_basket(tuple(a.bbox), a.workdir / "basket.zip")
        build_assets(zip_path, a.name, a.maxzoom, a.tilesdir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
