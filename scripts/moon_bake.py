#!/usr/bin/env python3
"""moon_bake.py — bake OUR OWN Moon (LROC WAC) tile pyramid onto OUR CDN.

WHY THIS EXISTS (Rendering & Motion Law II.8, CLAUDE.md Amendment 6):
  "Runtime never touches an upstream WMTS. All tiles come from our CDN.
   Upstream is a bake-time input only."

Today `client/src/lib/celestial/lroc.ts` points the browser straight at
NASA Trek's WMTS (`trek.nasa.gov/tiles/Moon/EQ/...`) on every close Moon
frame. That is a live runtime dependency on someone else's service: we
cannot cache it, cannot control its latency, cannot survive its outage, and
cannot pre-warm it. This script removes that dependency by baking the same
imagery from the ORIGINAL USGS source into our R2 bucket, in the EXACT tile
scheme lroc.ts already implements and tests — so switching the runtime over
is a `baseUrl` change, not a rewrite.

SOURCE (probed live 2026-08-13, evidence in research/rendering_motion_overhaul.md):
  url       https://planetarymaps.usgs.gov/mosaic/
            Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif
            (302 -> asc-pds-services.s3.us-west-2.amazonaws.com, Accept-Ranges: bytes)
  bytes     5,959,263,751  (5.55 GiB)
  raster    109164 x 54582, Byte, 1 band (Gray), UNCOMPRESSED
  blocks    109164 x 1  <-- ONE-SCANLINE STRIPS, and NO overviews. Windowed
            reads are therefore useless (any window costs full scanlines);
            only a sequential full pass is efficient. This is why the script
            downloads once and then re-reads locally, instead of using
            /vsicurl per level.
  crs       SimpleCylindrical MOON / plate carree, lon [-180,180], lat [-90,90]
  scale     100 m/px  (= 109164 px / 360 deg = 303.23 px/deg)
  licence   USGS/NASA LRO LROC WAC global mosaic — PUBLIC DOMAIN.

  Note the equivalence: Trek serves "LRO_WAC_Mosaic_Global_303ppd_v02", and
  303 ppd x 360 deg = 109,080 px — the SAME 100 m/px product this file is.
  Trek's deepest level (z8 = 131072 px wide) is therefore already an
  UPSAMPLE of this source, not extra real detail. Our native ceiling is z7
  (65536 px wide, a genuine downsample); z8 is offered only to match Trek's
  behaviour and is honestly recorded as interpolated in the manifest.

TILE SCHEME — deliberately IDENTICAL to NASA Trek EQ (lroc.ts):
  level z has MatrixWidth = 2^(z+1) cols, MatrixHeight = 2^z rows
  tiles are 256 x 256, grayscale JPEG, TopLeftCorner = (-180, +90)
  key order is /{z}/{y}/{x}.jpg  — ROW BEFORE COLUMN (WMTS TileRow/TileCol)
  a tile spans 180/2^z degrees in BOTH axes (square in degrees)

  Matching the scheme exactly is the whole design: lroc.ts's matrixWidth /
  degPerTile / tilesForBbox / tileUrl math is already written and tested
  against this convention, and the four sibling Trek bodies (Mars, Mercury,
  Venus, Ceres) share it. Inventing our own scheme would have forced a
  rewrite of tested math for zero gain.

HONESTY RAILS:
  - Nothing is invented. Every output pixel is an AREA AVERAGE (gdal -r
    average) of real source pixels; no interpolation beyond that, no
    inpainting, no synthetic poles.
  - Source nodata is 0 (real WAC illumination gaps near the poles). Those
    stay black. A gap is shown as a gap.
  - The manifest records which levels are true downsamples of the 100 m/px
    source and which (z8) are upsampled, so the client can never claim
    resolution the data does not have.

USAGE (session-side dev tool; Railway never runs this):
  python3 scripts/moon_bake.py plan --max-z 5
  python3 scripts/moon_bake.py fetch  --work /tmp/moon
  python3 scripts/moon_bake.py build  --work /tmp/moon --max-z 5
  python3 scripts/moon_bake.py upload --work /tmp/moon --max-z 5
  python3 scripts/moon_bake.py run    --work /tmp/moon --max-z 5     # all three

  --dry-run is honoured by upload (prints keys, transfers nothing).

COST: $0. This is CPU + bandwidth only and runs in-session. RunPod is NOT
needed for the pilot (levels 0-5). Per CLAUDE.md's RunPod section the pod is
for batch bakes when the local box is the bottleneck — for the full z0-7 set
(43,690 tiles) measure first, then decide; the cost-cap gate in
scripts/runpod_budget.py governs any pod launch.

DEPENDENCIES: gdal_translate (CLI), numpy, Pillow, boto3. No osgeo Python
bindings required — levels are written as flat ENVI rasters and read with
numpy.memmap, which keeps memory flat regardless of level size.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import io
import json
import os
import subprocess
import sys
import time
from typing import Iterator

# ── constants (the scheme; these are the contract with lroc.ts) ─────────────

SRC_URL = (
    "https://planetarymaps.usgs.gov/mosaic/"
    "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
)
SRC_BYTES = 5_959_263_751
SRC_W = 109164
SRC_H = 54582

TILE_PX = 256
#: deepest level that is a genuine DOWNSAMPLE of the 100 m/px source.
NATIVE_MAX_Z = 7
#: deepest level Trek itself serves (z8 is upsampled — offered for parity only).
TREK_MAX_Z = 8

R2_PREFIX = "moon/wac"
JPEG_QUALITY = 88

LICENSE = "Public Domain (USGS/NASA, LRO LROC WAC global mosaic, 100 m/px, June 2013)"
CREDIT = "Moon: NASA LRO · LROC WAC 100 m/px · USGS Astrogeology"


# ── pure tile math (mirrors lroc.ts — parity pinned by test_moon_bake.py) ───


def matrix_size(z: int) -> tuple[int, int]:
    """(cols, rows) at level z. Trek EQ: 2^(z+1) x 2^z."""
    if z < 0:
        raise ValueError(f"level must be >= 0, got {z}")
    return (1 << (z + 1), 1 << z)


def level_px(z: int, tile_px: int = TILE_PX) -> tuple[int, int]:
    """Full raster size of level z, in pixels."""
    cols, rows = matrix_size(z)
    return (cols * tile_px, rows * tile_px)


def deg_per_tile(z: int) -> float:
    """A tile spans this many degrees in BOTH axes (square in degrees)."""
    return 180.0 / (1 << z)


def tile_bounds_deg(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """(lon_min, lat_max, lon_max, lat_min) for tile (z,x,y). y=0 is NORTH."""
    d = deg_per_tile(z)
    return (-180.0 + x * d, 90.0 - y * d, -180.0 + (x + 1) * d, 90.0 - (y + 1) * d)


def tile_key(z: int, x: int, y: int, prefix: str = R2_PREFIX) -> str:
    """Object key. ROW BEFORE COLUMN — matches lroc.ts tileUrl()."""
    return f"{prefix}/{z}/{y}/{x}.jpg"


def tiles_at(z: int) -> int:
    cols, rows = matrix_size(z)
    return cols * rows


def total_tiles(max_z: int, min_z: int = 0) -> int:
    return sum(tiles_at(z) for z in range(min_z, max_z + 1))


def iter_tiles(z: int) -> Iterator[tuple[int, int]]:
    cols, rows = matrix_size(z)
    for y in range(rows):
        for x in range(cols):
            yield (x, y)


def native_ceiling_z(src_w: int = SRC_W, tile_px: int = TILE_PX) -> int:
    """Deepest level whose width does not exceed the source's — i.e. the last
    level that is a real downsample rather than an upsample."""
    z = 0
    while level_px(z + 1, tile_px)[0] <= src_w:
        z += 1
    return z


def is_upsampled(z: int, src_w: int = SRC_W, tile_px: int = TILE_PX) -> bool:
    return level_px(z, tile_px)[0] > src_w


def plan_levels(max_z: int, min_z: int = 0) -> list[dict]:
    out = []
    for z in range(min_z, max_z + 1):
        cols, rows = matrix_size(z)
        w, h = level_px(z)
        out.append(
            {
                "z": z,
                "cols": cols,
                "rows": rows,
                "tiles": cols * rows,
                "px_w": w,
                "px_h": h,
                "m_per_px": round(2 * 3.14159265358979 * 1737400 / w, 1),
                "upsampled": is_upsampled(z),
            }
        )
    return out


def manifest(max_z: int, min_z: int = 0, prefix: str = R2_PREFIX) -> dict:
    """The sidecar the client reads to learn what actually exists."""
    levels = plan_levels(max_z, min_z)
    return {
        "body": "moon",
        "product": "LRO_LROC_WAC_Mosaic_Global_100m_June2013",
        "scheme": {
            "kind": "trek-eq",
            "note": "MatrixWidth=2^(z+1), MatrixHeight=2^z, plate carree, "
            "TopLeftCorner=(-180,+90), key order {z}/{y}/{x}",
            "tile_px": TILE_PX,
            "ext": "jpg",
            "min_z": min_z,
            "max_z": max_z,
            "native_max_z": native_ceiling_z(),
        },
        "source": {
            "url": SRC_URL,
            "bytes": SRC_BYTES,
            "raster": [SRC_W, SRC_H],
            "m_per_px": 100.0,
            "license": LICENSE,
        },
        "credit": CREDIT,
        "resampling": "area average (gdal -r average); no interpolation, no inpainting",
        "nodata": "source nodata 0 (real WAC polar illumination gaps) is preserved as black",
        "levels": levels,
        "tiles_total": total_tiles(max_z, min_z),
    }


# ── stages ─────────────────────────────────────────────────────────────────


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[-2000:]}")


def src_path(work: str) -> str:
    return os.path.join(work, "wac_global_100m.tif")


def stage_fetch(work: str) -> str:
    """Download the source once, resumable. Verifies the final size."""
    os.makedirs(work, exist_ok=True)
    dst = src_path(work)
    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    if have == SRC_BYTES:
        print(f"  source already complete ({have:,} bytes)")
        return dst
    print(f"  fetching {SRC_URL}\n  -> {dst} (resume from {have:,})")
    _run(["curl", "-sS", "-L", "-C", "-", "--retry", "5", "--retry-delay", "3", "-o", dst, SRC_URL])
    got = os.path.getsize(dst)
    if got != SRC_BYTES:
        raise RuntimeError(f"source size {got:,} != expected {SRC_BYTES:,}")
    print(f"  ok ({got:,} bytes)")
    return dst


def level_raw_path(work: str, z: int) -> str:
    return os.path.join(work, f"level_{z}.img")


def stage_level(work: str, z: int, src: str) -> str:
    """Render level z as a flat ENVI grayscale raster (area-averaged).

    Each level is produced DIRECTLY from the source rather than by halving the
    level above it. That costs one extra sequential pass per level but makes
    every level independently correct — a bug at z5 cannot silently poison z4.
    GDAL streams this out-of-core, so memory stays flat.
    """
    w, h = level_px(z)
    out = level_raw_path(work, z)
    if os.path.exists(out) and os.path.getsize(out) == w * h:
        print(f"    z{z}: level raster cached ({w}x{h})")
        return out
    t0 = time.time()
    _run(
        [
            "gdal_translate", "-q",
            "-of", "ENVI",
            "-ot", "Byte",
            "-b", "1",
            "-r", "average",
            "-outsize", str(w), str(h),
            src, out,
        ]
    )
    got = os.path.getsize(out)
    if got != w * h:
        raise RuntimeError(f"z{z}: level raster is {got:,} bytes, expected {w*h:,}")
    print(f"    z{z}: {w}x{h} raster in {time.time()-t0:.1f}s")
    return out


def _encode_row(args) -> tuple[int, int, int]:
    """Cut and JPEG-encode one tile ROW. Returns (y, tiles, bytes)."""
    import numpy as np
    from PIL import Image

    work, z, y, cols, w, h, outdir, quality = args
    arr = np.memmap(level_raw_path(work, z), dtype=np.uint8, mode="r", shape=(h, w))
    band = arr[y * TILE_PX : (y + 1) * TILE_PX, :]
    n = 0
    nbytes = 0
    rowdir = os.path.join(outdir, str(z), str(y))
    os.makedirs(rowdir, exist_ok=True)
    for x in range(cols):
        tile = np.ascontiguousarray(band[:, x * TILE_PX : (x + 1) * TILE_PX])
        buf = io.BytesIO()
        Image.fromarray(tile, mode="L").save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        with open(os.path.join(rowdir, f"{x}.jpg"), "wb") as fh:
            fh.write(data)
        n += 1
        nbytes += len(data)
    del arr
    return (y, n, nbytes)


def stage_tiles(work: str, max_z: int, min_z: int, jobs: int, quality: int) -> dict:
    """Cut every level into 256px grayscale JPEG tiles under work/tiles/."""
    outdir = os.path.join(work, "tiles")
    os.makedirs(outdir, exist_ok=True)
    src = src_path(work)
    if not os.path.exists(src):
        raise SystemExit("source missing — run `fetch` first")
    totals = {"tiles": 0, "bytes": 0}
    for z in range(min_z, max_z + 1):
        stage_level(work, z, src)
        cols, rows = matrix_size(z)
        w, h = level_px(z)
        t0 = time.time()
        work_items = [(work, z, y, cols, w, h, outdir, quality) for y in range(rows)]
        zt = zb = 0
        with futures.ProcessPoolExecutor(max_workers=jobs) as ex:
            for _y, n, nb in ex.map(_encode_row, work_items):
                zt += n
                zb += nb
        totals["tiles"] += zt
        totals["bytes"] += zb
        print(
            f"    z{z}: {zt:,} tiles, {zb/1e6:.1f} MB "
            f"({zb/max(1,zt)/1024:.1f} KB avg) in {time.time()-t0:.1f}s"
        )
    man = manifest(max_z, min_z)
    man["baked_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    man["bytes_total"] = totals["bytes"]
    with open(os.path.join(outdir, "tiles.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    print(f"  wrote {outdir}/tiles.json")
    return totals


def _r2_client():
    import boto3
    from botocore.config import Config

    ca = "/root/.ccr/ca-bundle.crt"
    if os.path.exists(ca):
        os.environ.setdefault("AWS_CA_BUNDLE", ca)
    ep = (os.environ.get("R2_ENDPOINT") or "").strip().rstrip("/")
    if ep and not ep.startswith("http"):
        ep = f"https://{ep}.r2.cloudflarestorage.com"
    ak = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY")
    sk = os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET")
    if not (ep and ak and sk):
        raise SystemExit(
            "R2 credentials incomplete — need R2_ENDPOINT, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY. Run `python3 scripts/r2_verify.py` to diagnose."
        )
    return boto3.client(
        "s3", endpoint_url=ep, aws_access_key_id=ak, aws_secret_access_key=sk,
        region_name="auto", config=Config(retries={"max_attempts": 3}, max_pool_connections=32),
    )


def stage_upload(work: str, max_z: int, min_z: int, bucket: str, prefix: str,
                 jobs: int, dry: bool) -> int:
    outdir = os.path.join(work, "tiles")
    files: list[tuple[str, str]] = []
    for z in range(min_z, max_z + 1):
        cols, rows = matrix_size(z)
        for y in range(rows):
            for x in range(cols):
                p = os.path.join(outdir, str(z), str(y), f"{x}.jpg")
                if os.path.exists(p):
                    files.append((p, tile_key(z, x, y, prefix)))
    mp = os.path.join(outdir, "tiles.json")
    if os.path.exists(mp):
        files.append((mp, f"{prefix}/tiles.json"))

    print(f"  {len(files):,} objects -> s3://{bucket}/{prefix}/")
    if dry:
        for p, k in files[:5]:
            print(f"    (dry) {k}")
        print(f"    (dry) ... {len(files):,} total, nothing transferred")
        return len(files)

    s3 = _r2_client()
    done = [0]
    t0 = time.time()

    def put(item: tuple[str, str]) -> None:
        p, k = item
        ctype = "application/json" if k.endswith(".json") else "image/jpeg"
        with open(p, "rb") as fh:
            s3.put_object(
                Bucket=bucket, Key=k, Body=fh.read(), ContentType=ctype,
                CacheControl="public, max-age=31536000, immutable",
            )
        done[0] += 1
        if done[0] % 250 == 0:
            print(f"    {done[0]:,}/{len(files):,} ({time.time()-t0:.0f}s)")

    with futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(put, files))
    print(f"  uploaded {done[0]:,} objects in {time.time()-t0:.0f}s")
    return done[0]


# ── cli ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["plan", "fetch", "build", "upload", "run"])
    ap.add_argument("--work", default="/tmp/moon_bake")
    ap.add_argument("--max-z", type=int, default=5)
    ap.add_argument("--min-z", type=int, default=0)
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", "voltrade-tiles"))
    ap.add_argument("--prefix", default=R2_PREFIX)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--quality", type=int, default=JPEG_QUALITY)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.max_z > TREK_MAX_Z:
        raise SystemExit(f"--max-z {a.max_z} exceeds the Trek scheme ceiling z{TREK_MAX_Z}")

    if a.stage == "plan":
        man = manifest(a.max_z, a.min_z, a.prefix)
        print(f"MOON BAKE PLAN  z{a.min_z}..z{a.max_z}   (native ceiling z{native_ceiling_z()})")
        print(f"  {'z':>3} {'grid':>12} {'tiles':>9} {'raster':>15} {'m/px':>9}  note")
        for L in man["levels"]:
            note = "UPSAMPLED (not real detail)" if L["upsampled"] else ""
            print(
                f"  {L['z']:>3} {L['cols']:>5}x{L['rows']:<6} {L['tiles']:>9,} "
                f"{L['px_w']:>7}x{L['px_h']:<7} {L['m_per_px']:>9,.0f}  {note}"
            )
        print(f"  TOTAL {man['tiles_total']:,} tiles")
        print(f"  keys  {tile_key(a.max_z, 0, 0, a.prefix)} … (row before column)")
        return 0

    os.makedirs(a.work, exist_ok=True)

    if a.stage in ("fetch", "run"):
        print("── fetch ──")
        stage_fetch(a.work)
    if a.stage in ("build", "run"):
        print("── build ──")
        stage_tiles(a.work, a.max_z, a.min_z, a.jobs, a.quality)
    if a.stage in ("upload", "run"):
        print("── upload ──")
        stage_upload(a.work, a.max_z, a.min_z, a.bucket, a.prefix, min(a.jobs * 4, 32), a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
