"""test_gebco_seafloor_tiles.py — GEBCO v2 seafloor pipeline battery
(scripts/gebco_seafloor_tiles.py, EARTH TWIN E2 v2). NO NETWORK: every grid
here is a tiny SYNTHETIC FIXTURE (clearly labeled, test-only, never shipped as
data); the Esri ASCII header shape is VERBATIM from a real download-app basket
fetched 2026-07-16 (gebco_2026_n12.0_s11.0_w142.0_e143.0_ascii.asc). The
committed real assets are separately ratcheted by server/seafloorTiles.test.ts
through the real pmtiles JS reader.
"""
import gzip
import json
import os
import struct
import sys
import zlib
import importlib.util

import pytest

_spec = importlib.util.spec_from_file_location(
    "gebco_seafloor_tiles",
    os.path.join(os.path.dirname(__file__), "scripts", "gebco_seafloor_tiles.py"))
gst = importlib.util.module_from_spec(_spec)
sys.modules["gebco_seafloor_tiles"] = gst  # dataclasses need the module registered
_spec.loader.exec_module(gst)


# ── fixtures (synthetic, test-only — NEVER shipped as data) ──────────────────

# Header lines VERBATIM in shape from the real 2026-07-16 basket subset;
# the 3x3 value block is synthetic.
ASCII_FIXTURE = """ncols        3
nrows        3
xllcorner    142.000000000000
yllcorner    11.000000000000
cellsize     1.000000000000
NODATA_value -32767
-4497 -4480 -4466
-10931 -32767 -2063
-5000 -6000 -7000
"""

# Synthetic TID grid: one code from each documented group + the documented
# nodata value 127 (codes themselves are from the official decode table).
TID_FIXTURE = """ncols        3
nrows        3
xllcorner    142.000000000000
yllcorner    11.000000000000
cellsize     1.000000000000
NODATA_value 127
11 40 71
0 127 11
40 40 40
"""


def _grid(text=ASCII_FIXTURE):
    return gst.parse_esri_ascii(text)


# ── decode table (the shipped datacore/gebco/tid_decode.json) ────────────────

def test_decode_table_is_the_documented_set_and_nothing_else():
    t = gst.load_tid_decode()
    codes = sorted(int(k) for k in t["codes"])
    # The EXACT GEBCO_2026 documented code set — never collapsed, never invented
    assert codes == [0] + list(range(10, 18)) + list(range(40, 49)) + [70, 71, 72]
    assert int(t["nodataValue"]) == 127
    groups = {v["group"] for v in t["codes"].values()}
    assert groups == {"land", "direct", "indirect", "unknown"}
    # grouping is GEBCO's own: 0 land, 10-17 direct, 40-48 indirect, 70-72 unknown
    for k, v in t["codes"].items():
        c = int(k)
        expected = ("land" if c == 0 else
                    "direct" if 10 <= c <= 17 else
                    "indirect" if 40 <= c <= 48 else "unknown")
        assert v["group"] == expected, f"code {c} grouped as {v['group']}"
        assert len(v["definition"]) > 3
    # attribution + not-for-navigation honesty must ride with the table
    assert "doi:10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa" in t["source"]["attribution"]
    assert "NOT be used for navigation" in t["source"]["termsOfUse"]


# ── Esri ASCII parsing ───────────────────────────────────────────────────────

def test_parse_esri_ascii_header_and_values():
    g = _grid()
    assert (g.ncols, g.nrows) == (3, 3)
    assert (g.xll, g.yll, g.cellsize, g.nodata) == (142.0, 11.0, 1.0, -32767.0)
    assert g.cell(0, 0) == -4497           # first data row is the NORTH row
    assert g.cell(1, 0) == -10931
    assert g.xmax == 145.0 and g.ytop == 14.0
    with pytest.raises(ValueError):
        gst.parse_esri_ascii("ncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 1\n1 2 3\n")
    with pytest.raises(ValueError):
        gst.parse_esri_ascii("ncols 2\n1 2\n")


def test_grid_lonlat_to_cell_orientation():
    g = _grid()
    # cell (row 0, col 0) spans lon 142..143, lat 13..14 (top row = north)
    assert g.col_of(142.5) == 0 and g.row_of(13.5) == 0
    assert g.col_of(144.9) == 2 and g.row_of(11.1) == 2


def test_box_mean_excludes_nodata_and_clamps():
    g = _grid()
    # middle row: -10931, NODATA, -2063 -> mean of the two valid cells
    assert g.box_mean(1, 1, 0, 2) == pytest.approx((-10931 - 2063) / 2)
    # single nodata cell -> None
    assert g.box_mean(1, 1, 1, 1) is None
    # fully outside -> None; partial overlap clamps
    assert g.box_mean(5, 9, 0, 2) is None
    assert g.box_mean(-2, 0, -2, 0) == pytest.approx(-4497)


# ── terrarium encoding ───────────────────────────────────────────────────────

def test_terrarium_round_trip():
    for elev in (0.0, -1.0, -10931.0, 8848.0, -0.25, gst.TID_NO_DATA):
        r, g, b = gst.terrarium_encode(float(elev))
        assert all(0 <= c <= 255 for c in (r, g, b))
        assert gst.terrarium_decode(r, g, b) == pytest.approx(elev, abs=1 / 256)
    # the v1 layer's land-transparent anchor: elevation 0 encodes exactly
    assert gst.terrarium_encode(0.0) == (128, 0, 0)
    assert gst.terrarium_decode(128, 0, 0) == 0.0


# ── PMTiles primitives ───────────────────────────────────────────────────────

def test_varint_pins():
    assert gst.varint(0) == b"\x00"
    assert gst.varint(127) == b"\x7f"
    assert gst.varint(128) == b"\x80\x01"
    assert gst.varint(300) == b"\xac\x02"


def test_zxy_to_tileid_pins():
    # Pinned to the published PMTiles v3 Hilbert ordering (also cross-checked
    # against the real pmtiles JS lib in server/seafloorTiles.test.ts).
    assert gst.zxy_to_tileid(0, 0, 0) == 0
    assert gst.zxy_to_tileid(1, 0, 0) == 1
    assert gst.zxy_to_tileid(1, 0, 1) == 2
    assert gst.zxy_to_tileid(1, 1, 1) == 3
    assert gst.zxy_to_tileid(1, 1, 0) == 4
    assert gst.zxy_to_tileid(2, 0, 0) == 5
    # ids at one zoom never collide
    ids = {gst.zxy_to_tileid(3, x, y) for x in range(8) for y in range(8)}
    assert len(ids) == 64 and min(ids) == (4 ** 3 - 1) // 3


def _png_rows(png: bytes):
    """Minimal PNG decode for the writer's own output (filter 0 only)."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png[16:24])
    idat = b""
    off = 8
    while off < len(png):
        ln = struct.unpack(">I", png[off:off + 4])[0]
        tag = png[off + 4:off + 8]
        if tag == b"IDAT":
            idat += png[off + 8:off + 8 + ln]
        off += 12 + ln
    raw = zlib.decompress(idat)
    stride = w * 3 + 1
    rows = []
    for j in range(h):
        line = raw[j * stride:(j + 1) * stride]
        assert line[0] == 0, "writer emits filter 0 scanlines only"
        rows.append(line[1:])
    return w, h, rows


def test_png_encode_rgb_round_trip():
    rows = [bytes([j, 2 * j, 255 - j] * 4) for j in range(4)]
    png = gst.png_encode_rgb(4, 4, rows)
    w, h, got = _png_rows(png)
    assert (w, h) == (4, 4) and got == rows
    with pytest.raises(ValueError):
        gst.png_encode_rgb(4, 4, rows[:2])


def test_build_pmtiles_archive_parses_back():
    tiles = {
        gst.zxy_to_tileid(0, 0, 0): b"tile-zero",
        gst.zxy_to_tileid(1, 0, 1): b"tile-one",
    }
    bounds = (138.0, 7.0, 146.0, 15.0)
    pm = gst.build_pmtiles(tiles, {"name": "fixture"}, 0, 1, bounds)
    assert pm[:7] == b"PMTiles" and pm[7] == 3
    root_off, root_len = struct.unpack("<QQ", pm[8:24])
    meta_off, meta_len = struct.unpack("<QQ", pm[24:40])
    leaf_off, leaf_len = struct.unpack("<QQ", pm[40:56])
    tile_off, tile_len = struct.unpack("<QQ", pm[56:72])
    n_addr, n_entries, n_contents = struct.unpack("<QQQ", pm[72:96])
    assert (n_addr, n_entries, n_contents) == (2, 2, 2)
    assert pm[96] == 1 and pm[97] == 2 and pm[98] == 1 and pm[99] == 2
    assert (pm[100], pm[101]) == (0, 1)
    assert struct.unpack("<i", pm[102:106])[0] == 138_0000000
    assert struct.unpack("<i", pm[114:118])[0] == 15_0000000
    assert leaf_len == 0
    # decode the root directory and pull both tiles back out
    d = gzip.decompress(pm[root_off:root_off + root_len])
    pos = 0

    def rv():
        nonlocal pos
        shift = out = 0
        while True:
            b = d[pos]
            pos += 1
            out |= (b & 0x7F) << shift
            if not (b & 0x80):
                return out
            shift += 7

    n = rv()
    assert n == 2
    ids = []
    last = 0
    for i in range(n):
        last += rv()
        ids.append(last)
    assert ids == sorted(tiles.keys())
    runs = [rv() for _ in range(n)]
    lens = [rv() for _ in range(n)]
    offs = [rv() - 1 for _ in range(n)]
    assert runs == [1, 1]
    blobs = [pm[tile_off + o:tile_off + o + ln] for o, ln in zip(offs, lens)]
    assert blobs == [tiles[i] for i in ids]
    meta = json.loads(gzip.decompress(pm[meta_off:meta_off + meta_len]))
    assert meta["name"] == "fixture"
    assert tile_off + tile_len == len(pm)


# ── tile rendering ───────────────────────────────────────────────────────────

def test_render_depth_tile_transparent_outside_coverage():
    g = _grid()
    png = gst.render_depth_tile(g, 0, 0, 0)  # z0 world tile, 3x3-degree coverage
    assert png is not None
    w, h, rows = _png_rows(png)
    # pixel far outside coverage (lon -90) encodes elevation 0 -> transparent
    # under the client ramp
    i_out = int((-90.0 + 180.0) / 360.0 * 256)
    px = rows[128][i_out * 3:i_out * 3 + 3]
    assert gst.terrarium_decode(*px) == 0.0
    # pixel inside coverage decodes to a real negative depth from the fixture
    i_in = int((143.5 + 180.0) / 360.0 * 256)
    j_in = 128 - int(256 * (12.5 / 360.0) / 1.4)  # rough northern offset…
    found = False
    for j in range(0, 129):  # …so scan the column: at least one covered pixel
        px = rows[j][i_in * 3:i_in * 3 + 3]
        v = gst.terrarium_decode(*px)
        if v < -1000:
            found = True
            break
    assert found, "no covered pixel decoded to a fixture depth"


def test_render_depth_tile_none_when_no_overlap():
    g = _grid()
    # z3 tile at far west (lon -180..-135): no coverage -> None (tile skipped)
    assert gst.render_depth_tile(g, 3, 0, 3) is None


def test_render_tid_tile_nearest_and_nodata():
    t = gst.load_tid_decode()
    valid = {int(k) for k in t["codes"]}
    g = _grid(TID_FIXTURE)
    png = gst.render_tid_tile(g, 0, 0, 0, valid, 127)
    assert png is not None
    w, h, rows = _png_rows(png)
    seen = set()
    for j in range(h):
        for i in range(w):
            px = rows[j][i * 3:i * 3 + 3]
            seen.add(int(round(gst.terrarium_decode(*px))))
    # every rendered value is either a REAL fixture code or the -1 no-data
    # marker — codes are never averaged into nonsense like 25
    assert seen <= {gst.TID_NO_DATA, 0, 11, 40, 71}
    assert {11, 40}.issubset(seen)
    # the fixture's 127 nodata cell renders as -1, never as a fake code
    assert gst.TID_NO_DATA in seen


def test_render_tid_tile_rejects_undocumented_codes():
    t = gst.load_tid_decode()
    valid = {int(k) for k in t["codes"]}
    bad = TID_FIXTURE.replace("71", "99")  # 99 is not a documented TID code
    g = _grid(bad)
    with pytest.raises(ValueError, match="99"):
        gst.render_tid_tile(g, 0, 0, 0, valid, 127)
    with pytest.raises(ValueError, match="99"):
        gst.tid_histogram(g, valid, 127)


def test_tid_histogram_counts_and_skips_nodata():
    t = gst.load_tid_decode()
    valid = {int(k) for k in t["codes"]}
    g = _grid(TID_FIXTURE)
    hist = gst.tid_histogram(g, valid, 127)
    assert hist == {11: 2, 40: 4, 71: 1, 0: 1}  # 127 nodata excluded


# ── mercator helpers ─────────────────────────────────────────────────────────

def test_tiles_for_bbox():
    # world at z0/z1
    assert gst.tiles_for_bbox(0, -180, -85, 180, 85) == [(0, 0)]
    assert len(gst.tiles_for_bbox(1, -180, -85, 180, 85)) == 4
    # the Mariana demo bbox at z5: tile x spans 11.25 deg
    tiles = gst.tiles_for_bbox(5, 138, 7, 146, 15)
    assert all(28 <= x <= 28 for x, y in tiles)  # 138..146E all in x=28
    assert len(tiles) >= 1


def test_tile_bounds_round_trip():
    w, s, e, n = gst.tile_bounds(1, 1, 0)
    assert (w, e) == (0.0, 180.0)
    assert s == pytest.approx(0.0, abs=1e-9)
    assert n == pytest.approx(gst.MERC_MAX_LAT, abs=1e-6)


# ── download-app contract ────────────────────────────────────────────────────

def test_basket_body_contract():
    b = gst.basket_body(138.0, 7.0, 146.0, 15.0)
    assert b["processing_status"] == "new" and b["email"] is None
    (item,) = b["items"]
    # contract read from download.gebco.net's own JS on 2026-07-16
    assert item["grid_id"] == 1
    assert item["data_source_ids"] == [1, 3]     # Bathymetry + TID
    assert item["formats"] == [3]                # Esri ASCII
    assert (item["left"], item["right"]) == (138.0, 146.0)
    assert (item["bottom"], item["top"]) == (7.0, 15.0)
