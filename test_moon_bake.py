"""test_moon_bake.py — the Moon bake's tile scheme must match what the client reads.

The bake (scripts/moon_bake.py) writes object keys; the client
(client/src/lib/celestial/lroc.ts) computes the URLs it asks for. Those two
live in DIFFERENT LANGUAGES with no shared type, which is precisely the
failure class CLAUDE.md's READ BEFORE WRITE rule warns about: a scheme change
on one side fails SILENTLY AT RUNTIME (404s / black Moon), never in CI.

So the parity tests below do not restate the formulas from memory — they
PARSE lroc.ts and assert the TypeScript still says what the Python assumes.
If someone changes either side's scheme, this fails loudly and names the
mismatch.

The remaining tests pin the honesty rails: which levels are real detail vs.
upsampled, and that the manifest cannot silently overclaim resolution.
"""

from __future__ import annotations

import importlib.util
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_LROC = os.path.join(_HERE, "client", "src", "lib", "celestial", "lroc.ts")


def _load_bake():
    path = os.path.join(_HERE, "scripts", "moon_bake.py")
    spec = importlib.util.spec_from_file_location("moon_bake", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mb = _load_bake()


def _lroc_src() -> str:
    with open(_LROC, "r", encoding="utf-8") as fh:
        return fh.read()


# ── cross-language scheme parity (the silent-404 guard) ────────────────────


def test_lroc_module_exists():
    assert os.path.exists(_LROC), "lroc.ts is the client half of this contract"


def test_matrix_width_formula_matches_client():
    src = _lroc_src()
    m = re.search(r"export function matrixWidth\(z: number\): number \{\s*return ([^;]+);", src)
    assert m, "matrixWidth() not found in lroc.ts — scheme contract broken"
    assert m.group(1).strip() == "2 ** (z + 1)", (
        f"lroc.ts matrixWidth is now `{m.group(1).strip()}` but moon_bake.matrix_size "
        f"still assumes 2^(z+1). The bake would write keys the client never requests."
    )
    for z in range(0, 9):
        assert mb.matrix_size(z)[0] == 2 ** (z + 1)


def test_matrix_height_formula_matches_client():
    src = _lroc_src()
    m = re.search(r"export function matrixHeight\(z: number\): number \{\s*return ([^;]+);", src)
    assert m, "matrixHeight() not found in lroc.ts"
    assert m.group(1).strip() == "2 ** z"
    for z in range(0, 9):
        assert mb.matrix_size(z)[1] == 2 ** z


def test_deg_per_tile_matches_client():
    src = _lroc_src()
    m = re.search(r"export function degPerTile\(z: number\): number \{\s*return ([^;]+);", src)
    assert m, "degPerTile() not found in lroc.ts"
    assert m.group(1).strip() == "180 / 2 ** z"
    for z in range(0, 9):
        assert mb.deg_per_tile(z) == pytest.approx(180 / 2**z)


def test_key_order_is_row_before_column():
    """The single easiest thing to get backwards, and it 404s every tile."""
    src = _lroc_src()
    m = re.search(r"export function tileUrl\([^)]*\): string \{\s*return `([^`]+)`", src)
    assert m, "tileUrl() not found in lroc.ts"
    tmpl = m.group(1)
    assert "${t.z}/${t.y}/${t.x}" in tmpl, (
        f"lroc.ts tileUrl path is `{tmpl}` — moon_bake.tile_key writes {{z}}/{{y}}/{{x}}. "
        f"If the client now asks for {{z}}/{{x}}/{{y}} every baked tile is a 404."
    )
    assert mb.tile_key(3, 5, 2) == "moon/wac/3/2/5.jpg"  # z/y/x
    assert mb.tile_key(3, 5, 2, prefix="p") == "p/3/2/5.jpg"


def test_tile_px_matches_client_scheme():
    src = _lroc_src()
    m = re.search(r"export const MOON_TREK: TrekScheme = \{.*?tilePx: (\d+)", src, re.S)
    assert m, "MOON_TREK.tilePx not found in lroc.ts"
    assert int(m.group(1)) == mb.TILE_PX


# ── tile geometry ──────────────────────────────────────────────────────────


def test_level_zero_is_two_tiles_covering_the_globe():
    assert mb.matrix_size(0) == (2, 1)
    assert mb.tiles_at(0) == 2
    w, e = mb.tile_bounds_deg(0, 0, 0), mb.tile_bounds_deg(0, 1, 0)
    assert w[0] == -180.0 and w[2] == 0.0
    assert e[0] == 0.0 and e[2] == 180.0
    assert w[1] == 90.0 and w[3] == -90.0, "level 0 spans pole to pole"


def test_tiles_are_square_in_degrees():
    for z in range(0, 8):
        lon0, lat1, lon1, lat0 = mb.tile_bounds_deg(z, 0, 0)
        assert (lon1 - lon0) == pytest.approx(lat1 - lat0)


def test_row_zero_is_north():
    _, lat_max_top, _, _ = mb.tile_bounds_deg(2, 0, 0)
    _, lat_max_next, _, _ = mb.tile_bounds_deg(2, 0, 1)
    assert lat_max_top == 90.0
    assert lat_max_next < lat_max_top, "y increases SOUTHWARD (TopLeftCorner convention)"


def test_level_raster_is_exactly_the_tile_grid():
    for z in range(0, 8):
        cols, rows = mb.matrix_size(z)
        assert mb.level_px(z) == (cols * mb.TILE_PX, rows * mb.TILE_PX)


def test_iter_tiles_covers_the_grid_exactly_once():
    for z in (0, 1, 3):
        seen = list(mb.iter_tiles(z))
        assert len(seen) == mb.tiles_at(z)
        assert len(set(seen)) == len(seen)


def test_negative_level_rejected():
    with pytest.raises(ValueError):
        mb.matrix_size(-1)


# ── the pilot's own numbers (so a regression in the math is visible) ───────


def test_pilot_tile_count_is_2730():
    assert mb.total_tiles(5) == 2730


def test_full_native_bake_is_43690_tiles():
    assert mb.total_tiles(mb.NATIVE_MAX_Z) == 43690


# ── honesty rails ──────────────────────────────────────────────────────────


def test_native_ceiling_is_the_last_true_downsample():
    """z7 is 65536px wide against a 109164px source (a real downsample);
    z8 is 131072px wide, i.e. INTERPOLATED. Getting this backwards would let
    the product advertise resolution the data does not contain."""
    assert mb.native_ceiling_z() == 7
    assert mb.level_px(7)[0] <= mb.SRC_W
    assert mb.level_px(8)[0] > mb.SRC_W
    assert not mb.is_upsampled(7)
    assert mb.is_upsampled(8)


def test_manifest_flags_upsampled_levels():
    man = mb.manifest(8)
    by_z = {L["z"]: L for L in man["levels"]}
    assert by_z[7]["upsampled"] is False
    assert by_z[8]["upsampled"] is True, "z8 must be labelled interpolated, not real detail"


def test_manifest_records_public_domain_provenance():
    man = mb.manifest(5)
    assert "Public Domain" in man["source"]["license"]
    assert man["source"]["url"].startswith("https://planetarymaps.usgs.gov/")
    assert man["source"]["m_per_px"] == 100.0
    assert "average" in man["resampling"]
    assert "nodata" in man


def test_manifest_declares_the_scheme_it_was_baked_in():
    man = mb.manifest(5)
    s = man["scheme"]
    assert s["kind"] == "trek-eq"
    assert s["tile_px"] == 256
    assert s["max_z"] == 5
    assert s["native_max_z"] == 7
    assert man["tiles_total"] == 2730


def test_manifest_resolution_matches_the_grid():
    """m_per_px must be derived from the level's real pixel width, not typed in."""
    man = mb.manifest(7)
    moon_circumference_m = 2 * 3.14159265358979 * 1737400
    for L in man["levels"]:
        assert L["m_per_px"] == pytest.approx(moon_circumference_m / L["px_w"], rel=1e-3)


# ── the manifest must describe the BUCKET, not the invocation ──────────────
# REGRESSION (caught on the first real sub-range bake, 2026-08-13): running
# `build --min-z 6 --max-z 7` rewrote tiles.json to claim min_z=6, so the
# sidecar advertised that levels 0-5 did not exist while they were sitting in
# R2. The manifest is the ONLY thing the client trusts to know what is
# available, so a manifest that under-reports is a silent product outage.


def _fake_level(root, z, complete=True):
    """Materialise level z's tile tree on disk (optionally one tile short)."""
    cols, rows = mb.matrix_size(z)
    n = 0
    total = cols * rows
    for y in range(rows):
        d = os.path.join(root, str(z), str(y))
        os.makedirs(d, exist_ok=True)
        for x in range(cols):
            n += 1
            if not complete and n == total:
                continue
            open(os.path.join(d, f"{x}.jpg"), "wb").close()


def test_levels_on_disk_finds_every_complete_level(tmp_path):
    root = str(tmp_path)
    for z in (0, 1, 2):
        _fake_level(root, z)
    assert mb.levels_on_disk(root, max_probe=3) == [0, 1, 2]


def test_an_incomplete_level_is_not_advertised(tmp_path):
    """A half-present level renders as holes — excluded, never advertised."""
    root = str(tmp_path)
    _fake_level(root, 0)
    _fake_level(root, 1)
    _fake_level(root, 2, complete=False)  # one tile short
    assert mb.levels_on_disk(root, max_probe=3) == [0, 1]


def test_manifest_covers_all_levels_present_not_just_the_last_bake(tmp_path):
    root = str(tmp_path)
    for z in (0, 1, 2, 3):
        _fake_level(root, z)
    man = mb.write_manifest(root)
    assert man["scheme"]["min_z"] == 0, "a sub-range bake must not orphan the lower levels"
    assert man["scheme"]["max_z"] == 3
    assert man["levels_complete"] == [0, 1, 2, 3]
    assert man["tiles_total"] == mb.total_tiles(3)


def test_write_manifest_refuses_to_describe_an_empty_tree(tmp_path):
    with pytest.raises(SystemExit):
        mb.write_manifest(str(tmp_path))
