"""Pure-function tests for the FAA chart bake (scripts/faa_charts/): the
56-day cycle and edition discovery, edge geometry, the premultiplied raster
maths, WebP encoding, and the edge-reuse decision. Network/raster I/O is
exercised by the bake run itself (gated against the FAA mosaic, verify.py).
"""
import datetime as dt
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "faa_charts"))

import bake  # noqa: E402
import cycle  # noqa: E402
import edges  # noqa: E402


# ── cycle + discovery ───────────────────────────────────────────────────────

def test_cycle_dates_match_the_faa_editions_and_the_server_constants():
    # the service's own editions and the aeronav directory that exists
    for d in ("2026-05-14", "2026-07-09", "2026-09-03"):
        assert cycle.cycle_start(dt.date.fromisoformat(d)).isoformat() == d
    assert cycle.cycle_start(dt.date(2026, 10, 6)) == dt.date(2026, 9, 3)
    assert cycle.next_cycle(dt.date(2026, 10, 6)) == dt.date(2026, 10, 29)
    assert cycle.cycle_start(dt.date(2026, 9, 2)) == dt.date(2026, 7, 9)
    assert cycle.aeronav_date(dt.date(2026, 9, 3)) == "09-03-2026"
    # mirrors server/aeroCharts.ts (one cycle definition, two languages)
    ts = open(os.path.join(os.path.dirname(__file__), "server", "aeroCharts.ts")).read()
    assert f'FAA_CYCLE_ANCHOR = "{cycle.FAA_CYCLE_ANCHOR.isoformat()}"' in ts
    assert f"FAA_CYCLE_DAYS = {cycle.FAA_CYCLE_DAYS};" in ts


LISTING = """<html><body><pre>
<A HREF="/visual/09-03-2026/sectional-files/Seattle.zip">Seattle.zip</A>
<a href="/visual/09-03-2026/sectional-files/Klamath_Falls.zip">Klamath_Falls.zip</a>
<a href="/visual/09-03-2026/sectional-files/Seattle.zip">dup</a>
<a href="/visual/09-03-2026/">[To Parent Directory]</a>
<a href="https://s.go-mpulse.net/boomerang/x.js">boomerang</a>
</pre></body></html>"""


def test_listing_parser_keeps_only_this_familys_files():
    assert cycle.parse_listing(LISTING, cycle.FAMILIES["sectional"].file_re) == ["Klamath_Falls.zip", "Seattle.zip"]
    enr = '<a href="ENR_L05.zip">x</a><a href="ENR_H03.zip">x</a><a href="ENR_A01.zip">x</a>' \
          '<a href="ENR_AKL01.zip">x</a><a href="DELUS1.zip">x</a><a href="ENR_AKH01.zip">x</a>'
    assert cycle.parse_listing(enr, cycle.FAMILIES["ifrlow"].file_re) == ["ENR_A01.zip", "ENR_AKL01.zip", "ENR_L05.zip"]
    assert cycle.parse_listing(enr, cycle.FAMILIES["ifrhigh"].file_re) == ["ENR_AKH01.zip", "ENR_H03.zip"]
    tac = '<a href="Seattle_TAC.zip">x</a><a href="Seattle.zip">x</a>'
    assert cycle.parse_listing(tac, cycle.FAMILIES["tac"].file_re) == ["Seattle_TAC.zip"]


def test_newest_published_prefers_the_early_posted_next_cycle():
    fam = cycle.FAMILIES["sectional"]
    posted = {"09-03-2026"}

    def get(url):
        return LISTING if any(d in url for d in posted) else None

    today = dt.date(2026, 10, 6)
    assert cycle.newest_published(fam, today, get) == dt.date(2026, 9, 3)
    posted.add("10-29-2026")  # FAA posts ~20 days early
    assert cycle.newest_published(fam, today, get) == dt.date(2026, 10, 29)
    posted.clear()
    assert cycle.newest_published(fam, today, get) is None


# ── edge geometry ───────────────────────────────────────────────────────────

def test_dp_simplify_and_ring_simplify_drop_collinear_noise():
    line = [(0, 0), (1, 0.1), (2, -0.1), (3, 0), (10, 0)]
    assert edges.dp_simplify(line, 0.5) == [(0, 0), (10, 0)]
    ring = [(0, 0), (5, 0.2), (10, 0), (10, 5), (10, 10), (5, 9.9), (0, 10), (0, 5)]
    out = edges.simplify_ring(ring, 0.5)
    assert sorted(out) == sorted([(0, 0), (10, 0), (10, 10), (0, 10)])


def test_line_fit_intersection_and_step_location():
    (c, d) = edges.fit_line([(0, 1), (1, 1.01), (2, 0.99), (3, 1)])
    assert abs(c[1] - 1.0) < 0.01 and abs(abs(d[0]) - 1) < 1e-3
    p = edges.intersect(((0, 0), (1, 0)), ((5, 5), (0, 1)))
    assert p == pytest.approx((5, 0))
    assert edges.intersect(((0, 0), (1, 0)), ((0, 1), (1, 0))) is None  # parallel
    assert edges.step_location([True] * 7 + [False] * 5) == 7
    # robust to a mismatching label inside and a matching speck outside
    assert edges.step_location([True, True, False, True, True, True, False, False, True, False, False]) == 6


def test_point_in_ring_and_densify():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert edges.point_in_ring((5, 5), sq) and not edges.point_in_ring((11, 5), sq)
    d = edges.densify(sq, 2.5)
    assert len(d) == 16 and d[0] == (0, 0) and (5.0, 0.0) in d


def test_fingerprint_reuse_tolerates_float_noise_but_not_a_moved_chart():
    fp = {"width": 100, "height": 50, "crs": "LCC", "transform": [42.3, 0, -409375.3, 0, -42.3, 261032.1]}
    same = {**fp, "transform": [42.3000001, 0, -409375.2, 0, -42.3, 261032.0]}
    moved = {**fp, "transform": [42.3, 0, -409000.0, 0, -42.3, 261032.1]}
    rescaled = {**fp, "transform": [45.0, 0, -409375.3, 0, -45.0, 261032.1]}
    assert edges.fingerprint_matches(fp, same)
    assert not edges.fingerprint_matches(fp, moved)
    assert not edges.fingerprint_matches(fp, rescaled)
    assert not edges.fingerprint_matches(fp, {**fp, "crs": "other"})
    assert not edges.fingerprint_matches(None, fp)


# ── raster maths + encoding ─────────────────────────────────────────────────

def test_premultiplied_downsample_never_darkens_edges():
    a = np.zeros((4, 4, 4), np.float32)
    a[:, :2] = (200, 100, 50, 1.0)  # left half opaque orange, right half transparent black
    out = bake.downsample_premultiplied(a, 2)
    assert out.shape == (2, 2, 4)
    assert out[0, 0, :3] == pytest.approx([200, 100, 50]) and out[0, 0, 3] == 1.0
    assert out[0, 1, 3] == 0.0
    b = np.zeros((2, 2, 4), np.float32)
    b[0, 0] = (200, 100, 50, 1.0)  # one opaque pixel of four
    m = bake.downsample_premultiplied(b, 2)[0, 0]
    assert m[:3] == pytest.approx([200, 100, 50]) and m[3] == pytest.approx(0.25)


def test_compose_children_places_quadrants_and_keeps_missing_ones_transparent():
    red = np.zeros((256, 256, 4), np.float32)
    red[...] = (255, 0, 0, 1.0)
    out = bake.compose_children([red, None, None, red])
    assert out[10, 10, 0] == 255 and out[10, 10, 3] == 1.0
    assert out[10, 200, 3] == 0.0 and out[200, 10, 3] == 0.0
    assert out[200, 200, 0] == 255
    assert bake.compose_children([None] * 4) is None
    with pytest.raises(ValueError):
        bake.compose_children([red])


def test_encode_is_webp_keeps_alpha_only_when_needed_and_skips_empty_tiles():
    opaque = np.zeros((256, 256, 4), np.float32)
    opaque[...] = (10, 120, 200, 1.0)
    data = bake.encode(opaque)
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    back = bake.decode(data)
    assert back[..., 3].min() == 1.0
    assert np.abs(back[128, 128, :3] - [10, 120, 200]).max() < 8
    half = opaque.copy()
    half[:, 128:, 3] = 0
    back = bake.decode(bake.encode(half))
    assert back[10, 10, 3] == 1.0 and back[10, 200, 3] == 0.0, "alpha is lossless"
    assert bake.encode(np.zeros((256, 256, 4), np.float32)) is None


def test_parent_tiles_dedupes():
    assert bake.parent_tiles([(4, 6), (5, 6), (4, 7), (9, 9)], 5) == [(2, 3), (4, 4)]


# ── edge reuse decision ─────────────────────────────────────────────────────

class _Chart:
    def __init__(self, name, fp):
        self.name = name
        self._fp = fp


def test_edges_reusable_requires_every_chart_known_and_unmoved(monkeypatch):
    import run

    fp = {"width": 1, "height": 1, "crs": "C", "transform": [1, 0, 0, 0, -1, 0]}
    monkeypatch.setattr(edges, "fingerprint", lambda c: c._fp)
    stored = {"charts": {"Seattle SEC": {"fingerprint": fp}, "Klamath Falls SEC": {"fingerprint": fp}},
              "excluded": ["Seattle SEC Inset"]}
    charts = [_Chart("Seattle SEC", fp), _Chart("Klamath Falls SEC", fp), _Chart("Seattle SEC Inset", fp)]
    assert run.edges_reusable(stored, charts) == (True, [])
    ok, why = run.edges_reusable(stored, charts[:1] + charts[2:])
    assert not ok and "Klamath Falls SEC: no longer published" in why
    ok, why = run.edges_reusable(stored, charts + [_Chart("New SEC", fp)])
    assert not ok and "New SEC: new chart" in why
    moved = {**fp, "transform": [1, 0, 999, 0, -1, 0]}
    ok, why = run.edges_reusable(stored, [_Chart("Seattle SEC", moved), charts[1], charts[2]])
    assert not ok and "Seattle SEC: georeferencing changed" in why
    assert run.edges_reusable(None, charts) == (False, ["no stored edges"])


# ── antimeridian + family exclusions ────────────────────────────────────────

def test_antimeridian_boxes_and_wrapping():
    from common import ORIGIN, antimeridian_boxes, lonlat_to_m, m_to_tile, wrap_x

    one = antimeridian_boxes(-125, 44.5, -117, 49)
    assert len(one) == 1 and one[0][0] == pytest.approx(lonlat_to_m(-125, 0)[0])
    # Western Aleutian Islands East sectional: GDAL reports west > east
    two = antimeridian_boxes(177.16, 50.75, -172.35, 53.29)
    assert len(two) == 2
    assert two[0][2] == ORIGIN and two[1][0] == -ORIGIN
    assert two[0][0] == pytest.approx(lonlat_to_m(177.16, 0)[0])
    assert two[1][2] == pytest.approx(lonlat_to_m(-172.35, 0)[0])
    assert wrap_x(ORIGIN + 10) == pytest.approx(-ORIGIN + 10)
    assert m_to_tile(ORIGIN + 10, 0, 2) == (0, 2)


def test_unwrap_ring_keeps_a_date_line_polygon_continuous_on_both_sides():
    from common import ORIGIN

    near = 0.99 * ORIGIN
    ring = [(near, 0), (-near, 0), (-near, 10), (near, 10), (near, 0)]
    out = bake.unwrap_ring(ring)
    assert len(out) == 2
    east, west = out
    assert max(p[0] for p in east) - min(p[0] for p in east) < 0.1 * ORIGIN
    assert min(p[0] for p in east) == pytest.approx(near)
    assert west[0][0] == pytest.approx(near - 2 * ORIGIN)
    plain = [(0, 0), (10, 0), (10, 10), (0, 0)]
    assert bake.unwrap_ring(plain) == [plain]


def test_tac_layer_excludes_the_flyway_charts_shipped_in_its_zips():
    import re

    fam = cycle.FAMILIES["tac"]
    assert re.search(fam.exclude_re, "Los Angeles FLY")
    assert not re.search(fam.exclude_re, "Los Angeles TAC")
    assert not re.search(cycle.FAMILIES["sectional"].exclude_re, "Seattle SEC")


def test_snap_antimeridian_closes_the_cut_between_a_charts_two_halves():
    from common import ORIGIN

    ring = [(-ORIGIN + 1000, 0), (-ORIGIN + 5e5, 0), (ORIGIN - 2000, 5)]
    out = bake.snap_antimeridian(ring)
    assert out[0][0] == -ORIGIN and out[2][0] == ORIGIN
    assert out[1] == ring[1]
