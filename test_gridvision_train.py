"""test_gridvision_train.py — GRID VISION tower-detector v0 ON-POD pipeline battery
(scripts/gridvision_train/*). PURE functions only: bbox->pixel window, GSD
downsample factor, pixel-bbox->YOLO-normalized line, downsample label-invariance,
train/val split determinism, data.yaml, and the weight-sink URL/response parsing +
injected upload. NO network, NO rasterio, NO ultralytics, NO GPU — matching the
sibling test_gridvision_*.py convention (spec-load by path, deterministic).

train.py's top-level imports are all light (the heavy rasterio/ultralytics/torch
deps are imported INSIDE the on-pod functions), so loading train here also loads
its pure sibling modules (train.cog_reader / train.e2y / train.weight_sink), and
the whole --dry-run wiring runs offline in one test.
"""
import importlib.util
import os

import pytest

_ROOT = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "gridvision_train_entry",
    os.path.join(_ROOT, "scripts", "gridvision_train", "train.py"))
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

cog = train.cog_reader
e2y = train.e2y
ws = train.weight_sink


# ── (A) COG window pure geometry ────────────────────────────────────────────

def test_north_up_transform_and_guard():
    t = cog.north_up_transform(500000.0, 3550000.0, 0.6)
    assert t == (0.6, 0.0, 500000.0, 0.0, -0.6, 3550000.0)
    with pytest.raises(ValueError):
        cog.north_up_transform(0, 0, 0)


def test_pixel_window_from_bounds_matches_affine():
    # 0.6 m pixels, top-left at (500000, 3550000). A 300 m x 300 m box.
    t = cog.north_up_transform(500000.0, 3550000.0, 0.6)
    # box from x=500060..500360 (east), y=3549700..3550000 (a 300m tall window at top)
    col_off, row_off, w, h = cog.pixel_window_from_bounds(
        500060.0, 3549700.0, 500360.0, 3550000.0, t)
    # col starts at (500060-500000)/0.6 = 100; width 300/0.6 = 500 px
    assert col_off == 100
    assert w == 500
    # north edge is the top (row 0); south edge 300m down = 500 px
    assert row_off == 0
    assert h == 500


def test_pixel_window_clamps_to_raster_and_nonnegative():
    t = cog.north_up_transform(500000.0, 3550000.0, 0.6)
    # a bbox partly west/north of the origin -> offsets clamp to 0
    win = cog.pixel_window_from_bounds(499900.0, 3550100.0, 500060.0, 3550200.0,
                                       t, raster_size=(1000, 1000))
    assert win[0] == 0 and win[1] == 0
    assert win[2] >= 0 and win[3] >= 0
    with pytest.raises(ValueError):  # north-up assumption enforced
        cog.pixel_window_from_bounds(0, 0, 1, 1, (0.6, 0.1, 0, 0.1, -0.6, 0))


def test_downsample_factor():
    assert cog.downsample_factor(0.30, 0.60) == pytest.approx(2.0)  # coarser -> real downsample
    assert cog.downsample_factor(0.60, 0.60) == pytest.approx(1.0)
    assert cog.downsample_factor(0.60, 0.30) == pytest.approx(0.5)  # finer target -> <1, honest
    with pytest.raises(ValueError):
        cog.downsample_factor(0, 0.6)


def test_out_shape_for_window():
    assert cog.out_shape_for_window(500, 500, 2.0) == (250, 250)
    assert cog.out_shape_for_window(500, 500, 1.0) == (500, 500)
    assert cog.out_shape_for_window(0, 500, 2.0) == (0, 250)   # empty width stays empty
    assert cog.out_shape_for_window(1, 1, 4.0) == (1, 1)        # never rounds a box to 0
    with pytest.raises(ValueError):
        cog.out_shape_for_window(10, 10, 0)


# ── (B) ETDII -> YOLO pure conversion ───────────────────────────────────────

def test_pixel_bbox_to_yolo_line_center_and_norm():
    # a 48x48 box centered at (124,124) in a 512x512 image
    line = e2y.pixel_bbox_to_yolo_line([100, 100, 148, 148], 512, 512, 0)
    parts = line.split()
    assert parts[0] == "0"
    assert float(parts[1]) == pytest.approx(124 / 512, abs=1e-5)   # cx
    assert float(parts[2]) == pytest.approx(124 / 512, abs=1e-5)   # cy
    assert float(parts[3]) == pytest.approx(48 / 512, abs=1e-5)    # w
    assert float(parts[4]) == pytest.approx(48 / 512, abs=1e-5)    # h


def test_pixel_bbox_to_yolo_line_clamps_and_rejects():
    # overhanging box is clamped to the image, still valid
    line = e2y.pixel_bbox_to_yolo_line([-10, -10, 100, 100], 512, 512, 0)
    assert line is not None and all(0.0 <= float(x) <= 1.0 for x in line.split()[1:])
    # degenerate / outside -> None (never a zero-area box)
    assert e2y.pixel_bbox_to_yolo_line([50, 50, 50, 80], 512, 512) is None   # zero width
    assert e2y.pixel_bbox_to_yolo_line([600, 600, 700, 700], 512, 512) is None  # fully outside
    assert e2y.pixel_bbox_to_yolo_line(None, 512, 512) is None
    assert e2y.pixel_bbox_to_yolo_line([1, 2, 3, 4], 0, 512) is None          # bad dims


def test_downsample_label_invariance():
    # THE key GSD claim: normalizing makes the label identical at 0.30 and 0.60 m.
    box = [100, 100, 148, 148]
    native = e2y.pixel_bbox_to_yolo_line(box, 512, 512, 0)
    factor = cog.downsample_factor(0.30, 0.60)  # 2.0
    ow, oh = e2y.downsample_image_dims(512, 512, factor)
    assert (ow, oh) == (256, 256)
    ds = e2y.pixel_bbox_to_yolo_line([v / factor for v in box], ow, oh, 0)
    assert ds == native  # scale-invariant


def test_downsample_image_dims_guard():
    assert e2y.downsample_image_dims(512, 512, 2.0) == (256, 256)
    assert e2y.downsample_image_dims(3, 3, 4.0) == (1, 1)  # never below 1 px
    with pytest.raises(ValueError):
        e2y.downsample_image_dims(512, 512, 0)


def test_records_filter_tower_only():
    # reuse the REAL parser so this proves the reuse contract, not a re-impl
    etdii = train.etdii
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[
            [-110.9, 32.0], [-110.9, 32.001], [-110.899, 32.001], [-110.899, 32.0], [-110.9, 32.0]]]},
         "properties": {"label": "T", "filename": "img_1.geojson", "country": "USA",
                        "pixel_coordinates": [[100, 100], [100, 148], [148, 148], [148, 100]]}},
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[
            [-110.8, 32.1], [-110.8, 32.11], [-110.79, 32.11], [-110.79, 32.1], [-110.8, 32.1]]]},
         "properties": {"label": "SS", "filename": "img_1.geojson", "country": "USA",
                        "pixel_coordinates": [[10, 10], [10, 60], [60, 60], [60, 10]]}},
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[-110.0, 32.0], [-110.0, 32.01]]},
         "properties": {"label": "L", "filename": "img_1.geojson", "country": "USA"}},
    ]}
    recs = etdii.parse_geojson(gj, "ETDII", "CC BY 4.0")
    by_img = e2y.group_records_by_image(recs)
    assert set(by_img) == {"img_1.geojson"}
    # tower-only default: SS + L excluded (6-label substation is not a v0 class)
    lines = e2y.records_to_yolo_lines(by_img["img_1.geojson"], 512, 512)
    assert len(lines) == 1 and lines[0].startswith("0 ")
    # a substation feature carries a pixel box but is filtered by keep_classes
    kept = e2y.image_to_label_records(recs, keep_classes=("tower",))
    assert [c for c, _ in kept] == [0]


# ── (B2) v1 tiling: detect on native-GSD windows, not a downscaled full frame ─

def test_tile_windows_cover_edges_and_clamp():
    wins = e2y.tile_windows(1500, 1000, tile=640, stride=512)
    # every window is within the image and no bigger than the tile
    for x0, y0, tw, th in wins:
        assert 0 <= x0 and 0 <= y0 and 0 < tw <= 640 and 0 < th <= 640
        assert x0 + tw <= 1500 and y0 + th <= 1000
    # far edges are covered (a flush window on each axis)
    assert any(x0 + tw == 1500 for x0, y0, tw, th in wins)
    assert any(y0 + th == 1000 for x0, y0, tw, th in wins)
    # an image smaller than the tile -> a single full-extent window
    small = e2y.tile_windows(400, 300, tile=640, stride=512)
    assert small == [(0, 0, 400, 300)]
    with pytest.raises(ValueError):
        e2y.tile_windows(100, 100, tile=0, stride=512)


def test_aug_kwargs_presets():
    assert train.aug_kwargs("default") == {}
    strong = train.aug_kwargs("strong")
    # the cross-domain-critical knobs are actually cranked
    assert strong["hsv_v"] >= 0.4 and strong["hsv_s"] >= 0.6
    assert strong["flipud"] == 0.5          # nadir aerial: vertical flip is label-safe
    assert strong["mixup"] > 0 and strong["copy_paste"] > 0
    # returns a COPY, not the shared module dict (caller must not mutate global)
    strong["hsv_v"] = 0.0
    assert train.aug_kwargs("strong")["hsv_v"] >= 0.4


def test_region_of_longest_prefix():
    assert e2y.region_of("USA_AZ_Tucson_12") == "USA_AZ_Tucson"
    assert e2y.region_of("USA_KS_Colwich_Maize_3") == "USA_KS_Colwich_Maize"
    assert e2y.region_of("NZ_Dunedin_1") is None       # unknown region -> never bucketed
    # a held-out split is a clean partition: every AZ stem is val, every KS stem train
    stems = ["USA_AZ_Tucson_1", "USA_AZ_Tucson_2", "USA_KS_Colwich_Maize_1"]
    val = [s for s in stems if e2y.region_of(s) == "USA_AZ_Tucson"]
    train = [s for s in stems if e2y.region_of(s) != "USA_AZ_Tucson"]
    assert val == ["USA_AZ_Tucson_1", "USA_AZ_Tucson_2"]
    assert train == ["USA_KS_Colwich_Maize_1"]
    assert set(val) & set(train) == set()              # no leakage across the fold


def test_scale_bbox_halves_for_downsample():
    assert e2y.scale_bbox([1400, 600, 1440, 640], 2.0) == [700.0, 300.0, 720.0, 320.0]
    with pytest.raises(ValueError):
        e2y.scale_bbox([1, 2, 3, 4], 0)


def test_box_in_tile_clips_translates_and_drops():
    # fully-inside box -> tile-local normalized line (center of a 20px box at (710,310))
    ln = e2y.box_in_tile_yolo_line([700, 300, 720, 320], 512, 0, 640, 640, 0)
    parts = ln.split()
    assert parts[0] == "0"
    assert float(parts[1]) == pytest.approx((710 - 512) / 640, abs=1e-5)  # local cx
    assert float(parts[2]) == pytest.approx(310 / 640, abs=1e-5)          # local cy
    # box east of the tile -> no overlap -> None
    assert e2y.box_in_tile_yolo_line([700, 300, 720, 320], 0, 0, 640, 640, 0) is None
    # barely-clipped box (only ~1% inside the tile edge) is dropped by min_visible
    assert e2y.box_in_tile_yolo_line([636, 100, 736, 200], 0, 0, 640, 640, 0) is None
    # a box straddling the edge but mostly inside is KEPT and clipped in-bounds
    kept = e2y.box_in_tile_yolo_line([600, 100, 660, 160], 0, 0, 640, 640, 0)
    assert kept is not None and all(0.0 <= float(v) <= 1.0 for v in kept.split()[1:])


# ── (B) deterministic split ─────────────────────────────────────────────────

def test_train_val_split_deterministic_and_order_independent():
    ids = [f"img_{i}" for i in range(200)]
    tr1, va1 = e2y.train_val_split(ids, 0.2)
    tr2, va2 = e2y.train_val_split(list(reversed(ids)), 0.2)
    assert (tr1, va1) == (tr2, va2)           # same image -> same side, any order
    assert len(tr1) + len(va1) == 200
    assert 0 < len(va1) < 200                  # ~20% roughly
    assert 25 < len(va1) < 55                  # binomial slack around 40
    # edge fracs
    assert e2y.train_val_split(ids, 0.0)[1] == []
    assert e2y.train_val_split(ids, 1.0)[0] == []


def test_assign_split_stable():
    assert e2y.assign_split("img_7") == e2y.assign_split("img_7")
    assert e2y.assign_split("img_7") in ("train", "val")


# ── data.yaml ───────────────────────────────────────────────────────────────

def test_data_yaml_single_tower_class():
    y = e2y.data_yaml_dict("/workspace/ds")
    assert y["nc"] == 1 and y["names"] == ["tower"]
    assert y["train"] == "images/train" and y["val"] == "images/val"
    txt = e2y.dump_simple_yaml(y)
    assert "nc: 1" in txt and "names: ['tower']" in txt and "path: /workspace/ds" in txt


# ── (D) weight sink pure parsing + injected upload ──────────────────────────

def test_parse_upload_url():
    assert ws.parse_upload_url("https://0x0.st/abc.pt\n") == "https://0x0.st/abc.pt"
    assert ws.parse_upload_url("  http://transfer.sh/x/best.pt  ") == "http://transfer.sh/x/best.pt"
    assert ws.parse_upload_url("<html>error</html>") is None
    assert ws.parse_upload_url("") is None
    assert ws.parse_upload_url(None) is None


def test_estimate_retention_days_curve():
    tiny = ws.estimate_retention_days(6 * 1024 * 1024)      # ~6 MB detector
    near = ws.estimate_retention_days(int(0.99 * 512 * 1024 * 1024))
    assert 30.0 <= near <= tiny <= 365.0                    # smaller lives longer
    assert tiny > 350.0                                     # a small .pt lives ~a year
    assert ws.estimate_retention_days(0) == 365.0


def test_multipart_body_contains_file():
    ct, body = ws.multipart_body("file", "best.pt", b"WEIGHTS", boundary="B123")
    assert "boundary=B123" in ct
    assert b'filename="best.pt"' in body
    assert b"WEIGHTS" in body
    assert body.startswith(b"--B123") and body.rstrip().endswith(b"--B123--")


def test_result_line_survives_upload_failure():
    line = ws.result_line({"map50": 0.61}, weight_url=None)
    assert line.startswith("GRIDVISION_RESULT ")
    import json
    doc = json.loads(line[len("GRIDVISION_RESULT "):])
    assert doc["metrics"]["map50"] == 0.61 and doc["weight_url"] is None


def test_upload_weight_with_injected_uploader(tmp_path):
    p = tmp_path / "best.pt"
    p.write_bytes(b"x" * 1024)

    def ok_uploader(url, content_type, body, timeout=120):
        return "https://0x0.st/zzz.pt\n"

    res = ws.upload_weight(str(p), uploader=ok_uploader)
    assert res["ok"] and res["url"] == "https://0x0.st/zzz.pt"
    assert res["size_bytes"] == 1024 and res["retention_days_est"] > 0

    def dead_uploader(url, content_type, body, timeout=120):
        raise OSError("blocked")

    fail = ws.upload_weight(str(p), hosts=("https://0x0.st",), uploader=dead_uploader)
    assert fail["ok"] is False and fail["url"] is None and fail["errors"]


# ── (F) on-pod bootstrap pins a torch-2.1-compatible NumPy ──────────────────

def test_pip_bootstrap_pins_numpy_below_2(monkeypatch):
    """Regression for gv-detector-v0-1 (2026-07-09): the RunPod torch-2.1.0 base
    image + an unpinned `pip install ultralytics` pulls NumPy 2.x, and the first
    torch.from_numpy (ultralytics' AMP check) dies with 'Numpy is not available',
    burning a full GPU-hour before any epoch. The bootstrap must constrain numpy<2
    in the install command, ahead of ultralytics, so pip never resolves 2.x."""
    import subprocess

    captured = {}

    def fake_check_call(argv, *a, **k):
        captured["argv"] = list(argv)

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    train.pip_bootstrap()
    argv = captured["argv"]
    assert "numpy<2" in argv, argv
    # the numpy constraint must precede ultralytics in the resolve set
    assert argv.index("numpy<2") < argv.index("ultralytics"), argv


# ── (E) full dry-run wiring, offline ────────────────────────────────────────

def test_dry_run_wires_end_to_end_offline():
    report = train.dry_run()
    assert report["ok"] is True, report
    names = {c["name"]: c["ok"] for c in report["checks"]}
    # every wiring check green, and the real manifest's verified counts confirmed
    assert names["labels_manifest"] and names["yolo_conversion"]
    assert names["downsample_invariance"] and names["split_determinism"]
    assert names["cog_window_geometry"] and names["weight_sink_parse"]
    assert names["tiling"]  # v1 tiling path wired
    assert report["summary"]["manifest_tower"] == 1408
    assert report["summary"]["manifest_substation"] == 6
