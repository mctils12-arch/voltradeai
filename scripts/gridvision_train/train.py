#!/usr/bin/env python3
"""train.py — GRID VISION tower-detector v0 ON-POD training entrypoint.

Runs on a RunPod GPU pod (image runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-
ubuntu22.04). Fine-tunes an Ultralytics YOLOv8 detector from COCO-pretrained
weights on the CC-BY ETDII US TOWER labels, downsampled to 0.60 m GSD to match
NAIP, then pushes weights + metrics off the ephemeral pod.

  full pipeline:  pip-bootstrap -> fetch ETDII US zips -> build YOLO dataset (B)
                  -> [optional] materialize NAIP chips (A) -> train -> write-back (D)

HONEST v0 SCOPE (research/grid_vision_phaseb.md, pre-stated prior): TOWER ONLY.
1408 clean US towers; 6 substations is NOT a trainable class, so substation is
excluded (never blended in). Pre-stated prior: tower AP50 0.55-0.75 @0.6 m (below
the 0.73 @0.3 m bar — we train/infer at 2x coarser GSD on 2 US regions). This
script does not judge the result; it produces it and reports metrics honestly.

LICENSE FLAG (surfaced, must reach the human — this is stricter than the phase-B
note assumed): Ultralytics YOLOv8 CODE and its COCO-pretrained weights are
AGPL-3.0 (network copyleft), NOT the CC-BY-SA the phase-B weights-hygiene note
assumed for a Detectron2 starter. Under AGPL, SERVING detections from a
YOLOv8-derived model can trigger source-disclosure obligations. The fine-tuned
weights stay INTERNAL regardless (research/grid_vision_phaseb.md). If the plan is
to SELL served detections, a permissively-licensed detector (Detectron2 code is
Apache-2.0; PI-Detection MIT starter) is the clean-commercial path — a human
decision, flagged here, not silently taken.

----------------------------------------------------------------------------
ON-POD BOOTSTRAP RECIPE (F) — what the pod runs; no Dockerfile needed because the
base image already carries CUDA + torch 2.1. The launcher's --cmd wraps this in an
in-pod `timeout <cap>s` (scripts/runpod_launch.py build_create_body):

    cd /workspace && \
    git clone --depth 1 --branch claude/gridvision-train-container \
        https://github.com/<owner>/voltradeai.git repo && \
    cd repo && \
    python3 scripts/gridvision_train/train.py --epochs 60 --imgsz 512 \
        --model yolov8n.pt --out /workspace/gv_out

(train.py pip-installs ultralytics + rasterio + pillow itself on first run, so the
--cmd can also be just the clone + train.py call. The repo is public-cloneable; if
private, pass a token via the pod env — BLOCKED-FOR-MIKE. Simplest keyless path:
the launcher --cmd curls this repo's tarball for the branch instead of git+token.)
----------------------------------------------------------------------------

CLI:
  # provable-wired, CPU, NO GPU / rasterio / ultralytics needed:
  python3 scripts/gridvision_train/train.py --dry-run
  # real (on the pod):
  python3 scripts/gridvision_train/train.py --epochs 60 --out /workspace/gv_out
"""
import argparse
import json
import os
import sys
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                              # sibling train modules
sys.path.insert(0, os.path.dirname(_HERE))             # scripts/ (gridvision_etdii, _naip_stac)

import cog_reader                # noqa: E402  pure geometry (rasterio guarded inside)
import etdii_to_yolo as e2y      # noqa: E402  pure conversion
import weight_sink               # noqa: E402  pure parsing + guarded upload
import gridvision_etdii as etdii  # noqa: E402  reuse the verified parser

# ETDII zips (figshare 14935434) — file ids VERIFIED via the figshare API
# 2026-07-07 (US) and 2026-07-09 (NZ). US = the 2 in-scope regions; NZ = 5 more
# regions in the SAME geojson format (gridvision_etdii.parse_geojson), added as
# training-DIVERSITY to attack gate-1(a)'s cross-region FAIL. All small (14-120 MB).
ETDII_REGIONS = {
    "USA_AZ_Tucson": 28887792, "USA_KS_Colwich_Maize": 28887795,
    "NZ_Dunedin": 28887777, "NZ_Gisborne": 28887780,
    "NZ_Palmerston_North": 28887783, "NZ_Rotorua": 28887786, "NZ_Tauranga": 28887789,
}
ETDII_US_FILE_IDS = {k: v for k, v in ETDII_REGIONS.items() if k.startswith("USA_")}
ETDII_NDOWNLOAD = "https://ndownloader.figshare.com/files/{fid}"
ETDII_NATIVE_GSD = 0.30
TARGET_GSD = 0.60
IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


# ── pip bootstrap (on-pod) ──────────────────────────────────────────────────

# The RunPod base image ships torch 2.1.0, which is built against NumPy 1.x.
# `pip install ultralytics` pulls NumPy 2.x, and torch 2.1.0 cannot consume a
# 2.x array — the first `torch.from_numpy` (ultralytics' AMP check) dies with
# "RuntimeError: Numpy is not available", killing the run before a single epoch
# (observed on gv-detector-v0-1, 2026-07-09). Pinning numpy<2 in the SAME install
# command keeps the resolver on a torch-2.1-compatible NumPy. Pin lives here (not
# a requirements file) because this is the on-pod bootstrap and needs no image.
NUMPY_PIN = "numpy<2"


def pip_bootstrap(packages=("ultralytics", "rasterio", "pillow")):
    """Install on-pod deps. On-pod only; skipped by --dry-run. Kept here so the
    container needs no custom image. numpy<2 is pinned FIRST so pip resolves a
    NumPy the base image's torch 2.1.0 can actually use (see NUMPY_PIN)."""
    import subprocess
    pkgs = (NUMPY_PIN, *packages)
    print(f"[train] pip installing {pkgs} ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *pkgs])


# ── fetch ETDII US zips (on-pod) ────────────────────────────────────────────

def fetch_etdii_us(dest_dir, regions=None):
    """Download the selected ETDII region zips to dest_dir -> {region: zip_path}.
    On-pod (network). `regions` = iterable of region names (keys of ETDII_REGIONS);
    None => the 2 US regions (back-compat). Unknown names are skipped with a warning
    (never guessed). Uses stdlib urllib."""
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    names = list(regions) if regions else list(ETDII_US_FILE_IDS.keys())
    paths = {}
    for city in names:
        fid = ETDII_REGIONS.get(city)
        if fid is None:
            print(f"[train] WARNING unknown region {city!r} — skipped", flush=True)
            continue
        out = os.path.join(dest_dir, f"etdii_{city}_{fid}.zip")
        if not os.path.exists(out):
            url = ETDII_NDOWNLOAD.format(fid=fid)
            print(f"[train] fetching {city} ({url}) ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": weight_sink.UA})
            with urllib.request.urlopen(req, timeout=600) as r, open(out, "wb") as f:
                f.write(r.read())
        paths[city] = out
    return paths


# ── build the YOLO dataset (on-pod: unzip + read dims + resize + write) ──────

def _index_zip_images(names):
    """Map an image STEM -> its path inside the zip, for sibling-of-geojson lookup.
    Pure helper (list of names in -> dict out)."""
    idx = {}
    for name in names:
        stem, ext = os.path.splitext(os.path.basename(name))
        if ext.lower() in IMG_EXTS:
            idx.setdefault(stem, name)
    return idx


def build_yolo_dataset(zip_paths, out_root, val_frac=0.2, keep_classes=("tower",),
                       tile=640, stride=512, holdout_region=None):
    """Unzip ETDII US, downsample each image 0.30->0.60 m, then TILE it into
    `tile`-px windows at `stride` and write one image+label file per POSITIVE tile
    (>=1 tower). v1 fix for grid-detector-v0's gate-1 failure (experiments.md
    2026-07-09): v0 resized the WHOLE ortho to imgsz 512, collapsing few-px towers
    to ~1 px (recall 0.035). Tiling keeps each tower at its true 0.60 m size because
    the detector trains at imgsz == tile (no second downscale). Tiles inherit their
    PARENT image's train/val side so no scene straddles the split. On-pod (PIL).

    Honest unknown surfaced at runtime: the exact IMAGE filename convention inside
    the ETDII zips is verified ON-POD (the session could not download the 100 MB+
    zips). This matches every image geojson to a sibling raster by STEM across
    common extensions; images with no sibling raster are counted + skipped, never
    guessed."""
    from PIL import Image  # on-pod pip dep
    import io

    img_train = os.path.join(out_root, "images", "train")
    img_val = os.path.join(out_root, "images", "val")
    lbl_train = os.path.join(out_root, "labels", "train")
    lbl_val = os.path.join(out_root, "labels", "val")
    for d in (img_train, img_val, lbl_train, lbl_val):
        os.makedirs(d, exist_ok=True)

    factor = cog_reader.downsample_factor(ETDII_NATIVE_GSD, TARGET_GSD)  # 2.0

    # first pass: collect per-image geojson + sibling image across all zips
    per_image = {}  # stem -> {"records": ..., "img_name": ..., "zip": path}
    skipped_no_image = 0  # geojsons with no sibling raster — counted, never guessed
    for zpath in zip_paths.values():
        with zipfile.ZipFile(zpath) as z:
            names = z.namelist()
            img_idx = _index_zip_images(names)
            for name in names:
                if not name.endswith(".geojson"):
                    continue
                stem = os.path.splitext(os.path.basename(name))[0]
                img_name = img_idx.get(stem)
                if img_name is None:
                    skipped_no_image += 1  # no raster for this geojson -> cannot train on it
                    continue
                try:
                    gj = json.loads(z.read(name))
                except (ValueError, KeyError):
                    continue
                recs = etdii.parse_geojson(gj, "ETDII", "CC BY 4.0")
                per_image[stem] = {"records": recs, "img_name": img_name, "zip": zpath}

    # split by PARENT image (not tile) so tiles from one scene never straddle.
    # holdout_region (gate-1 item a): ALL images of that region go to val, the rest
    # to train — a true cross-region generalization test (train regions never
    # include the eval region). Otherwise the deterministic hash split.
    stems = list(per_image.keys())
    if holdout_region:
        val_ids = [s for s in stems if e2y.region_of(s) == holdout_region]
        train_ids = [s for s in stems if e2y.region_of(s) != holdout_region]
    else:
        train_ids, val_ids = e2y.train_val_split(stems, val_frac)
    split_of = {i: "train" for i in train_ids}
    split_of.update({i: "val" for i in val_ids})

    written = {"train": 0, "val": 0}   # POSITIVE tiles written per split
    boxes = {"train": 0, "val": 0}
    tiles_empty_skipped = 0            # background-only tiles dropped (positive-only v1)
    for stem, info in per_image.items():
        split = split_of[stem]
        with zipfile.ZipFile(info["zip"]) as z:
            raw = z.read(info["img_name"])
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        w0, h0 = im.size
        ow, oh = e2y.downsample_image_dims(w0, h0, factor)
        im = im.resize((ow, oh), Image.BILINEAR)                 # the 0.60 m image
        # tower boxes moved into the 0.60 m pixel space (labels are pixel, so scale)
        boxes06 = [(cid, e2y.scale_bbox(pb, factor))
                   for cid, pb in e2y.image_to_label_records(info["records"], keep_classes)]
        img_dir = img_train if split == "train" else img_val
        lbl_dir = lbl_train if split == "train" else lbl_val
        for (x0, y0, tw, th) in e2y.tile_windows(ow, oh, tile, stride):
            lines = []
            for cid, pb in boxes06:
                ln = e2y.box_in_tile_yolo_line(pb, x0, y0, tw, th, cid)
                if ln is not None:
                    lines.append(ln)
            if not lines:
                tiles_empty_skipped += 1
                continue                                          # positive tiles only
            crop = im.crop((x0, y0, x0 + tw, y0 + th))
            tname = f"{stem}_x{x0}_y{y0}"
            crop.save(os.path.join(img_dir, tname + ".png"))
            with open(os.path.join(lbl_dir, tname + ".txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            written[split] += 1
            boxes[split] += len(lines)

    ymap = e2y.data_yaml_dict(out_root, e2y.CLASS_NAMES)
    yaml_path = os.path.join(out_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(e2y.dump_simple_yaml(ymap))

    return {"data_yaml": yaml_path, "images": written, "boxes": boxes,
            "n_images": len(per_image), "skipped_no_image": skipped_no_image,
            "tiles_empty_skipped": tiles_empty_skipped, "tile": tile, "stride": stride,
            "downsample_factor": factor, "classes": e2y.CLASS_NAMES,
            "holdout_region": holdout_region,
            "split_mode": "holdout_region" if holdout_region else "image_hash",
            "n_train_images": len(train_ids), "n_val_images": len(val_ids)}


# ── optional: materialize NAIP chips for the OSM-weak-label augmentation (A) ─

def materialize_naip_chips(chip_index_path, out_dir, limit=None, target_gsd=TARGET_GSD):
    """Stream NAIP COG windows for chips in a chip index (gridvision_chips output)
    and save them as PNGs. On-pod (rasterio inside cog_reader.read_cog_window).
    OPTIONAL and OFF by default for v0: OSM tower seeds are the documented Phase-B
    gap (build_power_tiles.sh omits power=tower) and OSM boxes are point-weak, so
    v0 trains on ETDII labels; this path exists so the COG reader is real and can
    feed later weak-label / domain-adaptation experiments and inference."""
    import gridvision_naip_stac as naip
    from PIL import Image
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    with open(chip_index_path) as f:
        idx = json.load(f)
    chips = idx.get("chips", [])
    if limit:
        chips = chips[:limit]
    made, failed = 0, 0
    for i, chip in enumerate(chips):
        try:
            items, _ = naip.search(chip["chip_bbox"], *idx["meta"]["naip_date_window"], limit=10)
            best = naip.best_item(items)
            if not best:
                failed += 1
                continue
            signed, _ = naip.sign_href(best["image_href"])
            arr, meta = cog_reader.read_cog_window(signed, chip["chip_bbox"], target_gsd)
            if arr is None:
                failed += 1
                continue
            rgb = np.transpose(arr, (1, 2, 0)).astype("uint8")
            Image.fromarray(rgb).save(os.path.join(out_dir, f"chip_{i}_{chip.get('osm_id','x')}.png".replace("/", "_")))
            made += 1
        except Exception as e:
            print(f"[train] chip {i} failed: {e}", flush=True)
            failed += 1
    return {"chips_made": made, "chips_failed": failed, "chips_total": len(chips)}


# ── training (on-pod: ultralytics) ──────────────────────────────────────────

# STRONG augmentation preset — the cheap cross-region/-domain generalization lever
# the v1/v2 runs never pulled (they used Ultralytics defaults). Rationale for
# gate-1(a)'s held-out FAIL: a model that only saw AZ+KS appearance overfits that
# look. Heavy HSV jitter simulates other regions'/imagery's radiometry (the ortho->
# NAIP and desert->plains gaps); flipud=0.5 is valid for NADIR aerial (no canonical
# "up") and doubles orientation coverage free; scale/rotate/mosaic/mixup/copy_paste
# broaden object context. All are label-preserving for axis-aligned tower boxes.
STRONG_AUG = {
    "hsv_h": 0.02, "hsv_s": 0.8, "hsv_v": 0.5, "degrees": 10.0, "translate": 0.15,
    "scale": 0.6, "shear": 2.0, "perspective": 0.0005, "flipud": 0.5, "fliplr": 0.5,
    "mosaic": 1.0, "mixup": 0.15, "copy_paste": 0.2, "erasing": 0.4,
}


def aug_kwargs(preset):
    """Augmentation kwargs for Ultralytics train() by preset name. 'default' -> {}
    (Ultralytics defaults); 'strong' -> STRONG_AUG. Pure."""
    return dict(STRONG_AUG) if preset == "strong" else {}


def run_training(data_yaml, out_root, epochs=60, imgsz=512, model="yolov8n.pt",
                 device=0, aug="default"):
    """Fine-tune Ultralytics YOLO from COCO weights. On-pod. Returns
    (best_weights_path, metrics dict). `aug` selects the augmentation preset
    (default | strong) — strong is the cross-region generalization lever."""
    from ultralytics import YOLO
    yolo = YOLO(model)  # COCO-pretrained
    results = yolo.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device=device,
                         project=out_root, name="tower_v0", exist_ok=True,
                         verbose=True, **aug_kwargs(aug))
    # locate best.pt
    save_dir = getattr(results, "save_dir", os.path.join(out_root, "tower_v0"))
    best = os.path.join(str(save_dir), "weights", "best.pt")
    metrics = {}
    try:
        rd = getattr(results, "results_dict", {}) or {}
        # normalize the common Ultralytics metric keys to friendly names
        metrics = {
            "map50": rd.get("metrics/mAP50(B)"),
            "map50_95": rd.get("metrics/mAP50-95(B)"),
            "precision": rd.get("metrics/precision(B)"),
            "recall": rd.get("metrics/recall(B)"),
            "raw": rd,
        }
    except Exception as e:
        metrics = {"error": f"metric extraction failed: {e!r}"}
    return best, metrics


# ── dry-run (E): CPU, no GPU / rasterio / ultralytics ───────────────────────

def dry_run():
    """Exercise the WHOLE wiring on CPU with stdlib + repo pure modules only — no
    training, no GPU, no rasterio, no ultralytics. Proves the container is wired
    before a GPU dollar is spent. Returns a summary dict (also printed as JSON)."""
    report = {"dry_run": True, "checks": []}

    def check(name, ok, detail=""):
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        return ok

    # 1) the real labels manifest parses and carries the verified 1408/6 counts
    manifest_path = os.path.join(os.path.dirname(_HERE), "..", "datacore",
                                 "gridvision", "labels_manifest.json")
    towers = subs = None
    try:
        with open(manifest_path) as f:
            man = json.load(f)
        head = man["composition_headline"]["clean_us_labels_available_now"]
        towers, subs = head["tower"], head["substation"]
    except Exception as e:
        check("labels_manifest", False, f"read failed: {e!r} (regenerate with "
              "gridvision_etdii.py --write-manifest)")
    else:
        check("labels_manifest", towers == 1408 and subs == 6,
              f"clean US labels tower={towers} substation={subs} (expect 1408/6)")

    # 2) build a tiny YOLO dataset structure from a synthetic ETDII-shaped sample
    #    (reuses the REAL parser + the REAL converter, just no image bytes)
    sample = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Polygon", "coordinates": [[[-110.9, 32.0], [-110.9, 32.001],
                      [-110.899, 32.001], [-110.899, 32.0], [-110.9, 32.0]]]},
         "properties": {"label": "T", "filename": "USA_AZ_Tucson_1.geojson",
                        "country": "USA", "city/state": "AZ",
                        "pixel_coordinates": [[100, 100], [100, 148], [148, 148], [148, 100]]}},
        {"type": "Feature",  # a substation: in-scope but excluded from tower-v0 dataset
         "geometry": {"type": "Polygon", "coordinates": [[[-110.8, 32.1], [-110.8, 32.11],
                      [-110.79, 32.11], [-110.79, 32.1], [-110.8, 32.1]]]},
         "properties": {"label": "SS", "filename": "USA_AZ_Tucson_1.geojson",
                        "country": "USA", "pixel_coordinates": [[10, 10], [10, 60], [60, 60], [60, 10]]}},
        {"type": "Feature",  # a line: out of scope, must not become a box
         "geometry": {"type": "LineString", "coordinates": [[-110.0, 32.0], [-110.0, 32.01]]},
         "properties": {"label": "L", "filename": "USA_AZ_Tucson_1.geojson", "country": "USA"}},
    ]}
    recs = etdii.parse_geojson(sample, "ETDII", "CC BY 4.0")
    by_img = e2y.group_records_by_image(recs)
    img_w, img_h = 512, 512
    lines = e2y.records_to_yolo_lines(by_img["USA_AZ_Tucson_1.geojson"], img_w, img_h)
    # exactly ONE tower box; substation + line excluded from tower-v0
    check("yolo_conversion", len(lines) == 1 and lines[0].startswith("0 "),
          f"lines={lines}")

    # 3) downsample invariance: normalized YOLO label is identical at 0.30 and 0.60 m
    factor = cog_reader.downsample_factor(ETDII_NATIVE_GSD, TARGET_GSD)
    ow, oh = e2y.downsample_image_dims(img_w, img_h, factor)
    line_ds = e2y.pixel_bbox_to_yolo_line(
        [v / factor for v in [100, 100, 148, 148]], ow, oh, 0)
    check("downsample_invariance", line_ds == lines[0],
          f"factor={factor} dims={ow}x{oh} native='{lines[0]}' downsampled='{line_ds}'")

    # 4) train/val split determinism
    ids = [f"img_{i}" for i in range(50)]
    tr1, va1 = e2y.train_val_split(ids, 0.2)
    tr2, va2 = e2y.train_val_split(list(reversed(ids)), 0.2)
    check("split_determinism", (tr1, va1) == (tr2, va2) and 0 < len(va1) < len(ids),
          f"train={len(tr1)} val={len(va1)} (stable under input reorder)")

    # 5) pure COG window geometry (no rasterio): a north-up UTM-like transform
    transform = cog_reader.north_up_transform(500000.0, 3550000.0, 0.6)  # 0.6 m pixels
    win = cog_reader.pixel_window_from_bounds(
        500100.0, 3549700.0, 500300.0, 3550000.0, transform, raster_size=(10000, 10000))
    out_shape = cog_reader.out_shape_for_window(win[2], win[3], factor)
    check("cog_window_geometry", win[2] > 0 and win[3] > 0 and out_shape[0] > 0,
          f"window(col,row,w,h)={win} out_shape@0.6m={out_shape}")

    # 6) data.yaml serializes as a single-class 'tower' dataset
    ytxt = e2y.dump_simple_yaml(e2y.data_yaml_dict("/tmp/ds"))
    check("data_yaml", "names: ['tower']" in ytxt and "nc: 1" in ytxt, ytxt.strip())

    # 7) v1 tiling: a large frame yields edge-flush windows, and a box maps into
    #    its tile in tile-local normalized coords (the fix for v0's downscale bug)
    wins = e2y.tile_windows(1500, 1000, tile=640, stride=512)
    covers_right = any(x0 + tw == 1500 for (x0, y0, tw, th) in wins)
    covers_bottom = any(y0 + th == 1000 for (x0, y0, tw, th) in wins)
    # a 20px tower at (700,300) in 0.6m space -> lands in the tile starting x=512
    box06 = e2y.scale_bbox([1400, 600, 1440, 640], 2.0)  # -> [700,300,720,320]
    ln = e2y.box_in_tile_yolo_line(box06, 512, 0, 640, 640, 0)
    outside = e2y.box_in_tile_yolo_line(box06, 0, 0, 640, 640, 0)  # box is east of this tile
    check("tiling", bool(wins) and covers_right and covers_bottom
          and box06 == [700.0, 300.0, 720.0, 320.0]
          and ln is not None and ln.startswith("0 ") and outside is None,
          f"n_windows={len(wins)} sample_line={ln!r} outside={outside!r}")

    # 8) weight-sink URL parsing (no network)
    url = weight_sink.parse_upload_url("https://0x0.st/abcd.pt\n")
    check("weight_sink_parse", url == "https://0x0.st/abcd.pt", f"url={url}")

    report["ok"] = all(c["ok"] for c in report["checks"])
    report["summary"] = {"manifest_tower": towers, "manifest_substation": subs,
                         "tower_boxes_from_sample": len(lines),
                         "downsample_factor": factor}
    return report


# ── main ────────────────────────────────────────────────────────────────────

def run_full(args):
    """The real on-pod pipeline. Everything heavy is imported inside the callees."""
    pip_bootstrap()
    work = args.out
    os.makedirs(work, exist_ok=True)
    if args.regions == "all":
        regions = list(ETDII_REGIONS.keys())
    elif args.regions:
        regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    else:
        regions = None  # default: 2 US regions
    print(f"[train] training regions: {regions or list(ETDII_US_FILE_IDS.keys())}", flush=True)
    zips = fetch_etdii_us(os.path.join(work, "etdii"), regions=regions)
    ds = build_yolo_dataset(zips, os.path.join(work, "dataset"), args.val_frac,
                            tile=args.tile, stride=args.stride,
                            holdout_region=args.holdout_region)
    print("[train] dataset:", json.dumps(ds), flush=True)
    if args.imgsz < args.tile:
        # a smaller imgsz would re-downscale the tiles and re-create the v0 bug;
        # train at the tile resolution so towers keep their true 0.60 m size.
        print(f"[train] raising imgsz {args.imgsz}->{args.tile} to match tile (no re-downscale)", flush=True)
        args.imgsz = args.tile
    if args.chip_index:
        chip_report = materialize_naip_chips(
            args.chip_index, os.path.join(work, "naip_chips"), limit=args.chip_limit)
        print("[train] naip chips:", json.dumps(chip_report), flush=True)
    best, metrics = run_training(ds["data_yaml"], work, epochs=args.epochs,
                                 imgsz=args.imgsz, model=args.model, aug=args.aug)
    # write-back (D): push weights out + ALWAYS print metrics to stdout
    sink = {"ok": False, "url": None, "note": "no best.pt found"}
    if best and os.path.exists(best):
        sink = weight_sink.upload_weight(best)
        print("[train] weight upload:", json.dumps(sink), flush=True)
    else:
        print("[train] WARNING: best.pt not found; metrics-only survival.", flush=True)
    # the single machine-parseable survival line (metrics survive even if upload failed)
    print(weight_sink.result_line(metrics, weight_url=sink.get("url"),
                                  extra={"dataset": ds, "weight_sink": sink}), flush=True)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true",
                   help="CPU-only wiring proof: no GPU/rasterio/ultralytics, no training")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tile", type=int, default=640, help="tile window px (v1: detect on native-GSD tiles)")
    p.add_argument("--stride", type=int, default=512, help="tile stride px (< tile => overlap)")
    p.add_argument("--holdout-region", dest="holdout_region", default=None,
                   help="ETDII region name to hold out as val (gate-1 cross-region test); "
                        "e.g. USA_AZ_Tucson or USA_KS_Colwich_Maize")
    p.add_argument("--model", default="yolov8n.pt", help="COCO-pretrained starter")
    p.add_argument("--aug", default="default", choices=["default", "strong"],
                   help="augmentation preset — 'strong' is the cross-region generalization lever")
    p.add_argument("--regions", default=None,
                   help="'all' (7 ETDII regions: 2 US + 5 NZ, for diversity) or a "
                        "comma-list of region names; default = 2 US regions")
    p.add_argument("--val-frac", dest="val_frac", type=float, default=0.2)
    p.add_argument("--out", default="/workspace/gv_out", help="pod work/output dir")
    p.add_argument("--chip-index", default=None,
                   help="optional gridvision_chips index JSON -> materialize NAIP chips (A)")
    p.add_argument("--chip-limit", type=int, default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dry_run:
        report = dry_run()
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["ok"] else 1)
    sys.exit(run_full(args))


if __name__ == "__main__":
    main()
