#!/usr/bin/env python3
"""etdii_to_yolo.py — GRID VISION Phase B: ETDII US labels -> YOLO detector dataset.

Turns the CC-BY ETDII US tower labels (parsed by scripts/gridvision_etdii.py) into
a YOLO-format detection dataset: images/{train,val}/ + labels/{train,val}/*.txt
(one normalized `cls cx cy w h` line per box) + data.yaml with a single `tower`
class. Reuses gridvision_etdii.pixel_bbox / parse_geojson — the pixel_coordinates
quad ETDII ships on every feature IS the detection box; no parse logic is
re-implemented here.

HONEST v0 SCOPE (research/grid_vision_phaseb.md): TOWER-ONLY. The US ETDII set has
1408 clean towers but only 6 substation polygons across all 74 images (~0.4%) —
NOT a trainable substation class, so substation is deliberately excluded from the
v0 dataset (folding 6 labels in would fake a class). keep_classes defaults to
{"tower"}; the caller may widen it, but the docstring says why it should not yet.

GSD / DOWNSAMPLE NOTE (important, and why the labels don't change): ETDII imagery
is USGS ortho @0.30 m; the detector trains at 0.60 m to match NAIP. Downsampling
0.30->0.60 m HALVES the image pixel dimensions, but YOLO boxes are NORMALIZED by
image width/height — box and dims scale together, so the normalized label is
IDENTICAL before and after downsample. The resize happens to the IMAGE bytes
(on-pod, PIL); the label math here is scale-invariant. downsample_image_dims()
gives the resized pixel dims for that image resize; a test pins the invariance.

PURITY: every function in this file is pure and offline-tested. The only on-pod
steps (unzip, read image dims, PIL-resize, write files) live in the build_*
orchestration in train.py, never here.
"""
import hashlib
import os
import sys

# reuse the verified ETDII parser (pixel_bbox / parse_geojson) — no duplicate parse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
import gridvision_etdii as etdii  # noqa: E402

# v0 class table: tower only (index 0). Substation intentionally absent (6 labels).
CLASS_INDEX = {"tower": 0}
CLASS_NAMES = ["tower"]

# ETDII US region prefixes — the image stems begin with these. Used for the
# held-out-REGION generalization split (gate-1 item a): train on some regions,
# evaluate on a region never seen. Longest-prefix match so a more specific region
# name wins. Extend as more regions (Duke-US zips, new states) are added.
REGION_PREFIXES = ["USA_AZ_Tucson", "USA_KS_Colwich_Maize"]


def region_of(stem, prefixes=REGION_PREFIXES):
    """Region a training image belongs to, by longest matching prefix of its stem
    (e.g. 'USA_AZ_Tucson_12' -> 'USA_AZ_Tucson'). Returns None if no known region
    matches — such an image is never silently bucketed. Pure."""
    best = None
    for p in prefixes:
        if stem.startswith(p) and (best is None or len(p) > len(best)):
            best = p
    return best


# ── pure box conversion ─────────────────────────────────────────────────────

def pixel_bbox_to_yolo_line(pixel_bbox, img_w, img_h, cls_id=0):
    """[xmin,ymin,xmax,ymax] pixel box + image dims -> a YOLO label line
    'cls cx cy w h' (normalized, 6 dp), or None if the box is degenerate or falls
    entirely outside the image. The box is clamped to the image before
    normalization so a slightly-overhanging annotation still yields a valid line.
    Pure."""
    if not pixel_bbox or len(pixel_bbox) != 4:
        return None
    if not img_w or not img_h or img_w <= 0 or img_h <= 0:
        return None
    xmin, ymin, xmax, ymax = (float(v) for v in pixel_bbox)
    # clamp to the image extent
    xmin = min(max(xmin, 0.0), img_w)
    xmax = min(max(xmax, 0.0), img_w)
    ymin = min(max(ymin, 0.0), img_h)
    ymax = min(max(ymax, 0.0), img_h)
    if xmax <= xmin or ymax <= ymin:
        return None  # degenerate or fully outside -> skip, never emit a zero box
    cx = ((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    # numeric safety: keep everything inside [0,1]
    cx, cy, w, h = (min(max(v, 0.0), 1.0) for v in (cx, cy, w, h))
    return f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def downsample_image_dims(img_w, img_h, factor):
    """Resized (out_w, out_h) when an image is downsampled by `factor`
    (native-px per out-px; 2.0 for 0.30->0.60 m). Each dim >= 1. Pure. The YOLO
    labels do NOT change under this resize (they are normalized); this only sizes
    the PIL resize the on-pod builder performs."""
    if factor <= 0:
        raise ValueError("factor must be > 0")
    return (max(1, int(round(img_w / factor))), max(1, int(round(img_h / factor))))


# ── tiling (v1: detect on native-GSD windows, never a downscaled full frame) ──
#
# v0 (grid-detector-v0-2, experiments.md 2026-07-09) resized each WHOLE ortho to
# imgsz 512 and scored AP50 0.036 / recall 0.035 — dense few-px towers collapsed
# to ~1 px. v1 slides a fixed-pixel window over the 0.60 m image and trains on the
# windows at their own resolution (imgsz == tile), so a tower keeps its true 0.60 m
# size. Both helpers below are pure and offline-tested; train.py does the crop/IO.

def tile_windows(img_w, img_h, tile=640, stride=512):
    """Top-left origins tiling an (img_w,img_h) image with `tile` windows at
    `stride`. Always appends an edge-flush window on each axis so no right/bottom
    strip is dropped, and clamps the last window's size at the border. Returns a
    list of (x0, y0, tw, th) with tw,th <= tile. An image smaller than `tile` on an
    axis yields a single full-extent window on that axis. Pure."""
    if tile <= 0 or stride <= 0:
        raise ValueError("tile and stride must be > 0")
    if img_w <= 0 or img_h <= 0:
        return []

    def starts(extent):
        if extent <= tile:
            return [0]
        s = list(range(0, extent - tile + 1, stride))
        if s[-1] != extent - tile:
            s.append(extent - tile)  # flush the final window to the far edge
        return s

    out = []
    for y0 in starts(img_h):
        for x0 in starts(img_w):
            out.append((x0, y0, min(tile, img_w - x0), min(tile, img_h - y0)))
    return out


def box_in_tile_yolo_line(pixel_bbox, tile_x0, tile_y0, tile_w, tile_h, cls_id=0,
                          min_visible=0.3):
    """Clip a [xmin,ymin,xmax,ymax] box (image-pixel coords) to a tile window and,
    if at least `min_visible` of the box AREA lands inside the tile, return a YOLO
    line normalized to the tile (via pixel_bbox_to_yolo_line), else None. Dropping
    barely-clipped boxes keeps edge fragments from becoming mislabelled tiny boxes.
    Pure."""
    if not pixel_bbox or len(pixel_bbox) != 4:
        return None
    xmin, ymin, xmax, ymax = (float(v) for v in pixel_bbox)
    ix0 = max(xmin, tile_x0)
    iy0 = max(ymin, tile_y0)
    ix1 = min(xmax, tile_x0 + tile_w)
    iy1 = min(ymax, tile_y0 + tile_h)
    if ix1 <= ix0 or iy1 <= iy0:
        return None  # no overlap with this tile
    box_area = (xmax - xmin) * (ymax - ymin)
    if box_area <= 0 or ((ix1 - ix0) * (iy1 - iy0)) / box_area < min_visible:
        return None  # too little of the box is visible here
    # translate the clipped box into tile-local coords, then normalize to the tile
    return pixel_bbox_to_yolo_line(
        [ix0 - tile_x0, iy0 - tile_y0, ix1 - tile_x0, iy1 - tile_y0],
        tile_w, tile_h, cls_id)


def scale_bbox(pixel_bbox, factor):
    """Scale an original-resolution pixel box into the downsampled (0.60 m) space
    by dividing every coord by `factor` (2.0 for 0.30->0.60 m). Pure."""
    if factor <= 0:
        raise ValueError("factor must be > 0")
    return [float(v) / float(factor) for v in pixel_bbox]


def image_to_label_records(records, keep_classes=("tower",)):
    """From gridvision_etdii.parse_geojson() records for ONE image, keep only the
    in-scope, kept-class boxes that actually carry a pixel_bbox. Returns a list of
    (cls_id, pixel_bbox). Pure — filters, never fabricates a box."""
    keep = set(keep_classes)
    out = []
    for r in records:
        if not r.get("in_scope"):
            continue
        cls = r.get("cls")
        if cls not in keep or cls not in CLASS_INDEX:
            continue
        pb = r.get("pixel_bbox")
        if not pb:
            continue  # a feature with no pixel quad cannot be a detection box
        out.append((CLASS_INDEX[cls], pb))
    return out


def records_to_yolo_lines(records, img_w, img_h, keep_classes=("tower",)):
    """ETDII records for one image -> list of YOLO label lines (skips boxes that
    convert to None). Pure."""
    lines = []
    for cls_id, pb in image_to_label_records(records, keep_classes):
        line = pixel_bbox_to_yolo_line(pb, img_w, img_h, cls_id)
        if line is not None:
            lines.append(line)
    return lines


def group_records_by_image(records):
    """All parsed records -> {image_ref: [records]} (records with no image_ref are
    dropped — they cannot be attached to a training image). Pure, order-stable."""
    by_img = {}
    for r in records:
        ref = r.get("image_ref")
        if not ref:
            continue
        by_img.setdefault(ref, []).append(r)
    return by_img


# ── deterministic train/val split ───────────────────────────────────────────

def assign_split(image_id, val_frac=0.2, salt="gridvision-tower-v0"):
    """Deterministically assign an image to 'train' or 'val' by hashing its id.
    Hash-based (not order/random-state based) so the SAME image always lands on
    the SAME side regardless of input order, dict iteration, or Python version —
    reproducibility the promotion ladder needs. Pure."""
    if val_frac <= 0:
        return "train"
    if val_frac >= 1:
        return "val"
    h = hashlib.md5((salt + "|" + str(image_id)).encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 10000  # 4-hex-digit bucket in [0,9999]
    return "val" if bucket < int(round(val_frac * 10000)) else "train"


def train_val_split(image_ids, val_frac=0.2, salt="gridvision-tower-v0"):
    """image ids -> (sorted train ids, sorted val ids), deterministic. Pure."""
    train, val = [], []
    for iid in image_ids:
        (val if assign_split(iid, val_frac, salt) == "val" else train).append(iid)
    return sorted(train), sorted(val)


# ── data.yaml ───────────────────────────────────────────────────────────────

def data_yaml_dict(dataset_root, names=CLASS_NAMES):
    """The Ultralytics data.yaml mapping for the built dataset. Pure (dict in ->
    dict out); train.py serializes it. Paths are relative to dataset_root, the
    layout Ultralytics expects (images/train, images/val)."""
    return {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": list(names),
    }


def dump_simple_yaml(mapping):
    """Serialize the flat data.yaml mapping WITHOUT a yaml dependency (values are
    str/int/list-of-str only). Pure. Ultralytics parses this fine."""
    lines = []
    for k, v in mapping.items():
        if isinstance(v, list):
            inner = ", ".join(f"'{x}'" for x in v)
            lines.append(f"{k}: [{inner}]")
        elif isinstance(v, int):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines) + "\n"
