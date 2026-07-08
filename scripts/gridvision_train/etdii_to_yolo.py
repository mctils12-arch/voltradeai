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
