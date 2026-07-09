#!/usr/bin/env python3
"""gridvision_scene_eval.py — GRID VISION gate-1 item (b): SCENE-level accuracy.

The tile-level mAP Ultralytics reports counts a tower that straddles two
overlapping tiles twice and never penalizes a seam-split miss. Production runs on
whole scenes, so the honest number stitches tile detections back to scene-pixel
coords, de-duplicates across seams with global NMS, matches to scene ground truth
at IoU 0.5, and sweeps confidence for a PR curve -> scene AP50 + P/R at best F1.

This module is the PURE, offline-tested core (tile-offset parse, IoU, global NMS,
greedy IoU matching, AP from a PR sweep). The on-pod driver (loads the .pt, runs
inference per tile) feeds these — kept out of here so the whole thing unit-tests
with no torch/rasterio. Tile filenames are `{stem}_x{X}_y{Y}.png` (train.py).
"""
import os
import re

_TILE_RE = re.compile(r"^(?P<stem>.+)_x(?P<x>\d+)_y(?P<y>\d+)$")


def parse_tile_name(filename):
    """`USA_AZ_Tucson_3_x512_y0.png` -> ('USA_AZ_Tucson_3', 512, 0), or None if the
    name is not a tile. Pure."""
    base = os.path.splitext(os.path.basename(filename))[0]
    m = _TILE_RE.match(base)
    if not m:
        return None
    return m.group("stem"), int(m.group("x")), int(m.group("y"))


def tile_box_to_scene(box_xyxy, x0, y0):
    """A [xmin,ymin,xmax,ymax] box in TILE-local pixels -> scene pixels by adding
    the tile origin. Pure."""
    return [box_xyxy[0] + x0, box_xyxy[1] + y0, box_xyxy[2] + x0, box_xyxy[3] + y0]


def iou(a, b):
    """IoU of two [xmin,ymin,xmax,ymax] boxes. Pure."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def global_nms(boxes, scores, iou_thr=0.5):
    """Greedy non-max suppression across a whole scene (kills seam duplicates from
    overlapping tiles). Returns kept indices, highest score first. Pure."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    kept = []
    while order:
        i = order.pop(0)
        kept.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thr]
    return kept


def match_detections(det_boxes, det_scores, gt_boxes, iou_thr=0.5):
    """Greedy score-ordered matching of detections to ground truth at iou_thr.
    Each GT matches at most one detection. Returns (tp_flags, fp_flags, n_matched_gt)
    where tp_flags[i]/fp_flags[i] align with detections sorted by DESCENDING score.
    Pure — the scoring/AP layer consumes these."""
    order = sorted(range(len(det_boxes)), key=lambda i: det_scores[i], reverse=True)
    used = set()
    tp, fp = [], []
    for i in order:
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gt_boxes):
            if j in used:
                continue
            v = iou(det_boxes[i], g)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            used.add(best_j)
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    return tp, fp, len(used)


def ap_from_pr(tp_flags, fp_flags, n_gt):
    """Average Precision (VOC-style, all-points interpolation) from score-ordered
    TP/FP flags and the total GT count. Also returns precision/recall at the point
    of best F1. Pure. Empty GT -> AP 0."""
    if n_gt <= 0:
        return {"ap": 0.0, "best_f1": 0.0, "precision_at_best_f1": 0.0,
                "recall_at_best_f1": 0.0, "n_gt": 0, "n_det": len(tp_flags)}
    tp_cum = fp_cum = 0
    precisions, recalls = [], []
    best_f1 = best_p = best_r = 0.0
    for tp, fp in zip(tp_flags, fp_flags):
        tp_cum += tp
        fp_cum += fp
        p = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) else 0.0
        r = tp_cum / n_gt
        precisions.append(p)
        recalls.append(r)
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        if f1 > best_f1:
            best_f1, best_p, best_r = f1, p, r
    # all-points interpolation: precision envelope from the right
    ap = 0.0
    prev_r = 0.0
    # make precision monotone non-increasing from the end
    for k in range(len(precisions) - 2, -1, -1):
        precisions[k] = max(precisions[k], precisions[k + 1])
    for k in range(len(recalls)):
        dr = recalls[k] - prev_r
        if dr > 0:
            ap += precisions[k] * dr
            prev_r = recalls[k]
    return {"ap": round(ap, 6), "best_f1": round(best_f1, 6),
            "precision_at_best_f1": round(best_p, 6),
            "recall_at_best_f1": round(best_r, 6),
            "n_gt": n_gt, "n_det": len(tp_flags)}
