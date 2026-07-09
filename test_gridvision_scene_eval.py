"""test_gridvision_scene_eval.py — pure core of the SCENE-level evaluator (gate-1
item b): tile-offset parse, IoU, global NMS across seams, greedy matching, AP from
a PR sweep. No torch/rasterio/network."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "gv_scene", os.path.join(os.path.dirname(__file__), "scripts",
                             "gridvision_scene_eval.py"))
se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(se)


def test_parse_tile_name():
    assert se.parse_tile_name("USA_AZ_Tucson_3_x512_y0.png") == ("USA_AZ_Tucson_3", 512, 0)
    assert se.parse_tile_name("USA_KS_Colwich_Maize_7_x0_y1280.png") == ("USA_KS_Colwich_Maize_7", 0, 1280)
    assert se.parse_tile_name("not_a_tile.png") is None


def test_tile_box_to_scene():
    assert se.tile_box_to_scene([10, 20, 30, 40], 512, 128) == [522, 148, 542, 168]


def test_iou():
    assert se.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert se.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    # half-overlap on x: intersection 5x10=50, union 100+100-50=150
    assert abs(se.iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-9


def test_global_nms_kills_seam_duplicate():
    # same tower detected in two overlapping tiles -> near-identical scene boxes
    boxes = [[100, 100, 120, 120], [101, 101, 121, 121], [400, 400, 420, 420]]
    scores = [0.9, 0.8, 0.7]
    kept = se.global_nms(boxes, scores, iou_thr=0.5)
    assert set(kept) == {0, 2}          # the duplicate (idx 1) is suppressed


def test_match_detections_tp_fp():
    gt = [[100, 100, 120, 120], [400, 400, 420, 420]]
    det = [[101, 101, 121, 121], [402, 402, 422, 422], [800, 800, 820, 820]]
    scores = [0.9, 0.8, 0.95]
    tp, fp, nmatch = se.match_detections(det, scores, gt, iou_thr=0.5)
    assert nmatch == 2                  # both GTs matched
    assert sum(tp) == 2 and sum(fp) == 1  # the 800,800 det is a false positive


def test_ap_perfect_and_partial():
    # perfect: 2 dets both TP, 2 GT -> AP 1.0, recall 1.0
    perfect = se.ap_from_pr([1, 1], [0, 0], 2)
    assert perfect["ap"] == 1.0 and perfect["recall_at_best_f1"] == 1.0
    # half recall: 1 TP for 2 GT -> recall caps at 0.5
    half = se.ap_from_pr([1], [0], 2)
    assert half["recall_at_best_f1"] == 0.5
    assert half["ap"] <= 0.5
    # no GT -> AP 0, no crash
    assert se.ap_from_pr([1, 0], [0, 1], 0)["ap"] == 0.0
