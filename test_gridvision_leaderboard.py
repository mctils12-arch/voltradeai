"""test_gridvision_leaderboard.py — the accuracy-ratchet gate (Phase 2). Pure
functions only; no file IO, no network. The gate must: treat a first measurement
as baseline, block a real regression, pass a within-noise wobble, and flag a
suspiciously large jump for investigation rather than silently trusting it."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "gv_leaderboard", os.path.join(os.path.dirname(__file__), "scripts",
                                   "gridvision_leaderboard.py"))
lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb)


def _rec(version, key, map50):
    return {"version": version, "eval_key": key, "metrics": {"map50": map50}}


RECS = [_rec("v0", "image_split", 0.0358), _rec("v1", "image_split", 0.566),
        _rec("a", "holdout:USA_AZ_Tucson", 0.31)]


def test_best_for_key_and_champion():
    assert lb.best_for_key(RECS, "image_split") == 0.566
    assert lb.best_for_key(RECS, "holdout:USA_AZ_Tucson") == 0.31
    assert lb.best_for_key(RECS, "nope") is None
    assert lb.champion(RECS, "image_split")["version"] == "v1"


def test_regression_blocks_a_real_drop():
    v = lb.regression_check(RECS, "image_split", 0.50)   # 0.50 < 0.566 - 0.02
    assert v["status"] == "regression"


def test_within_noise_is_not_a_regression():
    v = lb.regression_check(RECS, "image_split", 0.555)  # within 0.02 of 0.566
    assert v["status"] == "within_noise"


def test_baseline_when_key_is_new():
    v = lb.regression_check(RECS, "scene:image_split", 0.42)
    assert v["status"] == "baseline" and v["prior_best"] is None


def test_suspicious_jump_is_flagged_not_celebrated():
    # v0->v1 style jump on a fresh low baseline
    recs = [_rec("v0", "k", 0.03)]
    v = lb.regression_check(recs, "k", 0.55)
    assert v["status"] == "suspicious_jump"
    assert "INVESTIGATE" in v["detail"]


def test_normal_improvement():
    v = lb.regression_check(RECS, "image_split", 0.58)   # +0.014, under suspicious
    assert v["status"] == "improved"
