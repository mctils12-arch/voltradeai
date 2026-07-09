#!/usr/bin/env python3
"""gridvision_leaderboard.py — GRID VISION accuracy ratchet (Phase 2).

The permanent, honest record of every tower-detector model version and its
measured metrics, plus the regression GATE that makes accuracy monotonic: a new
model may become champion only if it does not regress ANY previously-measured
eval key beyond a small noise band. A regression fails the build (exit 2).

An "eval key" identifies WHAT was measured so only like-for-like is compared:
  in-domain image-split  -> key "image_split"
  held-out region X      -> key "holdout:<region>"
  scene-level, split S   -> key "scene:<S>"
Each record carries its metrics under one eval key. `check` compares a candidate
against the BEST prior value for the same key.

HONESTY: metrics are whatever the run measured — never hand-edited up. A
suspiciously large jump (> SUSPICIOUS_JUMP) is flagged for investigation (a
circular/leaky eval is the usual cause), not celebrated.

Store: datacore/gridvision/leaderboard.json  (list of records, append-only in
spirit; git keeps history). Pure functions do the ranking/gating; file IO is thin.

CLI:
  python3 scripts/gridvision_leaderboard.py show
  python3 scripts/gridvision_leaderboard.py add --version v1 --job gv-detector-v1-2 \
      --eval-key image_split --model yolov8n --map50 0.566 --recall 0.499 \
      --precision 0.732 --notes "tiling 640/512, both regions in train"
  python3 scripts/gridvision_leaderboard.py check --eval-key image_split --map50 0.59
"""
import argparse
import json
import os
import sys

BOARD = os.path.join(os.path.dirname(__file__), "..", "datacore", "gridvision",
                     "leaderboard.json")
NOISE_BAND = 0.02        # a drop within this of the prior best is not a regression
SUSPICIOUS_JUMP = 0.30   # an improvement larger than this on one key -> investigate


# ── pure ranking / gating ───────────────────────────────────────────────────

def best_for_key(records, eval_key, metric="map50"):
    """Highest value of `metric` among records with this eval_key, or None."""
    vals = [r["metrics"].get(metric) for r in records
            if r.get("eval_key") == eval_key and r.get("metrics", {}).get(metric) is not None]
    return max(vals) if vals else None


def regression_check(records, eval_key, value, metric="map50",
                     noise=NOISE_BAND, suspicious=SUSPICIOUS_JUMP):
    """Compare a candidate `value` for `eval_key` against the prior best. Returns a
    dict verdict. Pure — the CLI turns 'regression' into a nonzero exit."""
    prior = best_for_key(records, eval_key, metric)
    if prior is None:
        return {"eval_key": eval_key, "metric": metric, "value": value,
                "prior_best": None, "status": "baseline",
                "detail": "first measurement for this key — recorded as baseline"}
    delta = round(value - prior, 6)
    if delta < -noise:
        status = "regression"
        detail = f"{metric} {value} < prior best {prior} by {-delta:.4f} (> {noise} band)"
    elif delta > suspicious:
        status = "suspicious_jump"
        detail = (f"{metric} +{delta:.4f} over {prior} exceeds {suspicious} — "
                  "INVESTIGATE for a leaky/circular eval before trusting")
    elif delta > 0:
        status = "improved"
        detail = f"{metric} +{delta:.4f} over prior best {prior}"
    else:
        status = "within_noise"
        detail = f"{metric} {value} within {noise} of prior best {prior}"
    return {"eval_key": eval_key, "metric": metric, "value": value,
            "prior_best": prior, "delta": delta, "status": status, "detail": detail}


def champion(records, eval_key="image_split", metric="map50"):
    """The record holding the best `metric` for `eval_key`, or None. Pure."""
    cand = [r for r in records if r.get("eval_key") == eval_key
            and r.get("metrics", {}).get(metric) is not None]
    return max(cand, key=lambda r: r["metrics"][metric]) if cand else None


# ── thin IO + CLI ───────────────────────────────────────────────────────────

def load(path=BOARD):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save(records, path=BOARD):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
        f.write("\n")


def _metrics_from_args(a):
    m = {}
    for k in ("map50", "map50_95", "precision", "recall", "scene_map50",
              "scene_precision", "scene_recall", "naip_recall_vs_osm"):
        v = getattr(a, k, None)
        if v is not None:
            m[k] = v
    return m


def _cmd_add(a):
    recs = load()
    rec = {"version": a.version, "job": a.job, "eval_key": a.eval_key,
           "model": a.model, "metrics": _metrics_from_args(a),
           "timestamp_utc": a.timestamp, "notes": a.notes or ""}
    ver = regression_check(recs, a.eval_key, rec["metrics"].get(a.metric),
                           a.metric) if rec["metrics"].get(a.metric) is not None else None
    rec["regression_verdict"] = ver
    recs.append(rec)
    save(recs)
    print(json.dumps({"added": rec}, indent=2))


def _cmd_check(a):
    ver = regression_check(load(), a.eval_key, a.map50, "map50")
    print(json.dumps(ver, indent=2))
    sys.exit(2 if ver["status"] == "regression" else 0)


def _cmd_show(a):
    recs = load()
    keys = sorted({r.get("eval_key") for r in recs})
    for k in keys:
        ch = champion(recs, k)
        print(f"[{k}] champion: {ch['version'] if ch else '-'} "
              f"map50={ch['metrics'].get('map50') if ch else '-'}")
    for r in recs:
        m = r.get("metrics", {})
        print(f"  {r.get('version'):>6}  {r.get('eval_key'):<22} "
              f"map50={m.get('map50')} recall={m.get('recall')} "
              f"scene_map50={m.get('scene_map50')}  {r.get('notes','')}")


def build_parser():
    p = argparse.ArgumentParser(description="GRID VISION detector leaderboard + regression gate")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    ca = sub.add_parser("add")
    ca.add_argument("--version", required=True)
    ca.add_argument("--job", default="")
    ca.add_argument("--eval-key", dest="eval_key", required=True)
    ca.add_argument("--model", default="yolov8n")
    ca.add_argument("--metric", default="map50")
    for k in ("map50", "map50_95", "precision", "recall", "scene_map50",
              "scene_precision", "scene_recall", "naip_recall_vs_osm"):
        ca.add_argument(f"--{k}", type=float, default=None)
    ca.add_argument("--timestamp", default="")
    ca.add_argument("--notes", default="")
    cc = sub.add_parser("check")
    cc.add_argument("--eval-key", dest="eval_key", required=True)
    cc.add_argument("--map50", type=float, required=True)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    {"show": _cmd_show, "add": _cmd_add, "check": _cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
