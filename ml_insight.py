"""
ml_insight — what the model is doing, from the fields the bot already persists.

PRODUCT 2026-10-16 (bot page v2, research/bot_cockpit_design_brief_v2.md §4).
Read-only. Every number here comes from an existing file or function:
  - model age / fallback line: ML_MODEL_PATH mtime vs ML_MAX_AGE_HOURS
  - blend weights: BLEND_TRACKER_PATH (diagnostics.run_diagnostics writes it)
  - live performance: diagnostics.check_model_health()["performance"]
  - recent scoring: the newest shadow records (shadow_portfolio) — the `ml`
    trace persisted since v1.0.1093, the regime block, and per-feature
    constancy over the window (the drift form the audit asked for: variance
    collapse / %-constant, not mean shift)
  - what the score predicts: shadow_portfolio.get_shadow_band_report()
Nothing is estimated; a value the bot does not record is returned as None.
"""
from __future__ import annotations

import os
import time
from collections import Counter

RECENT_WINDOW = 300          # newest shadow records used for the live-scoring view
CONSTANT_FRACTION = 0.98     # a feature whose modal value covers >= 98 % of the window is "constant"


def _model_block(now: float | None = None) -> dict:
    now = now or time.time()
    out = {"exists": False, "age_hours": None, "fallback_at_hours": None, "calibrator_newer_than_model": None}
    try:
        from storage_config import ML_MODEL_PATH
        import storage_config as _sc
        cal_path = getattr(_sc, "ML_CALIBRATOR_PATH", None)
    except Exception:
        ML_MODEL_PATH, cal_path = None, None
    try:
        from system_config import BASE_CONFIG
        out["fallback_at_hours"] = float(BASE_CONFIG.get("ML_MAX_AGE_HOURS", 26))
    except Exception:
        out["fallback_at_hours"] = None
    if ML_MODEL_PATH and os.path.exists(ML_MODEL_PATH):
        out["exists"] = True
        m_mtime = os.path.getmtime(ML_MODEL_PATH)
        out["age_hours"] = round((now - m_mtime) / 3600, 2)
        if cal_path and os.path.exists(cal_path):
            out["calibrator_newer_than_model"] = os.path.getmtime(cal_path) > m_mtime + 1
    return out


def _blend_block() -> dict:
    out = {"rule_weight": None, "ml_weight": None, "updated": None}
    try:
        import json
        from storage_config import BLEND_TRACKER_PATH
        if os.path.exists(BLEND_TRACKER_PATH):
            with open(BLEND_TRACKER_PATH) as f:
                b = json.load(f)
            out = {"rule_weight": b.get("rule_weight"), "ml_weight": b.get("ml_weight"), "updated": b.get("updated")}
    except Exception as e:  # unreadable tracker → weights reported as unknown, never as defaults
        out["error"] = f"{type(e).__name__}: {e}"[:120]
    return out


def recent_scoring(records: list, window: int = RECENT_WINDOW) -> dict:
    """Summarize the newest `window` shadow records: model trace histogram,
    regime inputs, feature constancy. Pure (tested)."""
    recs = [r for r in (records or []) if isinstance(r, dict)][-window:]
    n = len(recs)
    model_types = Counter()
    regimes = Counter()
    gated = 0
    blended = 0
    errors = 0
    traced = 0
    p_vals = []
    for r in recs:
        ml = r.get("ml")
        if not isinstance(ml, dict):
            model_types["not recorded"] += 1
            continue
        traced += 1
        model_types[str(ml.get("model_type") or ("error" if ml.get("error") else "not recorded"))] += 1
        if ml.get("regime_used"):
            regimes[str(ml["regime_used"])] += 1
        if ml.get("gated"):
            gated += 1
        if ml.get("blended"):
            blended += 1
        if ml.get("error"):
            errors += 1
        p = ml.get("win_probability")
        if isinstance(p, (int, float)):
            p_vals.append(float(p))

    # feature constancy over the window (variance collapse, the audit's drift form)
    feats = {}
    for r in recs:
        for k, v in (r.get("features") or {}).items():
            if isinstance(v, (int, float)):
                feats.setdefault(k, []).append(round(float(v), 6))
    feature_rows = []
    for k, vals in sorted(feats.items()):
        c = Counter(vals)
        mode_val, mode_n = c.most_common(1)[0]
        frac = mode_n / len(vals)
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        feature_rows.append({"name": k, "n": len(vals), "mode": mode_val, "mode_fraction": round(frac, 4),
                             "variance": round(var, 6), "constant": frac >= CONSTANT_FRACTION,
                             "last": vals[-1]})
    constant = [f["name"] for f in feature_rows if f["constant"]]

    last = recs[-1] if recs else {}
    regime_labels = Counter(str(r.get("regime_label")) for r in recs)
    vxx = [float(r["vxx_ratio"]) for r in recs if isinstance(r.get("vxx_ratio"), (int, float))]
    p_sorted = sorted(p_vals)

    def _pct(q):
        if not p_sorted:
            return None
        i = min(len(p_sorted) - 1, max(0, int(round(q * (len(p_sorted) - 1)))))
        return round(p_sorted[i], 4)

    return {
        "window": n,
        "traced": traced,
        "model_types": dict(model_types),
        "regime_used": dict(regimes),
        "gated": gated, "blended": blended, "errors": errors,
        "win_probability": {"n": len(p_sorted), "p10": _pct(0.10), "p50": _pct(0.50), "p90": _pct(0.90)},
        "features": feature_rows,
        "constant_features": constant,
        "feature_count": len(feature_rows),
        "regime_labels": dict(regime_labels),
        "vxx_ratio": {"last": vxx[-1] if vxx else None, "min": min(vxx) if vxx else None, "max": max(vxx) if vxx else None},
        "last_timestamp": last.get("timestamp"),
        "last_score": last.get("score"),
    }


def ml_insight(include_bands: bool = True, now: float | None = None) -> dict:
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now or time.time()))}
    out["model"] = _model_block(now)
    out["blend"] = _blend_block()
    try:
        from diagnostics import check_model_health
        h = check_model_health()
        out["performance"] = h.get("performance", {}) if isinstance(h, dict) else {}
    except Exception as e:
        out["performance"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    try:
        from shadow_portfolio import _load_shadow_log
        out["recent"] = recent_scoring(_load_shadow_log())
    except Exception as e:
        out["recent"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    if include_bands:
        try:
            from shadow_portfolio import get_shadow_band_report
            rep = get_shadow_band_report() or {}
            out["bands"] = {k: rep.get(k) for k in ("labeling", "score_quintile_edges", "score_quintile", "decision", "decision_x_band", "band_all") if k in rep}
        except Exception as e:
            out["bands"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(ml_insight(), default=str))
