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

[REPAIR] 2026-11-03 (approved item B3, serve_features.py) — the constancy
MEASURE is split by what it can honestly judge. `constant_features` keeps its
original definition (every feature whose modal value covers >= 98 % of the
newest RECENT_WINDOW records — "constant within the window"), but that
definition counts three different things as one:
  - per-ticker features that never move at serve while training varied them
    (the serve-path defect the panel is for),
  - market-wide features, which take ONE value per macro snapshot by nature —
    identical for every candidate of a scan, so a per-record mode fraction
    over a few hours of scans measures the scan cadence; they are judged
    ACROSS DAYS over the wider MARKET_WIDE_WINDOW instead (a stale/default
    reading, e.g. credit_spread pinned at 0.0, is constant across days),
  - training-constant features (0.0 / 0.5 on every training row by
    construction — serve_features.TRAINING_CONSTANT_FEATURES), which never
    "varied in training" and so cannot be a serve-path defect.
`constant_features_serve_bug` is the honest headline (first + second kinds,
each by its own measure); the raw list stays so nothing is hidden.
`constant_features_fixed` says how many of those would remain constant under
the corrected vector the fix records (`serve_fix` on the record).
"""
from __future__ import annotations

import os
import time
from collections import Counter

RECENT_WINDOW = 300          # newest shadow records used for the live-scoring view
CONSTANT_FRACTION = 0.98     # a feature whose modal value covers >= 98 % of the window is "constant"
MARKET_WIDE_WINDOW = 3000    # newest records used to judge MARKET-WIDE features across days (~a week of scans)


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


def _day(ts) -> str | None:
    s = str(ts or "")
    return s[:10] if len(s) >= 10 else None


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def _mode_fraction(vals: list) -> tuple:
    """(modal value, modal fraction, distinct count) of a non-empty list."""
    c = Counter(vals)
    mode_val, mode_n = c.most_common(1)[0]
    return mode_val, mode_n / len(vals), len(c)


def fixed_vector(rec: dict) -> dict | None:
    """The corrected feature vector the serve fix computed for this record, or
    None (no `serve_fix`, or the fix errored). `features` on a record is the
    vector the SERVED score saw: when the corrected one was served the record
    already holds it; otherwise it is `features` + `features_fixed_diff`."""
    sf = rec.get("serve_fix")
    feats = rec.get("features")
    if not isinstance(sf, dict) or not isinstance(feats, dict) or sf.get("error"):
        return None
    if sf.get("served") == "fixed":
        return feats
    diff = sf.get("features_fixed_diff")
    return {**feats, **diff} if isinstance(diff, dict) else dict(feats)


def _pct(sorted_vals: list, q: float):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return round(sorted_vals[i], 4)


def recent_scoring(records: list, window: int = RECENT_WINDOW,
                   market_wide_window: int = MARKET_WIDE_WINDOW) -> dict:
    """Summarize the newest `window` shadow records: model trace histogram,
    regime inputs, feature constancy. Pure (tested)."""
    from serve_features import feature_kind, MARKET_WIDE_FEATURES

    all_recs = [r for r in (records or []) if isinstance(r, dict)]
    recs = all_recs[-window:]
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
    feat_days = {}
    for r in recs:
        d = _day(r.get("timestamp"))
        for k, v in (r.get("features") or {}).items():
            if isinstance(v, (int, float)):
                feats.setdefault(k, []).append(round(float(v), 6))
                if d:
                    feat_days.setdefault(k, set()).add(d)

    # market-wide features: one value per macro snapshot by nature, so they are
    # judged ACROSS DAYS over the wider window (served values and, where the
    # fix recorded one, corrected values)
    mw_recs = all_recs[-market_wide_window:]
    mw_vals: dict = {}
    mw_fixed: dict = {}
    mw_days: set = set()
    for r in mw_recs:
        d = _day(r.get("timestamp"))
        if d:
            mw_days.add(d)
        f = r.get("features") or {}
        fv = fixed_vector(r)
        for k in MARKET_WIDE_FEATURES:
            v = f.get(k) if isinstance(f, dict) else None
            if _is_num(v):
                mw_vals.setdefault(k, []).append((d, round(float(v), 6)))
            if fv is not None and _is_num(fv.get(k)):
                mw_fixed.setdefault(k, []).append((d, round(float(fv[k]), 6)))

    # the serve fix's corrected vectors over the recent window (shadow-first
    # counterfactual: what the model would have said on the corrected inputs)
    fixed_vals: dict = {}
    fix_n = 0
    fix_modes = Counter()
    fix_served = Counter()
    fix_errors = 0
    deltas = []
    for r in recs:
        sf = r.get("serve_fix")
        if not isinstance(sf, dict):
            continue
        fix_n += 1
        fix_modes[str(sf.get("mode"))] += 1
        fix_served[str(sf.get("served"))] += 1
        if sf.get("error"):
            fix_errors += 1
        pc, pf = sf.get("ml_prob_current"), sf.get("ml_prob_fixed")
        if _is_num(pc) and _is_num(pf):
            deltas.append(float(pf) - float(pc))
        fv = fixed_vector(r)
        if fv is not None:
            for k, v in fv.items():
                if _is_num(v):
                    fixed_vals.setdefault(k, []).append(round(float(v), 6))

    feature_rows = []
    for k, vals in sorted(feats.items()):
        mode_val, frac, _distinct = _mode_fraction(vals)
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        kind = feature_kind(k)
        row = {"name": k, "n": len(vals), "mode": mode_val, "mode_fraction": round(frac, 4),
               "variance": round(var, 6), "constant": frac >= CONSTANT_FRACTION,
               "last": vals[-1], "kind": kind, "days": len(feat_days.get(k, ())),
               "judgeable": False, "constant_bug": False, "fixed_constant": None}
        if kind == "per_ticker":
            row["judgeable"] = True
            row["constant_bug"] = row["constant"]
            fv = fixed_vals.get(k)
            if fv:
                row["fixed_constant"] = _mode_fraction(fv)[1] >= CONSTANT_FRACTION
        elif kind == "market_wide":
            mv = mw_vals.get(k, [])
            days_across = {d for d, _ in mv if d}
            row["days_across"] = len(days_across)
            row["judgeable"] = len(days_across) >= 2
            if mv:
                _m, mfrac, mdistinct = _mode_fraction([v for _, v in mv])
                row["mode_fraction_across_days"] = round(mfrac, 4)
                row["distinct_values_across_days"] = mdistinct
                row["constant_bug"] = bool(row["judgeable"] and mfrac >= CONSTANT_FRACTION)
            fx = mw_fixed.get(k, [])
            if len({d for d, _ in fx if d}) >= 2:
                row["fixed_constant"] = _mode_fraction([v for _, v in fx])[1] >= CONSTANT_FRACTION
        # training_constant: never varied in training -> not judgeable, never a serve bug
        feature_rows.append(row)
    constant = [f["name"] for f in feature_rows if f["constant"]]
    serve_bug = [f["name"] for f in feature_rows if f["constant_bug"]]
    mw_window_constant = [f["name"] for f in feature_rows if f["kind"] == "market_wide" and f["constant"]]
    training_constant = [f["name"] for f in feature_rows if f["kind"] == "training_constant" and f["constant"]]
    fixed_constant = [f["name"] for f in feature_rows if f["constant_bug"] and f["fixed_constant"]] if fix_n else None
    fixed_unknown = [f["name"] for f in feature_rows if f["constant_bug"] and f["fixed_constant"] is None] if fix_n else None

    last = recs[-1] if recs else {}
    regime_labels = Counter(str(r.get("regime_label")) for r in recs)
    vxx = [float(r["vxx_ratio"]) for r in recs if isinstance(r.get("vxx_ratio"), (int, float))]
    p_sorted = sorted(p_vals)
    d_sorted = sorted(deltas)
    window_days = {d for d in (_day(r.get("timestamp")) for r in recs) if d}

    return {
        "window": n,
        "traced": traced,
        "model_types": dict(model_types),
        "regime_used": dict(regimes),
        "gated": gated, "blended": blended, "errors": errors,
        "win_probability": {"n": len(p_sorted), "p10": _pct(p_sorted, 0.10), "p50": _pct(p_sorted, 0.50), "p90": _pct(p_sorted, 0.90)},
        "features": feature_rows,
        "constant_features": constant,
        "feature_count": len(feature_rows),
        # [REPAIR] 2026-11-03 (B3): the honest split of `constant_features`
        "constant_features_serve_bug": serve_bug,
        "constant_features_market_wide_window": mw_window_constant,
        "constant_features_training_constant": training_constant,
        "constant_features_fixed": fixed_constant,
        "constant_features_fixed_unknown": fixed_unknown,
        "judgeable_feature_count": sum(1 for f in feature_rows if f["judgeable"]),
        "window_days": len(window_days),
        "market_wide_window": len(mw_recs),
        "market_wide_days": len(mw_days),
        "serve_fix": {
            "records": fix_n, "modes": dict(fix_modes), "served": dict(fix_served), "errors": fix_errors,
            "prob_delta": {"n": len(d_sorted),
                           "mean": round(sum(d_sorted) / len(d_sorted), 4) if d_sorted else None,
                           "p10": _pct(d_sorted, 0.10), "p50": _pct(d_sorted, 0.50), "p90": _pct(d_sorted, 0.90)},
        },
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
