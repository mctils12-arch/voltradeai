"""
score_composite.py — deep_score's composite v2 (2026-11-06, [RULE-REVIEW],
EXPLORATORY, kill date 2027-02-05; human refocus spec section 7 "cut what does
not work"). Configured by system_config.BASE_CONFIG["SCORE_COMPOSITE_V2"].

deep_score computes the LEGACY composite exactly as before (and logs it as
score_breakdown.composite_legacy). v2 starts from the same pre-clamp rules
total and:
  * holds the `neutralize` terms at their neutral values — the volume
    sub-score at 50 (its 15% weight then adds nothing either way), the
    vol-premium bonus and the relative-strength bonus at 0. Evidence: the
    2017-2026 point-in-time walk-forward (walkforward.py, 513 weekly dates,
    ~2,060 names a week, delisted names included) found each one's 5-day
    rank IC significantly NEGATIVE after Bonferroni (-0.015 / -0.017 /
    -0.020), the same sign at 20 days, and its top-10 picks behind random
    entry (signal_attribution.py shows the rows live);
  * with `ml_neutral`, holds the ML input of the blend at 50 — the value the
    legacy blend already uses when the model is "uncertain" — so the score
    keeps its scale and the MIN_SCORE thresholds keep their meaning. Every
    honest out-of-sample test of these feature families is null (v2 with
    date-keyed purged CV AUC 0.515, v1.0.1215; v3 AUC 0.513, 95% CI
    0.494-0.530) and the live v2 fit learns from per-scan copies of a few
    hundred candidate-days.
The reconstruction is the one source_validation.ablated_score uses (rules
total minus the removed points, clamped, then the logged blend), so the shadow
log can replay any variant. Pure: no I/O apart from reading the config.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

# neutral value and score weight of every term v2 may hold neutral
# (bot_engine.deep_score: volume_score enters at 0.15 around 50; the two
# bonuses are added points). walkforward.price_composite_v2 uses the same.
NEUTRAL: Dict[str, float] = {"volume": 50.0, "vol_premium": 0.0, "rel_strength": 0.0}
WEIGHT: Dict[str, float] = {"volume": 0.15, "vol_premium": 1.0, "rel_strength": 1.0}
ML_NEUTRAL_INPUT = 50.0

DEFAULT_CONFIG = {
    "enabled": True,
    "neutralize": ("volume", "vol_premium", "rel_strength"),
    "ml_neutral": True,
    "since": "2026-11-06",
    "kill_date": "2027-02-05",
}


def _num(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _clamp(v: float) -> float:
    return max(0.0, min(100.0, v))


def v2_config(cfg=None) -> dict:
    """The validated config: BASE_CONFIG["SCORE_COMPOSITE_V2"] (or `cfg`),
    defaults for missing keys, unknown `neutralize` names dropped (and listed
    under `ignored`) — a typo can never neutralize a term that does not exist."""
    if cfg is None:
        from system_config import BASE_CONFIG
        cfg = BASE_CONFIG.get("SCORE_COMPOSITE_V2")
    src = cfg if isinstance(cfg, dict) else {}
    out = dict(DEFAULT_CONFIG)
    out.update({k: v for k, v in src.items() if k in DEFAULT_CONFIG})
    if not isinstance(cfg, dict):
        out["enabled"] = False          # absent / malformed config: legacy served
    names = out.get("neutralize") or ()
    names = (names,) if isinstance(names, str) else tuple(names)
    out["neutralize"] = tuple(n for n in names if n in NEUTRAL)
    out["ignored"] = tuple(n for n in names if n not in NEUTRAL)
    out["enabled"] = bool(out.get("enabled"))
    out["ml_neutral"] = bool(out.get("ml_neutral"))
    return out


def removed_points(components: dict, neutralize) -> float:
    """Score points the neutralized terms contributed to the legacy rules
    total (positive = they had raised it)."""
    total = 0.0
    for k in neutralize:
        v = _num((components or {}).get(k))
        if v is not None:
            total += WEIGHT[k] * (v - NEUTRAL[k])
    return total


def composite_v2(rules_raw, components: dict, ml_trace: Optional[dict], cfg: dict) -> Optional[dict]:
    """The v2 score from deep_score's own terms. rules_raw = the pre-clamp
    rules total; components = the terms named in NEUTRAL; ml_trace = deep_score's
    _ml_trace (blended / gated / rule_weight / ml_weight / ml_only_score).
    None when rules_raw is missing (the caller serves the legacy score)."""
    raw = _num(rules_raw)
    if raw is None:
        return None
    removed = removed_points(components, cfg.get("neutralize") or ())
    rules = _clamp(raw - removed)
    ml = ml_trace if isinstance(ml_trace, dict) else {}
    out = {"rules": round(rules, 4), "removed": round(removed, 4), "blended": False, "ml_input": None}
    if ml.get("blended") is True:
        rw, mw = _num(ml.get("rule_weight")), _num(ml.get("ml_weight"))
        if rw is None or mw is None:
            return None
        if cfg.get("ml_neutral") or ml.get("gated"):
            ml_in = ML_NEUTRAL_INPUT
        else:
            ml_in = _num(ml.get("ml_only_score"))
            if ml_in is None:
                return None
        out.update({"blended": True, "ml_input": ml_in, "score": _clamp(rules * rw + ml_in * mw)})
    else:
        out["score"] = rules
    out["score"] = round(out["score"], 4)
    return out


def reason_line(v2: dict, legacy: float, cfg: dict) -> str:
    """The plain-English line deep_score adds to the candidate's reasons."""
    parts = []
    names = {"volume": "volume surge", "vol_premium": "vol-premium bonus", "rel_strength": "relative-strength bonus"}
    if cfg.get("neutralize"):
        held = ", ".join(names[n] for n in cfg["neutralize"])
        parts.append(f"{held} held neutral ({-v2['removed']:+.1f} pts; measured anti-predictive 2017–2026)")
    if cfg.get("ml_neutral") and v2.get("blended"):
        parts.append("ML input held at 50 (no validated out-of-sample skill)")
    head = f"Score v2 {v2['score']:.1f} (legacy {legacy:.1f})"
    return f"{head}: {'; '.join(parts)}" if parts else head
