#!/usr/bin/env python3
"""
Systemic Stress Index (2026-04-22)
===================================
6-component leading-indicator composite. Replaces tail hedging with
predictive de-risking.

Components:
  1. Yield curve stress (T10Y2Y inversion)     0.15
  2. Credit spread stress (BAMLC0A0CM)         0.25
  3. VIX term structure (VXX ratio)            0.20
  4. SPY realized vol (VIX proxy)              0.15
  5. Breadth (SPY vs MA50/MA200)               0.15
  6. Sector rotation (defensive vs offensive)  0.10

Output: composite 0-100 → calm/elevated/stressed/crisis
"""
import json
import os
import logging
import time
import math
from typing import Dict, Any

logger = logging.getLogger("voltrade.stress")

try:
    from storage_config import DATA_DIR
except ImportError:
    DATA_DIR = "/tmp"

STRESS_CACHE_PATH = os.path.join(DATA_DIR, "voltrade_stress_cache.json")
CACHE_TTL_SECONDS = 600


def _yield_curve_stress() -> float:
    try:
        from alt_data import get_fred_macro
        yc = get_fred_macro().get("yield_curve")
        if yc is None:
            return 50.0
        if yc < -0.5:
            return 95.0
        elif yc < 0.0:
            return 80.0 + (abs(yc) * 30)
        elif yc < 0.5:
            return 50.0 - (yc * 50)
        else:
            return max(0.0, 25.0 - (yc * 5))
    except Exception:
        return 50.0


def _credit_spread_stress() -> float:
    try:
        from alt_data import get_fred_macro
        cs = get_fred_macro().get("credit_spread")
        if cs is None:
            return 50.0
        if cs >= 10:
            return 95.0
        elif cs >= 8:
            return 80.0
        elif cs >= 6:
            return 65.0
        elif cs >= 5:
            return 45.0
        elif cs >= 4:
            return 30.0
        else:
            return 15.0
    except Exception:
        return 50.0


def _vix_term_structure_stress() -> float:
    try:
        from macro_data import get_macro_snapshot
        vxx_ratio = get_macro_snapshot().get("vxx_ratio", 1.0)
        if vxx_ratio >= 1.6:
            return 95.0
        elif vxx_ratio >= 1.4:
            return 80.0
        elif vxx_ratio >= 1.2:
            return 60.0
        elif vxx_ratio >= 1.0:
            return 35.0
        else:
            return 15.0
    except Exception:
        return 50.0


def _spy_realized_vol_stress() -> float:
    try:
        from macro_data import get_macro_snapshot
        vix = get_macro_snapshot().get("vix", 20)
        if vix >= 35:
            return 95.0
        elif vix >= 28:
            return 80.0
        elif vix >= 22:
            return 55.0
        elif vix >= 17:
            return 30.0
        else:
            return 15.0
    except Exception:
        return 50.0


def _breadth_stress() -> float:
    try:
        from macro_data import get_macro_snapshot
        m = get_macro_snapshot()
        spy_vs_ma50 = m.get("spy_vs_ma50", 1.0)
        spy_vs_ma200 = m.get("spy_vs_ma200", 1.0)
        if spy_vs_ma200 < 0.95:
            return 90.0
        elif spy_vs_ma200 < 1.0:
            return 75.0
        elif spy_vs_ma50 < 0.97:
            return 60.0
        elif spy_vs_ma50 < 1.0:
            return 40.0
        elif spy_vs_ma50 < 1.02:
            return 25.0
        else:
            return 10.0
    except Exception:
        return 50.0


def _sector_rotation_stress() -> float:
    try:
        from macro_data import get_macro_snapshot
        sectors = get_macro_snapshot().get("sector_momentum", {})
        if not sectors or len(sectors) < 4:
            return 50.0
        defensive = [sectors.get(s, 0) for s in ("Consumer Staples", "Utilities", "Healthcare") if s in sectors]
        offensive = [sectors.get(s, 0) for s in ("Technology", "Consumer Discretionary", "Financials") if s in sectors]
        if not defensive or not offensive:
            return 50.0
        rotation = (sum(defensive) / len(defensive)) - (sum(offensive) / len(offensive))
        if rotation > 2.0:
            return 80.0
        elif rotation > 0.5:
            return 65.0
        elif rotation > 0:
            return 50.0
        elif rotation > -0.5:
            return 35.0
        else:
            return 20.0
    except Exception:
        return 50.0


def compute_stress_index() -> Dict[str, Any]:
    if os.path.exists(STRESS_CACHE_PATH):
        try:
            with open(STRESS_CACHE_PATH) as f:
                cached = json.load(f)
            if time.time() - cached.get("_cached_at", 0) < CACHE_TTL_SECONDS:
                return cached
        except Exception:
            pass

    components = {
        "yield_curve": _yield_curve_stress(),
        "credit_spread": _credit_spread_stress(),
        "vix_term_structure": _vix_term_structure_stress(),
        "spy_realized_vol": _spy_realized_vol_stress(),
        "breadth": _breadth_stress(),
        "sector_rotation": _sector_rotation_stress(),
    }
    weights = {
        "yield_curve":         0.15,
        "credit_spread":       0.25,
        "vix_term_structure":  0.20,
        "spy_realized_vol":    0.15,
        "breadth":             0.15,
        "sector_rotation":     0.10,
    }
    index_val = round(sum(components[k] * weights[k] for k in components), 1)

    if index_val >= 75:
        level = "crisis"
        action = "halt_new"
    elif index_val >= 55:
        level = "stressed"
        action = "reduce_50pct"
    elif index_val >= 30:
        level = "elevated"
        action = "reduce_25pct"
    else:
        level = "calm"
        action = "full_deploy"

    result = {
        "index": index_val,
        "level": level,
        "action_suggestion": action,
        "components": components,
        "weights": weights,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "_cached_at": time.time(),
    }

    try:
        import tempfile
        dirname = os.path.dirname(STRESS_CACHE_PATH) or "."
        fd, tmp = tempfile.mkstemp(dir=dirname, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp, STRESS_CACHE_PATH)
    except Exception as e:
        logger.warning(f"stress cache write failed: {e}")

    return result


def get_stress_multiplier() -> float:
    result = compute_stress_index()
    mapping = {"calm": 1.0, "elevated": 0.75, "stressed": 0.5, "crisis": 0.0}
    return mapping.get(result.get("level", "elevated"), 0.75)


if __name__ == "__main__":
    r = compute_stress_index()
    print(f"Stress Index: {r['index']:.1f} ({r['level']})")
    print(f"Action: {r['action_suggestion']}")
    for k, v in r['components'].items():
        print(f"  {k}: {v:.1f} (weight {r['weights'][k]:.2f})")
