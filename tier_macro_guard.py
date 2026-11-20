#!/usr/bin/env python3
"""
Tier engine: a failed VXX fetch must not read as a calm market.

WHAT WAS WRONG (found 2026-11-20 in the production audit log)
  macro_data.get_macro_snapshot() writes vxx_ratio = 1.0 when its VXX fetch
  fails. The tier engine took that number as a reading: 1.0 classifies NEUTRAL
  (exposure cap 95%) and sits under the Tier 2-4 kill (1.40) and under the
  Tier-1 size steps (1.05 / 1.20 / 1.40; those are replaced by the regime
  engine's multiplier while REGIME_SIZING_DEFAULT names tier1). It is also the
  vxx_ratio handed to risk_kill_switch.check_kill_switches. On 2026-11-17 the engine's exposure
  cap blocked every scan from 13:35Z to 21:51Z under CAUTION (60%) and BEAR
  (50%) except two, 15:26Z and 16:10Z; both of those carried "regime=NEUTRAL,
  vxx=1.00" and both sold a put. The real readings four minutes either side
  were at or above 1.05. A data failure opened the gate.

WHAT THIS DOES
  resolve_vxx(macro) is called once per scan, before the kill switches:
    * a real reading (the snapshot carries VXX closes) is used and remembered;
    * a substitute is replaced by the last real reading while that reading is
      no older than TIER_VXX_HOLD_MAX_AGE_S ("held");
    * with no reading that recent, the scan proposes no new tier entry
      ("none", block=True) and says so. Exits are not the tier engine's and
      are untouched.
  It never invents a number: "held" is a value this process (or the one before
  a restart) read from Alpaca, with its age stated.

OFF SWITCH
  system_config BASE_CONFIG["TIER_VXX_FALLBACK_GUARD"] = False, or env
  VOLTRADE_TIER_VXX_GUARD=off (no deploy): the substitute is used as before.

The floors have had the same protection since 2026-10-13
(bot_engine._resolve_floor_regime); bot_engine._floor_macro_vxx_is_fallback
now calls vxx_is_fallback here so the two cannot drift apart.
"""

import json
import logging
import math
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("voltrade.tier_macro_guard")

DEFAULT_HOLD_MAX_AGE_S = 3600.0
STATE_FILE = "voltrade_tier_vxx_last.json"


def vxx_is_fallback(macro: Any) -> bool:
    """True when the macro carries no real VXX reading.

    macro_data.get_macro_snapshot() substitutes vxx_ratio=1.0 when the VXX
    fetch fails; it only sets vxx_closes / vxx_latest from real bars, and lists
    "vxx_ratio" in data_quality_failures when the closes are missing.
    """
    if not isinstance(macro, dict):
        return True
    if "vxx_ratio" in (macro.get("data_quality_failures") or []):
        return True
    if macro.get("vxx_closes"):
        return False
    return macro.get("vxx_latest") is None


def _state_path() -> str:
    try:
        from storage_config import DATA_DIR
    except ImportError:
        DATA_DIR = "/tmp"
    return os.path.join(DATA_DIR, STATE_FILE)


def guard_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    e = os.environ if env is None else env
    v = str(e.get("VOLTRADE_TIER_VXX_GUARD", "") or "").strip().lower()
    if v in ("off", "0", "false"):
        return False
    if v in ("on", "1", "true"):
        return True
    try:
        from system_config import BASE_CONFIG
        return bool(BASE_CONFIG.get("TIER_VXX_FALLBACK_GUARD", True))
    except ImportError as err:
        logger.warning(f"tier VXX guard: config unreadable, guard stays on: {err}")
        return True


def _hold_max_age_s() -> float:
    try:
        from system_config import BASE_CONFIG
        v = float(BASE_CONFIG.get("TIER_VXX_HOLD_MAX_AGE_S", DEFAULT_HOLD_MAX_AGE_S))
        return v if math.isfinite(v) and v >= 0 else DEFAULT_HOLD_MAX_AGE_S
    except (ImportError, TypeError, ValueError):
        return DEFAULT_HOLD_MAX_AGE_S


def _real_ratio(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _read_state(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            s = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as err:
        logger.warning(f"tier VXX guard: last reading unreadable ({err}); treated as none")
        return None
    if not isinstance(s, dict):
        return None
    ratio, at = _real_ratio(s.get("vxx_ratio")), _real_ratio(s.get("at"))
    if ratio is None or at is None:
        return None
    return {"vxx_ratio": ratio, "at": at}


def _write_state(path: str, ratio: float, at: float) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"vxx_ratio": ratio, "at": at}, f)
        os.replace(tmp, path)
    except OSError as err:
        # The reading in hand is still used this scan; only the memory is lost.
        logger.warning(f"tier VXX guard: could not remember the reading: {err}")


def resolve_vxx(macro: Any, now: Optional[float] = None, path: Optional[str] = None,
                max_age_s: Optional[float] = None, enabled: Optional[bool] = None) -> Dict[str, Any]:
    """The VXX ratio this scan's tier engine and kill switches should read.

    Returns {"vxx_ratio", "source", "age_s", "block", "reason"}:
      source "live"     the snapshot's own reading; age_s 0.
      source "held"     the snapshot had none; the last real reading, age_s old.
      source "none"     no real reading within max_age_s. vxx_ratio is the
                        substitute 1.0 (what the kill switches got before this
                        existed) and block is True: no new tier entry this scan.
      source "disabled" the off switch is set; the snapshot's number as it is.
    """
    now = time.time() if now is None else float(now)
    path = _state_path() if path is None else path
    max_age_s = _hold_max_age_s() if max_age_s is None else float(max_age_s)
    enabled = guard_enabled() if enabled is None else bool(enabled)

    raw = _real_ratio(macro.get("vxx_ratio")) if isinstance(macro, dict) else None

    if not enabled:
        return {"vxx_ratio": raw if raw is not None else 1.0, "source": "disabled",
                "age_s": None, "block": False, "reason": ""}

    if raw is not None and not vxx_is_fallback(macro):
        # The snapshot is served from a 300 s cache: its own stamp is when the
        # reading was taken, so a held reading's age counts from there.
        seen = _real_ratio(macro.get("timestamp"))
        at = seen if seen is not None and seen <= now else now
        prev = _read_state(path)
        if prev is None or prev["vxx_ratio"] != raw or prev["at"] != at:
            _write_state(path, raw, at)
        return {"vxx_ratio": raw, "source": "live", "age_s": 0.0, "block": False, "reason": ""}

    prev = _read_state(path)
    if prev is not None:
        age = max(0.0, now - prev["at"])
        if age <= max_age_s:
            return {"vxx_ratio": prev["vxx_ratio"], "source": "held", "age_s": round(age, 1), "block": False,
                    "reason": (f"no VXX reading this scan; using the last real one, "
                               f"{prev['vxx_ratio']:.2f} from {age / 60:.0f} min ago")}
        why = (f"no VXX reading this scan and the last real one is {age / 60:.0f} min old "
               f"(limit {max_age_s / 60:.0f} min)")
    else:
        why = "no VXX reading this scan and none remembered"
    return {"vxx_ratio": 1.0, "source": "none", "age_s": None, "block": True,
            "reason": f"{why}: no new tier entries until one arrives"}
