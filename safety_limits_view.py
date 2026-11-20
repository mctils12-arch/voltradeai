"""Read-only view of the engine's safety limits and where each one stands.

([PRODUCT] 2026-11-20, UI coverage batch A, third part.) The AI Engine page
showed one limit (the trading loop's drawdown kill) and none of the engine's
own: its kill switch and drawdown halt, the daily loss limit, the
losing-streak pause, the volatility stop, the free-buying-power floor, the
correlated-names cap, the per-position stop. This module reports each limit's
VALUE (the constants in risk_kill_switch.py and system_config.py, read, never
set) beside the STATE the guards themselves keep on the volume.

The three drawdown guards come from kill_state_probe.kill_state_report (the
`kill_state` diag probe's body, reused, not re-read). The two figures that
report lacks are read from the kill switch's own state file: the date its
day-loss figure belongs to, and the time of the last counted loss.

Why not risk_kill_switch.get_kill_switch_status(): that function (the daemon's
"risk_status", served by /api/monitoring/overview) reads the losing streak and
the day's loss from consec_losses.json and daily_pnl.json, two files nothing
in the repository writes, so it answers 0 for both whatever happened. The
kill switch keeps the real figures in voltrade_killswitch.json.

Nothing here changes a limit, a state file, or any guard's behaviour: no
write, no call into check_kill_switches. A value that is not on record is
None, never 0 ("no losing streak" and "not recorded" are different answers).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import kill_state_probe as ksp
import risk_kill_switch as rks


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _pct(fraction: Any) -> Optional[float]:
    f = _num(fraction)
    return None if f is None else round(f * 100, 2)


def _text(v: Any) -> Optional[str]:
    return v if isinstance(v, str) and v.strip() else None


def _engine_halt_limits() -> Dict[str, Any]:
    """bot_engine's drawdown halt reads these from system_config at each scan."""
    try:
        from system_config import BASE_CONFIG
        return {
            "engine_halt_pct": -abs(float(BASE_CONFIG.get("DRAWDOWN_HALT_PCT", 18.0))),
            "engine_halt_enabled": bool(BASE_CONFIG.get("DRAWDOWN_HALT_ENABLED", True)),
            "engine_halt_resume_within_pct": _num(BASE_CONFIG.get("DRAWDOWN_HALT_RESUME_EQUITY_PCT", 5.0)),
            "engine_halt_resume_regimes": [str(r) for r in BASE_CONFIG.get("DRAWDOWN_HALT_RESUME_REGIMES", ["BULL", "NEUTRAL"])],
        }
    except Exception:
        return {"engine_halt_pct": None, "engine_halt_enabled": None, "engine_halt_resume_within_pct": None, "engine_halt_resume_regimes": []}


def _events(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in raw if isinstance(raw, list) else []:
        if not isinstance(e, dict) or not _text(e.get("timestamp")) or not _text(e.get("reason")):
            continue
        d = e.get("details") if isinstance(e.get("details"), dict) else {}
        out.append({"at": e["timestamp"], "reason": e["reason"], "equity": _num(d.get("equity")), "drawdown_pct": _pct(d.get("dd_pct"))})
    return out[-5:]


def safety_limits(equity: Any = None, now: Optional[datetime] = None) -> Dict[str, Any]:
    """The limits, and the guards' own record of where each stands.

    `equity` (the caller's current account equity) lets each drawdown guard be
    measured from ITS OWN peak against one number. `now` is the engine's clock
    (naive local time, as risk_kill_switch uses it); it decides only whether
    the recorded day's loss belongs to today.
    """
    now = now or datetime.now()
    rep = ksp.kill_state_report(equity=equity if _num(equity) is not None else None)
    ks = rep.get("risk_kill_switch") or {}
    dd = rep.get("bot_engine_dd")

    raw = ksp._read_json(rks.KILLSWITCH_PATH)
    state: Dict[str, Any] = raw if isinstance(raw, dict) else {}
    state_read = isinstance(raw, dict)
    day = _text(state.get("daily_loss_date"))
    today = now.strftime("%Y-%m-%d")
    streak = _num(state.get("consecutive_losses"))
    peak = _num(ks.get("peak_equity"))

    return {
        "checked_at": rep.get("checked_at"),
        "equity": rep.get("equity"),
        # False = the kill switch has written no state file yet (or it is unreadable): its readings below are None
        "state_read": state_read,
        "kill_switch": {
            "killed": bool(ks.get("killed")) if state_read else None,
            "reason": str(ks.get("kill_reason") or "")[:200],
            "killed_at": _text(ks.get("killed_at")),
            "manual_kill_file": bool(ks.get("manual_kill_file")),
            "peak_equity": peak if peak and peak > 0 else None,
            "drawdown_pct": ks.get("drawdown_pct"),
            "kill_equity": ks.get("kill_equity"),
            "resume_equity": ks.get("resume_equity"),
        },
        # None = bot_engine has written no halt file yet
        "engine_halt": None if dd is None else {
            "halted": bool(dd.get("halted")),
            "reason": str(dd.get("halt_reason") or "")[:200],
            "halted_at": _text(dd.get("halt_started_at")),
            "peak_equity": dd.get("peak_equity") if _num(dd.get("peak_equity")) and dd["peak_equity"] > 0 else None,
            "drawdown_pct": dd.get("drawdown_pct"),
        },
        "streak": {
            "losses": int(streak) if state_read and streak is not None else None,
            "last_loss_at": _text(state.get("last_loss_ts")),
        },
        # the WORST day-P&L reading the engine saw on `date` (it keeps the minimum), in percent
        "day": {
            "worst_pct": _pct(state.get("daily_loss_pct")) if day else None,
            "date": day,
            "is_today": day == today,
        },
        "limits": {
            "kill_pct": _pct(rks.PORTFOLIO_DD_KILL),
            "daily_loss_pct": _pct(rks.DAILY_LOSS_LIMIT),
            "streak_losses": int(rks.CONSECUTIVE_LOSS_KILL),
            # the pause check_kill_switches applies after the streak limit (a literal there, not a constant)
            "streak_pause_hours": 24,
            "vxx_ratio_stop": _num(rks.REGIME_KILL_VXX),
            "min_free_bp_pct": _pct(rks.MIN_FREE_BP),
            "correlated_group_cap_pct": _pct(rks.CORRELATION_CAP),
            "position_stop_pct": _pct(rks.POSITION_KILL_LOSS_PCT),
            "position_warn_pct": _pct(rks.POSITION_WARN_LOSS_PCT),
            "resume_recovery_pct": _pct(rks.RESUME_REQUIRES_EQUITY_RECOVERY),
            "resume_cooldown_hours": _num(rks.RESUME_COOLDOWN_HOURS),
            **_engine_halt_limits(),
        },
        "recent_events": _events(ks.get("recent_kills")),
    }


if __name__ == "__main__":  # operator one-liner: python3 safety_limits_view.py [equity]
    import sys
    print(json.dumps(safety_limits(float(sys.argv[1]) if len(sys.argv) > 1 else None), indent=2, default=str))
