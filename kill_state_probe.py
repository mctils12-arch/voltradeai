"""
kill_state_probe.py — READ-ONLY report of every Python-side kill latch, so a
session (or /api/health) can see WHERE each drawdown kill is measured from.

WHY (2026-11-10, KNOWN BROKEN #49). Three drawdown guards persist their own
peak on the volume and none of them is visible from outside the container:

  1. risk_kill_switch.check_kill_switches()  -20% from its state's peak
     (voltrade_killswitch.json) — ratchets from the SHARED peak below, latches
     `killed`, can_auto_resume False, and suppresses every scanner entry and
     the whole tier engine while latched (bot_engine.suppress_entries_on_kill).
  2. the shared tiered/risk peak (voltrade_peak_equity.json) — max-ratchet
     writers only (tiered_strategy.update_peak_equity, risk_kill_switch.
     set_peak_equity); the paper auto-resume (paper_resume_sync.py) reports it
     and deliberately never lowers it.
  3. bot_engine's portfolio DD halt (voltrade_portfolio_dd.json, -18%) — the
     one halt paper_resume_sync DOES re-baseline.

Found live: /api/health read "active, -9.7% from the re-baselined $99,962
peak" while guard 1 had been latched "Portfolio DD -20.0%" against the stale
~$110.7K peak since 2026-11-06 — zero new entries for three trading days and
nothing surfaced it. This module reads the three files and the constants that
apply to them. It writes nothing, imports nothing heavy (no bot_engine: that
import is the ~300 MB cold start), and never raises for a missing or corrupt
file — a probe that crashes is a probe that hides the state it was built to
show.

Served by voltrade_daemon.py as the `kill_state` RPC method (python3 -c
fallback in server/bot.ts's `kill_state` diag probe).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import risk_kill_switch as rks

try:
    from storage_config import DATA_DIR
except ImportError:  # same fallback risk_kill_switch and bot_engine use
    DATA_DIR = "/data/voltrade" if os.path.isdir("/data") else "/tmp"

# bot_engine._DD_STATE_PATH, mirrored by name so this probe never imports
# bot_engine. test_kill_state_probe pins the two paths agree.
DD_STATE_FILENAME = "voltrade_portfolio_dd.json"
DD_STATE_PATH = os.path.join(DATA_DIR, DD_STATE_FILENAME)

# Mirror of bot_engine's DRAWDOWN_HALT_PCT default, reported only as the
# percentage the file's own `halt_reason` string also carries; the live value
# lives in system_config (DRAWDOWN_HALT_*), not here.
_DD_FIELDS = ("peak_equity", "halted", "halt_reason", "halt_started_at",
              "last_equity", "last_updated")


def _read_json(path: str):
    """A dict from `path`, or None when the file is missing, unreadable or not
    a JSON object. Never raises — see the module docstring."""
    try:
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def implied_levels(peak: float) -> dict:
    """The equity levels risk_kill_switch's constants imply for `peak`:
    where the -20% kill trips and where auto-resume's 95%-of-peak gate opens.
    Pure; constants are read from risk_kill_switch, never restated."""
    if peak <= 0:
        return {"kill_equity": None, "resume_equity": None}
    return {
        "kill_equity": round(peak * (1.0 + rks.PORTFOLIO_DD_KILL), 2),
        "resume_equity": round(peak * rks.RESUME_REQUIRES_EQUITY_RECOVERY, 2),
    }


def drawdown_pct(equity, peak) -> float | None:
    """(equity - peak) / peak in percent, or None without a usable pair."""
    e, p = _num(equity, -1.0), _num(peak, -1.0)
    if e <= 0 or p <= 0:
        return None
    return round((e - p) / p * 100.0, 2)


def kill_state_report(equity=None) -> dict:
    """Everything the three guards persist, read-only.

    `equity` (optional, the caller's current account equity) adds each guard's
    live drawdown from ITS OWN peak so the three readings can be compared on
    one line with the Node side's. Nothing is written."""
    ks = rks.get_kill_status()
    ks_peak = _num(ks.get("peak_equity"))
    shared_peak = _num(rks.get_peak_equity())
    dd_raw = _read_json(DD_STATE_PATH)
    dd = None
    if dd_raw is not None:
        dd = {k: dd_raw.get(k) for k in _DD_FIELDS}
        dd["peak_equity"] = _num(dd.get("peak_equity"))
        dd["halted"] = bool(dd.get("halted", False))
        dd["drawdown_pct"] = drawdown_pct(equity, dd["peak_equity"]) if equity is not None else None

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "equity": _num(equity) if equity is not None else None,
        "risk_kill_switch": {
            "killed": bool(ks.get("killed", False)),
            "kill_reason": str(ks.get("kill_reason") or ""),
            "killed_at": ks.get("killed_at"),
            "peak_equity": ks_peak,
            "consecutive_losses": ks.get("consecutive_losses", 0),
            "daily_loss_pct": ks.get("daily_loss_pct", 0),
            "manual_kill_file": bool(ks.get("manual_flag_present", False)),
            "can_auto_resume": False if str(ks.get("kill_reason") or "").startswith("Portfolio DD") else None,
            "drawdown_pct": drawdown_pct(equity, ks_peak) if equity is not None else None,
            **implied_levels(ks_peak),
            # last events of voltrade_kill_history.json (equity/peak/dd at the trip)
            "recent_kills": list(ks.get("recent_kills") or [])[-5:],
        },
        "shared_peak": {
            "peak_equity": shared_peak,
            "drawdown_pct": drawdown_pct(equity, shared_peak) if equity is not None else None,
            **implied_levels(shared_peak),
            "note": "voltrade_peak_equity.json — max-ratchet writers only; the tier engine's master kill (-20%) and risk_kill_switch ratchet from it",
        },
        "bot_engine_dd": dd,
        "constants": {
            "PORTFOLIO_DD_KILL": rks.PORTFOLIO_DD_KILL,
            "RESUME_REQUIRES_EQUITY_RECOVERY": rks.RESUME_REQUIRES_EQUITY_RECOVERY,
            "RESUME_COOLDOWN_HOURS": rks.RESUME_COOLDOWN_HOURS,
        },
    }
    return report


if __name__ == "__main__":  # operator one-liner: python3 kill_state_probe.py [equity]
    import sys
    eq = float(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(kill_state_report(eq), indent=2, default=str))
