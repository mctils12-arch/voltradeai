"""
paper_resume_sync.py — Python-side half of the PAPER-account drawdown-kill
AUTO-RESUME (server/killSwitchAutoResume.ts; human-directed 2026-09-28,
KNOWN BROKEN #42/#43).

WHY: clearing the Node latch (state.killSwitch) is not enough to turn the
loop back on. Two Python-side halts persist their own state on the volume
and can independently keep new entries blocked after the Node latch clears:

  1. bot_engine.py's portfolio DD halt (voltrade_portfolio_dd.json,
     DRAWDOWN_HALT_PCT=18%): scan_market() returns empty while `halted`, and
     its one-way ratchet only resumes within 5% of ITS OWN peak — the same
     stale ~$110.7K peak the Node side re-baselines. The 2026-09-09 reading
     (~-18.0%) very likely tripped it.
  2. risk_kill_switch.py's PORTFOLIO_DD_KILL (-20%, voltrade_killswitch.json,
     can_auto_resume False): suppresses every scanner entry and the tier
     engine while `killed`.

WHAT THIS DOES — ONLY through functions that already exist (no mechanism,
threshold or constant in either file is touched; risk_kill_switch.py is
not edited):
  - bot_engine: _load_dd_state() / _save_dd_state() re-baseline the DD peak
    to the same validated equity the Node side re-baselines to, and clear
    `halted` — only when the Node side re-baselines (basis DATA_ANOMALY or
    PAPER_REBASE). On RECOVERED the peak is real and nothing is written.
  - risk_kill_switch: reset_kill_state() (the module's own operator reset)
    — only when its persisted kill is a "Portfolio DD" kill AND no MANUAL
    kill file exists. A manual kill is never touched (reset_kill_state would
    delete the manual file, so it is never called in that case).
  - REPORTED, NOT CHANGED: the tiered_strategy / risk_kill_switch shared peak
    (voltrade_peak_equity.json). Only max-ratchet writers exist for it
    (tiered_strategy.update_peak_equity, risk_kill_switch.set_peak_equity),
    so it stays at the stale peak: tiered_strategy's T2 leverage gate (-8%)
    and master kill (-20%) keep measuring from it. Lowering it would need a
    new function — deliberately not written here; reported to the caller.

Refuses to write anything unless bot_engine's effective Alpaca base URL is
the paper host (the Node side checks its own URLs independently).
"""
from __future__ import annotations

import math
import os
from datetime import datetime
from urllib.parse import urlparse

PAPER_HOST = "paper-api.alpaca.markets"
REBASELINE_BASES = ("DATA_ANOMALY", "PAPER_REBASE")
REPORT_ONLY_BASES = ("RECOVERED",)


def _is_paper(url: object) -> bool:
    if not isinstance(url, str) or not url:
        return False
    try:
        u = urlparse(url)
    except ValueError:
        return False
    return u.scheme == "https" and u.hostname == PAPER_HOST


def rebaseline_python_halts(equity: float, basis: str, apply: bool = True) -> dict:
    """Bring the Python-side DD halts in line with a Node-side auto-resume.

    equity: the validated equity the Node side re-baselined its peak to.
    basis:  "DATA_ANOMALY" | "PAPER_REBASE" (re-baseline) or "RECOVERED"
            (report only — the peak is real).
    apply:  False = report current state only, write nothing.

    Returns a JSON-serializable report; never raises for expected failures.
    """
    out: dict = {
        "basis": basis, "equity": equity, "applied": False,
        "bot_engine_dd": None, "risk_kill_switch": None, "shared_peak": None,
        "notes": [],
    }
    try:
        eq = float(equity)
    except (TypeError, ValueError):
        out["notes"].append("refused: equity is not a number")
        return out
    if not math.isfinite(eq) or eq <= 0:
        out["notes"].append("refused: equity is not a credible positive number")
        return out
    if basis not in REBASELINE_BASES + REPORT_ONLY_BASES:
        out["notes"].append(f"refused: unknown basis {basis!r}")
        return out
    write = bool(apply) and basis in REBASELINE_BASES
    if bool(apply) and not write:
        out["notes"].append(f"{basis}: report only — the peak is real, nothing re-baselined")

    import bot_engine
    import risk_kill_switch as rks

    if write and not _is_paper(getattr(bot_engine, "ALPACA_BASE_URL", None)):
        out["notes"].append("refused: bot_engine.ALPACA_BASE_URL is not the paper endpoint — nothing written")
        write = False

    # 1. bot_engine portfolio DD halt
    dd_before = bot_engine._load_dd_state()
    dd_report = {
        "before": {
            "peak_equity": dd_before.get("peak_equity"),
            "halted": bool(dd_before.get("halted", False)),
            "halt_reason": dd_before.get("halt_reason", ""),
        },
        "rebaselined": False,
    }
    if write:
        new_state = dict(dd_before)
        new_state.update({
            "peak_equity": eq,
            "last_equity": eq,
            "last_updated": datetime.now().isoformat(),
            "halted": False,
            "halt_reason": "",
            "halt_started_at": None,
        })
        bot_engine._save_dd_state(new_state)
        dd_after = bot_engine._load_dd_state()
        dd_report["after"] = {
            "peak_equity": dd_after.get("peak_equity"),
            "halted": bool(dd_after.get("halted", False)),
        }
        dd_report["rebaselined"] = (
            abs(float(dd_after.get("peak_equity", 0) or 0) - eq) < 0.005
            and not dd_after.get("halted", False)
        )
        if not dd_report["rebaselined"]:
            out["notes"].append("bot_engine DD state did not persist (save failed?) — scan_market may still halt")
    out["bot_engine_dd"] = dd_report

    # 2. risk_kill_switch persisted kill (-20% PORTFOLIO_DD_KILL)
    ks = rks._load_state()
    manual = os.path.exists(rks.MANUAL_KILL_PATH)
    kill_reason = str(ks.get("kill_reason", "") or "")
    ks_report = {
        "killed": bool(ks.get("killed", False)),
        "kill_reason": kill_reason,
        "manual_kill_file": manual,
        "reset": False,
    }
    if ks_report["killed"]:
        if manual:
            out["notes"].append("risk_kill_switch: MANUAL kill file present — never auto-cleared")
        elif not kill_reason.startswith("Portfolio DD"):
            out["notes"].append(f"risk_kill_switch: kill reason {kill_reason!r} is not a portfolio-DD kill — left as is")
        elif write:
            rks.reset_kill_state()
            ks_report["reset"] = not bool(rks._load_state().get("killed", False))
            if not ks_report["reset"]:
                out["notes"].append("risk_kill_switch: reset did not persist — tier engine may still block entries")
    out["risk_kill_switch"] = ks_report

    # 3. shared tiered/risk peak — read only (no lowering function exists)
    shared = float(rks.get_peak_equity() or 0.0)
    out["shared_peak"] = {
        "peak_equity": shared,
        "drawdown_pct_vs_equity": round((eq - shared) / shared * 100.0, 2) if shared > 0 else None,
        "changed": False,
    }
    if shared > eq:
        out["notes"].append(
            "voltrade_peak_equity.json left at its stale peak (only max-ratchet writers exist): "
            "tiered_strategy T2 leverage gate (-8%) and master kill (-20%) still measure from it"
        )

    out["applied"] = write
    return out
