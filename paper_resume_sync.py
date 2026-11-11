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
  - the tiered_strategy / risk_kill_switch SHARED peak
    (voltrade_peak_equity.json): written DOWN to the same validated equity
    through tiered_strategy.rebaseline_peak_equity (2026-11-11, KNOWN BROKEN
    #49, the human's wishlist option (a)). Until then it was reported, never
    lowered, so risk_kill_switch's -20% kill, tiered_strategy's master kill
    (-20%) and T2 leverage gate (-8%) kept measuring from the pre-re-baseline
    high: the engine latched on 2026-11-06 at a -10.6% drawdown from the
    re-baselined peak. On RECOVERED (the peak is real) it is reported only.
  - catch_up_shared_peak(): the ONE-TIME catch-up for the 2026-09-28
    re-baseline, whose event fired before this file could lower the shared
    peak (server/peakCatchUp.ts runs it once, marker-guarded).

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

    # 3. shared tiered/risk peak — written down with the others on a
    #    re-baselining basis (2026-11-11, KNOWN BROKEN #49 option (a))
    shared = float(rks.get_peak_equity() or 0.0)
    out["shared_peak"] = {
        "peak_equity": shared,
        "drawdown_pct_vs_equity": round((eq - shared) / shared * 100.0, 2) if shared > 0 else None,
        "changed": False,
    }
    if shared > eq:
        if write:
            import tiered_strategy
            wd = tiered_strategy.rebaseline_peak_equity(eq, basis)
            after = float(rks.get_peak_equity() or 0.0)
            out["shared_peak"].update({"changed": bool(wd.get("changed")) and abs(after - eq) < 0.005,
                                       "after": after, "note": wd.get("note", "")})
            if not out["shared_peak"]["changed"]:
                out["notes"].append(f"shared peak NOT written down: {wd.get('note', '')}")
        else:
            out["notes"].append(
                "voltrade_peak_equity.json left at its peak (report only): "
                "tiered_strategy T2 leverage gate (-8%), master kill (-20%) and risk_kill_switch (-20%) measure from it"
            )

    out["applied"] = write
    return out


def catch_up_shared_peak(target_peak: float, equity: float, apply: bool = True, settle_s: float = 2.0) -> dict:
    """ONE-TIME catch-up for the 2026-09-28 paper re-baseline (KNOWN BROKEN #49,
    the human's wishlist option (a), 2026-11-11).

    That re-baseline lowered Node's equityPeak and bot_engine's DD-halt peak
    but left the shared peak (voltrade_peak_equity.json) at the old high, and
    Node is not latched, so no new re-baseline event will ever fire to fix it.
    This writes the shared peak DOWN to `target_peak` (Node's re-baselined
    equityPeak) and clears risk_kill_switch's persisted kill through its own
    reset_kill_state() — only a "Portfolio DD" kill, never a manual one, the
    same rules as rebaseline_python_halts.

    target_peak: Node's state.equityPeak. Must be finite, >= equity (a peak is
                 never below the current equity) and below the stored shared
                 peak; otherwise nothing is written.
    equity:      the current account equity (for the guard and the report).
    apply:       False = report only. settle_s: pause before the read-back that
                 catches a concurrent max-ratchet write (a Tier-2 scan's
                 update_peak_equity); one retry if it bounced back.

    Returns {"ok", "applied", "shared_peak", "risk_kill_switch", "levels",
    "notes"}; "ok" means the shared peak now equals the target (or already
    did) and no Portfolio-DD kill is left latched. Never raises for expected
    failures."""
    import time
    out: dict = {"ok": False, "applied": False, "target_peak": target_peak, "equity": equity,
                 "shared_peak": None, "risk_kill_switch": None, "levels": None, "notes": []}
    try:
        tgt, eq = float(target_peak), float(equity)
    except (TypeError, ValueError):
        out["notes"].append("refused: target_peak / equity is not a number")
        return out
    if not (math.isfinite(tgt) and math.isfinite(eq)) or tgt <= 0 or eq <= 0:
        out["notes"].append("refused: target_peak / equity is not a credible positive number")
        return out
    if tgt < eq:
        out["notes"].append(f"refused: target peak {tgt:.2f} is below the current equity {eq:.2f}")
        return out

    import bot_engine
    import risk_kill_switch as rks
    import tiered_strategy

    write = bool(apply)
    if write and not _is_paper(getattr(bot_engine, "ALPACA_BASE_URL", None)):
        out["notes"].append("refused: bot_engine.ALPACA_BASE_URL is not the paper endpoint — nothing written")
        write = False

    before = float(rks.get_peak_equity() or 0.0)
    sp: dict = {"before": before, "after": before, "changed": False}
    if before > tgt + 0.005 and write:
        for attempt in (1, 2):
            wd = tiered_strategy.rebaseline_peak_equity(tgt, "PAPER_REBASE_CATCHUP")
            if settle_s > 0:
                time.sleep(settle_s)
            after = float(rks.get_peak_equity() or 0.0)
            sp.update({"after": after, "changed": abs(after - tgt) < 0.005, "note": wd.get("note", ""), "attempts": attempt})
            if sp["changed"]:
                break
            out["notes"].append(f"attempt {attempt}: shared peak reads {after:.2f} after the write-down ({wd.get('note', '')})")
    elif before > tgt + 0.005:
        sp["note"] = "report only"
    else:
        sp["note"] = "already at or below the target: nothing to write"
    out["shared_peak"] = sp

    ks = rks._load_state()
    manual = os.path.exists(rks.MANUAL_KILL_PATH)
    kill_reason = str(ks.get("kill_reason", "") or "")
    kr = {"killed": bool(ks.get("killed", False)), "kill_reason": kill_reason,
          "state_peak": ks.get("peak_equity"), "manual_kill_file": manual, "reset": False}
    peak_ok = abs(float(sp["after"]) - tgt) < 0.005 or float(sp["after"]) <= tgt
    if kr["killed"]:
        if manual:
            out["notes"].append("risk_kill_switch: MANUAL kill file present — never auto-cleared")
        elif not kill_reason.startswith("Portfolio DD"):
            out["notes"].append(f"risk_kill_switch: kill reason {kill_reason!r} is not a portfolio-DD kill — left as is")
        elif write and peak_ok:
            rks.reset_kill_state()
            kr["reset"] = not bool(rks._load_state().get("killed", False))
            if not kr["reset"]:
                out["notes"].append("risk_kill_switch: reset did not persist")
        elif write:
            out["notes"].append("risk_kill_switch: kill left latched — the shared peak did not settle at the target")
    out["risk_kill_switch"] = kr

    new_peak = float(sp["after"])
    out["levels"] = {
        "peak": new_peak,
        "kill_equity": round(new_peak * (1 + rks.PORTFOLIO_DD_KILL), 2),
        "drawdown_pct": round((eq - new_peak) / new_peak * 100.0, 2) if new_peak > 0 else None,
    }
    still_dd_killed = kr["killed"] and kill_reason.startswith("Portfolio DD") and not manual and not kr["reset"]
    out["applied"] = write and (sp["changed"] or kr["reset"])
    out["ok"] = peak_ok and not still_dd_killed
    return out

