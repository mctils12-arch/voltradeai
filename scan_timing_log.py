"""
One line per full scan, kept by day (2026-11-24, KNOWN BROKEN #60 recurrence;
research/open_questions.md (al3)).

bot_engine writes each scan's phase timings to ONE file,
voltrade_scan_timings.json, and the next scan overwrites it. So the only
record of a scan that ran past the daemon's 300 s was the audit line of its
timeout, and nothing held how long the steps AFTER the options stage took on
the scans that did return. #60 gave the options stage a deadline of 240 s
from that gap in the record: it assumed 60 s was enough for what follows.

roll_previous() is called at the start of a scan, before the timing file is
cleared. It reads what the PREVIOUS scan left there, whether it finished or
not (a scan the daemon abandoned keeps running in its thread and usually
finishes its file), and appends one summary line to
scan_timing_log/<UTC day>.jsonl. It never raises and makes no request.

A record, not a rule: nothing on a trade path reads these files.
"""
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

LOG_DIR_NAME = "scan_timing_log"
KEEP_DAYS = 30
# The phase that ends the options stage and the one that starts the tier
# engine (bot_engine._scan_market_inner's _timing_log names).
OPTIONS_STAGE_END = "step6_trade_loop_and_options"
TIER_ENGINE_START = "tier_engine_start"
_TIER_SECONDS = ("tier1_sec", "tier2_sec", "tier3_sec", "tier4_sec", "allocator_sec",
                 "total_sec", "killed_at_sec")


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return round(float(v), 2)


def summarize(timing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The one-line summary of a timing file's content, or None when the file
    holds no scan (no start time, or no phase reached)."""
    if not isinstance(timing, dict):
        return None
    started = _num(timing.get("scan_started"))
    phases = timing.get("phases")
    if started is None or not isinstance(phases, list) or not phases:
        return None
    phase_sec: Dict[str, float] = {}
    at_end_of: Dict[str, float] = {}
    last_phase = None
    reached = None
    options: Dict[str, Any] = {}
    tier: Dict[str, Any] = {}
    layer2: Dict[str, Any] = {}
    for p in phases:
        if not isinstance(p, dict):
            continue
        name = str(p.get("phase", ""))
        if name == "options_stage_deadline":
            options = {"deadline_sec": _num(p.get("deadline_sec")),
                       "candidates": p.get("candidates"),
                       "not_reached": p.get("not_reached"),
                       # the per-ticker loop alone: the only part of the
                       # options step the deadline bounds
                       "loop_sec": _num(p.get("stage_secs"))}
            continue
        if name == "tier_engine_breakdown":
            tt = p.get("tier_timings") if isinstance(p.get("tier_timings"), dict) else {}
            tier = {k: _num(tt.get(k)) for k in _TIER_SECONDS if _num(tt.get(k)) is not None}
            pre = tt.get("csp_layer2_prefetch") if isinstance(tt.get("csp_layer2_prefetch"), dict) else {}
            if pre:
                layer2 = {"cache_hit": pre.get("cache_hit"),
                          "elapsed_sec": _num(pre.get("elapsed_sec")),
                          "completed": pre.get("completed"), "total": pre.get("total")}
            continue
        dur, tot = _num(p.get("duration_sec")), _num(p.get("total_sec"))
        if dur is None or tot is None:
            continue
        phase_sec[name] = dur
        at_end_of[name] = tot
        last_phase, reached = name, tot
    if last_phase is None:
        return None
    completed = timing.get("status") == "completed"
    total = _num(timing.get("total_sec")) if completed else None
    end = total if total is not None else reached
    stage_end = at_end_of.get(OPTIONS_STAGE_END)
    return {
        "scan_started": started,
        "day": time.strftime("%Y-%m-%d", time.gmtime(started)),
        "status": "completed" if completed else "unfinished",
        "total_sec": total,
        "reached_sec": reached,
        "last_phase": last_phase,
        "options_stage_end_sec": stage_end,
        "tier_engine_start_sec": at_end_of.get(TIER_ENGINE_START),
        # what the scan spent after the options stage ended (to its end, or to
        # the last phase it reached when it never finished)
        "after_options_sec": round(end - stage_end, 2) if stage_end is not None else None,
        "options": options,
        # the options step's seconds OUTSIDE its per-ticker loop (the earnings
        # pass and the candidate list before it, the sort after it): no
        # deadline covers these
        "options_outside_loop_sec": (round(phase_sec[OPTIONS_STAGE_END] - options["loop_sec"], 2)
                                     if OPTIONS_STAGE_END in phase_sec and options.get("loop_sec") is not None
                                     else None),
        "tier": tier,
        "layer2": layer2,
        "phase_sec": phase_sec,
    }


def _prune(log_dir: str, today: str, keep_days: int) -> List[str]:
    """Remove day files older than keep_days before `today` (names sort as dates)."""
    removed = []
    try:
        cutoff = time.strftime("%Y-%m-%d", time.gmtime(
            time.mktime(time.strptime(today, "%Y-%m-%d")) - keep_days * 86400))
        for name in sorted(os.listdir(log_dir)):
            if name.endswith(".jsonl") and name[:-6] < cutoff:
                os.remove(os.path.join(log_dir, name))
                removed.append(name)
    except Exception as e:
        print(f"[scan-timing-log] prune failed: {e}", file=sys.stderr, flush=True)
    return removed


def roll_previous(timing_path: str, data_dir: str, keep_days: int = KEEP_DAYS) -> Optional[Dict[str, Any]]:
    """Append the summary of the scan whose timings sit at timing_path to its
    day file under data_dir. Returns the row written, or None (no file, no
    scan in it, or the write failed: said on stderr, never raised)."""
    try:
        if not os.path.exists(timing_path):
            return None
        with open(timing_path) as f:
            row = summarize(json.load(f))
        if row is None:
            return None
        log_dir = os.path.join(data_dir, LOG_DIR_NAME)
        os.makedirs(log_dir, exist_ok=True)
        day_path = os.path.join(log_dir, row["day"] + ".jsonl")
        new_day = not os.path.exists(day_path)
        with open(day_path, "a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
        if new_day:
            _prune(log_dir, row["day"], keep_days)
        return row
    except Exception as e:
        print(f"[scan-timing-log] previous scan not recorded: {e}", file=sys.stderr, flush=True)
        return None
