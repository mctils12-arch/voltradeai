#!/usr/bin/env python3
"""
ladder_readiness_check.py — EDGE DOCTRINE #3 compiled-knowledge check for
gateN_pending ROOT VALIDATION LADDER roots (CLAUDE.md).

WHY THIS EXISTS: several datacore/signal_ladder.json roots are stuck at
gate2_pending not because nobody scoped the next test, but because a prior
session already ran it, found the sample too thin/young, and filed a
precise, dated re-run condition in research/open_questions.md's own NEXT
note (e.g. "re-run no earlier than 2026-08-15", ">=90 days of archive",
"~15-20 more weekly COT reports"). As of 2026-08-13, usaspending_contracts's
exact "unblocks 2026-08-15" condition alone had been manually re-derived
and re-stated in research/experiments.md over a dozen separate times across
sessions spanning 2026-07-26 through 2026-08-12 — each restatement is pure
LABOR (re-grepping prose, redoing date arithmetic), not JUDGMENT, exactly
the cost EDGE DOCTRINE #3 says should be compiled into code once, not paid
again every session.

This script reads datacore/signal_ladder.json's `readiness_trigger` field
(schema documented in that file's own _doc) and reports which gated roots
are READY (their stated condition is now satisfied) vs WAITING (with days
remaining), so a session's own AXIS SURVEY step can check this once instead
of re-deriving it. It does NOT invent new trigger conditions — a root only
gets a readiness_trigger when a prior session already committed one to
research/ in writing (datacore/signal_ladder.json's _doc says so
explicitly); this script is purely a reader/evaluator over that data.

HONESTY NOTE on 'weekly_reports' triggers: these are an ESTIMATE by
elapsed calendar time assuming the stated report cadence holds exactly
(no missed/delayed publications). A session that finds a 'weekly_reports'
root READY should still live-verify the actual published-report count
(e.g. via the source's own archive/API) before treating the root as truly
unblocked — this script cannot see live archive contents, only compute
calendar time, and says so in its own output.

Usage:
  python3 scripts/ladder_readiness_check.py             # human report
  python3 scripts/ladder_readiness_check.py --json       # machine JSON
  python3 scripts/ladder_readiness_check.py --ready-only # only READY roots
"""

import argparse
import json
import os
from datetime import date, datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER_PATH = os.path.join(REPO_ROOT, "datacore", "signal_ladder.json")


def _parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def evaluate_trigger(trigger, today):
    """Returns dict {ready: bool, days: int, detail: str} for one readiness_trigger.

    days is positive "days overdue" when ready, positive "days remaining" when not.
    Raises ValueError on an unrecognized trigger type — fail loud, never guess.
    """
    ttype = trigger.get("type")

    if ttype == "date":
        target = _parse_date(trigger["not_before"])
        delta = (today - target).days
        if delta >= 0:
            return {"ready": True, "days": delta, "detail": f"{delta}d past {target.isoformat()}"}
        return {"ready": False, "days": -delta, "detail": f"{-delta}d until {target.isoformat()}"}

    if ttype == "archive_days":
        since = _parse_date(trigger["since"])
        min_days = trigger["min_days"]
        elapsed = (today - since).days
        remaining = min_days - elapsed
        if remaining <= 0:
            return {"ready": True, "days": -remaining,
                     "detail": f"{elapsed}d elapsed since {since.isoformat()} (needs {min_days}d)"}
        return {"ready": False, "days": remaining,
                 "detail": f"{elapsed}d elapsed since {since.isoformat()} (needs {min_days}d)"}

    if ttype == "weekly_reports":
        since = _parse_date(trigger["since"])
        min_count = trigger["min_count"]
        cadence_days = trigger.get("cadence_days", 7)
        elapsed = (today - since).days
        est_count = elapsed // cadence_days
        remaining_reports = min_count - est_count
        detail = (f"ESTIMATE: ~{est_count} reports elapsed since {since.isoformat()} "
                  f"at {cadence_days}d cadence (needs ~{min_count}) — "
                  f"live-verify actual published count before trusting this")
        if remaining_reports <= 0:
            return {"ready": True, "days": 0, "detail": detail}
        remaining_days = remaining_reports * cadence_days
        return {"ready": False, "days": remaining_days, "detail": detail}

    raise ValueError(f"unrecognized readiness_trigger type: {ttype!r}")


def check_all(today=None):
    """Returns a list of {id, name, status, ready, days, detail, source_note} for every
    gated root that carries a readiness_trigger. Roots without one are omitted —
    this tool answers 'is a KNOWN condition now met', not 'what should be tested next'."""
    if today is None:
        today = date.today()

    with open(LADDER_PATH) as f:
        ladder = json.load(f)

    results = []
    for root in ladder["roots"]:
        trigger = root.get("readiness_trigger")
        if not trigger:
            continue
        outcome = evaluate_trigger(trigger, today)
        results.append({
            "id": root["id"],
            "name": root["name"],
            "status": root["status"],
            "trigger_type": trigger["type"],
            "ready": outcome["ready"],
            "days": outcome["days"],
            "detail": outcome["detail"],
            "source_note": trigger.get("source_note", ""),
        })
    return results


def _print_human_report(results, ready_only):
    ready = [r for r in results if r["ready"]]
    waiting = [r for r in results if not r["ready"]]

    print(f"READY: {len(ready)}/{len(results)} gated roots have their stated re-run condition met")
    for r in ready:
        est_flag = " [ESTIMATE]" if r["trigger_type"] == "weekly_reports" else ""
        print(f"  [READY{est_flag}] {r['id']} ({r['status']}): {r['detail']}")
        print(f"        {r['source_note']}")

    if not ready_only:
        print(f"\nWAITING: {len(waiting)}/{len(results)}")
        for r in waiting:
            print(f"  [waiting {r['days']}d] {r['id']} ({r['status']}): {r['detail']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a report")
    ap.add_argument("--ready-only", action="store_true", help="print only READY roots")
    args = ap.parse_args()

    results = check_all()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_human_report(results, args.ready_only)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
