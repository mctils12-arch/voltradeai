#!/usr/bin/env python3
"""runpod_reap.py — find + close RunPod pods orphaned by a dead watchdog session.

WHY THIS EXISTS: runpod_launch.py's OPTION A CAVEAT (research/runpod_ledger.md,
human-accepted 2026-07-08) is that the cost-cap watchdog lives in the launching
Claude Code session. This repo's remote sessions run in an EPHEMERAL container
that is reclaimed when the session ends — so if a session ends (or is
interrupted) before its watchdog loop reaches the terminate step, NOTHING is
left watching that pod. A concurrent session diagnosing the gv-div4-ks
train_rc=1 crash (2026-07-10) hit exactly this: its pod was still `RUNNING`
minutes after the training process had already crashed, with no live watchdog
attached in that session at the time it checked. (HONESTY NOTE, added on
recovery of this tool 2026-07-25: gv-div4-ks itself was ultimately closed by
its own launching session's watchdog at the 5h cap — `datacore/runpod/
ledger.jsonl`'s real record — not by a manual reap; the two sessions were
investigating the same incident from different angles and only one's account
made it into the merged ledger. The underlying risk this script defends
against — a session ending mid-watchdog, orphaning a still-billing pod — is
real and undisputed regardless of which specific run first exposed it.) The
in-pod `timeout` wrapper does NOT help here: it bounds the training PROCESS,
not the pod's billing (a pod bills until DELETEd — exiting its start command
does not stop the meter).

This script is the stopgap for Option A's caveat (Option B — a server-side
watchdog living in the always-on Railway process — is the real fix, filed in
research/wishlist.md; not built). It should be run:
  1. At the START of any session that is about to check on or launch a GRID
     VISION RunPod job (research/runpod_ledger.md documents this as the
     standing session-start check for that territory), and
  2. Any time `datacore/runpod/ledger.jsonl` shows an OPEN (unclosed) row that
     is older than a fresh session boot (a strong signal its watchdog died).

MATCHING: the ledger does not store the RunPod pod id (only job_id), but
runpod_launch.py always creates pods with `name == job_id` (build_create_body).
So an open ledger job_id is matched against live pods by `pod["name"]`.

Two orphan shapes, handled differently:
  - Pod still LIVE on RunPod (still billing) -> TERMINATE it (if --apply) and
    close the ledger with the real elapsed-time cost.
  - Pod already GONE (RunPod already reaped it, or a prior manual close missed
    the ledger) -> nothing to terminate; just close the ledger with a
    best-effort cost estimate (capped at the job's own worst-case reservation
    so a reap can never invent a cost above what was already authorized).

Default is a DRY RUN (report only, never terminates or writes the ledger) --
pass --apply to actually act. stdlib only.

CLI:
  python3 scripts/runpod_reap.py            # report
  python3 scripts/runpod_reap.py --apply     # terminate + close
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runpod_budget as rb
import runpod_launch as rl


def find_open_jobs(rows):
    """Ledger rows (reconciled to one per job_id) that are still open (no
    close row written yet). Pure — thin wrapper over rb.reconcile so this
    module's own tests don't need to re-derive the reconciliation rule."""
    return [r for r in rb.reconcile(rows) if r.get("actual_cost") is None]


def match_orphans(open_jobs, live_pods):
    """Pure matching: pair each open ledger job with its live RunPod pod (by
    name == job_id), if any. Returns a list of dicts:
      {job, pod|None, still_billing: bool}
    `pod` is None when RunPod has no pod by that name (already gone — ledger
    just needs closing, nothing to terminate)."""
    by_name = {p.get("name"): p for p in live_pods if p.get("name")}
    out = []
    for job in open_jobs:
        pod = by_name.get(job.get("job_id"))
        out.append({"job": job, "pod": pod, "still_billing": pod is not None})
    return out


def estimate_cost_for_gone_pod(job):
    """Best-effort cost when the pod is already gone and we have no live
    elapsed time to measure: cap at the job's own worst-case reservation (the
    number already authorized/logged) so a reap can never invent a cost higher
    than what was already accounted for. Pure."""
    return round(float(job.get("reserved_cost", 0.0) or 0.0), 6)


def elapsed_seconds_since(opened_utc, now_utc):
    """Pure: seconds between two _utcstamp()-format strings ('%Y-%m-%dT%H:%M:%SZ')."""
    from datetime import datetime, timezone
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    a = datetime.strptime(opened_utc, fmt).replace(tzinfo=timezone.utc)
    b = datetime.strptime(now_utc, fmt).replace(tzinfo=timezone.utc)
    return (b - a).total_seconds()


def _report_line(entry):
    job = entry["job"]
    jid = job.get("job_id", "?")
    if entry["still_billing"]:
        pod = entry["pod"]
        return (f"ORPHAN-LIVE  {jid}: pod {pod.get('id')} status="
                f"{rl._pod_status(pod)} STILL BILLING (opened {job.get('opened_utc')})")
    return f"ORPHAN-GONE  {jid}: no live pod found, ledger still open (opened {job.get('opened_utc')})"


def run_reap(apply=False, ledger_path=None):
    """Full check: load ledger, list live pods, match, optionally act. Returns
    a result dict (never raises on the no-key case -- mirrors run_launch's
    awaiting_key gate). `ledger_path` defaults to rb.LEDGER, read at CALL time
    (not import time) so tests can inject a tmp_path ledger."""
    key = rl.api_key()
    if not key:
        return {"ok": False, "state": "awaiting_key",
                "detail": "RUNPOD_API_KEY absent in this session."}

    path = ledger_path or rb.LEDGER
    rows = rb.load_ledger(path)
    open_jobs = find_open_jobs(rows)
    if not open_jobs:
        return {"ok": True, "state": "clean", "orphans": []}

    live_pods = rl.list_pods(key)
    matches = match_orphans(open_jobs, live_pods)
    orphans = [m for m in matches]  # every open job is reportable; not all are billing

    acted = []
    if apply:
        now = rb._utcstamp()
        for m in orphans:
            job = m["job"]
            if m["still_billing"]:
                pod = m["pod"]
                rl.terminate_pod(pod["id"], key)
                elapsed = elapsed_seconds_since(job["opened_utc"], now)
                cost = rl.actual_cost(elapsed, job.get("hourly", 0.0))
                note = f"runpod_reap.py: orphaned pod (dead watchdog session) TERMINATED + closed"
            else:
                cost = estimate_cost_for_gone_pod(job)
                note = f"runpod_reap.py: pod already gone; ledger closed with reserved-cost estimate"
            row = dict(job)
            row["actual_cost"] = cost
            row["closed_utc"] = now
            row["note"] = note
            rb._append_row(row, path)
            acted.append({"job_id": job.get("job_id"), "actual_cost": cost})

    return {"ok": True, "state": "orphans_found" if orphans else "clean",
            "orphans": [_report_line(m) for m in orphans], "acted": acted}


def _build_parser():
    p = argparse.ArgumentParser(description="Find + close RunPod pods orphaned by a dead watchdog session")
    p.add_argument("--apply", action="store_true",
                   help="actually terminate live orphans and close the ledger (default: dry-run report)")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    res = run_reap(apply=args.apply)
    if res["state"] == "awaiting_key":
        print(res["detail"])
        return 0
    if res["state"] == "clean":
        print("no open ledger jobs -- nothing to reap.")
        return 0
    for line in res["orphans"]:
        print(line)
    if not args.apply:
        print("\n(dry run -- pass --apply to terminate + close)")
    else:
        for a in res["acted"]:
            print(f"CLOSED {a['job_id']}: actual ${a['actual_cost']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
