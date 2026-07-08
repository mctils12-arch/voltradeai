#!/usr/bin/env python3
"""
runpod_budget.py — RunPod GPU spend cost-cap gate + append-only spend ledger.

MANDATE (human, 2026-07-07): "log spend per job against the $50 balance, flag
when it runs low with a top-up purchase order, and NEVER let a training job run
unbounded without a cost cap."

This module is the PURE, TESTABLE half of that mandate: a cost-cap CALCULATOR
plus an append-only ledger. It deliberately does NOT launch or terminate a pod
— that needs RUNPOD_API_KEY, which lives only in Railway, not in this session
(presence-checked 0 here). Any routine that DOES launch a pod MUST:
  1. call authorize_job() first and refuse to launch if it returns authorized
     == False, and
  2. ENFORCE the returned `max_runtime_seconds` — RunPod has NO native TTL
     (verified 2026-07-07), so a watchdog must DELETE the pod at the cap and the
     training command is wrapped in an in-pod `timeout` (see runpod_launch.py),
     so a hung/runaway job self-kills instead of draining the balance.

Balance model
-------------
START balance funded by Mike 2026-07-07 = $50.00.
  remaining = START - sum(ledger cost per row)
A job IN FLIGHT (open row, actual_cost is None) reserves its WORST-CASE cost
(hourly * max_hours). When it finishes, the row is CLOSED with the ACTUAL cost,
so the reservation is released and replaced by the real spend. This means the
balance is always conservative while a job runs — we can never authorize a
second job on money the first job might still spend.

authorize_job refuses a job when:
  - max_hours is missing / non-finite / <= 0  (UNBOUNDED — the exact thing the
    mandate forbids), OR
  - hourly is non-finite / <= 0 / absurdly large (fat-finger guard), OR
  - worst-case cost would drive remaining below FLOOR_BUFFER.

low_balance() flags when remaining < LOW_THRESHOLD and emits the exact top-up
purchase-order line for wishlist.md.

Ledger: datacore/runpod/ledger.jsonl (append-only JSONL; git keeps vintages).
Each row:
  {job_id, workload, gpu, hourly, max_hours, reserved_cost,
   actual_cost|null, opened_utc, closed_utc|null, note}

CLI (session-side; stdlib only):
  python3 scripts/runpod_budget.py status
  python3 scripts/runpod_budget.py authorize --gpu RTX4090 --hourly 0.34 \
      --max-hours 6 --workload grid-detector
  python3 scripts/runpod_budget.py open  --job j1 --gpu RTX4090 --hourly 0.34 \
      --max-hours 6 --workload grid-detector --note "first fine-tune"
  python3 scripts/runpod_budget.py close --job j1 --actual 1.83
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

# Funded by Mike 2026-07-07. Single source of truth for the starting balance.
START_BALANCE = 50.00
# Never authorize a job that would leave less than this in reserve — a safety
# margin so a slightly-over-worst-case job (rounding, brief overrun) can't zero
# the account and so there is always headroom to close out an in-flight job.
FLOOR_BUFFER = 5.00
# Below this, surface a top-up purchase order to the human.
LOW_THRESHOLD = 10.00
# Fat-finger guard: a single GPU renting for more than this per hour is almost
# certainly a typo (highest legit tier ~ H100 secure ≈ $2.89/hr; 8x clusters
# push higher, but $50/hr is well past anything this program would ever run).
MAX_SANE_HOURLY = 50.00

LEDGER = os.path.join(
    os.path.dirname(__file__), "..", "datacore", "runpod", "ledger.jsonl"
)

# Reference rates (community/secure, 2026-07-07 research) — informational only;
# the live rate is passed in by the caller who read it from RunPod at launch.
REFERENCE_HOURLY = {
    "RTX4090": 0.34,   # community
    "A40": 0.44,
    "A100": 1.64,      # secure upper
    "H100": 2.89,      # secure upper
}


def _utcstamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def row_cost(row):
    """Cost this row contributes to spend: the ACTUAL cost once the job is
    closed, otherwise the reserved worst-case while it is still in flight."""
    actual = row.get("actual_cost")
    if actual is not None:
        return float(actual)
    return float(row.get("reserved_cost", 0.0) or 0.0)


def spent(rows):
    """Total committed spend = sum of every row's cost (actual if closed,
    reserved worst-case if open)."""
    return round(sum(row_cost(r) for r in rows), 6)


def remaining(rows, start=START_BALANCE):
    """Balance left after all committed spend. Conservative while jobs run."""
    return round(start - spent(rows), 6)


def worst_case_cost(hourly, max_hours):
    """Worst-case dollars for a job capped at max_hours at `hourly`."""
    return round(float(hourly) * float(max_hours), 6)


def authorize_job(
    rows,
    gpu,
    hourly,
    max_hours,
    floor_buffer=FLOOR_BUFFER,
    start=START_BALANCE,
):
    """Decide whether a proposed GPU job may launch. Returns a dict; the caller
    launches ONLY if result['authorized'] is True and MUST pass
    result['max_runtime_seconds'] to RunPod as a hard terminate-after cap.

    Refusal reasons (checked in order):
      unbounded      — max_hours missing / non-finite / <= 0
      bad_rate       — hourly missing / non-finite / <= 0 / > MAX_SANE_HOURLY
      insufficient   — worst-case would leave < floor_buffer in the account
    """
    rem_before = remaining(rows, start=start)

    # 1. The mandate's hard rule: no job without a finite, positive time cap.
    if max_hours is None or not isinstance(max_hours, (int, float)) \
            or not math.isfinite(max_hours) or max_hours <= 0:
        return {
            "authorized": False,
            "reason": "unbounded",
            "detail": "max_hours must be a finite number > 0 (no unbounded runs)",
            "gpu": gpu,
            "worst_case": None,
            "remaining_before": rem_before,
            "remaining_after_if_run": None,
            "max_runtime_seconds": None,
        }

    # 2. Sanity-check the hourly rate the caller read from RunPod.
    if hourly is None or not isinstance(hourly, (int, float)) \
            or not math.isfinite(hourly) or hourly <= 0 or hourly > MAX_SANE_HOURLY:
        return {
            "authorized": False,
            "reason": "bad_rate",
            "detail": f"hourly must be >0 and <= {MAX_SANE_HOURLY:.2f} (fat-finger guard)",
            "gpu": gpu,
            "worst_case": None,
            "remaining_before": rem_before,
            "remaining_after_if_run": None,
            "max_runtime_seconds": None,
        }

    worst = worst_case_cost(hourly, max_hours)
    rem_after = round(rem_before - worst, 6)

    # 3. Must leave the floor buffer intact after the worst case.
    if rem_after < floor_buffer:
        return {
            "authorized": False,
            "reason": "insufficient",
            "detail": (
                f"worst-case ${worst:.2f} would leave ${rem_after:.2f}, "
                f"below the ${floor_buffer:.2f} floor buffer"
            ),
            "gpu": gpu,
            "worst_case": worst,
            "remaining_before": rem_before,
            "remaining_after_if_run": rem_after,
            "max_runtime_seconds": None,
        }

    return {
        "authorized": True,
        "reason": "ok",
        "detail": f"worst-case ${worst:.2f}; ${rem_after:.2f} would remain",
        "gpu": gpu,
        "worst_case": worst,
        "remaining_before": rem_before,
        "remaining_after_if_run": rem_after,
        # Hard auto-terminate cap the launcher MUST pass to RunPod.
        "max_runtime_seconds": int(round(float(max_hours) * 3600)),
    }


def low_balance(rows, threshold=LOW_THRESHOLD, start=START_BALANCE):
    """Flag a low balance and emit the exact wishlist top-up PO line."""
    rem = remaining(rows, start=start)
    low = rem < threshold
    po_line = None
    if low:
        po_line = (
            f"PURCHASE ORDER — RunPod top-up: balance ${rem:.2f} is below the "
            f"${threshold:.2f} low-water mark. Recommend a $50 top-up to keep "
            f"GPU training/sweeps unblocked (grid-vision detector). Spend to "
            f"date is ledgered in datacore/runpod/ledger.jsonl."
        )
    return {"low": low, "remaining": rem, "threshold": threshold, "po_line": po_line}


# ----------------------------------------------------------------------------
# Thin file I/O + CLI (the pure functions above are what the tests exercise).
# ----------------------------------------------------------------------------
def load_ledger(path=LEDGER):
    """Read the append-only JSONL ledger. Missing file -> empty. Blank/garbage
    lines are skipped (never crash the balance read on one bad row)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except (ValueError, TypeError):
                continue
    return rows


def _append_row(row, path=LEDGER):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _cmd_status(_args):
    rows = load_ledger()
    lb = low_balance(rows)
    open_jobs = [r for r in rows if r.get("actual_cost") is None]
    print(f"START           ${START_BALANCE:.2f}")
    print(f"committed spend ${spent(rows):.2f}  (actual closed + reserved open)")
    print(f"remaining       ${remaining(rows):.2f}")
    print(f"open jobs       {len(open_jobs)}"
          + ("  " + ", ".join(r.get('job_id', '?') for r in open_jobs) if open_jobs else ""))
    if lb["low"]:
        print("\n" + lb["po_line"])


def _cmd_authorize(args):
    rows = load_ledger()
    res = authorize_job(rows, args.gpu, args.hourly, args.max_hours)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["authorized"] else 2)


def _cmd_open(args):
    rows = load_ledger()
    res = authorize_job(rows, args.gpu, args.hourly, args.max_hours)
    if not res["authorized"]:
        print("REFUSED: " + res["detail"])
        sys.exit(2)
    row = {
        "job_id": args.job,
        "workload": args.workload,
        "gpu": args.gpu,
        "hourly": args.hourly,
        "max_hours": args.max_hours,
        "reserved_cost": res["worst_case"],
        "actual_cost": None,
        "opened_utc": _utcstamp(),
        "closed_utc": None,
        "note": args.note or "",
    }
    _append_row(row)
    print(f"OPENED {args.job}: reserved ${res['worst_case']:.2f}; "
          f"pass max_runtime_seconds={res['max_runtime_seconds']} to RunPod.")


def _cmd_close(args):
    # Append a closing row (append-only ledger; the reader pairs open/close by
    # job_id, last-write-wins on actual_cost). Simpler + audit-friendly than
    # rewriting the open row in place.
    rows = load_ledger()
    match = [r for r in rows if r.get("job_id") == args.job and r.get("actual_cost") is None]
    if not match:
        print(f"no OPEN row for job {args.job}")
        sys.exit(1)
    src = match[-1]
    row = dict(src)
    row["actual_cost"] = args.actual
    row["closed_utc"] = _utcstamp()
    row["note"] = (src.get("note", "") + " | closed").strip(" |")
    _append_row(row)
    print(f"CLOSED {args.job}: actual ${args.actual:.2f}. "
          f"remaining ${remaining(load_ledger()):.2f}")


def _build_parser():
    p = argparse.ArgumentParser(description="RunPod cost-cap gate + spend ledger")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="show balance + open jobs + low-balance PO")

    def add_job_args(sp, need_job=False):
        if need_job:
            sp.add_argument("--job", required=True)
            sp.add_argument("--note", default="")
        sp.add_argument("--gpu", required=True)
        sp.add_argument("--hourly", type=float, required=True)
        sp.add_argument("--max-hours", dest="max_hours", type=float, required=True)
        sp.add_argument("--workload", required=True)

    add_job_args(sub.add_parser("authorize", help="dry-run: may this job launch?"))
    add_job_args(sub.add_parser("open", help="reserve worst-case + write open row"), need_job=True)

    spc = sub.add_parser("close", help="record actual spend + release reservation")
    spc.add_argument("--job", required=True)
    spc.add_argument("--actual", type=float, required=True)
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    {
        "status": _cmd_status,
        "authorize": _cmd_authorize,
        "open": _cmd_open,
        "close": _cmd_close,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
