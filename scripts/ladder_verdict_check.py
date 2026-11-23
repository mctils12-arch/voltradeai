#!/usr/bin/env python3
"""
ladder_verdict_check.py — does datacore/signal_ladder.json say what the
registered gate tests have already decided?

WHY THIS EXISTS (EDGE DOCTRINE 3, compile knowledge into code): several
ROOT VALIDATION LADDER tests now run themselves on production on a weekly
schedule and write a REGISTERED verdict (the first sufficient run; later
runs are informational). The ladder file is a hand-kept snapshot, so a
verdict changed nothing until a session happened to curl the probe, read
the JSON and edit the file. On 2026-11-23 two such verdicts were sitting
unread: github_gate2 had registered FAIL five days earlier (2026-11-18)
while the ladder still read gate1_pass, and earnings_qoq registered FAIL
at 2026-11-22T23:05Z while the ladder read gate2_pending — which
server/signalProducts.ts publishes as "in validation". "Read tonight's
runs" had been carried in the NEXT/STARVED lines of several entries.
That is labour, not judgment: this script does the reading.

WHAT IT DOES: for every root carrying a `registered_verdict` block
  {"probe": "<diag probe>", "path": "<dotted path to the verdict>", "gate": N}
it reads GET /api/diag/<probe> (header x-diag-token) and classifies:
  AGREES      the ladder status is what the registered verdict implies
  DRIFT       a registered PASS/FAIL exists and the status does not say so
  NO_VERDICT  the probe answered and holds no registered verdict yet
  UNREADABLE  the probe did not answer (says so; never guessed)
A registered FAIL at gate N implies `gateN_fail` or `killed`. A registered
PASS at gate N implies `gateN_pass` or a status at a later gate.

WHAT IT DOES NOT DO: it never edits the ladder. A DRIFT row is a prompt for
a session to read the run, apply judgment (a mechanical PASS can still be
FRAGILE — the 2026-10-02 tone-level run is the precedent) and record the
outcome with its layer of death in research/experiments.md.

Usage:
  DIAG_TOKEN=... python3 scripts/ladder_verdict_check.py          # report
  DIAG_TOKEN=... python3 scripts/ladder_verdict_check.py --json
Exit code is always 0: informational, like ladder_readiness_check.py.
scripts/session_health_check.py runs the same check at every session start.
"""

import argparse
import json
import os
import re
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LADDER_PATH = os.path.join(REPO_ROOT, "datacore", "signal_ladder.json")
DEFAULT_BASE_URL = "https://voltradeai-production.up.railway.app"
TIMEOUT_S = 30

AGREES, DRIFT, NO_VERDICT, UNREADABLE = "AGREES", "DRIFT", "NO_VERDICT", "UNREADABLE"

_PROBE_RE = re.compile(r"^[a-z0-9_]+$")
_STATUS_RE = re.compile(r"^gate(\d)_(pass|fail|pending)$")


def dig(obj, path):
    """Value at a dotted path, or None when any step is missing."""
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def normalize_verdict(value):
    """'PASS' / 'FAIL' for a registered verdict; None for anything else
    (WAITING, INSUFFICIENT, null, a number): those are not verdicts."""
    if not isinstance(value, str):
        return None
    v = value.strip().upper()
    return v if v in ("PASS", "FAIL") else None


def status_agrees(status, verdict, gate):
    """Does a ladder status say what a registered verdict at `gate` implies?"""
    if verdict == "FAIL":
        return status in (f"gate{gate}_fail", "killed")
    if verdict == "PASS":
        m = _STATUS_RE.match(status or "")
        if not m:
            return False
        g, kind = int(m.group(1)), m.group(2)
        return (g == gate and kind == "pass") or g > gate
    raise ValueError(f"not a verdict: {verdict!r}")


def validate_spec(root_id, spec):
    """Fail loud on a malformed block: a silently skipped root is a verdict
    nobody will ever read."""
    if not isinstance(spec, dict):
        raise ValueError(f"{root_id}: registered_verdict must be an object")
    probe, path, gate = spec.get("probe"), spec.get("path"), spec.get("gate")
    if not isinstance(probe, str) or not _PROBE_RE.match(probe):
        raise ValueError(f"{root_id}: registered_verdict.probe must be a diag probe name, got {probe!r}")
    if not isinstance(path, str) or not path or path.startswith(".") or path.endswith("."):
        raise ValueError(f"{root_id}: registered_verdict.path must be a dotted path, got {path!r}")
    if isinstance(gate, bool) or not isinstance(gate, int) or not 1 <= gate <= 5:
        raise ValueError(f"{root_id}: registered_verdict.gate must be 1..5, got {gate!r}")
    return probe, path, gate


def classify(root, payload):
    """One root against its probe's answer (payload None = no answer)."""
    probe, path, gate = validate_spec(root["id"], root["registered_verdict"])
    base = {"id": root["id"], "status": root["status"], "probe": probe, "gate": gate, "verdict": None}
    if not isinstance(payload, dict):
        return {**base, "outcome": UNREADABLE,
                "detail": f"/api/diag/{probe} did not answer; the ladder status was not checked"}
    verdict = normalize_verdict(dig(payload, path))
    if verdict is None:
        return {**base, "outcome": NO_VERDICT,
                "detail": f"/api/diag/{probe} holds no registered PASS/FAIL at {path} yet"}
    base["verdict"] = verdict
    if status_agrees(root["status"], verdict, gate):
        return {**base, "outcome": AGREES,
                "detail": f"registered {verdict} at gate {gate}; the ladder reads {root['status']}"}
    return {**base, "outcome": DRIFT,
            "detail": (f"/api/diag/{probe} registered {verdict} at gate {gate} and the ladder still reads "
                       f"{root['status']}: read the run, record it in research/experiments.md "
                       f"and update datacore/signal_ladder.json")}


def load_roots(path=LADDER_PATH):
    with open(path, encoding="utf-8") as f:
        return [r for r in json.load(f)["roots"] if r.get("registered_verdict")]


def fetch_probe(base_url, probe, token, timeout=TIMEOUT_S):
    req = urllib.request.Request(f"{base_url}/api/diag/{probe}", headers={"x-diag-token": token})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def check_all(fetch, roots=None):
    """`fetch(probe)` -> the probe's JSON or None. One request per probe."""
    roots = load_roots() if roots is None else roots
    seen = {}
    out = []
    for root in roots:
        probe = validate_spec(root["id"], root["registered_verdict"])[0]
        if probe not in seen:
            seen[probe] = fetch(probe)
        out.append(classify(root, seen[probe]))
    return out


def summarize(results):
    """(needs_action, one line) for the session-start health report."""
    drift = [r for r in results if r["outcome"] == DRIFT]
    unreadable = [r for r in results if r["outcome"] == UNREADABLE]
    if drift:
        return True, "; ".join(r["detail"] for r in drift)
    agrees = sum(1 for r in results if r["outcome"] == AGREES)
    waiting = sum(1 for r in results if r["outcome"] == NO_VERDICT)
    line = f"{len(results)} root(s) with a self-running registered test: {agrees} recorded in the ladder, {waiting} with no verdict yet"
    if unreadable:
        line += f", {len(unreadable)} probe(s) did not answer ({', '.join(r['probe'] for r in unreadable)})"
    return False, line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("VOLTRADE_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("DIAG_TOKEN", "")
    if not token:
        print("[ladder_verdict_check] DIAG_TOKEN not set — nothing was read")
        return 0
    results = check_all(lambda probe: fetch_probe(args.base_url, probe, token))
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"[{r['outcome']:10s}] {r['id']}: {r['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
