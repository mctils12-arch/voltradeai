#!/usr/bin/env python3
"""clock_skew_sweep.py — find calendar time-bombs in the server test suite.

WHY (EDGE DOCTRINE #3, compile recurring reasoning into code): on 2026-11-02
server/gridDemand.test.ts started failing, turning CI red for EVERY unrelated
PR, because its fixture was hard-coded to 2026-07-05 and the code under test
only reads the last 120 days. Finding that took a full job-log download. The
class is "a fixed fixture date ages out of a windowed code path", and it
cannot be found by reading: it has to be run against a moved clock.

HOW: run the server suite once at skew 0 (baseline) and once per requested
forward skew with scripts/clock_skew_preload.cjs (shifts Date.now() and
`new Date()` only; explicit-arg Dates are untouched). A test FILE that passes
at skew 0 but fails at skew N has a time-bomb whose fuse is <= N days. Files
already failing at skew 0 are reported separately and excluded (a plain
failure is not a time-bomb).

Usage:
  python3 scripts/clock_skew_sweep.py                 # skews 45,120
  python3 scripts/clock_skew_sweep.py --skews 30,90,200 --strict
  python3 scripts/clock_skew_sweep.py --json out.json

Exit 0 = no bombs (or advisory mode); 1 = bombs found with --strict.
SKEW_INCOMPATIBLE lists files the tool cannot judge (subprocess real clock).
Pure parsing helpers (parse_tap_failures, find_bombs) are unit-tested in
test_clock_skew_sweep.py. Zero tokens, zero network.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRELOAD = ROOT / "scripts" / "clock_skew_preload.cjs"
# node's tap reporter prints one top-level `not ok N - <file path>` per failing
# test FILE (nested subtests are indented). Anchor at column 0 so only files count.
_FILE_FAIL = re.compile(r"^not ok \d+ - (.+?\.test\.[cm]?[tj]sx?)\s*$", re.M)


# Files whose subject shells out to a process using the REAL clock (the skew
# only moves Node's Date), so JS-side "now" and subprocess "now" disagree under
# skew. Found 2026-11-02: gatedTests builds dates from Date.now() but runs
# scripts/gated_tests.sh, which calls `date`. A failure there is the tool's
# blind spot, not a fixture bomb. Add entries only with a reason.
SKEW_INCOMPATIBLE = {
    "server/gatedTests.test.ts": "shells out to gated_tests.sh which uses the real `date`",
}


def parse_tap_failures(tap: str) -> set[str]:
    """Top-level failing test files from node:test TAP output."""
    out = set()
    for m in _FILE_FAIL.finditer(tap):
        p = m.group(1).strip()
        try:
            p = os.path.relpath(p, ROOT) if os.path.isabs(p) else p
        except ValueError:
            pass
        out.add(p)
    return out


def find_bombs(baseline: set[str], by_skew: dict[int, set[str]]) -> dict[str, int]:
    """file -> smallest skew (days) at which it newly fails vs baseline."""
    bombs: dict[str, int] = {}
    for skew in sorted(by_skew):
        for f in by_skew[skew] - baseline:
            bombs.setdefault(f, skew)
    return bombs


def run_suite(skew: int, timeout: int, files: list[str] | None = None) -> tuple[str, int]:
    env = dict(os.environ)
    env["SKEW_DAYS"] = str(skew)
    env["NODE_OPTIONS"] = (env.get("NODE_OPTIONS", "") + f" --require {PRELOAD}").strip()
    p = subprocess.run(
        "npx tsx --test --test-reporter=tap " + " ".join(files or ["server/*.test.ts"]),
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout, shell=True,
    )
    return p.stdout + p.stderr, p.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skews", default="45,120", help="comma-separated forward skews in days")
    ap.add_argument("--timeout", type=int, default=1500, help="per-run timeout seconds")
    ap.add_argument("--strict", action="store_true", help="exit 1 when bombs are found")
    ap.add_argument("--json", help="write the result to this path")
    a = ap.parse_args(argv)
    skews = sorted({int(s) for s in a.skews.split(",") if s.strip()})
    if not skews or any(s <= 0 for s in skews):
        ap.error("skews must be positive integers")

    base_tap, _ = run_suite(0, a.timeout)
    baseline = parse_tap_failures(base_tap)
    by_skew: dict[int, set[str]] = {}
    for s in skews:
        tap, _ = run_suite(s, a.timeout)
        by_skew[s] = parse_tap_failures(tap)
    bombs = find_bombs(baseline, by_skew)
    for f in SKEW_INCOMPATIBLE:
        bombs.pop(f, None)
    # Whole-suite runs are concurrent; a loaded box makes timing tests flake.
    # Re-run each candidate ALONE at its skew and at skew 0; only a file that
    # passes alone today and fails alone under skew is a real time-bomb.
    for f in list(bombs):
        if parse_tap_failures(run_suite(0, a.timeout, [f])[0]):
            del bombs[f]  # fails alone today: environment/flake, not a bomb
        elif not parse_tap_failures(run_suite(bombs[f], a.timeout, [f])[0]):
            del bombs[f]  # passes alone under skew: concurrency flake

    print(f"baseline (skew 0) failing files: {len(baseline)} (excluded from bomb list)")
    for s in skews:
        print(f"skew +{s}d: {len(by_skew[s])} failing files, {len(by_skew[s] - baseline)} new")
    if bombs:
        print("\nCALENDAR TIME-BOMBS (passes today, fails when the clock moves):")
        for f, s in sorted(bombs.items(), key=lambda kv: (kv[1], kv[0])):
            print(f"  fuse <= {s:>4}d  {f}")
        print("Fix: make fixture dates relative to now (see server/gridDemand.test.ts, 2026-11-02).")
    else:
        print("\nno calendar time-bombs found at the tested skews.")
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"baseline_failing": sorted(baseline),
             "by_skew": {str(k): sorted(v) for k, v in by_skew.items()},
             "bombs": bombs}, indent=2))
    return 1 if (bombs and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
