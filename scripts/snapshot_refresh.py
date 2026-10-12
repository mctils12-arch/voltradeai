#!/usr/bin/env python3
"""
snapshot_refresh.py — ONE command that refreshes the four bundled, git-versioned
whole-file snapshots (CLAUDE.md EDGE DOCTRINE #3: compile recurring work).

  eia_weekly.py        -> datacore/eia/weekly_series.json    (needs xlrd)
  cpc_degree_days.py   -> datacore/cpc/degree_days.json      (stdlib)
  stb_rail.py          -> datacore/rail/ep724_carloads.json  (needs openpyxl)
  jodi_oil.py          -> datacore/jodi/primary_stocks.json  (stdlib)

Why a wrapper: these were "session-run" with no flag; eiaweekly sat 99d stale
(found 2026-10-12) and a bare sandbox `python3` lacked xlrd, so a naive run
silently failed ("every series failed"). The wrapper picks an interpreter
that can import the deps or fails BEFORE any write, and each refresher keeps
its own all-fail-no-write guard. Staleness is flagged by
scripts/research_state_check.py (bundled_* archive_freshness entries).

Run:
  python3 scripts/snapshot_refresh.py --dry-run
  python3 scripts/snapshot_refresh.py [--only eia,cpc,rail,jodi]
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED_MODULES = ("xlrd", "openpyxl")
CANDIDATE_PYTHONS = ("/usr/bin/python3", "/usr/local/bin/python3", "python3")
STEPS = [("eia", "eia_weekly.py"), ("cpc", "cpc_degree_days.py"),
         ("rail", "stb_rail.py"), ("jodi", "jodi_oil.py")]


def build_plan(py, only=None):
    """Ordered [(key, argv)]. Pure — unit-tested."""
    return [(k, [py, os.path.join(ROOT, "scripts", f)]) for k, f in STEPS
            if only is None or k in only]


def interpreter_ok(py):
    try:
        return subprocess.run([py, "-c", "import " + ",".join(REQUIRED_MODULES)],
                              capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def pick_python(candidates=CANDIDATE_PYTHONS, ok=interpreter_ok):
    return next((c for c in candidates if ok(c)), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = [x for x in a.only.split(",") if x] or None
    bad = [x for x in (only or []) if x not in dict(STEPS)]
    if bad:
        sys.exit(f"unknown --only {bad}; choose from {[k for k, _ in STEPS]}")
    py = pick_python()
    if py is None:
        sys.exit(f"no interpreter among {list(CANDIDATE_PYTHONS)} can import "
                 f"{list(REQUIRED_MODULES)} — pip install them (nothing written)")
    failed = []
    for key, argv in build_plan(py, only):
        print(f"== {key}: {' '.join(argv[1:])}", flush=True)
        if not a.dry_run and subprocess.run(argv, cwd=ROOT).returncode != 0:
            failed.append(key)  # independent sources: one outage must not block the rest
    if failed:
        sys.exit(f"failed: {failed}; others completed. Inspect `git diff --stat datacore`.")
    print("done — review `git diff --stat datacore`, commit the snapshots.")


if __name__ == "__main__":
    main()
