#!/usr/bin/env python3
"""
sentinel2_refresh.py — ONE command for the whole Cushing tank-fill archive
catch-up (CLAUDE.md EDGE DOCTRINE #3: compile recurring reasoning into code).

Replaces the hand-run sequence a session had to remember (and which went
unrun for ~12 weeks once, then needed re-deriving):

  v1  sentinel2_tankfill.py --since D --compare        (keyless STAC)
  v2  cdse_chips.py --since D ; tankfill_estimator.py  (CDSE creds, append-only)
  v3  cdse_s1_chips.py --since D --redownload-missing  (rehydrate gitignored chips)
      tankfill_s1_estimator.py --no-gate               (whole-file rebuild; has
                                                        its own shrink guard)

Archive-only: no gate attempt, no trading path. Each archive stays
append-only in substance (v3 renormalizes per-tank medians on rebuild, a
documented property of that estimator).

Why a wrapper, not three habits:
 - the interpreter: numpy/xlrd/rasterio must be importable; a sandbox's bare
   `python3` may lack them (found 2026-10-11: 3.11 without numpy while
   /usr/bin/python3 had them). The wrapper re-execs under the first
   interpreter that can import them, or fails loudly BEFORE any write.
 - the --since date is derived from each archive's own newest record
   (minus an overlap), never from prose that goes stale.
 - v3 order is rehydrate THEN rebuild, always.

Run:
  python3 scripts/sentinel2_refresh.py --dry-run     # print the plan only
  python3 scripts/sentinel2_refresh.py               # execute
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2 = os.path.join(ROOT, "datacore", "sentinel2")
REQUIRED_MODULES = ("numpy", "xlrd", "rasterio")
CANDIDATE_PYTHONS = ("/usr/bin/python3", "/usr/local/bin/python3", "python3")
OVERLAP_DAYS = 5  # re-scan a few days back; every collector dedups by scene id


def newest_date(path, field="date", oldest=False):
    """Max (or, with oldest=True, min) `field` across a JSONL file, or None."""
    best = None
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line).get(field)
            except (ValueError, AttributeError) as e:
                print(f"[sentinel2_refresh] malformed line in {path}: {e}", file=sys.stderr)
                continue
            if d and (best is None or (d < best if oldest else d > best)):
                best = d
    return date.fromisoformat(best) if best else None


def since_for(newest, overlap=OVERLAP_DAYS, fallback="2026-06-20"):
    """ISO --since string: newest record minus overlap (fallback if empty)."""
    if newest is None:
        return fallback
    return (newest - timedelta(days=overlap)).isoformat()


def build_plan(py, since_v1, since_v2, since_v3):
    """Ordered list of (label, argv). Pure — unit-tested."""
    sc = os.path.join(ROOT, "scripts")
    return [
        ("v1 readings", [py, f"{sc}/sentinel2_tankfill.py", "--since", since_v1, "--compare"]),
        ("v2 chips", [py, f"{sc}/cdse_chips.py", "--since", since_v2]),
        ("v2 estimator", [py, f"{sc}/tankfill_estimator.py"]),
        ("v3 chips (rehydrate)", [py, f"{sc}/cdse_s1_chips.py", "--since", since_v3, "--redownload-missing"]),
        ("v3 estimator", [py, f"{sc}/tankfill_s1_estimator.py", "--no-gate"]),
    ]


def interpreter_ok(py):
    try:
        r = subprocess.run([py, "-c", "import " + ",".join(REQUIRED_MODULES)],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def pick_python(candidates=CANDIDATE_PYTHONS, ok=interpreter_ok):
    for c in candidates:
        if ok(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    py = pick_python()
    if py is None:
        sys.exit("no interpreter among %s can import %s — pip install them first "
                 "(nothing was written)" % (list(CANDIDATE_PYTHONS), list(REQUIRED_MODULES)))
    plan = build_plan(
        py,
        since_for(newest_date(os.path.join(S2, "readings.jsonl"))),
        since_for(newest_date(os.path.join(S2, "readings_v2.jsonl"))),
        # v3 rebuilds the WHOLE file, so every indexed chip must be on disk:
        # rehydrate from the OLDEST indexed scene, not the newest.
        since_for(newest_date(os.path.join(S2, "s1_chips_index.jsonl"), oldest=True), overlap=0),
    )
    for label, argv in plan:
        print(f"== {label}: {' '.join(argv[1:])}", flush=True)
        if a.dry_run:
            continue
        r = subprocess.run(argv, cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"step '{label}' failed (rc={r.returncode}); later steps skipped. "
                     "Inspect `git diff --stat datacore/sentinel2` before committing.")
    print("done — review `git diff --stat datacore/sentinel2`, commit the archives.")


if __name__ == "__main__":
    main()
