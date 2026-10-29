#!/usr/bin/env python3
"""scripts/source_validation.py — local CLI for the [RESEARCH] source
validation harness (repo-root source_validation.py, 2026-10-27).

Production runs the same code read-only through the diag probe
"source_validation" (the shadow log lives on the Railway volume; this
sandbox has none). scripts/ is not in the production image, so this file is
only a thin wrapper: point it at a COPY of voltrade_shadow_candidates.json.

Usage:
  python3 scripts/source_validation.py --log /path/to/voltrade_shadow_candidates.json
  python3 scripts/source_validation.py --log LOG --out report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import source_validation as sv  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", help="shadow candidates JSON (default: the storage_config path)")
    ap.add_argument("--out", help="write the full JSON report here")
    args = ap.parse_args(argv)

    if args.log:
        with open(args.log) as f:
            records = json.load(f)
    else:
        from shadow_portfolio import _load_shadow_log
        records = _load_shadow_log()
    if not isinstance(records, list):
        print("shadow log is not a JSON list", file=sys.stderr)
        return 2

    rep = sv.validate_sources(records)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1)
    pop = rep["population"][sv.PRIMARY_OUTCOME]
    print(f"{sv.PREREG_VERSION}: primary {sv.PRIMARY_OUTCOME}, n_unique={pop['n_unique']}, "
          f"days={pop['n_days']} ({pop['first_day']}..{pop['last_day']})")
    for verdict, names in rep["summary"].items():
        print(f"  {verdict:14s} {len(names):3d}  {', '.join(names)}")
    for nm, r in rep["sources"].items():
        if "ic" in r:
            print(f"  {nm:28s} n={r['n']:5d} IC={r['ic']:+.3f} CI={r.get('ci95')} q={r.get('q_bh')} {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
