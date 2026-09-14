#!/usr/bin/env python3
"""grid_generation_gate1_national_sweep.py — closes the long-open NEXT item
from research/open_questions.md's FUSION HYPOTHESES entry (filed 2026-09-12,
restated unclaimed across five subsequent sessions through 2026-09-14): "no
other BA/fuel pair from the original national gate-1 run has been
individually re-verified ... a future session could sweep the remaining
FAIL/INCONCLUSIVE cells across all tracked respondents rather than waiting
for each to surface one at a time."

Reuses grid_generation_gate1_ba.py's `build_report()` unchanged (EDGE
DOCTRINE #3 — no new fetch/reconcile logic) and adds exactly one thing: a
filter that separates REAL findings (an actual FAIL, or an INCONCLUSIVE with
a substantive reason) from the trivial "missing on one side" INCONCLUSIVE
cells that just mean a BA has no registry-matched capacity or no EIA-930
rows for that fuel bucket — noise this thread's prior manual JSON reads had
to skip past by eye each time.

Usage: python3 scripts/grid_generation_gate1_national_sweep.py
       [--days N] [--tolerance F] [--respondents CSV] [--source eia860|polygon]
(same flags as grid_generation_gate1_ba.py, passed straight through)
"""
import argparse
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location(
    "grid_generation_gate1_ba", os.path.join(_HERE, "grid_generation_gate1_ba.py"))
_gate1_ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate1_ba)


def is_real_finding(verdict):
    """verdict: one fuel bucket's entry under report["regions"][ba]["verdicts"].
    True for a FAIL, or an INCONCLUSIVE whose reason is something other than
    the trivial "missing on one side" (no registry capacity or no EIA-930
    generation on that side — nothing to reconcile, not a data quality
    finding)."""
    v = verdict.get("verdict")
    if v == "FAIL":
        return True
    if v == "INCONCLUSIVE":
        return verdict.get("reason") != "missing on one side"
    return False


def summarize_report(report):
    """report: grid_generation_gate1_ba.build_report()'s return value.
    Returns a flat list of {ba, fuel, verdict, ...verdict fields} for every
    real finding across every respondent, sorted by (ba, fuel) for a stable,
    diffable summary."""
    findings = []
    for ba, region in report.get("regions", {}).items():
        for fuel, verdict in region.get("verdicts", {}).items():
            if is_real_finding(verdict):
                findings.append({"ba": ba, "fuel": fuel, **verdict})
    findings.sort(key=lambda f: (f["ba"], f["fuel"]))
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--respondents", default=",".join(_gate1_ba.DEFAULT_RESPONDENTS))
    ap.add_argument("--source", choices=("eia860", "polygon"), default="eia860")
    args = ap.parse_args()

    respondents = [r.strip() for r in args.respondents.split(",") if r.strip()]
    report = _gate1_ba.build_report(respondents, days=args.days, tolerance=args.tolerance, source=args.source)
    findings = summarize_report(report)

    print(json.dumps({
        "window": report["window"],
        "source": report["source"],
        "respondents_checked": respondents,
        "real_findings_count": len(findings),
        "real_findings": findings,
    }, indent=2))


if __name__ == "__main__":
    main()
