#!/usr/bin/env python3
"""Build datacore/aircraft/nasr_airways.json from the FAA NASR 28-day CSV bundle.

FAA NASR is US-government public domain. Uses AWY_BASE.csv's AIRWAY_STRING
(the ordered fix/navaid idents of each airway). Usage, after extracting the
same bundle build_nasr_fixes.py reads:  python3 scripts/build_nasr_airways.py nasr
Output: {"cycle", "airways": {"J150": [["HTO", ..., "OOD"], ...]}} — an airway id
can exist in several regions (AWY_LOCATION C/A/H), so each id maps to a LIST of
variants. Deterministic (sorted keys, file order within an id).
"""
import csv, json, sys, os

def main(src):
    airways, cycle = {}, None
    with open(os.path.join(src, "AWY_BASE.csv"), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cycle = cycle or r["EFF_DATE"].replace("/", "-")
            pts = r["AIRWAY_STRING"].upper().split()
            if len(pts) >= 2:
                airways.setdefault(r["AWY_ID"].strip().upper(), []).append(pts)
    out = os.path.join(os.path.dirname(__file__), "..", "datacore", "aircraft", "nasr_airways.json")
    with open(out, "w") as f:
        json.dump({"cycle": cycle, "source": "FAA NASR AWY_BASE AIRWAY_STRING (public domain)",
                   "airways": dict(sorted(airways.items()))}, f, separators=(",", ":"))
    print(f"cycle {cycle}: {len(airways)} airway ids -> {out}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nasr")
