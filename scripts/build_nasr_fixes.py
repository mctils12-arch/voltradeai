#!/usr/bin/env python3
"""Build datacore/aircraft/nasr_fixes.json from the FAA NASR 28-day CSV bundle.

FAA NASR is US-government public-domain data. Usage:
  curl -o csv.zip https://nfdc.faa.gov/webContent/28DaySub/extra/<DD>_<Mon>_<YYYY>_CSV.zip
  unzip csv.zip -d nasr && python3 scripts/build_nasr_fixes.py nasr
Output: {"cycle": "YYYY-MM-DD", "fixes": {"ID": [lat, lon]}} (5 decimals ~1 m).
Fixes (FIX_BASE) win over navaids (NAV_BASE) on an id collision; the first
navaid row wins among duplicate navaid ids. Deterministic (sorted keys).
Also emits "magVar": {"ID": [deg_east_positive, survey_year]} for each navaid
whose POSITION came from NAV_BASE (NAV_BASE MAG_VARN/MAG_VARN_HEMIS/
MAG_VARN_YEAR) — a VOR radial is MAGNETIC, so true bearing = radial + magVar
(east positive). Existing consumers read only "fixes"; a navaid shadowed by a
FIX_BASE ident, or with no variation on file, gets no entry (never guessed).
"""
import csv, json, sys, os

def main(src, out=None):
    fixes, cycle, magvar = {}, None, {}
    with open(os.path.join(src, "FIX_BASE.csv"), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cycle = cycle or r["EFF_DATE"].replace("/", "-")
            fixes[r["FIX_ID"].strip().upper()] = [round(float(r["LAT_DECIMAL"]), 5), round(float(r["LONG_DECIMAL"]), 5)]
    nav_added = 0
    with open(os.path.join(src, "NAV_BASE.csv"), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            k = r["NAV_ID"].strip().upper()
            if not k or k in fixes or not r["LAT_DECIMAL"] or not r["LONG_DECIMAL"]:
                continue
            fixes[k] = [round(float(r["LAT_DECIMAL"]), 5), round(float(r["LONG_DECIMAL"]), 5)]
            nav_added += 1
            mv = (r.get("MAG_VARN") or "").strip()
            hem = (r.get("MAG_VARN_HEMIS") or "").strip().upper()
            if mv and hem in ("E", "W"):
                yr = (r.get("MAG_VARN_YEAR") or "").strip()
                magvar[k] = [float(mv) if hem == "E" else -float(mv), int(yr) if yr.isdigit() else None]
    out = out or os.path.join(os.path.dirname(__file__), "..", "datacore", "aircraft", "nasr_fixes.json")
    with open(out, "w") as f:
        json.dump({"cycle": cycle, "source": "FAA NASR FIX_BASE + NAV_BASE (public domain)",
                   "fixes": dict(sorted(fixes.items())), "magVar": dict(sorted(magvar.items()))}, f, separators=(",", ":"))
    print(f"cycle {cycle}: {len(fixes)} idents ({nav_added} navaids, {len(magvar)} with magVar) -> {out}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nasr")
