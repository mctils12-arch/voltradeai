#!/usr/bin/env python3
"""
cpc_degree_days.py — NOAA CPC population-weighted daily degree days,
CONUS states (BUILD ORDER 3 #3, 2026-07-05; access probed keyless the
same day). US government work, public domain; attribution "NOAA Climate
Prediction Center".

Source files (per year, full year-to-date in each):
  ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/<year>/
    StatesCONUS.Heating.txt / StatesCONUS.Cooling.txt
Format: three header lines (Product/Regions/Weights), then a pipe-
delimited date-column header and one row per state. Values are daily
degree days (integers).

Selection (stated here + manifest): StatesCONUS Heating + Cooling only,
2016 -> current year. Census-division and fuel-weighted variants exist
upstream and are deliberately not archived in v1.

Output: datacore/cpc/degree_days.json — keyed "STATE|H"/"STATE|C" ->
{dates aligned to a shared per-kind axis}; whole-file rebuild per run
(each pull re-serves the current year fully; git keeps vintages).

HYPOTHESIS (gate-locked): population-weighted degree-day departures lead
natgas/power demand and utility earnings surprises.

Run (session-side ~weekly; stdlib only):
  python3 scripts/cpc_degree_days.py
"""
import json
import os
import urllib.request
from datetime import datetime, timezone

BASE_URL = "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data"
OUT = os.path.join(os.path.dirname(__file__), "..", "datacore", "cpc", "degree_days.json")
FIRST_YEAR = 2016
KINDS = [("Heating", "H"), ("Cooling", "C")]


def parse_states_file(text):
    """One StatesCONUS file -> (dates, {state: [values]}). Non-numeric
    cells become None (gaps stay gaps); short rows pad with None; a
    malformed date header refuses the whole file."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    hdr_i = next((i for i, l in enumerate(lines) if l.startswith("Region|")), None)
    if hdr_i is None:
        raise ValueError("no Region| header — format changed, refuse to guess")
    raw_dates = lines[hdr_i].split("|")[1:]
    dates = []
    for d in raw_dates:
        d = d.strip()
        if len(d) == 8 and d.isdigit():
            dates.append(f"{d[0:4]}-{d[4:6]}-{d[6:8]}")
        else:
            raise ValueError(f"malformed date column {d!r} — refuse to guess")
    out = {}
    for l in lines[hdr_i + 1:]:
        parts = l.split("|")
        state = parts[0].strip()
        if not state or len(state) > 2:
            continue
        vals = []
        for c in parts[1:len(dates) + 1]:
            c = c.strip()
            try:
                vals.append(int(c))
            except ValueError:
                vals.append(None)
        vals += [None] * (len(dates) - len(vals))
        out[state] = vals
    return dates, out


def fetch_year(year, kind_name):
    url = f"{BASE_URL}/{year}/StatesCONUS.{kind_name}.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return parse_states_file(r.read().decode("utf-8", "replace"))


def main():
    now_year = datetime.now(timezone.utc).year
    series = {}   # "STATE|K" -> {date: value}
    axis = {"H": [], "C": []}
    failures = 0
    for year in range(FIRST_YEAR, now_year + 1):
        for kind_name, k in KINDS:
            try:
                dates, states = fetch_year(year, kind_name)
            except Exception as e:
                failures += 1
                print(f"  ! {year} {kind_name}: {e}")
                continue
            axis[k].extend(dates)
            for st, vals in states.items():
                d = series.setdefault(f"{st}|{k}", {})
                for i, v in enumerate(vals):
                    if v is not None:
                        d[dates[i]] = v
            print(f"  {year} {kind_name}: {len(states)} states x {len(dates)} days")
    if not series:
        raise SystemExit("nothing parsed — refusing to write an empty artifact")
    # emit aligned arrays on each kind's date axis (sorted, deduped)
    doc_series = {}
    axes = {k: sorted(set(a)) for k, a in axis.items()}
    for key, dmap in sorted(series.items()):
        k = key.split("|")[1]
        doc_series[key] = [dmap.get(d) for d in axes[k]]
    doc = {
        "built": datetime.now(timezone.utc).isoformat(),
        "source": "NOAA CPC population-weighted daily degree days (StatesCONUS)",
        "attribution": "NOAA Climate Prediction Center",
        "selection": "StatesCONUS Heating+Cooling 2016->current; census-division and fuel-weighted variants deliberately not archived in v1",
        "axes": axes,
        "n_series": len(doc_series),
        "series": doc_series,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT)} ({os.path.getsize(OUT)/1e6:.2f} MB): "
          f"{len(doc_series)} series, H axis {len(axes['H'])}d, C axis {len(axes['C'])}d, {failures} failures")


if __name__ == "__main__":
    main()
