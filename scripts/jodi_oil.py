#!/usr/bin/env python3
"""
jodi_oil.py — JODI World Primary oil database: closing-stock levels
(DATACORE MAXIMUS census build #3, 2026-07-06; source probed live the
same day: 23.3MB zip, 6,872,400 rows, 118 REF_AREAs, latest period
2026-04 — the ~2-month publication lag as documented).

Free with acknowledgment; attribution "JODI Oil World Database (Joint
Organisations Data Initiative)".

WHY (hypothesis, gate-locked): non-OECD crude/product stock builds
(Saudi/UAE/India) are invisible in EIA weeklies but feed Brent term
structure; moderately uncrowded because the 2-month lag scares off fast
money — the value is in LEVEL trends and revisions, not nowcasting.
DIRECT gate-1 partner for the tank-shadow root (a country whose JODI
stocks build should show filling tanks from orbit).

Verified source shape (probe 2026-07-06):
  REF_AREA,TIME_PERIOD,ENERGY_PRODUCT,FLOW_BREAKDOWN,UNIT_MEASURE,OBS_VALUE,ASSESSMENT_CODE
  - dense grid: 10 flows x 4 products x 5 units per area-month
  - OBS_VALUE carries non-numeric markers: '-' and 'N/A' = not
    available, 'x' = not applicable ('N/A' found by full-file scan,
    29,736 occurrences in the filtered subset — the header probe alone
    missed it). SKIPPED, never zero (zero is a real level).
  - ASSESSMENT_CODE preserved verbatim per point (JODI's own data-
    quality flag; we record it, we do not interpret it).

Selection (stated here + manifest): FLOW_BREAKDOWN == CLOSTLV (closing
stock LEVELS — the inventory signal), UNIT_MEASURE == KBBL (thousand
barrels, the natural stocks unit; CONVBBL is a derived conversion, KBD
is a flow rate, KL/KTONS duplicate in other units). All 4 primary
products kept (CRUDEOIL, NGL, OTHERCRUDE, TOTCRUDE) x all 118 areas,
full history 2002+. ~91k points (measured 91,167 on the probe file).

Output: datacore/jodi/primary_stocks.json — WHOLE-FILE REBUILD per run
(the source ships full history every month; git history keeps vintages,
and JODI's own revisions surface as diffs — eiaweekly precedent).
Refuses to write if parsing yields zero series (all-fail = no write).

Run (session-side, monthly after the ~19th):
  python3 scripts/jodi_oil.py [path-to-downloaded-zip]
(downloads the zip itself when no path is given)
"""
import csv
import io
import json
import os
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone

URL = "https://www.jodidata.org/_resources/files/downloads/oil-data/world_Primary_CSV.zip"
OUT = os.path.join(os.path.dirname(__file__), "..", "datacore", "jodi", "primary_stocks.json")
FLOW = "CLOSTLV"
UNIT = "KBBL"
PRODUCTS = ("CRUDEOIL", "NGL", "OTHERCRUDE", "TOTCRUDE")


def parse_rows(fh) -> dict:
    """CSV file handle -> {series_key: {points: [[period, value, assessment], ...]}}.

    Filters to closing-stock levels in KBBL; skips '-'/'x'/'' markers
    (not-available / not-applicable are NOT zero); preserves the
    assessment code verbatim. Rows shorter than the 7 probed columns or
    with unparseable values are counted, never silently dropped.
    """
    series: dict = {}
    skipped_markers = 0
    bad_rows = 0
    r = csv.reader(fh)
    header = next(r, None)
    if not header or header[0].strip().lstrip("﻿") != "REF_AREA":
        raise ValueError(f"unexpected header (source shape changed?): {header!r}")
    for row in r:
        if len(row) < 7:
            bad_rows += 1
            continue
        area, period, prod, flow, unit, val, code = row[:7]
        if flow != FLOW or unit != UNIT or prod not in PRODUCTS:
            continue
        if val in ("-", "x", "", "N/A"):
            skipped_markers += 1
            continue
        try:
            v = float(val)
        except ValueError:
            bad_rows += 1
            continue
        key = f"{area}|{prod}"
        series.setdefault(key, []).append([period, v, code])
    for pts in series.values():
        pts.sort(key=lambda p: p[0])
    return {"series": series, "skipped_markers": skipped_markers, "bad_rows": bad_rows}


def build_artifact(parsed: dict, source_last_modified: str | None, now_iso: str) -> dict:
    series = parsed["series"]
    if not series:
        raise RuntimeError("zero series parsed — refusing to write (all-fail = no write)")
    out_series = {}
    latest = ""
    for key in sorted(series):
        pts = series[key]
        out_series[key] = {
            "points": pts,
            "n": len(pts),
            "first": pts[0][0],
            "last": pts[-1][0],
        }
        if pts[-1][0] > latest:
            latest = pts[-1][0]
    return {
        "source": URL,
        "source_last_modified": source_last_modified,
        "built_at": now_iso,
        "selection": {
            "flow": FLOW + " (closing stock levels)",
            "unit": UNIT + " (thousand barrels)",
            "products": list(PRODUCTS),
            "markers_skipped_not_zeroed": parsed["skipped_markers"],
            "bad_rows": parsed["bad_rows"],
        },
        "attribution": "JODI Oil World Database (Joint Organisations Data Initiative)",
        "latest_period": latest,
        "series_count": len(out_series),
        "series": out_series,
    }


def main() -> int:
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
        last_modified = None
        print(f"using local zip {zip_path}")
    else:
        zip_path = "/tmp/jodi_primary.zip"
        print(f"downloading {URL} ...")
        req = urllib.request.Request(URL, headers={"User-Agent": "voltradeai-datacore/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(zip_path, "wb") as f:
            last_modified = resp.headers.get("Last-Modified")
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    z = zipfile.ZipFile(zip_path)
    member = z.namelist()[0]
    print(f"parsing {member} (streaming) ...")
    with z.open(member) as f:
        parsed = parse_rows(io.TextIOWrapper(f, "utf-8"))
    artifact = build_artifact(parsed, last_modified, datetime.now(timezone.utc).isoformat())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(artifact, f, separators=(",", ":"))
    size_mb = os.path.getsize(OUT) / 1e6
    print(
        f"WROTE {OUT}: {artifact['series_count']} series, latest {artifact['latest_period']}, "
        f"{size_mb:.1f}MB; markers skipped {artifact['selection']['markers_skipped_not_zeroed']}, "
        f"bad rows {artifact['selection']['bad_rows']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
