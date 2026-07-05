#!/usr/bin/env python3
"""
eia_weekly.py — EIA weekly petroleum + natural gas storage series (BUILD
ORDER 3 #2, 2026-07-05; all endpoints probed keyless the same day). US
government work, public domain; attribution "U.S. Energy Information
Administration".

The keyless hist_xls family is PROVEN (the tank-fill EIA comparator has
parsed W_EPC0_SAX_YCUOK since v1). Each XLS carries its own series title
in the Data sheet header — titles are READ FROM THE FILE and stored, never
assumed from our labels (a moved/renamed series shows up as a title change
in git diff, not silent mislabeling).

Selection (stated here + manifest): five weekly series covering the energy
regime axis — US commercial crude stocks, total gasoline, total distillate,
SPR crude, and lower-48 natgas working storage. Cushing is deliberately NOT
duplicated here (the tank-fill comparator owns it).

Output: datacore/eia/weekly_series.json — whole-file rebuild per run (the
source serves full history every pull; git history keeps vintages, and the
EIA's own revisions surface as diffs).

HYPOTHESIS (gate-locked): storage-vs-seasonal-band deltas condition the
energy regime the bot classifies; also the standing external-truth source
for any future inventory root.

Run (session-side ~weekly; dep xlrd like the comparator):
  python3 scripts/eia_weekly.py
"""
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone

OUT = os.path.join(os.path.dirname(__file__), "..", "datacore", "eia", "weekly_series.json")

SERIES = [
    {"key": "crude_stocks_us", "url": "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls"},
    {"key": "gasoline_stocks_us", "url": "https://www.eia.gov/dnav/pet/hist_xls/WGTSTUS1w.xls"},
    {"key": "distillate_stocks_us", "url": "https://www.eia.gov/dnav/pet/hist_xls/WDISTUS1w.xls"},
    {"key": "spr_crude_stocks", "url": "https://www.eia.gov/dnav/pet/hist_xls/WCSSTUS1w.xls"},
    {"key": "natgas_working_l48", "url": "https://www.eia.gov/dnav/ng/hist_xls/NW2_EPG0_SWO_R48_BCFw.xls"},
]


def parse_data_sheet(rows, datemode):
    """EIA 'Data 1' sheet rows -> (title, [[iso_date, value], ...]).
    Row layout (proven by the tank-fill comparator): a title row naming the
    series, then date|value pairs. Non-finite values are SKIPPED (gaps stay
    gaps, never zero)."""
    import xlrd
    title = None
    out = []
    for row in rows:
        if len(row) < 2:
            continue
        c0, c1 = row[0], row[1]
        if isinstance(c1, str) and c1.strip() and not isinstance(c0, float) and title is None:
            # header block: second cell carries the series description
            if "Weekly" in c1 or "weekly" in c1:
                title = c1.strip()
            continue
        if isinstance(c0, float) and isinstance(c1, (int, float)):
            try:
                d = xlrd.xldate_as_datetime(c0, datemode).date().isoformat()
            except Exception:
                continue
            out.append([d, float(c1)])
    return title, out


def fetch_series(url):
    import xlrd
    req = urllib.request.Request(url, headers={"User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)"})
    with tempfile.NamedTemporaryFile(suffix=".xls", delete=False) as tf:
        with urllib.request.urlopen(req, timeout=60) as r:
            tf.write(r.read())
        p = tf.name
    try:
        wb = xlrd.open_workbook(p)
        # data lives on the sheet named 'Data 1' (index 1); sheet 0 is Contents
        sh = wb.sheet_by_index(1)
        rows = [sh.row_values(i) for i in range(sh.nrows)]
        return parse_data_sheet(rows, wb.datemode)
    finally:
        os.unlink(p)


def main():
    doc = {
        "built": datetime.now(timezone.utc).isoformat(),
        "source": "EIA dnav hist_xls weekly series (keyless, public domain)",
        "attribution": "U.S. Energy Information Administration",
        "selection": "five weekly energy-regime series; Cushing deliberately not duplicated (tank-fill comparator owns it)",
        "series": {},
    }
    failures = 0
    for s in SERIES:
        try:
            title, points = fetch_series(s["url"])
            if not points:
                raise ValueError("no data points parsed")
            doc["series"][s["key"]] = {
                "title_as_published": title,
                "url": s["url"],
                "n": len(points),
                "first": points[0][0],
                "last": points[-1][0],
                "points": points,
            }
            print(f"  {s['key']:24s} {len(points):5d} pts  {points[0][0]} .. {points[-1][0]}  | {str(title)[:60]}")
        except Exception as e:
            failures += 1
            print(f"  ! {s['key']}: {e}")
    if failures == len(SERIES):
        raise SystemExit("every series failed — refusing to write an empty artifact")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT)} ({os.path.getsize(OUT)/1e6:.2f} MB), {failures} failures")


if __name__ == "__main__":
    main()
