#!/usr/bin/env python3
"""
build_entity_spine.py — aircraft entity spine v1 (GIP Part 3b; BUILD ORDER 2
#1, 2026-07-05). Joins the FAA Releasable Aircraft registry against OUR
archived icao24 hexes and emits the small committed artifact
datacore/aircraft/entity_spine.json — one identity per airframe we have
actually observed. NEVER commits the full 300k-row FAA dump (GIP scope rule).

Sources (US government works, public domain):
  - FAA Releasable Aircraft DB: registry.faa.gov ReleasableAircraft.zip
    (MASTER.txt col "MODE S CODE HEX" is the exact join key; ACFTREF.txt
    maps MFR MDL CODE -> manufacturer/model). GOTCHA (probed 2026-07-05):
    the download host 403s non-browser User-Agents — a browser UA is sent.
  - hex list: /api/data/aircraft/hexes on prod (or --hexes file, one hex
    per line, for offline runs).

Match basis is the exact Mode S hex — an inference-free join; the evidence
envelope on every record states exactly that. Non-US hexes simply don't
match (international registries are a filed follow-up, GIP Part 5a).

Run (session-side; stdlib only):
  python3 scripts/build_entity_spine.py --faa-dir /path/with/MASTER.txt \
      --hexes-url https://voltradeai.com/api/data/aircraft/hexes
"""
import argparse
import csv
import io
import json
import os
import urllib.request
from datetime import datetime, timezone

OUT = os.path.join(os.path.dirname(__file__), "..", "datacore", "aircraft", "entity_spine.json")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

REGISTRANT_TYPES = {
    "1": "individual", "2": "partnership", "3": "corporation", "4": "co-owned",
    "5": "government", "7": "llc", "8": "non-citizen-corporation", "9": "non-citizen-co-owned",
}


def parse_acftref(text):
    """ACFTREF.txt -> {code: {"mfr": ..., "model": ...}}."""
    out = {}
    rdr = csv.reader(io.StringIO(text))
    header = next(rdr, None)
    if not header:
        return out
    idx = {h.lstrip("﻿").strip().upper(): i for i, h in enumerate(header)}
    ci, mi, di = idx.get("CODE"), idx.get("MFR"), idx.get("MODEL")
    if ci is None or mi is None or di is None:
        return out
    for row in rdr:
        if len(row) <= max(ci, mi, di):
            continue
        code = row[ci].strip()
        if code:
            out[code] = {"mfr": row[mi].strip() or None, "model": row[di].strip() or None}
    return out


def _date8(s):
    s = (s or "").strip()
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else None


def parse_master(text, wanted_hexes, acftref):
    """MASTER.txt -> {hex: spine record} for hexes in `wanted_hexes` only.
    Column names are taken from the header row, not positions — the FAA
    reorders occasionally."""
    rdr = csv.reader(io.StringIO(text))
    header = next(rdr, None)
    if not header:
        return {}
    # header cells carry stray spaces and a BOM on the first cell
    idx = {h.lstrip("﻿").strip().upper(): i for i, h in enumerate(header)}
    need = ["N-NUMBER", "NAME", "TYPE REGISTRANT", "MFR MDL CODE", "YEAR MFR",
            "CERT ISSUE DATE", "EXPIRATION DATE", "MODE S CODE HEX"]
    cols = {k: idx.get(k) for k in need}
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise SystemExit(f"MASTER.txt header missing {missing} — format changed, refuse to guess")
    retrieved = datetime.now(timezone.utc).date().isoformat()
    out = {}
    for row in rdr:
        if len(row) <= cols["MODE S CODE HEX"]:
            continue
        hex_ = row[cols["MODE S CODE HEX"]].strip().lower()
        if not hex_ or hex_ not in wanted_hexes:
            continue
        ref = acftref.get(row[cols["MFR MDL CODE"]].strip(), {})
        year = row[cols["YEAR MFR"]].strip()
        out[hex_] = {
            "n_number": "N" + row[cols["N-NUMBER"]].strip(),
            "owner": row[cols["NAME"]].strip(),
            "registrant_type": REGISTRANT_TYPES.get(row[cols["TYPE REGISTRANT"]].strip()),
            "mfr": ref.get("mfr"),
            "model": ref.get("model"),
            "year_mfr": int(year) if year.isdigit() else None,
            "cert_issue": _date8(row[cols["CERT ISSUE DATE"]]),
            "expiration": _date8(row[cols["EXPIRATION DATE"]]),
            "evidence": {
                "source": "FAA Releasable Aircraft DB (US government work, public domain)",
                "retrieved": retrieved,
                "match": "exact Mode S hex",
            },
        }
    return out


def fetch_hexes(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        doc = json.load(r)
    return {h["i"].lower() for h in doc.get("hexes", []) if h.get("i")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faa-dir", required=True, help="directory containing MASTER.txt + ACFTREF.txt")
    ap.add_argument("--hexes-url", help="prod /api/data/aircraft/hexes URL")
    ap.add_argument("--hexes", help="file with one icao24 hex per line (offline)")
    args = ap.parse_args()

    if args.hexes:
        with open(args.hexes) as fh:
            wanted = {l.strip().lower() for l in fh if l.strip()}
    elif args.hexes_url:
        wanted = fetch_hexes(args.hexes_url)
    else:
        raise SystemExit("need --hexes or --hexes-url")
    print(f"{len(wanted)} archived hexes to match")

    with open(os.path.join(args.faa_dir, "ACFTREF.txt"), encoding="utf-8", errors="replace") as fh:
        acftref = parse_acftref(fh.read())
    with open(os.path.join(args.faa_dir, "MASTER.txt"), encoding="utf-8", errors="replace") as fh:
        entities = parse_master(fh.read(), wanted, acftref)
    print(f"matched {len(entities)} of {len(wanted)} hexes to FAA registrations "
          f"(unmatched = non-US or unassigned — expected, never guessed)")

    doc = {
        "built": datetime.now(timezone.utc).isoformat(),
        "source": "FAA Releasable Aircraft DB x voltradeai aircraft archive hexes",
        "license": "US government work, public domain; attribution 'FAA Aircraft Registry'",
        "archived_hexes": len(wanted),
        "matched": len(entities),
        "entities": dict(sorted(entities.items())),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"wrote {os.path.relpath(OUT)} ({os.path.getsize(OUT)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
