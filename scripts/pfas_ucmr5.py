#!/usr/bin/env python3
"""
pfas_ucmr5.py — EPA UCMR 5 PFAS drinking-water occurrence data
(Location Context Engine hazard layer — research/location_context_engine.md;
the queued "PFAS (EPA UCMR5/SDWIS)" item, logged as still-missing across at
least six prior session entries in research/experiments.md).

Source (verified live 2026-07-15): EPA's Fifth Unregulated Contaminant
Monitoring Rule national occurrence data file — keyless ZIP,
https://www.epa.gov/system/files/other-files/2023-08/ucmr5-occurrence-data.zip
(13MB zip / 308MB UCMR5_All.txt, cp1252-encoded tab-delimited — the µg/L unit
string decodes as garbage under utf-8, correctly under cp1252/latin-1;
Last-Modified 2026-02-12 at probe time — EPA updates this file periodically
as new monitoring batches arrive, not on a fixed daily/weekly cadence).
Probed full-file scan 2026-07-15: 1,928,117 rows, 30 distinct Contaminant
values = the 29 UCMR5 PFAS analytes + lithium (excluded below — lithium is
UCMR5's one non-PFAS analyte, unrelated to this layer).

RAW/FACTUAL layer (location_context_engine.md honesty rails): a monitoring
RESULT is EPA's own published federal record. We report DETECTED
concentrations exactly as EPA reports them (AnalyticalResultsSign=='=' rows
only; '<' rows are non-detects below the method reporting level and carry no
numeric value, so they contribute no detection here — a "not detected"
sample is not a zero-concentration claim). This is explicitly NOT a
health-risk or MCL-exceedance verdict: PFAS MCLs took effect under EPA's
2024 National Primary Drinking Water Regulation, but compliance MONITORING
for those MCLs doesn't begin until 2027-2029 — UCMR5 (2023-2025 monitoring)
predates and is separate from that compliance data, and this file carries no
exceedance flag of its own. predictive:false throughout the output.

GEOCODE JOIN: UCMR5's own file carries no coordinates, only PWSID. Joined
against EPA's "Community Water System Service Areas" ArcGIS FeatureServer
(same ArcGIS org, cJ9YHowT8TU7DUyn, that already hosts the Superfund NPL
layer server/superfund.ts uses) via `returnCentroid=true&returnGeometry=
false` — the boundary POLYGON's centroid stands in for a point location.
Service areas can be irregular, so a centroid is an APPROXIMATION of where
the system serves, not the exact intake/treatment-plant location — this is
stated in the output, never silently presented as exact. Queried ONLY for
PWSIDs with >=1 PFAS detection (verified 2026-07-15: 3,539 of 10,297
UCMR5-monitored systems), never the full ~124k-system boundary layer.
A PWSID absent from that FeatureServer (unmapped service area) is DROPPED
from the artifact, never geocoded by guesswork — counted honestly in
`totals.systems_ungeocoded`.

Output: datacore/pfas/ucmr5_detections.json — WHOLE-FILE REBUILD per run
(EPA republishes the full occurrence file each release, not a delta feed;
git history keeps vintages — JODI/eiaweekly precedent). Per-system
detections are aggregated BY CONTAMINANT (max detected value + first/last
detected date + sample count), not one row per lab sample, so the per-system
record is a useful summary instead of dozens of near-duplicate rows from
repeated sample points/events.

Run (session-side; EPA's own release cadence is roughly quarterly, not
daily/weekly, so this does not need a tight refresh loop):
  python3 scripts/pfas_ucmr5.py [path-to-downloaded-zip]
(downloads the zip itself when no path is given; the geocode join always
runs live regardless of whether the zip was local or downloaded)
"""
import io
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone

URL = "https://www.epa.gov/system/files/other-files/2023-08/ucmr5-occurrence-data.zip"
MEMBER = "UCMR5_All.txt"
OUT = os.path.join(os.path.dirname(__file__), "..", "datacore", "pfas", "ucmr5_detections.json")
GEOCODE_SERVICE = (
    "https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/"
    "Water_System_Boundaries/FeatureServer/0/query"
)
GEOCODE_BATCH = 150
NON_PFAS_CONTAMINANTS = {"lithium"}
REQUIRED_COLUMNS = [
    "PWSID", "PWSName", "State", "CollectionDate", "Contaminant",
    "MRL", "Units", "AnalyticalResultsSign", "AnalyticalResultValue",
]


def _to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _date_key(mdy):
    """'9/27/2023' -> (2023, 9, 27) for chronological comparison; an
    unparsable date sorts last so it never crashes the aggregate."""
    try:
        m, d, y = mdy.split("/")
        return (int(y), int(m), int(d))
    except (ValueError, AttributeError):
        return (9999, 12, 31)


def parse_ucmr5(fh) -> dict:
    """UCMR5_All.txt file handle (tab-delimited, header row) -> per-PWSID
    PFAS detection aggregate. Only AnalyticalResultsSign=='=' rows (an
    actual detection with a numeric value) contribute; '<' (non-detect)
    rows are counted in totals but never treated as a zero-concentration
    detection. lithium rows are dropped (UCMR5's one non-PFAS analyte)."""
    header = fh.readline().rstrip("\r\n").split("\t")
    idx = {h: i for i, h in enumerate(header)}
    missing = [c for c in REQUIRED_COLUMNS if c not in idx]
    if missing:
        raise ValueError(f"unexpected header (source shape changed?): missing {missing}")

    systems: dict = {}
    total_rows = 0
    pfas_rows = 0
    detect_rows = 0
    bad_rows = 0

    for line in fh:
        total_rows += 1
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < len(header):
            bad_rows += 1
            continue
        contaminant = parts[idx["Contaminant"]]
        if contaminant in NON_PFAS_CONTAMINANTS:
            continue
        pfas_rows += 1
        if parts[idx["AnalyticalResultsSign"]] != "=":
            continue
        value = _to_float(parts[idx["AnalyticalResultValue"]])
        if value is None:
            bad_rows += 1
            continue
        detect_rows += 1
        pwsid = parts[idx["PWSID"]]
        rec = systems.setdefault(pwsid, {
            "pws_name": parts[idx["PWSName"]],
            "state": parts[idx["State"]] or None,
            "analytes": {},
        })
        date = parts[idx["CollectionDate"]]
        a = rec["analytes"].setdefault(contaminant, {
            "units": parts[idx["Units"]],
            "mrl": _to_float(parts[idx["MRL"]]),
            "max_value": value,
            "first_detected": date,
            "last_detected": date,
            "n_detections": 0,
        })
        a["n_detections"] += 1
        if value > a["max_value"]:
            a["max_value"] = value
        if _date_key(date) < _date_key(a["first_detected"]):
            a["first_detected"] = date
        if _date_key(date) > _date_key(a["last_detected"]):
            a["last_detected"] = date

    return {
        "systems": systems,
        "total_rows": total_rows,
        "pfas_rows": pfas_rows,
        "detect_rows": detect_rows,
        "bad_rows": bad_rows,
    }


def geocode_batches(pwsids):
    """Yield batches of PWSIDs sized to stay well under any sane URL-length
    limit for the ArcGIS query's WHERE ... IN (...) clause."""
    ids = sorted(set(pwsids))
    for i in range(0, len(ids), GEOCODE_BATCH):
        yield ids[i:i + GEOCODE_BATCH]


def _http_post(url, form: dict):
    data = urllib.parse.urlencode(form).encode("ascii")
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "voltradeai-datacore/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def fetch_centroids(pwsids, post=None) -> dict:
    """PWSIDs -> {pwsid: {lat, lon, population_served, boundary_name}} via
    EPA's Community Water System Service Areas FeatureServer centroid query
    (batched, POSTed — a GET with 150 quoted PWSIDs in the WHERE clause can
    exceed the service's URL-length tolerance; queried ONLY for the given
    PWSIDs, never the full ~124k boundary layer). A PWSID with no matching
    boundary feature (unmapped service area) is simply absent from the
    result — callers must drop it, never guess a location for it."""
    post = post or _http_post
    out = {}
    for batch in geocode_batches(pwsids):
        where = "PWSID IN (" + ",".join("'" + p.replace("'", "''") + "'" for p in batch) + ")"
        form = {
            "where": where,
            "outFields": "PWSID,PWS_Name,Population_Served_Count",
            "returnCentroid": "true",
            "returnGeometry": "false",
            "f": "json",
        }
        body = post(GEOCODE_SERVICE, form)
        data = json.loads(body)
        for feat in data.get("features", []):
            attrs = feat.get("attributes", {}) or {}
            centroid = feat.get("centroid")
            pwsid = attrs.get("PWSID")
            if not pwsid or not centroid:
                continue
            lat, lon = centroid.get("y"), centroid.get("x")
            if lat is None or lon is None:
                continue
            out[pwsid] = {
                "lat": lat,
                "lon": lon,
                "population_served": attrs.get("Population_Served_Count"),
                "boundary_name": attrs.get("PWS_Name"),
            }
    return out


def build_artifact(parsed: dict, centroids: dict, source_last_modified, now_iso: str) -> dict:
    systems = parsed["systems"]
    if not systems:
        raise RuntimeError("zero PFAS-detecting systems parsed — refusing to write (all-fail = no write)")
    out_systems = []
    ungeocoded = 0
    for pwsid, rec in systems.items():
        geo = centroids.get(pwsid)
        if not geo:
            ungeocoded += 1
            continue
        detections = [
            {"contaminant": c, **vals}
            for c, vals in sorted(rec["analytes"].items())
        ]
        out_systems.append({
            "pwsid": pwsid,
            "name": rec["pws_name"],
            "state": rec["state"],
            "lat": geo["lat"],
            "lon": geo["lon"],
            "population_served": geo["population_served"],
            "n_analytes_detected": len(detections),
            "detections": detections,
        })
    out_systems.sort(key=lambda s: s["pwsid"])
    return {
        "source": URL,
        "source_last_modified": source_last_modified,
        "geocode_source": (
            GEOCODE_SERVICE + " (centroid of the mapped service-area boundary — "
            "an approximation of where the system serves, NOT the exact intake/"
            "treatment-plant location)"
        ),
        "built_at": now_iso,
        "attribution": (
            "U.S. EPA UCMR 5 (public domain); service-area boundaries: EPA "
            "Community Water System Service Areas"
        ),
        "predictive": False,
        "caveat": (
            "RAW monitoring detections as EPA publishes them under UCMR 5 "
            "(2023-2025 monitoring cycle). NOT an MCL-exceedance or health-risk "
            "claim -- PFAS MCL compliance monitoring under the 2024 NPDWR does "
            "not begin until 2027-2029, and UCMR5 predates and is separate from "
            "that compliance data. A non-detect ('<' rows) is not a zero-"
            "concentration claim, only 'below this sample's method reporting "
            "level' -- non-detects are not represented as detections here."
        ),
        "totals": {
            "rows_scanned": parsed["total_rows"],
            "pfas_rows": parsed["pfas_rows"],
            "detection_rows": parsed["detect_rows"],
            "bad_rows": parsed["bad_rows"],
            "systems_with_detection": len(systems),
            "systems_geocoded": len(out_systems),
            "systems_ungeocoded": ungeocoded,
        },
        "system_count": len(out_systems),
        "systems": out_systems,
    }


def main() -> int:
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
        last_modified = None
        print(f"using local zip {zip_path}")
    else:
        zip_path = "/tmp/ucmr5.zip"
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
    print(f"parsing {MEMBER} (streaming) ...")
    with z.open(MEMBER) as f:
        parsed = parse_ucmr5(io.TextIOWrapper(f, encoding="cp1252"))

    pwsids = list(parsed["systems"].keys())
    print(f"geocoding {len(pwsids)} PWSIDs with >=1 PFAS detection ...")
    centroids = fetch_centroids(pwsids)

    artifact = build_artifact(parsed, centroids, last_modified, datetime.now(timezone.utc).isoformat())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(artifact, f, separators=(",", ":"))
    size_mb = os.path.getsize(OUT) / 1e6
    t = artifact["totals"]
    print(
        f"WROTE {OUT}: {artifact['system_count']} systems "
        f"({t['systems_ungeocoded']} ungeocoded, dropped), "
        f"{t['detection_rows']} detection rows / {t['pfas_rows']} PFAS rows scanned, "
        f"{size_mb:.1f}MB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
