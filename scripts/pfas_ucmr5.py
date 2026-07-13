#!/usr/bin/env python3
"""
pfas_ucmr5.py — EPA UCMR 5 (Fifth Unregulated Contaminant Monitoring
Rule) PFAS occurrence data: which public water systems detected PFAS
in mandated 2023-2025 monitoring, joined to a facility lat/lon via
EPA ECHO/FRS (Location Context Engine hazard layer — the queued
"PFAS (EPA UCMR5/SDWIS)" item in research/location_context_engine.md).

WHY THIS SHAPE (verified live 2026-07-13, not assumed from memory):
UCMR5_All.txt (public domain, EPA) is a 300MB+ per-sample-per-analyte
file with PWSID/PWSName/State/CollectionDate/Contaminant/Units/
AnalyticalResultsSign/AnalyticalResultValue columns — but NO lat/lon.
A public water system is not a single-point facility like an NPDES
discharger, so geocoding needs a two-hop join through EPA ECHO's REST
family: PWSID -> RegistryID (sdw_rest_services.get_systems, exact
p_pid filter) -> FacLat/FacLong (echo_rest_services.get_facilities,
same p_pid pattern, the FRS Locational Reference Table). Both hops
verified live against real PWSIDs before this script was written
(e.g. WY5600011 CHEYENNE BOARD OF PUBLIC UTILITIES -> RegistryID
110012840104 -> 41.100082,-104.924884).

RAW/FACTUAL, not a SIGNAL: a system's PFAS detection is EPA's own lab
result for that sample, reported as-is with its collection date, MRL,
and result value in micrograms/liter. No health-risk, safety-score, or
"this water is unsafe" claim is made or implied here (that would be a
ladder-gated SIGNAL, and the EPA PFAS NPDWR's own enforceable MCLs
carry a 2029 compliance deadline that has not yet arrived as of this
run — treating a detection as a violation would be dishonest). Only
DETECTIONS (AnalyticalResultsSign == "=", i.e. above the method
reporting limit) are kept; non-detects are not mapped (mirrors "we
don't map the absence of a violation").

DATA QUALITY GATE: a PWSID whose ECHO/FRS geocode hop returns no
RegistryID, or a RegistryID with a null FacLat/FacLong, is quarantined
(counted, never guessed at a state/region centroid).

Run (session-side; UCMR5 monitoring is complete through 2025, EPA
posts periodic updates — re-run occasionally, not on a fixed cadence):
  python3 scripts/pfas_ucmr5.py [path-to-downloaded-zip]
(downloads the zip itself when no path is given; the two ECHO geocode
hops always run live regardless of whether the zip is local)
"""
import csv
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone

URL = "https://www.epa.gov/system/files/other-files/2023-08/ucmr5-occurrence-data.zip"
OUT = os.path.join(os.path.dirname(__file__), "..", "datacore", "pfas", "pfas_ucmr5.json")
SDW_BASE = "https://echodata.epa.gov/echo/sdw_rest_services"
ECHO_BASE = "https://echodata.epa.gov/echo/echo_rest_services"
SDW_BATCH = 300
# echo_rest_services.get_facilities 500s on a GET URL built from ~300
# comma-separated RegistryIDs (verified live 2026-07-13: 200 ids OK,
# 300 ids -> HTTP 500) — a smaller batch for this specific hop only.
ECHO_BATCH = 150
UA = {"User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)"}

# The 28 PFAS analyte names UCMR5 tests for (verified live against the
# real Contaminant column's distinct values 2026-07-13 — "lithium" is
# UCMR5's 29th analyte but is NOT a PFAS and is deliberately excluded).
PFAS_ANALYTES = {
    "11Cl-PF3OUdS", "4:2 FTS", "6:2 FTS", "8:2 FTS", "9Cl-PF3ONS", "ADONA",
    "HFPO-DA", "NEtFOSAA", "NFDHA", "NMeFOSAA", "PFBA", "PFBS", "PFDA",
    "PFDoA", "PFEESA", "PFHpA", "PFHpS", "PFHxA", "PFHxS", "PFMBA",
    "PFMPA", "PFNA", "PFOA", "PFOS", "PFPeA", "PFPeS", "PFTA", "PFTrDA",
    "PFUnA",
}


def parse_detections(fh) -> dict:
    """UCMR5_All.txt file handle -> {pwsid: {name, state, region, dets: [...]}}.

    Keeps only PFAS-analyte rows with AnalyticalResultsSign == "=" (a
    detection above the method reporting limit). Units are normalized
    to the literal string "µg/L" (confirmed the sole unit used for
    every PFAS analyte in this file) rather than round-tripped through
    the source's raw latin-1 byte, to sidestep encoding ambiguity.
    """
    by_pws: dict = {}
    total_pfas_rows = 0
    monitored_pws = set()
    bad_rows = 0
    r = csv.DictReader(fh, delimiter="\t")
    if r.fieldnames is None or "Contaminant" not in r.fieldnames:
        raise ValueError(f"unexpected header (source shape changed?): {r.fieldnames!r}")
    for row in r:
        c = row.get("Contaminant")
        if c not in PFAS_ANALYTES:
            continue
        pwsid = row.get("PWSID")
        if not pwsid:
            bad_rows += 1
            continue
        total_pfas_rows += 1
        monitored_pws.add(pwsid)
        if row.get("AnalyticalResultsSign") != "=":
            continue
        try:
            val = float(row["AnalyticalResultValue"])
            mrl = float(row["MRL"])
        except (KeyError, ValueError, TypeError):
            bad_rows += 1
            continue
        entry = by_pws.setdefault(pwsid, {
            "name": row.get("PWSName", "").strip(),
            "state": row.get("State", "").strip(),
            "region": row.get("Region", "").strip(),
            "dets": [],
        })
        entry["dets"].append({
            "contaminant": c,
            "result": val,
            "mrl": mrl,
            "units": "µg/L",
            "date": row.get("CollectionDate", "").strip(),
        })
    return {
        "by_pws": by_pws,
        "total_pfas_rows": total_pfas_rows,
        "monitored_pws": len(monitored_pws),
        "bad_rows": bad_rows,
    }


def _fetch_json(url: str, timeout=60) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def geocode_pwsids(pwsids: list) -> dict:
    """PWSID -> RegistryID -> FacLat/FacLong, two-hop ECHO join.

    Returns {pwsid: {lat, lon}} for every PWSID that resolved to a
    RegistryID AND that RegistryID carries a non-null FRS geocode.
    Never guesses; a failed hop just means that PWSID is absent from
    the return value (counted by the caller as missing_geocode).
    """
    pws_to_registry: dict = {}
    for batch in _chunks(pwsids, SDW_BATCH):
        ids = ",".join(batch)
        try:
            r1 = _fetch_json(f"{SDW_BASE}.get_systems?output=json&p_pid={urllib.parse.quote(ids)}&responseset=1000")
            qid = r1.get("Results", {}).get("QueryID")
            if not qid:
                continue
            r2 = _fetch_json(f"{SDW_BASE}.get_qid?output=json&qid={qid}&qcolumns=2,8")
            for w in r2.get("Results", {}).get("WaterSystems", []):
                if w.get("PWSId") and w.get("RegistryID"):
                    pws_to_registry[w["PWSId"]] = w["RegistryID"]
        except Exception as e:
            print(f"  [warn] sdw hop failed for a batch of {len(batch)}: {e}", file=sys.stderr)
        time.sleep(0.3)

    registry_ids = sorted(set(pws_to_registry.values()))
    registry_to_geo: dict = {}
    for batch in _chunks(registry_ids, ECHO_BATCH):
        ids = ",".join(batch)
        try:
            r1 = _fetch_json(f"{ECHO_BASE}.get_facilities?output=json&p_pid={urllib.parse.quote(ids)}")
            qid = r1.get("Results", {}).get("QueryID")
            if not qid:
                continue
            r2 = _fetch_json(f"{ECHO_BASE}.get_qid?output=json&qid={qid}&qcolumns=6,17,18")
            for f in r2.get("Results", {}).get("Facilities", []):
                lat, lon = f.get("FacLat"), f.get("FacLong")
                if f.get("RegistryID") and lat is not None and lon is not None:
                    try:
                        registry_to_geo[f["RegistryID"]] = (float(lat), float(lon))
                    except ValueError:
                        pass
        except Exception as e:
            print(f"  [warn] echo hop failed for a batch of {len(batch)}: {e}", file=sys.stderr)
        time.sleep(0.3)

    out = {}
    for pwsid, reg in pws_to_registry.items():
        geo = registry_to_geo.get(reg)
        if geo:
            out[pwsid] = {"lat": geo[0], "lon": geo[1], "registry_id": reg}
    return out


def build_artifact(parsed: dict, geo: dict, now_iso: str) -> dict:
    by_pws = parsed["by_pws"]
    if not by_pws:
        raise RuntimeError("zero PFAS detections parsed — refusing to write (all-fail = no write)")
    systems = []
    missing_geocode = 0
    for pwsid in sorted(by_pws):
        entry = by_pws[pwsid]
        g = geo.get(pwsid)
        if not g:
            missing_geocode += 1
            continue
        dets = sorted(entry["dets"], key=lambda d: (-d["result"], d["date"]))
        systems.append({
            "pws_id": pwsid,
            "name": entry["name"],
            "state": entry["state"],
            "region": entry["region"],
            "lat": g["lat"],
            "lon": g["lon"],
            "n_detections": len(dets),
            "max_result_ug_l": dets[0]["result"],
            "detections": dets,
        })
    return {
        "source": "EPA Fifth Unregulated Contaminant Monitoring Rule (UCMR 5), https://www.epa.gov/dwucmr",
        "geocode_source": "EPA ECHO / FRS Locational Reference Table (echodata.epa.gov)",
        "attribution": "US EPA, public domain",
        "built_at": now_iso,
        "selection": "PFAS analytes only (28 of UCMR5's 29 monitored substances; "
                      "lithium excluded as non-PFAS); AnalyticalResultsSign == '=' "
                      "(detections above the method reporting limit) only — non-detects not mapped.",
        "caveat": "RAW/FACTUAL lab results, not a risk or safety claim. EPA's PFAS "
                  "National Primary Drinking Water Regulation MCLs carry a 2029 "
                  "compliance deadline that has not arrived as of this build — a "
                  "detection here is not a regulatory violation.",
        "counts": {
            "pfas_rows_all_result_signs": parsed["total_pfas_rows"],
            "pws_monitored_for_pfas": parsed["monitored_pws"],
            "pws_with_detection": len(by_pws),
            "pws_geocoded": len(systems),
            "pws_missing_geocode": missing_geocode,
            "bad_rows": parsed["bad_rows"],
        },
        "systems": systems,
    }


def main() -> int:
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
        print(f"using local zip {zip_path}")
    else:
        zip_path = "/tmp/ucmr5.zip"
        print(f"downloading {URL} ...")
        req = urllib.request.Request(URL, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as resp, open(zip_path, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)

    z = zipfile.ZipFile(zip_path)
    member = next(n for n in z.namelist() if n.endswith("UCMR5_All.txt"))
    print(f"parsing {member} (streaming) ...")
    with z.open(member) as f:
        parsed = parse_detections(io.TextIOWrapper(f, encoding="latin-1", newline=""))
    print(
        f"  {parsed['total_pfas_rows']} PFAS rows across {parsed['monitored_pws']} monitored PWS; "
        f"{len(parsed['by_pws'])} PWS with >=1 detection; {parsed['bad_rows']} bad rows"
    )

    pwsids = sorted(parsed["by_pws"])
    print(f"geocoding {len(pwsids)} PWSIDs via ECHO (sdw batches of {SDW_BATCH}, echo batches of {ECHO_BATCH}) ...")
    geo = geocode_pwsids(pwsids)
    print(f"  {len(geo)}/{len(pwsids)} geocoded")

    artifact = build_artifact(parsed, geo, datetime.now(timezone.utc).isoformat())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(artifact, f, separators=(",", ":"), ensure_ascii=False)
    size_kb = os.path.getsize(OUT) / 1e3
    c = artifact["counts"]
    print(
        f"WROTE {OUT}: {c['pws_geocoded']} geocoded systems "
        f"({c['pws_missing_geocode']} missing geocode) out of {c['pws_with_detection']} with a "
        f"detection, {c['pws_monitored_for_pfas']} monitored; {size_kb:.1f}KB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
