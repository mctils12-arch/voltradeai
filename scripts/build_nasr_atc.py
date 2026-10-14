#!/usr/bin/env python3
"""Build datacore/aircraft/nasr_atc.json from the FAA NASR 28-day CSV bundle.

FAA NASR is US-government public domain. Reads, from the same extracted bundle
build_nasr_fixes.py uses:
  FRQ.csv       every published frequency (towers, ground, clearance, ATIS,
                approach/departure sectors, CTAF/UNICOM, center RCAG sites)
  APT_BASE.csv  airport ICAO id, reference point, magnetic variation, ARTCC
  ARB_SEG.csv   ARTCC boundary rings (LOW = below FL180, HIGH = FL180 and up;
                UNLIMITED = oceanic CTA/FIR)
Usage: python3 scripts/build_nasr_atc.py nasr
Output (compact; server/atcFrequencies.ts reads it):
  {"cycle", "source",
   "airports": [{"id","lid","la","lo","mv","twr","art","tc","ac","h",
                 "f": [[mhz, kind, sector, call?], ...]}],
   "rcag": [[lat, lon, artcc, mhz, strata, site]],
   "artcc": {"ZTL": {"n": "ATLANTA", "L": [[lat, lon], ...], "H": [...], "U": [[[lat, lon], ...], ...]}}}
Only the civil VHF airband (118.000-136.975 MHz) is kept: UHF/military rows,
navaid voice, flight-service and weather-broadcast rows are dropped. Kinds:
  T tower (LCL)  G ground  C clearance delivery  A ATIS / D-ATIS
  AD approach+departure  AP approach  DP departure  CT CTAF  UN UNICOM
Departure-procedure / STAR-specific frequencies are not kept (v1).
Deterministic (sorted ids, file order within an airport).
"""
import csv, json, os, re, sys

VHF_LO, VHF_HI = 118.0, 137.0
GUARD_MHZ = 121.5  # the emergency guard frequency is never "the controller"
FREQ_RE = re.compile(r"^(\d{3}\.\d{1,3})$")
STRATA = {"LOW": "L", "HIGH": "H", "LOW/HIGH": "LH", "ULTRA-HIGH": "U"}
# ARTCCs whose LOW/HIGH rings are checked against these reference airports at
# build time: a polygon that does not contain its own airports is a bad parse
ARTCC_CHECK = {"KCLT": "ZTL", "KATL": "ZTL", "KDEN": "ZDV", "KORD": "ZAU", "KJFK": "ZNY",
               "KLAX": "ZLA", "KSEA": "ZSE", "KMIA": "ZMA", "KDFW": "ZFW", "KBOS": "ZBW",
               "KMSP": "ZMP", "KSLC": "ZLC", "KIAD": "ZDC", "KMEM": "ZME", "KIAH": "ZHU"}


def kinds(use):
    """FREQ_USE -> the kinds we keep (one row can be both approach and departure)."""
    toks = [t.strip() for t in use.upper().split(";")]
    out = []
    for t in toks:
        apch, dep = "APCH/" in t, "DEP/" in t
        if apch and dep:
            out.append("AD")
        elif apch:
            out.append("AP")
        elif dep and not t.endswith(" DP"):
            out.append("DP")
        elif t.startswith("LCL/"):
            out.append("T")
        elif t.startswith("GND/"):
            out.append("G")
        elif t.startswith("CD/") or t == "CD PRE TAXI CLNC":
            out.append("C")
        elif t in ("ATIS", "D-ATIS"):
            out.append("A")
        elif t == "CTAF":
            out.append("CT")
        elif t == "UNICOM":
            out.append("UN")
    return out


def vhf(freq):
    m = FREQ_RE.match(freq.strip())
    if not m:
        return None
    f = float(m.group(1))
    return f if VHF_LO <= f < VHF_HI and f != GUARD_MHZ else None


def point_in_ring(lat, lon, ring):
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        yi, xi = ring[i]
        yj, xj = ring[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def main(src, out=None, check=True):
    apt = {}
    with open(os.path.join(src, "APT_BASE.csv"), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                mv = float(r["MAG_VARN"]) * (-1 if r["MAG_HEMIS"].strip() == "W" else 1)
            except ValueError:
                mv = None
            apt[r["ARPT_ID"].strip()] = {
                "icao": r["ICAO_ID"].strip() or None, "la": float(r["LAT_DECIMAL"]), "lo": float(r["LONG_DECIMAL"]),
                "mv": mv, "twr": r.get("TWR_TYPE_CODE", "").strip(), "art": r["RESP_ARTCC_ID"].strip() or None,
            }

    airports, rcag, cycle = {}, {}, None
    with open(os.path.join(src, "FRQ.csv"), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cycle = cycle or r["EFF_DATE"].replace("/", "-")
            mhz = vhf(r["FREQ"])
            if mhz is None:
                continue
            ftype = r["FACILITY_TYPE"].strip()
            site = r["SERVICED_SITE_TYPE"].strip()
            if ftype == "RCAG" and site == "RCAG" and r["ARTCC_OR_FSS_ID"].strip():
                st = STRATA.get(r["SECTORIZATION"].strip().upper())
                if st:
                    key = (r["SERVICED_FACILITY"].strip(), mhz, st)
                    rcag.setdefault(key, [round(float(r["LAT_DECIMAL"]), 4), round(float(r["LONG_DECIMAL"]), 4),
                                          r["ARTCC_OR_FSS_ID"].strip(), mhz, st, r["SERVICED_FACILITY"].strip()])
                continue
            if site not in ("AIRPORT", "HELIPORT", "SEAPLANE BASE") or ftype == "RCAG":
                continue
            ks = kinds(r["FREQ_USE"])
            if not ks:
                continue
            lid = r["SERVICED_FACILITY"].strip()
            a = apt.get(lid)
            if not a:
                continue
            e = airports.get(lid)
            if not e:
                e = airports[lid] = {
                    "id": a["icao"] or lid, "lid": lid, "la": round(a["la"], 5), "lo": round(a["lo"], 5),
                    "mv": a["mv"], "twr": a["twr"], "art": a["art"], "tc": None, "ac": None,
                    "h": r["TOWER_HRS"].strip() or None, "f": [], "_seen": set(),
                }
            e["h"] = e["h"] or r["TOWER_HRS"].strip() or None
            tc = r["TOWER_OR_COMM_CALL"].strip() or None
            ac = r["PRIMARY_APPROACH_RADIO_CALL"].strip() or None
            sector = re.sub(r"\s+", " ", r["SECTORIZATION"].strip())
            for k in ks:
                call = ac if k in ("AD", "AP", "DP") else tc
                if k in ("AD", "AP", "DP"):
                    e["ac"] = e["ac"] or ac
                elif k in ("T", "G", "C", "A"):
                    e["tc"] = e["tc"] or tc
                dedupe = (mhz, k, sector)
                if dedupe in e["_seen"]:
                    continue
                e["_seen"].add(dedupe)
                e["f"].append([mhz, k, sector, call])

    out_airports = []
    for lid in sorted(airports):
        e = airports.pop(lid)
        del e["_seen"]
        for row in e["f"]:  # drop the per-row call when it is the airport default
            default = e["ac"] if row[1] in ("AD", "AP", "DP") else e["tc"]
            if row[3] is None or row[3] == default:
                row.pop()
        out_airports.append({k: v for k, v in e.items() if v is not None})

    rings = {}
    names = {}
    with open(os.path.join(src, "ARB_SEG.csv"), newline="", encoding="utf-8-sig") as f:
        segs = [r for r in csv.DictReader(f)]
    segs.sort(key=lambda r: (r["LOCATION_ID"], r["ALTITUDE"], int(r["POINT_SEQ"])))
    for r in segs:
        loc, alt = r["LOCATION_ID"].strip(), r["ALTITUDE"].strip()
        if not loc.startswith("Z") or alt not in ("LOW", "HIGH", "UNLIMITED"):
            continue
        names.setdefault(loc, r["LOCATION_NAME"].strip())
        key = {"LOW": "L", "HIGH": "H", "UNLIMITED": "U"}[alt]
        lst = rings.setdefault(loc, {}).setdefault(key, [[]])
        lst[-1].append([round(float(r["LAT_DECIMAL"]), 4), round(float(r["LONG_DECIMAL"]), 4)])
        if "POINT OF BEGINNING" in r["BNDRY_PT_DESCRIP"].upper():
            lst.append([])
    artcc = {}
    for loc in sorted(rings):
        d = {"n": names[loc]}
        for key, lst in sorted(rings[loc].items()):
            lst = [ring for ring in lst if len(ring) >= 3]
            if not lst:
                continue
            # LOW/HIGH are single rings for every CONUS/AK/HI center; oceanic
            # UNLIMITED can be several
            d[key] = lst if key == "U" else lst[0]
        artcc[loc] = d

    by_id = {a["id"]: a for a in out_airports}
    for icao, want in ARTCC_CHECK.items() if check else ():
        a = by_id.get(icao)
        assert a, f"{icao} missing from the frequency table"
        for key in ("L", "H"):
            ring = artcc[want].get(key)
            assert ring and point_in_ring(a["la"], a["lo"], ring), f"{icao} not inside {want} {key}"

    out = out or os.path.join(os.path.dirname(__file__), "..", "datacore", "aircraft", "nasr_atc.json")
    with open(out, "w") as f:
        json.dump({"cycle": cycle,
                   "source": "FAA NASR FRQ + APT_BASE + ARB_SEG (public domain)",
                   "airports": out_airports,
                   "rcag": sorted(rcag.values(), key=lambda x: (x[2], x[5], x[3], x[4])),
                   "artcc": artcc}, f, separators=(",", ":"))
    print(f"cycle {cycle}: {len(out_airports)} airports, {len(rcag)} center transmitters, "
          f"{len(artcc)} ARTCC boundaries -> {out} ({os.path.getsize(out) // 1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nasr")
