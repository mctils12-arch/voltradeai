#!/usr/bin/env python3
"""FLIGHT PROGRAM gate-1: filed-route (SFDPS) vs live ADS-B cross-track.

Samples the live site: global aircraft -> /api/data/aircraft/plan/:hex for each
airline-callsign airborne aircraft -> reads the SERVER's deviation.crossTrackNm
for plans whose path is PLACED (pathEstimated=false). Do NOT recompute from
`points`: those are re-seamed to the aircraft's own position (forward-only
seam, v1.0.1013), so a cross-track against them is 0 by construction.
Excludes aircraft <60 nm from the destination (terminal vectoring) and below
12,000 ft. Read-only; raw overlay measurement, no trading.

  python3 scripts/flightplan_gate1.py [base_url] [max_aircraft]
"""
import concurrent.futures as cf, json, math, re, sys, urllib.request

EARTH_NM = 3440.065


def hav_nm(la1, lo1, la2, lo2):
    p = math.pi / 180
    x = math.sin((la2 - la1) * p / 2) ** 2 + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2
    return EARTH_NM * 2 * math.asin(math.sqrt(x))


def summarize(vals):
    v = sorted(vals)
    n = len(v)
    if not n:
        return {"n": 0}
    q = lambda f: v[min(n - 1, int(n * f))]
    return {"n": n, "median": q(0.5), "p75": q(0.75), "p90": q(0.9), "p95": q(0.95),
            "le10": sum(x <= 10 for x in v) / n, "le15": sum(x <= 15 for x in v) / n}


def eligible(row):
    # fields: hex lon lat altFt gsKt trk callsign type seenAt cat gnd reg src
    return bool(row[6]) and (row[3] or 0) > 12000 and not row[10] and re.match(r"^[A-Z]{3}\d", row[6])


def measure(plan, row):
    """crossTrackNm for a placed filed plan mid-route, else None."""
    if not plan or plan.get("source") != "FILED_FAA" or plan.get("pathEstimated") or not plan.get("points"):
        return None
    xt = (plan.get("deviation") or {}).get("crossTrackNm")
    d = plan["points"][-1]
    if xt is None or hav_nm(row[2], row[1], d["lat"], d["lon"]) < 60:
        return None
    return xt


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "https://voltradeai-production.up.railway.app"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    rows = [r for r in _get(base + "/api/data/aircraft/global?lamin=24&lamax=50&lomin=-125&lomax=-66")["rows"] if eligible(r)][:cap]

    def one(r):
        try:
            return measure(_get(f"{base}/api/data/aircraft/plan/{r[0]}?callsign={r[6]}&lat={r[2]}&lon={r[1]}&alt={r[3]}&trk={r[5]}"), r)
        except Exception:
            return None
    with cf.ThreadPoolExecutor(6) as ex:
        vals = [x for x in ex.map(one, rows) if x is not None]
    print(json.dumps(summarize(vals)))


if __name__ == "__main__":
    main()
