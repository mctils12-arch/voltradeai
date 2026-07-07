#!/usr/bin/env python3
"""gridvision_naip_stac.py — GRID VISION Phase B: a thin, VERIFIED Microsoft
Planetary Computer STAC client for NAIP. Given a bbox + date range it queries
the `naip` collection, picks the best (most-recent) item(s), and signs the COG
asset href so a downstream step can HTTP-range-read pixels — no bulk download.

Charter: research/grid_vision.md; imagery inventory: grid_vision_research.md
GV-A4 (§4.2). NAIP is the license-clean substrate: the Esri wall forbids ML on
Esri tiles, NAIP does not.

VERIFIED LIVE 2026-07-07 (through the agent proxy):
  STAC root:     https://planetarycomputer.microsoft.com/api/stac/v1
  collection id: naip  (temporal 2010-01-01 .. 2023-12-31; gsd 0.3/0.6/1.0 m)
  search:        POST {root}/search  {collections, bbox, datetime, limit}
  item shape:    id e.g. "tx_m_3009751_nw_14_060_20220608";
                 properties.datetime, naip:year, naip:state, gsd;
                 assets.image = COG (image/tiff; application=geotiff;
                 profile=cloud-optimized) at naipeuwest.blob.core.windows.net.
  signing:       GET {root}/../sas/v1/sign?href=<blob-href> -> {href (signed),
                 msft:expiry}  — NO account/token required (verified keyless).

LICENSE HONESTY: the MPC STAC collection's own `license` field literally reads
"proprietary" (an SPDX placeholder MPC did not fill in) — this is NOT the data
license. NAIP is USDA/FSA PUBLIC DOMAIN (attribution requested: "USDA-FSA-APFO
NAIP"); no ML or resale restriction (grid_vision_research.md §4.2, VERIFIED via
the USDA/GEE catalog). Every item this helper returns carries capture_date so a
downstream product can surface imagery freshness honestly.

FRESHNESS CAVEAT (baked into every record): MPC's `naip` collection ENDS at
2023 — 2024/2025 30-cm imagery is not here yet (needs USDA Box/GDG). Records
never claim to be newer than their capture_date.

CLI (session-side; stdlib only; network only when actually querying):
  python3 scripts/gridvision_naip_stac.py --bbox -97.80 30.20 -97.70 30.30 \
      --from 2018-01-01 --to 2023-12-31 --sign
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

STAC_ROOT = "https://planetarycomputer.microsoft.com/api/stac/v1"
SEARCH_URL = f"{STAC_ROOT}/search"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
NAIP_COLLECTION = "naip"
# The collection's own temporal extent (VERIFIED 2026-07-07). Used only to warn
# when a caller asks for imagery newer than the mirror can serve.
NAIP_TEMPORAL_END = "2023-12-31"


# ── request building + parsing (pure, unit-tested, no network) ──────────────

def build_search_body(bbox, date_from, date_to, limit=50, collection=NAIP_COLLECTION):
    """STAC POST /search body for a bbox [W,S,E,N] and an inclusive date range.
    datetime is RFC3339 interval per the STAC API spec."""
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        raise ValueError("bbox must be [west, south, east, north]")
    return {
        "collections": [collection],
        "bbox": [float(x) for x in bbox],
        "datetime": f"{date_from}T00:00:00Z/{date_to}T23:59:59Z",
        "limit": int(limit),
    }


def parse_items(payload):
    """STAC FeatureCollection -> normalized item records (freshness-honest).
    capture_date is on EVERY record; the raw signed href is filled later."""
    out = []
    for f in payload.get("features", []):
        p = f.get("properties", {}) or {}
        dt = p.get("datetime") or ""
        img = (f.get("assets", {}) or {}).get("image", {}) or {}
        out.append({
            "id": f.get("id"),
            "datetime": dt,
            "capture_date": dt[:10] if dt else None,
            "naip_year": p.get("naip:year"),
            "state": p.get("naip:state"),
            "gsd": p.get("gsd"),
            "bbox": f.get("bbox"),
            "image_href": img.get("href"),       # UNSIGNED blob href
            "image_type": img.get("type"),
            "signed_href": None,
        })
    return out


def best_item(items):
    """Pick the most recent usable NAIP item. NAIP has no cloud cover, so
    'best' = latest capture_date (freshest imagery), with a stable id tie-break
    for determinism. Items without a capture_date are ignored (never guessed).
    Returns None if none qualify."""
    usable = [it for it in items if it.get("capture_date") and it.get("image_href")]
    if not usable:
        return None
    return sorted(usable, key=lambda it: (it["capture_date"], it["id"] or ""))[-1]


def warn_if_stale_request(date_to, temporal_end=NAIP_TEMPORAL_END):
    """Return a warning string if the caller asked for imagery past the mirror's
    coverage end, else None. Keeps the freshness promise explicit."""
    if str(date_to) > str(temporal_end):
        return (f"requested end {date_to} is past the MPC naip mirror end "
                f"{temporal_end}; 2024/2025 imagery needs USDA Box/GDG")
    return None


# ── network (thin; injectable fetchers so tests never hit the wire) ─────────

def _urllib_post(url, body, timeout=60):
    import urllib.request
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"content-type": "application/json",
                 "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _urllib_get(url, timeout=60):
    import urllib.request
    req = urllib.request.Request(
        url, headers={"User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def search(bbox, date_from, date_to, limit=50, post=_urllib_post):
    """Query MPC STAC for NAIP over bbox/date range -> (items, warning).
    `post` is injectable: tests pass a fixture-returning callable."""
    body = build_search_body(bbox, date_from, date_to, limit)
    payload = post(SEARCH_URL, body)
    return parse_items(payload), warn_if_stale_request(date_to)


def sign_href(href, get=_urllib_get):
    """Sign a NAIP blob href via the keyless MPC SAS endpoint ->
    (signed_href, expiry). `get` is injectable for tests."""
    if not href:
        return None, None
    import urllib.parse
    resp = get(f"{SIGN_URL}?href={urllib.parse.quote(href, safe='')}")
    return resp.get("href"), resp.get("msft:expiry")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                    required=True)
    ap.add_argument("--from", dest="date_from", default="2018-01-01")
    ap.add_argument("--to", dest="date_to", default=NAIP_TEMPORAL_END)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--sign", action="store_true",
                    help="also sign the best item's COG href")
    args = ap.parse_args(argv)

    items, warning = search(args.bbox, args.date_from, args.date_to, args.limit)
    if warning:
        print(f"# WARN: {warning}", file=sys.stderr)
    best = best_item(items)
    if best is None:
        print(json.dumps({"n_items": len(items), "best": None}, indent=1))
        return
    if args.sign:
        best["signed_href"], best["signed_expiry"] = sign_href(best["image_href"])
    print(json.dumps({"n_items": len(items), "best": best,
                      "queried_utc": datetime.now(timezone.utc).isoformat()}, indent=1))


if __name__ == "__main__":
    main()
