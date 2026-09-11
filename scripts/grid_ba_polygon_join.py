#!/usr/bin/env python3
"""grid_ba_polygon_join.py — the plant -> balancing-authority polygon
join that scripts/grid_generation_gate1.py's own header names as the
missing ingredient for FUSION HYPOTHESIS (b)'s literal per-region gate-1
ground truth ("EIA-930 totals reconciling to registry capacity within
~5% per region", CLAUDE.md/research/open_questions.md). Two prior
sessions (2026-09-11) searched for a working national HIFLD "Control
Areas" mirror and abandoned: the documented `services1.arcgis.com/
Hp6G80Pky0om7QvQ/.../Control_Areas_gdb` URL 400s "Invalid URL" (a
retired/dead item), and a "Balancing Authority Areas" alternative that
DID respond turned out to be a small-extent non-national derivative
(wrong dataset). This session found a THIRD, live, national mirror of
the same HIFLD dataset via the ArcGIS `sharing/rest/search` API rather
than guessing at hub.arcgis.com item URLs (those are JS-rendered SPAs
with no static content to scrape):

    https://services5.arcgis.com/HDRa0B57OVrv2E1q/arcgis/rest/services/Control_Areas/FeatureServer/0

Verified live before trusting it: 71 features, geometryType polygon,
extent -158.28..-66.95 lon / 21.25..61.48 lat (CONUS+AK+HI — genuinely
national, not the small-extent trap the prior session hit), field set
matches HIFLD's documented schema (ID/NAME/STATE/COUNTRY/SOURCE/
SOURCEDATE/...). Fetched once this session (`maxAllowableOffset=0.005`
degrees, ~500m at mid-latitudes, server-side generalization — the
unsimplified fetch was 43MB, unreasonable to commit; the simplified
fetch is ~1.5MB, checked: no feature lost a real ring, smallest polygons
are genuine compact single-plant "embedded" BAs like Griffith Energy
and Arlington Valley, not simplification artifacts) and committed as
datacore/boundaries/hifld_control_areas.json, matching the existing
datacore/boundaries/ne_110m_admin0.json convention (FeatureCollection +
_doc, properties stripped to what's actually used).

CROSSWALK, stated honestly (no name match guessed): HIFLD's 71 entity
NAMEs don't carry EIA-930 respondent codes directly, so this module
matches them against EIA's own live respondent facet list
(`electricity/rto/region-data/facet/respondent`, 83 entries, fetched
live this session with the sandbox's existing EIA_API_KEY) by normalized
exact/substring match. 58 of 71 matched automatically this way; 13 did
not, of which 6 are real matches an automated string match cannot find
(entity renames, e.g. "PACIFICORP - EAST" -> EIA code PACE
"PacifiCorp East"; "MIDCONTINENT INDEPENDENT TRANSMISSION SYSTEM
OPERATOR" -> MISO "Midcontinent Independent System Operator"; "SOUTH
CAROLINA ELECTRIC & GAS COMPANY" -> SCEG "Dominion Energy South
Carolina, Inc.", the utility's 2021 rename) — these are hardcoded in
BA_CROSSWALK_OVERRIDES below, each with the real-world reason. The
remaining 7 are LEGITIMATELY unmatched, not a search failure: Chugach
Electric (AK) and Anchorage Municipal Light & Power (AK) and Hawaiian
Electric (HI) are outside EIA-930's Lower-48 reporting scope (confirmed
against EIA's own "US48" respondent id and the absence of any AK/HI
code in the live 83-entry list); New Brunswick System Operator is a
Canadian interconnection, not a US respondent; Gila River Power LLC,
Gridforce South, and Ohio Valley Electric Corporation are HIFLD-mapped
historical control areas with NO corresponding code in the CURRENT
EIA-930 respondent list (OVEC's own generation has been folded into
PJM's EIA-930 reporting for years; the two small Arizona merchant-plant
BAs likewise have no current standalone EIA-930 respondent) — recorded
in BA_EXCLUDED with the reason, never silently dropped or forced onto a
similar-sounding code.

SCOPE OF THIS MODULE: builds the crosswalk and the plant -> BA
assignment (point-in-polygon of each registry plant's lat/lon against
the 71 committed polygons, reusing grid_ba_capacity.py's own
point_in_rings ray-casting rather than re-deriving it, EDGE DOCTRINE
#3). It does NOT re-run grid_generation_gate1.py per-BA — that stays
its own follow-up PR (one logical change; this join is the prerequisite
that PR needs, not a bundled re-verdict).

Usage:
    python3 scripts/grid_ba_polygon_join.py \
        [--boundaries datacore/boundaries/hifld_control_areas.json] \
        [--registry datacore/powerplants/us_power_plants.json] \
        [--out datacore/powerplants/plant_balancing_authority.json]
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "grid_ba_capacity", os.path.join(_HERE, "grid_ba_capacity.py"))
_gbc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gbc)
point_in_rings = _gbc.point_in_rings


# Six real matches a normalized-name string match cannot find on its own
# (entity renames / informal-vs-official naming) — verified individually
# against EIA-930's live respondent list before being hardcoded, per this
# module's own docstring.
BA_CROSSWALK_OVERRIDES = {
    "56669": ("MISO", "Midcontinent Independent System Operator, Inc.",
              "MIDCONTINENT INDEPENDENT TRANSMISSION SYSTEM OPERATOR is MISO's HIFLD name variant"),
    "14379": ("PACE", "PacifiCorp East",
              "PACIFICORP - EAST is EIA-930 respondent PACE"),
    "14378": ("PACW", "PacifiCorp West",
              "PACIFICORP - WEST is EIA-930 respondent PACW"),
    "19610": ("WAUW", "Western Area Power Administration - Upper Great Plains West",
              "'UGP WEST' in the HIFLD name is 'Upper Great Plains West'"),
    "17539": ("SCEG", "Dominion Energy South Carolina, Inc.",
              "South Carolina Electric & Gas Company renamed to Dominion Energy South Carolina in 2021; same entity, same EIA-930 code SCEG"),
    "13485": ("NSB", "Utilities Commission of New Smyrna Beach",
              "word-order variant of the same entity name"),
}

# Seven real HIFLD control areas with NO current EIA-930 respondent code,
# each independently reasoned (not a batch guess) — see module docstring.
BA_EXCLUDED = {
    "3522": "Chugach Electric Assn (AK) — outside EIA-930 Lower-48 scope",
    "1": "New Brunswick System Operator — Canadian interconnection, not a US EIA-930 respondent",
    "599": "Anchorage Municipal Light & Power (AK) — outside EIA-930 Lower-48 scope",
    "14412": "Gila River Power, LLC — no current EIA-930 respondent code (small AZ merchant-plant BA)",
    "56545": "Gridforce South — no current EIA-930 respondent code (small AZ merchant-plant BA)",
    "19547": "Hawaiian Electric Co (HI) — outside EIA-930 Lower-48 scope",
    "14015": "Ohio Valley Electric Corporation — generation folded into PJM's EIA-930 reporting; no standalone OVEC respondent code remains",
}


def normalize_entity_name(name):
    """Strip punctuation and generic corporate suffixes so e.g.
    'PJM Interconnection, LLC' and 'PJM INTERCONNECTION, LLC' compare
    equal, without being so aggressive that distinct entities collide
    (kept intentionally narrow — verified against the real 71/83 lists,
    not tuned against a synthetic case)."""
    s = name.upper()
    s = re.sub(r"[.,]", "", s)
    s = re.sub(r"\bLLC\b|\bINC\b|\bCOMPANY\b|\bCO\b|\bCORP\b|\bCORPORATION\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_ba_crosswalk(hifld_features, eia_respondents,
                        overrides=BA_CROSSWALK_OVERRIDES, excluded=BA_EXCLUDED):
    """hifld_features: list of {id, name} (HIFLD ID + NAME).
    eia_respondents: list of {id, name} (EIA-930 respondent code + name).
    Returns (matched, unmatched):
      matched: list of {hifld_id, hifld_name, ba_code, ba_name, method}
      unmatched: list of {hifld_id, hifld_name, reason} — reason is
        either the excluded-map's stated cause or "no match found" for
        anything genuinely new (never silently dropped)."""
    eia_norm = [(e["id"], e["name"], normalize_entity_name(e["name"])) for e in eia_respondents]
    matched, unmatched = [], []
    for f in hifld_features:
        hid, hname = f["id"], f["name"]
        if hid in overrides:
            code, name, reason = overrides[hid]
            matched.append({"hifld_id": hid, "hifld_name": hname, "ba_code": code,
                             "ba_name": name, "method": "override:" + reason})
            continue
        hnorm = normalize_entity_name(hname)
        best = None
        for eid, ename, enorm in eia_norm:
            if hnorm == enorm:
                best = (eid, ename, "exact")
                break
            if enorm and (enorm in hnorm or hnorm in enorm):
                if best is None or best[2] != "exact":
                    best = (eid, ename, "substr")
        if best:
            matched.append({"hifld_id": hid, "hifld_name": hname, "ba_code": best[0],
                             "ba_name": best[1], "method": best[2]})
        else:
            reason = excluded.get(hid, "no match found")
            unmatched.append({"hifld_id": hid, "hifld_name": hname, "reason": reason})
    return matched, unmatched


def geometry_rings(geometry):
    """Flatten a GeoJSON Polygon/MultiPolygon into the list-of-rings
    shape point_in_rings expects (each ring itself a list of [lon,lat],
    outer + holes all included — point_in_rings's even-odd rule handles
    holes without needing orientation analysis)."""
    if geometry is None:
        return []
    if geometry["type"] == "Polygon":
        return list(geometry["coordinates"])
    if geometry["type"] == "MultiPolygon":
        rings = []
        for poly in geometry["coordinates"]:
            rings.extend(poly)
        return rings
    return []


def bbox_of_rings(rings):
    lons = [pt[0] for ring in rings for pt in ring]
    lats = [pt[1] for ring in rings for pt in ring]
    if not lons:
        return None
    return (min(lons), min(lats), max(lons), max(lats))


def build_ba_index(boundaries_features, ba_code_by_hifld_id):
    """Returns list of (ba_code, bbox, rings) for every HIFLD feature
    that has a resolved EIA-930 code — unmatched/excluded features are
    skipped here (they contribute no assignment, honestly, rather than
    matching under a null code that could look like a real bucket)."""
    index = []
    for feat in boundaries_features:
        hid = feat["properties"]["ba_id"]
        code = ba_code_by_hifld_id.get(hid)
        if not code:
            continue
        rings = geometry_rings(feat["geometry"])
        bbox = bbox_of_rings(rings)
        if bbox is None:
            continue
        index.append((code, bbox, rings))
    return index


def assign_plant_ba(plants, ba_index):
    """plants: list of [name, capacity_mw, fuel, owner, lat, lon, verified]
    (build_powerplants.py's row format). ba_index: build_ba_index() output.
    Returns list of {name, fuel, capacity_mw, ba_codes} — ba_codes is a
    list (usually length 0 or 1; length >1 flags a genuine polygon
    overlap or boundary-adjacency case, reported rather than arbitrarily
    resolved) so a caller can see ambiguity instead of a silently wrong
    single answer."""
    out = []
    for p in plants:
        name, capacity_mw, fuel, _owner, lat, lon, _verified = p
        codes = []
        if lat is not None and lon is not None:
            for code, bbox, rings in ba_index:
                if bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3] \
                        and point_in_rings(lon, lat, rings):
                    codes.append(code)
        out.append({"name": name, "fuel": fuel, "capacity_mw": capacity_mw, "ba_codes": codes})
    return out


def summarize_assignment(assignments):
    matched = sum(1 for a in assignments if len(a["ba_codes"]) == 1)
    unmatched = sum(1 for a in assignments if len(a["ba_codes"]) == 0)
    ambiguous = sum(1 for a in assignments if len(a["ba_codes"]) > 1)
    by_ba = defaultdict(lambda: {"count": 0, "capacity_mw": 0.0})
    for a in assignments:
        if len(a["ba_codes"]) == 1:
            b = by_ba[a["ba_codes"][0]]
            b["count"] += 1
            b["capacity_mw"] += a["capacity_mw"] or 0.0
    return {
        "total": len(assignments), "matched": matched, "unmatched": unmatched,
        "ambiguous": ambiguous, "by_ba": dict(by_ba),
    }


def run(boundaries_path, registry_path, out_path, eia_respondents_path=None):
    boundaries = json.load(open(boundaries_path))
    registry = json.load(open(registry_path))

    if eia_respondents_path:
        eia_respondents = json.load(open(eia_respondents_path))["response"]["facets"]
    else:
        import urllib.request
        api_key = os.environ.get("EIA_API_KEY", "")
        url = ("https://api.eia.gov/v2/electricity/rto/region-data/facet/respondent/"
               f"?api_key={api_key}")
        with urllib.request.urlopen(url, timeout=30) as resp:
            eia_respondents = json.load(resp)["response"]["facets"]

    hifld_features = [{"id": f["properties"]["ba_id"], "name": f["properties"]["name"]}
                       for f in boundaries["features"]]
    matched, unmatched = match_ba_crosswalk(hifld_features, eia_respondents)
    ba_code_by_hifld_id = {m["hifld_id"]: m["ba_code"] for m in matched}

    ba_index = build_ba_index(boundaries["features"], ba_code_by_hifld_id)
    assignments = assign_plant_ba(registry["plants"], ba_index)
    summary = summarize_assignment(assignments)

    out = {
        "_doc": ("Plant -> EIA-930 balancing-authority assignment via point-in-polygon "
                 "against datacore/boundaries/hifld_control_areas.json. Recomputed fresh "
                 "from the registry + boundaries each run (like entityGraph.ts's own "
                 "plantFacilityId) — not a stable cross-rebuild join key. Built by "
                 "scripts/grid_ba_polygon_join.py."),
        "source": "scripts/grid_ba_polygon_join.py",
        "boundaries_source": boundaries_path,
        "registry_source": registry_path,
        "crosswalk_matched": len(matched),
        "crosswalk_unmatched": unmatched,
        "summary": summary,
        "assignments": assignments,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"crosswalk: {len(matched)} matched / {len(unmatched)} unmatched "
          f"(of {len(hifld_features)} HIFLD control areas)")
    print(f"plants: {summary['matched']} matched / {summary['unmatched']} unmatched / "
          f"{summary['ambiguous']} ambiguous (of {summary['total']})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundaries", default="datacore/boundaries/hifld_control_areas.json")
    ap.add_argument("--registry", default="datacore/powerplants/us_power_plants.json")
    ap.add_argument("--out", default="datacore/powerplants/plant_balancing_authority.json")
    ap.add_argument("--eia-respondents", default=None,
                     help="local JSON of the EIA respondent facet response (session-fetched); "
                          "defaults to a live api.eia.gov call")
    args = ap.parse_args()
    run(args.boundaries, args.registry, args.out, args.eia_respondents)
