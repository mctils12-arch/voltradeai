#!/usr/bin/env python3
"""gridvision_geo.py — GRID VISION geo-referencing + OSM-recall core (pure).

Turns detector pixel boxes (from a NAIP window whose geographic bbox is known)
into lon/lat points, and scores recall against OSM tower nodes. Used by the NAIP
domain test (gate-1 c) and every state rollout pass. Pure + offline-tested; the
on-pod driver (streams NAIP, runs the .pt, fetches Overpass) feeds these.

HONESTY on the OSM metric (baked into the function names + docstrings): OSM
UNDER-maps towers, so recall-vs-OSM is a LOWER BOUND on true recall, and
"precision-vs-OSM" is NOT true precision (a real tower OSM never mapped reads as
a false positive). True precision needs a human-sampled pass. Never relabel these.
"""
import math


def pixel_to_lonlat(px, py, win_bbox, win_w, win_h):
    """Detector pixel (px,py) inside a NORTH-UP window covering geographic
    win_bbox=[W,S,E,N] at win_w x win_h pixels -> (lon, lat). Row 0 is the NORTH
    edge (standard image + north-up raster convention). Pure."""
    W, S, E, N = win_bbox
    if win_w <= 0 or win_h <= 0:
        raise ValueError("window pixel size must be > 0")
    lon = W + (px / win_w) * (E - W)
    lat = N - (py / win_h) * (N - S)
    return lon, lat


def box_center_lonlat(box_xyxy, win_bbox, win_w, win_h):
    """Center of a [xmin,ymin,xmax,ymax] pixel box -> (lon,lat). Pure."""
    cx = (box_xyxy[0] + box_xyxy[2]) / 2.0
    cy = (box_xyxy[1] + box_xyxy[3]) / 2.0
    return pixel_to_lonlat(cx, cy, win_bbox, win_w, win_h)


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in METERS between two lon/lat points. Pure."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def dedup_points(points, min_sep_m):
    """Collapse detections closer than min_sep_m (seam duplicates from overlapping
    NAIP windows) — greedy, keeps the first of each cluster. points = [(lon,lat,score)].
    Returns the kept list, highest score first. Pure."""
    order = sorted(points, key=lambda p: p[2], reverse=True)
    kept = []
    for lon, lat, sc in order:
        if all(haversine_m(lon, lat, k[0], k[1]) >= min_sep_m for k in kept):
            kept.append((lon, lat, sc))
    return kept


def recall_vs_osm(det_points, osm_points, match_m=30.0):
    """Recall of detections against OSM tower nodes: fraction of OSM points with a
    detection within match_m. Each OSM node matches at most one detection.
    det_points=[(lon,lat,score)] (or [(lon,lat)]), osm_points=[(lon,lat)].
    Returns {n_osm, n_matched, recall_lower_bound, n_det}. LOWER BOUND — OSM
    under-maps towers. Pure."""
    if not osm_points:
        return {"n_osm": 0, "n_matched": 0, "recall_lower_bound": None,
                "n_det": len(det_points), "note": "no OSM towers in bbox — recall undefined"}
    used = set()
    matched = 0
    for o in osm_points:
        for i, d in enumerate(det_points):
            if i in used:
                continue
            if haversine_m(d[0], d[1], o[0], o[1]) <= match_m:
                used.add(i)
                matched += 1
                break
    return {"n_osm": len(osm_points), "n_matched": matched,
            "recall_lower_bound": round(matched / len(osm_points), 4),
            "n_det": len(det_points)}


def osm_corroborated(det_points, osm_points, match_m=30.0, min_conf=0.5):
    """Detections that BOTH clear min_conf AND sit within match_m of an OSM tower —
    the high-confidence, OSM-corroborated set safe to promote to self-bootstrap
    training labels (never the model's own uncorroborated guesses). det_points =
    [(lon,lat,score)]. Returns the corroborated subset. Pure."""
    out = []
    for d in det_points:
        sc = d[2] if len(d) > 2 else 1.0
        if sc < min_conf:
            continue
        if any(haversine_m(d[0], d[1], o[0], o[1]) <= match_m for o in osm_points):
            out.append(d)
    return out
