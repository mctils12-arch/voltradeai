#!/usr/bin/env python3
"""Fetch Esri World Imagery mosaics for each strategic site with a crosshair
at the exact stored coordinate, for visual verification (DESIGN.md
REFERENCE DATA ACCURACY rule)."""
import json, math, os, sys, urllib.request
from PIL import Image, ImageDraw

Z = 15  # ~1.2 km/tile at mid-lat; 3x3 mosaic ~= 3.5 km context
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site_tiles")
os.makedirs(OUT, exist_ok=True)

def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y

def fetch(z, x, y):
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    req = urllib.request.Request(url, headers={"User-Agent": "voltradeai-site-verify/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        fn = os.path.join(OUT, f"t_{z}_{x}_{y}.png")
        open(fn, "wb").write(r.read())
        return fn

sites = json.load(open(sys.argv[1]))["sites"]
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
for s in sites:
    if only and s["id"] not in only:
        continue
    fx, fy = tile_xy(s["lat"], s["lon"], Z)
    cx, cy = int(fx), int(fy)
    mosaic = Image.new("RGB", (256 * 3, 256 * 3))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            try:
                fn = fetch(Z, cx + dx, cy + dy)
                mosaic.paste(Image.open(fn), ((dx + 1) * 256, (dy + 1) * 256))
            except Exception as e:
                print(f"  tile fail {s['id']} {dx},{dy}: {e}")
    px = int((fx - (cx - 1)) * 256)
    py = int((fy - (cy - 1)) * 256)
    d = ImageDraw.Draw(mosaic)
    d.line([(px - 26, py), (px + 26, py)], fill=(255, 40, 40), width=4)
    d.line([(px, py - 26), (px, py + 26)], fill=(255, 40, 40), width=4)
    d.ellipse([px - 34, py - 34, px + 34, py + 34], outline=(255, 240, 0), width=4)
    out = os.path.join(OUT, f"{s['id']}.jpg")
    mosaic.save(out, quality=82)
    print(f"{s['id']}: {out}  ({s['lat']},{s['lon']}) {s['category']}")
