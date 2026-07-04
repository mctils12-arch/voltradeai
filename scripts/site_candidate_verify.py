#!/usr/bin/env python3
"""Fetch a mosaic for arbitrary candidate coordinates: id lat lon zoom"""
import math, os, sys, urllib.request
from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site_tiles")
os.makedirs(OUT, exist_ok=True)

def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y

def fetch(z, x, y):
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    req = urllib.request.Request(url, headers={"User-Agent": "voltradeai-site-verify/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return Image.open(__import__("io").BytesIO(r.read())).convert("RGB")

def mosaic_for(sid, lat, lon, z):
    fx, fy = tile_xy(lat, lon, z)
    cx, cy = int(fx), int(fy)
    m = Image.new("RGB", (768, 768))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            try:
                m.paste(fetch(z, cx + dx, cy + dy), ((dx + 1) * 256, (dy + 1) * 256))
            except Exception as e:
                print(f"  tile fail {sid}: {e}")
    px, py = int((fx - cx + 1) * 256), int((fy - cy + 1) * 256)
    d = ImageDraw.Draw(m)
    d.line([(px - 26, py), (px + 26, py)], fill=(255, 40, 40), width=4)
    d.line([(px, py - 26), (px, py + 26)], fill=(255, 40, 40), width=4)
    d.ellipse([px - 34, py - 34, px + 34, py + 34], outline=(255, 240, 0), width=4)
    out = os.path.join(OUT, f"{sid}_z{z}.jpg")
    m.save(out, quality=82)
    print(f"{sid}: {out}")

# args: quads of id,lat,lon,z separated by spaces
args = sys.argv[1:]
for i in range(0, len(args), 4):
    mosaic_for(args[i], float(args[i + 1]), float(args[i + 2]), int(args[i + 3]))
