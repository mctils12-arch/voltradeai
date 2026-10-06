#!/usr/bin/env python3
"""Regenerate server/__fixtures__/faa_bake_fixture.pmtiles — a tiny archive
written by the SAME writer and encoder the bake uses (bake.write_pmtiles,
bake.encode), read back by server/aeroBake.test.ts with the npm `pmtiles`
reader the server uses. A contract test across the Python/Node boundary.

Tiles: 2/0/1 opaque red, 5/8/12 half-transparent green, 11/350/740 opaque blue.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pmtiles.tile as pt  # noqa: E402

import bake  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "server", "__fixtures__", "faa_bake_fixture.pmtiles")


def tile(rgb, left_only=False):
    a = np.zeros((256, 256, 4), np.float32)
    a[..., :3] = rgb
    a[..., 3] = 1.0
    if left_only:  # right half fully transparent
        a[:, 128:, 3] = 0.0
    return bake.encode(a)


def main():
    tiles = [((2, 0, 1), tile((255, 0, 0))), ((5, 8, 12), tile((0, 200, 0), left_only=True)),
             ((11, 350, 740), tile((0, 0, 255)))]
    entries = sorted((pt.zxy_to_tileid(*zxy), data) for zxy, data in tiles)
    bake.write_pmtiles(os.path.abspath(OUT), entries, {
        "min_zoom": 2, "max_zoom": 11, "min_lon_e7": -1250000000, "min_lat_e7": 400000000,
        "max_lon_e7": -1160000000, "max_lat_e7": 490000000, "center_zoom": 8,
        "center_lon_e7": -1210000000, "center_lat_e7": 445000000}, {"family": "sectional", "fixture": True})
    print(os.path.abspath(OUT), os.path.getsize(OUT))


if __name__ == "__main__":
    main()
