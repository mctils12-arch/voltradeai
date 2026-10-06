"""Quality gate for a baked chart family: compare our tiles with the FAA's own
tile-service mosaic at sampled tiles before anything is published.

A bake is published only if, across the sample:
  * COVERAGE — where the FAA draws chart, we draw chart (no holes, no
    missing chart): >= MIN_COVERAGE of FAA-opaque pixels are opaque in ours.
  * NO COLLAR LEAK — where the FAA draws nothing, we draw (almost) nothing:
    <= MAX_SPILL of our opaque pixels fall where the FAA is transparent
    (a leaked legend/title block shows up exactly here).
  * SAME CHART — the structural difference (edges.structural_diff) per
    sampled tile has median <= MAX_MEDIAN_DIFF. Our bake is the NEWER
    edition, so small differences are expected (airspace and frequency
    changes); a misregistered or wrong chart is ~100+.

The FAA mosaic can lag by an edition, which is exactly why we bake; that lag
is what the thresholds tolerate, measured on the 2026-09-03 sectional test
bake against the service's 2026-07-09 mosaic.
"""
from __future__ import annotations

import io
import random
from typing import Dict, List

import numpy as np

from common import TILE, fetch_faa_tile
from edges import structural_diff

MIN_COVERAGE = 0.995
MAX_SPILL = 0.005
MAX_MEDIAN_DIFF = 40.0


def verify_pmtiles(fam, pmtiles_path: str, faa_cache_dir: str, zoom: int, samples: int = 300,
                   seed: int = 7) -> Dict:
    from PIL import Image
    from pmtiles.reader import MmapSource, Reader, all_tiles

    with open(pmtiles_path, "rb") as f:
        r = Reader(MmapSource(f))
        keys = [zxy for zxy, _ in all_tiles(r.get_bytes) if zxy[0] == zoom]
        rnd = random.Random(seed)
        pick = rnd.sample(keys, min(samples, len(keys)))
        faa_opaque = ours_on_faa = ours_opaque = spill = 0
        diffs: List[float] = []
        worst: List[Dict] = []
        for (z, x, y) in pick:
            ours = np.asarray(Image.open(io.BytesIO(r.get(z, x, y))).convert("RGBA"))
            faa = fetch_faa_tile(fam.service, z, x, y, faa_cache_dir)
            fo = faa[..., 3] == 255
            oo = ours[..., 3] == 255
            faa_opaque += int(fo.sum())
            ours_on_faa += int((fo & oo).sum())
            ours_opaque += int(oo.sum())
            spill += int((oo & (faa[..., 3] == 0)).sum())
            both = fo & oo
            if both.sum() > TILE * TILE // 8:
                d = structural_diff(ours[..., :3].astype(np.float32), faa[..., :3].astype(np.float32))
                md = float(np.median(d[both]))
                diffs.append(md)
                worst.append({"tile": f"{z}/{x}/{y}", "median_diff": round(md, 1),
                              "coverage": round(float((fo & oo).sum()) / max(1, int(fo.sum())), 4)})
    coverage = ours_on_faa / faa_opaque if faa_opaque else 0.0
    spill_frac = spill / ours_opaque if ours_opaque else 0.0
    med = float(np.median(diffs)) if diffs else float("inf")
    worst.sort(key=lambda w: (w["coverage"], -w["median_diff"]))
    ok = coverage >= MIN_COVERAGE and spill_frac <= MAX_SPILL and med <= MAX_MEDIAN_DIFF and len(pick) > 0
    return {
        "ok": ok, "zoom": zoom, "sampled_tiles": len(pick),
        "coverage": round(coverage, 5), "min_coverage": MIN_COVERAGE,
        "spill": round(spill_frac, 5), "max_spill": MAX_SPILL,
        "median_structural_diff": round(med, 1), "max_median_diff": MAX_MEDIAN_DIFF,
        "worst_tiles": worst[:8],
    }
