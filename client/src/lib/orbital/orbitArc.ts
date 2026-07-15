// Orbit track sampling — EARTH TWIN O6-1 (human directive: click a
// satellite and "the track around the globe pops up for just that one …
// you see their path all across the globe").
//
// Pure math: sample ONE full orbital period with the same SGP4 kernel the
// live layer uses — the arc is the real propagated track from the epoch
// elements, not a drawn ellipse. Points come back as interleaved
// [mercX, mercY, altMeters] triplets ready for arcLayer; samples the kernel
// refuses (it never fakes a position) carry a negative-altitude sentinel so
// the layer breaks the line instead of bridging a gap.

import { propagate } from './propagate.js';
import { lonLatToMercator } from './satBuffer.js';
import type { GpRecord } from './tle.js';

/** Samples per arc — one per ~2° of the orbit reads smooth at globe zooms. */
export const ARC_SAMPLES = 181;

/** Sentinel altitude marking an unsampleable point (line breaks there). */
export const ARC_GAP = -1;

/**
 * Sample one full period starting at nowMs. Returns [x, y, altM] * samples,
 * or null when the object can't honestly be tracked (deep-space, incomplete
 * elements) or too few samples resolve (>25% gaps = no meaningful arc).
 */
export function sampleOrbitArc(
  gp: GpRecord,
  nowMs: number,
  samples: number = ARC_SAMPLES,
): Float32Array | null {
  if (gp.meanMotion == null || !(gp.meanMotion > 0)) return null;
  const periodMs = (1440 / gp.meanMotion) * 60_000;
  const out = new Float32Array(samples * 3);
  let ok = 0;
  for (let k = 0; k < samples; k++) {
    const p = propagate(gp, nowMs + (k / (samples - 1)) * periodMs);
    if (!p || Math.abs(p.latDeg) > 85) {
      // beyond-mercator latitudes can't land on the map either — honest gap
      out[k * 3 + 2] = ARC_GAP;
      continue;
    }
    const m = lonLatToMercator(p.lonDeg, p.latDeg);
    out[k * 3] = m.x;
    out[k * 3 + 1] = m.y;
    out[k * 3 + 2] = p.altKm * 1000;
    ok++;
  }
  return ok >= samples * 0.75 ? out : null;
}

/** RAAN (orbital plane) → a stable color, so a group's orbits read apart
 *  ("make it easy to detect what orbit is for what"). 8 hue buckets. */
export function raanColor(raanDeg: number | null): [number, number, number, number] {
  const PALETTE: [number, number, number][] = [
    [1.0, 0.82, 0.4], [0.4, 0.75, 1.0], [0.55, 1.0, 0.6], [1.0, 0.55, 0.55],
    [0.8, 0.6, 1.0], [0.45, 0.95, 0.95], [1.0, 0.7, 0.9], [0.85, 0.9, 0.55],
  ];
  const idx = raanDeg == null ? 0 : Math.min(7, Math.max(0, Math.floor(((raanDeg % 360 + 360) % 360) / 45)));
  const [r, g, b] = PALETTE[idx];
  return [r, g, b, 0.85];
}
