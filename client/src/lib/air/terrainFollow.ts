// TERRAIN-FOLLOWING + DEVICE-AWARE BUDGETS for the SELECTED aircraft's
// flown track and gray plan (human 2026-09-30: "with 3D terrain on, the line
// needs to follow the structure of the terrain — right now it just follows
// the flat curve of the earth"; "analyze the user's device to see what it
// can handle").
//
// Why the lines looked flat: the geometry is drawn as straight quads between
// vertices, and a quad's bottom edge interpolates LINEARLY between its two
// ground heights. The flown track was decimated uniformly to 512 GPU points
// (≈3 km apart on a 1,700 km flight — every ridge between two kept points
// vanished), the live tail was ONE quad from the last archived fix to the
// plane, and the plan's seam connector (incl. ATC vectoring) was ONE quad.
// This module holds the pure pieces that fix it:
//   - trackBudget(tier): vertex caps, ground-sampling step, subdivision
//     bounds and the fast-lane poll interval per device tier;
//   - refineForGround: after a uniform decimation, spend the remaining
//     vertex budget where the terrain deviates most from the straight bottom
//     edge (Douglas–Peucker on the ground profile), so ridges/valleys between
//     fixes survive the cap;
//   - subdivisionCount / lerpPoints: split a long straight segment into
//     pieces short enough for the caller to sample the rendered terrain.
//
// Law IV: the tier comes from lib/deviceTier.classifyDevice — the GPU
// renderer string (renderer capabilities), memory and cores the map already
// published on __vtDeviceTier — never user-agent sniffing.

import type { DeviceTier, TierReading } from '../deviceTier.js';
import { distMeters } from './trackModel.js';

export interface TrackBudget {
  /** GPU vertex cap for the flown track geometry (≤ FT_MAX_FEATURES). */
  trackMaxPoints: number;
  /** target ground-sampling step along long segments, meters. */
  groundStepM: number;
  /** max sub-segments for the live tail (last fix → plane). */
  tailMaxSubdiv: number;
  /** max sub-segments per seam/vectoring connector piece of the plan. */
  seamMaxSubdiv: number;
  /** densified plan vertex cap (≤ PLAN_MAX_POINTS) and finest spacing. */
  planMaxPoints: number;
  planMinSpacingM: number;
  /** while terrain tiles are still loading, re-read the rendered mesh for
   *  the plan this often (frame loop), and at most this many times. */
  meshRefreshMs: number;
  meshRefreshMax: number;
  /** selected-aircraft fast-lane poll interval. */
  fastPollMs: number;
}

const BUDGETS: Record<DeviceTier, TrackBudget> = {
  full: {
    trackMaxPoints: 2048, groundStepM: 300, tailMaxSubdiv: 24, seamMaxSubdiv: 16,
    planMaxPoints: 1536, planMinSpacingM: 1500, meshRefreshMs: 2000, meshRefreshMax: 30, fastPollMs: 2500,
  },
  reduced: {
    trackMaxPoints: 1024, groundStepM: 600, tailMaxSubdiv: 12, seamMaxSubdiv: 8,
    planMaxPoints: 1024, planMinSpacingM: 2500, meshRefreshMs: 3000, meshRefreshMax: 20, fastPollMs: 3000,
  },
  minimal: {
    trackMaxPoints: 512, groundStepM: 1500, tailMaxSubdiv: 4, seamMaxSubdiv: 4,
    planMaxPoints: 512, planMinSpacingM: 5000, meshRefreshMs: 5000, meshRefreshMax: 10, fastPollMs: 4000,
  },
};

/** Pure: the budget for a device tier (unknown → full: the frame governor
 *  still steps a mis-classified machine down). */
export function trackBudget(tier: DeviceTier | null | undefined): TrackBudget {
  return BUDGETS[tier ?? 'full'] ?? BUDGETS.full;
}

/** The tier datamap classified at startup (published on __vtDeviceTier). */
export function currentDeviceTier(): DeviceTier {
  const t = (globalThis as { __vtDeviceTier?: TierReading }).__vtDeviceTier?.tier;
  return t === 'reduced' || t === 'minimal' || t === 'full' ? t : 'full';
}

/** Ground deviation (display meters, before exaggeration) that earns an
 *  extra vertex in refineForGround. */
export const GROUND_TOL_M = 25;

/** Pure: how many pieces to split a segment of `distM` into so each is
 *  ≤ stepM, bounded to [1, maxSubdiv]. */
export function subdivisionCount(distM: number, stepM: number, maxSubdiv: number): number {
  if (!(distM > 0) || !(stepM > 0)) return 1;
  const n = Math.ceil(distM / stepM);
  return Math.max(1, Math.min(Math.max(1, Math.floor(maxSubdiv)), n));
}

/** Pure: the n−1 INTERIOR points of a segment split into n equal pieces
 *  (linear in whatever 2D space the caller passes — mercator here). */
export function lerpPoints(ax: number, ay: number, bx: number, by: number, n: number): Array<[number, number, number]> {
  const out: Array<[number, number, number]> = [];
  for (let k = 1; k < n; k++) {
    const u = k / n;
    out.push([ax + (bx - ax) * u, ay + (by - ay) * u, u]);
  }
  return out;
}

/** Pure: max |ground − straight bottom edge| over (a, b), and where. */
function worstDeviation(alongM: ArrayLike<number>, ground: ArrayLike<number>, a: number, b: number): { i: number; dev: number } {
  const span = alongM[b] - alongM[a];
  let best = -1, dev = 0;
  for (let i = a + 1; i < b; i++) {
    const g = ground[i];
    if (!Number.isFinite(g)) continue;
    const u = span > 0 ? (alongM[i] - alongM[a]) / span : 0;
    const d = Math.abs(g - (ground[a] + (ground[b] - ground[a]) * u));
    if (d > dev) { dev = d; best = i; }
  }
  return { i: best, dev };
}

/**
 * Pure: add vertices to an existing kept-index list (sorted, first and last
 * sample included) where the terrain deviates most from the straight bottom
 * edge, until `maxPoints` or until every remaining deviation ≤ tolM. The
 * returned list is sorted and a superset of `kept` — gap boundaries the
 * uniform pass kept are never dropped.
 */
export function refineForGround(
  kept: readonly number[],
  alongM: ArrayLike<number>,
  ground: ArrayLike<number>,
  maxPoints: number,
  tolM: number,
): number[] {
  const out = new Set<number>(kept);
  if (kept.length < 2 || out.size >= maxPoints) return Array.from(out).sort((x, y) => x - y);
  // max-heap of segments by worst deviation
  type Seg = { a: number; b: number; i: number; dev: number };
  const heap: Seg[] = [];
  const push = (s: Seg) => {
    heap.push(s);
    let c = heap.length - 1;
    while (c > 0) {
      const p = (c - 1) >> 1;
      if (heap[p].dev >= heap[c].dev) break;
      [heap[p], heap[c]] = [heap[c], heap[p]];
      c = p;
    }
  };
  const pop = (): Seg | undefined => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length && last) {
      heap[0] = last;
      let c = 0;
      for (;;) {
        const l = 2 * c + 1, r = l + 1;
        let m = c;
        if (l < heap.length && heap[l].dev > heap[m].dev) m = l;
        if (r < heap.length && heap[r].dev > heap[m].dev) m = r;
        if (m === c) break;
        [heap[m], heap[c]] = [heap[c], heap[m]];
        c = m;
      }
    }
    return top;
  };
  const consider = (a: number, b: number) => {
    if (b - a < 2) return;
    const w = worstDeviation(alongM, ground, a, b);
    if (w.i > a && w.dev > tolM) push({ a, b, i: w.i, dev: w.dev });
  };
  const sorted = Array.from(out).sort((x, y) => x - y);
  for (let k = 0; k + 1 < sorted.length; k++) consider(sorted[k], sorted[k + 1]);
  while (out.size < maxPoints && heap.length) {
    const s = pop();
    if (!s) break;
    out.add(s.i);
    consider(s.a, s.i);
    consider(s.i, s.b);
  }
  return Array.from(out).sort((x, y) => x - y);
}

/** Pure: cumulative along-track meters for lon/lat samples (trackModel's
 *  own great-circle distance — one earth model for the track). */
export function cumulativeAlongM(samples: ReadonlyArray<{ lat: number; lon: number }>): Float64Array {
  const out = new Float64Array(samples.length);
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1], b = samples[i];
    out[i] = out[i - 1] + distMeters(a.lat, a.lon, b.lat, b.lon);
  }
  return out;
}
