// Shared buffer-layout contract for the satellite render pipeline.
//
// This is the SINGLE SOURCE OF TRUTH for how the propagation worker
// (./satWorker) packs positions and how the GPU layer (./satLayer) reads them.
// It is deliberately PURE — zero orbital-math / propagate imports, zero DOM,
// zero side effects — so the main-thread layer can import the layout constants
// WITHOUT dragging the SGP4 kernel or the worker install-block into the main
// bundle (the worker file imports ./propagate; type-only imports of its
// message shapes are erased, so satLayer stays kernel-free). It is also fully
// hermetic to unit-test.

/**
 * Floats per satellite in the packed position buffer:
 *   [0] mercX      web-mercator X, normalized 0..1  (projectTile input)
 *   [1] mercY      web-mercator Y, normalized 0..1
 *   [2] altMeters  altitude above sea level in METERS (projectTileFor3D elev)
 *   [3] classCode  orbit-class code, ALSO the validity sentinel:
 *                    0 = LEO, 1 = MEO, 2 = GEO  (rendered)
 *                    SENTINEL_SKIP (-1) = not rendered (deep-space / invalid)
 *   [4] velX       d(mercX)/dt in normalized-mercator UNITS PER SECOND
 *                  (antimeridian-wrapped finite difference of two REAL
 *                  propagations — see satWorker.packPositions)
 *   [5] velY       d(mercY)/dt in normalized-mercator units per second
 *   [6] velAlt     d(altMeters)/dt in METERS PER SECOND (radial glide —
 *                  matters for eccentric orbits, e.g. Molniya)
 * The layer's vertex shader treats classCode < 0 as "skip" (gl_PointSize = 0).
 * Velocity fields are DISPLAY-ONLY glide inputs for the shader's between-tick
 * extrapolation (satLayer u_dtSec); every CPU consumer (pick / follow /
 * miniSelect / siteQuery) keeps reading the exact-tick position at [0..2] —
 * the physics truth stays the worker's tick, the glide never enters analysis.
 * Fields [0..3] keep their pre-glide offsets on purpose: stride-agnostic
 * readers (base + 0..3) survive the extension unchanged.
 */
export const SAT_STRIDE = 7;

/** classCode value marking a slot that must NOT be rendered (no real position). */
export const SENTINEL_SKIP = -1;

/** Orbit-class codes packed into the classCode field. */
export const CLASS_CODE = { LEO: 0, MEO: 1, GEO: 2 } as const;
export type OrbitClassCode = 0 | 1 | 2;

/**
 * Geodetic lon/lat (degrees) -> web-mercator normalized [0..1] x/y.
 * Matches MapLibre's projectTile input space when the defaultProjectionData
 * uniforms are bound. Y is clamped to [0,1] at the poles (mercator singularity).
 */
export function lonLatToMercator(lonDeg: number, latDeg: number): { x: number; y: number } {
  const x = (lonDeg + 180) / 360;
  const s = Math.sin((latDeg * Math.PI) / 180);
  // 0.5 - ln((1+sin)/(1-sin)) / (4π) === standard slippy-map Y.
  let y = 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
  if (y < 0) y = 0;
  else if (y > 1) y = 1;
  return { x, y };
}

/** Inverse of lonLatToMercator (exposed mainly for round-trip verification). */
export function mercatorToLonLat(x: number, y: number): { lonDeg: number; latDeg: number } {
  const lonDeg = x * 360 - 180;
  const n = Math.PI * (1 - 2 * y);
  const latDeg = (Math.atan(Math.sinh(n)) * 180) / Math.PI;
  return { lonDeg, latDeg };
}

/**
 * Shortest signed delta between two normalized-mercator X values, wrapping
 * the antimeridian (x is periodic with period 1). Without this, a finite
 * difference across lon ±180 (x 0.999 -> 0.001) would read as a near-full
 * westward world traverse instead of a tiny eastward step.
 */
export function wrapMercDelta(dx: number): number {
  if (dx > 0.5) return dx - 1;
  if (dx < -0.5) return dx + 1;
  return dx;
}

/**
 * Read one satellite slot out of a packed buffer (round-trip / CPU-picking
 * helper). mercX/mercY/altMeters are the EXACT-TICK propagated position (the
 * physics truth consumers analyze against); vel* are the display-glide rates
 * (see the SAT_STRIDE layout doc) — exposed for round-trip verification, not
 * for repositioning: the shader owns the glide.
 */
export function readSatAt(
  buffer: ArrayLike<number>,
  i: number,
): {
  mercX: number;
  mercY: number;
  altMeters: number;
  classCode: number;
  valid: boolean;
  velX: number;
  velY: number;
  velAlt: number;
} {
  const base = i * SAT_STRIDE;
  const classCode = buffer[base + 3];
  return {
    mercX: buffer[base],
    mercY: buffer[base + 1],
    altMeters: buffer[base + 2],
    classCode,
    valid: classCode >= 0,
    velX: buffer[base + 4],
    velY: buffer[base + 5],
    velAlt: buffer[base + 6],
  };
}
