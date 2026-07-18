// orbitPath — precomputed ORBIT ELLIPSE polylines for the space frame
// (celestial v2 B3, directive §3, 2026-07-18: "full orbit ellipses for
// Mercury–Neptune …, the Moon's orbit around Earth, and orbits for a
// curated set of major moons … Paths are polylines sampled …, cached,
// restyled — never resampled — on slider moves").
//
// SAMPLING REUSES THE EPHEMERIS (charter: never invent a second one): an
// orbit line is the locus of the body's REAL ephemeris positions across one
// orbital period — planets sampled heliocentrically, satellites as LOCAL
// offsets from their parent (body(t) − parent(t)). The line therefore
// carries the same perturbations the positions do; a sampled loop closes to
// the perturbation/precession scale, which the tests bound explicitly.
//
// SCALE PASS-THROUGH (B3 charter: paths flow through the scaleModel layout
// "exactly like body positions"): scaleModel compresses a PARENTLESS body's
// heliocentric radius (direction preserved) and attaches a PARENTED body to
// its compressed parent at the TRUE local offset. Orbit lines obey the same
// two rules: layoutOrbitPolyline() applies compressDistance per point to a
// heliocentric line (c = 0 is the bit-exact identity), and a satellite line
// is drawn as stored-true offsets translated to the parent's layout
// position (no per-point transform at all — the identity, like the body).
//
// CACHING / PERF (charter PERF RULES): sampling happens once per body
// (chunked by the caller — spaceFrame samples one body per macrotask, a few
// ms each, never a >16ms main-thread task); layout arrays are recomputed
// ONLY when the compression slider value changes (memoized per c), and
// resampling happens only when the simulation clock has moved far enough
// that precession would visibly rotate the line (staleness threshold below).
//
// Hermetic: pure math + a tiny persisted preference (localStorage, the
// scaleModel pattern). `npx tsx --test`-able.

import { compressDistance } from "./scaleModel.js";

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

/** Samples per planet/Moon loop (1.875° of mean motion per segment). */
export const ORBIT_SAMPLES_PLANET = 192;
/** Samples per curated-moon loop (near-circular ellipses need fewer). */
export const ORBIT_SAMPLES_MOON = 96;

/**
 * Resample a cached line when the sim clock is this far from its sample
 * instant. 30 days rotates the fastest-precessing curated ellipse
 * (Phobos' apsis, 1.1 yr period) ~27° — but at e 0.015 that reshapes the
 * near-circle invisibly; every slower element moves ≤4°. The Moon's
 * perturbed shape drifts on the same monthly scale. Cheap to redo: one
 * body per macrotask.
 */
export const ORBIT_RESAMPLE_MS = 30 * 86_400_000;

/**
 * One cached orbit line. `pts` is xyz-interleaved (3 floats per vertex,
 * length 3n), in METERS: heliocentric ecliptic for parentless bodies,
 * parent-centric TRUE offsets for satellites. The loop is NOT closed in
 * the array — the renderer joins last→first (closure gap = real
 * perturbation drift, bounded by tests).
 */
export interface OrbitPolyline {
  bodyId: string;
  parentId: string | null;
  pts: Float64Array;
  sampledAtMs: number;
  periodDays: number;
  /** max |vertex| (satellites: max offset radius) — the renderer's
   *  skip-when-sub-pixel bound. */
  maxRadiusM: number;
}

/**
 * Sample one body's orbit line from its REAL ephemeris across one period
 * centered on t0. `parentEph` null ⇒ heliocentric points; else local
 * offsets eph(t) − parentEph(t). Pure; cost = n ephemeris evaluations
 * (the caller chunks bodies across macrotasks).
 */
export function sampleOrbitPolyline(
  bodyId: string,
  parentId: string | null,
  eph: (timeMs: number) => Vec3,
  parentEph: ((timeMs: number) => Vec3) | null,
  t0Ms: number,
  periodDays: number,
  n: number,
): OrbitPolyline {
  const pts = new Float64Array(n * 3);
  const periodMs = periodDays * 86_400_000;
  let maxR = 0;
  for (let i = 0; i < n; i++) {
    const t = t0Ms + (i / n) * periodMs;
    const p = eph(t);
    let x = p.x;
    let y = p.y;
    let z = p.z;
    if (parentEph) {
      const q = parentEph(t);
      x -= q.x;
      y -= q.y;
      z -= q.z;
    }
    pts[i * 3] = x;
    pts[i * 3 + 1] = y;
    pts[i * 3 + 2] = z;
    const r = Math.hypot(x, y, z);
    if (r > maxR) maxR = r;
  }
  return { bodyId, parentId, pts, sampledAtMs: t0Ms, periodDays, maxRadiusM: maxR };
}

/**
 * The scaleModel pass-through for a HELIOCENTRIC line: per-point radial
 * compression, direction preserved — the identical rule applyDistanceCompression
 * applies to a parentless body position. c = 0 returns the points bit-exactly
 * (compressDistance's special-cased identity). `out` (same length) is filled
 * in place when given — the caller memoizes one buffer per line per c.
 * Satellite lines DO NOT pass through this (true local offsets ride the
 * parent's layout position — the parented rule), which is why the renderer
 * never calls it for them.
 */
export function layoutOrbitPolyline(pts: Float64Array, c: number, out?: Float64Array): Float64Array {
  const dst = out && out.length === pts.length ? out : new Float64Array(pts.length);
  for (let i = 0; i < pts.length; i += 3) {
    const x = pts[i];
    const y = pts[i + 1];
    const z = pts[i + 2];
    const r = Math.hypot(x, y, z);
    const k = r > 0 ? compressDistance(r, c) / r : 0;
    dst[i] = x * k;
    dst[i + 1] = y * k;
    dst[i + 2] = z * k;
  }
  return dst;
}

/** True when the sim clock has drifted far enough from the sample instant
 *  that the cached line should be resampled (see ORBIT_RESAMPLE_MS). */
export function orbitPolylineStale(sampledAtMs: number, timeMs: number): boolean {
  return Math.abs(timeMs - sampledAtMs) > ORBIT_RESAMPLE_MS;
}

// ── persisted preference (the scaleModel/localStorage pattern) ──────────────

export const ORBIT_PATHS_PREF_KEY = "vt-celestial-orbits";

/** Default ON: the computed real orbits are part of the premium default
 *  view (they are RAW computed geometry with a stated source, not a
 *  prediction — no ladder gate applies). */
export function readOrbitPathsPref(): boolean {
  try {
    const raw = globalThis.localStorage?.getItem(ORBIT_PATHS_PREF_KEY);
    if (raw === "0") return false;
    if (raw === "1") return true;
  } catch { /* no storage — default */ }
  return true;
}

let orbitsOn: boolean = readOrbitPathsPref();
const listeners = new Set<() => void>();

export function getOrbitPathsPref(): boolean {
  return orbitsOn;
}

export function setOrbitPathsPref(on: boolean): void {
  if (on === orbitsOn) return;
  orbitsOn = on;
  try {
    globalThis.localStorage?.setItem(ORBIT_PATHS_PREF_KEY, on ? "1" : "0");
  } catch { /* best-effort */ }
  for (const fn of Array.from(listeners)) {
    try { fn(); } catch { /* listener owns its errors */ }
  }
}

export function subscribeOrbitPathsPref(fn: () => void): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}
