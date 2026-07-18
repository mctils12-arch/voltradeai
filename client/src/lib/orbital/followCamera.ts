// FOLLOW-CAMERA INSPECT — satellite UX directive 2026-07-18 §3 (human, after
// two rejected attempts: "clicking Inspect, the satellite leaves the frame
// entirely"). A camera ATTACHED to the live, moving satellite. Time does not
// stop: getState() is fed from the SAME live worker buffer (exact-tick
// position + the worker's real velocity glide) that draws the map layer, so
// the craft keeps flying its real orbit while you inspect it.
//
// HISTORY / PROVENANCE OF THE MECHANICS: the retired O7 inspectScene
// (deleted at 2b7e69a — "INSPECT IS THE MAP"; recovered from git history as
// REFERENCE ONLY) had the right CAMERA MECHANICS (free orbit around the
// craft, dolly, real-ephemeris Sun/Moon/terminator, the honesty-caption
// discipline) but the wrong world attachment (a frozen snapshot fed once at
// entry). This module re-derives those mechanics with the live attachment:
//   · the craft anchor MOVES every frame (glidedCraftState below),
//   · ephemeris time is NOW every frame (sun/moon/terminator advance),
//   · "Back to map" is the parent's job (it restores the exact prior map
//     camera) — this module never touches the map.
//
// TWO VIEWS (directive §3 / design screen "1e inspect chrome"):
//   ORBIT   — camera orbits the CRAFT: drag = full-sphere orbit (yaw 360°,
//             pitch ±89° — from above, from behind, from the Earth side),
//             scroll/pinch = dolly from close structural detail out to the
//             craft small against the Earth limb.
//   ONBOARD — camera AT the craft looking out: free-look every direction
//             (drag), zoom = FOV narrowing toward whatever you face — the
//             Earth below, the real Moon/Sun, other catalogued satellites
//             as markers from the live buffer.
//
// HONESTY TIERS (each stated in the per-view provenance strings the parent
// shows on screen — the retired scene's caption discipline):
//   CRAFT  — the injected mesh (satModelLayer's real .vtm model when one
//            exists, else the catalogued class form, else a generic
//            bus+panels labeled representative; the parent's meshLabel
//            carries the derivation caption). Lit by the REAL sun direction
//            (subsolar ephemeris ~0.01°) — Lambert diffuse with a hard
//            n·L clamp as the self-shadow approximation (faces turned from
//            the sun go dark; no cast-shadow raytracing, stated) — plus an
//            earthshine fill from the nadir whose strength follows whether
//            the Earth face below is actually sunlit, and the whole sun
//            term scaled by the Earth-shadow factor below.
//   UMBRA  — B6 first slice: the craft goes DARK crossing into Earth's
//            shadow, computed from the REAL cone geometry (sun/Earth radii
//            at 1 AU — penumbra dim transition, umbra black; earthShadow()
//            below, unit-tested).
//   EARTH  — a sphere of TRUE ANGULAR SIZE from the craft's live altitude
//            (asin(R/(R+h))), in the nadir direction, textured with a
//            NASA-derived equirectangular surface map (threex.planets
//            earthmap1k, shipped at EARTH_TEXTURE_URL — a STATIC base map,
//            not live imagery of the moment; the credit line the parent
//            renders says so) and shaded by the REAL day/night terminator.
//            The texture is aligned through the SAME scene→ECEF rotation
//            the 15° graticule always used (sceneToEcefMat3, unit-pinned),
//            so the subsatellite point is geographically right. If the
//            texture cannot load, the scene falls back to the labeled
//            simplified graticule globe — never a broken texture.
//   MOON   — disc at the real ephemeris direction, true angular size,
//            sphere-shaded by the real sun direction (real phase +
//            lit-side orientation). Display-grade (~1°), labeled.
//   SUN    — bright disc at the true ~0.53° angular size.
//   OTHERS — onboard view only: every valid slot in the live position
//            buffer, drawn as a marker at its real direction and scaled
//            scene range (markers, not lit models — stated).
//   SKY    — pure black. No decorative starfield: real positions or absent.
//
// FRAME MATH CONVENTION (validated in followCamera.test.ts): the craft sits
// at the SCENE ORIGIN. Its local frame is ENU at its geodetic lat/lon
// (spherical Earth, R = 6371 km — same model as lib/orbital/geometry.ts):
// scene.x = East, scene.y = Up (zenith), scene.z = -North. Nadir is
// (0,-1,0). Celestial directions are geocentric unit vectors through their
// sub-points rotated into ENU (sun parallax < 0.003°, negligible; moon
// parallax ≲1° — same order as the ephemeris' own stated grade).
//
// SCALE HONESTY: the craft mesh is meters-sized; Earth/Moon/Sun are drawn
// by ANGULAR size at fixed scene distances (Earth 4000, Moon 9000, Sun 9500
// model units — Sun honestly behind the Moon so a real syzygy occludes, and
// Earth's sphere depth-occludes both plus the neighbor markers, so a body
// below the craft's horizon is honestly hidden). Angular sizes are exact
// from the origin; the camera orbits ≤ ~120 units, so off-origin parallax
// distorts apparent sizes ≲3% — a stated display approximation.
//
// GL DISCIPLINE (spaceFrame/celestialSky mount pattern): one canvas overlaid
// on the map container, handle with dispose() and a getRenderFailed() latch
// (any GL failure disables the scene and the page continues — the parent
// falls back to the plain close-orbit map ease), and `precision highp float`
// in EVERY shader stage — the O7 repair lesson (d27c5d3): a highp-vertex /
// mediump-fragment uniform precision mismatch fails the program LINK on
// real drivers.

import { subsolarPoint, moonState } from '../celestial/ephemeris.js';
import { readSatAt, mercatorToLonLat, SAT_STRIDE } from './satBuffer.js';
import type { Mesh } from './model3d.js';

// ── constants ───────────────────────────────────────────────────────────────

/** Mean Earth radius, meters — matches lib/orbital/geometry.ts (spherical). */
export const EARTH_RADIUS_M = 6_371_000;
/** Sun radius, meters (IAU nominal) — umbra/penumbra cone geometry. */
export const SUN_RADIUS_M = 696_000_000;
/** Mean Earth–Sun distance, meters (1 AU) — cone geometry display grade. */
export const AU_M = 149_597_870_700;
/** True mean solar angular RADIUS, radians (~0.533° diameter). */
export const SUN_ANGULAR_RADIUS_RAD = (0.533 / 2) * (Math.PI / 180);
/** Scene distance (model units) at which the Earth sphere is placed. */
export const EARTH_SCENE_DIST = 4000;
/** Scene distance of the Moon disc. */
export const MOON_SCENE_DIST = 9000;
/** Scene distance of the Sun disc — behind the Moon, so eclipses occlude. */
export const SUN_SCENE_DIST = 9500;
/** Default craft half-extent when no mesh is present (model3d builds ~1.2). */
export const DEFAULT_EXTENT = 1.2;
/** Honesty cap on between-tick glide extrapolation — matches satLayer's
 *  MAX_GLIDE_SEC: past this the display holds the last real tick. */
export const MAX_GLIDE_SEC = 2.5;
/** Onboard neighbor-marker cap (nearest first) — bounds the per-refresh
 *  upload; the count actually drawn vs total is reported honestly. */
export const NEIGHBOR_CAP = 4096;
/** NASA-derived equirectangular Earth surface map (threex.planets
 *  earthmap1k lineage), shipped in client/public/space — fetched lazily at
 *  mount, ~336 KB, zero bundle growth. Load failure falls back to the
 *  labeled simplified graticule globe (getEarthTexture() reports which). */
export const EARTH_TEXTURE_URL = '/space/earthmap1k.jpg';

/** Earth-texture lifecycle, surfaced so the parent's credit line never
 *  claims imagery that is not actually on screen. */
export type EarthTextureState = 'loading' | 'ready' | 'failed';

export type InspectView = 'orbit' | 'onboard';

/** Per-view provenance captions (the retired scene's discipline: every tier
 *  labeled, simplifications stated, nothing implied to be imagery). */
export const ORBIT_VIEW_PROVENANCE =
  'LIVE FOLLOW — the craft rides its real SGP4 orbit; time never stops. ' +
  'Craft lit by the real sun direction (subsolar ephemeris, ~0.01°): Lambert with hard-clamp self-shadow approximation, ' +
  'earthshine fill only while the Earth below is sunlit, and the craft darkens through penumbra into Earth’s umbra ' +
  '(real shadow-cone geometry). Earth at true angular size with NASA-derived surface imagery (a static base map, ' +
  'NOT live imagery of this moment) under the real day/night terminator. Moon at its real ephemeris direction and phase ' +
  '(display-grade, ~1°); Sun disc at true angular size (~0.53°). Black sky: no decorative starfield — real positions or absent.';
export const ONBOARD_VIEW_PROVENANCE =
  'LIVE FOLLOW, ONBOARD — free-look from the craft’s real moving SGP4 position; time never stops. ' +
  'Earth below at true angular size with NASA-derived surface imagery (a static base map, NOT live imagery of this ' +
  'moment) under the real day/night terminator. Moon/Sun at their real ephemeris directions and true angular sizes; ' +
  'other satellites are live catalogued positions from the same SGP4 buffer, drawn as markers (not lit models). ' +
  'Black sky: no decorative starfield — real positions or absent.';

/** Caption for the mesh fallback tier when neither a real model nor a
 *  catalogued class form exists (directive: "an honest generic bus+panels
 *  labeled representative"). */
export const GENERIC_MESH_LABEL =
  'Generic bus + solar panels — representative placeholder only (no catalogued class for this object); not imagery of this unit';

const DEG = Math.PI / 180;

// ── frame math (pure, exported for tests) ───────────────────────────────────

export interface EnuDir { e: number; n: number; u: number }

/** Unit ECEF vector through a geocentric lat/lon (spherical model). */
export function ecefUnit(latDeg: number, lonDeg: number): [number, number, number] {
  const lat = latDeg * DEG;
  const lon = lonDeg * DEG;
  const c = Math.cos(lat);
  return [c * Math.cos(lon), c * Math.sin(lon), Math.sin(lat)];
}

/** Rotate an ECEF vector into the ENU frame at (latDeg, lonDeg). */
export function ecefToEnu(v: [number, number, number], latDeg: number, lonDeg: number): EnuDir {
  const lat = latDeg * DEG;
  const lon = lonDeg * DEG;
  const sinLat = Math.sin(lat), cosLat = Math.cos(lat);
  const sinLon = Math.sin(lon), cosLon = Math.cos(lon);
  return {
    e: -sinLon * v[0] + cosLon * v[1],
    n: -sinLat * cosLon * v[0] - sinLat * sinLon * v[1] + cosLat * v[2],
    u: cosLat * cosLon * v[0] + cosLat * sinLon * v[1] + sinLat * v[2],
  };
}

/** ENU → scene axes: x=E, y=U(up), z=-N (right-handed, GL y-up). */
export function enuToScene(d: EnuDir): [number, number, number] {
  return [d.e, d.u, -d.n];
}

/** Column-major mat3 mapping SCENE directions → ECEF at (lat, lon) — the
 *  graticule shader uses it to find the real lat/lon under each Earth-sphere
 *  fragment. Columns are the scene axes expressed in ECEF: [E, U, -N]. */
export function sceneToEcefMat3(latDeg: number, lonDeg: number): Float32Array {
  const lat = latDeg * DEG, lon = lonDeg * DEG;
  const sinLat = Math.sin(lat), cosLat = Math.cos(lat);
  const sinLon = Math.sin(lon), cosLon = Math.cos(lon);
  // E = (-sinLon, cosLon, 0); N = (-sinLat cosLon, -sinLat sinLon, cosLat);
  // U = (cosLat cosLon, cosLat sinLon, sinLat)
  return new Float32Array([
    -sinLon, cosLon, 0,                                      // col 0: scene.x = E
    cosLat * cosLon, cosLat * sinLon, sinLat,                // col 1: scene.y = U
    sinLat * cosLon, sinLat * sinLon, -cosLat,               // col 2: scene.z = -N
  ]);
}

/** REAL sun direction in the craft's local ENU frame (see the retired
 *  scene's anchors, re-pinned in followCamera.test.ts). */
export function sunDirLocal(timeMs: number, craftLatDeg: number, craftLonDeg: number): EnuDir {
  const sp = subsolarPoint(timeMs);
  return ecefToEnu(ecefUnit(sp.latDeg, sp.lonDeg), craftLatDeg, craftLonDeg);
}

export interface MoonLocal {
  dir: EnuDir;
  angularRadiusRad: number;
  illuminatedFraction: number;
}

/** REAL moon direction + true angular size in the craft's local frame
 *  (geocentric direction — LEO topocentric parallax ≲1°, same order as the
 *  truncated lunar series' own grade; stated, never survey ephemeris). */
export function moonDirLocal(timeMs: number, craftLatDeg: number, craftLonDeg: number): MoonLocal {
  const m = moonState(timeMs);
  return {
    dir: ecefToEnu(ecefUnit(m.latDeg, m.lonDeg), craftLatDeg, craftLonDeg),
    angularRadiusRad: (m.angularSizeArcmin / 2 / 60) * DEG,
    illuminatedFraction: m.illuminatedFraction,
  };
}

/** Earth's true angular RADIUS from altitude h: asin(R/(R+h)). */
export function earthAngularRadiusRad(altMeters: number): number {
  const h = Math.max(1, altMeters);
  return Math.asin(EARTH_RADIUS_M / (EARTH_RADIUS_M + h));
}

// ── B6 first slice: Earth-shadow (umbra/penumbra) from real cone geometry ──

export interface EarthShadow {
  /** direct-sunlight factor 0..1 — 1 sunlit, 0 in umbra, between in penumbra. */
  factor: number;
  region: 'sunlit' | 'penumbra' | 'umbra';
}

/**
 * Where the craft sits relative to Earth's shadow, from REAL geometry: the
 * craft's geocentric position is (R+h)·ẑenith; with the sun direction ŝ in
 * the same ENU frame, the along-shadow-axis distance is d = -(R+h)·(ŝ·ẑ)
 * (positive on the anti-sun side) and the off-axis distance is
 * r = (R+h)·√(1-(ŝ·ẑ)²). The umbra cone converges at
 * tanα = (Rsun−Re)/AU, the penumbra diverges at tanβ = (Rsun+Re)/AU
 * (display grade: mean AU, spherical Earth — stated). Inside the umbra
 * radius the factor is 0; outside the penumbra radius it is 1; across the
 * band it ramps linearly (a stated display approximation of the real
 * partial-eclipse falloff).
 */
export function earthShadow(altMeters: number, sunDir: EnuDir): EarthShadow {
  const r = EARTH_RADIUS_M + Math.max(0, altMeters);
  const su = Math.max(-1, Math.min(1, sunDir.u)); // ŝ·ẑenith
  const dAlong = -r * su;                          // >0 = anti-sun side
  if (dAlong <= 0) return { factor: 1, region: 'sunlit' }; // sun above the shadow plane
  const rOff = r * Math.sqrt(Math.max(0, 1 - su * su));
  const tanUmbra = (SUN_RADIUS_M - EARTH_RADIUS_M) / AU_M;
  const tanPen = (SUN_RADIUS_M + EARTH_RADIUS_M) / AU_M;
  const rUmbra = EARTH_RADIUS_M - dAlong * tanUmbra;   // converging cone
  const rPen = EARTH_RADIUS_M + dAlong * tanPen;       // diverging cone
  if (rUmbra > 0 && rOff <= rUmbra) return { factor: 0, region: 'umbra' };
  if (rOff >= rPen) return { factor: 1, region: 'sunlit' };
  const span = Math.max(1, rPen - Math.max(0, rUmbra));
  const f = (rOff - Math.max(0, rUmbra)) / span;
  return { factor: Math.max(0, Math.min(1, f)), region: 'penumbra' };
}

// ── live craft state from the shared position buffer ────────────────────────

export interface CraftState {
  latDeg: number;
  lonDeg: number;
  altMeters: number;
  timeMs: number;
}

/**
 * The followed craft's DISPLAY position right now: the buffer's exact-tick
 * SGP4 position advanced along the worker's REAL velocity fields for the
 * time since the tick (the same first-order glide the map layer's shader
 * applies — satLayer u_dtSec), capped at MAX_GLIDE_SEC so a stale worker
 * holds rather than lies. This is what makes the inspect world LIVE between
 * 1 Hz physics ticks. Returns null on a sentinel slot (no honest position).
 */
export function glidedCraftState(
  buffer: ArrayLike<number> | null,
  index: number,
  tickAnchorMs: number | null,
  nowMs: number,
  timeMs: number,
): CraftState | null {
  if (!buffer || index < 0 || (index + 1) * SAT_STRIDE > buffer.length) return null;
  const s = readSatAt(buffer, index);
  if (!s.valid) return null;
  let dt = tickAnchorMs == null ? 0 : (nowMs - tickAnchorMs) / 1000;
  if (!Number.isFinite(dt) || dt < 0) dt = 0;
  if (dt > MAX_GLIDE_SEC) dt = MAX_GLIDE_SEC;
  let mx = s.mercX + s.velX * dt;
  mx -= Math.floor(mx); // wrap the antimeridian (x periodic with period 1)
  const my = Math.max(0, Math.min(1, s.mercY + s.velY * dt));
  const alt = s.altMeters + s.velAlt * dt;
  const ll = mercatorToLonLat(mx, my);
  if (!Number.isFinite(ll.lonDeg) || !Number.isFinite(ll.latDeg)) return null;
  return { latDeg: ll.latDeg, lonDeg: ll.lonDeg, altMeters: alt, timeMs };
}

// ── onboard neighbors: other catalogued objects from the live buffer ────────

export interface NeighborSet {
  /** packed xyz scene-frame unit directions, nearest-first. */
  dirs: Float32Array;
  /** real range in meters, aligned with dirs (dirs.length/3 entries). */
  rangesM: Float64Array;
  count: number;
  /** valid population before the cap (honesty readout). */
  total: number;
}

/**
 * Directions to every OTHER valid object in the live buffer, in the craft's
 * scene frame, nearest-first under NEIGHBOR_CAP. Real positions only:
 * sentinel slots and the craft's own slot are skipped, nothing interpolated.
 */
export function neighborDirsScene(
  buffer: ArrayLike<number> | null,
  selfIndex: number,
  craft: { latDeg: number; lonDeg: number; altMeters: number },
  cap = NEIGHBOR_CAP,
): NeighborSet {
  const empty: NeighborSet = { dirs: new Float32Array(0), rangesM: new Float64Array(0), count: 0, total: 0 };
  if (!buffer) return empty;
  const n = Math.floor(buffer.length / SAT_STRIDE);
  const selfR = EARTH_RADIUS_M + craft.altMeters;
  const selfU = ecefUnit(craft.latDeg, craft.lonDeg);
  const sx = selfU[0] * selfR, sy = selfU[1] * selfR, sz = selfU[2] * selfR;
  const cand: Array<{ d2: number; x: number; y: number; z: number }> = [];
  for (let i = 0; i < n; i++) {
    if (i === selfIndex) continue;
    const s = readSatAt(buffer, i);
    if (!s.valid) continue;
    const ll = mercatorToLonLat(s.mercX, s.mercY);
    const u = ecefUnit(ll.latDeg, ll.lonDeg);
    const r = EARTH_RADIUS_M + s.altMeters;
    const dx = u[0] * r - sx, dy = u[1] * r - sy, dz = u[2] * r - sz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < 1) continue; // co-located artifact — no honest direction
    cand.push({ d2, x: dx, y: dy, z: dz });
  }
  const total = cand.length;
  cand.sort((a, b) => a.d2 - b.d2);
  const kept = cand.slice(0, Math.max(0, cap));
  const dirs = new Float32Array(kept.length * 3);
  const ranges = new Float64Array(kept.length);
  for (let i = 0; i < kept.length; i++) {
    const c = kept[i];
    const len = Math.sqrt(c.d2);
    const enu = ecefToEnu([c.x / len, c.y / len, c.z / len], craft.latDeg, craft.lonDeg);
    const sc = enuToScene(enu);
    dirs[i * 3] = sc[0]; dirs[i * 3 + 1] = sc[1]; dirs[i * 3 + 2] = sc[2];
    ranges[i] = len;
  }
  return { dirs, rangesM: ranges, count: kept.length, total };
}

// ── orbit camera (retired-scene mechanics, re-pinned by tests) ──────────────

/** Pitch bound: ±89° — full range without the lookAt pole singularity. */
export const PITCH_LIMIT_RAD = 89 * DEG;
/** Orbit sensitivity: radians per CSS pixel of drag. */
export const DRAG_RAD_PER_PX = 0.0055;
/** Release-inertia decay time constant, seconds. */
export const DAMPING_TAU_S = 0.35;
/** Dolly smoothing time constant, seconds. */
export const DOLLY_TAU_S = 0.12;

export interface OrbitCam {
  yaw: number;
  pitch: number;
  dist: number;
  velYaw: number;
  velPitch: number;
  distTarget: number;
  minDist: number;
  maxDist: number;
  dragging: boolean;
}

/** Dolly bounds from the craft's half-extent: ~2× (structural detail) to
 *  ~50× (craft small against the Earth limb). */
export function camDollyBounds(extent: number): { min: number; max: number } {
  const e = Math.max(1e-6, extent);
  return { min: 2 * e, max: 50 * e };
}

export function wrapAngle(a: number): number {
  return ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

export function createOrbitCam(extent: number): OrbitCam {
  const b = camDollyBounds(extent);
  const dist = Math.min(b.max, Math.max(b.min, extent * 6));
  return {
    yaw: 0.6, pitch: 0.32, dist,
    velYaw: 0, velPitch: 0, distTarget: dist,
    minDist: b.min, maxDist: b.max, dragging: false,
  };
}

export function camApplyDrag(cam: OrbitCam, dxPx: number, dyPx: number): void {
  cam.yaw = wrapAngle(cam.yaw - dxPx * DRAG_RAD_PER_PX);
  cam.pitch = Math.max(-PITCH_LIMIT_RAD, Math.min(PITCH_LIMIT_RAD, cam.pitch + dyPx * DRAG_RAD_PER_PX));
  cam.velYaw = -dxPx * DRAG_RAD_PER_PX * 60;
  cam.velPitch = dyPx * DRAG_RAD_PER_PX * 60;
}

export function camApplyDolly(cam: OrbitCam, factor: number): void {
  if (!(factor > 0) || !Number.isFinite(factor)) return;
  cam.distTarget = Math.max(cam.minDist, Math.min(cam.maxDist, cam.distTarget * factor));
}

/** Advance inertia + smoothing by dtMs; true while still settling. */
export function camStep(cam: OrbitCam, dtMs: number): boolean {
  const dt = Math.min(0.1, Math.max(0, dtMs / 1000));
  if (!cam.dragging && (cam.velYaw !== 0 || cam.velPitch !== 0)) {
    cam.yaw = wrapAngle(cam.yaw + cam.velYaw * dt);
    const p = cam.pitch + cam.velPitch * dt;
    cam.pitch = Math.max(-PITCH_LIMIT_RAD, Math.min(PITCH_LIMIT_RAD, p));
    if (cam.pitch !== p) cam.velPitch = 0;
    const decay = Math.exp(-dt / DAMPING_TAU_S);
    cam.velYaw *= decay;
    cam.velPitch *= decay;
    if (Math.abs(cam.velYaw) < 0.004) cam.velYaw = 0;
    if (Math.abs(cam.velPitch) < 0.004) cam.velPitch = 0;
  }
  const gap = cam.distTarget - cam.dist;
  if (gap !== 0) {
    cam.dist += gap * (1 - Math.exp(-dt / DOLLY_TAU_S));
    if (Math.abs(cam.distTarget - cam.dist) < 1e-4 * cam.distTarget) cam.dist = cam.distTarget;
  }
  return cam.velYaw !== 0 || cam.velPitch !== 0 || cam.dist !== cam.distTarget;
}

/** Camera eye position: spherical orbit around the origin. pitch<0 = below
 *  the craft — looking at it from the Earth side. */
export function camEye(cam: OrbitCam): [number, number, number] {
  const cp = Math.cos(cam.pitch);
  return [
    cam.dist * cp * Math.sin(cam.yaw),
    cam.dist * Math.sin(cam.pitch),
    cam.dist * cp * Math.cos(cam.yaw),
  ];
}

// ── onboard free-look camera ────────────────────────────────────────────────

/** FOV zoom bounds, degrees — wide look-around down to a tight telephoto. */
export const LOOK_FOV_MAX_DEG = 75;
export const LOOK_FOV_MIN_DEG = 6;
/** Elevation bound ±89° (same pole guard as the orbit cam). */
export const LOOK_EL_LIMIT_DEG = 89;

export interface LookCam {
  /** azimuth, degrees — 0 = north, 90 = east (compass sense). */
  azDeg: number;
  /** elevation, degrees — +90 zenith, -90 nadir. */
  elDeg: number;
  fovDeg: number;
  fovTargetDeg: number;
}

export function createLookCam(): LookCam {
  // opening pose: over the shoulder toward the Earth limb ahead
  return { azDeg: 0, elDeg: -32, fovDeg: 60, fovTargetDeg: 60 };
}

/** Free-look drag: sensitivity scales with FOV so a zoomed view stays
 *  controllable. Drag right = look right (az increases eastward). */
export function lookApplyDrag(cam: LookCam, dxPx: number, dyPx: number): void {
  const k = 0.12 * (cam.fovDeg / 60);
  cam.azDeg = ((cam.azDeg + dxPx * k) % 360 + 360) % 360;
  cam.elDeg = Math.max(-LOOK_EL_LIMIT_DEG, Math.min(LOOK_EL_LIMIT_DEG, cam.elDeg - dyPx * k));
}

/** Wheel/pinch zoom: multiplies the target FOV, clamped. */
export function lookApplyZoom(cam: LookCam, factor: number): void {
  if (!(factor > 0) || !Number.isFinite(factor)) return;
  cam.fovTargetDeg = Math.max(LOOK_FOV_MIN_DEG, Math.min(LOOK_FOV_MAX_DEG, cam.fovTargetDeg * factor));
}

/** Smooth the FOV toward its target; true while still settling. */
export function lookStep(cam: LookCam, dtMs: number): boolean {
  const dt = Math.min(0.1, Math.max(0, dtMs / 1000));
  const gap = cam.fovTargetDeg - cam.fovDeg;
  if (gap !== 0) {
    cam.fovDeg += gap * (1 - Math.exp(-dt / DOLLY_TAU_S));
    if (Math.abs(cam.fovTargetDeg - cam.fovDeg) < 1e-3) cam.fovDeg = cam.fovTargetDeg;
  }
  return cam.fovDeg !== cam.fovTargetDeg;
}

/** Look direction in scene axes from az/el (az 0=north → scene -z). */
export function lookDirScene(cam: LookCam): [number, number, number] {
  const az = cam.azDeg * DEG, el = cam.elDeg * DEG;
  const ce = Math.cos(el);
  return [ce * Math.sin(az), Math.sin(el), -ce * Math.cos(az)];
}

/** Craft half-extent = max |coordinate| over the mesh (dolly reference). */
export function meshExtent(mesh: Mesh | null): number {
  if (!mesh || mesh.vertexCount === 0) return DEFAULT_EXTENT;
  let m = 0;
  const p = mesh.positions;
  for (let i = 0; i < p.length; i++) {
    const a = Math.abs(p[i]);
    if (a > m) m = a;
  }
  return m > 0 ? m : DEFAULT_EXTENT;
}

// ── tiny mat4 (column-major, internal) ──────────────────────────────────────

type Mat4 = Float32Array;

function mat4Perspective(fovyRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovyRad / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

function mat4LookAt(eye: [number, number, number], target: [number, number, number]): Mat4 {
  let fx = target[0] - eye[0], fy = target[1] - eye[1], fz = target[2] - eye[2];
  let l = Math.hypot(fx, fy, fz) || 1;
  fx /= l; fy /= l; fz /= l;
  let sx = -fz; const sy = 0; let sz = fx;
  l = Math.hypot(sx, sy, sz) || 1;
  sx /= l; sz /= l;
  const ux = sy * fz - sz * fy;
  const uy = sz * fx - sx * fz;
  const uz = sx * fy - sy * fx;
  const m = new Float32Array(16);
  m[0] = sx; m[4] = sy; m[8] = sz;
  m[1] = ux; m[5] = uy; m[9] = uz;
  m[2] = -fx; m[6] = -fy; m[10] = -fz;
  m[12] = -(sx * eye[0] + sy * eye[1] + sz * eye[2]);
  m[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
  m[14] = fx * eye[0] + fy * eye[1] + fz * eye[2];
  m[15] = 1;
  return m;
}

function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + row] * b[col * 4 + k];
      o[col * 4 + row] = s;
    }
  }
  return o;
}

// ── shaders — precision highp float in EVERY stage (the O7 link lesson) ─────

const MESH_VS = `#version 300 es
precision highp float;
in vec3 a_pos;
in vec3 a_normal;
in vec3 a_color;
uniform mat4 u_mvp;
out vec3 v_normal;
out vec3 v_color;
void main() {
  gl_Position = u_mvp * vec4(a_pos, 1.0);
  v_normal = a_normal;
  v_color = a_color;
}`;

// REAL sun key light × the Earth-shadow factor (u_shadow: 1 sunlit → 0
// umbra, penumbra between) + earthshine fill from the nadir whose strength
// (u_fillK, CPU-computed) follows whether the Earth below is sunlit. The
// n·L hard clamp is the stated self-shadow approximation.
const MESH_FS = `#version 300 es
precision highp float;
in vec3 v_normal;
in vec3 v_color;
uniform vec3 u_sun;     // scene-frame unit sun direction (real ephemeris)
uniform float u_shadow; // Earth-shadow factor 0..1 (umbra..sunlit)
uniform float u_fillK;  // earthshine strength 0..1
out vec4 o;
void main() {
  vec3 n = normalize(v_normal);
  float sun = max(dot(n, u_sun), 0.0) * u_shadow;
  float fill = max(dot(n, vec3(0.0, -1.0, 0.0)), 0.0);
  vec3 c = v_color * (0.05 + 1.02 * sun)
         + v_color * vec3(0.45, 0.55, 0.75) * (0.22 * u_fillK) * fill;
  o = vec4(c, 1.0);
}`;

const EARTH_VS = `#version 300 es
precision highp float;
in vec3 a_pos; // unit sphere
uniform mat4 u_mvp;
uniform vec3 u_center;
uniform float u_radius;
out vec3 v_normal;
out vec3 v_world;
void main() {
  vec3 wp = u_center + a_pos * u_radius;
  gl_Position = u_mvp * vec4(wp, 1.0);
  v_normal = a_pos;
  v_world = wp;
}`;

// Day/night by the REAL terminator (dot(surface normal, real sun dir)),
// over the NASA-derived equirect surface map sampled through the SAME
// scene→ECEF rotation the graticule always used (sceneToEcefMat3 — the
// alignment is unit-pinned, so the subsatellite geography is right).
// u_hasTex=0 (texture still loading / failed) keeps the labeled simplified
// graticule globe as the honest fallback.
const EARTH_FS = `#version 300 es
precision highp float;
in vec3 v_normal;
in vec3 v_world;
uniform vec3 u_sun;
uniform vec3 u_camPos;
uniform mat3 u_sceneToEcef;
uniform sampler2D u_tex; // NASA-derived equirect surface map (static)
uniform float u_hasTex;  // 1 once loaded; 0 = simplified-globe fallback
out vec4 o;
void main() {
  vec3 n = normalize(v_normal);
  float lit = dot(n, u_sun);
  float day = smoothstep(-0.08, 0.12, lit);
  // real lat/lon under this fragment (scene normal → ECEF)
  vec3 ne = normalize(u_sceneToEcef * n);
  float latDeg = degrees(asin(clamp(ne.z, -1.0, 1.0)));
  float lonDeg = degrees(atan(ne.y, ne.x));
  // equirect sample: u=0 at 180°W → lon/360+0.5; v=0 at the north pole
  // (image top row = +90°, default DOM upload order) → 0.5-lat/180
  vec3 texCol = texture(u_tex, vec2(lonDeg / 360.0 + 0.5, 0.5 - latDeg / 180.0)).rgb;
  vec3 flatDay = mix(vec3(0.07, 0.20, 0.42), vec3(0.16, 0.36, 0.60), 0.5 + 0.5 * n.y);
  vec3 dayCol = mix(flatDay, texCol, u_hasTex);
  // night side: near-black; with imagery, a faint earthshine-grade hint of
  // the same real surface (display shading — no fabricated city lights)
  vec3 nightCol = mix(vec3(0.004, 0.006, 0.012), texCol * vec3(0.020, 0.024, 0.034), u_hasTex);
  vec3 c = mix(nightCol, dayCol, day);
  // 15° graticule: only the no-texture fallback needs it to show motion
  vec2 g = vec2(latDeg / 15.0, lonDeg / 15.0);
  vec2 dg = fwidth(g);
  vec2 cell = abs(fract(g) - 0.5);
  vec2 line = 1.0 - smoothstep(vec2(0.0), dg * 1.4, cell);
  float grat = max(line.x, line.y) * (1.0 - u_hasTex);
  c += vec3(0.10, 0.16, 0.22) * grat * (0.25 + 0.75 * day);
  vec3 vDir = normalize(u_camPos - v_world);
  float rim = pow(1.0 - clamp(dot(n, vDir), 0.0, 1.0), 3.0);
  c += vec3(0.25, 0.45, 0.80) * rim * (0.10 + 0.55 * day);
  o = vec4(c, 1.0);
}`;

const DISC_VS = `#version 300 es
precision highp float;
in vec2 a_uv; // corner in [-1,1]
uniform mat4 u_mvp;
uniform vec3 u_centerPos;
uniform vec3 u_right;
uniform vec3 u_up2;
uniform float u_radius;
out vec2 v_uv;
void main() {
  vec3 wp = u_centerPos + (a_uv.x * u_right + a_uv.y * u_up2) * u_radius;
  gl_Position = u_mvp * vec4(wp, 1.0);
  v_uv = a_uv;
}`;

// mode 0 = SUN (bright disc, soft limb); mode 1 = MOON shaded as a sphere
// lit by the real sun direction — the phase falls out of real geometry.
const DISC_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform int u_mode;
uniform vec3 u_right;
uniform vec3 u_up2;
uniform vec3 u_facing; // toward the viewer (= -direction)
uniform vec3 u_sun;
out vec4 o;
void main() {
  float r2 = dot(v_uv, v_uv);
  if (r2 > 1.0) discard;
  if (u_mode == 0) {
    float edge = smoothstep(1.0, 0.80, sqrt(r2));
    o = vec4(vec3(1.0, 0.97, 0.86), edge);
  } else {
    float z = sqrt(max(0.0, 1.0 - r2));
    vec3 n = normalize(v_uv.x * u_right + v_uv.y * u_up2 + z * u_facing);
    float lit = dot(n, u_sun);
    float day = smoothstep(-0.02, 0.06, lit);
    vec3 c = mix(vec3(0.012, 0.012, 0.016), vec3(0.62, 0.61, 0.57), day);
    o = vec4(c, 1.0);
  }
}`;

// onboard neighbor markers: fixed-size GL points at real directions/ranges
const POINT_VS = `#version 300 es
precision highp float;
in vec3 a_pos;
uniform mat4 u_mvp;
void main() {
  gl_Position = u_mvp * vec4(a_pos, 1.0);
  gl_PointSize = 3.0;
}`;

const POINT_FS = `#version 300 es
precision highp float;
uniform vec4 u_color;
out vec4 o;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  if (dot(d, d) > 0.25) discard;
  o = u_color;
}`;

const LINE_VS = `#version 300 es
precision highp float;
in vec3 a_pos;
uniform mat4 u_mvp;
void main() { gl_Position = u_mvp * vec4(a_pos, 1.0); }`;

const LINE_FS = `#version 300 es
precision highp float;
uniform vec4 u_color;
out vec4 o;
void main() { o = u_color; }`;

// ── mount ───────────────────────────────────────────────────────────────────

export interface FollowCameraOptions {
  /** real or form mesh, injected by the parent (null = generic tier is the
   *  PARENT's call — this module draws whatever it is given). */
  mesh: Mesh | null;
  meshLabel: string;
  /** LIVE per-frame craft state (glidedCraftState above) — null ticks hold
   *  the last honest state, never guess. */
  getState: () => CraftState | null;
  /** live position buffer for onboard neighbor markers (raw tick preferred
   *  so a display filter never hides real objects here). */
  getNeighborBuffer?: () => ArrayLike<number> | null;
  /** the craft's own buffer index (skipped in the neighbor set). */
  selfIndex?: number;
  /** ephemeris clock — defaults to Date.now (the parent may thread a
   *  prod-inert test seam through here). */
  getTimeMs?: () => number;
  initialView?: InspectView;
}

export interface LightingReadout {
  sunEnu: EnuDir;
  shadow: EarthShadow;
  /** earthshine fill strength 0..1 actually fed to the shader. */
  earthshine: number;
}

export interface FollowCameraHandle {
  setView(v: InspectView): void;
  getView(): InspectView;
  setMesh(mesh: Mesh | null, label: string): void;
  getMeshLabel(): string;
  /** provenance caption for the CURRENT view (parent renders it). */
  getProvenance(): string;
  render(): void;
  dispose(): void;
  getRenderFailed(): boolean;
  getCamera(): { yaw: number; pitch: number; dist: number };
  getLook(): { azDeg: number; elDeg: number; fovDeg: number };
  /** Earth-imagery lifecycle — the parent's credit line renders from this,
   *  so "NASA imagery" is only ever claimed while it is actually drawn. */
  getEarthTexture(): EarthTextureState;
  /** drive/test seams — real internal state, prod-inert to read. */
  setLook(azDeg: number, elDeg: number): void;
  getLighting(): LightingReadout | null;
  getStateNow(): CraftState | null;
  getNeighborCount(): { drawn: number; total: number };
  /** median of the last render-loop frame CPU times, ms (perf readout). */
  getFrameStats(): { medianMs: number; samples: number };
}

interface GlProgram {
  prog: WebGLProgram;
  attrs: Record<string, number>;
  unis: Record<string, WebGLUniformLocation | null>;
}

export function mountFollowCamera(container: HTMLElement, opts: FollowCameraOptions): FollowCameraHandle {
  let disposed = false;
  let renderFailed = false;
  let mesh = opts.mesh;
  let meshLabel = opts.meshLabel;
  let extent = meshExtent(mesh);
  let view: InspectView = opts.initialView ?? 'orbit';
  const cam = createOrbitCam(extent);
  const look = createLookCam();
  const getTimeMs = opts.getTimeMs ?? (() => Date.now());

  let lastState: CraftState | null = null;
  let lastLighting: LightingReadout | null = null;
  let neighborCount = { drawn: 0, total: 0 };
  let earthTexState: EarthTextureState = 'loading';
  const frameTimes: number[] = [];

  const doc: Document | null =
    (container && (container as { ownerDocument?: Document }).ownerDocument) ??
    (typeof document !== 'undefined' ? document : null);

  const inert = (): FollowCameraHandle => ({
    setView(v) { view = v; },
    getView: () => view,
    setMesh(m, label) { mesh = m; meshLabel = label; },
    getMeshLabel: () => meshLabel,
    getProvenance: () => (view === 'orbit' ? ORBIT_VIEW_PROVENANCE : ONBOARD_VIEW_PROVENANCE),
    render() { /* disabled */ },
    dispose() { disposed = true; },
    getRenderFailed: () => renderFailed,
    getCamera: () => ({ yaw: cam.yaw, pitch: cam.pitch, dist: cam.dist }),
    getLook: () => ({ azDeg: look.azDeg, elDeg: look.elDeg, fovDeg: look.fovDeg }),
    getEarthTexture: () => 'failed', // no GL → nothing textured is drawn
    setLook(az, el) { look.azDeg = az; look.elDeg = Math.max(-LOOK_EL_LIMIT_DEG, Math.min(LOOK_EL_LIMIT_DEG, el)); },
    getLighting: () => lastLighting,
    getStateNow: () => lastState,
    getNeighborCount: () => neighborCount,
    getFrameStats: () => ({ medianMs: 0, samples: 0 }),
  });
  if (!doc) {
    renderFailed = true;
    return inert();
  }

  const canvas = doc.createElement('canvas');
  canvas.className = 'vt-follow-cam';
  try {
    canvas.style.position = 'absolute';
    canvas.style.inset = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.background = '#000000'; // pure black — no fake starfield
    canvas.style.cursor = 'grab';
    canvas.style.touchAction = 'none';
    container.appendChild(canvas);
  } catch {
    renderFailed = true;
    return inert();
  }

  // ── GL init, latched: any failure disables the scene, never the page ──
  let gl: WebGL2RenderingContext | null = null;
  let meshProg: GlProgram | null = null;
  let earthProg: GlProgram | null = null;
  let discProg: GlProgram | null = null;
  let pointProg: GlProgram | null = null;
  let lineProg: GlProgram | null = null;
  let bufMeshPos: WebGLBuffer | null = null;
  let bufMeshNor: WebGLBuffer | null = null;
  let bufMeshCol: WebGLBuffer | null = null;
  let meshDirty = true;
  let bufSphere: WebGLBuffer | null = null;
  let sphereVerts = 0;
  let bufQuad: WebGLBuffer | null = null;
  let bufPoints: WebGLBuffer | null = null;
  let bufLine: WebGLBuffer | null = null;
  let earthTex: WebGLTexture | null = null;
  let pointScene: Float32Array | null = null; // scene-space marker positions
  let lastNeighborRefreshMs = 0;
  let prevSample: CraftState | null = null;
  let prevSampleWallMs = 0;
  let lastVelDir: [number, number, number] | null = null;

  function compile(vsSrc: string, fsSrc: string, attrs: string[], unis: string[]): GlProgram {
    const g = gl!;
    const mk = (type: number, src: string): WebGLShader => {
      const sh = g.createShader(type);
      if (!sh) throw new Error('followCamera: createShader failed');
      g.shaderSource(sh, src);
      g.compileShader(sh);
      if (!g.getShaderParameter(sh, g.COMPILE_STATUS)) {
        const log = g.getShaderInfoLog(sh);
        g.deleteShader(sh);
        throw new Error('followCamera: shader compile failed: ' + log);
      }
      return sh;
    };
    const vs = mk(g.VERTEX_SHADER, vsSrc);
    const fs = mk(g.FRAGMENT_SHADER, fsSrc);
    const p = g.createProgram();
    if (!p) throw new Error('followCamera: createProgram failed');
    g.attachShader(p, vs);
    g.attachShader(p, fs);
    g.linkProgram(p);
    if (!g.getProgramParameter(p, g.LINK_STATUS)) {
      const log = g.getProgramInfoLog(p);
      g.deleteProgram(p);
      throw new Error('followCamera: program link failed: ' + log);
    }
    g.deleteShader(vs);
    g.deleteShader(fs);
    const out: GlProgram = { prog: p, attrs: {}, unis: {} };
    for (const a of attrs) out.attrs[a] = g.getAttribLocation(p, a);
    for (const u of unis) out.unis[u] = g.getUniformLocation(p, u);
    return out;
  }

  function buildSphere(stacks: number, slices: number): Float32Array {
    const v: number[] = [];
    const pt = (i: number, j: number): [number, number, number] => {
      const phi = (i / stacks) * Math.PI - Math.PI / 2;
      const th = (j / slices) * 2 * Math.PI;
      const c = Math.cos(phi);
      return [c * Math.cos(th), Math.sin(phi), c * Math.sin(th)];
    };
    for (let i = 0; i < stacks; i++) {
      for (let j = 0; j < slices; j++) {
        const a = pt(i, j), b = pt(i + 1, j), c = pt(i + 1, j + 1), d = pt(i, j + 1);
        v.push(...a, ...b, ...c, ...a, ...c, ...d);
      }
    }
    return new Float32Array(v);
  }

  try {
    gl = canvas.getContext('webgl2', { antialias: true }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error('followCamera: WebGL2 unavailable');
    meshProg = compile(MESH_VS, MESH_FS, ['a_pos', 'a_normal', 'a_color'], ['u_mvp', 'u_sun', 'u_shadow', 'u_fillK']);
    earthProg = compile(EARTH_VS, EARTH_FS, ['a_pos'],
      ['u_mvp', 'u_center', 'u_radius', 'u_sun', 'u_camPos', 'u_sceneToEcef', 'u_tex', 'u_hasTex']);
    discProg = compile(DISC_VS, DISC_FS, ['a_uv'],
      ['u_mvp', 'u_centerPos', 'u_right', 'u_up2', 'u_radius', 'u_mode', 'u_facing', 'u_sun']);
    pointProg = compile(POINT_VS, POINT_FS, ['a_pos'], ['u_mvp', 'u_color']);
    lineProg = compile(LINE_VS, LINE_FS, ['a_pos'], ['u_mvp', 'u_color']);
    const sphere = buildSphere(32, 48);
    sphereVerts = sphere.length / 3;
    bufSphere = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, bufSphere);
    gl.bufferData(gl.ARRAY_BUFFER, sphere, gl.STATIC_DRAW);
    bufQuad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, bufQuad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW);
    bufMeshPos = gl.createBuffer();
    bufMeshNor = gl.createBuffer();
    bufMeshCol = gl.createBuffer();
    bufPoints = gl.createBuffer();
    bufLine = gl.createBuffer();
  } catch (e) {
    renderFailed = true;
    // eslint-disable-next-line no-console
    console.error('followCamera: disabling after GL init failure (page continues):', e);
  }

  // ── Earth imagery: lazy fetch of the NASA-derived surface map. Any
  // failure (network, decode, upload) latches 'failed' and the simplified
  // graticule globe stays — the scene never breaks over a texture. ──
  if (gl && !renderFailed && typeof Image !== 'undefined') {
    const img = new Image();
    img.onload = () => {
      if (disposed || !gl || renderFailed) return;
      try {
        earthTex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, earthTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT); // lon seam at ±180°
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.bindTexture(gl.TEXTURE_2D, null);
        earthTexState = 'ready';
        drawFrame();
      } catch (e) {
        earthTexState = 'failed';
        // eslint-disable-next-line no-console
        console.error('followCamera: earth texture upload failed (simplified globe stays):', e);
      }
    };
    img.onerror = () => { earthTexState = 'failed'; };
    img.src = EARTH_TEXTURE_URL;
  } else {
    earthTexState = 'failed';
  }

  function resizeBacking(): void {
    // DPR capped at 1.5 — fill-rate control on software GL (perf-harness gate)
    const dpr = Math.min(1.5, (globalThis.devicePixelRatio as number | undefined) || 1);
    const w = canvas.clientWidth || container.clientWidth || 800;
    const h = canvas.clientHeight || container.clientHeight || 600;
    const bw = Math.max(1, Math.round(w * dpr));
    const bh = Math.max(1, Math.round(h * dpr));
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
  }

  function refreshNeighbors(craft: CraftState, nowWall: number): void {
    if (view !== 'onboard' || !opts.getNeighborBuffer) return;
    if (nowWall - lastNeighborRefreshMs < 1000 && pointScene) return; // tick cadence
    lastNeighborRefreshMs = nowWall;
    const buf = opts.getNeighborBuffer();
    const set = neighborDirsScene(buf ?? null, opts.selfIndex ?? -1, craft);
    neighborCount = { drawn: set.count, total: set.total };
    // scene distance from REAL range: scene-units-per-meter anchors on the
    // Earth placement (EARTH_SCENE_DIST ↔ R+h), so Earth's sphere
    // depth-occludes markers physically behind it.
    const scale = EARTH_SCENE_DIST / (EARTH_RADIUS_M + Math.max(0, craft.altMeters));
    const out = new Float32Array(set.count * 3);
    for (let i = 0; i < set.count; i++) {
      const d = Math.min(25_000, Math.max(extent * 4, set.rangesM[i] * scale));
      out[i * 3] = set.dirs[i * 3] * d;
      out[i * 3 + 1] = set.dirs[i * 3 + 1] * d;
      out[i * 3 + 2] = set.dirs[i * 3 + 2] * d;
    }
    pointScene = out;
    if (gl && bufPoints) {
      gl.bindBuffer(gl.ARRAY_BUFFER, bufPoints);
      gl.bufferData(gl.ARRAY_BUFFER, out, gl.DYNAMIC_DRAW);
    }
  }

  function drawFrame(): void {
    if (disposed || renderFailed || !gl) return;
    try {
      drawFrameInner();
    } catch (e) {
      renderFailed = true;
      // eslint-disable-next-line no-console
      console.error('followCamera: disabling after render failure (page continues):', e);
    }
  }

  /** Velocity direction derived from two live samples ≥400ms apart — the
   *  retired scene's derived-never-fabricated rule. */
  function deriveVelocity(st: CraftState, nowWall: number): void {
    if (!prevSample) {
      prevSample = { ...st };
      prevSampleWallMs = nowWall;
      return;
    }
    if (nowWall - prevSampleWallMs <= 400) return;
    const p0 = ecefUnit(prevSample.latDeg, prevSample.lonDeg);
    const p1 = ecefUnit(st.latDeg, st.lonDeg);
    const r0 = EARTH_RADIUS_M + prevSample.altMeters;
    const r1 = EARTH_RADIUS_M + st.altMeters;
    const d: [number, number, number] = [
      p1[0] * r1 - p0[0] * r0, p1[1] * r1 - p0[1] * r0, p1[2] * r1 - p0[2] * r0,
    ];
    prevSample = { ...st };
    prevSampleWallMs = nowWall;
    const len = Math.hypot(d[0], d[1], d[2]);
    if (len < 1) return; // below 1 m — never fabricate a heading from noise
    const enu = ecefToEnu([d[0] / len, d[1] / len, d[2] / len], st.latDeg, st.lonDeg);
    lastVelDir = enuToScene(enu);
  }

  function drawFrameInner(): void {
    const g = gl!;
    resizeBacking();
    g.viewport(0, 0, canvas.width, canvas.height);
    g.clearColor(0, 0, 0, 1);
    g.clearDepth(1);
    g.enable(g.DEPTH_TEST);
    g.depthFunc(g.LEQUAL);
    g.clear(g.COLOR_BUFFER_BIT | g.DEPTH_BUFFER_BIT);

    // LIVE world: fresh state every frame — a null tick holds the last
    // honest state (never guesses), the next valid tick resumes.
    const st = opts.getState() ?? lastState;
    if (!st) return; // nothing honest to draw yet
    lastState = st;
    const nowWall = getTimeMs();
    deriveVelocity(st, nowWall);

    const sunEnu = sunDirLocal(st.timeMs, st.latDeg, st.lonDeg);
    const sunScene = enuToScene(sunEnu);
    const moon = moonDirLocal(st.timeMs, st.latDeg, st.lonDeg);
    const moonScene = enuToScene(moon.dir);
    const earthAng = earthAngularRadiusRad(st.altMeters);
    const shadow = earthShadow(st.altMeters, sunEnu);
    // earthshine: the Earth face below reflects only while it is sunlit
    const fillK = Math.max(0, Math.min(1, sunScene[1]));
    lastLighting = { sunEnu, shadow, earthshine: fillK };
    const sceneToEcef = sceneToEcefMat3(st.latDeg, st.lonDeg);

    const aspect = canvas.width / Math.max(1, canvas.height);
    const isOrbit = view === 'orbit';
    const eye: [number, number, number] = isOrbit ? camEye(cam) : [0, extent * 0.9, 0];
    const fovRad = (isOrbit ? 50 : look.fovDeg) * DEG;
    const proj = mat4Perspective(fovRad, aspect, 0.05, 30000);
    const target: [number, number, number] = isOrbit
      ? [0, 0, 0]
      : (() => {
          const d = lookDirScene(look);
          return [eye[0] + d[0], eye[1] + d[1], eye[2] + d[2]] as [number, number, number];
        })();
    const viewM = mat4LookAt(eye, target);
    const mvp = mat4Multiply(proj, viewM);

    refreshNeighbors(st, nowWall);

    // ── EARTH: nadir direction, true angular size from LIVE altitude ──
    {
      const p = earthProg!;
      g.useProgram(p.prog);
      g.uniformMatrix4fv(p.unis.u_mvp, false, mvp);
      g.uniform3f(p.unis.u_center, 0, -EARTH_SCENE_DIST, 0);
      g.uniform1f(p.unis.u_radius, EARTH_SCENE_DIST * Math.sin(earthAng));
      g.uniform3f(p.unis.u_sun, sunScene[0], sunScene[1], sunScene[2]);
      g.uniform3f(p.unis.u_camPos, eye[0], eye[1], eye[2]);
      g.uniformMatrix3fv(p.unis.u_sceneToEcef, false, sceneToEcef);
      const texOn = earthTexState === 'ready' && earthTex != null;
      g.activeTexture(g.TEXTURE0);
      g.bindTexture(g.TEXTURE_2D, texOn ? earthTex : null);
      g.uniform1i(p.unis.u_tex, 0);
      g.uniform1f(p.unis.u_hasTex, texOn ? 1 : 0);
      g.bindBuffer(g.ARRAY_BUFFER, bufSphere);
      g.enableVertexAttribArray(p.attrs.a_pos);
      g.vertexAttribPointer(p.attrs.a_pos, 3, g.FLOAT, false, 0, 0);
      g.enable(g.CULL_FACE);
      g.cullFace(g.BACK);
      g.drawArrays(g.TRIANGLES, 0, sphereVerts);
      g.disable(g.CULL_FACE);
      g.disableVertexAttribArray(p.attrs.a_pos);
    }

    // ── CRAFT mesh at the origin (orbit view only — onboard stands ON it) ──
    if (isOrbit && mesh && mesh.vertexCount > 0) {
      const p = meshProg!;
      g.useProgram(p.prog);
      g.uniformMatrix4fv(p.unis.u_mvp, false, mvp);
      g.uniform3f(p.unis.u_sun, sunScene[0], sunScene[1], sunScene[2]);
      g.uniform1f(p.unis.u_shadow, shadow.factor);
      g.uniform1f(p.unis.u_fillK, fillK);
      const bind = (buf: WebGLBuffer | null, loc: number, data: Float32Array): void => {
        if (!buf || loc < 0) return;
        g.bindBuffer(g.ARRAY_BUFFER, buf);
        if (meshDirty) g.bufferData(g.ARRAY_BUFFER, data, g.STATIC_DRAW);
        g.enableVertexAttribArray(loc);
        g.vertexAttribPointer(loc, 3, g.FLOAT, false, 0, 0);
      };
      bind(bufMeshPos, p.attrs.a_pos, mesh.positions);
      bind(bufMeshNor, p.attrs.a_normal, mesh.normals);
      bind(bufMeshCol, p.attrs.a_color, mesh.colors);
      meshDirty = false;
      g.enable(g.CULL_FACE);
      g.drawArrays(g.TRIANGLES, 0, mesh.vertexCount);
      g.disable(g.CULL_FACE);
      for (const a of [p.attrs.a_pos, p.attrs.a_normal, p.attrs.a_color]) {
        if (a >= 0) g.disableVertexAttribArray(a);
      }
    }

    // ── velocity indicator (orbit view): derived from live samples ──
    if (isOrbit && lastVelDir && bufLine) {
      const p = lineProg!;
      const L = extent * 2.6;
      g.useProgram(p.prog);
      g.uniformMatrix4fv(p.unis.u_mvp, false, mvp);
      g.uniform4f(p.unis.u_color, 0.55, 0.75, 0.95, 0.5);
      g.bindBuffer(g.ARRAY_BUFFER, bufLine);
      g.bufferData(g.ARRAY_BUFFER, new Float32Array([
        -lastVelDir[0] * L, -lastVelDir[1] * L, -lastVelDir[2] * L,
        lastVelDir[0] * L, lastVelDir[1] * L, lastVelDir[2] * L,
      ]), g.DYNAMIC_DRAW);
      g.enableVertexAttribArray(p.attrs.a_pos);
      g.vertexAttribPointer(p.attrs.a_pos, 3, g.FLOAT, false, 0, 0);
      g.enable(g.BLEND);
      g.blendFunc(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA);
      g.drawArrays(g.LINES, 0, 2);
      g.disable(g.BLEND);
      g.disableVertexAttribArray(p.attrs.a_pos);
    }

    // ── onboard: other catalogued satellites as markers (live buffer) ──
    if (!isOrbit && pointScene && pointScene.length > 0 && bufPoints) {
      const p = pointProg!;
      g.useProgram(p.prog);
      g.uniformMatrix4fv(p.unis.u_mvp, false, mvp);
      g.uniform4f(p.unis.u_color, 0.75, 0.85, 1.0, 0.9);
      g.bindBuffer(g.ARRAY_BUFFER, bufPoints);
      g.enableVertexAttribArray(p.attrs.a_pos);
      g.vertexAttribPointer(p.attrs.a_pos, 3, g.FLOAT, false, 0, 0);
      g.enable(g.BLEND);
      g.blendFunc(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA);
      g.drawArrays(g.POINTS, 0, pointScene.length / 3);
      g.disable(g.BLEND);
      g.disableVertexAttribArray(p.attrs.a_pos);
    }

    // ── MOON then SUN discs (sun farther: real syzygy occludes) ──
    const drawDisc = (
      dir: [number, number, number], dist: number, angularRadius: number, mode: 0 | 1,
    ): void => {
      const p = discProg!;
      const ref: [number, number, number] = Math.abs(dir[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
      let rx = dir[1] * ref[2] - dir[2] * ref[1];
      let ry = dir[2] * ref[0] - dir[0] * ref[2];
      let rz = dir[0] * ref[1] - dir[1] * ref[0];
      const rl = Math.hypot(rx, ry, rz) || 1;
      rx /= rl; ry /= rl; rz /= rl;
      const ux = ry * dir[2] - rz * dir[1];
      const uy = rz * dir[0] - rx * dir[2];
      const uz = rx * dir[1] - ry * dir[0];
      g.useProgram(p.prog);
      g.uniformMatrix4fv(p.unis.u_mvp, false, mvp);
      g.uniform3f(p.unis.u_centerPos, dir[0] * dist, dir[1] * dist, dir[2] * dist);
      g.uniform3f(p.unis.u_right, rx, ry, rz);
      g.uniform3f(p.unis.u_up2, ux, uy, uz);
      g.uniform1f(p.unis.u_radius, dist * Math.tan(angularRadius));
      g.uniform1i(p.unis.u_mode, mode);
      g.uniform3f(p.unis.u_facing, -dir[0], -dir[1], -dir[2]);
      g.uniform3f(p.unis.u_sun, sunScene[0], sunScene[1], sunScene[2]);
      g.bindBuffer(g.ARRAY_BUFFER, bufQuad);
      g.enableVertexAttribArray(p.attrs.a_uv);
      g.vertexAttribPointer(p.attrs.a_uv, 2, g.FLOAT, false, 0, 0);
      if (mode === 0) {
        g.enable(g.BLEND);
        g.blendFunc(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA);
        g.depthMask(false);
      }
      g.drawArrays(g.TRIANGLES, 0, 6);
      if (mode === 0) {
        g.depthMask(true);
        g.disable(g.BLEND);
      }
      g.disableVertexAttribArray(p.attrs.a_uv);
    };
    drawDisc(moonScene, MOON_SCENE_DIST, moon.angularRadiusRad, 1);
    drawDisc(sunScene, SUN_SCENE_DIST, SUN_ANGULAR_RADIUS_RAD, 0);
  }

  // ── interaction: drag orbits/free-looks, wheel/pinch dollies/zooms ──
  const pointers = new Map<number, { x: number; y: number }>();
  let pinchDist = 0;

  const onPointerDown = (e: PointerEvent): void => {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 1) {
      cam.dragging = true;
      cam.velYaw = 0;
      cam.velPitch = 0;
      canvas.style.cursor = 'grabbing';
    } else if (pointers.size === 2) {
      const [a, b] = Array.from(pointers.values());
      pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
    }
    try { canvas.setPointerCapture(e.pointerId); } catch { /* older engines */ }
  };
  const onPointerMove = (e: PointerEvent): void => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      const [a, b] = Array.from(pointers.values());
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchDist > 0 && d > 0) {
        if (view === 'orbit') camApplyDolly(cam, pinchDist / d);
        else lookApplyZoom(look, pinchDist / d);
      }
      pinchDist = d;
      return;
    }
    if (!cam.dragging) return;
    if (view === 'orbit') camApplyDrag(cam, e.clientX - prev.x, e.clientY - prev.y);
    else lookApplyDrag(look, e.clientX - prev.x, e.clientY - prev.y);
  };
  const onPointerUp = (e: PointerEvent): void => {
    pointers.delete(e.pointerId);
    if (pointers.size === 0) {
      cam.dragging = false;
      canvas.style.cursor = 'grab';
    }
    pinchDist = 0;
  };
  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    e.stopPropagation(); // never leak into the map's zoom-seam listeners
    const f = Math.pow(1.0015, e.deltaY);
    if (view === 'orbit') camApplyDolly(cam, f);
    else lookApplyZoom(look, f);
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });

  let ro: ResizeObserver | null = null;
  if (typeof ResizeObserver !== 'undefined') {
    ro = new ResizeObserver(() => drawFrame());
    ro.observe(container);
  }

  // ── frame loop: continuous while mounted — the WORLD moves (live craft,
  // advancing terminator), bounded by dispose(). No RAF (tests/SSR) →
  // parent-driven handle.render() only. Frame CPU time sampled for the
  // perf readout (median over a rolling window). ──
  let rafId = 0;
  let lastFrameMs = 0;
  const hasRaf = typeof requestAnimationFrame !== 'undefined';
  const loop = (t: number): void => {
    if (disposed || renderFailed) return;
    const dt = lastFrameMs > 0 ? t - lastFrameMs : 16;
    lastFrameMs = t;
    const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    camStep(cam, dt);
    lookStep(look, dt);
    drawFrame();
    const t1 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    frameTimes.push(t1 - t0);
    if (frameTimes.length > 240) frameTimes.splice(0, frameTimes.length - 240);
    rafId = requestAnimationFrame(loop);
  };
  if (hasRaf && !renderFailed) rafId = requestAnimationFrame(loop);

  drawFrame();

  return {
    setView(v: InspectView): void {
      if (v === view) return;
      view = v;
      pointScene = null;             // onboard re-derives from the live buffer
      lastNeighborRefreshMs = 0;
      drawFrame();
    },
    getView: () => view,
    setMesh(m: Mesh | null, label: string): void {
      mesh = m;
      meshLabel = label;
      meshDirty = true;
      extent = meshExtent(m);
      const b = camDollyBounds(extent);
      cam.minDist = b.min;
      cam.maxDist = b.max;
      cam.dist = Math.max(b.min, Math.min(b.max, cam.dist));
      cam.distTarget = Math.max(b.min, Math.min(b.max, cam.distTarget));
      drawFrame();
    },
    getMeshLabel: () => meshLabel,
    getProvenance: () => (view === 'orbit' ? ORBIT_VIEW_PROVENANCE : ONBOARD_VIEW_PROVENANCE),
    render(): void {
      camStep(cam, 16);
      lookStep(look, 16);
      drawFrame();
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      if (hasRaf && rafId) cancelAnimationFrame(rafId);
      try { ro?.disconnect(); } catch { /* already gone */ }
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerUp);
      canvas.removeEventListener('wheel', onWheel);
      if (gl) {
        try {
          for (const b of [bufMeshPos, bufMeshNor, bufMeshCol, bufSphere, bufQuad, bufPoints, bufLine]) {
            if (b) gl.deleteBuffer(b);
          }
          if (earthTex) gl.deleteTexture(earthTex);
          for (const p of [meshProg, earthProg, discProg, pointProg, lineProg]) {
            if (p) gl.deleteProgram(p.prog);
          }
        } catch { /* context already lost */ }
      }
      try { canvas.remove(); } catch { /* detached */ }
    },
    getRenderFailed: () => renderFailed,
    getCamera: () => ({ yaw: cam.yaw, pitch: cam.pitch, dist: cam.dist }),
    getLook: () => ({ azDeg: look.azDeg, elDeg: look.elDeg, fovDeg: look.fovDeg }),
    getEarthTexture: () => earthTexState,
    setLook(az: number, el: number): void {
      look.azDeg = ((az % 360) + 360) % 360;
      look.elDeg = Math.max(-LOOK_EL_LIMIT_DEG, Math.min(LOOK_EL_LIMIT_DEG, el));
      drawFrame();
    },
    getLighting: () => lastLighting,
    getStateNow: () => lastState,
    getNeighborCount: () => neighborCount,
    getFrameStats: () => {
      if (!frameTimes.length) return { medianMs: 0, samples: 0 };
      const s = [...frameTimes].sort((a, b) => a - b);
      return { medianMs: s[Math.floor(s.length / 2)], samples: s.length };
    },
  };
}
