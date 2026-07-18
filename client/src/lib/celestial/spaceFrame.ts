// THE CONTINUOUS SPACE FRAME — one camera from the map's zoom floor to
// Neptune and back (human-approved design 2026-07-18, third iteration of the
// "no separate scenes" directive: "keep zooming out and the real map globe —
// with whatever layers you have on, terminator and all — shrinks into space
// in one continuous motion; zoom back in and you're seamlessly back on the
// map. Standing near the Moon, Earth is right there behind you — the live
// one."). Celestial v2 B1 (directive §1, 2026-07-18): there is NO entry
// button of any kind — the seam is reached only by zoom inputs themselves
// (wheel, pinch, the map's +/- buttons, keyboard +/-), all riding the same
// exponential curve (see zoomSeam.ts).
//
// WHAT THIS REPLACES: lib/celestial/solarView.ts (the separate top-down
// 2D-ortho solar-system scene) is RETIRED — same precedent as inspectScene.
// There is no scene swap any more: the LIVE MAPLIBRE CANVAS itself becomes
// the Earth in a 3D star-space frame. This module renders everything EXCEPT
// Earth (space, Sun, Moon, planets, markers, chrome) on a canvas placed
// UNDER the map canvas (which is transparent around the globe disc in globe
// projection — verified empirically), and each frame hands the parent an
// EarthAnchor {translate, scale, roll, opacity} that the parent applies as a
// CSS transform to the maplibre canvas. The map keeps rendering — layers,
// terminator and all; that IS the Earth the user sees — until its disc is
// small enough that a crossfade to a live-shaded impostor (real terminator
// from the real Sun vector) takes over. On approach the crossfade reverses
// and at the seam (CSS scale = 1) the camera is handed back to MapLibre at
// the zoom floor: the exact reverse of entry, no chip, no flash.
//
// USER-CONTROLLED SCALE (B2, directive §2, 2026-07-18 — supersedes v1's
// fixed true scale): every body's TRUE position still comes from the real
// ephemeris (lib/celestial/solarSystem.ts — Schlyter/van Flandern, arcmin-
// class), but the RENDER LAYOUT flows through lib/celestial/scaleModel.ts:
// a distance-compression slider c (0 = true 1:1, 1 = Mercury→Neptune in one
// view) and a body-size multiplier s (1–5000×, Sun capped, the map-anchor
// body exempt — scaleModel's header carries the full mapping rationale).
// THE RULE: the LAYOUT may compress; the NUMBERS never lie — labels, the
// status line and the scale bar always print TRUE ephemeris distances
// (dual-space camera: one (focus, dir, dist) pose evaluated against both
// the compressed and the true positions). At the TRUE SCALE preset
// (c=0, s=1) the mapping is the exact identity and this frame renders
// bit-identically to the B1 true-scale frame; satellites (the Moon) ride
// their parent at the TRUE local offset at every c, so the physical payoff
// stays free: from lunar distance Earth is a ~1.9° disc — the live one.
// Exponential zoom (each wheel tick MULTIPLIES camera distance) is what
// makes 12 orders of magnitude traversable.
//
// FLOATING ORIGIN (B1 precision audit, 2026-07-18 — directive §1 "camera-
// relative rendering with double-precision origin offsets on CPU, single-
// precision only for camera-relative GPU coords"): every world position,
// camera position, and difference is computed in JS DOUBLES on the CPU, and
// the only values that ever leave this module are CAMERA-RELATIVE — screen
// pixels handed to Canvas2D (projectPoint subtracts camPos in f64 FIRST:
// rel = world − camPos, then projects rel) and the CSS px of the Earth
// anchor. No absolute heliocentric meters ever reach a float32 path: there
// is no WebGL here, no f32 attribute/uniform upload, and Canvas2D receives
// only viewport-scale numbers. Residual f64 rounding at Neptune range
// (~4.5e12 m, mantissa resolution ~0.5 m) is ~1e-8 of the closest camera
// distance — orders of magnitude below a pixel; the jitter test pins this
// (sub-pixel projection stability at 30 AU under micro camera moves).
//
// HONESTY RAILS
// - Bodies smaller than MARKER_MAX_DISC_PX RENDERED on screen carry a
//   labeled marker (reticle + name + TRUE camera distance) — styled as a
//   marker, never a fake disc. Nothing is ever invisible; enlargement only
//   ever happens through the user's own size slider, and whenever the
//   layout is not true 1:1 the caption says so in amber, verbatim from the
//   directive: "distances/sizes compressed for visibility — labels always
//   show real values".
// - No starfield: a decorative random sky would violate "real position or
//   absent" (the retired solarView made the same choice; a real bright-star
//   catalog remains future work). Black is honest.
// - Body shading is geometric: a Lambert-lit sphere under the real Sun
//   direction, so the Moon's phase (lit fraction AND terminator orientation)
//   falls out of the ephemeris rather than being painted on.
// - Display colors are presentation (approximate naked-eye appearance),
//   never data. The persistent caption states scale, provenance and the
//   marker rule.
//
// KNOWN LIMITS (stated, not hidden): occlusion between the live-map Earth
// and a body passing IN FRONT of it (e.g. the Moon transiting Earth's disc
// as seen from behind the Moon) paints the map on top — the map canvas
// always composites above this frame; bodies behind Earth are correctly
// hidden by the opaque globe disc. Off-screen bodies get no edge pointers
// (click-to-fly + markers make everything reachable; scope decision).
//
// B3 — ORBITS / ROTATION / MOONS / TIME (directive §3, 2026-07-18): the
// registry gains the eight curated moons (Io, Europa, Ganymede, Callisto,
// Titan, Triton, Phobos, Deimos — JPL mean elements, moons.ts) riding their
// parents through compression; every body exposes its IAU rotation state
// (axisEcl + prime meridian W — rotation.ts, pck00011/WGCCRE 2015; the Moon
// tidally locked by the real constants) in getState() for B4/B5 surfaces to
// render from; orbit-ellipse polylines are sampled ONCE per body from the
// real ephemeris (one body per macrotask, cached, re-laid-out — never
// resampled — on scale changes) and drawn under the bodies; and the frame's
// time is fed by the ONE simulation clock (simClock.ts) via setTime — at 1×
// real time everything renders exactly as before. A satellite projecting
// inside its parent's drawn footprint folds into the parent's label as
// "+N moons" (nothing invisible; zooming resolves them back out).
//
// FUTURE-PROOF BODY REGISTRY — bodies are DECLARED, not hardcoded. Each body
// is a SpaceBodyDef {id, name, radiusKm, ephemeris(timeMs)→meters, color,
// emissive?, mapAnchor, parentId?, orbitPeriodDays?} and the frame renders
// whatever registry it is given (defaultBodyRegistry() = Sun/Moon/Earth +
// the 8 planets from solarSystem.ts + the curated moons from moons.ts).
// mapAnchor names which live map surface anchors at the
// body: "maplibre" (Earth today — THE map canvas, posed by the parent via
// applyEarthAnchor) or null (shaded true-scale sphere only). Because space
// is true-scale, a later Moon tile pyramid becomes a second anchor by
// declaration alone — no camera, projection, or seam change; the anchor
// machinery (pose + crossfade + seam) already runs off the registry entry,
// not off "Earth". The camera's up-reference stays Earth's spin axis (north
// up) for now; a per-anchor axis is a registry field away when a non-Earth
// anchor ships.
//
// Interop: mountSpaceFrame(container, opts) → handle (setTime/render/
// flyTo/flyHome/nudgeZoom/getState/dispose) — the celestialSky
// mount-handle idiom. All view math is exported pure for tests.

import {
  solarSystemState,
  BODY_RADIUS_M,
  BODY_ORDER,
  AU_M,
  type BodyId,
} from "./solarSystem.js";
import { gmstDeg } from "./ephemeris.js";
import { fmtKm, getUnits, type UnitSystem } from "../units.js";
import { ZOOM_STEP_PER_NOTCH, zoomStepFactor } from "./zoomSeam.js";
import {
  applyDistanceCompression,
  renderedDiscPx,
  clampScaleState,
  getCelestialScale,
  isTrueScale,
  SUN_SIZE_MULT_CAP,
  SIZE_REL_EARTH_KM,
  type ScaleState,
  type ScaleBodyIn,
} from "./scaleModel.js";
// B3 (directive §3, 2026-07-18): curated moons as real ephemeris bodies
// (JPL mean elements — moons.ts header carries the citations), IAU axial
// tilt + true rotation states (rotation.ts — pck00011/WGCCRE 2015), and
// precomputed orbit-ellipse polylines that flow through the same layout
// compression as body positions (orbitPath.ts).
import {
  MOON_IDS,
  MOONS,
  MOON_COLOR,
  MOON_NAME,
  moonLocalOffsetEclM,
  moonOrbitPeriodDays,
  type MoonId,
} from "./moons.js";
import {
  hasRotationModel,
  axisEclOfDate,
  primeMeridianDeg as iauPrimeMeridianDeg,
} from "./rotation.js";
import {
  sampleOrbitPolyline,
  layoutOrbitPolyline,
  orbitPolylineStale,
  getOrbitPathsPref,
  ORBIT_SAMPLES_PLANET,
  ORBIT_SAMPLES_MOON,
  type OrbitPolyline,
} from "./orbitPath.js";

// B2 scale system: the user-controlled layout mapping is re-exported so the
// frame's public surface keeps one name per contract (the zoomSeam pattern).
export {
  SCALE_PRESET_TRUE,
  SCALE_PRESET_VISIBLE,
  SIZE_MULT_MIN,
  SIZE_MULT_MAX,
  SUN_SIZE_MULT_CAP,
  SIZE_APPARENT_CAP_PX,
  compressDistance,
  renderedDiscPx,
  isTrueScale,
  type ScaleState,
} from "./scaleModel.js";

// the seam's zoom math is shared with datamap's input wiring (see
// zoomSeam.ts header) — re-exported so this module's public surface and
// its tests keep one name per contract.
export { ZOOM_STEP_PER_NOTCH, zoomStepFactor, wheelDeltaForFactor, ZOOM_BUTTON_DELTAY } from "./zoomSeam.js";

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

// ── tunables (exported so tests pin the contract) ───────────────────────────

/** Vertical field of view, degrees — matches the map camera's default fov so
 *  angular sizes are continuous across the seam. */
export const DEFAULT_FOV_DEG = 36.87;

/** A body whose true disc is smaller than this carries a labeled marker. */
export const MARKER_MAX_DISC_PX = 2;

/** Live-map ↔ impostor crossfade band, in Earth-disc CSS px. Above HI the
 *  live map is fully opaque (from lunar distance Earth is ~45px at 900px
 *  viewport height — deliberately above HI, so the Earth you see from the
 *  Moon is the live one). Below LO — where the map would be a sub-14px
 *  blob with no legible content — the impostor carries the disc alone. */
export const MAP_FADE_HI_PX = 24;
export const MAP_FADE_LO_PX = 14;

/** Fly-to arrival framing: the target's disc spans this fraction of the
 *  viewport's SHORT side (aspect-safe — phones frame as well as 1440), and
 *  the closest approach zoom-in allows, in body radii. At 1440×900 the
 *  framing lands ≈4.5 body radii out. */
export const FRAME_DISC_FRACTION = 0.68;
export const MIN_DISTANCE_RADII = 3.2;

/** Arrivals swing PAST the target and look back the way they came, offset
 *  sideways by this angle — so the place you left (Earth, when flying out)
 *  hangs beside the target instead of hiding behind your back or behind the
 *  target's disc. 15° sits inside the horizontal half-fov at 16:9. */
export const ARRIVAL_LOOKBACK_OFFSET_DEG = 15;

/** Hard camera-distance ceiling, meters — comfortably past Neptune (~30 AU);
 *  and the floor is per-body (MIN_DISTANCE_RADII). */
export const MAX_CAMERA_DISTANCE_M = 7e12;

/** One label row's height, px — vertical collision distance for stacking. */
export const LABEL_COLLIDE_PX = 14;

/** One label row's nominal text width, px — horizontal collision distance.
 *  (The retired solarView stacked on POINT distance only; a whole-system
 *  cluster puts anchors 15-25px apart with ~130px of text overprinting —
 *  drive round 1 showed exactly that, so the stacker is box-aware now.) */
export const LABEL_BOX_W_PX = 130;

/** Bodies drawn per-pixel are shaded at most at this sprite radius and
 *  upscaled beyond it (smooth sphere — visually lossless, 4x+ cheaper). */
export const SPRITE_MAX_SHADE_RADIUS = 180;

// ── pure view math ──────────────────────────────────────────────────────────

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

const v3 = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const sub = (a: Vec3, b: Vec3): Vec3 => v3(a.x - b.x, a.y - b.y, a.z - b.z);
const add = (a: Vec3, b: Vec3): Vec3 => v3(a.x + b.x, a.y + b.y, a.z + b.z);
const scale3 = (a: Vec3, s: number): Vec3 => v3(a.x * s, a.y * s, a.z * s);
const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: Vec3, b: Vec3): Vec3 =>
  v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
const len3 = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);
const norm3 = (a: Vec3): Vec3 => {
  const l = len3(a) || 1;
  return v3(a.x / l, a.y / l, a.z / l);
};

/** Perspective scale k: px per unit tan(angle) — (h/2)/tan(fov/2). */
export function perspectiveScalePx(fovDeg: number, viewportHPx: number): number {
  return viewportHPx / 2 / Math.tan((fovDeg / 2) * DEG);
}

/** TRUE on-screen disc diameter of a sphere, px: 2·k·tan(asin(R/d)).
 *  Exact spherical form (the visible limb, not the parallel projection). */
export function bodyDiscPx(radiusM: number, distM: number, k: number): number {
  if (!(distM > radiusM)) return Number.POSITIVE_INFINITY; // inside the body
  return 2 * k * Math.tan(Math.asin(radiusM / distM));
}

/** Inverse of bodyDiscPx: the camera distance at which a sphere's true disc
 *  spans discPx. Used for the entry seam and fly-to framing. */
export function distanceForDiscPx(radiusM: number, discPx: number, k: number): number {
  const theta = Math.atan(discPx / (2 * k));
  return radiusM / Math.sin(Math.max(1e-12, theta));
}

/** MapLibre globe apparent disc, CSS px: worldSize/π/cos(centerLat)
 *  (verified against maplibre-gl 5.24 — same formula the retired handoff
 *  used, and measured ≈51px at zoom −2 / lat 37.5). */
export function mapGlobeDiscPx(zoom: number, centerLatDeg: number): number {
  const worldSize = 512 * Math.pow(2, zoom);
  return worldSize / Math.PI / Math.cos(centerLatDeg * DEG);
}

/** Live-map opacity from Earth's true disc px — 1 at/above HI, 0 at/below
 *  LO, smoothstep between (the impostor always sits underneath, so total
 *  Earth visibility never dips). */
export function mapAnchorOpacity(discPx: number): number {
  if (discPx >= MAP_FADE_HI_PX) return 1;
  if (discPx <= MAP_FADE_LO_PX) return 0;
  const t = (discPx - MAP_FADE_LO_PX) / (MAP_FADE_HI_PX - MAP_FADE_LO_PX);
  return t * t * (3 - 2 * t);
}

/** Marker rule: sub-MARKER_MAX_DISC_PX bodies are flagged, never inflated. */
export function markerNeeded(discPx: number): boolean {
  return discPx < MARKER_MAX_DISC_PX;
}

/** Illuminated fraction of a sphere's disc as seen by THIS camera:
 *  (1 + cos α)/2 with α the Sun–body–camera angle. toSun/toCam are unit
 *  vectors FROM the body. From Earth this reproduces the familiar lunar
 *  illumination; near another body it is that viewpoint's real phase. */
export function apparentLitFraction(toSun: Vec3, toCam: Vec3): number {
  return (1 + dot(toSun, toCam)) / 2;
}

/** Mean obliquity of the ecliptic, degrees (Schlyter §5 — same series the
 *  ephemeris uses; d = days since 1999-12-31 00:00 UT). */
export function obliquityDeg(dateMs: number): number {
  const d = (dateMs - Date.UTC(1999, 11, 31)) / 86_400_000;
  return 23.4393 - 3.563e-7 * d;
}

/** Equatorial → ecliptic-of-date frame rotation (about the vernal-equinox
 *  x-axis). Earth's north celestial pole lands at ecliptic (0, sin ε, cos ε)
 *  — i.e. ecliptic lon 90°, lat 90−ε — the textbook check. */
export function eclFromEq(veq: Vec3, dateMs: number): Vec3 {
  const e = obliquityDeg(dateMs) * DEG;
  return v3(
    veq.x,
    Math.cos(e) * veq.y + Math.sin(e) * veq.z,
    -Math.sin(e) * veq.y + Math.cos(e) * veq.z,
  );
}

/** Ecliptic-of-date → equatorial (exact inverse of eclFromEq). */
export function eqFromEcl(vecl: Vec3, dateMs: number): Vec3 {
  const e = obliquityDeg(dateMs) * DEG;
  return v3(
    vecl.x,
    Math.cos(e) * vecl.y - Math.sin(e) * vecl.z,
    Math.sin(e) * vecl.y + Math.cos(e) * vecl.z,
  );
}

/** Earth's spin-axis (north pole) direction in the ecliptic frame. */
export function earthAxisEcl(dateMs: number): Vec3 {
  return eclFromEq(v3(0, 0, 1), dateMs);
}

/**
 * The camera direction (unit, Earth-center → camera, ecliptic frame) that
 * looks straight down at (latDeg, lonDeg) right now — the map's center point
 * at the moment of entry, via the real sidereal angle (ephemeris.ts gmstDeg).
 * This is what makes the seam geometrically true: the hemisphere the map was
 * showing is the hemisphere that faces the space camera, and the terminator
 * painted on the shrinking globe agrees with where the Sun hangs in the
 * frame.
 */
export function entryCameraDir(latDeg: number, lonDeg: number, dateMs: number): Vec3 {
  const raDeg = gmstDeg(dateMs) + lonDeg;
  const dec = latDeg * DEG;
  const ra = raDeg * DEG;
  const veq = v3(Math.cos(dec) * Math.cos(ra), Math.cos(dec) * Math.sin(ra), Math.sin(dec));
  return eclFromEq(veq, dateMs);
}

/** Inverse of entryCameraDir: the sub-camera point on Earth (the lat/lon the
 *  camera is directly above) — drives the live map's hemisphere alignment
 *  while flying (and honestly drifts ~0.25°/min of longitude when hovering
 *  inertially: Earth turns beneath you). */
export function subCameraLatLon(dirEcl: Vec3, dateMs: number): { latDeg: number; lonDeg: number } {
  const veq = eqFromEcl(norm3(dirEcl), dateMs);
  const latDeg = Math.asin(Math.max(-1, Math.min(1, veq.z))) * RAD;
  let lonDeg = Math.atan2(veq.y, veq.x) * RAD - gmstDeg(dateMs);
  lonDeg = ((lonDeg % 360) + 360) % 360;
  if (lonDeg > 180) lonDeg -= 360;
  return { latDeg, lonDeg };
}

/** Spherical interpolation between unit vectors, with an antiparallel
 *  fallback (detour through a perpendicular waypoint — the arc stays on the
 *  sphere and is deterministic). t clamped to [0,1]. */
export function slerpUnit(a: Vec3, b: Vec3, t: number): Vec3 {
  const s = t < 0 ? 0 : t > 1 ? 1 : t;
  const d = Math.max(-1, Math.min(1, dot(a, b)));
  if (d > 0.999999) return norm3(v3(a.x + (b.x - a.x) * s, a.y + (b.y - a.y) * s, a.z + (b.z - a.z) * s));
  if (d < -0.999999) {
    const helper = Math.abs(a.z) < 0.9 ? v3(0, 0, 1) : v3(1, 0, 0);
    const mid = norm3(cross(a, helper));
    return s < 0.5 ? slerpUnit(a, mid, s * 2) : slerpUnit(mid, b, s * 2 - 1);
  }
  const ang = Math.acos(d);
  const sa = Math.sin((1 - s) * ang) / Math.sin(ang);
  const sb = Math.sin(s * ang) / Math.sin(ang);
  return norm3(v3(a.x * sa + b.x * sb, a.y * sa + b.y * sb, a.z * sa + b.z * sb));
}

/** Rotate unit vector v about unit axis by angleDeg (Rodrigues). */
export function rotateAbout(vec: Vec3, axis: Vec3, angleDeg: number): Vec3 {
  const a = angleDeg * DEG;
  const c = Math.cos(a);
  const s = Math.sin(a);
  const k = norm3(axis);
  return add(
    add(scale3(vec, c), scale3(cross(k, vec), s)),
    scale3(k, dot(k, vec) * (1 - c)),
  );
}

export function easeInOutCubic(t: number): number {
  const s = t < 0 ? 0 : t > 1 ? 1 : t;
  return s < 0.5 ? 4 * s * s * s : 1 - Math.pow(-2 * s + 2, 3) / 2;
}

/** Log-space distance interpolation — the only ease that feels uniform
 *  across 6 orders of magnitude (exponential zoom's continuous twin). */
export function expLerp(a: number, b: number, t: number): number {
  return Math.exp(Math.log(a) + (Math.log(b) - Math.log(a)) * t);
}

/** Flight duration from how far (log distance ratio) and how much the view
 *  swings (deg). Clamped so nothing is ever a teleport or a slog. */
export function flyDurationMs(distRatio: number, swingDeg: number): number {
  const r = Math.abs(Math.log10(Math.max(1e-9, distRatio)));
  const ms = 900 + 260 * r + 9 * swingDeg;
  return Math.max(1100, Math.min(3200, ms));
}

export interface CamBasis {
  /** forward (into the screen), right, up — orthonormal, ecliptic frame. */
  f: Vec3;
  r: Vec3;
  u: Vec3;
}

/** Camera basis from dir (unit, target → camera) and an up-reference. The up
 *  axis is built FROM the reference (Earth's spin axis), so north is always
 *  screen-up and the anchor roll is 0 by construction — the map's own
 *  north-up drawing needs no CSS rotation until a free-roll camera exists. */
export function camBasis(dir: Vec3, upRef: Vec3): CamBasis {
  const f = scale3(dir, -1); // camera looks at the target
  let r = cross(f, upRef);
  if (len3(r) < 1e-6) r = cross(f, Math.abs(upRef.x) < 0.9 ? v3(1, 0, 0) : v3(0, 0, 1));
  r = norm3(r);
  const u = norm3(cross(r, f));
  return { f, r, u };
}

/** Screen roll of a world axis under a basis, degrees clockwise from
 *  screen-up. 0 when the axis lies in the (forward, up) plane — which
 *  camBasis guarantees for its own up-reference. */
export function northRollDeg(basis: CamBasis, axis: Vec3): number {
  return Math.atan2(dot(axis, basis.r), dot(axis, basis.u)) * RAD;
}

export interface ProjectedPoint {
  x: number;
  y: number;
  /** depth along the view axis, meters; <= 0 ⇒ behind the camera. */
  depth: number;
}

/**
 * The seam exit (and its scale clamp) is armed only when the camera is
 * actually HEADING for the map: idle at the anchor focus, or on the
 * fly-home flight. A flight to any OTHER body must never eject the user
 * mid-arc, however close the camera swings past the anchor — the phone
 * drive caught exactly that: the 2.2×-lunar → Moon arc passes near enough
 * to Earth that its true disc briefly exceeds the map disc, and the
 * un-gated seam ejected the flight to the map. While unarmed, the anchor
 * scale follows the TRUE disc (the live map swells past you, upscaled —
 * blurry but honest) instead of clamping at the seam size.
 */
export function seamExitArmed(
  flying: boolean,
  exitOnArrival: boolean,
  focusIsAnchor: boolean,
): boolean {
  return flying ? exitOnArrival : focusIsAnchor;
}

/** Perspective projection of a world point into container CSS px. */
export function projectPoint(
  world: Vec3,
  camPos: Vec3,
  basis: CamBasis,
  k: number,
  cxPx: number,
  cyPx: number,
): ProjectedPoint {
  const rel = sub(world, camPos);
  const depth = dot(rel, basis.f);
  if (depth <= 0) return { x: Number.NaN, y: Number.NaN, depth };
  return {
    x: cxPx + (k * dot(rel, basis.r)) / depth,
    y: cyPx - (k * dot(rel, basis.u)) / depth,
    depth,
  };
}

/**
 * Per-label vertical offsets so colliding labels STACK instead of
 * overprinting — markers/bodies never move, only where their text anchors
 * (the retired solarView's tested guarantee, upgraded from point-distance
 * to BOX collision: two label ROWS collide when their anchors are within a
 * row's width horizontally and a row's height vertically). Greedy in input
 * order: each label steps down past every earlier label's occupied row
 * until it finds a free slot, so any cluster resolves to a clean readable
 * stack with no overprint.
 *
 * TERMINATION GUARANTEE (the v1.0.396 freeze, root-caused 2026-07-18): a
 * move must STRICTLY INCREASE y or it is not a move. In doubles, yj +
 * LABEL_COLLIDE_PX can round DOWN so that (yj+14) − yj < 14 — e.g. yj =
 * 507.7892615010763 steps to 521.7892615010762, difference
 * 13.999999999999943 — so the un-guarded loop re-detected the collision it
 * had just resolved, reassigned the identical y, set moved, and spun the
 * `while` forever: one synchronous infinite loop inside draw() = Chrome's
 * "Page Unresponsive" kill dialog. The guard restores the proof: y only
 * ever steps to a strictly larger member of the finite set {yj + 14}, so
 * the loop is bounded at i moves per label regardless of rounding. A label
 * landing on the fixed point stays put — one float-ulp of overlap, invisible.
 */
export function layoutLabelStacks(
  anchors: { x: number; y: number }[],
  boxW = LABEL_BOX_W_PX,
): number[] {
  const offsets = new Array(anchors.length).fill(0);
  for (let i = 0; i < anchors.length; i++) {
    let y = anchors[i].y;
    let moved = true;
    while (moved) {
      moved = false;
      for (let j = 0; j < i; j++) {
        const yj = anchors[j].y + offsets[j];
        if (Math.abs(anchors[i].x - anchors[j].x) < boxW && Math.abs(y - yj) < LABEL_COLLIDE_PX) {
          const ny = yj + LABEL_COLLIDE_PX; // step below the occupied row, re-scan
          if (ny > y) {
            y = ny;
            moved = true;
          }
          // ny <= y ⇒ float fixed point (yj+14 rounded to ≤ y): already at
          // the required clearance to double precision — moving would not
          // advance y, and flagging `moved` here is exactly what froze v1
        }
      }
    }
    offsets[i] = y - anchors[i].y;
  }
  return offsets;
}

/** Distance label: km/mi via the units preference below 0.01 AU, AU above
 *  (AU is a fixed domain convention, like knots for vessels). */
export function fmtSpaceDistance(meters: number): string {
  if (meters < 0.01 * AU_M) return fmtKm(meters / 1000, 0);
  const au = meters / AU_M;
  return `${au.toFixed(au < 10 ? 3 : 2)} AU`;
}

// ── the scale bar (B1 §1: "mi → thousands of mi → AU, respecting the
// existing mi/km units toggle") ─────────────────────────────────────────────

/** Max bar width, px — matches MapLibre ScaleControl's default maxWidth so
 *  the bar reads as the same instrument on both sides of the seam. */
export const SCALE_BAR_MAX_W_PX = 100;

/** AU switch threshold for the BAR, same constant the distance labels use
 *  (0.01 AU ≈ 930,000 mi): below it the bar is mi/km per the units
 *  preference, above it AU (fixed domain convention). */
export const SCALE_BAR_AU_SWITCH_M = 0.01 * AU_M;

const M_PER_MI = 1609.344;

/** Largest "nice" value (1/2/3/5 × 10^n — MapLibre's own series) ≤ target.
 *  Exported for the test's exhaustive sweep. */
export function niceScaleFloor(target: number): number {
  if (!(target > 0) || !Number.isFinite(target)) return 1;
  const exp = Math.floor(Math.log10(target));
  const base = Math.pow(10, exp);
  for (const m of [5, 3, 2, 1]) if (m * base <= target) return m * base;
  return base; // unreachable (1×base ≤ target by construction), kept safe
}

/**
 * The space frame's scale bar: given meters-per-CSS-px at the focus plane,
 * pick a round distance whose bar fits in SCALE_BAR_MAX_W_PX and label it in
 * the current unit system (mi/km below the AU switch, AU above — the same
 * ladder as fmtSpaceDistance, so bar and labels never disagree on units).
 *
 * SEAM CONTINUITY (why metersPerPx = (dist − R)/k is the right input): the
 * map's own ScaleControl shows ground meters/px at the view center, which
 * for MapLibre equals cameraAltitude / cameraToCenterPx — and this frame's
 * perspective k IS cameraToCenterPx (same fov, same viewport), while
 * (dist − R) IS the camera altitude over the focus surface. At the seam the
 * two bars therefore show the same number by construction.
 */
export function spaceScaleBar(
  metersPerPx: number,
  system: UnitSystem = getUnits(),
  maxWidthPx: number = SCALE_BAR_MAX_W_PX,
): { widthPx: number; label: string } {
  const mpp = Math.max(1e-9, metersPerPx);
  const maxM = mpp * maxWidthPx;
  let unitM: number;
  let suffix: string;
  if (maxM >= SCALE_BAR_AU_SWITCH_M) {
    unitM = AU_M;
    suffix = " AU";
  } else if (system === "imperial") {
    unitM = M_PER_MI;
    suffix = " mi";
  } else {
    unitM = 1000;
    suffix = " km";
  }
  const nice = niceScaleFloor(maxM / unitM);
  const label =
    nice >= 1000
      ? `${nice.toLocaleString("en-US")}${suffix}`
      : `${Number(nice.toFixed(3))}${suffix}`; // 0.02 AU stays "0.02", never "0.020"
  return { widthPx: (nice * unitM) / mpp, label };
}

// ── the body registry ───────────────────────────────────────────────────────

/**
 * A body the frame renders. Everything the frame needs is declared here —
 * adding a body (or giving one a live map surface) is a registry entry,
 * never an architecture change.
 */
export interface SpaceBodyDef {
  id: string;
  name: string;
  /** true physical radius, km (the drawn size IS this, always). */
  radiusKm: number;
  /** heliocentric ecliptic-of-date position, METERS, at timeMs. Real
   *  ephemeris or nothing — a body with no computable position does not
   *  belong in the registry. */
  ephemeris: (timeMs: number) => Vec3;
  /** approximate naked-eye display color — presentation only, not data. */
  color: string;
  /** self-luminous: rendered as a glow (no phase) and used as the frame's
   *  light source. Exactly one emissive body per registry (the Sun). */
  emissive?: boolean;
  /** which live map surface anchors at this body: "maplibre" = the parent's
   *  map canvas (Earth today); null = shaded true-scale sphere only. A
   *  future Moon tile pyramid becomes an anchor here, nowhere else. */
  mapAnchor: "maplibre" | null;
  /** B2: satellites name their primary here and RIDE it through distance
   *  compression at the TRUE local offset (scaleModel.ts header states
   *  why); absent ⇒ the body's heliocentric radius is what compresses. */
  parentId?: string | null;
  /** B3: one full orbit-line period, days (planets: sidereal year; moons:
   *  the closed-ellipse anomalistic period). Absent/null ⇒ no orbit line
   *  (the Sun). */
  orbitPeriodDays?: number | null;
}

/** Approximate naked-eye display colors — presentation only, not data. */
const BODY_COLOR: Record<BodyId, string> = {
  sun: "#ffd76e",
  mercury: "#b5a9a0",
  venus: "#e6d5a8",
  earth: "#5b8fd9",
  moon: "#c9c9c9",
  mars: "#d97b5b",
  jupiter: "#d9b08c",
  saturn: "#e0cba8",
  uranus: "#9fd4d9",
  neptune: "#6e8fd9",
};

const NAMES: Record<BodyId, string> = {
  sun: "Sun", mercury: "Mercury", venus: "Venus", earth: "Earth",
  moon: "Moon", mars: "Mars", jupiter: "Jupiter", saturn: "Saturn",
  uranus: "Uranus", neptune: "Neptune",
};

/** Sidereal orbital periods, days (NSSDC planetary fact sheets) — the
 *  B3 orbit-line sampling windows. The Moon's is the sidereal month (its
 *  drawn line is one month of the real perturbed geocentric path). */
export const PLANET_ORBIT_PERIOD_DAYS: Record<Exclude<BodyId, "sun">, number> = {
  mercury: 87.969,
  venus: 224.701,
  earth: 365.256,
  moon: 27.3217,
  mars: 686.980,
  jupiter: 4332.589,
  saturn: 10759.22,
  uranus: 30685.4,
  neptune: 60189.0,
};

/**
 * The default registry: Sun, Moon, Earth and the 8 planets from the
 * solarSystem.ts ephemeris (Schlyter/van Flandern, arcmin-class), plus —
 * since B3 — the curated moons (Io, Europa, Ganymede, Callisto, Titan,
 * Triton, Phobos, Deimos) at their JPL mean-element positions riding
 * their parents (moons.ts carries the citations). All solar-system
 * ephemeris fns share one single-instant memo — solarSystemState computes
 * the whole system at once, and flights evaluate every frame; each moon
 * adds one cheap local Kepler solve on top of the shared memo.
 */
export function defaultBodyRegistry(): SpaceBodyDef[] {
  let memo: { t: number; pos: Record<string, Vec3> } | null = null;
  const at = (timeMs: number): Record<string, Vec3> => {
    if (!memo || memo.t !== timeMs) {
      const pos: Record<string, Vec3> = {};
      for (const b of solarSystemState(timeMs)) {
        pos[b.id] = v3(b.helioAu.x * AU_M, b.helioAu.y * AU_M, b.helioAu.z * AU_M);
      }
      memo = { t: timeMs, pos };
    }
    return memo.pos;
  };
  const core: SpaceBodyDef[] = BODY_ORDER.map((id) => ({
    id,
    name: NAMES[id],
    radiusKm: BODY_RADIUS_M[id] / 1000,
    ephemeris: (timeMs: number) => at(timeMs)[id],
    color: BODY_COLOR[id],
    emissive: id === "sun" ? true : undefined,
    mapAnchor: id === "earth" ? ("maplibre" as const) : null,
    parentId: id === "moon" ? "earth" : null,
    orbitPeriodDays: id === "sun" ? null : PLANET_ORBIT_PERIOD_DAYS[id],
  }));
  const moons: SpaceBodyDef[] = MOON_IDS.map((id: MoonId) => ({
    id,
    name: MOON_NAME[id],
    radiusKm: MOONS[id].radiusKm,
    ephemeris: (timeMs: number) => {
      const p = at(timeMs)[MOONS[id].parent];
      const o = moonLocalOffsetEclM(id, timeMs);
      return v3(p.x + o.x, p.y + o.y, p.z + o.z);
    },
    color: MOON_COLOR[id],
    mapAnchor: null,
    parentId: MOONS[id].parent,
    orbitPeriodDays: moonOrbitPeriodDays(id),
  }));
  return [...core, ...moons];
}

// ── mount ───────────────────────────────────────────────────────────────────

/** Live seam facts the parent reads off the real map each frame. */
export interface MapSeamState {
  zoom: number;
  minZoom: number;
  centerLatDeg: number;
  centerLonDeg: number;
}

/** Per-frame CSS pose for the live maplibre canvas — the Earth anchor. */
export interface EarthAnchor {
  /** false ⇒ Earth is off-screen/behind: parent hides the canvas (opacity
   *  0) but must NOT reset styles — that is dispose's applyEarthAnchor(null). */
  visible: boolean;
  /** translation of the canvas center to Earth's projected point, CSS px. */
  dxPx: number;
  dyPx: number;
  /** CSS scale that makes the drawn globe span Earth's true disc. Clamped
   *  at 1 only while the seam exit is ARMED (idle at the anchor / flying
   *  home) so the handback never overshoots; on flights past Earth it
   *  follows the true disc — the live map swells by, upscaled but honest. */
  scale: number;
  /** Earth-axis screen roll (0 while the camera up is axis-referenced —
   *  kept for a future free-roll camera). */
  rollDeg: number;
  /** live-map opacity: crossfades with the impostor as the disc shrinks. */
  opacity: number;
}

export interface SpaceFrameOptions {
  /** initial sky-clock ms (time axis; default Date.now()). */
  timeMs?: number;
  fovDeg?: number;
  /** B2 initial scale state (default: the persisted preference — VISIBLE on
   *  first run). Layout only; labels/scale bar stay true regardless. */
  scale?: ScaleState;
  /** B3 initial orbit-paths visibility (default: the persisted preference —
   *  ON on first run). */
  orbitPaths?: boolean;
  /** the body registry (default: defaultBodyRegistry() — Sun/Moon/Earth +
   *  the 8 planets). Must contain exactly one emissive body (the light
   *  source) and at most one mapAnchor body. */
  bodies?: SpaceBodyDef[];
  getMapSeam: () => MapSeamState;
  /** apply the anchor to the maplibre canvas; null ⇒ release (dispose). */
  applyEarthAnchor: (a: EarthAnchor | null) => void;
  /** align the live map's hemisphere to the sub-camera point (throttled by
   *  the frame; parent just jumpTo's). */
  recenterMap: (latDeg: number, lonDeg: number) => void;
  /** the seam crossed inward: parent restores map control. Fired once. */
  onExitToMap: () => void;
}

export interface SpaceFrameHandle {
  setTime(dateMs: number): void;
  render(): void;
  /** log-space fly to a body, arriving at the FRAME_DISC_FRACTION framing
   *  with the look-back offset (the place you left hangs beside the target). */
  flyTo(id: string): void;
  /** fly home to the seam and hand the camera back to the map. */
  flyHome(): void;
  /** wheel/pinch/button impulse — multiplies camera distance (exponential
   *  zoom; buttons/keys convert to deltaY via zoomSeam.ZOOM_BUTTON_DELTAY). */
  nudgeZoom(deltaY: number): void;
  /** B2: live scale update (slider drags). rAF-coalesced like any input —
   *  layout re-flows next frame; ephemeris and labels are untouched. */
  setScale(st: ScaleState): void;
  /** B3: toggle the orbit-ellipse polylines (cached lines restyle, never
   *  resample, on scale changes — the charter's slider rule). */
  setOrbitPaths(on: boolean): void;
  getState(): SpaceFrameState;
  dispose(): void;
}

export interface SpaceFrameBodyState {
  id: string;
  name: string;
  screenX: number;
  screenY: number;
  /** RENDERED disc px (B2: true disc through renderedDiscPx — the anchor
   *  and any body at/above the apparent cap render true). */
  discPx: number;
  behind: boolean;
  marker: boolean;
  /** camera-relative illuminated fraction (real phase from HERE). */
  litFraction: number;
  /** TRUE camera→body distance, meters — what the label prints. THE
   *  NUMBERS NEVER LIE: computed from real ephemeris positions with the
   *  camera at its real distance from the focus body, at every c/s. */
  distM: number;
  /** camera→body distance in the compressed LAYOUT, meters (what the
   *  projection used). Equals distM at c=0. */
  layoutDistM: number;
  /** B3: spin-axis unit vector (ecliptic of date) from the IAU model —
   *  null for bodies without a rotation model. Time-derived from the one
   *  simulation clock like everything else. */
  axisEcl: Vec3 | null;
  /** B3: IAU prime-meridian angle W, degrees [0,360) — the body's true
   *  rotation state (Moon tidally locked by the real constants). B4/B5
   *  surface rendering consumes axisEcl + this pair. */
  primeMeridianDeg: number | null;
  /** B3: true when this satellite's marker+label were folded into its
   *  parent's "+N moons" note because it projects inside the parent's
   *  drawn footprint (the body pixel itself still renders). */
  insideParent: boolean;
}

export interface SpaceFrameState {
  timeMs: number;
  focusId: string;
  distM: number;
  /** B2: the scale state the frame rendered with this frame. */
  scale: ScaleState;
  flying: boolean;
  /** true while the rAF loop is live (flights/input); idle frames cost 0. */
  animating: boolean;
  renderMsLast: number;
  markers: number;
  /** the scale bar as DRAWN this frame (render truth for the harness):
   *  mi/km per the units preference near bodies, AU at system range. */
  scaleBar: { widthPx: number; label: string } | null;
  /** B3 orbit-line render state: toggle + how many of the registry's
   *  lines are sampled and cached (they stream in one per macrotask). */
  orbitPaths: { on: boolean; ready: number; total: number };
  anchor: (EarthAnchor & { subLatDeg: number; subLonDeg: number }) | null;
  bodies: SpaceFrameBodyState[];
}

interface FlightState {
  fromId: string;
  toId: string;
  fromDir: Vec3;
  toDir: Vec3;
  fromDist: number;
  toDist: number;
  startedAt: number;
  durMs: number;
  /** hand the camera to the map when this flight lands (flyHome). */
  exitOnArrival: boolean;
}

export function mountSpaceFrame(container: HTMLElement, opts: SpaceFrameOptions): SpaceFrameHandle {
  const fovDeg = opts.fovDeg ?? DEFAULT_FOV_DEG;
  let timeMs = opts.timeMs ?? Date.now();
  let disposed = false;
  let exited = false;

  // ── canvas UNDER the map canvas: first child paints first, the (globe-
  // transparent) maplibre canvas composites on top of it ──
  const canvas = document.createElement("canvas");
  canvas.className = "vt-space-frame";
  canvas.style.position = "absolute";
  canvas.style.inset = "0";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  canvas.style.pointerEvents = "none"; // input rides the container (bubbling)
  container.insertBefore(canvas, container.firstChild);
  const ctx = canvas.getContext("2d");

  // ── the registry: which bodies exist, who lights them, who carries the
  // live map (Earth today) — declaration, not architecture ──
  const defs = opts.bodies && opts.bodies.length ? opts.bodies : defaultBodyRegistry();
  const defById = new Map(defs.map((d) => [d.id, d] as const));
  const sunDef = defs.find((d) => d.emissive) ?? defs[0];
  const anchorDef = defs.find((d) => d.mapAnchor === "maplibre") ?? defs[0];
  const radiusM = (id: string): number => (defById.get(id)?.radiusKm ?? 1) * 1000;

  // ── B2 scale state (layout only — the numbers never lie) ──
  let scaleSt: ScaleState = clampScaleState(opts.scale ?? getCelestialScale());

  // TRUE positions, memoized per instant (labels, scale bar, honesty).
  const trueMemo: { t: number; pos: Record<string, Vec3> } = { t: Number.NaN, pos: {} };
  const positionsNow = (t: number): Record<string, Vec3> => {
    if (trueMemo.t !== t) {
      for (const d of defs) trueMemo.pos[d.id] = d.ephemeris(t);
      trueMemo.t = t;
    }
    return trueMemo.pos;
  };
  // COMPRESSED layout positions, memoized per (instant, c) into preallocated
  // vectors — a slider drag re-flows ten pows and allocates nothing.
  const scaleInput: ScaleBodyIn[] = defs.map((d) => ({
    id: d.id,
    parentId: d.parentId ?? null,
    pos: { x: 0, y: 0, z: 0 },
  }));
  const layoutMemo: { t: number; c: number; pos: Record<string, Vec3> } = { t: Number.NaN, c: Number.NaN, pos: {} };
  for (const d of defs) layoutMemo.pos[d.id] = { x: 0, y: 0, z: 0 };
  const layoutNow = (t: number): Record<string, Vec3> => {
    if (layoutMemo.t !== t || layoutMemo.c !== scaleSt.c) {
      const truePos = positionsNow(t);
      for (const si of scaleInput) si.pos = truePos[si.id];
      applyDistanceCompression(scaleInput, scaleSt.c, layoutMemo.pos);
      layoutMemo.t = t;
      layoutMemo.c = scaleSt.c;
    }
    return layoutMemo.pos;
  };
  // rendered disc from the true radius at the LAYOUT distance, through the
  // B2 size pipeline (anchor exempt, Sun capped, apparent cap; s=1 identity;
  // reference response curve m^0.78·rel^−0.22 — rel in Earth radii)
  const discPxOf = (id: string, layoutDistM: number, k: number): number =>
    renderedDiscPx(
      bodyDiscPx(radiusM(id), layoutDistM, k),
      scaleSt.s,
      (defById.get(id)?.radiusKm ?? SIZE_REL_EARTH_KM) / SIZE_REL_EARTH_KM,
      defById.get(id)?.mapAnchor === "maplibre",
      !!defById.get(id)?.emissive,
    );

  // ── B3 orbit-ellipse polylines: sampled ONCE per body from the real
  // ephemeris (one body per macrotask — no >16ms main-thread work), cached,
  // and only RE-LAID-OUT when the compression slider moves (memoized per c).
  // Resampled only when the sim clock drifts past the staleness window. ──
  let orbitsOn = opts.orbitPaths ?? getOrbitPathsPref();
  interface OrbitEntry {
    line: OrbitPolyline;
    /** heliocentric lines: compressed layout memo (satellite lines never
     *  compress — true offsets ride the parent's layout position). */
    layout: Float64Array | null;
    layoutC: number;
  }
  const orbitCache = new Map<string, OrbitEntry>();
  const orbitTotal = defs.filter((d) => d.orbitPeriodDays).length;
  let orbitJob: ReturnType<typeof setTimeout> | null = null;
  function nextOrbitBody(): SpaceBodyDef | null {
    for (const d of defs) {
      if (!d.orbitPeriodDays) continue;
      const e = orbitCache.get(d.id);
      if (!e || orbitPolylineStale(e.line.sampledAtMs, timeMs)) return d;
    }
    return null;
  }
  function scheduleOrbitSampling(): void {
    if (orbitJob || disposed || !orbitsOn || !nextOrbitBody()) return;
    orbitJob = setTimeout(() => {
      orbitJob = null;
      if (disposed || !orbitsOn) return;
      const def = nextOrbitBody();
      if (!def) return;
      const parentId = def.parentId ?? null;
      const parentDef = parentId ? defById.get(parentId) : null;
      const line = sampleOrbitPolyline(
        def.id,
        parentId,
        def.ephemeris,
        parentDef ? parentDef.ephemeris : null,
        timeMs,
        def.orbitPeriodDays!,
        parentId ? ORBIT_SAMPLES_MOON : ORBIT_SAMPLES_PLANET,
      );
      orbitCache.set(def.id, { line, layout: null, layoutC: Number.NaN });
      kick(); // paint the new line
      scheduleOrbitSampling(); // chain: next body on its own macrotask
    }, 0);
  }

  /** rgba() from a #rrggbb presentation color at the path alpha. */
  function pathStroke(hex: string, alpha: number): string {
    return `rgba(${parseInt(hex.slice(1, 3), 16)},${parseInt(hex.slice(3, 5), 16)},${parseInt(hex.slice(5, 7), 16)},${alpha})`;
  }

  /**
   * Stroke every cached orbit line in the CURRENT layout space (same
   * dual-space rule as bodies: geometry compressed, numbers elsewhere stay
   * true). Inline projection — no per-vertex allocation. Segments crossing
   * behind the camera break the stroke instead of smearing across it.
   */
  function drawOrbitPaths(
    c2d: CanvasRenderingContext2D,
    pos: Record<string, Vec3>,
    camPos: Vec3,
    basis: CamBasis,
    k: number,
    cx: number,
    cy: number,
  ): void {
    c2d.lineWidth = 1;
    for (const def of defs) {
      const entry = orbitCache.get(def.id);
      if (!entry) continue;
      const parentId = entry.line.parentId;
      let px = 0;
      let py = 0;
      let pz = 0;
      let verts: Float64Array;
      if (parentId) {
        const pp = pos[parentId];
        if (!pp) continue;
        // skip when the whole local orbit is sub-pixel from here
        const pdx = pp.x - camPos.x;
        const pdy = pp.y - camPos.y;
        const pdz = pp.z - camPos.z;
        const pDist = Math.hypot(pdx, pdy, pdz);
        if (pDist > 0 && (entry.line.maxRadiusM / pDist) * k < 2) continue;
        px = pp.x;
        py = pp.y;
        pz = pp.z;
        verts = entry.line.pts; // TRUE offsets — the parented layout rule
      } else {
        if (entry.layoutC !== scaleSt.c) {
          entry.layout = layoutOrbitPolyline(entry.line.pts, scaleSt.c, entry.layout ?? undefined);
          entry.layoutC = scaleSt.c;
        }
        verts = entry.layout!;
      }
      c2d.strokeStyle = pathStroke(def.color, 0.3);
      c2d.beginPath();
      let open = false;
      const n = verts.length / 3;
      for (let i = 0; i <= n; i++) {
        const j = (i % n) * 3; // wrap: join last→first to close the loop
        const wx = px + verts[j] - camPos.x;
        const wy = py + verts[j + 1] - camPos.y;
        const wz = pz + verts[j + 2] - camPos.z;
        const depth = wx * basis.f.x + wy * basis.f.y + wz * basis.f.z;
        if (depth <= 0) {
          open = false;
          continue;
        }
        const sx = cx + (k * (wx * basis.r.x + wy * basis.r.y + wz * basis.r.z)) / depth;
        const sy = cy - (k * (wx * basis.u.x + wy * basis.u.y + wz * basis.u.z)) / depth;
        if (!open) {
          c2d.moveTo(sx, sy);
          open = true;
        } else {
          c2d.lineTo(sx, sy);
        }
      }
      c2d.stroke();
    }
  }

  // ── camera: focus body + unit dir (target→camera) + distance ──
  const seam0 = opts.getMapSeam();
  const entryDiscPx = mapGlobeDiscPx(seam0.zoom, seam0.centerLatDeg);
  let focusId: string = anchorDef.id;
  let dir = entryCameraDir(seam0.centerLatDeg, seam0.centerLonDeg, timeMs);
  let kNow = perspectiveScalePx(fovDeg, container.clientHeight || 600);
  let dist = distanceForDiscPx(radiusM(anchorDef.id), entryDiscPx, kNow);
  let flight: FlightState | null = null;
  let renderMsLast = 0;
  let lastState: SpaceFrameState | null = null;

  // recenter throttle (hemisphere alignment; only while the map is visible)
  let lastRecenterAt = 0;
  let lastRecenterLat = seam0.centerLatDeg;
  let lastRecenterLon = seam0.centerLonDeg;

  // ── sprite cache: per-body shaded sphere, keyed on quantized size+light ──
  const spriteCache = new Map<string, HTMLCanvasElement>();

  function shadedSprite(def: SpaceBodyDef, radiusPx: number, sunCam: Vec3): HTMLCanvasElement {
    // quantize: 6% radius steps, 0.06 light-direction steps — a flight
    // re-shades a handful of times, idle frames reuse the cache
    const rq = Math.max(2, Math.round(Math.exp(Math.round(Math.log(radiusPx) / 0.06) * 0.06)));
    const sq = (n: number): number => Math.round(n / 0.06) * 0.06;
    const key = `${def.id}|${rq}|${sq(sunCam.x)},${sq(sunCam.y)},${sq(sunCam.z)}`;
    const hit = spriteCache.get(key);
    if (hit) return hit;
    if (spriteCache.size > 48) spriteCache.clear(); // tiny working set, no LRU needed
    const shadeR = Math.min(rq, SPRITE_MAX_SHADE_RADIUS);
    const size = shadeR * 2 + 2;
    const c = document.createElement("canvas");
    c.width = size;
    c.height = size;
    const cc = c.getContext("2d")!;
    if (def.emissive) {
      // emissive: limb-darkened radial glow, no phase (presentation of a
      // light source; the SIZE is the true disc)
      const g = cc.createRadialGradient(shadeR + 1, shadeR + 1, 0, shadeR + 1, shadeR + 1, shadeR);
      g.addColorStop(0, "#fff6dd");
      g.addColorStop(0.75, "#ffd76e");
      g.addColorStop(1, "rgba(255,190,80,0.25)");
      cc.fillStyle = g;
      cc.beginPath();
      cc.arc(shadeR + 1, shadeR + 1, shadeR, 0, Math.PI * 2);
      cc.fill();
    } else {
      // Lambert-lit sphere under the REAL sun direction (camera frame):
      // the phase and terminator orientation are geometry, not paint.
      const img = cc.createImageData(size, size);
      const color = def.color;
      const cr = parseInt(color.slice(1, 3), 16);
      const cg = parseInt(color.slice(3, 5), 16);
      const cb = parseInt(color.slice(5, 7), 16);
      const s = norm3(sunCam);
      for (let py = 0; py < size; py++) {
        for (let px = 0; px < size; px++) {
          const nx = (px - shadeR - 1) / shadeR;
          const ny = -(py - shadeR - 1) / shadeR; // screen y down → up
          const nz2 = 1 - nx * nx - ny * ny;
          if (nz2 < 0) continue;
          const nz = Math.sqrt(nz2);
          const lit = nx * s.x + ny * s.y + nz * s.z;
          // soft terminator (±0.04 of the cosine) — display smoothing only
          const day = Math.max(0, Math.min(1, (lit + 0.03) / 0.07));
          const shade = 0.05 + 0.95 * Math.max(lit, 0);
          const w = 0.05 + 0.95 * (day * shade);
          const i = (py * size + px) * 4;
          img.data[i] = cr * w;
          img.data[i + 1] = cg * w;
          img.data[i + 2] = cb * w;
          img.data[i + 3] = 255;
        }
      }
      cc.putImageData(img, 0, 0);
    }
    spriteCache.set(key, c);
    return c;
  }

  function cssSize(): { w: number; h: number } {
    return { w: canvas.clientWidth || container.clientWidth || 1, h: canvas.clientHeight || container.clientHeight || 1 };
  }

  function resizeBacking(): void {
    const dpr = (globalThis.devicePixelRatio as number | undefined) || 1;
    const { w, h } = cssSize();
    const bw = Math.max(1, Math.round(w * dpr));
    const bh = Math.max(1, Math.round(h * dpr));
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // ── flight construction (always from the LIVE pose: cancels any flight
  // in progress first, freezing its blended camera as the new start) ──
  function beginFlight(toId: string, o?: { toDist?: number; toDir?: Vec3; exitOnArrival?: boolean }): void {
    cancelFlight();
    // flights fly through DRAWN space (layout) — a fly-to must land on the
    // body where it is rendered; framing math uses TRUE radii, and the
    // apparent-size cap renders true discs at arrival range, so the
    // FRAME_DISC_FRACTION landing is exact at any c/s
    const pos = layoutNow(timeMs);
    const from = pos[focusId];
    const to = pos[toId];
    let toDir: Vec3;
    let toDist: number;
    if (o?.toDir && o?.toDist) {
      toDir = o.toDir;
      toDist = o.toDist;
    } else {
      // arrival pose: swing PAST the target and look back the way you came,
      // offset sideways so the place you left hangs beside the target's limb
      // instead of hiding behind it (or behind your back)
      const away = norm3(sub(to, from)); // continue outbound past the target
      const axis = earthAxisEcl(timeMs);
      toDir = rotateAbout(away, camBasis(away, axis).u, ARRIVAL_LOOKBACK_OFFSET_DEG);
      const { w, h } = cssSize();
      toDist = o?.toDist ?? Math.max(
        MIN_DISTANCE_RADII * radiusM(toId),
        distanceForDiscPx(
          radiusM(toId),
          FRAME_DISC_FRACTION * Math.min(w, h),
          perspectiveScalePx(fovDeg, h),
        ),
      );
    }
    const swing = Math.acos(Math.max(-1, Math.min(1, dot(dir, toDir)))) * RAD;
    flight = {
      fromId: focusId,
      toId,
      fromDir: dir,
      toDir,
      fromDist: dist,
      toDist,
      startedAt: performance.now(),
      durMs: flyDurationMs(toDist / dist, swing),
      exitOnArrival: o?.exitOnArrival ?? false,
    };
    kick();
  }

  /** freeze a mid-flight pose into plain camera state (wheel takes over).
   *  Runs in layout space — dist becomes the camera's one shared distance-
   *  from-focus parameter, identical in both spaces thereafter. */
  function cancelFlight(): void {
    if (!flight) return;
    const pos = layoutNow(timeMs);
    const s = easeInOutCubic((performance.now() - flight.startedAt) / flight.durMs);
    const target = add(
      scale3(pos[flight.fromId], 1 - s),
      scale3(pos[flight.toId], s),
    );
    const d = slerpUnit(flight.fromDir, flight.toDir, s);
    const dd = expLerp(flight.fromDist, flight.toDist, s);
    const cam = add(target, scale3(d, dd));
    focusId = flight.toId;
    flight = null;
    // re-anchor the frozen pose on the new focus body
    const rel = sub(cam, pos[focusId]);
    dist = Math.max(len3(rel), MIN_DISTANCE_RADII * radiusM(focusId));
    dir = norm3(rel);
  }

  // ── the frame ──
  function draw(): void {
    if (disposed || !ctx || exited) return;
    const t0 = performance.now();
    resizeBacking();
    const { w, h } = cssSize();
    kNow = perspectiveScalePx(fovDeg, h);
    // B2 dual-space rule: LAYOUT positions (compressed) are what the frame
    // DRAWS; TRUE positions are what the frame SAYS (labels, status line,
    // scale bar). At c=0 the two are the same values, bit-exact.
    const posT = positionsNow(timeMs);
    const pos = layoutNow(timeMs);
    const axis = earthAxisEcl(timeMs);

    // flight progression — camera pose derived in BOTH spaces from the one
    // shared (focus/blend, dir, dist) parameterization
    let camPos: Vec3;
    let camTrue: Vec3;
    let viewDir: Vec3;
    if (flight) {
      const raw = (performance.now() - flight.startedAt) / flight.durMs;
      const s = easeInOutCubic(raw);
      const target = add(scale3(pos[flight.fromId], 1 - s), scale3(pos[flight.toId], s));
      viewDir = slerpUnit(flight.fromDir, flight.toDir, s);
      const dd = expLerp(flight.fromDist, flight.toDist, s);
      camPos = add(target, scale3(viewDir, dd));
      camTrue = add(
        add(scale3(posT[flight.fromId], 1 - s), scale3(posT[flight.toId], s)),
        scale3(viewDir, dd),
      );
      if (raw >= 1) {
        focusId = flight.toId;
        dir = flight.toDir;
        dist = flight.toDist;
        const exiting = flight.exitOnArrival;
        flight = null;
        if (exiting) {
          // landing at the seam: one final anchor apply below, then exit
          dist = distanceForDiscPx(
            radiusM(anchorDef.id),
            mapGlobeDiscPx(opts.getMapSeam().zoom, opts.getMapSeam().centerLatDeg) + 0.5,
            kNow,
          );
        }
        camPos = add(pos[focusId], scale3(dir, dist));
        camTrue = add(posT[focusId], scale3(dir, dist));
        viewDir = dir;
      }
    } else {
      camPos = add(pos[focusId], scale3(dir, dist));
      camTrue = add(posT[focusId], scale3(dir, dist));
      viewDir = dir;
    }
    const basis = camBasis(viewDir, axis);
    const cx = w / 2;
    const cy = h / 2;

    // ── paint space (black is honest — no fabricated starfield) ──
    ctx.fillStyle = "#020409";
    ctx.fillRect(0, 0, w, h);

    // ── B3 orbit ellipses (under the bodies) — cached polylines in the
    // current layout space; sampling streams in one body per macrotask ──
    if (orbitsOn) {
      scheduleOrbitSampling();
      drawOrbitPaths(ctx, pos, camPos, basis, kNow, cx, cy);
    }

    // ── project every body; depth-sort far→near for painter's occlusion.
    // Geometry (projection, disc, occlusion, shading) is LAYOUT space;
    // distM carried per body is the TRUE camera distance (label truth). ──
    const sunPos = pos[sunDef.id];
    const drawn: Array<{
      id: string; p: ProjectedPoint; discPx: number; distM: number;
      layoutDistM: number; lit: number; behind: boolean;
    }> = [];
    for (const def of defs) {
      const bp = pos[def.id];
      const rel = sub(bp, camPos);
      const layoutDistM = len3(rel);
      const p = projectPoint(bp, camPos, basis, kNow, cx, cy);
      const behind = !(p.depth > 0);
      const toSun = def.emissive ? v3(0, 0, 1) : norm3(sub(sunPos, bp));
      const toCam = norm3(scale3(rel, -1));
      drawn.push({
        id: def.id, p,
        distM: len3(sub(posT[def.id], camTrue)),
        layoutDistM,
        discPx: discPxOf(def.id, layoutDistM, kNow),
        lit: def.emissive ? 1 : apparentLitFraction(toSun, toCam),
        behind,
      });
    }
    drawn.sort((a, b) => b.layoutDistM - a.layoutDistM);

    const margin = 60;
    const onScreen = drawn.filter(
      (d) => !d.behind && d.p.x > -margin && d.p.x < w + margin && d.p.y > -margin && d.p.y < h + margin,
    );

    // bodies (far→near). Earth's DISC here is the impostor — the live map
    // composites above it and crossfades by opacity in the anchor below.
    for (const d of onScreen) {
      const r = d.discPx / 2;
      if (r >= 3) {
        // sun direction in camera coords for the sprite shader
        const def = defById.get(d.id)!;
        const toSunW = def.emissive ? basis.f : norm3(sub(sunPos, pos[d.id]));
        const sunCam = v3(dot(toSunW, basis.r), dot(toSunW, basis.u), -dot(toSunW, basis.f));
        const sprite = shadedSprite(def, r, sunCam);
        // the sprite's shaded disc has radius (sprite.width/2 − 1); scale the
        // whole sprite so THAT maps to exactly r (the +1 border must not
        // shrink the drawn body — true size is the contract)
        const shadeR = sprite.width / 2 - 1;
        const box = (r / shadeR) * sprite.width;
        ctx.drawImage(sprite, d.p.x - box / 2, d.p.y - box / 2, box, box);
      } else if (r >= 0.5) {
        ctx.fillStyle = defById.get(d.id)!.color;
        ctx.beginPath();
        ctx.arc(d.p.x, d.p.y, r, 0, Math.PI * 2);
        ctx.fill();
      } else if (r > 0) {
        // honestly sub-pixel: one dim pixel, never inflated (marker below)
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = defById.get(d.id)!.color;
        ctx.fillRect(Math.round(d.p.x), Math.round(d.p.y), 1, 1);
        ctx.globalAlpha = 1;
      }
    }

    // markers + labels (near→far so close bodies claim label space first);
    // stacking runs on the label ANCHORS (right of each disc/marker), box-
    // aware — a big disc's label can collide with a distant body's label.
    // The live-map anchor composites ABOVE this canvas, so its opaque disc
    // would swallow any label row it covers (drive round 2: the Moon's limb
    // label vanished under the live Earth) — seed PHANTOM occupied rows
    // spanning that disc so labels step around it. Capped: a near-seam disc
    // filling the screen needs no label choreography.
    // B3: a satellite projecting INSIDE its parent's drawn footprint is not
    // separable at this zoom — its reticle+label would overprint the parent
    // (four identical reticles on Jupiter's pixel at system zoom). Fold it
    // into the parent's label as "+N moons" instead: nothing is invisible
    // (the honesty rail), and zooming toward the parent resolves the moons
    // back into their own markers/labels the moment they separate.
    const insideParent = new Set<string>();
    const foldedMoons = new Map<string, number>();
    for (const d of onScreen) {
      const parentId = defById.get(d.id)?.parentId;
      if (!parentId) continue;
      const par = drawn.find((x) => x.id === parentId);
      if (!par || par.behind || !Number.isFinite(par.p.x)) continue;
      const sep = Math.hypot(d.p.x - par.p.x, d.p.y - par.p.y);
      if (sep < Math.max(par.discPx / 2, 8) + 4) {
        insideParent.add(d.id);
        foldedMoons.set(parentId, (foldedMoons.get(parentId) ?? 0) + 1);
      }
    }
    const labeled = [...onScreen].reverse().filter((d) => !insideParent.has(d.id));
    const anchorDrawn = onScreen.find((d) => d.id === anchorDef.id);
    const phantoms: { x: number; y: number }[] = [];
    if (anchorDrawn && anchorDrawn.discPx >= 16 && mapAnchorOpacity(anchorDrawn.discPx) > 0.05) {
      const reach = anchorDrawn.discPx / 2 + 7;
      for (let py = anchorDrawn.p.y - reach; py <= anchorDrawn.p.y + reach && phantoms.length < 14; py += LABEL_COLLIDE_PX) {
        phantoms.push({ x: anchorDrawn.p.x - LABEL_BOX_W_PX / 2, y: py });
      }
    }
    const offsets = layoutLabelStacks([
      ...phantoms,
      ...labeled.map((d) => ({ x: d.p.x + Math.max(d.discPx / 2, 8) + 6, y: d.p.y })),
    ]).slice(phantoms.length);
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    let markers = 0;
    const markerIds = new Set<string>(); // getState reports what was DRAWN
    labeled.forEach((d, i) => {
      const r = d.discPx / 2;
      const isMarker = markerNeeded(d.discPx);
      if (isMarker) {
        markers++;
        markerIds.add(d.id);
        // reticle: ring + 4 ticks OUTSIDE the (sub-pixel) body — reads as an
        // annotation, never as a disc
        const ring = 8;
        ctx.strokeStyle = "rgba(190,205,225,0.6)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(d.p.x, d.p.y, ring, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        for (const [tx, ty] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
          ctx.moveTo(d.p.x + tx * ring, d.p.y + ty * ring);
          ctx.lineTo(d.p.x + tx * (ring + 4), d.p.y + ty * (ring + 4));
        }
        ctx.stroke();
      }
      const anchorX = d.p.x + Math.max(r, 8) + 6;
      const ly = d.p.y + offsets[i];
      if (offsets[i] > 0) {
        ctx.strokeStyle = "rgba(190,205,225,0.35)";
        ctx.beginPath();
        ctx.moveTo(d.p.x + Math.max(r, 8), d.p.y);
        ctx.lineTo(anchorX, ly);
        ctx.stroke();
      }
      ctx.fillStyle = "rgba(210,222,238,0.92)";
      const extra =
        d.id === anchorDef.id && mapAnchorOpacity(d.discPx) < 0.5
          ? " · live map resumes on approach"
          : "";
      // B3: satellites folded into this body's footprint surface here —
      // never silently absent (zoom/fly closer and they resolve)
      const folded = foldedMoons.get(d.id) ?? 0;
      const moonsNote = folded ? ` · +${folded} moon${folded > 1 ? "s" : ""}` : "";
      ctx.fillText(`${defById.get(d.id)!.name} · ${fmtSpaceDistance(d.distM)}${extra}${moonsNote}`, anchorX, ly);
    });

    // ── chrome (x=72 clears the page's left button rail; the bottom-left
    // caption slot is free because the parent hides the map-meta overlays
    // — preset switch, imagery-date chip — while the frame is active, and
    // raises the surviving +/- zoom buttons above the caption via CSS) ──
    ctx.fillStyle = "rgba(210,222,238,0.85)";
    ctx.fillText(new Date(timeMs).toISOString().replace(/\.\d{3}Z$/, "Z"), 72, 22);
    const focus = drawn.find((d) => d.id === focusId)!;
    ctx.fillStyle = "rgba(160,175,198,0.8)";
    ctx.fillText(
      `${flight ? "flying to" : "at"} ${defById.get(flight ? flight.toId : focusId)!.name} · camera ${fmtSpaceDistance(flight ? len3(sub(camTrue, posT[flight.toId])) : focus.distM)} out`,
      72, 38,
    );
    // honesty caption — persistent, every frame; wraps to three lines on
    // narrow viewports so nothing ever clips off-screen (390px flawless).
    // B2: the caption tracks the scale state truthfully — TRUE SCALE keeps
    // the B1 caption; ANY compression switches to the directive's own
    // wording, amber (the SIGNAL/warning hue) so a compressed layout can
    // never pass itself off as the true-scale view.
    const trueNow = isTrueScale(scaleSt);
    const capLines = trueNow
      ? (w < 700
        ? [
            "TRUE SCALE — real ephemeris positions & sizes",
            "the camera does the compressing · sub-pixel bodies get markers",
            "Schlyter/van Flandern (~arcmin) · Moon phase real",
          ]
        : [
            "TRUE SCALE — real ephemeris positions & sizes · the camera does the compressing",
            "markers flag bodies smaller than a pixel · Schlyter/van Flandern (~arcmin) · Moon phase real",
          ])
      : (w < 700
        ? [
            "distances/sizes compressed for visibility",
            "— labels always show real values",
            `compression ${Math.round(scaleSt.c * 100)}% · size ×${Math.round(scaleSt.s)} · Sun cap ×${SUN_SIZE_MULT_CAP} · Earth (live map) true`,
          ]
        : [
            "distances/sizes compressed for visibility — labels always show real values",
            `compression ${Math.round(scaleSt.c * 100)}% · body size ×${Math.round(scaleSt.s)} · Sun cap ×${SUN_SIZE_MULT_CAP} · Earth (live map) always true · Schlyter/van Flandern (~arcmin)`,
          ]);
    capLines.forEach((line, i) => {
      ctx.fillStyle = trueNow
        ? (i === 0 ? "rgba(160,175,198,0.8)" : "rgba(140,155,178,0.65)")
        : (i < capLines.length - 1 ? "rgba(251,178,76,0.9)" : "rgba(251,178,76,0.6)");
      ctx.fillText(line, 16, h - 16 - (capLines.length - 1 - i) * 14);
    });

    // ── scale bar (B1 §1): the map's own scale control stops at the floor —
    // this continues the SAME instrument through mi → thousands of mi → AU,
    // switching with the units preference, styled after .maplibregl-ctrl-
    // scale (translucent box, hairline frame, end ticks). Input is meters/px
    // at the focus body's surface, which equals the map bar's ground m/px at
    // the seam by construction (see spaceScaleBar). Sits just above the
    // caption; the zoom buttons are raised above both via CSS. ──
    let scaleBarOut: { widthPx: number; label: string } | null = null;
    {
      const barId = flight ? flight.toId : focusId; // agree with the status line
      const focusForBar = drawn.find((d) => d.id === barId);
      if (focusForBar) {
        const mpp = Math.max(1e-9, (focusForBar.distM - radiusM(barId)) / kNow);
        const bar = spaceScaleBar(mpp, getUnits());
        scaleBarOut = bar;
        const bx = 16;
        const by = h - 16 - capLines.length * 14 - 12; // baseline of the bar
        ctx.fillStyle = "rgba(5,10,19,0.6)";
        ctx.fillRect(bx - 4, by - 20, bar.widthPx + 8, 24);
        ctx.strokeStyle = "rgba(102,128,160,0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath(); // left tick, bottom rule, right tick — the classic bar
        ctx.moveTo(bx + 0.5, by - 6);
        ctx.lineTo(bx + 0.5, by + 0.5);
        ctx.lineTo(bx + bar.widthPx - 0.5, by + 0.5);
        ctx.lineTo(bx + bar.widthPx - 0.5, by - 6);
        ctx.stroke();
        ctx.fillStyle = "rgba(170,185,205,0.9)";
        ctx.fillText(bar.label, bx + 4, by - 10);
      }
    }

    // ── the Earth anchor: pose the LIVE MAP as the Earth ──
    const seam = opts.getMapSeam();
    const earth = drawn.find((d) => d.id === anchorDef.id)!;
    const mapDisc = mapGlobeDiscPx(seam.zoom, seam.centerLatDeg);
    let anchorOut: SpaceFrameState["anchor"] = null;
    if (!earth.behind && Number.isFinite(earth.p.x)) {
      const cssScale = earth.discPx / mapDisc;
      const opacity = mapAnchorOpacity(earth.discPx);
      // exit is armed only when heading for the map (idle at the anchor, or
      // the fly-home flight) — a Moon-bound arc swinging past Earth must
      // neither eject the user nor freeze Earth at the seam size
      const exitArmed = seamExitArmed(!!flight, !!flight?.exitOnArrival, focusId === anchorDef.id);
      const sub_ = subCameraLatLon(norm3(sub(camPos, pos[anchorDef.id])), timeMs);
      const anchor: EarthAnchor = {
        visible: Math.abs(earth.p.x - cx) < w * 2 && Math.abs(earth.p.y - cy) < h * 2,
        dxPx: earth.p.x - cx,
        dyPx: earth.p.y - cy,
        // armed: clamp at the seam (no overshoot flash); unarmed: TRUE size
        scale: exitArmed ? Math.min(cssScale, 1) : cssScale,
        rollDeg: northRollDeg(basis, axis), // 0 by construction (axis-up camera)
        opacity,
      };
      opts.applyEarthAnchor(anchor);
      anchorOut = { ...anchor, subLatDeg: sub_.latDeg, subLonDeg: sub_.lonDeg };
      // hemisphere alignment: the live map shows the face that looks at YOU.
      // Throttled; skipped while the map is invisible (no fetch churn).
      const now = performance.now();
      if (
        opacity > 0.05 &&
        now - lastRecenterAt > 200 &&
        (Math.abs(sub_.latDeg - lastRecenterLat) > 1.5 ||
          Math.abs(((sub_.lonDeg - lastRecenterLon + 540) % 360) - 180) > 1.5)
      ) {
        lastRecenterAt = now;
        lastRecenterLat = sub_.latDeg;
        lastRecenterLon = sub_.lonDeg;
        opts.recenterMap(Math.max(-85, Math.min(85, sub_.latDeg)), sub_.lonDeg);
      }
      // ── the seam, inward: scale crossing 1 while ARMED hands the
      // camera back ──
      if (cssScale > 1.0001 && exitArmed && !exited) {
        exited = true;
        opts.onExitToMap();
      }
    } else {
      opts.applyEarthAnchor({ visible: false, dxPx: 0, dyPx: 0, scale: 1, rollDeg: 0, opacity: 0 });
    }

    renderMsLast = performance.now() - t0;
    lastState = {
      timeMs,
      focusId,
      distM: dist,
      scale: { c: scaleSt.c, s: scaleSt.s },
      flying: !!flight,
      animating: !!flight || performance.now() - lastInputAt < 200,
      renderMsLast,
      markers,
      scaleBar: scaleBarOut,
      orbitPaths: { on: orbitsOn, ready: orbitCache.size, total: orbitTotal },
      anchor: anchorOut,
      bodies: drawn.map((d) => ({
        id: d.id,
        name: defById.get(d.id)!.name,
        screenX: d.p.x,
        screenY: d.p.y,
        discPx: d.discPx,
        behind: d.behind,
        // true ⇔ this body's marker was actually DRAWN this frame (in front,
        // inside the render margin, sub-threshold) — the render truth,
        // never a recomputation that could drift from it
        marker: markerIds.has(d.id),
        litFraction: d.lit,
        distM: d.distM,
        layoutDistM: d.layoutDistM,
        // B3 rotation state (IAU pole + W at the sim-clock instant) — the
        // orientation truth B4/B5 surfaces will render from
        axisEcl: hasRotationModel(d.id) ? axisEclOfDate(d.id, timeMs) : null,
        primeMeridianDeg: hasRotationModel(d.id) ? iauPrimeMeridianDeg(d.id, timeMs) : null,
        insideParent: insideParent.has(d.id),
      })),
    };
  }

  // ── rAF only while something moves — idle frames cost nothing ──
  let rafId = 0;
  let lastInputAt = 0;
  function kick(): void {
    if (rafId || disposed || exited) return;
    const loop = (): void => {
      rafId = 0;
      if (disposed || exited) return;
      draw();
      if (flight || performance.now() - lastInputAt < 200) {
        rafId = requestAnimationFrame(loop);
      }
    };
    rafId = requestAnimationFrame(loop);
  }

  // ── input (container-level: events bubble up from the map canvas) ──
  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    lastInputAt = performance.now();
    nudge(e.deltaY);
  };
  function nudge(deltaY: number): void {
    if (exited) return;
    cancelFlight();
    // per-body closest approach; for Earth the seam (scale > 1 inside
    // draw()) hands back to the map long before this floor could bind
    dist = Math.min(
      MAX_CAMERA_DISTANCE_M,
      Math.max(MIN_DISTANCE_RADII * radiusM(focusId), dist * zoomStepFactor(deltaY)),
    );
    kick();
  }

  // drag = orbit the focus body; two pointers = pinch zoom
  const pointers = new Map<number, { x: number; y: number }>();
  let pinchDist = 0;
  let dragMoved = 0;
  const onPointerDown = (e: PointerEvent): void => {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    dragMoved = 0;
    if (pointers.size === 2) {
      const [a, b] = Array.from(pointers.values());
      pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
    }
  };
  const onPointerMove = (e: PointerEvent): void => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    lastInputAt = performance.now();
    if (pointers.size === 2) {
      const [a, b] = Array.from(pointers.values());
      const d2 = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchDist > 0 && d2 > 0) {
        cancelFlight();
        dist = Math.min(
          MAX_CAMERA_DISTANCE_M,
          Math.max(MIN_DISTANCE_RADII * radiusM(focusId), dist * (pinchDist / d2)),
        );
        kick();
      }
      pinchDist = d2;
      return;
    }
    dragMoved += Math.abs(dx) + Math.abs(dy);
    if (dragMoved < 3) return;
    cancelFlight();
    // orbit: yaw about the Earth axis, pitch about the camera right axis
    const axis = earthAxisEcl(timeMs);
    const b = camBasis(dir, axis);
    let nd = rotateAbout(dir, axis, dx * 0.25);
    nd = rotateAbout(nd, b.r, dy * 0.25);
    // clamp away from the poles so the axis-up basis never degenerates
    if (Math.abs(dot(nd, axis)) < 0.985) dir = norm3(nd);
    else dir = norm3(rotateAbout(dir, axis, dx * 0.25));
    kick();
  };
  const onPointerUp = (e: PointerEvent): void => {
    pointers.delete(e.pointerId);
    pinchDist = 0;
  };
  const onClick = (e: MouseEvent): void => {
    if (dragMoved >= 4 || exited || !lastState) return;
    const rect = container.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    // nearest-first hit test: disc radius or the marker reticle, min 10px.
    // Sorted by LAYOUT distance — visual nearest wins (what you clicked is
    // what you saw; true and layout ordering can differ under compression)
    const hits = lastState.bodies
      .filter((b) => !b.behind && Number.isFinite(b.screenX))
      .sort((a, b) => a.layoutDistM - b.layoutDistM);
    for (const b of hits) {
      const r = Math.max(b.discPx / 2, 10);
      if (Math.hypot(mx - b.screenX, my - b.screenY) <= r + 4) {
        if (b.id === anchorDef.id) flyHome();
        else if (b.id !== focusId || flight) beginFlight(b.id);
        return;
      }
    }
  };

  container.addEventListener("wheel", onWheel, { passive: false });
  container.addEventListener("pointerdown", onPointerDown);
  container.addEventListener("pointermove", onPointerMove);
  container.addEventListener("pointerup", onPointerUp);
  container.addEventListener("pointercancel", onPointerUp);
  container.addEventListener("click", onClick);

  let ro: ResizeObserver | null = null;
  if (typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(() => draw());
    ro.observe(container);
  }

  function flyHome(): void {
    if (exited) return;
    cancelFlight(); // freeze any flight first: home starts from the LIVE pose
    const seam = opts.getMapSeam();
    const target = distanceForDiscPx(
      radiusM(anchorDef.id),
      mapGlobeDiscPx(seam.zoom, seam.centerLatDeg) + 0.5, // land just inside the seam
      perspectiveScalePx(fovDeg, cssSize().h),
    );
    // come straight down the camera→anchor ray — the map recenters beneath
    // you, so you land above wherever you flew; leaving without flying
    // returns you exactly where you entered. Layout space: the ray must
    // point at the anchor where it is DRAWN (the seam itself is scale-
    // invariant — compression never touches the camera→anchor distance,
    // and the anchor body is never size-scaled).
    const pos = layoutNow(timeMs);
    const camNow = add(pos[focusId], scale3(dir, dist));
    const homeDir = norm3(sub(camNow, pos[anchorDef.id]));
    beginFlight(anchorDef.id, { toDir: homeDir, toDist: target, exitOnArrival: true });
  }

  draw(); // first frame: anchor at scale 1/opacity 1 — the seam, untouched

  return {
    setTime(dateMs: number): void {
      timeMs = dateMs;
      draw();
    },
    render(): void {
      draw();
    },
    flyTo(id: string): void {
      if (exited || !defById.has(id)) return;
      if (id === anchorDef.id) return flyHome();
      beginFlight(id);
    },
    flyHome,
    nudgeZoom(deltaY: number): void {
      lastInputAt = performance.now();
      nudge(deltaY);
    },
    setScale(st: ScaleState): void {
      const ns = clampScaleState(st);
      if (ns.c === scaleSt.c && ns.s === scaleSt.s) return;
      scaleSt = ns;
      // rAF-coalesced like any input: a slider drag firing faster than the
      // display refresh re-flows the layout at most once per frame
      lastInputAt = performance.now();
      kick();
    },
    setOrbitPaths(on: boolean): void {
      if (on === orbitsOn) return;
      orbitsOn = on;
      lastInputAt = performance.now();
      kick(); // paths appear (sampling streams in) or vanish next frame
    },
    getState(): SpaceFrameState {
      if (!lastState) draw();
      return lastState!;
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = 0;
      if (orbitJob) {
        clearTimeout(orbitJob);
        orbitJob = null;
      }
      try { ro?.disconnect(); } catch { /* already gone */ }
      container.removeEventListener("wheel", onWheel);
      container.removeEventListener("pointerdown", onPointerDown);
      container.removeEventListener("pointermove", onPointerMove);
      container.removeEventListener("pointerup", onPointerUp);
      container.removeEventListener("pointercancel", onPointerUp);
      container.removeEventListener("click", onClick);
      spriteCache.clear();
      opts.applyEarthAnchor(null); // release the map canvas (styles reset)
      try { canvas.remove(); } catch { /* detached */ }
    },
  };
}
