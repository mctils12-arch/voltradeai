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
// - Starfield: REAL bright stars only (Yale Bright Star Catalog, public
//   domain — starCatalog.ts). Each of the 9096 stars is drawn at its true
//   RA/Dec direction (converted to the scene's ecliptic-of-date frame, so it
//   registers with the panorama's own star field), sized/brightened by its
//   real V magnitude and tinted by its real color temperature. NEVER a
//   fabricated random sky (the "real position or absent" rail): positions and
//   colors are data; only magnitude→size and K→RGB are stated presentation
//   mappings. The field fades in as the camera leaves the inner solar system
//   (earlier than the faint galactic glow), so it is black near Earth.
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
  compressDistance,
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
  equatorNodeDirEclOfDate,
  rotationRateDegPerDay,
  rotationPeriodDays,
  type RotationBodyId,
} from "./rotation.js";
// SPACE VIEW VISUAL UPGRADE (human directive 2026-07-18 — handoff +
// reference in research/directives/): textured bodies, Saturn rings, the
// Milky Way backdrop, motion trails, the ecliptic grid, dot labels, the
// body card and the release gesture — all reproduced from the reference
// scene INSIDE this hand-rolled renderer (no three.js; the parent's
// architecture decision). spaceAssets owns assets/tiers/prefs;
// textureSphere owns the pure sampling/LUT/ring/sky math.
import {
  createSpaceTextureManager,
  milkyWayOpacity,
  starFieldOpacity,
  loadStarCatalog,
  getMilkyWayPref,
  getStarsPref,
  getEclipticGridPref,
  getLockHorizonPref,
  getMotionTrailsPref,
  getBodyLabelsPref,
  getRealisticLightingPref,
  MILKY_WAY_CREDIT,
  textureLonOffsetDeg,
  type SpaceTextureManager,
  type TexImage,
} from "./spaceAssets.js";
// GALAXY / STARFIELD DETAIL (human directive 2026-07-19): the REAL bright-star
// sky (Yale BSC, public domain) as GPU-cheap Canvas2D points, registered to the
// same ecliptic frame as the panorama — replaces the old "no starfield" note
// with real positions, never fabrication.
import {
  eqToEclDir,
  projectStarScreen,
  starSizePx,
  starIntensity,
  STAR_CATALOG_CREDIT,
  type StarCatalog,
} from "./starCatalog.js";
import {
  buildSphereLUT,
  createEmptySphereLUT,
  fillSphereLUTRows,
  composeTexturedSprite,
  ringBandsFromTextures,
  ringCircle3D,
  galacticBasisEcl,
  buildSkyRays,
  renderSkyToBuffer,
  spinObliquityDeg,
  SATURN_RING_INNER_REL,
  SATURN_RING_OUTER_REL,
  type SphereLUT,
  type RingBand,
  type GalacticBasis,
  type RingShadowParams,
} from "./textureSphere.js";
// B6 UNIVERSAL LIGHTING (directive §8): ONE Sun lights everything — real
// terminators/phases already come from the lambert shading below; this adds
// eclipse (umbra cone) + ring shadows, all from real positions at the sim clock.
import {
  shadowConeFactor,
  pointInSphereShadow,
} from "./bodyLighting.js";
import {
  sampleOrbitPolyline,
  layoutOrbitPolyline,
  orbitPolylineStale,
  getOrbitPathsPref,
  ORBIT_SAMPLES_PLANET,
  ORBIT_SAMPLES_MOON,
  type OrbitPolyline,
} from "./orbitPath.js";
// B4 "MOON FOR REAL": real LROC WAC tiles streamed into a perspective surface
// patch when the camera skims the Moon (the disc sprite caps out at ~448 px of
// texels; the surface patch shows real 100 m/px imagery where the eye looks).
import {
  createMoonTileManager,
  type MoonTileManager,
} from "./moonTiles.js";
// B5 "PLANET SURFACES": the same tile→patch path streams Mars/Mercury/Venus
// (and the B4 Moon) — trekBody() supplies each body's Trek scheme + credit.
import { trekBody } from "./lroc.js";
import {
  renderMoonSurfaceRows,
  surfaceLonLat,
  type MoonSurfaceView,
  type DetailOverlay,
} from "./moonSurface.js";

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

/** Manual (wheel/pinch) closest-approach floor, in body radii — SEPARATE
 *  from the fly-to arrival clamp (MIN_DISTANCE_RADII) so an arrival still
 *  frames the whole body (~4.5 radii) while the user can then push right in
 *  to skim the surface (human, 2026-07-18: "zoom in way more to the moon").
 *  1.05 sits just above the surface; the texture softens there until the
 *  LRO tile pyramid lands, but the APPROACH is no longer capped. */
export const MIN_ZOOM_RADII = 1.05;

/** Rotational inertia (the reference's damped OrbitControls feel — human:
 *  "exactly like the claude design"): after a drag releases, the orbit
 *  coasts and decays by this factor per frame (~0.9 ⇒ ≈1 s glide), and the
 *  rAF loop stays alive until |velocity| drops below the epsilon. Grabbing
 *  (pointerdown) zeroes it, exactly like OrbitControls. Pure, unit-tested. */
export const ORBIT_INERTIA_DAMP = 0.9;
export const ORBIT_INERTIA_EPS_DEG = 0.015;

/** North-lock polar clamp (radians from the up axis). Lock ON (default): the
 *  camera can tip near the top pole (0.12) but never swing under the ecliptic
 *  (π/2+0.42 past level). OFF only WIDENS the clamp (small pole guards keep
 *  the basis non-degenerate) — roll stays impossible either way, because
 *  camBasis rebuilds the up-vector every frame (see spaceFrame.test.ts
 *  "roll 0 by construction"). Pure + exported so the clamp is unit-testable. */
export const LOCK_POLAR_MIN = 0.12;
export const LOCK_POLAR_MAX = Math.PI / 2 + 0.42;
export const UNLOCK_POLAR_MIN = 0.05;
export const UNLOCK_POLAR_MAX = Math.PI - 0.05;

/** dot(dir,axis)=cos(polar) bounds for the current lock state. A candidate
 *  orbit direction is accepted iff dotLo ≤ dot ≤ dotHi. */
export function polarClampDots(locked: boolean): { dotHi: number; dotLo: number } {
  return {
    dotHi: Math.cos(locked ? LOCK_POLAR_MIN : UNLOCK_POLAR_MIN),
    dotLo: Math.cos(locked ? LOCK_POLAR_MAX : UNLOCK_POLAR_MAX),
  };
}

/** One coasting step: advance the residual angular velocity and decay it.
 *  Returns the applied step (deg) and the decayed velocity; below the
 *  epsilon it snaps to rest so the loop can idle. */
export function orbitInertiaStep(
  velDeg: number,
  damp: number = ORBIT_INERTIA_DAMP,
  epsDeg: number = ORBIT_INERTIA_EPS_DEG,
): { stepDeg: number; velDeg: number } {
  if (!Number.isFinite(velDeg) || Math.abs(velDeg) < epsDeg) return { stepDeg: 0, velDeg: 0 };
  return { stepDeg: velDeg, velDeg: velDeg * damp };
}

// ── ZOOM-TO-CURSOR + PAN — reconcile the orbit camera to the maplibre map's
// feel (human live feedback 2026-07-19: "it need to have the mechanics of the
// earth for zooming and panning ... hard to maneuver around the universe"). The
// map the human finds easy zooms toward the CURSOR and pans the surface; the
// praised OrbitControls reference adds damped orbit + right-drag pan. The
// bridge is a body-relative LOOK-AT OFFSET (focusOff, meters from the focus
// body's centre): the camera may look off-centre, so a wheel zoom pulls the
// cursor's surface point toward screen centre (the map's zoom-to-cursor payoff)
// and a pan-drag traverses the surface laterally — the map's mental model,
// inside the orbit camera. focusOff is 0 for centred orbiting and is reset on
// every focus change (fly-to / release / fly-home all run with focusOff = 0, so
// the seam, flights and follow contracts are untouched). All pure + unit-tested.

/** Wheel-zoom momentum: one eased step moves camera distance this fraction
 *  toward the target in LOG space (exponential zoom's uniform-feel twin) —
 *  ~4-5 frames to settle, responsive, never a hard snap. */
export const ZOOM_EASE_ALPHA = 0.34;

/** Per-notch gain of the cursor pull: the fraction of the residual toward the
 *  cursor's surface point the look-at moves each wheel notch (1 ⇒ the point
 *  reaches screen centre over a full zoom-in, matching the map's feel). */
export const CURSOR_PULL_GAIN = 1.0;

/** How far the look-at may slide off the body centre, in body radii — enough
 *  to pan / zoom-pull across the visible hemisphere to the limb, never beyond
 *  (the look-at can't fly off into empty space). */
export const PAN_MAX_OFFSET_RADII = 1.2;

/** Unnormalised world ray direction through screen pixel (mx,my) under a
 *  camera basis (forward f, right r, up u), perspective scale k and principal
 *  point (cx,cy). The exact inverse of projectPoint's mapping: right offset
 *  a = (mx−cx)/k, up offset b = (cy−my)/k (screen y is down), forward = 1. */
export function screenRayDir(
  mx: number, my: number, k: number, cx: number, cy: number, basis: CamBasis,
): Vec3 {
  const a = (mx - cx) / k;
  const b = (cy - my) / k;
  return v3(
    basis.f.x + a * basis.r.x + b * basis.u.x,
    basis.f.y + a * basis.r.y + b * basis.u.y,
    basis.f.z + a * basis.r.z + b * basis.u.z,
  );
}

/** Nearest FRONT intersection of ray (o + t·d, d need not be unit) with the
 *  sphere (centre c, radius R), or null on a miss / behind the camera — the
 *  cursor→surface raycast that anchors zoom-to-cursor. */
export function raycastSphere(o: Vec3, d: Vec3, c: Vec3, R: number): Vec3 | null {
  const ox = o.x - c.x;
  const oy = o.y - c.y;
  const oz = o.z - c.z;
  const A = d.x * d.x + d.y * d.y + d.z * d.z;
  const B = 2 * (ox * d.x + oy * d.y + oz * d.z);
  const C = ox * ox + oy * oy + oz * oz - R * R;
  const disc = B * B - 4 * A * C;
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  let t = (-B - sq) / (2 * A);
  if (t <= 0) t = (-B + sq) / (2 * A);
  if (t <= 0) return null;
  return v3(o.x + t * d.x, o.y + t * d.y, o.z + t * d.z);
}

/** Clamp a body-relative look-at offset to a maximum magnitude — keeps the
 *  look-at near the body so a pan or zoom-pull can never fling it away. */
export function clampOffset(off: Vec3, maxMag: number): Vec3 {
  const m = len3(off);
  if (m <= maxMag || m === 0) return off;
  const s = maxMag / m;
  return v3(off.x * s, off.y * s, off.z * s);
}

/** Cursor pull: move the look-at offset a fraction `alpha` toward the
 *  body-relative point `target` the cursor is over — bringing that point
 *  toward screen centre (the maplibre zoom-to-cursor payoff), clamped. */
export function cursorPullOffset(off: Vec3, target: Vec3, alpha: number, maxMag: number): Vec3 {
  const a = alpha < 0 ? 0 : alpha > 1 ? 1 : alpha;
  return clampOffset(
    v3(
      off.x + (target.x - off.x) * a,
      off.y + (target.y - off.y) * a,
      off.z + (target.z - off.z) * a,
    ),
    maxMag,
  );
}

/** Pan the look-at offset across the surface: a screen drag (dxPx,dyPx) at
 *  `metersPerPx` translates the offset in the camera's tangent plane so the
 *  surface point under the pointer follows it (grab-pan, the map's feel).
 *  Drag right ⇒ offset −right (content moves right); drag down ⇒ offset +up. */
export function panOffset(
  off: Vec3, dxPx: number, dyPx: number, basis: CamBasis, metersPerPx: number, maxMag: number,
): Vec3 {
  const kr = -dxPx * metersPerPx;
  const ku = dyPx * metersPerPx;
  return clampOffset(
    v3(
      off.x + kr * basis.r.x + ku * basis.u.x,
      off.y + kr * basis.r.y + ku * basis.u.y,
      off.z + kr * basis.r.z + ku * basis.u.z,
    ),
    maxMag,
  );
}

/** One eased zoom step in LOG distance: move `cur` a fraction `alpha` toward
 *  `target`, snapping when within snapFrac (relative). Wheel-zoom momentum
 *  that never lags behind intent — exponential zoom's uniform-feel ease. */
export function smoothZoomStep(cur: number, target: number, alpha: number, snapFrac = 0.004): number {
  if (!(cur > 0) || !(target > 0)) return target;
  if (Math.abs(Math.log(target / cur)) < snapFrac) return target;
  return Math.exp(Math.log(cur) + (Math.log(target) - Math.log(cur)) * alpha);
}

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

/** TEXTURED bodies shade to a larger cap (surface detail rewards it).
 *  Full-res sprites are NEVER built in one task: motion frames use the
 *  FAST tier synchronously, and the full tier streams in as row-chunked
 *  macrotasks (each bounded) once the pose settles. */
export const TEXTURE_MAX_SHADE_RADIUS = 224;

/** The synchronous sprite tier: small enough that a per-frame rebuild
 *  (time-warp spin, flights) stays a bounded main-thread task. */
export const FAST_SHADE_RADIUS = 112;

/** Row-chunk budget for the async full-res upgrade pipeline, pixels per
 *  macrotask (LUT fill ≈ compose ≈ a few ms each at this size). */
export const UPGRADE_CHUNK_PX = 45_000;

// ── B4 "MOON FOR REAL" — focused-close LROC surface patch ───────────────────

/** Camera within this many Moon radii → the perspective surface patch takes
 *  over from the billboard disc sprite (below it the true disc exceeds ~700px,
 *  where the ≤448px sprite is visibly soft). */
export const MOON_PATCH_ACTIVATE_RADII = 3.5;

/** Synchronous BOOTSTRAP tier: the patch buffer's long side, px. Since the
 *  2026-07-19 crisp-during-motion fix keeps the high-res buffer on screen
 *  through orbit/pan/zoom, this synchronous tier only ever shows on the FIRST
 *  approach to a body (before any full buffer exists) — so it is sized purely
 *  for a safe <16ms one-shot render, not for motion quality (mpFull carries
 *  that). Trimmed from 140 for headroom against the frame budget on a loaded
 *  SwiftShader machine (drive measured 15.3ms at 140 near 2 radii). */
export const MOON_PATCH_FAST_LONG_PX = 116;

/** Streamed high-res tier: the patch buffer's long side, px. Rendered in
 *  MOON_PATCH_CHUNK_PX row bands (each well under a frame), refreshed
 *  CONTINUOUSLY (during orbit/pan too — B5, not only when settled), then shown
 *  at full sharpness — real Trek imagery over the patch. patchBufDims caps the
 *  buffer at the disc's own pixel size, so a 900px disc stays 1:1; this larger
 *  cap only sharpens discs bigger than 900px (big/hi-dpi screens, very close).
 *  Kept moderate: a larger buffer is crisper but takes more bands to build, so
 *  it lags further behind fast motion before the fresh crisp frame swaps in. */
export const MOON_PATCH_FULL_LONG_PX = 1100;

/** During motion the last high-res buffer is kept on screen (no downgrade to
 *  the low-res fast tier) as long as the camera has rotated less than this from
 *  the pose it was rendered at — stored as a cosine for a cheap dot-product
 *  test. Beyond it (a fast fling) the registered fast tier takes over until a
 *  fresh high-res buffer lands. ~40° keeps orbit/pan crisp; the buffer refreshes
 *  well before the lag reaches the cap at normal drag speed. */
export const MOON_PATCH_MOTION_KEEP_COS = Math.cos((40 * Math.PI) / 180);

/** Row-chunk budget for the patch stream, px per macrotask (smaller than the
 *  sprite's UPGRADE_CHUNK_PX because a patch pixel costs a ray-sphere solve +
 *  atan2/asin). Banding is OUTPUT-IDENTICAL (proven by moonSurface's "chunked
 *  rows compose to the same buffer" test) — only the task granularity changes.
 *  Because the 2026-07-19 fix keeps the previous full buffer on screen while
 *  the next one builds, completion latency no longer costs crispness — so this
 *  can be SMALL for a hard <16ms guarantee: 6k (down from 12k) keeps a loaded
 *  SwiftShader band well under budget where 12k measured 17.4ms near 2 radii. */
export const MOON_PATCH_CHUNK_PX = 6_000;

/** Idle time (ms since the last input) after which the high-res patch buffer
 *  rebuilds for the settled pose. Below this the last crisp buffer is held on
 *  screen (see the REFRESH note in drawBodyPatch). Kept under the rAF loop's
 *  200ms input-active window so the settle rebuild always fires. */
export const MOON_PATCH_SETTLE_MS = 150;

/** The tile mosaic covers the viewport-visible surface arc times this factor
 *  (headroom past the viewport corners so the sharp region fills the frame AND
 *  a PREFETCH ring so panning/orbiting into new surface reveals already-resident
 *  tiles instead of soft/unloaded gaps). Widened to 2.8 (from 2.1) for the
 *  2026-07-19 crisp-during-motion gate: a pan/orbit sweeps the sub-camera point
 *  across the surface, and a wider resident window keeps the refreshed buffers
 *  sampling REAL tiles (not the soft base) through the motion. */
export const MOON_PATCH_COVER_MARGIN = 2.8;

/** Floor on the covered half-span, degrees — a mosaic never shrinks below a
 *  couple of degrees (avoids a degenerate one-texel fetch and gives the near
 *  edges of the patch real coverage). */
export const MOON_PATCH_MIN_HALFSPAN_DEG = 2;

/** A textured body's dot-label glyph renders while its disc is under this
 *  size (the reference's .lbl dot: a 9px annotation ring ON the body). */
export const LABEL_DOT_MAX_DISC_PX = 9;

/** The Sun's additive glow sprite spans this many disc diameters
 *  (reference: glow.scale = 7× the sun mesh). */
export const SUN_GLOW_SCALE = 7;

/** B6 shadow darkening: the residual brightness (0..1) of a ring quad in the
 *  planet's own shadow, and of a planet-disc pixel under the ring shadow.
 *  Not fully black — a little in-scatter/ring-shine keeps it readable. */
export const RING_SHADOW_STRENGTH = 0.32;

/** Ecliptic-grid AU rings (the reference's GRID_AU) + 12 bearing spokes. */
export const GRID_AU: readonly number[] = Object.freeze([0.5, 1, 2, 5, 10, 20, 40]);

/** Motion-trail arc: 60° of orbital phase trailing the body (reference). */
export const TRAIL_ARC_DEG = 60;

/** Milky Way display-opacity easing time constant, ms (the reference eases
 *  ~6%/frame at 60fps ≈ a 270ms constant; isolated draws snap). */
export const MILKY_WAY_EASE_MS = 270;

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
 *  (AU is a fixed domain convention, like knots for vessels). AU decimals
 *  follow the reference's fmtAU (2026-07-18): ≥10 → 1dp, ≥1 → 2dp,
 *  <1 → 3dp — so "4721.0 AU" at far range, "1.50 AU" near Earth. */
export function fmtSpaceDistance(meters: number): string {
  if (meters < 0.01 * AU_M) return fmtKm(meters / 1000, 0);
  const au = meters / AU_M;
  return `${au.toFixed(au >= 10 ? 1 : au >= 1 ? 2 : 3)} AU`;
}

/**
 * Re-anchor a camera pose on a new focus body WITHOUT moving the camera:
 * the released/re-focused camera keeps its world position exactly (above
 * the min-distance floor). This is the release gesture's math (click empty
 * space → back to the Sun frame) and cancelFlight's re-anchor — and the
 * FOLLOW contract's inverse: while focused, camPos = focusPos + dir·dist,
 * so the camera translates by exactly the body's per-frame motion delta.
 */
export function reanchorPose(camPos: Vec3, focusPos: Vec3, minDistM: number): { dir: Vec3; dist: number } {
  const rel = sub(camPos, focusPos);
  return { dir: norm3(rel), dist: Math.max(len3(rel), minDistM) };
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

// ── the body info card (reference #bodycard, 2026-07-18) ────────────────────

export interface BodyCardInfo {
  id: string;
  name: string;
  /** "STAR" | "PLANET" | "MOON · orbits <parent>" */
  typeLabel: string;
  /** display-ready rows in the reference's order: day length, orbit
   *  period, distance, radius, axial tilt. REAL values only — rotation
   *  from the IAU model, distances live from the ephemeris at timeMs,
   *  lengths through the units formatters. */
  rows: { label: string; value: string }[];
}

function fmtDayLength(id: string): string {
  if (!hasRotationModel(id)) return "—";
  const p = rotationPeriodDays(id as RotationBodyId);
  const retro = rotationRateDegPerDay(id as RotationBodyId) < 0 ? " · retrograde" : "";
  return (p < 2 ? `${(p * 24).toFixed(1)} h` : `${p.toFixed(1)} days`) + retro;
}

function fmtOrbitPeriod(days: number): string {
  if (days < 2) return `${(days * 24).toFixed(1)} h`;
  const yr = days / 365.25;
  if (yr < 1) return `${days < 100 ? days.toFixed(1) : String(Math.round(days))} days`;
  return yr < 20 ? `${yr.toFixed(1)} years` : `${Math.round(yr)} years`;
}

/**
 * Display-ready card data for a registry body at the sim-clock instant.
 * Pure (tested against fact-sheet values: Jupiter ≈ 9.9 h, Venus 243 days
 * retrograde, the Moon ~384,400 km from Earth). Distance rows are LIVE
 * true ephemeris distances — never catalog averages.
 */
export function bodyCardInfo(id: string, timeMs: number): BodyCardInfo | null {
  const isPlanet = (BODY_ORDER as readonly string[]).includes(id);
  const isCurated = (MOON_IDS as readonly string[]).includes(id);
  if (!isPlanet && !isCurated) return null;
  const name = isPlanet ? NAMES[id as BodyId] : MOON_NAME[id as MoonId];
  const helio = new Map(solarSystemState(timeMs).map((b) => [b.id as string, b.helioAu]));
  let typeLabel: string;
  let distRow: string;
  let orbitRow: string;
  let radiusKm: number;
  if (id === "sun") {
    typeLabel = "STAR";
    distRow = "—";
    orbitRow = "—";
    radiusKm = BODY_RADIUS_M.sun / 1000;
  } else if (id === "moon") {
    typeLabel = "MOON · orbits Earth";
    const e = helio.get("earth")!;
    const m = helio.get("moon")!;
    const dM = Math.hypot(m.x - e.x, m.y - e.y, m.z - e.z) * AU_M;
    distRow = `${fmtKm(dM / 1000, 0)} from Earth`;
    orbitRow = `${fmtOrbitPeriod(PLANET_ORBIT_PERIOD_DAYS.moon)} · around Earth`;
    radiusKm = BODY_RADIUS_M.moon / 1000;
  } else if (isCurated) {
    const el = MOONS[id as MoonId];
    const parentName = NAMES[el.parent as BodyId];
    typeLabel = `MOON · orbits ${parentName}`;
    const o = moonLocalOffsetEclM(id as MoonId, timeMs);
    distRow = `${fmtKm(Math.hypot(o.x, o.y, o.z) / 1000, 0)} from ${parentName}`;
    orbitRow = `${fmtOrbitPeriod(moonOrbitPeriodDays(id as MoonId))} · around ${parentName}`;
    radiusKm = el.radiusKm;
  } else {
    typeLabel = "PLANET";
    const p = helio.get(id)!;
    distRow = `${fmtSpaceDistance(Math.hypot(p.x, p.y, p.z) * AU_M)} from Sun`;
    orbitRow = fmtOrbitPeriod(PLANET_ORBIT_PERIOD_DAYS[id as Exclude<BodyId, "sun">]);
    radiusKm = BODY_RADIUS_M[id as BodyId] / 1000;
  }
  // stated frame: tilt is measured to the ECLIPTIC (fact sheets quote the
  // orbit plane — within ~2-3° for the planets, and honesty demands the
  // frame be named rather than approximated silently)
  const tilt = hasRotationModel(id)
    ? `${spinObliquityDeg(axisEclOfDate(id as RotationBodyId, timeMs), rotationRateDegPerDay(id as RotationBodyId)).toFixed(1)}° to ecliptic`
    : "—";
  return {
    id,
    name,
    typeLabel,
    rows: [
      { label: "day length", value: fmtDayLength(id) },
      { label: "orbit period", value: orbitRow },
      { label: "distance", value: distRow },
      { label: "radius", value: fmtKm(radiusKm, 0) },
      { label: "axial tilt", value: tilt },
    ],
  };
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
  /** Space-view visual toggles (defaults: the persisted preferences). */
  milkyWay?: boolean;
  /** Real bright-star point layer (Yale BSC). Default: the persisted pref (ON). */
  stars?: boolean;
  eclipticGrid?: boolean;
  /** Lock horizon — polar clamp keeps the view from swinging under the
   *  ecliptic (default: the persisted pref, ON). OFF only widens the clamp. */
  lockHorizon?: boolean;
  motionTrails?: boolean;
  bodyLabels?: boolean;
  /** B6 realistic (physically-lit) mode. Default: the persisted pref (ON).
   *  OFF = even-lit inspection mode (full-bright, no terminator). */
  realisticLighting?: boolean;
  /** the sim clock's current rate for the footer's "time ×N" (0 = paused).
   *  The frame never owns time — it reads the ONE clock's rate for display. */
  getTimeRate?: () => number;
  /** focus notifications for the parent's body card: a body id when a
   *  fly-to targets it, null on release/fly-home (card closes). */
  onFocusBody?: (id: string | null) => void;
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
  setMilkyWay(on: boolean): void;
  /** toggle the real bright-star point layer (Yale BSC). */
  setStars(on: boolean): void;
  setEclipticGrid(on: boolean): void;
  /** Lock horizon: live polar-clamp toggle (ON = never under the ecliptic). */
  setLockHorizon(on: boolean): void;
  setMotionTrails(on: boolean): void;
  setBodyLabels(on: boolean): void;
  /** B6: toggle realistic (physically-lit) mode. ON (default) = real
   *  terminators/phases/eclipse+ring shadows under the sim clock; OFF =
   *  even-lit inspection mode (full-bright, no terminator). The CELESTIAL-panel
   *  row that drives this is a parent datamap follow-up. */
  setRealisticLighting(on: boolean): void;
  /** DEBUG/DRIVE seam: project a sky direction (RA/Dec, J2000 degrees) into
   *  container CSS px using the live camera — the SAME transform stars +
   *  panorama register through. Proves a known star lands at its true place. */
  projectSkyRaDec(raDeg: number, decDeg: number): { x: number; y: number; front: boolean };
  /** display-ready card data for a body at the frame's current instant. */
  getBodyCard(id: string): BodyCardInfo | null;
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
  /** B6 eclipse/umbra: direct-sunlight factor 0..1 from real cone geometry
   *  (1 = fully sunlit, 0 = total umbra). < 1 ⇒ this body is eclipsed by its
   *  parent at the sim-clock instant (a lunar eclipse for the Moon). Pure
   *  geometry — reported regardless of the realistic-lighting toggle. */
  sunlitFactor: number;
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
  /** ZOOM-TO-CURSOR/PAN: the look-at offset from the focus body's centre,
   *  body-relative meters (0 = centred orbit). Grows toward the cursor side on
   *  a wheel zoom-in and traverses the surface on a pan drag. */
  focusOffM: Vec3;
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
  /** visual-upgrade render truth (2026-07-18): what the frame actually
   *  drew this frame — the drive asserts against these, never a recompute. */
  milkyWay: { on: boolean; opacity: number; target: number };
  /** real Yale BSC bright-star field render truth (galaxy detail 2026-07-19):
   *  toggle, eased opacity, fade target, whether the catalog is resident, its
   *  star count, how many points were drawn this frame, and the layer cost. */
  starfield: {
    on: boolean;
    opacity: number;
    target: number;
    loaded: boolean;
    count: number;
    rendered: number;
    msLast: number;
  };
  eclipticGrid: boolean;
  motionTrails: boolean;
  bodyLabels: boolean;
  /** B6: realistic (physically-lit) vs even-lit inspection mode. */
  realisticLighting: boolean;
  /** resident texture facts + the perf maxima the report needs. */
  textures: {
    loaded: number;
    ids: string[];
    moonTier: string;
    bytes: number;
    maxChunkMs: number;
    mainThreadDecode: boolean;
    spriteBuildMsMax: number;
    lutBuildMsMax: number;
    skyMsLast: number;
    /** body ids whose sprite composed from a texture THIS frame. */
    texturedThisFrame: string[];
  };
  /** clickable label text boxes as drawn (label→fly-to hit targets). */
  labelRects: { id: string; x: number; y: number; w: number; h: number }[];
  anchor: (EarthAnchor & { subLatDeg: number; subLonDeg: number }) | null;
  bodies: SpaceFrameBodyState[];
  /** B4: the focused-close Moon surface patch (real LROC tiles). Inactive
   *  when the camera is far — zero cost, nothing streamed. */
  moonPatch: {
    active: boolean;
    /** full streamed tier drawn this frame (vs the motion tier). */
    full: boolean;
    /** resident tile mosaic: native level, tiles composited, RGBA bytes. */
    mosaicZ: number | null;
    mosaicTiles: number;
    tileBytes: number;
    /** worst single main-thread patch-render band, ms (the ≤16ms proof). */
    patchMsMax: number;
    /** worst single main-thread tile-readback band, ms. */
    tileChunkMsMax: number;
    /** the credit line is shown (a mosaic is on screen). */
    credited: boolean;
  };
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
    // trails share the sampled lines — either consumer keeps sampling live
    if (orbitJob || disposed || (!orbitsOn && !trailsOn) || !nextOrbitBody()) return;
    orbitJob = setTimeout(() => {
      orbitJob = null;
      if (disposed || (!orbitsOn && !trailsOn)) return;
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

  /** project one layout-space point; returns null when behind the camera. */
  function projTo(
    wx: number, wy: number, wz: number,
    camPos: Vec3, basis: CamBasis, k: number, cx: number, cy: number,
  ): { x: number; y: number; depth: number } | null {
    const rx = wx - camPos.x;
    const ry = wy - camPos.y;
    const rz = wz - camPos.z;
    const depth = rx * basis.f.x + ry * basis.f.y + rz * basis.f.z;
    if (depth <= 0) return null;
    return {
      x: cx + (k * (rx * basis.r.x + ry * basis.r.y + rz * basis.r.z)) / depth,
      y: cy - (k * (rx * basis.u.x + ry * basis.u.y + rz * basis.u.z)) / depth,
      depth,
    };
  }

  /**
   * The Milky Way backdrop: the CC-BY panorama mapped through the REAL
   * galactic frame onto the current camera basis, cached against the basis
   * (pure zoom reuses it; only orbit drags repaint), upscaled to the canvas,
   * plus the reference's additive glow pass. Opacity = the eased fade.
   *
   * GALAXY DETAIL (2026-07-19): the working resolution is raised (≤900 px
   * wide, was ≤480) and the panorama is sampled BILINEARLY with a mild
   * exposure lift — the band no longer reads as soft wallpaper when it fills
   * the screen far out. The 8k source decodes at 4096×2048 now (spaceAssets
   * DECODE_CAPS), so there is real texel detail to interpolate. Cost is paid
   * only on orbit/resize (cached against the basis), well within the frame
   * budget; skyMsLast reports it.
   */
  function drawMilkyWay(w: number, h: number, basis: CamBasis): void {
    if (!ctx || milkyShown <= 0.003) return;
    const tex = texman.getMilkyWay();
    if (!tex) return;
    const skyW = Math.min(900, Math.max(240, Math.round(w / 1.6)));
    const skyH = Math.max(150, Math.round((skyW * h) / w));
    if (!skyRays || skyRays.w !== skyW || skyRays.h !== skyH) {
      skyRays = { w: skyW, h: skyH, rays: buildSkyRays(skyW, skyH, fovDeg) };
      skyReadyKey = "";
      skyWorkKey = "";
    }
    if (!galBasis) galBasis = galacticBasisEcl(timeMs);
    const key =
      `${skyW}x${skyH}|${q2(basis.f.x)},${q2(basis.f.y)},${q2(basis.f.z)}` +
      `|${q2(basis.u.x)},${q2(basis.u.y)},${q2(basis.u.z)}`;
    if (!skyCanvas || skyCanvas.width !== skyW || skyCanvas.height !== skyH) {
      skyCanvas = document.createElement("canvas");
      skyCanvas.width = skyW;
      skyCanvas.height = skyH;
      skyReadyKey = "";
      skyWorkKey = "";
    }
    if (skyReadyKey !== key) {
      // (re)start the build for this camera basis if it changed mid-flight
      if (skyWorkKey !== key) {
        skyWorkKey = key;
        skyWorkRow = 0;
        if (!skyWork || skyWork.length !== skyW * skyH * 4) skyWork = new Uint8ClampedArray(skyW * skyH * 4);
      }
      const t0 = performance.now();
      const BUDGET_MS = 8; // per-frame slice — keeps the sky task well under 16 ms
      const bandRows = Math.max(4, Math.floor(24_000 / skyW));
      while (skyWorkRow < skyH && performance.now() - t0 < BUDGET_MS) {
        const end = Math.min(skyH, skyWorkRow + bandRows);
        renderSkyToBuffer(skyRays.rays, skyW, skyH, basis.f, basis.r, basis.u, galBasis, tex, skyWork!, true, 1.12, skyWorkRow, end);
        skyWorkRow = end;
      }
      skyMsLast = performance.now() - t0;
      if (skyWorkRow >= skyH) {
        const scc = skyCanvas.getContext("2d")!;
        scc.putImageData(new ImageData(new Uint8ClampedArray(skyWork!), skyW, skyH), 0, 0);
        skyReadyKey = key;
        skyWorkKey = "";
        skyRenderPending = false;
      } else {
        skyRenderPending = true; // keep drawing until the build finishes
      }
    }
    // composite whatever completed buffer we have (during a fast orbit this may
    // be the previous basis for a frame or two — an invisible backdrop lag)
    if (skyReadyKey) {
      ctx.save();
      ctx.globalAlpha = milkyShown;
      ctx.drawImage(skyCanvas, 0, 0, w, h);
      // additive second pass — the reference's galaxyGlow (the band glows
      // instead of reading as dim wallpaper)
      ctx.globalCompositeOperation = "lighter";
      ctx.globalAlpha = milkyShown * 0.55;
      ctx.drawImage(skyCanvas, 0, 0, w, h);
      ctx.restore();
    }
  }

  // ── GALAXY / STARFIELD: the real Yale BSC bright stars ─────────────────────

  /** Lazily fetch the bundled star catalog once (space-view only — this mount
   *  runs only when the user has left Earth). Degrades to panorama-only. */
  function ensureStars(): void {
    if (starLoadStarted || disposed) return;
    starLoadStarted = true;
    void loadStarCatalog(undefined, starAbort.signal).then((cat) => {
      if (disposed || !cat) return;
      starCat = cat;
      const n = cat.count;
      starSizeArr = new Float32Array(n);
      starAlphaArr = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        starSizeArr[i] = starSizePx(cat.mag[i]);
        starAlphaArr[i] = starIntensity(cat.mag[i]);
      }
      computeStarEcl(timeMs);
      kick(); // repaint now that the real sky is resident
    });
  }

  /** Precompute each star's direction in the scene's ecliptic-of-date frame
   *  (registers with the panorama). Recomputed only if the sim epoch drifts a
   *  year — the obliquity change is otherwise sub-pixel. */
  function computeStarEcl(t: number): void {
    if (!starCat) return;
    const n = starCat.count;
    if (!starEclDir || starEclDir.length !== n * 3) starEclDir = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const e = eqToEclDir(
        { x: starCat.dirEq[i * 3], y: starCat.dirEq[i * 3 + 1], z: starCat.dirEq[i * 3 + 2] },
        t,
      );
      starEclDir[i * 3] = e.x;
      starEclDir[i * 3 + 1] = e.y;
      starEclDir[i * 3 + 2] = e.z;
    }
    starEclEpochMs = t;
    starKey = ""; // directions changed → invalidate the cached point layer
  }

  /**
   * Draw the real bright-star field. Points are projected gnomonically at
   * their true sky directions onto a full-resolution offscreen canvas cached
   * against the camera basis (stars are at infinity → they move only on
   * orbit/resize, never on zoom — the panorama's caching discipline), then
   * composited ADDITIVELY over the panorama. Size + brightness come from the
   * real V magnitude, color from the real K temperature. The brightest named
   * stars carry crisp live labels far out (labels toggle honored).
   */
  function drawStarfield(w: number, h: number, basis: CamBasis, k: number, cx: number, cy: number): void {
    if (!ctx || !starsOn || starsShown <= 0.003 || !starCat || !starEclDir || !starSizeArr || !starAlphaArr) {
      starRendered = 0;
      return;
    }
    if (Math.abs(timeMs - starEclEpochMs) > 400 * 86_400_000) computeStarEcl(timeMs);
    const key =
      `${w}x${h}|${Math.round(k)}|${q2(basis.f.x)},${q2(basis.f.y)},${q2(basis.f.z)}` +
      `|${q2(basis.u.x)},${q2(basis.u.y)},${q2(basis.u.z)}`;
    if (!starCanvas || starCanvas.width !== w || starCanvas.height !== h) {
      starCanvas = document.createElement("canvas");
      starCanvas.width = w;
      starCanvas.height = h;
      starKey = "";
    }
    if (starKey !== key) {
      const t0 = performance.now();
      const scc = starCanvas.getContext("2d")!;
      scc.clearRect(0, 0, w, h);
      const n = starCat.count;
      const fx = basis.f.x, fy = basis.f.y, fz = basis.f.z;
      const rx = basis.r.x, ry = basis.r.y, rz = basis.r.z;
      const ux = basis.u.x, uy = basis.u.y, uz = basis.u.z;
      let drew = 0;
      for (let i = 0; i < n; i++) {
        const dx = starEclDir[i * 3];
        const dy = starEclDir[i * 3 + 1];
        const dz = starEclDir[i * 3 + 2];
        const fwd = dx * fx + dy * fy + dz * fz;
        if (fwd <= 1e-6) continue;
        const sx = cx + (k * (dx * rx + dy * ry + dz * rz)) / fwd;
        const sy = cy - (k * (dx * ux + dy * uy + dz * uz)) / fwd;
        if (sx < -3 || sy < -3 || sx > w + 3 || sy > h + 3) continue;
        const size = starSizeArr[i];
        const ri = i * 3;
        scc.fillStyle = `rgba(${starCat.rgb[ri]},${starCat.rgb[ri + 1]},${starCat.rgb[ri + 2]},${starAlphaArr[i]})`;
        const side = size < 1.3 ? 1 : size < 2 ? 2 : Math.round(size);
        scc.fillRect(sx - side / 2, sy - side / 2, side, side);
        if (size > 2.1) {
          // the ~few dozen brightest get a faint round halo → they read as stars
          scc.globalAlpha = starAlphaArr[i] * 0.3;
          scc.beginPath();
          scc.arc(sx, sy, size * 1.4, 0, Math.PI * 2);
          scc.fill();
          scc.globalAlpha = 1;
        }
        drew++;
      }
      starRendered = drew;
      starKey = key;
      starMsLast = performance.now() - t0;
    }
    ctx.save();
    ctx.globalAlpha = starsShown;
    ctx.globalCompositeOperation = "lighter";
    ctx.drawImage(starCanvas, 0, 0);
    ctx.restore();
    // brightest named stars: crisp live labels (cheap — ≤~22), far-out only
    if (labelsOn && starsShown > 0.55 && starCat.names.size) {
      ctx.save();
      ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = `rgba(150,166,190,${(0.7 * starsShown).toFixed(3)})`;
      const sd = starEclDir;
      const c2 = ctx;
      starCat.names.forEach((name, i) => {
        if (i >= starCat!.count) return;
        const dx = sd[i * 3];
        const dy = sd[i * 3 + 1];
        const dz = sd[i * 3 + 2];
        const fwd = dx * basis.f.x + dy * basis.f.y + dz * basis.f.z;
        if (fwd <= 1e-6) return;
        const sx = cx + (k * (dx * basis.r.x + dy * basis.r.y + dz * basis.r.z)) / fwd;
        const sy = cy - (k * (dx * basis.u.x + dy * basis.u.y + dz * basis.u.z)) / fwd;
        if (sx < 0 || sy < 0 || sx > w || sy > h) return;
        c2.fillText(name, sx + 5, sy - 4);
      });
      ctx.restore();
    }
  }

  /** Ecliptic grid: AU range rings + 12 bearing spokes (reference, default
   *  OFF), drawn through the SAME layout compression as everything else,
   *  with dim AU labels at each ring's reference bearing. */
  function drawEclipticGrid(
    c2d: CanvasRenderingContext2D,
    sunPos: Vec3,
    camPos: Vec3,
    basis: CamBasis,
    k: number,
    cx: number,
    cy: number,
    w: number,
    h: number,
  ): void {
    c2d.strokeStyle = "rgba(30,42,68,0.85)";
    c2d.lineWidth = 1;
    c2d.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    const seg = 96;
    for (const a of GRID_AU) {
      const rr = compressDistance(a * AU_M, scaleSt.c);
      c2d.beginPath();
      let open = false;
      let labelPt: { x: number; y: number } | null = null;
      for (let i = 0; i <= seg; i++) {
        const t = (i / seg) * Math.PI * 2;
        const p = projTo(
          sunPos.x + Math.cos(t) * rr, sunPos.y + Math.sin(t) * rr, sunPos.z,
          camPos, basis, k, cx, cy,
        );
        if (!p) { open = false; continue; }
        if (!open) { c2d.moveTo(p.x, p.y); open = true; } else c2d.lineTo(p.x, p.y);
        if (i === 0) labelPt = p;
      }
      c2d.stroke();
      if (labelPt && labelPt.x > -20 && labelPt.x < w && labelPt.y > 0 && labelPt.y < h) {
        c2d.fillStyle = "rgba(61,74,102,0.9)";
        c2d.fillText(`${a} AU`, labelPt.x + 4, labelPt.y - 4);
      }
    }
    // bearing spokes: radial 3D segments stay straight under projection
    // AND under the radial compression map — endpoints suffice
    const r0 = compressDistance(0.3 * AU_M, scaleSt.c);
    const r1 = compressDistance(45 * AU_M, scaleSt.c);
    c2d.beginPath();
    for (let i = 0; i < 12; i++) {
      const t = i * 30 * DEG;
      const p0 = projTo(sunPos.x + Math.cos(t) * r0, sunPos.y + Math.sin(t) * r0, sunPos.z, camPos, basis, k, cx, cy);
      const p1 = projTo(sunPos.x + Math.cos(t) * r1, sunPos.y + Math.sin(t) * r1, sunPos.z, camPos, basis, k, cx, cy);
      if (!p0 || !p1) continue;
      c2d.moveTo(p0.x, p0.y);
      c2d.lineTo(p1.x, p1.y);
    }
    c2d.stroke();
  }

  /**
   * Motion trails (reference "Motion trails", separate toggle): a 60°
   * trailing arc of each body's REAL sampled orbit, brighter than the
   * orbit line, ending exactly on the body. Trailing = the PAST arc —
   * direction of travel reads at a glance.
   */
  function drawMotionTrails(
    c2d: CanvasRenderingContext2D,
    pos: Record<string, Vec3>,
    camPos: Vec3,
    basis: CamBasis,
    k: number,
    cx: number,
    cy: number,
  ): void {
    c2d.lineWidth = 1.5;
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
        const pdx = pp.x - camPos.x;
        const pdy = pp.y - camPos.y;
        const pdz = pp.z - camPos.z;
        const pDist = Math.hypot(pdx, pdy, pdz);
        if (pDist > 0 && (entry.line.maxRadiusM / pDist) * k < 2) continue;
        px = pp.x;
        py = pp.y;
        pz = pp.z;
        verts = entry.line.pts;
      } else {
        if (entry.layoutC !== scaleSt.c) {
          entry.layout = layoutOrbitPolyline(entry.line.pts, scaleSt.c, entry.layout ?? undefined);
          entry.layoutC = scaleSt.c;
        }
        verts = entry.layout!;
      }
      const n = verts.length / 3;
      const periodMs = entry.line.periodDays * 86_400_000;
      const f = ((((timeMs - entry.line.sampledAtMs) / periodMs) % 1) + 1) % 1;
      const count = Math.max(6, Math.round((n * TRAIL_ARC_DEG) / 360));
      const endIdx = Math.floor(f * n);
      c2d.strokeStyle = pathStroke(def.color, 0.7);
      c2d.beginPath();
      let open = false;
      for (let s = 0; s <= count; s++) {
        const j = ((((endIdx - count + s) % n) + n) % n) * 3;
        const p = projTo(px + verts[j], py + verts[j + 1], pz + verts[j + 2], camPos, basis, k, cx, cy);
        if (!p) { open = false; continue; }
        if (!open) { c2d.moveTo(p.x, p.y); open = true; } else c2d.lineTo(p.x, p.y);
      }
      // land the trail exactly on the body (sampling-granularity gap)
      const bp = pos[def.id];
      if (bp && open) {
        const p = projTo(bp.x, bp.y, bp.z, camPos, basis, k, cx, cy);
        if (p) c2d.lineTo(p.x, p.y);
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
  // ZOOM-TO-CURSOR + PAN (2026-07-19): the LOOK-AT is focus-body-centre +
  // focusOff (body-relative meters). focusOff = 0 ⇒ centred (the classic
  // orbit). Wheel-zoom eases dist toward distTarget and pulls focusOff toward
  // focusOffTarget (the cursor's surface point); pan writes focusOff directly.
  // Reset to 0 on every focus change so flights / seam / follow are untouched.
  let focusOff: Vec3 = v3(0, 0, 0);
  let focusOffTarget: Vec3 = v3(0, 0, 0);
  let distTarget = dist; // smooth-zoom (wheel) eases dist → distTarget
  let flight: FlightState | null = null;
  let renderMsLast = 0;
  let lastState: SpaceFrameState | null = null;

  // recenter throttle (hemisphere alignment; only while the map is visible)
  let lastRecenterAt = 0;
  let lastRecenterLat = seam0.centerLatDeg;
  let lastRecenterLon = seam0.centerLonDeg;

  // ── sprite cache: per-body shaded sphere, keyed on quantized size+light ──
  const spriteCache = new Map<string, HTMLCanvasElement>();

  // ── SPACE VIEW VISUAL UPGRADE state (2026-07-18) ──
  let milkyOn = opts.milkyWay ?? getMilkyWayPref();
  let starsOn = opts.stars ?? getStarsPref();
  let gridOn = opts.eclipticGrid ?? getEclipticGridPref();
  let trailsOn = opts.motionTrails ?? getMotionTrailsPref();
  let labelsOn = opts.bodyLabels ?? getBodyLabelsPref();
  // Lock horizon (2026-07-19): polar clamp state — see polarClampDots.
  let lockHorizon = opts.lockHorizon ?? getLockHorizonPref();
  // B6 universal lighting: ON = physically-lit (terminators/phases/eclipse+
  // ring shadows); OFF = even-lit inspection (full-bright, no terminator).
  let realistic = opts.realisticLighting ?? getRealisticLightingPref();
  let ringBands: RingBand[] | null = null;
  let spriteBuildMsMax = 0;
  let lutBuildMsMax = 0;
  let texturedIds: string[] = [];
  // milky way display opacity (eased toward the fade target on continuous
  // draws; isolated draws snap — deterministic either way)
  let milkyShown = 0;
  let milkyEasePending = false;
  let lastDrawAt = 0;
  let skyCanvas: HTMLCanvasElement | null = null;
  let skyRays: { w: number; h: number; rays: Float32Array } | null = null;
  let galBasis: GalacticBasis | null = null;
  let skyMsLast = 0;
  // progressive panorama render (galaxy detail 2026-07-19): the sky is built
  // at a high working resolution in row bands across frames so no single sky
  // repaint exceeds the budget; skyReadyKey holds the completed buffer's basis
  // key, skyWork accumulates the in-progress one, skyRenderPending keeps the
  // rAF alive until a build finishes.
  let skyWork: Uint8ClampedArray | null = null;
  let skyWorkRow = 0;
  let skyWorkKey = "";
  let skyReadyKey = "";
  let skyRenderPending = false;
  // GALAXY / STARFIELD (2026-07-19): the real Yale BSC bright stars. Loaded
  // lazily (space-view only), the point layer cached against the camera basis
  // like the panorama (stars are at infinity → they never move on zoom, only
  // on orbit/resize), composited additively over the panorama, and freed on
  // exit. eclDir is the per-star ecliptic-frame direction (recomputed only if
  // the sim epoch drifts a year — obliquity change is otherwise sub-pixel).
  let starCat: StarCatalog | null = null;
  let starLoadStarted = false;
  const starAbort = new AbortController();
  let starEclDir: Float32Array | null = null;
  let starSizeArr: Float32Array | null = null;
  let starAlphaArr: Float32Array | null = null;
  let starEclEpochMs = 0;
  let starCanvas: HTMLCanvasElement | null = null;
  let starKey = "";
  let starMsLast = 0;
  let starsShown = 0;
  let starEasePending = false;
  let starRendered = 0;
  // live camera capture (drive seam for projectSkyRaDec)
  let lastBasis: CamBasis | null = null;
  let lastK = 0;
  let lastCx = 0;
  let lastCy = 0;
  let glowSprite: HTMLCanvasElement | null = null;
  // texture LUT cache: per body, rebuilt only when the sprite size steps
  // or the axis moves in camera space (camera orbits) — a body SPINNING
  // under time warp only re-composes (no trig; textureSphere.ts header)
  const lutCache = new Map<string, { key: string; lut: SphereLUT }>();
  const q2 = (n: number): number => Math.round(n / 0.01) * 0.01;

  // progressive textures: nothing was fetched before this mount ran (B0
  // discipline — the map never pays); each texture that lands repaints
  const texman: SpaceTextureManager = createSpaceTextureManager();
  texman.onUpdate(() => {
    spriteCache.clear();
    ringBands = null;
    kick();
  });
  texman.start();

  // ── B4/B5: focused-close rocky bodies — real Trek tiles → surface patch ──
  // ONE tile manager per body (the B4 Moon + the B5 rocky planets Mars/Mercury/
  // Venus). Only one body is ever focused-close at a time (you cannot be within
  // a few radii of two bodies at once), so a single set of patch buffers below
  // is tagged with the body it belongs to (patchBodyId) and evicted on a body
  // switch. Each manager is built lazily from the body's own Trek scheme.
  const bodyTileManagers = new Map<string, MoonTileManager>();
  function tileManagerFor(bodyId: string): MoonTileManager | null {
    const tb = trekBody(bodyId);
    if (!tb) return null;
    let m = bodyTileManagers.get(bodyId);
    if (!m) {
      m = createMoonTileManager({ scheme: tb.scheme });
      // a fresher mosaic → repaint and let the pose key (its mosKey changed)
      // trigger a rebuild; DON'T blank mpFull — keep the current crisp buffer on
      // screen until the sharper one is ready (no mid-motion resolution drop)
      m.onUpdate(() => { kick(); });
      bodyTileManagers.set(bodyId, m);
    }
    return m;
  }
  const RAD = 180 / Math.PI;
  interface MoonPatchBuf {
    key: string;
    canvas: HTMLCanvasElement;
    img: ImageData;
    bufW: number;
    bufH: number;
    bx: number;
    by: number;
    bw: number;
    bh: number;
    view: MoonSurfaceView;
    base: TexImage;
    detail: DetailOverlay | null;
    row: number;
    ready: boolean;
    /** unit camera→body direction the buffer was rendered for (motion staleness
     *  test: keep showing this crisp buffer while the camera stays within
     *  MOON_PATCH_MOTION_KEEP_COS of it). */
    nCam: Vec3;
  }
  let mpFast: { canvas: HTMLCanvasElement; img: ImageData; bufW: number; bufH: number } | null = null;
  let mpFull: MoonPatchBuf | null = null;
  let mpBuilding: MoonPatchBuf | null = null;
  let mpTimer: ReturnType<typeof setTimeout> | null = null;
  let moonPatchOn = false;
  let moonPatchDrewThisFrame = false;
  let moonPatchFullThisFrame = false;
  let moonPatchMsMax = 0;
  let moonCreditOn = false;
  // which body the current patch buffers belong to, and its on-screen credit
  let patchBodyId: string | null = null;
  let patchCreditText = "";

  /** buffer dims fitting MOON_PATCH_*_LONG_PX on the bbox's long side. */
  function patchBufDims(bw: number, bh: number, longPx: number): { bufW: number; bufH: number } {
    const long = Math.max(bw, bh);
    const s = long > longPx ? longPx / long : 1;
    return { bufW: Math.max(2, Math.round(bw * s)), bufH: Math.max(2, Math.round(bh * s)) };
  }

  /** build a MoonSurfaceView mapping a bufW×bufH buffer onto canvas bbox. */
  function patchView(
    bufW: number, bufH: number, bx: number, by: number, bw: number, bh: number,
    camPos: Vec3, center: Vec3, R: number, basis: CamBasis, k: number, cx: number, cy: number,
    X: Vec3, Y: Vec3, Z: Vec3, wDeg: number, sun: Vec3, texLonOffsetDeg: number,
    shadowFactor: number,
  ): MoonSurfaceView {
    const stepX = bw / bufW;
    const stepY = bh / bufH;
    return {
      bw: bufW, bh: bufH,
      originX: bx + stepX / 2, originY: by + stepY / 2, stepX, stepY,
      cx, cy, k,
      r: basis.r, u: basis.u, f: basis.f,
      cam: camPos, center, radius: R,
      X, Y, Z, wDeg, sun, texLonOffsetDeg,
      // B6: realistic OFF ⇒ full-bright patch; the eclipse scalar matches the
      // far sprite so the terminator/eclipse read consistently across zoom.
      fullBright: !realistic,
      shadowFactor,
    };
  }

  function pumpMoonPatch(): void {
    mpTimer = null;
    if (!mpBuilding || disposed) return;
    const b = mpBuilding;
    const rows = Math.max(4, Math.round(MOON_PATCH_CHUNK_PX / b.bufW));
    const t0 = performance.now();
    const end = Math.min(b.bufH, b.row + rows);
    renderMoonSurfaceRows(b.view, b.base, b.detail, b.img.data, b.row, end);
    const ms = performance.now() - t0;
    if (ms > moonPatchMsMax) moonPatchMsMax = ms;
    b.row = end;
    if (b.row >= b.bufH) {
      b.canvas.getContext("2d")!.putImageData(b.img, 0, 0);
      b.ready = true;
      mpFull = b;
      mpBuilding = null;
      kick();
      return;
    }
    mpTimer = setTimeout(pumpMoonPatch, 0);
  }

  /** B6 eclipse/umbra factor for a body: a moon in its PARENT's real shadow
   *  cone darkens (lunar eclipse = Moon in Earth's umbra; a Jovian moon in
   *  Jupiter's) — everything else stays 1. Pure GEOMETRY from TRUE ecliptic
   *  positions (the physical cone, never the compressed layout) and the real
   *  Sun distance, so the darkening turns on and off at the correct sim-clock
   *  instants. Not gated on the realistic toggle — the shading paths already
   *  ignore the factor in full-bright inspection mode — so getState can report
   *  the honest eclipse geometry regardless of the display toggle. */
  function eclipseFactor(id: string, posT: Record<string, Vec3>): number {
    const parentId = defById.get(id)?.parentId;
    if (!parentId) return 1;
    const bc = posT[id];
    const pc = posT[parentId];
    const sc = posT[sunDef.id];
    if (!bc || !pc || !sc) return 1;
    const pRel = sub(bc, pc);
    const sunV = sub(sc, pc);
    const sunDist = len3(sunV);
    if (sunDist <= 0) return 1;
    const sunDir = { x: sunV.x / sunDist, y: sunV.y / sunDist, z: sunV.z / sunDist };
    return shadowConeFactor(pRel, sunDir, radiusM(parentId), radiusM(sunDef.id), sunDist).factor;
  }

  /** Draw a focused-close rocky body (Moon/Mars/Mercury/Venus) as a real-
   *  imagery perspective patch. Returns true when it handled the body (caller
   *  skips the disc sprite). Body-generic: geometry from radiusM(bodyId), pose
   *  from pos[bodyId], orientation from rotation.ts for `bodyId`, tiles from the
   *  body's own Trek manager — the exact B4 path, parameterised by bodyId.
   *  `shadowFactor` is the body's B6 eclipse/umbra scalar (1 = fully sunlit). */
  function drawBodyPatch(
    bodyId: string,
    d: { p: ProjectedPoint; layoutDistM: number },
    sunPos: Vec3, pos: Record<string, Vec3>, camPos: Vec3, basis: CamBasis,
    k: number, cx: number, cy: number, w: number, h: number,
    shadowFactor: number,
  ): boolean {
    if (!ctx) return false;
    const mgr = tileManagerFor(bodyId);
    if (!mgr) return false; // not a Trek-surfaced body
    const base = texman.get(bodyId);
    if (!base) return false; // no fallback texture yet → let the sprite draw
    // switching focused-close bodies: drop the prior body's patch + tiles first
    // (GPU budget — only one body's mosaic is ever resident)
    if (patchBodyId && patchBodyId !== bodyId) evictBodyPatch();
    patchBodyId = bodyId;
    const rid = bodyId as RotationBodyId;
    // source-map prime-meridian offset so the base fallback shows the SAME true
    // frame as the streamed Trek detail tiles (0 for the Moon, 180 for planets)
    const texLon = textureLonOffsetDeg(bodyId);
    const R = radiusM(bodyId);
    const center = pos[bodyId];
    const distC = Math.max(R * 1.0001, d.layoutDistM);
    // orientation (identical to the sprite path — alignment inherited): the
    // body's real IAU pole + prime meridian for THIS body (areographic for
    // Mars, etc.), exactly as the far-tier textured sprite uses
    const Z = norm3(axisEclOfDate(rid, timeMs));
    const X = norm3(equatorNodeDirEclOfDate(rid, timeMs));
    const Y = norm3(cross(Z, X));
    const wDeg = iauPrimeMeridianDeg(rid, timeMs);
    // sub-camera surface point → mosaic request (screen-matched resolution,
    // span = the geometric horizon cap + a small margin)
    const nCam = norm3(sub(camPos, center));
    const subPt = surfaceLonLat(nCam, X, Y, Z, wDeg);
    const horizonDeg = Math.acos(Math.min(1, R / distC)) * RAD;
    // true-geometry disc bbox, clamped to the canvas (decoupled from the B2
    // size slider up close — the ray-sphere IS the true sphere). Computed
    // FIRST because the tile request is sized to what this bbox actually
    // shows on screen, not the whole horizon cap.
    const rDisc = bodyDiscPx(R, distC, k) / 2;
    const bx = Math.max(0, Math.floor(d.p.x - rDisc));
    const by = Math.max(0, Math.floor(d.p.y - rDisc));
    const bw = Math.min(w, Math.ceil(d.p.x + rDisc)) - bx;
    const bh = Math.min(h, Math.ceil(d.p.y + rDisc)) - by;
    if (bw <= 0 || bh <= 0) return false;
    const sun = norm3(sub(sunPos, center));
    // request tiles at the on-screen SURFACE resolution over ONLY the patch
    // the viewport actually shows: when the disc dwarfs the viewport (skimming
    // the surface), the visible surface arc is tiny but at high resolution —
    // so cover that small arc at native tile detail (true ~100 m/px at the
    // floor), instead of the whole horizon cap at coarse detail. Span capped
    // by the geometric horizon; resolution matched to the true on-screen
    // px-per-surface-degree; the mosaic-px budget bounds the fetch.
    const pxPerSurfDeg = (k * R * DEG) / Math.max(1, distC - R);
    const bboxLongPx = Math.max(bw, bh);
    const coverHalfDeg = Math.min(
      horizonDeg,
      Math.max(MOON_PATCH_MIN_HALFSPAN_DEG, ((bboxLongPx / pxPerSurfDeg) / 2) * MOON_PATCH_COVER_MARGIN),
    );
    mgr.request(subPt.lonDeg, subPt.latDeg, pxPerSurfDeg, coverHalfDeg);
    const mos = mgr.current();
    const detail: DetailOverlay | null = mos
      ? { tex: mos.tex, lonMin: mos.lonMin, lonSpan: mos.lonSpan, latMax: mos.latMax, latSpan: mos.latSpan }
      : null;
    moonCreditOn = !!mos;
    patchCreditText = trekBody(bodyId)!.credit;
    // pose key: what makes the patch pixels change (body id included so a
    // buffer from a different body can never be mistaken for a match)
    const distBucket = Math.round(Math.log(distC) / 0.02);
    const mosKey = mos ? `${mos.z}:${mos.tiles}:${Math.round(mos.lonMin)}` : "0";
    const key =
      `${bodyId}|${q2(nCam.x)},${q2(nCam.y)},${q2(nCam.z)}|${distBucket}|${Math.round(wDeg * 2)}` +
      `|${q2(sun.x)},${q2(sun.y)},${q2(sun.z)}|${bx},${by},${bw},${bh}|${mosKey}` +
      // B6: fold the realistic toggle + eclipse scalar into the pose key so the
      // patch rebuilds when they change (a toggle flip / a moon entering umbra)
      `|L${realistic ? 1 : 0}|e${Math.round(shadowFactor * 40)}`;
    // ── DRAW: keep the crisp high-res buffer on screen, even DURING motion ───
    // The high-res buffer maps buffer pixels to the disc bbox; drawing it into
    // the CURRENT bbox keeps the silhouette exact while its surface texture is
    // at most ONE refresh-build stale (the refresh block below always keeps a
    // build in flight for the live pose, so a completed buffer is never more
    // than a fraction of a second behind the camera). DEFINITIVE FIX for "it
    // gets fuzzy when you move around it" (2026-07-19): once a full buffer
    // exists for this body we ALWAYS draw it during orbit/pan/zoom — never drop
    // to the low-res fast tier — because a stale-but-high-res image carries the
    // real detail-energy the eye reads, whereas the ≤140px fast tier (the only
    // synchronous <16ms option) is inescapably soft. B5's cos-40° cone dropped
    // to that fast tier the moment a sustained drag out-ran a build, which is
    // exactly the ~30-37% the drive measured; keeping the last full buffer
    // holds ~100% of settled detail through the motion, snapping exact on
    // settle.
    const fullUsable = !!(mpFull && mpFull.ready);
    if (fullUsable) {
      ctx.drawImage(mpFull!.canvas, bx, by, bw, bh);
      moonPatchFullThisFrame = true;
    } else {
      // bootstrap only (first approach at a body, or a fast fling past the
      // staleness cap): a synchronous bounded fast tier so SOMETHING registered
      // shows immediately while the first high-res buffer streams in
      const fd = patchBufDims(bw, bh, MOON_PATCH_FAST_LONG_PX);
      if (!mpFast || mpFast.bufW !== fd.bufW || mpFast.bufH !== fd.bufH) {
        const cv = document.createElement("canvas");
        cv.width = fd.bufW;
        cv.height = fd.bufH;
        mpFast = { canvas: cv, img: cv.getContext("2d")!.createImageData(fd.bufW, fd.bufH), bufW: fd.bufW, bufH: fd.bufH };
      }
      const fView = patchView(fd.bufW, fd.bufH, bx, by, bw, bh, camPos, center, R, basis, k, cx, cy, X, Y, Z, wDeg, sun, texLon, shadowFactor);
      const tf = performance.now();
      renderMoonSurfaceRows(fView, base, detail, mpFast.img.data, 0, fd.bufH);
      const fms = performance.now() - tf;
      if (fms > moonPatchMsMax) moonPatchMsMax = fms;
      mpFast.canvas.getContext("2d")!.putImageData(mpFast.img, 0, 0);
      ctx.drawImage(mpFast.canvas, bx, by, bw, bh);
    }
    // ── REFRESH the high-res buffer for the CURRENT pose ────────────────────
    // The subtlety at very close range (2026-07-19): the disc fills the screen
    // but the tile mosaic covers only the tiny visible arc, so an ACTIVE drag
    // sweeps the sub-camera point off the resident tiles faster than they can
    // stream — a buffer rebuilt mid-drag would sample the soft BASE for the
    // newly-revealed surface and REPLACE the crisp settled buffer with a soft
    // one. So we only rebuild when SETTLED (or when there is no full buffer yet,
    // to bootstrap the first crisp frame promptly): during motion the last
    // crisp buffer stays drawn into the moving bbox (silhouette exact, texture
    // stale-but-crisp), then snaps to the correct pose the instant the drag
    // stops. Away from the floor the mosaic covers the whole sweep, so "settled"
    // arrives ~150ms after the last input with the fresh pose already correct.
    const patchSettled = !flight && performance.now() - lastInputAt >= MOON_PATCH_SETTLE_MS;
    if ((!mpFull || mpFull.key !== key) && !mpBuilding && (patchSettled || !mpFull)) {
      const ld = patchBufDims(bw, bh, MOON_PATCH_FULL_LONG_PX);
      const cv = document.createElement("canvas");
      cv.width = ld.bufW;
      cv.height = ld.bufH;
      mpBuilding = {
        key, canvas: cv, img: cv.getContext("2d")!.createImageData(ld.bufW, ld.bufH),
        bufW: ld.bufW, bufH: ld.bufH, bx, by, bw, bh,
        view: patchView(ld.bufW, ld.bufH, bx, by, bw, bh, camPos, center, R, basis, k, cx, cy, X, Y, Z, wDeg, sun, texLon, shadowFactor),
        base, detail, row: 0, ready: false, nCam,
      };
      if (!mpTimer) mpTimer = setTimeout(pumpMoonPatch, 0);
    }
    return true;
  }

  /** Leaving the focused-close range (or switching bodies): evict the active
   *  body's tiles + patch buffers (GPU budget), exactly like the 8k tier's
   *  eviction. Clears only the currently-resident body's mosaic. */
  function evictBodyPatch(): void {
    if (patchBodyId) bodyTileManagers.get(patchBodyId)?.clear();
    mpFull = null;
    mpFast = null;
    mpBuilding = null;
    if (mpTimer) { clearTimeout(mpTimer); mpTimer = null; }
    patchBodyId = null;
    moonCreditOn = false;
    patchCreditText = "";
  }

  function ringBandsNow(): RingBand[] | null {
    if (ringBands) return ringBands;
    const rt = texman.getRing();
    if (!rt) return null;
    ringBands = ringBandsFromTextures(rt.color, rt.alpha, 14);
    return ringBands;
  }

  /** the reference's sun glow canvas (additive sprite, drawn at 7×). */
  function sunGlowSprite(): HTMLCanvasElement {
    if (glowSprite) return glowSprite;
    const c = document.createElement("canvas");
    c.width = c.height = 256;
    const g = c.getContext("2d")!;
    const gr = g.createRadialGradient(128, 128, 10, 128, 128, 128);
    gr.addColorStop(0, "rgba(255,220,150,0.9)");
    gr.addColorStop(0.25, "rgba(255,180,80,0.35)");
    gr.addColorStop(1, "rgba(255,150,50,0)");
    g.fillStyle = gr;
    g.fillRect(0, 0, 256, 256);
    glowSprite = c;
    return c;
  }

  /**
   * Textured body sprite — TWO TIERS so no main-thread task busts the
   * frame budget:
   *  · FAST tier (≤ FAST_SHADE_RADIUS): built synchronously — bounded
   *    cost even when rebuilt every frame (time-warp spin, flights);
   *  · FULL tier (≤ TEXTURE_MAX_SHADE_RADIUS): streamed by the upgrade
   *    pipeline below in row-chunked macrotasks (LUT fill then compose,
   *    ≤ UPGRADE_CHUNK_PX each) once the same pose is requested on two
   *    consecutive settled draws — a spinning warp body keeps its fast
   *    sprite and never wastes upgrade work.
   */
  interface SpriteUpgrade {
    key: string;
    def: SpaceBodyDef;
    shadeR: number;
    sunCam: Vec3;
    axisCam: Vec3;
    nodeCam: Vec3;
    wq: number;
    tex: TexImage;
    bump: TexImage | null;
    light: BodyLight;
    lut: SphereLUT;
    lutRow: number;
    composeRow: number;
    canvas: HTMLCanvasElement;
    img: ImageData;
  }
  let upgrade: SpriteUpgrade | null = null;
  let upgradeTimer: ReturnType<typeof setTimeout> | null = null;
  const lastFullReq = new Map<string, string>();

  function pumpUpgrade(): void {
    upgradeTimer = null;
    if (!upgrade || disposed) return;
    const u = upgrade;
    const rows = Math.max(4, Math.round(UPGRADE_CHUNK_PX / u.lut.size));
    const t0 = performance.now();
    if (u.lutRow < u.lut.size) {
      fillSphereLUTRows(u.lut, u.axisCam, u.nodeCam, u.lutRow, u.lutRow + rows);
      u.lutRow += rows;
      const ms = performance.now() - t0;
      if (ms > lutBuildMsMax) lutBuildMsMax = ms;
    } else if (u.composeRow < u.lut.size) {
      composeTexturedSprite(
        u.lut, u.tex, u.wq, u.def.emissive ? null : u.sunCam, u.img.data,
        {
          bump: u.bump, bumpStrength: 1.2,
          rowStart: u.composeRow, rowEnd: u.composeRow + rows,
          texLonOffsetDeg: textureLonOffsetDeg(u.def.id),
          fullBright: !u.light.realistic,
          shadowFactor: u.light.shadowFactor,
          ringShadow: u.light.ring,
        },
      );
      u.composeRow += rows;
      const ms = performance.now() - t0;
      if (ms > spriteBuildMsMax) spriteBuildMsMax = ms;
    } else {
      u.canvas.getContext("2d")!.putImageData(u.img, 0, 0);
      if (spriteCache.size > 48) spriteCache.clear();
      spriteCache.set(u.key, u.canvas);
      upgrade = null;
      kick(); // the next draw picks up the full-res sprite
      return;
    }
    upgradeTimer = setTimeout(pumpUpgrade, 0);
  }

  /** B6 light state that changes a body's shaded pixels beyond its pose:
   *  realistic on/off, the eclipse/umbra scalar, and the ring shadow. Folded
   *  into the sprite cache key so a toggle or an eclipse re-shades correctly. */
  interface BodyLight {
    realistic: boolean;
    shadowFactor: number;
    ring: RingShadowParams | null;
  }
  function lightKey(l: BodyLight): string {
    const sq = (n: number): number => Math.round(n / 0.06) * 0.06;
    const r = l.ring
      ? `r${sq(l.ring.sun.x)},${sq(l.ring.sun.y)},${sq(l.ring.sun.z)}`
      : "r0";
    return `L${l.realistic ? 1 : 0}|e${Math.round(l.shadowFactor * 40)}|${r}`;
  }

  function spriteKey(id: string, tier: string, shadeR: number, wq: number, sunCam: Vec3, axisCam: Vec3, light: BodyLight): string {
    const sq = (n: number): number => Math.round(n / 0.06) * 0.06;
    return (
      `${id}|t${tier}|${shadeR}|${wq}|${sq(sunCam.x)},${sq(sunCam.y)},${sq(sunCam.z)}` +
      `|${q2(axisCam.x)},${q2(axisCam.y)},${q2(axisCam.z)}|${lightKey(light)}`
    );
  }

  function texturedSprite(
    def: SpaceBodyDef,
    radiusPx: number,
    sunCam: Vec3,
    axisCam: Vec3,
    nodeCam: Vec3,
    wDeg: number,
    tex: TexImage,
    bump: TexImage | null,
    light: BodyLight,
  ): HTMLCanvasElement {
    const rq = Math.max(2, Math.round(Math.exp(Math.round(Math.log(radiusPx) / 0.06) * 0.06)));
    const fullR = Math.min(rq, TEXTURE_MAX_SHADE_RADIUS);
    const fastR = Math.min(rq, FAST_SHADE_RADIUS);
    const wq = Math.round(wDeg * 2) / 2; // 0.5° spin quantum (sub-texel @1k)
    const wantTangents = def.id === "moon" && !!bump;
    // 1) full-res already composed for this pose? use it
    const fullKey = spriteKey(def.id, tex.tier, fullR, wq, sunCam, axisCam, light);
    const fullHit = spriteCache.get(fullKey);
    if (fullHit) return fullHit;
    // 2) settled pose seen twice → stream the full tier in the background
    const settled = !flight && performance.now() - lastInputAt >= 160;
    if (settled && fullR > fastR) {
      if (lastFullReq.get(def.id) === fullKey) {
        if (!upgrade || upgrade.key !== fullKey) {
          const size = Math.max(2, Math.round(fullR)) * 2 + 2;
          const canvas = document.createElement("canvas");
          canvas.width = size;
          canvas.height = size;
          upgrade = {
            key: fullKey, def, shadeR: fullR, sunCam, axisCam, nodeCam, wq, tex, bump, light,
            lut: createEmptySphereLUT(fullR, wantTangents),
            lutRow: 0, composeRow: 0, canvas,
            img: canvas.getContext("2d")!.createImageData(size, size),
          };
          if (!upgradeTimer) upgradeTimer = setTimeout(pumpUpgrade, 0);
        }
      } else {
        lastFullReq.set(def.id, fullKey);
        kick(); // one more draw re-requests this pose and starts the stream
      }
    }
    // 3) synchronous FAST tier (bounded task; per-frame safe under warp)
    const key = spriteKey(def.id, tex.tier, fastR, wq, sunCam, axisCam, light);
    const hit = spriteCache.get(key);
    if (hit) return hit;
    if (spriteCache.size > 48) spriteCache.clear();
    const lutKey =
      `${fastR}|${q2(axisCam.x)},${q2(axisCam.y)},${q2(axisCam.z)}` +
      `|${q2(nodeCam.x)},${q2(nodeCam.y)},${q2(nodeCam.z)}|b${wantTangents ? 1 : 0}`;
    let ent = lutCache.get(def.id);
    if (!ent || ent.key !== lutKey) {
      const t0 = performance.now();
      ent = { key: lutKey, lut: buildSphereLUT(fastR, axisCam, nodeCam, wantTangents) };
      const ms = performance.now() - t0;
      if (ms > lutBuildMsMax) lutBuildMsMax = ms;
      if (lutCache.size > 4) lutCache.clear();
      lutCache.set(def.id, ent);
    }
    const size = ent.lut.size;
    const c = document.createElement("canvas");
    c.width = size;
    c.height = size;
    const cc = c.getContext("2d")!;
    const img = cc.createImageData(size, size);
    const t0 = performance.now();
    composeTexturedSprite(
      ent.lut,
      tex,
      wq,
      def.emissive ? null : sunCam,
      img.data,
      {
        ...(wantTangents ? { bump, bumpStrength: 1.2 } : {}),
        texLonOffsetDeg: textureLonOffsetDeg(def.id),
        fullBright: !light.realistic,
        shadowFactor: light.shadowFactor,
        ringShadow: light.ring,
      },
    );
    const ms = performance.now() - t0;
    if (ms > spriteBuildMsMax) spriteBuildMsMax = ms;
    cc.putImageData(img, 0, 0);
    spriteCache.set(key, c);
    return c;
  }

  function shadedSprite(def: SpaceBodyDef, radiusPx: number, sunCam: Vec3, light: BodyLight): HTMLCanvasElement {
    // quantize: 6% radius steps, 0.06 light-direction steps — a flight
    // re-shades a handful of times, idle frames reuse the cache
    const rq = Math.max(2, Math.round(Math.exp(Math.round(Math.log(radiusPx) / 0.06) * 0.06)));
    const sq = (n: number): number => Math.round(n / 0.06) * 0.06;
    const key = `${def.id}|${rq}|${sq(sunCam.x)},${sq(sunCam.y)},${sq(sunCam.z)}|${lightKey(light)}`;
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
          // soft terminator (±0.04 of the cosine) — display smoothing only.
          // B6: realistic OFF ⇒ full-bright (no terminator); ON ⇒ the lambert
          // curve × the eclipse/umbra factor (a moon in its parent's shadow).
          const day = Math.max(0, Math.min(1, (lit + 0.03) / 0.07));
          const shade = 0.05 + 0.95 * Math.max(lit, 0);
          const w = light.realistic ? (0.05 + 0.95 * (day * shade)) * light.shadowFactor : 1;
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
    // a fly-to lands looking at the target body's centre — drop any
    // zoom-to-cursor / pan look-at offset carried from the prior focus
    focusOff = v3(0, 0, 0);
    focusOffTarget = v3(0, 0, 0);
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
      // instead of hiding behind it (or behind your back). If from == to (a
      // fly-to targeting the CURRENT focus body — external callers may do this),
      // sub(to,from) is 0 and norm3 would degenerate to a zero look direction
      // (broken camera); keep the current view direction and just re-frame the
      // distance in that case.
      const outbound = sub(to, from);
      const away = len3(outbound) > 1 ? norm3(outbound) : dir;
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
    // body card hook: a fly-to opens the target's card; flying home (to
    // the seam) closes it
    opts.onFocusBody?.(o?.exitOnArrival ? null : toId);
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
    // new focus ⇒ recentre the look-at (a frozen flight lands looking at the
    // body centre; zoom-to-cursor/pan offsets belong to the prior body only)
    focusOff = v3(0, 0, 0);
    focusOffTarget = v3(0, 0, 0);
    distTarget = dist;
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
    moonPatchDrewThisFrame = false;
    moonPatchFullThisFrame = false;

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
        // arrived: look-at is the body centre (focusOff already 0), and the
        // smooth-zoom target tracks the landed distance (no idle drift)
        focusOff = v3(0, 0, 0);
        focusOffTarget = v3(0, 0, 0);
        distTarget = dist;
        camPos = add(pos[focusId], scale3(dir, dist));
        camTrue = add(posT[focusId], scale3(dir, dist));
        viewDir = dir;
      }
    } else {
      // idle: the camera looks at (focus centre + focusOff) — the look-at
      // offset is what zoom-to-cursor and pan move (0 for centred orbiting).
      camPos = add(add(pos[focusId], focusOff), scale3(dir, dist));
      camTrue = add(add(posT[focusId], focusOff), scale3(dir, dist));
      viewDir = dir;
    }
    const basis = camBasis(viewDir, axis);
    const cx = w / 2;
    const cy = h / 2;
    // capture the live camera for the drive's projectSkyRaDec seam
    lastBasis = basis;
    lastK = kNow;
    lastCx = cx;
    lastCy = cy;

    // ── paint space: black base + the Milky Way panorama past 8 AU + the
    // REAL Yale BSC bright stars (starCatalog.ts) as they leave the inner
    // solar system. Both are the one real sky, at true directions through the
    // real galactic/ecliptic frame — never a fabricated random starfield. ──
    ctx.fillStyle = "#020409";
    ctx.fillRect(0, 0, w, h);
    const nowDraw = performance.now();
    const easeDt = lastDrawAt ? nowDraw - lastDrawAt : 1e9;
    lastDrawAt = nowDraw;
    const camAUtrue = len3(camTrue) / AU_M;
    const milkyTarget = milkyOn ? milkyWayOpacity(camAUtrue) : 0;
    if (easeDt > 1000) milkyShown = milkyTarget;
    else milkyShown += (milkyTarget - milkyShown) * (1 - Math.exp(-easeDt / MILKY_WAY_EASE_MS));
    if (Math.abs(milkyTarget - milkyShown) < 0.005) milkyShown = milkyTarget;
    milkyEasePending = Math.abs(milkyTarget - milkyShown) > 0.01;
    drawMilkyWay(w, h, basis);
    // real bright stars fade in EARLIER than the faint galactic glow (they pop
    // as the camera leaves the inner system); lazily loaded once the field
    // would show, additively composited over the panorama.
    const starTarget = starsOn ? starFieldOpacity(camAUtrue) : 0;
    if (starTarget > 0) ensureStars();
    if (easeDt > 1000) starsShown = starTarget;
    else starsShown += (starTarget - starsShown) * (1 - Math.exp(-easeDt / MILKY_WAY_EASE_MS));
    if (Math.abs(starTarget - starsShown) < 0.005) starsShown = starTarget;
    starEasePending = Math.abs(starTarget - starsShown) > 0.01;
    drawStarfield(w, h, basis, kNow, cx, cy);

    // ── ecliptic grid (default OFF): AU rings + bearing spokes ──
    if (gridOn) drawEclipticGrid(ctx, pos[sunDef.id], camPos, basis, kNow, cx, cy, w, h);

    // ── B3 orbit ellipses (under the bodies) — cached polylines in the
    // current layout space; sampling streams in one body per macrotask ──
    if (orbitsOn || trailsOn) scheduleOrbitSampling();
    if (orbitsOn) drawOrbitPaths(ctx, pos, camPos, basis, kNow, cx, cy);
    // 60° trailing motion arcs over the orbit lines (their own toggle)
    if (trailsOn) drawMotionTrails(ctx, pos, camPos, basis, kNow, cx, cy);

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

    // moon texture tiers ride the render truth: what disc the Moon drew at
    // and whether it is the focus (2k/8k fetch + the 8k eviction contract)
    const moonDrawn = drawn.find((x) => x.id === "moon");
    texman.noteMoonView(moonDrawn ? moonDrawn.discPx : 0, focusId === "moon");
    texturedIds = [];

    // bodies (far→near). Earth's DISC here is the impostor — the live map
    // composites above it and crossfades by opacity in the anchor below.
    for (const d of onScreen) {
      const r = d.discPx / 2;
      // Saturn's rings: the back half paints under the disc, the front
      // half over it (painter's split at the planet's center depth)
      const rings = d.id === "saturn" && d.discPx >= 6 ? ringBandsNow() : null;
      let ringEdges: ({ x: number; y: number; depth: number } | null)[][] | null = null;
      // B6 planet-shadow-on-rings: per-edge×segment flag — true when that ring
      // vertex sits in Saturn's OWN shadow (its umbra cast across the rings),
      // from REAL geometry (Sun→planet direction, ring offset in real metres).
      let ringShadowGrid: boolean[][] | null = null;
      if (rings) {
        const trueDisc = bodyDiscPx(radiusM(d.id), d.layoutDistM, kNow);
        const enlarge = trueDisc > 0 && Number.isFinite(trueDisc) ? d.discPx / trueDisc : 1;
        const axisE = axisEclOfDate("saturn", timeMs);
        const SEGS = 40;
        const Rp = radiusM(d.id);
        const sunDirEcl = realistic ? norm3(sub(posT[sunDef.id], posT[d.id])) : null;
        ringEdges = [];
        ringShadowGrid = sunDirEcl ? [] : null;
        for (let e = 0; e <= rings.length; e++) {
          const rel = e === 0 ? rings[0].r0 : rings[e - 1].r1;
          const pts = ringCircle3D(pos[d.id], axisE, Rp * rel * enlarge, SEGS);
          const row: ({ x: number; y: number; depth: number } | null)[] = [];
          for (let i = 0; i < SEGS; i++) {
            row.push(projTo(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2], camPos, basis, kNow, cx, cy));
          }
          ringEdges.push(row);
          if (sunDirEcl && ringShadowGrid) {
            // real-metre ring offsets at this radius (same segment order) → is
            // each vertex behind the planet from the Sun?
            const off = ringCircle3D({ x: 0, y: 0, z: 0 }, axisE, Rp * rel, SEGS);
            const srow: boolean[] = [];
            for (let i = 0; i < SEGS; i++) {
              srow.push(pointInSphereShadow({ x: off[i * 3], y: off[i * 3 + 1], z: off[i * 3 + 2] }, sunDirEcl, Rp));
            }
            ringShadowGrid.push(srow);
          }
        }
      }
      const paintRings = (back: boolean): void => {
        if (!rings || !ringEdges) return;
        const segs = ringEdges[0].length;
        for (let b = 0; b < rings.length; b++) {
          const band = rings[b];
          if (band.alpha < 0.02) continue;
          for (let j = 0; j < segs; j++) {
            const j2 = (j + 1) % segs;
            const a = ringEdges[b][j];
            const bq = ringEdges[b][j2];
            const c = ringEdges[b + 1][j2];
            const dq = ringEdges[b + 1][j];
            if (!a || !bq || !c || !dq) continue;
            const isBack = (a.depth + bq.depth + c.depth + dq.depth) / 4 > d.p.depth;
            if (isBack !== back) continue;
            // B6: a ring quad mostly inside the planet's shadow reads much
            // darker (the planet's shadow falling across its own rings).
            let shade = 1;
            if (ringShadowGrid) {
              const nShadow =
                (ringShadowGrid[b][j] ? 1 : 0) + (ringShadowGrid[b][j2] ? 1 : 0) +
                (ringShadowGrid[b + 1][j2] ? 1 : 0) + (ringShadowGrid[b + 1][j] ? 1 : 0);
              if (nShadow >= 3) shade = RING_SHADOW_STRENGTH;
            }
            ctx.fillStyle = `rgba(${Math.round(band.r * shade)},${Math.round(band.g * shade)},${Math.round(band.b * shade)},${band.alpha})`;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(bq.x, bq.y);
            ctx.lineTo(c.x, c.y);
            ctx.lineTo(dq.x, dq.y);
            ctx.closePath();
            ctx.fill();
          }
        }
      };
      paintRings(true);
      if (r >= 3) {
        // sun direction in camera coords for the sprite shader
        const def = defById.get(d.id)!;
        // B4/B5: a focused-close rocky body (Moon + Mars/Mercury/Venus) → real
        // Trek surface patch instead of the capped disc sprite (the far tier).
        // Handles it entirely. Gas giants have no Trek surface (trekBody→null)
        // and fall through to the textured cloud-band sprite below. Saturn's
        // rings are painted above regardless; the rocky bodies carry none.
        // B6: the eclipse/umbra scalar (a moon in its parent's real shadow),
        // computed from TRUE positions at the sim-clock instant. 1 for planets
        // (no parent) and whenever realistic lighting is off.
        const eclipse = eclipseFactor(d.id, posT);
        if (
          trekBody(d.id) &&
          d.layoutDistM < MOON_PATCH_ACTIVATE_RADII * radiusM(d.id) &&
          drawBodyPatch(d.id, d, sunPos, pos, camPos, basis, kNow, cx, cy, w, h, eclipse)
        ) {
          moonPatchDrewThisFrame = true;
          continue;
        }
        const toSunW = def.emissive ? basis.f : norm3(sub(sunPos, pos[d.id]));
        const sunCam = v3(dot(toSunW, basis.r), dot(toSunW, basis.u), -dot(toSunW, basis.f));
        // the Sun's flare: the additive glow sprite at 7× the disc
        // (reference look), under the textured/emissive disc
        if (def.emissive) {
          const glow = sunGlowSprite();
          const gs = d.discPx * SUN_GLOW_SCALE;
          ctx.save();
          ctx.globalCompositeOperation = "lighter";
          ctx.drawImage(glow, d.p.x - gs / 2, d.p.y - gs / 2, gs, gs);
          ctx.restore();
        }
        // textured when the asset is resident AND the body has an IAU
        // rotation model (axisEcl + W drive the texture orientation —
        // B3's rotation state becomes VISIBLE here); Lambert otherwise
        const tex = texman.get(d.id);
        let sprite: HTMLCanvasElement;
        if (tex && hasRotationModel(d.id)) {
          const rid = d.id as RotationBodyId;
          const axisE = axisEclOfDate(rid, timeMs);
          const nodeE = equatorNodeDirEclOfDate(rid, timeMs);
          const axisCam = v3(dot(axisE, basis.r), dot(axisE, basis.u), -dot(axisE, basis.f));
          const nodeCam = v3(dot(nodeE, basis.r), dot(nodeE, basis.u), -dot(nodeE, basis.f));
          // B6 ring shadow-on-disc (Saturn): express the Sun in the LUT's body
          // frame (Z = spin axis, X = equator node, Y = Z×X) so compose can test
          // each pixel's surface point against the ring annulus.
          let ring: RingShadowParams | null = null;
          if (d.id === "saturn" && realistic && ringBandsNow()) {
            const Zc = norm3(axisCam);
            const Xc = norm3(nodeCam);
            const Yc = cross(Zc, Xc);
            ring = {
              sun: { x: dot(sunCam, Xc), y: dot(sunCam, Yc), z: dot(sunCam, Zc) },
              rInner: SATURN_RING_INNER_REL,
              rOuter: SATURN_RING_OUTER_REL,
              strength: RING_SHADOW_STRENGTH,
            };
          }
          const light: BodyLight = { realistic, shadowFactor: eclipse, ring };
          sprite = texturedSprite(
            def, r, sunCam, axisCam, nodeCam,
            iauPrimeMeridianDeg(rid, timeMs), tex,
            d.id === "moon" ? texman.getMoonBump() : null,
            light,
          );
          texturedIds.push(d.id);
        } else {
          sprite = shadedSprite(def, r, sunCam, { realistic, shadowFactor: eclipse, ring: null });
        }
        // the sprite's shaded disc has radius (sprite.width/2 − 1); scale the
        // whole sprite so THAT maps to exactly r (the +1 border must not
        // shrink the drawn body — true size is the contract)
        const shadeR = sprite.width / 2 - 1;
        const box = (r / shadeR) * sprite.width;
        ctx.drawImage(sprite, d.p.x - box / 2, d.p.y - box / 2, box, box);
        paintRings(false);
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
    const labelRects: SpaceFrameState["labelRects"] = [];
    labeled.forEach((d, i) => {
      const r = d.discPx / 2;
      const isMarker = markerNeeded(d.discPx);
      const isSunBody = !!defById.get(d.id)!.emissive;
      if (isMarker) {
        markers++;
        markerIds.add(d.id);
      }
      // circle-dot glyph ON the body (the reference's .lbl dot; the Sun's
      // is larger per .lbl.sun). It is ALSO the honesty marker: with
      // labels toggled off, sub-pixel bodies keep their dot — nothing is
      // ever invisible.
      if (d.discPx < LABEL_DOT_MAX_DISC_PX && (labelsOn || isMarker)) {
        const ring = isSunBody ? 7 : 4.5;
        ctx.beginPath();
        ctx.arc(d.p.x, d.p.y, ring, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(59,130,246,0.12)";
        ctx.fill();
        ctx.strokeStyle = isSunBody ? "rgba(232,217,168,0.95)" : "rgba(159,180,216,0.9)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.lineWidth = 1;
      }
      if (!labelsOn) return;
      const anchorX = d.p.x + Math.max(r, 8) + 6;
      const ly = d.p.y + offsets[i];
      if (offsets[i] > 0) {
        ctx.strokeStyle = "rgba(190,205,225,0.35)";
        ctx.beginPath();
        ctx.moveTo(d.p.x + Math.max(r, 8), d.p.y);
        ctx.lineTo(anchorX, ly);
        ctx.stroke();
      }
      const extra =
        d.id === anchorDef.id && mapAnchorOpacity(d.discPx) < 0.5
          ? " · live map resumes on approach"
          : "";
      // B3: satellites folded into this body's footprint surface here —
      // never silently absent (zoom/fly closer and they resolve)
      const folded = foldedMoons.get(d.id) ?? 0;
      const moonsNote = folded ? ` · +${folded} moon${folded > 1 ? "s" : ""}` : "";
      // reference label typography: bright name, dim real-distance readout
      // (the number is the TRUE camera→body distance — never layout space)
      const name = defById.get(d.id)!.name;
      const dim = ` · ${fmtSpaceDistance(d.distM)}${extra}${moonsNote}`;
      ctx.fillStyle = "rgba(210,222,238,0.92)";
      ctx.fillText(name, anchorX, ly);
      const nameW = ctx.measureText(name).width;
      ctx.fillStyle = "rgba(139,150,170,0.9)";
      ctx.fillText(dim, anchorX + nameW, ly);
      // the label text is a fly-to hit target (reference: click a label)
      labelRects.push({ id: d.id, x: anchorX, y: ly - 7, w: nameW + ctx.measureText(dim).width, h: 14 });
    });

    // ── chrome (x=72 clears the page's left button rail; the bottom-left
    // caption slot is free because the parent hides the map-meta overlays
    // — preset switch, imagery-date chip — while the frame is active, and
    // raises the surviving +/- zoom buttons above the caption via CSS) ──
    ctx.fillStyle = "rgba(210,222,238,0.85)";
    ctx.fillText(new Date(timeMs).toISOString().replace(/\.\d{3}Z$/, "Z"), 72, 22);
    const focus = drawn.find((d) => d.id === focusId)!;
    ctx.fillStyle = "rgba(160,175,198,0.8)";
    // reference HUD line 2: "at <body> · camera N AU out[ · tracking <name>]"
    // — the tracking suffix marks an explicit fly-to follow (not the seam
    // anchor, not the released Sun frame)
    const hudName = defById.get(flight ? flight.toId : focusId)!.name;
    const trackSuffix =
      !flight && focusId !== sunDef.id && focusId !== anchorDef.id ? ` · tracking ${hudName}` : "";
    ctx.fillText(
      `${flight ? "flying to" : "at"} ${hudName} · camera ${fmtSpaceDistance(flight ? len3(sub(camTrue, posT[flight.toId])) : focus.distM)} out${trackSuffix}`,
      72, 38,
    );
    // honesty caption — persistent, every frame; wraps to three lines on
    // narrow viewports so nothing ever clips off-screen (390px flawless).
    // B2: the caption tracks the scale state truthfully — TRUE SCALE keeps
    // the B1 caption; ANY compression switches to the directive's own
    // wording, amber (the SIGNAL/warning hue) so a compressed layout can
    // never pass itself off as the true-scale view.
    const trueNow = isTrueScale(scaleSt);
    // the reference footer adds the LIVE time multiplier (one sim clock —
    // the frame only READS the rate for display)
    const rate = opts.getTimeRate ? opts.getTimeRate() : 1;
    const rateStr = rate === 0 ? "paused" : `×${Math.round(rate).toLocaleString("en-US")}`;
    const capLines: { text: string; color: string }[] = [];
    // REQUIRED visible CC-BY credit whenever the panorama is on screen
    if (milkyShown > 0.02) capLines.push({ text: MILKY_WAY_CREDIT, color: "rgba(140,155,178,0.7)" });
    // public-domain courtesy credit whenever the real bright stars are visible
    if (starsShown > 0.02 && starCat) capLines.push({ text: STAR_CATALOG_CREDIT, color: "rgba(140,155,178,0.7)" });
    // B4/B5: the required visible source credit whenever real Trek tiles are on
    // screen — the ACTIVE body's own credit (Moon/Mars/Mercury/Venus), a
    // public-domain courtesy line, never claimed as live
    if (moonPatchDrewThisFrame && moonCreditOn && patchCreditText) {
      capLines.push({ text: patchCreditText, color: "rgba(140,155,178,0.7)" });
    }
    const amberHi = "rgba(251,178,76,0.9)";
    const amberLo = "rgba(251,178,76,0.6)";
    const grayHi = "rgba(160,175,198,0.8)";
    const grayLo = "rgba(140,155,178,0.65)";
    if (trueNow) {
      const timeNote = rate !== 1 ? ` · time ${rateStr}` : "";
      if (w < 700) {
        capLines.push(
          { text: "TRUE SCALE — real ephemeris positions & sizes", color: grayHi },
          { text: "the camera does the compressing · sub-pixel bodies get markers", color: grayLo },
          { text: `Schlyter/van Flandern (~arcmin) · Moon phase real${timeNote}`, color: grayLo },
        );
      } else {
        capLines.push(
          { text: "TRUE SCALE — real ephemeris positions & sizes · the camera does the compressing", color: grayHi },
          { text: `markers flag bodies smaller than a pixel · Schlyter/van Flandern (~arcmin) · Moon phase real${timeNote}`, color: grayLo },
        );
      }
    } else if (w < 700) {
      capLines.push(
        { text: "distances/sizes compressed for visibility", color: amberHi },
        { text: "— labels always show real values", color: amberHi },
        { text: `compression ${Math.round(scaleSt.c * 100)}% · size ×${Math.round(scaleSt.s)} · Sun cap ×${SUN_SIZE_MULT_CAP} · time ${rateStr} · Earth (live map) true`, color: amberLo },
      );
    } else {
      capLines.push(
        { text: "distances/sizes compressed for visibility — labels always show real values", color: amberHi },
        { text: `compression ${Math.round(scaleSt.c * 100)}% · body size ×${Math.round(scaleSt.s)} · Sun cap ×${SUN_SIZE_MULT_CAP} · time ${rateStr} · Earth (live map) always true · Schlyter/van Flandern (~arcmin)`, color: amberLo },
      );
    }
    capLines.forEach((line, i) => {
      ctx.fillStyle = line.color;
      ctx.fillText(line.text, 16, h - 16 - (capLines.length - 1 - i) * 14);
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

    // B4/B5 eviction: the moment no rocky body is the focused-close body, drop
    // its tile mosaic + patch buffers (GPU budget — matches the 8k tier's
    // eviction; __vtMoonTileBytes drops to 0)
    if (moonPatchDrewThisFrame) moonPatchOn = true;
    else if (moonPatchOn) { moonPatchOn = false; evictBodyPatch(); }

    renderMsLast = performance.now() - t0;
    lastState = {
      timeMs,
      focusId,
      distM: dist,
      focusOffM: { x: focusOff.x, y: focusOff.y, z: focusOff.z },
      scale: { c: scaleSt.c, s: scaleSt.s },
      flying: !!flight,
      animating: !!flight || performance.now() - lastInputAt < 200
        || Math.abs(orbitVelYaw) > ORBIT_INERTIA_EPS_DEG || Math.abs(orbitVelPitch) > ORBIT_INERTIA_EPS_DEG
        || zoomEaseActive(),
      renderMsLast,
      markers,
      scaleBar: scaleBarOut,
      orbitPaths: { on: orbitsOn, ready: orbitCache.size, total: orbitTotal },
      milkyWay: { on: milkyOn, opacity: milkyShown, target: milkyTarget },
      starfield: {
        on: starsOn,
        opacity: starsShown,
        target: starTarget,
        loaded: !!starCat,
        count: starCat ? starCat.count : 0,
        rendered: starRendered,
        msLast: starMsLast,
      },
      eclipticGrid: gridOn,
      motionTrails: trailsOn,
      bodyLabels: labelsOn,
      // B6: realistic (physically-lit) vs even-lit inspection mode.
      realisticLighting: realistic,
      textures: {
        ...texman.stats(),
        spriteBuildMsMax,
        lutBuildMsMax,
        skyMsLast,
        texturedThisFrame: texturedIds.slice(),
      },
      labelRects,
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
        // B6 eclipse/umbra: direct-sunlight factor from real geometry (1 =
        // fully sunlit, 0 = total umbra) — < 1 ⇒ this body is being eclipsed
        // by its parent at the sim-clock instant (a lunar eclipse for the Moon).
        sunlitFactor: eclipseFactor(d.id, posT),
        distM: d.distM,
        layoutDistM: d.layoutDistM,
        // B3 rotation state (IAU pole + W at the sim-clock instant) — the
        // orientation truth B4/B5 surfaces will render from
        axisEcl: hasRotationModel(d.id) ? axisEclOfDate(d.id, timeMs) : null,
        primeMeridianDeg: hasRotationModel(d.id) ? iauPrimeMeridianDeg(d.id, timeMs) : null,
        insideParent: insideParent.has(d.id),
      })),
      moonPatch: (() => {
        const mgr = patchBodyId ? bodyTileManagers.get(patchBodyId) : null;
        const st = mgr ? mgr.stats() : { z: null, tiles: 0, bytes: 0, maxChunkMs: 0 };
        return {
          active: moonPatchDrewThisFrame,
          full: moonPatchFullThisFrame,
          // which rocky body's surface is streaming (Moon/Mars/Mercury/Venus)
          body: patchBodyId,
          mosaicZ: st.z,
          mosaicTiles: st.tiles,
          tileBytes: st.bytes,
          patchMsMax: moonPatchMsMax,
          tileChunkMsMax: st.maxChunkMs,
          credited: moonPatchDrewThisFrame && moonCreditOn,
        };
      })(),
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
      // INERTIA COAST: when the drag has released but the orbit still has
      // residual velocity, advance + decay it before drawing (the reference's
      // damped-OrbitControls glide). Dragging drives dir directly, so skip.
      if (!dragging && (Math.abs(orbitVelYaw) > ORBIT_INERTIA_EPS_DEG || Math.abs(orbitVelPitch) > ORBIT_INERTIA_EPS_DEG)) {
        const y = orbitInertiaStep(orbitVelYaw);
        const p = orbitInertiaStep(orbitVelPitch);
        applyOrbit(y.stepDeg, p.stepDeg);
        orbitVelYaw = y.velDeg;
        orbitVelPitch = p.velDeg;
      }
      // SMOOTH ZOOM-TO-CURSOR (2026-07-19): a wheel notch sets distTarget +
      // focusOffTarget; each frame eases dist (log space) and the look-at
      // offset toward them — maplibre-style momentum, no hard snap. Pan writes
      // focusOff == focusOffTarget so this is a no-op mid-pan.
      if (!flight) advanceZoomEase();
      draw();
      const coasting = Math.abs(orbitVelYaw) > ORBIT_INERTIA_EPS_DEG || Math.abs(orbitVelPitch) > ORBIT_INERTIA_EPS_DEG;
      // sky fades (milky/star ease) + skyRenderPending (progressive panorama
      // build) + zoomEaseActive (smooth zoom-to-cursor easing to rest)
      if (flight || coasting || zoomEaseActive() || performance.now() - lastInputAt < 200 || milkyEasePending || starEasePending || skyRenderPending) {
        rafId = requestAnimationFrame(loop);
      }
    };
    rafId = requestAnimationFrame(loop);
  }

  /** true while the smooth-zoom distance OR the look-at offset is still
   *  easing toward its target — keeps the rAF loop alive until it settles. */
  function zoomEaseActive(): boolean {
    const dr = dist > 0 && distTarget > 0 ? Math.abs(Math.log(dist / distTarget)) : 0;
    const fx = focusOff.x - focusOffTarget.x;
    const fy = focusOff.y - focusOffTarget.y;
    const fz = focusOff.z - focusOffTarget.z;
    const fd = Math.hypot(fx, fy, fz);
    return dr > 0.004 || fd > radiusM(focusId) * 1e-4;
  }

  /** one eased step of the smooth-zoom distance + look-at offset toward their
   *  targets (log-space distance, linear offset), snapping when within eps. */
  function advanceZoomEase(): void {
    if (!zoomEaseActive()) {
      dist = distTarget;
      focusOff = focusOffTarget;
      return;
    }
    dist = smoothZoomStep(dist, distTarget, ZOOM_EASE_ALPHA);
    focusOff = v3(
      focusOff.x + (focusOffTarget.x - focusOff.x) * ZOOM_EASE_ALPHA,
      focusOff.y + (focusOffTarget.y - focusOff.y) * ZOOM_EASE_ALPHA,
      focusOff.z + (focusOffTarget.z - focusOff.z) * ZOOM_EASE_ALPHA,
    );
  }

  // ── input (container-level: events bubble up from the map canvas) ──

  /** clamp a distance to the manual zoom band [MIN_ZOOM_RADII·R, MAX]. */
  function clampDist(d: number): number {
    return Math.min(MAX_CAMERA_DISTANCE_M, Math.max(MIN_ZOOM_RADII * radiusM(focusId), d));
  }

  /** the LIVE camera pose (layout space) for a raycast: look-at = focus centre
   *  + focusOff, camPos = look-at + dir·dist, basis from the axis-up rule. */
  function liveCamera(): { camPos: Vec3; basis: CamBasis; k: number; center: Vec3 } {
    const pos = layoutNow(timeMs);
    const center = pos[focusId];
    const lookAt = add(center, focusOff);
    const camPos = add(lookAt, scale3(dir, dist));
    const basis = camBasis(dir, earthAxisEcl(timeMs));
    const k = perspectiveScalePx(fovDeg, cssSize().h);
    return { camPos, basis, k, center };
  }

  /** the body-relative surface point the cursor (container px) is over, or
   *  null on a miss — the anchor of zoom-to-cursor. */
  function cursorHitOffset(mx: number, my: number): Vec3 | null {
    const { camPos, basis, k, center } = liveCamera();
    const { w, h } = cssSize();
    const rd = screenRayDir(mx, my, k, w / 2, h / 2, basis);
    const hit = raycastSphere(camPos, rd, center, radiusM(focusId));
    return hit ? sub(hit, center) : null;
  }

  /** WHEEL / trackpad zoom — maplibre-parity: ease the camera distance toward
   *  a target (momentum, no snap) AND pull the cursor's surface point toward
   *  screen centre (zoom-to-cursor). Zoom-out un-winds the pull back to
   *  centred. A cursor that misses the body falls back to plain centred zoom. */
  function wheelZoom(deltaY: number, mx: number, my: number): void {
    if (exited) return;
    cancelFlight();
    const g = zoomStepFactor(deltaY);
    distTarget = clampDist(distTarget * g);
    // the Earth anchor hands the camera back to the LIVE MAP at the seam — keep
    // its zoom centred so the handoff frame is untouched (the map itself does
    // cursor-zoom once control returns). Every other body gets zoom-to-cursor.
    if (focusId === anchorDef.id) { kick(); return; }
    const maxOff = PAN_MAX_OFFSET_RADII * radiusM(focusId);
    if (g < 1) {
      // zoom IN: pull the look-at toward the cursor's surface point so that
      // point tracks toward centre — the map's "zoom in on the crater" feel.
      const hit = cursorHitOffset(mx, my);
      if (hit && Number.isFinite(hit.x + hit.y + hit.z)) {
        focusOffTarget = cursorPullOffset(focusOffTarget, hit, (1 - g) * CURSOR_PULL_GAIN, maxOff);
      }
    } else if (g > 1) {
      // zoom OUT: recentre — unwind the look-at offset back toward the body.
      focusOffTarget = cursorPullOffset(focusOffTarget, v3(0, 0, 0), (1 - 1 / g) * CURSOR_PULL_GAIN, maxOff);
    }
    kick();
  }

  /** Programmatic / button / keyboard / seam-entry zoom toward SCREEN CENTRE
   *  (no cursor). Immediate (discrete inputs snap), keeping distTarget in sync
   *  so a following wheel gesture eases from the right place; the look-at
   *  offset is untouched (centred zoom for the +/- buttons). */
  function nudge(deltaY: number): void {
    if (exited) return;
    cancelFlight();
    // per-body closest approach; for Earth the seam (scale > 1 inside
    // draw()) hands back to the map long before this floor could bind.
    // MIN_ZOOM_RADII (not the arrival clamp) — manual zoom skims the surface.
    dist = clampDist(dist * zoomStepFactor(deltaY));
    distTarget = dist;
    kick();
  }

  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    lastInputAt = performance.now();
    const rect = container.getBoundingClientRect();
    wheelZoom(e.deltaY, e.clientX - rect.left, e.clientY - rect.top);
  };

  // INPUT MAPPING (maplibre-parity, 2026-07-19):
  //  · left-drag (mouse) / one-finger (touch) = ORBIT (+ release inertia) —
  //    the OrbitControls-rotate gesture the human called "easy".
  //  · right-drag OR shift+left-drag (mouse) / two-finger drag (touch) = PAN —
  //    translate the look-at across the surface (OrbitControls-pan / the map's
  //    two-finger pan), keeping the surface point under the pointer.
  //  · wheel / trackpad = smooth zoom toward the CURSOR (see wheelZoom).
  //  · two-finger pinch (touch) = smooth zoom toward the pinch CENTROID.
  const pointers = new Map<number, { x: number; y: number }>();
  let pinchDist = 0;
  let pinchCx = 0;
  let pinchCy = 0;
  let dragMoved = 0;
  let dragging = false;
  let panMode = false; // this drag pans (right/shift/two-finger) vs orbits
  // residual angular velocity (deg/frame) for the post-release coast — the
  // reference's damped OrbitControls feel; decayed by orbitInertiaStep().
  let orbitVelYaw = 0;
  let orbitVelPitch = 0;
  const DRAG_DEG_PER_PX = 0.25;

  /** yaw about the ecliptic axis + pitch about the camera-right axis, with
   *  the lock-horizon polar clamp (never under the ecliptic while ON; OFF
   *  only widens — roll impossible in both, camBasis rebuilds up). Shared by
   *  live drag and the inertial coast — the ONLY pitch-bounding site. */
  function applyOrbit(yawDeg: number, pitchDeg: number): void {
    const axis = earthAxisEcl(timeMs);
    const b = camBasis(dir, axis);
    let nd = rotateAbout(dir, axis, yawDeg);
    nd = rotateAbout(nd, b.r, pitchDeg);
    // polar clamp: dot(dir,axis)=cos(polar). Out of bounds ⇒ yaw-only, so a
    // wild drag slides along the clamp instead of dying at it.
    const { dotHi, dotLo } = polarClampDots(lockHorizon);
    const d = dot(nd, axis);
    if (d >= dotLo && d <= dotHi) dir = norm3(nd);
    else dir = norm3(rotateAbout(dir, axis, yawDeg));
  }

  /** translate the look-at offset by a screen drag (grab-pan) — no cancelFlight
   *  / kick (callers own those). meters/px = dist/k at the look-at plane, so
   *  the surface moves 1:1 under the pointer, exactly like panning the map. */
  function panOffsetBy(dxPx: number, dyPx: number): void {
    const basis = camBasis(dir, earthAxisEcl(timeMs));
    const k = perspectiveScalePx(fovDeg, cssSize().h);
    const maxOff = PAN_MAX_OFFSET_RADII * radiusM(focusId);
    focusOff = panOffset(focusOff, dxPx, dyPx, basis, dist / k, maxOff);
    focusOffTarget = focusOff; // pan is direct manipulation — nothing to ease
  }

  const onPointerDown = (e: PointerEvent): void => {
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    dragMoved = 0;
    dragging = pointers.size === 1;
    // right-button or shift ⇒ this single-pointer drag PANS instead of orbiting
    panMode = pointers.size === 1 && (e.button === 2 || e.shiftKey);
    // grabbing kills the coast (OrbitControls parity: catch the spinning globe)
    orbitVelYaw = 0;
    orbitVelPitch = 0;
    if (pointers.size === 2) {
      const [a, b] = Array.from(pointers.values());
      pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
      pinchCx = (a.x + b.x) / 2;
      pinchCy = (a.y + b.y) / 2;
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
      dragging = false;
      const [a, b] = Array.from(pointers.values());
      const d2 = Math.hypot(a.x - b.x, a.y - b.y);
      const cx2 = (a.x + b.x) / 2;
      const cy2 = (a.y + b.y) / 2;
      cancelFlight();
      // pinch distance ⇒ zoom (immediate — direct manipulation); centroid
      // translation ⇒ pan. Together = zoom toward the pinch centroid + drag,
      // the map's two-finger gesture.
      if (pinchDist > 0 && d2 > 0) {
        dist = clampDist(dist * (pinchDist / d2));
        distTarget = dist;
      }
      panOffsetBy(cx2 - pinchCx, cy2 - pinchCy);
      pinchDist = d2;
      pinchCx = cx2;
      pinchCy = cy2;
      kick();
      return;
    }
    dragMoved += Math.abs(dx) + Math.abs(dy);
    if (dragMoved < 3) return;
    cancelFlight();
    if (panMode) {
      // PAN: traverse the surface; no orbit, no coast
      panOffsetBy(dx, dy);
      orbitVelYaw = 0;
      orbitVelPitch = 0;
      kick();
      return;
    }
    const yaw = dx * DRAG_DEG_PER_PX;
    const pitch = dy * DRAG_DEG_PER_PX;
    applyOrbit(yaw, pitch);
    // record the last delta as the coast velocity (released → glides on)
    orbitVelYaw = yaw;
    orbitVelPitch = pitch;
    kick();
  };
  const onPointerUp = (e: PointerEvent): void => {
    pointers.delete(e.pointerId);
    pinchDist = 0;
    if (pointers.size === 0) {
      dragging = false;
      // leave orbitVel* intact → the loop coasts it out (unless the drag
      // barely moved, in which case velocity is already ~0)
      if (Math.abs(orbitVelYaw) > ORBIT_INERTIA_EPS_DEG || Math.abs(orbitVelPitch) > ORBIT_INERTIA_EPS_DEG) kick();
    }
  };
  const onClick = (e: MouseEvent): void => {
    if (dragMoved >= 4 || exited || !lastState) return;
    const rect = container.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    // label text boxes fly to their body first (reference: "click a label
    // to fly to that body") — small, specific targets beat disc proximity
    for (const lr of lastState.labelRects) {
      if (mx >= lr.x - 2 && mx <= lr.x + lr.w + 2 && my >= lr.y - 2 && my <= lr.y + lr.h + 2) {
        if (lr.id === anchorDef.id) flyHome();
        else if (lr.id !== focusId || flight) beginFlight(lr.id);
        return;
      }
    }
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
    // EMPTY SPACE → release back to the Sun/heliocentric frame (the
    // reference's open todo, implemented): tracking stops, the camera
    // keeps its exact world position (reanchorPose invariant). Guarded at
    // the seam — releasing while focused on the map anchor would disarm
    // the seam exit under the user's feet.
    if (!flight && focusId !== sunDef.id && focusId !== anchorDef.id) {
      const posL = layoutNow(timeMs);
      // the camera keeps its exact world position (look-at offset included)
      const cam = add(add(posL[focusId], focusOff), scale3(dir, dist));
      const pose = reanchorPose(cam, posL[sunDef.id], MIN_DISTANCE_RADII * radiusM(sunDef.id));
      focusId = sunDef.id;
      dir = pose.dir;
      dist = pose.dist;
      // released to the Sun frame ⇒ centred look-at (offsets were the body's)
      focusOff = v3(0, 0, 0);
      focusOffTarget = v3(0, 0, 0);
      distTarget = dist;
      opts.onFocusBody?.(null);
      kick();
    }
  };

  // right-drag pans (not a browser context menu) while the space frame is live
  const onContextMenu = (e: MouseEvent): void => { e.preventDefault(); };

  container.addEventListener("wheel", onWheel, { passive: false });
  container.addEventListener("pointerdown", onPointerDown);
  container.addEventListener("pointermove", onPointerMove);
  container.addEventListener("pointerup", onPointerUp);
  container.addEventListener("pointercancel", onPointerUp);
  container.addEventListener("click", onClick);
  container.addEventListener("contextmenu", onContextMenu);

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
    const camNow = add(add(pos[focusId], focusOff), scale3(dir, dist));
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
    setMilkyWay(on: boolean): void {
      if (on === milkyOn) return;
      milkyOn = on;
      lastInputAt = performance.now();
      kick();
    },
    setStars(on: boolean): void {
      if (on === starsOn) return;
      starsOn = on;
      if (on) ensureStars();
      lastInputAt = performance.now();
      kick();
    },
    projectSkyRaDec(raDeg: number, decDeg: number): { x: number; y: number; front: boolean } {
      if (!lastState) draw();
      if (!lastBasis) return { x: NaN, y: NaN, front: false };
      const ra = raDeg * DEG;
      const dec = decDeg * DEG;
      const eq = v3(Math.cos(dec) * Math.cos(ra), Math.cos(dec) * Math.sin(ra), Math.sin(dec));
      const e = eqToEclDir(eq, timeMs);
      return projectStarScreen(e, lastBasis, lastK, lastCx, lastCy);
    },
    setEclipticGrid(on: boolean): void {
      if (on === gridOn) return;
      gridOn = on;
      lastInputAt = performance.now();
      kick();
    },
    setLockHorizon(on: boolean): void {
      if (on === lockHorizon) return;
      lockHorizon = on;
      lastInputAt = performance.now();
      kick();
    },
    setMotionTrails(on: boolean): void {
      if (on === trailsOn) return;
      trailsOn = on;
      lastInputAt = performance.now();
      kick(); // shares the orbit sampling stream (scheduleOrbitSampling)
    },
    setBodyLabels(on: boolean): void {
      if (on === labelsOn) return;
      labelsOn = on;
      lastInputAt = performance.now();
      kick();
    },
    setRealisticLighting(on: boolean): void {
      if (on === realistic) return;
      realistic = on;
      // shaded/textured sprites + surface patches re-key on the realistic bit,
      // so a flip re-shades every body next frame (terminator on ⇄ full-bright).
      lastInputAt = performance.now();
      kick();
    },
    getBodyCard(id: string): BodyCardInfo | null {
      return bodyCardInfo(id, timeMs);
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
      container.removeEventListener("contextmenu", onContextMenu);
      spriteCache.clear();
      lutCache.clear();
      upgrade = null;
      if (upgradeTimer) {
        clearTimeout(upgradeTimer);
        upgradeTimer = null;
      }
      skyCanvas = null;
      skyRays = null;
      skyWork = null;
      skyReadyKey = "";
      skyWorkKey = "";
      skyRenderPending = false;
      ringBands = null;
      // GALAXY: abort any in-flight catalog fetch and free the star layer +
      // catalog buffers (leaving space unloads the real sky like everything else)
      try { starAbort.abort(); } catch { /* already */ }
      starCanvas = null;
      starCat = null;
      starEclDir = null;
      starSizeArr = null;
      starAlphaArr = null;
      // B4/B5: free every rocky-body tile mosaic + patch buffers on exit
      // (__vtMoonTileBytes → 0, drive-observable) alongside the base textures
      if (mpTimer) { clearTimeout(mpTimer); mpTimer = null; }
      for (const m of bodyTileManagers.values()) m.dispose();
      bodyTileManagers.clear();
      patchBodyId = null;
      mpFull = null;
      mpFast = null;
      mpBuilding = null;
      // leaving space unloads every texture (the directive's eviction
      // contract; __vtSpaceTexBytes drops to 0 — drive-observable)
      texman.dispose();
      opts.applyEarthAnchor(null); // release the map canvas (styles reset)
      try { canvas.remove(); } catch { /* detached */ }
    },
  };
}
