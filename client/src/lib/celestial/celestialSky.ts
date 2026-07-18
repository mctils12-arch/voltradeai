// Realistic celestial sky — EARTH TWIN W1 (human sprint spec: "an always-on
// REALISTIC celestial system — Sun, Moon, planets rendered at correct APPARENT
// size / position / brightness on the globe's sky"). This REPLACES the cartoon
// DOM emoji Sun/Moon sprites in datamap.tsx; the day/night terminator
// (lib/celestial/ephemeris.ts nightPolygon) STAYS — this module never touches
// it.
//
// SOURCE OF THE EPHEMERIS — astronomy-engine (Don Cross, MIT license,
// https://github.com/cosinekitty/astronomy). Its VSOP87/ELP-derived series are
// validated by its author to arcsecond class against JPL Horizons — far tighter
// than the spec's ~0.1° position / ~1% angular-diameter bar. We do NOT
// re-derive orbital theory here (that would be strictly worse); this module is
// the honest COMPOSITION of astronomy-engine into topocentric az/el + apparent
// size + brightness + a WebGL2 sky scene. The tests validate our composition
// (frame convention, unit handling, angular-size formula, magnitude ordering)
// against astronomy-engine anchors at three epochs. astronomy-engine functions
// used: Observer, MakeTime, Equator (topocentric, of-date, aberration-
// corrected), Horizon (geometric — no atmospheric refraction), Illumination
// (magnitude + phase), Vector / RotateVector / HorizonFromVector +
// Rotation_ECT_EQD / Rotation_EQD_HOR (the ecliptic great circle and the
// per-body sky tracks).
//
// "REAL SCALE" HONESTY — the emptiness would break the renderer. The true
// scene distances span the Sun at 1 AU (1.5e11 m) to Neptune at 30 AU
// (4.5e12 m): >4 orders of magnitude, which destroys float32 precision and
// z-buffer resolution if drawn literally. So every body sits on ONE far sky
// sphere of fixed radius along its TRUE topocentric direction, drawn at its
// TRUE angular diameter and TRUE apparent magnitude. Apparent
// size + position + brightness IS the real, observable scale — what an eye at
// the observer actually sees — and that is exactly what this renders. (The
// literal-distance true-scale view already exists separately in
// lib/celestial/solarView.ts, for the zoomed-way-out solar-system mode.)
//
// Hermetic core: celestialState / the path builders / the geometry helpers are
// pure functions of (timeMs, observer) — no DOM, no clock reads, no network —
// unit-testable with `npx tsx --test`. Only mountCelestialSky touches WebGL.

// Dual-mode interop: astronomy-engine ships a CJS `main` (used by Node/tsx,
// where only the default = module.exports is reliably reachable) AND an ESM
// `module` build (used by the Vite/esbuild browser bundle, which exposes named
// exports but NO default). Importing the namespace and unwrapping an optional
// `.default` gives the real API object under BOTH toolchains — the one import
// form that survives the test runner and the browser build alike.
import * as AstronomyNS from "astronomy-engine";
const Astronomy: typeof AstronomyNS =
  (AstronomyNS as unknown as { default?: typeof AstronomyNS }).default ?? AstronomyNS;

/** Provenance caption the parent surfaces on the layer (verbatim per spec). */
export const CELESTIAL_PROVENANCE =
  "computed ephemeris (astronomy-engine) — apparent size and position accurate; not true scene distance";

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

// ── physical constants ───────────────────────────────────────────────────────

/** 1 astronomical unit in meters — exact by IAU 2012 Resolution B2. */
export const AU_M = 149_597_870_700;

/**
 * Volumetric mean radii, meters — NASA NSSDC planetary fact sheets
 * (nssdc.gsfc.nasa.gov/planetary/factsheet/); Sun = IAU 2015 Resolution B3
 * nominal solar radius. These set the TRUE apparent angular diameters. (Same
 * values as lib/celestial/solarSystem.ts BODY_RADIUS_M — kept local so this
 * astronomy-engine path carries no dependency on the Schlyter path.)
 */
export const BODY_RADIUS_M: Record<SkyBodyId, number> = {
  sun: 695_700_000,
  moon: 1_737_400,
  mercury: 2_439_700,
  venus: 6_051_800,
  mars: 3_389_500,
  jupiter: 69_911_000,
  saturn: 58_232_000,
  uranus: 25_362_000,
  neptune: 24_622_000,
};

export type SkyBodyId =
  | "sun" | "moon" | "mercury" | "venus" | "mars"
  | "jupiter" | "saturn" | "uranus" | "neptune";

interface SkyBodyDef {
  id: SkyBodyId;
  name: string;
  body: AstronomyNS.Body;
  /** approximate naked-eye display color — PRESENTATION only, not data. */
  color: [number, number, number];
}

/** Draw / listing order: Sun, Moon, then planets Sun-outward. Uranus/Neptune
 *  are included (spec: "optional") — they are naked-eye-marginal but real. */
export const SKY_BODIES: readonly SkyBodyDef[] = [
  { id: "sun", name: "Sun", body: Astronomy.Body.Sun, color: [1.0, 0.94, 0.72] },
  { id: "moon", name: "Moon", body: Astronomy.Body.Moon, color: [0.82, 0.82, 0.8] },
  { id: "mercury", name: "Mercury", body: Astronomy.Body.Mercury, color: [0.72, 0.66, 0.62] },
  { id: "venus", name: "Venus", body: Astronomy.Body.Venus, color: [0.95, 0.9, 0.72] },
  { id: "mars", name: "Mars", body: Astronomy.Body.Mars, color: [0.86, 0.42, 0.28] },
  { id: "jupiter", name: "Jupiter", body: Astronomy.Body.Jupiter, color: [0.86, 0.72, 0.56] },
  { id: "saturn", name: "Saturn", body: Astronomy.Body.Saturn, color: [0.88, 0.8, 0.62] },
  { id: "uranus", name: "Uranus", body: Astronomy.Body.Uranus, color: [0.62, 0.83, 0.86] },
  { id: "neptune", name: "Neptune", body: Astronomy.Body.Neptune, color: [0.43, 0.56, 0.86] },
] as const;

/** Fixed radius of the far sky sphere, scene units — arbitrary (angular size is
 *  radius-independent under perspective); large enough to sit at "infinity". */
export const SKY_SPHERE_RADIUS = 1000;

/**
 * Render floor for a body's angular diameter, degrees. The planets are
 * genuinely sub-arcminute (Jupiter ≈ 0.008–0.013°, Mars ≈ 0.001°); without a
 * floor they would be invisible sub-pixel specks. The floor is a RENDER
 * concern only — celestialState always reports the TRUE angular diameter; the
 * scene applies this floor so a planet is at least a legible point. ~0.35° is
 * roughly a comfortable naked-eye "bright point" and below the Sun/Moon's true
 * ~0.5°, so those two are never inflated.
 */
export const MIN_POINT_ANGULAR_DIAMETER_DEG = 0.35;

/** Naked-eye magnitude limit — bodies at/above this map to zero brightness. */
export const MAG_LIMIT = 6.5;
/** Reference bright magnitude (≈ Venus at brightest) for the brightness ramp. */
export const MAG_BRIGHT = -4.6;

// ── pure geometry (exported for tests) ───────────────────────────────────────

/**
 * Topocentric horizontal (az/el) → unit direction in the scene frame. Azimuth
 * is astronomy-engine's convention (degrees clockwise from geographic North,
 * through East); elevation is degrees above the horizon. The scene frame is the
 * right-handed GL y-up frame shared with lib/orbital/inspectScene.ts:
 *   x = East, y = Up (zenith), z = -North (south).
 * so az=0,el=0 (due north on the horizon) → (0,0,-1); az=90 → +x (east);
 * el=90 (zenith) → +y.
 */
export function directionVector(azDeg: number, elDeg: number): [number, number, number] {
  const a = azDeg * DEG;
  const e = elDeg * DEG;
  const ce = Math.cos(e);
  return [ce * Math.sin(a), Math.sin(e), -ce * Math.cos(a)];
}

/** True angular diameter, degrees: 2·asin(R/d). Exact spherical form; caps at
 *  180° if the observer is (nonphysically) inside the body. */
export function angularDiameterDeg(radiusM: number, distanceM: number): number {
  if (!(distanceM > radiusM)) return 180;
  return 2 * Math.asin(radiusM / distanceM) * RAD;
}

/**
 * Normalized linear interpolation between two unit direction vectors — the
 * per-frame smoothing between 1 Hz ephemeris updates. Bodies move ≲0.005°/s, so
 * a normalized lerp is visually identical to a slerp and cheaper; renormalizing
 * keeps the result on the unit sphere. t is clamped to [0,1].
 */
export function interpolateDirection(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  t: number,
): [number, number, number] {
  const s = t < 0 ? 0 : t > 1 ? 1 : t;
  const x = a[0] + (b[0] - a[0]) * s;
  const y = a[1] + (b[1] - a[1]) * s;
  const z = a[2] + (b[2] - a[2]) * s;
  const len = Math.hypot(x, y, z) || 1;
  return [x / len, y / len, z / len];
}

/**
 * Brightness 0..1 from apparent magnitude — monotonically DECREASING in
 * magnitude (brighter bodies have smaller/more-negative magnitudes). Linear in
 * magnitude between MAG_LIMIT (→0) and a floor below MAG_BRIGHT, then clamped;
 * the Sun (mag ≈ −26.7) and Moon saturate at 1. Used to scale point size/glow.
 */
export function magnitudePointScale(mag: number): number {
  const t = (MAG_LIMIT - mag) / (MAG_LIMIT - MAG_BRIGHT);
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

const dot3 = (a: readonly number[], b: readonly number[]): number =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * Sun glare/bloom intensity 0..1, growing as the camera LOOKS TOWARD the Sun
 * and ZOOMS IN (spec 3: "a bloom/lens-flare/glare that grows as the camera
 * looks toward it and zooms in. NO cartoon rays."). lookDir and sunDir are unit
 * scene vectors; fovDeg is the vertical field of view. Two multiplied factors:
 *   • centering — 1 when the Sun sits at frame center, ramping to 0 as its
 *     angular separation from the look axis approaches the fov half-angle;
 *   • zoom — grows as the fov narrows below a wide reference (a tight fov = a
 *     zoomed-in, magnified Sun = more glare).
 * Returns 0 when the Sun is outside the frame, so there is no glare when the
 * Sun is off-screen.
 */
export function glareIntensity(
  lookDir: readonly [number, number, number],
  sunDir: readonly [number, number, number],
  fovDeg: number,
): number {
  const sep = Math.acos(Math.max(-1, Math.min(1, dot3(lookDir, sunDir)))) * RAD;
  const half = Math.max(1, fovDeg * 0.5);
  const centering = sep >= half ? 0 : 1 - sep / half;
  const zoom = Math.max(0.15, Math.min(1, 45 / Math.max(1, fovDeg)));
  const g = centering * centering * zoom;
  return g < 0 ? 0 : g > 1 ? 1 : g;
}

// ── the core ephemeris function ──────────────────────────────────────────────

export interface CelestialBody {
  id: SkyBodyId;
  name: string;
  /** topocentric azimuth, degrees clockwise from North (astronomy-engine). */
  azDeg: number;
  /** topocentric elevation above the horizon, degrees (geometric, no
   *  atmospheric refraction — a ≤0.5° near-horizon effect, stated, not
   *  applied). Negative ⇒ below the horizon (behind the Earth from the
   *  observer). */
  elDeg: number;
  /** true apparent angular diameter, degrees (from the topocentric distance). */
  angularDiameterDeg: number;
  /** apparent visual magnitude (astronomy-engine Illumination). */
  apparentMagnitude: number;
  /** Sun–body phase angle, degrees. Most meaningful for the Moon and the inner
   *  planets (drives the illuminated fraction / terminator); ~0 for the Sun and
   *  small for the outer planets. */
  phaseAngle: number;
  /** illuminated fraction 0..1 of the disc facing the observer. */
  illuminatedFraction: number;
  /** TOPOCENTRIC distance observer→body, AU. Topocentric (not geocentric) so
   *  angularDiameterDeg = 2·asin(R/(distanceAu·AU)) holds exactly; differs from
   *  the geocentric distance by ≤1 Earth radius — negligible except the Moon
   *  (≤1.7%), where topocentric is the correct APPARENT value anyway. */
  distanceAu: number;
  /** unit direction in the scene frame (see directionVector). */
  dirScene: [number, number, number];
  /** convenience: elDeg < 0 (below the observer's horizon). */
  belowHorizon: boolean;
}

/**
 * Real-time topocentric state for every body, as seen from (observerLatDeg,
 * observerLonDeg) at dateMs. Pure function — no clock read, no DOM. This is the
 * function the tests pin against astronomy-engine anchors.
 */
export function celestialState(
  dateMs: number,
  observerLatDeg: number,
  observerLonDeg: number,
): CelestialBody[] {
  const time = Astronomy.MakeTime(new Date(dateMs));
  const observer = new Astronomy.Observer(observerLatDeg, observerLonDeg, 0);

  return SKY_BODIES.map((def) => {
    // topocentric, equator-of-date, aberration-corrected apparent place
    const eq = Astronomy.Equator(def.body, time, observer, true, true);
    // geometric horizontal (refraction omitted ⇒ airless az/el)
    const hor = Astronomy.Horizon(time, observer, eq.ra, eq.dec);
    const illum = Astronomy.Illumination(def.body, time);
    const distanceM = eq.dist * AU_M;
    const azDeg = hor.azimuth;
    const elDeg = hor.altitude;
    return {
      id: def.id,
      name: def.name,
      azDeg,
      elDeg,
      angularDiameterDeg: angularDiameterDeg(BODY_RADIUS_M[def.id], distanceM),
      apparentMagnitude: illum.mag,
      phaseAngle: illum.phase_angle,
      illuminatedFraction: illum.phase_fraction,
      distanceAu: eq.dist,
      dirScene: directionVector(azDeg, elDeg),
      belowHorizon: elDeg < 0,
    };
  });
}

// ── sky-render helpers (pure, exported for tests) ────────────────────────────

/** Position of a body on the far sky sphere: its unit direction × radius. */
export function skyPosition(
  dirScene: readonly [number, number, number],
  radius = SKY_SPHERE_RADIUS,
): [number, number, number] {
  return [dirScene[0] * radius, dirScene[1] * radius, dirScene[2] * radius];
}

/**
 * Billboard-disc radius (scene units) for a body on the sky sphere at the given
 * true angular diameter, with the render floor applied so a sub-arcminute
 * planet is still a legible point. radius·tan(angularRadius) renders at exactly
 * the intended angular size under the perspective projection.
 */
export function discRadiusOnSphere(
  trueAngularDiameterDeg: number,
  radius = SKY_SPHERE_RADIUS,
  minAngularDiameterDeg = MIN_POINT_ANGULAR_DIAMETER_DEG,
): number {
  const diam = Math.max(trueAngularDiameterDeg, minAngularDiameterDeg);
  return radius * Math.tan((diam * 0.5) * DEG);
}

// ── orbital-path builders (pure, exported for tests) ─────────────────────────
//
// spec 6: "functions returning the ecliptic great circle, the Moon's orbital
// track, and each planet's path across the sky as arrays of az/el … the parent
// can feed to its existing ArcLayer." All paths are for a SINGLE instant's sky:
// Earth's rotation is FROZEN at dateMs (the EQD→HOR rotation is evaluated once),
// so a path shows WHERE that curve lies on the observer's sky right now, rather
// than smearing it across the whole dome via diurnal motion.

export interface SkyPoint {
  azDeg: number;
  elDeg: number;
}

/** Default sky-track window per body, days (roughly the time to trace a full
 *  loop of its apparent path against the stars — the sidereal period, capped so
 *  the outer planets don't need centuries). */
const TRACK_WINDOW_DAYS: Record<SkyBodyId, number> = {
  sun: 365.256,
  moon: 27.321661,
  mercury: 87.969,
  venus: 224.701,
  mars: 686.98,
  jupiter: 365.256, // one year of its slow arc — a full 11.9 yr loop is uselessly long
  saturn: 365.256,
  uranus: 365.256,
  neptune: 365.256,
};

const MS_PER_DAY = 86_400_000;

/**
 * The ecliptic as a closed great circle on the observer's sky at dateMs:
 * ecliptic longitude 0→360° at ecliptic latitude 0, each rotated ecliptic-of-
 * date → equator-of-date → horizontal with the frozen sidereal time. The last
 * point repeats the first (closed ring). Being a great circle, every point is
 * exactly 90° from the ecliptic pole — the property the closure test checks.
 */
export function eclipticPath(
  dateMs: number,
  observerLatDeg: number,
  observerLonDeg: number,
  samples = 180,
): SkyPoint[] {
  const time = Astronomy.MakeTime(new Date(dateMs));
  const observer = new Astronomy.Observer(observerLatDeg, observerLonDeg, 0);
  const rotEctEqd = Astronomy.Rotation_ECT_EQD(time);
  const rotEqdHor = Astronomy.Rotation_EQD_HOR(time, observer);
  const out: SkyPoint[] = [];
  const n = Math.max(8, Math.floor(samples));
  for (let k = 0; k <= n; k++) {
    const lon = (k / n) * 2 * Math.PI;
    const ecl = new Astronomy.Vector(Math.cos(lon), Math.sin(lon), 0, time);
    const eqd = Astronomy.RotateVector(rotEctEqd, ecl);
    const hor = Astronomy.RotateVector(rotEqdHor, eqd);
    const sph = Astronomy.HorizonFromVector(hor, ""); // "" ⇒ geometric, no refraction
    out.push({ azDeg: sph.lon, elDeg: sph.lat });
  }
  return out;
}

/**
 * A body's orbital track across the observer's sky at dateMs: its apparent
 * position sampled forward/back over one window (TRACK_WINDOW_DAYS) and each
 * projected into the FROZEN sky (sidereal time fixed at dateMs). For the Moon
 * this is the ~5°-inclined monthly track; for a planet, its slow arc near the
 * ecliptic (retrograde loops included where the window spans them).
 */
export function bodySkyPath(
  dateMs: number,
  observerLatDeg: number,
  observerLonDeg: number,
  id: SkyBodyId,
  samples = 120,
  windowDays = TRACK_WINDOW_DAYS[id],
): SkyPoint[] {
  const def = SKY_BODIES.find((b) => b.id === id);
  if (!def) return [];
  const frozen = Astronomy.MakeTime(new Date(dateMs));
  const observer = new Astronomy.Observer(observerLatDeg, observerLonDeg, 0);
  const out: SkyPoint[] = [];
  const n = Math.max(8, Math.floor(samples));
  for (let k = 0; k <= n; k++) {
    const sampleMs = dateMs + ((k / n) - 0.5) * windowDays * MS_PER_DAY;
    const sampleTime = Astronomy.MakeTime(new Date(sampleMs));
    // the body's apparent ra/dec (equator-of-date) at the sample time …
    const eq = Astronomy.Equator(def.body, sampleTime, observer, true, true);
    // … placed in the sky using dateMs's FROZEN sidereal time
    const hor = Astronomy.Horizon(frozen, observer, eq.ra, eq.dec);
    out.push({ azDeg: hor.azimuth, elDeg: hor.altitude });
  }
  return out;
}

// ═════════════════════════════════════════════════════════════════════════════
// WebGL2 SKY SCENE
// ═════════════════════════════════════════════════════════════════════════════
//
// An always-on overlay canvas (the solarView / inspectScene handoff pattern:
// mount(container, opts) → { setTime, render, dispose, getRenderFailed,
// getState }). The parent aligns it to the map camera each frame via
// opts.getView(). Every FRAGMENT shader declares `precision highp float` and no
// shader mixes mediump/highp on a shared uniform — the mediump/highp link
// failure that rendered inspectScene BLACK must never recur (that is a tested
// contract below). GL failure LATCHES getRenderFailed and disables the scene;
// the page continues (modelLayer latch).

const BILLBOARD_VS = `#version 300 es
precision highp float;
in vec2 a_uv;              // quad corner in [-1,1]
uniform mat4 u_viewProj;
uniform vec3 u_center;     // body center on the sky sphere (scene units)
uniform vec3 u_right;      // billboard basis (unit)
uniform vec3 u_up;
uniform float u_radius;    // disc radius (scene units)
out vec2 v_uv;
void main() {
  vec3 world = u_center + (a_uv.x * u_right + a_uv.y * u_up) * u_radius;
  gl_Position = u_viewProj * vec4(world, 1.0);
  v_uv = a_uv;
}`;

// SUN — HDR limb-darkened disc + additive bloom/glare that grows with u_glare
// (CPU: glareIntensity). The quad is larger than the disc (u_coreFrac = disc /
// quad) so the glare halo extends well beyond the photosphere. NO rays.
const SUN_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform float u_coreFrac;  // disc radius as a fraction of the quad radius
uniform float u_glare;     // 0..1 bloom strength
uniform float u_alpha;     // horizon fade
out vec4 o;
void main() {
  float r = length(v_uv);
  if (r > 1.0) discard;
  // photosphere disc with limb darkening (cos-law approximation)
  float coreEdge = smoothstep(u_coreFrac, u_coreFrac * 0.82, r);
  float mu = sqrt(max(0.0, 1.0 - (r / max(u_coreFrac, 1e-4)) * (r / max(u_coreFrac, 1e-4))));
  float limb = 0.55 + 0.45 * mu;                 // brighter center, dimmer limb
  vec3 disc = vec3(1.0, 0.965, 0.86) * limb * coreEdge;
  // bloom: smooth radial falloff beyond the disc, scaled by glare + a faint
  // chromatic ring; additive, so it reads as light not a sprite.
  float halo = exp(-r * 3.2) * (0.35 + 1.65 * u_glare);
  float ring = exp(-abs(r - 0.5) * 9.0) * 0.12 * u_glare;
  vec3 glow = (vec3(1.0, 0.92, 0.78) * halo) + vec3(0.5, 0.7, 1.0) * ring;
  vec3 c = disc + glow;
  o = vec4(c * u_alpha, 1.0);                     // additive blend upstream
}`;

// MOON — the disc is shaded as a SPHERE lit by the real Sun direction, so the
// phase (illuminated fraction AND terminator orientation) falls out of the
// geometry rather than being painted on. A STYLIZED PROCEDURAL surface (a few
// fixed dark maria blotches) gives it texture — it is explicitly NOT lunar
// imagery and is labeled so; the load-bearing accuracy is the phase, size and
// position. Earthshine faintly lifts the dark limb when the phase is thin.
const MOON_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform vec3 u_right;
uniform vec3 u_up;
uniform vec3 u_facing;     // toward the viewer (unit)
uniform vec3 u_sun;        // scene-frame unit Sun direction
uniform float u_earthshine; // 0..1, strong only near new moon
uniform float u_alpha;     // horizon fade
out vec4 o;
// cheap stylized maria: darken near a few fixed disc positions (NOT imagery)
float maria(vec2 p) {
  float m = 0.0;
  m += smoothstep(0.42, 0.0, distance(p, vec2(-0.18,  0.22)));
  m += smoothstep(0.34, 0.0, distance(p, vec2( 0.10,  0.34)));
  m += smoothstep(0.30, 0.0, distance(p, vec2( 0.26, -0.06)));
  m += smoothstep(0.24, 0.0, distance(p, vec2(-0.30, -0.20)));
  return clamp(m, 0.0, 1.0);
}
void main() {
  float r2 = dot(v_uv, v_uv);
  if (r2 > 1.0) discard;
  float z = sqrt(max(0.0, 1.0 - r2));
  vec3 n = normalize(v_uv.x * u_right + v_uv.y * u_up + z * u_facing);
  float lit = dot(n, u_sun);
  float day = smoothstep(-0.04, 0.06, lit);       // real terminator
  float albedo = mix(0.62, 0.34, maria(v_uv));    // highlands vs maria
  vec3 sunlit = vec3(albedo) * (0.05 + 0.95 * max(lit, 0.0));
  vec3 earthlit = vec3(0.05, 0.07, 0.12) * u_earthshine; // ashen dark side
  vec3 c = mix(earthlit, sunlit, day);
  o = vec4(c * u_alpha, u_alpha);
}`;

// PLANET — a shaded sphere disc (lit by the Sun, so inner-planet phases show)
// with a magnitude-driven glow ring so a floored-to-visible point still reads
// as a colored star. Colors are presentation, not data.
const PLANET_FS = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform vec3 u_right;
uniform vec3 u_up;
uniform vec3 u_facing;
uniform vec3 u_sun;
uniform vec3 u_color;
uniform float u_bright;    // 0..1 magnitude brightness
uniform float u_alpha;     // horizon fade
out vec4 o;
void main() {
  float r2 = dot(v_uv, v_uv);
  if (r2 > 1.0) discard;
  float z = sqrt(max(0.0, 1.0 - r2));
  vec3 n = normalize(v_uv.x * u_right + v_uv.y * u_up + z * u_facing);
  float lit = max(dot(n, u_sun), 0.0);
  float shade = 0.18 + 0.82 * lit;                // phase-lit body
  float glow = (0.35 + 0.65 * u_bright) * exp(-length(v_uv) * 2.4);
  vec3 c = u_color * shade + u_color * glow * 0.6;
  o = vec4(c * u_alpha, u_alpha);
}`;

/** All fragment-shader sources, exported so the shader-contract test can assert
 *  every one declares highp and none uses mediump. */
export const CELESTIAL_FRAGMENT_SHADERS: readonly string[] = [SUN_FS, MOON_FS, PLANET_FS];
export const CELESTIAL_VERTEX_SHADERS: readonly string[] = [BILLBOARD_VS];

// ── camera view fed by the parent each frame ─────────────────────────────────

export interface SkyView {
  /** UTC epoch ms the sky is drawn for (parent may scrub time). */
  timeMs: number;
  observerLatDeg: number;
  observerLonDeg: number;
  /** where the camera looks, in the same topocentric az/el as the bodies. */
  lookAzDeg: number;
  lookElDeg: number;
  /** vertical field of view, degrees (narrows as the user zooms in). */
  fovDeg: number;
}

export interface CelestialSkyOptions {
  /** live per-frame camera source the parent wires to the map. */
  getView: () => SkyView;
  /** minimum ms between ephemeris updates (default 1000 = 1 Hz per spec). */
  updateIntervalMs?: number;
}

export interface CelestialSkyHandle {
  /** jump the sky clock (parent time-scrubber); realtime advance continues
   *  from here via getView().timeMs each frame. */
  setTime(dateMs: number): void;
  render(): void;
  dispose(): void;
  /** true after a GL failure — scene disabled, page continues (modelLayer latch). */
  getRenderFailed(): boolean;
  getState(): { bodies: number; lastEphemerisMs: number };
}

interface GlProgram {
  prog: WebGLProgram;
  attrs: Record<string, number>;
  unis: Record<string, WebGLUniformLocation | null>;
}

// tiny column-major mat4 (same math as lib/orbital/inspectScene.ts) ──────────
type Mat4 = Float32Array;
function mat4Perspective(fovyRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovyRad / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect; m[5] = f; m[10] = (far + near) * nf; m[11] = -1; m[14] = 2 * far * near * nf;
  return m;
}
function mat4LookDir(dir: [number, number, number]): Mat4 {
  // eye at origin looking along dir; up = +Y, falling back to +Z near the pole
  let fx = dir[0], fy = dir[1], fz = dir[2];
  let l = Math.hypot(fx, fy, fz) || 1; fx /= l; fy /= l; fz /= l;
  let upx = 0, upy = 1, upz = 0;
  if (Math.abs(fy) > 0.999) { upx = 0; upy = 0; upz = 1; }
  // s = f × up
  let sx = fy * upz - fz * upy, sy = fz * upx - fx * upz, sz = fx * upy - fy * upx;
  l = Math.hypot(sx, sy, sz) || 1; sx /= l; sy /= l; sz /= l;
  // u = s × f
  const ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;
  const m = new Float32Array(16);
  m[0] = sx; m[4] = sy; m[8] = sz;
  m[1] = ux; m[5] = uy; m[9] = uz;
  m[2] = -fx; m[6] = -fy; m[10] = -fz;
  m[15] = 1;
  return m; // eye at origin ⇒ no translation column
}
function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let col = 0; col < 4; col++)
    for (let row = 0; row < 4; row++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + row] * b[col * 4 + k];
      o[col * 4 + row] = s;
    }
  return o;
}

/**
 * Mount the always-on celestial sky over `container`. Self-contained WebGL2
 * overlay; the parent composites it behind the map markers and feeds the map
 * camera through opts.getView(). Ephemeris is evaluated at ≤ updateIntervalMs
 * (default 1 Hz) into a forward pair and interpolated per frame; the render
 * path allocates no per-frame arrays (all buffers/scratch preallocated).
 */
export function mountCelestialSky(
  container: HTMLElement,
  opts: CelestialSkyOptions,
): CelestialSkyHandle {
  const updateIntervalMs = opts.updateIntervalMs ?? 1000;
  let disposed = false;
  let renderFailed = false;

  // ── ephemeris cadence + interpolation state ──
  let lastEphemerisMs = 0;         // sky-clock ms of the current forward pair
  let stateA: CelestialBody[] = []; // sample at the tick
  let stateB: CelestialBody[] = []; // sample at tick + updateIntervalMs
  // preallocated per-body scratch (no per-frame allocation)
  const bodyDir = SKY_BODIES.map(() => [0, 0, 0] as [number, number, number]);
  const bodyScalars = SKY_BODIES.map(() => ({ diam: 0, mag: 0, illum: 1, phase: 0, el: 0 }));

  function refreshEphemeris(view: SkyView): void {
    stateA = celestialState(view.timeMs, view.observerLatDeg, view.observerLonDeg);
    stateB = celestialState(view.timeMs + updateIntervalMs, view.observerLatDeg, view.observerLonDeg);
    lastEphemerisMs = view.timeMs;
  }

  function interpolate(view: SkyView): void {
    // fraction through the current 1 Hz window (clamped; scrubs re-tick below)
    const frac = (view.timeMs - lastEphemerisMs) / updateIntervalMs;
    for (let i = 0; i < SKY_BODIES.length; i++) {
      const a = stateA[i], b = stateB[i];
      const d = interpolateDirection(a.dirScene, b.dirScene, frac);
      bodyDir[i][0] = d[0]; bodyDir[i][1] = d[1]; bodyDir[i][2] = d[2];
      const s = bodyScalars[i];
      const f = frac < 0 ? 0 : frac > 1 ? 1 : frac;
      s.diam = a.angularDiameterDeg + (b.angularDiameterDeg - a.angularDiameterDeg) * f;
      s.mag = a.apparentMagnitude + (b.apparentMagnitude - a.apparentMagnitude) * f;
      s.illum = a.illuminatedFraction + (b.illuminatedFraction - a.illuminatedFraction) * f;
      s.phase = a.phaseAngle + (b.phaseAngle - a.phaseAngle) * f;
      s.el = a.elDeg + (b.elDeg - a.elDeg) * f;
    }
  }

  const doc: Document | null =
    (container && (container as { ownerDocument?: Document }).ownerDocument) ??
    (typeof document !== "undefined" ? document : null);

  const inert = (): CelestialSkyHandle => ({
    setTime() { /* disabled */ },
    render() { /* disabled */ },
    dispose() { disposed = true; },
    getRenderFailed: () => renderFailed,
    getState: () => ({ bodies: SKY_BODIES.length, lastEphemerisMs }),
  });
  if (!doc) { renderFailed = true; return inert(); }

  const canvas = doc.createElement("canvas");
  canvas.className = "vt-celestial-sky";
  try {
    canvas.style.position = "absolute";
    canvas.style.inset = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.pointerEvents = "none"; // the sky never eats map gestures
    container.appendChild(canvas);
  } catch { renderFailed = true; return inert(); }

  let gl: WebGL2RenderingContext | null = null;
  let sunProg: GlProgram | null = null;
  let moonProg: GlProgram | null = null;
  let planetProg: GlProgram | null = null;
  let bufQuad: WebGLBuffer | null = null;

  function compile(vsSrc: string, fsSrc: string, attrs: string[], unis: string[]): GlProgram {
    const g = gl!;
    const mk = (type: number, src: string): WebGLShader => {
      const sh = g.createShader(type);
      if (!sh) throw new Error("celestialSky: createShader failed");
      g.shaderSource(sh, src); g.compileShader(sh);
      if (!g.getShaderParameter(sh, g.COMPILE_STATUS)) {
        const log = g.getShaderInfoLog(sh); g.deleteShader(sh);
        throw new Error("celestialSky: shader compile failed: " + log);
      }
      return sh;
    };
    const vs = mk(g.VERTEX_SHADER, vsSrc);
    const fs = mk(g.FRAGMENT_SHADER, fsSrc);
    const p = g.createProgram();
    if (!p) throw new Error("celestialSky: createProgram failed");
    g.attachShader(p, vs); g.attachShader(p, fs); g.linkProgram(p);
    if (!g.getProgramParameter(p, g.LINK_STATUS)) {
      const log = g.getProgramInfoLog(p); g.deleteProgram(p);
      throw new Error("celestialSky: program link failed: " + log);
    }
    g.deleteShader(vs); g.deleteShader(fs);
    const out: GlProgram = { prog: p, attrs: {}, unis: {} };
    for (const a of attrs) out.attrs[a] = g.getAttribLocation(p, a);
    for (const u of unis) out.unis[u] = g.getUniformLocation(p, u);
    return out;
  }

  const BILLBOARD_ATTRS = ["a_uv"];
  try {
    gl = canvas.getContext("webgl2", { antialias: true, alpha: true, premultipliedAlpha: false }) as
      WebGL2RenderingContext | null;
    if (!gl) throw new Error("celestialSky: WebGL2 unavailable");
    sunProg = compile(BILLBOARD_VS, SUN_FS, BILLBOARD_ATTRS,
      ["u_viewProj", "u_center", "u_right", "u_up", "u_radius", "u_coreFrac", "u_glare", "u_alpha"]);
    moonProg = compile(BILLBOARD_VS, MOON_FS, BILLBOARD_ATTRS,
      ["u_viewProj", "u_center", "u_right", "u_up", "u_radius", "u_facing", "u_sun", "u_earthshine", "u_alpha"]);
    planetProg = compile(BILLBOARD_VS, PLANET_FS, BILLBOARD_ATTRS,
      ["u_viewProj", "u_center", "u_right", "u_up", "u_radius", "u_facing", "u_sun", "u_color", "u_bright", "u_alpha"]);
    bufQuad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, bufQuad);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW);
  } catch (e) {
    renderFailed = true;
    // eslint-disable-next-line no-console
    console.error("celestialSky: disabling after GL init failure (page continues):", e);
  }

  function resizeBacking(): void {
    const dpr = (globalThis.devicePixelRatio as number | undefined) || 1;
    const w = canvas.clientWidth || container.clientWidth || 800;
    const h = canvas.clientHeight || container.clientHeight || 600;
    const bw = Math.max(1, Math.round(w * dpr));
    const bh = Math.max(1, Math.round(h * dpr));
    if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
  }

  // horizon fade: 1 well above the horizon, 0 below it (Earth blocks the body).
  const HORIZON_FADE_DEG = 3;
  const horizonAlpha = (elDeg: number): number => {
    if (elDeg <= -HORIZON_FADE_DEG) return 0;
    if (elDeg >= 0) return 1;
    return (elDeg + HORIZON_FADE_DEG) / HORIZON_FADE_DEG;
  };

  // scratch billboard basis (no per-frame allocation)
  const scratchRight: [number, number, number] = [0, 0, 0];
  const scratchUp: [number, number, number] = [0, 0, 0];
  function billboardBasis(dir: readonly [number, number, number]): void {
    // basis perpendicular to dir, facing the origin/camera
    const refX = Math.abs(dir[1]) > 0.9 ? 1 : 0;
    const refY = Math.abs(dir[1]) > 0.9 ? 0 : 1;
    let rx = dir[1] * 0 - dir[2] * refY;
    let ry = dir[2] * refX - dir[0] * 0;
    let rz = dir[0] * refY - dir[1] * refX;
    const rl = Math.hypot(rx, ry, rz) || 1; rx /= rl; ry /= rl; rz /= rl;
    scratchRight[0] = rx; scratchRight[1] = ry; scratchRight[2] = rz;
    scratchUp[0] = ry * dir[2] - rz * dir[1];
    scratchUp[1] = rz * dir[0] - rx * dir[2];
    scratchUp[2] = rx * dir[1] - ry * dir[0];
  }

  function drawFrame(): void {
    if (disposed || renderFailed || !gl) return;
    try { drawFrameInner(); }
    catch (e) {
      renderFailed = true;
      // eslint-disable-next-line no-console
      console.error("celestialSky: disabling after render failure (page continues):", e);
    }
  }

  function drawFrameInner(): void {
    const g = gl!;
    const view = opts.getView();
    // (re)tick the ephemeris at ≤1 Hz, or immediately after a time scrub
    if (stateA.length === 0 ||
        view.timeMs < lastEphemerisMs ||
        view.timeMs - lastEphemerisMs >= updateIntervalMs) {
      refreshEphemeris(view);
    }
    interpolate(view);

    resizeBacking();
    g.viewport(0, 0, canvas.width, canvas.height);
    g.clearColor(0, 0, 0, 0); // transparent — the map/starfield shows through
    g.clear(g.COLOR_BUFFER_BIT);
    g.disable(g.DEPTH_TEST); // all bodies on one far sphere: painter's order
    g.enable(g.BLEND);

    const aspect = canvas.width / Math.max(1, canvas.height);
    const lookDir = directionVector(view.lookAzDeg, view.lookElDeg);
    const proj = mat4Perspective(view.fovDeg * DEG, aspect, 0.1, SKY_SPHERE_RADIUS * 4);
    const viewM = mat4LookDir(lookDir);
    const viewProj = mat4Multiply(proj, viewM);

    const sun = bodyDir[0]; // Sun is index 0 in SKY_BODIES
    const glare = glareIntensity(lookDir, sun, view.fovDeg);

    const bindQuad = (p: GlProgram): void => {
      g.bindBuffer(g.ARRAY_BUFFER, bufQuad);
      g.enableVertexAttribArray(p.attrs.a_uv);
      g.vertexAttribPointer(p.attrs.a_uv, 2, g.FLOAT, false, 0, 0);
    };

    // ── SUN (additive bloom) ──
    {
      const i = 0, s = bodyScalars[i], dir = bodyDir[i];
      const alpha = horizonAlpha(s.el);
      if (alpha > 0) {
        const p = sunProg!;
        const discR = discRadiusOnSphere(s.diam);
        const quadR = discR * (2.5 + 6 * glare); // glare halo extends past the disc
        billboardBasis(dir);
        g.useProgram(p.prog);
        g.blendFunc(g.SRC_ALPHA, g.ONE); // additive light
        g.uniformMatrix4fv(p.unis.u_viewProj, false, viewProj);
        g.uniform3f(p.unis.u_center, dir[0] * SKY_SPHERE_RADIUS, dir[1] * SKY_SPHERE_RADIUS, dir[2] * SKY_SPHERE_RADIUS);
        g.uniform3f(p.unis.u_right, scratchRight[0], scratchRight[1], scratchRight[2]);
        g.uniform3f(p.unis.u_up, scratchUp[0], scratchUp[1], scratchUp[2]);
        g.uniform1f(p.unis.u_radius, quadR);
        g.uniform1f(p.unis.u_coreFrac, discR / quadR);
        g.uniform1f(p.unis.u_glare, glare);
        g.uniform1f(p.unis.u_alpha, alpha);
        bindQuad(p);
        g.drawArrays(g.TRIANGLES, 0, 6);
      }
    }

    // ── PLANETS (over-blend) ──
    g.blendFunc(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA);
    for (let i = 2; i < SKY_BODIES.length; i++) {
      const def = SKY_BODIES[i], s = bodyScalars[i], dir = bodyDir[i];
      const alpha = horizonAlpha(s.el);
      if (alpha <= 0) continue;
      const p = planetProg!;
      const discR = discRadiusOnSphere(s.diam);
      billboardBasis(dir);
      g.useProgram(p.prog);
      g.uniformMatrix4fv(p.unis.u_viewProj, false, viewProj);
      g.uniform3f(p.unis.u_center, dir[0] * SKY_SPHERE_RADIUS, dir[1] * SKY_SPHERE_RADIUS, dir[2] * SKY_SPHERE_RADIUS);
      g.uniform3f(p.unis.u_right, scratchRight[0], scratchRight[1], scratchRight[2]);
      g.uniform3f(p.unis.u_up, scratchUp[0], scratchUp[1], scratchUp[2]);
      g.uniform1f(p.unis.u_radius, discR);
      g.uniform3f(p.unis.u_facing, -dir[0], -dir[1], -dir[2]);
      g.uniform3f(p.unis.u_sun, sun[0], sun[1], sun[2]);
      g.uniform3f(p.unis.u_color, def.color[0], def.color[1], def.color[2]);
      g.uniform1f(p.unis.u_bright, magnitudePointScale(s.mag));
      g.uniform1f(p.unis.u_alpha, alpha);
      bindQuad(p);
      g.drawArrays(g.TRIANGLES, 0, 6);
    }

    // ── MOON (over-blend, last so it occludes during syzygy) ──
    {
      const i = 1, s = bodyScalars[i], dir = bodyDir[i];
      const alpha = horizonAlpha(s.el);
      if (alpha > 0) {
        const p = moonProg!;
        const discR = discRadiusOnSphere(s.diam);
        // earthshine strong only near new moon (phase near 180° ⇒ thin crescent)
        const earthshine = Math.max(0, Math.min(1, (s.phase - 90) / 90)) * 0.8;
        billboardBasis(dir);
        g.useProgram(p.prog);
        g.uniformMatrix4fv(p.unis.u_viewProj, false, viewProj);
        g.uniform3f(p.unis.u_center, dir[0] * SKY_SPHERE_RADIUS, dir[1] * SKY_SPHERE_RADIUS, dir[2] * SKY_SPHERE_RADIUS);
        g.uniform3f(p.unis.u_right, scratchRight[0], scratchRight[1], scratchRight[2]);
        g.uniform3f(p.unis.u_up, scratchUp[0], scratchUp[1], scratchUp[2]);
        g.uniform1f(p.unis.u_radius, discR);
        g.uniform3f(p.unis.u_facing, -dir[0], -dir[1], -dir[2]);
        g.uniform3f(p.unis.u_sun, sun[0], sun[1], sun[2]);
        g.uniform1f(p.unis.u_earthshine, earthshine);
        g.uniform1f(p.unis.u_alpha, alpha);
        bindQuad(p);
        g.drawArrays(g.TRIANGLES, 0, 6);
      }
    }

    g.disable(g.BLEND);
  }

  let ro: ResizeObserver | null = null;
  if (typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(() => drawFrame());
    ro.observe(container);
  }

  // continuous RAF while mounted (the sky MOVES); parent-driven render() also
  // works where there is no RAF (tests/SSR).
  let rafId = 0;
  const hasRaf = typeof requestAnimationFrame !== "undefined";
  const loop = (): void => {
    if (disposed || renderFailed) return;
    drawFrame();
    rafId = requestAnimationFrame(loop);
  };
  if (hasRaf && !renderFailed) rafId = requestAnimationFrame(loop);
  drawFrame();

  return {
    setTime(dateMs: number): void {
      // force a re-tick at the new sky time on the next frame
      lastEphemerisMs = dateMs - updateIntervalMs - 1;
      drawFrame();
    },
    render(): void { drawFrame(); },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      if (hasRaf && rafId) cancelAnimationFrame(rafId);
      try { ro?.disconnect(); } catch { /* already gone */ }
      if (gl) {
        try {
          if (bufQuad) gl.deleteBuffer(bufQuad);
          for (const p of [sunProg, moonProg, planetProg]) if (p) gl.deleteProgram(p.prog);
        } catch { /* context already lost */ }
      }
      try { canvas.remove(); } catch { /* detached */ }
    },
    getRenderFailed: () => renderFailed,
    getState: () => ({ bodies: SKY_BODIES.length, lastEphemerisMs }),
  };
}
