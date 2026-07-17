// INSPECT MODE — EARTH TWIN self-contained WebGL2 inspection scene (human
// directive: "when you're clicked on a sat it should have different pan zoom
// mechanics … look at the sat render model and have the camera angle looking
// away from the earth … if the space station was passing on the moon side
// you could see it and look at the moon with the ISS in view").
//
// WHAT THIS IS: a free-orbit camera around the CRAFT, not the map camera.
// mount(container, opts) / handle.setMesh() / handle.render() / dispose() —
// the same handoff pattern as lib/celestial/solarView.ts. The parent owns
// the entry/exit wiring (inspect chip, map pause/resume, live getState feed).
//
// HONESTY TIERS (each also stated in INSPECT_PROVENANCE, which the parent
// shows on the card):
//   CRAFT  — the injected mesh (real .vtm model or class-representative
//            form; the parent's meshLabel carries its own derivation
//            caption), lit by the REAL sun direction computed from the
//            subsolar-point ephemeris (~0.01°) at the craft's position,
//            plus a soft earthshine fill from the nadir whose strength
//            follows whether the Earth below is actually sunlit.
//   EARTH  — a sphere of TRUE ANGULAR SIZE as seen from the craft's
//            altitude (2·asin(R/(R+h))), in the nadir direction, shaded by
//            the REAL day/night terminator (dot of surface normal with the
//            sun direction). The day side is a simple blue-marble-ish
//            gradient, the night side near-black with a thin atmosphere
//            rim. It is a simplified globe — NEVER live imagery, and never
//            claimed to be.
//   MOON   — a disc at the real ephemeris direction (moonState) and true
//            angular size, shaded as a sphere lit by the real sun
//            direction, so the phase (illuminated fraction AND lit-side
//            orientation) is the real one. This is the "ISS passing on the
//            moon side" moment — the position is real ephemeris, never
//            decorative.
//   SUN    — a bright disc at the real solar direction, true angular size
//            (~0.53° diameter).
//   SKY    — pure black. No fake starfield: "real position or absent"
//            (the solarView precedent). A real bright-star catalog is
//            future work.
//   VELOCITY — an optional thin direction line DERIVED from successive
//            getState positions (finite difference). Absent until two
//            distinct samples exist — derived, never fabricated.
//
// FRAME MATH CONVENTION (validated in inspectScene.test.ts):
//   The craft sits at the SCENE ORIGIN. Its local frame is ENU at its
//   geocentric lat/lon (spherical Earth, R = 6371 km — the same model as
//   lib/orbital/geometry.ts): E = east, N = north, U = zenith (radially
//   away from Earth's center). Scene axes are the right-handed GL mapping
//     scene.x = E,  scene.y = U (up),  scene.z = -N (south)
//   so nadir is scene (0,-1,0). A celestial direction is taken as the
//   GEOCENTRIC unit vector through its sub-point (ecefUnit of the
//   subsolar/sublunar lat/lon), then rotated into ENU (ecefToEnu). For the
//   SUN the craft-vs-geocenter parallax is < 0.003° (Earth radius vs 1 AU)
//   — negligible. For the MOON see moonDirLocal's honesty note.
//
// SCALE HONESTY: the craft mesh is meters-sized; Earth/Moon/Sun are drawn
// by ANGULAR size at fixed scene distances (Earth 4000, Moon 9000, Sun
// 9500 model units — the Sun honestly behind the Moon, so a real syzygy
// occludes correctly). Angular sizes are exact from the origin; the camera
// orbits up to ~50× the craft extent (~120 units), so off-origin parallax
// distorts apparent sizes by ≲3% — a stated display approximation, not a
// data claim. Earth's sphere also depth-occludes the Moon and Sun discs,
// so a body below the craft's horizon is honestly hidden.

import { subsolarPoint, moonState } from '../celestial/ephemeris.js';
import type { Mesh } from './model3d.js';

// ── constants ───────────────────────────────────────────────────────────────

/** Mean Earth radius, meters — matches lib/orbital/geometry.ts (spherical). */
export const EARTH_RADIUS_M = 6_371_000;
/** True mean solar angular RADIUS, radians (~0.533° diameter). */
export const SUN_ANGULAR_RADIUS_RAD = (0.533 / 2) * (Math.PI / 180);
/** Scene distance (model units) at which the Earth sphere is placed. */
export const EARTH_SCENE_DIST = 4000;
/** Scene distance of the Moon disc. */
export const MOON_SCENE_DIST = 9000;
/** Scene distance of the Sun disc — behind the Moon, so eclipses occlude. */
export const SUN_SCENE_DIST = 9500;
/** Default craft half-extent when no mesh is present (model3d builds to ~1.2). */
export const DEFAULT_EXTENT = 1.2;

/** Provenance caption the parent shows on the card — every tier labeled. */
export const INSPECT_PROVENANCE =
  'COMPUTED EPHEMERIS VIEW — craft lit by the real sun direction (subsolar ephemeris, ~0.01°) ' +
  'with earthshine fill only when the Earth below is sunlit. ' +
  'Earth shown as a simplified globe with the real day/night terminator — not live imagery. ' +
  'Moon at its real ephemeris direction, true angular size, real phase (display-grade, ~1°). ' +
  'Sun disc at true angular size (~0.53°). Black sky: no decorative starfield — real positions or absent.';

const DEG = Math.PI / 180;

// ── frame math (pure, exported for tests) ───────────────────────────────────

/** A direction in the craft's local East/North/Up frame (unit-ish). */
export interface EnuDir {
  e: number;
  n: number;
  u: number;
}

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

/**
 * REAL sun direction in the craft's local ENU frame: the geocentric unit
 * vector through the subsolar point, rotated to ENU at the craft. Sun
 * parallax between the craft and the geocenter is < 0.003° (R⊕/1 AU) —
 * far below the ephemeris' own ~0.01° grade.
 * Anchors (see tests): craft AT the subsolar point → sun on the zenith
 * axis {0,0,1} (sub-point directly below the craft, sun co-linear beyond);
 * craft at the antipode → nadir {0,0,-1}; subsolar 90° east on the
 * equator → due east on the horizon {1,0,0}.
 */
export function sunDirLocal(timeMs: number, craftLatDeg: number, craftLonDeg: number): EnuDir {
  const sp = subsolarPoint(timeMs);
  return ecefToEnu(ecefUnit(sp.latDeg, sp.lonDeg), craftLatDeg, craftLonDeg);
}

export interface MoonLocal {
  dir: EnuDir;
  /** true angular RADIUS from the instantaneous distance, radians. */
  angularRadiusRad: number;
  /** illuminated fraction 0..1 (ephemeris passthrough, display grade). */
  illuminatedFraction: number;
}

/**
 * REAL moon direction + true angular size in the craft's local frame.
 * HONESTY: the direction is the GEOCENTRIC one (through the sublunar
 * point). Topocentric parallax from a LEO craft displaced ~6771 km from
 * the geocenter can reach ~1.0° (asin(6771/385000)) — the same order as
 * the truncated lunar series' own stated sublunar accuracy (~1°), so the
 * approximation does not degrade the display grade. The ANGULAR SIZE
 * error from the same displacement is < 2% of a 0.52° disc (< 0.02°).
 * Display-grade, labeled as such; never survey ephemeris.
 */
export function moonDirLocal(timeMs: number, craftLatDeg: number, craftLonDeg: number): MoonLocal {
  const m = moonState(timeMs);
  return {
    dir: ecefToEnu(ecefUnit(m.latDeg, m.lonDeg), craftLatDeg, craftLonDeg),
    angularRadiusRad: (m.angularSizeArcmin / 2 / 60) * DEG,
    illuminatedFraction: m.illuminatedFraction,
  };
}

/** Earth's true angular RADIUS from altitude h: asin(R/(R+h)). The full
 *  disc is twice this — ~140° at LEO (400 km), ~17.4° at GEO. */
export function earthAngularRadiusRad(altMeters: number): number {
  const h = Math.max(1, altMeters);
  return Math.asin(EARTH_RADIUS_M / (EARTH_RADIUS_M + h));
}

/** Geometric illuminated fraction from the moon/sun separation angle —
 *  (1 − cos(separation))/2, the same elongation form the ephemeris uses;
 *  exported so tests can pin shader-geometry ↔ ephemeris consistency. */
export function phaseFractionFromDirs(moonDir: EnuDir, sunDir: EnuDir): number {
  const dot = moonDir.e * sunDir.e + moonDir.n * sunDir.n + moonDir.u * sunDir.u;
  return (1 - Math.max(-1, Math.min(1, dot))) / 2;
}

export interface CraftState {
  latDeg: number;
  lonDeg: number;
  altMeters: number;
  timeMs: number;
}

/**
 * Velocity direction (scene frame, unit) DERIVED from two position
 * samples — finite difference of ECEF positions, expressed in the current
 * craft's ENU. Returns null when the motion is below 1 m (never
 * fabricate a heading from noise or identical samples).
 */
export function deriveVelocityDirScene(
  prev: Pick<CraftState, 'latDeg' | 'lonDeg' | 'altMeters'>,
  curr: Pick<CraftState, 'latDeg' | 'lonDeg' | 'altMeters'>,
): [number, number, number] | null {
  const p0 = ecefUnit(prev.latDeg, prev.lonDeg);
  const p1 = ecefUnit(curr.latDeg, curr.lonDeg);
  const r0 = EARTH_RADIUS_M + prev.altMeters;
  const r1 = EARTH_RADIUS_M + curr.altMeters;
  const d: [number, number, number] = [
    p1[0] * r1 - p0[0] * r0,
    p1[1] * r1 - p0[1] * r0,
    p1[2] * r1 - p0[2] * r0,
  ];
  const len = Math.hypot(d[0], d[1], d[2]);
  if (len < 1) return null;
  const enu = ecefToEnu([d[0] / len, d[1] / len, d[2] / len], curr.latDeg, curr.lonDeg);
  return enuToScene(enu);
}

// ── free-orbit camera (pure, exported for tests) ────────────────────────────
//
// THE POINT of inspect mode: the camera orbits the CRAFT, not the ground.
// Yaw is a full 360°; pitch spans ±89° INCLUDING the nadir side (camera
// below the craft looking up — Earth behind your back, craft against open
// space / the Moon). Dolly runs 2×–50× the craft extent. There is NO
// ground constraint and NO snap-back: the map camera's re-centering fight
// is exactly the failure this replaces.

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
  /** release inertia, rad/s (integrated by camStep when not dragging). */
  velYaw: number;
  velPitch: number;
  distTarget: number;
  minDist: number;
  maxDist: number;
  dragging: boolean;
}

/** Dolly bounds from the craft's half-extent: ~2× to ~50×. */
export function camDollyBounds(extent: number): { min: number; max: number } {
  const e = Math.max(1e-6, extent);
  return { min: 2 * e, max: 50 * e };
}

export function wrapAngle(a: number): number {
  const w = ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return w;
}

export function createOrbitCam(extent: number): OrbitCam {
  const b = camDollyBounds(extent);
  // opening pose: slightly above and beside the craft, Earth in frame
  const dist = Math.min(b.max, Math.max(b.min, extent * 6));
  return {
    yaw: 0.6,
    pitch: 0.32,
    dist,
    velYaw: 0,
    velPitch: 0,
    distTarget: dist,
    minDist: b.min,
    maxDist: b.max,
    dragging: false,
  };
}

/** Pointer drag: immediate orbit + a velocity estimate for release inertia. */
export function camApplyDrag(cam: OrbitCam, dxPx: number, dyPx: number): void {
  cam.yaw = wrapAngle(cam.yaw - dxPx * DRAG_RAD_PER_PX);
  cam.pitch = Math.max(-PITCH_LIMIT_RAD, Math.min(PITCH_LIMIT_RAD, cam.pitch + dyPx * DRAG_RAD_PER_PX));
  // rad/s assuming ~60 Hz pointer cadence — only consumed after release
  cam.velYaw = -dxPx * DRAG_RAD_PER_PX * 60;
  cam.velPitch = dyPx * DRAG_RAD_PER_PX * 60;
}

/** Wheel/pinch dolly: multiplies the target distance, clamped to bounds. */
export function camApplyDolly(cam: OrbitCam, factor: number): void {
  if (!(factor > 0) || !Number.isFinite(factor)) return;
  cam.distTarget = Math.max(cam.minDist, Math.min(cam.maxDist, cam.distTarget * factor));
}

/**
 * Advance inertia + smoothing by dtMs. Returns true while still moving
 * (the RAF loop may idle when everything has settled and no state feed
 * demands a redraw). Damping is exponential — converges, never rings.
 */
export function camStep(cam: OrbitCam, dtMs: number): boolean {
  const dt = Math.min(0.1, Math.max(0, dtMs / 1000));
  if (!cam.dragging && (cam.velYaw !== 0 || cam.velPitch !== 0)) {
    cam.yaw = wrapAngle(cam.yaw + cam.velYaw * dt);
    const p = cam.pitch + cam.velPitch * dt;
    cam.pitch = Math.max(-PITCH_LIMIT_RAD, Math.min(PITCH_LIMIT_RAD, p));
    if (cam.pitch !== p) cam.velPitch = 0; // hit the limit → stop, no bounce
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

/** Camera eye position (scene units): spherical orbit around the origin.
 *  pitch>0 = above the craft (zenith side); pitch<0 = BELOW it — Earth at
 *  your back, looking up past the craft into space. */
export function camEye(cam: OrbitCam): [number, number, number] {
  const cp = Math.cos(cam.pitch);
  return [
    cam.dist * cp * Math.sin(cam.yaw),
    cam.dist * Math.sin(cam.pitch),
    cam.dist * cp * Math.cos(cam.yaw),
  ];
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
  // up = +Y; pitch is clamped to ±89° so the basis never degenerates
  let fx = target[0] - eye[0], fy = target[1] - eye[1], fz = target[2] - eye[2];
  let l = Math.hypot(fx, fy, fz) || 1;
  fx /= l; fy /= l; fz /= l;
  // s = normalize(f × up) with up = (0,1,0) → f × up = (-fz, 0, fx)
  let sx = -fz; const sy = 0; let sz = fx;
  l = Math.hypot(sx, sy, sz) || 1;
  sx /= l; sz /= l;
  // u = s × f
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

// ── shaders ─────────────────────────────────────────────────────────────────

const MESH_VS = `#version 300 es
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

// REAL sun key light + earthshine fill from the nadir (0,-1,0), whose
// strength (u_fillK, CPU-computed) is ~max(sunUp, 0): the Earth face below
// the craft reflects only when it is actually sunlit.
const MESH_FS = `#version 300 es
precision mediump float;
in vec3 v_normal;
in vec3 v_color;
uniform vec3 u_sun;    // scene-frame unit sun direction (real ephemeris)
uniform float u_fillK; // earthshine strength 0..1
out vec4 o;
void main() {
  vec3 n = normalize(v_normal);
  float sun = max(dot(n, u_sun), 0.0);
  float fill = max(dot(n, vec3(0.0, -1.0, 0.0)), 0.0);
  vec3 c = v_color * (0.08 + 1.02 * sun)
         + v_color * vec3(0.45, 0.55, 0.75) * (0.22 * u_fillK) * fill;
  o = vec4(c, 1.0);
}`;

const EARTH_VS = `#version 300 es
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

// Day/night by the REAL terminator: dot(surface normal, real sun dir).
// Day side = simple blue-marble-ish gradient (simplified globe — NOT
// imagery, and the provenance string says so); night near-black; thin
// atmosphere rim at the limb, brighter toward the day side.
const EARTH_FS = `#version 300 es
precision mediump float;
in vec3 v_normal;
in vec3 v_world;
uniform vec3 u_sun;
uniform vec3 u_camPos;
out vec4 o;
void main() {
  vec3 n = normalize(v_normal);
  float lit = dot(n, u_sun);
  float day = smoothstep(-0.08, 0.12, lit);
  vec3 dayCol = mix(vec3(0.07, 0.20, 0.42), vec3(0.16, 0.36, 0.60), 0.5 + 0.5 * n.y);
  vec3 nightCol = vec3(0.004, 0.006, 0.012);
  vec3 c = mix(nightCol, dayCol, day);
  vec3 vDir = normalize(u_camPos - v_world);
  float rim = pow(1.0 - clamp(dot(n, vDir), 0.0, 1.0), 3.0);
  c += vec3(0.25, 0.45, 0.80) * rim * (0.10 + 0.55 * day);
  o = vec4(c, 1.0);
}`;

const DISC_VS = `#version 300 es
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

// mode 0 = SUN (bright disc, soft limb). mode 1 = MOON: the disc is shaded
// as a SPHERE lit by the real sun direction — the phase's illuminated
// fraction AND lit-side orientation fall out of the real geometry instead
// of being painted on.
const DISC_FS = `#version 300 es
precision mediump float;
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

const LINE_VS = `#version 300 es
in vec3 a_pos;
uniform mat4 u_mvp;
void main() { gl_Position = u_mvp * vec4(a_pos, 1.0); }`;

const LINE_FS = `#version 300 es
precision mediump float;
uniform vec4 u_color;
out vec4 o;
void main() { o = u_color; }`;

// ── mount ───────────────────────────────────────────────────────────────────

export interface InspectOptions {
  /** real or form mesh, injected by the parent (null = no craft claim). */
  mesh: Mesh | null;
  meshLabel: string;
  /** live per-frame source the parent wires to the follow tick. */
  getState: () => CraftState;
}

export interface InspectHandle {
  setMesh(mesh: Mesh | null, label: string): void;
  render(): void;
  dispose(): void;
  /** true after a GL failure — scene disabled, page continues (modelLayer latch). */
  getRenderFailed(): boolean;
  /** current camera pose (for tests / parent chrome). */
  getCamera(): { yaw: number; pitch: number; dist: number };
  getMeshLabel(): string;
}

interface GlProgram {
  prog: WebGLProgram;
  attrs: Record<string, number>;
  unis: Record<string, WebGLUniformLocation | null>;
}

export function mount(container: HTMLElement, opts: InspectOptions): InspectHandle {
  let disposed = false;
  let renderFailed = false;
  let mesh = opts.mesh;
  let meshLabel = opts.meshLabel;
  let extent = meshExtent(mesh);
  const cam = createOrbitCam(extent);

  // Velocity derivation: keep a trailing sample ≥400 ms old.
  let prevSample: CraftState | null = null;
  let prevSampleWallMs = 0;

  const doc: Document | null =
    (container && (container as { ownerDocument?: Document }).ownerDocument) ??
    (typeof document !== 'undefined' ? document : null);

  // Inert handle when there is no DOM at all (SSR/tests without a canvas).
  const inert = (): InspectHandle => ({
    setMesh(m, label) { mesh = m; meshLabel = label; },
    render() { /* disabled */ },
    dispose() { disposed = true; },
    getRenderFailed: () => renderFailed,
    getCamera: () => ({ yaw: cam.yaw, pitch: cam.pitch, dist: cam.dist }),
    getMeshLabel: () => meshLabel,
  });
  if (!doc) {
    renderFailed = true;
    return inert();
  }

  const canvas = doc.createElement('canvas');
  canvas.className = 'vt-inspect-scene';
  // Inline overlay styling so the scene is drop-in (solarView precedent);
  // the integration PR may move this into index.css.
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
  let lineProg: GlProgram | null = null;
  let bufMeshPos: WebGLBuffer | null = null;
  let bufMeshNor: WebGLBuffer | null = null;
  let bufMeshCol: WebGLBuffer | null = null;
  let meshDirty = true;
  let bufSphere: WebGLBuffer | null = null;
  let sphereVerts = 0;
  let bufQuad: WebGLBuffer | null = null;
  let bufLine: WebGLBuffer | null = null;
  let lastVelDir: [number, number, number] | null = null;

  function compile(vsSrc: string, fsSrc: string, attrs: string[], unis: string[]): GlProgram {
    const g = gl!;
    const mk = (type: number, src: string): WebGLShader => {
      const sh = g.createShader(type);
      if (!sh) throw new Error('inspectScene: createShader failed');
      g.shaderSource(sh, src);
      g.compileShader(sh);
      if (!g.getShaderParameter(sh, g.COMPILE_STATUS)) {
        const log = g.getShaderInfoLog(sh);
        g.deleteShader(sh);
        throw new Error('inspectScene: shader compile failed: ' + log);
      }
      return sh;
    };
    const vs = mk(g.VERTEX_SHADER, vsSrc);
    const fs = mk(g.FRAGMENT_SHADER, fsSrc);
    const p = g.createProgram();
    if (!p) throw new Error('inspectScene: createProgram failed');
    g.attachShader(p, vs);
    g.attachShader(p, fs);
    g.linkProgram(p);
    if (!g.getProgramParameter(p, g.LINK_STATUS)) {
      const log = g.getProgramInfoLog(p);
      g.deleteProgram(p);
      throw new Error('inspectScene: program link failed: ' + log);
    }
    g.deleteShader(vs);
    g.deleteShader(fs);
    const out: GlProgram = { prog: p, attrs: {}, unis: {} };
    for (const a of attrs) out.attrs[a] = g.getAttribLocation(p, a);
    for (const u of unis) out.unis[u] = g.getUniformLocation(p, u);
    return out;
  }

  function buildSphere(stacks: number, slices: number): Float32Array {
    // flat triangle list on the unit sphere (positions double as normals)
    const v: number[] = [];
    const pt = (i: number, j: number): [number, number, number] => {
      const phi = (i / stacks) * Math.PI - Math.PI / 2; // -90..90
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
    if (!gl) throw new Error('inspectScene: WebGL2 unavailable');
    meshProg = compile(MESH_VS, MESH_FS, ['a_pos', 'a_normal', 'a_color'], ['u_mvp', 'u_sun', 'u_fillK']);
    earthProg = compile(EARTH_VS, EARTH_FS, ['a_pos'], ['u_mvp', 'u_center', 'u_radius', 'u_sun', 'u_camPos']);
    discProg = compile(DISC_VS, DISC_FS, ['a_uv'],
      ['u_mvp', 'u_centerPos', 'u_right', 'u_up2', 'u_radius', 'u_mode', 'u_facing', 'u_sun']);
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
    bufLine = gl.createBuffer();
  } catch (e) {
    renderFailed = true;
    // eslint-disable-next-line no-console
    console.error('inspectScene: disabling after GL init failure (page continues):', e);
  }

  function resizeBacking(): void {
    const dpr = (globalThis.devicePixelRatio as number | undefined) || 1;
    const w = canvas.clientWidth || container.clientWidth || 800;
    const h = canvas.clientHeight || container.clientHeight || 600;
    const bw = Math.max(1, Math.round(w * dpr));
    const bh = Math.max(1, Math.round(h * dpr));
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
  }

  function drawFrame(): void {
    if (disposed || renderFailed || !gl) return;
    try {
      drawFrameInner();
    } catch (e) {
      renderFailed = true;
      // eslint-disable-next-line no-console
      console.error('inspectScene: disabling after render failure (page continues):', e);
    }
  }

  function drawFrameInner(): void {
    const g = gl!;
    resizeBacking();
    g.viewport(0, 0, canvas.width, canvas.height);
    g.clearColor(0, 0, 0, 1); // pure black sky — real positions or absent
    g.clearDepth(1);
    g.enable(g.DEPTH_TEST);
    g.depthFunc(g.LEQUAL);
    g.clear(g.COLOR_BUFFER_BIT | g.DEPTH_BUFFER_BIT);

    const st = opts.getState();
    const nowWall = Date.now();
    if (!prevSample) {
      prevSample = { ...st };
      prevSampleWallMs = nowWall;
    }

    // real directions in the scene frame
    const sunScene = enuToScene(sunDirLocal(st.timeMs, st.latDeg, st.lonDeg));
    const moon = moonDirLocal(st.timeMs, st.latDeg, st.lonDeg);
    const moonScene = enuToScene(moon.dir);
    const earthAng = earthAngularRadiusRad(st.altMeters);
    // earthshine strength: the Earth face below is sunlit only when the
    // sun has a zenith-ward component at the craft (scene y = up)
    const fillK = Math.max(0, Math.min(1, sunScene[1]));

    const aspect = canvas.width / Math.max(1, canvas.height);
    const eye = camEye(cam);
    const proj = mat4Perspective(50 * DEG, aspect, 0.05, 30000);
    const view = mat4LookAt(eye, [0, 0, 0]);
    const mvp = mat4Multiply(proj, view);

    // ── EARTH: nadir direction, true angular size from altitude ──
    {
      const p = earthProg!;
      g.useProgram(p.prog);
      g.uniformMatrix4fv(p.unis.u_mvp, false, mvp);
      g.uniform3f(p.unis.u_center, 0, -EARTH_SCENE_DIST, 0);
      g.uniform1f(p.unis.u_radius, EARTH_SCENE_DIST * Math.sin(earthAng));
      g.uniform3f(p.unis.u_sun, sunScene[0], sunScene[1], sunScene[2]);
      g.uniform3f(p.unis.u_camPos, eye[0], eye[1], eye[2]);
      g.bindBuffer(g.ARRAY_BUFFER, bufSphere);
      g.enableVertexAttribArray(p.attrs.a_pos);
      g.vertexAttribPointer(p.attrs.a_pos, 3, g.FLOAT, false, 0, 0);
      g.enable(g.CULL_FACE);
      g.cullFace(g.BACK);
      g.drawArrays(g.TRIANGLES, 0, sphereVerts);
      g.disable(g.CULL_FACE);
      g.disableVertexAttribArray(p.attrs.a_pos);
    }

    // ── CRAFT mesh at the origin, real sun + earthshine ──
    if (mesh && mesh.vertexCount > 0) {
      const p = meshProg!;
      g.useProgram(p.prog);
      g.uniformMatrix4fv(p.unis.u_mvp, false, mvp);
      g.uniform3f(p.unis.u_sun, sunScene[0], sunScene[1], sunScene[2]);
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

    // ── VELOCITY indicator: derived from getState deltas, never fabricated ──
    {
      if (nowWall - prevSampleWallMs > 400) {
        const dir = deriveVelocityDirScene(prevSample, st);
        prevSample = { ...st };
        prevSampleWallMs = nowWall;
        lastVelDir = dir ?? lastVelDir;
      }
      if (lastVelDir && bufLine) {
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
    }

    // ── MOON then SUN discs (sun farther: real syzygy occludes) ──
    const drawDisc = (
      dir: [number, number, number],
      dist: number,
      angularRadius: number,
      mode: 0 | 1,
    ): void => {
      const p = discProg!;
      // billboard basis perpendicular to dir (facing the origin/craft)
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

  // ── interaction: free orbit — no ground constraint, no snap-back ──
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
      // pinch dolly
      const [a, b] = Array.from(pointers.values());
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinchDist > 0 && d > 0) camApplyDolly(cam, pinchDist / d);
      pinchDist = d;
      return;
    }
    if (!cam.dragging) return;
    camApplyDrag(cam, e.clientX - prev.x, e.clientY - prev.y);
    kick();
  };
  const onPointerUp = (e: PointerEvent): void => {
    pointers.delete(e.pointerId);
    if (pointers.size === 0) {
      cam.dragging = false; // release: inertia decays via camStep
      canvas.style.cursor = 'grab';
    }
    pinchDist = 0;
  };
  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    camApplyDolly(cam, Math.pow(1.0015, e.deltaY));
    kick();
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

  // ── frame loop: continuous while mounted (the craft and the sky MOVE —
  // getState is live), bounded by dispose(). No RAF in this environment
  // (tests/SSR) → parent-driven handle.render() only. ──
  let rafId = 0;
  let lastFrameMs = 0;
  const hasRaf = typeof requestAnimationFrame !== 'undefined';
  const loop = (t: number): void => {
    if (disposed || renderFailed) return;
    const dt = lastFrameMs > 0 ? t - lastFrameMs : 16;
    lastFrameMs = t;
    camStep(cam, dt);
    drawFrame();
    rafId = requestAnimationFrame(loop);
  };
  const kick = (): void => { /* loop is continuous; kick kept for future idle mode */ };
  if (hasRaf && !renderFailed) rafId = requestAnimationFrame(loop);

  drawFrame();

  return {
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
    render(): void {
      camStep(cam, 16);
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
          for (const b of [bufMeshPos, bufMeshNor, bufMeshCol, bufSphere, bufQuad, bufLine]) {
            if (b) gl.deleteBuffer(b);
          }
          for (const p of [meshProg, earthProg, discProg, lineProg]) {
            if (p) gl.deleteProgram(p.prog);
          }
        } catch { /* context already lost */ }
      }
      try { canvas.remove(); } catch { /* detached */ }
    },
    getRenderFailed: () => renderFailed,
    getCamera: () => ({ yaw: cam.yaw, pitch: cam.pitch, dist: cam.dist }),
    getMeshLabel: () => meshLabel,
  };
}
