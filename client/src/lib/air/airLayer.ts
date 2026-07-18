// TRUE-ALTITUDE AIRCRAFT — EARTH TWIN E3 (research/earth_twin_program.md V2;
// the original directive's "see the planes at altitude").
//
// A MapLibre CustomLayerInterface (satLayer/modelLayer are the template:
// same projection prelude, variant caching, far-side cull, failure latch)
// that draws every live aircraft as a heading-oriented plane SILHOUETTE at
// its REAL barometric altitude via projectTileFor3D — with pitch + 3D
// terrain, planes visibly fly above the ground instead of being painted on
// it. WebGL2 instancing: a small static silhouette mesh PER CLASS (decoded
// from the broadcast ADS-B emitter category — symbols-not-dots, in 3D) + a
// 6-float instance record per aircraft (~240KB at the 10k cap — a trivial
// per-tick upload), drawn one instanced call per contiguous class group,
// plus one LINES pass drawing every aircraft's altitude drop-line to the
// surface (fades toward the ground end; degenerate for ground traffic).
//
// LOD SPLIT (charter E3): below AIR_3D_MIN_ZOOM the existing 2D class-icon
// symbol layer covers the map (altitude is sub-pixel at continent zooms and
// the icon system carries CLASS — symbols-not-dots preserved); at/above the
// split the icons hand off to these 3D silhouettes. One representation at a
// time, never both.
//
// ALTITUDE HONESTY: baro altitude rendered as-is (msl, registry altitudeRef);
// on-ground aircraft sit at 0. When 3D terrain is enabled the altitude is
// scaled by the SAME exaggeration factor the terrain mesh uses (setAltScale)
// so the vertical datum matches — a plane above a peak stays above the
// exaggerated peak ("never intersect mountains" by construction for honest
// baro; junk baro is a data property the card already exposes, never
// silently corrected). Color = the SAME altitude bands the 2D layer uses
// (ground grey / <3000m amber / cruise blue) — one decode, both renderers.

import type {
  CustomLayerInterface,
  CustomRenderMethodInput,
  Map as MapLibreMap,
} from 'maplibre-gl';
import { lonLatToMercator } from '../orbital/satBuffer.js';
import { mercatorToSphere } from '../orbital/occlusion.js';
import { MAX_AIR_GLIDE_SEC, airGlideDtSec, mercVelPerSec, shouldGlidePerFrame } from './airGlide.js';

/** The 2D↔3D hand-off zoom: symbols below, silhouettes at/above. */
export const AIR_3D_MIN_ZOOM = 8;

/** floats per instance: mercX, mercY, altMeters, headingDeg, band, shape,
 *  velX, velY. vel* are DISPLAY-ONLY dead-reckoning rates in normalized-
 *  mercator units per second (airGlide.mercVelPerSec — broadcast track +
 *  ground speed; 0 = frozen: on-ground / not broadcast). The shader glides
 *  base + vel·u_dtSec between polls, capped at MAX_AIR_GLIDE_SEC; fields
 *  [0..5] keep their pre-glide offsets so stride-agnostic readers survive
 *  the extension (the satBuffer velocity-extension precedent). */
export const AIR_INST_STRIDE = 8;

/** altitude band codes (match the 2D layer's ALT_COLOR semantics). */
export const AIR_BAND = { GROUND: 0, LOW: 1, CRUISE: 2 } as const;

/** silhouette shape codes — decoded from the broadcast ADS-B emitter
 *  category ONLY (a catalogued field; no category → the generic default,
 *  never a guessed class). */
export const AIR_SHAPE = { DEFAULT: 0, LIGHT: 1, HEAVY: 2, PERF: 3, ROTOR: 4 } as const;

/** ADS-B emitter category (DO-260 A1..A7, as broadcast) → silhouette.
 *  A1 light / A2-A4 small-to-large fixed-wing (the generic airliner) /
 *  A5 heavy / A6 high-performance / A7 rotorcraft. Anything else —
 *  missing, or the B/C categories (gliders, UAVs, surface vehicles) —
 *  stays DEFAULT. */
export function shapeForCategory(category: string | null | undefined): number {
  switch (String(category || '').toUpperCase()) {
    case 'A1': return AIR_SHAPE.LIGHT;
    case 'A5': return AIR_SHAPE.HEAVY;
    case 'A6': return AIR_SHAPE.PERF;
    case 'A7': return AIR_SHAPE.ROTOR;
    default: return AIR_SHAPE.DEFAULT;
  }
}

/**
 * Top-view airliner silhouette as a local-space triangle soup (x,y pairs;
 * nose +Y, wingspan X, extent within [-0.62, 0.62]). Deterministic and
 * generic by design — the CLASS-specific detail lives in the 2D icons and
 * the click card; this is the "silhouette" rung of the charter's
 * icon → silhouette → model ramp.
 */
export const PLANE_SILHOUETTE: Float32Array = new Float32Array([
  // fuselage (two triangles, slim rectangle)
  -0.06, -0.48, 0.06, -0.48, 0.06, 0.44,
  -0.06, -0.48, 0.06, 0.44, -0.06, 0.44,
  // nose (triangle to the tip)
  -0.06, 0.44, 0.06, 0.44, 0.0, 0.62,
  // left wing (swept)
  -0.06, 0.14, -0.06, -0.06, -0.62, -0.26,
  -0.06, 0.14, -0.62, -0.26, -0.62, -0.18,
  // right wing (swept)
  0.06, -0.06, 0.06, 0.14, 0.62, -0.26,
  0.06, 0.14, 0.62, -0.18, 0.62, -0.26,
  // left tailplane
  -0.05, -0.34, -0.05, -0.46, -0.28, -0.52,
  // right tailplane
  0.05, -0.46, 0.05, -0.34, 0.28, -0.52,
]);

// tiny deterministic builder for the class silhouettes (x,y pairs, nose +Y)
function silhouette(build: (quad: (x0: number, y0: number, x1: number, y1: number) => void,
                            tri: (ax: number, ay: number, bx: number, by: number, cx: number, cy: number) => void) => void): Float32Array {
  const v: number[] = [];
  const tri = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number) => { v.push(ax, ay, bx, by, cx, cy); };
  const quad = (x0: number, y0: number, x1: number, y1: number) => {
    tri(x0, y0, x1, y0, x1, y1);
    tri(x0, y0, x1, y1, x0, y1);
  };
  build(quad, tri);
  return new Float32Array(v);
}

/** Per-class top-view silhouettes, indexed by AIR_SHAPE code. Class comes
 *  from the broadcast emitter category (decode above) — the shape encodes
 *  KIND (SYMBOLS-NOT-DOTS, in 3D), color keeps encoding altitude band.
 *  All extents within ±0.62 (the DEFAULT airliner's envelope). */
export const AIR_SILHOUETTES: Float32Array[] = [
  PLANE_SILHOUETTE, // DEFAULT: the generic airliner
  // LIGHT (A1): small, straight (unswept) wing, prop-plane proportions
  silhouette((quad, tri) => {
    quad(-0.045, -0.38, 0.045, 0.30);   // fuselage
    tri(-0.045, 0.30, 0.045, 0.30, 0, 0.42); // nose
    quad(-0.55, 0.02, 0.55, 0.13);      // straight wing
    quad(-0.22, -0.36, 0.22, -0.28);    // tailplane
  }),
  // HEAVY (A5): wide-body — longer/wider fuselage, big swept wings
  silhouette((quad, tri) => {
    quad(-0.085, -0.55, 0.085, 0.48);   // fuselage
    tri(-0.085, 0.48, 0.085, 0.48, 0, 0.62);
    tri(-0.085, 0.16, -0.085, -0.10, -0.62, -0.34); // left wing
    tri(-0.085, 0.16, -0.62, -0.34, -0.62, -0.22);
    tri(0.085, -0.10, 0.085, 0.16, 0.62, -0.34);    // right wing
    tri(0.085, 0.16, 0.62, -0.22, 0.62, -0.34);
    tri(-0.06, -0.40, -0.06, -0.54, -0.34, -0.60);  // left tailplane
    tri(0.06, -0.54, 0.06, -0.40, 0.34, -0.60);     // right tailplane
  }),
  // PERF (A6): high-performance — slim body, delta wing set far back
  silhouette((quad, tri) => {
    quad(-0.05, -0.45, 0.05, 0.35);     // fuselage
    tri(-0.05, 0.35, 0.05, 0.35, 0, 0.58);
    tri(-0.05, 0.10, -0.05, -0.38, -0.45, -0.38);   // left delta
    tri(0.05, -0.38, 0.05, 0.10, 0.45, -0.38);      // right delta
    tri(-0.03, -0.40, -0.03, -0.45, -0.16, -0.47);  // left fin
    tri(0.03, -0.45, 0.03, -0.40, 0.16, -0.47);     // right fin
  }),
  // ROTOR (A7): rotorcraft — cabin, tail boom, rotor cross
  silhouette((quad, tri) => {
    quad(-0.09, -0.05, 0.09, 0.25);     // cabin
    tri(-0.09, 0.25, 0.09, 0.25, 0, 0.33);
    quad(-0.03, -0.45, 0.03, -0.05);    // tail boom
    quad(-0.10, -0.47, 0.10, -0.43);    // tail rotor
    quad(-0.035, -0.33, 0.035, 0.53);   // main rotor blade N-S (disc centered 0.10)
    quad(-0.43, 0.065, 0.43, 0.135);    // main rotor blade E-W
  }),
];

export interface AircraftInstanceInput {
  lon?: number | null;
  lat?: number | null;
  altitude_m?: number | null;
  heading?: number | null;
  on_ground?: boolean | null;
  category?: string | null;
  /** broadcast ground speed (m/s) — the glide rate input; null = frozen. */
  velocity_ms?: number | null;
}

export interface AirShapeGroup { shape: number; start: number; count: number }

/**
 * Pure: aircraft payload rows → packed instance buffer + the index-aligned
 * rows (for CPU picking → the click card) + contiguous per-shape draw
 * groups. Rows are STABLE-sorted by shape so each class renders as one
 * instanced draw over a contiguous range. Rows without a finite position
 * are skipped — nothing rendered from a guessed position.
 */
export function buildAircraftInstances<T extends AircraftInstanceInput>(
  aircraft: T[],
): { inst: Float32Array; rows: T[]; groups: AirShapeGroup[] } {
  const rows: T[] = [];
  for (const a of aircraft) {
    if (a == null || !Number.isFinite(a.lon as number) || !Number.isFinite(a.lat as number)) continue;
    if ((a.lat as number) < -85 || (a.lat as number) > 85) continue; // outside mercator
    rows.push(a);
  }
  rows.sort((a, b) => shapeForCategory(a.category) - shapeForCategory(b.category)); // stable
  const inst = new Float32Array(rows.length * AIR_INST_STRIDE);
  const groups: AirShapeGroup[] = [];
  for (let i = 0; i < rows.length; i++) {
    const a = rows[i];
    const m = lonLatToMercator(a.lon as number, a.lat as number);
    const ground = !!a.on_ground;
    const alt = ground ? 0 : Math.max(0, a.altitude_m ?? 0);
    const shape = shapeForCategory(a.category);
    const vel = mercVelPerSec(a.lon, a.lat, a.heading, a.velocity_ms, a.on_ground);
    inst[i * AIR_INST_STRIDE] = m.x;
    inst[i * AIR_INST_STRIDE + 1] = m.y;
    inst[i * AIR_INST_STRIDE + 2] = alt;
    inst[i * AIR_INST_STRIDE + 3] = a.heading ?? 0;
    inst[i * AIR_INST_STRIDE + 4] = ground ? AIR_BAND.GROUND : alt < 3000 ? AIR_BAND.LOW : AIR_BAND.CRUISE;
    inst[i * AIR_INST_STRIDE + 5] = shape;
    inst[i * AIR_INST_STRIDE + 6] = vel.vx;
    inst[i * AIR_INST_STRIDE + 7] = vel.vy;
    const g = groups[groups.length - 1];
    if (g && g.shape === shape) g.count += 1;
    else groups.push({ shape, start: i, count: 1 });
  }
  return { inst, rows, groups };
}

/** Glided mercator X of instance i at dtSec (wraps the antimeridian like
 *  the shader's fract()). */
function glidedX(inst: Float32Array, base: number, dtSec: number): number {
  const x = inst[base] + inst[base + 6] * dtSec;
  return ((x % 1) + 1) % 1;
}

/** Glided mercator Y of instance i at dtSec (pole-clamped like the shader). */
function glidedY(inst: Float32Array, base: number, dtSec: number): number {
  const y = inst[base + 1] + inst[base + 7] * dtSec;
  return y < 0 ? 0 : y > 1 ? 1 : y;
}

/** Pure: nearest instance to a mercator point within tolerance, else -1.
 *  `dtSec` applies the SAME display glide the shader drew with (pass the
 *  layer's getGlideDtSec()) so the pick agrees with the pixels — a plane
 *  25 s into a glide sits many pixels from its tick position at z8+. The
 *  resolved index maps back to the REAL payload row; the glide never enters
 *  what the card shows. */
export function pickNearestAircraft(
  inst: Float32Array | null,
  mercX: number,
  mercY: number,
  tolMercUnits: number,
  dtSec = 0,
): number {
  if (!inst || inst.length < AIR_INST_STRIDE) return -1;
  const n = Math.floor(inst.length / AIR_INST_STRIDE);
  let best = -1;
  let bestD2 = tolMercUnits * tolMercUnits;
  for (let i = 0; i < n; i++) {
    const base = i * AIR_INST_STRIDE;
    const dx = glidedX(inst, base, dtSec) - mercX;
    const dy = glidedY(inst, base, dtSec) - mercY;
    const d2 = dx * dx + dy * dy;
    if (d2 <= bestD2) { bestD2 = d2; best = i; }
  }
  return best;
}

/**
 * SCREEN-SPACE picking for full globe mode (live bug: at 45° tilt a plane
 * renders displaced from its ground point by altitude — ground-mercator
 * picking missed it entirely). Projects each instance with the SAME frame
 * matrix the shader used (AirLayer.getGlobeProjection) — the pick agrees
 * with the pixels. Mirrors pick.ts's satellite fix.
 */
export function pickNearestAircraftScreen(
  inst: Float32Array | null,
  matrix: ArrayLike<number>,
  clickX: number,
  clickY: number,
  width: number,
  height: number,
  tolerancePx: number,
  dtSec = 0,
): number {
  if (!inst || inst.length < AIR_INST_STRIDE) return -1;
  const n = Math.floor(inst.length / AIR_INST_STRIDE);
  let best = -1;
  let bestD2 = tolerancePx * tolerancePx;
  for (let i = 0; i < n; i++) {
    const base = i * AIR_INST_STRIDE;
    const p = mercatorToSphere(glidedX(inst, base, dtSec), glidedY(inst, base, dtSec), inst[base + 2]);
    const w = matrix[3] * p[0] + matrix[7] * p[1] + matrix[11] * p[2] + matrix[15];
    if (!(w > 0)) continue;
    const cx = (matrix[0] * p[0] + matrix[4] * p[1] + matrix[8] * p[2] + matrix[12]) / w;
    const cy = (matrix[1] * p[0] + matrix[5] * p[1] + matrix[9] * p[2] + matrix[13]) / w;
    const dx = ((cx + 1) / 2) * width - clickX;
    const dy = ((1 - cy) / 2) * height - clickY;
    const d2 = dx * dx + dy * dy;
    if (d2 <= bestD2) { bestD2 = d2; best = i; }
  }
  return best;
}

/** Silhouette length on screen in px (the focused-traffic size at z8+). */
export const AIR_PIXELS = 30;
const LOCAL_EXTENT = 1.24; // silhouette spans ~[-0.62, 0.62]

/** Exported for airLayer.test.ts — pins the projection/cull/rotation contract. */
export const AIR_VERT_SRC = (prelude: string, define: string): string => `#version 300 es
${prelude}
${define}
in vec2 a_local;           // silhouette vertex (per-vertex)
in float a_altFrac;        // per-vertex altitude fraction: 1 for silhouettes;
                           // 0→1 for the ground drop-line (constant-attrib trick)
in vec4 a_inst;            // per-instance: mercX, mercY, altMeters, headingDeg
in float a_band;           // per-instance: altitude band (0 ground / 1 low / 2 cruise)
in vec2 a_vel;             // per-instance GLIDE: d/dt of (mercX, mercY) per second
uniform float u_bearing;   // map bearing (deg) — silhouettes rotate with the map like icons
uniform float u_altScale;  // 1, or the terrain exaggeration so vertical datums match
uniform float u_scaleClip; // clip units per local unit at w=1 (pixel-constant via *w)
uniform float u_dtSec;     // seconds since the poll anchor, CPU-clamped to MAX_AIR_GLIDE_SEC
uniform float u_aspect;
uniform vec4 u_bandColors[3];
out vec4 v_color;
void main() {
  // PER-FRAME GLIDE (the satLayer pattern): poll position + broadcast-track
  // dead-reckoning rate × capped elapsed time. fract() wraps X across the
  // antimeridian; Y clamps at the poles like lonLatToMercator's own pack.
  // a_vel is 0 for frozen rows (on-ground / velocity not broadcast).
  vec2 g_xy = vec2(fract(a_inst.x + a_vel.x * u_dtSec),
                   clamp(a_inst.y + a_vel.y * u_dtSec, 0.0, 1.0));
#ifdef GLOBE
  // anchor far-side cull — identical to satLayer/modelLayer (inert at the
  // z>=8 zooms this layer draws at, kept for contract parity; 0.998001 =
  // OCCLUSION_RADIUS²). Uses the GLIDED position — cull where DRAWN.
  if (u_projection_transition > 0.999 && u_projection_clipping_plane.w < 0.0) {
    vec3 satPos = projectToSphere(g_xy) * (1.0 + a_inst.z / GLOBE_RADIUS);
    vec3 cam = u_projection_clipping_plane.xyz * (-1.0 / u_projection_clipping_plane.w);
    vec3 v = satPos - cam;
    float t = -dot(cam, v) / dot(v, v);
    if (t > 0.0 && t < 1.0) {
      vec3 closest = cam + t * v;
      if (dot(closest, closest) < 0.998001) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_color = vec4(0.0);
        return;
      }
    }
  }
#endif
  vec4 anchor = projectTileFor3D(g_xy, a_inst.z * u_altScale * a_altFrac);
  float rot = radians(a_inst.w - u_bearing);
  float c = cos(rot);
  float s = sin(rot);
  vec2 p = vec2(a_local.x * c + a_local.y * s, -a_local.x * s + a_local.y * c);
  gl_Position = anchor + vec4(p.x * u_scaleClip / u_aspect, p.y * u_scaleClip, 0.0, 0.0) * anchor.w;
  int band = int(clamp(a_band, 0.0, 2.0));
  v_color = u_bandColors[band];
  // the drop-line fades toward the ground end (altFrac 0); silhouettes
  // (altFrac 1 everywhere) keep full band alpha
  v_color.a *= mix(0.25, 1.0, a_altFrac);
}`;

const AIR_FRAG_SRC = `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 o;
void main() { o = v_color; }`;

type AnyGl = WebGLRenderingContext | WebGL2RenderingContext;

// altitude bands — same meanings as the 2D layer's ALT_COLOR.
// Exported (2026-07-16) so the aircraft altitude-curtain default ramp in
// orbital/arcLayer.ts can be TEST-PINNED against these exact stops — map
// silhouettes and curtain wall must always agree on band colors.
export const BAND_COLORS: [number, number, number, number][] = [
  [0.4, 0.5, 0.63, 0.95],   // ground grey-blue (#6680a0)
  [0.98, 0.7, 0.3, 0.95],   // low amber (#fbb24c)
  [0.3, 0.62, 1.0, 0.95],   // cruise blue (#4d9fff)
];

export class AirLayer implements CustomLayerInterface {
  readonly id: string;
  readonly type = 'custom' as const;
  readonly renderingMode = '2d' as const;

  private map: MapLibreMap | null = null;
  private program: WebGLProgram | null = null;
  private cachedVariant: string | null = null;
  private geomBuffers: (WebGLBuffer | null)[] = [];
  private lineBuffer: WebGLBuffer | null = null;
  private instBuffer: WebGLBuffer | null = null;
  private aLocal = -1;
  private aAltFrac = -1;
  private aInst = -1;
  private aBand = -1;
  private aVel = -1;
  private uBearing: WebGLUniformLocation | null = null;
  private uAltScale: WebGLUniformLocation | null = null;
  private uScale: WebGLUniformLocation | null = null;
  private uDtSec: WebGLUniformLocation | null = null;
  private uAspect: WebGLUniformLocation | null = null;
  private uBandColors: WebGLUniformLocation | null = null;
  private uProjMatrix: WebGLUniformLocation | null = null;
  private uProjTile: WebGLUniformLocation | null = null;
  private uProjClip: WebGLUniformLocation | null = null;
  private uProjTrans: WebGLUniformLocation | null = null;
  private uProjFallback: WebGLUniformLocation | null = null;

  private inst: Float32Array | null = null;
  private groups: AirShapeGroup[] = [];
  private instDirty = false;
  private geomUploaded = false;
  private count = 0;
  private altScale = 1;
  private renderFailed = false;
  private lastMainMatrix: Float32Array | null = null;
  private lastTransition = 0;
  // GLIDE anchor: clock reading (this.now domain, ms) at which the current
  // instance buffer's positions were received — set by the parent via
  // setTickTime() beside setInstances. null = glide not wired: u_dtSec
  // stays 0 and rendering is bit-identical to the pre-glide layer.
  private tickAnchorMs: number | null = null;
  private now: () => number;

  constructor(opts: { id?: string; now?: () => number } = {}) {
    this.id = opts.id ?? 'aircraft-3d';
    this.now = opts.now ?? (() => performance.now());
  }

  onAdd(map: MapLibreMap, _gl: AnyGl): void {
    this.map = map;
  }

  onRemove(_map: MapLibreMap, gl: AnyGl): void {
    if (this.program) gl.deleteProgram(this.program);
    for (const b of [...this.geomBuffers, this.lineBuffer, this.instBuffer]) if (b) gl.deleteBuffer(b);
    this.program = null;
    this.geomBuffers = [];
    this.lineBuffer = this.instBuffer = null;
    this.cachedVariant = null;
    this.geomUploaded = false;
    this.instDirty = this.inst != null;
    this.map = null;
  }

  /** Fresh tick's instance buffer + contiguous per-shape draw groups
   *  (null clears). Groups default to one DEFAULT-shape run for callers
   *  that don't classify. */
  setInstances(inst: Float32Array | null, groups?: AirShapeGroup[]): void {
    this.inst = inst;
    this.count = inst ? Math.floor(inst.length / AIR_INST_STRIDE) : 0;
    this.groups = inst
      ? (groups && groups.length ? groups : [{ shape: AIR_SHAPE.DEFAULT, start: 0, count: this.count }])
      : [];
    this.instDirty = inst != null;
    this.map?.triggerRepaint();
  }

  /** terrain exaggeration match (1 without terrain). */
  setAltScale(s: number): void {
    if (s === this.altScale) return;
    this.altScale = s;
    this.map?.triggerRepaint();
  }

  /** GLIDE anchor: WHEN (this layer's clock, ms) the current instance
   *  positions were true — the parent calls it beside setInstances on every
   *  fresh payload (satLayer.setTickTime precedent). Omit the argument to
   *  anchor "now". Display-only: picking maps back to real payload rows. */
  setTickTime(tMs?: number): void {
    this.tickAnchorMs = tMs ?? this.now();
    this.map?.triggerRepaint();
  }

  /** Current glide anchor (ms in the layer clock) — wiring checks/tests. */
  getTickTime(): number | null {
    return this.tickAnchorMs;
  }

  /** Seconds of display glide in effect right now (0 when unwired) — the
   *  CPU pick paths pass this into pickNearestAircraft* so picks agree with
   *  the glided pixels. */
  getGlideDtSec(): number {
    return this.tickAnchorMs != null ? airGlideDtSec(this.now(), this.tickAnchorMs) : 0;
  }

  /** Low-rate repaint tick (parent interval, AIR_GLIDE_STEP_MS): repaint so
   *  the shader re-evaluates u_dtSec while a glide is in progress. No-ops
   *  below the hand-off zoom (layer not drawn), with no planes, past the
   *  honesty cap (frozen — an identical frame), or unwired. At close zooms
   *  renderInner's per-frame self-repaint takes over (shouldGlidePerFrame);
   *  the extra tick there coalesces harmlessly. */
  glideRepaintTick(): void {
    if (!this.map || this.tickAnchorMs == null || this.count === 0) return;
    if (this.map.getZoom() < AIR_3D_MIN_ZOOM) return;
    if (airGlideDtSec(this.now(), this.tickAnchorMs) >= MAX_AIR_GLIDE_SEC) return;
    this.map.triggerRepaint();
  }

  getCounts(): { total: number; drawn: boolean } {
    const zoomOk = (this.map?.getZoom() ?? 0) >= AIR_3D_MIN_ZOOM;
    return { total: this.count, drawn: zoomOk && this.count > 0 };
  }

  getInstances(): Float32Array | null {
    return this.inst;
  }

  getRenderFailed(): boolean {
    return this.renderFailed;
  }

  /** Last frame's projection matrix — non-null only in full globe mode
   *  (the CPU sphere math mirrors the GPU there). Null = mercator pick. */
  getGlobeProjection(): Float32Array | null {
    if (!this.lastMainMatrix || this.lastTransition <= 0.999) return null;
    return this.lastMainMatrix;
  }

  render(gl: AnyGl, args: CustomRenderMethodInput): void {
    if (this.renderFailed) return;
    if (!this.inst || this.count === 0) return;
    // LOD hand-off: the 2D symbol layer owns everything below the split
    if ((this.map?.getZoom() ?? 0) < AIR_3D_MIN_ZOOM) return;
    if (!(gl instanceof WebGL2RenderingContext)) { this.renderFailed = true; return; } // instancing needs GL2 (maplibre v5 guarantees it)
    try {
      this.renderInner(gl, args);
    } catch (e) {
      this.renderFailed = true;
      // eslint-disable-next-line no-console
      console.error('AirLayer: disabling after render failure (map continues):', e);
    }
  }

  private renderInner(gl: WebGL2RenderingContext, args: CustomRenderMethodInput): void {
    const sd = args.shaderData;
    if (this.program == null || this.cachedVariant !== sd.variantName) {
      this.compile(gl, sd.vertexShaderPrelude, sd.define, sd.variantName);
      this.geomUploaded = false;
      this.instDirty = true;
    }
    if (this.program == null) return;
    gl.useProgram(this.program);

    const pd = args.defaultProjectionData;
    // pick fix: cache this frame's projection so CPU picking projects
    // instances EXACTLY like the shader (altitude included)
    if (!this.lastMainMatrix) this.lastMainMatrix = new Float32Array(16);
    this.lastMainMatrix.set(pd.mainMatrix as ArrayLike<number>);
    this.lastTransition = pd.projectionTransition;
    if (this.uProjMatrix) gl.uniformMatrix4fv(this.uProjMatrix, false, pd.mainMatrix);
    if (this.uProjTile) {
      gl.uniform4f(this.uProjTile,
        pd.tileMercatorCoords[0], pd.tileMercatorCoords[1],
        pd.tileMercatorCoords[2], pd.tileMercatorCoords[3]);
    }
    if (this.uProjClip) {
      gl.uniform4f(this.uProjClip,
        pd.clippingPlane[0], pd.clippingPlane[1], pd.clippingPlane[2], pd.clippingPlane[3]);
    }
    if (this.uProjTrans) gl.uniform1f(this.uProjTrans, pd.projectionTransition);
    if (this.uProjFallback) gl.uniformMatrix4fv(this.uProjFallback, false, pd.fallbackMatrix);

    if (this.uBearing) gl.uniform1f(this.uBearing, this.map?.getBearing() ?? 0);
    if (this.uAltScale) gl.uniform1f(this.uAltScale, this.altScale);
    // GLIDE: capped seconds since the poll anchor (0 when the parent hasn't
    // wired setTickTime — renders raw poll positions, pre-glide behavior).
    const dtSec = this.tickAnchorMs != null ? airGlideDtSec(this.now(), this.tickAnchorMs) : 0;
    if (this.uDtSec) gl.uniform1f(this.uDtSec, dtSec);
    const h = gl.drawingBufferHeight || 1;
    const w = gl.drawingBufferWidth || 1;
    const pxPerUnit = AIR_PIXELS / LOCAL_EXTENT;
    if (this.uScale) gl.uniform1f(this.uScale, (pxPerUnit * 2) / h);
    if (this.uAspect) gl.uniform1f(this.uAspect, w / h);
    if (this.uBandColors) gl.uniform4fv(this.uBandColors, BAND_COLORS.flat());

    if (!this.geomUploaded) {
      for (let s = 0; s < AIR_SILHOUETTES.length; s++) {
        if (!this.geomBuffers[s]) this.geomBuffers[s] = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.geomBuffers[s]);
        gl.bufferData(gl.ARRAY_BUFFER, AIR_SILHOUETTES[s], gl.STATIC_DRAW);
      }
      if (!this.lineBuffer) this.lineBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 1]), gl.STATIC_DRAW); // altFrac endpoints
      this.geomUploaded = true;
    }

    if (!this.instBuffer) this.instBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuffer);
    if (this.instDirty && this.inst) {
      gl.bufferData(gl.ARRAY_BUFFER, this.inst, gl.DYNAMIC_DRAW);
      this.instDirty = false;
    }
    gl.enableVertexAttribArray(this.aInst);
    gl.vertexAttribDivisor(this.aInst, 1);
    gl.enableVertexAttribArray(this.aBand);
    gl.vertexAttribDivisor(this.aBand, 1);
    if (this.aVel >= 0) {
      gl.enableVertexAttribArray(this.aVel);
      gl.vertexAttribDivisor(this.aVel, 1);
    }

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // ── pass 1: ALTITUDE DROP-LINES for every instance (one draw). The
    // line's two vertices are altFrac 0→1 from lineBuffer; a_local is held
    // constant at (0,0) so the line runs surface→aircraft with no offset.
    // Ground aircraft (alt 0) produce a degenerate, invisible line.
    gl.disableVertexAttribArray(this.aLocal);
    gl.vertexAttrib2f(this.aLocal, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.enableVertexAttribArray(this.aAltFrac);
    gl.vertexAttribPointer(this.aAltFrac, 1, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(this.aAltFrac, 0);
    this.bindInstPointers(gl, 0);
    gl.drawArraysInstanced(gl.LINES, 0, 2, this.count);

    // ── pass 2: per-class silhouettes, one instanced draw per contiguous
    // shape group (a_altFrac held constant at 1 = full altitude, full alpha).
    gl.disableVertexAttribArray(this.aAltFrac);
    gl.vertexAttrib1f(this.aAltFrac, 1);
    for (const g of this.groups) {
      const geom = this.geomBuffers[g.shape] ?? this.geomBuffers[AIR_SHAPE.DEFAULT];
      const verts = (AIR_SILHOUETTES[g.shape] ?? PLANE_SILHOUETTE).length / 2;
      gl.bindBuffer(gl.ARRAY_BUFFER, geom);
      gl.enableVertexAttribArray(this.aLocal);
      gl.vertexAttribPointer(this.aLocal, 2, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(this.aLocal, 0);
      this.bindInstPointers(gl, g.start);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, verts, g.count);
    }

    // reset divisors so we never corrupt MapLibre's own attribute state
    gl.vertexAttribDivisor(this.aInst, 0);
    gl.vertexAttribDivisor(this.aBand, 0);
    gl.disableVertexAttribArray(this.aLocal);
    gl.disableVertexAttribArray(this.aAltFrac);
    gl.disableVertexAttribArray(this.aInst);
    gl.disableVertexAttribArray(this.aBand);
    if (this.aVel >= 0) {
      gl.vertexAttribDivisor(this.aVel, 0);
      gl.disableVertexAttribArray(this.aVel);
    }

    // GLIDE self-repaint (satLayer's needsGlide pattern): while a glide is
    // in progress AND one AIR_GLIDE_STEP_MS step would move a worst-case
    // plane a visible (≥1px) jump at this camera, request the next frame —
    // 60fps motion exactly where the cheap 4Hz tick would look stepped.
    // Stops at the honesty cap (frozen frame) and at far zooms (the parent
    // step tick covers sub-pixel-step drift there).
    if (
      this.map &&
      this.tickAnchorMs != null &&
      dtSec < MAX_AIR_GLIDE_SEC &&
      shouldGlidePerFrame(this.map.getCenter().lat, this.map.getZoom())
    ) {
      this.map.triggerRepaint();
    }
  }

  /** (Re)point the per-instance attributes at a group's contiguous range. */
  private bindInstPointers(gl: WebGL2RenderingContext, startInstance: number): void {
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuffer);
    const base = startInstance * AIR_INST_STRIDE * 4;
    gl.vertexAttribPointer(this.aInst, 4, gl.FLOAT, false, AIR_INST_STRIDE * 4, base);
    gl.vertexAttribPointer(this.aBand, 1, gl.FLOAT, false, AIR_INST_STRIDE * 4, base + 16);
    if (this.aVel >= 0) gl.vertexAttribPointer(this.aVel, 2, gl.FLOAT, false, AIR_INST_STRIDE * 4, base + 24);
  }

  private compile(gl: WebGL2RenderingContext, prelude: string, define: string, variant: string): void {
    if (this.program) {
      gl.deleteProgram(this.program);
      this.program = null;
    }
    const mk = (type: number, src: string): WebGLShader => {
      const sh = gl.createShader(type);
      if (!sh) throw new Error('AirLayer: createShader failed');
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error('AirLayer: shader compile failed: ' + log);
      }
      return sh;
    };
    const vs = mk(gl.VERTEX_SHADER, AIR_VERT_SRC(prelude, define));
    const fs = mk(gl.FRAGMENT_SHADER, AIR_FRAG_SRC);
    const p = gl.createProgram();
    if (!p) throw new Error('AirLayer: createProgram failed');
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error('AirLayer: program link failed: ' + log);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    this.program = p;
    this.cachedVariant = variant;
    this.aLocal = gl.getAttribLocation(p, 'a_local');
    this.aAltFrac = gl.getAttribLocation(p, 'a_altFrac');
    this.aInst = gl.getAttribLocation(p, 'a_inst');
    this.aBand = gl.getAttribLocation(p, 'a_band');
    this.aVel = gl.getAttribLocation(p, 'a_vel');
    this.uBearing = gl.getUniformLocation(p, 'u_bearing');
    this.uAltScale = gl.getUniformLocation(p, 'u_altScale');
    this.uScale = gl.getUniformLocation(p, 'u_scaleClip');
    this.uDtSec = gl.getUniformLocation(p, 'u_dtSec');
    this.uAspect = gl.getUniformLocation(p, 'u_aspect');
    this.uBandColors = gl.getUniformLocation(p, 'u_bandColors');
    this.uProjMatrix = gl.getUniformLocation(p, 'u_projection_matrix');
    this.uProjTile = gl.getUniformLocation(p, 'u_projection_tile_mercator_coords');
    this.uProjClip = gl.getUniformLocation(p, 'u_projection_clipping_plane');
    this.uProjTrans = gl.getUniformLocation(p, 'u_projection_transition');
    this.uProjFallback = gl.getUniformLocation(p, 'u_projection_fallback_matrix');
  }
}
