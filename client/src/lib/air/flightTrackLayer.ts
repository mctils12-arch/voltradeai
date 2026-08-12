// FLIGHT TRACK 3D LAYER — the handoff-approved flight-track visualization
// (design_handoff_flight_track_3d, installed 2026-07-20). REPLACES the
// previous ArcLayer arcs+walls rendering of the selected aircraft's track.
//
// One MapLibre CustomLayerInterface (satLayer/airLayer/arcLayer are the
// template: same projection prelude, variant caching, failure latch) that
// draws, in order:
//   1. GROUND TRACE — the track draped on the topography (terrain +16 m),
//      a screen-space extruded ribbon ~3 px, #1e5fd6. Drawn through
//      altitude gaps (position history is real even when altitude wasn't
//      broadcast).
//   2. THE CURTAIN — one vertical world-space strip: top edge = the flight
//      path at its REAL recorded altitude, bottom edge = the terrain under
//      the track MINUS 40 m (drape rule: overlap into the ground so ridges
//      never open gaps — the old builder clamped bottoms TO the ground and
//      relied on sea-level fallbacks, which is exactly what broke over
//      relief). Vertex colors: the altitude ramp at 34% opacity on top,
//      the same color × [0.25, 0.28, 0.45] at the bottom.
//   3. ALTITUDE LINE — the track at altitude, ~4 px ribbon, vertex-colored
//      by the ramp (teal→blue→violet, min→max across the visible track).
//   4. MARKER — the selected aircraft (or its playback/scrub replay
//      position): class silhouette glyph #8fd0ff nose-along-track, a 1 px
//      AGL drop line (#bfe0ff @ 65%) and a soft ground dot.
//
// THE CRITICAL FIX (why the old curtain vanished at grazing angles) — the
// three rules from the handoff, expressed in MapLibre terms:
//   a. DRAPE THE BOTTOM EDGE 40 m *below* the re-sampled terrain (never a
//      constant sea-level base, never clamped to the top).
//   b. DOUBLE-SIDED: gl.disable(CULL_FACE) — a single-sided strip is
//      back-face-culled the moment the camera tilts past its plane.
//   c. TRANSPARENT GEOMETRY MUST NOT WRITE DEPTH, BUT MUST TEST IT:
//      MapLibre gives '2d' custom layers NO depth interaction at all
//      (painter: DepthMode.disabled above the opaque cutoff), while the
//      terrain mesh DOES write real depth into the main framebuffer — so
//      this layer explicitly sets gl.enable(DEPTH_TEST) + depthFunc(LEQUAL)
//      + depthMask(false) + depthRange(0,1) each frame. The curtain depth-
//      tests against the terrain (ridges in front occlude it correctly,
//      the below-ground overlap hides inside the mountain) and never
//      z-fights it (no writes). The painter's setDirty() after custom
//      render re-syncs MapLibre's own state tracking.
//
// Vertex layout, gap/antimeridian rules, GLOBE far-side fragment-discard
// and the constant-pixel-width ribbon extrusion are the proven arcLayer
// patterns (13-float layout), self-contained here so the orbital arc layer
// and the flight track can evolve independently.

import type {
  CustomLayerInterface,
  CustomRenderMethodInput,
  Map as MapLibreMap,
} from 'maplibre-gl';
import {
  ALT_LINE_WIDTH_PX,
  CURTAIN_ALPHA,
  CURTAIN_BELOW_TERRAIN_M,
  CURTAIN_BOTTOM_MUL,
  TRACE_ABOVE_TERRAIN_M,
  TRACE_RGBA,
  TRACE_WIDTH_PX,
  TZ_MARK_RGBA,
  TZ_MARK_WIDTH_PX,
  altRampColor,
} from './trackModel.js';
import { AIR_SILHOUETTES, AIR_SHAPE } from './airLayer.js';
import { mercatorToSphere, mercatorZFromAltitude } from '../orbital/occlusion.js';
import { VT_PROJ_ELEV_GLSL } from '../glElev.js';

type AnyGl = WebGLRenderingContext | WebGL2RenderingContext;

/** floats per geometry vertex (the arcLayer-proven layout):
 *  [0..2] a_pos (mercX, mercY, z meters in the DISPLAY datum)
 *  [3..5] a_other (segment's other endpoint; = self for wall vertices)
 *  [6..8] a_ext (side −1|0|+1, dirSign, widthPx)
 *  [9..12] rgba */
export const FT_VERT_STRIDE = 13;
export const FT_VERTS_PER_SEG = 4;
export const FT_INDICES_PER_SEG = 6;

/** marker drop-line color #bfe0ff @ 65% (handoff §4). */
export const MARKER_LINE_RGBA: [number, number, number, number] =
  [0xbf / 255, 0xe0 / 255, 0xff / 255, 0.65];

/** marker glyph color #8fd0ff (handoff §4). */
export const MARKER_GLYPH_RGBA: [number, number, number, number] =
  [0x8f / 255, 0xd0 / 255, 0xff / 255, 1];

/** soft ground dot under the marker, #bfe0ff @ 80% (handoff §4). */
export const MARKER_DOT_RGBA: [number, number, number, number] =
  [0xbf / 255, 0xe0 / 255, 0xff / 255, 0.8];

/** glyph size on screen, px (airLayer's AIR_PIXELS sizing convention). */
export const MARKER_PIXELS = 30;
/** ground dot radius on screen, px. */
export const MARKER_DOT_PX = 5;

/** Everything the geometry builder needs — parallel arrays, one entry per
 *  densified track sample (trackModel.buildTrackSamples + the caller's
 *  terrain queries; the layer itself never touches the map API). */
export interface TrackGeomInput {
  /** normalized-mercator [x,y] pairs, length 2n. */
  merc: Float32Array;
  /** REAL altitude meters MSL per sample; NaN = altitude gap. length n. */
  altM: Float32Array;
  /** ground elevation per sample in the DISPLAY datum (queryTerrainElevation
   *  output — ALREADY exaggeration-scaled; 0 = terrain off / sea level). */
  groundZ: Float32Array;
  /** ramp domain over the visible track (REAL meters). */
  altMin: number;
  altMax: number;
  /** REAL meters the curtain bottom sits below groundZ — the drape overlap.
   *  Callers pass CURTAIN_BELOW_TERRAIN_M with terrain ON (scaled by
   *  altScale internally so it stays 40 real meters under the exaggerated
   *  mesh) and 0 with terrain OFF (a flat sea-level base has no ridges to
   *  seal against). Default: CURTAIN_BELOW_TERRAIN_M. */
  drapeBelowM?: number;
  /** sample indices where the local UTC offset changed (tzCrossings.ts) —
   *  each gets a vertical amber mark line through the curtain, ground to
   *  altitude ("display it on … the curtain as a different color line",
   *  human 2026-08-11). Indices without altitude are skipped honestly. */
  tzMarkIdx?: number[];
}

/** Pure: index buffer for `segCount` independent quads (two triangles each;
 *  no index crosses a quad boundary, so no triangle can span a gap). */
export function buildQuadIndices(segCount: number): Uint32Array {
  const idx = new Uint32Array(segCount * FT_INDICES_PER_SEG);
  for (let s = 0; s < segCount; s++) {
    const v = s * FT_VERTS_PER_SEG;
    const o = s * FT_INDICES_PER_SEG;
    idx[o] = v; idx[o + 1] = v + 1; idx[o + 2] = v + 2;
    idx[o + 3] = v + 2; idx[o + 4] = v + 1; idx[o + 5] = v + 3;
  }
  return idx;
}

const wrapOk = (x0: number, x1: number): boolean => Math.abs(x0 - x1) <= 0.5;

/**
 * Pure: track samples → packed vertex array containing the ground trace,
 * the curtain and the altitude line (in that draw order), 4 verts per
 * surviving segment. `altScale` is the terrain exaggeration factor (1 when
 * terrain is off) — REAL altitudes are scaled into the display datum here;
 * groundZ is already in that datum.
 *
 * Segment rules:
 *  - trace: emitted between every consecutive sample pair on the same side
 *    of the antimeridian (position history is real through altitude gaps);
 *  - curtain + line: additionally require BOTH ends to have altitude
 *    (honest gaps).
 */
export function buildTrackVertices(input: TrackGeomInput, altScale: number): Float32Array {
  const n = Math.min(
    input.merc.length >> 1,
    input.altM.length,
    input.groundZ.length,
  );
  const marks = input.tzMarkIdx ?? [];
  const maxSegs = Math.max(0, n - 1) * 3 + marks.length;
  const out = new Float32Array(maxSegs * FT_VERTS_PER_SEG * FT_VERT_STRIDE);
  let o = 0;
  const put = (
    x: number, y: number, z: number,
    ox: number, oy: number, oz: number,
    side: number, dir: number, w: number,
    r: number, g: number, b: number, a: number,
  ) => {
    out[o++] = x; out[o++] = y; out[o++] = z;
    out[o++] = ox; out[o++] = oy; out[o++] = oz;
    out[o++] = side; out[o++] = dir; out[o++] = w;
    out[o++] = r; out[o++] = g; out[o++] = b; out[o++] = a;
  };
  // ribbon quad between two endpoints (screen-space extrusion, arc pattern)
  const ribbon = (
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    w: number,
    ca: [number, number, number, number], cb: [number, number, number, number],
  ) => {
    put(ax, ay, az, bx, by, bz, -1, +1, w, ca[0], ca[1], ca[2], ca[3]);
    put(ax, ay, az, bx, by, bz, +1, +1, w, ca[0], ca[1], ca[2], ca[3]);
    put(bx, by, bz, ax, ay, az, -1, -1, w, cb[0], cb[1], cb[2], cb[3]);
    put(bx, by, bz, ax, ay, az, +1, -1, w, cb[0], cb[1], cb[2], cb[3]);
  };

  const { merc, altM, groundZ, altMin, altMax } = input;
  const traceLift = TRACE_ABOVE_TERRAIN_M * altScale;
  const curtainDrop = (input.drapeBelowM ?? CURTAIN_BELOW_TERRAIN_M) * altScale;

  // 1) ground trace (draws through altitude gaps)
  for (let i = 0; i + 1 < n; i++) {
    const ax = merc[i * 2], ay = merc[i * 2 + 1];
    const bx = merc[i * 2 + 2], by = merc[i * 2 + 3];
    if (!wrapOk(ax, bx)) continue;
    ribbon(ax, ay, groundZ[i] + traceLift, bx, by, groundZ[i + 1] + traceLift,
      TRACE_WIDTH_PX, TRACE_RGBA, TRACE_RGBA);
  }
  // 2) the curtain (world-space wall quads: [top_i, bottom_i, top_j, bottom_j])
  for (let i = 0; i + 1 < n; i++) {
    if (Number.isNaN(altM[i]) || Number.isNaN(altM[i + 1])) continue; // honest gap
    const ax = merc[i * 2], ay = merc[i * 2 + 1];
    const bx = merc[i * 2 + 2], by = merc[i * 2 + 3];
    if (!wrapOk(ax, bx)) continue;
    const ca = altRampColor(altM[i], altMin, altMax);
    const cb = altRampColor(altM[i + 1], altMin, altMax);
    const topA = altM[i] * altScale, topB = altM[i + 1] * altScale;
    const botA = groundZ[i] - curtainDrop, botB = groundZ[i + 1] - curtainDrop;
    // side/dir/width all 0: side 0 zeroes the screen extrusion (world-space
    // quad) and the fragment edge rim; a_other = self.
    put(ax, ay, topA, ax, ay, topA, 0, 0, 0, ca[0], ca[1], ca[2], CURTAIN_ALPHA);
    put(ax, ay, botA, ax, ay, botA, 0, 0, 0,
      ca[0] * CURTAIN_BOTTOM_MUL[0], ca[1] * CURTAIN_BOTTOM_MUL[1], ca[2] * CURTAIN_BOTTOM_MUL[2],
      CURTAIN_ALPHA);
    put(bx, by, topB, bx, by, topB, 0, 0, 0, cb[0], cb[1], cb[2], CURTAIN_ALPHA);
    put(bx, by, botB, bx, by, botB, 0, 0, 0,
      cb[0] * CURTAIN_BOTTOM_MUL[0], cb[1] * CURTAIN_BOTTOM_MUL[1], cb[2] * CURTAIN_BOTTOM_MUL[2],
      CURTAIN_ALPHA);
  }
  // 3) altitude line (ramp-colored ribbon at altitude)
  for (let i = 0; i + 1 < n; i++) {
    if (Number.isNaN(altM[i]) || Number.isNaN(altM[i + 1])) continue;
    const ax = merc[i * 2], ay = merc[i * 2 + 1];
    const bx = merc[i * 2 + 2], by = merc[i * 2 + 3];
    if (!wrapOk(ax, bx)) continue;
    const ca = altRampColor(altM[i], altMin, altMax);
    const cb = altRampColor(altM[i + 1], altMin, altMax);
    ribbon(ax, ay, altM[i] * altScale, bx, by, altM[i + 1] * altScale,
      ALT_LINE_WIDTH_PX, [ca[0], ca[1], ca[2], 1], [cb[0], cb[1], cb[2], 1]);
  }
  // 4) tz-crossing marks: a vertical screen-extruded ribbon at the crossing
  // sample, curtain base → the flight path (the ribbon extrusion works for
  // vertical segments too — self/other differ only in z, so dirPx is the
  // screen-projected vertical and the normal offsets sideways). Drawn last
  // so the amber sits over the curtain fill it annotates.
  for (const mi of marks) {
    if (!(mi >= 0 && mi < n)) continue;
    if (Number.isNaN(altM[mi])) continue; // honest: no altitude, no mark line
    const x = merc[mi * 2], y = merc[mi * 2 + 1];
    const top = altM[mi] * altScale;
    const bot = groundZ[mi] - curtainDrop;
    if (top <= bot) continue;
    ribbon(x, y, bot, x, y, top, TZ_MARK_WIDTH_PX, TZ_MARK_RGBA, TZ_MARK_RGBA);
  }
  return o === out.length ? out : out.slice(0, o);
}

/** The moving live tail (followed aircraft): one trace + curtain + line
 *  segment from the last archived/crumb sample to the plane's glided
 *  position — rebuilt every glide tick without touching the main buffer. */
export interface TrackTail {
  fromMercX: number; fromMercY: number; fromAltM: number; fromGroundZ: number;
  toMercX: number; toMercY: number; toAltM: number; toGroundZ: number;
  altMin: number; altMax: number;
  drapeBelowM?: number;
}

/** Pure: tail → the same packed layout as buildTrackVertices (≤3 quads). */
export function buildTailVertices(tail: TrackTail, altScale: number): Float32Array {
  const merc = new Float32Array([tail.fromMercX, tail.fromMercY, tail.toMercX, tail.toMercY]);
  const altM = new Float32Array([tail.fromAltM, tail.toAltM]);
  const groundZ = new Float32Array([tail.fromGroundZ, tail.toGroundZ]);
  return buildTrackVertices(
    { merc, altM, groundZ, altMin: tail.altMin, altMax: tail.altMax, drapeBelowM: tail.drapeBelowM },
    altScale,
  );
}

/** Exported for flightTrackLayer.test.ts — pins the projection/extrusion/
 *  cull contract (identical semantics to the arcLayer ribbon shader). */
export const FT_VERT_SRC = (prelude: string, define: string): string => `#version 300 es
${prelude}
${define}
${VT_PROJ_ELEV_GLSL}
in vec3 a_pos;
in vec3 a_other;
in vec3 a_ext;    // x: side (-1|0|+1; 0 = world-space wall vertex), y: dirSign, z: widthPx
in vec4 a_color;
uniform vec2 u_viewport;
out vec4 v_color;
out float v_cull;
out float v_edge;
void main() {
  v_cull = 0.0;
#ifdef GLOBE
  // far-side test — identical to satLayer/airLayer/arcLayer (0.998001 =
  // OCCLUSION_RADIUS²), flagged to the FRAGMENT stage: ribbons/walls fade
  // at the horizon instead of stretching to a snapped vertex. Runs through
  // the WHOLE globe↔mercator transition, not just full globe (same fix as
  // satLayer v1.0.563 — the transition is zoom-driven and PERSISTENT).
  if (u_projection_transition > 0.0 && u_projection_clipping_plane.w < 0.0) {
    vec3 satPos = projectToSphere(a_pos.xy) * (1.0 + a_pos.z / GLOBE_RADIUS);
    vec3 cam = u_projection_clipping_plane.xyz * (-1.0 / u_projection_clipping_plane.w);
    vec3 v = satPos - cam;
    float t = -dot(cam, v) / dot(v, v);
    if (t > 0.0 && t < 1.0) {
      vec3 closest = cam + t * v;
      if (dot(closest, closest) < 0.998001) v_cull = 1.0;
    }
  }
#endif
  vec4 self = projectTileFor3D(a_pos.xy, vtProjElev(a_pos.z, a_pos.y));
  vec4 other = projectTileFor3D(a_other.xy, vtProjElev(a_other.z, a_other.y));
  vec2 ndcSelf = self.xy / max(abs(self.w), 1e-9);
  vec2 ndcOther = other.xy / max(abs(other.w), 1e-9);
  vec2 dirPx = (ndcOther - ndcSelf) * a_ext.y * u_viewport;
  float len = length(dirPx);
  dirPx = len < 1e-6 ? vec2(1.0, 0.0) : dirPx / len;
  vec2 normalPx = vec2(-dirPx.y, dirPx.x) * a_ext.x;
  vec2 offs = normalPx * (a_ext.z * 0.5) * 2.0 / u_viewport;
  gl_Position = self + vec4(offs * self.w, 0.0, 0.0);
  v_color = a_color;
  v_edge = a_ext.x;
}`;

/** Exported for flightTrackLayer.test.ts — fragment discard + ribbon rim
 *  (side 0 wall vertices have v_edge 0, so the rim never eats the
 *  curtain's 34% alpha). */
export const FT_FRAG_SRC = `#version 300 es
precision mediump float;
in vec4 v_color;
in float v_cull;
in float v_edge;
out vec4 o;
void main() {
  if (v_cull > 0.01) discard;
  o = v_color;
  o.a *= mix(1.0, 0.55, abs(v_edge));
}`;

/** Marker program: one anchored glyph/dot/drop-line in local pixel space
 *  (airLayer's anchor.w screen-constant pattern, single instance via
 *  uniforms). a_altFrac 0 = ground end, 1 = altitude end. Exported for
 *  flightTrackLayer.test.ts. */
export const FT_MARKER_VERT_SRC = (prelude: string, define: string): string => `#version 300 es
${prelude}
${define}
${VT_PROJ_ELEV_GLSL}
in vec2 a_local;
in float a_altFrac;
uniform vec2 u_anchor;     // normalized mercator
uniform float u_altZ;      // altitude end, display-datum meters
uniform float u_groundZ;   // ground end, display-datum meters
uniform float u_rot;       // radians: heading - map bearing
uniform float u_scaleClip; // clip units per local px at w=1
uniform float u_aspect;
uniform vec4 u_color;
out vec4 v_color;
void main() {
#ifdef GLOBE
  // same whole-transition far-side cull as FT_VERT_SRC above (satLayer
  // v1.0.563 fix) — the marker anchor must not draw as a far-side ghost
  // mid-transition either.
  if (u_projection_transition > 0.0 && u_projection_clipping_plane.w < 0.0) {
    vec3 satPos = projectToSphere(u_anchor) * (1.0 + u_altZ / GLOBE_RADIUS);
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
  vec4 anchor = projectTileFor3D(u_anchor, vtProjElev(mix(u_groundZ, u_altZ, a_altFrac), u_anchor.y));
  float c = cos(u_rot);
  float s = sin(u_rot);
  vec2 p = vec2(a_local.x * c + a_local.y * s, -a_local.x * s + a_local.y * c);
  gl_Position = anchor + vec4(p.x * u_scaleClip / u_aspect, p.y * u_scaleClip, 0.0, 0.0) * anchor.w;
  v_color = u_color;
}`;

const FT_MARKER_FRAG_SRC = `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 o;
void main() { o = v_color; }`;

export interface TrackMarker {
  mercX: number;
  mercY: number;
  /** REAL altitude meters (scaled by altScale in the layer); NaN = altitude
   *  gap — glyph rides the ground trace, no drop line. */
  altM: number;
  /** display-datum ground meters under the marker. */
  groundZ: number;
  /** track-tangent heading, deg (0 = N, clockwise). */
  headingDeg: number;
  /** AIR_SHAPE code — the selected aircraft's class silhouette
   *  (symbols-not-dots holds for the marker too). */
  shape: number;
}

// dot: a small circle fan in local px space (unit radius, scaled by u_scaleClip)
const DOT_SEGS = 20;
function buildDotFan(radiusPx: number): Float32Array {
  const v: number[] = [];
  for (let i = 0; i < DOT_SEGS; i++) {
    const a0 = (i / DOT_SEGS) * Math.PI * 2;
    const a1 = ((i + 1) / DOT_SEGS) * Math.PI * 2;
    v.push(0, 0, Math.cos(a0) * radiusPx, Math.sin(a0) * radiusPx,
      Math.cos(a1) * radiusPx, Math.sin(a1) * radiusPx);
  }
  return new Float32Array(v);
}

export class FlightTrackLayer implements CustomLayerInterface {
  readonly id: string;
  readonly type = 'custom' as const;
  readonly renderingMode = '2d' as const;

  private map: MapLibreMap | null = null;
  // geometry program (trace + curtain + line + tail)
  private program: WebGLProgram | null = null;
  private cachedVariant: string | null = null;
  private buffer: WebGLBuffer | null = null;
  private indexBuffer: WebGLBuffer | null = null;
  private tailBuffer: WebGLBuffer | null = null;
  private tailIndexBuffer: WebGLBuffer | null = null;
  private aPos = -1; private aOther = -1; private aExt = -1; private aColor = -1;
  private uViewport: WebGLUniformLocation | null = null;
  private geomProj: Record<string, WebGLUniformLocation | null> = {};
  // marker program
  private mProgram: WebGLProgram | null = null;
  private mGlyphBuffer: WebGLBuffer | null = null;
  private mDotBuffer: WebGLBuffer | null = null;
  private mLineBuffer: WebGLBuffer | null = null;
  private mGlyphVerts = 0;
  private mGlyphShape = -1;
  private mALocal = -1; private mAAltFrac = -1;
  private mU: Record<string, WebGLUniformLocation | null> = {};
  private markerProj: Record<string, WebGLUniformLocation | null> = {};

  private verts: Float32Array | null = null;
  private indices: Uint32Array | null = null;
  private tailVerts: Float32Array | null = null;
  private tailIndices: Uint32Array | null = null;
  private marker: TrackMarker | null = null;
  private altScale = 1;
  private dirty = false;
  private tailDirty = false;
  // SELF-HEALING render failures (live report 2026-07-20: "the 3d trail/
  // curtain does not show" — the old boolean latch meant ONE transient GL
  // exception, e.g. a context loss during a terrain hiccup, killed the
  // layer for the whole session even after the context recovered). A
  // failure now DROPS the GL objects (stale handles from a lost context
  // would throw forever) and counts a streak; new data (setTrack — every
  // poll) re-arms a retry, which rebuilds on the healthy context. Only a
  // persistent streak with no successes stays disabled.
  private failStreak = 0;
  private static readonly MAX_FAIL_STREAK = 5;
  // last frame's projection (for CPU screen-projection of the floating tag
  // — the pickNearestAircraftScreen precedent, extended to mercator via the
  // prelude's own formula: matrix × [mercX, mercY, elevationMeters, 1])
  private lastMainMatrix: Float32Array | null = null;
  private lastTransition = 0;

  constructor(opts: { id?: string } = {}) {
    this.id = opts.id ?? 'flight-track-3d';
  }

  onAdd(map: MapLibreMap, _gl: AnyGl): void {
    this.map = map;
  }

  onRemove(_map: MapLibreMap, gl: AnyGl): void {
    this.dropGlObjects(gl);
    this.map = null;
  }

  /** Forget every GL handle (deleting on a possibly-lost context is safe —
   *  deletes are no-ops there) so the next render rebuilds from scratch. */
  private dropGlObjects(gl?: AnyGl): void {
    try {
      if (gl) {
        for (const p of [this.program, this.mProgram]) if (p) gl.deleteProgram(p);
        for (const b of [this.buffer, this.indexBuffer, this.tailBuffer, this.tailIndexBuffer,
          this.mGlyphBuffer, this.mDotBuffer, this.mLineBuffer]) if (b) gl.deleteBuffer(b);
      }
    } catch { /* context gone — handles are already invalid */ }
    this.program = this.mProgram = null;
    this.buffer = this.indexBuffer = this.tailBuffer = this.tailIndexBuffer = null;
    this.mGlyphBuffer = this.mDotBuffer = this.mLineBuffer = null;
    this.mGlyphShape = -1;
    this.cachedVariant = null;
    this.dirty = this.verts != null;
    this.tailDirty = this.tailVerts != null;
  }

  /** Replace the displayed track (null clears — the layer then costs
   *  nothing). altScale = terrain exaggeration (1 when terrain is off);
   *  input.groundZ is already in the exaggerated datum. */
  setTrack(input: TrackGeomInput | null, altScale = 1): void {
    this.altScale = altScale;
    this.verts = input ? buildTrackVertices(input, altScale) : null;
    this.indices = this.verts && this.verts.length
      ? buildQuadIndices(this.verts.length / FT_VERT_STRIDE / FT_VERTS_PER_SEG)
      : null;
    this.dirty = this.verts != null;
    this.failStreak = 0; // new data re-arms rendering after transient GL failures
    this.map?.triggerRepaint();
  }

  /** Update the live moving tail only (cheap per-glide-tick path). */
  setTail(tail: TrackTail | null): void {
    this.tailVerts = tail ? buildTailVertices(tail, this.altScale) : null;
    this.tailIndices = this.tailVerts && this.tailVerts.length
      ? buildQuadIndices(this.tailVerts.length / FT_VERT_STRIDE / FT_VERTS_PER_SEG)
      : null;
    this.tailDirty = this.tailVerts != null;
    this.map?.triggerRepaint();
  }

  /** Place/move the aircraft marker (null hides it). Cheap: uniforms only. */
  setMarker(m: TrackMarker | null): void {
    this.marker = m;
    this.map?.triggerRepaint();
  }

  getMarker(): TrackMarker | null {
    return this.marker;
  }

  /** Rendered geometry vertex count (harness/test seam, arcLayer parity). */
  getVertexCount(): number {
    return (this.verts ? this.verts.length / FT_VERT_STRIDE : 0)
      + (this.tailVerts ? this.tailVerts.length / FT_VERT_STRIDE : 0);
  }

  getRenderFailed(): boolean {
    return this.failStreak >= FlightTrackLayer.MAX_FAIL_STREAK;
  }

  /**
   * Project a 3D track point (normalized mercator + REAL altitude meters)
   * to screen pixels using the SAME matrix the shader drew with last frame
   * — the floating tag anchors exactly above the drawn marker. Returns null
   * before the first frame or when the point is behind the camera /
   * far side of the globe.
   */
  projectToScreen(
    mercX: number,
    mercY: number,
    altM: number,
    widthPx: number,
    heightPx: number,
  ): { x: number; y: number } | null {
    const m = this.lastMainMatrix;
    if (!m) return null;
    const z = Number.isNaN(altM) ? 0 : altM * this.altScale;
    let p: [number, number, number];
    if (this.lastTransition > 0.999) {
      p = mercatorToSphere(mercX, mercY, z) as [number, number, number];
    } else {
      // pd.mainMatrix consumes MERCATOR-unit z, never meters — with raw
      // meters the tag flew off-screen in tilted mercator views (same
      // latent bug as the picks; pinned 2026-07-20)
      p = [mercX, mercY, mercatorZFromAltitude(z, mercY)];
    }
    const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    if (!(w > 0)) return null;
    const cx = (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w;
    const cy = (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w;
    if (cx < -1.3 || cx > 1.3 || cy < -1.3 || cy > 1.3) return null;
    return { x: ((cx + 1) / 2) * widthPx, y: ((1 - cy) / 2) * heightPx };
  }

  render(gl: AnyGl, args: CustomRenderMethodInput): void {
    if (this.failStreak >= FlightTrackLayer.MAX_FAIL_STREAK) return;
    const hasGeom = this.verts != null && this.verts.length > 0;
    const hasTail = this.tailVerts != null && this.tailVerts.length > 0;
    if (!hasGeom && !hasTail && !this.marker) return; // nothing set → zero cost
    try {
      this.renderInner(gl as WebGL2RenderingContext, args, hasGeom, hasTail);
      this.failStreak = 0; // a clean frame ends any failure streak
    } catch (e) {
      this.failStreak++;
      this.dropGlObjects(gl); // stale handles from a lost context throw forever
      // eslint-disable-next-line no-console
      console.error(
        `FlightTrackLayer: render failure ${this.failStreak}/${FlightTrackLayer.MAX_FAIL_STREAK} ` +
        '(GL objects dropped for a clean retry; map continues):', e);
    }
  }

  private renderInner(
    gl: WebGL2RenderingContext,
    args: CustomRenderMethodInput,
    hasGeom: boolean,
    hasTail: boolean,
  ): void {
    const sd = args.shaderData;
    if (this.program == null || this.cachedVariant !== sd.variantName) {
      this.compile(gl, sd.vertexShaderPrelude, sd.define, sd.variantName);
      this.dirty = this.verts != null;
      this.tailDirty = this.tailVerts != null;
      this.mGlyphShape = -1;
    }
    if (this.program == null || this.mProgram == null) return;

    // cache this frame's projection for the tag's CPU screen-projection
    const pd0 = args.defaultProjectionData;
    if (!this.lastMainMatrix) this.lastMainMatrix = new Float32Array(16);
    this.lastMainMatrix.set(pd0.mainMatrix as ArrayLike<number>);
    this.lastTransition = pd0.projectionTransition;

    // THE CRITICAL FIX (c), amended 2026-07-20 (probe-bisected): WITHOUT
    // terrain the depth buffer is the normal cleared-far one — depth-test
    // LEQUAL works and costs nothing. WITH terrain enabled MapLibre leaves
    // NO usable depth for '2d' custom layers (every fragment failed: the
    // trail was invisible, census 223 vs 2,502) — so the test is SKIPPED
    // there: the curtain X-rays through ridges rather than vanishing.
    // Never writes depth either way. MapLibre re-syncs its own tracked GL
    // state after custom layers via setDirty().
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const terrainOn = !!(this.map && (this.map as unknown as { getTerrain?: () => unknown }).getTerrain?.());
    if (terrainOn) gl.disable(gl.DEPTH_TEST);
    else gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(false);
    gl.depthRange(0, 1);
    // THE CRITICAL FIX (b): double-sided — never back-face-culled.
    gl.disable(gl.CULL_FACE);

    if (hasGeom || hasTail) {
      gl.useProgram(this.program);
      this.bindProjection(gl, args, this.geomProj);
      if (this.uViewport) {
        gl.uniform2f(this.uViewport, gl.drawingBufferWidth || 1, gl.drawingBufferHeight || 1);
      }
      if (hasGeom) {
        if (!this.buffer) this.buffer = gl.createBuffer();
        if (!this.indexBuffer) this.indexBuffer = gl.createBuffer();
        this.drawGeom(gl, this.buffer, this.indexBuffer, this.verts as Float32Array,
          this.indices as Uint32Array, this.dirty);
        this.dirty = false;
      }
      if (hasTail) {
        if (!this.tailBuffer) this.tailBuffer = gl.createBuffer();
        if (!this.tailIndexBuffer) this.tailIndexBuffer = gl.createBuffer();
        this.drawGeom(gl, this.tailBuffer, this.tailIndexBuffer, this.tailVerts as Float32Array,
          this.tailIndices as Uint32Array, this.tailDirty);
        this.tailDirty = false;
      }
      gl.disableVertexAttribArray(this.aPos);
      gl.disableVertexAttribArray(this.aOther);
      gl.disableVertexAttribArray(this.aExt);
      gl.disableVertexAttribArray(this.aColor);
    }

    if (this.marker) this.drawMarker(gl, args);
  }

  private drawGeom(
    gl: WebGL2RenderingContext,
    buf: WebGLBuffer | null,
    ibuf: WebGLBuffer | null,
    verts: Float32Array,
    indices: Uint32Array,
    upload: boolean,
  ): void {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf);
    if (upload) {
      gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    }
    const strideB = FT_VERT_STRIDE * 4;
    gl.enableVertexAttribArray(this.aPos);
    gl.vertexAttribPointer(this.aPos, 3, gl.FLOAT, false, strideB, 0);
    gl.enableVertexAttribArray(this.aOther);
    gl.vertexAttribPointer(this.aOther, 3, gl.FLOAT, false, strideB, 12);
    gl.enableVertexAttribArray(this.aExt);
    gl.vertexAttribPointer(this.aExt, 3, gl.FLOAT, false, strideB, 24);
    gl.enableVertexAttribArray(this.aColor);
    gl.vertexAttribPointer(this.aColor, 4, gl.FLOAT, false, strideB, 36);
    gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_INT, 0);
  }

  private drawMarker(gl: WebGL2RenderingContext, args: CustomRenderMethodInput): void {
    const m = this.marker as TrackMarker;
    gl.useProgram(this.mProgram);
    this.bindProjection(gl, args, this.markerProj);

    const h = gl.drawingBufferHeight || 1;
    const w = gl.drawingBufferWidth || 1;
    if (this.mU.scaleClip) gl.uniform1f(this.mU.scaleClip, 2 / h);
    if (this.mU.aspect) gl.uniform1f(this.mU.aspect, w / h);
    if (this.mU.anchor) gl.uniform2f(this.mU.anchor, m.mercX, m.mercY);
    const hasAlt = !Number.isNaN(m.altM);
    const altZ = hasAlt ? m.altM * this.altScale : m.groundZ + TRACE_ABOVE_TERRAIN_M * this.altScale;
    if (this.mU.altZ) gl.uniform1f(this.mU.altZ, altZ);
    if (this.mU.groundZ) gl.uniform1f(this.mU.groundZ, m.groundZ + 14 * this.altScale);
    const bearing = this.map?.getBearing() ?? 0;
    if (this.mU.rot) gl.uniform1f(this.mU.rot, ((m.headingDeg - bearing) * Math.PI) / 180);

    if (!this.mDotBuffer) {
      this.mDotBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.mDotBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, buildDotFan(MARKER_DOT_PX), gl.STATIC_DRAW);
    }
    if (!this.mLineBuffer) {
      this.mLineBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.mLineBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 1]), gl.STATIC_DRAW);
    }
    const shape = Number.isFinite(m.shape) ? m.shape : AIR_SHAPE.DEFAULT;
    if (!this.mGlyphBuffer || this.mGlyphShape !== shape) {
      if (!this.mGlyphBuffer) this.mGlyphBuffer = gl.createBuffer();
      const sil = AIR_SILHOUETTES[shape] ?? AIR_SILHOUETTES[AIR_SHAPE.DEFAULT];
      // silhouettes span ~±0.62 local units — scale to MARKER_PIXELS px
      const px = new Float32Array(sil.length);
      const k = MARKER_PIXELS / 1.24;
      for (let i = 0; i < sil.length; i++) px[i] = sil[i] * k;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.mGlyphBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, px, gl.STATIC_DRAW);
      this.mGlyphVerts = px.length / 2;
      this.mGlyphShape = shape;
    }

    // 1) AGL drop line (only when altitude is known — never invent AGL)
    if (hasAlt) {
      if (this.mU.color) gl.uniform4fv(this.mU.color, MARKER_LINE_RGBA);
      gl.disableVertexAttribArray(this.mALocal);
      gl.vertexAttrib2f(this.mALocal, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.mLineBuffer);
      gl.enableVertexAttribArray(this.mAAltFrac);
      gl.vertexAttribPointer(this.mAAltFrac, 1, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, 2);
      gl.disableVertexAttribArray(this.mAAltFrac);
    }
    // 2) soft ground dot (altFrac 0)
    if (this.mU.color) gl.uniform4fv(this.mU.color, MARKER_DOT_RGBA);
    gl.vertexAttrib1f(this.mAAltFrac, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.mDotBuffer);
    gl.enableVertexAttribArray(this.mALocal);
    gl.vertexAttribPointer(this.mALocal, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, DOT_SEGS * 3);
    // 3) the glyph at altitude (altFrac 1), nose along the track tangent
    if (this.mU.color) gl.uniform4fv(this.mU.color, MARKER_GLYPH_RGBA);
    gl.vertexAttrib1f(this.mAAltFrac, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.mGlyphBuffer);
    gl.vertexAttribPointer(this.mALocal, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, this.mGlyphVerts);
    gl.disableVertexAttribArray(this.mALocal);
  }

  private bindProjection(
    gl: WebGL2RenderingContext,
    args: CustomRenderMethodInput,
    u: Record<string, WebGLUniformLocation | null>,
  ): void {
    const pd = args.defaultProjectionData;
    if (u.matrix) gl.uniformMatrix4fv(u.matrix, false, pd.mainMatrix);
    if (u.tile) {
      gl.uniform4f(u.tile, pd.tileMercatorCoords[0], pd.tileMercatorCoords[1],
        pd.tileMercatorCoords[2], pd.tileMercatorCoords[3]);
    }
    if (u.clip) {
      gl.uniform4f(u.clip, pd.clippingPlane[0], pd.clippingPlane[1],
        pd.clippingPlane[2], pd.clippingPlane[3]);
    }
    if (u.trans) gl.uniform1f(u.trans, pd.projectionTransition);
    if (u.fallback) gl.uniformMatrix4fv(u.fallback, false, pd.fallbackMatrix);
  }

  private compile(gl: WebGL2RenderingContext, prelude: string, define: string, variant: string): void {
    for (const p of [this.program, this.mProgram]) if (p) gl.deleteProgram(p);
    this.program = this.mProgram = null;
    const mk = (type: number, src: string): WebGLShader => {
      const sh = gl.createShader(type);
      if (!sh) throw new Error('FlightTrackLayer: createShader failed');
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error('FlightTrackLayer: shader compile failed: ' + log);
      }
      return sh;
    };
    const link = (vsSrc: string, fsSrc: string): WebGLProgram => {
      const vs = mk(gl.VERTEX_SHADER, vsSrc);
      const fs = mk(gl.FRAGMENT_SHADER, fsSrc);
      const p = gl.createProgram();
      if (!p) throw new Error('FlightTrackLayer: createProgram failed');
      gl.attachShader(p, vs);
      gl.attachShader(p, fs);
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(p);
        gl.deleteProgram(p);
        throw new Error('FlightTrackLayer: program link failed: ' + log);
      }
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      return p;
    };
    const proj = (p: WebGLProgram): Record<string, WebGLUniformLocation | null> => ({
      matrix: gl.getUniformLocation(p, 'u_projection_matrix'),
      tile: gl.getUniformLocation(p, 'u_projection_tile_mercator_coords'),
      clip: gl.getUniformLocation(p, 'u_projection_clipping_plane'),
      trans: gl.getUniformLocation(p, 'u_projection_transition'),
      fallback: gl.getUniformLocation(p, 'u_projection_fallback_matrix'),
    });
    const gp = link(FT_VERT_SRC(prelude, define), FT_FRAG_SRC);
    this.program = gp;
    this.aPos = gl.getAttribLocation(gp, 'a_pos');
    this.aOther = gl.getAttribLocation(gp, 'a_other');
    this.aExt = gl.getAttribLocation(gp, 'a_ext');
    this.aColor = gl.getAttribLocation(gp, 'a_color');
    this.uViewport = gl.getUniformLocation(gp, 'u_viewport');
    this.geomProj = proj(gp);

    const mp = link(FT_MARKER_VERT_SRC(prelude, define), FT_MARKER_FRAG_SRC);
    this.mProgram = mp;
    this.mALocal = gl.getAttribLocation(mp, 'a_local');
    this.mAAltFrac = gl.getAttribLocation(mp, 'a_altFrac');
    this.mU = {
      anchor: gl.getUniformLocation(mp, 'u_anchor'),
      altZ: gl.getUniformLocation(mp, 'u_altZ'),
      groundZ: gl.getUniformLocation(mp, 'u_groundZ'),
      rot: gl.getUniformLocation(mp, 'u_rot'),
      scaleClip: gl.getUniformLocation(mp, 'u_scaleClip'),
      aspect: gl.getUniformLocation(mp, 'u_aspect'),
      color: gl.getUniformLocation(mp, 'u_color'),
    };
    this.markerProj = proj(mp);
    this.cachedVariant = variant;
  }
}
