// FLEET REPLAY LAYER — the Time Machine's curtain fleet (FLIGHT PROGRAM,
// human directive 2026-09-28; earth_twin_program.md TIME MACHINE v2 T-3:
// "a MultiTrackLayer batching N track geometries into one buffer/draw …
// one GL layer, not N layers").
//
// One MapLibre CustomLayerInterface, FOUR draws total no matter how many
// aircraft are replayed:
//   1. THIN   — every thin-LOD track's altitude polyline, ONE batched buffer
//   2. FULL   — every full-LOD track's ground trace + curtain + altitude
//               line, ONE batched buffer (flightTrackLayer's handoff styling)
//   3. CONN   — close-approach connector ribbons (tiny, per frame)
//   4. HEADS  — every head as ONE instanced draw of a heading-aligned
//               chevron (symbols, not dots)
//
// LAW I: geometry is built ONCE per LOD membership change (fleetModel.ts,
// driven from the frameCore loop by fleetReplayController.ts) and never on a
// camera event. The replay PLAYHEAD moves per frame as a uniform: every
// vertex carries its fix time, the fragment stage discards the future and
// fades the trail past FLEET_TRAIL_SEC — the curtains grow with the replay
// without a single rebuild. Heads are an instance buffer re-filled per frame
// (≤ a few thousand × 32 B).
//
// Depth/cull/blend state and the globe far-side cull are FlightTrackLayer's
// proven rules (the CRITICAL FIX a/b/c comments there): double-sided, no
// depth writes, depth-test only when terrain is off.
//
// LAW IV: exports maxFeatures (tracks), vramBudget (MB, derived from the
// full-tier FLEET_BUDGETS — fleetWorstCaseBytes, pinned by the test), and
// the class implements dispose().

import type { CustomLayerInterface, CustomRenderMethodInput, Map as MapLibreMap } from "maplibre-gl";
import { VT_PROJ_ELEV_GLSL } from "../glElev.js";
import { mercatorToSphere, mercatorZFromAltitude } from "../orbital/occlusion.js";
import { setGauge } from "../../render/perfMetrics.js";
import { buildQuadIndices } from "./flightTrackLayer.js";
import {
  FLEET_BUDGETS, FLEET_STRIDE, FLEET_VERTS_PER_SEG, FLEET_TRAIL_SEC, HEAD_STRIDE, type FleetBudget,
} from "./fleetModel.js";

type AnyGl = WebGLRenderingContext | WebGL2RenderingContext;

/** Law IV: most tracks one replay renders (= the route's `max` bound) */
export const FLEET_MAX_TRACKS = 2000;
/** close-approach connectors drawn at once (server caps the list at 200) */
export const FLEET_MAX_CONNECTORS = 200;

/** DESIGN.md --accent-red #ff5a6e: the close-approach highlight. Distinct
 *  from the amber replay/tz marks (#f5a524/#ffcc66) and the teal→violet
 *  altitude ramp, so a flagged pair never reads as ordinary track data. */
export const FLEET_WARN_RGBA: [number, number, number, number] = [0xff / 255, 0x5a / 255, 0x6e / 255, 1];
/** alpha multiplier for every non-highlighted track while a pair is focused */
export const FLEET_DIM = 0.22;
/** head glyph size, px (nose to tail) */
export const FLEET_HEAD_PX = 13;

/** worst-case bytes for a budget: FULL segments × 3 quads + THIN segments ×
 *  1 quad (verts + Uint32 indices), heads, connectors */
export function fleetWorstCaseBytes(b: FleetBudget = FLEET_BUDGETS.full): number {
  const quadV = FLEET_VERTS_PER_SEG * FLEET_STRIDE * 4;
  const quadI = 6 * 4;
  const fullQuads = b.fullSegments * 3, thinQuads = b.thinSegments, connQuads = FLEET_MAX_CONNECTORS;
  return (fullQuads + thinQuads + connQuads) * (quadV + quadI) + b.heads * HEAD_STRIDE * 4 + 1024;
}

export const maxFeatures = FLEET_MAX_TRACKS;
export const vramBudget = 40; // MB — fleetWorstCaseBytes(full tier) ≈ 39.8 MB

/** geometry vertex shader: flightTrackLayer's FT_VERT_SRC + per-vertex fix
 *  time and track slot (playhead cut + highlight). Exported for the test. */
export const FLEET_VERT_SRC = (prelude: string, define: string): string => `#version 300 es
${prelude}
${define}
${VT_PROJ_ELEV_GLSL}
in vec3 a_pos;
in vec3 a_other;
in vec3 a_ext;
in vec4 a_color;
in float a_t;
in float a_slot;
uniform vec2 u_viewport;
uniform float u_hlOn;
uniform float u_hlA;
uniform float u_hlB;
uniform vec4 u_hlColor;
uniform float u_dim;
out vec4 v_color;
out float v_cull;
out float v_edge;
out float v_t;
void main() {
  v_cull = 0.0;
#ifdef GLOBE
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
  vec4 c = a_color;
  if (u_hlOn > 0.5) {
    bool hl = abs(a_slot - u_hlA) < 0.5 || abs(a_slot - u_hlB) < 0.5;
    if (hl) c = vec4(mix(c.rgb, u_hlColor.rgb, 0.85), min(1.0, c.a * 1.6));
    else c.a *= u_dim;
  }
  v_color = c;
  v_edge = a_ext.x;
  v_t = a_t;
}`;

/** playhead cut: future discarded, trail faded over its last 40% */
export const FLEET_FRAG_SRC = `#version 300 es
precision highp float;
in vec4 v_color;
in float v_cull;
in float v_edge;
in float v_t;
uniform float u_now;
uniform float u_trail;
out vec4 o;
void main() {
  if (v_cull > 0.01) discard;
  if (v_t > u_now + 0.001) discard;
  float age = u_now - v_t;
  float fade = 1.0 - smoothstep(u_trail * 0.6, u_trail, age);
  if (fade <= 0.002) discard;
  o = v_color;
  o.a *= fade * mix(1.0, 0.55, abs(v_edge));
}`;

/** instanced head glyph: a_inst = (mercX, mercY, zMeters, headingRad) */
export const FLEET_HEAD_VERT_SRC = (prelude: string, define: string): string => `#version 300 es
${prelude}
${define}
${VT_PROJ_ELEV_GLSL}
in vec2 a_local;
in vec4 a_inst;
in vec4 a_icolor;
uniform float u_scaleClip;
uniform float u_aspect;
uniform float u_bearing;
out vec4 v_color;
void main() {
#ifdef GLOBE
  if (u_projection_transition > 0.0 && u_projection_clipping_plane.w < 0.0) {
    vec3 satPos = projectToSphere(a_inst.xy) * (1.0 + a_inst.z / GLOBE_RADIUS);
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
  vec4 anchor = projectTileFor3D(a_inst.xy, vtProjElev(a_inst.z, a_inst.y));
  float rot = a_inst.w - u_bearing;
  float c = cos(rot);
  float s = sin(rot);
  vec2 p = vec2(a_local.x * c + a_local.y * s, -a_local.x * s + a_local.y * c);
  gl_Position = anchor + vec4(p.x * u_scaleClip / u_aspect, p.y * u_scaleClip, 0.0, 0.0) * anchor.w;
  v_color = a_icolor;
}`;

const FLEET_HEAD_FRAG_SRC = `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 o;
void main() { o = v_color; }`;

/** chevron glyph (nose +y), local px — two triangles */
export function headGlyph(px = FLEET_HEAD_PX): Float32Array {
  const k = px / 14;
  return new Float32Array([
    0, 7 * k, -5 * k, -6 * k, 0, -2.5 * k,
    0, 7 * k, 0, -2.5 * k, 5 * k, -6 * k,
  ]);
}

interface Batch {
  verts: Float32Array | null;
  indices: Uint32Array | null;
  buf: WebGLBuffer | null;
  ibuf: WebGLBuffer | null;
  dirty: boolean;
}
const emptyBatch = (): Batch => ({ verts: null, indices: null, buf: null, ibuf: null, dirty: false });

export interface FleetUniformState {
  /** playhead, seconds relative to the geometry's base time */
  nowRel: number;
  trailSec: number;
  /** highlighted track slots (-1 = none) */
  hlA: number;
  hlB: number;
}

export class FleetReplayLayer implements CustomLayerInterface {
  readonly id: string;
  readonly type = "custom" as const;
  readonly renderingMode = "2d" as const;

  private map: MapLibreMap | null = null;
  private glRef: AnyGl | null = null;
  private program: WebGLProgram | null = null;
  private hProgram: WebGLProgram | null = null;
  private cachedVariant: string | null = null;
  private a: Record<string, number> = {};
  private u: Record<string, WebGLUniformLocation | null> = {};
  private gProj: Record<string, WebGLUniformLocation | null> = {};
  private ha: Record<string, number> = {};
  private hu: Record<string, WebGLUniformLocation | null> = {};
  private hProj: Record<string, WebGLUniformLocation | null> = {};

  private full = emptyBatch();
  private thin = emptyBatch();
  private conn = emptyBatch();
  private heads: Float32Array | null = null;
  private headCount = 0;
  private headsDirty = false;
  private headBuf: WebGLBuffer | null = null;
  private headCap = 0;
  private glyphBuf: WebGLBuffer | null = null;
  private glyphVerts = 0;

  private uni: FleetUniformState = { nowRel: 0, trailSec: FLEET_TRAIL_SEC, hlA: -1, hlB: -1 };
  private failStreak = 0;
  private drawnFrames = 0;
  private deleteErrors = 0;
  private static readonly MAX_FAIL_STREAK = 5;
  private lastMainMatrix: Float32Array | null = null;
  private lastTransition = 0;
  private altScale = 1;

  constructor(opts: { id?: string } = {}) {
    this.id = opts.id ?? "fleet-replay-3d";
  }

  onAdd(map: MapLibreMap, gl: AnyGl): void {
    this.map = map;
    this.glRef = gl;
    // a re-add (context restore / style reload) must re-upload everything
    for (const b of [this.full, this.thin, this.conn]) b.dirty = b.verts != null;
    this.headsDirty = this.heads != null;
  }

  onRemove(_map: MapLibreMap, gl: AnyGl): void {
    this.dropGlObjects(gl);
    this.map = null;
  }

  /** Law IV explicit teardown (idempotent, safe on a lost context). */
  dispose(): void {
    const gl = this.glRef;
    this.glRef = null;
    if (gl) this.dropGlObjects(gl);
    this.map = null;
    this.full = emptyBatch();
    this.thin = emptyBatch();
    this.conn = emptyBatch();
    this.heads = null;
    this.headCount = 0;
  }

  private dropGlObjects(gl?: AnyGl): void {
    try {
      if (gl) {
        for (const p of [this.program, this.hProgram]) if (p) gl.deleteProgram(p);
        for (const b of [this.full, this.thin, this.conn]) {
          if (b.buf) gl.deleteBuffer(b.buf);
          if (b.ibuf) gl.deleteBuffer(b.ibuf);
        }
        if (this.headBuf) gl.deleteBuffer(this.headBuf);
        if (this.glyphBuf) gl.deleteBuffer(this.glyphBuf);
      }
    } catch {
      // lost context: handles are already invalid — count it, keep tearing down
      this.deleteErrors++;
      setGauge("fleetReplay.glDeleteErrors", this.deleteErrors);
    }
    this.program = this.hProgram = null;
    for (const b of [this.full, this.thin, this.conn]) {
      b.buf = b.ibuf = null;
      b.dirty = b.verts != null;
    }
    this.headBuf = this.glyphBuf = null;
    this.headCap = 0;
    this.headsDirty = this.heads != null;
    this.cachedVariant = null;
  }

  private setBatch(b: Batch, verts: Float32Array | null): void {
    b.verts = verts && verts.length ? verts : null;
    b.indices = b.verts ? buildQuadIndices(b.verts.length / FLEET_STRIDE / FLEET_VERTS_PER_SEG) : null;
    b.dirty = b.verts != null;
    this.failStreak = 0;
    this.map?.triggerRepaint();
  }

  /** batched FULL-LOD geometry (one draw) */
  setFull(verts: Float32Array | null): void { this.setBatch(this.full, verts); }
  /** batched THIN-LOD geometry (one draw) */
  setThin(verts: Float32Array | null): void { this.setBatch(this.thin, verts); }
  /** close-approach connectors (rebuilt per frame while any is shown) */
  setConnectors(verts: Float32Array | null): void { this.setBatch(this.conn, verts); }

  /** per-frame head instances (HEAD_STRIDE floats each) */
  setHeads(inst: Float32Array | null, count: number): void {
    this.heads = inst;
    this.headCount = inst ? Math.max(0, Math.min(count, Math.floor(inst.length / HEAD_STRIDE))) : 0;
    this.headsDirty = true;
    this.map?.triggerRepaint();
  }

  setUniforms(s: Partial<FleetUniformState>): void {
    this.uni = { ...this.uni, ...s };
    this.map?.triggerRepaint();
  }

  setAltScale(k: number): void {
    this.altScale = Number.isFinite(k) && k > 0 ? k : 1;
  }

  /** harness/test seam */
  getCounts(): { full: number; thin: number; conn: number; heads: number } {
    const v = (b: Batch) => (b.verts ? b.verts.length / FLEET_STRIDE : 0);
    return { full: v(this.full), thin: v(this.thin), conn: v(this.conn), heads: this.headCount };
  }

  getRenderFailed(): boolean {
    return this.failStreak >= FleetReplayLayer.MAX_FAIL_STREAK;
  }

  /** frames that completed with at least one draw, and the current failure
   *  streak — proof the programs compiled AND drew (harness seam) */
  getDrawStats(): { drawnFrames: number; failStreak: number } {
    return { drawnFrames: this.drawnFrames, failStreak: this.failStreak };
  }

  /** screen px of a mercator point at REAL altitude, using last frame's
   *  matrix (FlightTrackLayer.projectToScreen — same formula) */
  projectToScreen(mx: number, my: number, altM: number, widthPx: number, heightPx: number): { x: number; y: number } | null {
    const m = this.lastMainMatrix;
    if (!m) return null;
    const z = Number.isNaN(altM) ? 0 : altM * this.altScale;
    const p: [number, number, number] = this.lastTransition > 0.999
      ? (mercatorToSphere(mx, my, z) as [number, number, number])
      : [mx, my, mercatorZFromAltitude(z, my)];
    const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    if (!(w > 0)) return null;
    const cx = (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w;
    const cy = (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w;
    if (cx < -1.2 || cx > 1.2 || cy < -1.2 || cy > 1.2) return null;
    return { x: ((cx + 1) / 2) * widthPx, y: ((1 - cy) / 2) * heightPx };
  }

  render(gl: AnyGl, args: CustomRenderMethodInput): void {
    if (this.failStreak >= FleetReplayLayer.MAX_FAIL_STREAK) return;
    if (!this.full.verts && !this.thin.verts && !this.conn.verts && !this.headCount) return;
    try {
      this.renderInner(gl as WebGL2RenderingContext, args);
      this.failStreak = 0;
    } catch (e) {
      this.failStreak++;
      this.dropGlObjects(gl);
      // eslint-disable-next-line no-console
      console.error(`FleetReplayLayer: render failure ${this.failStreak}/${FleetReplayLayer.MAX_FAIL_STREAK} (GL objects dropped; map continues):`, e);
    }
  }

  private renderInner(gl: WebGL2RenderingContext, args: CustomRenderMethodInput): void {
    const sd = args.shaderData;
    if (this.program == null || this.cachedVariant !== sd.variantName) {
      this.compile(gl, sd.vertexShaderPrelude, sd.define, sd.variantName);
      for (const b of [this.full, this.thin, this.conn]) b.dirty = b.verts != null;
      this.headsDirty = this.heads != null;
    }
    if (!this.program || !this.hProgram) return;
    const pd = args.defaultProjectionData;
    if (!this.lastMainMatrix) this.lastMainMatrix = new Float32Array(16);
    this.lastMainMatrix.set(pd.mainMatrix as ArrayLike<number>);
    this.lastTransition = pd.projectionTransition;

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const terrainOn = !!(this.map && (this.map as unknown as { getTerrain?: () => unknown }).getTerrain?.());
    if (terrainOn) gl.disable(gl.DEPTH_TEST); else gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(false);
    gl.depthRange(0, 1);
    gl.disable(gl.CULL_FACE);

    const W = gl.drawingBufferWidth || 1, H = gl.drawingBufferHeight || 1;
    // geometry batches
    gl.useProgram(this.program);
    this.bindProjection(gl, args, this.gProj);
    if (this.u.viewport) gl.uniform2f(this.u.viewport, W, H);
    if (this.u.now) gl.uniform1f(this.u.now, this.uni.nowRel);
    if (this.u.trail) gl.uniform1f(this.u.trail, this.uni.trailSec);
    const hlOn = this.uni.hlA >= 0 || this.uni.hlB >= 0;
    if (this.u.hlOn) gl.uniform1f(this.u.hlOn, hlOn ? 1 : 0);
    if (this.u.hlA) gl.uniform1f(this.u.hlA, this.uni.hlA);
    if (this.u.hlB) gl.uniform1f(this.u.hlB, this.uni.hlB);
    if (this.u.hlColor) gl.uniform4fv(this.u.hlColor, FLEET_WARN_RGBA);
    if (this.u.dim) gl.uniform1f(this.u.dim, FLEET_DIM);
    let drawn = 0;
    for (const b of [this.thin, this.full, this.conn]) {
      if (!b.verts || !b.indices) continue;
      if (!b.buf) b.buf = gl.createBuffer();
      if (!b.ibuf) b.ibuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b.buf);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b.ibuf);
      if (b.dirty) {
        gl.bufferData(gl.ARRAY_BUFFER, b.verts, gl.DYNAMIC_DRAW);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, b.indices, gl.DYNAMIC_DRAW);
        b.dirty = false;
      }
      const sB = FLEET_STRIDE * 4;
      const attr = (loc: number, size: number, off: number) => {
        if (loc < 0) return;
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, sB, off);
      };
      attr(this.a.pos, 3, 0);
      attr(this.a.other, 3, 12);
      attr(this.a.ext, 3, 24);
      attr(this.a.color, 4, 36);
      attr(this.a.t, 1, 52);
      attr(this.a.slot, 1, 56);
      gl.drawElements(gl.TRIANGLES, b.indices.length, gl.UNSIGNED_INT, 0);
      drawn++;
    }
    for (const k of ["pos", "other", "ext", "color", "t", "slot"]) if (this.a[k] >= 0) gl.disableVertexAttribArray(this.a[k]);

    // heads: ONE instanced draw
    if (this.headCount > 0 && this.heads) {
      gl.useProgram(this.hProgram);
      this.bindProjection(gl, args, this.hProj);
      if (this.hu.scaleClip) gl.uniform1f(this.hu.scaleClip, 2 / H);
      if (this.hu.aspect) gl.uniform1f(this.hu.aspect, W / H);
      if (this.hu.bearing) gl.uniform1f(this.hu.bearing, ((this.map?.getBearing() ?? 0) * Math.PI) / 180);
      if (!this.glyphBuf) {
        this.glyphBuf = gl.createBuffer();
        // u_scaleClip is per DEVICE pixel; size the glyph in CSS px (DPR ≤ 2,
        // the tier cap datamap applies to the canvas)
        const dpr = Math.min(2, Math.max(1, (globalThis as { devicePixelRatio?: number }).devicePixelRatio || 1));
        const g = headGlyph(FLEET_HEAD_PX * dpr);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.glyphBuf);
        gl.bufferData(gl.ARRAY_BUFFER, g, gl.STATIC_DRAW);
        this.glyphVerts = g.length / 2;
      }
      if (!this.headBuf) { this.headBuf = gl.createBuffer(); this.headCap = 0; this.headsDirty = true; }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.headBuf);
      if (this.headsDirty) {
        const floats = this.headCount * HEAD_STRIDE;
        if (floats > this.headCap) {
          // grow to the instance array's capacity (the controller reuses one
          // budget-sized array, so this reallocates at most a handful of times)
          gl.bufferData(gl.ARRAY_BUFFER, this.heads.byteLength, gl.DYNAMIC_DRAW);
          this.headCap = this.heads.length;
        }
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.heads, 0, floats);
        this.headsDirty = false;
      }
      const sB = HEAD_STRIDE * 4;
      if (this.ha.inst >= 0) {
        gl.enableVertexAttribArray(this.ha.inst);
        gl.vertexAttribPointer(this.ha.inst, 4, gl.FLOAT, false, sB, 0);
        gl.vertexAttribDivisor(this.ha.inst, 1);
      }
      if (this.ha.icolor >= 0) {
        gl.enableVertexAttribArray(this.ha.icolor);
        gl.vertexAttribPointer(this.ha.icolor, 4, gl.FLOAT, false, sB, 16);
        gl.vertexAttribDivisor(this.ha.icolor, 1);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.glyphBuf);
      if (this.ha.local >= 0) {
        gl.enableVertexAttribArray(this.ha.local);
        gl.vertexAttribPointer(this.ha.local, 2, gl.FLOAT, false, 0, 0);
        gl.vertexAttribDivisor(this.ha.local, 0);
      }
      gl.drawArraysInstanced(gl.TRIANGLES, 0, this.glyphVerts, this.headCount);
      drawn++;
      // shared GL state: divisors back to 0 (MapLibre's own draws follow)
      for (const k of ["inst", "icolor", "local"]) {
        if (this.ha[k] >= 0) {
          gl.vertexAttribDivisor(this.ha[k], 0);
          gl.disableVertexAttribArray(this.ha[k]);
        }
      }
    }
    setGauge("fleetReplay.draws", drawn);
    if (drawn > 0) this.drawnFrames++;
  }

  private bindProjection(gl: WebGL2RenderingContext, args: CustomRenderMethodInput, u: Record<string, WebGLUniformLocation | null>): void {
    const pd = args.defaultProjectionData;
    if (u.matrix) gl.uniformMatrix4fv(u.matrix, false, pd.mainMatrix);
    if (u.tile) gl.uniform4f(u.tile, pd.tileMercatorCoords[0], pd.tileMercatorCoords[1], pd.tileMercatorCoords[2], pd.tileMercatorCoords[3]);
    if (u.clip) gl.uniform4f(u.clip, pd.clippingPlane[0], pd.clippingPlane[1], pd.clippingPlane[2], pd.clippingPlane[3]);
    if (u.trans) gl.uniform1f(u.trans, pd.projectionTransition);
    if (u.fallback) gl.uniformMatrix4fv(u.fallback, false, pd.fallbackMatrix);
  }

  private compile(gl: WebGL2RenderingContext, prelude: string, define: string, variant: string): void {
    for (const p of [this.program, this.hProgram]) if (p) gl.deleteProgram(p);
    this.program = this.hProgram = null;
    const mk = (type: number, src: string): WebGLShader => {
      const sh = gl.createShader(type);
      if (!sh) throw new Error("FleetReplayLayer: createShader failed");
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error("FleetReplayLayer: shader compile failed: " + log);
      }
      return sh;
    };
    const link = (vs: string, fs: string): WebGLProgram => {
      const v = mk(gl.VERTEX_SHADER, vs), f = mk(gl.FRAGMENT_SHADER, fs);
      const p = gl.createProgram();
      if (!p) throw new Error("FleetReplayLayer: createProgram failed");
      gl.attachShader(p, v); gl.attachShader(p, f); gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(p);
        gl.deleteProgram(p);
        throw new Error("FleetReplayLayer: program link failed: " + log);
      }
      gl.deleteShader(v); gl.deleteShader(f);
      return p;
    };
    const proj = (p: WebGLProgram) => ({
      matrix: gl.getUniformLocation(p, "u_projection_matrix"),
      tile: gl.getUniformLocation(p, "u_projection_tile_mercator_coords"),
      clip: gl.getUniformLocation(p, "u_projection_clipping_plane"),
      trans: gl.getUniformLocation(p, "u_projection_transition"),
      fallback: gl.getUniformLocation(p, "u_projection_fallback_matrix"),
    });
    const gp = link(FLEET_VERT_SRC(prelude, define), FLEET_FRAG_SRC);
    this.program = gp;
    this.a = {
      pos: gl.getAttribLocation(gp, "a_pos"), other: gl.getAttribLocation(gp, "a_other"),
      ext: gl.getAttribLocation(gp, "a_ext"), color: gl.getAttribLocation(gp, "a_color"),
      t: gl.getAttribLocation(gp, "a_t"), slot: gl.getAttribLocation(gp, "a_slot"),
    };
    this.u = {
      viewport: gl.getUniformLocation(gp, "u_viewport"), now: gl.getUniformLocation(gp, "u_now"),
      trail: gl.getUniformLocation(gp, "u_trail"), hlOn: gl.getUniformLocation(gp, "u_hlOn"),
      hlA: gl.getUniformLocation(gp, "u_hlA"), hlB: gl.getUniformLocation(gp, "u_hlB"),
      hlColor: gl.getUniformLocation(gp, "u_hlColor"), dim: gl.getUniformLocation(gp, "u_dim"),
    };
    this.gProj = proj(gp);
    const hp = link(FLEET_HEAD_VERT_SRC(prelude, define), FLEET_HEAD_FRAG_SRC);
    this.hProgram = hp;
    this.ha = {
      local: gl.getAttribLocation(hp, "a_local"), inst: gl.getAttribLocation(hp, "a_inst"),
      icolor: gl.getAttribLocation(hp, "a_icolor"),
    };
    this.hu = {
      scaleClip: gl.getUniformLocation(hp, "u_scaleClip"), aspect: gl.getUniformLocation(hp, "u_aspect"),
      bearing: gl.getUniformLocation(hp, "u_bearing"),
    };
    this.hProj = proj(hp);
    this.cachedVariant = variant;
  }
}
