// Orbit arc layer — EARTH TWIN O6-1, upgraded 2026-07-16 (round 10b:
// human report "the line of objects need to be more clear or visible").
// A MapLibre CustomLayerInterface (satLayer/modelLayer/airLayer are the
// template: same projection prelude, variant caching, failure latch,
// GLOBE-guarded far-side handling) that draws orbit tracks as EXTRUDED
// SCREEN-SPACE RIBBONS at TRUE altitude via projectTileFor3D.
//
// RIBBONS, not GL_LINES (the original implementation): GL line width is
// clamped to 1px on nearly every GPU/driver, so orbit arcs and aircraft
// 3D trails rendered as invisible hairlines. Each surviving segment now
// becomes a QUAD (two triangles): every vertex carries BOTH segment
// endpoints plus a side/end selector, the vertex shader projects both
// endpoints, derives the screen-space normal, and extrudes by
// (widthPx * 0.5) * w / viewport — CONSTANT PIXEL WIDTH at every zoom and
// tilt (the anchor.w scaling pattern airLayer/modelLayer use). Default
// width ~3px; each arc may override via `widthPx`. All consumers gain the
// upgrade for free: satellite orbit arcs, group orbits, aircraft 3D
// trails, curtain ribs.
//
// 2026-07-20: the ALTITUDE CURTAIN WALL draw mode (setWalls, added
// 2026-07-16 for the aircraft trail) was REMOVED with the flight-track
// replacement (design_handoff_flight_track_3d): the aircraft curtain now
// lives in lib/air/flightTrackLayer.ts with the handoff's contract (40m
// below-terrain drape, 34% alpha, teal→violet ramp, explicit depth-test/
// no-depth-write state). This layer serves satellite orbit RIBBONS only.
//
// Line-specific choices (unchanged semantics from the GL_LINES version):
// - Segments, not strips: a quad is emitted only between two consecutive
//   GOOD samples that don't jump the antimeridian (|dx| > 0.5 in mercator)
//   — ARC_GAP sentinels break the ribbon honestly instead of bridging it;
//   no quad ever spans a gap.
// - Far-side handling is a FRAGMENT discard (v_cull varying) rather than
//   the point layers' vertex snap-out: snapping one vertex of a ribbon
//   stretches the quad across the screen; discarding fades the arc at
//   the horizon cleanly. Vertices are NEVER snapped or dropped for culling.
// - Edge rim: the fragment fades alpha 1 (core) → 0.55 (edges) along the
//   across-ribbon coordinate, so the ribbon reads with a subtle darker rim
//   on bright imagery — one pass, no extra geometry.
// - COST: zero when no arcs are set; static buffers per setArcs call
//   (vertex + index, no per-frame allocation); no self-repaint (arcs move
//   only when re-sampled by the caller).
// - BUFFER GROWTH vs GL_LINES: a segment was 2 verts × 7 floats = 56 B;
//   it is now 4 verts × 13 floats = 208 B + 6 × 4 B indices = 232 B —
//   ~4.1x per segment (4x vertices, wider stride, plus the index buffer).
//   Index type is UNSIGNED_INT (WebGL2 core — MapLibre v5 requires GL2;
//   on a GL1 context OES_element_index_uint is requested, and its absence
//   trips the existing failure latch instead of crashing the map).

import type {
  CustomLayerInterface,
  CustomRenderMethodInput,
  Map as MapLibreMap,
} from 'maplibre-gl';
import { ARC_GAP } from './orbitArc.js';
import { VT_PROJ_ELEV_GLSL } from '../glElev.js';

type AnyGl = WebGLRenderingContext | WebGL2RenderingContext;

export interface Arc {
  /** interleaved [mercX, mercY, altMeters] triplets (orbitArc.sampleOrbitArc). */
  pts: Float32Array;
  color: [number, number, number, number];
  /** ribbon width in px for THIS arc; absent/0 = the layer default (~3px). */
  widthPx?: number;
}

/** default ribbon width in pixels (constant on screen at every zoom/tilt). */
export const ARC_DEFAULT_WIDTH_PX = 3;

/** floats per ribbon vertex (layout pinned by arcLayer.test.ts):
 *  [0..2] a_pos   — THIS endpoint: mercX, mercY, altMeters
 *  [3..5] a_other — the segment's OTHER endpoint (same triplet)
 *  [6]    side    — across-ribbon extrusion side: -1 | +1
 *  [7]    dirSign — +1 when this vertex sits at the segment START, -1 at
 *                   the END (keeps the screen direction consistent A→B
 *                   across the whole quad)
 *  [8]    widthPx — per-arc override; 0 = use the u_width uniform
 *  [9..12] r,g,b,a
 */
export const ARC_VERT_STRIDE = 13;

/** vertices per surviving segment (one quad). */
export const ARC_VERTS_PER_SEG = 4;

/** indices per surviving segment (two triangles). */
export const ARC_INDICES_PER_SEG = 6;

/** Pure: arcs → packed ribbon vertex array, 4 verts per surviving segment
 *  (exported for tests). Gap/antimeridian split rules are IDENTICAL to the
 *  GL_LINES version: a quad is only emitted between two consecutive good
 *  samples on the same side of the antimeridian. */
export function buildArcVertices(arcs: Arc[]): Float32Array {
  const out: number[] = [];
  for (const arc of arcs) {
    // 0 = "no override" sentinel → the shader falls back to u_width
    const w = arc.widthPx && arc.widthPx > 0 ? arc.widthPx : 0;
    const [r, g, bl, al] = arc.color;
    const n = Math.floor(arc.pts.length / 3);
    for (let i = 0; i + 1 < n; i++) {
      const a = i * 3, b = (i + 1) * 3;
      if (arc.pts[a + 2] === ARC_GAP || arc.pts[b + 2] === ARC_GAP) continue; // honest gap
      if (Math.abs(arc.pts[a] - arc.pts[b]) > 0.5) continue; // antimeridian jump
      const ax = arc.pts[a], ay = arc.pts[a + 1], az = arc.pts[a + 2];
      const bx = arc.pts[b], by = arc.pts[b + 1], bz = arc.pts[b + 2];
      // quad: A(-1), A(+1), B(-1), B(+1) — sides pair up across the segment
      out.push(
        ax, ay, az, bx, by, bz, -1, +1, w, r, g, bl, al,
        ax, ay, az, bx, by, bz, +1, +1, w, r, g, bl, al,
        bx, by, bz, ax, ay, az, -1, -1, w, r, g, bl, al,
        bx, by, bz, ax, ay, az, +1, -1, w, r, g, bl, al,
      );
    }
  }
  return new Float32Array(out);
}

/** Pure: index buffer for `segCount` quads — two triangles each, sharing the
 *  quad's 4 vertices (exported for tests). No index ever crosses a quad
 *  boundary, so no triangle can span a gap by construction. */
export function buildArcIndices(segCount: number): Uint32Array {
  const idx = new Uint32Array(segCount * ARC_INDICES_PER_SEG);
  for (let s = 0; s < segCount; s++) {
    const v = s * ARC_VERTS_PER_SEG;
    const o = s * ARC_INDICES_PER_SEG;
    idx[o] = v;
    idx[o + 1] = v + 1;
    idx[o + 2] = v + 2;
    idx[o + 3] = v + 2;
    idx[o + 4] = v + 1;
    idx[o + 5] = v + 3;
  }
  return idx;
}

/** Exported for arcLayer.test.ts / orbitArc.test.ts — pins the
 *  projection/cull/extrusion contract. */
export const ARC_VERT_SRC = (prelude: string, define: string): string => `#version 300 es
${prelude}
${define}
${VT_PROJ_ELEV_GLSL}
in vec3 a_pos;    // THIS endpoint: mercX, mercY, altMeters
in vec3 a_other;  // the segment's OTHER endpoint
in vec3 a_ext;    // x: side (-1|+1), y: dirSign (+1 start / -1 end), z: widthPx (0 = u_width)
in vec4 a_color;
uniform float u_width;    // default ribbon width in px
uniform vec2 u_viewport;  // drawing buffer size in px
out vec4 v_color;
out float v_cull;
out float v_edge;
void main() {
  v_cull = 0.0;
#ifdef GLOBE
  // far-side test — identical formula to satLayer/modelLayer/airLayer
  // (0.998001 = OCCLUSION_RADIUS²) but flagged to the FRAGMENT stage:
  // ribbons must fade at the horizon, not stretch to a snapped vertex.
  // Runs through the whole globe↔mercator transition, matching satLayer
  // (2026-07-31): a camera parked mid-blend must not see far-side ghosts.
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
  // screen-space segment direction, consistent start→end for the whole
  // quad via a_ext.y; abs(w) guards a behind-camera endpoint from
  // flipping the ribbon inside out.
  vec2 ndcSelf = self.xy / max(abs(self.w), 1e-9);
  vec2 ndcOther = other.xy / max(abs(other.w), 1e-9);
  vec2 dirPx = (ndcOther - ndcSelf) * a_ext.y * u_viewport;
  float len = length(dirPx);
  dirPx = len < 1e-6 ? vec2(1.0, 0.0) : dirPx / len;
  vec2 normalPx = vec2(-dirPx.y, dirPx.x) * a_ext.x;
  float widthPx = a_ext.z > 0.0 ? a_ext.z : u_width;
  // extrude half the width: px → NDC is 2/viewport, then * self.w keeps
  // the ribbon a CONSTANT PIXEL WIDTH at every zoom/tilt (the anchor.w
  // pattern airLayer/modelLayer use).
  vec2 offs = normalPx * (widthPx * 0.5) * 2.0 / u_viewport;
  gl_Position = self + vec4(offs * self.w, 0.0, 0.0);
  v_color = a_color;
  v_edge = a_ext.x;
}`;

/** Exported for arcLayer.test.ts — pins the fragment discard + edge rim. */
export const ARC_FRAG_SRC = `#version 300 es
precision mediump float;
in vec4 v_color;
in float v_cull;
in float v_edge;
out vec4 o;
void main() {
  if (v_cull > 0.01) discard;
  o = v_color;
  // subtle rim: alpha 1 at the ribbon core → 0.55 at the edges reads as a
  // darker rim on bright imagery (one pass, honest color underneath).
  o.a *= mix(1.0, 0.55, abs(v_edge));
}`;

export class ArcLayer implements CustomLayerInterface {
  readonly id: string;
  readonly type = 'custom' as const;
  readonly renderingMode = '2d' as const;

  private map: MapLibreMap | null = null;
  private program: WebGLProgram | null = null;
  private cachedVariant: string | null = null;
  private buffer: WebGLBuffer | null = null;
  private indexBuffer: WebGLBuffer | null = null;
  private aPos = -1;
  private aOther = -1;
  private aExt = -1;
  private aColor = -1;
  private uWidth: WebGLUniformLocation | null = null;
  private uViewport: WebGLUniformLocation | null = null;
  private uProjMatrix: WebGLUniformLocation | null = null;
  private uProjTile: WebGLUniformLocation | null = null;
  private uProjClip: WebGLUniformLocation | null = null;
  private uProjTrans: WebGLUniformLocation | null = null;
  private uProjFallback: WebGLUniformLocation | null = null;

  private verts: Float32Array | null = null;
  private indices: Uint32Array | null = null;
  private dirty = false;
  // SELF-HEALING failures (stability audit 2026-07-20; the flightTrackLayer
  // pattern — see satLayer for the full rationale): drop GL objects on a
  // throw, count a streak, re-arm on fresh arcs.
  private failStreak = 0;
  private static readonly MAX_FAIL_STREAK = 5;
  private defaultWidthPx: number;

  constructor(opts: { id?: string; widthPx?: number } = {}) {
    this.id = opts.id ?? 'orbital-arcs';
    this.defaultWidthPx = opts.widthPx ?? ARC_DEFAULT_WIDTH_PX;
  }

  /** Cached GL context so `dispose()` needs no arguments (Law IV). */
  private glRef: AnyGl | null = null;

  onAdd(map: MapLibreMap, gl: AnyGl): void {
    this.map = map;
    // Law IV: cache the context so dispose() can free GL objects without the
    // caller holding a `gl`. MapLibre only hands one to onAdd/onRemove/render,
    // so a no-arg teardown is impossible without this.
    this.glRef = gl;
  }

  onRemove(_map: MapLibreMap, gl: AnyGl): void {
    if (this.program) gl.deleteProgram(this.program);
    if (this.buffer) gl.deleteBuffer(this.buffer);
    if (this.indexBuffer) gl.deleteBuffer(this.indexBuffer);
    this.program = null;
    this.buffer = null;
    this.indexBuffer = null;
    this.cachedVariant = null;
    this.dirty = this.verts != null;
    this.map = null;
  }

  /**
   * Law IV explicit teardown. Delegates to the layer's own onRemove path
   * using the cached context, so a view unmount (or the harness's
   * toggle-on/toggle-off leak assertion) can free everything without
   * holding a `gl`. Idempotent — onRemove nulls every handle, and GL
   * deletes on a lost context are no-ops.
   */
  dispose(): void {
    const gl = this.glRef;
    this.glRef = null;
    if (gl) this.onRemove(null as unknown as MapLibreMap, gl);
    this.map = null;
  }

  /** Replace the displayed arcs (null/[] clears — layer costs nothing). */
  setArcs(arcs: Arc[] | null): void {
    const v = arcs && arcs.length ? buildArcVertices(arcs) : null;
    this.verts = v && v.length ? v : null;
    this.indices = this.verts
      ? buildArcIndices(this.verts.length / ARC_VERT_STRIDE / ARC_VERTS_PER_SEG)
      : null;
    this.dirty = this.verts != null;
    this.failStreak = 0; // fresh data re-arms rendering after transient GL failures
    this.map?.triggerRepaint();
  }

  getVertexCount(): number {
    return this.verts ? this.verts.length / ARC_VERT_STRIDE : 0;
  }

  getRenderFailed(): boolean {
    return this.failStreak >= ArcLayer.MAX_FAIL_STREAK;
  }

  /** Forget every GL handle (deletes on a lost context are safe no-ops) so
   *  the next render rebuilds from scratch on the recovered context. */
  private dropGlObjects(gl?: AnyGl): void {
    try {
      if (gl) {
        if (this.program) gl.deleteProgram(this.program);
        if (this.buffer) gl.deleteBuffer(this.buffer);
        if (this.indexBuffer) gl.deleteBuffer(this.indexBuffer);
      }
    } catch { /* context gone — handles are already invalid */ }
    this.program = null;
    this.buffer = null;
    this.indexBuffer = null;
    this.cachedVariant = null;
    this.dirty = this.verts != null;
  }

  render(gl: AnyGl, args: CustomRenderMethodInput): void {
    if (this.failStreak >= ArcLayer.MAX_FAIL_STREAK) return;
    if (!this.verts || this.verts.length === 0) return; // nothing set → zero cost
    try {
      this.renderInner(gl, args);
      this.failStreak = 0; // a clean frame ends any failure streak
    } catch (e) {
      this.failStreak++;
      this.dropGlObjects(gl); // stale handles from a lost context throw forever
      // eslint-disable-next-line no-console
      console.error(
        `ArcLayer: render failure ${this.failStreak}/${ArcLayer.MAX_FAIL_STREAK} ` +
        '(GL objects dropped for a clean retry; map continues):', e);
    }
  }

  private renderInner(gl: AnyGl, args: CustomRenderMethodInput): void {
    const sd = args.shaderData;
    if (this.program == null || this.cachedVariant !== sd.variantName) {
      this.compile(gl, sd.vertexShaderPrelude, sd.define, sd.variantName);
      this.dirty = true;
    }
    if (this.program == null || !this.indices) return;
    gl.useProgram(this.program);

    const pd = args.defaultProjectionData;
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

    if (this.uWidth) gl.uniform1f(this.uWidth, this.defaultWidthPx);
    if (this.uViewport) {
      gl.uniform2f(this.uViewport, gl.drawingBufferWidth || 1, gl.drawingBufferHeight || 1);
    }

    if (!this.buffer) this.buffer = gl.createBuffer();
    if (!this.indexBuffer) this.indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    if (this.dirty && this.verts) {
      gl.bufferData(gl.ARRAY_BUFFER, this.verts, gl.STATIC_DRAW);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.indices, gl.STATIC_DRAW);
      this.dirty = false;
    }
    const strideB = ARC_VERT_STRIDE * 4;
    gl.enableVertexAttribArray(this.aPos);
    gl.vertexAttribPointer(this.aPos, 3, gl.FLOAT, false, strideB, 0);
    gl.enableVertexAttribArray(this.aOther);
    gl.vertexAttribPointer(this.aOther, 3, gl.FLOAT, false, strideB, 12);
    gl.enableVertexAttribArray(this.aExt);
    gl.vertexAttribPointer(this.aExt, 3, gl.FLOAT, false, strideB, 24);
    gl.enableVertexAttribArray(this.aColor);
    gl.vertexAttribPointer(this.aColor, 4, gl.FLOAT, false, strideB, 36);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    // screen-space extrusion has no stable winding — never let a stale
    // CULL_FACE from a prior layer eat half the triangles
    gl.disable(gl.CULL_FACE);
    if (!(gl instanceof WebGL2RenderingContext)) {
      // MapLibre v5 guarantees WebGL2; on a GL1 context UNSIGNED_INT
      // indices need this extension — absence trips the failure latch.
      gl.getExtension('OES_element_index_uint');
    }
    gl.drawElements(gl.TRIANGLES, this.indices.length, gl.UNSIGNED_INT, 0);

    gl.disableVertexAttribArray(this.aPos);
    gl.disableVertexAttribArray(this.aOther);
    gl.disableVertexAttribArray(this.aExt);
    gl.disableVertexAttribArray(this.aColor);
  }

  private compile(gl: AnyGl, prelude: string, define: string, variant: string): void {
    if (this.program) {
      gl.deleteProgram(this.program);
      this.program = null;
    }
    const mk = (type: number, src: string): WebGLShader => {
      const sh = gl.createShader(type);
      if (!sh) throw new Error('ArcLayer: createShader failed');
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error('ArcLayer: shader compile failed: ' + log);
      }
      return sh;
    };
    const vs = mk(gl.VERTEX_SHADER, ARC_VERT_SRC(prelude, define));
    const fs = mk(gl.FRAGMENT_SHADER, ARC_FRAG_SRC);
    const p = gl.createProgram();
    if (!p) throw new Error('ArcLayer: createProgram failed');
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error('ArcLayer: program link failed: ' + log);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    this.program = p;
    this.cachedVariant = variant;
    this.aPos = gl.getAttribLocation(p, 'a_pos');
    this.aOther = gl.getAttribLocation(p, 'a_other');
    this.aExt = gl.getAttribLocation(p, 'a_ext');
    this.aColor = gl.getAttribLocation(p, 'a_color');
    this.uWidth = gl.getUniformLocation(p, 'u_width');
    this.uViewport = gl.getUniformLocation(p, 'u_viewport');
    this.uProjMatrix = gl.getUniformLocation(p, 'u_projection_matrix');
    this.uProjTile = gl.getUniformLocation(p, 'u_projection_tile_mercator_coords');
    this.uProjClip = gl.getUniformLocation(p, 'u_projection_clipping_plane');
    this.uProjTrans = gl.getUniformLocation(p, 'u_projection_transition');
    this.uProjFallback = gl.getUniformLocation(p, 'u_projection_fallback_matrix');
  }
}

// ── Law IV budget declaration ───────────────────────────────────────────────
// ARC_VERT_STRIDE(13) floats x ARC_VERTS_PER_SEG(4) per segment. Orbit arcs
// are drawn for selected/followed objects, not the whole catalog, so the
// working set is single digits; 256 is a runaway guard well above any real
// selection.
export const maxFeatures = 256;
export const vramBudget = 2; // MB
