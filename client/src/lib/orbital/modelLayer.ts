// On-map 3D spacecraft layer — EARTH TWIN / ORBITAL O5-2b (human directive:
// the 3D rendering shows ON THE WORLD MAP at the satellite's position, not
// in a side viewer). A MapLibre CustomLayerInterface (satLayer.ts is the
// template: same projection prelude, same variant caching, same failure
// latch, same far-side cull) that draws ONE class-representative mesh
// (lib/orbital/model3d) anchored at the FOLLOWED satellite's live
// SGP4-projected position — lit, slowly tumbling, riding the orbit.
//
// SCALE HONESTY: a real satellite is meters long — invisible at true scale
// from any map camera — so the form renders at a CONSTANT SCREEN SIZE
// (clip-space offsets scaled by anchor.w hold ~pixel size at any zoom),
// the standard focused-object treatment. The detail card carries the
// "representative form — derived, not imagery of this unit" caption.
//
// COST: draws ONLY while a satellite is followed (no anchor → render is a
// no-op and requests no repaint). While drawing it self-requests repaint
// for the tumble — bounded by the follow mode itself, which the user exits
// by dragging (and which LOD exits when the layer hides).

import type {
  CustomLayerInterface,
  CustomRenderMethodInput,
  Map as MapLibreMap,
} from 'maplibre-gl';
import { buildFormMesh, rotationMat3, type FormKind, type Mesh } from './model3d.js';
import { VT_PROJ_ELEV_GLSL } from '../glElev.js';

type AnyGl = WebGLRenderingContext | WebGL2RenderingContext;

export interface ModelAnchor {
  mercX: number;
  mercY: number;
  altMeters: number;
}

/** Exported for modelLayer.test.ts — pins the projection/cull contract the
 *  same way satLayer.test.ts pins its shader (GLSL can't import TS). */
export const MODEL_VERT_SRC = (prelude: string, define: string): string => `#version 300 es
${prelude}
${define}
${VT_PROJ_ELEV_GLSL}
in vec3 a_pos;             // model space, ~[-1.2, 1.2]
in vec3 a_normal;
in vec3 a_color;
uniform vec3 u_anchor;     // mercX (0..1), mercY (0..1), altMeters
uniform mat3 u_rot;        // tumble rotation
uniform float u_scaleClip; // clip units per model unit at w=1 (pixel-constant via *w)
uniform float u_aspect;    // drawingBufferWidth / drawingBufferHeight
out vec3 v_color;
out vec3 v_normal;
void main() {
#ifdef GLOBE
  // FAR-SIDE CULL on the anchor — identical test to satLayer.ts (mirrors
  // ./occlusion; 0.998001 = OCCLUSION_RADIUS²): a model whose satellite is
  // behind the earth must vanish with it. Runs through the whole
  // globe↔mercator transition, matching satLayer (2026-07-31).
  if (u_projection_transition > 0.0 && u_projection_clipping_plane.w < 0.0) {
    vec3 satPos = projectToSphere(u_anchor.xy) * (1.0 + u_anchor.z / GLOBE_RADIUS);
    vec3 cam = u_projection_clipping_plane.xyz * (-1.0 / u_projection_clipping_plane.w);
    vec3 v = satPos - cam;
    float t = -dot(cam, v) / dot(v, v);
    if (t > 0.0 && t < 1.0) {
      vec3 closest = cam + t * v;
      if (dot(closest, closest) < 0.998001) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_color = vec3(0.0);
        v_normal = vec3(0.0, 0.0, 1.0);
        return;
      }
    }
  }
#endif
  vec4 anchor = projectTileFor3D(u_anchor.xy, vtProjElev(u_anchor.z, u_anchor.y));
  vec3 p = u_rot * a_pos;
  // constant-pixel-size: clip-space offset scaled by w cancels perspective
  gl_Position = anchor + vec4(p.x * u_scaleClip / u_aspect, p.y * u_scaleClip, 0.0, 0.0) * anchor.w;
  // tiny self-depth bias so the mesh's own triangles sort against each other
  gl_Position.z -= p.z * 0.0002 * anchor.w;
  v_color = a_color;
  v_normal = u_rot * a_normal;
}`;

const MODEL_FRAG_SRC = `#version 300 es
precision mediump float;
in vec3 v_color;
in vec3 v_normal;
out vec4 o;
void main() {
  vec3 light = normalize(vec3(0.45, 0.65, 0.62)); // fixed key light
  float diff = max(dot(normalize(v_normal), light), 0.0);
  o = vec4(v_color * (0.35 + 0.75 * diff), 1.0);
}`;

/** Base on-screen model height in pixels (the focused-object size). */
export const MODEL_PIXELS = 72;
/** Zoom-to-inspect ceiling — the model grows past the focus zoom (2^Δzoom)
 *  up to this, so "get really close" actually gets close. */
export const MODEL_MAX_PIXELS = 1600;
/** On-screen size of an O6-2 auto-3D mini (browse-zoom form, not focused). */
export const MINI_PIXELS = 34;
/** Focus ring RADIUS in px — drawn at the anchor's rendered altitude. */
export const RING_PIXELS = 17;
/** Model-space half-extent the meshes are built to (model3d contract). */
export const MODEL_HALF_EXTENT = 1.2;

/** One O6-2 mini: a class form at a live buffer position (see miniSelect). */
export interface MiniModel {
  mercX: number;
  mercY: number;
  altMeters: number;
  form: FormKind;
  /** stable per-object index → deterministic static orientation. */
  index: number;
}

export class SatModelLayer implements CustomLayerInterface {
  readonly id: string;
  readonly type = 'custom' as const;
  readonly renderingMode = '3d' as const; // depth buffer for mesh self-occlusion

  private map: MapLibreMap | null = null;
  private program: WebGLProgram | null = null;
  private cachedVariant: string | null = null;
  private bufPos: WebGLBuffer | null = null;
  private bufNor: WebGLBuffer | null = null;
  private bufCol: WebGLBuffer | null = null;
  private aPos = -1;
  private aNor = -1;
  private aCol = -1;
  private uAnchor: WebGLUniformLocation | null = null;
  private uRot: WebGLUniformLocation | null = null;
  private uScale: WebGLUniformLocation | null = null;
  private uAspect: WebGLUniformLocation | null = null;
  private uProjMatrix: WebGLUniformLocation | null = null;
  private uProjTile: WebGLUniformLocation | null = null;
  private uProjClip: WebGLUniformLocation | null = null;
  private uProjTrans: WebGLUniformLocation | null = null;
  private uProjFallback: WebGLUniformLocation | null = null;

  private mesh: Mesh | null = null;
  private realMesh: Mesh | null = null;
  private meshDirty = false;
  private form: FormKind | null = null;
  private anchor: ModelAnchor | null = null;
  // SELF-HEALING failures (stability audit 2026-07-20; the flightTrackLayer
  // pattern — see satLayer for the full rationale): drop GL objects on a
  // throw, count a streak, re-arm on fresh anchors/minis from the worker.
  private failStreak = 0;
  private static readonly MAX_FAIL_STREAK = 5;

  // O6-2 minis: class forms drawn at browse zoom without a click. Static
  // per-form GPU buffers (uploaded once), one small uniform-anchored draw
  // per mini, NO self-repaint (minis move only when the worker tick
  // replaces them — static orientation, no tumble animation).
  private minis: MiniModel[] = [];
  private formBuffers = new Map<FormKind, { pos: WebGLBuffer; nor: WebGLBuffer; col: WebGLBuffer; count: number }>();
  // O6 ring fix: the focus ring is drawn HERE, at the anchor's rendered
  // (altitude-projected) position — the old geojson circle sat at the
  // GROUND point, so it visibly detached from the satellite whenever the
  // camera tilted or panned away (parallax between nadir and object).
  private ringBuffer: WebGLBuffer | null = null;

  constructor(opts: { id?: string } = {}) {
    this.id = opts.id ?? 'orbital-sat-model';
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
    for (const b of [this.bufPos, this.bufNor, this.bufCol]) if (b) gl.deleteBuffer(b);
    for (const fb of Array.from(this.formBuffers.values())) {
      gl.deleteBuffer(fb.pos); gl.deleteBuffer(fb.nor); gl.deleteBuffer(fb.col);
    }
    this.formBuffers.clear();
    if (this.ringBuffer) gl.deleteBuffer(this.ringBuffer);
    this.ringBuffer = null;
    this.program = null;
    this.bufPos = this.bufNor = this.bufCol = null;
    this.cachedVariant = null;
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

  /** O6-2: replace the auto-3D minis (empty/null clears). */
  setMinis(minis: MiniModel[] | null): void {
    this.minis = minis ?? [];
    this.failStreak = 0; // fresh data re-arms rendering after transient GL failures
    this.map?.triggerRepaint();
  }

  getMiniCount(): number {
    return this.minis.length;
  }

  /** The form to draw at focus, or null = draw nothing (unknown class stays
   *  an honest ring-only follow — never a guessed spacecraft). */
  setForm(form: FormKind | null): void {
    if (form === this.form) return;
    this.form = form;
    this.mesh = form ? buildFormMesh(form) : null;
    this.meshDirty = true;
    this.failStreak = 0; // new focus target re-arms after transient GL failures
    this.map?.triggerRepaint();
  }

  getForm(): FormKind | null {
    return this.form;
  }

  /** O5-3b: a REAL model of the followed unit (decoded .vtm from a verified
   *  public asset — see lib/orbital/realMesh). Takes precedence over the
   *  class-representative form while set; null falls back to the form.
   *  Callers clear this when the follow target changes so one satellite's
   *  real model can never appear on another. */
  setRealMesh(mesh: Mesh | null): void {
    if (mesh === this.realMesh) return;
    this.realMesh = mesh;
    this.meshDirty = true;
    this.map?.triggerRepaint();
  }

  getRealMesh(): Mesh | null {
    return this.realMesh;
  }

  /** The mesh render() actually draws: real model first, then the form. */
  getActiveMesh(): Mesh | null {
    return this.realMesh ?? this.mesh;
  }

  /** The followed satellite's live position (null = not following).
   *  NOTE: called per FRAME by the smooth-follow loop — deliberately does
   *  NOT reset failStreak (a per-frame re-arm would turn a persistent GL
   *  failure into an endless throw/recompile loop; setMinis and setForm,
   *  on tick/user cadence, are the re-arm points). */
  setAnchor(a: ModelAnchor | null): void {
    this.anchor = a;
    this.map?.triggerRepaint();
  }

  getAnchor(): ModelAnchor | null {
    return this.anchor;
  }

  getRenderFailed(): boolean {
    return this.failStreak >= SatModelLayer.MAX_FAIL_STREAK;
  }

  /** Forget every GL handle (deletes on a lost context are safe no-ops) so
   *  the next render rebuilds from scratch on the recovered context. */
  private dropGlObjects(gl?: AnyGl): void {
    try {
      if (gl) {
        if (this.program) gl.deleteProgram(this.program);
        for (const b of [this.bufPos, this.bufNor, this.bufCol]) if (b) gl.deleteBuffer(b);
        for (const fb of Array.from(this.formBuffers.values())) {
          gl.deleteBuffer(fb.pos); gl.deleteBuffer(fb.nor); gl.deleteBuffer(fb.col);
        }
        if (this.ringBuffer) gl.deleteBuffer(this.ringBuffer);
      }
    } catch { /* context gone — handles are already invalid */ }
    this.program = null;
    this.cachedVariant = null;
    this.bufPos = this.bufNor = this.bufCol = null;
    this.formBuffers.clear();
    this.ringBuffer = null;
    this.meshDirty = this.getActiveMesh() != null;
  }

  render(gl: AnyGl, args: CustomRenderMethodInput): void {
    if (this.failStreak >= SatModelLayer.MAX_FAIL_STREAK) return;
    const focused = !!(this.anchor && this.getActiveMesh());
    // an anchor with NO mesh (unknown class) still draws the focus RING
    if (!this.anchor && this.minis.length === 0) return; // nothing to draw → zero cost, no repaint
    try {
      this.renderInner(gl, args, focused);
      this.failStreak = 0; // a clean frame ends any failure streak
      // NO self-repaint: the attitude is camera-driven (gestures repaint the
      // map themselves) and position updates arrive via setAnchor each tick.
    } catch (e) {
      this.failStreak++;
      this.dropGlObjects(gl); // stale handles from a lost context throw forever
      // eslint-disable-next-line no-console
      console.error(
        `SatModelLayer: render failure ${this.failStreak}/${SatModelLayer.MAX_FAIL_STREAK} ` +
        '(GL objects dropped for a clean retry; map continues):', e);
    }
  }

  private renderInner(gl: AnyGl, args: CustomRenderMethodInput, focused: boolean): void {
    const sd = args.shaderData;
    if (this.program == null || this.cachedVariant !== sd.variantName) {
      this.compile(gl, sd.vertexShaderPrelude, sd.define, sd.variantName);
      this.meshDirty = true;
    }
    if (this.program == null) return;
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

    const h = gl.drawingBufferHeight || 1;
    const w = gl.drawingBufferWidth || 1;
    if (this.uAspect) gl.uniform1f(this.uAspect, w / h);

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.disable(gl.BLEND); // opaque meshes

    const enableAttrs = (loc: number) => {
      if (loc < 0) return;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    };

    // ── the FOCUSED model (dynamic mesh: real > form) — CAMERA-DRIVEN
    // attitude (human directive: "stop the rotating … it's like YOU'RE
    // doing the rotating"): the model holds still in space; rotating/
    // pitching the MAP views it from a different side. No animation, no
    // self-repaint — the map repaints during gestures anyway. ──
    const mesh = this.getActiveMesh();
    const anchor = this.anchor;
    if (focused && mesh && anchor) {
      if (this.uAnchor) gl.uniform3f(this.uAnchor, anchor.mercX, anchor.mercY, anchor.altMeters);
      const bearingRad = ((this.map?.getBearing() ?? 0) * Math.PI) / 180;
      const pitchRad = ((this.map?.getPitch() ?? 0) * Math.PI) / 180;
      // base 3/4 attitude + the camera's own bearing/pitch — drag-rotate
      // orbits the craft, tilt looks over/under it
      if (this.uRot) gl.uniformMatrix3fv(this.uRot, false, rotationMat3(0.6 - bearingRad, 0.3 + pitchRad * 0.7));
      // zoom-to-inspect (human directive: "get really close and see it"):
      // past the focus zoom the model grows with the camera, up to a cap
      const zoom = this.map?.getZoom() ?? 0;
      const px = Math.min(MODEL_MAX_PIXELS, Math.max(MODEL_PIXELS, MODEL_PIXELS * Math.pow(2, zoom - 4.3)));
      if (this.uScale) gl.uniform1f(this.uScale, ((px / (2 * MODEL_HALF_EXTENT)) * 2) / h);
      const bind = (buf: WebGLBuffer | null, loc: number, data: Float32Array) => {
        if (!buf || loc < 0) return;
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        if (this.meshDirty) gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
      };
      if (!this.bufPos) this.bufPos = gl.createBuffer();
      if (!this.bufNor) this.bufNor = gl.createBuffer();
      if (!this.bufCol) this.bufCol = gl.createBuffer();
      bind(this.bufPos, this.aPos, mesh.positions);
      bind(this.bufNor, this.aNor, mesh.normals);
      bind(this.bufCol, this.aCol, mesh.colors);
      this.meshDirty = false;
      gl.drawArrays(gl.TRIANGLES, 0, mesh.vertexCount);
    }

    // ── O6-2 MINIS: static per-form buffers, one small draw per mini,
    // deterministic per-object orientation (no animation) ──
    if (this.minis.length) {
      if (this.uScale) gl.uniform1f(this.uScale, ((MINI_PIXELS / (2 * MODEL_HALF_EXTENT)) * 2) / h);
      for (const mini of this.minis) {
        let fb = this.formBuffers.get(mini.form);
        if (!fb) {
          const m = buildFormMesh(mini.form);
          const mk = (data: Float32Array): WebGLBuffer => {
            const b = gl.createBuffer();
            if (!b) throw new Error('SatModelLayer: createBuffer failed');
            gl.bindBuffer(gl.ARRAY_BUFFER, b);
            gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
            return b;
          };
          fb = { pos: mk(m.positions), nor: mk(m.normals), col: mk(m.colors), count: m.vertexCount };
          this.formBuffers.set(mini.form, fb);
        }
        if (this.uAnchor) gl.uniform3f(this.uAnchor, mini.mercX, mini.mercY, mini.altMeters);
        if (this.uRot) gl.uniformMatrix3fv(this.uRot, false, rotationMat3(mini.index * 2.399, 0.35));
        gl.bindBuffer(gl.ARRAY_BUFFER, fb.pos); enableAttrs(this.aPos);
        gl.bindBuffer(gl.ARRAY_BUFFER, fb.nor); enableAttrs(this.aNor);
        gl.bindBuffer(gl.ARRAY_BUFFER, fb.col); enableAttrs(this.aCol);
        gl.drawArrays(gl.TRIANGLES, 0, fb.count);
      }
    }

    // ── FOCUS RING at the anchor's rendered altitude — ONLY when there is
    // no model to see (unknown class): the human was clear that a circle
    // around a visible spacecraft is clutter ("i don't want a circle
    // around the sat"); with a mesh drawn, the model IS the focus marker ──
    if (this.anchor && !(focused && mesh)) {
      if (!this.ringBuffer) {
        const segs = 36;
        const ring = new Float32Array(segs * 3);
        for (let i = 0; i < segs; i++) {
          const a = (i / segs) * Math.PI * 2;
          ring[i * 3] = Math.cos(a) * MODEL_HALF_EXTENT;
          ring[i * 3 + 1] = Math.sin(a) * MODEL_HALF_EXTENT;
          ring[i * 3 + 2] = 0;
        }
        this.ringBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.ringBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, ring, gl.STATIC_DRAW);
      }
      if (this.uAnchor) gl.uniform3f(this.uAnchor, this.anchor.mercX, this.anchor.mercY, this.anchor.altMeters);
      if (this.uRot) gl.uniformMatrix3fv(this.uRot, false, new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]));
      if (this.uScale) gl.uniform1f(this.uScale, ((RING_PIXELS / MODEL_HALF_EXTENT) * 2) / h);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.ringBuffer);
      if (this.aPos >= 0) {
        gl.enableVertexAttribArray(this.aPos);
        gl.vertexAttribPointer(this.aPos, 3, gl.FLOAT, false, 0, 0);
      }
      // constant-attribute trick (airLayer precedent): flat normal + the
      // focus amber, no per-vertex buffers needed for a chrome ring
      if (this.aNor >= 0) { gl.disableVertexAttribArray(this.aNor); gl.vertexAttrib3f(this.aNor, 0, 0, 1); }
      if (this.aCol >= 0) { gl.disableVertexAttribArray(this.aCol); gl.vertexAttrib3f(this.aCol, 1.0, 0.82, 0.4); }
      gl.drawArrays(gl.LINE_LOOP, 0, 36);
    }

    gl.disable(gl.CULL_FACE);
    for (const loc of [this.aPos, this.aNor, this.aCol]) {
      if (loc >= 0) gl.disableVertexAttribArray(loc);
    }
  }

  private compile(gl: AnyGl, prelude: string, define: string, variant: string): void {
    if (this.program) {
      gl.deleteProgram(this.program);
      this.program = null;
    }
    const mk = (type: number, src: string): WebGLShader => {
      const sh = gl.createShader(type);
      if (!sh) throw new Error('SatModelLayer: createShader failed');
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error('SatModelLayer: shader compile failed: ' + log);
      }
      return sh;
    };
    const vs = mk(gl.VERTEX_SHADER, MODEL_VERT_SRC(prelude, define));
    const fs = mk(gl.FRAGMENT_SHADER, MODEL_FRAG_SRC);
    const p = gl.createProgram();
    if (!p) throw new Error('SatModelLayer: createProgram failed');
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error('SatModelLayer: program link failed: ' + log);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    this.program = p;
    this.cachedVariant = variant;
    this.aPos = gl.getAttribLocation(p, 'a_pos');
    this.aNor = gl.getAttribLocation(p, 'a_normal');
    this.aCol = gl.getAttribLocation(p, 'a_color');
    this.uAnchor = gl.getUniformLocation(p, 'u_anchor');
    this.uRot = gl.getUniformLocation(p, 'u_rot');
    this.uScale = gl.getUniformLocation(p, 'u_scaleClip');
    this.uAspect = gl.getUniformLocation(p, 'u_aspect');
    this.uProjMatrix = gl.getUniformLocation(p, 'u_projection_matrix');
    this.uProjTile = gl.getUniformLocation(p, 'u_projection_tile_mercator_coords');
    this.uProjClip = gl.getUniformLocation(p, 'u_projection_clipping_plane');
    this.uProjTrans = gl.getUniformLocation(p, 'u_projection_transition');
    this.uProjFallback = gl.getUniformLocation(p, 'u_projection_fallback_matrix');
  }
}

// ── Law IV budget declaration ───────────────────────────────────────────────
// Per-form mesh buffers (position/normal/colour), one set per distinct
// satellite form rather than per instance. Meshes are the heaviest thing
// this renderer uploads, hence the larger budget against a much smaller
// feature cap.
export const maxFeatures = 64;
export const vramBudget = 8; // MB
