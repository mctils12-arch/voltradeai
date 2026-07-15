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
  // behind the earth must vanish with it.
  if (u_projection_transition > 0.999 && u_projection_clipping_plane.w < 0.0) {
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
  vec4 anchor = projectTileFor3D(u_anchor.xy, u_anchor.z);
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

/** Constant on-screen model height in pixels (the focused-object size). */
export const MODEL_PIXELS = 72;
/** Model-space half-extent the meshes are built to (model3d contract). */
export const MODEL_HALF_EXTENT = 1.2;

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
  private meshDirty = false;
  private form: FormKind | null = null;
  private anchor: ModelAnchor | null = null;
  private renderFailed = false; // same self-disable latch as satLayer

  constructor(opts: { id?: string } = {}) {
    this.id = opts.id ?? 'orbital-sat-model';
  }

  onAdd(map: MapLibreMap, _gl: AnyGl): void {
    this.map = map;
  }

  onRemove(_map: MapLibreMap, gl: AnyGl): void {
    if (this.program) gl.deleteProgram(this.program);
    for (const b of [this.bufPos, this.bufNor, this.bufCol]) if (b) gl.deleteBuffer(b);
    this.program = null;
    this.bufPos = this.bufNor = this.bufCol = null;
    this.cachedVariant = null;
    this.map = null;
  }

  /** The form to draw at focus, or null = draw nothing (unknown class stays
   *  an honest ring-only follow — never a guessed spacecraft). */
  setForm(form: FormKind | null): void {
    if (form === this.form) return;
    this.form = form;
    this.mesh = form ? buildFormMesh(form) : null;
    this.meshDirty = true;
    this.map?.triggerRepaint();
  }

  getForm(): FormKind | null {
    return this.form;
  }

  /** The followed satellite's live position (null = not following). */
  setAnchor(a: ModelAnchor | null): void {
    this.anchor = a;
    this.map?.triggerRepaint();
  }

  getAnchor(): ModelAnchor | null {
    return this.anchor;
  }

  getRenderFailed(): boolean {
    return this.renderFailed;
  }

  render(gl: AnyGl, args: CustomRenderMethodInput): void {
    if (this.renderFailed) return;
    if (!this.anchor || !this.mesh) return; // not following → zero cost, no repaint
    try {
      this.renderInner(gl, args);
      this.map?.triggerRepaint(); // animate the tumble ONLY while visible
    } catch (e) {
      this.renderFailed = true;
      // eslint-disable-next-line no-console
      console.error('SatModelLayer: disabling after render failure (map continues):', e);
    }
  }

  private renderInner(gl: AnyGl, args: CustomRenderMethodInput): void {
    const mesh = this.mesh;
    const anchor = this.anchor;
    if (!mesh || !anchor) return;
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

    if (this.uAnchor) gl.uniform3f(this.uAnchor, anchor.mercX, anchor.mercY, anchor.altMeters);
    // slow tumble, time-based (display animation only — clearly not telemetry)
    const t = (typeof performance !== 'undefined' ? performance.now() : 0) / 1000;
    if (this.uRot) gl.uniformMatrix3fv(this.uRot, false, rotationMat3(t * 0.5, 0.4));
    const h = gl.drawingBufferHeight || 1;
    const w = gl.drawingBufferWidth || 1;
    // clip units per model unit so the form spans ~MODEL_PIXELS on screen
    const pxPerUnit = MODEL_PIXELS / (2 * MODEL_HALF_EXTENT);
    if (this.uScale) gl.uniform1f(this.uScale, (pxPerUnit * 2) / h);
    if (this.uAspect) gl.uniform1f(this.uAspect, w / h);

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

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.disable(gl.BLEND); // opaque mesh
    gl.drawArrays(gl.TRIANGLES, 0, mesh.vertexCount);
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
