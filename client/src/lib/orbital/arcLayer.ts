// Orbit arc layer — EARTH TWIN O6-1. A MapLibre CustomLayerInterface
// (satLayer/modelLayer/airLayer are the template: same projection prelude,
// variant caching, failure latch, GLOBE-guarded far-side handling) that
// draws orbit tracks as GL_LINES at TRUE altitude via projectTileFor3D.
//
// Line-specific choices:
// - Segments, not strips: a pair is emitted only between two consecutive
//   GOOD samples that don't jump the antimeridian (|dx| > 0.5 in mercator)
//   — gaps break the line honestly instead of bridging it.
// - Far-side handling is a FRAGMENT discard (v_cull varying) rather than
//   the point layers' vertex snap-out: snapping one endpoint of a line
//   stretches the segment across the screen; discarding fades the arc at
//   the horizon cleanly.
// - COST: zero when no arcs are set; static buffer per setArcs call; no
//   self-repaint (arcs move only when re-sampled by the caller).

import type {
  CustomLayerInterface,
  CustomRenderMethodInput,
  Map as MapLibreMap,
} from 'maplibre-gl';
import { ARC_GAP } from './orbitArc.js';

type AnyGl = WebGLRenderingContext | WebGL2RenderingContext;

export interface Arc {
  /** interleaved [mercX, mercY, altMeters] triplets (orbitArc.sampleOrbitArc). */
  pts: Float32Array;
  color: [number, number, number, number];
}

/** floats per line vertex: x, y, altM, r, g, b, a. */
export const ARC_VERT_STRIDE = 7;

/** Pure: arcs → packed GL_LINES vertex array (exported for tests). */
export function buildArcVertices(arcs: Arc[]): Float32Array {
  const segs: number[] = [];
  for (const arc of arcs) {
    const n = Math.floor(arc.pts.length / 3);
    for (let i = 0; i + 1 < n; i++) {
      const a = i * 3, b = (i + 1) * 3;
      if (arc.pts[a + 2] === ARC_GAP || arc.pts[b + 2] === ARC_GAP) continue; // honest gap
      if (Math.abs(arc.pts[a] - arc.pts[b]) > 0.5) continue; // antimeridian jump
      segs.push(
        arc.pts[a], arc.pts[a + 1], arc.pts[a + 2], ...arc.color,
        arc.pts[b], arc.pts[b + 1], arc.pts[b + 2], ...arc.color,
      );
    }
  }
  return new Float32Array(segs);
}

/** Exported for arcLayer.test.ts — pins the projection/cull contract. */
export const ARC_VERT_SRC = (prelude: string, define: string): string => `#version 300 es
${prelude}
${define}
in vec3 a_pos;    // mercX, mercY, altMeters
in vec4 a_color;
out vec4 v_color;
out float v_cull;
void main() {
  v_cull = 0.0;
#ifdef GLOBE
  // far-side test — identical formula to satLayer/modelLayer/airLayer
  // (0.998001 = OCCLUSION_RADIUS²) but flagged to the FRAGMENT stage:
  // lines must fade at the horizon, not stretch to a snapped vertex.
  if (u_projection_transition > 0.999 && u_projection_clipping_plane.w < 0.0) {
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
  gl_Position = projectTileFor3D(a_pos.xy, a_pos.z);
  v_color = a_color;
}`;

const ARC_FRAG_SRC = `#version 300 es
precision mediump float;
in vec4 v_color;
in float v_cull;
out vec4 o;
void main() {
  if (v_cull > 0.01) discard;
  o = v_color;
}`;

export class ArcLayer implements CustomLayerInterface {
  readonly id: string;
  readonly type = 'custom' as const;
  readonly renderingMode = '2d' as const;

  private map: MapLibreMap | null = null;
  private program: WebGLProgram | null = null;
  private cachedVariant: string | null = null;
  private buffer: WebGLBuffer | null = null;
  private aPos = -1;
  private aColor = -1;
  private uProjMatrix: WebGLUniformLocation | null = null;
  private uProjTile: WebGLUniformLocation | null = null;
  private uProjClip: WebGLUniformLocation | null = null;
  private uProjTrans: WebGLUniformLocation | null = null;
  private uProjFallback: WebGLUniformLocation | null = null;

  private verts: Float32Array | null = null;
  private dirty = false;
  private renderFailed = false;

  constructor(opts: { id?: string } = {}) {
    this.id = opts.id ?? 'orbital-arcs';
  }

  onAdd(map: MapLibreMap, _gl: AnyGl): void {
    this.map = map;
  }

  onRemove(_map: MapLibreMap, gl: AnyGl): void {
    if (this.program) gl.deleteProgram(this.program);
    if (this.buffer) gl.deleteBuffer(this.buffer);
    this.program = null;
    this.buffer = null;
    this.cachedVariant = null;
    this.dirty = this.verts != null;
    this.map = null;
  }

  /** Replace the displayed arcs (null/[] clears — layer costs nothing). */
  setArcs(arcs: Arc[] | null): void {
    this.verts = arcs && arcs.length ? buildArcVertices(arcs) : null;
    this.dirty = this.verts != null;
    this.map?.triggerRepaint();
  }

  getVertexCount(): number {
    return this.verts ? this.verts.length / ARC_VERT_STRIDE : 0;
  }

  getRenderFailed(): boolean {
    return this.renderFailed;
  }

  render(gl: AnyGl, args: CustomRenderMethodInput): void {
    if (this.renderFailed) return;
    if (!this.verts || this.verts.length === 0) return; // nothing set → zero cost
    try {
      this.renderInner(gl, args);
    } catch (e) {
      this.renderFailed = true;
      // eslint-disable-next-line no-console
      console.error('ArcLayer: disabling after render failure (map continues):', e);
    }
  }

  private renderInner(gl: AnyGl, args: CustomRenderMethodInput): void {
    const sd = args.shaderData;
    if (this.program == null || this.cachedVariant !== sd.variantName) {
      this.compile(gl, sd.vertexShaderPrelude, sd.define, sd.variantName);
      this.dirty = true;
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

    if (!this.buffer) this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    if (this.dirty && this.verts) {
      gl.bufferData(gl.ARRAY_BUFFER, this.verts, gl.STATIC_DRAW);
      this.dirty = false;
    }
    gl.enableVertexAttribArray(this.aPos);
    gl.vertexAttribPointer(this.aPos, 3, gl.FLOAT, false, ARC_VERT_STRIDE * 4, 0);
    gl.enableVertexAttribArray(this.aColor);
    gl.vertexAttribPointer(this.aColor, 4, gl.FLOAT, false, ARC_VERT_STRIDE * 4, 12);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.LINES, 0, this.verts!.length / ARC_VERT_STRIDE);

    gl.disableVertexAttribArray(this.aPos);
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
    this.aColor = gl.getAttribLocation(p, 'a_color');
    this.uProjMatrix = gl.getUniformLocation(p, 'u_projection_matrix');
    this.uProjTile = gl.getUniformLocation(p, 'u_projection_tile_mercator_coords');
    this.uProjClip = gl.getUniformLocation(p, 'u_projection_clipping_plane');
    this.uProjTrans = gl.getUniformLocation(p, 'u_projection_transition');
    this.uProjFallback = gl.getUniformLocation(p, 'u_projection_fallback_matrix');
  }
}
