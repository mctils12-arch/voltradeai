// TRUE-ALTITUDE AIRCRAFT — EARTH TWIN E3 (research/earth_twin_program.md V2;
// the original directive's "see the planes at altitude").
//
// A MapLibre CustomLayerInterface (satLayer/modelLayer are the template:
// same projection prelude, variant caching, far-side cull, failure latch)
// that draws every live aircraft as a heading-oriented plane SILHOUETTE at
// its REAL barometric altitude via projectTileFor3D — with pitch + 3D
// terrain, planes visibly fly above the ground instead of being painted on
// it. WebGL2 instancing: ONE static silhouette mesh + a 5-float instance
// record per aircraft (~200KB at the 10k cap — a trivial per-tick upload).
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

/** The 2D↔3D hand-off zoom: symbols below, silhouettes at/above. */
export const AIR_3D_MIN_ZOOM = 8;

/** floats per instance: mercX, mercY, altMeters, headingDeg, band. */
export const AIR_INST_STRIDE = 5;

/** altitude band codes (match the 2D layer's ALT_COLOR semantics). */
export const AIR_BAND = { GROUND: 0, LOW: 1, CRUISE: 2 } as const;

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

export interface AircraftInstanceInput {
  lon?: number | null;
  lat?: number | null;
  altitude_m?: number | null;
  heading?: number | null;
  on_ground?: boolean | null;
}

/**
 * Pure: aircraft payload rows → packed instance buffer + the index-aligned
 * rows (for CPU picking → the click card). Rows without a finite position
 * are skipped (counted by the caller via rows.length vs input length —
 * nothing rendered from a guessed position).
 */
export function buildAircraftInstances<T extends AircraftInstanceInput>(
  aircraft: T[],
): { inst: Float32Array; rows: T[] } {
  const rows: T[] = [];
  for (const a of aircraft) {
    if (a == null || !Number.isFinite(a.lon as number) || !Number.isFinite(a.lat as number)) continue;
    if ((a.lat as number) < -85 || (a.lat as number) > 85) continue; // outside mercator
    rows.push(a);
  }
  const inst = new Float32Array(rows.length * AIR_INST_STRIDE);
  for (let i = 0; i < rows.length; i++) {
    const a = rows[i];
    const m = lonLatToMercator(a.lon as number, a.lat as number);
    const ground = !!a.on_ground;
    const alt = ground ? 0 : Math.max(0, a.altitude_m ?? 0);
    inst[i * AIR_INST_STRIDE] = m.x;
    inst[i * AIR_INST_STRIDE + 1] = m.y;
    inst[i * AIR_INST_STRIDE + 2] = alt;
    inst[i * AIR_INST_STRIDE + 3] = a.heading ?? 0;
    inst[i * AIR_INST_STRIDE + 4] = ground ? AIR_BAND.GROUND : alt < 3000 ? AIR_BAND.LOW : AIR_BAND.CRUISE;
  }
  return { inst, rows };
}

/** Pure: nearest instance to a mercator point within tolerance, else -1. */
export function pickNearestAircraft(
  inst: Float32Array | null,
  mercX: number,
  mercY: number,
  tolMercUnits: number,
): number {
  if (!inst || inst.length < AIR_INST_STRIDE) return -1;
  const n = Math.floor(inst.length / AIR_INST_STRIDE);
  let best = -1;
  let bestD2 = tolMercUnits * tolMercUnits;
  for (let i = 0; i < n; i++) {
    const dx = inst[i * AIR_INST_STRIDE] - mercX;
    const dy = inst[i * AIR_INST_STRIDE + 1] - mercY;
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
in vec4 a_inst;            // per-instance: mercX, mercY, altMeters, headingDeg
in float a_band;           // per-instance: altitude band (0 ground / 1 low / 2 cruise)
uniform float u_bearing;   // map bearing (deg) — silhouettes rotate with the map like icons
uniform float u_altScale;  // 1, or the terrain exaggeration so vertical datums match
uniform float u_scaleClip; // clip units per local unit at w=1 (pixel-constant via *w)
uniform float u_aspect;
uniform vec4 u_bandColors[3];
out vec4 v_color;
void main() {
#ifdef GLOBE
  // anchor far-side cull — identical to satLayer/modelLayer (inert at the
  // z>=8 zooms this layer draws at, kept for contract parity; 0.998001 =
  // OCCLUSION_RADIUS²)
  if (u_projection_transition > 0.999 && u_projection_clipping_plane.w < 0.0) {
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
  vec4 anchor = projectTileFor3D(a_inst.xy, a_inst.z * u_altScale);
  float rot = radians(a_inst.w - u_bearing);
  float c = cos(rot);
  float s = sin(rot);
  vec2 p = vec2(a_local.x * c + a_local.y * s, -a_local.x * s + a_local.y * c);
  gl_Position = anchor + vec4(p.x * u_scaleClip / u_aspect, p.y * u_scaleClip, 0.0, 0.0) * anchor.w;
  int band = int(clamp(a_band, 0.0, 2.0));
  v_color = u_bandColors[band];
}`;

const AIR_FRAG_SRC = `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 o;
void main() { o = v_color; }`;

type AnyGl = WebGLRenderingContext | WebGL2RenderingContext;

// altitude bands — same meanings as the 2D layer's ALT_COLOR
const BAND_COLORS: [number, number, number, number][] = [
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
  private geomBuffer: WebGLBuffer | null = null;
  private instBuffer: WebGLBuffer | null = null;
  private aLocal = -1;
  private aInst = -1;
  private aBand = -1;
  private uBearing: WebGLUniformLocation | null = null;
  private uAltScale: WebGLUniformLocation | null = null;
  private uScale: WebGLUniformLocation | null = null;
  private uAspect: WebGLUniformLocation | null = null;
  private uBandColors: WebGLUniformLocation | null = null;
  private uProjMatrix: WebGLUniformLocation | null = null;
  private uProjTile: WebGLUniformLocation | null = null;
  private uProjClip: WebGLUniformLocation | null = null;
  private uProjTrans: WebGLUniformLocation | null = null;
  private uProjFallback: WebGLUniformLocation | null = null;

  private inst: Float32Array | null = null;
  private instDirty = false;
  private geomUploaded = false;
  private count = 0;
  private altScale = 1;
  private renderFailed = false;

  constructor(opts: { id?: string } = {}) {
    this.id = opts.id ?? 'aircraft-3d';
  }

  onAdd(map: MapLibreMap, _gl: AnyGl): void {
    this.map = map;
  }

  onRemove(_map: MapLibreMap, gl: AnyGl): void {
    if (this.program) gl.deleteProgram(this.program);
    for (const b of [this.geomBuffer, this.instBuffer]) if (b) gl.deleteBuffer(b);
    this.program = null;
    this.geomBuffer = this.instBuffer = null;
    this.cachedVariant = null;
    this.geomUploaded = false;
    this.instDirty = this.inst != null;
    this.map = null;
  }

  /** Fresh tick's instance buffer (null clears). */
  setInstances(inst: Float32Array | null): void {
    this.inst = inst;
    this.count = inst ? Math.floor(inst.length / AIR_INST_STRIDE) : 0;
    this.instDirty = inst != null;
    this.map?.triggerRepaint();
  }

  /** terrain exaggeration match (1 without terrain). */
  setAltScale(s: number): void {
    if (s === this.altScale) return;
    this.altScale = s;
    this.map?.triggerRepaint();
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
    const h = gl.drawingBufferHeight || 1;
    const w = gl.drawingBufferWidth || 1;
    const pxPerUnit = AIR_PIXELS / LOCAL_EXTENT;
    if (this.uScale) gl.uniform1f(this.uScale, (pxPerUnit * 2) / h);
    if (this.uAspect) gl.uniform1f(this.uAspect, w / h);
    if (this.uBandColors) gl.uniform4fv(this.uBandColors, BAND_COLORS.flat());

    if (!this.geomBuffer) this.geomBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.geomBuffer);
    if (!this.geomUploaded) {
      gl.bufferData(gl.ARRAY_BUFFER, PLANE_SILHOUETTE, gl.STATIC_DRAW);
      this.geomUploaded = true;
    }
    gl.enableVertexAttribArray(this.aLocal);
    gl.vertexAttribPointer(this.aLocal, 2, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(this.aLocal, 0);

    if (!this.instBuffer) this.instBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instBuffer);
    if (this.instDirty && this.inst) {
      gl.bufferData(gl.ARRAY_BUFFER, this.inst, gl.DYNAMIC_DRAW);
      this.instDirty = false;
    }
    gl.enableVertexAttribArray(this.aInst);
    gl.vertexAttribPointer(this.aInst, 4, gl.FLOAT, false, AIR_INST_STRIDE * 4, 0);
    gl.vertexAttribDivisor(this.aInst, 1);
    gl.enableVertexAttribArray(this.aBand);
    gl.vertexAttribPointer(this.aBand, 1, gl.FLOAT, false, AIR_INST_STRIDE * 4, 16);
    gl.vertexAttribDivisor(this.aBand, 1);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, PLANE_SILHOUETTE.length / 2, this.count);

    // reset divisors so we never corrupt MapLibre's own attribute state
    gl.vertexAttribDivisor(this.aInst, 0);
    gl.vertexAttribDivisor(this.aBand, 0);
    gl.disableVertexAttribArray(this.aLocal);
    gl.disableVertexAttribArray(this.aInst);
    gl.disableVertexAttribArray(this.aBand);
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
    this.aInst = gl.getAttribLocation(p, 'a_inst');
    this.aBand = gl.getAttribLocation(p, 'a_band');
    this.uBearing = gl.getUniformLocation(p, 'u_bearing');
    this.uAltScale = gl.getUniformLocation(p, 'u_altScale');
    this.uScale = gl.getUniformLocation(p, 'u_scaleClip');
    this.uAspect = gl.getUniformLocation(p, 'u_aspect');
    this.uBandColors = gl.getUniformLocation(p, 'u_bandColors');
    this.uProjMatrix = gl.getUniformLocation(p, 'u_projection_matrix');
    this.uProjTile = gl.getUniformLocation(p, 'u_projection_tile_mercator_coords');
    this.uProjClip = gl.getUniformLocation(p, 'u_projection_clipping_plane');
    this.uProjTrans = gl.getUniformLocation(p, 'u_projection_transition');
    this.uProjFallback = gl.getUniformLocation(p, 'u_projection_fallback_matrix');
  }
}
