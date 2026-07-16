// GPU satellite render layer — MapLibre v5 CustomLayerInterface for the
// ORBITAL program (research/orbital_program.md → O1 LOCKED APPROACH). Draws the
// full satellite population as instanced gl.POINTS, projected onto the globe
// (or mercator) via MapLibre's shader prelude with per-object ALTITUDE, so
// LEO/MEO/GEO shells are visually distinct. Ported from the O1 feasibility
// spike (orbital-spike/render_harness.html), upgraded to:
//   - projectTileFor3D + altitude (LEO/MEO/GEO shells) instead of flat projectTile
//   - color-by-orbit-class (packed classCode field)
//   - a validity sentinel: slots with NO real position (deep-space / invalid)
//     render at size 0 — NEVER a fabricated point (charter honesty rule)
//   - FAR-SIDE CULL (globe mode): satellites occluded by the earth are hidden
//     — projectTileFor3D itself applies no clipping, so without this the far
//     hemisphere's objects draw on top of the globe. Exact segment–sphere
//     test, altitude-aware (GEO stays visible past the limb); math mirrored
//     and unit-tested in ./occlusion, pinned by ./satLayer.test.ts. The CPU
//     pick path applies the same cull via getGlobeCamera().
//
// It consumes the Float32Array produced by ./satWorker (SAT_STRIDE floats per
// object; see ./satBuffer for the layout). This file is KERNEL-FREE: it imports
// only the pure buffer layout from ./satBuffer and TYPE-ONLY message shapes from
// ./satWorker (erased at compile time), so the SGP4 kernel never enters the main
// bundle — propagation lives solely in the worker.
//
// PERF / LOD (charter): O1 proved the point-field stays smooth past 100k, so
// the FULL population always renders as points — no silent decimation. The
// setRenderCap() hook exists only as a defensive lever for a struggling device;
// whenever it is set, getCounts() reports rendered < total and capped=true so
// the parent surfaces "showing N of M" (no-silent-caps rule). Progressive
// resolve on zoom (labels / 3D models / click targets) is a SEPARATE layer the
// parent adds — it never thins THIS field.
//
// PICKING (parent's job): custom layers have no queryRenderedFeatures. The
// buffer is index-aligned to the worker's GP order (sentinels included), so a
// CPU nearest-point lookup over getPositions() resolves a click to index i and
// straight into the parent's parallel metadata (noradId, epoch age / freshness,
// name…). getPositions()/getStride() expose exactly what that needs.

import type {
  CustomLayerInterface,
  CustomRenderMethodInput,
  Map as MapLibreMap,
} from 'maplibre-gl';
import { SAT_STRIDE, readSatAt } from './satBuffer.js';
import { cameraFromClippingPlane, type Vec3 } from './occlusion.js';
import type { SatPositionsMessage } from './satWorker.js';
import { metersPerPixel } from '../lod.js';

type AnyGl = WebGLRenderingContext | WebGL2RenderingContext;

/** RGBA in 0..1 (non-premultiplied; blend is SRC_ALPHA). */
export type Rgba = [number, number, number, number];

export interface SatLayerOptions {
  id?: string;
  /** point diameter in pixels (device-independent). */
  pointSize?: number;
  /** color for LEO-class objects. */
  colorLEO?: Rgba;
  /** color for MEO-class objects. */
  colorMEO?: Rgba;
  /** color for GEO-class objects. */
  colorGEO?: Rgba;
}

/** Subset of the worker positions message the layer echoes for the honesty panel. */
export interface PositionMeta {
  shown: number;
  deepSpaceSkipped: number;
  invalidSkipped: number;
}

export interface SatLayerCounts {
  /** slots in the buffer (== population size, index-aligned to worker GP order). */
  total: number;
  /** points actually issued to the GPU this frame (== min(cap, total)). */
  rendered: number;
  /** true when a render cap is hiding some objects — surface "showing N of M". */
  capped: boolean;
  /** slots carrying a real propagated position (from the worker; null until first update). */
  shown: number | null;
  /** objects skipped as deep-space (need SDP4). */
  deepSpaceSkipped: number | null;
  /** objects skipped for other reasons (missing elements / decayed). */
  invalidSkipped: number | null;
}

const DEFAULT_COLORS = {
  LEO: [0.3, 0.62, 1.0, 0.9] as Rgba, // cyan-blue (matches the O1 spike)
  MEO: [1.0, 0.72, 0.25, 0.9] as Rgba, // amber
  GEO: [0.85, 0.45, 1.0, 0.9] as Rgba, // violet
};

// PERF (scale_program.md queue item (c), "1Hz orbital repaint"): the worker
// ticks positions once a second for the WHOLE population, and every tick
// forced a full-map triggerRepaint — at low zoom (the default globe view)
// a satellite's per-second ground-track motion is a fraction of a screen
// pixel, so the map redrew every second for a change nobody could see,
// keeping a weak GPU from ever idling. MAX_GROUND_TRACK_SPEED_MPS is a
// conservative worst-case bound (ISS orbital/ground-track speed ~7.66 km/s;
// no catalogued object exceeds this at LEO, and MEO/GEO move far slower) —
// used to decide whether the WORST-CASE object in the population could have
// moved a visible amount, never a per-object check. Displacement
// ACCUMULATES across skipped ticks (never reset until an actual repaint
// fires), so staleness is bounded to <1px of worst-case drift at all
// times — this cannot silently freeze the layer indefinitely. The
// underlying position buffer is always updated regardless of this decision;
// only the forced GPU redraw is skipped, and any other repaint trigger
// (camera move, another layer) picks up the fresh data for free via
// dataDirty.
export const MAX_GROUND_TRACK_SPEED_MPS = 8000;

/**
 * True when the worst-case ground-track displacement over `elapsedSec`
 * (at the map's current center latitude/zoom) would round to under one
 * screen pixel — i.e. it is safe to skip forcing a repaint for this tick.
 * Pure function of camera state; exported for testing without a live map.
 */
export function shouldSkipTickRepaint(latDeg: number, zoom: number, elapsedSec: number): boolean {
  if (!Number.isFinite(latDeg) || !Number.isFinite(zoom) || !Number.isFinite(elapsedSec) || elapsedSec <= 0) {
    return false; // fail open — never suppress a repaint on broken input
  }
  const worstCaseDisplacementM = MAX_GROUND_TRACK_SPEED_MPS * elapsedSec;
  return metersPerPixel(latDeg, zoom) > worstCaseDisplacementM;
}

/** Exported for satLayer.test.ts, which pins the far-side-cull block to the
 * CPU mirror in ./occlusion (the shader inlines that module's math — GLSL
 * can't import TS, so the test is the sync mechanism). */
export const VERT_SRC = (prelude: string, define: string): string => `#version 300 es
${prelude}
${define}
in vec4 a_data;            // x=mercX(0..1) y=mercY(0..1) z=altMeters w=classCode
in float a_shape;          // SYMBOLS NOT DOTS: 0=unidentified(dot) 1=payload 2=rocket body 3=debris
uniform float u_size;
uniform float u_opacity;   // LOD envelope fade (EARTH TWIN A1) — 1 = fully visible
uniform vec4 u_colorLEO;
uniform vec4 u_colorMEO;
uniform vec4 u_colorGEO;
out vec4 v_color;
out float v_shape;
void main() {
  float cls = a_data.w;
  if (cls < 0.0) {
    // Sentinel slot: no real position (deep-space / invalid). Cull it —
    // never fabricate a location. Kept in the buffer only for index alignment.
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    v_color = vec4(0.0);
    return;
  }
#ifdef GLOBE
  // FAR-SIDE CULL (globe only): MapLibre's projectTileFor3D applies NO
  // occlusion (interpolateProjectionFor3D skips globeComputeClippingZ), so
  // without this, satellites physically behind the earth draw on top of the
  // globe. Exact segment–sphere test against the camera reconstructed from
  // the clipping plane (C = plane.xyz · -1/plane.w) — a plain hemisphere
  // test would wrongly hide high-altitude (GEO) objects visible past the
  // limb. Mirrors ./occlusion.ts (earthOccludes / cameraFromClippingPlane);
  // 0.998001 = OCCLUSION_RADIUS² (0.999², limb anti-flicker bias). Skipped
  // mid globe↔mercator transition (positions blend toward flat; there is no
  // far side to hide) and on a degenerate plane (w must be < 0).
  if (u_projection_transition > 0.999 && u_projection_clipping_plane.w < 0.0) {
    vec3 satPos = projectToSphere(a_data.xy) * (1.0 + a_data.z / GLOBE_RADIUS);
    vec3 cam = u_projection_clipping_plane.xyz * (-1.0 / u_projection_clipping_plane.w);
    vec3 v = satPos - cam;
    float t = -dot(cam, v) / dot(v, v);
    if (t > 0.0 && t < 1.0) {
      vec3 closest = cam + t * v;
      if (dot(closest, closest) < 0.998001) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        v_color = vec4(0.0);
        return;
      }
    }
  }
#endif
  gl_Position = projectTileFor3D(a_data.xy, a_data.z);
  // identified objects draw as type glyphs and need more pixels to read
  gl_PointSize = a_shape > 0.5 ? u_size * 2.6 : u_size;
  v_color = cls < 0.5 ? u_colorLEO : (cls < 1.5 ? u_colorMEO : u_colorGEO);
  v_color.a *= u_opacity;
  v_shape = a_shape;
}`;

// SYMBOLS NOT DOTS (human-directed 2026-07-15 + the standing 2026-07-12
// rule): identified objects draw as SDF type glyphs — payload = body with
// solar wings, rocket body = vertical capsule, debris = shard. A plain dot
// now MEANS "not yet identified" (SATCAT still loading, or the object is
// genuinely absent from the catalog) — shape is a catalogued fact, never a
// guess. Color stays the orbit class (the second dimension).
const FRAG_SRC = `#version 300 es
precision mediump float;
in vec4 v_color;
in float v_shape;
out vec4 o;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float a;
  if (v_shape < 0.5) {
    // unidentified: the original round dot
    float r = dot(d, d);
    if (r > 0.25) discard;
    a = smoothstep(0.25, 0.10, r);
  } else if (v_shape < 1.5) {
    // payload: square bus + horizontal solar wings
    float mBody = 1.0 - smoothstep(0.11, 0.15, max(abs(d.x), abs(d.y)));
    float mWing = (1.0 - smoothstep(0.06, 0.09, abs(d.y))) * (1.0 - smoothstep(0.40, 0.46, abs(d.x)));
    a = max(mBody, mWing);
    if (a < 0.05) discard;
  } else if (v_shape < 2.5) {
    // rocket body: vertical capsule
    a = (1.0 - smoothstep(0.09, 0.13, abs(d.x))) * (1.0 - smoothstep(0.34, 0.40, abs(d.y)));
    if (a < 0.05) discard;
  } else {
    // debris: diamond shard
    float m = abs(d.x) + abs(d.y);
    if (m > 0.32) discard;
    a = smoothstep(0.32, 0.20, m);
  }
  o = vec4(v_color.rgb, v_color.a * a);
}`;

/**
 * The satellite GPU point layer. Construct once, `map.addLayer(layer)`, then
 * feed it worker output via `updatePositions(buf, meta)`.
 */
export class SatLayer implements CustomLayerInterface {
  readonly id: string;
  readonly type = 'custom' as const;
  readonly renderingMode = '2d' as const;

  private map: MapLibreMap | null = null;
  private gl: AnyGl | null = null;
  private program: WebGLProgram | null = null;
  private buffer: WebGLBuffer | null = null;
  private cachedVariant: string | null = null;

  // uniform / attribute locations (resolved on compile)
  private aData = -1;
  private aShape = -1;
  private uSize: WebGLUniformLocation | null = null;
  private uOpacity: WebGLUniformLocation | null = null;
  private uColorLEO: WebGLUniformLocation | null = null;
  private uColorMEO: WebGLUniformLocation | null = null;
  private uColorGEO: WebGLUniformLocation | null = null;
  private uProjMatrix: WebGLUniformLocation | null = null;
  private uProjTile: WebGLUniformLocation | null = null;
  private uProjClip: WebGLUniformLocation | null = null;
  private uProjTrans: WebGLUniformLocation | null = null;
  private uProjFallback: WebGLUniformLocation | null = null;

  // position data + render state
  private data: Float32Array | null = null;
  private dataDirty = false;
  private uploadedFloats = -1;
  // SYMBOLS NOT DOTS: one shape code per object, index-aligned to the GP
  // order like everything else. null = catalog not joined yet → all dots.
  private shapeBuffer: WebGLBuffer | null = null;
  private shapeCodes: Float32Array | null = null;
  private shapeDirty = false;
  private total = 0;
  private renderCap: number | null = null;
  private meta: PositionMeta | null = null;
  // PERF: seconds of tick time skipped (no forced repaint) since the last
  // actual repaint — see shouldSkipTickRepaint / MAX_GROUND_TRACK_SPEED_MPS.
  private skippedRepaintSec = 0;

  // last frame's globe projection state, mirrored for the CPU pick path so
  // picking applies the SAME far-side cull the GPU applied (see ./occlusion).
  private lastClippingPlane: [number, number, number, number] | null = null;
  private lastTransition = 0;
  private lastMainMatrix: Float32Array | null = null;

  private pointSize: number;
  // LOD envelope fade (EARTH TWIN A1): 1 = fully visible. At 0 render()
  // skips the draw entirely — an out-of-envelope layer costs no GPU work.
  private globalOpacity = 1;
  private colorLEO: Rgba;
  private colorMEO: Rgba;
  private colorGEO: Rgba;

  // Set true if render throws (e.g. shaders fail to compile/link on a
  // constrained GL — a SwiftShader harness, an old device). render() is called
  // by MapLibre inside its per-frame loop; an uncaught throw there would break
  // the ENTIRE map every frame, not just this layer. So we catch once, disable
  // ourselves, and let the base map carry on (mirrors the globeSupport
  // "unavailable" degrade pattern). getRenderFailed() surfaces it for honesty.
  private renderFailed = false;

  constructor(opts: SatLayerOptions = {}) {
    this.id = opts.id ?? 'orbital-sats';
    this.pointSize = opts.pointSize ?? 3.0;
    this.colorLEO = opts.colorLEO ?? DEFAULT_COLORS.LEO;
    this.colorMEO = opts.colorMEO ?? DEFAULT_COLORS.MEO;
    this.colorGEO = opts.colorGEO ?? DEFAULT_COLORS.GEO;
  }

  // --- CustomLayerInterface ------------------------------------------------

  onAdd(map: MapLibreMap, gl: AnyGl): void {
    this.map = map;
    this.gl = gl;
    this.buffer = gl.createBuffer();
    // Program is compiled lazily in render() — it needs the projection
    // shaderData (prelude/variant), which is only available per-frame.
    this.uploadedFloats = -1;
    if (this.data) this.dataDirty = true; // data set before add -> upload on first render
  }

  render(gl: AnyGl, args: CustomRenderMethodInput): void {
    if (this.renderFailed) return; // disabled after a prior failure — never crash the map
    if (this.globalOpacity <= 0) return; // fully faded out (LOD envelope) — zero draw calls
    if (!this.data || this.total === 0) return;
    try {
      this.renderInner(gl, args);
    } catch (e) {
      this.renderFailed = true;
      // eslint-disable-next-line no-console
      console.error('SatLayer: disabling after render failure (map continues):', e);
    }
  }

  /** True once render has failed and the layer has self-disabled (for the honesty panel). */
  getRenderFailed(): boolean {
    return this.renderFailed;
  }

  private renderInner(gl: AnyGl, args: CustomRenderMethodInput): void {
    if (!this.data || this.total === 0) return; // re-narrow for TS (render() already guarded)
    const data = this.data;
    const sd = args.shaderData;
    if (this.program == null || this.cachedVariant !== sd.variantName) {
      this.compile(gl, sd.vertexShaderPrelude, sd.define, sd.variantName);
    }
    if (this.program == null || this.buffer == null) return;

    gl.useProgram(this.program);

    // Projection uniforms (globe + mercator + altitude); skip any the current
    // variant does not declare (mercator variant lacks the globe-only ones).
    const pd = args.defaultProjectionData;
    this.lastClippingPlane = [
      pd.clippingPlane[0], pd.clippingPlane[1], pd.clippingPlane[2], pd.clippingPlane[3],
    ];
    this.lastTransition = pd.projectionTransition;
    // O6 pick fix: cache this frame's projection matrix so CPU picking can
    // project candidates EXACTLY like the shader (including altitude — a
    // MEO object renders far from its ground point; ground-mercator picking
    // selected whatever LEO object's nadir sat under the cursor).
    if (!this.lastMainMatrix) this.lastMainMatrix = new Float32Array(16);
    this.lastMainMatrix.set(pd.mainMatrix as ArrayLike<number>);
    if (this.uProjMatrix) gl.uniformMatrix4fv(this.uProjMatrix, false, pd.mainMatrix);
    if (this.uProjTile) {
      gl.uniform4f(
        this.uProjTile,
        pd.tileMercatorCoords[0],
        pd.tileMercatorCoords[1],
        pd.tileMercatorCoords[2],
        pd.tileMercatorCoords[3],
      );
    }
    if (this.uProjClip) {
      gl.uniform4f(
        this.uProjClip,
        pd.clippingPlane[0],
        pd.clippingPlane[1],
        pd.clippingPlane[2],
        pd.clippingPlane[3],
      );
    }
    if (this.uProjTrans) gl.uniform1f(this.uProjTrans, pd.projectionTransition);
    if (this.uProjFallback) gl.uniformMatrix4fv(this.uProjFallback, false, pd.fallbackMatrix);

    // Style uniforms.
    if (this.uSize) gl.uniform1f(this.uSize, this.pointSize);
    if (this.uOpacity) gl.uniform1f(this.uOpacity, this.globalOpacity);
    if (this.uColorLEO) gl.uniform4f(this.uColorLEO, ...this.colorLEO);
    if (this.uColorMEO) gl.uniform4f(this.uColorMEO, ...this.colorMEO);
    if (this.uColorGEO) gl.uniform4f(this.uColorGEO, ...this.colorGEO);

    // Non-premultiplied alpha (matches the fragment output).
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    if (this.dataDirty) {
      if (data.length !== this.uploadedFloats) {
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
        this.uploadedFloats = data.length;
      } else {
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
      }
      this.dataDirty = false;
    }
    gl.enableVertexAttribArray(this.aData);
    gl.vertexAttribPointer(this.aData, 4, gl.FLOAT, false, SAT_STRIDE * 4, 0);

    // Shape codes: only honored when index-aligned to the population —
    // a mismatched buffer would put the wrong glyph on the wrong object,
    // so misalignment falls back to honest dots, never a mislabel.
    const shapesValid = this.shapeCodes != null && this.shapeCodes.length === this.total;
    if (this.aShape >= 0) {
      if (shapesValid) {
        if (!this.shapeBuffer) this.shapeBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.shapeBuffer);
        if (this.shapeDirty) {
          gl.bufferData(gl.ARRAY_BUFFER, this.shapeCodes!, gl.STATIC_DRAW);
          this.shapeDirty = false;
        }
        gl.enableVertexAttribArray(this.aShape);
        gl.vertexAttribPointer(this.aShape, 1, gl.FLOAT, false, 0, 0);
      } else {
        gl.disableVertexAttribArray(this.aShape);
        (gl as WebGL2RenderingContext).vertexAttrib1f?.(this.aShape, 0); // constant 0 = dot
      }
    }

    const count = this.renderCap != null ? Math.min(this.renderCap, this.total) : this.total;
    gl.drawArrays(gl.POINTS, 0, count);
    gl.disableVertexAttribArray(this.aData);
    if (this.aShape >= 0) gl.disableVertexAttribArray(this.aShape);
  }

  onRemove(_map: MapLibreMap, gl: AnyGl): void {
    if (this.program) gl.deleteProgram(this.program);
    if (this.buffer) gl.deleteBuffer(this.buffer);
    if (this.shapeBuffer) gl.deleteBuffer(this.shapeBuffer);
    this.shapeBuffer = null;
    this.shapeDirty = this.shapeCodes != null; // re-upload if re-added
    this.program = null;
    this.buffer = null;
    this.cachedVariant = null;
    this.uploadedFloats = -1;
    this.gl = null;
    this.map = null;
  }

  // --- Public API (parent wiring) -----------------------------------------

  /**
   * Feed a freshly propagated population from the worker. `data` is the
   * SAT_STRIDE-packed Float32Array; `meta` (the worker's shown/skipped counts)
   * is echoed through getCounts() for the honesty panel. The position buffer
   * is always updated; the forced repaint is skipped when `tickIntervalSec`
   * is given and the worst-case ground-track displacement accumulated since
   * the last repaint is still sub-pixel at the current camera (see
   * shouldSkipTickRepaint) — pass it for self-driven worker ticks, omit it
   * (or pass none) for one-off updates that should always redraw.
   */
  updatePositions(data: Float32Array, meta?: PositionMeta, tickIntervalSec?: number): void {
    this.data = data;
    this.total = Math.floor(data.length / SAT_STRIDE);
    this.meta = meta ?? this.meta;
    this.dataDirty = true;
    if (tickIntervalSec == null || !this.map) {
      this.skippedRepaintSec = 0;
      this.map?.triggerRepaint();
      return;
    }
    this.skippedRepaintSec += tickIntervalSec;
    const skip = shouldSkipTickRepaint(this.map.getCenter().lat, this.map.getZoom(), this.skippedRepaintSec);
    if (!skip) {
      this.skippedRepaintSec = 0;
      this.map.triggerRepaint();
    }
  }

  /** Set point diameter in pixels. */
  setPointSize(px: number): void {
    this.pointSize = px;
    this.map?.triggerRepaint();
  }

  /** Color points by orbit class. */
  setClassColors(colors: { leo?: Rgba; meo?: Rgba; geo?: Rgba }): void {
    if (colors.leo) this.colorLEO = colors.leo;
    if (colors.meo) this.colorMEO = colors.meo;
    if (colors.geo) this.colorGEO = colors.geo;
    this.map?.triggerRepaint();
  }

  /** Flat color mode: paint every class the same color. */
  setColor(rgba: Rgba): void {
    this.colorLEO = rgba;
    this.colorMEO = rgba;
    this.colorGEO = rgba;
    this.map?.triggerRepaint();
  }

  /**
   * LOD envelope fade (EARTH TWIN A1): whole-layer opacity 0..1. At 0 the
   * layer draws nothing at all (render() early-outs). This is a RENDER
   * choice, reversible by zoom — the parent must surface the hidden state
   * on-panel (never a silently vanished layer) and may pause the worker
   * while fully hidden.
   */
  setGlobalOpacity(o: number): void {
    const clamped = Math.max(0, Math.min(1, o));
    if (clamped === this.globalOpacity) return;
    this.globalOpacity = clamped;
    this.map?.triggerRepaint();
  }

  /** Current LOD fade (for wiring checks and the honesty panel). */
  getGlobalOpacity(): number {
    return this.globalOpacity;
  }

  /**
   * SYMBOLS NOT DOTS: one shape code per object (0 dot/unidentified,
   * 1 payload, 2 rocket body, 3 debris), index-aligned to the worker's GP
   * order. null clears back to all-dots. A buffer whose length doesn't
   * match the population is ignored at draw time (dots, never mislabels).
   */
  setShapeCodes(codes: Float32Array | null): void {
    this.shapeCodes = codes;
    this.shapeDirty = codes != null;
    this.map?.triggerRepaint();
  }

  /** The active shape codes (wiring checks). */
  getShapeCodes(): Float32Array | null {
    return this.shapeCodes;
  }

  /**
   * Decimation lever (defensive; O1 says the point field needs none). null =
   * render the full population. When set, only the first `cap` objects draw and
   * getCounts() reports capped=true / rendered<total so the caller can surface
   * "showing N of M" — never a silent drop.
   */
  setRenderCap(cap: number | null): void {
    this.renderCap = cap != null && cap >= 0 ? Math.floor(cap) : null;
    this.map?.triggerRepaint();
  }

  /** Shown-vs-total accounting for the honesty panel. */
  getCounts(): SatLayerCounts {
    const rendered = this.renderCap != null ? Math.min(this.renderCap, this.total) : this.total;
    return {
      total: this.total,
      rendered,
      capped: rendered < this.total,
      shown: this.meta?.shown ?? null,
      deepSpaceSkipped: this.meta?.deepSpaceSkipped ?? null,
      invalidSkipped: this.meta?.invalidSkipped ?? null,
    };
  }

  /** The last position buffer (for CPU nearest-point picking). Index-aligned to worker GP order. */
  getPositions(): Float32Array | null {
    return this.data;
  }

  /**
   * Camera position in unit-sphere space, reconstructed from the last frame's
   * clipping plane — non-null ONLY when the map is fully in globe mode (the
   * only mode where the shader far-side cull runs). Pass it to
   * pickNearestSatellite so clicks can't select satellites hidden behind the
   * earth. Null (mercator / mid-transition / no frame yet) = don't filter.
   */
  getGlobeCamera(): Vec3 | null {
    if (!this.lastClippingPlane || this.lastTransition <= 0.999) return null;
    return cameraFromClippingPlane(this.lastClippingPlane);
  }

  /** O6 pick fix: last frame's projection matrix (column-major, the exact
   *  matrix the shader used) — non-null only in full globe mode, where the
   *  CPU sphere math (occlusion.mercatorToSphere) mirrors the GPU. Null =
   *  caller falls back to ground-mercator picking. */
  getGlobeProjection(): Float32Array | null {
    if (!this.lastMainMatrix || this.lastTransition <= 0.999) return null;
    return this.lastMainMatrix;
  }

  /** Floats per object in the position buffer. */
  getStride(): number {
    return SAT_STRIDE;
  }

  /** Read one packed slot (mercX, mercY, altMeters, classCode, valid). */
  readSat(i: number): ReturnType<typeof readSatAt> | null {
    if (!this.data || i < 0 || i >= this.total) return null;
    return readSatAt(this.data, i);
  }

  // --- internals -----------------------------------------------------------

  private compile(gl: AnyGl, prelude: string, define: string, variant: string): void {
    // Drop any prior program (projection variant changed).
    if (this.program) {
      gl.deleteProgram(this.program);
      this.program = null;
    }
    const vs = this.mkShader(gl, gl.VERTEX_SHADER, VERT_SRC(prelude, define));
    const fs = this.mkShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
    const p = gl.createProgram();
    if (!p) throw new Error('SatLayer: createProgram failed');
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error('SatLayer: program link failed: ' + log);
    }
    // Shaders can be freed once linked.
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    this.program = p;
    this.cachedVariant = variant;
    this.aData = gl.getAttribLocation(p, 'a_data');
    this.aShape = gl.getAttribLocation(p, 'a_shape');
    this.uSize = gl.getUniformLocation(p, 'u_size');
    this.uOpacity = gl.getUniformLocation(p, 'u_opacity');
    this.uColorLEO = gl.getUniformLocation(p, 'u_colorLEO');
    this.uColorMEO = gl.getUniformLocation(p, 'u_colorMEO');
    this.uColorGEO = gl.getUniformLocation(p, 'u_colorGEO');
    this.uProjMatrix = gl.getUniformLocation(p, 'u_projection_matrix');
    this.uProjTile = gl.getUniformLocation(p, 'u_projection_tile_mercator_coords');
    this.uProjClip = gl.getUniformLocation(p, 'u_projection_clipping_plane');
    this.uProjTrans = gl.getUniformLocation(p, 'u_projection_transition');
    this.uProjFallback = gl.getUniformLocation(p, 'u_projection_fallback_matrix');
  }

  private mkShader(gl: AnyGl, type: number, src: string): WebGLShader {
    const sh = gl.createShader(type);
    if (!sh) throw new Error('SatLayer: createShader failed');
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      throw new Error('SatLayer: shader compile failed: ' + log);
    }
    return sh;
  }
}
