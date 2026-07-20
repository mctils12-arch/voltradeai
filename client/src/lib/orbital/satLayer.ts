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
//   - PER-FRAME VELOCITY GLIDE: between 1 Hz worker ticks the vertex shader
//     extrapolates each point along its packed real finite-difference
//     velocity (see MAX_GLIDE_SEC / setTickTime) — smooth 60 fps motion from
//     honest physics, display-only (CPU consumers keep tick positions)
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
  /** injectable monotonic clock for the glide (ms) — defaults to
   *  performance.now(); tests inject a fake to run without a browser. */
  now?: () => number;
  /** injectable timer pair for the PACED glide repaint (pulse fix) —
   *  defaults to setTimeout/clearTimeout; tests inject fakes. */
  setTimeoutFn?: (cb: () => void, ms: number) => unknown;
  clearTimeoutFn?: (handle: unknown) => void;
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

// ── PER-FRAME VELOCITY GLIDE (human directive 2026-07-17: "make them move
// smoothly … interpret where they would go … but still use the data we get").
// The worker still ticks REAL SGP4/SDP4 positions at ~1 Hz; between ticks the
// vertex shader extrapolates each point along its packed per-second velocity
// (a finite difference of two real propagations — satWorker.VEL_SAMPLE_MS):
// position = tick position + velocity · u_dtSec. First-order honest motion,
// never a made-up path.
//
// HONESTY CAP: past MAX_GLIDE_SEC of extrapolation the first-order glide
// stops being physics and starts being fiction (a stale worker would send
// points sailing off their orbits), so u_dtSec clamps there — a stale layer
// HOLDS at the cap rather than running away, and the next real tick snaps
// everything back to truth.
export const MAX_GLIDE_SEC = 2.5;

/** Conservative frame interval (seconds) used to decide whether ONE frame of
 *  glide motion could even be visible at the current camera (~30fps bound —
 *  a longer interval than real 60fps frames, so the test errs toward
 *  repainting). */
export const GLIDE_FRAME_SEC = 1 / 30;

/**
 * Seconds of glide to apply this frame: elapsed time since the tick anchor,
 * clamped to [0, MAX_GLIDE_SEC] (see the cap note above). Non-finite input
 * (no anchor yet, broken clock) glides zero — the raw tick positions render,
 * which is exactly the pre-glide behavior.
 */
export function glideDtSec(nowMs: number, tickAnchorMs: number): number {
  return glideDtSecWarp(nowMs, tickAnchorMs, 1);
}

/**
 * B3 SIM-CLOCK generalization of glideDtSec: elapsed real seconds since the
 * anchor, scaled by the simulation rate (u_dtSec is SIMULATED seconds — the
 * worker's finite-difference velocity is per simulated second, so warped
 * time glides rate× faster), still clamped to the same MAX_GLIDE_SEC
 * honesty cap (the cap bounds linear-extrapolation FICTION in simulated
 * seconds of orbital motion; warping time does not loosen it). timeScale 1
 * is the exact pre-B3 math (pinned by test); timeScale 0 (paused) glides
 * zero — frozen positions, honestly.
 */
export function glideDtSecWarp(nowMs: number, tickAnchorMs: number, timeScale: number): number {
  const dt = ((nowMs - tickAnchorMs) / 1000) * timeScale;
  if (!Number.isFinite(dt) || dt <= 0) return 0;
  return dt < MAX_GLIDE_SEC ? dt : MAX_GLIDE_SEC;
}

/**
 * PERF (resolves scale_program.md's "1Hz orbital repaint reconsideration"):
 * gliding needs continuous repaint, but ONLY while a frame's worth of motion
 * is actually visible at the current camera — zoomed out on the default
 * globe, per-frame satellite motion is sub-pixel and the map may idle
 * exactly as the 1Hz tick-skip logic already allows (the 1 Hz buffer swap
 * still repaints via updatePositions when its ACCUMULATED drift becomes
 * visible). So the continuous-repaint cost is paid only when the orbital
 * layer is on AND the camera is close enough for glide to be seen.
 * Pure function of camera state, exported for hermetic tests.
 */
export function shouldRequestGlideFrame(latDeg: number, zoom: number): boolean {
  return !shouldSkipTickRepaint(latDeg, zoom, GLIDE_FRAME_SEC);
}

/**
 * PULSE FIX (live report 2026-07-18, "rhythmic pulsing"): the binary
 * shouldRequestGlideFrame gate left a DEAD BAND — the zoom range where one
 * FRAME of worst-case motion is sub-pixel (so no glide frames were ever
 * requested) but one TICK of motion is 1..30px (so the only repaint was the
 * 1 Hz buffer swap). In that band satellites moved in pure 1 Hz snaps: the
 * pulse WAS the tick (measured: ~10px steps once/second with zero motion
 * between, drive_pulse.mjs, z7 equator). At the equator the band spanned
 * zoom ~3.3..8.2 — most of the useful orbital viewing range.
 *
 * This function replaces the binary decision with a PACED one: how long
 * until the worst-case object has moved ~one pixel, i.e. how long a repaint
 * may wait without the motion becoming a visible step.
 *   0     -> per-frame motion is visible: request every frame (old behavior)
 *   d > 0 -> schedule ONE repaint d seconds out (paced glide — repaint
 *            exactly as often as motion crosses a pixel, ~1..30 Hz across
 *            the former dead band, scaling with zoom)
 *   null  -> even a full refresh horizon of motion is sub-pixel: no glide
 *            repaint needed (tick-accumulation repaints still cover it, and
 *            the map may idle — the perf win this pacing preserves).
 * `horizonSec` is the time base already covered by other repaint sources —
 * the worker tick interval when streaming (updatePositions passes it), else
 * MAX_GLIDE_SEC (past the cap the glide freezes, so no frames are owed).
 * Same worst-case bound as shouldSkipTickRepaint (MAX_GROUND_TRACK_SPEED_MPS
 * at the map-CENTER latitude — a near-pole object on a flat mercator view
 * can exceed it by its 1/cos(lat) stretch; accepted, the globe view where
 * low zooms actually happen has no such stretch). Fail-open: broken camera
 * numbers return 0 (repaint), never a silent freeze. Pure; hermetic tests.
 */
export function glideRepaintDelaySec(
  latDeg: number,
  zoom: number,
  tickIntervalSec: number | null,
): number | null {
  if (shouldRequestGlideFrame(latDeg, zoom)) return 0; // includes the fail-open paths
  const mpp = metersPerPixel(latDeg, zoom);
  if (!Number.isFinite(mpp) || mpp <= 0) return 0; // fail open — repaint
  const secPerPixel = mpp / MAX_GROUND_TRACK_SPEED_MPS;
  const horizonSec =
    tickIntervalSec != null && tickIntervalSec > 0 ? tickIntervalSec : MAX_GLIDE_SEC;
  return secPerPixel < horizonSec ? secPerPixel : null;
}

/**
 * GLIDE ANCHOR FROM THE WORKER EPOCH (pulse fix, secondary): the worker
 * stamps each positions message with the Date.now() epoch it propagated TO
 * (timeMs, captured BEFORE the ~60-120ms 16k×2-SGP4 pack), while the layer's
 * glide clock is performance.now(). Anchoring at ARRIVAL (the old wiring)
 * told the shader "these positions are true now", hiding the pack+transfer
 * latency: the display permanently lagged truth by that latency, and its
 * tick-to-tick JITTER rendered as small backward snaps at close zooms. This
 * maps the worker epoch into the layer clock so dt measures from the real
 * propagation instant. A non-finite/negative/absurd lag (clock skew, NTP
 * jump) falls back to the arrival anchor — the pre-fix behavior, never
 * worse. Pure; all three clocks injected for tests.
 */
export const MAX_ANCHOR_LAG_MS = 10_000;
export function tickAnchorFromEpoch(
  epochMs: number,
  dateNowMs: number,
  perfNowMs: number,
): number {
  const lag = dateNowMs - epochMs;
  if (!Number.isFinite(lag) || lag < 0 || lag > MAX_ANCHOR_LAG_MS) return perfNowMs;
  return perfNowMs - lag;
}

/**
 * B3 SIM-CLOCK generalization of tickAnchorFromEpoch: the worker propagated
 * to a SIMULATED epoch; the lag to the current simulated now is in
 * simulated ms, so it maps into the layer's real (performance.now) clock
 * divided by the rate. With rate 1 and simulated ≡ real this is EXACTLY
 * tickAnchorFromEpoch (pinned by test). The same MAX_ANCHOR_LAG_MS guard
 * applies to the REAL-mapped lag (the guard is about wall-clock skew);
 * rate ≤ 0 (paused) anchors at arrival — the glide dt is zero there anyway
 * (glideDtSecWarp × 0).
 */
export function tickAnchorFromSimEpoch(
  epochSimMs: number,
  simNowMs: number,
  perfNowMs: number,
  rate: number,
): number {
  if (!Number.isFinite(rate) || rate <= 0) return perfNowMs;
  const lagReal = (simNowMs - epochSimMs) / rate;
  if (!Number.isFinite(lagReal) || lagReal < 0 || lagReal > MAX_ANCHOR_LAG_MS) return perfNowMs;
  return perfNowMs - lagReal;
}

/** Exported for satLayer.test.ts, which pins the far-side-cull block to the
 * CPU mirror in ./occlusion (the shader inlines that module's math — GLSL
 * can't import TS, so the test is the sync mechanism). */
export const VERT_SRC = (prelude: string, define: string): string => `#version 300 es
${prelude}
${define}
in vec4 a_data;            // x=mercX(0..1) y=mercY(0..1) z=altMeters w=classCode
in vec3 a_vel;             // GLIDE: d/dt of (mercX, mercY, altMeters) per second
in float a_shape;          // SYMBOLS NOT DOTS: 0=unidentified(dot) 1=payload 2=rocket body 3=debris
uniform float u_size;
uniform float u_dtSec;     // seconds since the worker tick, CPU-clamped to MAX_GLIDE_SEC
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
  // PER-FRAME GLIDE: first-order motion between real physics ticks —
  // tick position + real finite-difference velocity × capped elapsed time
  // (u_dtSec; beyond the cap the extrapolation would be fiction, so the CPU
  // clamps it — see MAX_GLIDE_SEC). fract() wraps X across the antimeridian
  // (mercator X is periodic, and GLSL fract of a negative wraps correctly);
  // Y clamps at the poles like the worker's own pack does.
  vec2 g_xy = vec2(fract(a_data.x + a_vel.x * u_dtSec),
                   clamp(a_data.y + a_vel.y * u_dtSec, 0.0, 1.0));
  float g_alt = a_data.z + a_vel.z * u_dtSec;
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
  // far side to hide) and on a degenerate plane (w must be < 0). Uses the
  // GLIDED position (g_xy/g_alt) — the cull must hide the point where it is
  // DRAWN, or objects would pop at the limb mid-glide.
  if (u_projection_transition > 0.999 && u_projection_clipping_plane.w < 0.0) {
    vec3 satPos = projectToSphere(g_xy) * (1.0 + g_alt / GLOBE_RADIUS);
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
  gl_Position = projectTileFor3D(g_xy, g_alt);
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
  private aVel = -1;
  private aShape = -1;
  private uSize: WebGLUniformLocation | null = null;
  private uDtSec: WebGLUniformLocation | null = null;
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
  // GLIDE: clock reading (this.now domain, ms) at which the current position
  // buffer was CURRENT — set by the parent via setTickTime() on each worker
  // positions message. null = glide not wired (parent never called it):
  // u_dtSec stays 0 and rendering is bit-identical to the pre-glide layer.
  private tickAnchorMs: number | null = null;
  // B3 SIM CLOCK: simulated seconds per real second for the glide (1 =
  // exact pre-B3 behavior; set by the parent only while the simulation
  // clock is warped). See glideDtSecWarp.
  private timeScaleK = 1;
  private now: () => number;
  // PACED GLIDE (pulse fix): at most ONE pending delayed repaint at a time —
  // scheduled by render() when per-frame motion is sub-pixel but per-tick
  // motion is not (see glideRepaintDelaySec). Cleared on onRemove.
  private glideTimer: unknown = null;
  private setTimeoutFn: (cb: () => void, ms: number) => unknown;
  private clearTimeoutFn: (handle: unknown) => void;
  // Last tick interval the position stream declared (updatePositions) — the
  // repaint horizon already covered by the tick path. null until a streaming
  // caller declares one (one-off updates keep the last known stream value).
  private lastTickIntervalSec: number | null = null;

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
    this.now = opts.now ?? (() => performance.now());
    this.setTimeoutFn = opts.setTimeoutFn ?? ((cb, ms) => setTimeout(cb, ms));
    this.clearTimeoutFn =
      opts.clearTimeoutFn ?? ((h) => clearTimeout(h as ReturnType<typeof setTimeout>));
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
    // GLIDE: capped SIMULATED seconds since the worker tick (0 when the
    // parent hasn't wired setTickTime — renders the raw tick positions,
    // pre-glide behavior). timeScaleK is 1 except under B3 time warp, where
    // glideDtSecWarp(…, 1) === glideDtSec — bit-identical at realtime.
    const dtSec = this.tickAnchorMs != null
      ? glideDtSecWarp(this.now(), this.tickAnchorMs, this.timeScaleK)
      : 0;
    if (this.uDtSec) gl.uniform1f(this.uDtSec, dtSec);
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
    // GLIDE velocity: floats [4..6] of the same interleaved buffer (byte
    // offset 16). Location can be -1 if the driver dead-strips it (u_dtSec
    // permanently 0 would allow that) — guard like a_shape.
    if (this.aVel >= 0) {
      gl.enableVertexAttribArray(this.aVel);
      gl.vertexAttribPointer(this.aVel, 3, gl.FLOAT, false, SAT_STRIDE * 4, 16);
    }

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
    if (this.aVel >= 0) gl.disableVertexAttribArray(this.aVel);
    if (this.aShape >= 0) gl.disableVertexAttribArray(this.aShape);

    // GLIDE self-repaint ("needsGlide"): while satellites are present, glide
    // is wired (tick anchor set), and motion is visible at this camera,
    // keep frames coming — the established self-animating custom-layer
    // pattern (triggerRepaint from render schedules exactly one more frame;
    // MapLibre coalesces). PULSE FIX (2026-07-18): the request is now PACED
    // by glideRepaintDelaySec instead of the old binary gate, which left a
    // dead band (zoom ~3.3-8.2 at the equator) where NO glide frame was ever
    // requested while per-tick motion was 1..30px — satellites moved in pure
    // 1 Hz snaps there (the reported rhythmic pulsing):
    //   delay 0    -> per-frame motion visible: request the next frame;
    //   delay d>0  -> per-frame motion sub-pixel but per-tick motion is not:
    //                 schedule ONE repaint for when the worst case crosses
    //                 ~a pixel (repaint rate scales with zoom, ~1..30 Hz);
    //   delay null -> even a tick of motion is sub-pixel: no glide repaint
    //                 owed; the tick-accumulation path still repaints when
    //                 drift becomes visible, and the map may idle (the perf
    //                 win of the 1Hz repaint-skip work, preserved).
    // dtSec at MAX_GLIDE_SEC (stale worker) stops the loop either way: the
    // glide is frozen at the honesty cap, so repainting would redraw an
    // identical frame; the next real tick's updatePositions restarts it.
    // Cost note (charter '1Hz orbital repaint reconsideration' — resolved):
    // repaint pressure exists ONLY while this layer has data, i.e. while
    // the orbital toggle is on; layer off/empty = zero repaint pressure.
    if (this.map && this.tickAnchorMs != null && dtSec < MAX_GLIDE_SEC) {
      const delay = glideRepaintDelaySec(
        this.map.getCenter().lat,
        this.map.getZoom(),
        this.lastTickIntervalSec,
      );
      if (delay === 0) {
        this.map.triggerRepaint();
      } else if (delay != null && this.glideTimer == null) {
        this.glideTimer = this.setTimeoutFn(() => {
          this.glideTimer = null;
          this.map?.triggerRepaint();
        }, delay * 1000);
      }
    }
  }

  onRemove(_map: MapLibreMap, gl: AnyGl): void {
    if (this.glideTimer != null) {
      this.clearTimeoutFn(this.glideTimer);
      this.glideTimer = null;
    }
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
    // Remember the stream's tick interval for the paced-glide horizon (a
    // one-off call between ticks — e.g. a filter repush — keeps the last
    // known stream value rather than forgetting it).
    if (tickIntervalSec != null && tickIntervalSec > 0) {
      this.lastTickIntervalSec = tickIntervalSec;
    }
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

  /**
   * GLIDE anchor: tell the layer WHEN (in this layer's clock — performance
   * .now() unless injected) the current position buffer was true. The parent
   * calls this on every worker `positions` message, right beside
   * updatePositions; each frame then renders tick position + velocity ×
   * elapsed-since-anchor (capped at MAX_GLIDE_SEC — see glideDtSec).
   * Omit the argument to anchor "now" (the normal case: message latency from
   * the worker is ~ms, negligible against a 1 s tick).
   *
   * DISPLAY-ONLY: the glide exists purely in the vertex shader. getPositions
   * / readSat / every CPU consumer (pick, follow, miniSelect, siteQuery)
   * keeps returning the exact-tick physics positions.
   */
  setTickTime(tMs?: number): void {
    this.tickAnchorMs = tMs ?? this.now();
    this.map?.triggerRepaint(); // (re)start the glide frame loop
  }

  /** Current glide anchor (ms in the layer clock) — wiring checks/tests. */
  getTickTime(): number | null {
    return this.tickAnchorMs;
  }

  /**
   * B3 SIM CLOCK: simulated-seconds-per-real-second for the shader glide.
   * The parent sets this alongside its worker-drive mode (1 while the
   * simulation clock is realtime — the exact pre-B3 render math; the
   * clock's rate under warp; 0 while paused). Non-finite/negative clamps
   * to 0 — a broken rate freezes rather than fabricates.
   */
  setTimeScale(k: number): void {
    const kk = Number.isFinite(k) && k > 0 ? k : 0;
    if (kk === this.timeScaleK) return;
    this.timeScaleK = kk;
    this.map?.triggerRepaint();
  }

  /** Current glide time scale (tests/wiring checks). */
  getTimeScale(): number {
    return this.timeScaleK;
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
  /** Mercator-mode complement of getGlobeProjection — screen picking must
   *  work there too (tilted terrain views: objects render displaced from
   *  their ground point by altitude; ground picking misses them). */
  getMercatorProjection(): Float32Array | null {
    if (!this.lastMainMatrix || this.lastTransition > 0.999) return null;
    return this.lastMainMatrix;
  }

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
    this.aVel = gl.getAttribLocation(p, 'a_vel');
    this.aShape = gl.getAttribLocation(p, 'a_shape');
    this.uSize = gl.getUniformLocation(p, 'u_size');
    this.uDtSec = gl.getUniformLocation(p, 'u_dtSec');
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
