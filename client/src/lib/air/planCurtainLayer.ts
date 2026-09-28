// PLANNED-ROUTE CURTAIN — the GRAY 3D curtain of where the selected
// aircraft is planned to fly NEXT (FLIGHT PROGRAM 2026-09-28, human vision
// items 2-4: "click a plane → a gray 3D curtain shows its planned route from
// where it is NOW to where it's going; as it flies, the gray ahead turns into
// the existing colored live curtain behind it").
//
// One MapLibre CustomLayerInterface built on flightTrackLayer's proven
// construction so the two curtains read as one object:
//   1. GROUND TRACE — thin gray ribbon draped on the ground (terrain +16 m).
//   2. THE CURTAIN — a world-space vertical strip: top edge = the planned
//      path at its planned altitude, bottom edge = the terrain MINUS the
//      drape overlap (the caller passes CURTAIN_BELOW_TERRAIN_M × exag with
//      terrain on, 0 with terrain off — exactly the live curtain's rule).
//      Double-sided (CULL_FACE off), depth-tested-never-written, and the
//      same terrain-on depth exemption as the live curtain.
//   3. TOP EDGE — a gray ribbon at the planned altitude. Segments whose
//      altitude is the server's ESTIMATED profile (altEstimated) are drawn
//      more transparent with a DASHED top edge — the plan's own honesty
//      flag, visible without clicking.
//   4. ORIGINAL PLAN (OFF_PLAN only) — the plan before re-planning, a faint
//      thin gray line at its altitude, no curtain: filed-vs-actual.
//
// THE SEAM ("gray turns into live"): the fixed plan geometry is built ONCE
// per plan (world space, one upload). Every frame — frameCore's rAF loop,
// never a map event (Law I) — the layer reads the SEAM SOURCE: the exact
// endpoint the live curtain's moving tail was last drawn to (datamap passes
// FlightTrackLayer.getTailEnd(), or the track's last sample when no tail is
// up). When it moved, the layer locates it on the plan (nearest segment,
// progress-monotonic), draws the fixed geometry only from the first plan
// vertex AHEAD of it (a sub-range drawElements — no rebuild), and rebuilds
// the tiny seam piece seam → that vertex (≤6 quads, bufferSubData). The part
// the plane has passed therefore disappears from the gray curtain exactly
// where the colored curtain grows: no gap, no overlap, by construction.
//
// A NEW plan (e.g. REPLANNED) crossfades over PLAN_CROSSFADE_MS (a lerp
// toward target alpha, stepped in the frame loop — Law I), first
// appearance fades in the same way. Far-side-of-globe fragments are culled
// with the same shader test as every other custom layer; the destination
// label is hidden by the same CPU occlusion test.
//
// Law IV: maxFeatures = PLAN_MAX_POINTS densified vertices (hard cap — the
// densifier never exceeds it and reports input decimation to the perf HUD),
// vramBudget derived below, dispose() frees every GL object, the frame
// registration and the label reference.

import type {
  CustomLayerInterface,
  CustomRenderMethodInput,
  Map as MapLibreMap,
} from 'maplibre-gl';
import { buildQuadIndices, FT_VERTS_PER_SEG, FT_INDICES_PER_SEG } from './flightTrackLayer.js';
import { TRACE_ABOVE_TERRAIN_M, distMeters } from './trackModel.js';
import { lonLatToMercator, mercatorToLonLat } from '../orbital/satBuffer.js';
import {
  cameraFromClippingPlane,
  earthOccludes,
  mercatorToSphere,
  mercatorZFromAltitude,
} from '../orbital/occlusion.js';
import { VT_PROJ_ELEV_GLSL } from '../glElev.js';
import { metersPerPixel } from '../lod.js';
import { bump, setGauge } from '../../render/perfMetrics.js';
import { frameCore, PRIORITY, type FrameLoop } from '../../render/frameCore.js';

type AnyGl = WebGLRenderingContext | WebGL2RenderingContext;

const reportedPlanErrors = new Set<string>();

/**
 * No silent swallows: every caught failure in the planned-route path is
 * COUNTED on the perf HUD (`planCurtain.err.<where>`) and logged ONCE per
 * site — a per-frame failure must never flood the console, and never vanish
 * either. The caller then degrades to its documented fallback.
 */
export function reportPlanError(where: string, err: unknown): void {
  bump(`planCurtain.err.${where}`);
  if (reportedPlanErrors.has(where)) return;
  reportedPlanErrors.add(where);
  // eslint-disable-next-line no-console
  console.warn(`[planCurtain] ${where} failed (counted; falling back):`, err);
}

// ── constants ───────────────────────────────────────────────────────────────

/** floats per vertex: flightTrackLayer's 13-float layout + 1 —
 *  [0..2] a_pos (mercX, mercY, z display meters)
 *  [3..5] a_other (ribbon's other endpoint; = self for wall vertices)
 *  [6..8] a_ext (side −1|0|+1, dirSign, widthPx — NEGATIVE width = dashed)
 *  [9..12] rgba
 *  [13] a_along (great-circle meters from the plan start — dash phase) */
export const PC_VERT_STRIDE = 14;

/** Law IV hard cap: densified plan vertices per plan (a 20,000 km route
 *  densifies to ~13 km spacing; a domestic hop to PLAN_MIN_SPACING_M). */
export const PLAN_MAX_POINTS = 1536;
/** Finest densification spacing (great-circle meters between vertices). */
export const PLAN_MIN_SPACING_M = 1500;
/** vertices reserved inside the cap for antimeridian split pairs. */
const AM_RESERVE = 16;
/** seam piece capacity: trace + curtain + edge, ×2 if it crosses the
 *  antimeridian. */
export const SEAM_MAX_QUADS = 6;

/** new plan / first appearance crossfade (brief: 250 ms). */
export const PLAN_CROSSFADE_MS = 250;

/** DESIGN.md --text-secondary (#b3c2d8): the theme's neutral cool gray —
 *  numeric channels, so the palette stays single-sourced in DESIGN.md. */
export const PLAN_GRAY: [number, number, number] = [0xb3 / 255, 0xc2 / 255, 0xd8 / 255];
/** DESIGN.md --text-tertiary (#6680a0): the dimmer gray for the ground
 *  trace and the original-plan line. */
export const PLAN_GRAY_DIM: [number, number, number] = [0x66 / 255, 0x80 / 255, 0xa0 / 255];

/** curtain fill alpha: planned altitude / ESTIMATED altitude (fainter —
 *  both below the live curtain's 34% so live always reads as primary). */
export const PLAN_CURTAIN_ALPHA = 0.24;
export const PLAN_CURTAIN_ALPHA_EST = 0.13;
/** bottom-vertex darkening (neutral: gray stays gray toward the ground). */
export const PLAN_BOTTOM_MUL = 0.3;
/** top edge: width px, alpha solid / estimated (estimated is also dashed). */
export const PLAN_EDGE_WIDTH_PX = 2.5;
export const PLAN_EDGE_ALPHA = 0.9;
export const PLAN_EDGE_ALPHA_EST = 0.6;
/** ground trace. */
export const PLAN_TRACE_WIDTH_PX = 1.5;
export const PLAN_TRACE_ALPHA = 0.7;
/** original (pre-replan) plan line. */
export const PLAN_ORIG_WIDTH_PX = 1.25;
export const PLAN_ORIG_ALPHA = 0.38;
/** dash period on screen (CSS px at the view center) and its "on" share. */
export const PLAN_DASH_PX = 12;
export const PLAN_DASH_ON = 0.6;

const D2R = Math.PI / 180;
const M_PER_DEG = 111_320;

const wrapOk = (x0: number, x1: number): boolean => Math.abs(x0 - x1) <= 0.5;
const wrapLonDeg = (d: number): number => ((((d + 180) % 360) + 360) % 360) - 180;

// ── densification (pure) ────────────────────────────────────────────────────

export interface PlanPointLike {
  lon: number;
  lat: number;
  altM: number | null;
  altEstimated?: boolean;
}

/** The plan densified along great circles — one entry per render vertex. */
export interface DensePlan {
  n: number;
  lon: Float64Array;
  lat: Float64Array;
  /** REAL meters MSL; NaN = no altitude (no curtain there — honest gap). */
  altM: Float32Array;
  /** 1 = the altitude here is the server's estimate (or interpolated from
   *  an estimated end) — drawn fainter and dashed. */
  est: Uint8Array;
  /** cumulative great-circle meters from the first vertex. */
  alongM: Float64Array;
  /** valid input points (finite, in-range positions). */
  inputCount: number;
  /** input points dropped to fit PLAN_MAX_POINTS (0 in practice; reported). */
  decimated: number;
}

const valid = (p: PlanPointLike | null | undefined): p is PlanPointLike =>
  !!p && Number.isFinite(p.lon) && Number.isFinite(p.lat) && Math.abs(p.lat) <= 90 && Math.abs(p.lon) <= 180;

function toVec(lon: number, lat: number): [number, number, number] {
  const la = lat * D2R, lo = lon * D2R;
  return [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)];
}

/**
 * Densify plan points along GREAT CIRCLES (the path an airway leg between
 * two fixes actually follows — a straight mercator line would be wrong on
 * a globe) to ~max(PLAN_MIN_SPACING_M, total / budget) spacing, never more
 * than `maxPoints` vertices. Altitude interpolates linearly between two
 * KNOWN ends (either end unknown → NaN, an honest gap); the estimated flag
 * is inherited from either end. Antimeridian crossings get an exact
 * split pair at lon ±180 (mercator x 1 | 0) so the seam there is zero-width
 * — flightTrackLayer's wrap rule then skips only that zero-length hop.
 */
export function densifyPlan(
  points: readonly (PlanPointLike | null | undefined)[],
  maxPoints: number = PLAN_MAX_POINTS,
  minSpacingM: number = PLAN_MIN_SPACING_M,
): DensePlan {
  let pts = points.filter(valid);
  const inputCount = pts.length;
  const cap = Math.max(2, maxPoints - AM_RESERVE);
  let decimated = 0;
  if (pts.length > cap) {
    // Law IV: keep first/last + evenly spaced waypoints, and SAY so
    const keep: PlanPointLike[] = [];
    for (let i = 0; i < cap; i++) keep.push(pts[Math.round((i * (pts.length - 1)) / (cap - 1))]);
    decimated = pts.length - cap;
    pts = keep;
  }
  setGauge('planCurtain.decimated', decimated);

  const oLon: number[] = [], oLat: number[] = [], oAlt: number[] = [], oEst: number[] = [];
  const pushRaw = (lon: number, lat: number, alt: number, est: boolean) => {
    oLon.push(lon); oLat.push(lat); oAlt.push(alt); oEst.push(est ? 1 : 0);
  };
  const push = (lon: number, lat: number, alt: number, est: boolean) => {
    const k = oLon.length;
    if (k > 0 && Math.abs(lon - oLon[k - 1]) > 180) {
      // antimeridian: exact split pair (only when the cap has room — past it
      // the one hop is simply skipped by the wrap rule, never mis-drawn)
      if (k + 3 <= maxPoints) {
        const pLon = oLon[k - 1], pLat = oLat[k - 1], pAlt = oAlt[k - 1];
        const east = lon < pLon; // e.g. 179 → -179 heads east across +180
        const qLon = east ? lon + 360 : lon - 360;
        const edge = east ? 180 : -180;
        const f = (edge - pLon) / (qLon - pLon);
        const cLat = pLat + (lat - pLat) * f;
        const cAlt = Number.isNaN(pAlt) || Number.isNaN(alt) ? NaN : pAlt + (alt - pAlt) * f;
        const cEst = est || oEst[k - 1] === 1;
        pushRaw(edge, cLat, cAlt, cEst);
        pushRaw(-edge, cLat, cAlt, cEst);
      }
    }
    if (oLon.length < maxPoints) pushRaw(lon, lat, alt, est);
  };

  if (pts.length >= 1) {
    const a0 = pts[0];
    push(a0.lon, a0.lat, a0.altM == null ? NaN : a0.altM, !!a0.altEstimated && a0.altM != null);
  }
  if (pts.length >= 2) {
    let total = 0;
    const legs: number[] = [];
    for (let i = 0; i + 1 < pts.length; i++) {
      const d = distMeters(pts[i].lat, pts[i].lon, pts[i + 1].lat, pts[i + 1].lon);
      legs.push(d);
      total += d;
    }
    const budget = cap - pts.length;
    const spacing = Math.max(minSpacingM, budget > 0 ? total / budget : Infinity);
    let used = 0;
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = pts[i], b = pts[i + 1];
      const aAlt = a.altM == null ? NaN : a.altM;
      const bAlt = b.altM == null ? NaN : b.altM;
      const est = (!!a.altEstimated && a.altM != null) || (!!b.altEstimated && b.altM != null);
      let steps = Number.isFinite(spacing) ? Math.floor(legs[i] / spacing) : 0;
      steps = Math.max(0, Math.min(steps, budget - used));
      used += steps;
      if (steps > 0) {
        const va = toVec(a.lon, a.lat), vb = toVec(b.lon, b.lat);
        const dot = Math.max(-1, Math.min(1, va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]));
        const om = Math.acos(dot);
        const so = Math.sin(om);
        for (let s = 1; s <= steps; s++) {
          const u = s / (steps + 1);
          let x: number, y: number, z: number;
          if (so > 1e-9) {
            const ka = Math.sin((1 - u) * om) / so, kb = Math.sin(u * om) / so;
            x = ka * va[0] + kb * vb[0]; y = ka * va[1] + kb * vb[1]; z = ka * va[2] + kb * vb[2];
          } else {
            x = va[0] + (vb[0] - va[0]) * u; y = va[1] + (vb[1] - va[1]) * u; z = va[2] + (vb[2] - va[2]) * u;
          }
          const lat = Math.atan2(z, Math.hypot(x, y)) / D2R;
          const lon = Math.atan2(y, x) / D2R;
          const alt = Number.isNaN(aAlt) || Number.isNaN(bAlt) ? NaN : aAlt + (bAlt - aAlt) * u;
          push(lon, lat, alt, est && !Number.isNaN(alt));
        }
      }
      push(b.lon, b.lat, bAlt, !!b.altEstimated && b.altM != null);
    }
  }

  const n = oLon.length;
  const along = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    along[i] = along[i - 1] + (Math.abs(oLon[i] - oLon[i - 1]) > 180
      ? 0 // the split pair is one meridian — zero distance
      : distMeters(oLat[i - 1], oLon[i - 1], oLat[i], oLon[i]));
  }
  return {
    n,
    lon: Float64Array.from(oLon),
    lat: Float64Array.from(oLat),
    altM: Float32Array.from(oAlt),
    est: Uint8Array.from(oEst),
    alongM: along,
    inputCount,
    decimated,
  };
}

// ── geometry (pure) ─────────────────────────────────────────────────────────

/** Everything the plan geometry builder needs, in the DISPLAY datum (the
 *  same datum the live curtain uses — datamap's displayAltReal rule:
 *  terrain on → MSL clamped above the displayed mesh, terrain off → height
 *  above the flat plane). */
export interface PlanGeomInput {
  dense: DensePlan;
  /** display altitude per dense vertex (NaN = no curtain/edge there). */
  altDisp: Float32Array;
  /** display ground per dense vertex. */
  groundZ: Float32Array;
  /** meters the curtain bottom sits below groundZ (the drape overlap). */
  drapeBelowM: number;
}

/** Draw groups, in draw order (the live curtain's order). */
export const PG_TRACE = 0;
export const PG_CURTAIN = 1;
export const PG_EDGE = 2;

export interface PlanGeometry {
  dense: DensePlan;
  merc: Float32Array;
  altDisp: Float32Array;
  groundZ: Float32Array;
  drapeBelowM: number;
  verts: Float32Array;
  indices: Uint32Array;
  nSeg: number;
  quads: number;
  /** per draw group: first quad index for segments ≥ s (length nSeg+1) —
   *  "draw from the plane forward" is one sub-range per group. */
  segQuad: [Uint32Array, Uint32Array, Uint32Array];
  /** per draw group: one past its last quad. */
  groupEnd: [number, number, number];
}

type RGBA = [number, number, number, number];

/** Vertex packer shared by the plan, seam and original-line builders. */
class Packer {
  out: Float32Array;
  o = 0;
  constructor(maxQuads: number) {
    this.out = new Float32Array(maxQuads * FT_VERTS_PER_SEG * PC_VERT_STRIDE);
  }
  get quads(): number { return this.o / (PC_VERT_STRIDE * FT_VERTS_PER_SEG); }
  put(x: number, y: number, z: number, ox: number, oy: number, oz: number,
      side: number, dir: number, w: number, c: RGBA, along: number): void {
    const a = this.out;
    a[this.o++] = x; a[this.o++] = y; a[this.o++] = z;
    a[this.o++] = ox; a[this.o++] = oy; a[this.o++] = oz;
    a[this.o++] = side; a[this.o++] = dir; a[this.o++] = w;
    a[this.o++] = c[0]; a[this.o++] = c[1]; a[this.o++] = c[2]; a[this.o++] = c[3];
    a[this.o++] = along;
  }
  /** screen-extruded ribbon (arc pattern); negative w = dashed. */
  ribbon(ax: number, ay: number, az: number, bx: number, by: number, bz: number,
         w: number, c: RGBA, alongA: number, alongB: number): void {
    this.put(ax, ay, az, bx, by, bz, -1, +1, w, c, alongA);
    this.put(ax, ay, az, bx, by, bz, +1, +1, w, c, alongA);
    this.put(bx, by, bz, ax, ay, az, -1, -1, w, c, alongB);
    this.put(bx, by, bz, ax, ay, az, +1, -1, w, c, alongB);
  }
  /** world-space wall quad [top_a, bottom_a, top_b, bottom_b]. */
  wall(ax: number, ay: number, topA: number, botA: number,
       bx: number, by: number, topB: number, botB: number,
       est: boolean, alongA: number, alongB: number): void {
    const alpha = est ? PLAN_CURTAIN_ALPHA_EST : PLAN_CURTAIN_ALPHA;
    const top: RGBA = [PLAN_GRAY[0], PLAN_GRAY[1], PLAN_GRAY[2], alpha];
    const bot: RGBA = [PLAN_GRAY[0] * PLAN_BOTTOM_MUL, PLAN_GRAY[1] * PLAN_BOTTOM_MUL, PLAN_GRAY[2] * PLAN_BOTTOM_MUL, alpha];
    this.put(ax, ay, topA, ax, ay, topA, 0, 0, 0, top, alongA);
    this.put(ax, ay, botA, ax, ay, botA, 0, 0, 0, bot, alongA);
    this.put(bx, by, topB, bx, by, topB, 0, 0, 0, top, alongB);
    this.put(bx, by, botB, bx, by, botB, 0, 0, 0, bot, alongB);
  }
  done(): Float32Array {
    return this.o === this.out.length ? this.out : this.out.slice(0, this.o);
  }
}

const TRACE_RGBA_PLAN: RGBA = [PLAN_GRAY_DIM[0], PLAN_GRAY_DIM[1], PLAN_GRAY_DIM[2], PLAN_TRACE_ALPHA];
const edgeRgba = (est: boolean): RGBA =>
  [PLAN_GRAY[0], PLAN_GRAY[1], PLAN_GRAY[2], est ? PLAN_EDGE_ALPHA_EST : PLAN_EDGE_ALPHA];
const edgeWidth = (est: boolean): number => (est ? -PLAN_EDGE_WIDTH_PX : PLAN_EDGE_WIDTH_PX);

/** a segment is "estimated" when either end's altitude is. */
const segEst = (d: DensePlan, i: number, j: number): boolean => d.est[i] === 1 || d.est[j] === 1;

/**
 * Pure: the whole plan → ONE packed buffer, grouped trace | curtain | edge
 * (the live curtain's draw order), each group ordered by segment so the
 * layer can draw "from the plane forward" as one sub-range per group.
 * Segment rules match flightTrackLayer: the trace draws wherever the
 * position is real (through altitude gaps); curtain + edge need BOTH ends'
 * altitude; nothing spans the antimeridian.
 */
export function buildPlanGeometry(input: PlanGeomInput): PlanGeometry {
  const { dense, altDisp, groundZ } = input;
  const n = Math.min(dense.n, altDisp.length, groundZ.length);
  const merc = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const m = lonLatToMercator(dense.lon[i], dense.lat[i]);
    merc[i * 2] = m.x;
    merc[i * 2 + 1] = m.y;
  }
  const nSeg = Math.max(0, n - 1);
  const pk = new Packer(nSeg * 3);
  const segQuad: [Uint32Array, Uint32Array, Uint32Array] =
    [new Uint32Array(nSeg + 1), new Uint32Array(nSeg + 1), new Uint32Array(nSeg + 1)];
  const groupEnd: [number, number, number] = [0, 0, 0];
  const drop = input.drapeBelowM;
  const lift = TRACE_ABOVE_TERRAIN_M;
  const A = dense.alongM;

  // 1) ground trace
  for (let s = 0; s < nSeg; s++) {
    segQuad[PG_TRACE][s] = pk.quads;
    const ax = merc[s * 2], ay = merc[s * 2 + 1], bx = merc[s * 2 + 2], by = merc[s * 2 + 3];
    if (!wrapOk(ax, bx)) continue;
    pk.ribbon(ax, ay, groundZ[s] + lift, bx, by, groundZ[s + 1] + lift,
      PLAN_TRACE_WIDTH_PX, TRACE_RGBA_PLAN, A[s], A[s + 1]);
  }
  segQuad[PG_TRACE][nSeg] = groupEnd[PG_TRACE] = pk.quads;
  // 2) curtain
  for (let s = 0; s < nSeg; s++) {
    segQuad[PG_CURTAIN][s] = pk.quads;
    if (Number.isNaN(altDisp[s]) || Number.isNaN(altDisp[s + 1])) continue; // honest gap
    const ax = merc[s * 2], ay = merc[s * 2 + 1], bx = merc[s * 2 + 2], by = merc[s * 2 + 3];
    if (!wrapOk(ax, bx)) continue;
    pk.wall(ax, ay, altDisp[s], groundZ[s] - drop, bx, by, altDisp[s + 1], groundZ[s + 1] - drop,
      segEst(dense, s, s + 1), A[s], A[s + 1]);
  }
  segQuad[PG_CURTAIN][nSeg] = groupEnd[PG_CURTAIN] = pk.quads;
  // 3) top edge
  for (let s = 0; s < nSeg; s++) {
    segQuad[PG_EDGE][s] = pk.quads;
    if (Number.isNaN(altDisp[s]) || Number.isNaN(altDisp[s + 1])) continue;
    const ax = merc[s * 2], ay = merc[s * 2 + 1], bx = merc[s * 2 + 2], by = merc[s * 2 + 3];
    if (!wrapOk(ax, bx)) continue;
    const est = segEst(dense, s, s + 1);
    pk.ribbon(ax, ay, altDisp[s], bx, by, altDisp[s + 1], edgeWidth(est), edgeRgba(est), A[s], A[s + 1]);
  }
  segQuad[PG_EDGE][nSeg] = groupEnd[PG_EDGE] = pk.quads;

  const verts = pk.done();
  const quads = verts.length / (PC_VERT_STRIDE * FT_VERTS_PER_SEG);
  return {
    dense, merc, altDisp, groundZ, drapeBelowM: drop,
    verts,
    indices: buildQuadIndices(quads),
    nSeg, quads, segQuad, groupEnd,
  };
}

/** Pure: the ORIGINAL (pre-replan) plan as one faint thin gray line at its
 *  planned altitude (ground + trace lift where the altitude is unknown) —
 *  no curtain. Same wrap rule. */
export function buildOriginalLineVertices(
  dense: DensePlan, altDisp: Float32Array, groundZ: Float32Array,
): Float32Array {
  const n = Math.min(dense.n, altDisp.length, groundZ.length);
  const pk = new Packer(Math.max(0, n - 1));
  const c: RGBA = [PLAN_GRAY_DIM[0], PLAN_GRAY_DIM[1], PLAN_GRAY_DIM[2], PLAN_ORIG_ALPHA];
  let px = 0, py = 0, pz = 0;
  for (let i = 0; i < n; i++) {
    const m = lonLatToMercator(dense.lon[i], dense.lat[i]);
    const z = Number.isNaN(altDisp[i]) ? groundZ[i] + TRACE_ABOVE_TERRAIN_M : altDisp[i];
    if (i > 0 && wrapOk(px, m.x)) {
      pk.ribbon(px, py, pz, m.x, m.y, z, PLAN_ORIG_WIDTH_PX, c, dense.alongM[i - 1], dense.alongM[i]);
    }
    px = m.x; py = m.y; pz = z;
  }
  return pk.done();
}

// ── locating the plane on the plan (pure) ──────────────────────────────────

/** The live curtain's drawn endpoint, display datum. */
export interface PlanSeam {
  mercX: number;
  mercY: number;
  /** display altitude (NaN = unknown → the seam piece has no curtain). */
  altM: number;
  groundZ: number;
}

export interface PlanLocation {
  /** first dense vertex AHEAD of the plane (0 = before the plan start,
   *  n = past the destination → nothing left to draw). */
  ahead: number;
  /** distance from the plane to the drawn plan, meters (cross-track). */
  crossTrackM: number;
  /** along-route meters of the plane's projection onto the plan. */
  projAlongM: number;
  /** the nearest segment. */
  seg: number;
}

/**
 * Where is the plane on the plan? Nearest segment in a local
 * equirectangular frame centred on the plane (exact to <1% within a few
 * hundred km — the only distances where "nearest" is contested), with
 * PROGRESS-MONOTONIC preference: when a segment near the previous result
 * (`hint`) is within max(2 km, 1.25×) of the global best, it wins — a route
 * that doubles back must not make the seam jump to a parallel leg.
 */
export function locateOnPlan(dense: DensePlan, lon: number, lat: number, hint = -1): PlanLocation | null {
  const n = dense.n;
  if (n < 2 || !Number.isFinite(lon) || !Number.isFinite(lat)) return null;
  const cosLat = Math.max(0.01, Math.cos(lat * D2R));
  const L = dense.lon, La = dense.lat;
  let gBest = -1, gD = Infinity, gT = 0, gRaw = 0;
  let hBest = -1, hD = Infinity, hT = 0, hRaw = 0;
  const lo = hint >= 0 ? Math.max(0, hint - 3) : -1;
  const hi = hint >= 0 ? Math.min(n - 2, hint + 64) : -2;
  for (let s = 0; s + 1 < n; s++) {
    if (Math.abs(L[s + 1] - L[s]) > 180) continue; // the antimeridian split pair
    const ax = wrapLonDeg(L[s] - lon) * cosLat * M_PER_DEG, ay = (La[s] - lat) * M_PER_DEG;
    const bx = wrapLonDeg(L[s + 1] - lon) * cosLat * M_PER_DEG, by = (La[s + 1] - lat) * M_PER_DEG;
    const dx = bx - ax, dy = by - ay;
    const l2 = dx * dx + dy * dy;
    const tRaw = l2 > 0 ? -(ax * dx + ay * dy) / l2 : 0;
    const t = tRaw < 0 ? 0 : tRaw > 1 ? 1 : tRaw;
    const d = Math.hypot(ax + t * dx, ay + t * dy);
    if (d < gD) { gD = d; gBest = s; gT = t; gRaw = tRaw; }
    if (s >= lo && s <= hi && d < hD) { hD = d; hBest = s; hT = t; hRaw = tRaw; }
  }
  if (gBest < 0) return null;
  let s = gBest, t = gT, tRaw = gRaw, d = gD;
  if (hBest >= 0 && hD <= Math.max(2000, gD * 1.25)) { s = hBest; t = hT; tRaw = hRaw; d = hD; }
  const A = dense.alongM;
  const projAlongM = A[s] + t * (A[s + 1] - A[s]);
  let ahead: number;
  if (s === 0 && tRaw < 0) ahead = 0;
  else {
    ahead = s + 1;
    while (ahead < n && A[ahead] <= projAlongM + 1e-6) ahead++;
  }
  return { ahead, crossTrackM: d, projAlongM, seg: s };
}

/**
 * Pure: the seam piece — live curtain end → the first plan vertex ahead,
 * trace + curtain + edge (≤3 quads; ≤6 when it crosses the antimeridian,
 * split exactly at the meridian). Empty when the plane is past the end.
 */
export function buildSeamVertices(seam: PlanSeam, geom: PlanGeometry, loc: PlanLocation): Float32Array {
  const d = geom.dense;
  const j = loc.ahead;
  const n = Math.min(d.n, geom.altDisp.length);
  if (!(j >= 0 && j < n)) return new Float32Array(0);
  const est = j > 0 ? segEst(d, j - 1, j) : d.est[0] === 1;
  const B = {
    x: geom.merc[j * 2], y: geom.merc[j * 2 + 1],
    alt: geom.altDisp[j], g: geom.groundZ[j], along: d.alongM[j],
  };
  const S = { x: seam.mercX, y: seam.mercY, alt: seam.altM, g: seam.groundZ, along: loc.projAlongM };
  const pieces: [typeof S, typeof S][] = [];
  if (wrapOk(S.x, B.x)) pieces.push([S, B]);
  else {
    const bxU = B.x + (B.x < S.x ? 1 : -1); // unwrapped across the seam
    const edge = bxU > 1 ? 1 : 0;
    const f = (edge - S.x) / (bxU - S.x);
    const mid = (x: number) => ({
      x, y: S.y + (B.y - S.y) * f,
      alt: S.alt + (B.alt - S.alt) * f, g: S.g + (B.g - S.g) * f,
      along: S.along + (B.along - S.along) * f,
    });
    pieces.push([S, mid(edge)], [mid(1 - edge), B]);
  }
  const pk = new Packer(SEAM_MAX_QUADS);
  const drop = geom.drapeBelowM;
  for (const [a, b] of pieces) {
    pk.ribbon(a.x, a.y, a.g + TRACE_ABOVE_TERRAIN_M, b.x, b.y, b.g + TRACE_ABOVE_TERRAIN_M,
      PLAN_TRACE_WIDTH_PX, TRACE_RGBA_PLAN, a.along, b.along);
  }
  for (const [a, b] of pieces) {
    if (Number.isNaN(a.alt) || Number.isNaN(b.alt)) continue;
    pk.wall(a.x, a.y, a.alt, a.g - drop, b.x, b.y, b.alt, b.g - drop, est, a.along, b.along);
  }
  for (const [a, b] of pieces) {
    if (Number.isNaN(a.alt) || Number.isNaN(b.alt)) continue;
    pk.ribbon(a.x, a.y, a.alt, b.x, b.y, b.alt, edgeWidth(est), edgeRgba(est), a.along, b.along);
  }
  return pk.done();
}

/** Smoothstep ease for the crossfade (0→1, zero slope at both ends). */
export function crossfadeEase(u: number): number {
  const x = u <= 0 ? 0 : u >= 1 ? 1 : u;
  return x * x * (3 - 2 * x);
}

/**
 * CPU projection of a mercator point + display altitude (meters) to CSS
 * pixels with LAST FRAME's matrix — the flightTrackLayer.projectToScreen
 * rule (globe: sphere position; mercator: mercator-unit z). null when behind
 * the camera or off screen.
 */
export function projectMercToScreen(
  m: ArrayLike<number>, transition: number,
  mercX: number, mercY: number, zMeters: number,
  widthPx: number, heightPx: number,
): { x: number; y: number } | null {
  const p: readonly number[] = transition > 0.999
    ? mercatorToSphere(mercX, mercY, zMeters)
    : [mercX, mercY, mercatorZFromAltitude(zMeters, mercY)];
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
  if (!(w > 0)) return null;
  const cx = (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w;
  const cy = (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w;
  if (cx < -1 || cx > 1 || cy < -1 || cy > 1) return null;
  return { x: ((cx + 1) / 2) * widthPx, y: ((1 - cy) / 2) * heightPx };
}

// ── shaders ─────────────────────────────────────────────────────────────────

/** flightTrackLayer's FT_VERT_SRC (identical projection, extrusion and
 *  whole-transition far-side cull) + a crossfade alpha uniform and the
 *  dash varyings. Exported for planCurtainLayer.test.ts. */
export const PC_VERT_SRC = (prelude: string, define: string): string => `#version 300 es
${prelude}
${define}
${VT_PROJ_ELEV_GLSL}
in vec3 a_pos;
in vec3 a_other;
in vec3 a_ext;    // x: side (-1|0|+1; 0 = world-space wall vertex), y: dirSign, z: widthPx (<0 = dashed)
in vec4 a_color;
in float a_along; // great-circle meters from the plan start
uniform vec2 u_viewport;
uniform float u_alpha;     // crossfade multiplier
uniform float u_alongRef;  // the plane's along-route meters (keeps the dash phase small)
out vec4 v_color;
out float v_cull;
out float v_edge;
out highp float v_along;
out float v_dash;
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
  float widthPx = abs(a_ext.z);
  vec2 offs = normalPx * (widthPx * 0.5) * 2.0 / u_viewport;
  gl_Position = self + vec4(offs * self.w, 0.0, 0.0);
  v_color = vec4(a_color.rgb, a_color.a * u_alpha);
  v_edge = a_ext.x;
  v_along = a_along - u_alongRef;
  v_dash = a_ext.z < 0.0 ? 1.0 : 0.0;
}`;

/** FT_FRAG_SRC + screen-scaled dashing for estimated top edges. */
export const PC_FRAG_SRC = `#version 300 es
precision mediump float;
in vec4 v_color;
in float v_cull;
in float v_edge;
in highp float v_along;
in float v_dash;
uniform highp float u_dashM; // dash period in meters at the view center
out vec4 o;
void main() {
  if (v_cull > 0.01) discard;
  if (v_dash > 0.5 && fract(v_along / u_dashM) > ${PLAN_DASH_ON.toFixed(2)}) discard;
  o = v_color;
  o.a *= mix(1.0, 0.55, abs(v_edge));
}`;

// ── the layer ───────────────────────────────────────────────────────────────

interface Slot {
  geom: PlanGeometry;
  buf: WebGLBuffer | null;
  ibuf: WebGLBuffer | null;
  dirty: boolean;
  /** first dense vertex ahead of the plane; −1 = not located (not drawn). */
  ahead: number;
}

/** A DOM label anchored at a world point (the destination), positioned in
 *  render() from the same matrix the frame drew with. */
export interface PlanLabelAnchor {
  el: { style: { transform: string; display: string } };
  mercX: number;
  mercY: number;
  /** display meters. */
  z: number;
}

const seamEq = (a: PlanSeam | null, b: PlanSeam | null): boolean => {
  if (a === b) return true;
  if (!a || !b) return false;
  const same = (x: number, y: number) => x === y || (Number.isNaN(x) && Number.isNaN(y));
  return same(a.mercX, b.mercX) && same(a.mercY, b.mercY) && same(a.altM, b.altM) && same(a.groundZ, b.groundZ);
};

export class PlanCurtainLayer implements CustomLayerInterface {
  readonly id: string;
  readonly type = 'custom' as const;
  readonly renderingMode = '2d' as const;

  private map: MapLibreMap | null = null;
  private glRef: AnyGl | null = null;
  private readonly loop: FrameLoop | null;
  private unregisterFrame: (() => void) | null = null;

  private program: WebGLProgram | null = null;
  private cachedVariant: string | null = null;
  private aPos = -1; private aOther = -1; private aExt = -1; private aColor = -1; private aAlong = -1;
  private u: Record<string, WebGLUniformLocation | null> = {};
  private proj: Record<string, WebGLUniformLocation | null> = {};

  private cur: Slot | null = null;
  private prev: Slot | null = null;
  private fade = 1;
  private orig: { verts: Float32Array; indices: Uint32Array; buf: WebGLBuffer | null; ibuf: WebGLBuffer | null; dirty: boolean } | null = null;

  private seamSource: (() => PlanSeam | null) | null = null;
  private lastSeam: PlanSeam | null = null;
  private seamStale = true;
  private seamVerts: Float32Array | null = null;
  private seamDirty = false;
  private seamBuf: WebGLBuffer | null = null;
  private seamIBuf: WebGLBuffer | null = null;
  private hint = -1;
  private lastLoc: PlanLocation | null = null;
  private onLocate: ((loc: PlanLocation | null) => void) | null = null;
  private onFrame: (() => void) | null = null;

  private garbage: WebGLBuffer[] = [];
  private label: PlanLabelAnchor | null = null;
  private labelXY = '';

  private failStreak = 0;
  private static readonly MAX_FAIL_STREAK = 5;
  private disposed = false;

  constructor(opts: { id?: string; loop?: FrameLoop | null } = {}) {
    this.id = opts.id ?? 'flight-plan-curtain';
    this.loop = opts.loop === undefined ? null : opts.loop;
  }

  // ── lifecycle ────────────────────────────────────────────────────────────

  onAdd(map: MapLibreMap, gl: AnyGl): void {
    this.map = map;
    this.glRef = gl;
  }

  onRemove(_map: MapLibreMap, gl: AnyGl): void {
    this.dropGlObjects(gl);
    this.map = null;
  }

  /** Law IV explicit teardown: every GL object, the frame registration,
   *  every CPU buffer and the label reference. Idempotent. */
  dispose(): void {
    this.disposed = true;
    this.unregisterFrame?.();
    this.unregisterFrame = null;
    const gl = this.glRef;
    this.glRef = null;
    if (gl) this.onRemove(null as unknown as MapLibreMap, gl);
    this.hideLabel();
    this.cur = this.prev = null;
    this.orig = null;
    this.seamVerts = null;
    this.lastSeam = null;
    this.lastLoc = null;
    this.garbage = [];
    this.label = null;
    this.seamSource = null;
    this.onLocate = null;
    this.onFrame = null;
    this.map = null;
    setGauge('planCurtain.verts', 0);
  }

  /** Forget every GL handle (deletes on a lost context are no-ops) so the
   *  next render rebuilds cleanly. */
  private dropGlObjects(gl?: AnyGl): void {
    const bufs: (WebGLBuffer | null)[] = [
      this.seamBuf, this.seamIBuf,
      this.cur?.buf ?? null, this.cur?.ibuf ?? null,
      this.prev?.buf ?? null, this.prev?.ibuf ?? null,
      this.orig?.buf ?? null, this.orig?.ibuf ?? null,
      ...this.garbage,
    ];
    try {
      if (gl) {
        if (this.program) gl.deleteProgram(this.program);
        for (const b of bufs) if (b) gl.deleteBuffer(b);
      }
    } catch (e) {
      reportPlanError('gl-delete', e); // context gone — the handles are invalid anyway
    }
    this.program = null;
    this.cachedVariant = null;
    this.garbage = [];
    this.seamBuf = this.seamIBuf = null;
    for (const s of [this.cur, this.prev]) if (s) { s.buf = s.ibuf = null; s.dirty = true; }
    if (this.orig) { this.orig.buf = this.orig.ibuf = null; this.orig.dirty = true; }
    this.seamDirty = this.seamVerts != null;
  }

  private ensureFrame(): void {
    if (this.unregisterFrame || this.disposed) return;
    const loop = this.loop ?? frameCore();
    this.unregisterFrame = loop.register((dt) => this.frame(dt), PRIORITY.SIM, { label: 'planCurtain' });
  }

  // ── inputs ───────────────────────────────────────────────────────────────

  /** Where the live curtain currently ends (read every frame). */
  setSeamSource(fn: (() => PlanSeam | null) | null): void {
    this.seamSource = fn;
    this.seamStale = true;
    if (fn) this.ensureFrame();
  }

  /** Called whenever the plane is re-located on the plan (null = not
   *  located) — the controller's >10 nm re-fetch trigger. */
  setOnLocate(fn: ((loc: PlanLocation | null) => void) | null): void {
    this.onLocate = fn;
  }

  /** Extra per-frame hook (the controller's cheap datum-change check). */
  setOnFrame(fn: (() => void) | null): void {
    this.onFrame = fn;
  }

  /** Install a new plan (null clears). A replaced plan crossfades out while
   *  the new one fades in; geometry is uploaded once. `crossfade: false`
   *  swaps the geometry in place (a datum refinement of the SAME plan — a
   *  late DEM tile — is not a new plan and must not re-fade). The seam is
   *  located synchronously, so no frame ever draws the new slot unplaced. */
  setPlan(geom: PlanGeometry | null, opts: { crossfade?: boolean } = {}): void {
    if (this.disposed) return;
    if (!geom || geom.dense.n < 2) {
      this.retire(this.cur);
      this.retire(this.prev);
      this.cur = this.prev = null;
      this.seamVerts = null;
      this.lastLoc = null;
      this.hint = -1;
      this.map?.triggerRepaint();
      return;
    }
    const crossfade = opts.crossfade !== false;
    if (this.cur && crossfade) {
      this.retire(this.prev);
      // the outgoing plan only fades if it was actually visible
      this.prev = this.cur.ahead >= 0 ? this.cur : null;
      if (!this.prev) this.retire(this.cur);
      this.fade = 0;
    } else if (this.cur) {
      this.retire(this.cur); // in-place swap: fade state carries over
    } else {
      this.fade = 0;
    }
    this.cur = { geom, buf: null, ibuf: null, dirty: true, ahead: -1 };
    this.hint = -1;
    this.failStreak = 0; // new data re-arms rendering after transient GL failures
    const s = this.seamSource ? this.seamSource() : null;
    this.lastSeam = s ? { ...s } : null;
    this.seamStale = false;
    this.updateSeam(s);
    this.ensureFrame();
    setGauge('planCurtain.verts', this.getVertexCount()); // ?perf=1 HUD
    this.map?.triggerRepaint();
  }

  /** The original (pre-replan) plan line, or null. */
  setOriginal(verts: Float32Array | null): void {
    if (this.orig) this.garbage.push(...[this.orig.buf, this.orig.ibuf].filter((b): b is WebGLBuffer => !!b));
    this.orig = verts && verts.length
      ? { verts, indices: buildQuadIndices(verts.length / (PC_VERT_STRIDE * FT_VERTS_PER_SEG)), buf: null, ibuf: null, dirty: true }
      : null;
    this.map?.triggerRepaint();
  }

  /** Anchor a DOM label (the destination) — positioned every drawn frame. */
  setLabel(label: PlanLabelAnchor | null): void {
    if (this.label && (!label || label.el !== this.label.el)) this.hideLabel();
    this.label = label;
    this.labelXY = '';
    this.map?.triggerRepaint();
  }

  private retire(s: Slot | null): void {
    if (!s) return;
    if (s.buf) this.garbage.push(s.buf);
    if (s.ibuf) this.garbage.push(s.ibuf);
    s.buf = s.ibuf = null;
  }

  // ── read-outs (tests / controller / harness) ─────────────────────────────

  getLocation(): PlanLocation | null { return this.lastLoc; }
  getFade(): number { return this.fade; }
  hasPlan(): boolean { return this.cur != null; }
  getRenderFailed(): boolean { return this.failStreak >= PlanCurtainLayer.MAX_FAIL_STREAK; }
  /** Vertices currently held for drawing (plan slots + seam + original). */
  getVertexCount(): number {
    const v = (a: Float32Array | null | undefined) => (a ? a.length / PC_VERT_STRIDE : 0);
    return v(this.cur?.geom.verts) + v(this.prev?.geom.verts) + v(this.seamVerts) + v(this.orig?.verts);
  }
  isFrameRegistered(): boolean { return this.unregisterFrame != null; }

  // ── the frame (frameCore rAF — Law I) ────────────────────────────────────

  /** One frame: re-seam if the live end moved, advance the crossfade.
   *  Public for the headless test (production reaches it via frameCore). */
  frame(dtMs: number): void {
    if (this.disposed) return;
    try { this.onFrame?.(); } catch (e) { reportPlanError('frame-hook', e); }
    const cur = this.cur;
    if (!cur) return;
    let repaint = false;
    const s = this.seamSource ? this.seamSource() : null;
    if (this.seamStale || !seamEq(s, this.lastSeam)) {
      this.lastSeam = s ? { ...s } : null;
      this.seamStale = false;
      this.updateSeam(s);
      repaint = true;
    }
    if (cur.ahead >= 0 && this.fade < 1) {
      // lerp toward the target alpha at a fixed rate: PLAN_CROSSFADE_MS end to end
      this.fade = Math.min(1, this.fade + Math.max(0, dtMs) / PLAN_CROSSFADE_MS);
      if (this.fade >= 1) { this.retire(this.prev); this.prev = null; }
      repaint = true;
    }
    if (repaint) this.map?.triggerRepaint();
  }

  private updateSeam(s: PlanSeam | null): void {
    const cur = this.cur;
    if (!cur) return;
    let loc: PlanLocation | null = null;
    if (s && Number.isFinite(s.mercX) && Number.isFinite(s.mercY)) {
      const ll = mercatorToLonLat(s.mercX, s.mercY);
      loc = locateOnPlan(cur.geom.dense, ll.lonDeg, ll.latDeg, this.hint);
    }
    this.lastLoc = loc;
    try { this.onLocate?.(loc); } catch (e) { reportPlanError('locate-hook', e); }
    if (!s || !loc) {
      cur.ahead = -1; // no live end yet → the gray waits for the live curtain
      this.seamVerts = null;
      return;
    }
    this.hint = loc.ahead;
    cur.ahead = loc.ahead;
    const v = buildSeamVertices(s, cur.geom, loc);
    this.seamVerts = v.length ? v : null;
    this.seamDirty = this.seamVerts != null;
  }

  // ── render ───────────────────────────────────────────────────────────────

  render(gl: AnyGl, args: CustomRenderMethodInput): void {
    if (this.failStreak >= PlanCurtainLayer.MAX_FAIL_STREAK) return;
    const curOn = !!this.cur && this.cur.ahead >= 0;
    if (!curOn && !this.prev && this.garbage.length === 0) { this.hideLabel(); return; }
    try {
      this.renderInner(gl as WebGL2RenderingContext, args, curOn);
      this.failStreak = 0;
    } catch (e) {
      this.failStreak++;
      this.dropGlObjects(gl);
      // eslint-disable-next-line no-console
      console.error(
        `PlanCurtainLayer: render failure ${this.failStreak}/${PlanCurtainLayer.MAX_FAIL_STREAK} ` +
        '(GL objects dropped for a clean retry; map continues):', e);
    }
  }

  private renderInner(gl: WebGL2RenderingContext, args: CustomRenderMethodInput, curOn: boolean): void {
    if (this.garbage.length) {
      for (const b of this.garbage) gl.deleteBuffer(b);
      this.garbage = [];
    }
    if (!curOn && !this.prev) { this.hideLabel(); return; }
    const sd = args.shaderData;
    if (this.program == null || this.cachedVariant !== sd.variantName) {
      this.compile(gl, sd.vertexShaderPrelude, sd.define, sd.variantName);
      for (const s of [this.cur, this.prev]) if (s) s.dirty = true;
      if (this.orig) this.orig.dirty = true;
      this.seamDirty = this.seamVerts != null;
    }
    if (this.program == null) return;

    // the live curtain's GL state, verbatim (THE CRITICAL FIX b + c)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const terrainOn = !!(this.map && (this.map as unknown as { getTerrain?: () => unknown }).getTerrain?.());
    if (terrainOn) gl.disable(gl.DEPTH_TEST);
    else gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(false);
    gl.depthRange(0, 1);
    gl.disable(gl.CULL_FACE);

    gl.useProgram(this.program);
    this.bindProjection(gl, args);
    if (this.u.viewport) gl.uniform2f(this.u.viewport, gl.drawingBufferWidth || 1, gl.drawingBufferHeight || 1);
    if (this.u.dashM) gl.uniform1f(this.u.dashM, this.dashPeriodM());
    if (this.u.alongRef) gl.uniform1f(this.u.alongRef, this.lastLoc?.projAlongM ?? 0);

    const e = crossfadeEase(this.fade);
    if (this.prev && this.prev.ahead >= 0) this.drawSlot(gl, this.prev, 1 - e);
    if (curOn && this.cur) {
      if (this.orig) {
        if (!this.orig.buf) this.orig.buf = gl.createBuffer();
        if (!this.orig.ibuf) this.orig.ibuf = gl.createBuffer();
        this.setAlpha(gl, e);
        this.bindAndMaybeUpload(gl, this.orig.buf, this.orig.ibuf, this.orig.verts, this.orig.indices, this.orig.dirty, false);
        this.orig.dirty = false;
        gl.drawElements(gl.TRIANGLES, this.orig.indices.length, gl.UNSIGNED_INT, 0);
      }
      this.drawSlot(gl, this.cur, e);
      if (this.seamVerts && this.seamVerts.length) this.drawSeam(gl, e);
    }
    gl.disableVertexAttribArray(this.aPos);
    gl.disableVertexAttribArray(this.aOther);
    gl.disableVertexAttribArray(this.aExt);
    gl.disableVertexAttribArray(this.aColor);
    if (this.aAlong >= 0) gl.disableVertexAttribArray(this.aAlong);

    this.positionLabel(args, curOn);
  }

  private dashPeriodM(): number {
    try {
      const m = this.map as unknown as { getCenter?: () => { lat: number }; getZoom?: () => number } | null;
      const c = m?.getCenter?.();
      const z = m?.getZoom?.();
      if (c && Number.isFinite(c.lat) && Number.isFinite(z)) {
        return Math.max(1, PLAN_DASH_PX * metersPerPixel(c.lat, z as number));
      }
    } catch (e) {
      reportPlanError('dash-period', e); // a fixed 1 km period still reads as dashed
    }
    return 1000;
  }

  private setAlpha(gl: WebGL2RenderingContext, a: number): void {
    if (this.u.alpha) gl.uniform1f(this.u.alpha, a);
  }

  private bindAndMaybeUpload(
    gl: WebGL2RenderingContext, buf: WebGLBuffer | null, ibuf: WebGLBuffer | null,
    verts: Float32Array, indices: Uint32Array, upload: boolean, dynamic: boolean,
  ): void {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibuf);
    if (upload) {
      if (dynamic) gl.bufferSubData(gl.ARRAY_BUFFER, 0, verts);
      else {
        gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
      }
    }
    const strideB = PC_VERT_STRIDE * 4;
    gl.enableVertexAttribArray(this.aPos);
    gl.vertexAttribPointer(this.aPos, 3, gl.FLOAT, false, strideB, 0);
    gl.enableVertexAttribArray(this.aOther);
    gl.vertexAttribPointer(this.aOther, 3, gl.FLOAT, false, strideB, 12);
    gl.enableVertexAttribArray(this.aExt);
    gl.vertexAttribPointer(this.aExt, 3, gl.FLOAT, false, strideB, 24);
    gl.enableVertexAttribArray(this.aColor);
    gl.vertexAttribPointer(this.aColor, 4, gl.FLOAT, false, strideB, 36);
    if (this.aAlong >= 0) {
      gl.enableVertexAttribArray(this.aAlong);
      gl.vertexAttribPointer(this.aAlong, 1, gl.FLOAT, false, strideB, 52);
    }
  }

  /** The fixed plan from the plane forward: one sub-range per draw group. */
  private drawSlot(gl: WebGL2RenderingContext, s: Slot, alpha: number): void {
    const g = s.geom;
    if (!g.quads) return;
    if (!s.buf) s.buf = gl.createBuffer();
    if (!s.ibuf) s.ibuf = gl.createBuffer();
    this.setAlpha(gl, alpha);
    this.bindAndMaybeUpload(gl, s.buf, s.ibuf, g.verts, g.indices, s.dirty, false);
    s.dirty = false;
    const from = Math.min(Math.max(0, s.ahead), g.nSeg);
    for (let grp = 0; grp < 3; grp++) {
      const first = g.segQuad[grp][from];
      const end = g.groupEnd[grp];
      if (end > first) {
        gl.drawElements(gl.TRIANGLES, (end - first) * FT_INDICES_PER_SEG, gl.UNSIGNED_INT,
          first * FT_INDICES_PER_SEG * 4);
      }
    }
  }

  /** The seam piece: a fixed-capacity DYNAMIC buffer, sub-updated only when
   *  the live end moved. */
  private drawSeam(gl: WebGL2RenderingContext, alpha: number): void {
    const v = this.seamVerts as Float32Array;
    if (!this.seamBuf) {
      this.seamBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.seamBuf);
      gl.bufferData(gl.ARRAY_BUFFER, SEAM_MAX_QUADS * FT_VERTS_PER_SEG * PC_VERT_STRIDE * 4, gl.DYNAMIC_DRAW);
      this.seamDirty = true;
    }
    if (!this.seamIBuf) {
      this.seamIBuf = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.seamIBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, buildQuadIndices(SEAM_MAX_QUADS), gl.STATIC_DRAW);
    }
    this.setAlpha(gl, alpha);
    this.bindAndMaybeUpload(gl, this.seamBuf, this.seamIBuf, v, SEAM_INDICES, this.seamDirty, true);
    this.seamDirty = false;
    const quads = v.length / (PC_VERT_STRIDE * FT_VERTS_PER_SEG);
    gl.drawElements(gl.TRIANGLES, quads * FT_INDICES_PER_SEG, gl.UNSIGNED_INT, 0);
  }

  // ── destination label ────────────────────────────────────────────────────

  private hideLabel(): void {
    if (this.label && this.labelXY !== 'hidden') {
      this.label.el.style.display = 'none';
      this.labelXY = 'hidden';
    }
  }

  /** Positioned from THIS frame's matrix inside the render pass (never a
   *  map-event handler): the label can never lag its anchor. Hidden behind
   *  the globe (the shader's cull, on the CPU) and off screen. */
  private positionLabel(args: CustomRenderMethodInput, visible: boolean): void {
    const L = this.label;
    if (!L) return;
    if (!visible || !this.map) { this.hideLabel(); return; }
    const pd = args.defaultProjectionData;
    const t = pd.projectionTransition;
    if (t > 0) {
      const cam = cameraFromClippingPlane(pd.clippingPlane as ArrayLike<number>);
      if (cam && earthOccludes(cam, mercatorToSphere(L.mercX, L.mercY, L.z))) { this.hideLabel(); return; }
    }
    let w = 0, h = 0;
    try {
      const c = (this.map as unknown as { getCanvas: () => { clientWidth: number; clientHeight: number } }).getCanvas();
      w = c.clientWidth; h = c.clientHeight;
    } catch (e) {
      reportPlanError('label-canvas', e); // no canvas → the label hides below
    }
    const p = w > 0 && h > 0
      ? projectMercToScreen(pd.mainMatrix as ArrayLike<number>, t, L.mercX, L.mercY, L.z, w, h)
      : null;
    if (!p) { this.hideLabel(); return; }
    const key = `${Math.round(p.x * 2) / 2},${Math.round(p.y * 2) / 2}`;
    if (key === this.labelXY) return;
    this.labelXY = key;
    L.el.style.transform = `translate3d(${p.x.toFixed(1)}px, ${p.y.toFixed(1)}px, 0)`;
    L.el.style.display = '';
  }

  // ── GL plumbing ──────────────────────────────────────────────────────────

  private bindProjection(gl: WebGL2RenderingContext, args: CustomRenderMethodInput): void {
    const pd = args.defaultProjectionData;
    const u = this.proj;
    if (u.matrix) gl.uniformMatrix4fv(u.matrix, false, pd.mainMatrix);
    if (u.tile) {
      gl.uniform4f(u.tile, pd.tileMercatorCoords[0], pd.tileMercatorCoords[1],
        pd.tileMercatorCoords[2], pd.tileMercatorCoords[3]);
    }
    if (u.clip) {
      gl.uniform4f(u.clip, pd.clippingPlane[0], pd.clippingPlane[1], pd.clippingPlane[2], pd.clippingPlane[3]);
    }
    if (u.trans) gl.uniform1f(u.trans, pd.projectionTransition);
    if (u.fallback) gl.uniformMatrix4fv(u.fallback, false, pd.fallbackMatrix);
  }

  private compile(gl: WebGL2RenderingContext, prelude: string, define: string, variant: string): void {
    if (this.program) gl.deleteProgram(this.program);
    this.program = null;
    const mk = (type: number, src: string): WebGLShader => {
      const sh = gl.createShader(type);
      if (!sh) throw new Error('PlanCurtainLayer: createShader failed');
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error('PlanCurtainLayer: shader compile failed: ' + log);
      }
      return sh;
    };
    const vs = mk(gl.VERTEX_SHADER, PC_VERT_SRC(prelude, define));
    const fs = mk(gl.FRAGMENT_SHADER, PC_FRAG_SRC);
    const p = gl.createProgram();
    if (!p) throw new Error('PlanCurtainLayer: createProgram failed');
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error('PlanCurtainLayer: program link failed: ' + log);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    this.program = p;
    this.aPos = gl.getAttribLocation(p, 'a_pos');
    this.aOther = gl.getAttribLocation(p, 'a_other');
    this.aExt = gl.getAttribLocation(p, 'a_ext');
    this.aColor = gl.getAttribLocation(p, 'a_color');
    this.aAlong = gl.getAttribLocation(p, 'a_along');
    this.u = {
      viewport: gl.getUniformLocation(p, 'u_viewport'),
      alpha: gl.getUniformLocation(p, 'u_alpha'),
      alongRef: gl.getUniformLocation(p, 'u_alongRef'),
      dashM: gl.getUniformLocation(p, 'u_dashM'),
    };
    this.proj = {
      matrix: gl.getUniformLocation(p, 'u_projection_matrix'),
      tile: gl.getUniformLocation(p, 'u_projection_tile_mercator_coords'),
      clip: gl.getUniformLocation(p, 'u_projection_clipping_plane'),
      trans: gl.getUniformLocation(p, 'u_projection_transition'),
      fallback: gl.getUniformLocation(p, 'u_projection_fallback_matrix'),
    };
    this.cachedVariant = variant;
  }
}

const SEAM_INDICES = buildQuadIndices(SEAM_MAX_QUADS);

// ── Law IV budget declaration ───────────────────────────────────────────────
// Per plan slot, worst case: (PLAN_MAX_POINTS − 1) segments × 3 quads
// (trace + curtain + edge) × FT_VERTS_PER_SEG(4) × PC_VERT_STRIDE(14) × 4 B
//   = 1535 × 3 × 224 B ≈ 1.03 MB, + indices 1535 × 3 × 6 × 4 B ≈ 110 KB.
// Two slots are resident only during the 250 ms crossfade (≈ 2.3 MB), plus
// the original-plan line (1535 quads ≈ 344 KB + 37 KB indices) and the seam
// (6 quads, < 6 KB): ≈ 2.7 MB worst case → a 4 MB budget.
// maxFeatures counts DENSIFIED PLAN VERTICES of the one selected aircraft.
export const maxFeatures = PLAN_MAX_POINTS;
export const vramBudget = 4; // MB

/** Worst-case resident bytes from this module's own constants (the
 *  layerContract test checks vramBudget against it). */
export function planWorstCaseBytes(): number {
  const segs = PLAN_MAX_POINTS - 1;
  const quadBytes = FT_VERTS_PER_SEG * PC_VERT_STRIDE * 4 + FT_INDICES_PER_SEG * 4;
  return (2 * segs * 3 + segs + SEAM_MAX_QUADS) * quadBytes;
}
