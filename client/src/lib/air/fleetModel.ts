// FLEET REPLAY MODEL — the pure half of the Time Machine's "all planes in
// view, with tracks and curtains" replay (FLIGHT PROGRAM, human directive
// 2026-09-28; earth_twin_program.md TIME MACHINE v2 T-3 "curtain fleet").
//
// No GL, no DOM, no React — unit-tested with `npx tsx --test`. The GL layer
// (fleetReplayLayer.ts) and the per-frame driver (fleetReplayController.ts)
// consume what this module computes:
//
//  - prepareFleet: window payload → typed per-track arrays + HONEST GAP
//    flags. The server marks every returned point that follows a REAL raw
//    archive hole > 10 min (`gaps`, server/aircraftWindow.ts) — decimation
//    spacing is never mistaken for signal loss. Payloads without that field
//    (older cache entries) fall back to "hole > max(10 min, 2 × step)".
//  - headAt: the playhead position of one track, LINEAR between the two
//    bracketing REAL fixes; null outside the track's time span or across an
//    honest gap (a plane is never drawn where we did not record it).
//  - LOD: budgets by device tier (deviceTier.ts classification), importance
//    = nearness to the camera eye (≈ on-screen size) with an on-screen bonus,
//    and HYSTERESIS on every class boundary so a track near the cutoff does
//    not flicker between full curtain / thin line / head-only.
//  - buildFleetVerts: one track → the packed vertex layout the fleet layer
//    batches into ONE buffer (one draw) per LOD class. Every vertex carries
//    its fix time, so the shader cuts the trail at the playhead per frame
//    (the curtain grows smoothly with the replay — geometry is built once,
//    never rebuilt on a camera event or a playhead tick; Law I).

import {
  altRampColor, distMeters, CURTAIN_ALPHA, CURTAIN_BOTTOM_MUL, TRACE_RGBA, TRACE_ABOVE_TERRAIN_M,
  CURTAIN_BELOW_TERRAIN_M,
} from "./trackModel.ts";

/** window payload hex (server/aircraftWindow.ts WindowHex) */
export interface WindowHexIn {
  i: string;
  c?: string;
  rg?: string;
  ty?: string;
  points: Array<[number, number, number, number | null]>;
  gaps?: number[];
}

export interface FleetTrack {
  id: string;
  label: string;
  n: number;
  t: Float64Array;     // epoch sec
  lat: Float64Array;
  lon: Float64Array;
  alt: Float64Array;   // meters MSL, NaN = not broadcast
  /** 1 = an honest gap precedes this point (never draw/interpolate across) */
  gapBefore: Uint8Array;
  /** geometry-cache signature (points changed ⇒ rebuild) */
  sig: string;
}

/** the client's honest-gap threshold — must equal server WINDOW_GAP_SEC */
export const FLEET_GAP_SEC = 600;

/** seconds of trail drawn behind each head (fades over the last 40%) */
export const FLEET_TRAIL_SEC = 45 * 60;

/** fixed altitude-ramp domain for the whole fleet (meters) — colors must be
 *  comparable ACROSS planes, unlike the single-track ramp (min→max of one) */
export const FLEET_ALT_DOMAIN_M: [number, number] = [0, 12500];

export function prepareFleet(hexes: WindowHexIn[], stepSec: number): FleetTrack[] {
  const hasGapInfo = hexes.some((h) => Array.isArray(h.gaps));
  const fallbackGap = Math.max(FLEET_GAP_SEC, 2 * Math.max(0, stepSec || 0));
  const out: FleetTrack[] = [];
  for (const h of hexes) {
    if (!h || typeof h.i !== "string" || !Array.isArray(h.points)) continue;
    const pts = h.points.filter((p) => Array.isArray(p) && Number.isFinite(p[0]) &&
      Number.isFinite(p[1]) && Number.isFinite(p[2]));
    if (pts.length === 0) continue;
    const n = pts.length;
    const tr: FleetTrack = {
      id: h.i, label: h.c || h.rg || h.i, n,
      t: new Float64Array(n), lat: new Float64Array(n), lon: new Float64Array(n),
      alt: new Float64Array(n), gapBefore: new Uint8Array(n),
      sig: `${n}:${pts[0][0]}:${pts[n - 1][0]}`,
    };
    const gapSet = new Set(h.gaps || []);
    for (let k = 0; k < n; k++) {
      const p = pts[k];
      tr.t[k] = p[0]; tr.lat[k] = p[1]; tr.lon[k] = p[2];
      tr.alt[k] = p[3] == null || !Number.isFinite(p[3] as number) ? NaN : (p[3] as number);
      if (k > 0) {
        const dt = p[0] - pts[k - 1][0];
        tr.gapBefore[k] = (hasGapInfo ? gapSet.has(p[0]) : dt > fallbackGap) ? 1 : 0;
      }
    }
    out.push(tr);
  }
  return out;
}

/** full-fidelity override points (the /api/data/track endpoint's shape) */
export function trackFromPoints(id: string, label: string,
  points: Array<{ t: number; la: number; lo: number; al?: number | null }>): FleetTrack | null {
  const hex: WindowHexIn = {
    i: id, c: label,
    points: points.filter((p) => Number.isFinite(p.t)).sort((a, b) => a.t - b.t)
      .map((p) => [p.t, p.la, p.lo, p.al ?? null] as [number, number, number, number | null]),
  };
  // raw fixes: the gap rule applies directly to consecutive fixes
  const gaps: number[] = [];
  for (let k = 1; k < hex.points.length; k++) {
    if (hex.points[k][0] - hex.points[k - 1][0] > FLEET_GAP_SEC) gaps.push(hex.points[k][0]);
  }
  hex.gaps = gaps;
  return prepareFleet([hex], 0)[0] ?? null;
}

/**
 * Splice full-fidelity fixes into a window track: inside the override's own
 * time span its fixes replace the window's (decimated) ones; outside it the
 * window points stay. Returns null — keep the window track — unless the
 * override actually covers `mustCoverSec` (every instant listed has a head),
 * so a response for the wrong time or place can never replace real window
 * data. Gap flags: the window's where both neighbors are window points, the
 * raw > FLEET_GAP_SEC rule at and inside the override.
 */
export function spliceOverride(base: FleetTrack | undefined, ov: FleetTrack, mustCoverSec: number[]): FleetTrack | null {
  if (ov.n === 0) return null;
  for (const t of mustCoverSec) if (!headAt(ov, t)) return null;
  const lo = ov.t[0], hi = ov.t[ov.n - 1];
  type P = { t: number; la: number; lo: number; al: number; src: "b" | "o"; g: number };
  const pts: P[] = [];
  if (base) {
    for (let k = 0; k < base.n; k++) {
      if (base.t[k] >= lo && base.t[k] <= hi) continue;
      pts.push({ t: base.t[k], la: base.lat[k], lo: base.lon[k], al: base.alt[k], src: "b", g: base.gapBefore[k] });
    }
  }
  for (let k = 0; k < ov.n; k++) pts.push({ t: ov.t[k], la: ov.lat[k], lo: ov.lon[k], al: ov.alt[k], src: "o", g: 0 });
  pts.sort((a, b) => a.t - b.t);
  const gaps: number[] = [];
  for (let k = 1; k < pts.length; k++) {
    const both = pts[k].src === "b" && pts[k - 1].src === "b";
    if (both ? pts[k].g === 1 : pts[k].t - pts[k - 1].t > FLEET_GAP_SEC) gaps.push(pts[k].t);
  }
  return prepareFleet([{
    i: ov.id, c: ov.label,
    points: pts.map((p) => [p.t, p.la, p.lo, Number.isFinite(p.al) ? p.al : null] as [number, number, number, number | null]),
    gaps,
  }], 0)[0] ?? null;
}

export interface Head {
  lon: number;
  lat: number;
  /** meters MSL, NaN = altitude not broadcast at this instant */
  altM: number;
  /** track heading, degrees clockwise from north */
  hdg: number;
  /** seconds to the nearest real fix (interpolation honesty readout) */
  nearestFixSec: number;
}

const D2R = Math.PI / 180;

function bearingDeg(la1: number, lo1: number, la2: number, lo2: number): number {
  let dLon = lo2 - lo1;
  if (dLon > 180) dLon -= 360;
  if (dLon < -180) dLon += 360;
  const x = dLon * Math.cos(((la1 + la2) / 2) * D2R);
  const y = la2 - la1;
  if (x === 0 && y === 0) return NaN;
  return (Math.atan2(x, y) / D2R + 360) % 360;
}

/** index of the last fix with t <= tSec (-1 if none) */
export function fixIndexAtOrBefore(tr: FleetTrack, tSec: number): number {
  let lo = 0, hi = tr.n - 1;
  if (tr.n === 0 || tSec < tr.t[0]) return -1;
  if (tSec >= tr.t[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (tr.t[mid] <= tSec) lo = mid; else hi = mid;
  }
  return lo;
}

/** a head is HELD (not moved, never extrapolated) at its last real fix for
 *  up to this long past it — 2× the archive's 75 s cruise cadence — so the
 *  window's "now" end still shows planes whose newest fix lags it by one
 *  poll; past that (landed, out of coverage, gap) the head disappears. */
export const FLEET_HOLD_SEC = 150;

/**
 * The head position at the playhead: linear between the bracketing REAL
 * fixes; held at a fix for ≤ FLEET_HOLD_SEC when nothing follows it (end of
 * track or an honest gap); null before the first fix or beyond the hold.
 */
export function headAt(tr: FleetTrack, tSec: number): Head | null {
  const k = fixIndexAtOrBefore(tr, tSec);
  if (k < 0) return null;
  const hdgFrom = (a: number, b: number): number => {
    const h = bearingDeg(tr.lat[a], tr.lon[a], tr.lat[b], tr.lon[b]);
    return Number.isNaN(h) ? 0 : h;
  };
  const held = (): Head | null => {
    const since = tSec - tr.t[k];
    if (since > FLEET_HOLD_SEC) return null; // beyond the hold: not drawn
    let hdg = 0;
    if (since === 0 && k + 1 < tr.n && !tr.gapBefore[k + 1]) hdg = hdgFrom(k, k + 1);
    else if (k > 0 && !tr.gapBefore[k]) hdg = hdgFrom(k - 1, k);
    else if (k + 1 < tr.n && !tr.gapBefore[k + 1]) hdg = hdgFrom(k, k + 1);
    return { lon: tr.lon[k], lat: tr.lat[k], altM: tr.alt[k], hdg, nearestFixSec: since };
  };
  if (tSec === tr.t[k] || k === tr.n - 1) return held();
  const j = k + 1;
  if (tr.gapBefore[j]) return held(); // honest gap — never bridged (held briefly, then gone)
  const span = tr.t[j] - tr.t[k];
  const u = span > 0 ? (tSec - tr.t[k]) / span : 0;
  let dLon = tr.lon[j] - tr.lon[k];
  if (dLon > 180) dLon -= 360;
  if (dLon < -180) dLon += 360;
  let lon = tr.lon[k] + dLon * u;
  if (lon > 180) lon -= 360;
  if (lon < -180) lon += 360;
  const altOk = Number.isFinite(tr.alt[k]) && Number.isFinite(tr.alt[j]);
  return {
    lon,
    lat: tr.lat[k] + (tr.lat[j] - tr.lat[k]) * u,
    altM: altOk ? tr.alt[k] + (tr.alt[j] - tr.alt[k]) * u : NaN,
    hdg: hdgFrom(k, j),
    nearestFixSec: Math.min(tSec - tr.t[k], tr.t[j] - tSec),
  };
}

// ── LOD ──────────────────────────────────────────────────────────────────────

export const LOD_FULL = 0;
export const LOD_THIN = 1;
export const LOD_HEAD = 2;
export const LOD_HIDDEN = 3;
export type Lod = 0 | 1 | 2 | 3;

export interface FleetBudget {
  /** tracks drawn with the full curtain + altitude line + ground trace */
  full: number;
  /** tracks drawn as a thin altitude polyline */
  thin: number;
  /** heads drawn (every LOD class above HIDDEN gets a head when airborne) */
  heads: number;
  /** vertex-budget caps: segments across ALL full / thin tracks */
  fullSegments: number;
  thinSegments: number;
}

/** Law IV budgets per device tier (deviceTier.ts DeviceTier). The full-tier
 *  numbers are the declared worst case (fleetReplayLayer.vramBudget). */
export const FLEET_BUDGETS: Record<"full" | "reduced" | "minimal", FleetBudget> = {
  full: { full: 150, thin: 1000, heads: 2000, fullSegments: 30_000, thinSegments: 60_000 },
  reduced: { full: 50, thin: 400, heads: 1000, fullSegments: 10_000, thinSegments: 24_000 },
  minimal: { full: 12, thin: 120, heads: 400, fullSegments: 3_000, thinSegments: 8_000 },
};

/** window `max` hex cap to request per tier (route bounds: 50..2000) */
export const FLEET_FETCH_MAX: Record<"full" | "reduced" | "minimal", number> = {
  full: 2000, reduced: 1000, minimal: 400,
};

export function fleetBudgetForTier(tier: string | undefined | null): FleetBudget {
  return tier === "full" || tier === "minimal" ? FLEET_BUDGETS[tier] : FLEET_BUDGETS.reduced;
}

/** LOD boundary hysteresis: a member keeps its class while ranked within
 *  (1 + H) × budget; a newcomer enters only when ranked within (1 − H) × budget */
export const LOD_HYSTERESIS = 0.2;

/**
 * Select the top-`budget` indices by score with hysteresis against the
 * previous selection. `must` indices (already in a better class) are always
 * included. Never selects a non-finite score. Deterministic (ties by index).
 */
function selectWithHysteresis(
  ranked: number[], rankOf: Int32Array, prevIn: (i: number) => boolean,
  must: Uint8Array, budget: number, h: number,
): Uint8Array {
  const sel = new Uint8Array(rankOf.length);
  const keepLimit = Math.ceil(budget * (1 + h));
  const enterLimit = Math.floor(budget * (1 - h));
  let count = 0;
  for (const i of ranked) {
    const r = rankOf[i];
    if (must[i] || (prevIn(i) && r < keepLimit) || r < enterLimit) { sel[i] = 1; count++; }
  }
  if (count > budget) {
    // trim the worst-ranked non-mandatory members
    for (let k = ranked.length - 1; k >= 0 && count > budget; k--) {
      const i = ranked[k];
      if (sel[i] && !must[i]) { sel[i] = 0; count--; }
    }
  } else if (count < budget) {
    for (const i of ranked) {
      if (count >= budget) break;
      if (!sel[i]) { sel[i] = 1; count++; }
    }
  }
  return sel;
}

/**
 * Assign every track a LOD class. `scores` higher = more important;
 * -Infinity = nothing of this track is visible at the playhead (HIDDEN).
 * `prev` = last assignment (same indexing) for hysteresis.
 */
export function assignLods(
  scores: Float64Array, prev: Uint8Array | null, budget: FleetBudget, h = LOD_HYSTERESIS,
): Uint8Array {
  const n = scores.length;
  const ranked: number[] = [];
  for (let i = 0; i < n; i++) if (Number.isFinite(scores[i])) ranked.push(i);
  ranked.sort((a, b) => scores[b] - scores[a] || a - b);
  const rankOf = new Int32Array(n).fill(0x7fffffff);
  ranked.forEach((i, r) => { rankOf[i] = r; });
  const prevLod = (i: number): number => (prev && i < prev.length ? prev[i] : LOD_HIDDEN);
  const none = new Uint8Array(n);
  const inFull = selectWithHysteresis(ranked, rankOf, (i) => prevLod(i) <= LOD_FULL, none, Math.min(budget.full, ranked.length), h);
  const inThin = selectWithHysteresis(ranked, rankOf, (i) => prevLod(i) <= LOD_THIN, inFull,
    Math.min(budget.full + budget.thin, ranked.length), h);
  // heads budget counts EVERY drawn track (full and thin tracks get heads too)
  const inHead = selectWithHysteresis(ranked, rankOf, (i) => prevLod(i) <= LOD_HEAD, inThin,
    Math.min(Math.max(budget.heads, budget.full + budget.thin), ranked.length), h);
  const out = new Uint8Array(n).fill(LOD_HIDDEN);
  for (let i = 0; i < n; i++) {
    if (inFull[i]) out[i] = LOD_FULL;
    else if (inThin[i]) out[i] = LOD_THIN;
    else if (inHead[i]) out[i] = LOD_HEAD;
  }
  return out;
}

/** straight-line (3D) km from the camera eye to a point — the on-screen
 *  size proxy (perspective: nearer = larger) */
export function eyeDistanceKm(
  eye: { lat: number; lon: number; heightM: number }, lat: number, lon: number, altM: number,
): number {
  const ground = distMeters(eye.lat, eye.lon, lat, lon) / 1000;
  const dz = (eye.heightM - (Number.isFinite(altM) ? altM : 0)) / 1000;
  return Math.sqrt(ground * ground + dz * dz);
}

/** importance score: -distance, with a large penalty when off-screen */
export function trackScore(distKm: number, onScreen: boolean): number {
  return onScreen ? -distKm : -distKm - 1e7;
}

/**
 * Indices to keep so a track fits `maxPts`, always keeping the first/last
 * point and both sides of every honest gap (a dropped boundary would bridge
 * a gap). Uniform in index otherwise — every kept point is a real fix, so
 * "linear between real fixes" still holds.
 */
export function decimateIdx(tr: FleetTrack, maxPts: number): number[] {
  const n = tr.n;
  const all = () => Array.from({ length: n }, (_, i) => i);
  if (n <= maxPts || maxPts < 2) return all();
  const stride = (n - 1) / (maxPts - 1);
  const keep = new Uint8Array(n);
  for (let k = 0; k < maxPts; k++) keep[Math.round(k * stride)] = 1;
  keep[0] = 1; keep[n - 1] = 1;
  for (let k = 1; k < n; k++) if (tr.gapBefore[k]) { keep[k] = 1; keep[k - 1] = 1; }
  const out: number[] = [];
  for (let k = 0; k < n; k++) if (keep[k]) out.push(k);
  return out;
}

// ── vertex packing (the fleet layer's layout) ────────────────────────────────

/** floats per vertex: [0..2] pos (mercX, mercY, z) · [3..5] other ·
 *  [6..8] ext (side, dir, widthPx) · [9..12] rgba · [13] t (sec rel. base) ·
 *  [14] track slot (highlight lookup) */
export const FLEET_STRIDE = 15;
export const FLEET_VERTS_PER_SEG = 4;

export const FLEET_THIN_WIDTH_PX = 1.5;
export const FLEET_LINE_WIDTH_PX = 3;
export const FLEET_TRACE_WIDTH_PX = 2;

export function mercX(lon: number): number { return (lon + 180) / 360; }
export function mercY(lat: number): number {
  const c = Math.max(-85.05113, Math.min(85.05113, lat)) * D2R;
  return (1 - Math.log(Math.tan(Math.PI / 4 + c / 2)) / Math.PI) / 2;
}

export type FleetMode = "full" | "thin";

/** does a segment between kept indices a < b cross an honest gap? */
function gapBetween(tr: FleetTrack, a: number, b: number): boolean {
  for (let k = a + 1; k <= b; k++) if (tr.gapBefore[k]) return true;
  return false;
}

/**
 * One track → packed vertices (FLEET_STRIDE) for its LOD class. `idx` = the
 * kept point indices (decimateIdx). FULL = ground trace + curtain + altitude
 * line; THIN = the altitude line (or the ground trace where altitude is not
 * broadcast — position history is real even then). Segments never span an
 * honest gap or the antimeridian. `groundZ` (display datum, per kept point)
 * is only used by FULL with terrain on; null = sea level.
 */
export function buildFleetVerts(
  tr: FleetTrack, idx: number[], mode: FleetMode, slot: number, baseSec: number,
  altScale: number, groundZ: Float32Array | null = null, drapeBelowM = 0,
): Float32Array {
  const segs = Math.max(0, idx.length - 1);
  const out = new Float32Array(segs * (mode === "full" ? 3 : 1) * FLEET_VERTS_PER_SEG * FLEET_STRIDE);
  let o = 0;
  const put = (x: number, y: number, z: number, ox: number, oy: number, oz: number,
    side: number, dir: number, w: number, r: number, g: number, b: number, a: number, t: number) => {
    out[o++] = x; out[o++] = y; out[o++] = z;
    out[o++] = ox; out[o++] = oy; out[o++] = oz;
    out[o++] = side; out[o++] = dir; out[o++] = w;
    out[o++] = r; out[o++] = g; out[o++] = b; out[o++] = a;
    out[o++] = t; out[o++] = slot;
  };
  const ribbon = (ax: number, ay: number, az: number, at: number, bx: number, by: number, bz: number, bt: number,
    w: number, ca: number[], cb: number[]) => {
    put(ax, ay, az, bx, by, bz, -1, +1, w, ca[0], ca[1], ca[2], ca[3], at);
    put(ax, ay, az, bx, by, bz, +1, +1, w, ca[0], ca[1], ca[2], ca[3], at);
    put(bx, by, bz, ax, ay, az, -1, -1, w, cb[0], cb[1], cb[2], cb[3], bt);
    put(bx, by, bz, ax, ay, az, +1, -1, w, cb[0], cb[1], cb[2], cb[3], bt);
  };
  const [aMin, aMax] = FLEET_ALT_DOMAIN_M;
  const lift = TRACE_ABOVE_TERRAIN_M * altScale;
  const drop = drapeBelowM * altScale;
  for (let s = 0; s + 1 < idx.length; s++) {
    const a = idx[s], b = idx[s + 1];
    if (gapBetween(tr, a, b)) continue;
    const ax = mercX(tr.lon[a]), ay = mercY(tr.lat[a]);
    const bx = mercX(tr.lon[b]), by = mercY(tr.lat[b]);
    if (Math.abs(ax - bx) > 0.5) continue; // antimeridian: honest break
    const at = tr.t[a] - baseSec, bt = tr.t[b] - baseSec;
    const gA = groundZ ? groundZ[s] : 0, gB = groundZ ? groundZ[s + 1] : 0;
    const hasAlt = Number.isFinite(tr.alt[a]) && Number.isFinite(tr.alt[b]);
    const ca = hasAlt ? altRampColor(tr.alt[a], aMin, aMax) : null;
    const cb = hasAlt ? altRampColor(tr.alt[b], aMin, aMax) : null;
    if (mode === "full") {
      const tr1 = [TRACE_RGBA[0], TRACE_RGBA[1], TRACE_RGBA[2], 0.7];
      ribbon(ax, ay, gA + lift, at, bx, by, gB + lift, bt, FLEET_TRACE_WIDTH_PX, tr1, tr1);
      if (hasAlt && ca && cb) {
        const topA = tr.alt[a] * altScale, topB = tr.alt[b] * altScale;
        const botA = gA - drop, botB = gB - drop;
        const m = CURTAIN_BOTTOM_MUL;
        put(ax, ay, topA, ax, ay, topA, 0, 0, 0, ca[0], ca[1], ca[2], CURTAIN_ALPHA, at);
        put(ax, ay, botA, ax, ay, botA, 0, 0, 0, ca[0] * m[0], ca[1] * m[1], ca[2] * m[2], CURTAIN_ALPHA, at);
        put(bx, by, topB, bx, by, topB, 0, 0, 0, cb[0], cb[1], cb[2], CURTAIN_ALPHA, bt);
        put(bx, by, botB, bx, by, botB, 0, 0, 0, cb[0] * m[0], cb[1] * m[1], cb[2] * m[2], CURTAIN_ALPHA, bt);
        ribbon(ax, ay, topA, at, bx, by, topB, bt, FLEET_LINE_WIDTH_PX, [ca[0], ca[1], ca[2], 1], [cb[0], cb[1], cb[2], 1]);
      }
    } else if (hasAlt && ca && cb) {
      ribbon(ax, ay, tr.alt[a] * altScale, at, bx, by, tr.alt[b] * altScale, bt, FLEET_THIN_WIDTH_PX,
        [ca[0], ca[1], ca[2], 0.75], [cb[0], cb[1], cb[2], 0.75]);
    } else {
      const c = [TRACE_RGBA[0], TRACE_RGBA[1], TRACE_RGBA[2], 0.6];
      ribbon(ax, ay, lift, at, bx, by, lift, bt, FLEET_THIN_WIDTH_PX, c, c);
    }
  }
  return o === out.length ? out : out.slice(0, o);
}

/** segments a track contributes after decimation (budget arithmetic) */
export function segmentCount(idxLen: number): number {
  return Math.max(0, idxLen - 1);
}

/** per-track point cap so `count` tracks fit `segmentBudget` */
export function pointsPerTrack(segmentBudget: number, count: number, floor = 16): number {
  if (count <= 0) return Number.MAX_SAFE_INTEGER;
  return Math.max(floor, Math.floor(segmentBudget / count) + 1);
}

/** instance layout for heads: mercX, mercY, z (display m), heading rad,
 *  r, g, b, a */
export const HEAD_STRIDE = 8;

/** head color: fleet altitude ramp; altitude unknown → neutral slate */
export function headColor(altM: number): [number, number, number] {
  if (!Number.isFinite(altM)) return [0x9f / 255, 0xb3 / 255, 0xc8 / 255];
  return altRampColor(altM, FLEET_ALT_DOMAIN_M[0], FLEET_ALT_DOMAIN_M[1]);
}

export { CURTAIN_BELOW_TERRAIN_M };
