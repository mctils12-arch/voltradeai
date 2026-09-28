// CLOSE APPROACHES — "if two planes came close you can see it in replay"
// (FLIGHT PROGRAM, human directive 2026-09-28, vision item 5).
//
// Pure module: given per-hex tracks from ONE window read (the Time Machine's
// /api/data/aircraft/window, server/aircraftWindow.ts), find aircraft pairs
// whose RECORDED positions came within the standard en-route separation
// minima at the same instant: < 5 nm horizontally AND < 1,000 ft vertically.
//
// WHAT THIS IS NOT (the honesty rails, stated in every result's `basis`):
//  - NOT an official loss-of-separation / near-mid-air-collision report.
//    Those come from ATC/FAA investigations with radar + voice data. This
//    is "per our recorded ADS-B data" — broadcast positions archived at
//    30–75 s cadence, linearly interpolated between REAL fixes.
//  - 5 nm / 1,000 ft is the IFR EN-ROUTE standard. Terminal-area radar
//    separation is 3 nm, parallel/simultaneous approaches, formation
//    flights, air refuelling and same-airport pattern/ground traffic are
//    routinely closer BY DESIGN and are legal. VFR traffic has no ATC
//    separation minimum from other VFR traffic at all. A flagged pair is
//    "these two were this close per our data", nothing more.
//
// THE INTERPOLATION RULE (honesty, from the build brief): a position at
// time t is only ever LINEAR between two consecutive real archived fixes,
// and only where that pair of fixes is at most 2 x FIX_WINDOW_SEC (180 s)
// apart — so every evaluated instant has a real fix within ±90 s AND is
// never an interpolation across a signal gap. No extrapolation, no smoothing,
// no invented curve. A fix without a broadcast altitude breaks the track for
// this purpose (vertical separation unknown => not evaluated, never assumed).
//
// CONFIDENCE from fix density at the reported instant: each aircraft's
// distance to its NEAREST real fix — high <= 20 s, medium <= 60 s, low <= 90 s;
// the pair takes the worse of the two.
//
// AIRPORT / LOW-LEVEL EXCLUSION (approximation, documented): a pair is not
// flagged while BOTH aircraft are below LOW_ALT_FT (2,000 ft) above a
// reference elevation — the field elevation of the nearest open airport
// within 5 nm of the pair when an `airportNear` lookup is supplied, else
// sea level. That removes same-airport ground/pattern/approach traffic
// (where 5 nm/1,000 ft does not apply) without a terrain model. Aircraft
// reported on the ground carry no altitude in the archive, so ground pairs
// never enter at all. Non-ICAO addresses ("~"-prefixed TIS-B/ADS-R
// rebroadcasts) are excluded: a radar-derived echo of an ADS-B target
// would otherwise pair with its own aircraft as a false 0 nm "collision".
//
// ALGORITHM (O(n) in fixes for realistic traffic, not O(n^2) in hexes):
//  1. every eligible consecutive-fix segment becomes a PIECE (linear
//     position+altitude over [t0,t1]); isolated eligible fixes are
//     degenerate pieces;
//  2. pieces are cut into BUCKET_SEC time buckets; within one bucket each
//     piece is inserted into a lat/lon grid (cell = the horizontal
//     threshold; per-row longitude width widened by 1/cos(poleward edge),
//     so any two positions within threshold always share a probed cell);
//  3. each piece probes the cells of its bbox expanded by the threshold;
//     each distinct piece pair gets an EXACT test: on the pieces' common
//     time interval both motions are linear, so the minimum horizontal
//     distance subject to (|dz| <= 1000 ft) and the low-level exclusion is
//     a clamped quadratic minimum — no sampling, no missed crossings;
//  4. flagged instants per aircraft pair are merged into ENCOUNTERS (a new
//     encounter when > ENCOUNTER_GAP_SEC apart), reporting the minimum.
// `useGrid: false` runs the same exact test over every time-overlapping
// piece pair — the brute-force reference the test pins the grid against.

import { EARTH_RADIUS_NM } from "../shared/flightPlanGeometry";

export type Fix = [number, number, number, number | null]; // [t sec, lat, lon, alt METERS | null]

export interface CloseApproachTrack {
  /** icao24 hex */
  i: string;
  /** callsign (last seen in the window), for display */
  c?: string;
  /** time-ascending archived fixes (the window reader's dedup'd, UN-decimated points) */
  points: Fix[];
}

export type Confidence = "high" | "medium" | "low";

export interface CloseApproach {
  /** hex of the first aircraft (lexically smaller) */
  a: string;
  /** hex of the second aircraft */
  b: string;
  /** callsigns when known (display only) */
  ca?: string;
  cb?: string;
  /** instant of minimum horizontal separation within the encounter, epoch MS */
  t: number;
  horizNm: number;
  vertFt: number;
  confidence: Confidence;
  basis: string;
  /** midpoint of the two positions at t (for framing the camera) */
  lat: number;
  lon: number;
  /** both aircraft's interpolated altitudes at t, feet MSL (baro, as broadcast) */
  altAFt: number;
  altBFt: number;
}

export interface CloseApproachResult {
  approaches: CloseApproach[];
  /** encounters found before the output cap */
  found: number;
  capped: boolean;
  /** tracks that contributed at least one eligible piece */
  evaluated_hexes: number;
  /** non-ICAO (~) addresses skipped */
  excluded_non_icao: number;
  pieces: number;
}

export const CLOSE_APPROACH_BASIS =
  "separation below 5 nm / 1,000 ft per our recorded ADS-B data — not an official loss-of-separation report";

export const CA_DEFAULTS = {
  /** en-route IFR horizontal minimum, nautical miles */
  HORIZ_NM: 5,
  /** vertical minimum, feet (RVSM and below FL290 alike) */
  VERT_FT: 1000,
  /** every evaluated instant has a real fix within ± this many seconds;
   *  interpolation spans at most 2x this (never across a signal gap) */
  FIX_WINDOW_SEC: 90,
  /** both aircraft below this height above the reference elevation =
   *  terminal/airport/low-level traffic, not evaluated against en-route minima */
  LOW_ALT_FT: 2000,
  /** airport reference radius for the low-level exclusion, nm */
  AIRPORT_RADIUS_NM: 5,
  /** result cap (sorted by minimum separation) */
  CAP: 200,
  /** flagged instants further apart than this = separate encounters */
  ENCOUNTER_GAP_SEC: 600,
  /** time-bucket width for the spatial grid */
  BUCKET_SEC: 60,
} as const;

/** highest open airports are ~14,500 ft; above LOW_ALT_FT + this no airport
 *  lookup can matter, so cruise pairs never pay for one. */
const MAX_AIRPORT_ELEV_FT = 15_000;

const M_TO_FT = 3.28084;
const NM_PER_DEG_LAT = 60;
// one source of truth for the Earth radius in nm (D11 dup_precise_literal)
const R_EARTH_NM = EARTH_RADIUS_NM;
const D2R = Math.PI / 180;

export interface CloseApproachOptions {
  horizNm?: number;
  vertFt?: number;
  fixWindowSec?: number;
  lowAltFt?: number;
  cap?: number;
  encounterGapSec?: number;
  bucketSec?: number;
  /** false = brute-force reference (tests). Default true. */
  useGrid?: boolean;
  /** optional field-elevation lookup for the low-level exclusion: returns
   *  the nearest open airport within `radiusNm` of (lat, lon), elevation in
   *  FEET (null when unpublished = treated as sea level). */
  airportNear?: (lat: number, lon: number, radiusNm: number) => { elevFt: number | null } | null;
  airportRadiusNm?: number;
}

/** haversine, nautical miles */
export function distNm(la1: number, lo1: number, la2: number, lo2: number): number {
  const p1 = la1 * D2R, p2 = la2 * D2R;
  const dp = (la2 - la1) * D2R, dl = (lo2 - lo1) * D2R;
  const a = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * R_EARTH_NM * Math.asin(Math.min(1, Math.sqrt(a)));
}

const wrap180 = (d: number): number => {
  let x = ((d + 180) % 360 + 360) % 360 - 180;
  if (x === -180 && d > 0) x = 180;
  return x;
};

/** fix-density confidence from the WORSE aircraft's nearest-fix distance */
export function confidenceFor(nearestFixSecA: number, nearestFixSecB: number): Confidence {
  const w = Math.max(nearestFixSecA, nearestFixSecB);
  if (w <= 20) return "high";
  if (w <= 60) return "medium";
  return "low";
}

// ── pieces (struct-of-arrays for speed) ─────────────────────────────────────

interface Pieces {
  n: number;
  hex: Int32Array;
  t0: Float64Array; t1: Float64Array;
  la0: Float64Array; lo0: Float64Array;
  la1: Float64Array; lo1: Float64Array; // lo1 UNWRAPPED relative to lo0
  z0: Float64Array; z1: Float64Array;   // feet
}

function buildPieces(tracks: CloseApproachTrack[], maxSpan: number): {
  pieces: Pieces; hexIds: string[]; calls: Array<string | undefined>; evaluated: number; nonIcao: number;
} {
  const hexIds: string[] = [];
  const calls: Array<string | undefined> = [];
  const tmp: number[] = []; // flat: hex,t0,t1,la0,lo0,la1,lo1,z0,z1
  let evaluated = 0;
  let nonIcao = 0;
  for (const tr of tracks) {
    if (!tr || typeof tr.i !== "string" || !Array.isArray(tr.points)) continue;
    if (tr.i.startsWith("~")) { nonIcao++; continue; }
    const pts = tr.points.filter((p) => Array.isArray(p) && Number.isFinite(p[0]) &&
      Number.isFinite(p[1]) && Number.isFinite(p[2]));
    if (pts.length === 0) continue;
    const h = hexIds.length;
    let added = 0;
    const hasAlt = (p: Fix) => p[3] != null && Number.isFinite(p[3] as number);
    const segOk = (a: Fix, b: Fix) => hasAlt(a) && hasAlt(b) && b[0] > a[0] && b[0] - a[0] <= maxSpan;
    for (let j = 0; j < pts.length; j++) {
      const p = pts[j];
      if (j + 1 < pts.length && segOk(p, pts[j + 1])) {
        const q = pts[j + 1];
        tmp.push(h, p[0], q[0], p[1], p[2], q[1], p[2] + wrap180(q[2] - p[2]),
          (p[3] as number) * M_TO_FT, (q[3] as number) * M_TO_FT);
        added++;
      }
      // isolated eligible fix (no qualifying segment on either side): a
      // degenerate piece — the instant of a real fix is always evaluable.
      const prevOk = j > 0 && segOk(pts[j - 1], p);
      const nextOk = j + 1 < pts.length && segOk(p, pts[j + 1]);
      if (!prevOk && !nextOk && hasAlt(p)) {
        const z = (p[3] as number) * M_TO_FT;
        tmp.push(h, p[0], p[0], p[1], p[2], p[1], p[2], z, z);
        added++;
      }
    }
    if (added > 0) {
      hexIds.push(tr.i);
      calls.push(tr.c);
      evaluated++;
    }
  }
  const n = tmp.length / 9;
  const pieces: Pieces = {
    n,
    hex: new Int32Array(n),
    t0: new Float64Array(n), t1: new Float64Array(n),
    la0: new Float64Array(n), lo0: new Float64Array(n),
    la1: new Float64Array(n), lo1: new Float64Array(n),
    z0: new Float64Array(n), z1: new Float64Array(n),
  };
  for (let k = 0; k < n; k++) {
    const o = k * 9;
    pieces.hex[k] = tmp[o];
    pieces.t0[k] = tmp[o + 1]; pieces.t1[k] = tmp[o + 2];
    pieces.la0[k] = tmp[o + 3]; pieces.lo0[k] = tmp[o + 4];
    pieces.la1[k] = tmp[o + 5]; pieces.lo1[k] = tmp[o + 6];
    pieces.z0[k] = tmp[o + 7]; pieces.z1[k] = tmp[o + 8];
  }
  return { pieces, hexIds, calls, evaluated, nonIcao };
}

/** position of piece k at time t (linear; lon unwrapped) */
function posAt(P: Pieces, k: number, t: number): [number, number, number] {
  const span = P.t1[k] - P.t0[k];
  const u = span > 0 ? Math.min(1, Math.max(0, (t - P.t0[k]) / span)) : 0;
  return [
    P.la0[k] + (P.la1[k] - P.la0[k]) * u,
    P.lo0[k] + (P.lo1[k] - P.lo0[k]) * u,
    P.z0[k] + (P.z1[k] - P.z0[k]) * u,
  ];
}

/** interval of s in [0,1] where a + b*s satisfies lo <= . <= hi */
function linInterval(a: number, b: number, lo: number, hi: number): [number, number] | null {
  let s0 = 0, s1 = 1;
  if (Math.abs(b) < 1e-12) {
    return a >= lo && a <= hi ? [0, 1] : null;
  }
  const sa = (lo - a) / b, sb = (hi - a) / b;
  const smin = Math.min(sa, sb), smax = Math.max(sa, sb);
  s0 = Math.max(s0, smin); s1 = Math.min(s1, smax);
  return s0 <= s1 ? [s0, s1] : null;
}

const intersect = (x: [number, number] | null, y: [number, number] | null): [number, number] | null => {
  if (!x || !y) return null;
  const a = Math.max(x[0], y[0]), b = Math.min(x[1], y[1]);
  return a <= b ? [a, b] : null;
};

interface Hit { t: number; horiz: number; vert: number; laP: number; loP: number; laQ: number; loQ: number; zP: number; zQ: number; nfP: number; nfQ: number }

/**
 * EXACT test for one piece pair: min horizontal separation over the common
 * time interval subject to |dz| <= vert and "not both low". Returns the hit
 * when that minimum is below both thresholds, else null.
 */
function testPair(P: Pieces, p: number, q: number, o: Required<Pick<CloseApproachOptions,
  "horizNm" | "vertFt" | "lowAltFt" | "airportRadiusNm">> & { airportNear?: CloseApproachOptions["airportNear"]; aptCache: Map<string, number> }): Hit | null {
  const lo = Math.max(P.t0[p], P.t0[q]);
  const hi = Math.min(P.t1[p], P.t1[q]);
  if (lo > hi) return null;
  const A0 = posAt(P, p, lo), A1 = posAt(P, p, hi);
  const B0 = posAt(P, q, lo), B1 = posAt(P, q, hi);
  // quick vertical reject: dz linear, so its range is its endpoints
  const dz0 = B0[2] - A0[2], dz1 = B1[2] - A1[2];
  if (Math.min(dz0, dz1) >= o.vertFt || Math.max(dz0, dz1) <= -o.vertFt) return null;

  // local planar frame at A(lo)
  const lat0 = A0[0], lon0 = A0[1];
  const cphi = Math.cos(lat0 * D2R);
  const xy = (la: number, ln: number): [number, number] =>
    [wrap180(ln - lon0) * NM_PER_DEG_LAT * cphi, (la - lat0) * NM_PER_DEG_LAT];
  const [ax0, ay0] = xy(A0[0], A0[1]), [ax1, ay1] = xy(A1[0], A1[1]);
  const [bx0, by0] = xy(B0[0], B0[1]), [bx1, by1] = xy(B1[0], B1[1]);
  const rx0 = bx0 - ax0, ry0 = by0 - ay0;
  const rvx = (bx1 - ax1) - rx0, rvy = (by1 - ay1) - ry0;

  // quick horizontal reject: min over s of |r0 + rv s| on [0,1]
  const vv = rvx * rvx + rvy * rvy;
  const sFree = vv > 1e-18 ? Math.min(1, Math.max(0, -(rx0 * rvx + ry0 * rvy) / vv)) : 0;
  const minFree = Math.hypot(rx0 + rvx * sFree, ry0 + rvy * sFree);
  if (minFree >= o.horizNm * 1.01 + 0.01) return null;

  // vertical feasibility interval
  const vI = linInterval(dz0, dz1 - dz0, -o.vertFt, o.vertFt);
  if (!vI) return null;

  // low-level exclusion: flagged only where at least one aircraft is at or
  // above (reference elevation + lowAltFt)
  let refFt = 0;
  const minZ = Math.min(A0[2], A1[2], B0[2], B1[2]);
  if (o.airportNear && minZ < o.lowAltFt + MAX_AIRPORT_ELEV_FT) {
    const mla = (A0[0] + B0[0]) / 2, mlo = wrap180((A0[1] + B0[1]) / 2);
    const key = `${mla.toFixed(2)}|${mlo.toFixed(2)}`;
    let el = o.aptCache.get(key);
    if (el === undefined) {
      let got: { elevFt: number | null } | null = null;
      try { got = o.airportNear(mla, mlo, o.airportRadiusNm); } catch { got = null; }
      el = got && got.elevFt != null && Number.isFinite(got.elevFt) ? got.elevFt : 0;
      o.aptCache.set(key, el);
    }
    refFt = el;
  }
  const low = refFt + o.lowAltFt;
  const iA = intersect(vI, linInterval(A0[2], A1[2] - A0[2], low, Infinity));
  const iB = intersect(vI, linInterval(B0[2], B1[2] - B0[2], low, Infinity));

  let best: { s: number; d: number } | null = null;
  for (const iv of [iA, iB]) {
    if (!iv) continue;
    const s = vv > 1e-18 ? Math.min(iv[1], Math.max(iv[0], -(rx0 * rvx + ry0 * rvy) / vv)) : iv[0];
    const d = Math.hypot(rx0 + rvx * s, ry0 + rvy * s);
    if (!best || d < best.d - 1e-12) best = { s, d };
  }
  if (!best) return null;
  const t = lo + (hi - lo) * best.s;
  const a = posAt(P, p, t), b = posAt(P, q, t);
  const horiz = distNm(a[0], a[1], b[0], b[1]);
  const vert = Math.abs(b[2] - a[2]);
  if (!(horiz < o.horizNm) || !(vert < o.vertFt)) return null;
  const nf = (k: number) => P.t1[k] > P.t0[k] ? Math.min(t - P.t0[k], P.t1[k] - t) : 0;
  return { t, horiz, vert, laP: a[0], loP: a[1], laQ: b[0], loQ: b[1], zP: a[2], zQ: b[2], nfP: nf(p), nfQ: nf(q) };
}

// ── grid ────────────────────────────────────────────────────────────────────

const LAT_ROWS_OFFSET = 90;

function rowOf(lat: number, cellDeg: number): number {
  return Math.floor((Math.max(-90, Math.min(90, lat)) + LAT_ROWS_OFFSET) / cellDeg);
}

/** longitude cell width for a row — widened by 1/cos of the row's POLEWARD
 *  edge so a threshold-wide longitude span never skips a cell */
function rowWidthDeg(row: number, cellDeg: number): number {
  const latA = row * cellDeg - LAT_ROWS_OFFSET, latB = latA + cellDeg;
  const poleward = Math.min(89.9, Math.max(Math.abs(latA), Math.abs(latB)));
  return Math.min(360, cellDeg / Math.max(0.01, Math.cos(poleward * D2R)));
}

/** every [row, col] cell covering lat [la0,la1] × lon [lw,le] (lon may be
 *  outside ±180 — columns wrap) */
function cellsFor(la0: number, la1: number, lw: number, le: number, cellDeg: number, out: number[]): void {
  out.length = 0;
  const r0 = rowOf(la0, cellDeg), r1 = rowOf(la1, cellDeg);
  for (let r = r0; r <= r1; r++) {
    const w = rowWidthDeg(r, cellDeg);
    const ncol = Math.ceil(360 / w);
    const c0 = Math.floor((lw + 180) / w), c1 = Math.floor((le + 180) / w);
    if (c1 - c0 + 1 >= ncol) {
      for (let c = 0; c < ncol; c++) out.push(r * 100_000 + c);
    } else {
      for (let c = c0; c <= c1; c++) out.push(r * 100_000 + (((c % ncol) + ncol) % ncol));
    }
  }
}

/**
 * Find close approaches across a window's tracks. Deterministic for a given
 * input (sorted output, stable tie-breaks). Synchronous — tests and small
 * inputs; the HTTP path uses findCloseApproachesAsync so a dense window
 * never blocks the shared event loop (the trading loop runs in this process).
 */
export function findCloseApproaches(tracks: CloseApproachTrack[], opts: CloseApproachOptions = {}): CloseApproachResult {
  const it = runCloseApproaches(tracks, opts);
  for (;;) {
    const r = it.next();
    if (r.done) return r.value;
  }
}

/** Same result as findCloseApproaches, yielding to the event loop whenever
 *  a slice of work exceeds `sliceMs` (default 12 ms). */
export async function findCloseApproachesAsync(
  tracks: CloseApproachTrack[], opts: CloseApproachOptions = {}, sliceMs = 12,
): Promise<CloseApproachResult> {
  const it = runCloseApproaches(tracks, opts);
  let sliceStart = Date.now();
  for (;;) {
    const r = it.next();
    if (r.done) return r.value;
    if (Date.now() - sliceStart >= sliceMs) {
      await new Promise<void>((res) => setImmediate(res));
      sliceStart = Date.now();
    }
  }
}

/** the work, as a generator that yields between units (time buckets /
 *  brute-force rows) so the async driver can interleave other requests */
function* runCloseApproaches(tracks: CloseApproachTrack[], opts: CloseApproachOptions): Generator<void, CloseApproachResult, void> {
  const horizNm = opts.horizNm ?? CA_DEFAULTS.HORIZ_NM;
  const vertFt = opts.vertFt ?? CA_DEFAULTS.VERT_FT;
  const fixWindow = opts.fixWindowSec ?? CA_DEFAULTS.FIX_WINDOW_SEC;
  const cap = opts.cap ?? CA_DEFAULTS.CAP;
  const gap = opts.encounterGapSec ?? CA_DEFAULTS.ENCOUNTER_GAP_SEC;
  const bucketSec = opts.bucketSec ?? CA_DEFAULTS.BUCKET_SEC;
  const testOpts = {
    horizNm, vertFt,
    lowAltFt: opts.lowAltFt ?? CA_DEFAULTS.LOW_ALT_FT,
    airportRadiusNm: opts.airportRadiusNm ?? CA_DEFAULTS.AIRPORT_RADIUS_NM,
    airportNear: opts.airportNear,
    aptCache: new Map<string, number>(),
  };

  const { pieces: P, hexIds, calls, evaluated, nonIcao } = buildPieces(tracks, 2 * fixWindow);
  const hitsByPair = new Map<number, Hit[]>(); // key = hexA * H + hexB (hexA < hexB by index)
  const H = hexIds.length;

  const consider = (p: number, q: number) => {
    const hp = P.hex[p], hq = P.hex[q];
    if (hp === hq) return;
    const [x, y] = hp < hq ? [p, q] : [q, p];
    const hit = testPair(P, x, y, testOpts);
    if (!hit) return;
    const key = Math.min(hp, hq) * H + Math.max(hp, hq);
    const arr = hitsByPair.get(key);
    if (arr) arr.push(hit); else hitsByPair.set(key, [hit]);
  };

  if (opts.useGrid === false) {
    // brute-force reference: every time-overlapping piece pair
    const order = Array.from({ length: P.n }, (_, k) => k).sort((a, b) => P.t0[a] - P.t0[b] || a - b);
    for (let ii = 0; ii < order.length; ii++) {
      const p = order[ii];
      for (let jj = ii + 1; jj < order.length; jj++) {
        const q = order[jj];
        if (P.t0[q] > P.t1[p]) break;
        consider(p, q);
      }
      if ((ii & 255) === 255) yield;
    }
  } else {
    const cellDeg = horizNm / NM_PER_DEG_LAT;
    const dLat = cellDeg; // threshold expansion in latitude
    // (piece, bucket) entries sorted by bucket
    const entK: number[] = [];
    const entP: number[] = [];
    for (let k = 0; k < P.n; k++) {
      const b0 = Math.floor(P.t0[k] / bucketSec), b1 = Math.floor(P.t1[k] / bucketSec);
      for (let b = b0; b <= b1; b++) { entK.push(b); entP.push(k); }
    }
    const idx = Array.from({ length: entK.length }, (_, i) => i).sort((a, b) => entK[a] - entK[b] || entP[a] - entP[b]);
    const cells: number[] = [];
    let g = 0;
    while (g < idx.length) {
      // per-bucket dedupe (a pair spanning two buckets is re-tested there —
      // an identical hit, harmless to the encounter minimum; a global set
      // would grow with every candidate pair in dense terminal airspace)
      const seen = new Set<number>();
      const bucket = entK[idx[g]];
      let e = g;
      while (e < idx.length && entK[idx[e]] === bucket) e++;
      // bboxes of each piece's sub-interval inside this bucket
      const members: number[] = [];
      const boxes: number[] = []; // la0, la1, lw, le per member
      for (let m = g; m < e; m++) {
        const k = entP[idx[m]];
        const s0 = Math.max(P.t0[k], bucket * bucketSec);
        const s1 = Math.min(P.t1[k], (bucket + 1) * bucketSec);
        const a = posAt(P, k, s0), b = posAt(P, k, s1);
        members.push(k);
        boxes.push(Math.min(a[0], b[0]), Math.max(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[1], b[1]));
      }
      const grid = new Map<number, number[]>();
      for (let m = 0; m < members.length; m++) {
        cellsFor(boxes[m * 4], boxes[m * 4 + 1], boxes[m * 4 + 2], boxes[m * 4 + 3], cellDeg, cells);
        for (const c of cells) {
          const arr = grid.get(c);
          if (arr) arr.push(m); else grid.set(c, [m]);
        }
      }
      for (let m = 0; m < members.length; m++) {
        const la0 = boxes[m * 4] - dLat, la1 = boxes[m * 4 + 1] + dLat;
        const poleward = Math.min(89.9, Math.max(Math.abs(la0), Math.abs(la1)));
        const dLon = dLat / Math.max(0.01, Math.cos(poleward * D2R));
        cellsFor(la0, la1, boxes[m * 4 + 2] - dLon, boxes[m * 4 + 3] + dLon, cellDeg, cells);
        const p = members[m];
        for (const c of cells) {
          const arr = grid.get(c);
          if (!arr) continue;
          for (const mm of arr) {
            const q = members[mm];
            if (q === p || P.hex[q] === P.hex[p]) continue;
            const key = Math.min(p, q) * P.n + Math.max(p, q);
            if (seen.has(key)) continue;
            seen.add(key);
            consider(p, q);
          }
        }
      }
      g = e;
      yield;
    }
  }

  // encounters: per aircraft pair, split flagged instants by gap, keep minimum
  const out: CloseApproach[] = [];
  const keys = Array.from(hitsByPair.keys()).sort((a, b) => a - b);
  for (const key of keys) {
    const ha = Math.floor(key / H), hb = key % H;
    const hits = (hitsByPair.get(key) as Hit[]).slice().sort((x, y) => x.t - y.t || x.horiz - y.horiz);
    let group: Hit[] = [];
    const flush = () => {
      if (!group.length) return;
      let m = group[0];
      for (const h of group) if (h.horiz < m.horiz - 1e-9 || (Math.abs(h.horiz - m.horiz) <= 1e-9 && h.vert < m.vert)) m = h;
      // hex strings ordered for a stable a/b; piece order put the lower hex INDEX first
      let hexA = hexIds[ha], hexB = hexIds[hb];
      let callA = calls[ha], callB = calls[hb];
      let zA = m.zP, zB = m.zQ;
      if (hexA > hexB) {
        [hexA, hexB] = [hexB, hexA];
        [callA, callB] = [callB, callA];
        [zA, zB] = [zB, zA];
      }
      out.push({
        a: hexA, b: hexB,
        ...(callA ? { ca: callA } : {}), ...(callB ? { cb: callB } : {}),
        t: Math.round(m.t * 1000),
        horizNm: Math.round(m.horiz * 100) / 100,
        vertFt: Math.round(m.vert),
        confidence: confidenceFor(m.nfP, m.nfQ),
        basis: CLOSE_APPROACH_BASIS,
        lat: Math.round(((m.laP + m.laQ) / 2) * 1e5) / 1e5,
        lon: Math.round(wrap180(m.loP + wrap180(m.loQ - m.loP) / 2) * 1e5) / 1e5,
        altAFt: Math.round(zA),
        altBFt: Math.round(zB),
      });
      group = [];
    };
    for (const h of hits) {
      if (group.length && h.t - group[group.length - 1].t > gap) flush();
      group.push(h);
    }
    flush();
  }
  out.sort((x, y) => x.horizNm - y.horizNm || x.vertFt - y.vertFt || x.t - y.t ||
    (x.a < y.a ? -1 : x.a > y.a ? 1 : 0) || (x.b < y.b ? -1 : x.b > y.b ? 1 : 0));
  return {
    approaches: out.slice(0, cap),
    found: out.length,
    capped: out.length > cap,
    evaluated_hexes: evaluated,
    excluded_non_icao: nonIcao,
    pieces: P.n,
  };
}
