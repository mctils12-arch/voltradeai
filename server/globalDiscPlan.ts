// globalDiscPlan.ts — the STATIC world disc plan for the global sweep
// (FLIGHT PROGRAM B1, 2026-09-28). Pure, deterministic, no I/O.
//
// adsb.lol's only geographic query is a point + radius (hard max 250nm).
// To see "every plane on Earth" through it, the sweep rotates through a
// fixed set of 250nm discs covering where aircraft can actually be HEARD:
// adsb.lol is a network of volunteer GROUND receivers (no satellite ADS-B),
// so a disc in the middle of an ocean, or over Antarctica, returns nothing
// by construction — polling it is pure cost to a free community API.
//
// MASK SOURCE (honest): the production archive's historical density would
// be the ideal mask, but it lives on the Railway volume and is not
// available at build time, so the mask is a HAND-AUTHORED list of lat/lon
// boxes (TRAFFIC_MASK below): every inhabited landmass with receiver
// coverage, the island airports long-haul traffic uses, and the major
// oceanic corridors whose coastal ends ARE heard (North Atlantic Tracks,
// North Pacific, US–Hawaii, Tasman). Remote open ocean (mid-South-Atlantic,
// central Indian Ocean, South Pacific), the high Arctic above 72°N and
// Antarctica are deliberately NOT in the plan — the global endpoint's
// coverage block says so on every response. Each box carries a `prior`
// (rough aircraft-per-disc expectation) used ONLY to order the first sweep
// pass (dense regions first); after one visit the scheduler learns the real
// per-disc counts.
//
// GEOMETRY: a zonal grid of CELLS, each served by the disc at its center.
// Rows are CELL_HEIGHT_NM tall (r·√2·safety — the square cell a radius-r
// disc covers exactly); within a row the cell width is solved so the
// cell's EQUATORWARD corners (the farthest points: a degree of longitude
// is widest there) sit exactly on the disc edge by great-circle distance,
// then rounded so the row closes around the globe. Every point of every
// cell is therefore within r of its own cell's center — no reliance on
// hex interleaving between rows of different sizes (an earlier hex-lattice
// draft left measurable gaps where adjacent rows had different counts).
// A cell is KEPT iff it intersects a mask box, which makes the cover exact:
// any point inside any box lies in some kept cell, hence in its disc. The
// guarantee is VERIFIED, not assumed: globalDiscPlan.test.ts samples every
// box on a dense grid plus ~60 major airports with haversine distance.

import { EARTH_RADIUS_NM } from "../shared/flightPlanGeometry";

export const PLAN_RADIUS_NM = 250;
/** Shrink factor on the cell height + corner radius (numeric margin; the
 *  cover itself is exact by construction and test-verified). */
export const LATTICE_SAFETY = 0.98;
/** Plan latitude extent (the sweep never goes poleward of this). */
export const PLAN_LAT_MIN = -56;
export const PLAN_LAT_MAX = 72;

// one source of truth for the Earth radius in nm (D11 dup_precise_literal)
const R_EARTH_NM = EARTH_RADIUS_NM;
const NM_PER_DEG = 60;
const toRad = (d: number) => (d * Math.PI) / 180;

export interface MaskBox {
  name: string;
  lat0: number; lat1: number;
  /** lon0 < lon1, both in [-180, 180] (antimeridian boxes are split) */
  lon0: number; lon1: number;
  /** rough aircraft-per-disc expectation — FIRST-PASS ORDER ONLY */
  prior: number;
}

/** The hand-authored traffic mask (see header). */
export const TRAFFIC_MASK: MaskBox[] = [
  // ── North America ──
  { name: "us-contiguous", lat0: 24.5, lat1: 49.5, lon0: -125, lon1: -66.5, prior: 300 },
  { name: "canada-south", lat0: 42, lat1: 56, lon0: -130, lon1: -52, prior: 80 },
  { name: "canada-north-polar-routes", lat0: 56, lat1: 72, lon0: -141, lon1: -55, prior: 8 },
  { name: "alaska", lat0: 51, lat1: 71.5, lon0: -170, lon1: -129, prior: 15 },
  { name: "mexico-central-america", lat0: 7, lat1: 32.7, lon0: -118, lon1: -77, prior: 40 },
  { name: "caribbean", lat0: 10, lat1: 27.5, lon0: -86, lon1: -59, prior: 30 },
  { name: "bermuda", lat0: 31, lat1: 33.5, lon0: -66, lon1: -63.5, prior: 3 },
  { name: "hawaii", lat0: 18.5, lat1: 22.5, lon0: -160.5, lon1: -154.5, prior: 20 },
  { name: "us-hawaii-corridor", lat0: 20, lat1: 38, lon0: -158, lon1: -122, prior: 3 },
  // ── South America ──
  { name: "south-america-north", lat0: -20, lat1: 13, lon0: -82, lon1: -34, prior: 30 },
  { name: "south-america-south", lat0: -56, lat1: -20, lon0: -76, lon1: -40, prior: 20 },
  // ── Europe + North Atlantic ──
  { name: "europe", lat0: 35, lat1: 71.5, lon0: -11, lon1: 45, prior: 300 },
  { name: "atlantic-islands", lat0: 14.5, lat1: 40, lon0: -32, lon1: -12, prior: 10 },
  { name: "iceland-faroe", lat0: 62, lat1: 67.5, lon0: -25, lon1: -6, prior: 10 },
  { name: "north-atlantic-tracks", lat0: 44, lat1: 62, lon0: -58, lon1: -10, prior: 10 },
  { name: "greenland-south", lat0: 59, lat1: 71, lon0: -56, lon1: -20, prior: 3 },
  // ── Africa + Middle East ──
  { name: "north-africa", lat0: 15, lat1: 37.5, lon0: -18, lon1: 36, prior: 30 },
  { name: "sub-saharan-africa", lat0: -35, lat1: 15, lon0: -18, lon1: 52, prior: 15 },
  { name: "middle-east", lat0: 12, lat1: 42, lon0: 34, lon1: 63, prior: 120 },
  { name: "indian-ocean-islands", lat0: -26, lat1: -3, lon0: 44, lon1: 64, prior: 5 },
  // ── Asia ──
  { name: "russia-west-central-asia", lat0: 40, lat1: 70, lon0: 45, lon1: 90, prior: 30 },
  { name: "siberia-far-east", lat0: 42, lat1: 72, lon0: 90, lon1: 180, prior: 10 },
  { name: "south-asia", lat0: -1, lat1: 37, lon0: 60, lon1: 98, prior: 80 },
  { name: "east-asia", lat0: 18, lat1: 54, lon0: 98, lon1: 146, prior: 150 },
  { name: "southeast-asia", lat0: -11.5, lat1: 23, lon0: 92, lon1: 141, prior: 60 },
  { name: "micronesia-guam", lat0: 6, lat1: 21, lon0: 130, lon1: 147, prior: 5 },
  { name: "north-pacific-corridor-w", lat0: 45, lat1: 60, lon0: 145, lon1: 180, prior: 3 },
  { name: "north-pacific-corridor-e", lat0: 45, lat1: 60, lon0: -180, lon1: -160, prior: 3 },
  // ── Oceania ──
  { name: "australia", lat0: -44, lat1: -10, lon0: 112, lon1: 154, prior: 40 },
  { name: "new-zealand", lat0: -47.5, lat1: -34, lon0: 166, lon1: 179, prior: 15 },
  { name: "tasman", lat0: -45, lat1: -30, lon0: 150, lon1: 175, prior: 3 },
  { name: "south-pacific-islands-w", lat0: -23, lat1: -12, lon0: 162, lon1: 180, prior: 3 },
  { name: "south-pacific-islands-e", lat0: -23, lat1: -12, lon0: -180, lon1: -168, prior: 3 },
  { name: "tahiti", lat0: -18.5, lat1: -16, lon0: -152, lon1: -148, prior: 2 },
];

export interface PlanDisc {
  /** stable id (index in the deterministic plan) */
  id: number;
  lat: number;
  lon: number;
  radiusNm: number;
  /** first-pass ordering prior (max prior of the boxes the disc reaches) */
  prior: number;
}

/** Great-circle distance in nautical miles. */
export function haversineNm(aLat: number, aLon: number, bLat: number, bLon: number): number {
  const dLat = toRad(bLat - aLat);
  const dLon = toRad(bLon - aLon);
  const s = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(aLat)) * Math.cos(toRad(bLat)) * Math.sin(dLon / 2) ** 2;
  return 2 * R_EARTH_NM * Math.asin(Math.min(1, Math.sqrt(s)));
}

/** Wrap a longitude into [-180, 180). */
export function wrapLon(lon: number): number {
  const w = ((((lon + 180) % 360) + 360) % 360) - 180;
  return w;
}

/** Point-in-box (longitude wrapped into [-180, 180); a box ending at 180
 *  also owns -180, the same meridian). */
export function inBox(lat: number, lon: number, b: MaskBox): boolean {
  if (lat < b.lat0 || lat > b.lat1) return false;
  const l = wrapLon(lon);
  return (l >= b.lon0 && l <= b.lon1) || (b.lon1 === 180 && l === -180);
}

/** Largest half-width (degrees of longitude) such that a point at
 *  (edgeLat, lon ± w) is within `radiusNm` of (centerLat, lon). */
function halfWidthDeg(centerLat: number, edgeLat: number, radiusNm: number): number {
  let lo = 0, hi = 180;
  for (let i = 0; i < 48; i++) {
    const mid = (lo + hi) / 2;
    if (haversineNm(centerLat, 0, edgeLat, mid) <= radiusNm) lo = mid; else hi = mid;
  }
  return lo;
}

/** Does the lon interval [a0, a1] (a0 < a1, may exceed ±180) intersect box b? */
function lonIntersects(a0: number, a1: number, b: MaskBox): boolean {
  for (const k of [-360, 0, 360]) {
    if (a0 + k <= b.lon1 && a1 + k >= b.lon0) return true;
  }
  return false;
}

/**
 * Build the deterministic world plan from the mask. Same input → same
 * output, forever (the plan is part of the coverage contract).
 */
export function buildWorldPlan(
  mask: MaskBox[] = TRAFFIC_MASK,
  radiusNm: number = PLAN_RADIUS_NM,
  safety: number = LATTICE_SAFETY,
): PlanDisc[] {
  const cellHDeg = (radiusNm * Math.SQRT2 * safety) / NM_PER_DEG;
  const out: PlanDisc[] = [];
  for (let la0 = PLAN_LAT_MIN; la0 < PLAN_LAT_MAX - 1e-9; la0 += cellHDeg) {
    const la1 = Math.min(PLAN_LAT_MAX, la0 + cellHDeg);
    const clat = (la0 + la1) / 2;
    // the equatorward edge is where the cell is widest in nm per degree
    const eqLat = Math.abs(la0) < Math.abs(la1) ? la0 : la1;
    const w = halfWidthDeg(clat, eqLat, radiusNm * safety);
    const n = Math.max(1, Math.ceil(360 / (2 * w)));
    const step = 360 / n; // exact division: the row closes around the globe
    for (let k = 0; k < n; k++) {
      const lo0 = -180 + k * step, lo1 = lo0 + step;
      let prior = -1;
      for (const b of mask) {
        if (la0 > b.lat1 || la1 < b.lat0) continue;
        if (lonIntersects(lo0, lo1, b)) prior = Math.max(prior, b.prior);
      }
      if (prior < 0) continue;
      out.push({
        id: out.length, lat: +clat.toFixed(4), lon: +wrapLon(lo0 + step / 2).toFixed(4),
        radiusNm, prior,
      });
    }
  }
  return out;
}

/** True iff (lat, lon) lies within some plan disc. */
export function coveredByPlan(lat: number, lon: number, plan: PlanDisc[]): boolean {
  for (const d of plan) {
    if (Math.abs(d.lat - lat) * NM_PER_DEG > d.radiusNm + 1) continue; // cheap lat reject
    if (haversineNm(lat, lon, d.lat, d.lon) <= d.radiusNm) return true;
  }
  return false;
}

/** The plan every process builds (memoized — pure function of constants). */
let _worldPlan: PlanDisc[] | null = null;
export function worldPlan(): PlanDisc[] {
  if (!_worldPlan) _worldPlan = buildWorldPlan();
  return _worldPlan;
}
