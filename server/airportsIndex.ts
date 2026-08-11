// QC-2 airports index (human directive 2026-08-11: "the plane need to take
// off from an airport at its height and the land at an airport at the
// height we need software for that"). Loads the pruned OurAirports dataset
// (datacore/aircraft/airports_min.json — 72k open fields worldwide with
// elevations, public-domain dedication; built by scripts/build_airports.mjs)
// into a degree-grid for fast nearest-airport lookups. Lazy singleton: the
// 6.5MB JSON parses once on first use, never at server boot.

import fs from "fs";
import path from "path";
import { repoDataPath } from "./repoFiles";

export interface Airport {
  id: string;            // OurAirports ident (ICAO-style where assigned)
  n: string;             // name
  la: number; lo: number;
  el: number | null;     // field elevation METERS (null = not published)
  ty: "L" | "M" | "S" | "W" | "H"; // large/medium/small/seaplane/heliport
}

export interface AirportMatch extends Airport { dist_km: number }

const GRID_DEG = 0.5;

let grid: Map<string, Airport[]> | null = null;
let loadedCount = 0;

function cellKey(la: number, lo: number): string {
  return `${Math.floor(la / GRID_DEG)}|${Math.floor(lo / GRID_DEG)}`;
}

export function loadAirports(jsonPath?: string): number {
  if (grid) return loadedCount;
  // repoDataPath (R14 class): dev reads the repo tree, Railway reads the
  // dist/ copy the build ships — never a bare cwd join.
  const p = jsonPath || repoDataPath(path.join("datacore", "aircraft", "airports_min.json"));
  try {
    const d = JSON.parse(fs.readFileSync(p, "utf-8"));
    grid = new Map();
    for (const a of d.airports || []) {
      const k = cellKey(a.la, a.lo);
      const arr = grid.get(k) || [];
      arr.push(a);
      grid.set(k, arr);
    }
    loadedCount = (d.airports || []).length;
  } catch (e: any) {
    console.error("[airports] load:", e?.message || e);
    grid = new Map(); // degrade to no-matches, never throw at call sites
    loadedCount = 0;
  }
  return loadedCount;
}

export function haversineKm(la1: number, lo1: number, la2: number, lo2: number): number {
  const R = 6371;
  const dLa = (la2 - la1) * Math.PI / 180;
  const dLo = (lo2 - lo1) * Math.PI / 180;
  const a = Math.sin(dLa / 2) ** 2
    + Math.cos(la1 * Math.PI / 180) * Math.cos(la2 * Math.PI / 180) * Math.sin(dLo / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

/** Nearest open airport within maxKm (default 6 — generous for big fields'
 *  reference-point offset, tight enough to reject mid-route points).
 *  `types` restricts matches (fixed-wing trips shouldn't "land" at a
 *  heliport by proximity accident — pass ["L","M","S","W"]). */
export function nearestAirport(
  la: number, lo: number, maxKm = 6,
  types?: ReadonlyArray<Airport["ty"]>,
): AirportMatch | null {
  if (!grid) loadAirports();
  if (!grid || !Number.isFinite(la) || !Number.isFinite(lo)) return null;
  const span = Math.ceil(maxKm / 111 / GRID_DEG) + 1;
  const cla = Math.floor(la / GRID_DEG), clo = Math.floor(lo / GRID_DEG);
  let best: AirportMatch | null = null;
  for (let dy = -span; dy <= span; dy++) {
    for (let dx = -span; dx <= span; dx++) {
      const arr = grid.get(`${cla + dy}|${clo + dx}`);
      if (!arr) continue;
      for (const a of arr) {
        if (types && !types.includes(a.ty)) continue;
        const d = haversineKm(la, lo, a.la, a.lo);
        if (d <= maxKm && (!best || d < best.dist_km)) best = { ...a, dist_km: +d.toFixed(2) };
      }
    }
  }
  return best;
}

/** test hook */
export function _resetAirports(): void { grid = null; loadedCount = 0; }
