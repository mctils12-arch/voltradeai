// plantsUnderAlerts — Pillar-1 inference cross-tie (worldview-globe backend
// inference). Which US generating capacity currently sits INSIDE an active NWS
// severe-weather warning polygon — a live, keyless storm-risk-to-generation read
// distinct from the water-risk river cross-tie. Live NWS alerts (api.weather.gov,
// US-government public domain) × the WRI Global Power Plant DB (CC BY 4.0).
//
// PURE + tested (server/plantsUnderAlerts.test.ts). Makes NO predictive claim —
// the "generation under a warning → operator/grid disruption" hypothesis is
// gate-2 and filed in research/open_questions.md. Point-in-polygon is a standard
// even-odd ray cast; alert display rings are OUTER rings only (holes already
// dropped upstream in nwsAlerts.simplifyRings), so ring containment = coverage.
import type { PlantTuple } from "./riverPlants";

export type Ring = Array<[number, number]>; // [lon, lat] pairs
export interface AlertLike {
  id: string; event: string; severity: string | null; area: string | null;
  ends: string | null; rings: Ring[];
}

export interface PlantHit { name: string; fuel: string; capacity_mw: number; operator: string }
export interface AlertPlantHit {
  id: string; event: string; severity: string | null; area: string | null; ends: string | null;
  plants: PlantHit[]; plant_count: number; total_capacity_mw: number;
  capacity_by_fuel: Record<string, number>;
}

/** Even-odd ray-cast point-in-polygon. `ring` is [lon,lat] pairs; the point is
 *  (lon,lat). Boundary cases are not meaningful at plant/warning scale. Pure. */
export function pointInRing(lon: number, lat: number, ring: Ring): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1];
    const xj = ring[j][0], yj = ring[j][1];
    const intersect = (yi > lat) !== (yj > lat) &&
      lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

/** True if the point falls inside ANY of the alert's outer rings. Pure. */
export function pointInAnyRing(lon: number, lat: number, rings: Ring[]): boolean {
  for (const r of rings) if (r && r.length >= 3 && pointInRing(lon, lat, r)) return true;
  return false;
}

/** Axis-aligned bbox of a set of rings, or null if empty. Cheap pre-filter so a
 *  plant far from an alert never runs the full ray cast. Pure. */
export function ringsBbox(rings: Ring[]): [number, number, number, number] | null {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const r of rings) for (const [x, y] of r) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  return Number.isFinite(minX) ? [minX, minY, maxX, maxY] : null;
}

/**
 * For each alert with polygon geometry, the power plants inside it — with
 * aggregate capacity and a per-fuel breakdown. Returns only alerts that cover at
 * least one plant, ordered by MOST exposed capacity first. Pure — caller passes
 * the cached alert display list + the powerplant table. A non-numeric MW counts
 * as 0, never a fabricated guess.
 */
export function plantsUnderAlerts(alerts: AlertLike[], plants: PlantTuple[]): AlertPlantHit[] {
  const out: AlertPlantHit[] = [];
  for (const a of alerts) {
    const rings = (a.rings || []).filter((r) => r && r.length >= 3);
    if (!rings.length) continue; // zone-only alert (no polygon) — skip, honestly
    const bbox = ringsBbox(rings);
    if (!bbox) continue;
    const [minX, minY, maxX, maxY] = bbox;
    const inside: PlantHit[] = [];
    for (const p of plants) {
      const lat = p[4], lon = p[5];
      if (typeof lat !== "number" || typeof lon !== "number") continue;
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
      if (lon < minX || lon > maxX || lat < minY || lat > maxY) continue; // bbox pre-filter
      if (!pointInAnyRing(lon, lat, rings)) continue;
      inside.push({
        name: p[0], fuel: p[2],
        capacity_mw: typeof p[1] === "number" && Number.isFinite(p[1]) ? p[1] : 0,
        operator: p[3],
      });
    }
    if (!inside.length) continue;
    const byFuel: Record<string, number> = {};
    let total = 0;
    for (const p of inside) { byFuel[p.fuel] = (byFuel[p.fuel] || 0) + p.capacity_mw; total += p.capacity_mw; }
    inside.sort((x, y) => y.capacity_mw - x.capacity_mw);
    out.push({
      id: a.id, event: a.event, severity: a.severity, area: a.area, ends: a.ends,
      plants: inside, plant_count: inside.length,
      total_capacity_mw: Math.round(total), capacity_by_fuel: byFuel,
    });
  }
  out.sort((a, b) => b.total_capacity_mw - a.total_capacity_mw);
  return out;
}
