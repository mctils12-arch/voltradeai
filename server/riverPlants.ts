// riverPlants — Pillar-1 inference cross-tie (worldview-globe backend inference).
// Which US power plants sit on/near a barge-corridor river gauge, so a LOW-WATER
// reading immediately names the generating capacity exposed to (a) cooling-water
// limits at river-cooled thermal plants and (b) barge draft restrictions on coal
// delivery. Live keyless data only — USGS NWIS (public domain) × the WRI Global
// Power Plant DB (CC BY 4.0). Reuses haversineKm from firesFacilities.ts.
//
// PURE + tested, same discipline as firesFacilities.ts (#388). It computes the
// PROXIMITY join only — it makes NO predictive claim; the "low water near
// dependent plants → operator/utility returns" hypothesis is gate-2 and filed
// in research/open_questions.md.
import { haversineKm } from "./firesFacilities";

export interface GaugePoint { site: string; name: string | null; lat: number; lon: number }
// WRI powerplant tuple as stored in datacore/powerplants/us_power_plants.json:
// [name, capacityMW, fuel, operator, lat, lon, flag?]
export type PlantTuple = [string, number, string, string, number, number, (number | undefined)?];

export interface PlantNear {
  name: string; fuel: string; capacity_mw: number; operator: string; distance_km: number;
}
export interface GaugePlantHit {
  site: string; name: string | null; lat: number; lon: number;
  plants: PlantNear[]; plant_count: number; total_capacity_mw: number;
  capacity_by_fuel: Record<string, number>;
}

/** The plant tuples out of the shipped datacore/powerplants/us_power_plants.json
 *  module, which wraps them in `{_doc, source, fuels, count, verified_count,
 *  plants}` — casting that object straight to PlantTuple[] compiles but is not
 *  iterable at runtime, which 500'd both cross-tie routes. Returns [] rather
 *  than throwing if the shape ever changes again. Pure. */
export function powerplantTable(raw: unknown): PlantTuple[] {
  if (Array.isArray(raw)) return raw as PlantTuple[];
  const plants = (raw as { plants?: unknown } | null)?.plants;
  return Array.isArray(plants) ? (plants as PlantTuple[]) : [];
}

/** Dedupe gauge observations to one POINT per site (a site publishes stage AND
 *  discharge as separate series; the cross-tie only needs its location once).
 *  Skips gauges with missing/non-finite coordinates. Pure. */
export function gaugePoints(
  gauges: Array<{ site: string; name: string | null; lat: number | null; lon: number | null }>,
): GaugePoint[] {
  const seen = new Map<string, GaugePoint>();
  for (const g of gauges) {
    if (!g || typeof g.lat !== "number" || typeof g.lon !== "number") continue;
    if (!Number.isFinite(g.lat) || !Number.isFinite(g.lon)) continue;
    if (!seen.has(g.site)) seen.set(g.site, { site: g.site, name: g.name ?? null, lat: g.lat, lon: g.lon });
  }
  return Array.from(seen.values());
}

/** For each gauge, the power plants within radiusKm — nearest-first — with
 *  aggregate capacity and a per-fuel breakdown. Returns only gauges that have at
 *  least one plant nearby, ordered by MOST exposed capacity first (the
 *  operationally interesting ordering). Pure — caller passes the cached gauge
 *  points + the powerplant table + radius. Never fabricates a capacity: a
 *  non-numeric MW counts as 0, not a guess. */
export function plantsNearGauges(
  gauges: GaugePoint[], plants: PlantTuple[], radiusKm: number,
): GaugePlantHit[] {
  const out: GaugePlantHit[] = [];
  for (const g of gauges) {
    const near: PlantNear[] = [];
    for (const p of plants) {
      const lat = p[4], lon = p[5];
      if (typeof lat !== "number" || typeof lon !== "number") continue;
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
      const d = haversineKm(g.lat, g.lon, lat, lon);
      if (d <= radiusKm) {
        near.push({
          name: p[0], fuel: p[2],
          capacity_mw: typeof p[1] === "number" && Number.isFinite(p[1]) ? p[1] : 0,
          operator: p[3], distance_km: Math.round(d * 10) / 10,
        });
      }
    }
    if (!near.length) continue;
    near.sort((a, b) => a.distance_km - b.distance_km);
    const byFuel: Record<string, number> = {};
    let total = 0;
    for (const p of near) { byFuel[p.fuel] = (byFuel[p.fuel] || 0) + p.capacity_mw; total += p.capacity_mw; }
    out.push({
      site: g.site, name: g.name, lat: g.lat, lon: g.lon,
      plants: near, plant_count: near.length,
      total_capacity_mw: Math.round(total), capacity_by_fuel: byFuel,
    });
  }
  out.sort((a, b) => b.total_capacity_mw - a.total_capacity_mw);
  return out;
}
