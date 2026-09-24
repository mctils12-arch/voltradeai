/**
 * gemLngCarriers.ts — GEM "Global LNG Carrier Tracker", RAW-DATA overlay
 * (CLAUDE.md RAW-vs-SIGNAL surface rule): a catalogued observation, no
 * predictive claim.
 *
 * Source: datacore/gem/lng_carriers.json, already ingested by
 * scripts/gem_ingest.py from GEM's release (CC BY 4.0, per the release's
 * own Copyright sheet — see datacore/manifests/gem.json for the full
 * artifact provenance). STATIC reference dataset, same seeded pattern as
 * gemChemicals.ts / gemIronSteelPlants.ts / gemIronOreMines.ts /
 * gemCoalTerminals.ts: GEM ships a new release ~2x/year and a human
 * re-runs the ingest script on delivery — no boot-poll loop, no
 * archive-append machinery needed here.
 *
 * HONESTY FRAMING (this is why this file is NOT a per-carrier point layer,
 * the shape every sibling GEM module in this family uses): the only lat/lon
 * this release carries per carrier is "Yard location latitude/longitude" —
 * where the ship was BUILT, not its current position. Live-verified this
 * session (research/open_questions.md's 2026-09-22 GEM-suite backlog entry
 * flagged this exact gap, "needs the shipyard honesty framing before
 * shipping"): of 1,143 carriers with usable yard coordinates, there are
 * only 32 DISTINCT (lat, lon) pairs, and each one maps to exactly one
 * "Shipbuilder" name and one "Shipbuilder yard country/area" — i.e. the
 * per-row coordinate is really a per-SHIPYARD fact repeated once per hull
 * built there, not 1,143 independent locations. Plotting 1,143 points would
 * either stack ~1,100 invisible duplicates on 32 pixels or imply 1,143
 * distinct vessel positions — both dishonest. Instead this module
 * AGGREGATES by shipyard: one point per shipbuilding yard (32 total, 1,125
 * located carriers), carrying the count of carriers built there by
 * lifecycle status (active / on order / proposed) and the summed nameplate
 * capacity of the carriers with a known capacity. This is a genuinely
 * different, and arguably more useful, fact: where the world's LNG carrier
 * fleet came from (global shipbuilding capacity concentration), not a
 * vessel tracker. 18 of the release's 1,143 carriers have no yard
 * coordinates on file (all "proposed" — no yard has been assigned yet) and
 * are dropped, honestly, not guessed.
 *
 * Unlike the packed "lat, lon" Coordinates string other GEM releases in
 * this family use, this release already carries separate numeric
 * "Yard location latitude"/"Yard location longitude" fields — no string
 * parsing needed.
 */
import fs from "fs";
import { repoDataPath } from "./repoFiles";

export type LngCarrierStatus = "active" | "on_order" | "proposed" | "other";

const KNOWN_STATUSES: Record<string, LngCarrierStatus> = {
  active: "active",
  "on order": "on_order",
  proposed: "proposed",
};

/** Normalizes GEM's free-text "Status" column into one of the 3 lifecycle
 *  buckets exhaustively observed in the release (live-verified this
 *  session). An unrecognized/blank value falls back to "other" honestly,
 *  never guessed as "active". */
export function classifyCarrierStatus(raw: string | null | undefined): LngCarrierStatus {
  const s = String(raw || "").trim().toLowerCase();
  return KNOWN_STATUSES[s] || "other";
}

function toStrOrNull(v: unknown): string | null {
  if (v == null) return null;
  const s = String(v).trim();
  return s.length > 0 ? s : null;
}
function toNumOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Slugifies a shipbuilder name into a stable id. Live-verified this
 *  session: exactly one shipbuilder per distinct yard coordinate across
 *  the whole release (32 of each), so the slug is unique by construction —
 *  no counter-suffix collision handling needed. */
export function slugifyShipyardId(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export interface LngShipyard {
  id: string; // slug of Shipbuilder name — unique across the release (verified)
  shipbuilder: string;
  country: string | null;
  lat: number;
  lon: number;
  coordinateAccuracy: string | null;
  carrierCount: number;
  activeCount: number;
  onOrderCount: number;
  proposedCount: number;
  otherCount: number;
  totalCapacityCbm: number | null; // sum of known "Capacity" (cbm) across ALL carriers built here, any status; null if none reported
  knownCapacityCount: number;
}

/** Aggregates GEM's per-carrier rows into one row per shipbuilding yard.
 *  Drops rows with no usable yard coordinates or no shipbuilder name —
 *  nothing to key or place on a map (same drop-not-infer rule every
 *  sibling GEM module in this family uses). */
export function normalizeLngShipyards(rows: Record<string, unknown>[]): LngShipyard[] {
  type Group = {
    shipbuilder: string;
    country: string | null;
    lat: number;
    lon: number;
    coordinateAccuracy: string | null;
    active: number;
    onOrder: number;
    proposed: number;
    other: number;
    capacitySum: number;
    capacityCount: number;
  };
  const groups = new Map<string, Group>();
  for (const r of rows || []) {
    const lat = toNumOrNull(r["Yard location latitude"]);
    const lon = toNumOrNull(r["Yard location longitude"]);
    const shipbuilder = toStrOrNull(r["Shipbuilder"]);
    if (lat == null || lon == null || !shipbuilder) continue;
    const key = `${lat},${lon}`;
    let g = groups.get(key);
    if (!g) {
      g = {
        shipbuilder,
        country: toStrOrNull(r["Shipbuilder yard country/area"]),
        lat,
        lon,
        coordinateAccuracy: toStrOrNull(r["Yard location accuracy"]),
        active: 0, onOrder: 0, proposed: 0, other: 0,
        capacitySum: 0, capacityCount: 0,
      };
      groups.set(key, g);
    }
    const status = classifyCarrierStatus(toStrOrNull(r["Status"]));
    if (status === "active") g.active++;
    else if (status === "on_order") g.onOrder++;
    else if (status === "proposed") g.proposed++;
    else g.other++;
    const cap = toNumOrNull(r["Capacity"]);
    if (cap != null) { g.capacitySum += cap; g.capacityCount++; }
  }
  const out: LngShipyard[] = [];
  for (const g of groups.values()) {
    const carrierCount = g.active + g.onOrder + g.proposed + g.other;
    out.push({
      id: slugifyShipyardId(g.shipbuilder),
      shipbuilder: g.shipbuilder,
      country: g.country,
      lat: g.lat,
      lon: g.lon,
      coordinateAccuracy: g.coordinateAccuracy,
      carrierCount,
      activeCount: g.active,
      onOrderCount: g.onOrder,
      proposedCount: g.proposed,
      otherCount: g.other,
      totalCapacityCbm: g.capacityCount > 0 ? g.capacitySum : null,
      knownCapacityCount: g.capacityCount,
    });
  }
  out.sort((a, b) => b.carrierCount - a.carrierCount);
  return out;
}

export interface LngShipyardsResult {
  release: string | null;
  attribution: string;
  license: string;
  shipyards: LngShipyard[];
  totalCarriers: number;
}

/** Reads datacore/gem/lng_carriers.json. Never throws: a missing/corrupt
 *  file degrades to null, matching loadGemChemicals's fetch-failure-
 *  degrades precedent, so the route can serve an honest "unavailable"
 *  instead of crashing the process. */
export function loadGemLngShipyards(
  fp: string = repoDataPath("datacore/gem/lng_carriers.json"),
): LngShipyardsResult | null {
  try {
    const raw = JSON.parse(fs.readFileSync(fp, "utf8"));
    const prov = raw.provenance || {};
    const shipyards = normalizeLngShipyards(raw.carriers || []);
    return {
      release: toStrOrNull(prov.release),
      attribution: typeof prov.attribution === "string" ? prov.attribution : "Global Energy Monitor",
      license: typeof prov.license === "string" ? prov.license : "CC BY 4.0",
      shipyards,
      totalCarriers: shipyards.reduce((sum, s) => sum + s.carrierCount, 0),
    };
  } catch {
    return null;
  }
}

// In-memory cache — same rationale as gemChemicals.ts's `cached`: static
// reference data, parse+aggregate once per process lifetime.
let cached: LngShipyardsResult | null | undefined;

export function cachedGemLngShipyards(): LngShipyardsResult | null {
  if (cached === undefined) cached = loadGemLngShipyards();
  return cached;
}

/** Test-only: clears the module cache so a test can inject a different fp. */
export function _resetGemLngShipyardsCacheForTests(): void {
  cached = undefined;
}
