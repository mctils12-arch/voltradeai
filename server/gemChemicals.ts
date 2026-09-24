/**
 * gemChemicals.ts — GEM "Global Chemicals Inventory", RAW-DATA overlay
 * (CLAUDE.md RAW-vs-SIGNAL surface rule): a catalogued observation, no
 * predictive claim.
 *
 * Source: datacore/gem/chemicals.json, already ingested by
 * scripts/gem_ingest.py from GEM's release (CC BY 4.0, per the release's
 * own Copyright sheet — see datacore/manifests/gem.json for the full
 * artifact provenance). STATIC reference dataset, same seeded pattern as
 * gemIronSteelPlants.ts / gemIronOreMines.ts / gemCoalTerminals.ts: GEM
 * ships a new release ~2x/year and a human re-runs the ingest script on
 * delivery — no boot-poll loop, no archive-append machinery needed here.
 *
 * 868 chemical plants worldwide. This release, like iron_steel_plants.json,
 * has no single lifecycle-status column — live-verified this session
 * (every one of the 868 rows' own column set read directly, see the
 * `Counter` scan in this session's experiments.md entry). The signal_ladder
 * note for this root already recorded that "Primary products" was screened
 * and REJECTED as the color dimension (68 distinct free-text multi-product
 * combinations, no clean small bucket set). "Feedstock" is the alternative
 * used here instead: also free-text and semicolon-separated, but the top
 * single tokens (natural gas 255, coal 153, naphtha 147, ethane 86, crude
 * oil 70, propane 53...) cleanly cover the overwhelming majority of rows
 * into a small number of real feedstock families, standard industry
 * groupings (fossil-solid / fossil-gas / petroleum-liquid / NGL /
 * low-carbon), not invented here. classifyFeedstockFamily buckets by
 * simple presence-priority over the semicolon list (coal beats natural gas
 * beats petroleum-liquids beats NGLs beats low-carbon beats a residual
 * "other" that also catches the "unknown" sentinel and pure downstream-
 * chemical feedstocks like methanol/ethylene/benzene) — never infers a
 * feedstock the row doesn't state. Coordinates/date/sentinel handling is
 * byte-identical in shape to gemIronSteelPlants.ts's own conventions (same
 * GEM release family): "Coordinates" packs lat/lon into one "lat, lon"
 * string, parsed by parseGemCoordinates (duplicated locally rather than
 * imported, matching this file family's established one-module-per-file
 * convention).
 */
import fs from "fs";
import { repoDataPath } from "./repoFiles";

export type ChemicalFeedstockFamily = "coal" | "natural_gas" | "petroleum" | "ngl" | "low_carbon" | "other";

export interface ChemicalPlant {
  id: string; // "GEM plant ID"
  name: string;
  feedstockFamily: ChemicalFeedstockFamily;
  feedstockRaw: string | null;
  primaryProducts: string | null;
  secondaryProducts: string | null;
  country: string | null;
  region: string | null;
  municipality: string | null;
  subnationalUnit: string | null;
  coordinateAccuracy: string | null;
  owner: string | null;
  wiki: string | null;
  lat: number;
  lon: number;
}

// Sentinel strings the release uses for "no value reported" (case-
// insensitive) — matches gemIronSteelPlants.ts's NULL_SENTINELS exactly
// (same GEM release family, same convention).
const NULL_SENTINELS = new Set(["", "-", "--", "n/a", "unknown", "unkonwn"]);

function toStrOrNull(v: unknown): string | null {
  if (v == null) return null;
  const s = String(v).trim();
  return s.length > 0 && !NULL_SENTINELS.has(s.toLowerCase()) ? s : null;
}

/** Parses GEM's packed "lat, lon" Coordinates string. Returns null on
 *  anything that doesn't cleanly split into two finite numbers — nothing
 *  to place on a map, dropped by the caller rather than guessed. Same
 *  logic as gemIronSteelPlants.ts's parseGemCoordinates. */
export function parseGemCoordinates(raw: unknown): { lat: number; lon: number } | null {
  if (typeof raw !== "string") return null;
  const parts = raw.split(",").map((p) => p.trim());
  if (parts.length !== 2) return null;
  const lat = Number(parts[0]);
  const lon = Number(parts[1]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  return { lat, lon };
}

// Presence-priority token sets. A token belongs to exactly one set; the
// FIRST matching set in priority order wins when a row lists several
// feedstocks (e.g. "ethane; naphtha" -> petroleum, "coal; natural gas" ->
// coal). "Coke oven gas" is a coal-derived byproduct but is bucketed with
// natural_gas here: chemically it behaves as a fuel gas feedstock, not the
// solid-coal route the "coal" bucket is meant to isolate — a deliberate,
// stated choice, not an artifact.
const NATURAL_GAS_TOKENS = new Set(["natural gas", "methane", "liquefied natural gas (lng)", "coke oven gas"]);
const PETROLEUM_TOKENS = new Set([
  "naphtha", "crude oil", "condensate", "heavy fuel oil", "fuel oil", "gas oil",
  "pyrolysis oil", "pyrolysis gasoline", "reformate",
]);
const NGL_TOKENS = new Set([
  "ethane", "propane", "butane", "liquid petroleum gas (lpg)", "natural gas liquids (ngl)", "mixed c4",
]);
const LOW_CARBON_TOKENS = new Set(["green hydrogen", "hydrogen", "biomass", "bioethanol", "carbon dioxide"]);

/** Buckets GEM's free-text, semicolon-separated "Feedstock" column into one
 *  of six real feedstock families. Anything unmatched (the "unknown"
 *  sentinel, blank, or a pure downstream-chemical feedstock like methanol/
 *  ethylene/benzene/acetic acid) falls to "other" honestly, never guessed. */
export function classifyFeedstockFamily(raw: string | null | undefined): ChemicalFeedstockFamily {
  const tokens = String(raw || "")
    .split(";")
    .map((t) => t.trim().toLowerCase())
    .filter(Boolean);
  const has = (set: Set<string>) => tokens.some((t) => set.has(t));
  if (tokens.includes("coal")) return "coal";
  if (has(NATURAL_GAS_TOKENS)) return "natural_gas";
  if (has(PETROLEUM_TOKENS)) return "petroleum";
  if (has(NGL_TOKENS)) return "ngl";
  if (has(LOW_CARBON_TOKENS)) return "low_carbon";
  return "other";
}

/** Normalizes the release's raw per-row columns into the clean API
 *  schema. Drops rows with no usable coordinates, no name, or no id —
 *  nothing to key or place on a map (same drop-not-infer rule as
 *  normalizeIronSteelPlants). */
export function normalizeChemicalPlants(rows: Record<string, unknown>[]): ChemicalPlant[] {
  const out: ChemicalPlant[] = [];
  for (const r of rows || []) {
    const id = toStrOrNull(r["GEM plant ID"]);
    const name = toStrOrNull(r["Plant name (English)"]);
    const coords = parseGemCoordinates(r["Coordinates"]);
    if (!id || !name || !coords) continue;
    const feedstockRaw = toStrOrNull(r["Feedstock"]);
    out.push({
      id,
      name,
      feedstockFamily: classifyFeedstockFamily(feedstockRaw),
      feedstockRaw,
      primaryProducts: toStrOrNull(r["Primary products"]),
      secondaryProducts: toStrOrNull(r["Secondary products"]),
      country: toStrOrNull(r["Country/area"]),
      region: toStrOrNull(r["Region"]),
      municipality: toStrOrNull(r["Municipality"]),
      subnationalUnit: toStrOrNull(r["Subnational unit"]),
      coordinateAccuracy: toStrOrNull(r["Coordinate accuracy"]),
      owner: toStrOrNull(r["Owner (English)"]),
      wiki: toStrOrNull(r["GEM wiki page"]),
      lat: coords.lat,
      lon: coords.lon,
    });
  }
  return out;
}

export interface ChemicalPlantsResult {
  release: string | null;
  attribution: string;
  license: string;
  plants: ChemicalPlant[];
}

/** Reads datacore/gem/chemicals.json. Never throws: a missing/corrupt file
 *  degrades to null, matching loadGemIronSteelPlants's fetch-failure-
 *  degrades precedent, so the route can serve an honest "unavailable"
 *  instead of crashing the process. */
export function loadGemChemicals(
  fp: string = repoDataPath("datacore/gem/chemicals.json"),
): ChemicalPlantsResult | null {
  try {
    const raw = JSON.parse(fs.readFileSync(fp, "utf8"));
    const prov = raw.provenance || {};
    return {
      release: toStrOrNull(prov.release),
      attribution: typeof prov.attribution === "string" ? prov.attribution : "Global Energy Monitor",
      license: typeof prov.license === "string" ? prov.license : "CC BY 4.0",
      plants: normalizeChemicalPlants(raw.plants || []),
    };
  } catch {
    return null;
  }
}

// In-memory cache — same rationale as gemIronSteelPlants.ts's `cached`:
// static reference data, parse once per process lifetime.
let cached: ChemicalPlantsResult | null | undefined;

export function cachedGemChemicals(): ChemicalPlantsResult | null {
  if (cached === undefined) cached = loadGemChemicals();
  return cached;
}

/** Test-only: clears the module cache so a test can inject a different fp. */
export function _resetGemChemicalsCacheForTests(): void {
  cached = undefined;
}
