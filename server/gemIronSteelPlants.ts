/**
 * gemIronSteelPlants.ts — GEM "Global Iron and Steel Tracker", RAW-DATA
 * overlay (CLAUDE.md RAW-vs-SIGNAL surface rule): a catalogued
 * observation, no predictive claim.
 *
 * Source: datacore/gem/iron_steel_plants.json, already ingested by
 * scripts/gem_ingest.py from GEM's release (CC BY 4.0, per the release's
 * own Copyright sheet — see datacore/manifests/gem.json for the full
 * artifact provenance). STATIC reference dataset, same seeded pattern as
 * gemIronOreMines.ts / gemCoalTerminals.ts: GEM ships a new release
 * ~2x/year and a human re-runs the ingest script on delivery — no
 * boot-poll loop, no archive-append machinery needed here.
 *
 * 1,293 iron and steel plants worldwide. This release has no single
 * "Operating status" lifecycle column the way iron_ore_mines.json does —
 * live-verified this session (`grep`-checked every column name across a
 * 200-row sample) — so the SYMBOLS NOT DOTS color dimension here instead
 * comes from "Main production equipment" (a catalogued, semicolon-
 * separated list of real steelmaking technology codes: BF = blast
 * furnace, BOF = basic oxygen furnace, DRI = direct reduced iron, EAF =
 * electric arc furnace, IF = induction furnace — standard industry
 * abbreviations, not invented here). classifyProductionTechnology buckets
 * that list into the plant's PRIMARY reduction route by simple presence-
 * priority (BF beats DRI beats EAF beats IF beats a lone BOF), never
 * infers a technology the row doesn't state. Like iron_ore_mines.json,
 * "Coordinates" packs lat/lon into one string ("lat, lon") — parsed by
 * parseGemCoordinates, duplicated locally rather than imported, matching
 * this file family's established one-module-per-file convention (see
 * gemCoalTerminals.ts / gemIronOreMines.ts, each carrying its own copy).
 *
 * GEM's date columns (Start/Announced/Construction/Idled/Retired/
 * Pre-retirement-announcement date) mix real ISO date strings with
 * year-only numbers (e.g. 1983) AND the literal sentinel string
 * "unknown" for unreported values — live-verified via a full-column type
 * scan this session (928 float years + 365 date strings for "Start
 * date" alone, "unknown" the single dominant string value in every date
 * column). toDateOrNull handles both real shapes; a bare year formats
 * to its integer string, an ISO string passes through unchanged, and any
 * sentinel/unrecognized value degrades to null.
 */
import fs from "fs";
import { repoDataPath } from "./repoFiles";

export type SteelProductionTechnology = "bf_bof" | "dri" | "eaf" | "if" | "other";

export interface IronSteelPlant {
  id: string; // "GEM plant ID"
  name: string;
  technology: SteelProductionTechnology;
  technologyRaw: string | null;
  categorySteelProduct: string | null; // crude / semi-finished / finished rolled (as catalogued)
  country: string | null;
  region: string | null;
  municipality: string | null;
  subnationalUnit: string | null;
  coordinateAccuracy: string | null;
  workforceSize: number | null;
  startDate: string | null;
  retiredDate: string | null;
  idledDate: string | null;
  owner: string | null;
  parent: string | null;
  soeStatus: string | null;
  wiki: string | null;
  lat: number;
  lon: number;
}

// Sentinel strings the release uses for "no value reported" (case-
// insensitive) — matches gemIronOreMines.ts's NULL_SENTINELS exactly
// (same GEM release family, same convention, live-verified separately
// against this file rather than assumed).
const NULL_SENTINELS = new Set(["", "-", "--", "n/a", "unknown", "unkonwn"]);

function toStrOrNull(v: unknown): string | null {
  if (v == null) return null;
  const s = String(v).trim();
  return s.length > 0 && !NULL_SENTINELS.has(s.toLowerCase()) ? s : null;
}
function toNumOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
/** Handles GEM's mixed year-number / ISO-date-string / "unknown" date
 *  columns. A bare year (e.g. 1983) formats to "1983"; an ISO string
 *  passes through; a sentinel or anything else degrades to null. */
export function toDateOrNull(v: unknown): string | null {
  if (typeof v === "number" && Number.isFinite(v)) return String(Math.trunc(v));
  return toStrOrNull(v);
}

/** Parses GEM's packed "lat, lon" Coordinates string. Returns null on
 *  anything that doesn't cleanly split into two finite numbers — nothing
 *  to place on a map, dropped by the caller rather than guessed. Same
 *  logic as gemIronOreMines.ts's parseGemCoordinates. */
export function parseGemCoordinates(raw: unknown): { lat: number; lon: number } | null {
  if (typeof raw !== "string") return null;
  const parts = raw.split(",").map((p) => p.trim());
  if (parts.length !== 2) return null;
  const lat = Number(parts[0]);
  const lon = Number(parts[1]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  return { lat, lon };
}

/** Buckets GEM's free-text "Main production equipment" column (a
 *  semicolon-separated list of real steelmaking technology codes) into
 *  the plant's PRIMARY reduction route. Priority order — BF (blast
 *  furnace, the integrated route) beats DRI (direct-reduced-iron route)
 *  beats EAF (electric arc, scrap-based) beats IF (induction furnace)
 *  beats a lone BOF (basic oxygen furnace with no stated BF, kept in the
 *  same integrated-route bucket since BOF steelmaking always consumes
 *  molten iron from somewhere) — chosen because BF/DRI are the
 *  technologies that actually distinguish a plant's route (integrated
 *  vs. green-steel-adjacent), never inferred beyond the codes the row
 *  itself lists. Anything else (blank, "unknown", "Iron other/
 *  unspecified", "Steel other/unspecified") falls to "other" honestly. */
export function classifyProductionTechnology(raw: string | null | undefined): SteelProductionTechnology {
  const tokens = new Set(
    String(raw || "")
      .split(";")
      .map((t) => t.trim().toLowerCase())
      .filter(Boolean),
  );
  if (tokens.has("bf")) return "bf_bof";
  if (tokens.has("dri")) return "dri";
  if (tokens.has("eaf")) return "eaf";
  if (tokens.has("if")) return "if";
  if (tokens.has("bof")) return "bf_bof";
  return "other";
}

/** Normalizes the release's raw per-row columns into the clean API
 *  schema. Drops rows with no usable coordinates, no name, or no id —
 *  nothing to key or place on a map (same drop-not-infer rule as
 *  normalizeIronOreMines). */
export function normalizeIronSteelPlants(rows: Record<string, unknown>[]): IronSteelPlant[] {
  const out: IronSteelPlant[] = [];
  for (const r of rows || []) {
    const id = toStrOrNull(r["GEM plant ID"]);
    const name = toStrOrNull(r["Plant name (English)"]);
    const coords = parseGemCoordinates(r["Coordinates"]);
    if (!id || !name || !coords) continue;
    const technologyRaw = toStrOrNull(r["Main production equipment"]);
    out.push({
      id,
      name,
      technology: classifyProductionTechnology(technologyRaw),
      technologyRaw,
      categorySteelProduct: toStrOrNull(r["Category steel product"]),
      country: toStrOrNull(r["Country/area"]),
      region: toStrOrNull(r["Region"]),
      municipality: toStrOrNull(r["Municipality"]),
      subnationalUnit: toStrOrNull(r["Subnational unit"]),
      coordinateAccuracy: toStrOrNull(r["Coordinate accuracy"]),
      workforceSize: toNumOrNull(r["Workforce size"]),
      startDate: toDateOrNull(r["Start date"]),
      retiredDate: toDateOrNull(r["Retired date"]),
      idledDate: toDateOrNull(r["Idled date"]),
      owner: toStrOrNull(r["Owner"]),
      parent: toStrOrNull(r["Parent (English)"]),
      soeStatus: toStrOrNull(r["SOE status"]),
      wiki: toStrOrNull(r["GEM wiki page"]),
      lat: coords.lat,
      lon: coords.lon,
    });
  }
  return out;
}

export interface IronSteelPlantsResult {
  release: string | null;
  attribution: string;
  license: string;
  plants: IronSteelPlant[];
}

/** Reads datacore/gem/iron_steel_plants.json. Never throws: a missing/
 *  corrupt file degrades to null, matching loadGemIronOreMines's
 *  fetch-failure-degrades precedent, so the route can serve an honest
 *  "unavailable" instead of crashing the process. */
export function loadGemIronSteelPlants(
  fp: string = repoDataPath("datacore/gem/iron_steel_plants.json"),
): IronSteelPlantsResult | null {
  try {
    const raw = JSON.parse(fs.readFileSync(fp, "utf8"));
    const prov = raw.provenance || {};
    return {
      release: toStrOrNull(prov.release),
      attribution: typeof prov.attribution === "string" ? prov.attribution : "Global Energy Monitor",
      license: typeof prov.license === "string" ? prov.license : "CC BY 4.0",
      plants: normalizeIronSteelPlants(raw.plants || []),
    };
  } catch {
    return null;
  }
}

// In-memory cache — same rationale as gemIronOreMines.ts's `cached`:
// static reference data, parse once per process lifetime.
let cached: IronSteelPlantsResult | null | undefined;

export function cachedGemIronSteelPlants(): IronSteelPlantsResult | null {
  if (cached === undefined) cached = loadGemIronSteelPlants();
  return cached;
}

/** Test-only: clears the module cache so a test can inject a different fp. */
export function _resetGemIronSteelPlantsCacheForTests(): void {
  cached = undefined;
}
