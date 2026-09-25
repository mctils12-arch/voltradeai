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
 *
 * FURNACE-UNIT ENRICHMENT (added later — detail-panel data, not a new
 * top-level layer): datacore/gem/steel_units.json is the same release
 * family's unit-level file, keyed to this one by "GEM plant ID"
 * (live-verified: its first EAF row IS plant P100000120882, the same
 * plant this file's own test fixture uses). It carries 3,554 individual
 * furnaces across 4 GEM-native sheets (eaf/bof/induction/open_hearth —
 * counts.eaf=1408, counts.bof=1531, counts.induction=605,
 * counts.open_hearth=10), each with its own lifecycle status, current
 * capacity (ttpa), and dates — real per-furnace detail a plant-level row
 * alone cannot carry (a plant can mix e.g. an idle BF with a newly
 * commissioned EAF). loadGemSteelUnits/normalizeSteelUnits/
 * groupSteelUnitsByPlant load and key this file independently of the
 * plant loader above (same one-file-does-one-thing shape as
 * gemSteelRawMaterials.ts's separate joinCountryChoropleth step);
 * cachedGemIronSteelPlants performs the actual join so
 * loadGemIronSteelPlants itself stays a pure, single-file function
 * (unchanged default single-arg behavior — its own tests still call it
 * with only a plants fixture and get units:[] on every row, exactly as
 * before this enrichment existed). "Unit status" is GEM's own 8-value
 * catalogued set (operating / operating pre-retirement / announced /
 * construction / mothballed / mothballed pre-retirement / retired /
 * cancelled — live Counter-verified across all 4 sheets); classifyUnit
 * Status buckets to one of those or "unknown", never invents a status
 * the row doesn't state (same discipline as classifyMineStatus in
 * gemIronOreMines.ts).
 */
import fs from "fs";
import { repoDataPath } from "./repoFiles";

export type SteelProductionTechnology = "bf_bof" | "dri" | "eaf" | "if" | "other";

// GEM's own 4 furnace-unit sheet names in steel_units.json.
export type SteelFurnaceType = "eaf" | "bof" | "induction" | "open_hearth";

// GEM's own 8 catalogued "Unit status" values (live Counter-verified
// across all 4 sheets), plus "unknown" for anything else — never a
// status the row doesn't state.
export type SteelUnitStatus =
  | "operating"
  | "operating pre-retirement"
  | "announced"
  | "construction"
  | "mothballed"
  | "mothballed pre-retirement"
  | "retired"
  | "cancelled"
  | "unknown";

const KNOWN_UNIT_STATUSES = new Set<string>([
  "operating",
  "operating pre-retirement",
  "announced",
  "construction",
  "mothballed",
  "mothballed pre-retirement",
  "retired",
  "cancelled",
]);

export interface SteelFurnaceUnit {
  plantId: string; // "GEM plant ID" — the join key back to IronSteelPlant.id
  unitId: string; // "GEM unit ID"
  name: string | null;
  furnaceType: SteelFurnaceType;
  status: SteelUnitStatus;
  capacityTtpa: number | null; // "Current capacity (ttpa)"
  startDate: string | null;
  retiredDate: string | null;
  manufacturer: string | null;
}

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
  units: SteelFurnaceUnit[]; // furnace-level detail, keyed from steel_units.json; [] when none joined
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
 *  normalizeIronOreMines). `unitsByPlantId`, when supplied, attaches
 *  each plant's furnace-level detail from steel_units.json; omitted
 *  (the default) every plant gets `units: []` — this is what keeps the
 *  function's existing single-arg callers (including its own tests)
 *  unchanged. */
export function normalizeIronSteelPlants(
  rows: Record<string, unknown>[],
  unitsByPlantId?: Map<string, SteelFurnaceUnit[]>,
): IronSteelPlant[] {
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
      units: unitsByPlantId?.get(id) || [],
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
 *  "unavailable" instead of crashing the process. `unitsByPlantId` is
 *  optional (see normalizeIronSteelPlants) so this stays a pure,
 *  single-file loader that a test can call with just a plants fixture. */
export function loadGemIronSteelPlants(
  fp: string = repoDataPath("datacore/gem/iron_steel_plants.json"),
  unitsByPlantId?: Map<string, SteelFurnaceUnit[]>,
): IronSteelPlantsResult | null {
  try {
    const raw = JSON.parse(fs.readFileSync(fp, "utf8"));
    const prov = raw.provenance || {};
    return {
      release: toStrOrNull(prov.release),
      attribution: typeof prov.attribution === "string" ? prov.attribution : "Global Energy Monitor",
      license: typeof prov.license === "string" ? prov.license : "CC BY 4.0",
      plants: normalizeIronSteelPlants(raw.plants || [], unitsByPlantId),
    };
  } catch {
    return null;
  }
}

/** Normalizes GEM's free-text "Unit status" column into one of the 8
 *  buckets the release itself uses (live-verified via a Counter scan
 *  across all 4 sheets). Never infers a bucket the string doesn't
 *  state — an unrecognized/blank value falls back to "unknown", not
 *  guessed as "operating" (same discipline as classifyMineStatus in
 *  gemIronOreMines.ts). */
export function classifyUnitStatus(raw: string | null | undefined): SteelUnitStatus {
  const s = String(raw || "").trim().toLowerCase();
  return (KNOWN_UNIT_STATUSES.has(s) ? s : "unknown") as SteelUnitStatus;
}

/** Normalizes one furnace-type sheet's raw rows (steel_units.json's
 *  eaf/bof/induction/open_hearth arrays) into the clean per-unit schema.
 *  Drops rows with no plant id or no unit id — nothing to key a plant's
 *  furnace list by (same drop-not-infer rule as normalizeIronSteelPlants). */
export function normalizeSteelUnits(
  furnaceType: SteelFurnaceType,
  rows: Record<string, unknown>[],
): SteelFurnaceUnit[] {
  const out: SteelFurnaceUnit[] = [];
  for (const r of rows || []) {
    const plantId = toStrOrNull(r["GEM plant ID"]);
    const unitId = toStrOrNull(r["GEM unit ID"]);
    if (!plantId || !unitId) continue;
    out.push({
      plantId,
      unitId,
      name: toStrOrNull(r["Unit name"]),
      furnaceType,
      status: classifyUnitStatus(toStrOrNull(r["Unit status"])),
      capacityTtpa: toNumOrNull(r["Current capacity (ttpa)"]),
      startDate: toDateOrNull(r["Start date"]),
      retiredDate: toDateOrNull(r["Retired date"]),
      manufacturer: toStrOrNull(r["Furnace manufacturer"]),
    });
  }
  return out;
}

const STEEL_UNIT_SHEETS: SteelFurnaceType[] = ["eaf", "bof", "induction", "open_hearth"];

export interface SteelUnitsResult {
  release: string | null;
  attribution: string;
  license: string;
  units: SteelFurnaceUnit[];
}

/** Reads datacore/gem/steel_units.json (all 4 furnace-type sheets).
 *  Never throws: a missing/corrupt file degrades to null, same
 *  fetch-failure-degrades precedent as loadGemIronSteelPlants — the
 *  plant loader join simply omits units rather than erroring. */
export function loadGemSteelUnits(
  fp: string = repoDataPath("datacore/gem/steel_units.json"),
): SteelUnitsResult | null {
  try {
    const raw = JSON.parse(fs.readFileSync(fp, "utf8"));
    const prov = raw.provenance || {};
    const units: SteelFurnaceUnit[] = [];
    for (const sheet of STEEL_UNIT_SHEETS) {
      units.push(...normalizeSteelUnits(sheet, raw[sheet] || []));
    }
    return {
      release: toStrOrNull(prov.release),
      attribution: typeof prov.attribution === "string" ? prov.attribution : "Global Energy Monitor",
      license: typeof prov.license === "string" ? prov.license : "CC BY 4.0",
      units,
    };
  } catch {
    return null;
  }
}

/** Groups a flat furnace-unit list by "GEM plant ID" — the shape
 *  normalizeIronSteelPlants's unitsByPlantId param expects. */
export function groupSteelUnitsByPlant(units: SteelFurnaceUnit[]): Map<string, SteelFurnaceUnit[]> {
  const m = new Map<string, SteelFurnaceUnit[]>();
  for (const u of units) {
    const arr = m.get(u.plantId);
    if (arr) arr.push(u);
    else m.set(u.plantId, [u]);
  }
  return m;
}

// In-memory cache — same rationale as gemIronOreMines.ts's `cached`:
// static reference data, parse once per process lifetime.
let cached: IronSteelPlantsResult | null | undefined;

export function cachedGemIronSteelPlants(): IronSteelPlantsResult | null {
  if (cached === undefined) {
    const unitsHit = loadGemSteelUnits();
    const unitsByPlantId = unitsHit ? groupSteelUnitsByPlant(unitsHit.units) : undefined;
    cached = loadGemIronSteelPlants(undefined, unitsByPlantId);
  }
  return cached;
}

/** Test-only: clears the module cache so a test can inject a different fp. */
export function _resetGemIronSteelPlantsCacheForTests(): void {
  cached = undefined;
}
