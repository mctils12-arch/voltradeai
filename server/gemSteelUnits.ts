/**
 * gemSteelUnits.ts — GEM "Global Iron and Steel Tracker" furnace-UNIT
 * detail (datacore/gem/steel_units.json), a RAW-DATA enrichment joined
 * onto the existing iron_steel_plants layer (server/gemIronSteelPlants.ts)
 * by "GEM plant ID" — deliberately NOT a new top-level map layer: this
 * release's units carry no coordinates of their own (only their parent
 * plant's), and a plant already has one point on the map, so a second,
 * co-located symbol per unit would just be visual noise. This module
 * only adds a `furnaceUnits` summary onto each plant row's existing
 * detail popup (server/routes.ts's /api/data/iron-steel-plants route),
 * per research/open_questions.md's 2026-09-24 GEM-suite backlog note
 * ("steel_units.json — furnace-level unit attribute data keyed to
 * iron_steel_plants.json ... a plausible follow-up detail-panel
 * enrichment rather than a new top-level layer").
 *
 * Source: datacore/gem/steel_units.json (same GEM release/ingest family
 * as iron_steel_plants.json, CC BY 4.0). Four furnace-technology arrays,
 * one object shape per type — EAF (electric arc, 1,408 rows), BOF (basic
 * oxygen, 1,531), induction (605), and open-hearth (10 rows — an
 * essentially extinct technology, kept for completeness rather than
 * dropped). Live-verified this session: every one of the 1,210 distinct
 * "GEM plant ID"s referenced across all four arrays already exists in
 * iron_steel_plants.json's own 1,293-plant list (0 orphans, a clean join
 * key, no fuzzy matching needed) — 83 plants in the release have no
 * catalogued furnace unit at all, which is a real gap in GEM's own data,
 * not a bug here; those plants simply get `furnaceUnits: null`.
 *
 * "Unit status" is a real lifecycle field (operating / operating
 * pre-retirement / announced / construction / retired / mothballed /
 * mothballed pre-retirement / cancelled — live-verified via a full
 * value-count scan this session) — aggregateUnitsByPlant counts units by
 * furnace type across EVERY status (so a plant whose 3 EAFs already
 * retired still honestly shows "3 EAF" rather than silently dropping to
 * 0), but sums capacity only across operating/operating-pre-retirement
 * units — summing retired/cancelled/announced capacity into a "current
 * capacity" figure would overstate what the plant can actually produce
 * today, the same operating-vs-catalogued distinction
 * gemIronOreMines.ts's own production-tonnage fields already respect.
 */
import fs from "fs";
import { repoDataPath } from "./repoFiles";

export type SteelFurnaceType = "eaf" | "bof" | "induction" | "open_hearth";

export interface SteelUnit {
  plantId: string;
  unitId: string | null;
  name: string | null;
  furnaceType: SteelFurnaceType;
  status: string | null;
  capacityTtpa: number | null;
}

export interface PlantFurnaceSummary {
  eaf: number;
  bof: number;
  induction: number;
  openHearth: number;
  operatingCount: number;
  // sum of known "Current capacity (ttpa)" across operating/operating-
  // pre-retirement units only; null if this plant has no operating unit
  // with a reported capacity (never fabricated as 0 — see module header).
  operatingCapacityTtpa: number | null;
}

// Matches gemIronSteelPlants.ts's NULL_SENTINELS exactly (same GEM
// release family, same convention, duplicated locally per this file
// family's established one-module-per-file precedent).
const NULL_SENTINELS = new Set(["", "-", "--", "n/a", "unknown", "unkonwn"]);

function toStrOrNull(v: unknown): string | null {
  if (v == null) return null;
  const s = String(v).trim();
  return s.length > 0 && !NULL_SENTINELS.has(s.toLowerCase()) ? s : null;
}
function toNumOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

const OPERATING_STATUSES = new Set(["operating", "operating pre-retirement"]);

/** Normalizes one of the release's four raw furnace-type arrays into the
 *  clean common SteelUnit schema. Drops rows with no plant ID — nothing
 *  to join to (same drop-not-infer rule as normalizeIronSteelPlants). */
export function normalizeSteelUnits(
  rows: Record<string, unknown>[],
  furnaceType: SteelFurnaceType,
): SteelUnit[] {
  const out: SteelUnit[] = [];
  for (const r of rows || []) {
    const plantId = toStrOrNull(r["GEM plant ID"]);
    if (!plantId) continue;
    out.push({
      plantId,
      unitId: toStrOrNull(r["GEM unit ID"]),
      name: toStrOrNull(r["Unit name"]),
      furnaceType,
      status: toStrOrNull(r["Unit status"]),
      capacityTtpa: toNumOrNull(r["Current capacity (ttpa)"]),
    });
  }
  return out;
}

/** Aggregates the flat unit list into one summary per plant ID. Pure —
 *  testable independent of the real ~3,554-row release. */
export function aggregateUnitsByPlant(units: SteelUnit[]): Map<string, PlantFurnaceSummary> {
  const out = new Map<string, PlantFurnaceSummary>();
  for (const u of units) {
    let s = out.get(u.plantId);
    if (!s) {
      s = { eaf: 0, bof: 0, induction: 0, openHearth: 0, operatingCount: 0, operatingCapacityTtpa: null };
      out.set(u.plantId, s);
    }
    if (u.furnaceType === "eaf") s.eaf++;
    else if (u.furnaceType === "bof") s.bof++;
    else if (u.furnaceType === "induction") s.induction++;
    else s.openHearth++;
    if (u.status != null && OPERATING_STATUSES.has(u.status.toLowerCase())) {
      s.operatingCount++;
      if (u.capacityTtpa != null) s.operatingCapacityTtpa = (s.operatingCapacityTtpa ?? 0) + u.capacityTtpa;
    }
  }
  return out;
}

export interface SteelUnitsResult {
  release: string | null;
  attribution: string;
  license: string;
  units: SteelUnit[];
}

/** Reads datacore/gem/steel_units.json. Never throws: a missing/corrupt
 *  file degrades to null, matching loadGemIronSteelPlants's own
 *  fetch-failure-degrades precedent, so the route can serve the base
 *  plant list with `furnaceUnits: null` on every row instead of crashing
 *  the process. */
export function loadGemSteelUnits(
  fp: string = repoDataPath("datacore/gem/steel_units.json"),
): SteelUnitsResult | null {
  try {
    const raw = JSON.parse(fs.readFileSync(fp, "utf8"));
    const prov = raw.provenance || {};
    const units = [
      ...normalizeSteelUnits(raw.eaf || [], "eaf"),
      ...normalizeSteelUnits(raw.bof || [], "bof"),
      ...normalizeSteelUnits(raw.induction || [], "induction"),
      ...normalizeSteelUnits(raw.open_hearth || [], "open_hearth"),
    ];
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

// In-memory caches — same rationale as gemIronSteelPlants.ts's `cached`:
// static reference data, parse/aggregate once per process lifetime.
let cached: SteelUnitsResult | null | undefined;
let cachedSummaries: Map<string, PlantFurnaceSummary> | null | undefined;

export function cachedGemSteelUnits(): SteelUnitsResult | null {
  if (cached === undefined) cached = loadGemSteelUnits();
  return cached;
}

/** Cached plant-ID -> furnace summary map, used to join onto each plant
 *  row in the /api/data/iron-steel-plants route. Returns null only when
 *  the underlying steel_units.json artifact itself failed to load —
 *  callers should treat that as "no enrichment available", not an error
 *  (the base plant list still serves fine without it). */
export function cachedPlantFurnaceSummaries(): Map<string, PlantFurnaceSummary> | null {
  if (cachedSummaries !== undefined) return cachedSummaries;
  const hit = cachedGemSteelUnits();
  cachedSummaries = hit ? aggregateUnitsByPlant(hit.units) : null;
  return cachedSummaries;
}

/** Test-only: clears both module caches so a test can inject a different fp. */
export function _resetGemSteelUnitsCacheForTests(): void {
  cached = undefined;
  cachedSummaries = undefined;
}
