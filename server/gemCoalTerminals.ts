/**
 * gemCoalTerminals.ts — GEM "Global Coal Terminals Tracker", RAW-DATA
 * overlay (CLAUDE.md RAW-vs-SIGNAL surface rule): a catalogued
 * observation, no predictive claim.
 *
 * Source: datacore/gem/coal_terminals.json, already ingested by
 * scripts/gem_ingest.py from GEM's release (CC BY 4.0, per the release's
 * own Copyright sheet — see datacore/manifests/gem.json for the full
 * artifact provenance). STATIC reference dataset, same seeded pattern as
 * gemCoalMineFeatures.ts: GEM ships a new release ~2x/year and a human
 * re-runs the ingest script on delivery — no boot-poll loop, no
 * archive-append machinery needed here.
 *
 * 521 port coal-handling terminals worldwide, each with a lifecycle
 * Status (Operating/Construction/Proposed/Shelved/Mothballed/Retired/
 * Cancelled) and a free-text Terminal Type (Imports/Exports/Domestic, or
 * a comma-joined combination for multi-role terminals — classified
 * honestly into 5 buckets by classifyTerminalType() below, never
 * inferring a role the source string doesn't state). This ties directly
 * into the existing AIS-derived port_dwell_maritime_transit /
 * shadow_fleet_maritime coal-corridor vessel data (CROSS-SYSTEM
 * INTEGRATION PRINCIPLE) — a real geographic join future gate-2 work can
 * use, not attempted here.
 *
 * "GEM Terminal ID" is NOT unique in the release (53 collisions,
 * live-verified — multiple berths/phases at one physical terminal share
 * it); "GEM Unit/Phase ID" IS unique across all 521 rows and is this
 * module's id field. "GEM Terminal ID" is kept separately as
 * `terminalId` (the parent-terminal grouping key).
 */
import fs from "fs";
import { repoDataPath } from "./repoFiles";

export type CoalTerminalTypeClass = "exports" | "imports" | "domestic" | "mixed" | "unstated";

export interface CoalTerminal {
  id: string; // "GEM Unit/Phase ID" — the only column verified unique
  terminalId: string | null; // "GEM Terminal ID" — parent-terminal grouping, NOT unique alone
  name: string;
  parentPort: string | null;
  status: string | null;
  typeClass: CoalTerminalTypeClass;
  typeRaw: string | null;
  productType: string | null;
  capacityMt: number | null;
  owner: string | null;
  country: string | null;
  region: string | null;
  startYear: string | null;
  retiredYear: string | null;
  locationAccuracy: string | null;
  wiki: string | null;
  lat: number;
  lon: number;
}

function toStrOrNull(v: unknown): string | null {
  if (v == null) return null;
  const s = String(v).trim();
  return s.length > 0 && s !== "-" ? s : null;
}
function toNumOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Classifies GEM's free-text "Terminal Type" column (a single value or a
 *  comma-joined combination, e.g. "Exports, Imports") into one of five
 *  honest buckets. Never infers a role the string doesn't state — an
 *  unrecognized/blank value is "unstated", not guessed as "domestic". */
export function classifyTerminalType(raw: string | null | undefined): CoalTerminalTypeClass {
  const s = String(raw || "").toLowerCase();
  const hasExports = s.includes("export");
  const hasImports = s.includes("import");
  const hasDomestic = s.includes("domestic");
  const roleCount = [hasExports, hasImports, hasDomestic].filter(Boolean).length;
  if (roleCount >= 2) return "mixed";
  if (hasExports) return "exports";
  if (hasImports) return "imports";
  if (hasDomestic) return "domestic";
  return "unstated";
}

/** Normalizes GEM's raw per-row columns into the clean API schema. Drops
 *  rows with no usable lat/lon, no name, or no unique id — nothing to
 *  key or place on a map (same drop-not-infer rule as
 *  normalizeCoalMineFeatures). */
export function normalizeCoalTerminals(rows: Record<string, unknown>[]): CoalTerminal[] {
  const out: CoalTerminal[] = [];
  for (const r of rows || []) {
    const lat = toNumOrNull(r["Latitude"]);
    const lon = toNumOrNull(r["Longitude"]);
    const id = toStrOrNull(r["GEM Unit/Phase ID"]);
    const name = toStrOrNull(r["Coal Terminal Name"]);
    if (lat == null || lon == null || !id || !name) continue;
    const typeRaw = toStrOrNull(r["Terminal Type"]);
    out.push({
      id,
      terminalId: toStrOrNull(r["GEM Terminal ID"]),
      name,
      parentPort: toStrOrNull(r["Parent Port Name"]),
      status: toStrOrNull(r["Status"]),
      typeClass: classifyTerminalType(typeRaw),
      typeRaw,
      productType: toStrOrNull(r["Product Type"]),
      capacityMt: toNumOrNull(r["Capacity (Mt)"]),
      owner: toStrOrNull(r["Owner"]),
      country: toStrOrNull(r["Country/Area"]),
      region: toStrOrNull(r["Region"]),
      startYear: toStrOrNull(r["Start Year"]),
      retiredYear: toStrOrNull(r["Retired Year"]),
      locationAccuracy: toStrOrNull(r["Location Accuracy"]),
      wiki: toStrOrNull(r["Wiki URL"]),
      lat,
      lon,
    });
  }
  return out;
}

export interface CoalTerminalsResult {
  release: string | null;
  attribution: string;
  license: string;
  terminals: CoalTerminal[];
}

/** Reads datacore/gem/coal_terminals.json. Never throws: a missing/corrupt
 *  file degrades to null, matching loadGemCoalMineFeatures's
 *  fetch-failure-degrades precedent, so the route can serve an honest
 *  "unavailable" instead of crashing the process. */
export function loadGemCoalTerminals(
  fp: string = repoDataPath("datacore/gem/coal_terminals.json"),
): CoalTerminalsResult | null {
  try {
    const raw = JSON.parse(fs.readFileSync(fp, "utf8"));
    const prov = raw.provenance || {};
    return {
      release: toStrOrNull(prov.release),
      attribution: typeof prov.attribution === "string" ? prov.attribution : "Global Energy Monitor",
      license: typeof prov.license === "string" ? prov.license : "CC BY 4.0",
      terminals: normalizeCoalTerminals(raw.terminals || []),
    };
  } catch {
    return null;
  }
}

// In-memory cache — same rationale as gemCoalMineFeatures.ts's `cached`:
// static reference data, parse once per process lifetime.
let cached: CoalTerminalsResult | null | undefined;

export function cachedGemCoalTerminals(): CoalTerminalsResult | null {
  if (cached === undefined) cached = loadGemCoalTerminals();
  return cached;
}

/** Test-only: clears the module cache so a test can inject a different fp. */
export function _resetGemCoalTerminalsCacheForTests(): void {
  cached = undefined;
}
