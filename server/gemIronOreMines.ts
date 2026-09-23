/**
 * gemIronOreMines.ts — GEM "Global Iron Ore Mines Tracker", RAW-DATA
 * overlay (CLAUDE.md RAW-vs-SIGNAL surface rule): a catalogued
 * observation, no predictive claim.
 *
 * Source: datacore/gem/iron_ore_mines.json, already ingested by
 * scripts/gem_ingest.py from GEM's release (CC BY 4.0, per the release's
 * own Copyright sheet — see datacore/manifests/gem.json for the full
 * artifact provenance). STATIC reference dataset, same seeded pattern as
 * gemCoalTerminals.ts / gemCoalMineFeatures.ts: GEM ships a new release
 * ~2x/year and a human re-runs the ingest script on delivery — no
 * boot-poll loop, no archive-append machinery needed here.
 *
 * 949 iron ore mines worldwide, each with a lowercase "Operating status"
 * lifecycle bucket (operating/proposed/mothballed/retired/unknown/
 * shelved/cancelled, live-verified exhaustive across the release — a
 * value outside this set degrades to "unknown" honestly rather than
 * guessed) plus catalogued production tonnage (2022-2024, thousand
 * tonnes/year) and reserve/resource tonnage (thousand metric tonnes) —
 * FACTS as reported by GEM, never a forecast or an implied trading
 * signal. Unlike coal_terminals.json's separate Latitude/Longitude
 * columns, this release packs both into one "Coordinates" string
 * ("lat, lon") that must be parsed.
 *
 * GEM's own numeric columns mix real numbers with the literal sentinel
 * strings "N/A"/"unknown"/"unkonwn" (a typo present in the source release
 * itself, live-verified) for unreported values — toNumOrNull's typeof
 * check treats any of these as null without needing to match the exact
 * sentinel spelling. "GEM Asset ID" is unique across all 949 rows
 * (live-verified, no collision class like coal_terminals' "GEM Terminal
 * ID"), so it is used directly as this module's id.
 */
import fs from "fs";
import { repoDataPath } from "./repoFiles";

export type IronOreMineStatus =
  | "operating" | "proposed" | "mothballed" | "retired" | "shelved" | "cancelled" | "unknown";

const KNOWN_STATUSES = new Set<string>([
  "operating", "proposed", "mothballed", "retired", "shelved", "cancelled", "unknown",
]);

export interface IronOreMine {
  id: string; // "GEM Asset ID" — verified unique across the release
  name: string;
  status: IronOreMineStatus;
  statusRaw: string | null;
  country: string | null;
  region: string | null;
  municipality: string | null;
  subnationalUnit: string | null;
  coordinateAccuracy: string | null;
  production2024Kt: number | null;
  production2023Kt: number | null;
  production2022Kt: number | null;
  designCapacityKt: number | null;
  totalReservesKt: number | null;
  totalResourceKt: number | null;
  startDate: string | null;
  stopDate: string | null;
  owner: string | null;
  parent: string | null;
  wiki: string | null;
  lat: number;
  lon: number;
}

// Sentinel strings the release uses for "no value reported" (case-
// insensitive), including "unkonwn" — GEM's own typo, present verbatim
// in the checked-in release, not a mistake introduced here.
const NULL_SENTINELS = new Set(["", "-", "--", "n/a", "unknown", "unkonwn"]);

function toStrOrNull(v: unknown): string | null {
  if (v == null) return null;
  const s = String(v).trim();
  return s.length > 0 && !NULL_SENTINELS.has(s.toLowerCase()) ? s : null;
}
function toNumOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Normalizes GEM's free-text "Operating status" column into one of the
 *  7 buckets the release itself uses. Never infers a bucket the string
 *  doesn't state — an unrecognized/blank value falls back to "unknown"
 *  (itself one of GEM's own catalogued values), not guessed as
 *  "operating". */
export function classifyMineStatus(raw: string | null | undefined): IronOreMineStatus {
  const s = String(raw || "").trim().toLowerCase();
  return (KNOWN_STATUSES.has(s) ? s : "unknown") as IronOreMineStatus;
}

/** Parses GEM's packed "lat, lon" Coordinates string. Returns null on
 *  anything that doesn't cleanly split into two finite numbers — nothing
 *  to place on a map, dropped by the caller rather than guessed. */
export function parseGemCoordinates(raw: unknown): { lat: number; lon: number } | null {
  if (typeof raw !== "string") return null;
  const parts = raw.split(",").map((p) => p.trim());
  if (parts.length !== 2) return null;
  const lat = Number(parts[0]);
  const lon = Number(parts[1]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  return { lat, lon };
}

/** Normalizes the release's raw per-row columns into the clean API
 *  schema. Drops rows with no usable coordinates, no name, or no id —
 *  nothing to key or place on a map (same drop-not-infer rule as
 *  normalizeCoalTerminals). */
export function normalizeIronOreMines(rows: Record<string, unknown>[]): IronOreMine[] {
  const out: IronOreMine[] = [];
  for (const r of rows || []) {
    const id = toStrOrNull(r["GEM Asset ID"]);
    const name = toStrOrNull(r["Asset name (English)"]);
    const coords = parseGemCoordinates(r["Coordinates"]);
    if (!id || !name || !coords) continue;
    const statusRaw = toStrOrNull(r["Operating status"]);
    out.push({
      id,
      name,
      status: classifyMineStatus(statusRaw),
      statusRaw,
      country: toStrOrNull(r["Country/Area"]),
      region: toStrOrNull(r["Region"]),
      municipality: toStrOrNull(r["Municipality"]),
      subnationalUnit: toStrOrNull(r["Subnational unit"]),
      coordinateAccuracy: toStrOrNull(r["Coordinate accuracy"]),
      production2024Kt: toNumOrNull(r["Production 2024 (ttpa)"]),
      production2023Kt: toNumOrNull(r["Production 2023 (ttpa)"]),
      production2022Kt: toNumOrNull(r["Production 2022 (ttpa)"]),
      designCapacityKt: toNumOrNull(r["Design capacity (ttpa)"]),
      totalReservesKt: toNumOrNull(r["Total reserves (proven and probable, thousand metric tonnes)"]),
      totalResourceKt: toNumOrNull(r["Total resource (inferred, indicated and measured, thousand metric tonnes)"]),
      startDate: toStrOrNull(r["Start date"]),
      stopDate: toStrOrNull(r["Stop date"]),
      owner: toStrOrNull(r["Owner"]),
      parent: toStrOrNull(r["Parent"]),
      wiki: toStrOrNull(r["GEM wiki page URL"]),
      lat: coords.lat,
      lon: coords.lon,
    });
  }
  return out;
}

export interface IronOreMinesResult {
  release: string | null;
  attribution: string;
  license: string;
  mines: IronOreMine[];
}

/** Reads datacore/gem/iron_ore_mines.json. Never throws: a missing/
 *  corrupt file degrades to null, matching loadGemCoalTerminals's
 *  fetch-failure-degrades precedent, so the route can serve an honest
 *  "unavailable" instead of crashing the process. */
export function loadGemIronOreMines(
  fp: string = repoDataPath("datacore/gem/iron_ore_mines.json"),
): IronOreMinesResult | null {
  try {
    const raw = JSON.parse(fs.readFileSync(fp, "utf8"));
    const prov = raw.provenance || {};
    return {
      release: toStrOrNull(prov.release),
      attribution: typeof prov.attribution === "string" ? prov.attribution : "Global Energy Monitor",
      license: typeof prov.license === "string" ? prov.license : "CC BY 4.0",
      mines: normalizeIronOreMines(raw.mines || []),
    };
  } catch {
    return null;
  }
}

// In-memory cache — same rationale as gemCoalTerminals.ts's `cached`:
// static reference data, parse once per process lifetime.
let cached: IronOreMinesResult | null | undefined;

export function cachedGemIronOreMines(): IronOreMinesResult | null {
  if (cached === undefined) cached = loadGemIronOreMines();
  return cached;
}

/** Test-only: clears the module cache so a test can inject a different fp. */
export function _resetGemIronOreMinesCacheForTests(): void {
  cached = undefined;
}
