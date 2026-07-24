/**
 * gemCoalMineFeatures.ts — GEM "Coal Mine Boundaries and Methane Sources"
 * catalogued geometry (mine boundary polygons, ventilation/degasification
 * points, other mapped infrastructure), RAW-DATA overlay (CLAUDE.md
 * RAW-vs-SIGNAL surface rule): a catalogued observation, no predictive
 * claim.
 *
 * Source: datacore/gem/coal_mine_features.geojson.gz, already ingested by
 * scripts/gem_ingest.py from GEM's release (CC BY 4.0, per the release's
 * own Copyright sheet — see datacore/manifests/gem.json for the full
 * artifact provenance). STATIC reference dataset, same seeded pattern as
 * gemMethane.ts / gemMethaneAssets.ts: GEM ships a new release ~2x/year
 * and a human re-runs the ingest script on delivery — no boot-poll loop,
 * no archive-append machinery needed here.
 *
 * This file has sat in datacore/gem/ unrouted since the 2026-07-07 GEM
 * ingest (wishlist.md's own follow-up note: "coal-mine map layer from the
 * geojson.gz"); this module + its route close the wishlist-named "last
 * unclaimed shipped-data-no-map-layer gap" (2026-07-21 DATACORE MAXIMUS
 * resume block) at the API-boundary layer. The client map layer itself is
 * a separate follow-up (same earthquakes/buoys/GMET-plumes precedent —
 * pipeline+API ships first, the UI layer is its own PR).
 *
 * Unlike methane_emitters.json.gz, this FeatureCollection carries no
 * top-level provenance object — release metadata (GEM's own
 * "build_version" string) is embedded per-feature instead, so it is read
 * off the first feature that has it.
 *
 * Every feature carries GEM's own per-feature "id" (e.g. "M0086.B1",
 * distinct from the parent "GEM Mine ID" it belongs to). The release's
 * own free-text "notes" column (citation/provenance prose, frequently
 * 1-2KB per row) is deliberately dropped from the API payload to keep the
 * response lean — "description" (GEM's own short human-readable summary)
 * is kept. Features with no id or no geometry (nothing to key or place)
 * are dropped, never guessed — same drop-not-infer rule as
 * normalizePlumes/normalizeCoalMines.
 */
import fs from "fs";
import zlib from "zlib";
import { repoDataPath } from "./repoFiles";

export type CoalMineFeatureGeometry =
  | { type: "Point"; coordinates: [number, number] }
  | { type: "Polygon"; coordinates: number[][][] }
  | { type: "MultiLineString"; coordinates: number[][][] }
  | { type: string; coordinates: any };

export interface CoalMineFeature {
  id: string;
  mineId: string | null;
  mineName: string | null;
  category: string | null;
  subcategory: string | null;
  description: string | null;
  owners: string | null;
  parent: string | null;
  country: string | null;
  coalGrade: string | null;
  wiki: string | null;
  dataSourceDate: string | null;
  geometry: CoalMineFeatureGeometry;
}

function toStrOrNull(v: any): string | null {
  return typeof v === "string" && v.length > 0 ? v : null;
}

interface RawGeoJsonFeature {
  properties?: Record<string, any>;
  geometry?: any;
}

/** Normalizes the release's raw GeoJSON features into the clean API
 *  schema. Drops features with no "id" or no geometry — nothing to key
 *  or place on a map. */
export function normalizeCoalMineFeatures(features: RawGeoJsonFeature[]): CoalMineFeature[] {
  const out: CoalMineFeature[] = [];
  for (const f of features || []) {
    const p = f.properties || {};
    const id = toStrOrNull(p["id"]);
    if (!id || !f.geometry) continue;
    out.push({
      id,
      mineId: toStrOrNull(p["GEM Mine ID"]),
      mineName: toStrOrNull(p["Mine Name"]),
      category: toStrOrNull(p["mine feature category"]),
      subcategory: toStrOrNull(p["mine feature subcategory"]),
      description: toStrOrNull(p["description"]),
      owners: toStrOrNull(p["Owners"]),
      parent: toStrOrNull(p["Parent Company"]),
      country: toStrOrNull(p["Country / Area"]),
      coalGrade: toStrOrNull(p["Coal Grade"]),
      wiki: toStrOrNull(p["GEM Wiki Page (ENG)"]),
      dataSourceDate: toStrOrNull(p["data source date"]),
      geometry: f.geometry,
    });
  }
  return out;
}

/** Finds the release's "build_version" string off the first feature that
 *  carries one — the file has no top-level provenance object. */
export function findBuildVersion(features: RawGeoJsonFeature[]): string | null {
  for (const f of features || []) {
    const v = f.properties && f.properties["build_version"];
    if (typeof v === "string" && v.length > 0) return v;
  }
  return null;
}

export interface CoalMineFeaturesResult {
  buildVersion: string | null;
  features: CoalMineFeature[];
}

/** Reads + gunzips datacore/gem/coal_mine_features.geojson.gz. Never
 *  throws: a missing/corrupt file degrades to null, matching
 *  loadGemMethane's fetch-failure-degrades precedent, so the route can
 *  serve an honest "unavailable" instead of crashing the process. */
export function loadGemCoalMineFeatures(
  fp: string = repoDataPath("datacore/gem/coal_mine_features.geojson.gz"),
): CoalMineFeaturesResult | null {
  try {
    const raw = JSON.parse(zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8"));
    const rawFeatures: RawGeoJsonFeature[] = raw.features || [];
    return {
      buildVersion: findBuildVersion(rawFeatures),
      features: normalizeCoalMineFeatures(rawFeatures),
    };
  } catch {
    return null;
  }
}

// In-memory cache — same rationale as gemMethane.ts's cachedGemMethane:
// static reference data, parse once per process lifetime.
let cached: CoalMineFeaturesResult | null | undefined;

export function cachedGemCoalMineFeatures(): CoalMineFeaturesResult | null {
  if (cached === undefined) cached = loadGemCoalMineFeatures();
  return cached;
}

/** Test-only: clears the module cache so a test can inject a different fp. */
export function _resetGemCoalMineFeaturesCacheForTests(): void {
  cached = undefined;
}
