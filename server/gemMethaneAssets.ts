/**
 * gemMethaneAssets.ts — GEM oil/gas-extraction and coal-mine registries,
 * normalized to a common point-asset shape for the methane-plume proximity
 * join (server/gemMethaneProximity.ts). Filed against
 * research/open_questions.md's "GEM METHANE-PLUME × EXTRACTION-REGISTRY
 * PROXIMITY" hypothesis (gate 0/2 at filing) — this module supplies the
 * asset side of that join's gate-2(a) step.
 *
 * Sources (both CC BY 4.0, same GEM release family as methane_emitters.
 * json.gz — see datacore/manifests/gem.json):
 *   - datacore/gem/oil_gas_extraction.json.gz, the `fields` sheet (7,673
 *     rows; wellpad/offshore-platform/plant-class infrastructure).
 *   - datacore/gem/coal_mine_tracker.json.gz, the `non_closed` sheet
 *     (5,382 active mines — matches the plume dataset's single largest
 *     infrastructureType bucket, "coal mine").
 * Pipeline (projects/gas_pipelines) geometry is WKT/route data, not a
 * single lat/lon per asset — proximity-joining a plume to a pipeline
 * ROUTE is a different (harder, unscoped) problem and is deliberately NOT
 * attempted here; pipeline-tagged plumes (~48 of 3,474) simply find no
 * asset match, honestly, rather than a fabricated nearest-point guess.
 *
 * Same "seeded pattern" as gemMethane.ts: static reference data, no live
 * poll, re-ingested only when a human re-runs scripts/gem_ingest.py on a
 * new GEM release.
 */
import fs from "fs";
import zlib from "zlib";
import { repoDataPath } from "./repoFiles";
import type { GemProvenance } from "./gemMethane";

export type GemAssetKind = "oil_gas_extraction" | "coal_mine";

export interface GemAsset {
  id: string;
  kind: GemAssetKind;
  name: string | null;
  operator: string | null;
  owner: string | null;
  parent: string | null;
  lat: number;
  lon: number;
}

function toNumOrNull(v: any): number | null {
  const n = typeof v === "number" ? v : parseFloat(v);
  return Number.isFinite(n) ? n : null;
}

function toStrOrNull(v: any): string | null {
  return typeof v === "string" && v.length > 0 ? v : null;
}

/** Normalizes GEM oil/gas-extraction `fields` rows. Drops rows with no
 *  Unit ID or no coordinates — same drop-not-infer rule as normalizePlumes. */
export function normalizeOilGasFields(rows: Array<Record<string, any>>): GemAsset[] {
  const out: GemAsset[] = [];
  for (const r of rows || []) {
    const id = toStrOrNull(r["Unit ID"]);
    const lat = toNumOrNull(r["Latitude"]);
    const lon = toNumOrNull(r["Longitude"]);
    if (!id || lat === null || lon === null) continue;
    out.push({
      id, kind: "oil_gas_extraction",
      name: toStrOrNull(r["Unit Name"]),
      operator: toStrOrNull(r["Operator"]),
      owner: toStrOrNull(r["Owner(s)"]),
      parent: toStrOrNull(r["Parent(s)"]),
      lat, lon,
    });
  }
  return out;
}

/** Normalizes GEM coal-mine-tracker `non_closed` rows. The tracker has no
 *  separate "Operator" column (Owners doubles as the operating party) —
 *  operator is left null rather than guessed from owner. */
export function normalizeCoalMines(rows: Array<Record<string, any>>): GemAsset[] {
  const out: GemAsset[] = [];
  for (const r of rows || []) {
    const id = toStrOrNull(r["GEM Mine ID"]);
    const lat = toNumOrNull(r["Latitude"]);
    const lon = toNumOrNull(r["Longitude"]);
    if (!id || lat === null || lon === null) continue;
    out.push({
      id, kind: "coal_mine",
      name: toStrOrNull(r["Mine Name"]),
      operator: null,
      owner: toStrOrNull(r["Owners"]),
      parent: toStrOrNull(r["Parent Company"]),
      lat, lon,
    });
  }
  return out;
}

interface GemGzFile {
  provenance: GemProvenance;
  [sheet: string]: any;
}

function readGz(fp: string): GemGzFile | null {
  try {
    return JSON.parse(zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8"));
  } catch {
    return null;
  }
}

export interface GemAssetsProvenance {
  oilGas: GemProvenance | null;
  coalMine: GemProvenance | null;
}

export interface GemAssetsResult {
  assets: GemAsset[];
  provenance: GemAssetsProvenance;
}

/** Loads + normalizes both registries into one combined asset list. Never
 *  throws: a missing/corrupt file drops that registry's assets (the other
 *  still loads) rather than failing the whole join. */
export function loadGemAssets(
  oilGasFp: string = repoDataPath("datacore/gem/oil_gas_extraction.json.gz"),
  coalMineFp: string = repoDataPath("datacore/gem/coal_mine_tracker.json.gz"),
): GemAssetsResult {
  const oilGas = readGz(oilGasFp);
  const coalMine = readGz(coalMineFp);
  const assets: GemAsset[] = [
    ...normalizeOilGasFields(oilGas?.fields ?? []),
    ...normalizeCoalMines(coalMine?.non_closed ?? []),
  ];
  return {
    assets,
    provenance: { oilGas: oilGas?.provenance ?? null, coalMine: coalMine?.provenance ?? null },
  };
}

// In-memory cache — same rationale as gemMethane.ts's cachedGemMethane:
// static reference data, parse once per process lifetime.
let cached: GemAssetsResult | undefined;

export function cachedGemAssets(): GemAssetsResult {
  if (cached === undefined) cached = loadGemAssets();
  return cached;
}

/** Test-only: clears the module cache so a test can inject different fps. */
export function _resetGemAssetsCacheForTests(): void {
  cached = undefined;
}
