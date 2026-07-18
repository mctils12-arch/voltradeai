/**
 * gemMethane.ts — GEM Methane Emitters Tracker (GMET) dated satellite plume
 * detections, RAW-DATA overlay (CLAUDE.md RAW-vs-SIGNAL surface rule): a
 * catalogued observation, no predictive claim.
 *
 * Source: datacore/gem/methane_emitters.json.gz, already ingested by
 * scripts/gem_ingest.py from GEM's GMET_V3_12-12-2025.xlsx release
 * (CC BY 4.0, per the release's own Copyright sheet — see
 * datacore/manifests/gem.json for the full artifact provenance). This is a
 * STATIC reference dataset, not a live-polled feed: GEM ships new tracker
 * releases ~2x/year and a human re-runs the ingest script on delivery (same
 * "seeded pattern" precedent as military_installations.json / suite_skips)
 * — no boot-poll loop, no archive-append machinery needed here.
 *
 * SCOPE (2026-07-18 session): only the `plumes` sheet (3,474 dated
 * CarbonMapper/GHGSat-class satellite methane-plume observations) is
 * surfaced. The same file also carries pipelines/lng_terminals/coal_mines/
 * extraction_areas/reserves sheets — those are separate, unrelated
 * datasets and are NOT read here; a future session can build them as their
 * own routes when there's a concrete use (per CLAUDE.md's one-logical-
 * change rule).
 *
 * GEM's raw column names are the release's literal spreadsheet headers
 * (spaces, units-in-parens) — normalized here to a clean API schema.
 * Every plume carries GEM's own stable "GEM Methane Plume ID"; fields GEM
 * itself leaves blank for a given row (e.g. no modeled emissions rate yet,
 * no inferred infrastructure type) stay `null`, never fabricated.
 */
import fs from "fs";
import zlib from "zlib";
import { repoDataPath } from "./repoFiles";

export interface MethanePlume {
  id: string;
  name: string | null;
  wiki: string | null;
  provider: string | null;
  instrument: string | null;
  observedAt: string | null; // ISO 8601, as reported by GEM
  emissionsKgHr: number | null;
  emissionsUncertaintyKgHr: number | null;
  infrastructureType: string | null;
  infrastructureNotes: string | null;
  subnationalUnit: string | null;
  country: string | null;
  lat: number | null;
  lon: number | null;
}

interface GemMethaneFile {
  provenance: { attribution: string; license: string; release: string; built_at?: string };
  plumes: Array<Record<string, any>>;
}

function toNumOrNull(v: any): number | null {
  const n = typeof v === "number" ? v : parseFloat(v);
  return Number.isFinite(n) ? n : null;
}

function toStrOrNull(v: any): string | null {
  return typeof v === "string" && v.length > 0 ? v : null;
}

/** Normalizes one raw GMET spreadsheet row into the clean API schema. Rows
 *  with no "GEM Methane Plume ID" (nothing to key on) or no origin
 *  coordinates (nothing to place on a map) are dropped — matches the
 *  "drop, never infer" rule other static-reference layers use. */
export function normalizePlumes(rows: Array<Record<string, any>>): MethanePlume[] {
  const out: MethanePlume[] = [];
  for (const r of rows || []) {
    const id = toStrOrNull(r["GEM Methane Plume ID"]);
    const lat = toNumOrNull(r["Plume Origin Latitude"]);
    const lon = toNumOrNull(r["Plume Origin Longitude"]);
    if (!id || lat === null || lon === null) continue;
    out.push({
      id,
      name: toStrOrNull(r["Name"]),
      wiki: toStrOrNull(r["GEM Wiki"]),
      provider: toStrOrNull(r["Satellite Data Provider"]),
      instrument: toStrOrNull(r["Instrument"]),
      observedAt: toStrOrNull(r["Observation Date"]),
      emissionsKgHr: toNumOrNull(r["Emissions (kg/hr)"]),
      emissionsUncertaintyKgHr: toNumOrNull(r["Emissions Uncertainty (kg/hr)"]),
      infrastructureType: toStrOrNull(r["Type of Infrastructure"]),
      infrastructureNotes: toStrOrNull(r["Infrastructure Notes"]),
      subnationalUnit: toStrOrNull(r["Subnational Unit"]),
      country: toStrOrNull(r["Country/Area"]),
      lat,
      lon,
    });
  }
  return out;
}

export interface MethanePlumesResult {
  provenance: GemMethaneFile["provenance"];
  plumes: MethanePlume[];
}

/** Reads + gunzips datacore/gem/methane_emitters.json.gz (~1.9MB gz /
 *  ~14.6MB raw — the whole GMET file, all six sheets; only `plumes` is
 *  extracted). Never throws: a missing/corrupt file degrades to null,
 *  matching loadGemOwnership's (entityGraph.ts) fetch-failure-degrades
 *  precedent, so a route can serve an honest "unavailable" instead of
 *  crashing the process. */
export function loadGemMethane(
  fp: string = repoDataPath("datacore/gem/methane_emitters.json.gz"),
): MethanePlumesResult | null {
  try {
    const raw: GemMethaneFile = JSON.parse(zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8"));
    return { provenance: raw.provenance, plumes: normalizePlumes(raw.plumes) };
  } catch {
    return null;
  }
}

// In-memory cache — the source file only changes on a manual re-ingest
// (session-run script, not a running-process concern), so parsing the
// 14.6MB JSON once per process lifetime (not once per HTTP request) is
// the correct cost/freshness tradeoff for a static reference dataset.
let cached: MethanePlumesResult | null | undefined;

export function cachedGemMethane(): MethanePlumesResult | null {
  if (cached === undefined) cached = loadGemMethane();
  return cached;
}

/** Test-only: clears the module cache so a test can inject a different fp. */
export function _resetGemMethaneCacheForTests(): void {
  cached = undefined;
}
