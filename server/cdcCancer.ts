/**
 * cdcCancer.ts — NCI State Cancer Profiles county-level cancer incidence
 * + mortality rates (Location Context Engine hazard layer #5,
 * research/location_context_engine.md — the last hazard layer on that
 * file's list, queued across roughly six prior session entries before
 * this build).
 *
 * Source: datacore/cdc_cancer/county_rates.json, built by
 * scripts/cdc_cancer_rates.py from NCI's State Cancer Profiles CSV export
 * (public domain SEER+NPCR data), GATE-1 cross-checked against CDC's own
 * separately-published national headline figures at build time — see the
 * script's own header. Static artifact, WHOLE-FILE REBUILD per run (NCI
 * republishes on its own multi-year cadence, not a live poll target),
 * same pattern as pfas.ts.
 *
 * Geometry: datacore/cdc_cancer/counties-10m.topo.json — US Census county
 * cartographic boundaries (1:10m simplified), redistributed verbatim via
 * the `us-atlas` npm package (ISC license; underlying Census TIGER data
 * is public domain). Converted to GeoJSON with topojson-client (already a
 * project dependency — see client/src/components/DataWorldMap.tsx for the
 * client-side precedent of the same library).
 *
 * ECOLOGICAL FALLACY GUARD (non-negotiable, per location_context_engine.md):
 * every value here describes an ENTIRE COUNTY's reported population over a
 * multi-year window — never a specific address or resident. This module
 * exposes exactly two shapes, both county-scoped, never a point value:
 *   1. the full county-choropleth GeoJSON (`cancerRatesGeoJSON()`)
 *   2. "which county contains this clicked point, and its rate" via
 *      `countyFipsAt(lat, lon)` (FCC's free Census Block API) — the
 *      dossier renders this exactly like femaFlood.ts's flood_zone: a
 *      point-in-polygon LOOKUP, never a radius list, with the same
 *      ready:false/ready:true-but-null distinction.
 *
 * RAW/FACTUAL: an age-adjusted rate is NCI's own published federal
 * surveillance statistic — not a risk claim about any location; that
 * distinction lives in `meta.caveat`, echoed verbatim by every API
 * response and by the dossier section's own caveat text.
 *
 * DATA QUALITY GATE: re-validated at serve time (FIPS format, rate
 * bounds) even though the build script already gates at ingest — defense
 * in depth, matching every other hazard layer's contract.
 */
import * as topojson from "topojson-client";
import {
  coordValid, partitionValid, freshnessStatus, layerHealth,
  type LayerHealth, type QualityIssue,
} from "./dataQuality";
import countyRatesJson from "../datacore/cdc_cancer/county_rates.json";
import countiesTopoJson from "../datacore/cdc_cancer/counties-10m.topo.json";

export interface CountyRate {
  fips: string;
  county: string;
  state: string | null;
  incidence_rate: number | null;
  incidence_ci: [number, number] | null;
  incidence_avg_annual_count: number | null;
  incidence_trend: string | null;
  incidence_suppressed: boolean;
  mortality_rate: number | null;
  mortality_ci: [number, number] | null;
  mortality_avg_annual_count: number | null;
  mortality_trend: string | null;
  mortality_suppressed: boolean;
}

export interface Gate1Result {
  pulled_national_incidence_rate: number | null;
  pulled_national_mortality_rate: number | null;
  reference: { incidence_rate: number; mortality_rate: number; as_of: string; tolerance_pct: number };
  incidence_pct_diff: number | null;
  mortality_pct_diff: number | null;
  passed: boolean;
}

export interface CountyRatesArtifact {
  source: string;
  attribution: string;
  predictive: boolean;
  caveat: string;
  built_at: string;
  county_count: number;
  quarantined_count: number;
  gate1: Gate1Result;
  counties: any[];
}

const SCHEMA_FIPS = /^\d{5}$/;
const RATE_MIN = 0;
const RATE_MAX = 5000; // sanity ceiling only, see scripts/cdc_cancer_rates.py's own comment

function rateFieldOk(v: any): boolean {
  return v == null || (typeof v === "number" && Number.isFinite(v) && v >= RATE_MIN && v <= RATE_MAX);
}

/** One artifact record -> validated CountyRate, or the quality issues that
 *  quarantine it. Rate fields are legitimately nullable (NCI suppression)
 *  — only format/bounds are enforced, never "required". */
export function countyFromRecord(r: any): { county?: CountyRate; issues: QualityIssue[] } {
  const issues: QualityIssue[] = [];
  if (typeof r?.fips !== "string" || !SCHEMA_FIPS.test(r.fips)) issues.push({ field: "fips", rule: "pattern", detail: String(r?.fips) });
  if (typeof r?.county !== "string" || !r.county) issues.push({ field: "county", rule: "required" });
  if (!rateFieldOk(r?.incidence_rate)) issues.push({ field: "incidence_rate", rule: "bounds", detail: String(r?.incidence_rate) });
  if (!rateFieldOk(r?.mortality_rate)) issues.push({ field: "mortality_rate", rule: "bounds", detail: String(r?.mortality_rate) });
  if (issues.length) return { issues };
  return {
    issues: [],
    county: {
      fips: r.fips,
      county: r.county,
      state: r.state ?? null,
      incidence_rate: r.incidence_rate ?? null,
      incidence_ci: Array.isArray(r.incidence_ci) ? [Number(r.incidence_ci[0]), Number(r.incidence_ci[1])] : null,
      incidence_avg_annual_count: r.incidence_avg_annual_count ?? null,
      incidence_trend: r.incidence_trend ?? null,
      incidence_suppressed: Boolean(r.incidence_suppressed),
      mortality_rate: r.mortality_rate ?? null,
      mortality_ci: Array.isArray(r.mortality_ci) ? [Number(r.mortality_ci[0]), Number(r.mortality_ci[1])] : null,
      mortality_avg_annual_count: r.mortality_avg_annual_count ?? null,
      mortality_trend: r.mortality_trend ?? null,
      mortality_suppressed: Boolean(r.mortality_suppressed),
    },
  };
}

/** Full artifact -> {clean counties, suspect count}. Pure — testable on
 *  any artifact shape, not just the committed one. */
export function parseCountyRates(artifact: Pick<CountyRatesArtifact, "counties">): { counties: CountyRate[]; suspect: number } {
  const part = partitionValid(artifact.counties || [], (r) => countyFromRecord(r).issues);
  const counties = part.clean.map((r) => countyFromRecord(r).county!).filter(Boolean);
  return { counties, suspect: part.suspect.length };
}

/** topojson `objects.counties` -> GeoJSON FeatureCollection, one Feature
 *  per county, `id` = 5-digit FIPS (us-atlas's own key). Pure wrapper over
 *  topojson-client so it's swappable/testable without the real 800KB
 *  topology loaded. */
export function countyGeometry(topo: any): GeoJSON.FeatureCollection {
  return topojson.feature(topo, topo.objects.counties) as unknown as GeoJSON.FeatureCollection;
}

/** Join county geometry + rate records by FIPS. A geometry with no rate
 *  match (Puerto Rico, territories NCI's export excludes, and a handful of
 *  historical FIPS-boundary edge cases — 89 of 3,231 counties, live-
 *  confirmed 2026-08-11) keeps its shape with `has_data:false` and every
 *  rate field null — NEVER dropped from the map (a hole would read as
 *  "zero risk here", the exact false claim this layer must not make) and
 *  NEVER fabricated. Pure — testable on small fixtures. */
export function joinGeometryAndRates(geo: GeoJSON.FeatureCollection, counties: CountyRate[]): GeoJSON.FeatureCollection {
  const byFips = new Map(counties.map((c) => [c.fips, c]));
  return {
    type: "FeatureCollection",
    features: geo.features.map((f) => {
      const fips = String((f as any).id ?? "");
      const rate = byFips.get(fips) ?? null;
      return {
        ...f,
        properties: {
          fips,
          county: rate?.county ?? (f.properties as any)?.name ?? null,
          state: rate?.state ?? null,
          has_data: Boolean(rate),
          incidence_rate: rate?.incidence_rate ?? null,
          incidence_trend: rate?.incidence_trend ?? null,
          incidence_suppressed: rate?.incidence_suppressed ?? null,
          mortality_rate: rate?.mortality_rate ?? null,
          mortality_trend: rate?.mortality_trend ?? null,
          mortality_suppressed: rate?.mortality_suppressed ?? null,
        },
      };
    }),
  };
}

const rawArtifact = countyRatesJson as unknown as CountyRatesArtifact;
const FRESH_BUDGET_MS = 400 * 86400_000; // NCI publishes multi-year windows on an irregular, slow cadence

const cached = (() => {
  const { counties, suspect } = parseCountyRates(rawArtifact);
  const geo = joinGeometryAndRates(countyGeometry(countiesTopoJson as any), counties);
  const builtAtMs = Date.parse(rawArtifact.built_at);
  const health = layerHealth(
    rawArtifact.attribution,
    { clean: counties, suspect: Array(suspect).fill(null) as any },
    freshnessStatus(Number.isFinite(builtAtMs) ? builtAtMs : null, FRESH_BUDGET_MS, Date.now()),
    rawArtifact.built_at,
  );
  const byFips = new Map(counties.map((c) => [c.fips, c]));
  return { counties, byFips, geo, health };
})();

export function latestCancerRates(): { counties: CountyRate[]; geo: GeoJSON.FeatureCollection; health: LayerHealth; meta: CountyRatesArtifact } {
  return { counties: cached.counties, geo: cached.geo, health: cached.health, meta: rawArtifact };
}

export function cancerRateFor(fips: string | null): CountyRate | null {
  if (!fips) return null;
  return cached.byFips.get(fips) ?? null;
}

// ── point -> county FIPS lookup (FCC Census Block API, free, no key) ────
export interface CountyLookupResult {
  fips: string | null;
  county_name: string | null;
  state_code: string | null;
  rate: CountyRate | null;
  source: string;
  /** false while the point-in-county lookup hasn't resolved yet (network
   *  failure) — mirrors femaFlood.ts's FloodZoneResult.ready contract so
   *  the dossier never confuses "not looked up" with "looked up, no data". */
  ready: boolean;
}

const FCC_SOURCE = "FCC Census Block Conversions API (geo.fcc.gov), county FIPS lookup";

function unavailable(): CountyLookupResult {
  return { fips: null, county_name: null, state_code: null, rate: null, source: FCC_SOURCE, ready: false };
}

function noCoverage(): CountyLookupResult {
  return { fips: null, county_name: null, state_code: null, rate: null, source: FCC_SOURCE, ready: true };
}

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export function fccQueryUrl(lat: number, lon: number): string {
  const q = new URLSearchParams({ lat: String(lat), lon: String(lon), format: "json" });
  return `https://geo.fcc.gov/api/census/area?${q.toString()}`;
}

/** One FCC `/api/census/area` JSON response -> CountyLookupResult (before
 *  the rate join, which the cached wrapper below adds). Pure. */
export function parseFccResponse(json: any): CountyLookupResult {
  const blocks: any[] = Array.isArray(json?.results) ? json.results : [];
  if (!blocks.length) return noCoverage();
  const b = blocks[0];
  const fips = typeof b.county_fips === "string" ? b.county_fips : null;
  if (!fips || !SCHEMA_FIPS.test(fips)) return noCoverage();
  return {
    fips,
    county_name: b.county_name || null,
    state_code: b.state_code || null,
    rate: null, // joined by the cached wrapper, which has access to the rate table
    source: FCC_SOURCE,
    ready: true,
  };
}

export async function fetchCountyAt(lat: number, lon: number, fetchImpl: FetchFn = fetch as any): Promise<CountyLookupResult> {
  if (!coordValid(lat, lon)) return unavailable();
  try {
    const r = await fetchImpl(fccQueryUrl(lat, lon), {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(10000) as any,
    });
    if (!r.ok) { console.error(`[datacore] cdcCancer -> ${r.status}`); return unavailable(); }
    const parsed = parseFccResponse(JSON.parse(await r.text()));
    return parsed.ready && parsed.fips ? { ...parsed, rate: cancerRateFor(parsed.fips) } : parsed;
  } catch (e: any) {
    console.error("[datacore] cdcCancer:", e?.message || e);
    return unavailable();
  }
}

// per-point cache — counties don't move between clicks a few meters apart.
const CACHE_TTL_MS = 30 * 86400_000;
const CACHE_MAX = 2000;
const cache = new Map<string, { at: number; result: CountyLookupResult }>();
const cacheKey = (lat: number, lon: number) => `${lat.toFixed(3)},${lon.toFixed(3)}`;

export async function countyFipsAt(lat: number, lon: number, fetchImpl?: FetchFn, nowMs?: number): Promise<CountyLookupResult> {
  const now = nowMs ?? Date.now();
  const key = cacheKey(lat, lon);
  const hit = cache.get(key);
  if (hit && now - hit.at < CACHE_TTL_MS) return hit.result;
  const result = await fetchCountyAt(lat, lon, fetchImpl);
  if (result.ready) {
    if (cache.size >= CACHE_MAX) {
      const oldest = cache.keys().next().value;
      if (oldest !== undefined) cache.delete(oldest);
    }
    cache.set(key, { at: now, result });
  }
  return result;
}

export function clearCountyCache(): void { cache.clear(); }
