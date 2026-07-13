/**
 * femaFlood.ts — FEMA National Flood Hazard Layer (NFHL), Flood Hazard
 * Zones (research/location_context_engine.md — "the closest direct Zillow
 * parallel"). Third hazard layer for the Location Context Engine, after
 * superfund.ts and waterViolators.ts.
 *
 * Source: FEMA's public NFHL ArcGIS MapServer (hazards.fema.gov), layer 28
 * "Flood Hazard Zones" — public domain, no key required, CORS-open (server
 * reflects the request Origin). Two independent uses of the same service:
 *
 * 1. MAP OVERLAY (client/src/pages/datamap.tsx): a raster tile source that
 *    hits the MapServer's `export` operation directly via MapLibre's
 *    `{bbox-epsg-3857}` template token — FEMA renders the tile server-side,
 *    zero cost/code on our end (same "tiles from someone else's public
 *    service" pattern as the surfacewater/forest layers). Nothing in this
 *    file drives that.
 * 2. DOSSIER POINT LOOKUP (this file, via server/dossier.ts): "is THIS
 *    clicked point in a flood zone" cannot be answered by a tile — it needs
 *    a server-side spatial query at the exact anchor coordinate. That is
 *    what `floodZoneAt` does.
 *
 * RAW/FACTUAL layer: FLD_ZONE, ZONE_SUBTY, and SFHA_TF are FEMA's own
 * published fields — SFHA (Special Flood Hazard Area) status is read
 * directly off SFHA_TF, never inferred from the zone code ourselves. The
 * `meaning` text is FEMA's own published zone-code glossary (FIRM/NFHL data
 * dictionary), stated as a fact about the zone designation — never a
 * property-specific risk/impact claim (that would be a ladder-gated SIGNAL).
 *
 * DATA QUALITY GATE: -9999 is FEMA's own null-value sentinel for BFE/DEPTH
 * fields (confirmed live) — these are converted to `null`, never displayed
 * as real elevations. An unrecognized FLD_ZONE is quarantined (returned as
 * `unavailable`, never guessed into the nearest-looking bucket). A point
 * outside NFHL's mapped footprint (zero features — confirmed live over
 * interior Alaska) returns `zone: null` with an honest "not on the NFHL"
 * note, never fabricated as X/minimal-risk.
 */
import { coordValid } from "./dataQuality";

const SERVICE = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query";

/** FEMA's own published FIRM/NFHL zone-code glossary — facts about what the
 *  designation means, not a claim about any specific property. */
export const FLOOD_ZONE_MEANINGS: Record<string, string> = {
  A: "1% annual-chance flood zone (Special Flood Hazard Area); no base flood elevations determined.",
  AE: "1% annual-chance flood zone (Special Flood Hazard Area); base flood elevations determined.",
  AH: "1% annual-chance shallow flooding, typically ponding, 1-3 ft average depth (Special Flood Hazard Area).",
  AO: "1% annual-chance shallow flooding, typically sheet flow on sloping terrain, 1-3 ft average depth (Special Flood Hazard Area).",
  AR: "Area formerly protected from the 1% annual-chance flood by a flood-control system now being restored (Special Flood Hazard Area).",
  A99: "Area to be protected from the 1% annual-chance flood by a federal flood-control system under construction (Special Flood Hazard Area).",
  V: "Coastal 1% annual-chance flood zone with additional hazard from storm-induced velocity wave action; no base flood elevations determined (Special Flood Hazard Area).",
  VE: "Coastal 1% annual-chance flood zone with additional hazard from storm-induced velocity wave action; base flood elevations determined (Special Flood Hazard Area).",
  X: "0.2% annual-chance flood zone (moderate hazard) or area of minimal flood hazard, per ZONE_SUBTY — outside the Special Flood Hazard Area.",
  D: "Flood hazard undetermined — area not yet studied.",
  "OPEN WATER": "Open water (lake, pond, or similar) — not flood-hazard classified in the usual sense.",
};

export const KNOWN_FLOOD_ZONES = Object.keys(FLOOD_ZONE_MEANINGS);

const NO_DATA_SENTINEL = -9999;

export interface FloodZoneResult {
  /** null when the point falls outside NFHL's mapped footprint (an
   *  honest coverage gap, never guessed as "minimal risk"). */
  zone: string | null;
  subtype: string | null;
  /** FEMA's own SFHA_TF field, read directly — never inferred from `zone`. */
  sfha: boolean | null;
  base_flood_elevation_ft: number | null;
  meaning: string | null;
  source_citation: string | null;
  source: string;
  /** false while a query is outstanding/failed and no cached result exists
   *  yet — mirrors HazardSection's ready flag so the UI shows nothing
   *  rather than a false "no flood risk". */
  ready: boolean;
}

const SOURCE = "FEMA National Flood Hazard Layer (NFHL), public domain";

function noCoverage(): FloodZoneResult {
  return { zone: null, subtype: null, sfha: null, base_flood_elevation_ft: null, meaning: null, source_citation: null, source: SOURCE, ready: true };
}

function unavailable(): FloodZoneResult {
  return { zone: null, subtype: null, sfha: null, base_flood_elevation_ft: null, meaning: null, source_citation: null, source: SOURCE, ready: false };
}

const num = (v: any): number | null => (typeof v === "number" && Number.isFinite(v) && v !== NO_DATA_SENTINEL ? v : null);

/** One ArcGIS query-result feature -> FloodZoneResult, or `unavailable()`
 *  if FLD_ZONE isn't a code we recognize (quarantine, never guess). */
export function parseFloodFeature(f: any): FloodZoneResult {
  const a = (f && f.attributes) || {};
  const zone = typeof a.FLD_ZONE === "string" ? a.FLD_ZONE.trim().toUpperCase() : null;
  if (!zone || !(zone in FLOOD_ZONE_MEANINGS)) return unavailable();
  const sfhaRaw = typeof a.SFHA_TF === "string" ? a.SFHA_TF.trim().toUpperCase() : null;
  return {
    zone,
    subtype: a.ZONE_SUBTY || null,
    sfha: sfhaRaw === "T" ? true : sfhaRaw === "F" ? false : null,
    base_flood_elevation_ft: num(a.STATIC_BFE),
    meaning: FLOOD_ZONE_MEANINGS[zone],
    source_citation: a.SOURCE_CIT || null,
    source: SOURCE,
    ready: true,
  };
}

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export function queryUrl(lat: number, lon: number): string {
  const q = new URLSearchParams({
    f: "json",
    geometry: `${lon},${lat}`,
    geometryType: "esriGeometryPoint",
    inSR: "4326",
    spatialRel: "esriSpatialRelIntersects",
    outFields: "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,SOURCE_CIT",
    returnGeometry: "false",
    resultRecordCount: "1",
  });
  return `${SERVICE}?${q.toString()}`;
}

/** Point-in-polygon flood zone lookup at (lat, lon). Never throws — a
 *  network/parse failure degrades to `unavailable()` (ready:false), and a
 *  point outside NFHL's footprint degrades to `noCoverage()` (ready:true,
 *  zone:null) — the two honest-gap cases the DATA QUALITY GATE requires
 *  never be confused with each other or with a real "minimal risk" reading. */
export async function fetchFloodZone(lat: number, lon: number, fetchImpl: FetchFn = fetch as any): Promise<FloodZoneResult> {
  if (!coordValid(lat, lon)) return unavailable();
  try {
    const r = await fetchImpl(queryUrl(lat, lon), {
      headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
      signal: AbortSignal.timeout(10000) as any,
    });
    if (!r.ok) { console.error(`[datacore] femaFlood -> ${r.status}`); return unavailable(); }
    const json = JSON.parse(await r.text());
    const feats: any[] = json?.features || [];
    if (!feats.length) return noCoverage();
    return parseFloodFeature(feats[0]);
  } catch (e: any) {
    console.error("[datacore] femaFlood:", e?.message || e);
    return unavailable();
  }
}

// ── per-point cache: flood zones don't move between clicks a few meters
// apart, and this keeps the dossier fast + FEMA's public service lightly
// loaded. Keyed to ~1km grid cells (3 decimal places), which is far finer
// than any real flood-zone boundary uncertainty matters at click scale. ──
const CACHE_TTL_MS = 30 * 86400_000; // NFHL revises slowly (per-county FIRM updates, not daily)
const CACHE_MAX = 2000;
const cache = new Map<string, { at: number; result: FloodZoneResult }>();

function cacheKey(lat: number, lon: number): string {
  return `${lat.toFixed(3)},${lon.toFixed(3)}`;
}

/** Cached wrapper around fetchFloodZone — what server/routes.ts calls before
 *  handing the result into buildDossier's (pure, sync) opts. */
export async function floodZoneAt(lat: number, lon: number, fetchImpl?: FetchFn, nowMs?: number): Promise<FloodZoneResult> {
  const now = nowMs ?? Date.now();
  const key = cacheKey(lat, lon);
  const hit = cache.get(key);
  if (hit && now - hit.at < CACHE_TTL_MS) return hit.result;
  const result = await fetchFloodZone(lat, lon, fetchImpl);
  // Only cache a resolved answer (ready:true, whether zone:null "no coverage"
  // or a real zone) — never cache a transient fetch failure, so the next
  // click retries instead of freezing an outage into a false "unavailable".
  if (result.ready) {
    if (cache.size >= CACHE_MAX) {
      const oldest = cache.keys().next().value;
      if (oldest !== undefined) cache.delete(oldest);
    }
    cache.set(key, { at: now, result });
  }
  return result;
}

export function clearFloodZoneCache(): void { cache.clear(); }
