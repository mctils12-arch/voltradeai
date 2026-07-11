/**
 * superfund.ts — EPA Superfund National Priorities List (NPL) sites
 * (Location Context Engine, first hazard layer — research/location_context_engine.md).
 *
 * Source: EPA's authoritative "Superfund National Priorities List (NPL) Sites
 * with Status Information" ArcGIS FeatureServer (owner whumbert_EPA, public
 * domain). ~1,840 sites with HRS score, status, listing dates, location.
 *
 * RAW/FACTUAL layer: a site's location + status + score are facts, displayed
 * as-is with source + date. No risk/impact interpretation is made here (that
 * would be a SIGNAL requiring ladder gate 2). predictive:false.
 *
 * DATA QUALITY GATE (server/dataQuality.ts): every site is validated at ingest
 * — valid coordinates (range + null-island), a name, a known status, an HRS
 * score in [0,100]. Failures are QUARANTINED (never served as clean); the
 * response carries a health summary {clean, suspect, freshness} so the UI shows
 * how trusted + fresh the layer is. Refreshes server-side (Railway), weekly —
 * the NPL changes slowly; a failed refresh keeps the last good snapshot and
 * marks it stale rather than serving nothing.
 */
import { validateRecord, coordValid, partitionValid, freshnessStatus, layerHealth, type LayerHealth, type QualityIssue } from "./dataQuality";

const SERVICE =
  "https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/" +
  "Superfund_National_Priorities_List_(NPL)_Sites_with_Status_Information/FeatureServer/0/query";

const OUT_FIELDS = "Site_Name,Site_Score,Site_EPA_ID,State,City,County,Status,Longitude,Latitude,Listing_Date,Proposed_Date,Construction_Completion_Date";

/** Known NPL status values (EPA). A value outside this set is a data error. */
export const NPL_STATUS = [
  "NPL Site", "Proposed NPL Site", "Deleted NPL Site",
  "Partial NPL Deletion", "Site in General Superfund Program",
];

export interface SuperfundSite {
  name: string;
  epa_id: string | null;
  hrs_score: number | null;   // Hazard Ranking System score 0-100
  status: string;
  state: string | null;
  city: string | null;
  county: string | null;
  lat: number;
  lon: number;
  listed: string | null;      // listing date (as published)
}

const SCHEMA = {
  Site_Name: { required: true },
  Site_Score: { numeric: true, min: 0, max: 100 },  // HRS scores are 0-100
};

/** One ArcGIS geojson feature -> validated SuperfundSite, or the quality issues
 *  that quarantine it. Coordinates checked with the null-island guard. */
export function siteFromFeature(f: any): { site?: SuperfundSite; issues: QualityIssue[] } {
  const p = (f && f.properties) || {};
  const issues = validateRecord(p, SCHEMA);
  const lon = p.Longitude, lat = p.Latitude;
  if (!coordValid(lat, lon)) issues.push({ field: "coordinates", rule: "coordValid", detail: `${lat},${lon}` });
  if (p.Status && !NPL_STATUS.includes(p.Status)) issues.push({ field: "Status", rule: "oneOf", detail: String(p.Status) });
  if (issues.length) return { issues };
  const score = p.Site_Score == null || p.Site_Score === "" ? null : Number(p.Site_Score);
  return {
    issues: [],
    site: {
      name: String(p.Site_Name),
      epa_id: p.Site_EPA_ID || null,
      hrs_score: Number.isFinite(score) ? score : null,
      status: p.Status || "NPL Site",
      state: p.State || null,
      city: p.City || null,
      county: p.County || null,
      lat: Number(lat),
      lon: Number(lon),
      listed: p.Listing_Date || p.Proposed_Date || null,
    },
  };
}

/** Parse a full FeatureCollection -> {clean sites, quarantined count}. */
export function parseSites(geojson: any): { sites: SuperfundSite[]; suspect: number } {
  const feats: any[] = (geojson && geojson.features) || [];
  const part = partitionValid(feats, (f) => siteFromFeature(f).issues);
  const sites = part.clean.map((f) => siteFromFeature(f).site!).filter(Boolean);
  return { sites, suspect: part.suspect.length };
}

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const PAGE = 1000;

export function pageUrl(offset: number): string {
  const q = new URLSearchParams({
    where: "1=1", outFields: OUT_FIELDS, outSR: "4326",
    resultOffset: String(offset), resultRecordCount: String(PAGE), f: "geojson",
  });
  return `${SERVICE}?${q.toString()}`;
}

/** Fetch all NPL sites (paginated), validate, quarantine bad rows. */
export async function fetchSuperfund(fetchImpl: FetchFn = fetch as any): Promise<{ sites: SuperfundSite[]; suspect: number }> {
  const all: SuperfundSite[] = [];
  let suspect = 0, offset = 0;
  for (let guard = 0; guard < 50; guard++) {
    let json: any;
    try {
      const r = await fetchImpl(pageUrl(offset), {
        headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
        signal: AbortSignal.timeout(30000) as any,
      });
      if (!r.ok) { console.error(`[datacore] superfund -> ${r.status}`); break; }
      json = JSON.parse(await r.text());
    } catch (e: any) { console.error("[datacore] superfund:", e?.message || e); break; }
    const feats: any[] = json?.features || [];
    if (!feats.length) break;
    const { sites, suspect: s } = parseSites(json);
    all.push(...sites); suspect += s;
    if (feats.length < PAGE) break;
    offset += PAGE;
  }
  return { sites: all, suspect };
}

// ── cache + weekly server-side refresh (NPL changes slowly) ──────────────────

interface SuperfundCache { at: number; sites: SuperfundSite[]; health: LayerHealth }
let cache: SuperfundCache | null = null;
let polling = false;
const FRESH_BUDGET_MS = 14 * 86400_000; // two weeks before a snapshot is "stale"

export function latestSuperfund(): SuperfundCache | null { return cache; }

export async function refreshSuperfund(fetchImpl?: FetchFn, nowMs?: number): Promise<void> {
  try {
    const { sites, suspect } = await fetchSuperfund(fetchImpl);
    if (!sites.length) {
      // failed/empty fetch: keep the last good snapshot, mark it stale — never
      // serve nothing or wipe good data on a transient outage.
      if (cache) cache.health = { ...cache.health, freshness: "stale" };
      return;
    }
    const now = nowMs ?? Date.now();
    cache = {
      at: now, sites,
      health: {
        source: "EPA Superfund NPL (SEMS), public domain",
        clean: sites.length, suspect,
        freshness: freshnessStatus(now, FRESH_BUDGET_MS, now),
        source_date: new Date(now).toISOString().slice(0, 10),
      },
    };
  } catch (e: any) { console.error("[datacore] superfund refresh:", e?.message || e); }
}

export function bootSuperfundPoll(intervalMs = 7 * 86400_000): void {
  if (polling) return;
  polling = true;
  refreshSuperfund().catch((e) => console.error("[datacore] superfund boot:", e?.message || e));
  setInterval(() => { refreshSuperfund(); }, intervalMs).unref?.();
}
