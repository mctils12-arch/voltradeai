/**
 * ambientRadiation.ts — ambient gamma dose-rate monitors, four national
 * networks with clean licenses (hazards wave 3: the human's "natural
 * radiation levels" directive, 2026-07-12).
 *
 * Sources (all verified live 2026-07-12; endpoints tested from this repo):
 *  - Germany BfS ODL: ~1,681 stations, hourly, GeoJSON WFS. License
 *    DL-DE-BY-2.0 (attribution: Bundesamt für Strahlenschutz).
 *  - Health Canada FPSN: 66 stations, ~15-min lag, ESRI GeoJSON. Open
 *    Government Licence – Canada.
 *  - Finland STUK via FMI open data: ~248 stations, 10-min cadence, WFS
 *    simple-feature XML. CC BY 4.0 (attribute FMI/STUK).
 *  - US EPA RadNet: ~76 city monitors, several updates daily, per-city CSV.
 *    Public domain. RadNet publishes CITY names only — coordinates come from
 *    the Census-gazetteer join in datacore/radnet_stations.json and every
 *    US marker is flagged approx:true (label as city-approximate).
 *
 * HONESTY: observed station readings displayed as-is with per-network
 * attribution + measurement timestamp. No interpolation, no modeling, no
 * "safety" claims — a raw overlay. Dose is served in µSv/h ONLY where the
 * network itself publishes dose (nSv/h -> µSv/h is unit arithmetic); EPA
 * stations that publish only gamma count rates are served as CPM (window
 * sum, labeled) — we never convert CPM to dose.
 *
 * EXCLUDED: EURDEP/JRC European mirror (license requires provider
 * permission — see research 2026-07-12); do not add it here.
 */
import { coordValid, freshnessStatus, type LayerHealth } from "./dataQuality";
// static import -> bundled into the CJS build (import.meta paths don't
// survive the esbuild CJS bundle; the table is ~5KB so bundling is right)
import radnetStationsTable from "../datacore/radnet_stations.json";

export interface RadiationStation {
  id: string;
  name: string;
  network: "bfs-de" | "hc-ca" | "stuk-fi" | "radnet-us";
  lat: number;
  lon: number;
  unit: "uSv/h" | "cpm";
  value: number;
  time: string | null;   // ISO timestamp of the measurement (as published)
  approx?: boolean;      // true = city-centroid location (EPA RadNet)
}

const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };

export const BFS_URL =
  "https://www.imis.bfs.de/ogc/opendata/ows?service=WFS&version=2.0.0&request=GetFeature&typeName=opendata:odlinfo_odl_1h_latest&outputFormat=application/json";
export const HC_URL =
  "https://maps-cartes.services.geo.ca/server_serveur/rest/services/HC/fpsn_en/MapServer/1/query?where=1%3D1&outFields=*&f=geojson&outSR=4326";
export const FMI_URL =
  "https://opendata.fmi.fi/wfs?request=getFeature&storedquery_id=stuk::observations::external-radiation::latest::simple";
export const radnetCsvUrl = (year: number, st: string, city: string) =>
  `https://radnet.epa.gov/cdx-radnet-rest/api/rest/csv/${year}/fixed/${encodeURIComponent(st)}/${encodeURIComponent(city)}`;

// ── pure parsers (unit-tested, no network) ──────────────────────────────────

export function parseBfs(geojson: any): RadiationStation[] {
  const out: RadiationStation[] = [];
  for (const f of geojson?.features ?? []) {
    const p = f?.properties ?? {};
    const c = f?.geometry?.coordinates;
    const v = Number(p.value);
    if (!Array.isArray(c) || !coordValid(c[1], c[0]) || !Number.isFinite(v) || v < 0) continue;
    out.push({
      id: `de-${p.kenn || p.id || out.length}`,
      name: String(p.name || "ODL station"),
      network: "bfs-de", lat: Number(c[1]), lon: Number(c[0]),
      unit: "uSv/h", value: v,
      time: p.end_measure ? String(p.end_measure) : null,
    });
  }
  return out;
}

export function parseHealthCanada(geojson: any): RadiationStation[] {
  const out: RadiationStation[] = [];
  for (const f of geojson?.features ?? []) {
    const p = f?.properties ?? {};
    const c = f?.geometry?.coordinates;
    const nsv = Number(p.latest_value_nSvHr);
    if (!Array.isArray(c) || !coordValid(c[1], c[0]) || !Number.isFinite(nsv) || nsv < 0) continue;
    out.push({
      id: `ca-${p.OBJECTID ?? out.length}`,
      name: String(p.location_name || "FPSN station"),
      network: "hc-ca", lat: Number(c[1]), lon: Number(c[0]),
      unit: "uSv/h", value: nsv / 1000,
      time: Number.isFinite(Number(p.time)) ? new Date(Number(p.time)).toISOString() : null,
    });
  }
  return out;
}

/** FMI WFS "simple" XML: BsWfsElement blocks with Location gml:pos,
 *  ParameterName, ParameterValue, Time. Keep DR_PT10M_avg (dose rate µSv/h). */
export function parseFmi(xml: string): RadiationStation[] {
  const out: RadiationStation[] = [];
  const blocks = xml.split("<BsWfs:BsWfsElement").slice(1);
  for (const b of blocks) {
    const name = /<BsWfs:ParameterName>([^<]+)</.exec(b)?.[1];
    if (name !== "DR_PT10M_avg") continue;
    const pos = /<gml:pos>([-\d.]+)\s+([-\d.]+)/.exec(b);
    const val = /<BsWfs:ParameterValue>([-\d.]+)</.exec(b)?.[1];
    const time = /<BsWfs:Time>([^<]+)</.exec(b)?.[1];
    const v = Number(val);
    if (!pos || !Number.isFinite(v) || v < 0 || !coordValid(pos[1], pos[2])) continue;
    out.push({
      id: `fi-${pos[1]}-${pos[2]}`,
      name: "STUK monitor",
      network: "stuk-fi", lat: Number(pos[1]), lon: Number(pos[2]),
      unit: "uSv/h", value: v, time: time || null,
    });
  }
  return out;
}

/** One RadNet YTD CSV -> latest APPROVED reading. Dose column may be blank
 *  (instrumentation varies by city) — fall back to the SUM of the gamma
 *  energy-window count rates, served as CPM and labeled a window sum. */
export function parseRadnetCsv(csv: string, id: string, name: string, lat: number, lon: number): RadiationStation | null {
  const lines = csv.split("\n").filter((l) => l.trim().length);
  if (lines.length < 2) return null;
  const header = lines[0].split(",").map((h) => h.trim());
  const doseIdx = header.findIndex((h) => h.startsWith("DOSE EQUIVALENT RATE"));
  const timeIdx = header.findIndex((h) => h.startsWith("SAMPLE COLLECTION TIME"));
  const statusIdx = header.length - 1;
  const winIdx = header.map((h, i) => (/^GAMMA COUNT RATE R0\d/.test(h) ? i : -1)).filter((i) => i >= 0);
  for (let i = lines.length - 1; i >= 1; i--) {
    const cells = lines[i].split(",");
    if ((cells[statusIdx] || "").trim() !== "APPROVED") continue;
    const timeRaw = (cells[timeIdx] || "").trim();  // "MM/DD/YYYY HH:MM:SS" UTC
    const m = /^(\d{2})\/(\d{2})\/(\d{4}) (\d{2}:\d{2}:\d{2})$/.exec(timeRaw);
    const iso = m ? `${m[3]}-${m[1]}-${m[2]}T${m[4]}Z` : null;
    const dose = Number(cells[doseIdx]);
    if (Number.isFinite(dose) && cells[doseIdx]?.trim() !== "" && dose >= 0) {
      return { id, name, network: "radnet-us", lat, lon, unit: "uSv/h", value: dose / 1000, time: iso, approx: true };
    }
    const winVals = winIdx.map((j) => Number(cells[j])).filter((v) => Number.isFinite(v) && v >= 0);
    if (winVals.length) {
      const sum = winVals.reduce((a, b) => a + b, 0);
      return { id, name, network: "radnet-us", lat, lon, unit: "cpm", value: Math.round(sum), time: iso, approx: true };
    }
  }
  return null;
}

// ── fetch + cache ────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

let radnetCities: Array<{ key: string; st: string; city: string; lat: number; lon: number }> | null = null;
export function loadRadnetCities(): Array<{ key: string; st: string; city: string; lat: number; lon: number }> {
  if (radnetCities) return radnetCities;
  try {
    const raw = radnetStationsTable as any;
    radnetCities = Object.entries(raw.stations as Record<string, { lat: number; lon: number }>).map(([key, c]) => {
      const [st, city] = key.split("/");
      return { key, st, city, lat: c.lat, lon: c.lon };
    });
  } catch (e: any) {
    console.error("[datacore] radnet stations table:", e?.message || e);
    radnetCities = [];
  }
  return radnetCities;
}

async function fetchJson(fetchImpl: FetchFn, url: string, timeoutMs = 60000): Promise<any | null> {
  try {
    const r = await fetchImpl(url, { headers: UA, signal: AbortSignal.timeout(timeoutMs) as any });
    if (!r.ok) { console.error(`[datacore] radiation ${url.slice(0, 60)} -> ${r.status}`); return null; }
    return JSON.parse(await r.text());
  } catch (e: any) { console.error("[datacore] radiation fetch:", e?.message || e); return null; }
}

export async function fetchAmbientRadiation(fetchImpl: FetchFn = fetch as any, nowMs?: number):
    Promise<{ stations: RadiationStation[]; networks: Record<string, number> }> {
  const stations: RadiationStation[] = [];
  const networks: Record<string, number> = {};

  const bfs = await fetchJson(fetchImpl, BFS_URL);
  if (bfs) { const s = parseBfs(bfs); stations.push(...s); networks["bfs-de"] = s.length; }

  const hc = await fetchJson(fetchImpl, HC_URL);
  if (hc) { const s = parseHealthCanada(hc); stations.push(...s); networks["hc-ca"] = s.length; }

  try {
    const r = await fetchImpl(FMI_URL, { headers: UA, signal: AbortSignal.timeout(60000) as any });
    if (r.ok) { const s = parseFmi(await r.text()); stations.push(...s); networks["stuk-fi"] = s.length; }
  } catch (e: any) { console.error("[datacore] radiation fmi:", e?.message || e); }

  // EPA RadNet: one small CSV per city, sequential with a soft time budget so
  // a slow EPA day degrades to partial US coverage instead of hanging boot.
  const year = new Date(nowMs ?? Date.now()).getUTCFullYear();
  const budgetEnd = Date.now() + 150_000;
  let us = 0;
  for (const c of loadRadnetCities()) {
    if (Date.now() > budgetEnd) { console.error(`[datacore] radnet time budget hit after ${us} cities`); break; }
    try {
      const r = await fetchImpl(radnetCsvUrl(year, c.st, c.city), { headers: UA, signal: AbortSignal.timeout(20000) as any });
      if (!r.ok) continue;
      const s = parseRadnetCsv(await r.text(), `us-${c.key}`, `${c.city}, ${c.st} (city-approximate)`, c.lat, c.lon);
      if (s) { stations.push(s); us++; }
    } catch { /* one bad city never kills the sweep */ }
  }
  networks["radnet-us"] = us;
  return { stations, networks };
}

interface RadCache { at: number; stations: RadiationStation[]; networks: Record<string, number>; health: LayerHealth }
let cache: RadCache | null = null;
let polling = false;
const FRESH_BUDGET_MS = 12 * 3600_000;

export function latestAmbientRadiation(): RadCache | null { return cache; }

export async function refreshAmbientRadiation(fetchImpl?: FetchFn, nowMs?: number): Promise<void> {
  try {
    const { stations, networks } = await fetchAmbientRadiation(fetchImpl, nowMs);
    if (!stations.length) {
      if (cache) cache.health = { ...cache.health, freshness: "stale" };
      return;
    }
    const now = nowMs ?? Date.now();
    cache = {
      at: now, stations, networks,
      health: {
        source: "BfS ODL (DL-DE-BY-2.0) · Health Canada FPSN (OGL-Canada) · STUK/FMI (CC BY 4.0) · EPA RadNet (public domain)",
        clean: stations.length, suspect: 0,
        freshness: freshnessStatus(now, FRESH_BUDGET_MS, now),
        source_date: new Date(now).toISOString().slice(0, 10),
      },
    };
  } catch (e: any) { console.error("[datacore] radiation refresh:", e?.message || e); }
}

export function bootAmbientRadiationPoll(intervalMs = 3 * 3600_000): void {
  if (polling) return;
  polling = true;
  refreshAmbientRadiation().catch((e) => console.error("[datacore] radiation boot:", e?.message || e));
  setInterval(() => { refreshAmbientRadiation(); }, intervalMs).unref?.();
}
