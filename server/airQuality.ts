/**
 * airQuality.ts — Google Air Quality API, current conditions archived at our
 * strategic sites (EDGE-DOCTRINE data root; T-DATACORE).
 *
 * Source: Google Maps Platform Air Quality API
 * (airquality.googleapis.com/v1). Current conditions + 30-day hourly
 * history + heatmap tiles; 70+ pollutants incl. PM2.5/NO2 at 500m,
 * 100+ countries. Contract mirrored from the DOCUMENTED reference
 * (developers.google.com/maps/documentation/air-quality) — NO live call
 * is made in this build or its tests. Request body:
 * {universalAqi, location:{latitude,longitude}, extraComputations[],
 * languageCode}; response: dateTime (RFC3339 UTC, rounded to the hour),
 * indexes[]{code,aqi,aqiDisplay,category,dominantPollutant,color},
 * pollutants[]{code,displayName,fullName,concentration{value,units}}.
 * Universal AQI index code "uaqi"; US EPA index (via LOCAL_AQI) code
 * "usa_epa"; pollutant codes "pm25"/"no2".
 *
 * TWO honest gate states (never a crash, never fabricated data):
 *  - awaiting_key  : GOOGLE_MAPS_API_KEY absent (poll loop never starts).
 *  - awaiting_enable: key present but the Air Quality API is NOT enabled on
 *    the GCP project — a live call returns HTTP 403 with
 *    status PERMISSION_DENIED / reason SERVICE_DISABLED. Surfaced as a
 *    first-class state on the route; activates automatically the moment the
 *    human flips the console toggle and the next poll succeeds.
 *
 * FREE-TIER BUDGET GUARD (5,000 calls/month free ≈ 166/day):
 *   16 strategic sites × current-conditions poll every 3h
 *     = 8 calls/day/site = 128 calls/day ≈ 3,968/month  → fits with headroom.
 *   DAILY_CALL_BUDGET = 150 (cap below the 166/day free-tier line).
 *   cyclesPerDay = 24h / 3h = 8;  maxSitesPerCycle = floor(150/8) = 18.
 *   With ≤18 sites every site is polled every cycle (limited=false). If the
 *   site list GROWS past 18, the poller self-limits to a ROTATING subset of
 *   18 sites/cycle (projected 18×8 = 144 calls/day < 150) and SAYS SO in the
 *   route envelope (budget.rotating_subset=true) — never silently exceeds,
 *   never fabricates a reading for an unpolled site (its last reading stands).
 *
 * HYPOTHESIS (gate-locked; drafted in open_questions.md): NO2/PM2.5 over an
 * industrial site is a combustion/activity proxy that fuses with our
 * power-plant, strategic-site and grid layers. We ARCHIVE readings over time
 * — accumulation is the moat (Google exposes only 30d of history; every day
 * we don't record is permanently lost). RAW archive only until ladder gates.
 *
 * Boundary: pure fetch/parse/archive — no imports from trading logic
 * (datacore spinout rule). Key never logged, never written into any archived
 * record or issue string (it travels in the URL query per Google's contract,
 * so every recorded URL/error is scrubbed).
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";
import strategicSites from "../datacore/sites/strategic_sites.json";

// ── key gate ─────────────────────────────────────────────────────────────────
export function airQualityKey(env: NodeJS.ProcessEnv = process.env): string {
  return env.GOOGLE_MAPS_API_KEY || "";
}
export function airQualityEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(airQualityKey(env));
}

// ── strategic sites (the same list siteTimeline/analyst import) ──────────────
export interface AqSite { id: string; lat: number; lon: number; name?: string }
export const SITES: AqSite[] = ((strategicSites as any).sites || [])
  .filter((s: any) => typeof s?.lat === "number" && typeof s?.lon === "number" && s?.id)
  .map((s: any) => ({ id: String(s.id), lat: s.lat, lon: s.lon, name: s.name }));

// ── budget guard (see header math) ───────────────────────────────────────────
export const POLL_INTERVAL_MS = 3 * 60 * 60_000; // 3h
export const DAILY_CALL_BUDGET = 150;            // < 166/day free-tier line
export const MONTHLY_FREE_TIER = 5000;
const CALL_SPACING_MS = 250;                     // politeness spacing between calls

export function cyclesPerDay(intervalMs = POLL_INTERVAL_MS): number {
  return Math.max(1, Math.round(86400_000 / intervalMs));
}
export function maxSitesPerCycle(budget = DAILY_CALL_BUDGET, intervalMs = POLL_INTERVAL_MS): number {
  return Math.max(1, Math.floor(budget / cyclesPerDay(intervalMs)));
}

/** Pick this cycle's sites. ≤maxPerCycle → poll all (limited=false); more →
 *  a rotating window so every site is still visited within a few cycles and
 *  the daily-call ceiling is never breached. Never fabricates. */
export function selectCycleSites<T>(sites: T[], cycleIndex: number, maxPerCycle: number): { batch: T[]; limited: boolean } {
  if (sites.length === 0) return { batch: [], limited: false };
  if (sites.length <= maxPerCycle) return { batch: sites.slice(), limited: false };
  const start = ((cycleIndex % Math.ceil(sites.length / maxPerCycle)) * maxPerCycle) % sites.length;
  const batch: T[] = [];
  for (let i = 0; i < maxPerCycle; i++) batch.push(sites[(start + i) % sites.length]);
  return { batch, limited: true };
}

export function callBudgetState(numSites: number, sitesThisCycle: number, limited: boolean): {
  monthly_free_tier: number; daily_call_budget: number; poll_interval_hours: number;
  cycles_per_day: number; site_count: number; max_sites_per_cycle: number;
  sites_this_cycle: number; rotating_subset: boolean; projected_calls_per_day: number;
} {
  const cpd = cyclesPerDay();
  const perCycle = limited ? maxSitesPerCycle() : numSites;
  return {
    monthly_free_tier: MONTHLY_FREE_TIER,
    daily_call_budget: DAILY_CALL_BUDGET,
    poll_interval_hours: POLL_INTERVAL_MS / 3600_000,
    cycles_per_day: cpd,
    site_count: numSites,
    max_sites_per_cycle: maxSitesPerCycle(),
    sites_this_cycle: sitesThisCycle,
    rotating_subset: limited,
    projected_calls_per_day: perCycle * cpd,
  };
}

// ── request contract (documented; no live call in this build) ────────────────
export function currentConditionsUrl(key: string): string {
  return `https://airquality.googleapis.com/v1/currentConditions:lookup?key=${encodeURIComponent(key)}`;
}

/** LOCAL_AQI adds the country-native index (US → code "usa_epa");
 *  POLLUTANT_CONCENTRATION adds pm25/no2 concentration values. */
export function currentConditionsBody(lat: number, lon: number): any {
  return {
    universalAqi: true,
    location: { latitude: lat, longitude: lon },
    extraComputations: ["LOCAL_AQI", "POLLUTANT_CONCENTRATION", "DOMINANT_POLLUTANT_CONCENTRATION"],
    languageCode: "en",
  };
}

/** Heatmap tile URL template (map overlay — a client layer is a LATER PR):
 *  GET https://airquality.googleapis.com/v1/mapTypes/{mapType}/heatmapTiles/{z}/{x}/{y}?key=KEY
 *  mapType ∈ {UAQI_INDIGO_PERSIAN, UAQI_RED_GREEN, PM25_INDIGO_PERSIAN,
 *  GBR_DEFRA, US_AQI, NO2_INDIGO_PERSIAN, ...}. */
export function heatmapTileUrl(mapType: string, z: number, x: number, y: number, key: string): string {
  return `https://airquality.googleapis.com/v1/mapTypes/${encodeURIComponent(mapType)}` +
    `/heatmapTiles/${z}/${x}/${y}?key=${encodeURIComponent(key)}`;
}

/** 403 PERMISSION_DENIED / SERVICE_DISABLED = the API is not enabled on the
 *  project (distinct from a bad key or a transient error). */
export function isServiceDisabled(status: number, body: string): boolean {
  if (status !== 403) return false;
  return /SERVICE_DISABLED/.test(body) || /PERMISSION_DENIED/.test(body) ||
         /has not been used in project/i.test(body);
}

// ── parse ────────────────────────────────────────────────────────────────────
export interface AqReading {
  siteId: string;
  dateTime: string;                 // RFC3339 UTC, hour-rounded (from API)
  uaqi: number | null;              // Universal AQI
  uaqi_category: string | null;
  uaqi_dominant: string | null;
  epa_aqi: number | null;           // US EPA AQI (LOCAL_AQI "usa_epa")
  epa_category: string | null;
  epa_dominant: string | null;
  pm25: number | null;              // concentration value
  pm25_units: string | null;
  no2: number | null;
  no2_units: string | null;
  indexes: any[];                   // full index list, kept verbatim
  pollutants: any[];                // full pollutant list, kept verbatim
  rt: string;                       // as-seen UTC date
}

const numOrNull = (v: any): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** currentConditions response → AqReading, or null if the doc has no
 *  dateTime (never guessed/interpolated). Extracts uaqi + usa_epa indexes and
 *  pm25 + no2 concentrations; keeps the full index/pollutant lists too. */
export function parseCurrentConditions(json: any, siteId: string, rt: string): AqReading | null {
  if (!json || typeof json.dateTime !== "string" || !json.dateTime) return null;
  const indexes: any[] = Array.isArray(json.indexes) ? json.indexes : [];
  const pollutants: any[] = Array.isArray(json.pollutants) ? json.pollutants : [];
  const idx = (code: string) => indexes.find((x: any) => x?.code === code) || null;
  const poll = (code: string) => pollutants.find((x: any) => x?.code === code) || null;
  const uaqi = idx("uaqi");
  const epa = idx("usa_epa");
  const pm25 = poll("pm25");
  const no2 = poll("no2");
  return {
    siteId,
    dateTime: json.dateTime,
    uaqi: uaqi ? numOrNull(uaqi.aqi) : null,
    uaqi_category: uaqi?.category ?? null,
    uaqi_dominant: uaqi?.dominantPollutant ?? null,
    epa_aqi: epa ? numOrNull(epa.aqi) : null,
    epa_category: epa?.category ?? null,
    epa_dominant: epa?.dominantPollutant ?? null,
    pm25: pm25?.concentration ? numOrNull(pm25.concentration.value) : null,
    pm25_units: pm25?.concentration?.units ?? null,
    no2: no2?.concentration ? numOrNull(no2.concentration.value) : null,
    no2_units: no2?.concentration?.units ?? null,
    indexes,
    pollutants,
    rt,
  };
}

// ── key scrub (the key rides the URL query — never let it land in a record,
//    issue string, or log line) ────────────────────────────────────────────────
export function scrubKey(s: string, key: string): string {
  let out = s;
  if (key) out = out.split(key).join("REDACTED");
  return out.replace(/([?&]key=)[^&\s"']+/gi, "$1REDACTED");
}

// ── archive (append-only JSONL day-file per UTC day; dedup siteId|dateTime) ───
const seenAq = new Set<string>();
let seeded = false;
const aqKey = (r: AqReading) => `${r.siteId}|${r.dateTime}`;

function aqDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "airquality");
}

export const SEED_WINDOW_DAYS = 45; // > the API's 30-day history; heap bound

export function seedFileInWindow(fileName: string, nowMs: number): boolean {
  const day = Date.parse(fileName.slice(0, 10));
  return Number.isFinite(day) && nowMs - day <= SEED_WINDOW_DAYS * 86400_000;
}

function seedSeen(dir: string): void {
  try {
    const nowMs = Date.now();
    for (const f of fs.readdirSync(dir)) {
      if (!/^\d{4}-\d{2}-\d{2}\.jsonl(\.gz)?$/.test(f)) continue;
      if (!seedFileInWindow(f, nowMs)) continue;
      const fp = path.join(dir, f);
      let text: string;
      try {
        text = f.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try {
          const r = JSON.parse(line);
          if (r?.siteId && r?.dateTime) seenAq.add(`${r.siteId}|${r.dateTime}`);
        } catch {}
      }
    }
  } catch {}
}

/** Appends UNSEEN readings to day-files keyed by the reading's UTC day. */
export function archiveAirQuality(readings: AqReading[], baseDir?: string): number {
  if (!readings.length) return 0;
  const dir = aqDir(baseDir);
  if (!seeded) { seedSeen(dir); seeded = true; }
  const fresh = readings.filter((r) => r.dateTime && !seenAq.has(aqKey(r)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const byDay = new Map<string, AqReading[]>();
    for (const r of fresh) {
      const day = r.dateTime.slice(0, 10);
      if (!byDay.has(day)) byDay.set(day, []);
      byDay.get(day)!.push(r);
    }
    byDay.forEach((rows, day) => {
      fs.appendFileSync(path.join(dir, `${day}.jsonl`),
                        rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
    });
    fresh.forEach((r) => seenAq.add(aqKey(r)));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] airquality archive:", e?.message || e);
    return 0;
  }
}

/** gz day-files older than 3 days; a late reading can append a plain file
 *  AFTER the day was gz'd — merge, never overwrite. */
export function gzipOldAqDays(baseDir?: string, nowMs?: number): number {
  const dir = aqDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      if (now - Date.parse(f.slice(0, 10)) < 3 * 86400_000) continue;
      const fp = path.join(dir, f);
      const gzPath = `${fp}.gz`;
      const prior = fs.existsSync(gzPath) ? zlib.gunzipSync(fs.readFileSync(gzPath)) : Buffer.alloc(0);
      fs.writeFileSync(gzPath, zlib.gzipSync(Buffer.concat([prior, fs.readFileSync(fp)])));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── fetch + poll ─────────────────────────────────────────────────────────────
type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export interface AqCache {
  at: number;
  readings: AqReading[];                  // latest reading per site (sorted by siteId)
  issues: Record<string, string>;
  awaiting_enable: boolean;
  budget: ReturnType<typeof callBudgetState>;
}

const latestBySite = new Map<string, AqReading>();
const lastIssues: Record<string, string> = {};
let cycleIndex = 0;
let cache: AqCache | null = null;
let polling = false;

export function latestAirQuality(): AqCache | null { return cache; }

export async function refreshAirQuality(
  fetchImpl: FetchFn = fetch as any,
  env: NodeJS.ProcessEnv = process.env,
  nowMs?: number, baseDir?: string,
  spacingMs = CALL_SPACING_MS,
  sites: AqSite[] = SITES,
): Promise<void> {
  try {
    if (!airQualityEnabled(env)) return;
    const key = airQualityKey(env);
    const now = nowMs ?? Date.now();
    const rt = new Date(now).toISOString().slice(0, 10);
    const { batch, limited } = selectCycleSites(sites, cycleIndex, maxSitesPerCycle());
    cycleIndex++;
    for (const k of Object.keys(lastIssues)) delete lastIssues[k];
    const fresh: AqReading[] = [];
    let disabled = false;

    for (const site of batch) {
      try {
        const r = await fetchImpl(currentConditionsUrl(key), {
          method: "POST",
          headers: { "Content-Type": "application/json", ...UA },
          body: JSON.stringify(currentConditionsBody(site.lat, site.lon)),
          signal: AbortSignal.timeout(30000) as any,
        });
        const text = await r.text();
        if (isServiceDisabled(r.status, text)) {
          // API not enabled on the project — every site would return the same;
          // stop the cycle (don't burn calls), record an honest issue, archive
          // nothing. Activates automatically once the human enables the API.
          disabled = true;
          lastIssues[site.id] = "api_not_enabled: Air Quality API disabled on the GCP project (enable it in the Google Cloud console; PERMISSION_DENIED/SERVICE_DISABLED)";
          break;
        }
        if (!r.ok) {
          lastIssues[site.id] = scrubKey(`http ${r.status}`, key).slice(0, 200);
          continue;
        }
        let json: any;
        try { json = JSON.parse(text); } catch { lastIssues[site.id] = "unparseable response"; continue; }
        const reading = parseCurrentConditions(json, site.id, rt);
        if (!reading) { lastIssues[site.id] = "response carried no dateTime"; continue; }
        fresh.push(reading);
        latestBySite.set(site.id, reading);
      } catch (e: any) {
        lastIssues[site.id] = scrubKey(String(e?.message || e), key).slice(0, 200);
      }
      if (spacingMs > 0) await sleep(spacingMs);
    }

    if (fresh.length) archiveAirQuality(fresh, baseDir);
    const readings = Array.from(latestBySite.values()).sort((a, b) => a.siteId.localeCompare(b.siteId));
    cache = {
      at: now,
      readings,
      issues: { ...lastIssues },
      awaiting_enable: disabled,
      budget: callBudgetState(sites.length, batch.length, limited),
    };
    gzipOldAqDays(baseDir, now);
  } catch (e: any) {
    console.error("[datacore] airquality refresh:", scrubKey(String(e?.message || e), airQualityKey(env)));
  }
}

/** Eager boot, key-gated (no point polling with no key). 3h cadence keeps us
 *  well under the free tier — see header math. */
export function bootAirQualityPoll(intervalMs = POLL_INTERVAL_MS, env: NodeJS.ProcessEnv = process.env): void {
  if (polling || !airQualityEnabled(env)) return;
  polling = true;
  refreshAirQuality();
  setInterval(() => { refreshAirQuality(); }, intervalMs).unref?.();
}
