/**
 * ndbcBuoys.ts — NOAA National Data Buoy Center "latest observations" feed
 * (all reporting buoys/C-MAN stations worldwide, one row per station, most
 * recent report). RAW-DATA overlay only (CLAUDE.md RAW-vs-SIGNAL surface
 * rule): displays NDBC-reported values as-is with source attribution, no
 * predictive claim — ships ungated, no key required.
 *
 * EDGE DOCTRINE #1 (build data, don't buy it): keyless, single global file
 * (no per-station requests), US government public domain
 * (noaa.gov/information-technology/open-data-policy). Source:
 * https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt — live-verified
 * at build time: 889 stations, whitespace-delimited fixed-column text,
 * missing readings marked "MM" (kept as null, never coerced to 0 — a
 * station with no wave sensor is not a station reporting 0m seas).
 *
 * Format quirk: the file is whitespace-delimited (NOT fixed-width) despite
 * the header's column alignment — station ids are 4 or 5 characters and
 * lat/lon can be negative, but splitting on whitespace handles both without
 * special-casing, confirmed against the live sample.
 *
 * Each row is a STATION SNAPSHOT, not a discrete event like a USGS quake —
 * the same station id reappears every poll. Archive dedup is therefore keyed
 * on (station, observation timestamp): a station's row is only appended when
 * its own YYYY-MM-DD hh:mm advances past what we last recorded, so a buoy
 * that hasn't phoned in a new reading yet doesn't pad the archive with
 * duplicate rows every poll cycle.
 *
 * ANGLE-HUNTING NOTE (filed alongside in open_questions.md): sea-state
 * (WVHT/DPD) and pressure-tendency (PTDY) fields are the standing
 * "foreign-field import" candidate — marine forecasting techniques applied
 * to shipping-cost/insurance exposure hypotheses. Gate 1/2 work, unattempted
 * here.
 *
 * Boundary: no imports from trading logic — pure fetch/parse/archive, same
 * day-file-JSONL shape as usgsQuakes.ts/nasaFirms.ts.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export const BUOYS_FEED_URL = "https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt";

export interface BuoyObs {
  station: string;
  lat: number | null;
  lon: number | null;
  time: number | null;    // observation time, ms epoch UTC (from YYYY MM DD hh mm)
  windDir: number | null; // degT
  windSpeed: number | null; // m/s
  gust: number | null;      // m/s
  waveHeight: number | null; // m
  dominantPeriod: number | null; // sec (DPD)
  avgPeriod: number | null;      // sec (APD)
  waveDir: number | null;        // degT (MWD)
  pressure: number | null;       // hPa
  pressureTendency: number | null; // hPa (PTDY, 3h change)
  airTemp: number | null;   // degC
  waterTemp: number | null; // degC
  dewpoint: number | null;  // degC
  visibility: number | null; // nmi
  tide: number | null;       // ft
  rt: string; // as-fetched UTC date
}

function num(tok: string | undefined): number | null {
  if (tok === undefined || tok === "MM") return null;
  const n = parseFloat(tok);
  return Number.isFinite(n) ? n : null;
}

/** Parses the NDBC latest_obs.txt whitespace-delimited feed. Lines starting
 *  with "#" (the two header rows) are skipped; a row with a missing
 *  station id or unparseable timestamp is dropped (nothing to key an
 *  archive dedup on). "MM" tokens map to null, never 0 — a station missing
 *  a sensor must never look like it read a real zero. */
export function parseLatestObs(text: string, rt: string): BuoyObs[] {
  const out: BuoyObs[] = [];
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t || t.startsWith("#")) continue;
    const f = t.split(/\s+/);
    if (f.length < 8) continue;
    const station = f[0];
    if (!station) continue;
    const [yr, mo, day, hr, mn] = [f[3], f[4], f[5], f[6], f[7]];
    const time = Date.parse(`${yr}-${mo}-${day}T${hr}:${mn}:00Z`);
    out.push({
      station,
      lat: num(f[1]),
      lon: num(f[2]),
      time: Number.isFinite(time) ? time : null,
      windDir: num(f[8]),
      windSpeed: num(f[9]),
      gust: num(f[10]),
      waveHeight: num(f[11]),
      dominantPeriod: num(f[12]),
      avgPeriod: num(f[13]),
      waveDir: num(f[14]),
      pressure: num(f[15]),
      pressureTendency: num(f[16]),
      airTemp: num(f[17]),
      waterTemp: num(f[18]),
      dewpoint: num(f[19]),
      visibility: num(f[20]),
      tide: num(f[21]),
      rt,
    });
  }
  return out;
}

// ── fetch (injectable, mirrors usgsQuakes.ts/nasaFirms.ts) ─────────────────
type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };

export async function fetchBuoys(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<BuoyObs[]> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const r = await fetchImpl(BUOYS_FEED_URL, { headers: UA, signal: AbortSignal.timeout(20000) as any });
  if (!r.ok) throw new Error(`NDBC buoys ${r.status}`);
  return parseLatestObs(await r.text(), rt);
}

// ── archive (day-file JSONL, dedup by station+observation time, gzipped
// after 2 days — identical shape to usgsQuakes.ts) ──────────────────────────
const archivedTime = new Map<string, number>();
let seeded = false;

function buoysDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "buoys");
}

function seedSeen(dir: string, nowMs: number): void {
  for (const dayMs of [nowMs, nowMs - 86400_000]) {
    const fp = path.join(dir, `${new Date(dayMs).toISOString().slice(0, 10)}.jsonl`);
    try {
      for (const line of fs.readFileSync(fp, "utf8").split("\n")) {
        if (!line) continue;
        try {
          const row = JSON.parse(line);
          const prev = archivedTime.get(row.station);
          if (prev === undefined || (row.time ?? 0) > prev) archivedTime.set(row.station, row.time ?? 0);
        } catch {}
      }
    } catch {}
  }
}

export function archiveBuoys(obs: BuoyObs[], baseDir?: string, nowMs?: number): number {
  const dir = buoysDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) {
    seedSeen(dir, now);
    seeded = true;
  }
  const fresh = obs.filter((o) => {
    const prev = archivedTime.get(o.station);
    if (prev === undefined) return true;
    return (o.time ?? 0) > prev;
  });
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((o) => JSON.stringify(o)).join("\n") + "\n");
    fresh.forEach((o) => archivedTime.set(o.station, o.time ?? 0));
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] buoys archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldBuoyDays(baseDir?: string, nowMs?: number): number {
  const dir = buoysDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      if (now - Date.parse(f.slice(0, 10)) < 2 * 86400_000) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

// ── in-memory cache + poll loop (mirrors usgsQuakes.ts's boot pattern) ─────
let cache: { at: number; obs: BuoyObs[] } | null = null;
let polling = false;

export function latestBuoys(): { at: number; obs: BuoyObs[] } | null {
  return cache;
}

export async function refreshBuoysCache(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const obs = await fetchBuoys(fetchImpl, nowMs);
    if (obs.length || !cache) cache = { at: Date.now(), obs };
    try { archiveBuoys(obs, undefined, nowMs); } catch {}
    try { gzipOldBuoyDays(undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] buoys refresh:", e?.message || e);
  }
}

/** Keyless, no gating — boots eagerly at server start (KNOWN BROKEN #9's
 *  lesson: a lazy first-request connect leaves a cold gap in the archive).
 *  30-min cadence: individual NDBC stations typically report hourly, so
 *  polling faster than that only re-confirms unchanged readings — the
 *  station+time dedup in archiveBuoys makes over-polling cheap (no-op
 *  writes) rather than harmful, and 30 min keeps a fresh cache for the
 *  eventual map layer without hammering a courtesy keyless feed. */
export function bootBuoysPoll(intervalMs = 30 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshBuoysCache();
  setInterval(() => { refreshBuoysCache(); }, intervalMs).unref?.();
}
