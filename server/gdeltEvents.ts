/**
 * gdeltEvents.ts — GDELT 2.0 event-stream pipeline, filtered to unrest/
 * strike events near OUR tracked facilities. Stream #7 of the DATA STREAM
 * EXPANSION build order (hypothesis + ladder filed BEFORE the build;
 * format live-verified 2026-07-05 against a real export file).
 *
 * HYPOTHESIS (gate 2, not attempted): geo-tagged unrest/strike bursts
 * near tracked facilities as an ALERT TRIGGER joined to our own sensors
 * (imagery/AIS/FIRMS) — a verification prompt, never a direct trade.
 * HONESTY: CAMEO is an actor-actor political taxonomy — it captures
 * strikes/protests/unrest well but NOT clean industrial accidents (FIRMS
 * is our fire sensor); GDELT geocoding is city/ADM-approximate, never
 * facility-exact.
 *
 * GOTCHA (verified): data.gdeltproject.org serves an INVALID HTTPS cert —
 * the 15-min files are fetched over plain http:// by necessity. Content
 * is public event data; integrity risk is acknowledged in the manifest.
 *
 * Filter at ingest keeps the archive KB-scale: CAMEO root codes
 * 14 (protest), 17 (coerce), 18 (assault), 19 (fight), 20 (mass
 * violence) + code 143x (strikes/boycotts), inside ~0.5° boxes around
 * the strategic sites. License: GDELT is free for unlimited use incl.
 * commercial, redistribution permitted; attribution "The GDELT Project".
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { unzipSync } from "fflate";
import { archiveBaseDir } from "./datacoreArchive";
import strategicSites from "../datacore/sites/strategic_sites.json";

export const GDELT_BBOX_DEG = 0.5;
/** CAMEO EventRootCodes kept (plus any EventCode starting "143"). */
export const GDELT_ROOT_CODES = new Set(["14", "17", "18", "19", "20"]);

// v2 Events column indices — confirmed against a real export 2026-07-05
const COL = { id: 0, day: 1, code: 26, root: 28, gold: 30, mentions: 31, tone: 34, lat: 56, lon: 57, url: 60 };

export interface GdeltEvent {
  id: string;        // GlobalEventID (dedup key)
  day: string;       // YYYYMMDD event day
  code: string;      // CAMEO EventCode
  root: string;      // CAMEO EventRootCode
  gold: number | null;   // GoldsteinScale
  tone: number | null;   // AvgTone
  mentions: number | null;
  lat: number;
  lon: number;
  site: string;      // matched facility id (join key to the graph)
  url: string | null;
  rt: string;        // as-seen UTC timestamp of the 15-min file
}

export interface FacilityBox { id: string; lat: number; lon: number }

export function facilityBoxes(): FacilityBox[] {
  const sites: any[] = (strategicSites as any).sites || [];
  return sites
    .filter((s) => Number.isFinite(s?.lat) && Number.isFinite(s?.lon))
    .map((s) => ({ id: s.id, lat: s.lat, lon: s.lon }));
}

export function nearestFacility(lat: number, lon: number, boxes: FacilityBox[]): string | null {
  for (const b of boxes) {
    if (Math.abs(lat - b.lat) <= GDELT_BBOX_DEG && Math.abs(lon - b.lon) <= GDELT_BBOX_DEG) return b.id;
  }
  return null;
}

/** Parses one 15-min export CSV (tab-separated, 61 cols), applying the
 *  CAMEO + facility-bbox filter. */
export function parseGdeltExport(csv: string, boxes: FacilityBox[], rt: string): GdeltEvent[] {
  const out: GdeltEvent[] = [];
  for (const line of csv.split("\n")) {
    if (!line) continue;
    const c = line.split("\t");
    if (c.length < 61) continue;
    const root = c[COL.root] || "";
    const code = c[COL.code] || "";
    if (!GDELT_ROOT_CODES.has(root) && !code.startsWith("143")) continue;
    const lat = parseFloat(c[COL.lat]);
    const lon = parseFloat(c[COL.lon]);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    const site = nearestFacility(lat, lon, boxes);
    if (!site) continue;
    const num = (s: string) => { const n = parseFloat(s); return Number.isFinite(n) ? n : null; };
    out.push({
      id: c[COL.id], day: c[COL.day], code, root,
      gold: num(c[COL.gold]), tone: num(c[COL.tone]), mentions: num(c[COL.mentions]),
      lat, lon, site, url: c[COL.url] || null, rt,
    });
  }
  return out;
}

/** Parses lastupdate.txt — three lines "size md5 url"; returns the export
 *  CSV zip URL (normalized to http://, see the cert gotcha above). */
export function parseLastUpdate(text: string): string | null {
  for (const line of String(text || "").split("\n")) {
    const url = line.trim().split(/\s+/)[2];
    if (url && url.includes(".export.CSV.zip")) return url.replace(/^https:/, "http:");
  }
  return null;
}

// ── Fetch ───────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string>; arrayBuffer(): Promise<ArrayBuffer> }>;
const LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt";
let lastExportUrl: string | null = null;

export async function fetchGdeltEvents(
  fetchImpl: FetchFn = fetch as any, nowMs?: number,
): Promise<GdeltEvent[]> {
  const rt = new Date(nowMs ?? Date.now()).toISOString();
  const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };
  const lu = await fetchImpl(LASTUPDATE_URL, { headers: UA, signal: AbortSignal.timeout(15000) as any });
  if (!lu.ok) throw new Error(`lastupdate -> ${lu.status}`);
  const url = parseLastUpdate(await lu.text());
  if (!url) throw new Error("no export url in lastupdate.txt");
  if (url === lastExportUrl) return []; // GDELT republishes; skip identical files
  const zr = await fetchImpl(url, { headers: UA, signal: AbortSignal.timeout(30000) as any });
  if (!zr.ok) throw new Error(`export -> ${zr.status}`);
  const files = unzipSync(new Uint8Array(await zr.arrayBuffer()));
  const name = Object.keys(files)[0];
  if (!name) throw new Error("empty export zip");
  const csv = Buffer.from(files[name]).toString("utf8");
  lastExportUrl = url;
  return parseGdeltExport(csv, facilityBoxes(), rt);
}

// ── Archive (dedup by GlobalEventID) ────────────────────────────────────────

const archivedIds = new Set<string>();
let seeded = false;

function gdeltDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "gdelt");
}

function seedSeen(dir: string, nowMs: number): void {
  for (let i = 0; i < 3; i++) {
    const day = new Date(nowMs - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { archivedIds.add(JSON.parse(line).id); } catch {}
      }
    }
  }
}

export function archiveGdeltEvents(events: GdeltEvent[], baseDir?: string, nowMs?: number): number {
  const dir = gdeltDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = events.filter((e) => e.id && !archivedIds.has(e.id));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((e) => JSON.stringify(e)).join("\n") + "\n");
    fresh.forEach((e) => archivedIds.add(e.id));
    if (archivedIds.size > 100_000) archivedIds.clear();
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] gdelt archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldGdeltDays(baseDir?: string, nowMs?: number): number {
  const dir = gdeltDir(baseDir);
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

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; events: GdeltEvent[] } | null = null;
let polling = false;

export function latestGdeltEvents(): { at: number; events: GdeltEvent[] } | null {
  return cache;
}

export async function refreshGdeltCache(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const events = await fetchGdeltEvents(fetchImpl, nowMs);
    // rolling 48h window of matched events for the API surface
    const prior = (cache?.events || []).filter((e) => Date.now() - Date.parse(e.rt) < 48 * 3600_000);
    const ids = new Set(prior.map((e) => e.id));
    cache = { at: Date.now(), events: [...prior, ...events.filter((e) => !ids.has(e.id))] };
    try { archiveGdeltEvents(events, undefined, nowMs); } catch {}
    try { gzipOldGdeltDays(undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] gdelt refresh:", e?.message || e);
  }
}

/** 15-min poll matching GDELT's publication cadence; identical republished
 *  files are skipped by URL comparison. */
export function bootGdeltPoll(intervalMs = 15 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshGdeltCache();
  setInterval(() => { refreshGdeltCache(); }, intervalMs).unref?.();
}
