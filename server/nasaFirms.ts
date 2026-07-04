/**
 * nasaFirms.ts — NASA FIRMS (Fire Information for Resource Management System)
 * active-fire detections, the Tier-1(c) geospatial root queued in
 * research/wishlist.md's GEOSPATIAL LICENSING REGISTER (verdict (c) FIRES:
 * free MAP_KEY, commercial-lawful, VIIRS 375m, NRT ~3h latency, LANCE
 * attribution + "not for safety-of-life" disclaimer required; NO free
 * history exists, so archiving starts from day one).
 *
 * RAW-DATA overlay only (CLAUDE.md RAW-vs-SIGNAL surface rule): displays
 * detections as-is with source attribution, no predictive claim — ships
 * ungated. A future gate-1/gate-2 signal (e.g. wildfire-adjacent insurer/
 * utility/timber exposure) is a separate, later hypothesis.
 *
 * Key-gated exactly like vesselStream.ts: without NASA_FIRMS_MAP_KEY the
 * layer reports awaiting_key and the poll loop never starts (no point
 * hitting an endpoint with no key). Boots eagerly at server start (KNOWN
 * BROKEN #9's lesson: lazy-first-request connect leaves a cold gap).
 *
 * Boundary: no imports from trading logic — pure fetch/parse/archive, same
 * shape as edgarForm4.ts (discrete dated events, not continuous tracks, so
 * it reuses that module's day-file-JSONL-with-dedup archive shape rather
 * than datacoreArchive.ts's adaptive-thinning position-track shape).
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

/** The key env var accepts BOTH names: the module shipped reading
 *  NASA_FIRMS_MAP_KEY, but the human set the key in Railway as
 *  FIRMS_MAP_KEY (2026-07-04) — the code adapts to the action already
 *  taken rather than requiring a Railway rename. */
export function firmsKey(env: NodeJS.ProcessEnv = process.env): string {
  return env.NASA_FIRMS_MAP_KEY || env.FIRMS_MAP_KEY || "";
}

export function firmsEnabled(env: NodeJS.ProcessEnv = process.env): boolean {
  return Boolean(firmsKey(env));
}

// VIIRS 375m NRT is the higher-resolution source the licensing register
// names; NOAA-20 is used as a second satellite pass (more detections/day)
// but this module only queries SNPP today — adding NOAA-20 is a one-line
// extension of AREA_SOURCES if the single-satellite detection rate proves
// too sparse for a future gate-1 test.
export const FIRMS_SOURCE = "VIIRS_SNPP_NRT";
export const FIRMS_AREA = "-180,-90,180,90"; // world bbox
export const FIRMS_DAY_RANGE = 1; // most recent day only — dedup handles overlap across polls

export function firmsUrl(mapKey: string, source = FIRMS_SOURCE, area = FIRMS_AREA, dayRange = FIRMS_DAY_RANGE): string {
  return `https://firms.modaps.eosdis.nasa.gov/api/area/csv/${encodeURIComponent(mapKey)}/${source}/${area}/${dayRange}`;
}

export type FireConfidence = "low" | "nominal" | "high";

export interface FireDetection {
  id: string;            // our own dedup key — see fireDetectionId()
  lat: number;
  lon: number;
  brightness: number | null;   // VIIRS bright_ti4 (K) / MODIS brightness (K)
  confidence: FireConfidence;
  frp: number | null;          // fire radiative power, MW — rough intensity proxy
  acq_date: string;            // YYYY-MM-DD, UTC
  acq_time: string;            // HHMM, UTC
  satellite: string;
  daynight: "D" | "N" | null;
}

/** VIIRS reports confidence as a letter (l/n/h); MODIS reports 0-100. Both
 *  collapse to the same three-bucket scale so the client renders one legend
 *  regardless of which source fed a given row. */
export function classifyFirmsConfidence(raw: string): FireConfidence {
  const s = raw.trim().toLowerCase();
  if (s === "l" || s === "low") return "low";
  if (s === "h" || s === "high") return "high";
  if (s === "n" || s === "nominal") return "nominal";
  const n = parseFloat(s);
  if (Number.isFinite(n)) {
    if (n < 30) return "low";
    if (n >= 80) return "high";
    return "nominal";
  }
  return "nominal";
}

function toNumOrNull(s: string | undefined): number | null {
  if (s == null || s === "") return null;
  const n = parseFloat(s);
  return Number.isFinite(n) ? n : null;
}

/** Our own dedup identity for a detection: FIRMS re-serves the same
 *  detection across overlapping day-range windows on every poll, and has no
 *  stable per-row id of its own. Satellite + rounded position + acquisition
 *  timestamp is stable across re-fetches of the same underlying pass while
 *  still separating genuinely distinct nearby fires. */
export function fireDetectionId(d: { satellite: string; lat: number; lon: number; acq_date: string; acq_time: string }): string {
  return `${d.satellite}:${d.lat.toFixed(3)}:${d.lon.toFixed(3)}:${d.acq_date}:${d.acq_time}`;
}

/** Parses a FIRMS area-CSV response (VIIRS or MODIS column layout — both
 *  share latitude/longitude/acq_date/acq_time/satellite/confidence/frp;
 *  VIIRS uses bright_ti4 where MODIS uses brightness). Column-name lookup
 *  (not fixed indices) so either source's header row works unchanged. */
export function parseFirmsCsv(csv: string): FireDetection[] {
  const lines = csv.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const idx = (name: string) => header.indexOf(name);
  const iLat = idx("latitude"), iLon = idx("longitude");
  const iBright = idx("bright_ti4") >= 0 ? idx("bright_ti4") : idx("brightness");
  const iConf = idx("confidence");
  const iFrp = idx("frp");
  const iDate = idx("acq_date");
  const iTime = idx("acq_time");
  const iSat = idx("satellite");
  const iDN = idx("daynight");
  if (iLat < 0 || iLon < 0) return [];
  const out: FireDetection[] = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const cols = lines[i].split(",");
    const lat = toNumOrNull(cols[iLat]);
    const lon = toNumOrNull(cols[iLon]);
    if (lat == null || lon == null) continue;
    const satellite = (iSat >= 0 ? cols[iSat] : "").trim() || "unknown";
    const acq_date = (iDate >= 0 ? cols[iDate] : "").trim();
    const acq_time = (iTime >= 0 ? cols[iTime] : "").trim();
    const dn = (iDN >= 0 ? cols[iDN] : "").trim().toUpperCase();
    const det = {
      lat, lon,
      brightness: iBright >= 0 ? toNumOrNull(cols[iBright]) : null,
      confidence: classifyFirmsConfidence(iConf >= 0 ? cols[iConf] : ""),
      frp: iFrp >= 0 ? toNumOrNull(cols[iFrp]) : null,
      acq_date, acq_time, satellite,
      daynight: dn === "D" || dn === "N" ? (dn as "D" | "N") : null,
    };
    out.push({ id: fireDetectionId(det), ...det });
  }
  return out;
}

// ── fetch layer (injectable for tests, mirrors edgarForm4.ts's FetchFn) ────
type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const FIRMS_UA = { "User-Agent": "voltradeai-datacore/1.0 (research@voltradeai.com)" };

export async function fetchFirmsDetections(mapKey: string, fetchImpl: FetchFn = fetch as any): Promise<FireDetection[]> {
  const r = await fetchImpl(firmsUrl(mapKey), { headers: FIRMS_UA, signal: AbortSignal.timeout(15000) as any });
  if (!r.ok) throw new Error(`FIRMS ${r.status}`);
  return parseFirmsCsv(await r.text());
}

// ── archive (COLLECT-EVERYTHING: NASA FIRMS has NO free history — every
// detection not archived today is permanently lost, per the licensing
// register). Append-only JSONL per UTC day under <archive>/fires/, deduped
// by our own id, gzipped after 2 days — identical shape to edgarForm4.ts's
// filings archive (both are discrete dated events, not continuous tracks).
const archivedIds = new Set<string>();
let seeded = false;

function firesDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "fires");
}

function seedSeen(dir: string, nowMs: number): void {
  for (const dayMs of [nowMs, nowMs - 86400_000]) {
    const fp = path.join(dir, `${new Date(dayMs).toISOString().slice(0, 10)}.jsonl`);
    try {
      for (const line of fs.readFileSync(fp, "utf8").split("\n")) {
        if (!line) continue;
        try { archivedIds.add(JSON.parse(line).id); } catch {}
      }
    } catch {}
  }
}

export function archiveFireDetections(dets: FireDetection[], baseDir?: string, nowMs?: number): number {
  const dir = firesDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = dets.filter((d) => d.id && !archivedIds.has(d.id));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((d) => JSON.stringify(d)).join("\n") + "\n");
    fresh.forEach((d) => archivedIds.add(d.id));
    if (archivedIds.size > 200_000) archivedIds.clear(); // bound memory
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] fires archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldFireDays(baseDir?: string, nowMs?: number): number {
  const dir = firesDir(baseDir);
  const now = nowMs ?? Date.now();
  let n = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith(".jsonl")) continue;
      const day = f.slice(0, 10);
      if (now - Date.parse(day) < 2 * 86400_000) continue;
      const fp = path.join(dir, f);
      fs.writeFileSync(`${fp}.gz`, zlib.gzipSync(fs.readFileSync(fp)));
      fs.unlinkSync(fp);
      n++;
    }
  } catch {}
  return n;
}

/** Reads archived detections for the last N days (for a future history/
 *  density view — not yet surfaced on the map, which shows the live cache). */
export function readFireHistory(days = 7, baseDir?: string, nowMs?: number, maxDetections = 20_000): FireDetection[] {
  const dir = firesDir(baseDir);
  const now = nowMs ?? Date.now();
  const out: FireDetection[] = [];
  const seen = new Set<string>();
  for (let d = 0; d < days && out.length < maxDetections; d++) {
    const day = new Date(now - d * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try {
          const det = JSON.parse(line);
          if (det.id && !seen.has(det.id)) { seen.add(det.id); out.push(det); }
        } catch {}
      }
    }
  }
  return out.slice(0, maxDetections);
}

// ── in-memory cache + poll loop (mirrors edgarForm4.ts's boot pattern) ─────
let cache: { at: number; detections: FireDetection[] } | null = null;
let polling = false;

export function latestFirms(): { at: number; detections: FireDetection[] } | null {
  return cache;
}

export async function refreshFirmsCache(env: NodeJS.ProcessEnv = process.env): Promise<void> {
  const key = firmsKey(env);
  if (!key) return;
  try {
    const detections = await fetchFirmsDetections(key);
    if (detections.length > 0 || !cache) cache = { at: Date.now(), detections };
    try { archiveFireDetections(detections); } catch {}
    try { gzipOldFireDays(); } catch {}
  } catch (e: any) {
    console.error("[datacore] FIRMS refresh:", e?.message || e);
  }
}

/** Starts the poll loop once at boot, key-gated (no point polling with no
 *  key — mirrors vesselStream.ts's enabled-check-before-connect shape).
 *  NRT latency is ~3h upstream, so a 30-min cadence is plenty responsive
 *  without wasting the shared MAP_KEY's request budget. */
export function bootFirmsPoll(env: NodeJS.ProcessEnv = process.env, intervalMs = 30 * 60_000): void {
  if (polling || !firmsEnabled(env)) return;
  polling = true;
  refreshFirmsCache(env);
  setInterval(() => { refreshFirmsCache(env); }, intervalMs).unref?.();
}
