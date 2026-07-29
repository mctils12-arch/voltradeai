/**
 * spaceWeather.ts — NOAA SWPC geomagnetic + solar X-ray flux (RESEARCH:
 * "Live-data gap sweep" 2026-07-27, research/open_questions.md "GRID-
 * ADJACENT FUTURE ROOTS" #1). RAW overlay per the raw-vs-signals rule:
 * official published readings + their official classification formulas
 * (Kp -> NOAA G-scale, X-ray flux -> GOES flare class) displayed as-is,
 * no predictive claim. Mike's GIC hypothesis ("geomagnetic storms induce
 * grid-damaging currents") stays gate-1-unvalidated — this ships the
 * archiver + display only; the DOE OE-417 outage-correlation study is a
 * filed follow-up, not attempted this session.
 *
 * services.swpc.noaa.gov: US government work, public domain. Both feeds
 * are keyless static JSON (no auth, no User-Agent requirement observed,
 * but one is sent anyway per good-citizen convention matching nwsAlerts).
 *
 * Two independent streams, archived separately (different cadence/shape):
 *  - K-index: 3-hourly planetary Kp, ~60 rows/response (7.5-day window).
 *  - X-ray flux: ~1-min GOES XRS readings, two energy bands, ~1-day window.
 * Both dedup against the immutable upstream time_tag (X-ray keys on
 * time_tag+energy since two bands share a time_tag).
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const KINDEX_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json";
const XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json";
const XRAY_LONG_BAND = "0.1-0.8nm";

export interface KIndexRec {
  time_tag: string;
  kp: number;
  a_running: number | null;
  station_count: number | null;
}

export interface XrayRec {
  time_tag: string;
  satellite: number | null;
  energy: string;
  flux: number;
  electron_contamination: boolean | null;
}

/** Official NOAA Kp -> G-scale storm classification (published formula, not a fit). */
export function classifyKp(kp: number): { gScale: number | null; label: string } {
  if (kp >= 9) return { gScale: 5, label: "G5 — extreme" };
  if (kp >= 8) return { gScale: 4, label: "G4 — severe" };
  if (kp >= 7) return { gScale: 3, label: "G3 — strong" };
  if (kp >= 6) return { gScale: 2, label: "G2 — moderate" };
  if (kp >= 5) return { gScale: 1, label: "G1 — minor" };
  if (kp >= 4) return { gScale: null, label: "active" };
  if (kp >= 3) return { gScale: null, label: "unsettled" };
  return { gScale: null, label: "quiet" };
}

/** Official GOES flare classification from 0.1-0.8nm flux in W/m^2 (published formula). */
export function classifyFlare(flux: number): { cls: "A" | "B" | "C" | "M" | "X"; magnitude: number; label: string } {
  const bands: Array<["A" | "B" | "C" | "M" | "X", number]> = [
    ["X", 1e-4], ["M", 1e-5], ["C", 1e-6], ["B", 1e-7], ["A", 1e-8],
  ];
  for (const [cls, floor] of bands) {
    if (flux >= floor) {
      const magnitude = +(flux / floor).toFixed(2);
      return { cls, magnitude, label: `${cls}${magnitude}` };
    }
  }
  return { cls: "A", magnitude: +(flux / 1e-8).toFixed(2), label: `A${(flux / 1e-8).toFixed(2)}` };
}

// ── Parse ────────────────────────────────────────────────────────────────

export function parseKIndex(json: any): KIndexRec[] {
  const recs: KIndexRec[] = [];
  for (const row of Array.isArray(json) ? json : []) {
    if (!row?.time_tag || typeof row.Kp !== "number") continue;
    recs.push({
      time_tag: row.time_tag,
      kp: row.Kp,
      a_running: typeof row.a_running === "number" ? row.a_running : null,
      station_count: typeof row.station_count === "number" ? row.station_count : null,
    });
  }
  return recs;
}

export function parseXray(json: any): XrayRec[] {
  const recs: XrayRec[] = [];
  for (const row of Array.isArray(json) ? json : []) {
    if (!row?.time_tag || typeof row.flux !== "number" || !row.energy) continue;
    recs.push({
      time_tag: row.time_tag,
      satellite: typeof row.satellite === "number" ? row.satellite : null,
      energy: row.energy,
      flux: row.flux,
      // NOAA's own field is misspelled "electron_contaminaton" upstream —
      // read defensively, normalize the name on our side.
      electron_contamination: typeof row.electron_contaminaton === "boolean" ? row.electron_contaminaton : null,
    });
  }
  return recs;
}

// ── Fetch ────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

async function fetchJson(url: string, fetchImpl: FetchFn): Promise<any> {
  const r = await fetchImpl(url, {
    headers: { "User-Agent": "voltradeai-datacore/1.0 (mctils12@gmail.com)" },
    signal: AbortSignal.timeout(25000) as any,
  });
  if (!r.ok) throw new Error(`space weather -> ${r.status}`);
  return JSON.parse(await r.text());
}

export async function fetchKIndex(fetchImpl: FetchFn = fetch as any): Promise<KIndexRec[]> {
  return parseKIndex(await fetchJson(KINDEX_URL, fetchImpl));
}

export async function fetchXray(fetchImpl: FetchFn = fetch as any): Promise<XrayRec[]> {
  return parseXray(await fetchJson(XRAY_URL, fetchImpl));
}

// ── Archive (dedup by time_tag — SWPC readings are immutable once posted) ──

function streamDir(name: string, baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), name);
}

function seedSeenKeys(dir: string, nowMs: number, keyOf: (rec: any) => string, seen: Set<string>): void {
  for (let i = 0; i < 3; i++) {
    const day = new Date(nowMs - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz") ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8") : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { seen.add(keyOf(JSON.parse(line))); } catch {}
      }
    }
  }
}

function trimIfLarge(seen: Set<string>): void {
  if (seen.size <= 200_000) return;
  // oldest-half trim, never clear() (nasaFirms lesson, audit #10b)
  const it = seen.values();
  const drop = seen.size >> 1;
  for (let i = 0; i < drop; i++) {
    const v = it.next();
    if (v.done) break;
    seen.delete(v.value);
  }
}

function makeArchiver<T>(streamName: string, keyOf: (rec: T) => string) {
  const seenKeys = new Set<string>();
  let seeded = false;
  return function archive(recs: T[], baseDir?: string, nowMs?: number): number {
    const dir = streamDir(streamName, baseDir);
    const now = nowMs ?? Date.now();
    if (!seeded) { seedSeenKeys(dir, now, keyOf as any, seenKeys); seeded = true; }
    const fresh = recs.filter((r) => !seenKeys.has(keyOf(r)));
    if (!fresh.length) return 0;
    try {
      fs.mkdirSync(dir, { recursive: true });
      const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
      fs.appendFileSync(fp, fresh.map((r) => JSON.stringify(r)).join("\n") + "\n");
      fresh.forEach((r) => seenKeys.add(keyOf(r)));
      trimIfLarge(seenKeys);
      return fresh.length;
    } catch (e: any) {
      console.error(`[datacore] ${streamName} archive:`, e?.message || e);
      return 0;
    }
  };
}

export const archiveKIndex = makeArchiver<KIndexRec>("swpckindex", (r) => r.time_tag);
export const archiveXray = makeArchiver<XrayRec>("swpcxray", (r) => `${r.time_tag}|${r.energy}`);

export function gzipOldDays(streamName: string, baseDir?: string, nowMs?: number): number {
  const dir = streamDir(streamName, baseDir);
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

export interface SpaceWeatherCache {
  at: number;
  kindex: { latest: KIndexRec | null; gScale: ReturnType<typeof classifyKp> | null; history: KIndexRec[] };
  xray: { latest: XrayRec | null; flare: ReturnType<typeof classifyFlare> | null; history: XrayRec[] };
}

let cache: SpaceWeatherCache | null = null;
let polling = false;

export function latestSpaceWeather(): SpaceWeatherCache | null {
  return cache;
}

// Sparkline history caps: 60 K-index rows is the full upstream window
// (7.5 days @ 3h); X-ray history is capped to the most recent 180
// long-band minutes (3h) — enough for a meaningful sparkline without
// shipping the full 1-day/2-band payload to the client every poll.
const XRAY_HISTORY_CAP = 180;

export async function refreshSpaceWeatherCache(fetchImpl: FetchFn = fetch as any, nowMs?: number, baseDir?: string): Promise<void> {
  const results = await Promise.allSettled([fetchKIndex(fetchImpl), fetchXray(fetchImpl)]);
  const kindexRecs = results[0].status === "fulfilled" ? results[0].value : null;
  const xrayRecs = results[1].status === "fulfilled" ? results[1].value : null;
  if (results[0].status === "rejected") console.error("[datacore] swpckindex refresh:", (results[0] as any).reason?.message || results[0]);
  if (results[1].status === "rejected") console.error("[datacore] swpcxray refresh:", (results[1] as any).reason?.message || results[1]);
  if (!kindexRecs && !xrayRecs) return;

  const kLatest = kindexRecs?.length ? kindexRecs[kindexRecs.length - 1] : cache?.kindex.latest ?? null;
  const xLongHistory = (xrayRecs || []).filter((r) => r.energy === XRAY_LONG_BAND);
  const xLatest = xLongHistory.length ? xLongHistory[xLongHistory.length - 1] : cache?.xray.latest ?? null;

  cache = {
    at: Date.now(),
    kindex: {
      latest: kLatest,
      gScale: kLatest ? classifyKp(kLatest.kp) : null,
      history: kindexRecs?.length ? kindexRecs.slice(-60) : cache?.kindex.history ?? [],
    },
    xray: {
      latest: xLatest,
      flare: xLatest ? classifyFlare(xLatest.flux) : null,
      history: xLongHistory.length ? xLongHistory.slice(-XRAY_HISTORY_CAP) : cache?.xray.history ?? [],
    },
  };

  try { if (kindexRecs) archiveKIndex(kindexRecs, baseDir, nowMs); } catch {}
  try { if (xrayRecs) archiveXray(xrayRecs, baseDir, nowMs); } catch {}
  try { gzipOldDays("swpckindex", baseDir, nowMs); gzipOldDays("swpcxray", baseDir, nowMs); } catch {}
}

/** 5-min poll — X-ray flux moves on minute scales; eager boot (KNOWN
 *  BROKEN #9 rule: pollers fire once at startup, never wait). */
export function bootSpaceWeatherPoll(intervalMs = 5 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshSpaceWeatherCache();
  setInterval(() => { refreshSpaceWeatherCache(); }, intervalMs).unref?.();
}
