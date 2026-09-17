/**
 * usgsWater.ts — USGS river-gauge pipeline (Mississippi barge corridor +
 * Ohio/Missouri/Illinois tributaries). Stream #6 of the DATA STREAM
 * EXPANSION build order (hypothesis + ladder filed BEFORE the build;
 * gauges live-verified 2026-07-05 — see the research annex).
 *
 * HYPOTHESIS (gate 2, not attempted): Mississippi low water → barge
 * freight stress → grain/fertilizer basis moves. Conditional signal —
 * fires only in drought years. This module surfaces RAW gauge readings
 * only; the low-water interpretation stays gate-2-locked.
 *
 * Gauges publish either stage (00065, ft) or discharge (00060, ft³/s) —
 * Memphis and Vicksburg are discharge-only (verified), so both params
 * are requested and whichever a site returns is stored. Provisional
 * readings (q=P) revise to approved (A): a revision appends a NEW row
 * (same site+param+timestamp, new value/qualifier, new rt) — the same
 * vintage discipline as fredMacro.
 *
 * Keyless; public domain; attribution "U.S. Geological Survey".
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

/** Live-verified 2026-07-05 (each returned a CURRENT reading; the dead
 *  Metropolis gauge 03611500 — 2010-stale — was caught and excluded). */
export const USGS_GAUGES: Array<{ site: string; name: string }> = [
  { site: "07010000", name: "Mississippi R at St. Louis, MO" },
  { site: "05587450", name: "Mississippi R at Grafton, IL" },
  { site: "07020500", name: "Mississippi R at Chester, IL" },
  { site: "07022000", name: "Mississippi R at Thebes, IL" },
  { site: "07032000", name: "Mississippi R at Memphis, TN" },        // discharge-only
  { site: "07289000", name: "Mississippi R at Vicksburg, MS" },      // discharge-only
  { site: "07374000", name: "Mississippi R at Baton Rouge, LA" },
  { site: "07374525", name: "Mississippi R at Belle Chasse, LA" },
  { site: "06934500", name: "Missouri R at Hermann, MO" },
  { site: "05586100", name: "Illinois R at Valley City, IL" },
  { site: "03216070", name: "Ohio R at Ironton, OH" },
  { site: "03277200", name: "Ohio R at Markland Dam, KY" },
  { site: "03303280", name: "Ohio R at Cannelton Dam, IN" },
  { site: "03612600", name: "Ohio R at Olmsted, IL" },
];

export interface GaugeObs {
  site: string;
  name: string | null;
  param: "00065" | "00060";
  d: string;            // observation timestamp (site-local ISO with offset)
  v: number;
  q: string | null;     // USGS qualifier: P provisional, A approved
  lat: number | null;
  lon: number | null;
  rt: string;           // as-seen UTC date
}

/** Parses the WaterML-JSON envelope (value.timeSeries[]). Takes the LATEST
 *  value per (site, param) series — the poll cadence owns history. */
export function parseGaugeObs(json: any, rt: string): GaugeObs[] {
  const out: GaugeObs[] = [];
  for (const ts of json?.value?.timeSeries || []) {
    const site = ts?.sourceInfo?.siteCode?.[0]?.value;
    const param = ts?.variable?.variableCode?.[0]?.value;
    if (!site || (param !== "00065" && param !== "00060")) continue;
    const geo = ts?.sourceInfo?.geoLocation?.geogLocation;
    const vals: any[] = ts?.values?.[0]?.value || [];
    const last = vals[vals.length - 1];
    if (!last?.dateTime) continue;
    const v = parseFloat(last.value);
    if (!Number.isFinite(v) || v <= -999999) continue; // USGS sentinel for missing
    out.push({
      site,
      name: ts?.sourceInfo?.siteName || null,
      param,
      d: last.dateTime,
      v,
      q: Array.isArray(last.qualifiers) ? last.qualifiers[0] ?? null : null,
      lat: geo?.latitude ?? null,
      lon: geo?.longitude ?? null,
      rt,
    });
  }
  return out;
}

// ── Fetch (one request covers all sites) ────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export async function fetchGauges(
  fetchImpl: FetchFn = fetch as any, nowMs?: number,
): Promise<GaugeObs[]> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const url = "https://waterservices.usgs.gov/nwis/iv/?format=json" +
    `&sites=${USGS_GAUGES.map((g) => g.site).join(",")}` +
    "&parameterCd=00065,00060&siteStatus=active";
  const r = await fetchImpl(url, {
    headers: { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" },
    signal: AbortSignal.timeout(20000) as any,
  });
  if (!r.ok) throw new Error(`usgs -> ${r.status}`);
  return parseGaugeObs(JSON.parse(await r.text()), rt);
}

// ── Archive (dedup by site|param|d|v|q — revisions append) ─────────────────

const archivedKeys = new Set<string>();
let seeded = false;

function usgsDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "usgswater");
}

function keyOf(o: GaugeObs): string {
  return `${o.site}|${o.param}|${o.d}|${o.v}|${o.q ?? ""}`;
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
        try { archivedKeys.add(keyOf(JSON.parse(line))); } catch {}
      }
    }
  }
}

export function archiveGaugeObs(obs: GaugeObs[], baseDir?: string, nowMs?: number): number {
  const dir = usgsDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = obs.filter((o) => !archivedKeys.has(keyOf(o)));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((o) => JSON.stringify(o)).join("\n") + "\n");
    fresh.forEach((o) => archivedKeys.add(keyOf(o)));
    if (archivedKeys.size > 100_000) archivedKeys.clear();
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] usgswater archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldUsgsDays(baseDir?: string, nowMs?: number): number {
  const dir = usgsDir(baseDir);
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

/** Reconstructs the latest known reading per (site, param) from the
 *  revision-append archive: keeps the row with the newest observation
 *  timestamp `d` (ties broken by `rt`, the vintage a provisional->approved
 *  revision appends under) — a revision adds a NEW row rather than
 *  replacing one, so a naive "last line wins" read would risk surfacing a
 *  stale provisional value over a later approved one if they ever land out
 *  of file order. Same cold-cache-no-disk-backfill fix already shipped for
 *  cbpBorderWait.ts/faaStatus.ts/edgarForm4.ts (research/open_questions.md's
 *  cold-cache-no-disk-backfill table named usgsWater.ts as one of the
 *  remaining "new reader needed" modules).
 *
 *  NOT routed through `cacheBackfill.ts`'s shared `resolveCacheItems`, for
 *  the same reason faaStatus.ts isn't: `refreshGaugeCache`'s existing
 *  behavior already trusts a successful-but-empty poll as a real state
 *  (`if (gauges.length || !cache)`) rather than treating it as ambiguous
 *  with a failed one — `resolveCacheItems` would silently change that. The
 *  gap this fix closes is narrower: only the THROW path (`fetchGauges`
 *  rejects — bad HTTP status or a network error) currently leaves `cache`
 *  untouched forever on a cold boot. */
export function backfillGaugesFromArchive(baseDir?: string, nowMs?: number, days = 3): GaugeObs[] {
  const dir = usgsDir(baseDir);
  const now = nowMs ?? Date.now();
  const latest = new Map<string, GaugeObs>();
  for (let i = 0; i < days; i++) {
    const day = new Date(now - i * 86400_000).toISOString().slice(0, 10);
    for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
      let text: string | null = null;
      try {
        text = fp.endsWith(".gz")
          ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
          : fs.readFileSync(fp, "utf8");
      } catch { continue; }
      for (const line of text.split("\n")) {
        if (!line) continue;
        let o: GaugeObs;
        try { o = JSON.parse(line); } catch { continue; }
        if (!o || !o.site || !o.param) continue;
        const identity = `${o.site}|${o.param}`;
        const prev = latest.get(identity);
        const newer = !prev
          || Date.parse(o.d) > Date.parse(prev.d)
          || (Date.parse(o.d) === Date.parse(prev.d) && (o.rt || "") > (prev.rt || ""));
        if (newer) latest.set(identity, o);
      }
    }
  }
  return Array.from(latest.values());
}

// ── Cache + poll ────────────────────────────────────────────────────────────

let cache: { at: number; gauges: GaugeObs[] } | null = null;
let polling = false;

export function latestGauges(): { at: number; gauges: GaugeObs[] } | null {
  return cache;
}

export async function refreshGaugeCache(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const gauges = await fetchGauges(fetchImpl, nowMs);
    if (gauges.length || !cache) cache = { at: Date.now(), gauges };
    try { archiveGaugeObs(gauges, undefined, nowMs); } catch {}
    try { gzipOldUsgsDays(undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] usgswater refresh:", e?.message || e);
    // transport error (thrown by fetchGauges) — keep the last snapshot, or
    // backfill from disk on a cold cache so a boot-time outage doesn't leave
    // /api/data/rivergauges warming_up forever despite a real archive on disk.
    if (!cache) {
      const backfilled = backfillGaugesFromArchive(undefined, nowMs);
      if (backfilled.length) cache = { at: Date.now(), gauges: backfilled };
    }
  }
}

/** Test-only reset — `cache`/`polling` are module-singleton state (same
 *  problem faaStatus.ts's `_resetFaaCacheForTests` solves for its own
 *  module), so a test exercising the cold-cache backfill path must be able
 *  to force `cache` back to null rather than rely on file execution order. */
export function _resetGaugeCacheForTests(): void {
  cache = null;
  polling = false;
}

/** 1h poll — river stage moves slowly; one request covers all 14 sites. */
export function bootUsgsPoll(intervalMs = 60 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshGaugeCache();
  setInterval(() => { refreshGaugeCache(); }, intervalMs).unref?.();
}
