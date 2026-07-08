/**
 * usgsQuakes.ts — USGS real-time earthquake feed (M2.5+, global, rolling
 * 24h window). RAW-DATA overlay only (CLAUDE.md RAW-vs-SIGNAL surface
 * rule): displays USGS-reported events as-is with source attribution, no
 * predictive claim — ships ungated, no key required.
 *
 * EDGE DOCTRINE #1 (build data, don't buy it): keyless, no documented rate
 * limit, US government public domain (usgs.gov/information-policies-and-
 * notices/copyrights-and-credits — "generally free from copyright ...
 * unless specifically annotated otherwise"). Source: USGS Earthquake
 * Hazards Program GeoJSON summary feed
 * (earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson).
 * M2.5+ is USGS's own recommended default threshold for general use (below
 * it, dense regional clusters — e.g. California, Alaska swarms — swamp a
 * global feed with little added signal per event); live-verified at build
 * time: 39 events/day at M2.5+ vs 242/day unfiltered ("all" feed) vs
 * 17/day at M4.5+ (too sparse for archive-density purposes).
 *
 * Every event carries USGS's own stable `id` (e.g. "us6000taui") — unlike
 * nasaFirms.ts/edgarForm4.ts, no composite dedup key needs constructing.
 * USGS also publishes a full free historical query API of its own
 * (fdsnws/event), so unlike FIRMS this archive is continuity/convenience,
 * not the only source of history — still recorded from day one per the
 * standing archive-first doctrine (a day not recorded locally still costs
 * a future session an extra live API round-trip to backfill).
 *
 * ANGLE-HUNTING NOTE (filed alongside in open_questions.md): pairs with the
 * existing entity_map.json (facility/operator table) and NASA FIRMS as a
 * hazard-adjacent RAW layer — insurer (P&C), utility, and supply-chain
 * exposure hypotheses are gate-1/gate-2 work, not attempted here.
 *
 * Boundary: no imports from trading logic — pure fetch/parse/archive, same
 * day-file-JSONL-with-dedup shape as nasaFirms.ts/edgarForm4.ts.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

export const QUAKES_FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson";

export interface QuakeEvent {
  id: string;              // USGS's own stable event id
  mag: number | null;
  place: string | null;
  lat: number | null;
  lon: number | null;
  depth: number | null;    // km
  time: number | null;     // origin time, ms epoch UTC
  updated: number | null;  // ms epoch UTC — USGS revises magnitude/location as review proceeds
  tsunami: boolean;
  sig: number | null;      // USGS significance score, 0-1000
  net: string | null;      // reporting network (us, ci, nc, ...)
  magType: string | null;
  type: string;            // "earthquake" | "quarry blast" | ... — kept as-reported, never filtered
  status: string | null;   // "automatic" | "reviewed" | "deleted"
  url: string | null;      // USGS event page
  rt: string;              // as-seen UTC date (day WE fetched it)
}

function toNumOrNull(v: any): number | null {
  const n = typeof v === "number" ? v : parseFloat(v);
  return Number.isFinite(n) ? n : null;
}

/** Parses a USGS earthquake GeoJSON FeatureCollection. Events with no `id`
 *  are dropped (nothing to dedup or archive against) — everything else is
 *  kept as-reported, including non-"earthquake" types (quarry blasts,
 *  explosions) and `status:"deleted"` rows the network later retracted,
 *  matching the fires.json precedent ("archive keeps ALL levels, display
 *  filters, the record does not"). */
export function parseQuakesGeoJson(json: any, rt: string): QuakeEvent[] {
  const out: QuakeEvent[] = [];
  for (const f of json?.features || []) {
    const id = f?.id ? String(f.id) : null;
    if (!id) continue;
    const p = f?.properties || {};
    const coords = f?.geometry?.coordinates || [];
    out.push({
      id,
      mag: toNumOrNull(p.mag),
      place: p.place || null,
      lon: toNumOrNull(coords[0]),
      lat: toNumOrNull(coords[1]),
      depth: toNumOrNull(coords[2]),
      time: toNumOrNull(p.time),
      updated: toNumOrNull(p.updated),
      tsunami: p.tsunami === 1 || p.tsunami === true,
      sig: toNumOrNull(p.sig),
      net: p.net || null,
      magType: p.magType || null,
      type: p.type || "earthquake",
      status: p.status || null,
      url: p.url || null,
      rt,
    });
  }
  return out;
}

// ── fetch (injectable, mirrors nasaFirms.ts/fdaEvents.ts) ──────────────────
type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;
const UA = { "User-Agent": "voltradeai-datacore/1.0 (+https://voltradeai.com)" };

export async function fetchQuakes(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<QuakeEvent[]> {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const r = await fetchImpl(QUAKES_FEED_URL, { headers: UA, signal: AbortSignal.timeout(15000) as any });
  if (!r.ok) throw new Error(`USGS quakes ${r.status}`);
  return parseQuakesGeoJson(JSON.parse(await r.text()), rt);
}

// ── archive (day-file JSONL, dedup by USGS's own id, gzipped after 2 days —
// identical shape to nasaFirms.ts/edgarForm4.ts) ────────────────────────────
const archivedIds = new Set<string>();
let seeded = false;

function quakesDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "earthquakes");
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

/** Re-archives an event whenever USGS's `updated` timestamp moves past what
 *  we last recorded (magnitude/location revisions during review) — an
 *  earthquake's id never changes, but its properties legitimately do, so a
 *  pure id-dedup would freeze every event at its first-seen (often
 *  "automatic", unreviewed) values forever. */
const archivedUpdated = new Map<string, number>();

export function archiveQuakes(events: QuakeEvent[], baseDir?: string, nowMs?: number): number {
  const dir = quakesDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) {
    seedSeen(dir, now);
    archivedIds.forEach((id) => archivedUpdated.set(id, 0));
    seeded = true;
  }
  const fresh = events.filter((e) => {
    if (!e.id) return false;
    const prevUpdated = archivedUpdated.get(e.id);
    if (prevUpdated === undefined) return true;
    return (e.updated ?? 0) > prevUpdated;
  });
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((e) => JSON.stringify(e)).join("\n") + "\n");
    fresh.forEach((e) => { archivedIds.add(e.id); archivedUpdated.set(e.id, e.updated ?? 0); });
    if (archivedIds.size > 100_000) {
      let drop = archivedIds.size >> 1;
      const it = archivedIds.values();
      let cur = it.next();
      while (!cur.done && drop-- > 0) {
        archivedUpdated.delete(cur.value);
        archivedIds.delete(cur.value);
        cur = it.next();
      }
    }
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] quakes archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldQuakeDays(baseDir?: string, nowMs?: number): number {
  const dir = quakesDir(baseDir);
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

// ── in-memory cache + poll loop (mirrors nasaFirms.ts's boot pattern) ──────
let cache: { at: number; events: QuakeEvent[] } | null = null;
let polling = false;

export function latestQuakes(): { at: number; events: QuakeEvent[] } | null {
  return cache;
}

export async function refreshQuakesCache(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const events = await fetchQuakes(fetchImpl, nowMs);
    if (events.length || !cache) cache = { at: Date.now(), events };
    try { archiveQuakes(events, undefined, nowMs); } catch {}
    try { gzipOldQuakeDays(undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] quakes refresh:", e?.message || e);
  }
}

/** Keyless, no gating — boots eagerly at server start (KNOWN BROKEN #9's
 *  lesson: a lazy first-request connect leaves a cold gap in the archive).
 *  15-min cadence: the upstream feed itself refreshes ~every 5 min, and
 *  near-real-time hazard events are more time-sensitive than FIRMS' ~3h
 *  NRT-latency 30-min cadence, while staying far under any reasonable
 *  courtesy-use rate for a keyless public feed. */
export function bootQuakesPoll(intervalMs = 15 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshQuakesCache();
  setInterval(() => { refreshQuakesCache(); }, intervalMs).unref?.();
}
