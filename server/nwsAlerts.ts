/**
 * nwsAlerts.ts — NWS active severe-weather alerts (BUILD ORDER 2 #3,
 * 2026-07-05; access live-probed keyless the same day). RAW overlay per the
 * standing raw-vs-signals rule: alerts are official warnings displayed
 * as-is with attribution — no predictive claim, no ladder gate needed for
 * display. The trading hypothesis (alert clusters over strategic sites
 * lead sector moves) stays gate-locked in the build order.
 *
 * api.weather.gov: US government work, public domain; wants a contact
 * User-Agent. Alert messages are immutable — updates arrive as NEW alert
 * ids referencing the old — so the archive dedups by id alone.
 *
 * Geometry honesty: many alerts carry geometry:null (zone-coded only).
 * v1 renders polygon-carrying alerts and COUNTS the zone-only rest in the
 * payload (zone_only) — a visible cap, never a silent one. Zone-polygon
 * resolution is a filed follow-up. Archive stores centroid + bbox, not
 * full polygons (space); the live route serves DISPLAY-simplified rings.
 */

import fs from "fs";
import path from "path";
import zlib from "zlib";
import { archiveBaseDir } from "./datacoreArchive";

const ALERTS_URL = "https://api.weather.gov/alerts/active?status=actual&message_type=alert,update";

export interface AlertRec {
  id: string;
  event: string;
  severity: string | null;   // Extreme | Severe | Moderate | Minor | Unknown
  urgency: string | null;
  certainty: string | null;
  sent: string | null;
  onset: string | null;
  ends: string | null;
  area: string | null;       // areaDesc, truncated
  lat: number | null;        // polygon centroid when geometry exists
  lon: number | null;
  bbox: [number, number, number, number] | null;
  rt: string;                // as-seen UTC date
}

export interface AlertDisplay {
  id: string;
  event: string;
  severity: string | null;
  area: string | null;
  ends: string | null;
  /** display-simplified outer rings (<= MAX_RING_PTS points each) */
  rings: Array<Array<[number, number]>>;
}

const AREA_MAX = 200;
const MAX_RING_PTS = 64;

/** Outer rings of a GeoJSON Polygon/MultiPolygon, decimated for display. */
export function simplifyRings(geometry: any, maxPts = MAX_RING_PTS): Array<Array<[number, number]>> {
  const polys: any[] =
    geometry?.type === "Polygon" ? [geometry.coordinates]
    : geometry?.type === "MultiPolygon" ? geometry.coordinates
    : [];
  const rings: Array<Array<[number, number]>> = [];
  for (const poly of polys) {
    const outer: any[] = poly?.[0] || [];
    if (outer.length < 4) continue;
    const step = Math.max(1, Math.ceil(outer.length / maxPts));
    const ring: Array<[number, number]> = [];
    for (let i = 0; i < outer.length; i += step) ring.push([outer[i][0], outer[i][1]]);
    const last = outer[outer.length - 1];
    const head = ring[0];
    if (ring[ring.length - 1][0] !== last[0] || ring[ring.length - 1][1] !== last[1]) ring.push([last[0], last[1]]);
    if (ring[ring.length - 1][0] !== head[0] || ring[ring.length - 1][1] !== head[1]) ring.push([head[0], head[1]]);
    if (ring.length >= 4) rings.push(ring);
  }
  return rings;
}

function centroidAndBbox(rings: Array<Array<[number, number]>>): { lat: number; lon: number; bbox: [number, number, number, number] } | null {
  let sx = 0, sy = 0, n = 0;
  let w = Infinity, s = Infinity, e = -Infinity, nn = -Infinity;
  for (const ring of rings) {
    for (const [x, y] of ring) {
      sx += x; sy += y; n++;
      if (x < w) w = x;
      if (x > e) e = x;
      if (y < s) s = y;
      if (y > nn) nn = y;
    }
  }
  if (!n) return null;
  return { lon: +(sx / n).toFixed(4), lat: +(sy / n).toFixed(4), bbox: [+w.toFixed(4), +s.toFixed(4), +e.toFixed(4), +nn.toFixed(4)] };
}

/** GeoJSON FeatureCollection -> archive records + display alerts. */
export function parseAlerts(json: any, rt: string): { recs: AlertRec[]; display: AlertDisplay[]; zoneOnly: number } {
  const recs: AlertRec[] = [];
  const display: AlertDisplay[] = [];
  let zoneOnly = 0;
  for (const f of json?.features || []) {
    const p = f?.properties || {};
    if (!p.id || !p.event) continue;
    const rings = simplifyRings(f?.geometry);
    const cb = rings.length ? centroidAndBbox(rings) : null;
    if (!cb) zoneOnly++;
    recs.push({
      id: p.id,
      event: p.event,
      severity: p.severity ?? null,
      urgency: p.urgency ?? null,
      certainty: p.certainty ?? null,
      sent: p.sent ?? null,
      onset: p.onset ?? null,
      ends: p.ends ?? null,
      area: typeof p.areaDesc === "string" ? p.areaDesc.slice(0, AREA_MAX) : null,
      lat: cb?.lat ?? null,
      lon: cb?.lon ?? null,
      bbox: cb?.bbox ?? null,
      rt,
    });
    if (cb) {
      display.push({
        id: p.id, event: p.event, severity: p.severity ?? null,
        area: typeof p.areaDesc === "string" ? p.areaDesc.slice(0, AREA_MAX) : null,
        ends: p.ends ?? null, rings,
      });
    }
  }
  return { recs, display, zoneOnly };
}

// ── Fetch ───────────────────────────────────────────────────────────────────

type FetchFn = (url: string, init?: any) => Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

export async function fetchAlerts(fetchImpl: FetchFn = fetch as any, nowMs?: number) {
  const rt = new Date(nowMs ?? Date.now()).toISOString().slice(0, 10);
  const r = await fetchImpl(ALERTS_URL, {
    headers: {
      "User-Agent": "voltradeai-datacore/1.0 (mctils12@gmail.com)",
      "Accept": "application/geo+json",
    },
    signal: AbortSignal.timeout(25000) as any,
  });
  if (!r.ok) throw new Error(`nws alerts -> ${r.status}`);
  return parseAlerts(JSON.parse(await r.text()), rt);
}

// ── Archive (dedup by alert id — messages are immutable) ───────────────────

const archivedIds = new Set<string>();
let seeded = false;

function alertsDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "nwsalerts");
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

export function archiveAlerts(recs: AlertRec[], baseDir?: string, nowMs?: number): number {
  const dir = alertsDir(baseDir);
  const now = nowMs ?? Date.now();
  if (!seeded) { seedSeen(dir, now); seeded = true; }
  const fresh = recs.filter((r) => !archivedIds.has(r.id));
  if (!fresh.length) return 0;
  try {
    fs.mkdirSync(dir, { recursive: true });
    const fp = path.join(dir, `${new Date(now).toISOString().slice(0, 10)}.jsonl`);
    fs.appendFileSync(fp, fresh.map((r) => JSON.stringify(r)).join("\n") + "\n");
    fresh.forEach((r) => archivedIds.add(r.id));
    if (archivedIds.size > 100_000) {
      // oldest-half trim, never clear() (nasaFirms lesson, audit #10b)
      const it = archivedIds.values();
      const drop = archivedIds.size >> 1;
      for (let i = 0; i < drop; i++) {
        const v = it.next();
        if (v.done) break;
        archivedIds.delete(v.value);
      }
    }
    return fresh.length;
  } catch (e: any) {
    console.error("[datacore] nwsalerts archive:", e?.message || e);
    return 0;
  }
}

export function gzipOldAlertDays(baseDir?: string, nowMs?: number): number {
  const dir = alertsDir(baseDir);
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

let cache: { at: number; display: AlertDisplay[]; zoneOnly: number } | null = null;
let polling = false;

export function latestAlerts() {
  return cache;
}

export async function refreshAlertCache(fetchImpl: FetchFn = fetch as any, nowMs?: number): Promise<void> {
  try {
    const { recs, display, zoneOnly } = await fetchAlerts(fetchImpl, nowMs);
    if (recs.length || !cache) cache = { at: Date.now(), display, zoneOnly };
    try { archiveAlerts(recs, undefined, nowMs); } catch {}
    try { gzipOldAlertDays(undefined, nowMs); } catch {}
  } catch (e: any) {
    console.error("[datacore] nwsalerts refresh:", e?.message || e);
  }
}

/** 10-min poll — warnings issue and expire on hour scales; eager boot
 *  (KNOWN BROKEN #9 rule: pollers fire once at startup, never wait). */
export function bootAlertsPoll(intervalMs = 10 * 60_000): void {
  if (polling) return;
  polling = true;
  refreshAlertCache();
  setInterval(() => { refreshAlertCache(); }, intervalMs).unref?.();
}
