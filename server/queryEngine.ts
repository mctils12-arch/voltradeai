/**
 * queryEngine.ts — ANALYST CONSOLE W4: cross-layer geo-temporal query.
 * Given a point + radius + day window + layer set, return everything the
 * platform's OWN ARCHIVES know about that neighborhood, with per-layer
 * provenance and freshness. Pure archive reads — no network at query time,
 * no trading imports (datacore boundary rules). Same fold-from-archives
 * pattern as siteTimeline.ts: one scan pass per layer per request, day
 * files only inside the window, plain .jsonl and .jsonl.gz both readable.
 *
 * Archive layouts mirrored here (read the writers, not guessed):
 *  - positions (datacoreArchive.ts): base/{aircraft,vessels,trains}/
 *    YYYY-MM-DD-HH.jsonl(.gz), lines {t: unix sec, i: id, la, lo, ...}.
 *  - events: base/fires (nasaFirms.ts FireDetection), base/nwsalerts
 *    (nwsAlerts.ts AlertRec), base/usgswater (usgsWater.ts GaugeObs) —
 *    one file per UTC day, YYYY-MM-DD.jsonl(.gz).
 *
 * HONESTY: counts are ARCHIVED-POINT counts under adaptive thinning
 * (near-site traffic is archived at full resolution, oceanic/cruise
 * sparser — see datacoreArchive sampling intervals), so they are traffic
 * LOWER BOUNDS, comparable day-over-day near sites; absent days are
 * absent, never zero-filled; the queryable window is bounded by the
 * RAW_RETENTION_DAYS raw week (older raw is rolled up and not served
 * here); every cap is stated in the payload, never silent.
 */
import fs from "fs";
import path from "path";
import zlib from "zlib";
import readline from "readline";
import { archiveBaseDir, RAW_RETENTION_DAYS } from "./datacoreArchive";
import { parseBbox, filterByViewport } from "./viewport";

// ── layers ───────────────────────────────────────────────────────────────────
// Exactly the archives that exist on disk today (datacoreArchive position
// kinds + the three geo event streams siteTimeline already folds). Do not
// add a name here without a writer that populates its directory.
export const SUPPORTED_LAYERS = ["aircraft", "vessels", "trains", "fires", "alerts", "gauges"] as const;
export type LayerName = (typeof SUPPORTED_LAYERS)[number];

interface LayerSource {
  dir: string;                       // directory under the archive base
  mode: "position" | "event";       // hourly track files vs. daily event files
  provenance: string;                // short source label, mirrors /api/data/* attributions
}

const LAYER_SOURCES: Record<LayerName, LayerSource> = {
  aircraft: { dir: "aircraft", mode: "position", provenance: "own position archive (ADS-B community chain: adsb.lol primary)" },
  vessels:  { dir: "vessels",  mode: "position", provenance: "own position archive (aisstream.io AIS)" },
  trains:   { dir: "trains",   mode: "position", provenance: "own position archive (Digitraffic FI CC BY 4.0 + Entur NO NLOD)" },
  fires:    { dir: "fires",    mode: "event",    provenance: "NASA FIRMS / LANCE archive (VIIRS 375m NRT, not for safety-of-life use)" },
  alerts:   { dir: "nwsalerts", mode: "event",   provenance: "NWS alerts archive (api.weather.gov, public domain)" },
  gauges:   { dir: "usgswater", mode: "event",   provenance: "USGS Water Services archive (public domain)" },
};

// ── caps (all stated in every response — no silent truncation) ──────────────
export const RADIUS_KM_DEFAULT = 50;
export const RADIUS_KM_CAP = 250;
export const DAYS_DEFAULT = 7;
export const DAYS_CAP = RAW_RETENTION_DAYS; // raw week; older raw is rolled up
export const TOP_ENTITIES_CAP = 10;
export const EVENTS_CAP = 50;

// ── API shapes ───────────────────────────────────────────────────────────────
export interface QueryRequest {
  lat: number;
  lon: number;
  radiusKm?: number;   // default RADIUS_KM_DEFAULT, hard cap RADIUS_KM_CAP
  days?: number;       // default DAYS_DEFAULT, hard cap DAYS_CAP
  layers?: string[];   // subset of SUPPORTED_LAYERS; default all
}

export interface LayerResult {
  points: number;                          // archived points in window+radius
  byDay: Record<string, number>;           // day -> count (absent days absent, never zero-filled)
  topEntities?: Array<{ id: string; points: number; firstSeen: string; lastSeen: string }>; // position layers only
  events?: Array<{ t: string; label: string; severity?: string | null; value?: number | null }>; // event layers only
  freshness: string | null;                // newest archived timestamp among MATCHED points (null = nothing matched)
  provenance: string;
}

export interface QueryResult {
  generated_at: string;
  query: { lat: number; lon: number; radiusKm: number; days: number; layers: string[] };
  rejected_layers: string[];               // unknown layer names, filtered — stated, never silent
  layers: Record<string, LayerResult>;
  caps: { radiusKm: number; days: number; topEntities: number; events: number };
  note: string;
}

const HONESTY_NOTE =
  "counts are ARCHIVED-POINT counts under adaptive thinning (near-site traffic archived at full " +
  "resolution, oceanic/cruise sparser) — traffic lower bounds, not raw feed volume; absent days are " +
  "absent, never zero-filled; window bounded by the " + RAW_RETENTION_DAYS + "-day raw retention " +
  "(older raw is rolled up and not queryable here); all caps stated in `caps`; freshness = newest " +
  "archived timestamp among matched points (null when nothing matched)";

// ── shared helpers (same math/reading as siteTimeline.ts) ────────────────────
const kmBetween = (aLat: number, aLon: number, bLat: number, bLon: number) => {
  const R = 6371, dLat = ((bLat - aLat) * Math.PI) / 180, dLon = ((bLon - aLon) * Math.PI) / 180;
  const s = Math.sin(dLat / 2) ** 2 +
    Math.cos((aLat * Math.PI) / 180) * Math.cos((bLat * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
};

function lastDays(nowMs: number, n: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < n; i++) out.push(new Date(nowMs - i * 86400_000).toISOString().slice(0, 10));
  return out;
}

function readJsonlDay(dir: string, day: string): any[] {
  const out: any[] = [];
  for (const fp of [path.join(dir, `${day}.jsonl`), path.join(dir, `${day}.jsonl.gz`)]) {
    let text: string | null = null;
    try {
      text = fp.endsWith(".gz")
        ? zlib.gunzipSync(fs.readFileSync(fp)).toString("utf8")
        : fs.readFileSync(fp, "utf8");
    } catch { continue; }
    for (const line of text.split("\n")) {
      if (!line) continue;
      try { out.push(JSON.parse(line)); } catch {}
    }
  }
  return out;
}

// ── request normalization ────────────────────────────────────────────────────
interface NormalizedQuery {
  lat: number; lon: number; radiusKm: number; days: number;
  layers: LayerName[]; rejected: string[];
}

export function normalizeQuery(req: QueryRequest): NormalizedQuery {
  const { lat, lon } = req;
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
    throw new Error("invalid lat/lon");
  }
  const radiusKm = Math.min(RADIUS_KM_CAP, Math.max(1,
    Number.isFinite(req.radiusKm as number) ? (req.radiusKm as number) : RADIUS_KM_DEFAULT));
  const days = Math.min(DAYS_CAP, Math.max(1,
    Number.isFinite(req.days as number) ? Math.floor(req.days as number) : DAYS_DEFAULT));
  const requested = req.layers && req.layers.length ? req.layers : [...SUPPORTED_LAYERS];
  const layers: LayerName[] = [];
  const rejected: string[] = [];
  for (const l of requested) {
    if ((SUPPORTED_LAYERS as readonly string[]).includes(l)) {
      if (!layers.includes(l as LayerName)) layers.push(l as LayerName);
    } else if (!rejected.includes(l)) rejected.push(l);
  }
  return { lat, lon, radiusKm, days, layers, rejected };
}

// ── position layers: stream hourly track files inside the window ────────────
function scanPositionLayer(dir: string, provenance: string, q: NormalizedQuery,
                           daySet: Set<string>): Promise<LayerResult> {
  const byDay: Record<string, number> = {};
  const entities = new Map<string, { points: number; t0: number; t1: number }>();
  let points = 0, maxT = -Infinity;
  let files: string[] = [];
  try {
    files = fs.readdirSync(dir).filter((f) =>
      (f.endsWith(".jsonl") || f.endsWith(".jsonl.gz")) && daySet.has(f.slice(0, 10)));
  } catch { /* absent archive dir -> zero result, never a throw */ }
  const chain = files.sort().map((f) => () => new Promise<void>((resolve) => {
    try {
      let stream: NodeJS.ReadableStream = fs.createReadStream(path.join(dir, f));
      if (f.endsWith(".gz")) stream = stream.pipe(zlib.createGunzip());
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      rl.on("line", (l) => {
        if (!l.trim()) return;
        let r: any;
        try { r = JSON.parse(l); } catch { return; }
        if (r?.la == null || r?.lo == null || typeof r.t !== "number") return;
        const day = new Date(r.t * 1000).toISOString().slice(0, 10);
        if (!daySet.has(day)) return; // record-level window guard, not just file names
        if (kmBetween(r.la, r.lo, q.lat, q.lon) > q.radiusKm) return;
        points++;
        byDay[day] = (byDay[day] || 0) + 1;
        maxT = Math.max(maxT, r.t);
        const id = String(r.i ?? "unknown");
        const e = entities.get(id);
        if (e) { e.points++; e.t0 = Math.min(e.t0, r.t); e.t1 = Math.max(e.t1, r.t); }
        else entities.set(id, { points: 1, t0: r.t, t1: r.t });
      });
      rl.on("close", () => resolve());
      stream.on("error", () => resolve());
    } catch { resolve(); }
  }));
  return chain.reduce((p, next) => p.then(next), Promise.resolve()).then(() => {
    const topEntities = Array.from(entities.entries())
      .sort((a, b) => b[1].points - a[1].points)
      .slice(0, TOP_ENTITIES_CAP)
      .map(([id, e]) => ({
        id, points: e.points,
        firstSeen: new Date(e.t0 * 1000).toISOString(),
        lastSeen: new Date(e.t1 * 1000).toISOString(),
      }));
    return {
      points, byDay, topEntities,
      freshness: Number.isFinite(maxT) ? new Date(maxT * 1000).toISOString() : null,
      provenance,
    };
  });
}

// ── event layers: fold daily event files inside the window ──────────────────
function eventFromRecord(layer: LayerName, rec: any, day: string):
    { t: string; label: string; severity?: string | null; value?: number | null } | null {
  if (rec?.lat == null || rec?.lon == null) return null; // zone-only alerts etc. — no geo, excluded per stream honesty
  if (layer === "alerts") {
    return { t: rec.sent || day, label: String(rec.event ?? "Alert"), severity: rec.severity ?? null };
  }
  if (layer === "fires") {
    const t = rec.acq_date && typeof rec.acq_time === "string" && rec.acq_time.length === 4
      ? `${rec.acq_date}T${rec.acq_time.slice(0, 2)}:${rec.acq_time.slice(2)}:00Z`
      : rec.acq_date || day;
    return { t, label: `Fire detection (${rec.confidence || "unknown"} confidence)`, value: rec.frp ?? null };
  }
  // gauges (GaugeObs)
  return { t: rec.d || day, label: String(rec.name || rec.site || "Gauge"), value: rec.v ?? null };
}

function scanEventLayer(layer: LayerName, dir: string, provenance: string,
                        q: NormalizedQuery, days: string[]): LayerResult {
  const byDay: Record<string, number> = {};
  const events: Array<{ t: string; label: string; severity?: string | null; value?: number | null }> = [];
  let points = 0, freshMs = -Infinity, freshness: string | null = null;
  for (const day of days) {
    for (const rec of readJsonlDay(dir, day)) {
      const ev = eventFromRecord(layer, rec, day);
      if (!ev) continue;
      if (kmBetween(rec.lat, rec.lon, q.lat, q.lon) > q.radiusKm) continue;
      points++;
      byDay[day] = (byDay[day] || 0) + 1; // archive day = when we recorded it
      events.push(ev);
      const ms = Date.parse(ev.t);
      if (Number.isFinite(ms) ? ms > freshMs : freshness === null) {
        if (Number.isFinite(ms)) freshMs = ms;
        freshness = ev.t;
      }
    }
  }
  events.sort((a, b) => (a.t < b.t ? 1 : a.t > b.t ? -1 : 0));
  return { points, byDay, events: events.slice(0, EVENTS_CAP), freshness, provenance };
}

// ── the query ────────────────────────────────────────────────────────────────
export async function queryWindow(req: QueryRequest, base = archiveBaseDir(),
                                  nowMs = Date.now()): Promise<QueryResult> {
  const q = normalizeQuery(req);
  const days = lastDays(nowMs, q.days);
  const daySet = new Set(days);
  const layers: Record<string, LayerResult> = {};
  for (const layer of q.layers) {
    const src = LAYER_SOURCES[layer];
    layers[layer] = src.mode === "position"
      ? await scanPositionLayer(path.join(base, src.dir), src.provenance, q, daySet)
      : scanEventLayer(layer, path.join(base, src.dir), src.provenance, q, days);
  }
  return {
    generated_at: new Date(nowMs).toISOString(),
    query: { lat: q.lat, lon: q.lon, radiusKm: q.radiusKm, days: q.days, layers: q.layers },
    rejected_layers: q.rejected,
    layers,
    caps: { radiusKm: RADIUS_KM_CAP, days: DAYS_CAP, topEntities: TOP_ENTITIES_CAP, events: EVENTS_CAP },
    note: HONESTY_NOTE,
  };
}

// ── cached wrapper (LRU, TTL 5 min, max 50 entries) ─────────────────────────
const CACHE_TTL_MS = 5 * 60_000;
const CACHE_MAX = 50;
const cache = new Map<string, { at: number; result: QueryResult }>();

function cacheKey(q: ReturnType<typeof normalizeQuery>): string {
  // coords rounded to 0.05 deg — nearby repeat queries share a scan
  const la = (Math.round(q.lat / 0.05) * 0.05).toFixed(2);
  const lo = (Math.round(q.lon / 0.05) * 0.05).toFixed(2);
  return `${[...q.layers].sort().join(",")}|${la}|${lo}|${q.radiusKm}|${q.days}`;
}

export async function queryWindowCached(req: QueryRequest, base = archiveBaseDir(),
                                        nowMs = Date.now()): Promise<QueryResult> {
  const key = cacheKey(normalizeQuery(req)); // validates before touching the cache
  const hit = cache.get(key);
  if (hit && nowMs - hit.at <= CACHE_TTL_MS) {
    cache.delete(key); cache.set(key, hit); // LRU bump
    return hit.result;
  }
  const result = await queryWindow(req, base, nowMs);
  cache.set(key, { at: nowMs, result });
  while (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value as string);
  return result;
}

export function _resetQueryCache() { cache.clear(); }

// ── W3 TIME SCRUBBER: one archived hour/day bucket, replayed ────────────────
// A snapshot is a single point-in-time read of ONE layer's archive — the
// building block for "scrub a slider, watch the world move" (console_charter
// W3). Position layers (aircraft/vessels/trains) bucket by HOUR (their write
// grain, readJsonlDay reused with an hour stamp); event layers (fires/alerts/
// gauges) bucket by DAY (their write grain) — an `at` timestamp is floored to
// whichever grain the layer uses. Bounded to the same RAW_RETENTION_DAYS raw
// window queryWindow uses (older raw is rolled up to lossy per-entity
// summaries and not point-replayable here) — outside the window is a stated
// error, never a silent empty result.
export const SNAPSHOT_POINTS_CAP = 3000;

export interface SnapshotPoint {
  id: string | null;
  lat: number;
  lon: number;
  label?: string;
  severity?: string | null;
  value?: number | null;
}

export interface SnapshotResult {
  layer: LayerName;
  mode: "position" | "event";
  requested_at: string;   // the `at` the caller passed, verbatim
  bucket_at: string;      // the resolved hour/day bucket start, ISO
  data: SnapshotPoint[];
  count: number;
  count_before_viewport: number;
  count_dropped_offscreen: number;
  viewport_filtered: boolean;
  capped: boolean;
  freshness: string | null;
  provenance: string;
  window: { min_iso: string; max_iso: string; days: number };
  note: string;
}

const SNAPSHOT_NOTE =
  "one archived hour (position layers) or day (event layers) bucket, replayed exactly as recorded — " +
  "not live; freshness = newest archived timestamp among matched points; bbox optional (SCALE S1 viewport " +
  "pattern), point cap stated when hit, never silent; bounded to the raw retention window (older raw is " +
  "rolled up to lossy per-entity summaries and not point-replayable here)";

export function snapshotWindow(nowMs = Date.now()): { minMs: number; maxMs: number } {
  return { minMs: nowMs - RAW_RETENTION_DAYS * 86400_000, maxMs: nowMs };
}

export function querySnapshot(layerName: string, atIso: string, bboxStr: unknown,
                              base = archiveBaseDir(), nowMs = Date.now()): SnapshotResult {
  if (!(SUPPORTED_LAYERS as readonly string[]).includes(layerName)) {
    throw new Error(`unknown layer "${layerName}" — supported: ${SUPPORTED_LAYERS.join(", ")}`);
  }
  const layer = layerName as LayerName;
  const atMs = Date.parse(atIso);
  if (!Number.isFinite(atMs)) throw new Error(`invalid "at" timestamp: "${atIso}"`);
  const { minMs, maxMs } = snapshotWindow(nowMs);
  if (atMs < minMs || atMs > maxMs) {
    throw new Error(`"at" outside the retained raw window (${new Date(minMs).toISOString()} .. ${new Date(maxMs).toISOString()})`);
  }
  const src = LAYER_SOURCES[layer];
  const dir = path.join(base, src.dir);
  const bbox = parseBbox(bboxStr);
  const d = new Date(atMs);
  let bucketAt: string;
  let raw: any[];
  if (src.mode === "position") {
    bucketAt = d.toISOString().slice(0, 13) + ":00:00.000Z";
    const stamp = d.toISOString().slice(0, 13).replace("T", "-"); // YYYY-MM-DD-HH
    raw = readJsonlDay(dir, stamp);
  } else {
    const day = d.toISOString().slice(0, 10);
    bucketAt = day + "T00:00:00.000Z";
    raw = readJsonlDay(dir, day);
  }

  let points: SnapshotPoint[] = [];
  let maxT = -Infinity;
  if (src.mode === "position") {
    for (const r of raw) {
      if (r?.la == null || r?.lo == null || typeof r.t !== "number") continue;
      points.push({ id: r.i != null ? String(r.i) : null, lat: r.la, lon: r.lo });
      maxT = Math.max(maxT, r.t * 1000);
    }
  } else {
    const day = bucketAt.slice(0, 10);
    for (const r of raw) {
      const ev = eventFromRecord(layer, r, day);
      if (!ev) continue; // no geometry (e.g. zone-only alerts) — excluded, same as queryWindow
      points.push({ id: null, lat: r.lat, lon: r.lon, label: ev.label, severity: ev.severity ?? null, value: ev.value ?? null });
      const ms = Date.parse(ev.t);
      if (Number.isFinite(ms)) maxT = Math.max(maxT, ms);
    }
  }

  const countBefore = points.length;
  let viewportFiltered = false, droppedOffscreen = 0;
  if (bbox) {
    const inView = filterByViewport(points, bbox, (p) => [p.lon, p.lat]);
    droppedOffscreen = countBefore - inView.length;
    points = inView;
    viewportFiltered = true;
  }
  const capped = points.length > SNAPSHOT_POINTS_CAP;
  if (capped) points = points.slice(0, SNAPSHOT_POINTS_CAP);

  return {
    layer, mode: src.mode, requested_at: atIso, bucket_at: bucketAt,
    data: points, count: points.length,
    count_before_viewport: countBefore, count_dropped_offscreen: droppedOffscreen,
    viewport_filtered: viewportFiltered, capped,
    freshness: Number.isFinite(maxT) ? new Date(maxT).toISOString() : null,
    provenance: src.provenance,
    window: { min_iso: new Date(minMs).toISOString(), max_iso: new Date(maxMs).toISOString(), days: RAW_RETENTION_DAYS },
    note: SNAPSHOT_NOTE,
  };
}
