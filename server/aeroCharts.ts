// aeroCharts.ts — FAA aeronautical chart base views for the /data map
// (VFR Sectional, VFR Terminal Area, IFR Low, IFR Enroute High), served
// from OUR origin at /tiles/aero/:chart/:z/:x/:y.
//
// SOURCE: the FAA Aeronautical Information Services ArcGIS tile caches
// (tiles.arcgis.com/.../ssFJjBXIUyZDrSYZ/.../{service}/MapServer). FAA
// charts are US-government works — public domain. 256px Web Mercator
// (wkid 3857) tiles, format MIXED: JPEG where the chart is solid, a
// transparent PNG at coverage edges / over empty cells, 404 outside the
// chart's LOD band or coverage. Verified 2026-09-30:
//   VFR_Sectional  LOD 8-12  (CONUS + AK + HI + PR)
//   VFR_Terminal   LOD 10-12 (only around TAC-covered Class B airports)
//   IFR_AreaLow    LOD 7-12  — NOT area charts only: z7-11 over Kansas (no
//                  area chart there) carries Victor airways + MOAs, i.e. the
//                  ENROUTE LOW series; z12 is populated only inside Area
//                  chart footprints (DFW yes, Kansas empty). Named honestly
//                  as "IFR Low (Enroute + Area)".
//   IFR_High       LOD 5-9   (jet routes, Q-routes, >= FL180)
//
// RENDERING & MOTION LAW II.8 ("runtime never touches an upstream WMTS") —
// THE COMPROMISE, stated plainly: the browser never talks to FAA/ArcGIS;
// every tile comes from our origin. But there is no offline bake yet, so the
// bake happens LAZILY: the first request for a tile (per chart edition) is a
// read-through fetch upstream, after which the tile lives in our cache —
// R2 (aero/<chart>/<edition>/<z>/<y>/<x>) when R2 is configured, else a
// bounded LRU under os.tmpdir() (NEVER the /data volume, which is nearly
// full). With R2 configured a background prefetch bakes CONUS z<=9 for each
// new edition at a bounded request rate, so the common views are warm
// before anyone asks. One upstream request per tile per edition, shared by
// every visitor (in-flight dedup) — never per-visitor fan-out. R2 writes are
// capped per UTC day (AERO_R2_MAX_PUTS_PER_DAY) so a crawler cannot turn the
// pyramid into millions of Class-A ops; past the cap tiles still serve and
// overflow to the tmp LRU.
//
// EDITION AWARENESS: the service metadata's documentInfo.subject carries
// "Updated with the latest charts on MM-DD-YYYY" — that date is the chart
// cycle the tiles were cut from, and it is part of every cache key, so a
// new cycle is a new key space (old keys simply stop being read). The FAA
// publishes VFR and IFR enroute charts on a 56-day cycle; effective/expiry
// are derived from that schedule, and the status says so. If the service
// lags the FAA's current cycle (it does: on 2026-09-30 the service reports
// 07-09 while the FAA cycle began 09-03) the status and the on-map badge say
// "behind current FAA cycle" rather than implying currency. NOT FOR
// NAVIGATION — surfaced on every response that reaches a person.

import fs from "fs";
import os from "os";
import path from "path";
import type { Express, Request, Response } from "express";
import { createR2Client, errText, r2ConfigDiagnostics, r2ConfigFromEnv, type R2Client } from "./r2Client";

// ── chart catalogue ─────────────────────────────────────────────────────────

export const AERO_UPSTREAM_BASE = "https://tiles.arcgis.com/tiles/ssFJjBXIUyZDrSYZ/arcgis/rest/services";

export interface AeroChartDef {
  id: AeroChartId;
  /** ArcGIS service name */
  service: string;
  label: string;
  short: string;
  /** verified LOD band — refreshed from ?f=json minLOD/maxLOD at runtime */
  minzoom: number;
  maxzoom: number;
  coverage: string;
}

export const AERO_CHART_IDS = ["sectional", "tac", "ifrlow", "ifrhigh"] as const;
export type AeroChartId = (typeof AERO_CHART_IDS)[number];

export const AERO_CHARTS: Record<AeroChartId, AeroChartDef> = {
  sectional: {
    id: "sectional", service: "VFR_Sectional", label: "VFR Sectional", short: "Sectional",
    minzoom: 8, maxzoom: 12,
    coverage: "US sectional series (CONUS, Alaska, Hawaii, Puerto Rico); zoom 8-12",
  },
  tac: {
    id: "tac", service: "VFR_Terminal", label: "VFR Terminal Area (TAC)", short: "Terminal",
    minzoom: 10, maxzoom: 12,
    coverage: "only around TAC-charted Class B/C terminal areas; zoom 10-12",
  },
  ifrlow: {
    id: "ifrlow", service: "IFR_AreaLow", label: "IFR Low (Enroute + Area)", short: "IFR Low",
    minzoom: 7, maxzoom: 12,
    coverage: "IFR Enroute Low series (Victor airways, MOAs) to zoom 11, plus Area charts to zoom 12 where published",
  },
  ifrhigh: {
    id: "ifrhigh", service: "IFR_High", label: "IFR Enroute High", short: "IFR High",
    minzoom: 5, maxzoom: 9,
    coverage: "IFR Enroute High series (jet routes, Q-routes, FL180+); zoom 5-9",
  },
};

export function isAeroChartId(s: string): s is AeroChartId {
  return (AERO_CHART_IDS as readonly string[]).includes(s);
}

// ── pure: URLs, keys, tile-range validation ─────────────────────────────────

/** ArcGIS cache path order is {z}/{y}/{x} (row before column). */
export function aeroUpstreamTileUrl(chart: AeroChartId, z: number, x: number, y: number): string {
  return `${AERO_UPSTREAM_BASE}/${AERO_CHARTS[chart].service}/MapServer/tile/${z}/${y}/${x}`;
}

export function aeroMetadataUrl(chart: AeroChartId): string {
  return `${AERO_UPSTREAM_BASE}/${AERO_CHARTS[chart].service}/MapServer?f=json`;
}

/** Cache key (R2 object key and tmp-LRU key). No extension: the upstream is
 *  MIXED JPEG/PNG and the stored bytes are sniffed on read. The edition
 *  segment is what makes a new chart cycle a fresh key space. */
export function aeroCacheKey(chart: AeroChartId, edition: string, z: number, x: number, y: number): string {
  return `aero/${chart}/${edition}/${z}/${y}/${x}`;
}

/** Our origin's tile URL template for the client (edition-pinned so the
 *  browser may cache it immutably). */
export function aeroClientTileTemplate(chart: AeroChartId, edition: string | null): string {
  return `/tiles/aero/${chart}/{z}/{x}/{y}${edition ? `?e=${encodeURIComponent(edition)}` : ""}`;
}

/** Integer z/x/y inside the Web Mercator pyramid. */
export function validTile(z: number, x: number, y: number): boolean {
  if (![z, x, y].every((n) => Number.isInteger(n) && n >= 0)) return false;
  if (z > 22) return false;
  const n = 2 ** z;
  return x < n && y < n;
}

export type TileImageType = "image/jpeg" | "image/png" | null;

export function sniffImageType(b: Uint8Array): TileImageType {
  if (b.length >= 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return "image/jpeg";
  if (b.length >= 8 && b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) return "image/png";
  return null;
}

/** 1x1 fully transparent PNG — what "no chart here" is served as, so the
 *  map shows the satellite beneath instead of an error tile or a hole. */
export const AERO_EMPTY_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "base64",
);

// ── pure: editions + the FAA 56-day cycle ───────────────────────────────────

export const FAA_CYCLE_DAYS = 56;
/** A published 56-day chart effective date (VFR + IFR enroute moved to the
 *  common 56-day cycle); every cycle is this +/- k*56 days. Cross-checked
 *  against the service's own reported editions 2026-05-14 and 2026-07-09. */
export const FAA_CYCLE_ANCHOR = "2026-01-22";
const DAY_MS = 86_400_000;

function isoDay(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

/** "Updated with the latest charts on 07-09-2026." -> "2026-07-09". Accepts
 *  MM-DD-YYYY or MM/DD/YYYY; null when the text carries no such date. */
export function parseEditionFromSubject(subject: string | null | undefined): string | null {
  if (!subject) return null;
  const m = /(\d{1,2})[-/](\d{1,2})[-/](\d{4})/.exec(subject);
  if (!m) return null;
  const mm = Number(m[1]), dd = Number(m[2]), yyyy = Number(m[3]);
  if (mm < 1 || mm > 12 || dd < 1 || dd > 31) return null;
  const d = new Date(Date.UTC(yyyy, mm - 1, dd));
  if (d.getUTCMonth() !== mm - 1) return null; // 02-31 etc.
  return isoDay(d.getTime());
}

/** Start date of the FAA 56-day cycle containing `nowMs`. */
export function faaCycleStart(nowMs: number): string {
  const anchor = Date.parse(`${FAA_CYCLE_ANCHOR}T00:00:00Z`);
  const k = Math.floor((nowMs - anchor) / (FAA_CYCLE_DAYS * DAY_MS));
  return isoDay(anchor + k * FAA_CYCLE_DAYS * DAY_MS);
}

export function addDays(isoDate: string, days: number): string {
  return isoDay(Date.parse(`${isoDate}T00:00:00Z`) + days * DAY_MS);
}

export interface EditionInfo {
  /** the cache-key segment */
  edition: string;
  /** "service-metadata" = parsed from the FAA tile service; "unverified" =
   *  metadata unreachable, keyed by the FAA cycle date but NOT claimed as
   *  the tiles' edition */
  source: "service-metadata" | "unverified";
  effective: string | null;
  expires: string | null;
  expired: boolean;
  currentFaaCycle: string;
  behindCurrentCycle: boolean;
}

export function editionInfo(serviceEdition: string | null, nowMs: number): EditionInfo {
  const cur = faaCycleStart(nowMs);
  if (!serviceEdition) {
    return {
      edition: `unverified-${cur}`, source: "unverified", effective: null, expires: null,
      expired: false, currentFaaCycle: cur, behindCurrentCycle: false,
    };
  }
  const expires = addDays(serviceEdition, FAA_CYCLE_DAYS);
  return {
    edition: serviceEdition, source: "service-metadata", effective: serviceEdition, expires,
    expired: isoDay(nowMs) >= expires, currentFaaCycle: cur, behindCurrentCycle: serviceEdition < cur,
  };
}

/** Browser cache policy: a request pinned to the CURRENT edition is
 *  immutable (the URL changes when the edition does); anything else gets a
 *  short life so a stale-pinned tab re-asks soon. */
export function aeroCacheControl(requestedEdition: string | undefined, currentEdition: string): string {
  return requestedEdition && requestedEdition === currentEdition
    ? "public, max-age=31536000, immutable"
    : "public, max-age=3600";
}

// ── pure: prefetch tile enumeration ─────────────────────────────────────────

/** Contiguous US bounding box for the background bake. Deliberately coarse
 *  (whole degrees); tiles outside coverage just cache as empty. */
export const AERO_CONUS_BBOX = { west: -125, south: 24, east: -66, north: 50 } as const;
export const AERO_PREFETCH_MAX_ZOOM = 9;

export function lonToTileX(lon: number, z: number): number {
  return Math.floor(((lon + 180) / 360) * 2 ** z);
}
export function latToTileY(lat: number, z: number): number {
  const r = (lat * Math.PI) / 180;
  return Math.floor(((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * 2 ** z);
}

export interface TileXYZ { z: number; x: number; y: number }

/** Every tile of `chart` over the CONUS box from its minzoom to
 *  min(maxzoom, AERO_PREFETCH_MAX_ZOOM), coarse levels first. */
export function prefetchTiles(def: Pick<AeroChartDef, "minzoom" | "maxzoom">, bbox = AERO_CONUS_BBOX,
  maxZoom = AERO_PREFETCH_MAX_ZOOM): TileXYZ[] {
  const out: TileXYZ[] = [];
  for (let z = def.minzoom; z <= Math.min(def.maxzoom, maxZoom); z++) {
    const x0 = lonToTileX(bbox.west, z), x1 = lonToTileX(bbox.east, z);
    const y0 = latToTileY(bbox.north, z), y1 = latToTileY(bbox.south, z);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) out.push({ z, x, y });
  }
  return out;
}

// ── bounded tmp LRU (the no-R2 cache) ───────────────────────────────────────

export const AERO_TMP_DEFAULT_MAX_BYTES = 256 * 1024 * 1024;

export interface TmpLru {
  get(key: string): Buffer | null;
  put(key: string, body: Buffer): void;
  stats(): { entries: number; bytes: number; maxBytes: number; evictions: number; errors: number };
}

/** Every entry is charged at least this much against the byte cap, so the
 *  0-byte "empty" markers are bounded in COUNT too (an inode is not free). */
export const AERO_TMP_ENTRY_FLOOR_BYTES = 512;

/** Disk LRU under `dir`, size-capped. The index is in memory (Map insertion
 *  order = recency) and the directory is wiped at construction, so index
 *  and disk can never disagree across restarts. A 0-byte file is a cached
 *  "no tile here". */
export function createTmpLru(dir: string, maxBytes: number): TmpLru {
  const index = new Map<string, number>();
  let bytes = 0, evictions = 0, errors = 0;
  const fileOf = (key: string) => path.join(dir, key.replace(/\//g, "_"));
  const noteErr = (where: string, e: unknown) => {
    errors++;
    if (errors === 1) console.warn(`[aero] tmp cache ${where} failed (further failures counted in /api/data/aero/status): ${errText(e)}`);
  };
  try {
    fs.rmSync(dir, { recursive: true, force: true });
    fs.mkdirSync(dir, { recursive: true });
  } catch (e: unknown) {
    noteErr("init", e);
  }
  function drop(key: string): void {
    const sz = index.get(key);
    if (sz === undefined) return;
    index.delete(key);
    bytes -= sz;
    try { fs.rmSync(fileOf(key), { force: true }); } catch (e: unknown) { noteErr("evict", e); }
  }
  return {
    get(key) {
      const sz = index.get(key);
      if (sz === undefined) return null;
      try {
        const b = fs.readFileSync(fileOf(key));
        index.delete(key); index.set(key, sz); // touch
        return b;
      } catch (e: unknown) {
        noteErr("read", e);
        index.delete(key); bytes -= sz;
        return null;
      }
    },
    put(key, body) {
      if (body.length > maxBytes) return;
      drop(key);
      try {
        fs.writeFileSync(fileOf(key), body);
      } catch (e: unknown) {
        noteErr("write", e);
        return;
      }
      const charged = Math.max(body.length, AERO_TMP_ENTRY_FLOOR_BYTES);
      index.set(key, charged);
      bytes += charged;
      while (bytes > maxBytes && index.size) {
        const oldest = index.keys().next().value as string;
        drop(oldest);
        evictions++;
      }
    },
    stats: () => ({ entries: index.size, bytes, maxBytes, evictions, errors }),
  };
}

// ── service ─────────────────────────────────────────────────────────────────

export interface AeroDeps {
  fetchImpl?: typeof fetch;
  now?: () => number;
  env?: NodeJS.ProcessEnv;
  r2?: R2Client;
  tmpDir?: string;
  sleep?: (ms: number) => Promise<void>;
}

interface ChartState {
  minzoom: number;
  maxzoom: number;
  serviceEdition: string | null;
  metadataFetchedAt: number | null;
  metadataError: string | null;
  hits: number;
  misses: number;
}

export type TileOutcome =
  | { kind: "tile"; body: Buffer; contentType: "image/jpeg" | "image/png"; from: "r2" | "tmp" | "upstream" }
  | { kind: "empty"; from: "r2" | "tmp" | "upstream" | "out-of-range" | "negative" }
  | { kind: "error"; error: string };

export const AERO_METADATA_TTL_MS = 6 * 3600_000;
export const AERO_METADATA_RETRY_MS = 5 * 60_000;
export const AERO_UPSTREAM_TIMEOUT_MS = 10_000;
export const AERO_MAX_UPSTREAM_INFLIGHT = 6;
export const AERO_PREFETCH_DEFAULT_RPS = 2;
/** Class-A (PUT) ops per UTC day. One edition's CONUS z<=9 bake is ~16k
 *  tiles across the charts, so a bake day still leaves organic headroom;
 *  25k/day saturated every day would be ~750k/month, inside R2's 1M free. */
export const AERO_R2_DEFAULT_MAX_PUTS_PER_DAY = 25_000;
const USER_AGENT = "VolTradeAI-datamap/1.0 (+aero chart cache; one fetch per tile per edition)";

export function createAeroChartService(deps: AeroDeps = {}) {
  const fetchImpl = deps.fetchImpl ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  const now = deps.now ?? (() => Date.now());
  const env = deps.env ?? process.env;
  const sleep = deps.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  // Tile-path R2 client: short timeout, no retries — a slow R2 must fall
  // through to upstream, never stall a map tile for 90s.
  const r2 = deps.r2 ?? createR2Client(r2ConfigFromEnv(env), { timeoutMs: 5000, maxRetries: 0 });
  const tmpMax = Number(env.AERO_TILE_CACHE_MAX_BYTES) > 0 ? Number(env.AERO_TILE_CACHE_MAX_BYTES) : AERO_TMP_DEFAULT_MAX_BYTES;
  let tmp: TmpLru | null = null;
  const tmpCache = () => (tmp ||= createTmpLru(deps.tmpDir ?? path.join(os.tmpdir(), "voltrade-aero-tiles"), tmpMax));

  const charts = Object.fromEntries(AERO_CHART_IDS.map((id) => [id, {
    minzoom: AERO_CHARTS[id].minzoom, maxzoom: AERO_CHARTS[id].maxzoom,
    serviceEdition: null, metadataFetchedAt: null, metadataError: null, hits: 0, misses: 0,
  } as ChartState])) as Record<AeroChartId, ChartState>;

  const counters = {
    r2Hits: 0, tmpHits: 0, upstreamFetches: 0, upstream404: 0, upstreamErrors: 0,
    r2Errors: 0, r2Puts: 0, r2PutErrors: 0, dedupJoins: 0,
    lastUpstreamError: null as string | null, lastUpstreamErrorAt: null as number | null,
  };
  const inflight = new Map<string, Promise<TileOutcome>>();
  let upstreamActive = 0;
  const upstreamWaiters: Array<() => void> = [];
  const metaInflight = new Map<AeroChartId, Promise<void>>();

  async function withUpstreamSlot<T>(fn: () => Promise<T>): Promise<T> {
    while (upstreamActive >= AERO_MAX_UPSTREAM_INFLIGHT) await new Promise<void>((r) => upstreamWaiters.push(r));
    upstreamActive++;
    try {
      return await fn();
    } finally {
      upstreamActive--;
      upstreamWaiters.shift()?.();
    }
  }

  async function timedFetch(url: string): Promise<globalThis.Response> {
    const ac = new AbortController();
    const t = setTimeout(() => ac.abort(), AERO_UPSTREAM_TIMEOUT_MS);
    try {
      return await fetchImpl(url, { signal: ac.signal, headers: { "user-agent": USER_AGENT } });
    } finally {
      clearTimeout(t);
    }
  }

  async function refreshMetadata(chart: AeroChartId): Promise<void> {
    const st = charts[chart];
    try {
      const res = await timedFetch(aeroMetadataUrl(chart));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = (await res.json()) as {
        minLOD?: number; maxLOD?: number; documentInfo?: { Subject?: string; subject?: string };
      };
      const subject = j.documentInfo?.Subject ?? j.documentInfo?.subject ?? null;
      const ed = parseEditionFromSubject(subject);
      if (ed) st.serviceEdition = ed;
      if (Number.isInteger(j.minLOD) && Number.isInteger(j.maxLOD) && (j.minLOD as number) <= (j.maxLOD as number)) {
        st.minzoom = j.minLOD as number;
        st.maxzoom = j.maxLOD as number;
      }
      st.metadataError = ed ? null : "service metadata carried no edition date";
    } catch (e: unknown) {
      st.metadataError = errText(e);
    }
    st.metadataFetchedAt = now();
  }

  /** Current edition for a chart; refreshes metadata when stale (deduped). */
  async function edition(chart: AeroChartId): Promise<EditionInfo> {
    const st = charts[chart];
    // a failed/edition-less fetch retries in minutes, a good one every 6h
    const ttl = st.serviceEdition ? AERO_METADATA_TTL_MS : AERO_METADATA_RETRY_MS;
    const stale = st.metadataFetchedAt === null || now() - st.metadataFetchedAt > ttl;
    if (stale) {
      let p = metaInflight.get(chart);
      if (!p) {
        p = refreshMetadata(chart).finally(() => metaInflight.delete(chart));
        metaInflight.set(chart, p);
      }
      // First-ever lookup waits (there is nothing to key by yet); a refresh
      // of a known edition happens in the background.
      if (st.metadataFetchedAt === null) await p;
    }
    return editionInfo(st.serviceEdition, now());
  }

  async function fetchUpstream(chart: AeroChartId, z: number, x: number, y: number): Promise<TileOutcome> {
    counters.upstreamFetches++;
    try {
      const res = await withUpstreamSlot(() => timedFetch(aeroUpstreamTileUrl(chart, z, x, y)));
      if (res.status === 404) {
        counters.upstream404++;
        await res.arrayBuffer().catch((e: unknown) => { counters.lastUpstreamError = `drain: ${errText(e)}`; });
        return { kind: "empty", from: "upstream" };
      }
      if (!res.ok) throw new Error(`upstream HTTP ${res.status}`);
      const body = Buffer.from(await res.arrayBuffer());
      const type = sniffImageType(body);
      if (!type) throw new Error(`upstream returned a non-image body (${body.length} bytes)`);
      return { kind: "tile", body, contentType: type, from: "upstream" };
    } catch (e: unknown) {
      counters.upstreamErrors++;
      counters.lastUpstreamError = errText(e);
      counters.lastUpstreamErrorAt = now();
      return { kind: "error", error: errText(e) };
    }
  }

  // R2 WRITE BUDGET: a crawler enumerating the pyramid must not turn into
  // millions of Class-A PUTs (or a multi-GB z12 mirror). Past the daily cap
  // tiles are still served — they just land in the bounded tmp LRU instead.
  // "No tile here" markers only persist to R2 inside the prefetch band.
  const r2PutBudget = Number(env.AERO_R2_MAX_PUTS_PER_DAY) > 0
    ? Math.floor(Number(env.AERO_R2_MAX_PUTS_PER_DAY)) : AERO_R2_DEFAULT_MAX_PUTS_PER_DAY;
  const budget = { day: "", used: 0 };
  function r2BudgetLeft(): number {
    const day = new Date(now()).toISOString().slice(0, 10);
    if (budget.day !== day) { budget.day = day; budget.used = 0; }
    return r2PutBudget - budget.used;
  }

  function storeOutcome(key: string, z: number, o: TileOutcome): void {
    if (o.kind !== "tile" && !(o.kind === "empty" && o.from === "upstream")) return;
    const body = o.kind === "tile" ? o.body : Buffer.alloc(0);
    const persistEmpty = o.kind === "tile" || z <= AERO_PREFETCH_MAX_ZOOM;
    if (r2.configured && persistEmpty && r2BudgetLeft() > 0) {
      budget.used++;
      void r2.putObject(key, body, o.kind === "tile" ? o.contentType : "application/octet-stream").then((r) => {
        if (r.ok) counters.r2Puts++;
        else counters.r2PutErrors++;
      });
    } else {
      tmpCache().put(key, body);
    }
  }

  function fromCachedBytes(b: Buffer, from: "r2" | "tmp"): TileOutcome {
    if (b.length === 0) return { kind: "empty", from };
    const type = sniffImageType(b);
    return type ? { kind: "tile", body: b, contentType: type, from } : { kind: "error", error: "cached object is not an image" };
  }

  /** Read-through: cache -> upstream -> cache. Deduped per key. */
  async function getTile(chart: AeroChartId, z: number, x: number, y: number): Promise<{ outcome: TileOutcome; edition: EditionInfo }> {
    const ed = await edition(chart);
    const st = charts[chart];
    if (!validTile(z, x, y) || z < st.minzoom || z > st.maxzoom) {
      return { outcome: { kind: "empty", from: "out-of-range" }, edition: ed };
    }
    const key = aeroCacheKey(chart, ed.edition, z, x, y);
    const existing = inflight.get(key);
    if (existing) {
      counters.dedupJoins++;
      return { outcome: await existing, edition: ed };
    }
    const p = (async (): Promise<TileOutcome> => {
      if (r2.configured) {
        const got = await r2.getObject(key, { maxBytes: 2 * 1024 * 1024 });
        if (got.ok && got.body) {
          const o = fromCachedBytes(got.body, "r2");
          if (o.kind !== "error") { counters.r2Hits++; st.hits++; return o; }
        } else if (got.status !== 404) {
          counters.r2Errors++;
        }
      }
      // the tmp LRU is the whole cache without R2, and the overflow tier with
      // it (write budget spent / deep "no tile" markers)
      const b = tmpCache().get(key);
      if (b) {
        const o = fromCachedBytes(b, "tmp");
        if (o.kind !== "error") { counters.tmpHits++; st.hits++; return o; }
      }
      st.misses++;
      const o = await fetchUpstream(chart, z, x, y);
      storeOutcome(key, z, o);
      return o;
    })().finally(() => inflight.delete(key));
    inflight.set(key, p);
    return { outcome: await p, edition: ed };
  }

  // ── background prefetch (R2 only) ──────────────────────────────────────────
  const prefetch = {
    running: false, chart: null as AeroChartId | null, edition: null as string | null,
    done: 0, total: 0, fetched: 0, skipped: 0, errors: 0,
    lastRunAt: null as number | null, lastCompleted: [] as string[], note: "" as string,
  };
  let stopRequested = false;

  async function runPrefetch(): Promise<void> {
    if (prefetch.running) return;
    if (!r2.configured) { prefetch.note = "disabled: R2 not configured (tmp cache is lazy-only)"; return; }
    const rps = Number(env.AERO_PREFETCH_RPS) > 0 ? Math.min(Number(env.AERO_PREFETCH_RPS), 10) : AERO_PREFETCH_DEFAULT_RPS;
    prefetch.running = true; stopRequested = false; prefetch.lastRunAt = now(); prefetch.note = "";
    try {
      for (const chart of AERO_CHART_IDS) {
        const ed = await edition(chart);
        if (ed.source !== "service-metadata") { prefetch.note = `skipped ${chart}: edition unverified`; continue; }
        const marker = `aero/${chart}/${ed.edition}/_prefetch_done`;
        const head = await r2.headObject(marker);
        if (head.ok && head.exists) { prefetch.lastCompleted.push(`${chart}@${ed.edition}`); continue; }
        const tiles = prefetchTiles(charts[chart]);
        Object.assign(prefetch, { chart, edition: ed.edition, done: 0, total: tiles.length });
        let failures = 0;
        for (const t of tiles) {
          if (stopRequested) return;
          const key = aeroCacheKey(chart, ed.edition, t.z, t.x, t.y);
          const h = await r2.headObject(key);
          if (h.ok && h.exists) { prefetch.skipped++; prefetch.done++; continue; }
          if (r2BudgetLeft() <= 0) {
            // no upstream fetch whose result could not be persisted anyway
            prefetch.note = `paused at ${chart} ${prefetch.done}/${tiles.length}: daily R2 write budget (${r2PutBudget}) reached — resumes next run`;
            return;
          }
          const o = await fetchUpstream(chart, t.z, t.x, t.y);
          if (o.kind === "error") { prefetch.errors++; failures++; }
          else { storeOutcome(key, t.z, o); prefetch.fetched++; }
          prefetch.done++;
          await sleep(1000 / rps);
        }
        if (failures === 0) {
          const put = await r2.putObject(marker, Buffer.from(new Date(now()).toISOString()), "text/plain");
          if (put.ok) prefetch.lastCompleted.push(`${chart}@${ed.edition}`);
        } else {
          prefetch.note = `${chart}: ${failures} tile fetches failed — next run retries them`;
        }
      }
    } finally {
      prefetch.running = false;
      prefetch.chart = null;
      prefetch.lastCompleted = prefetch.lastCompleted.slice(-8);
    }
  }

  async function status() {
    const t = now();
    const rows = [];
    for (const id of AERO_CHART_IDS) {
      const ed = await edition(id);
      const st = charts[id];
      rows.push({
        id, label: AERO_CHARTS[id].label, short: AERO_CHARTS[id].short, service: AERO_CHARTS[id].service,
        edition: ed.edition, editionSource: ed.source, effective: ed.effective, expires: ed.expires,
        expired: ed.expired, currentFaaCycle: ed.currentFaaCycle, behindCurrentCycle: ed.behindCurrentCycle,
        minzoom: st.minzoom, maxzoom: st.maxzoom, coverage: AERO_CHARTS[id].coverage,
        tiles: aeroClientTileTemplate(id, ed.source === "service-metadata" ? ed.edition : null),
        metadataFetchedAt: st.metadataFetchedAt ? new Date(st.metadataFetchedAt).toISOString() : null,
        metadataError: st.metadataError, hits: st.hits, misses: st.misses,
      });
    }
    return {
      charts: rows,
      cache: {
        mode: r2.configured ? "r2" : "tmp-lru",
        // with R2 the tmp LRU is the overflow tier; stats only once it exists
        tmp: r2.configured && !tmp ? null : tmpCache().stats(),
        ...counters,
        lastUpstreamErrorAt: counters.lastUpstreamErrorAt ? new Date(counters.lastUpstreamErrorAt).toISOString() : null,
        upstreamInflight: upstreamActive,
        r2PutBudgetPerDay: r2.configured ? r2PutBudget : null,
        r2PutsToday: r2.configured ? r2PutBudget - r2BudgetLeft() : null,
      },
      prefetch: { ...prefetch, lastRunAt: prefetch.lastRunAt ? new Date(prefetch.lastRunAt).toISOString() : null },
      r2Config: r2ConfigDiagnostics(env),
      source: "FAA Aeronautical Information Services chart tile caches (public domain), re-served from this origin",
      cycleNote: `effective/expires follow the FAA ${FAA_CYCLE_DAYS}-day chart cycle from the service-reported edition; behindCurrentCycle = the FAA's published cycle is newer than what the tile service carries`,
      lawNote: "Law II.8 compromise: lazy bake — first request per tile per edition reads through to the FAA service, then serves from our cache; with R2 configured a background job pre-bakes CONUS zoom<=9",
      notForNavigation: true,
      generated_at: new Date(t).toISOString(),
    };
  }

  /** The imagery row of /api/data/layers gains `aeroCharts` — synchronous,
   *  from whatever edition metadata is already cached (never a fetch on the
   *  registry path). The client refreshes from /api/data/aero/status. */
  function registryEntries() {
    const t = now();
    return AERO_CHART_IDS.map((id) => {
      const st = charts[id];
      const ed = editionInfo(st.serviceEdition, t);
      return {
        id, label: AERO_CHARTS[id].label, short: AERO_CHARTS[id].short,
        edition: ed.source === "service-metadata" ? ed.edition : null,
        effective: ed.effective, expires: ed.expires, expired: ed.expired, behindCurrentCycle: ed.behindCurrentCycle,
        minzoom: st.minzoom, maxzoom: st.maxzoom,
        tiles: aeroClientTileTemplate(id, ed.source === "service-metadata" ? ed.edition : null),
      };
    });
  }

  return {
    getTile, edition, status, runPrefetch, registryEntries,
    stopPrefetch: () => { stopRequested = true; },
    _charts: charts, _counters: counters,
  };
}

export type AeroChartService = ReturnType<typeof createAeroChartService>;

let service: AeroChartService | null = null;

/** Adds `aeroCharts` to the imagery (base) row of the layers registry. Pure
 *  pass-through for every other row, and when the service is not mounted. */
export function attachAeroChartEditions<T extends { id: string }>(layers: T[]): T[] {
  if (!service) return layers;
  const entries = service.registryEntries();
  return layers.map((l) => (l.id === "imagery" ? { ...l, aeroCharts: entries } : l));
}

export async function handleAeroTile(svc: AeroChartService, req: Request, res: Response): Promise<void> {
  const chart = String(req.params.chart || "");
  if (!isAeroChartId(chart)) { res.status(404).json({ error: "unknown chart" }); return; }
  const z = Number(req.params.z), x = Number(req.params.x), y = Number(req.params.y);
  if (!validTile(z, x, y)) { res.status(400).json({ error: "bad tile coordinates" }); return; }
  const requested = typeof req.query.e === "string" ? req.query.e : undefined;
  const { outcome, edition } = await svc.getTile(chart, z, x, y);
  res.setHeader("x-aero-edition", edition.edition);
  if (outcome.kind === "error") {
    // no-store: a transient upstream failure must not be cached as "empty"
    res.setHeader("cache-control", "no-store");
    res.status(502).json({ error: "chart tile unavailable", detail: outcome.error });
    return;
  }
  res.setHeader("cache-control", aeroCacheControl(requested, edition.edition));
  res.setHeader("x-aero-cache", outcome.from);
  if (outcome.kind === "empty") {
    res.setHeader("content-type", "image/png");
    res.status(200).end(AERO_EMPTY_PNG);
    return;
  }
  res.setHeader("content-type", outcome.contentType);
  res.status(200).end(outcome.body);
}

export const AERO_PREFETCH_FIRST_DELAY_MS = 10 * 60_000;
export const AERO_PREFETCH_INTERVAL_MS = 12 * 3600_000;

/** Boot wiring: ONE call from server/routes.ts. */
export function registerAeroChartRoutes(app: Express, deps: AeroDeps & { startTimers?: boolean } = {}): AeroChartService {
  const svc = createAeroChartService(deps);
  service = svc;
  app.get("/tiles/aero/:chart/:z/:x/:y", (req, res) => {
    handleAeroTile(svc, req, res).catch((e: unknown) => {
      if (!res.headersSent) res.status(500).json({ error: errText(e) });
    });
  });
  app.get("/api/data/aero/status", (_req, res) => {
    svc.status().then((s) => res.json(s), (e: unknown) => res.status(500).json({ error: errText(e) }));
  });
  if (deps.startTimers !== false) {
    // warm the edition metadata so the layers registry can badge freshness
    for (const id of AERO_CHART_IDS) void svc.edition(id);
    const kick = () => {
      svc.runPrefetch().catch((e: unknown) => console.warn(`[aero] prefetch failed: ${errText(e)}`));
    };
    const first = setTimeout(kick, AERO_PREFETCH_FIRST_DELAY_MS);
    const every = setInterval(kick, AERO_PREFETCH_INTERVAL_MS);
    first.unref?.();
    every.unref?.();
  }
  return svc;
}
