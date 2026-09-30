// GLOBAL AIRCRAFT FEED (FLIGHT PROGRAM B1, 2026-09-28) — the zoomed-out
// source for the live aircraft layer.
//
// The precise viewport feed (/api/data/aircraft) covers a view with at most
// MAX_DISCS_PER_REFRESH (8) 250nm point-query discs; past that it honestly
// says "viewport too large — showing central region". At those wide views
// this module switches the SAME layer to /api/data/aircraft/global — the
// server's worldwide snapshot (adsb.lol sweep + viewers' discs + optional
// OpenSky) — and switches back to the precise feed once the view fits in 8
// discs again (hysteresis: enter above 8, leave at ≤ 6, so the boundary
// can't flap). The global payload is adapted into the EXACT row shape the
// viewport feed produces, so every renderer downstream (2D symbols,
// vectors, glide, the 3D layer, the airline filter, click cards) is
// untouched.
//
// HONESTY: each global row carries its own fix time (seenAt); rows whose
// fix is older than STALE_ROW_MS (2 min) are flagged `stale` and drawn
// DIMMER (datamap ALT_COLOR dim variant), and the layer note states the
// fresh share + that remote oceans may be uncovered. Data age is computed
// against the SERVER's `at` clock, never the device clock.
//
// Law I / Law II.6: fetches are triggered only by the existing poll tick /
// camera-settle path (wireLivePoints.load); every request is abortable and
// a superseding request or dispose() aborts the previous one.

/** Mirrors server/aircraftTiling.ts (pinned by globalFeed.test.ts). */
export const MAX_DISCS_PER_REFRESH = 8;
export const DISC_RADIUS_MAX_NM = 250;
/** leave global mode only once the view fits in this many discs */
export const GLOBAL_EXIT_DISCS = 6;
/** below this zoom the view is global regardless of bounds (globe views
 *  report unreliable bounds) */
export const GLOBAL_FORCE_BELOW_ZOOM = 3;
/** global poll cadence. 20 s -> 8 s (2026-09-30, human: "faster refresh on
 *  every plane without bogging the system down"): polls after the first are
 *  DELTAS (`changed=<previous at>`, rows that changed in the server snapshot
 *  since the last answer) merged into the held set, so a faster cadence
 *  costs a fraction of the old full 1.2 MB snapshot. Browser<->our-server
 *  traffic only — the snapshot is fed by the server sweep either way. */
export const GLOBAL_FEED_POLL_MS = 8_000;
/** same-bbox dedupe: a camera settle right after a poll reuses it */
export const GLOBAL_FEED_MIN_SPACING_MS = 4_000;
/** a full (non-delta) resync at least this often: reconciles rows the
 *  server dropped at its cap and any merge drift */
export const GLOBAL_RESYNC_MS = 90_000;
/** mirrors server SNAPSHOT_EVICT_MS: the server drops a row whose fix is
 *  older than this, and a delta cannot say so — the client applies the same
 *  rule to its held set */
export const GLOBAL_EVICT_MS = 10 * 60_000;
/** re-query (instead of reusing the held bbox) once the held query area is
 *  this many times the view's own padded query — zooming in shrinks the
 *  payload rather than dragging a continent along */
export const HELD_BBOX_MAX_AREA_RATIO = 4;
/** rows older than this render dimmer (stale position, not live) */
export const STALE_ROW_MS = 120_000;
/** bbox padding for the server-side filter (pans inside it need no refetch) */
export const BBOX_PAD_FRACTION = 0.25;

/** Dimmed twins of datamap's ALT_COLOR bands (ground #6680a0 / low <3000 m
 *  #fbb24c / cruise #4d9fff) at ~55% brightness — SAME band thresholds, so
 *  a stale row keeps its altitude meaning and only reads as "older". */
export const STALE_BAND_COLORS = { ground: "#38465a", low: "#8a6229", cruise: "#2a578c" } as const;
export const STALE_ALT_COLOR: unknown[] = ["case",
  ["get", "ground"], STALE_BAND_COLORS.ground,
  ["<", ["coalesce", ["get", "alt"], 99999], 3000], STALE_BAND_COLORS.low,
  STALE_BAND_COLORS.cruise];

const NM_PER_DEG_LAT = 60;
const GRID_SPACING_NM = DISC_RADIUS_MAX_NM * Math.SQRT2;

export interface Bounds { north: number; south: number; east: number; west: number }

/** How many 250nm discs a full cover of this view needs — the SAME math
 *  as server planDiscs().needed (local approximation, cos floor 0.1). */
export function discsNeeded(b: Bounds): number {
  const laLo = Math.min(b.south, b.north), laHi = Math.max(b.south, b.north);
  const loLo = Math.min(b.west, b.east), loHi = Math.max(b.west, b.east);
  const clat = (Math.max(-85, laLo) + Math.min(85, laHi)) / 2;
  const cosLat = Math.max(0.1, Math.cos((clat * Math.PI) / 180));
  const latSpanNm = (Math.min(85, laHi) - Math.max(-85, laLo)) * NM_PER_DEG_LAT;
  const lonSpanNm = (Math.min(180, loHi) - Math.max(-180, loLo)) * NM_PER_DEG_LAT * cosLat;
  const neededNm = Math.ceil(Math.hypot(latSpanNm, lonSpanNm) / 2);
  if (neededNm <= DISC_RADIUS_MAX_NM) return 1;
  return Math.max(1, Math.ceil(latSpanNm / GRID_SPACING_NM)) * Math.max(1, Math.ceil(lonSpanNm / GRID_SPACING_NM));
}

/** Pure mode decision with hysteresis. */
export function wantsGlobalFeed(currentlyGlobal: boolean, b: Bounds, zoom: number): boolean {
  if (!Number.isFinite(zoom) || zoom < GLOBAL_FORCE_BELOW_ZOOM) return true;
  if (![b.north, b.south, b.east, b.west].every(Number.isFinite)) return currentlyGlobal;
  if (Math.abs(b.east - b.west) >= 360) return true;
  const n = discsNeeded(b);
  return currentlyGlobal ? n > GLOBAL_EXIT_DISCS : n > MAX_DISCS_PER_REFRESH;
}

const wrap = (lon: number) => ((((lon + 180) % 360) + 360) % 360) - 180;

/** Padded, whole-degree bbox for the server filter; null = whole world. */
export function globalQueryBBox(b: Bounds): { lamin: number; lamax: number; lomin: number; lomax: number } | null {
  if (![b.north, b.south, b.east, b.west].every(Number.isFinite)) return null;
  const lonSpan = Math.abs(b.east - b.west);
  const latSpan = Math.abs(b.north - b.south);
  if (lonSpan >= 240 || latSpan >= 120) return null; // effectively global — skip the filter
  const pLat = latSpan * BBOX_PAD_FRACTION, pLon = lonSpan * BBOX_PAD_FRACTION;
  const lamin = Math.max(-90, Math.floor(Math.min(b.south, b.north) - pLat));
  const lamax = Math.min(90, Math.ceil(Math.max(b.south, b.north) + pLat));
  const w = Math.floor(Math.min(b.west, b.east) - pLon);
  const e = Math.ceil(Math.max(b.west, b.east) + pLon);
  if (e - w >= 360) return { lamin, lamax, lomin: -180, lomax: 180 };
  // unwrapped map bounds (e.g. east 200) → wrapped; west > east = antimeridian bbox (server handles)
  return { lamin, lamax, lomin: wrap(w), lomax: e === 180 ? 180 : wrap(e) };
}

export type QueryBBox = { lamin: number; lamax: number; lomin: number; lomax: number };

export function queryUrl(q: QueryBBox | null): string {
  return q
    ? `/api/data/aircraft/global?lamin=${q.lamin}&lamax=${q.lamax}&lomin=${q.lomin}&lomax=${q.lomax}`
    : "/api/data/aircraft/global";
}

export function globalFeedUrl(b: Bounds): string {
  return queryUrl(globalQueryBBox(b));
}

const qArea = (q: QueryBBox) => (q.lamax - q.lamin) * (q.lomin <= q.lomax ? q.lomax - q.lomin : 360 - q.lomin + q.lomax);

/**
 * May a view keep polling the HELD query bbox (so the next poll can be a
 * delta) instead of a fresh full query? Yes when the view's own padded query
 * lies inside the held one (a pan inside the padding) and the held area is
 * not wildly larger (a deep zoom-in re-queries to shrink the payload).
 * Antimeridian-crossing boxes simply re-query (rare, always correct).
 */
export function heldQueryCovers(held: QueryBBox | null, next: QueryBBox | null): boolean {
  if (held === null) return next === null; // world covers only a world view (a smaller view re-queries smaller)
  if (next === null) return false;
  if (held.lomin > held.lomax || next.lomin > next.lomax) return false;
  if (next.lamin < held.lamin || next.lamax > held.lamax || next.lomin < held.lomin || next.lomax > held.lomax) return false;
  return qArea(held) <= HELD_BBOX_MAX_AREA_RATIO * Math.max(1, qArea(next));
}

/** Held rows of the global feed, keyed by hex, merged from full + delta
 *  answers (pure; the feed wraps it). */
export interface HeldSet {
  q: QueryBBox | null;
  /** server `at` of the last answer (the next delta cursor) */
  at: number;
  lastFullAt: number;
  fields: string[];
  rows: Map<string, unknown[]>;
  /** earliest server time at which a held fresh row turns stale (dimming
   *  must update even when no row changed) */
  nextStaleFlipAt: number;
}

/**
 * Merge a /global answer into the held set. A full answer replaces it; a
 * delta (full:false) upserts by hex. Either way rows whose fix is older than
 * the server's own evict rule are dropped (a delta cannot carry evictions).
 * `changed` = anything a renderer would draw differently (incl. a row
 * crossing the 2-min stale line, so dimming stays honest with no new data).
 */
export function mergeGlobalAnswer(held: HeldSet | null, raw: unknown, q: QueryBBox | null, localNow: number): { held: HeldSet; changed: boolean } {
  const d = (raw && typeof raw === "object" ? raw : {}) as { at?: unknown; fields?: unknown; rows?: unknown; full?: unknown };
  const fields: string[] = Array.isArray(d.fields) ? d.fields.map(String) : (held?.fields ?? []);
  const iHex = fields.indexOf("hex"), iSeen = fields.indexOf("seenAt");
  const at = asNum(d.at) ?? localNow;
  const isDelta = d.full === false && held !== null && held.fields.join(",") === fields.join(",");
  const rows = isDelta ? held!.rows : new Map<string, unknown[]>();
  let changed = !isDelta;
  for (const row of Array.isArray(d.rows) ? d.rows : []) {
    if (!Array.isArray(row)) continue;
    const hex = iHex >= 0 ? asStr(row[iHex]) : null;
    if (!hex) continue;
    rows.set(hex, row);
    changed = true;
  }
  let nextFlip = Infinity;
  if (iSeen >= 0) {
    const cut = at - GLOBAL_EVICT_MS;
    for (const [hex, row] of Array.from(rows)) {
      const seen = asNum(row[iSeen]);
      if (seen == null) continue;
      if (seen < cut) { rows.delete(hex); changed = true; continue; }
      const flip = seen + STALE_ROW_MS;
      if (flip > at && flip < nextFlip) nextFlip = flip;
    }
  }
  if (isDelta && !changed && at >= held!.nextStaleFlipAt) changed = true;
  return {
    held: { q, at, lastFullAt: isDelta ? held!.lastFullAt : localNow, fields, rows, nextStaleFlipAt: nextFlip },
    changed,
  };
}

const KT_TO_MS = 0.5144; // the pipeline's own factor (server mapPointAircraft)
const FT_TO_M = 0.3048;

/** One live aircraft row in the viewport feed's shape (what every
 *  renderer downstream of wireLivePoints reads). */
export interface AdaptedAircraft {
  icao24: string | null;
  callsign: string;
  registration: string;
  lon: number;
  lat: number;
  altitude_m: number | null;
  on_ground: boolean;
  velocity_ms: number | null;
  heading: number | null;
  type: string | null;
  category: string | null;
  provider: string | null;
  /** fix age in seconds at the server's `at` (the viewport feed's seen_pos) */
  seen_pos: number | null;
  /** fix older than STALE_ROW_MS — rendered dimmer */
  stale: boolean;
}

export interface AdaptedPayload {
  source: string;
  kind: "raw";
  time: string;
  global: true;
  at: number;
  count: number;
  coverage: "partial";
  coverage_note: string;
  aircraft: AdaptedAircraft[];
}

export interface UnchangedPayload { unchanged: true }

const asNum = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const asStr = (v: unknown): string | null => (typeof v === "string" && v ? v : null);

/**
 * Global endpoint payload → the viewport feed's payload shape. Rows are
 * decoded by the response's own `fields` list (never by assumed order).
 * `time` is namespaced ("g…") so the viewport feed's delta cursor can never
 * mistake a global snapshot for one of its own.
 */
export function adaptGlobalPayload(raw: unknown): AdaptedPayload {
  const d = (raw && typeof raw === "object" ? raw : {}) as {
    at?: unknown; fields?: unknown; rows?: unknown; coverage?: { opensky?: { enabled?: unknown } };
  };
  const fields: string[] = Array.isArray(d.fields) ? d.fields.map(String) : [];
  const ix = (f: string) => fields.indexOf(f);
  const I = {
    hex: ix("hex"), lon: ix("lon"), lat: ix("lat"), altFt: ix("altFt"), gsKt: ix("gsKt"),
    trk: ix("trk"), callsign: ix("callsign"), type: ix("type"), seenAt: ix("seenAt"),
    cat: ix("cat"), gnd: ix("gnd"), reg: ix("reg"), src: ix("src"),
  };
  const at = asNum(d.at) ?? Date.now();
  const get = (row: unknown[], i: number): unknown => (i >= 0 ? row[i] : null);
  let fresh = 0;
  const aircraft: AdaptedAircraft[] = [];
  for (const row of Array.isArray(d.rows) ? d.rows : []) {
    if (!Array.isArray(row)) continue;
    const lon = asNum(get(row, I.lon)), lat = asNum(get(row, I.lat));
    if (lon == null || lat == null) continue;
    const altFt = asNum(get(row, I.altFt));
    const gsKt = asNum(get(row, I.gsKt));
    const seenAt = asNum(get(row, I.seenAt));
    const ageMs = seenAt == null ? null : Math.max(0, at - seenAt);
    const stale = ageMs == null || ageMs > STALE_ROW_MS;
    if (!stale) fresh++;
    const gnd = get(row, I.gnd);
    aircraft.push({
      icao24: asStr(get(row, I.hex)),
      callsign: asStr(get(row, I.callsign)) ?? "",
      registration: asStr(get(row, I.reg)) ?? "",
      lon, lat,
      altitude_m: altFt == null ? null : altFt * FT_TO_M,
      on_ground: gnd === 1 || gnd === true,
      velocity_ms: gsKt == null ? null : gsKt * KT_TO_MS,
      heading: asNum(get(row, I.trk)),
      type: asStr(get(row, I.type)),
      category: asStr(get(row, I.cat)),
      provider: asStr(get(row, I.src)),
      seen_pos: ageMs == null ? null : ageMs / 1000,
      stale,
    });
  }
  const n = aircraft.length;
  const pct = n ? Math.round((fresh / n) * 100) : 0;
  const opensky = d.coverage?.opensky?.enabled === true;
  return {
    source: "worldwide snapshot (adsb.lol sweep" + (opensky ? " + OpenSky" : "") + ")",
    kind: "raw",
    time: `g${at}`,
    global: true,
    at,
    count: n,
    coverage: "partial",
    coverage_note: `worldwide feed (zoomed out): ${pct}% of ${n.toLocaleString()} positions updated <2 min — older ones dimmed; remote oceans may be uncovered — zoom in for the live viewport feed`
      + (opensky ? " · includes The OpenSky Network data (opensky-network.org, non-commercial)" : ""),
    aircraft,
  };
}

export interface AircraftFeed {
  /** stateful (hysteresis) mode decision for the current camera */
  shouldUseGlobal(b: Bounds, zoom: number): boolean;
  /** poll cadence while global (null = use the caller's own cadence) */
  pollMs(): number | null;
  /** abortable fetch of the global snapshot, adapted to the viewport shape;
   *  { unchanged: true } when superseded or deduped */
  fetchGlobal(b: Bounds, bypassCache?: boolean): Promise<AdaptedPayload | UnchangedPayload>;
  isGlobal(): boolean;
  dispose(): void;
}

const UNCHANGED: UnchangedPayload = { unchanged: true };

export function createAircraftFeed(opts: { fetchImpl?: typeof fetch; now?: () => number } = {}): AircraftFeed {
  const fetchImpl = opts.fetchImpl ?? ((...a: Parameters<typeof fetch>) => fetch(...a));
  const now = opts.now ?? (() => Date.now());
  let global = false;
  let inflight: AbortController | null = null;
  let lastUrl = "";
  let lastAt = 0;
  let disposed = false;
  let held: HeldSet | null = null;
  return {
    shouldUseGlobal(b, zoom) {
      global = wantsGlobalFeed(global, b, zoom);
      if (!global) { inflight?.abort(); inflight = null; lastUrl = ""; held = null; }
      return global;
    },
    pollMs() { return global ? GLOBAL_FEED_POLL_MS : null; },
    isGlobal() { return global; },
    async fetchGlobal(b, bypassCache = false) {
      if (disposed) return UNCHANGED;
      const t = now();
      const want = globalQueryBBox(b);
      // keep the held query while the view stays inside it -> delta polls
      const keep = held !== null && heldQueryCovers(held.q, want);
      const q = keep ? held!.q : want;
      const url = queryUrl(q);
      if (url === lastUrl && t - lastAt < GLOBAL_FEED_MIN_SPACING_MS) return UNCHANGED;
      const delta = keep && !bypassCache && t - held!.lastFullAt < GLOBAL_RESYNC_MS;
      const reqUrl = delta ? `${url}${url.includes("?") ? "&" : "?"}changed=${held!.at}` : url;
      inflight?.abort(); // a newer request supersedes the older one
      const ac = new AbortController();
      inflight = ac;
      try {
        const r = await fetchImpl(reqUrl, { signal: ac.signal, ...(bypassCache ? { cache: "reload" as RequestCache } : {}) });
        if (!r.ok) throw new Error(String(r.status));
        const d: unknown = await r.json();
        if (ac.signal.aborted) return UNCHANGED;
        lastUrl = url;
        lastAt = t;
        const m = mergeGlobalAnswer(delta ? held : null, d, q, t);
        held = m.held;
        // nothing on screen changed -> the caller skips the whole rebuild
        // (features, setData, 3D instances): the cheapest poll is no work
        if (!m.changed) return UNCHANGED;
        return adaptGlobalPayload({ ...(d as Record<string, unknown>), fields: held.fields, rows: Array.from(held.rows.values()) });
      } catch (e) {
        // superseded/disposed requests are not errors — the newer one delivers
        if (ac.signal.aborted || (e as { name?: unknown } | null)?.name === "AbortError") return UNCHANGED;
        throw e;
      } finally {
        if (inflight === ac) inflight = null;
      }
    },
    dispose() {
      disposed = true;
      inflight?.abort();
      inflight = null;
      held = null;
    },
  };
}
