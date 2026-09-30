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
/** global poll cadence (the snapshot's type lane refreshes ~60s) */
export const GLOBAL_FEED_POLL_MS = 20_000;
/** same-bbox dedupe: a camera settle right after a poll reuses it */
export const GLOBAL_FEED_MIN_SPACING_MS = 10_000;
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

export function globalFeedUrl(b: Bounds): string {
  const q = globalQueryBBox(b);
  return q
    ? `/api/data/aircraft/global?lamin=${q.lamin}&lamax=${q.lamax}&lomin=${q.lomin}&lomax=${q.lomax}`
    : "/api/data/aircraft/global";
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
  return {
    shouldUseGlobal(b, zoom) {
      global = wantsGlobalFeed(global, b, zoom);
      if (!global) { inflight?.abort(); inflight = null; lastUrl = ""; }
      return global;
    },
    pollMs() { return global ? GLOBAL_FEED_POLL_MS : null; },
    isGlobal() { return global; },
    async fetchGlobal(b, bypassCache = false) {
      if (disposed) return UNCHANGED;
      const url = globalFeedUrl(b);
      const t = now();
      if (url === lastUrl && t - lastAt < GLOBAL_FEED_MIN_SPACING_MS) return UNCHANGED;
      inflight?.abort(); // a newer request supersedes the older one
      const ac = new AbortController();
      inflight = ac;
      try {
        const r = await fetchImpl(url, { signal: ac.signal, ...(bypassCache ? { cache: "reload" as RequestCache } : {}) });
        if (!r.ok) throw new Error(String(r.status));
        const d: unknown = await r.json();
        if (ac.signal.aborted) return UNCHANGED;
        lastUrl = url;
        lastAt = t;
        return adaptGlobalPayload(d);
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
    },
  };
}

/**
 * The REAL fix time (epoch seconds) of the selected plane's row in a feed
 * payload, or null when it must not be stamped as a live breadcrumb.
 * Viewport payloads: the snapshot's own `time` (seconds). Worldwide
 * payloads (2026-09-30, "the selected plane's track must survive the
 * zoomed-out feed"): each row carries its own age, so the fix time is the
 * server's `at` minus that age — and only rows fresher than STALE_ROW_MS
 * count (an older row is already dimmed as not-live on the map).
 */
export function feedFixTimeSec(
  payload: { global?: unknown; time?: unknown; at?: unknown },
  row: { seen_pos?: number | null; stale?: boolean },
  nowMs: number,
): number | null {
  if (payload.global === true) {
    if (row.stale === true || row.seen_pos == null || !Number.isFinite(row.seen_pos)) return null;
    if (row.seen_pos * 1000 > STALE_ROW_MS) return null;
    const at = typeof payload.at === 'number' && Number.isFinite(payload.at) ? payload.at : nowMs;
    return (at - row.seen_pos * 1000) / 1000;
  }
  return typeof payload.time === 'number' && Number.isFinite(payload.time) ? payload.time : nowMs / 1000;
}
