/**
 * owmTiles.ts — OpenWeatherMap global weather-field tiles (Tier-1(b) global
 * half, unblocked 2026-07-04 when the human set OPENWEATHERMAP_KEY).
 *
 * The key stays SERVER-SIDE: the client fetches /api/data/wxtile/... and the
 * server proxies to tile.openweathermap.org with the key appended, caching
 * tiles in memory. Two reasons this is a proxy and not a client-side key:
 * (1) the free tier is 60 calls/min — one visitor panning the map would blow
 * it; a shared cache bounds upstream calls to unique-tiles-per-TTL across ALL
 * visitors (DESIGN.md: one upstream request shared, never per-visitor
 * fan-out); (2) keys in client tile URLs are scrapeable.
 *
 * FRESH-KEY HONESTY (human note 2026-07-04): OWM activates new keys within
 * ~2h; until then the API returns 401. A 401 therefore classifies as
 * "activating" (retry, not broken) — the layer shows a retry note, never an
 * error state, for a fresh-key delay. If it persists well past activation
 * (~2h), the same note tells the human to re-check the key.
 *
 * Pure module: no fetch here — URL building, validation, classification, and
 * the tile cache are all testable without network; routes.ts does the I/O.
 */

export const VALID_WX_LAYERS = ["temp_new", "wind_new"] as const;
export type WxLayer = (typeof VALID_WX_LAYERS)[number];

/** Tile budget: z <= 7 bounds the cache universe and is plenty for
 *  continental-scale temperature/wind fields (they're smooth); OWM free
 *  tiles exist to higher zooms but serve no extra signal for fields. */
export const MAX_WX_ZOOM = 7;
export const TILE_TTL_MS = 10 * 60_000;       // OWM field tiles update ~10 min
export const NEGATIVE_TTL_MS = 5 * 60_000;    // don't hammer OWM while a fresh key activates

export function validateWxTile(layer: string, z: number, x: number, y: number): layer is WxLayer {
  if (!(VALID_WX_LAYERS as readonly string[]).includes(layer)) return false;
  if (!Number.isInteger(z) || !Number.isInteger(x) || !Number.isInteger(y)) return false;
  if (z < 0 || z > MAX_WX_ZOOM) return false;
  const n = 2 ** z;
  return x >= 0 && x < n && y >= 0 && y < n;
}

export function owmTileUrl(layer: WxLayer, z: number, x: number, y: number, key: string): string {
  return `https://tile.openweathermap.org/map/${layer}/${z}/${x}/${y}.png?appid=${encodeURIComponent(key)}`;
}

export type OwmStatus = "ok" | "awaiting_key" | "activating" | "error";

/** Classify an upstream OWM HTTP status. 401/403 = key not (yet) accepted —
 *  for a fresh key that is normal for up to ~2h, so it is "activating"
 *  (retryable), never a hard error. */
export function classifyOwmStatus(httpStatus: number | null, keyPresent: boolean): OwmStatus {
  if (!keyPresent) return "awaiting_key";
  if (httpStatus === 200) return "ok";
  if (httpStatus === 401 || httpStatus === 403) return "activating";
  return "error";
}

export function owmStatusNote(s: OwmStatus): string {
  switch (s) {
    case "ok": return "Weather data © OpenWeatherMap";
    case "awaiting_key": return "OPENWEATHERMAP_KEY not set";
    case "activating":
      return "key set — OpenWeatherMap activates fresh keys within ~2h; " +
             "auto-retrying. If this persists well past 2h, re-check the key.";
    case "error": return "upstream error — retrying";
  }
}

/** Tiny TTL cache with FIFO eviction — bounded memory for tile buffers. */
export function makeTileCache<T>(maxEntries = 2000) {
  const m = new Map<string, { at: number; ttl: number; v: T }>();
  return {
    get(k: string, now = Date.now()): T | undefined {
      const e = m.get(k);
      if (!e) return undefined;
      if (now - e.at > e.ttl) { m.delete(k); return undefined; }
      return e.v;
    },
    set(k: string, v: T, ttl = TILE_TTL_MS, now = Date.now()) {
      if (m.size >= maxEntries) {
        const oldest = m.keys().next().value;
        if (oldest !== undefined) m.delete(oldest);
      }
      m.set(k, { at: now, ttl, v });
    },
    size() { return m.size; },
  };
}
