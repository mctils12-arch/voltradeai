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

// ── tile visibility amplification (v1.0.69 root-cause fix) ────────────────
// MEASURED ROOT CAUSE of "temp/wind not rendering" (2026-07-04, prod tiles
// pixel-analyzed): OWM Weather Maps 1.0 tiles are intrinsically
// near-transparent pastels built for LIGHT basemaps — temp_new is uniform
// ~76/255 alpha, wind_new ~15-53/255 with ZERO pixels above 120/255. Over
// our dark satellite base, and further scaled by client raster-opacity,
// effective visibility was 3-18% — invisible. GL raster paint can only
// REDUCE opacity below the texture's baked-in alpha, so the fix must happen
// before the texture: the proxy amplifies alpha (and saturation, the
// palette is pale) once per tile per TTL. Factors chosen from the measured
// per-layer alpha floors.
import { PNG } from "pngjs";

export const WX_ALPHA_FACTOR: Record<WxLayer, number> = {
  temp_new: 3.2,   // 76/255 avg -> ~95% ceiling-capped visible field
  wind_new: 5.5,   // 15-53/255 -> calm air stays faint, real wind reads
};
export const WX_ALPHA_CAP = 230;      // never fully opaque — basemap context survives
export const WX_SATURATION = 1.6;     // pale 1.0-palette colors get usable chroma

/** Amplifies a Weather-Maps-1.0 PNG's alpha + saturation. Pure buffer
 *  transform (pngjs, no native deps); throws on non-PNG input — the route
 *  falls back to the untouched buffer rather than serving nothing. */
export function amplifyWeatherTile(buf: Buffer, layer: WxLayer): Buffer {
  const png = PNG.sync.read(buf);
  const f = WX_ALPHA_FACTOR[layer] ?? 1;
  const d = png.data;
  for (let i = 0; i < d.length; i += 4) {
    const a = d[i + 3];
    if (a === 0) continue;
    d[i + 3] = Math.min(WX_ALPHA_CAP, Math.round(a * f));
    // saturation boost around luma (Rec.601), clamped
    const r = d[i], g = d[i + 1], b = d[i + 2];
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    d[i]     = Math.max(0, Math.min(255, Math.round(y + (r - y) * WX_SATURATION)));
    d[i + 1] = Math.max(0, Math.min(255, Math.round(y + (g - y) * WX_SATURATION)));
    d[i + 2] = Math.max(0, Math.min(255, Math.round(y + (b - y) * WX_SATURATION)));
  }
  return PNG.sync.write(png);
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
