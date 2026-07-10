/**
 * NASA GIBS raster-layer factory (worldview_globe.md Phase G2).
 *
 * Access pattern verified live against gibs.earthdata.nasa.gov 2026-07-08
 * (VIIRS_SNPP_DayNightBand_At_Sensor_Radiance, GoogleMapsCompatible_Level8 —
 * Level9 rejected by the server with an explicit
 * "TILEMATRIXSET is invalid for LAYER" WMTS exception, not a network fluke).
 * Public domain, no key. Every G2 layer (G2a night-lights first, G2b-h to
 * follow) shares this factory instead of hand-rolling its own URL builder.
 */

export const GIBS_TILE_BASE = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best";

export interface GibsLayerSpec {
  /** GIBS layer identifier, e.g. "VIIRS_SNPP_DayNightBand_At_Sensor_Radiance". */
  layer: string;
  /** GoogleMapsCompatible_LevelN — native max resolution differs per layer;
   *  verify live before assuming a level works for a new layer (see header). */
  tileMatrixSet: string;
  ext: "png" | "jpg";
}

/** Build the WMTS REST tile URL template for a dated GIBS layer. dateISO is
 *  a plain YYYY-MM-DD (daily cadence — the only cadence any G2 layer here
 *  uses so far); sub-daily layers would need a full ISO timestamp instead. */
export function gibsTileUrl(spec: GibsLayerSpec, dateISO: string): string {
  return `${GIBS_TILE_BASE}/${spec.layer}/default/${dateISO}/${spec.tileMatrixSet}/{z}/{y}/{x}.${spec.ext}`;
}

const toISODate = (d: Date): string => d.toISOString().slice(0, 10);

/** Charter: "default the time-slider to yesterday." Daily GIBS layers are
 *  never available same-day, so "today" is never an offerable date.
 *
 *  `latencyDays` is the layer's typical processing lag in days (1 = a
 *  true-daily layer whose newest data is yesterday). NOT every GIBS product
 *  is same-day-minus-one: SMAP soil moisture lands ~6 days back, MODIS/VIIRS
 *  multi-day composites lag their compositing window. Passing the real lag
 *  keeps the scrubber from DEFAULTING to a guaranteed-blank tile — the same
 *  honesty rule as never offering "today". Default 1 preserves every existing
 *  caller's behavior unchanged. */
export function gibsDefaultDate(nowMs: number, latencyDays = 1): string {
  const d = new Date(nowMs);
  d.setUTCDate(d.getUTCDate() - Math.max(1, Math.floor(latencyDays)));
  return toISODate(d);
}

export function gibsStepDate(dateISO: string, deltaDays: number): string {
  const d = new Date(`${dateISO}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + deltaDays);
  return toISODate(d);
}

/** A date is honestly un-offerable once it reaches the layer's latest
 *  AVAILABLE date (now minus its processing lag) — GIBS does not carry data
 *  newer than that. Used to disable the scrubber's "next day" control rather
 *  than silently serving a guaranteed-blank tile. `latencyDays` must match the
 *  value passed to gibsDefaultDate for the same layer (default 1 = yesterday). */
export function gibsIsLatestAvailable(dateISO: string, nowMs: number, latencyDays = 1): boolean {
  return dateISO >= gibsDefaultDate(nowMs, latencyDays);
}

/** G2b (fires, `GOES-East_ABI_FireTemp`): a genuinely sub-daily, IRREGULAR
 *  cadence (~10-min scans with real gaps) — none of the day-granularity
 *  scrubber machinery above applies. `gibsTileUrl` already accepts the WMTS
 *  REST spec's literal "default" token in place of a dateISO, which the
 *  server resolves to its freshest available scan (live-verified
 *  2026-07-10). What "default" resolved to isn't otherwise observable from
 *  the client — GIBS exposes it as a `layer-time-actual` response header
 *  (CORS-exposed; confirmed live), so a lightweight HEAD request against the
 *  z=0/y=0/x=0 tile (always in-domain for every GIBS layer, near-zero bytes)
 *  reads the real scan time without downloading a full tile. Returns null on
 *  any failure — freshness is honestly "unknown" rather than fabricated;
 *  callers should render that as such, never as "just now". */
export async function gibsLatestScanTime(spec: GibsLayerSpec): Promise<string | null> {
  try {
    const url = gibsTileUrl(spec, "default").replace("{z}/{y}/{x}", "0/0/0");
    const res = await fetch(url, { method: "HEAD" });
    if (!res.ok) return null;
    return res.headers.get("layer-time-actual");
  } catch {
    return null;
  }
}
