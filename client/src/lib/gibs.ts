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
 *  never available same-day, so "today" is never an offerable date. */
export function gibsDefaultDate(nowMs: number): string {
  const d = new Date(nowMs);
  d.setUTCDate(d.getUTCDate() - 1);
  return toISODate(d);
}

export function gibsStepDate(dateISO: string, deltaDays: number): string {
  const d = new Date(`${dateISO}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + deltaDays);
  return toISODate(d);
}

/** A date is honestly un-offerable once it reaches "yesterday" relative to
 *  now — GIBS daily layers do not carry today's data. Used to disable the
 *  scrubber's "next day" control rather than silently serving a guaranteed-
 *  blank tile. */
export function gibsIsLatestAvailable(dateISO: string, nowMs: number): boolean {
  return dateISO >= gibsDefaultDate(nowMs);
}
