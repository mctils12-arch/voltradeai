// mapBaseConfig — the Earth base map's motion/raster configuration
// (Rendering & Motion Law, Laws I and II) in one testable place.
//
// The Earth base is MapLibre's own tile pipeline, so tileCore does not
// drive it — MapLibre already implements the ready-gate, the abortable
// requests and the LRU internally. What it does NOT do by default is
// match our crossfade and cache policy, and the values it was configured
// with in datamap.tsx actively fought Law II. This module holds those
// knobs as constants with their reasoning attached, so the next session
// changing a fade duration has to argue with a comment instead of a
// magic number.
//
// THREE ITEMS IN THE WORK ORDER'S PR4 ARE NOT APPLIED, each for a
// verified reason. They are recorded here rather than silently dropped:
//
//  1. `tileSize: 512` on the imagery source. Our imagery source is Esri
//     World_Imagery, which serves 256px tiles. Declaring 512 does not
//     make the server send bigger tiles — it makes MapLibre request one
//     zoom level COARSER and upscale it, i.e. a permanently blurrier
//     base map. 512 is correct only for a source that actually bakes
//     512px tiles, which is what the RunPod pyramid bake will produce.
//     Revisit when the base moves to our own CDN.
//  2. `map.setPrefetchZoomDelta(4)`. This method does not exist in
//     MapLibre GL — it is a Mapbox GL JS API. Verified against
//     maplibre-gl 5.24.0: zero occurrences of `prefetchZoomDelta` in the
//     shipped bundle. Calling it would throw.
//  3. `transformRequest` rewriting tiles to our own CDN (Law II.8).
//     There is no CDN mirror of the imagery yet — that IS the RunPod
//     bake pipeline. Pointing the base map at a CDN path that does not
//     exist would 404 the entire Earth surface. This is a real
//     outstanding Law II.8 violation, tracked rather than faked.

import { ZOOM } from "./zoomInput.ts";

/**
 * Raster crossfade for the base imagery, ms.
 *
 * It was 0, chosen for "perceived speed" — tiles landing instantly with
 * no fade that reads as loading. That trade is exactly the failure mode
 * Law II names: `raster-fade-duration: 0` produces HARD POPS. A tile
 * appearing instantly at full opacity over its upscaled parent is a
 * visible snap on every zoom step; the 200-300ms fade is what turns that
 * snap into a resolve. Sits at the top of Law II's 150-300ms band because
 * the base map's parent tile is always present, so a longer fade costs
 * nothing in blankness.
 */
export const RASTER_FADE_MS = 300;

/**
 * Label/symbol fade, ms. A DIFFERENT knob from `raster-fade-duration`
 * despite the similar name — this one is the Map constructor's
 * `fadeDuration` and governs symbol collision fading. Zero is correct
 * here: labels have no parent to crossfade from, so their fade is pure
 * latency, and a label fading in during a pan is the flicker Law I
 * complains about.
 */
export const SYMBOL_FADE_MS = 0;

/**
 * Hard ceiling on cached tiles.
 *
 * MapLibre's default is derived from viewport size and is effectively
 * unbounded on a large screen. This file's history records context loss
 * on shared-memory integrated GPUs (the eviction signature: the map's GL
 * context dies alone while the GPU process survives), and an unbounded
 * tile cache is the largest single contributor.
 *
 * 500 is a ceiling, not a squeeze: at 256px tiles a 1440×900 viewport
 * shows ~24 tiles per level, and `maxTileCacheZoomLevels` retains 8
 * levels, so the working set is ~200. The cap only bites during
 * pathological pan-and-zoom churn, which is precisely when it should.
 */
export const MAX_TILE_CACHE_TILES = 500;

/**
 * Never re-request a tile because its Cache-Control expired. Base imagery
 * is effectively immutable at our timescales, and an expiry-driven
 * re-fetch is indistinguishable to the user from the tile pipeline
 * reloading for no reason.
 */
export const REFRESH_EXPIRED_TILES = false;

/** Constructor options this module owns. Spread into `new maplibregl.Map`. */
export function baseMapMotionOptions(): {
  fadeDuration: number;
  maxTileCacheSize: number;
  refreshExpiredTiles: boolean;
} {
  return {
    fadeDuration: SYMBOL_FADE_MS,
    maxTileCacheSize: MAX_TILE_CACHE_TILES,
    refreshExpiredTiles: REFRESH_EXPIRED_TILES,
  };
}

/** Paint properties for a raster base layer. */
export function baseRasterPaint(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return { "raster-fade-duration": RASTER_FADE_MS, ...extra };
}

/** Minimal surface of the bits of MapLibre's Map this module touches. */
export interface WheelZoomTarget {
  scrollZoom?: { setWheelZoomRate?(rate: number): void };
}

/**
 * Pin the map's wheel rate to the same constant zoomInput uses.
 *
 * HONEST NOTE: MapLibre 5.24's own default wheelZoomRate is ALREADY
 * 1/450 (verified in the shipped bundle), so this call changes nothing
 * today. It is here so the Earth base and the celestial views cannot
 * drift apart silently — if a future MapLibre release retunes its
 * default, this keeps the two feels identical, and the accompanying test
 * fails loudly if ZOOM.WHEEL_RATE is ever changed without thinking about
 * the map.
 */
export function applyWheelZoomRate(map: WheelZoomTarget, rate: number = ZOOM.WHEEL_RATE): boolean {
  const fn = map.scrollZoom?.setWheelZoomRate;
  if (typeof fn !== "function") return false;
  fn.call(map.scrollZoom, rate);
  return true;
}

/**
 * Guard: a raster paint block that fades in 0ms is a Law II violation.
 * Exported so the harness and the unit test check the same predicate the
 * map is configured through.
 */
export function rasterFadeViolations(paint: Record<string, unknown> | undefined | null): string[] {
  if (!paint) return [];
  const v = paint["raster-fade-duration"];
  if (v === 0) return ["raster-fade-duration: 0 produces hard pops on every zoom step (Law II.1)"];
  if (typeof v === "number" && (v < 150 || v > 300)) {
    return [`raster-fade-duration ${v}ms is outside Law II's 150-300ms crossfade band`];
  }
  return [];
}
