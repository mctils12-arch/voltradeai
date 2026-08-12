// mapBaseConfig.test.ts — the Earth base's Law II knobs.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  MAX_TILE_CACHE_TILES,
  RASTER_FADE_MS,
  REFRESH_EXPIRED_TILES,
  SYMBOL_FADE_MS,
  applyWheelZoomRate,
  baseMapMotionOptions,
  baseRasterPaint,
  rasterFadeViolations,
} from "./mapBaseConfig.ts";
import { STREAM } from "./tileCore.ts";
import { ZOOM } from "./zoomInput.ts";

test("the raster crossfade sits inside Law II's 150-300ms band, and is never 0", () => {
  // `raster-fade-duration: 0` is called out by name in the work order's
  // failure modes: it makes a tile appear at full opacity over its
  // upscaled parent, i.e. a hard pop on every zoom step.
  assert.notEqual(RASTER_FADE_MS, 0);
  assert.ok(RASTER_FADE_MS >= 150 && RASTER_FADE_MS <= 300, `${RASTER_FADE_MS}ms outside the band`);
});

test("symbol fade and raster fade are DIFFERENT knobs and may legitimately differ", () => {
  // The classic mix-up: `fadeDuration` (symbol collision fade) vs
  // `raster-fade-duration` (tile crossfade). Zero is right for one and
  // wrong for the other.
  assert.equal(SYMBOL_FADE_MS, 0);
  assert.notEqual(RASTER_FADE_MS, SYMBOL_FADE_MS);
});

test("baseMapMotionOptions carries exactly the constructor knobs this module owns", () => {
  assert.deepEqual(baseMapMotionOptions(), {
    fadeDuration: 0,
    maxTileCacheSize: MAX_TILE_CACHE_TILES,
    refreshExpiredTiles: REFRESH_EXPIRED_TILES,
  });
  assert.equal(REFRESH_EXPIRED_TILES, false);
});

test("the tile ceiling is above a realistic working set, so it bounds churn and not normal use", () => {
  // 1440x900 at 256px tiles ≈ 24 tiles per level; maxTileCacheZoomLevels
  // retains 8. If the cap dropped below that the base map would re-fetch
  // during ordinary panning.
  const workingSet = Math.ceil(1440 / 256) * Math.ceil(900 / 256) * 8;
  assert.ok(MAX_TILE_CACHE_TILES > workingSet * 2, `${MAX_TILE_CACHE_TILES} too tight for a ${workingSet}-tile working set`);
});

test("baseRasterPaint sets the fade and preserves caller paint properties", () => {
  const paint = baseRasterPaint({ "raster-saturation": 0.18, "raster-contrast": 0.12 });
  assert.equal(paint["raster-fade-duration"], RASTER_FADE_MS);
  assert.equal(paint["raster-saturation"], 0.18);
  assert.equal(paint["raster-contrast"], 0.12);
  assert.deepEqual(rasterFadeViolations(paint), [], "the paint this module produces must pass its own guard");
});

test("rasterFadeViolations catches a zero fade and an out-of-band one", () => {
  assert.equal(rasterFadeViolations({ "raster-fade-duration": 0 }).length, 1);
  assert.match(rasterFadeViolations({ "raster-fade-duration": 0 })[0], /hard pops/);
  assert.equal(rasterFadeViolations({ "raster-fade-duration": 1000 }).length, 1);
  assert.equal(rasterFadeViolations({ "raster-fade-duration": 200 }).length, 0);
  assert.deepEqual(rasterFadeViolations({}), [], "an unset fade uses MapLibre's own default and is not a violation");
  assert.deepEqual(rasterFadeViolations(null), []);
});

test("the base map's crossfade agrees with tileCore's", () => {
  // Two raster pipelines with visibly different fade lengths read as two
  // different products on one screen.
  assert.ok(Math.abs(RASTER_FADE_MS - STREAM.CROSSFADE_MS) <= 100, "base and tileCore crossfades have drifted apart");
});

test("applyWheelZoomRate pins the map to the SAME rate zoomInput uses", () => {
  const seen: number[] = [];
  const map = { scrollZoom: { setWheelZoomRate: (r: number) => seen.push(r) } };
  assert.equal(applyWheelZoomRate(map), true);
  assert.deepEqual(seen, [ZOOM.WHEEL_RATE]);
  assert.equal(ZOOM.WHEEL_RATE, 1 / 450);
});

test("applyWheelZoomRate degrades safely when the handler is absent", () => {
  // scrollZoom is absent when the map is constructed with scrollZoom:false,
  // and setWheelZoomRate would be absent on an older bundle. Neither may
  // throw during map init.
  assert.equal(applyWheelZoomRate({}), false);
  assert.equal(applyWheelZoomRate({ scrollZoom: {} }), false);
});
