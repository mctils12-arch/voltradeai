// Bathymetry palette — EARTH TWIN E2-1. Pins the one-source-of-truth
// contract between the map ramp and the legend, and the "land transparent"
// overlay behavior that makes the layer honest over satellite imagery.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  BATHYMETRY_STOPS,
  BATHYMETRY_LAND_TRANSPARENT,
  bathymetryColorRelief,
} from "./bathymetry.js";

test("stops: strictly ascending elevations, all below sea level, labeled for the legend", () => {
  assert.ok(BATHYMETRY_STOPS.length >= 4, "enough zones to read depth");
  for (let i = 0; i < BATHYMETRY_STOPS.length; i++) {
    const s = BATHYMETRY_STOPS[i];
    assert.ok(s.elevM < 0, `${s.label}: bathymetry stops are below sea level`);
    assert.ok(/^#[0-9a-f]{6}$/i.test(s.color), `${s.label}: solid hex color (opacity belongs to color-relief-opacity)`);
    assert.ok(s.label.length > 0 && s.depthM >= 0);
    if (i > 0) assert.ok(s.elevM > BATHYMETRY_STOPS[i - 1].elevM,
      "elevations must strictly ascend — MapLibre interpolate requires ordered stops");
  }
  assert.equal(BATHYMETRY_STOPS[0].elevM, -11000, "deepest stop covers the hadal trenches (Challenger Deep ~10,935 m)");
});

test("ramp: interpolate-over-elevation expression, stops in order, land transparent at 0", () => {
  const expr = bathymetryColorRelief() as unknown[];
  assert.deepEqual(expr.slice(0, 3), ["interpolate", ["linear"], ["elevation"]],
    "color-relief-color must interpolate over the elevation input");
  const pairs = expr.slice(3);
  assert.equal(pairs.length, (BATHYMETRY_STOPS.length + 1) * 2, "every stop + the land stop");
  for (let i = 0; i < BATHYMETRY_STOPS.length; i++) {
    assert.equal(pairs[i * 2], BATHYMETRY_STOPS[i].elevM, "ramp elevation comes from the shared stops");
    assert.equal(pairs[i * 2 + 1], BATHYMETRY_STOPS[i].color, "ramp color comes from the shared stops (legend parity)");
  }
  assert.equal(pairs[pairs.length - 2], 0, "sea level anchors the land stop");
  assert.equal(pairs[pairs.length - 1], BATHYMETRY_LAND_TRANSPARENT,
    "land must be fully transparent — the layer drains the ocean, never paints the continents");
  const alpha = /rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*0\s*\)/;
  assert.ok(alpha.test(BATHYMETRY_LAND_TRANSPARENT), "land stop carries alpha 0");
});
