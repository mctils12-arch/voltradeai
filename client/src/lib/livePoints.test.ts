// Live-points tick helpers — PERF session #2. Pins the three waste-source
// guards: vector-build gating, count quantization, redundant-refetch skip.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  VEC_MIN_ZOOM,
  VEC_BUILD_ZOOM,
  shouldBuildVectors,
  quantizeLiveCount,
  fetchFootprint,
  needsRefetch,
} from "./livePoints.js";

test("shouldBuildVectors: skips below the build zoom, builds just before layer visibility", () => {
  assert.equal(VEC_BUILD_ZOOM, VEC_MIN_ZOOM - 0.5, "build margin sits under the layer minzoom");
  assert.equal(shouldBuildVectors(3.6), false, "default continent zoom: vectors are pure waste");
  assert.equal(shouldBuildVectors(VEC_BUILD_ZOOM), true);
  assert.equal(shouldBuildVectors(10), true);
  assert.equal(shouldBuildVectors(Number.NaN), false, "broken zoom fails CLOSED here — skipping a hidden layer loses nothing");
});

test("quantizeLiveCount: exact for small counts, nearest-25 above 500, display-only semantics", () => {
  assert.equal(quantizeLiveCount(0), 0);
  assert.equal(quantizeLiveCount(499), 499, "small counts stay exact — they are information");
  assert.equal(quantizeLiveCount(3_493), 3_500);
  assert.equal(quantizeLiveCount(3_487), 3_475);
  assert.equal(quantizeLiveCount(3_500), 3_500);
  // the point of the quantization: consecutive jiggling ticks map to the
  // same value so React's no-op status bail engages
  assert.equal(quantizeLiveCount(3_493), quantizeLiveCount(3_502));
  assert.equal(quantizeLiveCount(-5), 0, "negative/NaN guard");
});

test("needsRefetch: jitter pans inside coverage skip; real moves and zoom-band changes refetch", () => {
  // ~continent view: halfSpan larger than the 250nm serve ceiling → the
  // effective served radius is the ceiling (250nm ≈ 4.17°); 20% ≈ 0.83°
  const prev = fetchFootprint(40, -100, 3.6, 55, 25, -70, -130);
  assert.equal(needsRefetch(null, prev), true, "first fetch always runs");
  const jitter = fetchFootprint(40.2, -100.3, 3.6, 55, 25, -70, -130);
  assert.equal(needsRefetch(prev, jitter), false, "sub-degree jitter re-downloads nothing");
  const realMove = fetchFootprint(43, -110, 3.6, 58, 28, -80, -140);
  assert.equal(needsRefetch(prev, realMove), true, "leaving coverage refetches");
  const zoomed = fetchFootprint(40, -100, 4.2, 50, 30, -80, -120);
  assert.equal(needsRefetch(prev, zoomed), true, "zoom-band change refetches (density/radius changes)");

  // city view: halfSpan below the 50nm floor → floor radius (≈0.83°); 20% ≈ 0.17°
  const city = fetchFootprint(34, -118, 10, 34.2, 33.8, -117.7, -118.3);
  const cityJitter = fetchFootprint(34.05, -118.05, 10, 34.25, 33.85, -117.75, -118.35);
  assert.equal(needsRefetch(city, cityJitter), false, "city-scale jitter inside the 50nm floor skips");
  const cityMove = fetchFootprint(34.5, -118, 10, 34.7, 34.3, -117.7, -118.3);
  assert.equal(needsRefetch(city, cityMove), true, "half-degree move at city scale leaves the floor radius");
});
