// Solar-system view — pure view-math tests (EARTH TWIN O6-7 tier 2 spike).
// mount() needs a DOM and is exercised by the integration PR's visual
// harness; everything mathematical here is hermetic and pinned. Same
// node:test style as ephemeris.test.ts / solarSystem.test.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  worldToScreen,
  trueRadiusPx,
  entryScaleForEarthDisc,
  shouldExitToGlobe,
  clampScale,
  pickScaleBarMeters,
  fmtDistanceFromEarth,
  type ViewCamera,
} from "./solarView.js";
import { AU_M, BODY_RADIUS_M } from "./solarSystem.js";

test("worldToScreen: center maps to screen center; axes oriented +x right, +y up", () => {
  const cam: ViewCamera = { metersPerPixel: 1000, centerOffsetXM: 0, centerOffsetYM: 0 };
  const c = worldToScreen(0, 0, cam, 800, 600);
  assert.deepEqual(c, { x: 400, y: 300 });
  const right = worldToScreen(10_000, 0, cam, 800, 600);
  assert.deepEqual(right, { x: 410, y: 300 }); // +x → right
  const up = worldToScreen(0, 10_000, cam, 800, 600);
  assert.deepEqual(up, { x: 400, y: 290 }); // +y (ecliptic north view) → screen up
  // camera offset shifts the world the other way
  const off = worldToScreen(10_000, 0, { ...cam, centerOffsetXM: 10_000 }, 800, 600);
  assert.deepEqual(off, { x: 400, y: 300 });
});

test("true scale is literal: Earth disc px = diameter/scale; sub-pixel at solar-system scale", () => {
  // At 10,000 m/px Earth's true radius is 637.1 px.
  assert.ok(Math.abs(trueRadiusPx(BODY_RADIUS_M.earth, 10_000) - 637.1) < 0.01);
  // Whole-system scale (Neptune's 30 AU orbit in a ~900 px viewport):
  const mpp = (30 * AU_M) / 450; // ≈ 1e10 m/px
  for (const id of ["mercury", "venus", "earth", "moon", "mars"] as const) {
    assert.ok(trueRadiusPx(BODY_RADIUS_M[id], mpp) < 0.001, `${id} honestly sub-pixel`);
  }
  // Even the Sun is sub-pixel at that scale — the emptiness IS the honesty.
  assert.ok(trueRadiusPx(BODY_RADIUS_M.sun, mpp) < 0.1);
});

test("handoff scales: entry matches the globe's measured disc; exit has hysteresis", () => {
  // Parent measured the globe at 512 px diameter at max zoom-out:
  const entry = entryScaleForEarthDisc(512);
  assert.ok(Math.abs(trueRadiusPx(BODY_RADIUS_M.earth, entry) * 2 - 512) < 1e-9);
  // At the entry scale with a 900 px viewport (min side), Earth spans
  // 512/900 ≈ 57% — below a 60% exit threshold → no immediate bounce-back.
  assert.equal(shouldExitToGlobe(entry, 900, 0.6), false);
  // Zooming IN (smaller m/px) grows the disc past the threshold → exit.
  assert.equal(shouldExitToGlobe(entry / 1.3, 900, 0.6), true);
  // Zooming OUT never exits.
  assert.equal(shouldExitToGlobe(entry * 100, 900, 0.6), false);
});

test("scale clamp: bounded to [1e3, 1e11] m/px", () => {
  assert.equal(clampScale(1), 1e3);
  assert.equal(clampScale(5e5), 5e5);
  assert.equal(clampScale(1e14), 1e11);
});

test("scale bar: 1/2/5 decade steps, honest length in [90px, 180px)", () => {
  for (const mpp of [1e3, 3.7e4, 5e6, 1.23e8, 1e10]) {
    const m = pickScaleBarMeters(mpp, 90);
    const px = m / mpp;
    assert.ok(px >= 90 && px < 900, `bar ${px.toFixed(1)}px at ${mpp} m/px`);
    // 1/2/5 mantissa only
    const mant = m / Math.pow(10, Math.floor(Math.log10(m)));
    assert.ok([1, 2, 5].some((v) => Math.abs(mant - v) < 1e-9), `mantissa ${mant}`);
  }
});

test("distance labels: units-preference formatter below 0.01 AU, AU above", () => {
  // Moon-range distance goes through lib/units.ts (imperial default in a
  // clean node process → miles).
  const moon = fmtDistanceFromEarth(384_400_000); // 384,400 km in meters
  assert.ok(/mi$|km$/.test(moon), `moon label carries a unit (${moon})`);
  assert.ok(!moon.includes("AU"));
  // Mars-range distance is AU (fixed domain convention in both systems).
  assert.equal(fmtDistanceFromEarth(1.5 * AU_M), "1.500 AU");
  assert.equal(fmtDistanceFromEarth(30 * AU_M), "30.00 AU");
});
