// Hermetic tests for orbital satellite click-picking (pure geometry, no
// DOM/WebGL/map). Run: npx tsx --test client/src/lib/orbital/pick.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pickNearestSatellite, pixelToleranceToMercUnits } from './pick.ts';
import { SENTINEL_SKIP, CLASS_CODE } from './satBuffer.ts';
import type { GpRecord } from './tle.ts';

function gp(noradId: number): GpRecord {
  return {
    noradId,
    name: `SAT-${noradId}`,
    epoch: null,
    inclination: null,
    raan: null,
    ecc: null,
    argp: null,
    meanAnomaly: null,
    meanMotion: null,
    bstar: null,
  };
}

test('pickNearestSatellite: finds the closest rendered point within tolerance', () => {
  const positions = new Float32Array([
    0.10, 0.10, 550000, CLASS_CODE.LEO, // i=0
    0.50, 0.50, 20000000, CLASS_CODE.MEO, // i=1 — click lands near here
    0.51, 0.49, 35786000, CLASS_CODE.GEO, // i=2 — slightly farther
  ]);
  const gps = [gp(1), gp(2), gp(3)];
  const hit = pickNearestSatellite(positions, 4, gps, 0.499, 0.499, 0.05);
  assert.ok(hit);
  assert.equal(hit!.index, 1);
  assert.equal(hit!.gp.noradId, 2);
  assert.equal(hit!.classCode, CLASS_CODE.MEO);
  assert.equal(hit!.altMeters, 20000000);
});

test('pickNearestSatellite: never snaps to a point outside tolerance (honest miss)', () => {
  const positions = new Float32Array([
    0.90, 0.90, 550000, CLASS_CODE.LEO,
  ]);
  const gps = [gp(1)];
  const hit = pickNearestSatellite(positions, 4, gps, 0.10, 0.10, 0.01);
  assert.equal(hit, null);
});

test('pickNearestSatellite: skips SENTINEL_SKIP slots even when nearest in raw distance', () => {
  const positions = new Float32Array([
    0.50, 0.50, 0, SENTINEL_SKIP, // i=0 — closest to the click, but not rendered
    0.55, 0.55, 550000, CLASS_CODE.LEO, // i=1 — farther, but the honest pick
  ]);
  const gps = [gp(1), gp(2)];
  const hit = pickNearestSatellite(positions, 4, gps, 0.50, 0.50, 0.2);
  assert.ok(hit);
  assert.equal(hit!.index, 1);
  assert.equal(hit!.gp.noradId, 2);
});

test('pickNearestSatellite: empty/all-sentinel population is always a clean miss', () => {
  const positions = new Float32Array([0, 0, 0, SENTINEL_SKIP]);
  assert.equal(pickNearestSatellite(positions, 4, [gp(1)], 0, 0, 1), null);
  assert.equal(pickNearestSatellite(new Float32Array(0), 4, [], 0.5, 0.5, 1), null);
});

test('pickNearestSatellite: caps the search at min(gp.length, buffer slots) — never reads past either', () => {
  // Buffer has 2 slots but gp only has 1 entry — must not throw or read gp[1].
  const positions = new Float32Array([
    0.5, 0.5, 550000, CLASS_CODE.LEO,
    0.6, 0.6, 550000, CLASS_CODE.LEO,
  ]);
  const gps = [gp(42)];
  const hit = pickNearestSatellite(positions, 4, gps, 0.5, 0.5, 0.05);
  assert.ok(hit);
  assert.equal(hit!.gp.noradId, 42);
});

// ── far-side occlusion filter (globe mode): cameraSphere parameter ──
// Camera over lon 0 / lat 0 at 3 earth radii from center — the same value
// SatLayer.getGlobeCamera() reconstructs from MapLibre's clipping plane
// [0, 0, 1, -1/3] (see occlusion.test.ts).
const CAM: readonly [number, number, number] = [0, 0, 3];

test('occlusion filter: a click on a far-side satellite is an honest miss in globe mode', () => {
  // lon 180 LEO — directly behind the earth from CAM.
  const positions = new Float32Array([1.0, 0.5, 550000, CLASS_CODE.LEO]);
  const gps = [gp(1)];
  // without the camera (mercator view): pickable as before
  const flat = pickNearestSatellite(positions, 4, gps, 1.0, 0.5, 0.05);
  assert.ok(flat);
  // with the camera (globe view): hidden behind the planet — never pickable
  const globe = pickNearestSatellite(positions, 4, gps, 1.0, 0.5, 0.05, CAM);
  assert.equal(globe, null);
});

test('occlusion filter is altitude-aware: at the same map spot past the limb, LEO is hidden but GEO is pickable', () => {
  // lon 110 is ~20° past the visible limb for a camera over lon 0 at d=3.
  // A LEO there is behind the disk; a GEO at the SAME lon/lat is far enough
  // out to be seen around the side of the earth — the case a plain
  // hemisphere test would get wrong.
  const mercX110 = (110 + 180) / 360;
  const positions = new Float32Array([
    mercX110, 0.5, 550000, CLASS_CODE.LEO, // i=0 — occluded
    mercX110, 0.5, 35786000, CLASS_CODE.GEO, // i=1 — visible past the limb
  ]);
  const gps = [gp(1), gp(2)];
  const hit = pickNearestSatellite(positions, 4, gps, mercX110, 0.5, 0.05, CAM);
  assert.ok(hit);
  assert.equal(hit!.index, 1);
  assert.equal(hit!.classCode, CLASS_CODE.GEO);
});

test('occlusion filter: null/omitted camera keeps the pre-globe behavior identical', () => {
  const positions = new Float32Array([1.0, 0.5, 550000, CLASS_CODE.LEO]);
  const gps = [gp(1)];
  const a = pickNearestSatellite(positions, 4, gps, 1.0, 0.5, 0.05);
  const b = pickNearestSatellite(positions, 4, gps, 1.0, 0.5, 0.05, null);
  assert.deepEqual(a, b);
  assert.ok(a);
});

test('pixelToleranceToMercUnits: halves per zoom level (slippy-map relation)', () => {
  const z5 = pixelToleranceToMercUnits(10, 5);
  const z6 = pixelToleranceToMercUnits(10, 6);
  assert.ok(Math.abs(z6 - z5 / 2) < 1e-12);
  // zoom 0: whole world is 256px, so 256px tolerance == the full [0,1] unit span.
  assert.ok(Math.abs(pixelToleranceToMercUnits(256, 0) - 1) < 1e-12);
});
