// Hermetic tests for orbital satellite click-picking (pure geometry, no
// DOM/WebGL/map). Run: npx tsx --test client/src/lib/orbital/pick.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pickNearestSatellite, pickNearestSatelliteScreen, pickNearestSatelliteScreenMercator, pixelToleranceToMercUnits } from './pick.ts';
import { SAT_STRIDE, SENTINEL_SKIP, CLASS_CODE } from './satBuffer.ts';
import type { GpRecord } from './tle.ts';
import { mercatorToSphere, mercatorZFromAltitude } from './occlusion.ts';

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

/** Pack [mercX, mercY, altMeters, classCode] rows into a SAT_STRIDE buffer
 *  (velocity fields zero — picking reads only the exact-tick position). */
function buf(rows: Array<[number, number, number, number]>): Float32Array {
  const b = new Float32Array(rows.length * SAT_STRIDE);
  rows.forEach((r, i) => b.set(r, i * SAT_STRIDE));
  return b;
}

test('pickNearestSatellite: finds the closest rendered point within tolerance', () => {
  const positions = buf([
    [0.10, 0.10, 550000, CLASS_CODE.LEO], // i=0
    [0.50, 0.50, 20000000, CLASS_CODE.MEO], // i=1 — click lands near here
    [0.51, 0.49, 35786000, CLASS_CODE.GEO], // i=2 — slightly farther
  ]);
  const gps = [gp(1), gp(2), gp(3)];
  const hit = pickNearestSatellite(positions, SAT_STRIDE, gps, 0.499, 0.499, 0.05);
  assert.ok(hit);
  assert.equal(hit!.index, 1);
  assert.equal(hit!.gp.noradId, 2);
  assert.equal(hit!.classCode, CLASS_CODE.MEO);
  assert.equal(hit!.altMeters, 20000000);
});

test('pickNearestSatellite: never snaps to a point outside tolerance (honest miss)', () => {
  const positions = buf([
    [0.90, 0.90, 550000, CLASS_CODE.LEO],
  ]);
  const gps = [gp(1)];
  const hit = pickNearestSatellite(positions, SAT_STRIDE, gps, 0.10, 0.10, 0.01);
  assert.equal(hit, null);
});

test('pickNearestSatellite: skips SENTINEL_SKIP slots even when nearest in raw distance', () => {
  const positions = buf([
    [0.50, 0.50, 0, SENTINEL_SKIP], // i=0 — closest to the click, but not rendered
    [0.55, 0.55, 550000, CLASS_CODE.LEO], // i=1 — farther, but the honest pick
  ]);
  const gps = [gp(1), gp(2)];
  const hit = pickNearestSatellite(positions, SAT_STRIDE, gps, 0.50, 0.50, 0.2);
  assert.ok(hit);
  assert.equal(hit!.index, 1);
  assert.equal(hit!.gp.noradId, 2);
});

test('pickNearestSatellite: empty/all-sentinel population is always a clean miss', () => {
  const positions = buf([[0, 0, 0, SENTINEL_SKIP]]);
  assert.equal(pickNearestSatellite(positions, SAT_STRIDE, [gp(1)], 0, 0, 1), null);
  assert.equal(pickNearestSatellite(new Float32Array(0), SAT_STRIDE, [], 0.5, 0.5, 1), null);
});

test('pickNearestSatellite: caps the search at min(gp.length, buffer slots) — never reads past either', () => {
  // Buffer has 2 slots but gp only has 1 entry — must not throw or read gp[1].
  const positions = buf([
    [0.5, 0.5, 550000, CLASS_CODE.LEO],
    [0.6, 0.6, 550000, CLASS_CODE.LEO],
  ]);
  const gps = [gp(42)];
  const hit = pickNearestSatellite(positions, SAT_STRIDE, gps, 0.5, 0.5, 0.05);
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
  const positions = buf([[1.0, 0.5, 550000, CLASS_CODE.LEO]]);
  const gps = [gp(1)];
  // without the camera (mercator view): pickable as before
  const flat = pickNearestSatellite(positions, SAT_STRIDE, gps, 1.0, 0.5, 0.05);
  assert.ok(flat);
  // with the camera (globe view): hidden behind the planet — never pickable
  const globe = pickNearestSatellite(positions, SAT_STRIDE, gps, 1.0, 0.5, 0.05, CAM);
  assert.equal(globe, null);
});

test('occlusion filter is altitude-aware: at the same map spot past the limb, LEO is hidden but GEO is pickable', () => {
  // lon 110 is ~20° past the visible limb for a camera over lon 0 at d=3.
  // A LEO there is behind the disk; a GEO at the SAME lon/lat is far enough
  // out to be seen around the side of the earth — the case a plain
  // hemisphere test would get wrong.
  const mercX110 = (110 + 180) / 360;
  const positions = buf([
    [mercX110, 0.5, 550000, CLASS_CODE.LEO], // i=0 — occluded
    [mercX110, 0.5, 35786000, CLASS_CODE.GEO], // i=1 — visible past the limb
  ]);
  const gps = [gp(1), gp(2)];
  const hit = pickNearestSatellite(positions, SAT_STRIDE, gps, mercX110, 0.5, 0.05, CAM);
  assert.ok(hit);
  assert.equal(hit!.index, 1);
  assert.equal(hit!.classCode, CLASS_CODE.GEO);
});

test('occlusion filter: null/omitted camera keeps the pre-globe behavior identical', () => {
  const positions = buf([[1.0, 0.5, 550000, CLASS_CODE.LEO]]);
  const gps = [gp(1)];
  const a = pickNearestSatellite(positions, SAT_STRIDE, gps, 1.0, 0.5, 0.05);
  const b = pickNearestSatellite(positions, SAT_STRIDE, gps, 1.0, 0.5, 0.05, null);
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

test('pickNearestSatelliteScreen: altitude displacement respected — the MEO object under the cursor wins over the LEO whose nadir is there', () => {
  // Identity-like projection: clip = position (orthographic stand-in). Two
  // objects share the SAME ground point; the high one renders displaced
  // along the sphere normal — in screen space they are far apart.
  const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const gp = [{ noradId: 1, name: 'LEO' }, { noradId: 2, name: 'MEO' }] as any[];
  // same mercator ground point, wildly different altitudes
  const positions = buf([
    [0.25, 0.5, 400_000, CLASS_CODE.LEO],
    [0.25, 0.5, 20_000_000, CLASS_CODE.MEO],
  ]);
  const W = 1000, H = 1000;
  const toScreen = (p: number[]) => [((p[0] + 1) / 2) * W, ((1 - p[1]) / 2) * H];
  const sMeo = toScreen(mercatorToSphere(0.25, 0.5, 20_000_000));
  const sLeo = toScreen(mercatorToSphere(0.25, 0.5, 400_000));
  assert.ok(Math.hypot(sMeo[0] - sLeo[0], sMeo[1] - sLeo[1]) > 50,
    'the two RENDER far apart on screen (the whole point of the fix)');
  // click exactly on the MEO's rendered position → MEO, not the LEO
  const hit = pickNearestSatelliteScreen(positions, SAT_STRIDE, gp, I, sMeo[0], sMeo[1], W, H, 12);
  assert.equal(hit?.gp.name, 'MEO', 'the object under the cursor wins');
  const hit2 = pickNearestSatelliteScreen(positions, SAT_STRIDE, gp, I, sLeo[0], sLeo[1], W, H, 12);
  assert.equal(hit2?.gp.name, 'LEO');
  // far from both = honest miss; sentinel slots stay unpickable
  assert.equal(pickNearestSatelliteScreen(positions, SAT_STRIDE, gp, I, 5, 5, W, H, 12), null);
  const masked = positions.slice(); masked[SAT_STRIDE + 3] = SENTINEL_SKIP;
  assert.equal(pickNearestSatelliteScreen(masked, SAT_STRIDE, gp, I, sMeo[0], sMeo[1], W, H, 12), null,
    'sentinel (filtered/unrendered) slots are never pickable');
});

test('pickNearestSatelliteScreenMercator: mercator-unit z (never meters) — the tilted-terrain click fix', () => {
  // Matrix with an x-shear on z so altitude visibly displaces the screen
  // position: clip.x = mx + 1000·z, clip.y = my, w = 1. With z in MERCATOR
  // units (alt/(2πR·cos φ)) the shear is a modest on-screen shift; feeding
  // raw METERS would fling it off-screen by ~10⁵ px (the 2026-07-20 bug).
  const M = [1, 0, 0, 0, 0, 1, 0, 0, 1000, 0, 1, 0, 0, 0, 0, 1];
  const gp = [{ noradId: 1, name: 'SAT' }] as any[];
  const positions = buf([[0.25, 0.5, 400_000, CLASS_CODE.LEO]]);
  const W = 1000, H = 1000;
  const z = mercatorZFromAltitude(400_000, 0.5);
  assert.ok(z > 0.0099 && z < 0.0101, `mercZ at the equator ≈ alt/2πR (${z})`);
  const sx = ((0.25 + 1000 * z + 1) / 2) * W;   // displaced by the shear
  const sy = ((1 - 0.5) / 2) * H;
  const hit = pickNearestSatelliteScreenMercator(positions, SAT_STRIDE, gp, M, sx, sy, W, H, 12);
  assert.equal(hit?.gp.name, 'SAT', 'click at the DISPLACED render position hits');
  const groundSx = ((0.25 + 1) / 2) * W; // where the nadir would be — no object drawn there
  assert.equal(pickNearestSatelliteScreenMercator(positions, SAT_STRIDE, gp, M, groundSx, sy, W, H, 12), null,
    'the ground point is an honest miss — the object does not render there');
  // sentinel stays unpickable in the mercator path too
  const masked = positions.slice(); masked[3] = SENTINEL_SKIP;
  assert.equal(pickNearestSatelliteScreenMercator(masked, SAT_STRIDE, gp, M, sx, sy, W, H, 12), null);
});

test('pickNearestSatelliteScreenMercator: cameraSphere excludes far-side satellites mid-transition (2026-07-31)', () => {
  // The mercator screen path serves the whole zoom-driven globe↔mercator
  // transition band — a camera parked there had clicks selecting satellites
  // on the other side of the planet. Identity matrix: screen pos derives
  // straight from mercator coords, so the click lands on the LEO at lon 180
  // — which is directly behind the earth from a camera over lon 0 (CAM).
  const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const gpArr = [{ noradId: 1, name: 'FARSIDE-LEO' }] as any[];
  const positions = buf([[1.0, 0.5, 550_000, CLASS_CODE.LEO]]);
  const W = 1000, H = 1000;
  const sx = ((1.0 + 1) / 2) * W;
  const sy = ((1 - 0.5) / 2) * H;
  // without the camera (pure mercator): pickable, exactly as before
  const flat = pickNearestSatelliteScreenMercator(positions, SAT_STRIDE, gpArr, I, sx, sy, W, H, 12);
  assert.ok(flat, 'pure mercator keeps the whole-sky behavior');
  assert.equal(pickNearestSatelliteScreenMercator(positions, SAT_STRIDE, gpArr, I, sx, sy, W, H, 12, null)?.gp.name,
    'FARSIDE-LEO', 'explicit null camera = identical');
  // with the mid-transition camera: hidden behind the planet — honest miss
  assert.equal(pickNearestSatelliteScreenMercator(positions, SAT_STRIDE, gpArr, I, sx, sy, W, H, 12, CAM), null,
    'a satellite the shader culls must never be click-selected');
});

test('mercatorZFromAltitude: latitude scaling + pole guard', () => {
  const zEq = mercatorZFromAltitude(1000, 0.5);            // equator
  const zMid = mercatorZFromAltitude(1000, 0.3);           // ~55°N → smaller circumference → bigger z
  assert.ok(zMid > zEq, 'higher latitude → larger mercator z for the same meters');
  assert.ok(Number.isFinite(mercatorZFromAltitude(1000, 0)), 'pole clamp keeps it finite');
  assert.equal(mercatorZFromAltitude(0, 0.42), 0, 'zero altitude → zero z');
});
