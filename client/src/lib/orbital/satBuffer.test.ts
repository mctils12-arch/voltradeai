// Hermetic tests for the shared satellite buffer layout (pure geometry +
// pack/unpack primitives). No Worker, no WebGL, no network, no clock.
// Run: npx tsx --test client/src/lib/orbital/satBuffer.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  SAT_STRIDE,
  SENTINEL_SKIP,
  CLASS_CODE,
  lonLatToMercator,
  mercatorToLonLat,
  readSatAt,
  wrapMercDelta,
} from './satBuffer.ts';

// --------------------------------------------------------------------------
// Layout constants
// --------------------------------------------------------------------------

test('layout constants are the locked contract', () => {
  // 7 since the glide extension (2026-07-17): [0..3] position+class keep
  // their historical offsets; [4..6] are the display-glide velocity.
  assert.equal(SAT_STRIDE, 7);
  assert.equal(SENTINEL_SKIP, -1);
  assert.deepEqual(CLASS_CODE, { LEO: 0, MEO: 1, GEO: 2 });
});

// --------------------------------------------------------------------------
// Mercator forward / inverse
// --------------------------------------------------------------------------

test('mercator: known anchor points', () => {
  // (lon=-180, lat=0) -> top-left-ish x=0, y=0.5 ; (lon=180) -> x=1
  const a = lonLatToMercator(-180, 0);
  assert.ok(Math.abs(a.x - 0) < 1e-9, `x ${a.x}`);
  assert.ok(Math.abs(a.y - 0.5) < 1e-9, `y ${a.y}`);
  const b = lonLatToMercator(180, 0);
  assert.ok(Math.abs(b.x - 1) < 1e-9, `x ${b.x}`);
  const c = lonLatToMercator(0, 0);
  assert.ok(Math.abs(c.x - 0.5) < 1e-9 && Math.abs(c.y - 0.5) < 1e-9);
});

test('mercator: round-trips across mid latitudes', () => {
  for (const lon of [-179, -90, -12.5, 0, 37.7, 121, 179]) {
    for (const lat of [-70, -33.9, 0, 12, 45.1, 66]) {
      const m = lonLatToMercator(lon, lat);
      const back = mercatorToLonLat(m.x, m.y);
      assert.ok(Math.abs(back.lonDeg - lon) < 1e-6, `lon ${lon} -> ${back.lonDeg}`);
      assert.ok(Math.abs(back.latDeg - lat) < 1e-6, `lat ${lat} -> ${back.latDeg}`);
    }
  }
});

test('mercator: Y clamps at the poles (no infinity)', () => {
  const np = lonLatToMercator(10, 89.9999);
  const sp = lonLatToMercator(10, -89.9999);
  assert.ok(np.y >= 0 && np.y <= 1 && Number.isFinite(np.y));
  assert.ok(sp.y >= 0 && sp.y <= 1 && Number.isFinite(sp.y));
});

// --------------------------------------------------------------------------
// readSatAt pack/unpack round-trip + sentinel
// --------------------------------------------------------------------------

test('readSatAt: reads the right stride slot and validity flag', () => {
  const buf = new Float32Array([
    // mercX, mercY, altMeters, classCode, velX, velY, velAlt
    0.25, 0.5, 550000, CLASS_CODE.LEO, 1e-4, -2e-4, 12.5, // i=0 valid LEO
    0, 0, 0, SENTINEL_SKIP, 0, 0, 0, // i=1 skipped
    0.75, 0.4, 20000000, CLASS_CODE.MEO, -5e-5, 3e-5, -1.25, // i=2 valid MEO
  ]);
  const a = readSatAt(buf, 0);
  assert.deepEqual(
    [a.mercX, a.mercY, a.altMeters, a.classCode, a.valid],
    [0.25, 0.5, 550000, 0, true],
  );
  const b = readSatAt(buf, 1);
  assert.equal(b.valid, false);
  assert.equal(b.classCode, SENTINEL_SKIP);
  const c = readSatAt(buf, 2);
  assert.equal(c.valid, true);
  assert.equal(c.classCode, CLASS_CODE.MEO);
});

test('readSatAt: velocity fields round-trip (float32 exactness for exact values)', () => {
  const buf = new Float32Array(2 * SAT_STRIDE);
  buf.set([0.5, 0.5, 400_000, CLASS_CODE.LEO, 0.0001220703125, -0.25, 12.5], 0);
  buf.set([0.1, 0.9, 35_786_000, CLASS_CODE.GEO, 0, 0, 0], SAT_STRIDE);
  const a = readSatAt(buf, 0);
  assert.equal(a.velX, 0.0001220703125, 'velX at [4]');
  assert.equal(a.velY, -0.25, 'velY at [5]');
  assert.equal(a.velAlt, 12.5, 'velAlt at [6]');
  const b = readSatAt(buf, 1);
  assert.deepEqual([b.velX, b.velY, b.velAlt], [0, 0, 0], 'zero velocity = honest hold');
});

// --------------------------------------------------------------------------
// wrapMercDelta — antimeridian-aware finite-difference delta
// --------------------------------------------------------------------------

test('wrapMercDelta: passes small deltas through, wraps antimeridian crossings', () => {
  assert.equal(wrapMercDelta(0.001), 0.001);
  assert.equal(wrapMercDelta(-0.001), -0.001);
  assert.equal(wrapMercDelta(0.5), 0.5, 'exactly half stays (boundary is exclusive)');
  // eastward crossing: x 0.999 -> 0.001 reads as +0.002, not -0.998
  assert.ok(Math.abs(wrapMercDelta(0.001 - 0.999) - 0.002) < 1e-12);
  // westward crossing: x 0.001 -> 0.999 reads as -0.002, not +0.998
  assert.ok(Math.abs(wrapMercDelta(0.999 - 0.001) - -0.002) < 1e-12);
});
