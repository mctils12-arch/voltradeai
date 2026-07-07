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
} from './satBuffer.ts';

// --------------------------------------------------------------------------
// Layout constants
// --------------------------------------------------------------------------

test('layout constants are the locked contract', () => {
  assert.equal(SAT_STRIDE, 4);
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
    0.25, 0.5, 550000, CLASS_CODE.LEO, // i=0 valid LEO
    0, 0, 0, SENTINEL_SKIP, // i=1 skipped
    0.75, 0.4, 20000000, CLASS_CODE.MEO, // i=2 valid MEO
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
