// Elevation lib pure core — tile addressing, terrarium decode, bilinear.
// Run: npx tsx --test client/src/lib/elevation.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { tileCoord, terrariumDecode, bilinearSample, ELEV_TILE_Z } from './elevation.js';

const near = (a: number, b: number, tol: number, msg: string) =>
  assert.ok(Math.abs(a - b) <= tol, `${msg} (${a} vs ${b})`);

test('tileCoord: known anchors on the web-mercator grid', () => {
  // lon 0 / lat 0 sits exactly at the grid center
  const c = tileCoord(0, 0, 9);
  near(c.x, 256, 1e-9, 'x center of 512 tiles at z9');
  near(c.y, 256, 1e-9, 'y center');
  // Denver (-104.99, 39.74) lands in z9 tile 106/194 (standard slippy math)
  const d = tileCoord(-104.99, 39.74, 9);
  assert.equal(Math.floor(d.x), 106, 'Denver tile x');
  assert.equal(Math.floor(d.y), 194, 'Denver tile y');
  // out-of-range latitude clamps instead of exploding
  const p = tileCoord(0, 89.9, 9);
  assert.ok(p.y >= 0 && Number.isFinite(p.y), 'polar latitude clamped');
});

test('terrarium decode: documented reference values', () => {
  near(terrariumDecode(128, 0, 0), 0, 1e-9, 'sea level = (128,0,0)');
  near(terrariumDecode(134, 160, 0), 1696, 1e-9, '1696 m');
  near(terrariumDecode(127, 255, 0), -1, 1e-9, '-1 m (one below sea level)');
  near(terrariumDecode(128, 0, 128), 0.5, 1e-9, 'blue channel = 1/256 m steps');
});

test('bilinear sample: interpolates and clamps at the edges', () => {
  // 2×2 grid: 0 10 / 20 30
  const g = new Int16Array([0, 10, 20, 30]);
  near(bilinearSample(g, 2, 0, 0), 0, 1e-9, 'corner exact');
  near(bilinearSample(g, 2, 0.999, 0.999), 30, 0.1, 'opposite corner');
  near(bilinearSample(g, 2, 0.5, 0.5), 15, 1e-9, 'center average');
  near(bilinearSample(g, 2, -5, -5), 0, 1e-9, 'clamps low');
  near(bilinearSample(g, 2, 99, 99), 30, 1e-9, 'clamps high (exact edge, no NaN)');
  // 1×1 grid (a fulfilled single-pixel tile) must return the pixel, not NaN
  const one = new Int16Array([500]);
  near(bilinearSample(one, 1, 0, 0), 500, 1e-9, '1×1 grid samples its only pixel');
  near(bilinearSample(one, 1, 0.7, 0.3), 500, 1e-9, '1×1 grid anywhere');
});

test('sampling zoom is coarse enough to keep long tracks in a few tiles', () => {
  // one z9 tile spans ~78 km at the equator — a 1,000 km track stays
  // within the cache cap with plenty of headroom
  assert.ok(ELEV_TILE_Z <= 10, 'AGL readout zoom stays coarse');
});
