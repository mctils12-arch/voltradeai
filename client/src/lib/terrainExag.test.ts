// Hermetic tests for the terrain vertical-exaggeration pref clamp (pure
// logic; no DOM, no localStorage — readTerrainExag's storage I/O is a thin
// try/catch shell over clampTerrainExag, which is the behavior under test).
// Run: npx tsx --test client/src/lib/terrainExag.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  clampTerrainExag,
  TERRAIN_EXAG_MIN,
  TERRAIN_EXAG_MAX,
  TERRAIN_EXAG_DEFAULT,
} from './terrainExag';

test('default is the historic production constant 1.3', () => {
  assert.equal(TERRAIN_EXAG_DEFAULT, 1.3);
});

test('in-range values pass through untouched', () => {
  assert.equal(clampTerrainExag(1.0), 1.0);
  assert.equal(clampTerrainExag(1.3), 1.3);
  assert.equal(clampTerrainExag(2.4), 2.4);
  assert.equal(clampTerrainExag(3.0), 3.0);
});

test('out-of-range values clamp to the documented bounds', () => {
  assert.equal(clampTerrainExag(0), TERRAIN_EXAG_MIN);
  assert.equal(clampTerrainExag(0.5), TERRAIN_EXAG_MIN);
  assert.equal(clampTerrainExag(-7), TERRAIN_EXAG_MIN);
  assert.equal(clampTerrainExag(3.1), TERRAIN_EXAG_MAX);
  assert.equal(clampTerrainExag(999), TERRAIN_EXAG_MAX);
});

test('non-finite input falls back to the default, never NaN', () => {
  assert.equal(clampTerrainExag(NaN), TERRAIN_EXAG_DEFAULT);
  assert.equal(clampTerrainExag(Infinity), TERRAIN_EXAG_DEFAULT);
  assert.equal(clampTerrainExag(-Infinity), TERRAIN_EXAG_DEFAULT);
});

test('bounds are sane: min ≤ default ≤ max', () => {
  assert.ok(TERRAIN_EXAG_MIN <= TERRAIN_EXAG_DEFAULT);
  assert.ok(TERRAIN_EXAG_DEFAULT <= TERRAIN_EXAG_MAX);
});
