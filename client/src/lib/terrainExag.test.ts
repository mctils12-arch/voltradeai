// Hermetic tests for the /data map's terrain vertical-exaggeration pref
// store (3D TERRAIN UPGRADES handoff §A). Guards the invariant that matters
// downstream: the persisted multiplier is the SAME number the aircraft/trail
// altitude datum locks to, so it must always be a finite value in [1,3] and
// must default to the shipped 1.3× when unset — never silently drift.
// Run: npx tsx --test client/src/lib/terrainExag.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  TERRAIN_EXAG_KEY, TERRAIN_EXAG_DEFAULT, TERRAIN_EXAG_MIN, TERRAIN_EXAG_MAX,
  clampTerrainExag, readTerrainExag, writeTerrainExag,
} from './terrainExag.ts';

// Minimal localStorage shim (node has none) — installed per-test so a case
// can also exercise the no-storage / throwing paths.
function installStore(): Record<string, string> {
  const backing: Record<string, string> = {};
  (globalThis as any).window = {
    localStorage: {
      getItem: (k: string) => (k in backing ? backing[k] : null),
      setItem: (k: string, v: string) => { backing[k] = String(v); },
      removeItem: (k: string) => { delete backing[k]; },
    },
  };
  return backing;
}

test('default is the shipped 1.3× and the band is 1..3', () => {
  assert.equal(TERRAIN_EXAG_DEFAULT, 1.3);
  assert.equal(TERRAIN_EXAG_MIN, 1);
  assert.equal(TERRAIN_EXAG_MAX, 3);
});

test('clamp holds the [1,3] band; non-finite falls back to default', () => {
  assert.equal(clampTerrainExag(1.3), 1.3);
  assert.equal(clampTerrainExag(2.4), 2.4);
  assert.equal(clampTerrainExag(0.2), 1);      // below floor
  assert.equal(clampTerrainExag(9), 3);        // above ceiling
  assert.equal(clampTerrainExag(1), 1);        // exact edges preserved
  assert.equal(clampTerrainExag(3), 3);
  assert.equal(clampTerrainExag(NaN), TERRAIN_EXAG_DEFAULT);
  assert.equal(clampTerrainExag(Infinity), TERRAIN_EXAG_DEFAULT);
});

test('read: default when unset, no window, or corrupt; clamps stored values', () => {
  delete (globalThis as any).window;           // no window at all → default
  assert.equal(readTerrainExag(), TERRAIN_EXAG_DEFAULT);

  const store = installStore();
  assert.equal(readTerrainExag(), TERRAIN_EXAG_DEFAULT); // unset key → default

  store[TERRAIN_EXAG_KEY] = 'not-a-number';
  assert.equal(readTerrainExag(), TERRAIN_EXAG_DEFAULT); // corrupt → default

  store[TERRAIN_EXAG_KEY] = '2.5';
  assert.equal(readTerrainExag(), 2.5);

  store[TERRAIN_EXAG_KEY] = '99';               // out-of-band persisted value
  assert.equal(readTerrainExag(), 3);           // read clamps
});

test('write persists a clamped value that read round-trips (reload persists)', () => {
  const store = installStore();
  writeTerrainExag(2.1);
  assert.equal(store[TERRAIN_EXAG_KEY], '2.1');
  assert.equal(readTerrainExag(), 2.1);

  writeTerrainExag(7);                           // clamp on the way in
  assert.equal(store[TERRAIN_EXAG_KEY], '3');
  assert.equal(readTerrainExag(), 3);

  writeTerrainExag(0);                           // clamp low
  assert.equal(store[TERRAIN_EXAG_KEY], '1');
  assert.equal(readTerrainExag(), 1);
});

test('write swallows storage failures (private mode) without throwing', () => {
  (globalThis as any).window = {
    localStorage: { setItem: () => { throw new Error('QuotaExceeded'); }, getItem: () => null },
  };
  assert.doesNotThrow(() => writeTerrainExag(2.0));
});
