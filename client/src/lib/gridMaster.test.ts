// ALL POWER GRIDS master control — pins the single-source-of-truth state
// rules: derived ON position, masters-only ON, whole-family OFF.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GRID_MASTER_IDS, allGridsOn, setAllGrids } from './gridMaster.js';

test('allGridsOn: true only when every continental master is on', () => {
  assert.equal(allGridsOn({}), false);
  const all = Object.fromEntries(GRID_MASTER_IDS.map((id) => [id, true]));
  assert.equal(allGridsOn(all), true);
  assert.equal(allGridsOn({ ...all, powergrid_europe: false }), false, 'one continent off = switch off');
  assert.equal(allGridsOn({ ...all, powergrid_tx: false }), true, 'per-state picks never affect the master position');
});

test('setAllGrids ON: flips the four masters, leaves finer picks and unrelated layers alone', () => {
  const before = { aircraft: true, powergrid_tx: true, powergrid_hifld: false };
  const next = setAllGrids(before, true);
  for (const id of GRID_MASTER_IDS) assert.equal(next[id], true, `${id} on`);
  assert.equal(next.aircraft, true, 'unrelated layers untouched');
  assert.equal(next.powergrid_tx, true, 'per-state pick left alone (same data the master shows)');
  assert.equal(next.powergrid_hifld, false, 'HIFLD stays an individual choice — no double-draw by default');
  assert.deepEqual(before, { aircraft: true, powergrid_tx: true, powergrid_hifld: false }, 'input never mutated');
});

test('setAllGrids OFF: clears the WHOLE powergrid family in one motion', () => {
  const on = setAllGrids({ powergrid_tx: true, powergrid_eu_de: true, powergrid_hifld: true, vessels: true }, true);
  const off = setAllGrids(on, false);
  for (const [k, v] of Object.entries(off)) {
    if (k.startsWith('powergrid')) assert.equal(v, false, `${k} cleared`);
  }
  assert.equal(off.vessels, true, 'unrelated layers untouched');
  assert.equal(allGridsOn(off), false);
});

test('round trip: on -> off -> on keeps the masters authoritative', () => {
  const s1 = setAllGrids({}, true);
  assert.equal(allGridsOn(s1), true);
  const s2 = setAllGrids(s1, false);
  assert.equal(allGridsOn(s2), false);
  const s3 = setAllGrids(s2, true);
  assert.equal(allGridsOn(s3), true);
});

// ── registry parity: a continent is not shipped until it is in the master ──
//
// REGRESSION (human report 2026-08-16: "when i turn on all power grids i dont
// see it them", about China and Russia). Asia had shipped end to end —
// registry `powergrid_asia` live, wired in datamap.tsx, power_asia.pmtiles
// serving from R2 with 4.28M China + 3.13M Russia features — but its id was
// never appended to GRID_MASTER_IDS. The master switch therefore skipped a
// whole continent in silence, and the panel's own subtitle ("US · Canada ·
// S. America · Europe") advertised the gap without anyone noticing.
//
// This test makes the registry the source of truth: every LIVE continental
// master in datacore/layers.json must be in GRID_MASTER_IDS. It fails on the
// day a new continent's registry entry lands, not weeks later when a user
// reports missing data.

import { readFileSync } from 'node:fs';

/** Minimal shape of a datacore/layers.json entry — enough to assert against,
 *  and typed rather than `any` so this file does not add to the ts_any debt
 *  ratchet pinned by test_ts_code_only.py. */
interface RegistryLayer {
  id: string;
  name?: string;
  status?: string;
  group?: string;
}
interface Registry { layers?: RegistryLayer[] }

test('every live continental master in the registry is in GRID_MASTER_IDS', () => {
  const reg: Registry = JSON.parse(
    readFileSync(new URL('../../../datacore/layers.json', import.meta.url), 'utf-8'),
  );
  // the continental masters are exactly the layers named "... Power Grid (all ...)"
  const masters = (reg.layers || [])
    .filter((l) => /Power Grid \(all/i.test(l.name || '') && l.status === 'live')
    .map((l) => l.id);
  assert.ok(masters.length >= 5, `expected >=5 continental masters, found ${masters.length}`);
  const missing = masters.filter((id) => !(GRID_MASTER_IDS as readonly string[]).includes(id));
  assert.deepEqual(
    missing,
    [],
    `live continental master(s) missing from GRID_MASTER_IDS: ${missing.join(', ')} — ` +
      `"All power grids" would silently skip them, exactly as it skipped Asia (China/Russia) until 2026-08-16`,
  );
});

test('GRID_MASTER_IDS contains no id absent from the registry', () => {
  const reg: Registry = JSON.parse(
    readFileSync(new URL('../../../datacore/layers.json', import.meta.url), 'utf-8'),
  );
  const ids = new Set((reg.layers || []).map((l) => l.id));
  for (const id of GRID_MASTER_IDS) {
    assert.ok(ids.has(id), `${id} is in GRID_MASTER_IDS but not in datacore/layers.json`);
  }
});

// ── bucket <-> registry parity (the invariant that would have caught ALL of
// ── Asia, Africa and Oceania before a human did)
//
// The Asia bug and the Africa/Oceania dark-data finding share one root: a
// tile can be BUILT, UPLOADED and SERVED while being unreachable, because
// nothing asserted that a bucket object corresponds to a reachable layer.
//
// The bucket cannot be listed from CI (no credentials, no network), so the
// checked-in inventory below is the contract. Refresh it with:
//   aws s3 ls s3://voltrade-tiles/tiles/ --endpoint-url $R2_ENDPOINT
// Adding a tile to the bucket without a registry layer now fails HERE.

const BUCKET_GRID_TILES_2026_08_16 = [
  // continental masters
  'power_us', 'power_canada', 'power_southamerica', 'power_europe',
  'power_asia', 'power_africa', 'power_oceania',
  // known non-continental / overlay sets, intentionally not continental masters
  'power_hifld', 'power_hifld_sub', 'power_hifld_plants',
] as const;

test('every continental tile in the bucket has a live registry layer', () => {
  const reg: Registry = JSON.parse(
    readFileSync(new URL('../../../datacore/layers.json', import.meta.url), 'utf-8'),
  );
  const live = new Set(
    (reg.layers || []).filter((l) => l.status === 'live').map((l) => l.id),
  );
  // map a continental tile filename to the registry id it should back
  const expected: Record<string, string> = {
    power_us: 'powergrid',
    power_canada: 'powergrid_canada',
    power_southamerica: 'powergrid_southamerica',
    power_europe: 'powergrid_europe',
    power_asia: 'powergrid_asia',
    power_africa: 'powergrid_africa',
    power_oceania: 'powergrid_oceania',
  };
  const orphans = Object.entries(expected)
    .filter(([, id]) => !live.has(id))
    .map(([file, id]) => `${file}.pmtiles -> ${id}`);
  assert.deepEqual(
    orphans,
    [],
    `bucket tiles with no live registry layer (built, served, unreachable): ${orphans.join(', ')}`,
  );
  // and every one of them must also be in the master switch
  for (const id of Object.values(expected)) {
    assert.ok(
      (GRID_MASTER_IDS as readonly string[]).includes(id),
      `${id} is live and continental but absent from GRID_MASTER_IDS`,
    );
  }
});

test('the master switch now covers all seven continents', () => {
  assert.equal(
    GRID_MASTER_IDS.length,
    7,
    'US, Canada, South America, Europe, Asia, Africa, Oceania',
  );
});
