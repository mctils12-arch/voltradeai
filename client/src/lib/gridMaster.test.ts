// ALL POWER GRIDS master control — pins the single-source-of-truth state
// rules: derived ON position, masters-only ON, whole-family OFF.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { GRID_MASTER_IDS, allGridsOn, setAllGrids } from './gridMaster.js';

const repoRoot = path.resolve(import.meta.dirname, '..', '..', '..');
const LAYERS_JSON = path.join(repoRoot, 'datacore', 'layers.json');

// Every continent the OSM power-grid pipeline could plausibly ship a
// dedicated master for. Not every name here exists yet — the test below
// only complains about the ones that ARE live in the registry.
const CONTINENT_SUFFIXES = ['canada', 'southamerica', 'europe', 'asia', 'africa', 'oceania'];

test('allGridsOn: true only when every continental master is on', () => {
  assert.equal(allGridsOn({}), false);
  const all = Object.fromEntries(GRID_MASTER_IDS.map((id) => [id, true]));
  assert.equal(allGridsOn(all), true);
  assert.equal(allGridsOn({ ...all, powergrid_europe: false }), false, 'one continent off = switch off');
  assert.equal(allGridsOn({ ...all, powergrid_tx: false }), true, 'per-state picks never affect the master position');
});

test('setAllGrids ON: flips every GRID_MASTER_IDS master, leaves finer picks and unrelated layers alone', () => {
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

// REGISTRY PARITY (2026-08-31, closing the item found 2026-08-19 in PR #817
// and re-confirmed still present 2026-08-28: `powergrid_asia` shipped live
// 2026-08-13 and sat unlisted here for 18 days — the master switch and the
// "All power grids" toggle both silently omitted an entire live continent).
// This asserts the drift can't recur silently: any future continental
// master (powergrid_africa, powergrid_oceania, ...) that ships `status:
// "live"` in the registry MUST be added to GRID_MASTER_IDS in the same PR,
// or this test fails the build instead of the gap sitting unnoticed again.
test('GRID_MASTER_IDS lists every live continental master in datacore/layers.json', () => {
  const registry = JSON.parse(fs.readFileSync(LAYERS_JSON, 'utf8')) as {
    layers: Array<{ id: string; status?: string }>;
  };
  const byId = new Map(registry.layers.map((l) => [l.id, l]));
  const masterSet = new Set<string>(GRID_MASTER_IDS);

  assert.ok(byId.get('powergrid')?.status === 'live', 'sanity: the base US master is live in the registry');
  assert.ok(masterSet.has('powergrid'), 'sanity: the base US master is in GRID_MASTER_IDS');

  const missing: string[] = [];
  for (const suffix of CONTINENT_SUFFIXES) {
    const id = `powergrid_${suffix}`;
    const layer = byId.get(id);
    if (layer && layer.status === 'live' && !masterSet.has(id)) missing.push(id);
  }
  assert.deepEqual(missing, [], `live continental master(s) missing from GRID_MASTER_IDS: ${missing.join(', ')}`);
});
