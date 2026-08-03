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
