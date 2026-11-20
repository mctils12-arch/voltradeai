// ALL POWER GRIDS master control — pins the single-source-of-truth state
// rules: derived ON position, masters-only ON, whole-family OFF.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {
  GRID_MASTER_IDS, allGridsOn, setAllGrids,
  isUsStateGridId, gridGroupState, countGridGroupOn, setGridGroup, toggleGridGroup,
} from './gridMaster.js';

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

// ── ALL STATES control (2026-11-19) — the tri-state rule over the panel's
// own `enabled` map, and the registry facts it rests on. ──
const STATES3 = ['powergrid_tx', 'powergrid_oh', 'powergrid_nd'];

test('gridGroupState: all / some / none, derived from the per-state values only', () => {
  assert.equal(gridGroupState({}, STATES3), 'none');
  assert.equal(gridGroupState({ powergrid_tx: false, powergrid_oh: false }, STATES3), 'none');
  assert.equal(gridGroupState({ powergrid_tx: true }, STATES3), 'some');
  assert.equal(gridGroupState({ powergrid_tx: true, powergrid_oh: true }, STATES3), 'some');
  assert.equal(gridGroupState({ powergrid_tx: true, powergrid_oh: true, powergrid_nd: true }, STATES3), 'all');
  assert.equal(countGridGroupOn({ powergrid_tx: true, powergrid_oh: true }, STATES3), 2);
  // the US master, HIFLD and other regions never move the states' control
  assert.equal(gridGroupState({ powergrid: true, powergrid_hifld: true, powergrid_ca_ab: true }, STATES3), 'none',
    'the US master on is not "all states on"');
  assert.equal(gridGroupState({}, []), 'none', 'an empty group has nothing on');
  assert.equal(gridGroupState({ powergrid_tx: true }, []), 'none');
});

test('setGridGroup: sets exactly the listed ids, leaves masters / other regions / unrelated layers alone', () => {
  const before = { aircraft: true, powergrid: true, powergrid_hifld: true, powergrid_ca_ab: true, powergrid_tx: true };
  const on = setGridGroup(before, STATES3, true);
  for (const id of STATES3) assert.equal(on[id], true, `${id} on`);
  assert.equal(gridGroupState(on, STATES3), 'all');
  const off = setGridGroup(on, STATES3, false);
  for (const id of STATES3) assert.equal(off[id], false, `${id} off`);
  for (const s of [on, off]) {
    assert.equal(s.aircraft, true, 'unrelated layer untouched');
    assert.equal(s.powergrid, true, 'US master untouched — off here is NOT the whole-family off');
    assert.equal(s.powergrid_hifld, true, 'HIFLD untouched');
    assert.equal(s.powergrid_ca_ab, true, 'another region untouched');
  }
  assert.deepEqual(before, { aircraft: true, powergrid: true, powergrid_hifld: true, powergrid_ca_ab: true, powergrid_tx: true },
    'input never mutated');
});

test('toggleGridGroup: none -> all, some -> all (the picks complete), all -> none', () => {
  const s1 = toggleGridGroup({}, STATES3);
  assert.equal(gridGroupState(s1, STATES3), 'all', 'none -> all');
  const s2 = toggleGridGroup(s1, STATES3);
  assert.equal(gridGroupState(s2, STATES3), 'none', 'all -> none');
  const some = { powergrid_tx: true, powergrid_oh: false, powergrid_nd: true };
  const s3 = toggleGridGroup(some, STATES3);
  assert.equal(gridGroupState(s3, STATES3), 'all', 'some -> all');
  assert.equal(gridGroupState(toggleGridGroup(s3, STATES3), STATES3), 'none', 'and the next press is a clean off');
});

test('a user\'s own state picks survive: they read "some" and stay on until the control is pressed', () => {
  const picked = { powergrid_tx: true, powergrid_oh: true, powergrid_nd: true, powergrid_ak: false };
  const all = [...STATES3, 'powergrid_ak'];
  assert.equal(gridGroupState(picked, all), 'some');
  assert.equal(countGridGroupOn(picked, all), 3);
  // the whole-family master switch going ON leaves them exactly as they were
  const afterMasterOn = setAllGrids(picked, true);
  for (const id of STATES3) assert.equal(afterMasterOn[id], true);
  assert.equal(gridGroupState(afterMasterOn, all), 'some');
  // and its OFF (the whole-family clear) reads "none" here
  assert.equal(gridGroupState(setAllGrids(afterMasterOn, false), all), 'none');
});

test('isUsStateGridId: a two-letter state code and nothing else', () => {
  for (const id of ['powergrid_tx', 'powergrid_ca', 'powergrid_dc', 'powergrid_nd']) assert.equal(isUsStateGridId(id), true, id);
  for (const id of ['powergrid', 'powergrid_canada', 'powergrid_hifld', 'powergrid_hifld_sub', 'powergrid_ca_ab',
    'powergrid_eu_de', 'powergrid_as_gcc', 'powergrid_asia', 'powergrid_all', 'grid_demand', 'powergrid_TX', 'xpowergrid_tx']) {
    assert.equal(isUsStateGridId(id), false, id);
  }
});

// REGISTRY PARITY for the by-state drop-down: the panel builds the section
// from the registry's `group: "grid"` and the All-states control from the
// members whose id is a state id. Those two sets must be the same 51 live
// layers — a non-state layer filed under "grid" would sit in the by-state
// list without being a state, and a state filed elsewhere would be missed
// by "all states".
test('registry: group "grid" is exactly the 50 states + DC, every one a live state id', () => {
  const registry = JSON.parse(fs.readFileSync(LAYERS_JSON, 'utf8')) as {
    layers: Array<{ id: string; status?: string; group?: string }>;
  };
  const inGroup = registry.layers.filter((l) => l.group === 'grid').map((l) => l.id).sort();
  const byShape = registry.layers.filter((l) => isUsStateGridId(l.id)).map((l) => l.id).sort();
  assert.deepEqual(inGroup, byShape, 'group "grid" and the state-id shape name the same layers');
  assert.equal(inGroup.length, 51, '50 states + DC');
  assert.ok(inGroup.includes('powergrid_dc') && inGroup.includes('powergrid_tx'));
  const notLive = registry.layers.filter((l) => l.group === 'grid' && l.status !== 'live').map((l) => l.id);
  assert.deepEqual(notLive, [], 'every state layer is live (the control only switches live layers)');
  // the non-state grid rows stay out of the by-state list
  for (const id of ['powergrid', 'powergrid_hifld', 'powergrid_hifld_sub', 'powergrid_hifld_plants', 'grid_demand', 'grid_generation']) {
    const l = registry.layers.find((x) => x.id === id);
    assert.ok(l, `${id} is in the registry`);
    assert.notEqual(l!.group, 'grid', `${id} is not filed in the by-state group`);
  }
});
