// Aircraft operator filter — EARTH TWIN O6-4. Pins the prefix semantics and
// the honest disclosure the transform must carry.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { AIRLINE_PRESETS, filterAircraftByPrefix, applyAirlineFilter } from './airFilter.js';

const ROWS = [
  { callsign: 'AAL123', lat: 1 },
  { callsign: 'AAL9', lat: 2 },
  { callsign: 'UAL55', lat: 3 },
  { callsign: 'aal77', lat: 4 },   // lowercase feed value still matches
  { callsign: '', lat: 5 },
  { callsign: null, lat: 6 },
] as any[];

test('filterAircraftByPrefix: prefix match, case-insensitive, null/empty prefix = passthrough', () => {
  assert.equal(filterAircraftByPrefix(ROWS, 'AAL').length, 3);
  assert.equal(filterAircraftByPrefix(ROWS, 'ual').length, 1);
  assert.equal(filterAircraftByPrefix(ROWS, null), ROWS);
  assert.equal(filterAircraftByPrefix(ROWS, '  '), ROWS, 'blank = no filter');
  assert.equal(filterAircraftByPrefix(ROWS, 'ZZZ').length, 0, 'no match = honest empty, not an error');
});

test('applyAirlineFilter: filters the payload, corrects the count, discloses what happened', () => {
  const d = { aircraft: ROWS, count: 6, time: 't1', source: 'x' };
  const out = applyAirlineFilter(d, 'AAL');
  assert.equal(out.aircraft.length, 3);
  assert.equal(out.count, 3);
  assert.ok(/callsign filter AAL\*/.test(out.filter_note), 'names the filter');
  assert.ok(/3 of 6/.test(out.filter_note), 'names both numbers');
  assert.equal(out.time, 't1', 'delta cursor fields untouched');
  assert.equal(applyAirlineFilter(d, null), d, 'no filter = identity');
  assert.equal(d.aircraft.length, 6, 'input never mutated');
});

test('presets: ICAO designators, unique keys and prefixes', () => {
  const keys = new Set(AIRLINE_PRESETS.map((p) => p.key));
  const prefixes = new Set(AIRLINE_PRESETS.map((p) => p.prefix));
  assert.equal(keys.size, AIRLINE_PRESETS.length);
  assert.equal(prefixes.size, AIRLINE_PRESETS.length);
  for (const p of AIRLINE_PRESETS) assert.match(p.prefix, /^[A-Z]{3}$/, `${p.key}: 3-letter ICAO designator`);
  assert.ok(AIRLINE_PRESETS.some((p) => p.prefix === 'AAL'), 'the directive\'s "all American flights" preset exists');
});
