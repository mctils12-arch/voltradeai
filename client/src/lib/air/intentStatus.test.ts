// AIRCRAFT INTENT C1 — the card's plain-English status line: only fresh
// broadcast fields, never "autopilot off" for an aircraft that sends no mode
// bits, inferred states flagged, units through lib/units.
// Run: npx tsx --test client/src/lib/air/intentStatus.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { intentStatusParts, fmtSelectedAlt, type IntentViewWire } from './intentStatus.js';

const txt = (v: IntentViewWire | null, sys: 'imperial' | 'metric' = 'imperial') =>
  intentStatusParts(v, sys).map((p) => p.text + (p.inferred ? ' [inferred]' : ''));

test('European airliner (everything sent): autopilot + modes, level at FL370, FMS, heading, QNH, wind, OAT, Mach', () => {
  const v: IntentViewWire = { values: {
    nav_modes: ['autopilot', 'vnav', 'lnav', 'tcas'], nav_altitude_mcp: 37008, nav_altitude_fms: 37000,
    nav_heading: 87.9, nav_qnh: 1013.6, wd: 270, ws: 50, oat: -53, mach: 0.788, ias: 258, alt_baro: 37000, baro_rate: 0,
  } };
  assert.deepEqual(txt(v), [
    'Autopilot on · VNAV LNAV TCAS', 'Level at FL370 (selected)', 'FMS FL370', 'Selected heading 088°',
    'QNH 1014 hPa', 'Wind 270° 50 kt', 'OAT -63 °F', 'M0.79',
  ]);
  assert.ok(txt(v, 'metric').includes('OAT -53 °C'), 'temperature follows the units preference');
});

test('US airliner (selected altitude + QNH only): climbing/descending wording; no autopilot claim without mode bits', () => {
  const climb: IntentViewWire = { values: { nav_altitude_mcp: 10016, nav_qnh: 1008.2, alt_baro: 6000, baro_rate: 1800 } };
  assert.deepEqual(txt(climb), ['Climbing to 10000 ft (selected)', 'QNH 1008 hPa']);
  assert.deepEqual(txt(climb, 'metric'), ['Climbing to 3048 m (selected)', 'QNH 1008 hPa']);
  const desc: IntentViewWire = { values: { nav_altitude_mcp: 24000, alt_baro: 31000, baro_rate: -2200 } };
  assert.deepEqual(txt(desc), ['Descending to FL240 (selected)']);
  const notMoving: IntentViewWire = { values: { nav_altitude_mcp: 24000, alt_baro: 31000, baro_rate: 0 } };
  assert.deepEqual(txt(notMoving), ['Selected altitude FL240']);
  assert.ok(!txt(climb).some((t) => /autopilot/i.test(t)), 'never "autopilot off" for an aircraft that sends no mode bits');
});

test('mode bits sent with none engaged -> "Autopilot off"; vectored is flagged as an inference', () => {
  assert.deepEqual(txt({ values: { nav_modes: [] } }), ['Autopilot off']);
  const vec: IntentViewWire = {
    values: { nav_modes: ['autopilot', 'vnav'], nav_heading: 270 },
    inferred: { vectored: { active: true, reason: 'vectored', heading_source: 'track', deviation_deg: 2 } },
  };
  assert.deepEqual(txt(vec), ['Autopilot on · VNAV', 'Selected heading 270°', 'Vectored [inferred]']);
  const notVec = { ...vec, inferred: { vectored: { active: false, reason: 'lnav_engaged', heading_source: null, deviation_deg: null } } };
  assert.ok(!txt(notVec).some((t) => t.startsWith('Vectored')));
});

test('alerts first: TCAS RA and emergency status', () => {
  const parts = intentStatusParts({ values: { acas_ra: { advisory: 'Climb' }, emergency: 'general', nav_qnh: 1013 } }, 'imperial');
  assert.deepEqual(parts.map((p) => [p.text, !!p.alert]), [['TCAS RA: Climb', true], ['Emergency: general', true], ['QNH 1013 hPa', false]]);
  assert.deepEqual(txt({ values: { emergency: 'none' } }), []);
});

test('nothing fresh -> nothing shown; FL formatting', () => {
  assert.deepEqual(intentStatusParts(null), []);
  assert.deepEqual(intentStatusParts({ values: {} }), []);
  assert.equal(fmtSelectedAlt(18000, 'imperial'), 'FL180');
  assert.equal(fmtSelectedAlt(5016, 'imperial'), '5000 ft');
  assert.equal(fmtSelectedAlt(9000, 'imperial'), '9000 ft');
});
