// Hermetic tests for the site-wide unit-system preference + formatters
// (human directive 2026-07-13: imperial/metric switch, applied everywhere,
// inherited by every new data source).
// Run: npx tsx --test client/src/lib/units.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  fmtKm, fmtMeters, fmtMetersSmall, fmtMetersPerSec, fmtKmh, fmtCelsius, fmtKilograms, splitUnit,
  getUnits, setUnits, subscribeUnits, readUnitsPref,
} from './units.ts';

test('conversions are correct in both systems (explicit system arg)', () => {
  // 85 km ≈ 52.8 mi (screenshot's "Port of Charleston · 85 km")
  assert.equal(fmtKm(85, 0, 'imperial'), '53 mi');
  assert.equal(fmtKm(85, 0, 'metric'), '85 km');
  // 0.6 m wave ≈ 2.0 ft
  assert.equal(fmtMetersSmall(0.6, 'imperial'), '2.0 ft');
  assert.equal(fmtMetersSmall(0.6, 'metric'), '0.6 m');
  // 3.0 m/s ≈ 6.7 mph
  assert.equal(fmtMetersPerSec(3.0, 'imperial'), '6.7 mph');
  assert.equal(fmtMetersPerSec(3.0, 'metric'), '3.0 m/s');
  // 121 km/h ≈ 75 mph
  assert.equal(fmtKmh(121, 'imperial'), '75 mph');
  assert.equal(fmtKmh(121, 'metric'), '121 km/h');
  // 28.3 °C ≈ 82.9 °F
  assert.equal(fmtCelsius(28.3, 1, 'imperial'), '82.9 °F');
  assert.equal(fmtCelsius(28.3, 1, 'metric'), '28.3 °C');
  // altitude: 10000 m ≈ 32808 ft
  assert.equal(fmtMeters(10000, 0, 'imperial'), '32808 ft');
});

test('missing sensor readings stay "no data" — never coerced to 0', () => {
  for (const f of [
    () => fmtKm(null), () => fmtMeters(undefined), () => fmtMetersSmall(null),
    () => fmtMetersPerSec(null), () => fmtKmh(undefined), () => fmtCelsius(NaN),
  ]) assert.equal(f(), 'no data');
});

test('preference: default imperial, set/get round-trip, subscribers fire once per change', () => {
  assert.equal(readUnitsPref(), 'imperial'); // no localStorage in node → default
  let fires = 0;
  const unsub = subscribeUnits(() => { fires += 1; });
  setUnits('metric');
  assert.equal(getUnits(), 'metric');
  assert.equal(fmtKm(85), '85 km');          // formatters follow the live pref
  setUnits('metric');                         // no-op — must not re-fire
  assert.equal(fires, 1);
  setUnits('imperial');
  assert.equal(fires, 2);
  assert.equal(fmtKm(85), '53 mi');
  unsub();
  setUnits('metric');
  assert.equal(fires, 2);                     // unsubscribed
  setUnits('imperial');                       // restore default for other tests
});

// ── splitUnit (satellite-UX 2026-07-18: unit moves into the chip LABEL) ──
test('splitUnit re-typesets formatter output without changing the conversion', () => {
  assert.deepEqual(splitUnit('254 mi'), { num: '254', unit: 'mi' });
  assert.deepEqual(splitUnit('17139 mph'), { num: '17,139', unit: 'mph' });
  assert.deepEqual(splitUnit('27600 km/h'), { num: '27,600', unit: 'km/h' });
  assert.deepEqual(splitUnit('92.9 min'), { num: '92.9', unit: 'min' });
  assert.deepEqual(splitUnit('no data'), { num: 'no data', unit: null });
  assert.deepEqual(splitUnit('51.6°'), { num: '51.6°', unit: null });
});

// ── fmtKilograms (Census FT920 containerized vessel weight, 2026-08-20) ──
test('mass converts to short tons or metric tonnes and scales compactly', () => {
  // one live port-month: Los Angeles, 3,940,035,515 kg containerized
  assert.equal(fmtKilograms(3_940_035_515, 'metric'), '3.9M t');
  assert.equal(fmtKilograms(3_940_035_515, 'imperial'), '4.3M short tons');
  // the national aggregate is an order of magnitude larger and must not
  // collapse into the same bucket
  assert.equal(fmtKilograms(18_173_621_123, 'metric'), '18.2M t');
  // each decade of the scale ladder
  assert.equal(fmtKilograms(1_000, 'metric'), '1 t');
  assert.equal(fmtKilograms(1_500_000, 'metric'), '1.5K t');
  assert.equal(fmtKilograms(2_500_000_000_000, 'metric'), '2.5B t');
  assert.equal(fmtKilograms(0, 'metric'), '0 t');
  assert.equal(fmtKilograms(0, 'imperial'), '0 short tons');
});

test('a US short ton and a metric tonne differ by ~10% and are labelled apart', () => {
  // 907.18474 kg is exactly one short ton — and 0.91 of a metric tonne, which
  // is precisely why "tons" alone would be an ambiguous label
  assert.equal(fmtKilograms(907.18474, 'imperial'), '1 short tons');
  assert.equal(fmtKilograms(907.18474, 'metric'), '0.91 t');
  assert.notEqual(fmtKilograms(1e9, 'imperial'), fmtKilograms(1e9, 'metric'));
});

test('a missing mass is "no data", never a zero weight', () => {
  assert.equal(fmtKilograms(null), 'no data');
  assert.equal(fmtKilograms(undefined), 'no data');
  assert.equal(fmtKilograms(NaN), 'no data');
});

test('mass follows the live preference like every other formatter', () => {
  setUnits('metric');
  assert.equal(fmtKilograms(1_000_000), '1.0K t');
  setUnits('imperial');
  assert.ok(fmtKilograms(1_000_000).endsWith('short tons'));
});
