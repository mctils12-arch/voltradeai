// Hermetic tests for the site-wide unit-system preference + formatters
// (human directive 2026-07-13: imperial/metric switch, applied everywhere,
// inherited by every new data source).
// Run: npx tsx --test client/src/lib/units.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  fmtKm, fmtMeters, fmtMetersSmall, fmtMetersPerSec, fmtKmh, fmtCelsius,
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
