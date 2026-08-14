// Hermetic tests for the orbit-derived card values (satellite-UX 2026-07-18).
// Ground truth: ISS elements from the human-approved design doc's TLE
// (n=15.49401687 rev/d, e=0.0004791) — known-good published values:
// period ~92.9 min, altitude ~410-420 km, speed ~27,600 km/h (~7.66 km/s).
// Run: npx tsx --test client/src/lib/orbital/satDerived.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  semiMajorAxisKm, apsidesKm, orbitalSpeedKmh, periodMinutes, EARTH_EQUATORIAL_RADIUS_KM,
} from './satDerived.ts';

const ISS_N = 15.49401687;
const ISS_E = 0.0004791;

test('ISS semi-major axis / apsides match published values', () => {
  const a = semiMajorAxisKm(ISS_N)!;
  assert.ok(Math.abs(a - 6795) < 15, `a=${a} expected ~6795 km`);
  const aps = apsidesKm(ISS_N, ISS_E)!;
  assert.ok(aps.apogeeKm > aps.perigeeKm, 'apogee above perigee');
  assert.ok(Math.abs(aps.apogeeKm - 420) < 25, `apogee=${aps.apogeeKm} expected ~420 km`);
  assert.ok(Math.abs(aps.perigeeKm - 413) < 25, `perigee=${aps.perigeeKm} expected ~413 km`);
});

test('ISS period ~92.9 min', () => {
  const p = periodMinutes(ISS_N)!;
  assert.ok(Math.abs(p - 92.9) < 0.2, `period=${p}`);
});

test('ISS speed ~27,600 km/h, vis-viva at live altitude close to circular', () => {
  const vCirc = orbitalSpeedKmh(ISS_N, null)!;
  assert.ok(Math.abs(vCirc - 27580) < 300, `vCirc=${vCirc}`);
  const vLive = orbitalSpeedKmh(ISS_N, 417)!;
  assert.ok(Math.abs(vLive - vCirc) < 200, `vLive=${vLive} vs vCirc=${vCirc}`);
});

test('GEO sanity: n=1.0027 rev/d -> ~35,786 km apsides, ~11,070 km/h', () => {
  const aps = apsidesKm(1.0027, 0.0001)!;
  assert.ok(Math.abs(aps.apogeeKm - 35786) < 120, `geo apogee=${aps.apogeeKm}`);
  const v = orbitalSpeedKmh(1.0027, 35786)!;
  assert.ok(Math.abs(v - 11068) < 150, `geo v=${v}`);
});

test('degenerate elements return null, never NaN', () => {
  assert.equal(semiMajorAxisKm(null), null);
  assert.equal(semiMajorAxisKm(0), null);
  assert.equal(apsidesKm(15.5, null), null);
  assert.equal(apsidesKm(15.5, 1.2), null);
  assert.equal(orbitalSpeedKmh(null, 400), null);
  assert.equal(periodMinutes(-3), null);
});
