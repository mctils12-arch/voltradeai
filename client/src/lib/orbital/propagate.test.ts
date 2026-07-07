// Hermetic tests for SGP4 propagation. Fixtures only — NO network, NO
// wall-clock (all times passed in as dateMs). The near-earth kernel is
// validated against the canonical SGP4 verification object (NORAD 88888,
// Spacetrack Report #3) at the ECI level and against a real ISS element set
// at the geodetic level. (Both were cross-checked to machine precision vs
// satellite.js@7.0.1 during development.)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  propagate,
  orbitClassFromAltKm,
  epochAgeDays,
  isDeepSpace,
  _sgp4Internal,
} from './propagate.ts';
import type { GpRecord } from './tle.ts';

const { sgp4init, sgp4, DEG2RAD, XPDOTP } = _sgp4Internal;

// Real ISS element set (CelesTrak, 2026-07-07, NORAD 25544).
const ISS: GpRecord = {
  noradId: 25544,
  name: 'ISS (ZARYA)',
  epoch: '2026-07-07T12:12:01.987776',
  inclination: 51.6304,
  raan: 199.5144,
  ecc: 0.00066874,
  argp: 267.6545,
  meanAnomaly: 92.3678,
  meanMotion: 15.48933372,
  bstar: 0.00011369272,
};

const ISS_EPOCH_MS = Date.parse('2026-07-07T12:12:01.987Z');

// --- ISS propagation: plausible LEO position ------------------------------

test('propagate: ISS resolves to a plausible LEO position', () => {
  // 90 minutes past epoch (deterministic dateMs, no wall-clock).
  const dateMs = ISS_EPOCH_MS + 90 * 60000;
  const p = propagate(ISS, dateMs);
  assert.ok(p, 'propagation returns a position');

  // Latitude bounded by inclination (~51.63°); a valid geodetic lon.
  assert.ok(Math.abs(p!.latDeg) <= 52.0, `lat ${p!.latDeg} within inclination`);
  assert.ok(p!.lonDeg >= -180 && p!.lonDeg <= 180, `lon ${p!.lonDeg} in range`);
  // ISS altitude band.
  assert.ok(p!.altKm > 400 && p!.altKm < 430, `alt ${p!.altKm} km ~ ISS band`);
  assert.equal(orbitClassFromAltKm(p!.altKm), 'LEO');

  // Regression lock against the validated reference (matched satellite.js
  // to machine precision): lat≈-8.858°, lon≈61.109°, alt≈421.44 km.
  assert.ok(Math.abs(p!.latDeg - -8.8584) < 0.02, `lat ${p!.latDeg}`);
  assert.ok(Math.abs(p!.lonDeg - 61.1089) < 0.02, `lon ${p!.lonDeg}`);
  assert.ok(Math.abs(p!.altKm - 421.437) < 0.5, `alt ${p!.altKm}`);
});

test('propagate: ISS at epoch (tsince=0) is LEO at expected point', () => {
  const p = propagate(ISS, ISS_EPOCH_MS);
  assert.ok(p);
  assert.ok(p!.altKm > 400 && p!.altKm < 430);
  assert.equal(orbitClassFromAltKm(p!.altKm), 'LEO');
  // Validated reference: lat≈0.0°, lon≈91.03°, alt≈420.78 km.
  assert.ok(Math.abs(p!.latDeg - 0.0) < 0.05, `lat ${p!.latDeg}`);
  assert.ok(Math.abs(p!.lonDeg - 91.028) < 0.05, `lon ${p!.lonDeg}`);
  assert.ok(Math.abs(p!.altKm - 420.776) < 0.5, `alt ${p!.altKm}`);
});

// --- Kernel correctness vs Spacetrack Report #3 (NORAD 88888) -------------

test('sgp4: NORAD 88888 ECI matches the Spacetrack #3 verification vector', () => {
  // Elements decoded from the canonical test TLE:
  //   1 88888U          80275.98708465 ...  66816-4 ...
  //   2 88888  72.8435 115.9689 0086731  52.6988 110.5714 16.05824518
  const s = sgp4init(0, {
    inclo: 72.8435 * DEG2RAD,
    nodeo: 115.9689 * DEG2RAD,
    argpo: 52.6988 * DEG2RAD,
    mo: 110.5714 * DEG2RAD,
    ecco: 0.0086731,
    noKozai: 16.05824518 / XPDOTP,
    bstar: 0.66816e-4,
  });
  assert.equal(s.deepSpace, false);

  const rv = sgp4(s, 0);
  assert.ok(rv, 'sgp4 returns a state vector at t=0');
  // Reference ECI (km) at t=0 (Spacetrack Report #3; == satellite.js@7.0.1).
  assert.ok(Math.abs(rv!.position.x - 2328.96975) < 0.02, `x ${rv!.position.x}`);
  assert.ok(Math.abs(rv!.position.y - -5995.22051) < 0.02, `y ${rv!.position.y}`);
  assert.ok(Math.abs(rv!.position.z - 1719.97297) < 0.02, `z ${rv!.position.z}`);

  // Sanity at a later time: still a bound LEO radius (~6700-7100 km).
  const rv2 = sgp4(s, 360)!;
  const r = Math.hypot(rv2.position.x, rv2.position.y, rv2.position.z);
  assert.ok(r > 6600 && r < 7200, `radius ${r} km`);
});

// --- Deep-space handling (honest null, never faked) -----------------------

test('propagate: deep-space (GEO) returns null and isDeepSpace is true', () => {
  const geo: GpRecord = {
    noradId: 40000,
    name: 'GEO-COMM',
    epoch: '2026-07-07T00:00:00.000000',
    inclination: 0.05,
    raan: 95.0,
    ecc: 0.0002,
    argp: 270.0,
    meanAnomaly: 12.0,
    meanMotion: 1.00271, // geostationary -> period ~1436 min -> deep space
    bstar: 0,
  };
  assert.equal(isDeepSpace(geo), true);
  assert.equal(propagate(geo, Date.parse('2026-07-07T06:00:00.000Z')), null);

  // A LEO set is not deep-space.
  assert.equal(isDeepSpace(ISS), false);
});

// --- orbitClassFromAltKm boundaries ---------------------------------------

test('orbitClassFromAltKm: boundary behavior', () => {
  assert.equal(orbitClassFromAltKm(400), 'LEO');
  assert.equal(orbitClassFromAltKm(1999), 'LEO');
  assert.equal(orbitClassFromAltKm(2000), 'MEO'); // LEO < 2000
  assert.equal(orbitClassFromAltKm(20200), 'MEO');
  assert.equal(orbitClassFromAltKm(34999), 'MEO');
  assert.equal(orbitClassFromAltKm(35000), 'GEO'); // MEO < 35000
  assert.equal(orbitClassFromAltKm(35786), 'GEO');
});

// --- epochAgeDays ---------------------------------------------------------

test('epochAgeDays: deterministic age from passed-in nowMs', () => {
  const epoch = '2026-07-07T12:00:00.000Z';
  const epochMs = Date.parse(epoch);
  assert.equal(epochAgeDays(epoch, epochMs), 0);
  assert.equal(epochAgeDays(epoch, epochMs + 5 * 86400000), 5);
  assert.equal(epochAgeDays(epoch, epochMs + 0.5 * 86400000), 0.5);
  // Handles CelesTrak's timezone-less UTC epochs too.
  assert.equal(
    epochAgeDays('2026-07-07T12:00:00.000', epochMs + 2 * 86400000),
    2,
  );
});

test('epochAgeDays: null / unparseable epoch -> null', () => {
  assert.equal(epochAgeDays(null, 0), null);
  assert.equal(epochAgeDays('', 0), null);
  assert.equal(epochAgeDays('not-a-date', 0), null);
});

// --- Incomplete elements --------------------------------------------------

test('propagate: incomplete elements or bad epoch -> null (never fabricated)', () => {
  const missingMm: GpRecord = { ...ISS, meanMotion: null };
  assert.equal(propagate(missingMm, ISS_EPOCH_MS), null);
  const missingEcc: GpRecord = { ...ISS, ecc: null };
  assert.equal(propagate(missingEcc, ISS_EPOCH_MS), null);
  const badEpoch: GpRecord = { ...ISS, epoch: null };
  assert.equal(propagate(badEpoch, ISS_EPOCH_MS), null);
});
