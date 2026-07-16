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

// --- Deep-space handling (SDP4) --------------------------------------------
// (Until 2026-07: deep-space returned null. The SDP4 port now propagates
// these objects for real, so the old "returns null" expectation is replaced
// by physics-invariant assertions — a strictly stronger check.)

const GEO: GpRecord = {
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
const GEO_EPOCH_MS = Date.parse('2026-07-07T00:00:00.000Z');

test('propagate: deep-space (GEO) resolves via SDP4; isDeepSpace still labels it', () => {
  assert.equal(isDeepSpace(GEO), true);
  const p = propagate(GEO, Date.parse('2026-07-07T06:00:00.000Z'));
  assert.ok(p, 'deep-space propagation returns a real position');
  assert.ok(Math.abs(p!.altKm - 35786) < 500, `alt ${p!.altKm} km ~ GEO band`);
  assert.equal(orbitClassFromAltKm(p!.altKm), 'GEO');

  // A LEO set is not deep-space.
  assert.equal(isDeepSpace(ISS), false);
});

// --- SDP4 physics invariants (hermetic — fixed epochs, no network) ---------

test('SDP4 GEO: geostationary band, latitude, and longitude hold across 24h', () => {
  const p0 = propagate(GEO, GEO_EPOCH_MS);
  assert.ok(p0, 'position at epoch');
  for (let h = 0; h <= 24; h++) {
    const p = propagate(GEO, GEO_EPOCH_MS + h * 3600000);
    assert.ok(p, `position at +${h}h`);
    // Geostationary altitude ~35786 km, held across the day.
    assert.ok(Math.abs(p!.altKm - 35786) < 500, `alt ${p!.altKm} km at +${h}h`);
    // Latitude bounded by the (tiny) inclination + lunar-solar periodics.
    assert.ok(Math.abs(p!.latDeg) < 1.0, `lat ${p!.latDeg} at +${h}h (incl 0.05°)`);
    // Geostationary: sub-satellite longitude must not drift.
    let dLon = p!.lonDeg - p0!.lonDeg;
    if (dLon > 180) dLon -= 360;
    if (dLon < -180) dLon += 360;
    assert.ok(Math.abs(dLon) < 3.0, `lon drift ${dLon}° at +${h}h`);
  }
});

const GPS: GpRecord = {
  noradId: 41019,
  name: 'GPS-LIKE MEO',
  epoch: '2026-07-07T00:00:00.000000',
  inclination: 55.0,
  raan: 120.0,
  ecc: 0.01,
  argp: 40.0,
  meanAnomaly: 300.0,
  meanMotion: 2.00565, // semi-synchronous -> period ~717.97 min
  bstar: 0,
};

test('SDP4 GPS-like MEO: altitude band across 12h + period matches meanMotion', () => {
  const epochMs = GEO_EPOCH_MS;
  for (let m = 0; m <= 720; m += 30) {
    const p = propagate(GPS, epochMs + m * 60000);
    assert.ok(p, `position at +${m}min`);
    assert.ok(Math.abs(p!.altKm - 20200) < 1500, `alt ${p!.altKm} km at +${m}min`);
    assert.ok(Math.abs(p!.latDeg) <= 56.0, `lat ${p!.latDeg} within inclination`);
  }

  // Period consistency, checked in ECI (independent of earth rotation):
  // after one mean period the satellite must be back near its start point,
  // and far from it at half a period.
  const s = sgp4init(epochMs, {
    inclo: 55.0 * DEG2RAD,
    nodeo: 120.0 * DEG2RAD,
    argpo: 40.0 * DEG2RAD,
    mo: 300.0 * DEG2RAD,
    ecco: 0.01,
    noKozai: 2.00565 / XPDOTP,
    bstar: 0,
  });
  const periodMin = 1440 / 2.00565;
  assert.ok(Math.abs(periodMin - 717.97) < 0.1, `period ${periodMin} min`);
  const angleDeg = (
    a: { x: number; y: number; z: number },
    b: { x: number; y: number; z: number },
  ) => {
    const dot = a.x * b.x + a.y * b.y + a.z * b.z;
    const na = Math.hypot(a.x, a.y, a.z);
    const nb = Math.hypot(b.x, b.y, b.z);
    return (Math.acos(Math.max(-1, Math.min(1, dot / (na * nb)))) * 180) / Math.PI;
  };
  const r0 = sgp4(s, 0)!.position;
  const rFull = sgp4(s, periodMin)!.position;
  const rHalf = sgp4(s, periodMin / 2)!.position;
  assert.ok(angleDeg(r0, rFull) < 3.0, `returns to start after one period (${angleDeg(r0, rFull)}°)`);
  assert.ok(angleDeg(r0, rHalf) > 90.0, `far away at half period (${angleDeg(r0, rHalf)}°)`);
});

const MOLNIYA: GpRecord = {
  noradId: 21118,
  name: 'MOLNIYA-LIKE HEO',
  epoch: '2026-07-07T00:00:00.000000',
  inclination: 63.4,
  raan: 80.0,
  ecc: 0.72,
  argp: 270.0,
  meanAnomaly: 0.0, // at perigee at epoch
  meanMotion: 2.006, // 12h resonance with e >= 0.5 -> irez=2 path exercised
  bstar: 0,
};

test('SDP4 Molniya: eccentric geometry survives dpper/dspace (perigee<4000, apogee>35000)', () => {
  let minAlt = Infinity;
  let maxAlt = -Infinity;
  let maxLat = -Infinity;
  for (let m = 0; m <= 720; m += 10) {
    const p = propagate(MOLNIYA, GEO_EPOCH_MS + m * 60000);
    assert.ok(p, `position at +${m}min`);
    minAlt = Math.min(minAlt, p!.altKm);
    maxAlt = Math.max(maxAlt, p!.altKm);
    maxLat = Math.max(maxLat, p!.latDeg);
  }
  assert.ok(minAlt < 4000, `perigee region reached (min alt ${minAlt} km)`);
  assert.ok(minAlt > 200, `perigee stays above the atmosphere (min alt ${minAlt} km)`);
  assert.ok(maxAlt > 35000, `apogee region reached (max alt ${maxAlt} km)`);
  assert.ok(maxLat > 30, `apogee dwells in the north (argp 270°): max lat ${maxLat}°`);
});

test('SDP4: deterministic — repeated and out-of-order calls agree exactly', () => {
  // Public API: identical inputs -> identical outputs.
  const t1 = GEO_EPOCH_MS + 24 * 3600000;
  assert.deepEqual(propagate(GEO, t1), propagate(GEO, t1));
  const t2 = GEO_EPOCH_MS + 123 * 60000;
  assert.deepEqual(propagate(MOLNIYA, t2), propagate(MOLNIYA, t2));

  // Kernel level: the resonance integrator must not leak state between
  // calls on the SAME satrec — out-of-order times reproduce exactly.
  const mk = () =>
    sgp4init(GEO_EPOCH_MS, {
      inclo: 0.05 * DEG2RAD,
      nodeo: 95.0 * DEG2RAD,
      argpo: 270.0 * DEG2RAD,
      mo: 12.0 * DEG2RAD,
      ecco: 0.0002,
      noKozai: 1.00271 / XPDOTP,
      bstar: 0,
    });
  const sA = mk();
  const first = sgp4(sA, 3000)!; // > 720 min: forces integrator stepping
  assert.ok(first);
  sgp4(sA, 60); // interleaved earlier time
  const again = sgp4(sA, 3000)!;
  assert.deepEqual(again.position, first.position);
  assert.deepEqual(again.velocity, first.velocity);
  // Fresh satrec at the same t matches the reused one.
  const fresh = sgp4(mk(), 3000)!;
  assert.deepEqual(fresh.position, first.position);
});

test('SDP4: subterranean perigee -> honest null (decay error path, never junk)', () => {
  // e=0.80 at n=2.006 rev/day puts perigee ~5300 km from the geocenter —
  // inside the earth. At epoch (mean anomaly 0 = perigee) SDP4 must refuse.
  const doomed: GpRecord = { ...MOLNIYA, ecc: 0.8 };
  assert.equal(isDeepSpace(doomed), true);
  assert.equal(propagate(doomed, GEO_EPOCH_MS), null);
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
