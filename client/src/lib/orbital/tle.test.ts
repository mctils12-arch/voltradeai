// Hermetic tests for the CelesTrak parse/join layer. Fixtures only — NO
// network, NO wall-clock. Field names & sample rows mirror the REAL CelesTrak
// schemas (verified live 2026-07-07).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  gpUrl,
  satcatUrl,
  parseGp,
  parseGpCsv,
  parseSatcat,
  joinGpSatcat,
  classifyOrbit,
  fetchGp,
  type GpRecord,
  type SatcatRecord,
} from './tle.ts';

// --- Fixtures -------------------------------------------------------------

// GP/OMM array, shaped exactly like gp.php?FORMAT=json.
// ISS uses REAL elements pulled from CelesTrak 2026-07-07 (NORAD 25544).
const GP_FIXTURE = [
  {
    OBJECT_NAME: 'ISS (ZARYA)',
    OBJECT_ID: '1998-067A',
    EPOCH: '2026-07-07T12:12:01.987776',
    MEAN_MOTION: 15.48933372,
    ECCENTRICITY: 0.00066874,
    INCLINATION: 51.6304,
    RA_OF_ASC_NODE: 199.5144,
    ARG_OF_PERICENTER: 267.6545,
    MEAN_ANOMALY: 92.3678,
    EPHEMERIS_TYPE: 0,
    CLASSIFICATION_TYPE: 'U',
    NORAD_CAT_ID: 25544,
    ELEMENT_SET_NO: 999,
    REV_AT_EPOCH: 57490,
    BSTAR: 0.00011369272,
    MEAN_MOTION_DOT: 5.806e-5,
    MEAN_MOTION_DDOT: 0,
  },
  {
    // GP-only object (no SATCAT row in the fixture). One field blank -> null.
    OBJECT_NAME: 'GPONLY-SAT',
    OBJECT_ID: '2024-100A',
    EPOCH: '2026-07-06T00:00:00.000000',
    MEAN_MOTION: 15.2,
    ECCENTRICITY: 0.001,
    INCLINATION: 53.0,
    RA_OF_ASC_NODE: 100.0,
    ARG_OF_PERICENTER: 90.0,
    MEAN_ANOMALY: 270.0,
    NORAD_CAT_ID: 99001,
    BSTAR: '', // blank string -> must normalize to null
  },
  {
    // Missing NORAD id -> unusable, must be dropped.
    OBJECT_NAME: 'NO-CATALOG-ID',
    EPOCH: '2026-07-06T00:00:00.000000',
    MEAN_MOTION: 15.0,
  },
];

// GP CSV, shaped exactly like gp.php?FORMAT=csv — same OMM fields as the JSON
// form, CSV-encoded (CRLF, as CelesTrak serves it). Mirrors GP_FIXTURE: ISS
// with real elements, one row with a blank BSTAR, one row missing its NORAD id.
const GP_CSV_FIXTURE = [
  'OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT',
  'ISS (ZARYA),1998-067A,2026-07-07T12:12:01.987776,15.48933372,.00066874,51.6304,199.5144,267.6545,92.3678,0,U,25544,999,57490,.11369272E-3,.5806E-4,0',
  'GPONLY-SAT,2024-100A,2026-07-06T00:00:00.000000,15.2,.001,53.0,100.0,90.0,270.0,0,U,99001,,,,,',
  'NO-CATALOG-ID,,2026-07-06T00:00:00.000000,15.0,,,,,,,,,,,,,',
].join('\r\n');

// SATCAT CSV. Header + rows are REAL-shaped (first two are actual catalog
// rows). ISS row + one SATCAT-only object (60000) exercise the join.
const SATCAT_FIXTURE = [
  'OBJECT_NAME,OBJECT_ID,NORAD_CAT_ID,OBJECT_TYPE,OPS_STATUS_CODE,OWNER,LAUNCH_DATE,LAUNCH_SITE,DECAY_DATE,PERIOD,INCLINATION,APOGEE,PERIGEE,RCS,DATA_STATUS_CODE,ORBIT_CENTER,ORBIT_TYPE',
  'SL-1 R/B,1957-001A,1,R/B,D,CIS,1957-10-04,TYMSC,1957-12-01,96.19,65.10,938,214,20.4200,,EA,IMP',
  'SPUTNIK 1,1957-001B,2,PAY,D,CIS,1957-10-04,TYMSC,1958-01-03,96.10,65.00,1080,64,,,EA,IMP',
  'ISS (ZARYA),1998-067A,25544,PAY,+,ISS,1998-11-20,TYMSC,,92.90,51.64,420,413,399.0500,,EA,ORB',
  'FAKE DEBRIS,2001-999ZZ,60000,DEB,,PRC,2001-05-05,XSC,,,,,,,,EA,ORB',
].join('\r\n'); // CelesTrak uses CRLF

// --- parseGp --------------------------------------------------------------

test('parseGp: ISS record normalized with real elements', () => {
  const recs = parseGp(GP_FIXTURE);
  const iss = recs.find((r) => r.noradId === 25544);
  assert.ok(iss, 'ISS present');
  assert.equal(iss!.name, 'ISS (ZARYA)');
  assert.equal(iss!.epoch, '2026-07-07T12:12:01.987776');
  assert.equal(iss!.meanMotion, 15.48933372);
  assert.equal(iss!.inclination, 51.6304);
  assert.equal(iss!.raan, 199.5144);
  assert.equal(iss!.ecc, 0.00066874);
  assert.equal(iss!.argp, 267.6545);
  assert.equal(iss!.meanAnomaly, 92.3678);
  assert.equal(iss!.bstar, 0.00011369272);
});

test('parseGp: blank field -> null, missing NORAD id -> dropped', () => {
  const recs = parseGp(GP_FIXTURE);
  // NO-CATALOG-ID dropped: only 2 of 3 survive.
  assert.equal(recs.length, 2);
  const gponly = recs.find((r) => r.noradId === 99001);
  assert.ok(gponly);
  assert.equal(gponly!.bstar, null); // blank string normalized to null
});

test('parseGp: accepts a JSON string as well as an array', () => {
  const recs = parseGp(JSON.stringify(GP_FIXTURE));
  assert.equal(recs.length, 2);
});

test('parseGp: garbage input yields empty array, never throws', () => {
  assert.deepEqual(parseGp('not json'), []);
  assert.deepEqual(parseGp(null), []);
  assert.deepEqual(parseGp(42), []);
  assert.deepEqual(parseGp({}), []);
});

// --- parseGpCsv -----------------------------------------------------------

test('parseGpCsv: ISS row normalized to the same GpRecord shape as parseGp', () => {
  const recs = parseGpCsv(GP_CSV_FIXTURE);
  const iss = recs.find((r) => r.noradId === 25544);
  assert.ok(iss, 'ISS present');
  assert.equal(iss!.name, 'ISS (ZARYA)');
  assert.equal(iss!.epoch, '2026-07-07T12:12:01.987776');
  assert.equal(iss!.meanMotion, 15.48933372);
  assert.equal(iss!.inclination, 51.6304);
  assert.equal(iss!.raan, 199.5144);
  assert.equal(iss!.ecc, 0.00066874);
  assert.equal(iss!.argp, 267.6545);
  assert.equal(iss!.meanAnomaly, 92.3678);
  assert.equal(iss!.bstar, 0.00011369272); // CelesTrak's .11369272E-3 form
});

test('parseGpCsv: blank field -> null, missing NORAD id -> dropped', () => {
  const recs = parseGpCsv(GP_CSV_FIXTURE);
  assert.equal(recs.length, 2); // NO-CATALOG-ID dropped
  const gponly = recs.find((r) => r.noradId === 99001);
  assert.ok(gponly);
  assert.equal(gponly!.bstar, null); // blank -> null
});

test('parseGpCsv: garbage / non-GP CSV yields empty array, never throws', () => {
  assert.deepEqual(parseGpCsv(''), []);
  assert.deepEqual(parseGpCsv('   '), []);
  // a body without a NORAD_CAT_ID column is not GP CSV (e.g. an error page)
  assert.deepEqual(parseGpCsv('foo,bar\n1,2'), []);
});

// --- parseSatcat ----------------------------------------------------------

test('parseSatcat: ISS row normalized (type/owner/orbitClass/rcs)', () => {
  const recs = parseSatcat(SATCAT_FIXTURE);
  const iss = recs.find((r) => r.noradId === 25544);
  assert.ok(iss);
  assert.equal(iss!.name, 'ISS (ZARYA)');
  assert.equal(iss!.intlDes, '1998-067A');
  assert.equal(iss!.objectType, 'PAYLOAD'); // PAY -> PAYLOAD
  assert.equal(iss!.owner, 'ISS');
  assert.equal(iss!.country, 'ISS'); // OWNER doubles as country/agency code
  assert.equal(iss!.opStatus, '+');
  assert.equal(iss!.launchDate, '1998-11-20');
  assert.equal(iss!.orbitClass, 'LEO'); // derived from apogee 420 / perigee 413
  assert.equal(iss!.rcsMeters2, 399.05);
  assert.equal(iss!.rcsSize, 'LARGE'); // > 1 m^2
});

test('parseSatcat: object-type codes and blank RCS', () => {
  const recs = parseSatcat(SATCAT_FIXTURE);
  const rb = recs.find((r) => r.noradId === 1);
  const deb = recs.find((r) => r.noradId === 60000);
  const sputnik = recs.find((r) => r.noradId === 2);
  assert.equal(rb!.objectType, 'ROCKET BODY'); // R/B
  assert.equal(deb!.objectType, 'DEBRIS'); // DEB
  assert.equal(sputnik!.objectType, 'PAYLOAD');
  assert.equal(sputnik!.rcsMeters2, null); // blank RCS column -> null
  assert.equal(sputnik!.rcsSize, null);
  assert.equal(deb!.opStatus, null); // blank ops status -> null
  assert.equal(deb!.orbitClass, null); // no apogee/perigee -> null (not guessed)
});

test('parseSatcat: garbage / non-satcat input -> empty array', () => {
  assert.deepEqual(parseSatcat(''), []);
  assert.deepEqual(parseSatcat('a,b,c\n1,2,3'), []); // no NORAD_CAT_ID column
});

// --- classifyOrbit boundaries ---------------------------------------------

test('classifyOrbit: LEO/MEO/GEO/HEO and null', () => {
  assert.equal(classifyOrbit(420, 413), 'LEO');
  assert.equal(classifyOrbit(20200, 20180), 'MEO'); // GPS band
  assert.equal(classifyOrbit(35793, 35779), 'GEO'); // geostationary
  assert.equal(classifyOrbit(39000, 500), 'HEO'); // Molniya-like
  assert.equal(classifyOrbit(null, 500), null);
  assert.equal(classifyOrbit(500, null), null);
});

// --- joinGpSatcat ---------------------------------------------------------

test('joinGpSatcat: union keeps GP-only and SATCAT-only records', () => {
  const gp = parseGp(GP_FIXTURE);
  const satcat = parseSatcat(SATCAT_FIXTURE);
  const joined = joinGpSatcat(gp, satcat);

  // Union of ids: GP {25544, 99001} ∪ SATCAT {1, 2, 25544, 60000}
  //             = {25544, 99001, 1, 2, 60000} = 5 records, none dropped.
  assert.equal(joined.length, 5);
  const ids = new Set(joined.map((j) => j.noradId));
  assert.deepEqual(ids, new Set([25544, 99001, 1, 2, 60000]));

  const iss = joined.find((j) => j.noradId === 25544)!;
  assert.equal(iss.hasGp, true);
  assert.equal(iss.hasSatcat, true);
  assert.equal(iss.meanMotion, 15.48933372); // from GP
  assert.equal(iss.objectType, 'PAYLOAD'); // from SATCAT
  assert.equal(iss.orbitClass, 'LEO');

  const gponly = joined.find((j) => j.noradId === 99001)!;
  assert.equal(gponly.hasGp, true);
  assert.equal(gponly.hasSatcat, false);
  assert.equal(gponly.meanMotion, 15.2); // orbital present
  assert.equal(gponly.objectType, null); // metadata null, NOT dropped
  assert.equal(gponly.owner, null);
  assert.equal(gponly.orbitClass, null);

  const satonly = joined.find((j) => j.noradId === 60000)!;
  assert.equal(satonly.hasGp, false);
  assert.equal(satonly.hasSatcat, true);
  assert.equal(satonly.meanMotion, null); // orbital null, NOT dropped
  assert.equal(satonly.epoch, null);
  assert.equal(satonly.objectType, 'DEBRIS'); // metadata present
});

test('joinGpSatcat: name falls back GP -> SATCAT and is never undefined', () => {
  const gp: GpRecord[] = [];
  const satcat: SatcatRecord[] = parseSatcat(SATCAT_FIXTURE);
  const joined = joinGpSatcat(gp, satcat);
  assert.equal(joined.length, 4); // all SATCAT ids, no GP
  for (const j of joined) {
    assert.notEqual(j.name, undefined);
    assert.equal(j.hasGp, false);
    assert.equal(j.hasSatcat, true);
  }
});

// --- endpoints & fetch wiring (hermetic: injected fetch, no network) ------

test('gpUrl / satcatUrl build the expected CelesTrak endpoints', () => {
  assert.equal(
    gpUrl('active'),
    'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json',
  );
  assert.equal(gpUrl(), gpUrl('active')); // default group + format
  assert.equal(
    gpUrl('gps-ops'),
    'https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=json',
  );
  // csv format — what fetchGp requests (smaller transfer)
  assert.equal(
    gpUrl('active', 'csv'),
    'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=csv',
  );
  assert.equal(satcatUrl(), 'https://celestrak.org/pub/satcat.csv');
});

test('fetchGp: fetches the CSV endpoint and parses it', async () => {
  let calledUrl = '';
  const fakeFetch = async (url: string) => {
    calledUrl = url;
    return { ok: true, status: 200, json: async () => GP_FIXTURE, text: async () => GP_CSV_FIXTURE };
  };
  const recs = await fetchGp('active', fakeFetch);
  assert.equal(calledUrl, gpUrl('active', 'csv')); // CSV, not JSON
  assert.equal(recs.length, 2); // NO-CATALOG-ID row dropped
  assert.equal(recs[0].noradId, 25544);
  assert.equal(recs[0].name, 'ISS (ZARYA)');
});

test('fetchGp: a non-2xx response (e.g. 403 rate-limit) throws, never returns []', async () => {
  // Regression: CelesTrak's courtesy rate-limit returns HTTP 403 with a non-GP
  // body. It must throw (→ resilient-load backoff), not silently parse to [] —
  // the empty parse used to masquerade as "no elements" and loop forever.
  const rateLimited = async () => ({
    ok: false,
    status: 403,
    json: async () => ({}),
    text: async () => 'Rate limit exceeded',
  });
  await assert.rejects(fetchGp('active', rateLimited), /403/);
});
