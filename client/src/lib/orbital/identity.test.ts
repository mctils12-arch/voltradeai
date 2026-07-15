// Satellite identity formatting — EARTH TWIN E4-1. Pins the honesty
// language: derived claims labeled, gaps honest, decode table documented.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  OPS_STATUS_DECODE,
  nameStemForOperator,
  satelliteIdentityLines,
} from './identity.js';
import type { SatcatRecord } from './tle.js';

const base: SatcatRecord = {
  noradId: 44713,
  name: 'STARLINK-1007',
  intlDes: '2019-074A',
  owner: 'US',
  country: 'US',
  launchDate: '2019-11-11',
  objectType: 'PAYLOAD',
  opStatus: '+',
  orbitClass: 'LEO',
  rcsMeters2: null,
  rcsSize: null,
};

test('nameStemForOperator: constellation stems derive as lookup keys, junk stays null', () => {
  assert.equal(nameStemForOperator('STARLINK-3021'), 'STARLINK');
  assert.equal(nameStemForOperator('IRIDIUM 167'), 'IRIDIUM');
  assert.equal(nameStemForOperator('ISS (ZARYA)'), 'ISS');
  assert.equal(nameStemForOperator('COSMOS 2251 DEB'), 'COSMOS');
  assert.equal(nameStemForOperator('ONEWEB-0424'), 'ONEWEB');
  assert.equal(nameStemForOperator(null), null);
  assert.equal(nameStemForOperator('X'), null, 'a 1-char stem is too weak to be a lookup key');
});

test('CubeSat-class language: ONLY for SMALL-RCS payloads, and always labeled derived', () => {
  const small = satelliteIdentityLines({ ...base, rcsMeters2: 0.05, rcsSize: 'SMALL' }, null);
  const typeLine = small.find((l) => l.startsWith('Object type:'));
  assert.ok(typeLine?.includes('CubeSat-class'), 'small payload reads as CubeSat-class size');
  assert.ok(typeLine?.includes('derived'), 'the class claim must be labeled derived, never a catalog fact');

  const large = satelliteIdentityLines({ ...base, rcsMeters2: 25, rcsSize: 'LARGE' }, null);
  assert.ok(!large.find((l) => l.includes('CubeSat')), 'large payloads never get CubeSat language');
  const rb = satelliteIdentityLines({ ...base, objectType: 'ROCKET BODY', rcsSize: 'SMALL' }, null);
  assert.ok(!rb.find((l) => l.includes('CubeSat')), 'a small rocket body is not a CubeSat');
});

test('status decode: documented CelesTrak codes, raw code always shown, unknown codes honest', () => {
  assert.equal(OPS_STATUS_DECODE['+'], 'operational');
  assert.equal(OPS_STATUS_DECODE['D'], 'decayed');
  const ok = satelliteIdentityLines(base, null);
  assert.ok(ok.find((l) => l === 'Status: operational (SATCAT code +)'));
  const weird = satelliteIdentityLines({ ...base, opStatus: 'Z' }, null);
  assert.ok(weird.find((l) => l.includes('code not in decode table') && l.includes('code Z')),
    'an undocumented code is stated as such, never guessed');
});

test('operator line: ticker when public, honest no-equity note when private, provenance always shown', () => {
  const priv = satelliteIdentityLines(base, {
    company: 'SpaceX (Space Exploration Technologies Corp.)',
    ticker: null, exchange: null, category: 'launch-provider' as any,
    country: 'US', notes: '', is_public: false, matched_on: 'starlink', asof: '2026-07-07',
  });
  const opLine = priv.find((l) => l.startsWith('Operator:'));
  assert.ok(opLine?.includes('no public equity'), 'private operator states the no-ticker truth');
  assert.ok(opLine?.includes('matched on "starlink"'), 'join provenance surfaces on the card');

  const pub = satelliteIdentityLines(base, {
    company: 'Iridium Communications Inc.', ticker: 'IRDM', exchange: 'NASDAQ',
    category: 'comms' as any, country: 'US', notes: '', is_public: true,
    matched_on: 'iridium', asof: '2026-07-07',
  });
  assert.ok(pub.find((l) => l.includes('IRDM') && l.includes('NASDAQ')));
});

test('missing catalog: loading vs unavailable read differently, and never invent identity', () => {
  const loading = satelliteIdentityLines(null, null, 'loading');
  assert.equal(loading.length, 1);
  assert.ok(loading[0].includes('still downloading'));
  const err = satelliteIdentityLines(null, null, 'error');
  assert.ok(err[0].includes('unavailable'));
  const sparse = satelliteIdentityLines(
    { ...base, objectType: null, owner: null, launchDate: null, opStatus: null, intlDes: null, rcsSize: null, rcsMeters2: null },
    null,
  );
  assert.deepEqual(sparse, [], 'an all-null SATCAT row yields NO lines — gaps are omitted, never fabricated');
});
