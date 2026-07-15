// Find & group — EARTH TWIN O6-3. Pins search behavior, group decode
// honesty (name-prefix only), sentinel-mask semantics, and the arc cap.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  searchSats, SAT_GROUPS, groupMask, maskCount, applyGroupSentinel,
  spreadIndices, GROUP_ARC_CAP,
} from './satFind.js';
import { SAT_STRIDE } from './satBuffer.js';

const GP = [
  { noradId: 25544, name: 'ISS (ZARYA)' },
  { noradId: 44713, name: 'STARLINK-1007' },
  { noradId: 44714, name: 'STARLINK-1008' },
  { noradId: 43013, name: 'ONEWEB-0012' },
  { noradId: 90001, name: 'FLOCK 4X-1' },
  { noradId: 20580, name: 'HST' },
] as any[];

test('searchSats: substring + NORAD id, short queries refuse, limit holds', () => {
  assert.deepEqual(searchSats(GP, 'starlink').map((h) => h.noradId), [44713, 44714]);
  assert.deepEqual(searchSats(GP, 'iss').map((h) => h.noradId), [25544]);
  assert.deepEqual(searchSats(GP, '20580').map((h) => h.noradId), [20580], 'numeric = NORAD id');
  assert.deepEqual(searchSats(GP, 'x'), [], 'single char = noise, no results');
  assert.equal(searchSats(GP, 'STARLINK', 1).length, 1, 'limit');
  assert.deepEqual(searchSats(null, 'iss'), []);
});

test('groups: name-prefix decodes; ISS chip finds the station; unknown key = null', () => {
  const starlink = groupMask(GP as any, 'starlink')!;
  assert.deepEqual([...starlink], [0, 1, 1, 0, 0, 0]);
  assert.equal(maskCount(starlink), 2);
  const iss = groupMask(GP as any, 'iss')!;
  assert.deepEqual([...iss], [1, 0, 0, 0, 0, 0]);
  assert.equal(groupMask(GP as any, 'gps'), null, 'deep-space constellations deliberately absent (no live position — an empty sky would read as a bug)');
  for (const g of SAT_GROUPS) assert.ok(g.label && g.test('X') === false, `${g.key}: honest negative on arbitrary names`);
});

test('applyGroupSentinel: non-members get the sentinel class, members untouched, input never mutated', () => {
  const buf = new Float32Array(3 * SAT_STRIDE);
  for (let i = 0; i < 3; i++) { buf[i * SAT_STRIDE] = 0.5; buf[i * SAT_STRIDE + 3] = 0; }
  const out = applyGroupSentinel(buf, new Uint8Array([1, 0, 1]));
  assert.equal(out[0 * SAT_STRIDE + 3], 0);
  assert.equal(out[1 * SAT_STRIDE + 3], -1, 'filtered out via the layer\'s own sentinel semantics');
  assert.equal(out[2 * SAT_STRIDE + 3], 0);
  assert.equal(buf[1 * SAT_STRIDE + 3], 0, 'source buffer untouched');
  assert.equal(applyGroupSentinel(buf, null), buf, 'no mask = passthrough, no copy');
});

test('spreadIndices: even deterministic sample under the cap', () => {
  const members = Array.from({ length: 200 }, (_, i) => i);
  const chosen = spreadIndices(members);
  assert.equal(chosen.length, GROUP_ARC_CAP);
  assert.equal(chosen[0], 0);
  assert.ok(chosen[GROUP_ARC_CAP - 1] >= 190, 'covers the tail, not just the head');
  assert.deepEqual(spreadIndices([1, 2, 3]), [1, 2, 3], 'small groups pass through whole');
});
