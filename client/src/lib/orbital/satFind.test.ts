// Find & group — EARTH TWIN O6-3. Pins search behavior, group decode
// honesty (name-prefix only), sentinel-mask semantics, and the arc cap.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  searchSats, SAT_GROUPS, groupMask, maskCount, applyGroupSentinel,
  applyFollowSolo,
  spreadIndices, GROUP_ARC_CAP, collapseStationComplexes, isStationComplex,
  groupMemberHits,
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
  // deep-space groups exist since the SDP4 port (O6-5); GLONASS stays out
  // honestly (bare "COSMOS NNNN" names — a prefix decode would lie)
  const gps = groupMask([{ noradId: 1, name: 'NAVSTAR 82 (USA 343)' }, ...GP] as any, 'gps')!;
  assert.equal(gps[0], 1, 'GPS group decodes NAVSTAR names');
  assert.equal(groupMask(GP as any, 'glonass'), null, 'GLONASS absent — no honest name decode');
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

test('applyFollowSolo: only the followed slot survives, input never mutated, null = passthrough', () => {
  const buf = new Float32Array(3 * SAT_STRIDE);
  for (let i = 0; i < 3; i++) { buf[i * SAT_STRIDE] = 0.5; buf[i * SAT_STRIDE + 3] = 0; }
  const out = applyFollowSolo(buf, 1);
  assert.equal(out[0 * SAT_STRIDE + 3], -1, 'neighbor hidden — cannot be accidentally re-picked');
  assert.equal(out[1 * SAT_STRIDE + 3], 0, 'followed craft untouched');
  assert.equal(out[2 * SAT_STRIDE + 3], -1, 'neighbor hidden');
  assert.equal(buf[0 * SAT_STRIDE + 3], 0, 'source buffer untouched');
  assert.equal(applyFollowSolo(buf, null), buf, 'no follow = passthrough, no copy');
  // composes after a group mask: solo wins over an already-filtered sky
  const grouped = applyGroupSentinel(buf, new Uint8Array([1, 1, 0]));
  const solo = applyFollowSolo(grouped, 0);
  assert.equal(solo[0 * SAT_STRIDE + 3], 0, 'followed member survives the composition');
  assert.equal(solo[1 * SAT_STRIDE + 3], -1);
  assert.equal(solo[2 * SAT_STRIDE + 3], -1);
});

test('groupMemberHits: members as clickable hits, capped with an honest total, buffer-aligned indices', () => {
  const { hits, total } = groupMemberHits(GP as any, 'starlink');
  assert.deepEqual(hits.map((h) => h.noradId), [44713, 44714]);
  assert.equal(total, 2);
  const capped = groupMemberHits(GP as any, 'starlink', 1);
  assert.equal(capped.hits.length, 1, 'cap limits the list');
  assert.equal(capped.total, 2, 'cap never hides the true count');
  assert.deepEqual(groupMemberHits(GP as any, 'glonass'), { hits: [], total: 0 }, 'unknown group = empty, never a guess');
  assert.deepEqual(groupMemberHits(null, 'starlink'), { hits: [], total: 0 });
  assert.equal(groupMemberHits(GP as any, 'iss').hits[0].index, 0, 'index aligned with gp (worker-buffer contract)');
  assert.equal(groupMemberHits(GP as any, 'planet').hits[0].index, 4, 'index aligned for later entries too');
});

test('spreadIndices: even deterministic sample under the cap', () => {
  const members = Array.from({ length: 200 }, (_, i) => i);
  const chosen = spreadIndices(members);
  assert.equal(chosen.length, GROUP_ARC_CAP);
  assert.equal(chosen[0], 0);
  assert.ok(chosen[GROUP_ARC_CAP - 1] >= 190, 'covers the tail, not just the head');
  assert.deepEqual(spreadIndices([1, 2, 3]), [1, 2, 3], 'small groups pass through whole');
});

test("station collapse: one physical station = one object; keeper = lowest NORAD; strangers untouched", () => {
  const mk = (noradId: number, name: string) => ({ noradId, name } as any);
  const gp = [
    mk(25575, "ISS (UNITY)"),
    mk(25544, "ISS (ZARYA)"),
    mk(49044, "ISS (NAUKA)"),
    mk(48274, "CSS (TIANHE)"),
    mk(54216, "CSS (MENGTIAN)"),
    mk(52085, "PROGRESS-MS 19"),   // visiting vehicle — own name, stays
    mk(44713, "STARLINK-1007"),
  ];
  const out = collapseStationComplexes(gp);
  assert.deepEqual(out.map((g: any) => g.noradId), [25544, 48274, 52085, 44713],
    "ZARYA + TIANHE keep their stations; modules dropped; others untouched");
  // no station entries at all → identity (no accidental filtering)
  const plain = [mk(1, "A"), mk(2, "B")];
  assert.deepEqual(collapseStationComplexes(plain), plain);
  assert.equal(isStationComplex("ISS (ZARYA)"), true);
  assert.equal(isStationComplex("CSS (TIANHE)"), true);
  assert.equal(isStationComplex("PROGRESS-MS 19"), false);
  assert.equal(isStationComplex(null), false);
});
