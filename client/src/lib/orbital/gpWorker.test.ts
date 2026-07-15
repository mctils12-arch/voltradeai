// GP fetch worker — PERF session #2. Hermetic: injected fetch, captured
// post, no Worker/DOM/network. Mirrors satcatWorker's contract.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createGpWorker, type GpWorkerOutbound } from './gpWorker.js';
import type { GpRecord } from './tle.js';

const gp = (noradId: number): GpRecord => ({
  noradId, name: `SAT-${noradId}`, epoch: '2026-07-15T00:00:00',
  meanMotion: 15.5, ecc: 0.001, inclination: 53, raan: 10, argp: 20,
  meanAnomaly: 30, bstar: 0.0001,
} as GpRecord);

test('fetch message: passes the requested group and posts parsed rows through', async () => {
  const posted: GpWorkerOutbound[] = [];
  let seenGroup = '';
  const w = createGpWorker((m) => posted.push(m), {
    fetchGpImpl: async (group) => { seenGroup = group; return [gp(1), gp(2)]; },
  });
  await w.handle({ type: 'fetch', group: 'active' });
  assert.equal(seenGroup, 'active');
  assert.equal(posted.length, 1);
  assert.equal(posted[0].type, 'rows');
  assert.equal((posted[0] as any).rows.length, 2);
});

test('group defaults to active; failures post an error, never throw across the boundary', async () => {
  const posted: GpWorkerOutbound[] = [];
  let seenGroup = '';
  const w = createGpWorker((m) => posted.push(m), {
    fetchGpImpl: async (group) => { seenGroup = group; throw new Error('celestrak 429'); },
  });
  await w.handle({ type: 'fetch' });
  assert.equal(seenGroup, 'active');
  assert.deepEqual(posted, [{ type: 'error', message: 'celestrak 429' }]);
});

test('unknown messages are ignored (forward-compat)', async () => {
  const posted: GpWorkerOutbound[] = [];
  const w = createGpWorker((m) => posted.push(m));
  await w.handle({ type: 'nonsense' } as any);
  assert.equal(posted.length, 0);
});
