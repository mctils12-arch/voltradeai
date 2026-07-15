// SATCAT worker — EARTH TWIN E4-2 perf slice. Hermetic: injected fetch,
// captured post, no Worker/DOM/network.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createSatcatWorker, type SatcatWorkerOutbound } from './satcatWorker.js';
import type { SatcatRecord } from './tle.js';

const row = (noradId: number): SatcatRecord => ({
  noradId, name: `SAT-${noradId}`, intlDes: null, owner: null, country: null,
  launchDate: null, objectType: 'PAYLOAD', opStatus: null, orbitClass: null,
  rcsMeters2: null, rcsSize: null,
});

test('fetch message: posts parsed rows through unchanged (caller owns the empty-rule)', async () => {
  const posted: SatcatWorkerOutbound[] = [];
  const w = createSatcatWorker((m) => posted.push(m), {
    fetchSatcatImpl: async () => [row(1), row(2)],
  });
  await w.handle({ type: 'fetch' });
  assert.equal(posted.length, 1);
  assert.equal(posted[0].type, 'rows');
  assert.equal((posted[0] as any).rows.length, 2);
});

test('fetch failure: posts an error message, never throws across the worker boundary', async () => {
  const posted: SatcatWorkerOutbound[] = [];
  const w = createSatcatWorker((m) => posted.push(m), {
    fetchSatcatImpl: async () => { throw new Error('celestrak 503'); },
  });
  await w.handle({ type: 'fetch' });
  assert.deepEqual(posted, [{ type: 'error', message: 'celestrak 503' }]);
});

test('unknown messages are ignored (forward-compat, satWorker convention)', async () => {
  const posted: SatcatWorkerOutbound[] = [];
  const w = createSatcatWorker((m) => posted.push(m));
  await w.handle({ type: 'nonsense' } as any);
  assert.equal(posted.length, 0);
});
