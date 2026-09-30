// Selected-aircraft fast lane, client half: decode, poll lifecycle (one
// request at a time, abort on stop, paused while hidden), and only NEWER
// fixes delivered (an unchanged upstream never re-stamps freshness).
// Run: npx tsx --test client/src/lib/air/selectedFastLane.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { decodeFastFix, startSelectedFastLane, FAST_POLL_MS, type FastFix } from './selectedFastLane.js';

const FIELDS = ['hex', 'lon', 'lat', 'altFt', 'gsKt', 'trk', 'callsign', 'type', 'seenAt', 'cat', 'gnd', 'reg', 'src', 'baroRate'];
const AT = 1_790_799_400_000;
const payload = (seenAt: number, baro: number | null = -64) => ({
  at: AT, hex: 'ab8c8e', source: 'upstream', fields: FIELDS,
  rows: [['ab8c8e', -83.9, 36.1, 43000, 466, 186, 'N843S', 'FA7X', seenAt, 'A2', 0, 'N843S', 'adsblol', baro]],
});

test('decodeFastFix: row via the global adapter + the fix time + broadcast vertical rate', () => {
  const f = decodeFastFix(payload(AT - 1_500));
  assert.ok(f);
  assert.equal(f?.row.callsign, 'N843S');
  assert.equal(f?.fixMs, AT - 1_500);
  assert.equal(f?.baroRateFpm, -64);
  assert.equal(f?.source, 'upstream');
  assert.equal(decodeFastFix(payload(AT, null))?.baroRateFpm, null);
  assert.equal(decodeFastFix({ at: AT, fields: FIELDS, rows: [] }), null);
  assert.equal(FAST_POLL_MS, 2_500, 'the requested 2-3 s cadence');
});

function harness(bodies: unknown[]) {
  const timers: { fn: () => void; ms: number }[] = [];
  const signals: AbortSignal[] = [];
  const urls: string[] = [];
  let hidden = false;
  const gates: (() => void)[] = [];
  let hold = false;
  const fetchImpl = (async (url: string, init?: RequestInit) => {
    urls.push(url);
    signals.push(init!.signal!);
    if (hold) await new Promise<void>((r) => gates.push(r));
    const body = bodies.length > 1 ? bodies.shift() : bodies[0];
    return { ok: true, status: 200, json: async () => body } as Response;
  }) as unknown as typeof fetch;
  return {
    timers, signals, urls, gates,
    setHidden: (v: boolean) => { hidden = v; },
    setHold: (v: boolean) => { hold = v; },
    opts: {
      fetchImpl, isHidden: () => hidden,
      setTimeout: ((fn: () => void, ms: number) => { timers.push({ fn, ms }); return timers.length; }) as (fn: () => void, ms: number) => unknown,
      clearTimeout: ((h: unknown) => { const t = timers[(h as number) - 1]; if (t) t.fn = () => {}; }) as (h: unknown) => void,
    },
  };
}
const flush = () => new Promise((r) => setTimeout(r, 0));
async function runNext(h: ReturnType<typeof harness>) { const t = h.timers.shift(); t?.fn(); await flush(); await flush(); }

test('lifecycle: first poll immediately, then every FAST_POLL_MS, one request at a time; only NEWER fixes delivered', async () => {
  const h = harness([payload(AT - 1_000), payload(AT - 1_000), payload(AT - 200)]);
  const fixes: FastFix[] = [];
  const polls: (FastFix | null)[] = [];
  const lane = startSelectedFastLane({ hex: 'AB8C8E', onFix: (f) => fixes.push(f), onPoll: (f) => polls.push(f), ...h.opts });
  assert.equal(h.timers[0].ms, 0, 'first poll on the next task');
  await runNext(h);
  assert.deepEqual(h.urls, ['/api/data/aircraft/live/ab8c8e']);
  assert.equal(fixes.length, 1);
  assert.equal(h.timers.length, 1);
  assert.equal(h.timers[0].ms, FAST_POLL_MS, 'next poll scheduled only after this one finished');
  await runNext(h);
  assert.equal(polls.length, 2);
  assert.equal(fixes.length, 1, 'same fix time -> not re-delivered (no fake freshness)');
  await runNext(h);
  assert.equal(fixes.length, 2, 'a newer fix is delivered');
  assert.equal(fixes[1].fixMs, AT - 200);
  lane.stop();
  assert.equal(lane.isStopped(), true);
});

test('lifecycle: paused while the tab is hidden; stop() aborts the in-flight request and schedules nothing', async () => {
  const h = harness([payload(AT)]);
  const fixes: FastFix[] = [];
  const lane = startSelectedFastLane({ hex: 'ab8c8e', onFix: (f) => fixes.push(f), ...h.opts });
  h.setHidden(true);
  await runNext(h);
  assert.equal(h.urls.length, 0, 'hidden tab: no request');
  assert.equal(h.timers.length, 1, 'but the loop keeps ticking so it resumes when visible');
  h.setHidden(false);
  h.setHold(true);
  const t = h.timers.shift();
  t?.fn();
  await flush();
  assert.equal(h.urls.length, 1);
  lane.stop();
  assert.equal(h.signals[0].aborted, true, 'deselect aborts the request');
  h.gates.forEach((g) => g());
  await flush(); await flush();
  assert.equal(fixes.length, 0, 'an aborted answer is never delivered');
  assert.equal(h.timers.length, 0, 'nothing scheduled after stop');
});

test('invalid hex never polls; a failed poll reports null and the loop continues', async () => {
  const bad = harness([payload(AT)]);
  const l1 = startSelectedFastLane({ hex: 'zz', onFix: () => {}, ...bad.opts });
  await runNext(bad);
  assert.equal(bad.urls.length, 0);
  l1.stop();
  const polls: (FastFix | null)[] = [];
  const failing = {
    ...bad.opts,
    fetchImpl: (async () => ({ ok: false, status: 502, json: async () => ({}) })) as unknown as typeof fetch,
  };
  const timers2: { fn: () => void; ms: number }[] = [];
  const l2 = startSelectedFastLane({
    hex: 'ab8c8e', onFix: () => {}, onPoll: (f) => polls.push(f), ...failing,
    setTimeout: ((fn: () => void, ms: number) => { timers2.push({ fn, ms }); return timers2.length; }) as (fn: () => void, ms: number) => unknown,
  });
  timers2.shift()?.fn();
  await flush(); await flush();
  assert.deepEqual(polls, [null]);
  assert.equal(timers2.length, 1, 'retries on the next tick');
  l2.stop();
});
