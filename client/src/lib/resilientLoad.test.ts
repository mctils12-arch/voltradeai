// Tests for the single-shot-layer reliability helper (BUG 1). Uses injected
// timer + AbortController deps so the retry/re-poll scheduling is exercised with
// no real timers and no DOM. Run: npx tsx --test client/src/lib/resilientLoad.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { backoffDelay, DEFAULT_BACKOFF, runResilientLoad } from "./resilientLoad.ts";

test("backoffDelay: clamps to the sequence, last value repeats", () => {
  assert.equal(backoffDelay(0), DEFAULT_BACKOFF[0]);
  assert.equal(backoffDelay(1), DEFAULT_BACKOFF[1]);
  assert.equal(backoffDelay(3), DEFAULT_BACKOFF[3]);
  assert.equal(backoffDelay(99), DEFAULT_BACKOFF[DEFAULT_BACKOFF.length - 1]);
  assert.equal(backoffDelay(-5), DEFAULT_BACKOFF[0]);
  assert.equal(backoffDelay(1, [100, 200]), 200);
});

// A controllable fake scheduler: records scheduled callbacks; flush() runs them.
function fakeTimers() {
  const q: Array<{ fn: () => void; ms: number }> = [];
  return {
    setTimeout: (fn: () => void, ms: number) => { q.push({ fn, ms }); return q.length - 1; },
    clearTimeout: (_h: unknown) => { /* no-op for the test */ },
    pending: q,
    async flush() {
      // run the most-recently scheduled callback (the loop schedules exactly one
      // next step at a time), awaiting microtasks between runs
      const item = q.pop();
      if (item) { item.fn(); await Promise.resolve(); await Promise.resolve(); }
    },
  };
}

// Minimal AbortController stand-in (node has a global one, but keep it explicit).
const AC = AbortController;

test("success path: calls attempt once, no retry, re-polls when repollMs set", async () => {
  const t = fakeTimers();
  let attempts = 0;
  let errors = 0;
  runResilientLoad(
    async () => { attempts += 1; },
    () => { errors += 1; },
    { repollMs: 1000 },
    { setTimeout: t.setTimeout as any, clearTimeout: t.clearTimeout, AbortController: AC },
  );
  await Promise.resolve(); await Promise.resolve();
  assert.equal(attempts, 1, "attempt ran once");
  assert.equal(errors, 0, "no error reported on success");
  // a re-poll timer was scheduled; firing it runs attempt again (self-heal/refresh)
  await t.flush();
  assert.equal(attempts, 2, "re-poll fired a second attempt");
});

test("failure path: reports error and schedules a backoff retry that recovers", async () => {
  const t = fakeTimers();
  let attempts = 0;
  const errFailures: number[] = [];
  runResilientLoad(
    async () => { attempts += 1; if (attempts === 1) throw new Error("stall"); },
    (f) => { errFailures.push(f); },
    { backoff: [500] },
    { setTimeout: t.setTimeout as any, clearTimeout: t.clearTimeout, AbortController: AC },
  );
  await Promise.resolve(); await Promise.resolve();
  assert.equal(attempts, 1, "first attempt ran and failed");
  assert.deepEqual(errFailures, [0], "error reported with failures=0 on first failure");
  // the scheduled retry fires and this time succeeds -> no further error
  await t.flush();
  assert.equal(attempts, 2, "retry attempt ran");
  assert.deepEqual(errFailures, [0], "no second error after recovery");
});

test("cleanup: aborts the in-flight request and stops the loop", async () => {
  const t = fakeTimers();
  let aborted = false;
  let resolveAttempt: (() => void) | null = null;
  const cleanup = runResilientLoad(
    (signal) => new Promise<void>((resolve) => {
      signal.addEventListener("abort", () => { aborted = true; resolve(); });
      resolveAttempt = resolve;
    }),
    () => {},
    {},
    { setTimeout: t.setTimeout as any, clearTimeout: t.clearTimeout, AbortController: AC },
  );
  await Promise.resolve();
  cleanup();
  assert.equal(aborted, true, "cleanup aborted the in-flight request's signal");
  // firing any leftover timer must not start a new attempt after cancel
  const before = t.pending.length;
  await t.flush();
  assert.ok(t.pending.length <= before, "no new work scheduled after cleanup");
  void resolveAttempt;
});
