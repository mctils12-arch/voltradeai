import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { guardedRefresh } from "./crashSafeRefresh";

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "crashsaferefresh-test-"));
}

test("guardedRefresh: runs fn on a cold start (no prior marker)", async () => {
  const dir = tmpDir();
  let calls = 0;
  const result = await guardedRefresh("test-job", 3600_000, async () => { calls++; }, dir, 1000);
  assert.equal(result.ran, true);
  assert.equal(calls, 1);
});

test("guardedRefresh: a fold that never completes (simulated crash) is not retried within the cooldown", async () => {
  const dir = tmpDir();
  let calls = 0;
  const crashingFn = async () => { calls++; throw new Error("simulated OOM kill mid-fold"); };
  // guardedRefresh only leaves an unresolved marker behind if the process
  // dies before the `finally` runs — an ordinary thrown error still
  // resolves the marker (see the next test). To simulate an actual crash,
  // write the "started, never completed" marker directly, the way a real
  // process death would leave it, then call guardedRefresh again.
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "voltrade_refresh_attempt_test-job.json"), JSON.stringify({ startedAt: 1000 }));

  const workingFn = async () => { calls++; };
  const result = await guardedRefresh("test-job", 3600_000, workingFn, dir, 1000 + 30_000);
  assert.equal(result.ran, false);
  assert.equal(result.reason, "cooldown");
  assert.equal(calls, 0);
  void crashingFn; // referenced for documentation of the scenario being simulated
});

test("guardedRefresh: an ORDINARY caught error still resolves the marker — not treated as a suspected crash", async () => {
  const dir = tmpDir();
  let calls = 0;
  const failingFn = async () => { calls++; throw new Error("ordinary network error"); };
  await assert.rejects(() => guardedRefresh("test-job", 3600_000, failingFn, dir, 1000));
  assert.equal(calls, 1);

  // Next call, moments later — should run again immediately, no cooldown,
  // because the finally block resolved the marker despite the thrown error.
  const workingFn = async () => { calls++; };
  const result = await guardedRefresh("test-job", 3600_000, workingFn, dir, 1000 + 5000);
  assert.equal(result.ran, true);
  assert.equal(calls, 2);
});

test("guardedRefresh: retries once the cooldown window has fully elapsed after a suspected crash", async () => {
  const dir = tmpDir();
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "voltrade_refresh_attempt_test-job.json"), JSON.stringify({ startedAt: 1000 }));

  let calls = 0;
  const workingFn = async () => { calls++; };
  const COOLDOWN_MS = 3600_000;
  const result = await guardedRefresh("test-job", COOLDOWN_MS, workingFn, dir, 1000 + COOLDOWN_MS + 1);
  assert.equal(result.ran, true);
  assert.equal(calls, 1);
});

test("guardedRefresh: a successful run does not block the very next scheduled tick", async () => {
  const dir = tmpDir();
  let calls = 0;
  const workingFn = async () => { calls++; };
  const r1 = await guardedRefresh("test-job", 3600_000, workingFn, dir, 1000);
  // 10 minutes later, the normal poller cadence — must NOT be deferred just
  // because it's within the (crash-only) cooldown window.
  const r2 = await guardedRefresh("test-job", 3600_000, workingFn, dir, 1000 + 10 * 60_000);
  assert.equal(r1.ran, true);
  assert.equal(r2.ran, true);
  assert.equal(calls, 2);
});

test("guardedRefresh: separate job names never interfere with each other's markers", async () => {
  const dir = tmpDir();
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "voltrade_refresh_attempt_job-a.json"), JSON.stringify({ startedAt: 1000 }));

  let calls = 0;
  const workingFn = async () => { calls++; };
  const result = await guardedRefresh("job-b", 3600_000, workingFn, dir, 1000 + 5000);
  assert.equal(result.ran, true);
  assert.equal(calls, 1);
});
