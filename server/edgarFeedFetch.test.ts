// REGRESSION (2026-11-19/20): each EDGAR poller asked for its feed page once
// with a 12 s limit, and one slow answer lost the whole poll. Production
// 2026-11-19: about 20 failed polls of 78 on each of the three streams;
// 2026-11-20 02:34Z: Form 4 at 2 failed of 6, last error "The operation was
// aborted due to timeout". The feed request is now retried
// (server/edgarFeedFetch.ts) and the retries are counted on the probe.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import {
  EDGAR_FEED_ATTEMPT_TIMEOUTS_MS, EDGAR_FEED_RETRY_PAUSE_MS, edgarFeedFailureIsRetryable, fetchEdgarFeed,
  type EdgarFeedFetchFn,
} from "./edgarFeedFetch";
import { _resetEdgarPollHealthForTests, edgarPollHealthStatus, type EdgarPollHealthRow } from "./edgarPollHealth";
import { refreshForm4Cache, _resetForm4CacheForTests, edgarCurrentFeedUrl } from "./edgarForm4";
import { sanitizeDiag } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
const URL4 = edgarCurrentFeedUrl("4", 75);
const UA = { "User-Agent": "test" };
const FAST = { pauseMs: 0 };

function row(stream: string): EdgarPollHealthRow {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "edgar-feed-fetch-"));
  const r = (edgarPollHealthStatus(dir).streams as EdgarPollHealthRow[]).find((s) => s.stream === stream);
  assert.ok(r, `${stream} in the probe`);
  return r;
}
function timeoutError(): Error {
  return new DOMException("The operation was aborted due to timeout", "TimeoutError");
}
/** A fetch that answers each call from a script; the last step repeats. */
function scripted(steps: Array<number | "timeout" | "torn">, body = "<feed></feed>") {
  const calls: Array<{ url: string; hasSignal: boolean }> = [];
  const impl: EdgarFeedFetchFn = async (url, init) => {
    const step = steps[Math.min(calls.length, steps.length - 1)];
    calls.push({ url, hasSignal: !!init?.signal });
    if (step === "timeout") throw timeoutError();
    if (step === "torn") return { ok: true, status: 200, text: async () => { throw new Error("terminated"); } };
    return { ok: step >= 200 && step < 300, status: step, text: async () => body };
  };
  return { impl, calls };
}

test("a feed request that times out is asked again, and the poll is counted as saved by the retry", async () => {
  _resetEdgarPollHealthForTests();
  const f = scripted(["timeout", 200], "<feed>page</feed>");
  assert.equal(await fetchEdgarFeed(f.impl, URL4, UA, "filings", FAST), "<feed>page</feed>");
  assert.equal(f.calls.length, 2);
  assert.ok(f.calls.every((c) => c.url === URL4 && c.hasSignal), "the same URL, each attempt with its own time limit");
  const r = row("filings");
  assert.equal(r.feed_attempts_failed, 1);
  assert.equal(r.polls_saved_by_retry, 1);
  assert.equal(row("filings13f").feed_attempts_failed, 0, "streams are counted apart");
});

test("a first answer costs no retry and counts nothing", async () => {
  _resetEdgarPollHealthForTests();
  const f = scripted([200]);
  await fetchEdgarFeed(f.impl, URL4, UA, "earnings8k", FAST);
  assert.equal(f.calls.length, 1);
  assert.equal(row("earnings8k").feed_attempts_failed, 0);
  assert.equal(row("earnings8k").polls_saved_by_retry, 0);
});

test("a 5xx and a body that breaks mid-read are retried", async () => {
  _resetEdgarPollHealthForTests();
  const f = scripted([503, "torn", 200], "ok");
  assert.equal(await fetchEdgarFeed(f.impl, URL4, UA, "filings13f", FAST), "ok");
  assert.equal(f.calls.length, 3);
  assert.equal(row("filings13f").feed_attempts_failed, 2);
  assert.equal(row("filings13f").polls_saved_by_retry, 1, "one poll, however many attempts it took");
});

test("a status a retry cannot change is not asked again: 400 (2026-11-16), 403, 404, 429", async () => {
  for (const status of [400, 403, 404, 429]) {
    _resetEdgarPollHealthForTests();
    const f = scripted([status, 200]);
    await assert.rejects(fetchEdgarFeed(f.impl, URL4, UA, "filings", FAST), new RegExp(`sec\\.gov.* -> ${status}$`));
    assert.equal(f.calls.length, 1, `${status}: one request`);
    assert.equal(row("filings").feed_attempts_failed, 0, `${status}: not counted as a retried attempt`);
  }
  assert.equal(edgarFeedFailureIsRetryable(null), true);
  assert.equal(edgarFeedFailureIsRetryable(500), true);
  assert.equal(edgarFeedFailureIsRetryable(599), true);
  assert.equal(edgarFeedFailureIsRetryable(600), false);
  assert.equal(edgarFeedFailureIsRetryable(499), false);
});

test("when every attempt fails the last error is thrown after exactly the configured number of requests", async () => {
  _resetEdgarPollHealthForTests();
  const f = scripted(["timeout"]);
  await assert.rejects(fetchEdgarFeed(f.impl, URL4, UA, "filings", FAST), /aborted due to timeout/);
  assert.equal(f.calls.length, EDGAR_FEED_ATTEMPT_TIMEOUTS_MS.length);
  assert.equal(row("filings").feed_attempts_failed, EDGAR_FEED_ATTEMPT_TIMEOUTS_MS.length);
  assert.equal(row("filings").polls_saved_by_retry, 0);
});

test("each attempt is cut off at its own limit, and a hung connection cannot hold the poll", async () => {
  _resetEdgarPollHealthForTests();
  const waited: number[] = [];
  const hang: EdgarFeedFetchFn = (_url, init) => new Promise((_res, rej) => {
    const t0 = Date.now();
    init.signal.addEventListener("abort", () => { waited.push(Date.now() - t0); rej(init.signal.reason); });
  });
  // AbortSignal.timeout's timer does not hold the event loop open (a real
  // request's socket does); this one stands in for the socket.
  const socket = setInterval(() => {}, 1000);
  try {
    await assert.rejects(fetchEdgarFeed(hang, URL4, UA, "filings", { timeoutsMs: [30, 120], pauseMs: 0 }), /timeout/i);
  } finally {
    clearInterval(socket);
  }
  assert.equal(waited.length, 2);
  assert.ok(waited[0] < 110, `the first attempt stopped at its limit (${waited[0]} ms)`);
  assert.ok(waited[1] >= 100, `the second attempt was given longer (${waited[1]} ms)`);
});

test("the limits: three attempts, none shorter than the one before, under two minutes in all", () => {
  const t = EDGAR_FEED_ATTEMPT_TIMEOUTS_MS;
  assert.equal(t.length, 3);
  assert.equal(t[0], 12_000, "the first attempt keeps the limit the pollers had");
  for (let i = 1; i < t.length; i++) assert.ok(t[i] >= t[i - 1]);
  const worst = t.reduce((a, b) => a + b, 0) + (t.length - 1) * EDGAR_FEED_RETRY_PAUSE_MS;
  assert.ok(worst < 120_000, "far inside the 15-minute poll interval");
});

test("REGRESSION: the Form 4 poll of 2026-11-20 02:32Z (feed timed out once) is now a good poll", async () => {
  _resetEdgarPollHealthForTests();
  _resetForm4CacheForTests();
  const f = scripted(["timeout", 200]);
  const errors: unknown[][] = [];
  const orig = console.error;
  console.error = (...a: unknown[]) => { errors.push(a); };
  try {
    await refreshForm4Cache(25, f.impl);
  } finally {
    console.error = orig;
  }
  const r = row("filings");
  assert.equal(r.polls_ok, 1);
  assert.equal(r.polls_failed, 0);
  assert.equal(r.consecutive_failures, 0);
  assert.equal(r.polls_saved_by_retry, 1);
  assert.equal(errors.length, 0, "no failed-poll line");
});

test("all three pollers ask for their feed page through the retrying request, and the probe serves its counters", () => {
  for (const [file, stream] of [["edgarForm4.ts", "filings"], ["edgar13f.ts", "filings13f"], ["sec8kEarnings.ts", "earnings8k"]] as const) {
    const src = fs.readFileSync(path.join(here, file), "utf8");
    assert.ok(src.includes(`await fetchEdgarFeed(fetchImpl, feedUrl, SEC_UA, "${stream}")`), `${file} retries its feed request`);
    assert.ok(!/fetchText\(fetchImpl, feedUrl\)/.test(src), `${file} has no single-attempt feed request left`);
  }
  _resetEdgarPollHealthForTests();
  const status = edgarPollHealthStatus(fs.mkdtempSync(path.join(os.tmpdir(), "edgar-feed-fetch-")));
  for (const s of status.streams as EdgarPollHealthRow[]) {
    assert.equal(s.feed_attempts_failed, 0);
    assert.equal(s.polls_saved_by_retry, 0);
  }
  assert.deepEqual(sanitizeDiag(status), status, "the new counters pass the diag scrub");
});
