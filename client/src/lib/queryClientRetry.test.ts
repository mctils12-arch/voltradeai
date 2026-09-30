// Field bug 2026-09-30: every stock tab showed "Something went wrong" during a
// Railway restart window (queries had retry: false). Transient failures now
// retry with backoff; real errors still surface immediately.
import { test } from "node:test";
import assert from "node:assert/strict";
import { isTransientError, retryTransient, transientRetryDelay, TRANSIENT_MAX_RETRIES } from "./queryClient";

test("transient: proxy 502/503/504 and pre-response network failures", () => {
  assert.equal(isTransientError(new Error("502: Bad Gateway")), true);
  assert.equal(isTransientError(new Error("503: Service Unavailable")), true);
  assert.equal(isTransientError(new Error("504: Gateway Timeout")), true);
  assert.equal(isTransientError(new TypeError("Failed to fetch")), true);
  assert.equal(isTransientError(new Error("NetworkError when attempting to fetch resource.")), true);
});

test("not transient: real answers surface immediately", () => {
  for (const m of ["400: bad ticker", "401: unauthorized", "404: Not Found", "500: internal error", "429: rate limit"]) {
    assert.equal(isTransientError(new Error(m)), false, m);
  }
  assert.equal(retryTransient(0, new Error("404: Not Found")), false);
});

test("retries are bounded and back off to ~45 s total", () => {
  const e = new Error("502: Bad Gateway");
  assert.equal(retryTransient(0, e), true);
  assert.equal(retryTransient(TRANSIENT_MAX_RETRIES - 1, e), true);
  assert.equal(retryTransient(TRANSIENT_MAX_RETRIES, e), false);
  const total = [0, 1, 2, 3].reduce((s, a) => s + transientRetryDelay(a), 0);
  assert.equal(total, 3000 + 6000 + 12000 + 24000);
  assert.equal(transientRetryDelay(10), 24000);
});
