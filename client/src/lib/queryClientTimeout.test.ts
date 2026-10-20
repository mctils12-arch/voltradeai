// Field bug 2026-10-20: the Bot page stayed "not loaded" until a manual
// reload. Root cause, reproduced: one request that never answers (a
// connection held open across a Railway deploy) blocks react-query polling
// for good — an interval refetch joins the in-flight request instead of
// starting a new one. Polled reads now pass a timeout; the abort surfaces as
// a "timeout:" error that retries like a dropped connection.
// Run: npx tsx --test client/src/lib/queryClientTimeout.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { QueryClient, QueryObserver } from "@tanstack/query-core";
import { apiRequest, isTransientError, retryTransient } from "./queryClient";

type FetchFn = typeof fetch;
function withFetch(fn: FetchFn, body: () => Promise<void>): Promise<void> {
  const orig = globalThis.fetch;
  globalThis.fetch = fn;
  return body().finally(() => { globalThis.fetch = orig; });
}
/** first call never answers (unless aborted); later calls answer 200 {ok:true} */
function hangThenOk(): { fetch: FetchFn; calls: () => number } {
  let n = 0;
  const f = ((_url: string, init?: RequestInit) => {
    n++;
    if (n === 1) {
      return new Promise<Response>((_res, rej) => {
        init?.signal?.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError")));
      });
    }
    return Promise.resolve(new Response(JSON.stringify({ ok: true }), { status: 200 }));
  }) as FetchFn;
  return { fetch: f, calls: () => n };
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

test("a GET with timeoutMs that gets no answer rejects with a transient 'timeout:' error", async () => {
  const h = hangThenOk();
  await withFetch(h.fetch, async () => {
    const err = await apiRequest("GET", "/api/bot/positions", undefined, { timeoutMs: 50 }).catch((e) => e);
    assert.match(String(err?.message), /^timeout: no answer from \/api\/bot\/positions in 50 ms/);
    assert.equal(isTransientError(err), true, "retried like a dropped connection");
    const ok = await apiRequest("GET", "/api/bot/positions", undefined, { timeoutMs: 50 });
    assert.equal(ok.status, 200);
  });
});

test("no timeoutMs: behavior unchanged (slow GETs and mutations are never cut off)", async () => {
  await withFetch((async () => { await sleep(80); return new Response("{}", { status: 200 }); }) as FetchFn, async () => {
    const r = await apiRequest("POST", "/api/bot/toggle", { on: true });
    assert.equal(r.status, 200, "an 80 ms answer with no timeout set is still awaited");
  });
});

test("the reproduced bug: without a timeout one hung request stalls polling forever; with it, polling recovers", async () => {
  // before: queryFn without a timeout — the interval keeps joining the hung request
  const stuck = hangThenOk();
  await withFetch(stuck.fetch, async () => {
    const qc = new QueryClient();
    const obs = new QueryObserver(qc, {
      queryKey: ["stuck"], refetchInterval: 50, retry: false,
      queryFn: async () => (await apiRequest("GET", "/api/bot/status")).json(),
    });
    let data: unknown;
    const off = obs.subscribe((r) => { data = r.data; });
    await sleep(600);
    off(); qc.clear();
    assert.equal(stuck.calls(), 1, "polling never issued a second request");
    assert.equal(data, undefined);
  });
  // after: the same hang with the Bot page's timeout — data arrives without a reload
  const healed = hangThenOk();
  await withFetch(healed.fetch, async () => {
    const qc = new QueryClient();
    const obs = new QueryObserver(qc, {
      queryKey: ["healed"], refetchInterval: 50, retry: retryTransient, retryDelay: 10,
      queryFn: async () => (await apiRequest("GET", "/api/bot/status", undefined, { timeoutMs: 100 })).json(),
    });
    let data: unknown;
    const off = obs.subscribe((r) => { data = r.data; });
    await sleep(600);
    off(); qc.clear();
    assert.deepEqual(data, { ok: true }, "recovered on its own");
    assert.ok(healed.calls() >= 2);
  });
});
