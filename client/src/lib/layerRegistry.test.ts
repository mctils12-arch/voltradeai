// Regression ratchet for the 2026-10-14 repair: the /data layers panel
// showed NO layer groups for the whole session after one failed registry
// fetch at startup (human report: "I can't see the options for the
// variables... it seems to do this on start up"). The old code was a
// one-shot fetch with `.catch(() => setLayers([]))` and no retry.
// Run: npx tsx --test client/src/lib/layerRegistry.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { fetchLayerRegistry, parseLayerRegistry } from "./layerRegistry.ts";
import { runResilientLoad } from "./resilientLoad.ts";

const GOOD = { layers: [{ id: "imagery" }, { id: "aircraft" }], server_version: "1.0.1065" };

const resp = (status: number, body: unknown, badJson = false) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => { if (badJson) throw new SyntaxError("Unexpected token <"); return body; },
});

test("parseLayerRegistry: accepts a real registry and reads the server version", () => {
  const r = parseLayerRegistry<{ id: string }>(GOOD);
  assert.deepEqual(r.layers.map((l) => l.id), ["imagery", "aircraft"]);
  assert.equal(r.serverVersion, "1.0.1065");
  assert.equal(parseLayerRegistry({ layers: [{ id: "x" }] }).serverVersion, null);
});

test("parseLayerRegistry: every shape that would empty the panel throws (so it retries)", () => {
  for (const bad of [null, undefined, {}, { layers: null }, { layers: "x" }, { layers: [] }, { error: "boom" }]) {
    assert.throws(() => parseLayerRegistry(bad), /layer registry/, JSON.stringify(bad));
  }
});

test("fetchLayerRegistry: non-2xx and non-JSON bodies throw instead of yielding []", async () => {
  await assert.rejects(fetchLayerRegistry(async () => resp(502, GOOD)), /HTTP 502/);
  await assert.rejects(fetchLayerRegistry(async () => resp(200, null, true)), /Unexpected token/);
  await assert.rejects(fetchLayerRegistry(async () => { throw new TypeError("Failed to fetch"); }), /Failed to fetch/);
  const ok = await fetchLayerRegistry<{ id: string }>(async (url) => {
    assert.equal(url, "/api/data/layers");
    return resp(200, GOOD);
  });
  assert.equal(ok.layers.length, 2);
});

test("startup failure recovers: network error, then 502, then success fills the panel", async () => {
  const q: Array<() => void> = [];
  const deps = {
    setTimeout: (fn: () => void, _ms: number) => { q.push(fn); return q.length; },
    clearTimeout: () => {},
  };
  const script = [
    async () => { throw new TypeError("Failed to fetch"); }, // tab restored before the network is up
    async () => resp(502, "<html>bad gateway</html>", true), // deploy switching containers
    async () => resp(200, GOOD),
  ];
  let calls = 0;
  const fetchFn = (_u: string) => script[Math.min(calls++, script.length - 1)]();
  let layers: unknown[] = [];
  const retries: number[] = [];
  const stop = runResilientLoad(async (signal) => {
    const reg = await fetchLayerRegistry(fetchFn as any, signal);
    layers = reg.layers;
  }, (f) => retries.push(f), {}, deps);

  const settle = async () => { for (let i = 0; i < 6; i++) await Promise.resolve(); };
  await settle();
  assert.equal(layers.length, 0, "first attempt failed");
  assert.deepEqual(retries, [0], "the failure is reported (panel shows the retry note)");
  // run the scheduled retries: the timeout timers are no-ops here, the last
  // queued fn is always the loop's next attempt
  q.pop()!(); await settle();
  assert.equal(layers.length, 0);
  assert.deepEqual(retries, [0, 1]);
  q.pop()!(); await settle();
  assert.equal(layers.length, 2, "registry landed after retrying, so the groups render");
  assert.equal(calls, 3);
  stop();
});

test("datamap.tsx loads the registry through the retry loop, never a bare one-shot", () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const src = readFileSync(join(here, "..", "pages", "datamap.tsx"), "utf8");
  // the exact defect: a single fetch whose failure pinned the panel to []
  assert.doesNotMatch(src, /\.catch\(\(\)\s*=>\s*setLayers\(\[\]\)\)/);
  assert.doesNotMatch(src, /fetch\("\/api\/data\/layers"\)/);
  assert.match(src, /runResilientLoad\(async \(signal\) => \{\s*const reg = await fetchLayerRegistry/);
  // a failed load is visible, not silent
  assert.match(src, /data-vt-registry-retry/);
});
