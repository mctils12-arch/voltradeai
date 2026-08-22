// cusipResolver.test.ts — pure-logic + fs-roundtrip coverage, mocked
// fetch throughout (no live network in CI, matching this codebase's own
// stated convention — see dtccSwaps.test.ts's header note). A one-off live
// smoke check against real OpenFIGI + a real DTCC-observed CUSIP was run
// manually this session (see research/experiments.md); it is NOT re-run
// here since a CI test must not depend on a third-party service being up.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  chunk, pickBestMatch, parseOpenFigiBatch,
  loadCusipCache, saveCusipCache, resolveCusips,
} from "./cusipResolver";

type FetchInit = { method: string; headers: Record<string, string>; body: string };
type OpenFigiJob = { idType: string; idValue: string };

test("chunk splits into groups of size, last group may be short", () => {
  assert.deepEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]);
  assert.deepEqual(chunk([], 3), []);
  assert.deepEqual(chunk([1], 10), [[1]]);
});

test("pickBestMatch prefers the US composite exchCode over other venues", () => {
  const data = [
    { figi: "F1", name: "APPLE INC", ticker: "AAPL", exchCode: "UN" },
    { figi: "F2", name: "APPLE INC", ticker: "AAPL", exchCode: "US" },
    { figi: "F3", name: "APPLE INC", ticker: "AAPL", exchCode: "UC" },
  ];
  const best = pickBestMatch(data);
  assert.equal(best?.exchCode, "US");
  assert.equal(best?.figi, "F2");
});

test("pickBestMatch falls back to the first entry when no US listing exists", () => {
  const data = [{ figi: "F1", name: "FOO LTD", ticker: "FOO", exchCode: "LN" }];
  const best = pickBestMatch(data);
  assert.equal(best?.exchCode, "LN");
});

test("pickBestMatch returns null for an empty listing array", () => {
  assert.equal(pickBestMatch([]), null);
});

test("parseOpenFigiBatch order-aligns response jobs to requested CUSIPs, error jobs -> null", () => {
  const cusips = ["037833100", "BADCUSIP1", "594918104"];
  const body = [
    { data: [{ figi: "F1", name: "APPLE INC", ticker: "AAPL", exchCode: "US" }] },
    { error: "No identifier found." },
    { data: [{ figi: "F2", name: "MICROSOFT CORP", ticker: "MSFT", exchCode: "US" }] },
  ];
  const out = parseOpenFigiBatch(cusips, body as any);
  assert.equal(out.get("037833100")?.ticker, "AAPL");
  assert.equal(out.get("BADCUSIP1"), null);
  assert.equal(out.get("594918104")?.ticker, "MSFT");
});

test("cache round-trips through disk; a missing file loads as empty", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "cusip-cache-"));
  const file = path.join(dir, "sub", "cusip_ticker_cache.json");
  assert.deepEqual(loadCusipCache(file), {});
  const cache = {
    "037833100": { result: { ticker: "AAPL", name: "APPLE INC", figi: "F1", exchCode: "US" }, resolvedAt: "2026-08-22T00:00:00.000Z" },
    "BADCUSIP1": { result: null, resolvedAt: "2026-08-22T00:00:00.000Z" },
  };
  saveCusipCache(file, cache);
  assert.deepEqual(loadCusipCache(file), cache);
});

test("resolveCusips: cache hits (positive and negative) never call fetch", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "cusip-cache-"));
  const cachePath = path.join(dir, "cache.json");
  saveCusipCache(cachePath, {
    "037833100": { result: { ticker: "AAPL", name: "APPLE INC", figi: "F1", exchCode: "US" }, resolvedAt: "x" },
    "BADCUSIP1": { result: null, resolvedAt: "x" },
  });
  let calls = 0;
  const fetchImpl = async () => { calls++; return { ok: true, status: 200, json: async () => [] }; };
  const out = await resolveCusips(["037833100", "BADCUSIP1"], { cachePath, fetchImpl });
  assert.equal(calls, 0);
  assert.equal(out.get("037833100")?.ticker, "AAPL");
  assert.equal(out.get("BADCUSIP1"), null);
});

test("resolveCusips: queries only the uncached subset, batches, dedupes, and persists results", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "cusip-cache-"));
  const cachePath = path.join(dir, "cache.json");
  saveCusipCache(cachePath, {
    "594918104": { result: { ticker: "MSFT", name: "MICROSOFT CORP", figi: "F2", exchCode: "US" }, resolvedAt: "x" },
  });
  const requestBodies: OpenFigiJob[][] = [];
  const fetchImpl = async (_url: string, init: FetchInit) => {
    const jobs: OpenFigiJob[] = JSON.parse(init.body);
    requestBodies.push(jobs);
    return {
      ok: true, status: 200,
      json: async () => jobs.map((j) =>
        j.idValue === "037833100"
          ? { data: [{ figi: "F1", name: "APPLE INC", ticker: "AAPL", exchCode: "US" }] }
          : { error: "No identifier found." }),
    };
  };
  // duplicate CUSIP in the input list must be deduped before querying
  const out = await resolveCusips(
    ["037833100", "594918104", "037833100", "UNKNOWNX1"],
    { cachePath, fetchImpl, nowIso: () => "2026-08-22T01:00:00.000Z" },
  );
  assert.equal(requestBodies.length, 1);
  assert.equal(requestBodies[0].length, 2); // only the 2 truly-uncached CUSIPs, deduped
  assert.equal(out.get("037833100")?.ticker, "AAPL");
  assert.equal(out.get("594918104")?.ticker, "MSFT"); // came from cache
  assert.equal(out.get("UNKNOWNX1"), null);

  const persisted = loadCusipCache(cachePath);
  assert.equal(persisted["037833100"].result?.ticker, "AAPL");
  assert.equal(persisted["UNKNOWNX1"].result, null);
  assert.equal(persisted["594918104"].resolvedAt, "x"); // untouched, not re-written
});

test("resolveCusips: an HTTP failure on one batch leaves those CUSIPs uncached (not a crash, not a false negative)", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "cusip-cache-"));
  const cachePath = path.join(dir, "cache.json");
  const fetchImpl = async () => ({ ok: false, status: 429, json: async () => [] });
  const out = await resolveCusips(["037833100"], { cachePath, fetchImpl });
  assert.equal(out.has("037833100"), false); // not resolved AND not falsely cached as null
  assert.deepEqual(loadCusipCache(cachePath), {});
});

test("resolveCusips: sends the API key header only when a key is provided", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "cusip-cache-"));
  const cachePath = path.join(dir, "cache.json");
  let seenHeaders: Record<string, string> = {};
  const fetchImpl = async (_url: string, init: FetchInit) => {
    seenHeaders = init.headers;
    return { ok: true, status: 200, json: async () => [{ error: "No identifier found." }] };
  };
  await resolveCusips(["037833100"], { cachePath, fetchImpl, apiKey: "test-key-123" });
  assert.equal(seenHeaders["X-OPENFIGI-APIKEY"], "test-key-123");

  await resolveCusips(["594918104"], { cachePath, fetchImpl });
  assert.equal("X-OPENFIGI-APIKEY" in seenHeaders, false);
});
