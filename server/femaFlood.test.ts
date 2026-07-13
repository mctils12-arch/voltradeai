// femaFlood.test.ts — FEMA NFHL flood-zone point lookup (Location Context
// Engine hazard layer #3). Pure-function + injected-fetch coverage, same
// convention as superfund.test.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseFloodFeature, fetchFloodZone, floodZoneAt, clearFloodZoneCache, queryUrl,
  FLOOD_ZONE_MEANINGS, KNOWN_FLOOD_ZONES,
} from "./femaFlood";

test("every FLOOD_ZONE_MEANINGS entry has a non-empty, distinct meaning (data-error guard: no code silently maps to '')", () => {
  const seen = new Set<string>();
  for (const zone of KNOWN_FLOOD_ZONES) {
    const meaning = FLOOD_ZONE_MEANINGS[zone];
    assert.ok(meaning && meaning.length > 10, `${zone} has no real meaning text`);
    assert.ok(!seen.has(meaning), `${zone} duplicates another zone's meaning`);
    seen.add(meaning);
  }
});

test("parseFloodFeature: a real SFHA feature (VE, coastal) decodes correctly, BFE preserved", () => {
  const out = parseFloodFeature({ attributes: { FLD_ZONE: "VE", ZONE_SUBTY: null, SFHA_TF: "T", STATIC_BFE: 9.0, SOURCE_CIT: "12086C_FIS1" } });
  assert.equal(out.zone, "VE");
  assert.equal(out.sfha, true);
  assert.equal(out.base_flood_elevation_ft, 9.0);
  assert.equal(out.ready, true);
  assert.ok(out.meaning!.includes("velocity wave action"));
});

test("parseFloodFeature: FEMA's -9999 no-data sentinel is converted to null, never displayed as a real elevation", () => {
  const out = parseFloodFeature({ attributes: { FLD_ZONE: "X", ZONE_SUBTY: "AREA OF MINIMAL FLOOD HAZARD", SFHA_TF: "F", STATIC_BFE: -9999.0, SOURCE_CIT: "40081C_STUDY1" } });
  assert.equal(out.base_flood_elevation_ft, null);
  assert.equal(out.sfha, false);
  assert.equal(out.zone, "X");
});

test("parseFloodFeature: an unrecognized FLD_ZONE is quarantined (unavailable), never guessed", () => {
  const out = parseFloodFeature({ attributes: { FLD_ZONE: "ZZ_NOT_A_REAL_CODE", SFHA_TF: "T" } });
  assert.equal(out.zone, null);
  assert.equal(out.ready, false);
});

test("parseFloodFeature: missing/null FLD_ZONE is quarantined, not crashed on", () => {
  const out = parseFloodFeature({ attributes: {} });
  assert.equal(out.ready, false);
  const out2 = parseFloodFeature({});
  assert.equal(out2.ready, false);
});

test("parseFloodFeature: an unrecognized SFHA_TF value degrades sfha to null rather than guessing true/false", () => {
  const out = parseFloodFeature({ attributes: { FLD_ZONE: "AE", SFHA_TF: "?" } });
  assert.equal(out.sfha, null);
});

test("queryUrl: point geometry is lon,lat order (Esri convention) and targets layer 28", () => {
  const url = queryUrl(35.94, -96.75);
  assert.ok(url.includes("MapServer/28/query"));
  assert.ok(url.includes("geometry=-96.75%2C35.94") || url.includes("geometry=-96.75,35.94"));
  assert.ok(url.includes("geometryType=esriGeometryPoint"));
});

test("fetchFloodZone: a real feature returned by the mock fetch parses through", async () => {
  const mockFetch = async () => ({
    ok: true, status: 200,
    text: async () => JSON.stringify({ features: [{ attributes: { FLD_ZONE: "AE", SFHA_TF: "T", STATIC_BFE: 12.5 } }] }),
  });
  const out = await fetchFloodZone(29.95, -90.07, mockFetch as any);
  assert.equal(out.zone, "AE");
  assert.equal(out.sfha, true);
});

test("fetchFloodZone: zero features (confirmed live over interior Alaska) is an honest 'no NFHL coverage', not a fabricated 'minimal risk'", async () => {
  const mockFetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify({ features: [] }) });
  const out = await fetchFloodZone(65.0, -153.0, mockFetch as any);
  assert.equal(out.zone, null);
  assert.equal(out.ready, true); // resolved answer: genuinely unmapped, not a failure
});

test("fetchFloodZone: an HTTP error degrades to ready:false, never throws", async () => {
  const mockFetch = async () => ({ ok: false, status: 503, text: async () => "" });
  const out = await fetchFloodZone(29.95, -90.07, mockFetch as any);
  assert.equal(out.ready, false);
});

test("fetchFloodZone: a network exception degrades to ready:false, never throws", async () => {
  const mockFetch = async () => { throw new Error("timeout"); };
  const out = await fetchFloodZone(29.95, -90.07, mockFetch as any);
  assert.equal(out.ready, false);
});

test("fetchFloodZone: invalid coordinates never reach the network", async () => {
  let called = false;
  const mockFetch = async () => { called = true; return { ok: true, status: 200, text: async () => "{}" }; };
  const out = await fetchFloodZone(999, 999, mockFetch as any);
  assert.equal(called, false);
  assert.equal(out.ready, false);
});

test("floodZoneAt: caches a resolved result — second call within TTL never hits fetch again", async () => {
  clearFloodZoneCache();
  let calls = 0;
  const mockFetch = async () => { calls++; return { ok: true, status: 200, text: async () => JSON.stringify({ features: [{ attributes: { FLD_ZONE: "X", SFHA_TF: "F" } }] }) }; };
  const a = await floodZoneAt(30.0, -90.0, mockFetch as any, 1000);
  const b = await floodZoneAt(30.0, -90.0, mockFetch as any, 2000);
  assert.equal(calls, 1);
  assert.equal(a.zone, "X");
  assert.deepEqual(a, b);
});

test("floodZoneAt: a cache entry past TTL is refetched, not served stale forever", async () => {
  clearFloodZoneCache();
  let calls = 0;
  const mockFetch = async () => { calls++; return { ok: true, status: 200, text: async () => JSON.stringify({ features: [{ attributes: { FLD_ZONE: "X", SFHA_TF: "F" } }] }) }; };
  await floodZoneAt(31.0, -91.0, mockFetch as any, 0);
  await floodZoneAt(31.0, -91.0, mockFetch as any, 31 * 86400_000 + 1);
  assert.equal(calls, 2);
});

test("floodZoneAt: a failed lookup (ready:false) is NOT cached — the next call retries instead of freezing an outage", async () => {
  clearFloodZoneCache();
  let calls = 0;
  const mockFetch = async () => {
    calls++;
    if (calls === 1) return { ok: false, status: 503, text: async () => "" };
    return { ok: true, status: 200, text: async () => JSON.stringify({ features: [{ attributes: { FLD_ZONE: "AE", SFHA_TF: "T" } }] }) };
  };
  const a = await floodZoneAt(32.0, -92.0, mockFetch as any, 0);
  assert.equal(a.ready, false);
  const b = await floodZoneAt(32.0, -92.0, mockFetch as any, 100);
  assert.equal(calls, 2);
  assert.equal(b.ready, true);
  assert.equal(b.zone, "AE");
});
