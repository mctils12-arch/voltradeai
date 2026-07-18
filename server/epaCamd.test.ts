import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import zlib from "zlib";
import { fileURLToPath } from "url";
import {
  parseFacilityAttrs, parseDailyEmissions, aggregateByFacility,
  quarterOf, quarterBounds, latestClosedQuarter, priorQuarters,
  epaCamdKey, epaCamdUsingDemoKey,
  archiveQuarterRows, quarterSealed, gzipSealedQuarters, pickQuarterToFetch,
  fetchFacilityAttrs, fetchQuarterDaily,
} from "./epaCamd";

const here = path.dirname(fileURLToPath(import.meta.url));

// Real response shapes, live-verified against api.epa.gov/easey 2026-07-18
// (DEMO_KEY, stateCode=TX).
const FACILITY_ATTRS = [
  { stateCode: "TX", facilityName: "Copper Station", facilityId: 9, unitId: "CTG-1",
    year: 2026, latitude: 31.7569, longitude: -106.375,
    ownerOperator: "El Paso Electric Company (Owner)|El Paso Electric Company (Operator)",
    primaryFuelInfo: "Pipeline Natural Gas" },
  { stateCode: "TX", facilityName: "Limestone", facilityId: 298, unitId: "LIM1",
    year: 2026, latitude: 31.4219, longitude: -96.2525,
    ownerOperator: "NRG Energy, Inc (Owner)|NRG Energy, Inc (Operator)",
    primaryFuelInfo: "Coal" },
];

const DAILY = [
  { stateCode: "TX", facilityName: "Copper Station", facilityId: 9, unitId: "CTG-1",
    date: "2026-06-01", countOpTime: 19, sumOpTime: 17.46, grossLoad: 809.08,
    steamLoad: null, so2Mass: null, co2Mass: null, noxMass: 1.52, noxRate: 0.2299,
    heatInput: 12421.736, primaryFuelInfo: "Pipeline Natural Gas", unitType: "Combustion turbine" },
  { stateCode: "TX", facilityName: "Limestone", facilityId: 298, unitId: "LIM1",
    date: "2026-06-01", countOpTime: 24, sumOpTime: 24, grossLoad: 1096,
    steamLoad: null, so2Mass: 0.004, co2Mass: 787, noxMass: 0.614,
    heatInput: 13240.3, primaryFuelInfo: "Pipeline Natural Gas", unitType: "Dry bottom wall-fired boiler" },
  { stateCode: "TX", facilityName: "Limestone", facilityId: 298, unitId: "LIM1",
    date: "2026-06-02", countOpTime: 0, sumOpTime: 0, grossLoad: null,
    heatInput: null, primaryFuelInfo: "Pipeline Natural Gas", unitType: "Dry bottom wall-fired boiler" },
];

test("parseFacilityAttrs: keys by facilityId|unitId, geo + owner joined", () => {
  const m = parseFacilityAttrs(FACILITY_ATTRS);
  assert.equal(m.size, 2);
  assert.deepEqual(m.get("9|CTG-1"), {
    lat: 31.7569, lon: -106.375,
    ownerOperator: "El Paso Electric Company (Owner)|El Paso Electric Company (Operator)",
  });
});

test("parseDailyEmissions: joins facility attrs onto each row, nulls where lookup misses", () => {
  const attrs = parseFacilityAttrs(FACILITY_ATTRS);
  const rows = parseDailyEmissions(DAILY, "2026-07-18", attrs);
  assert.equal(rows.length, 3);
  assert.equal(rows[0].lat, 31.7569);
  assert.equal(rows[0].ownerOperator, "El Paso Electric Company (Owner)|El Paso Electric Company (Operator)");
  assert.equal(rows[0].rt, "2026-07-18");
  assert.equal(rows[2].grossLoad, null, "explicit null preserved, never coerced to 0");
});

test("parseDailyEmissions: unknown unit gets null geo, never fabricated", () => {
  const attrs = parseFacilityAttrs(FACILITY_ATTRS);
  const rows = parseDailyEmissions(
    [{ facilityId: 999, unitId: "X", date: "2026-06-01" }], "2026-07-18", attrs,
  );
  assert.equal(rows[0].lat, null);
  assert.equal(rows[0].ownerOperator, null);
});

test("aggregateByFacility: sums grossLoad/opTime per facility, distinct unit count, sorted by output", () => {
  const attrs = parseFacilityAttrs(FACILITY_ATTRS);
  const rows = parseDailyEmissions(DAILY, "2026-07-18", attrs);
  const agg = aggregateByFacility(rows);
  assert.equal(agg.length, 2);
  assert.equal(agg[0].facilityId, 298, "Limestone (higher total grossLoad) sorts first");
  const limestone = agg.find((f) => f.facilityId === 298)!;
  assert.equal(limestone.sumGrossLoad, 1096, "null grossLoad day contributes 0, not NaN");
  assert.equal(limestone.sumOpTime, 24);
  assert.equal(limestone.unitCount, 1);
});

test("quarter math: quarterOf/quarterBounds/latestClosedQuarter/priorQuarters", () => {
  assert.deepEqual(quarterOf(new Date("2026-07-18T00:00:00Z")), { year: 2026, quarter: 3 });
  assert.deepEqual(quarterBounds(2026, 2), { begin: "2026-04-01", end: "2026-06-30" });
  assert.deepEqual(quarterBounds(2026, 1), { begin: "2026-01-01", end: "2026-03-31" });
  assert.deepEqual(
    latestClosedQuarter(new Date("2026-07-18T00:00:00Z")), { year: 2026, quarter: 2 },
    "the CURRENT quarter is never closed — live-probed 2026-07-18: Q3 rejected by the real API",
  );
  assert.deepEqual(
    latestClosedQuarter(new Date("2026-01-15T00:00:00Z")), { year: 2025, quarter: 4 },
    "year rollover",
  );
  const pq = priorQuarters({ year: 2026, quarter: 2 }, 4);
  assert.deepEqual(pq, [
    { year: 2026, quarter: 2 }, { year: 2026, quarter: 1 },
    { year: 2025, quarter: 4 }, { year: 2025, quarter: 3 },
  ]);
});

test("key gating: DEMO_KEY fallback when EPA_CAMD_API_KEY unset, real key otherwise", () => {
  assert.equal(epaCamdKey({}), "DEMO_KEY");
  assert.equal(epaCamdUsingDemoKey({}), true);
  assert.equal(epaCamdKey({ EPA_CAMD_API_KEY: "real123" }), "real123");
  assert.equal(epaCamdUsingDemoKey({ EPA_CAMD_API_KEY: "real123" }), false);
});

test("fetchFacilityAttrs / fetchQuarterDaily: correct URL shape, injected fetch", async () => {
  let seenAttrsUrl = "";
  await fetchFacilityAttrs(2026, "DEMO_KEY", (async (url: string) => {
    seenAttrsUrl = url;
    return { ok: true, status: 200, text: async () => JSON.stringify(FACILITY_ATTRS) };
  }) as any);
  assert.ok(seenAttrsUrl.includes("stateCode=TX") && seenAttrsUrl.includes("year=2026")
    && seenAttrsUrl.includes("api_key=DEMO_KEY"));

  let seenDailyUrl = "";
  const rows = await fetchQuarterDaily(2026, 2, "DEMO_KEY", parseFacilityAttrs(FACILITY_ATTRS), (async (url: string) => {
    seenDailyUrl = url;
    return { ok: true, status: 200, text: async () => JSON.stringify(DAILY) };
  }) as any, Date.parse("2026-07-18T00:00:00Z"));
  assert.ok(seenDailyUrl.includes("beginDate=2026-04-01") && seenDailyUrl.includes("endDate=2026-06-30"));
  assert.equal(rows.length, 3);
  assert.equal(rows[0].rt, "2026-07-18");
});

test("fetchQuarterDaily: non-200 throws (never silently swallowed)", async () => {
  await assert.rejects(
    fetchQuarterDaily(2026, 2, "DEMO_KEY", new Map(), (async () => ({ ok: false, status: 429, text: async () => "" })) as any),
    /429/,
  );
});

test("archive: dedup by (facility,unit,date,values) — revision appends, identical re-fetch no-ops", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtepacamd-"));
  const attrs = parseFacilityAttrs(FACILITY_ATTRS);
  const rows = parseDailyEmissions(DAILY, "2026-07-19", attrs);
  assert.equal(archiveQuarterRows(rows, 2026, 2, dir), 3);
  assert.equal(archiveQuarterRows(rows, 2026, 2, dir), 0, "identical values never re-archive");
  const revised = rows.map((r) => r.date === "2026-06-01" && r.facilityId === 298
    ? { ...r, grossLoad: 1100, rt: "2026-08-01" } : r);
  assert.equal(archiveQuarterRows(revised, 2026, 2, dir), 1, "changed value appends a new vintage row");
  const lines = fs.readFileSync(path.join(dir, "epacamd", "TX_2026Q2.jsonl"), "utf8").trim().split("\n");
  assert.equal(lines.length, 4, "original 3 + 1 revision, never overwritten");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("quarterSealed: not sealed until 45+ days past quarter-end AND archived", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtepacamd-"));
  const attrs = parseFacilityAttrs(FACILITY_ATTRS);
  const rows = parseDailyEmissions(DAILY, "2026-07-19", attrs);
  archiveQuarterRows(rows, 2026, 2, dir);
  const q2End = Date.parse("2026-06-30T00:00:00Z");
  assert.equal(quarterSealed(2026, 2, q2End + 30 * 86_400_000, dir), false, "within 45d window");
  assert.equal(quarterSealed(2026, 2, q2End + 46 * 86_400_000, dir), true, "past window + archived");
  assert.equal(quarterSealed(2026, 1, q2End + 46 * 86_400_000, dir), false, "never archived -> not sealed");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("gzipSealedQuarters: gzips + removes only sealed quarter files, sealed check still finds the .gz", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtepacamd-"));
  const attrs = parseFacilityAttrs(FACILITY_ATTRS);
  const rows = parseDailyEmissions(DAILY, "2026-07-19", attrs);
  archiveQuarterRows(rows, 2026, 2, dir);
  archiveQuarterRows(rows, 2026, 3, dir);
  const q2Sealed = Date.parse("2026-06-30T00:00:00Z") + 46 * 86_400_000;
  const n = gzipSealedQuarters(q2Sealed, dir);
  assert.equal(n, 1, "only Q2 is sealed at this timestamp; Q3 (not yet 45d past its own quarter-end) stays untouched");
  assert.ok(fs.existsSync(path.join(dir, "epacamd", "TX_2026Q2.jsonl.gz")));
  assert.ok(!fs.existsSync(path.join(dir, "epacamd", "TX_2026Q2.jsonl")));
  const gz = zlib.gunzipSync(fs.readFileSync(path.join(dir, "epacamd", "TX_2026Q2.jsonl.gz"))).toString("utf8");
  assert.equal(gz.trim().split("\n").length, 3);
  assert.equal(quarterSealed(2026, 2, q2Sealed, dir), true, "sealed check finds the .gz file after rotation");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("pickQuarterToFetch: chronological backfill first, then recheck-latest, then null at steady state", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vtepacamd-"));
  const now = Date.parse("2026-07-18T00:00:00Z"); // latestClosedQuarter = 2026Q2
  assert.deepEqual(
    pickQuarterToFetch(now, dir, 3), { year: 2025, quarter: 4 },
    "nothing archived yet -> oldest quarter in the 3-quarter backfill window first",
  );
  const attrs = parseFacilityAttrs(FACILITY_ATTRS);
  const rows = parseDailyEmissions(DAILY, "2026-07-18", attrs);
  archiveQuarterRows(rows, 2025, 4, dir);
  assert.deepEqual(pickQuarterToFetch(now, dir, 3), { year: 2026, quarter: 1 }, "next-oldest unbackfilled quarter");
  archiveQuarterRows(rows, 2026, 1, dir);
  assert.deepEqual(
    pickQuarterToFetch(now, dir, 3), { year: 2026, quarter: 2 },
    "last unbackfilled quarter is the latest-closed one itself",
  );
  archiveQuarterRows(rows, 2026, 2, dir);
  assert.deepEqual(
    pickQuarterToFetch(now, dir, 3), { year: 2026, quarter: 2 },
    "fully backfilled but latest quarter still within its 45d correction window -> recheck",
  );
  const wellPastSeal = Date.parse("2026-06-30T00:00:00Z") + 46 * 86_400_000;
  assert.equal(
    pickQuarterToFetch(wellPastSeal, dir, 3), null,
    "latest quarter sealed and everything in range archived -> steady state, nothing to fetch",
  );
  fs.rmSync(dir, { recursive: true, force: true });
});

test("routes.ts boots the EPA CAMD poll and registers /api/data/plant-operations; manifest present", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes("bootEpaCamdPoll()"), "poll must boot eagerly");
  assert.ok(routes.includes('"/api/data/plant-operations"'), "route registered");
  const manifest = JSON.parse(fs.readFileSync(path.join(here, "..", "datacore", "manifests", "epacamd.json"), "utf8"));
  assert.equal(manifest.stream, "epacamd");
  assert.deepEqual(manifest.geo_fields, ["lat", "lon"]);
  assert.ok(String(manifest.confidence_model).includes("gate-2"), "downstream trading signal stays gated");
  assert.ok(String(manifest.license).toLowerCase().includes("public domain"));
});
