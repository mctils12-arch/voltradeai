// Drought Monitor battery (BUILD ORDER 2 #5, 2026-07-05). Fixture values
// copied from the live probes of 2026-07-05 (CONUS + Iowa by FIPS).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  parseDrought, archiveDrought, gzipOldDroughtDays, DROUGHT_AOIS,
  refreshDroughtCache, latestDrought, _resetDroughtForTests,
} from "./droughtMonitor";

const CONUS_ROW = {
  mapDate: "2026-06-30T00:00:00", areaOfInterest: "CONUS",
  none: 32.98, d0: 67.02, d1: 47.84, d2: 30.70, d3: 10.89, d4: 1.00,
  validStart: "2026-06-30T00:00:00", validEnd: "2026-07-06T23:59:59", statisticFormatID: 1,
};
const IA_ROW = {
  mapDate: "2026-06-30T00:00:00", stateAbbreviation: "IA",
  none: 66.66, d0: 33.34, d1: 12.15, d2: 0.0, d3: 0.0, d4: 0.0,
  validStart: "2026-06-30T00:00:00", validEnd: "2026-07-06T23:59:59", statisticFormatID: 1,
};

test("parseDrought: real rows normalize; DSCI is the labeled derived sum", () => {
  const recs = parseDrought([CONUS_ROW], "CONUS", "2026-07-05");
  assert.equal(recs.length, 1);
  const c = recs[0];
  assert.equal(c.map_date, "2026-06-30");
  assert.equal(c.d0, 67.02);
  assert.equal(c.d4, 1.0);
  assert.equal(c.dsci, +(67.02 + 47.84 + 30.7 + 10.89 + 1.0).toFixed(2));
  assert.ok(c.dsci > 0 && c.dsci <= 500);
  const ia = parseDrought([IA_ROW], "IA", "2026-07-05")[0];
  assert.equal(ia.aoi, "IA");
  assert.equal(ia.dsci, +(33.34 + 12.15).toFixed(2));
});

test("parseDrought: label honesty — the us endpoint's 'Total' rows drop, never relabel", () => {
  // live-caught 2026-07-05: aoi=us returns BOTH CONUS and Total (incl.
  // AK/HI/PR) per week; mislabeling Total as CONUS silently mixed two series
  const TOTAL_ROW = { ...CONUS_ROW, areaOfInterest: "Total", d2: 25.66 };
  const recs = parseDrought([CONUS_ROW, TOTAL_ROW], "CONUS", "2026-07-05");
  assert.equal(recs.length, 1);
  assert.equal(recs[0].d2, 30.7, "only the row whose own label matches survives");
  // state rows verify stateAbbreviation the same way
  assert.equal(parseDrought([IA_ROW], "NE", "x").length, 0);
});

test("parseDrought: malformed percentage drops the whole row (no partial DSCI)", () => {
  assert.equal(parseDrought([{ ...CONUS_ROW, d2: "not-a-number" }], "CONUS", "x").length, 0);
  assert.equal(parseDrought([{ ...CONUS_ROW, d2: 130 }], "CONUS", "x").length, 0, ">100% is malformed");
  assert.equal(parseDrought([{ ...CONUS_ROW, mapDate: null }], "CONUS", "x").length, 0);
  assert.deepEqual(parseDrought(null, "CONUS", "x"), []);
});

test("aoi table: CONUS + 8 ag/water states, states by FIPS (probed: abbreviations return empty)", () => {
  assert.equal(DROUGHT_AOIS.length, 9);
  assert.equal(DROUGHT_AOIS[0].aoi, "us");
  for (const a of DROUGHT_AOIS.slice(1)) {
    assert.match(a.aoi, /^\d{2}$/, `state ${a.key} must use a 2-digit FIPS code`);
    assert.equal(a.api, "StateStatistics");
  }
});

test("archive: dedup aoi|map_date across polls; gz lifecycle", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "usdm-"));
  const now = Date.parse("2026-07-05T12:00:00Z");
  const recs = [
    ...parseDrought([CONUS_ROW], "CONUS", "2026-07-05"),
    ...parseDrought([IA_ROW], "IA", "2026-07-05"),
  ];
  assert.equal(archiveDrought(recs, base, now), 2);
  assert.equal(archiveDrought(recs, base, now), 0, "same aoi+map_date never re-archives");
  const nextWeek = parseDrought([{ ...IA_ROW, mapDate: "2026-07-07T00:00:00" }], "IA", "2026-07-12");
  assert.equal(archiveDrought(nextWeek, base, now), 1);
  assert.equal(gzipOldDroughtDays(base, now + 3 * 86400_000), 1);
  const day = path.join(base, "drought", "2026-07-05.jsonl");
  assert.ok(!fs.existsSync(day) && fs.existsSync(`${day}.gz`));
});

test("COLD-CACHE-NO-DISK-BACKFILL FIX: a live fetch that returns zero rows on a cold boot must restore from the on-disk archive instead of caching an empty, non-warming_up result forever", async () => {
  _resetDroughtForTests();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "usdm-"));
  // Simulate a PRIOR process run that already archived a week, then this
  // process restarts with a cold in-memory cache (real container-restart
  // shape: the disk archive survives, the in-memory cache does not).
  const priorWeek = [
    ...parseDrought([CONUS_ROW], "CONUS", "2026-08-01"),
    ...parseDrought([IA_ROW], "IA", "2026-08-01"),
  ];
  archiveDrought(priorWeek, dir, Date.parse("2026-08-01T12:00:00Z"));
  _resetDroughtForTests();
  assert.equal(latestDrought(), null, "cache starts cold after the simulated restart");

  // Every AOI request fails at boot (network blip) -> fetchDrought's own
  // per-AOI try/catch swallows every call; all-failed throws, the exact
  // "drought.length === 0 on the first-ever cycle" case the pre-fix
  // `if (drought.length || !cache) cache = {...}` mishandled.
  const allDown = async () => { throw new Error("network down"); };
  await refreshDroughtCache(allDown as any, Date.parse("2026-08-02T12:00:00Z"), dir);

  const hit = latestDrought();
  assert.ok(hit, "must restore from the on-disk archive instead of staying (or worse, caching empty) forever");
  assert.equal(hit!.drought.length, priorWeek.length);
  assert.equal(hit!.drought[0].aoi, "CONUS");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("refreshDroughtCache: cold boot + empty live fetch + no on-disk archive at all stays honestly null, never a fabricated empty cache", async () => {
  _resetDroughtForTests();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "usdm-"));
  const allDown = async () => { throw new Error("network down"); };
  await refreshDroughtCache(allDown as any, Date.parse("2026-08-03T12:00:00Z"), dir);
  assert.equal(latestDrought(), null, "nothing to restore -> stays null, so /api/data/drought still honestly reports warming_up rather than a silent empty count:0");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("refreshDroughtCache: a transient empty fetch during steady state (cache already populated) never clobbers the existing cache", async () => {
  _resetDroughtForTests();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "usdm-"));
  const ok = async () => ({ ok: true, status: 200, text: async () => JSON.stringify([CONUS_ROW]) });
  await refreshDroughtCache(ok as any, Date.parse("2026-08-04T12:00:00Z"), dir);
  const populated = latestDrought();
  assert.ok(populated && populated.drought.length, "cache populated by a successful fetch");

  const allDown = async () => { throw new Error("network down"); };
  await refreshDroughtCache(allDown as any, Date.parse("2026-08-04T13:00:00Z"), dir);
  assert.strictEqual(latestDrought(), populated, "a later transient failure must not overwrite good cache with anything, restored or empty");
  fs.rmSync(dir, { recursive: true, force: true });
});
