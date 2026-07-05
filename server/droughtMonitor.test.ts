// Drought Monitor battery (BUILD ORDER 2 #5, 2026-07-05). Fixture values
// copied from the live probes of 2026-07-05 (CONUS + Iowa by FIPS).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { parseDrought, archiveDrought, gzipOldDroughtDays, DROUGHT_AOIS } from "./droughtMonitor";

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
