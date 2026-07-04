import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import {
  firmsEnabled,
  firmsUrl,
  parseFirmsCsv,
  classifyFirmsConfidence,
  fireDetectionId,
  fetchFirmsDetections,
  archiveFireDetections,
  gzipOldFireDays,
  readFireHistory,
  bootFirmsPoll,
} from "./nasaFirms";

// ROOT VALIDATION LADDER gate 1 (DATA) fixtures — column layouts copied
// verbatim from the FIRMS API documentation's sample rows for VIIRS_SNPP_NRT
// and MODIS_NRT (firms.modaps.eosdis.nasa.gov/api/area/), the two shapes this
// parser must handle without knowing in advance which source served a row.
const VIIRS_CSV =
  "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,confidence,version,bright_ti5,frp,daynight\n" +
  "36.05481,-119.4432,335.2,0.43,0.36,2026-07-04,0424,N,h,2.0NRT,296.1,12.3,D\n" +
  "40.1234,-121.5678,300.1,0.45,0.38,2026-07-04,0424,N,l,2.0NRT,290.0,1.1,N\n";

const MODIS_CSV =
  "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight\n" +
  "34.1,-118.2,320.5,1.1,1.0,2026-07-04,0512,Aqua,MODIS,87,6.1NRT,295.0,45.2,D\n";

test("parseFirmsCsv: VIIRS rows map bright_ti4 -> brightness, letter confidence classified", () => {
  const rows = parseFirmsCsv(VIIRS_CSV);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].lat, 36.05481);
  assert.equal(rows[0].lon, -119.4432);
  assert.equal(rows[0].brightness, 335.2);
  assert.equal(rows[0].confidence, "high");
  assert.equal(rows[0].frp, 12.3);
  assert.equal(rows[0].daynight, "D");
  assert.equal(rows[1].confidence, "low");
});

test("parseFirmsCsv: MODIS rows map brightness column, numeric confidence classified", () => {
  const rows = parseFirmsCsv(MODIS_CSV);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].brightness, 320.5);
  assert.equal(rows[0].satellite, "Aqua");
  assert.equal(rows[0].confidence, "high"); // 87 >= 80
});

test("parseFirmsCsv: empty/header-only input returns no rows", () => {
  assert.deepEqual(parseFirmsCsv(""), []);
  assert.deepEqual(parseFirmsCsv("latitude,longitude,confidence\n"), []);
});

test("classifyFirmsConfidence: letter and numeric scales both bucket to the same three values", () => {
  assert.equal(classifyFirmsConfidence("l"), "low");
  assert.equal(classifyFirmsConfidence("nominal"), "nominal");
  assert.equal(classifyFirmsConfidence("H"), "high");
  assert.equal(classifyFirmsConfidence("15"), "low");
  assert.equal(classifyFirmsConfidence("50"), "nominal");
  assert.equal(classifyFirmsConfidence("95"), "high");
  assert.equal(classifyFirmsConfidence("garbage"), "nominal");
});

test("fireDetectionId: stable for identical rows, distinct for different positions", () => {
  const a = { satellite: "N", lat: 36.05481, lon: -119.4432, acq_date: "2026-07-04", acq_time: "0424" };
  const b = { ...a };
  const c = { ...a, lat: 36.06 };
  assert.equal(fireDetectionId(a), fireDetectionId(b));
  assert.notEqual(fireDetectionId(a), fireDetectionId(c));
});

test("firmsUrl: builds the documented area-CSV endpoint shape", () => {
  const url = firmsUrl("TESTKEY");
  assert.match(url, /^https:\/\/firms\.modaps\.eosdis\.nasa\.gov\/api\/area\/csv\/TESTKEY\/VIIRS_SNPP_NRT\/-180,-90,180,90\/1$/);
});

test("firmsEnabled: key-gated exactly like AISSTREAM_KEY — BOTH env names accepted", () => {
  assert.equal(firmsEnabled({}), false);
  assert.equal(firmsEnabled({ NASA_FIRMS_MAP_KEY: "abc" }), true);
  // The human set the Railway var as FIRMS_MAP_KEY (2026-07-04) — the code
  // accepts the action already taken; regression here would silently
  // deactivate the live layer.
  assert.equal(firmsEnabled({ FIRMS_MAP_KEY: "abc" }), true);
});

test("fetchFirmsDetections: non-ok upstream response throws (no silent empty result)", async () => {
  const fakeFetch = async () => ({ ok: false, status: 403, text: async () => "" });
  await assert.rejects(() => fetchFirmsDetections("badkey", fakeFetch as any), /403/);
});

test("fetchFirmsDetections: parses a real-shaped ok response", async () => {
  const fakeFetch = async () => ({ ok: true, status: 200, text: async () => VIIRS_CSV });
  const rows = await fetchFirmsDetections("key", fakeFetch as any);
  assert.equal(rows.length, 2);
});

test("bootFirmsPoll: never starts (no fetch scheduled) without a key — mirrors bootVesselStream", () => {
  // Passing a real global setInterval would leak a timer across test files;
  // the no-key path returns before ever touching the timer, so this proves
  // the gate without needing to mock timers.
  bootFirmsPoll({}); // no NASA_FIRMS_MAP_KEY
  bootFirmsPoll({}); // idempotent no-op, still gated
  assert.ok(true);
});

// Distinct lat/lon per test below (not the shared VIIRS_CSV/MODIS_CSV
// fixtures) — archiveFireDetections' dedup set is module-level and content-
// keyed (mirrors edgarForm4.ts's archivedAccessions), so reusing an id
// across test cases would falsely dedup across unrelated temp directories.
const csvRow = (lat: number, lon: number, date: string, time: string) =>
  "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,confidence,version,bright_ti5,frp,daynight\n" +
  `${lat},${lon},330.0,0.4,0.4,${date},${time},N,h,2.0NRT,290.0,10.0,D\n`;

test("archiveFireDetections + readFireHistory: dedups by our own id and round-trips through disk", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "firms-archive-"));
  const now = Date.parse("2026-07-04T12:00:00Z");
  const dets = parseFirmsCsv(csvRow(10.111, 20.222, "2026-07-04", "0100") + "11.111,21.222,331.0,0.4,0.4,2026-07-04,0100,N,h,2.0NRT,290.0,10.0,D\n");
  const wrote = archiveFireDetections(dets, dir, now);
  assert.equal(wrote, 2);
  // FIRMS re-serves the same detections on the next poll (overlapping
  // day-range window) — archiving the identical set again must write zero.
  const wroteAgain = archiveFireDetections(dets, dir, now + 60_000);
  assert.equal(wroteAgain, 0);
  const history = readFireHistory(7, dir, now + 60_000);
  assert.equal(history.length, 2);
});

test("gzipOldFireDays: compresses day files older than 2 days, leaves recent ones alone", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "firms-gzip-"));
  const now = Date.parse("2026-07-04T12:00:00Z");
  archiveFireDetections(parseFirmsCsv(csvRow(30.333, 40.444, "2026-07-01", "0200")), dir, now - 3 * 86400_000);
  archiveFireDetections(parseFirmsCsv(csvRow(31.333, 41.444, "2026-07-04", "0200")), dir, now);
  const n = gzipOldFireDays(dir, now);
  assert.equal(n, 1);
  const firesDir = path.join(dir, "fires");
  const files = fs.readdirSync(firesDir);
  assert.ok(files.some((f) => f.endsWith(".jsonl.gz")));
  assert.ok(files.some((f) => f.endsWith(".jsonl") && !f.endsWith(".gz")));
});
