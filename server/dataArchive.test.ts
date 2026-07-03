import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import {
  dayKeyUTC,
  shouldSample,
  compactAircraftRecord,
  compactVesselRecord,
  buildSampleLine,
  recordAircraftSample,
  recordVesselSample,
  getArchiveStats,
  archiveFilePath,
  SAMPLE_INTERVAL_MS,
} from "./dataArchive";

// MAP V2 ROADMAP R1 (open_questions.md): position archive must self-throttle
// writes regardless of client poll rate, store compact positional (not
// object) records, and report honest stats for the volume-watch wishlist item.

test("dayKeyUTC returns YYYY-MM-DD in UTC", () => {
  assert.equal(dayKeyUTC(Date.parse("2026-07-03T23:59:00Z")), "2026-07-03");
  assert.equal(dayKeyUTC(Date.parse("2026-07-04T00:00:01Z")), "2026-07-04");
});

test("shouldSample gates on interval, not every call", () => {
  assert.equal(shouldSample(undefined, 1000, SAMPLE_INTERVAL_MS), true, "first call always samples");
  assert.equal(shouldSample(1000, 1000 + SAMPLE_INTERVAL_MS - 1, SAMPLE_INTERVAL_MS), false);
  assert.equal(shouldSample(1000, 1000 + SAMPLE_INTERVAL_MS, SAMPLE_INTERVAL_MS), true);
});

test("compactAircraftRecord rounds to ~11m precision and preserves nulls", () => {
  const rec = compactAircraftRecord({ icao24: "abc123", lat: 40.64134567, lon: -73.77812345, altitude_m: 11000.6, heading: 269.6 });
  assert.deepEqual(rec, ["abc123", 40.6413, -73.7781, 11000.6, 270]);
  const recNulls = compactAircraftRecord({ icao24: "def456", lat: 1, lon: 2, altitude_m: null, heading: null });
  assert.deepEqual(recNulls, ["def456", 1, 2, null, null]);
});

test("compactVesselRecord shape matches aircraft convention", () => {
  const rec = compactVesselRecord({ mmsi: "366123456", lat: 29.5, lon: -90.1, sog: 12.3, cog: 180.4 });
  assert.deepEqual(rec, ["366123456", 29.5, -90.1, 12.3, 180]);
});

test("buildSampleLine is valid JSON round-tripping records", () => {
  const line = buildSampleLine(1000, [["a", 1, 2, null, null]]);
  const parsed = JSON.parse(line);
  assert.equal(parsed.t, 1000);
  assert.equal(parsed.n, 1);
  assert.deepEqual(parsed.recs, [["a", 1, 2, null, null]]);
});

test("recordAircraftSample writes a throttled, appended, per-day JSONL file and updates stats", () => {
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "voltrade-archive-test-"));
  const t0 = Date.parse("2026-07-03T12:00:00Z");
  const aircraft = [{ icao24: "aaa111", lat: 40.0, lon: -74.0, altitude_m: 1000, heading: 90 }];

  recordAircraftSample(dataDir, aircraft, t0);
  // A second call inside the same sampling window must NOT write again.
  recordAircraftSample(dataDir, aircraft, t0 + 1000);
  // A call past the interval must write a second line into the SAME day file.
  recordAircraftSample(dataDir, aircraft, t0 + SAMPLE_INTERVAL_MS);

  const file = archiveFilePath(dataDir, "aircraft", "2026-07-03");
  const lines = fs.readFileSync(file, "utf8").trim().split("\n");
  assert.equal(lines.length, 2, "throttled: 3 calls in one interval window should yield 2 samples");

  const stats = getArchiveStats(dataDir);
  assert.equal(stats.kinds.aircraft.totalSamples, 2);
  assert.equal(stats.kinds.aircraft.totalRecords, 2);
  assert.equal(stats.kinds.aircraft.firstDay, "2026-07-03");
  assert.ok(stats.kinds.aircraft.approxBytesOnDisk > 0);
  assert.equal(stats.kinds.vessels.totalSamples, 0, "vessels untouched by an aircraft-only test");

  fs.rmSync(dataDir, { recursive: true, force: true });
});

test("recordVesselSample skips writing (but still throttles) on an empty feed", () => {
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "voltrade-archive-test-"));
  const t0 = Date.parse("2026-07-03T12:00:00Z");
  recordVesselSample(dataDir, [], t0);
  const stats = getArchiveStats(dataDir);
  assert.equal(stats.kinds.vessels.totalSamples, 0);
  fs.rmSync(dataDir, { recursive: true, force: true });
});
