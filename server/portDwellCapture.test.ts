import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  earliestAttemptableWeekIndex, nextCaptureTarget, captureIfDue, loadCapturedSnapshots,
} from "./portDwellCapture";
import { ARCHIVE_START_MS, weekBounds, mergeWeeklySnapshot, extractWeeklySnapshot, type WeeklySnapshot } from "./portDwellWeekly";
import type { PortDwellStats, PortDef } from "./portDwell";

const WEEK_MS = 7 * 24 * 3600_000;

test("earliestAttemptableWeekIndex: null floor (no known raw retention boundary) falls back to week 0", () => {
  assert.equal(earliestAttemptableWeekIndex(null), 0);
});

test("earliestAttemptableWeekIndex: derives the first week whose full 7 days sit at or after the floor", () => {
  const floorMs = ARCHIVE_START_MS + 10 * WEEK_MS + 3 * 24 * 3600_000; // mid-way through week 10
  assert.equal(earliestAttemptableWeekIndex(floorMs), 11); // week 10 itself is not fully covered
});

test("earliestAttemptableWeekIndex: never returns negative even if the floor predates the archive start", () => {
  assert.equal(earliestAttemptableWeekIndex(ARCHIVE_START_MS - WEEK_MS), 0);
});

function fakeStats(overrides: Partial<PortDwellStats> = {}): PortDwellStats {
  return {
    window_hours: 168, vessels_seen: 400, visits_completed: 500, in_port_now: 90, anomaly_count: 12,
    caveat: "test fixture",
    ports: [{
      id: "port_la", name: "Port of Los Angeles", lat: 33.74, lon: -118.272,
      visits_completed: 280, unique_vessels: 120, in_port_now: 48,
      dwell_median_h: 8.1, dwell_p90_h: 45.4, dwell_max_h: 122.5,
      anomaly_count: 55, anomaly_examples: [],
    }],
    ...overrides,
  };
}

const PORTS: PortDef[] = [{ id: "port_la", name: "Port of Los Angeles", lat: 33.74, lon: -118.272, radius_km: 5 }];

test("nextCaptureTarget: picks the oldest missing week not already skipped", () => {
  const nowMs = weekBounds(8).startMs; // weeks 0-7 completed
  const have: WeeklySnapshot[] = mergeWeeklySnapshot([], extractWeeklySnapshot(fakeStats(), weekBounds(4), weekBounds(4).endMs));
  const idx = nextCaptureTarget(have, new Set([0, 1, 2, 3]), null, nowMs);
  assert.equal(idx, 5); // 0-3 skipped, 4 captured, 5 is next
});

test("nextCaptureTarget: null when every week in range is captured or skipped", () => {
  const nowMs = weekBounds(3).startMs; // weeks 0-2 completed
  assert.equal(nextCaptureTarget([], new Set([0, 1, 2]), null, nowMs), null);
});

test("nextCaptureTarget: null when there is nothing to capture yet (no completed week)", () => {
  assert.equal(nextCaptureTarget([], new Set(), null, ARCHIVE_START_MS), null);
});

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "portdwellcapture-test-"));
}

test("captureIfDue: no_missing_week when nothing is due, no compute call made", async () => {
  const dir = tmpDir();
  let called = false;
  const result = await captureIfDue(PORTS, null, ARCHIVE_START_MS, dir, async () => { called = true; return { ...fakeStats(), pointsScanned: 0, elapsedMs: 0 }; });
  assert.equal(result.action, "no_missing_week");
  assert.equal(called, false);
});

test("captureIfDue: captures the oldest due week and persists it", async () => {
  const dir = tmpDir();
  const nowMs = weekBounds(1).startMs; // week 0 just completed
  const result = await captureIfDue(PORTS, null, nowMs, dir, async () => ({ ...fakeStats(), pointsScanned: 12345, elapsedMs: 678 }));
  assert.equal(result.action, "captured");
  assert.equal(result.week_index, 0);
  const saved = loadCapturedSnapshots(dir);
  assert.equal(saved.length, 1);
  assert.equal(saved[0].week_index, 0);
  assert.equal(saved[0].ports[0].dwell_median_h, 8.1);
});

test("captureIfDue: a second call after capture moves on to the next week, never re-captures the first", async () => {
  const dir = tmpDir();
  let calls = 0;
  const compute = async () => { calls++; return { ...fakeStats(), pointsScanned: 1, elapsedMs: 1 }; };
  const r1 = await captureIfDue(PORTS, null, weekBounds(1).startMs, dir, compute);
  const r2 = await captureIfDue(PORTS, null, weekBounds(2).startMs, dir, compute);
  assert.equal(r1.week_index, 0);
  assert.equal(r2.week_index, 1);
  assert.equal(calls, 2);
  assert.deepEqual(loadCapturedSnapshots(dir).map((s) => s.week_index), [0, 1]);
});

test("captureIfDue: a week whose start predates the raw-retention floor is never attempted at all — jumps straight to the first reachable week", async () => {
  const dir = tmpDir();
  const floorMs = weekBounds(3).startMs; // weeks 0-2 are all before the floor, permanently unreachable
  let calls = 0;
  const compute = async () => { calls++; return { ...fakeStats(), pointsScanned: 1, elapsedMs: 1 }; };
  const r1 = await captureIfDue(PORTS, floorMs, weekBounds(4).startMs, dir, compute);
  assert.equal(r1.action, "captured");
  assert.equal(r1.week_index, 3); // not 0 -- earliestAttemptableWeekIndex excluded 0-2 up front
  assert.equal(calls, 1); // exactly one fold, for the one reachable week

  // A later tick moves on to week... nothing further completed yet.
  const r2 = await captureIfDue(PORTS, floorMs, weekBounds(4).startMs, dir, compute);
  assert.equal(r2.action, "no_missing_week");
  assert.equal(calls, 1); // week 3 already captured, weeks 0-2 never reconsidered
});

test("captureIfDue: skips and remembers a degenerate all-zero read, does not retry it", async () => {
  const dir = tmpDir();
  const degenerateStats = { ...fakeStats(), vessels_seen: 900, ports: fakeStats().ports.map((p) => ({ ...p, visits_completed: 0, in_port_now: 0 })) };
  let calls = 0;
  const compute = async () => { calls++; return { ...degenerateStats, pointsScanned: 1, elapsedMs: 1 }; };
  const r1 = await captureIfDue(PORTS, null, weekBounds(1).startMs, dir, compute);
  assert.equal(r1.action, "skipped_degenerate");
  assert.equal(r1.week_index, 0);
  assert.equal(calls, 1);

  const r2 = await captureIfDue(PORTS, null, weekBounds(1).startMs, dir, compute);
  assert.equal(r2.action, "no_missing_week"); // week 0 skipped-remembered, no other week completed yet
  assert.equal(calls, 1); // not called again
  assert.equal(loadCapturedSnapshots(dir).length, 0);
});

test("captureIfDue: bounded to one fold per call even with many weeks outstanding", async () => {
  const dir = tmpDir();
  let calls = 0;
  const compute = async () => { calls++; return { ...fakeStats(), pointsScanned: 1, elapsedMs: 1 }; };
  await captureIfDue(PORTS, null, weekBounds(20).startMs, dir, compute); // weeks 0-19 all outstanding
  assert.equal(calls, 1);
});
