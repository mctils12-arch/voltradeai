import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ARCHIVE_START_MS, weekBounds, lastCompletedWeekIndex,
  extractWeeklySnapshot, mergeWeeklySnapshot, missingWeekIndices, isDegenerateAllZeroRead,
  WeeklySnapshot,
} from "./portDwellWeekly";
import type { PortDwellStats } from "./portDwell";

test("weekBounds: week 0 starts exactly at the archive start", () => {
  const w = weekBounds(0);
  assert.equal(w.startMs, ARCHIVE_START_MS);
  assert.equal(w.endMs - w.startMs, 7 * 24 * 3600_000);
});

test("weekBounds: consecutive weeks are contiguous, no gap or overlap", () => {
  const w3 = weekBounds(3);
  const w4 = weekBounds(4);
  assert.equal(w3.endMs, w4.startMs);
});

test("lastCompletedWeekIndex: exactly at a week boundary counts that week as NOT yet completed (still in progress)", () => {
  const w5 = weekBounds(5);
  // "now" == the instant week 5 starts: weeks 0-4 are fully elapsed, week 5 just began.
  assert.equal(lastCompletedWeekIndex(w5.startMs), 4);
});

test("lastCompletedWeekIndex: one ms before a week's end, that week is not yet completed", () => {
  const w5 = weekBounds(5);
  assert.equal(lastCompletedWeekIndex(w5.endMs - 1), 4);
});

test("lastCompletedWeekIndex: at a week's end instant, that week counts as completed", () => {
  const w5 = weekBounds(5);
  assert.equal(lastCompletedWeekIndex(w5.endMs), 5);
});

function fakeStats(): PortDwellStats {
  return {
    window_hours: 168,
    vessels_seen: 400,
    visits_completed: 500,
    in_port_now: 90,
    anomaly_count: 12,
    caveat: "test fixture",
    ports: [
      {
        id: "port_la", name: "Port of Los Angeles", lat: 33.74, lon: -118.272,
        visits_completed: 280, unique_vessels: 120, in_port_now: 48,
        dwell_median_h: 8.1, dwell_p90_h: 45.4, dwell_max_h: 122.5,
        anomaly_count: 55,
        anomaly_examples: [{ mmsi: "215558000", name: "APL QINGDAO", dwell_h: 122.5, median_h: 8.1 }],
      },
    ],
  };
}

test("extractWeeklySnapshot: strips per-vessel identity, keeps only aggregate per-port fields", () => {
  const week = weekBounds(6);
  const snap = extractWeeklySnapshot(fakeStats(), week, week.endMs + 3600_000);
  assert.equal(snap.week_index, 6);
  assert.equal(snap.week_start, new Date(week.startMs).toISOString());
  assert.equal(snap.week_end, new Date(week.endMs).toISOString());
  assert.equal(snap.ports.length, 1);
  const p = snap.ports[0];
  assert.equal(p.id, "port_la");
  assert.equal(p.dwell_median_h, 8.1);
  assert.equal(p.visits_completed, 280);
  // no per-vessel identity or geo field leaked through -- exactly the
  // aggregate fields WeeklyPortSnapshot declares, nothing else
  assert.deepEqual(Object.keys(p).sort(), [
    "dwell_max_h", "dwell_median_h", "dwell_p90_h", "id", "in_port_now", "name", "unique_vessels", "visits_completed",
  ]);
});

test("mergeWeeklySnapshot: appends a new week and keeps the array sorted by week_index", () => {
  const s6 = extractWeeklySnapshot(fakeStats(), weekBounds(6), weekBounds(6).endMs);
  const s4 = extractWeeklySnapshot(fakeStats(), weekBounds(4), weekBounds(4).endMs);
  let arr: WeeklySnapshot[] = [];
  arr = mergeWeeklySnapshot(arr, s6);
  arr = mergeWeeklySnapshot(arr, s4);
  assert.deepEqual(arr.map((s) => s.week_index), [4, 6]);
});

test("mergeWeeklySnapshot: never overwrites an already-captured week, even with different data", () => {
  const original = extractWeeklySnapshot(fakeStats(), weekBounds(5), weekBounds(5).endMs);
  const laterRead = extractWeeklySnapshot(
    { ...fakeStats(), ports: [{ ...fakeStats().ports[0], dwell_median_h: 999 }] },
    weekBounds(5), weekBounds(5).endMs + 30 * 24 * 3600_000,
  );
  let arr: WeeklySnapshot[] = mergeWeeklySnapshot([], original);
  arr = mergeWeeklySnapshot(arr, laterRead);
  assert.equal(arr.length, 1);
  assert.equal(arr[0].ports[0].dwell_median_h, 8.1); // original value, not 999
});

test("mergeWeeklySnapshot: does not mutate the input array", () => {
  const s4 = extractWeeklySnapshot(fakeStats(), weekBounds(4), weekBounds(4).endMs);
  const original: WeeklySnapshot[] = [];
  const result = mergeWeeklySnapshot(original, s4);
  assert.equal(original.length, 0);
  assert.equal(result.length, 1);
});

test("missingWeekIndices: returns every uncaptured week in [earliest, lastCompleted]", () => {
  const nowMs = weekBounds(8).startMs; // weeks 0-7 completed
  const have = mergeWeeklySnapshot([], extractWeeklySnapshot(fakeStats(), weekBounds(5), weekBounds(5).endMs));
  const missing = missingWeekIndices(have, 4, nowMs);
  assert.deepEqual(missing, [4, 6, 7]);
});

test("isDegenerateAllZeroRead: true when every port reads zero completed and zero ongoing despite vessels seen", () => {
  const s = fakeStats();
  s.vessels_seen = 1349;
  s.ports = s.ports.map((p) => ({ ...p, visits_completed: 0, in_port_now: 0 }));
  assert.equal(isDegenerateAllZeroRead(s), true);
});

test("isDegenerateAllZeroRead: false when at least one port has real activity", () => {
  assert.equal(isDegenerateAllZeroRead(fakeStats()), false); // fakeStats' one port has visits_completed 280
});

test("isDegenerateAllZeroRead: false when vessels_seen itself is zero (a coverage question, not a degenerate-read one)", () => {
  const s = fakeStats();
  s.vessels_seen = 0;
  s.ports = s.ports.map((p) => ({ ...p, visits_completed: 0, in_port_now: 0 }));
  assert.equal(isDegenerateAllZeroRead(s), false);
});

test("isDegenerateAllZeroRead: false when every port is zero-completed but one has an ongoing (in-port-now) visit", () => {
  const s = fakeStats();
  s.vessels_seen = 500;
  s.ports = s.ports.map((p, i) => ({ ...p, visits_completed: 0, in_port_now: i === 0 ? 1 : 0 }));
  assert.equal(isDegenerateAllZeroRead(s), false);
});

test("missingWeekIndices: empty when every week in range is already captured", () => {
  const nowMs = weekBounds(6).startMs; // weeks 0-5 completed
  let have: WeeklySnapshot[] = [];
  have = mergeWeeklySnapshot(have, extractWeeklySnapshot(fakeStats(), weekBounds(4), weekBounds(4).endMs));
  have = mergeWeeklySnapshot(have, extractWeeklySnapshot(fakeStats(), weekBounds(5), weekBounds(5).endMs));
  assert.deepEqual(missingWeekIndices(have, 4, nowMs), []);
});
