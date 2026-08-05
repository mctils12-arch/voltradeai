import { test } from "node:test";
import assert from "node:assert/strict";
import { weeklySampleDays, SAMPLE_START, SAMPLE_WEEKS } from "./occ_volume_gate2";
import { DISJOINT_SAMPLE_START, DISJOINT_SAMPLE_WEEKS } from "./occ_volume_gate2_clustered_disjoint";

// Ratchet for LADDER PATH STEP (2)'s own stated requirement: "the
// 2026-01-07..04-22 window used [in step (1)] must not be reused as its
// own confirmation." This is a plain assertion the disjoint window really
// is disjoint from — and chronologically after — the original one, so a
// future edit to either window's constants can't silently reintroduce
// overlap without a failing test.

test("disjoint window shares zero sample days with the original step (1) window", () => {
  const original = weeklySampleDays(SAMPLE_START, SAMPLE_WEEKS);
  const disjoint = weeklySampleDays(DISJOINT_SAMPLE_START, DISJOINT_SAMPLE_WEEKS);
  const overlap = disjoint.filter((d) => original.includes(d));
  assert.deepEqual(overlap, []);
});

test("disjoint window starts strictly after the original window's last sample day", () => {
  const original = weeklySampleDays(SAMPLE_START, SAMPLE_WEEKS);
  const disjoint = weeklySampleDays(DISJOINT_SAMPLE_START, DISJOINT_SAMPLE_WEEKS);
  const originalLast = original[original.length - 1];
  assert.ok(disjoint[0] > originalLast, `expected disjoint window to start after ${originalLast}, got ${disjoint[0]}`);
});

test("disjoint window is 9 weekly Wednesdays, 2026-05-06 through 2026-07-01", () => {
  const disjoint = weeklySampleDays(DISJOINT_SAMPLE_START, DISJOINT_SAMPLE_WEEKS);
  assert.equal(disjoint.length, 9);
  assert.equal(disjoint[0], "2026-05-06");
  assert.equal(disjoint[disjoint.length - 1], "2026-07-01");
  for (const d of disjoint) {
    const weekday = new Date(d + "T00:00:00Z").getUTCDay();
    assert.equal(weekday, 3, `expected ${d} to be a Wednesday (UTC day 3), got day ${weekday}`); // matches step (1)'s own weekly-Wednesday cadence
  }
});

test("disjoint window's last sample day leaves a positive settle buffer before the run date (2026-08-05)", () => {
  const disjoint = weeklySampleDays(DISJOINT_SAMPLE_START, DISJOINT_SAMPLE_WEEKS);
  const lastSampleDay = new Date(disjoint[disjoint.length - 1] + "T00:00:00Z").getTime();
  const runDate = new Date("2026-08-05T00:00:00Z").getTime();
  const calendarDaysBuffer = (runDate - lastSampleDay) / 86_400_000;
  // +20 trading days is ~28 calendar days; require a positive buffer beyond
  // that so the +20d forward return is fully realized, not still pending.
  assert.ok(calendarDaysBuffer > 28, `expected >28 calendar days of buffer, got ${calendarDaysBuffer}`);
});
