import { test } from "node:test";
import assert from "node:assert/strict";
import { mergeCapturedWeeks } from "./portdwell_weekly_snapshot.ts";
import type { WeeklySnapshot } from "../server/portDwellWeekly.ts";

function fakeWeek(index: number, dwellMedianH: number): WeeklySnapshot {
  return {
    week_index: index,
    week_start: new Date(index * 1000).toISOString(),
    week_end: new Date((index + 1) * 1000).toISOString(),
    captured_at: new Date(0).toISOString(),
    ports: [
      {
        id: "port_test", name: "Test Port",
        dwell_median_h: dwellMedianH, dwell_p90_h: null, dwell_max_h: null,
        visits_completed: 1, unique_vessels: 1, in_port_now: 0,
      },
    ],
  };
}

test("mergeCapturedWeeks: an empty existing file picks up every server-captured week", () => {
  const merged = mergeCapturedWeeks([], [fakeWeek(6, 8), fakeWeek(7, 9), fakeWeek(8, 10)]);
  assert.deepEqual(merged.map((s) => s.week_index), [6, 7, 8]);
});

test("mergeCapturedWeeks: a week already present in the file is never overwritten by the server's copy", () => {
  const localWeek6 = fakeWeek(6, 8);
  const serverWeek6 = fakeWeek(6, 999); // same index, different (hypothetically fresher-looking) content
  const merged = mergeCapturedWeeks([localWeek6], [serverWeek6]);
  assert.equal(merged.length, 1);
  assert.equal(merged[0].ports[0].dwell_median_h, 8, "the pre-existing local snapshot must win, per mergeWeeklySnapshot's own never-overwrite contract");
});

test("mergeCapturedWeeks: only genuinely new weeks are added, existing untouched entries survive alongside them", () => {
  const merged = mergeCapturedWeeks([fakeWeek(5, 3)], [fakeWeek(5, 999), fakeWeek(6, 8), fakeWeek(7, 9)]);
  assert.deepEqual(merged.map((s) => s.week_index), [5, 6, 7]);
  assert.equal(merged.find((s) => s.week_index === 5)!.ports[0].dwell_median_h, 3);
});

test("mergeCapturedWeeks: order of the captured list does not matter, result is always sorted by week_index", () => {
  const merged = mergeCapturedWeeks([], [fakeWeek(8, 1), fakeWeek(6, 2), fakeWeek(7, 3)]);
  assert.deepEqual(merged.map((s) => s.week_index), [6, 7, 8]);
});

test("mergeCapturedWeeks: an empty captured list leaves existing unchanged", () => {
  const existing = [fakeWeek(6, 8)];
  const merged = mergeCapturedWeeks(existing, []);
  assert.deepEqual(merged, existing);
});
