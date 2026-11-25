// Run: npx tsx --test client/src/lib/dtccSourceDays.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { dtccSourceDaysLine, dtccEdgeLine, type DtccSourceDaysSummary } from "./dtccSourceDays";

const base: DtccSourceDaysSummary = {
  enabled: true, floor: "2026-08-21", through: "2026-11-24", expected: 96, read: 96,
  unavailable: 0, pending: 0, refilledDays: 0, refilledRows: 0, lastRefilled: null,
};

test("no summary, or no file read yet: no line", () => {
  assert.equal(dtccSourceDaysLine(undefined), null);
  assert.equal(dtccSourceDaysLine(null), null);
  assert.equal(dtccSourceDaysLine({ ...base, through: null, expected: 0, read: 0 }), null);
});

test("every day read and none missed: says so, and only then", () => {
  assert.equal(dtccSourceDaysLine(base), "Vendor files 2026-08-21 to 2026-11-24: 96 of 96 days read · no day was missed");
});

test("days still to check: 'no day was missed' is NOT said while any day is unchecked", () => {
  const line = dtccSourceDaysLine({ ...base, read: 3, pending: 93 })!;
  assert.match(line, /3 of 96 days read/);
  assert.match(line, /93 days not yet checked \(two are read after each poll\)/);
  assert.doesNotMatch(line, /no day was missed/);
});

test("the catch-up switched off: the unchecked days are said to stay unchecked", () => {
  const line = dtccSourceDaysLine({ ...base, enabled: false, read: 1, pending: 95 })!;
  assert.match(line, /95 days not checked \(the catch-up is switched off\)/);
});

test("a missed day read late is counted with its events; singular and plural", () => {
  const one = dtccSourceDaysLine({ ...base, refilledDays: 1, refilledRows: 112100, lastRefilled: { date: "2026-09-14", at: "2026-11-26T01:00:00.000Z", newRows: 112100 } })!;
  assert.match(one, /1 day had been missed and was read late \(112,100 events added\)/);
  assert.doesNotMatch(one, /no day was missed/);
  const two = dtccSourceDaysLine({ ...base, refilledDays: 2, refilledRows: 5 })!;
  assert.match(two, /2 days had been missed and were read late \(5 events added\)/);
});

test("a day the vendor did not serve is said, and blocks 'no day was missed'", () => {
  const line = dtccSourceDaysLine({ ...base, read: 95, unavailable: 1 })!;
  assert.match(line, /1 day the vendor did not serve/);
  assert.doesNotMatch(line, /no day was missed/);
});

test("edge line: nothing until a day at the vendor's two-year limit has been read; then the days, the span and the events", () => {
  assert.equal(dtccEdgeLine(undefined), null);
  assert.equal(dtccEdgeLine(base), null, "an older server sends no edge block");
  assert.equal(dtccEdgeLine({ ...base, edge: { enabled: true, days: 0, rows: 0, oldest: null, newest: null } }), null);
  assert.equal(
    dtccEdgeLine({ ...base, edge: { enabled: true, days: 1, rows: 64749, oldest: "2024-11-25", newest: "2024-11-25" } }),
    "The vendor stops serving a file two years after it is written: 1 day at that limit read in time (2024-11-25, 64,749 events)",
  );
  const many = dtccEdgeLine({ ...base, edge: { enabled: false, days: 3, rows: 130000, oldest: "2024-11-24", newest: "2024-11-26" } })!;
  assert.match(many, /3 days at that limit read in time \(2024-11-24 to 2024-11-26, 130,000 events\)/);
  assert.match(many, /these reads are switched off now$/);
});
