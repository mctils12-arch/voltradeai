import { test } from "node:test";
import assert from "node:assert/strict";
import {
  toWeeklyTotal, computeWeeklyDeltas, bucketWeek, nearestTradingDayOnOrBefore,
  meanOf, MIN_PANEL_FOR_WEEK, type OrgWeek,
} from "./github_activity_gate2";
import { toSeries } from "./occ_volume_gate2";

test("toWeeklyTotal sums mergedPRs + commits", () => {
  const w = toWeeklyTotal({ weekStart: "2026-07-20", weekEnd: "2026-07-26", mergedPRs: 137, commits: 573 });
  assert.equal(w.total, 710);
});

test("toWeeklyTotal returns null total when either field is missing", () => {
  assert.equal(toWeeklyTotal({ weekStart: "w", weekEnd: "w", mergedPRs: null, commits: 5 }).total, null);
  assert.equal(toWeeklyTotal({ weekStart: "w", weekEnd: "w", mergedPRs: 5, commits: null }).total, null);
});

function week(ticker: string, weekStart: string, weekEnd: string, total: number | null): OrgWeek {
  return { ticker, weekStart, weekEnd, total };
}

test("computeWeeklyDeltas computes each ticker's pct change against its OWN prior week only", () => {
  const map = new Map<string, OrgWeek[]>([
    ["AAA", [week("AAA", "2026-07-20", "2026-07-26", 100), week("AAA", "2026-07-27", "2026-08-02", 150)]],
    ["BBB", [week("BBB", "2026-07-20", "2026-07-26", 1000), week("BBB", "2026-07-27", "2026-08-02", 900)]],
  ]);
  const deltas = computeWeeklyDeltas(map);
  assert.equal(deltas.length, 2);
  const aaa = deltas.find((d) => d.ticker === "AAA")!;
  const bbb = deltas.find((d) => d.ticker === "BBB")!;
  assert.ok(Math.abs(aaa.pctChange - 0.5) < 1e-9);
  assert.ok(Math.abs(bbb.pctChange - -0.1) < 1e-9);
});

test("computeWeeklyDeltas skips a week whose prior total is null or zero", () => {
  const map = new Map<string, OrgWeek[]>([
    ["AAA", [week("AAA", "w0", "w0", null), week("AAA", "w1", "w1", 100)]],
    ["ZERO", [week("ZERO", "w0", "w0", 0), week("ZERO", "w1", "w1", 50)]],
  ]);
  assert.equal(computeWeeklyDeltas(map).length, 0);
});

test("computeWeeklyDeltas never compares across tickers", () => {
  const map = new Map<string, OrgWeek[]>([
    ["AAA", [week("AAA", "2026-07-20", "2026-07-26", 100)]],
    ["BBB", [week("BBB", "2026-07-27", "2026-08-02", 900)]],
  ]);
  // Each series has only ONE week — no prior week exists for either ticker,
  // so no delta can be computed even though BBB's single week postdates AAA's.
  assert.equal(computeWeeklyDeltas(map).length, 0);
});

function delta(ticker: string, weekEnd: string, pctChange: number) {
  return { ticker, weekEnd, pctChange };
}

test("bucketWeek splits a full 15-org panel into clean 5/5/5 terciles, ranked descending", () => {
  const rows = Array.from({ length: 15 }, (_, i) => delta(`T${i}`, "2026-08-02", i - 7)); // -7..7
  const b = bucketWeek("2026-08-02", rows)!;
  assert.equal(b.top.length, 5);
  assert.equal(b.mid.length, 5);
  assert.equal(b.bottom.length, 5);
  assert.deepEqual(b.top, ["T14", "T13", "T12", "T11", "T10"]); // largest pctChange first
  assert.deepEqual(b.bottom, ["T4", "T3", "T2", "T1", "T0"]);
});

test("bucketWeek drops a week with fewer than MIN_PANEL_FOR_WEEK valid deltas", () => {
  const rows = Array.from({ length: MIN_PANEL_FOR_WEEK - 1 }, (_, i) => delta(`T${i}`, "2026-08-02", i));
  assert.equal(bucketWeek("2026-08-02", rows), null);
});

test("bucketWeek ignores rows from other weeks", () => {
  const rows = [
    ...Array.from({ length: MIN_PANEL_FOR_WEEK }, (_, i) => delta(`T${i}`, "2026-08-02", i)),
    delta("OTHERWEEK", "2026-08-09", 999),
  ];
  const b = bucketWeek("2026-08-02", rows)!;
  assert.ok(![...b.top, ...b.mid, ...b.bottom].includes("OTHERWEEK"));
});

test("nearestTradingDayOnOrBefore finds the latest series date <= the target", () => {
  const s = toSeries(new Map([["2026-07-24", 100], ["2026-07-27", 101], ["2026-07-28", 102]]));
  assert.equal(nearestTradingDayOnOrBefore(s, "2026-07-26"), "2026-07-24"); // weekend, falls back to Friday
  assert.equal(nearestTradingDayOnOrBefore(s, "2026-07-27"), "2026-07-27"); // exact match
  assert.equal(nearestTradingDayOnOrBefore(s, "2026-07-28"), "2026-07-28");
});

test("nearestTradingDayOnOrBefore returns null when the series starts after the target", () => {
  const s = toSeries(new Map([["2026-08-01", 100]]));
  assert.equal(nearestTradingDayOnOrBefore(s, "2026-07-01"), null);
});

test("meanOf ignores nulls and returns null for an all-null input", () => {
  assert.equal(meanOf([1, null, 3]), 2);
  assert.equal(meanOf([null, null]), null);
  assert.equal(meanOf([]), null);
});
