import { test } from "node:test";
import assert from "node:assert/strict";
import {
  shapeSeries, sumOrNull, goodExcellentSeries, rankedShareSeries, ratingCountSeries,
  githubSeries, weeksWithDifferentCoverage,
} from "./historySeries";

const pts = (vals: Array<number | null>) => vals.map((v, i) => ({ label: `d${i}`, value: v }));

test("a missing reading breaks the line; it is never joined across or drawn as zero", () => {
  const s = shapeSeries(pts([10, 20, null, 30, 40]), 100, 40)!;
  assert.equal(s.segments.length, 2);
  assert.equal(s.dots.length, 0);
  assert.equal(s.present, 4);
  assert.equal(s.missing, 1);
  assert.equal(s.lo, 10); // not 0: the gap contributed no value
  assert.equal(s.hi, 40);
  for (const seg of s.segments) assert.equal(seg.split("L").length, 2); // two points each
});

test("a single reading between two gaps is kept as a dot", () => {
  const s = shapeSeries(pts([1, 2, null, 5, null, 3, 4]), 100, 40)!;
  assert.equal(s.segments.length, 2);
  assert.equal(s.dots.length, 1);
});

test("fewer than two readings is no chart, not a flat line", () => {
  assert.equal(shapeSeries(pts([]), 100, 40), null);
  assert.equal(shapeSeries(pts([null, 7, null]), 100, 40), null);
  assert.equal(shapeSeries(pts([null, null]), 100, 40), null);
});

test("first and last are the first and last readings PRESENT, with their own labels", () => {
  const s = shapeSeries(pts([null, 5, 6, 9, null]), 100, 40)!;
  assert.deepEqual(s.first, { label: "d1", value: 5 });
  assert.deepEqual(s.last, { label: "d3", value: 9 });
});

test("geometry: x spans the whole window (gaps keep their place), high is at the top", () => {
  const s = shapeSeries(pts([0, 10]), 100, 40, 0)!;
  assert.equal(s.segments[0], "M0.0,40.0 L100.0,0.0");
  // A leading gap still occupies its slot: the first reading is not pulled to x=0.
  const g = shapeSeries(pts([null, 0, 10]), 100, 40, 0)!;
  assert.equal(g.segments[0], "M50.0,40.0 L100.0,0.0");
});

test("a flat series is drawn on the middle line and yields finite numbers", () => {
  const s = shapeSeries(pts([7, 7, 7]), 100, 40)!;
  assert.match(s.segments[0], /^M3\.0,20\.0 /);
  assert.ok(!s.segments[0].includes("NaN"));
});

test("sumOrNull: any missing part makes the sum missing", () => {
  assert.equal(sumOrNull([57, 10]), 67);
  assert.equal(sumOrNull([57, null]), null);
  assert.equal(sumOrNull([57, undefined]), null);
  assert.equal(sumOrNull([Number.NaN, 1]), null);
  assert.equal(sumOrNull([]), null);
});

test("crop conditions: an out-of-season week (no ratings) is a gap, not 0% good/excellent", () => {
  // Live shape, 2026-11-20: soybeans is {} from mid-October on.
  const trend = [
    { week_ending: "2026-10-11", corn: { EXCELLENT: 13, GOOD: 44, FAIR: 26, POOR: 12, "VERY POOR": 5 }, soybeans: { EXCELLENT: 11, GOOD: 47 } },
    { week_ending: "2026-10-18", corn: { EXCELLENT: 13, GOOD: 43, FAIR: 27, POOR: 12, "VERY POOR": 5 }, soybeans: {} },
    { week_ending: "2026-10-25", corn: { GOOD: 43 } },
  ];
  assert.deepEqual(goodExcellentSeries(trend, "corn").map((p) => p.value), [57, 56, null]);
  assert.deepEqual(goodExcellentSeries(trend, "soybeans").map((p) => p.value), [58, null, null]);
  assert.equal(goodExcellentSeries(trend, "corn")[0].label, "2026-10-11");
});

test("App Store: the ranked share uses the day's own denominator; no fetched slots is a gap", () => {
  const trend = [
    { date: "2026-11-18", ranked_slots: 18, total_slots: 48, total_rating_count: 161000000 },
    { date: "2026-11-19", ranked_slots: 0, total_slots: 0, total_rating_count: null },
    { date: "2026-11-20", ranked_slots: 12, total_slots: 24, total_rating_count: 161042389 },
  ];
  assert.deepEqual(rankedShareSeries(trend).map((p) => p.value), [37.5, null, 50]);
  assert.deepEqual(ratingCountSeries(trend).map((p) => p.value), [161000000, null, 161042389]);
});

test("GitHub: a null weekly total stays null; weeks summed over a different org count are counted", () => {
  const trend = [
    { weekStart: "2026-10-26", orgs_reporting: 14, total_merged_prs: 6000, total_commits: null },
    { weekStart: "2026-11-02", orgs_reporting: 15, total_merged_prs: 7046, total_commits: 9843 },
    { weekStart: "2026-11-09", orgs_reporting: 15, total_merged_prs: 7036, total_commits: 9806 },
  ];
  assert.deepEqual(githubSeries(trend, "total_commits").map((p) => p.value), [null, 9843, 9806]);
  assert.deepEqual(githubSeries(trend, "total_merged_prs").map((p) => p.value), [6000, 7046, 7036]);
  assert.equal(weeksWithDifferentCoverage(trend), 1);
  assert.equal(weeksWithDifferentCoverage([]), 0);
});
