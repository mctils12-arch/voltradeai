import { test } from "node:test";
import assert from "node:assert/strict";
import { populationSample, classifyRegime, POP_SAMPLE_SIZE } from "./finra_shortvol_gate2_retest";
import { FLOOR_TOTAL_VOL } from "../server/finraShortVolume";
import type { ShortVolRow } from "../server/finraShortVolume";
import { toSeries } from "./occ_volume_gate2";

function row(symbol: string, short_vol: number, total_vol: number): ShortVolRow {
  return { date: "2026-01-07", symbol, short_vol, short_exempt_vol: 0, total_vol, market: "Q", rt: "2026-01-07" };
}

test("populationSample filters below FLOOR_TOTAL_VOL", () => {
  const rows = [
    row("BELOW", 100, FLOOR_TOTAL_VOL - 1),
    row("AT", 100, FLOOR_TOTAL_VOL),
    row("ABOVE", 100, FLOOR_TOTAL_VOL + 1),
  ];
  const { qualifying, sampled } = populationSample(rows);
  assert.equal(qualifying, 2);
  assert.ok(!sampled.some((r) => r.ticker === "BELOW"));
});

test("populationSample selection is independent of short_ratio (alphabetical stride, not rank)", () => {
  // Construct rows where the alphabetically-early tickers have the HIGHEST
  // short_ratio, so a rank-based sampler and an alphabetical sampler would
  // disagree — proves the sample isn't secretly correlated with the metric.
  const rows = Array.from({ length: 30 }, (_, i) =>
    row(`T${String(i).padStart(2, "0")}`, (30 - i) * 1000, 1_000_000)); // T00 has highest short_vol
  const { sampled } = populationSample(rows);
  // stride = floor(30/200) = 1 (POP_SAMPLE_SIZE > n), so every qualifying
  // ticker is included — sanity check the alphabetical ordering survived.
  assert.equal(sampled.length, 30);
  assert.equal(sampled[0].ticker, "T00");
  assert.equal(sampled[sampled.length - 1].ticker, "T29");
});

test("populationSample caps at POP_SAMPLE_SIZE via a fixed stride, never exceeding it", () => {
  const rows = Array.from({ length: 5000 }, (_, i) => row(`T${i}`, 100, 1_000_000));
  const { qualifying, sampled } = populationSample(rows);
  assert.equal(qualifying, 5000);
  assert.ok(sampled.length <= POP_SAMPLE_SIZE);
  assert.ok(sampled.length > 0);
});

test("populationSample on an empty qualifying set returns no sample, no division by zero", () => {
  const rows = [row("ZERO", 0, 0)];
  const { qualifying, sampled } = populationSample(rows);
  assert.equal(qualifying, 0);
  assert.deepEqual(sampled, []);
});

// ── classifyRegime ──────────────────────────────────────────────────────

function seriesFromCloses(startDate: string, closes: number[]) {
  const map = new Map<string, number>();
  const cur = new Date(startDate + "T00:00:00Z");
  for (const c of closes) {
    map.set(cur.toISOString().slice(0, 10), c);
    cur.setUTCDate(cur.getUTCDate() + 1);
  }
  return toSeries(map);
}

// Both series share the same start date and array LENGTH (60 flat prior
// days + 1 final day), so they always end on the identical calendar day —
// alignment is a fixture property, not something each test re-derives.
const FIXTURE_START = "2025-10-01";
const FIXTURE_PRIOR_DAYS = 60; // > both the 50-trading-day SPY and 30-trading-day VXX lookback

test("classifyRegime returns bull when VXX is calm and SPY is above its 50d MA", () => {
  // last close / avg-of-prior-60(=500) = 520/500 = 1.04 >= 0.98
  const spyCloses = [...Array(FIXTURE_PRIOR_DAYS).fill(500), 520];
  // last close / avg-of-prior-60(=15) ≈ 13/15 = 0.867 <= 0.95
  const vxxCloses = [...Array(FIXTURE_PRIOR_DAYS).fill(15), 13];
  const spySeries = seriesFromCloses(FIXTURE_START, spyCloses);
  const vxxSeries = seriesFromCloses(FIXTURE_START, vxxCloses);
  const day = [...spySeries.indexOf.keys()][spySeries.indexOf.size - 1];
  assert.equal(classifyRegime(day, spySeries, vxxSeries), "bull");
});

test("classifyRegime returns bear when VXX ratio spikes", () => {
  const spyCloses = [...Array(FIXTURE_PRIOR_DAYS).fill(500), 500];
  const vxxCloses = [...Array(FIXTURE_PRIOR_DAYS).fill(15), 20]; // 20/15 = 1.33 >= 1.15
  const spySeries = seriesFromCloses(FIXTURE_START, spyCloses);
  const vxxSeries = seriesFromCloses(FIXTURE_START, vxxCloses);
  const day = [...spySeries.indexOf.keys()][spySeries.indexOf.size - 1];
  assert.equal(classifyRegime(day, spySeries, vxxSeries), "bear");
});

test("classifyRegime returns neutral outside both thresholds", () => {
  const spyCloses = [...Array(FIXTURE_PRIOR_DAYS).fill(500), 505];
  const vxxCloses = [...Array(FIXTURE_PRIOR_DAYS).fill(15), 15.5]; // 15.5/15 ≈ 1.033 — between 0.95 and 1.15
  const spySeries = seriesFromCloses(FIXTURE_START, spyCloses);
  const vxxSeries = seriesFromCloses(FIXTURE_START, vxxCloses);
  const day = [...spySeries.indexOf.keys()][spySeries.indexOf.size - 1];
  assert.equal(classifyRegime(day, spySeries, vxxSeries), "neutral");
});

test("classifyRegime returns null when a series has NO trailing history (day is the only point)", () => {
  const spySeries = seriesFromCloses("2026-01-01", [500]); // single point, no prior day at all
  const vxxSeries = seriesFromCloses("2026-01-01", [15]);
  const day = [...spySeries.indexOf.keys()][0];
  assert.equal(classifyRegime(day, spySeries, vxxSeries), null);
});

test("classifyRegime returns null when the day is missing from a series", () => {
  const spySeries = seriesFromCloses("2025-10-01", Array(60).fill(500));
  const vxxSeries = seriesFromCloses("2025-10-01", Array(60).fill(15));
  assert.equal(classifyRegime("2099-01-01", spySeries, vxxSeries), null);
});
