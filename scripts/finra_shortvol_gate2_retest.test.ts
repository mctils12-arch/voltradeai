import { test } from "node:test";
import assert from "node:assert/strict";
import {
  bucketDayRetest, classifyRegime, mulberry32, daySeed, seededShuffle,
  BUCKET_SIZE, BASELINE_SIZE,
} from "./finra_shortvol_gate2_retest";
import { FLOOR_TOTAL_VOL } from "../server/finraShortVolume";
import type { ShortVolRow } from "../server/finraShortVolume";
import type { Series } from "./occ_volume_gate2";

function row(symbol: string, short_vol: number, total_vol: number): ShortVolRow {
  return { date: "2026-01-07", symbol, short_vol, short_exempt_vol: 0, total_vol, market: "Q", rt: "2026-01-07" };
}

test("mulberry32 is deterministic for a fixed seed", () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  const seqA = Array.from({ length: 5 }, () => a());
  const seqB = Array.from({ length: 5 }, () => b());
  assert.deepEqual(seqA, seqB);
});

test("mulberry32 produces values in [0, 1)", () => {
  const rng = mulberry32(7);
  for (let i = 0; i < 100; i++) {
    const v = rng();
    assert.ok(v >= 0 && v < 1, `value ${v} out of range`);
  }
});

test("daySeed is deterministic and differs across days", () => {
  assert.equal(daySeed("2026-01-07"), daySeed("2026-01-07"));
  assert.notEqual(daySeed("2026-01-07"), daySeed("2026-01-14"));
});

test("seededShuffle is a reproducible permutation of the input (no drops/dupes)", () => {
  const input = Array.from({ length: 20 }, (_, i) => i);
  const out1 = seededShuffle(input, mulberry32(1));
  const out2 = seededShuffle(input, mulberry32(1));
  assert.deepEqual(out1, out2);
  assert.deepEqual([...out1].sort((a, b) => a - b), input);
});

test("bucketDayRetest filters below FLOOR_TOTAL_VOL", () => {
  const rows = [
    row("BELOW", 100, FLOOR_TOTAL_VOL - 1),
    row("AT", 100, FLOOR_TOTAL_VOL),
    row("ABOVE", 100, FLOOR_TOTAL_VOL + 1),
  ];
  const b = bucketDayRetest("2026-01-07", rows);
  assert.equal(b.qualifying, 2);
  const tickers = [...b.highShort, ...b.baseline].map((r) => r.ticker);
  assert.ok(!tickers.includes("BELOW"));
});

test("bucketDayRetest ranks HIGH_SHORT as the highest short_ratio names", () => {
  const rows = Array.from({ length: 200 }, (_, i) => row(`T${i}`, i * 100, 1_000_000));
  const b = bucketDayRetest("2026-01-07", rows);
  assert.equal(b.highShort.length, BUCKET_SIZE);
  const highRatios = b.highShort.map((r) => r.short_ratio);
  // highest-ratio 40 tickers are T160..T199 (short_ratio = i*100/1_000_000)
  assert.ok(Math.min(...highRatios) >= 159 * 100 / 1_000_000 - 1e-9);
});

test("bucketDayRetest HIGH_SHORT and BASELINE never overlap", () => {
  const rows = Array.from({ length: 300 }, (_, i) => row(`T${i}`, i * 100, 1_000_000));
  const b = bucketDayRetest("2026-01-07", rows);
  const highSet = new Set(b.highShort.map((r) => r.ticker));
  for (const r of b.baseline) assert.ok(!highSet.has(r.ticker), `${r.ticker} in both buckets`);
});

test("bucketDayRetest baseline caps at BASELINE_SIZE when population is large", () => {
  const rows = Array.from({ length: 1000 }, (_, i) => row(`T${i}`, i * 10, 1_000_000));
  const b = bucketDayRetest("2026-01-07", rows);
  assert.equal(b.highShort.length, BUCKET_SIZE);
  assert.equal(b.baseline.length, BASELINE_SIZE);
});

test("bucketDayRetest baseline shrinks honestly (never fabricated) when remainder is small", () => {
  // 50 qualifying rows -> highSize = floor(50/3) = 16, remainder = 34 < BASELINE_SIZE
  const rows = Array.from({ length: 50 }, (_, i) => row(`T${i}`, i * 100, 1_000_000));
  const b = bucketDayRetest("2026-01-07", rows);
  assert.equal(b.highShort.length, Math.floor(50 / 3));
  assert.equal(b.baseline.length, 50 - Math.floor(50 / 3));
});

test("bucketDayRetest is deterministic across repeated calls (same day, same input)", () => {
  const rows = Array.from({ length: 300 }, (_, i) => row(`T${i}`, i * 100, 1_000_000));
  const b1 = bucketDayRetest("2026-01-07", rows);
  const b2 = bucketDayRetest("2026-01-07", rows);
  assert.deepEqual(b1.baseline.map((r) => r.ticker), b2.baseline.map((r) => r.ticker));
});

test("bucketDayRetest baseline draw differs across different days (not a fixed pick)", () => {
  const rows = Array.from({ length: 300 }, (_, i) => row(`T${i}`, i * 100, 1_000_000));
  const bA = bucketDayRetest("2026-01-07", rows);
  const bB = bucketDayRetest("2026-01-14", rows);
  assert.notDeepEqual(bA.baseline.map((r) => r.ticker), bB.baseline.map((r) => r.ticker));
});

function series(pairs: [string, number][]): Series {
  const dates = pairs.map((p) => p[0]);
  const closes = pairs.map((p) => p[1]);
  const indexOf = new Map(dates.map((d, i) => [d, i]));
  return { dates, closes, indexOf };
}

test("classifyRegime returns UP when SPY's trailing 20d return is non-negative", () => {
  const dates = Array.from({ length: 25 }, (_, i) => `2026-01-${String(i + 1).padStart(2, "0")}`);
  const closes = dates.map((_, i) => 100 + i); // rising
  const s = series(dates.map((d, i) => [d, closes[i]] as [string, number]));
  assert.equal(classifyRegime(s, dates[24]), "UP");
});

test("classifyRegime returns DOWN when SPY's trailing 20d return is negative", () => {
  const dates = Array.from({ length: 25 }, (_, i) => `2026-01-${String(i + 1).padStart(2, "0")}`);
  const closes = dates.map((_, i) => 200 - i * 3); // falling
  const s = series(dates.map((d, i) => [d, closes[i]] as [string, number]));
  assert.equal(classifyRegime(s, dates[24]), "DOWN");
});

test("classifyRegime returns null when there isn't 20 trading days of lookback", () => {
  const dates = Array.from({ length: 10 }, (_, i) => `2026-01-${String(i + 1).padStart(2, "0")}`);
  const s = series(dates.map((d, i) => [d, 100 + i] as [string, number]));
  assert.equal(classifyRegime(s, dates[5]), null);
});

test("classifyRegime returns null for a day not present in the series", () => {
  const dates = Array.from({ length: 25 }, (_, i) => `2026-01-${String(i + 1).padStart(2, "0")}`);
  const s = series(dates.map((d, i) => [d, 100 + i] as [string, number]));
  assert.equal(classifyRegime(s, "2099-01-01"), null);
});
