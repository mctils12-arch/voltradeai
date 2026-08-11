import { test } from "node:test";
import assert from "node:assert/strict";
import {
  splitPopulationAndTop,
  computeSpread,
  mulberry32,
  sampleWithoutReplacement,
  POPULATION_SAMPLE_SIZE,
} from "./finra_shortvol_gate2_retest";
import { BUCKET_SIZE } from "./finra_shortvol_gate2";
import { FLOOR_TOTAL_VOL } from "../server/finraShortVolume";
import type { ShortVolRow } from "../server/finraShortVolume";

function row(symbol: string, short_vol: number, total_vol: number): ShortVolRow {
  return { date: "2026-01-07", symbol, short_vol, short_exempt_vol: 0, total_vol, market: "Q", rt: "2026-01-07" };
}

test("splitPopulationAndTop filters below FLOOR_TOTAL_VOL", () => {
  const rows = [
    row("BELOW", 100, FLOOR_TOTAL_VOL - 1),
    row("AT", 100, FLOOR_TOTAL_VOL),
    row("ABOVE", 100, FLOOR_TOTAL_VOL + 1),
  ];
  const s = splitPopulationAndTop("2026-01-07", rows);
  assert.equal(s.qualifying.length, 2);
  assert.ok(!s.qualifying.some((r) => r.ticker === "BELOW"));
});

test("splitPopulationAndTop ranks highShort as the highest short_ratio names", () => {
  const rows = Array.from({ length: 30 }, (_, i) => row(`T${i}`, i * 1000, 1_000_000)); // short_ratio ascending
  const s = splitPopulationAndTop("2026-01-07", rows);
  assert.equal(s.qualifying.length, 30);
  const bucketSize = Math.floor(30 / 3);
  assert.equal(s.highShort.length, bucketSize);
  // highShort must be exactly the top-ratio names (T29 downward)
  const highTickers = new Set(s.highShort.map((r) => r.ticker));
  assert.ok(highTickers.has("T29"));
  assert.ok(!highTickers.has("T0"));
});

test("splitPopulationAndTop caps highShort at BUCKET_SIZE for a large universe", () => {
  const rows = Array.from({ length: 500 }, (_, i) => row(`T${i}`, i * 100, 1_000_000));
  const s = splitPopulationAndTop("2026-01-07", rows);
  assert.equal(s.highShort.length, BUCKET_SIZE);
});

test("computeSpread returns null when either side is empty", () => {
  assert.equal(computeSpread([], [0.01]), null);
  assert.equal(computeSpread([0.01], []), null);
});

test("computeSpread: top mean minus population mean, correct sign", () => {
  const r = computeSpread([0.0, 0.02], [0.1, 0.1]); // pop mean=0.01, top mean=0.1
  assert.ok(r);
  assert.ok(Math.abs(r!.popMean - 0.01) < 1e-9);
  assert.ok(Math.abs(r!.topMean - 0.1) < 1e-9);
  assert.ok(Math.abs(r!.spread - 0.09) < 1e-9);
});

test("mulberry32 is deterministic for a fixed seed", () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  const seqA = Array.from({ length: 5 }, () => a());
  const seqB = Array.from({ length: 5 }, () => b());
  assert.deepEqual(seqA, seqB);
  for (const v of seqA) { assert.ok(v >= 0 && v < 1); }
});

test("sampleWithoutReplacement never repeats an element and respects the pool size", () => {
  const rng = mulberry32(7);
  const pool = Array.from({ length: 10 }, (_, i) => ({ ticker: `T${i}`, short_ratio: i, total_vol: 1 }));
  const sample = sampleWithoutReplacement(pool, 5, rng);
  assert.equal(sample.length, 5);
  const tickers = sample.map((r) => r.ticker);
  assert.equal(new Set(tickers).size, 5);
});

test("sampleWithoutReplacement caps at the pool size when k exceeds it", () => {
  const rng = mulberry32(1);
  const pool = [{ ticker: "A", short_ratio: 0, total_vol: 1 }, { ticker: "B", short_ratio: 0, total_vol: 1 }];
  const sample = sampleWithoutReplacement(pool, POPULATION_SAMPLE_SIZE, rng);
  assert.equal(sample.length, 2);
});

test("sampleWithoutReplacement is deterministic across independent runs with the same seed sequence", () => {
  const pool = Array.from({ length: 100 }, (_, i) => ({ ticker: `T${i}`, short_ratio: i, total_vol: 1 }));
  const rngA = mulberry32(20260811);
  const rngB = mulberry32(20260811);
  const sampleA = sampleWithoutReplacement(pool, 10, rngA);
  const sampleB = sampleWithoutReplacement(pool, 10, rngB);
  assert.deepEqual(sampleA.map((r) => r.ticker), sampleB.map((r) => r.ticker));
});
