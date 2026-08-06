import { test } from "node:test";
import assert from "node:assert/strict";
import { bucketDay, BUCKET_SIZE } from "./finra_shortvol_gate2";
import { FLOOR_TOTAL_VOL } from "../server/finraShortVolume";
import type { ShortVolRow } from "../server/finraShortVolume";

function row(symbol: string, short_vol: number, total_vol: number): ShortVolRow {
  return { date: "2026-01-07", symbol, short_vol, short_exempt_vol: 0, total_vol, market: "Q", rt: "2026-01-07" };
}

test("bucketDay filters below FLOOR_TOTAL_VOL", () => {
  const rows = [
    row("BELOW", 100, FLOOR_TOTAL_VOL - 1),
    row("AT", 100, FLOOR_TOTAL_VOL),
    row("ABOVE", 100, FLOOR_TOTAL_VOL + 1),
  ];
  const b = bucketDay("2026-01-07", rows);
  assert.equal(b.qualifying, 2);
  const tickers = [...b.highShort, ...b.lowShort, ...b.neutral].map((r) => r.ticker);
  assert.ok(!tickers.includes("BELOW"));
});

test("bucketDay ranks HIGH_SHORT as highest short_ratio, LOW_SHORT as lowest", () => {
  const rows = Array.from({ length: 30 }, (_, i) =>
    row(`T${i}`, i * 1000, 1_000_000)); // short_ratio ascending 0 .. 0.029
  const b = bucketDay("2026-01-07", rows);
  assert.equal(b.qualifying, 30);
  const highRatios = b.highShort.map((r) => r.short_ratio);
  const lowRatios = b.lowShort.map((r) => r.short_ratio);
  assert.ok(Math.min(...highRatios) >= Math.max(...lowRatios));
  // bucket size caps at floor(n/3) when n < BUCKET_SIZE*3
  assert.equal(b.highShort.length, Math.floor(30 / 3));
  assert.equal(b.lowShort.length, Math.floor(30 / 3));
  assert.equal(b.neutral.length, Math.floor(30 / 3));
});

test("bucketDay bucket size caps at BUCKET_SIZE when the qualifying universe is large", () => {
  const rows = Array.from({ length: 500 }, (_, i) =>
    row(`T${i}`, i * 100, 1_000_000));
  const b = bucketDay("2026-01-07", rows);
  assert.equal(b.qualifying, 500);
  assert.equal(b.highShort.length, BUCKET_SIZE);
  assert.equal(b.lowShort.length, BUCKET_SIZE);
  assert.equal(b.neutral.length, BUCKET_SIZE);
});

test("bucketDay buckets never overlap and neutral sits between them", () => {
  const rows = Array.from({ length: 200 }, (_, i) =>
    row(`T${i}`, i * 100, 1_000_000));
  const b = bucketDay("2026-01-07", rows);
  const lowMax = Math.max(...b.lowShort.map((r) => r.short_ratio));
  const neutralVals = b.neutral.map((r) => r.short_ratio);
  const highMin = Math.min(...b.highShort.map((r) => r.short_ratio));
  assert.ok(lowMax <= Math.min(...neutralVals));
  assert.ok(Math.max(...neutralVals) <= highMin);
});

test("bucketDay handles a total_vol=0 row without dividing by zero", () => {
  const rows = [row("ZERO", 0, 0), row("REAL", 1000, FLOOR_TOTAL_VOL)];
  const b = bucketDay("2026-01-07", rows);
  assert.equal(b.qualifying, 1);
  assert.ok(![...b.highShort, ...b.lowShort, ...b.neutral].some((r) => r.ticker === "ZERO"));
});
