// spyBenchmark — the account vs buy-and-hold SPY (GOAL priority 3's own
// yardstick). Run: npx tsx --test server/spyBenchmark.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { benchmarkVsSpy, closeOnOrBefore, BENCH_SHORT_WINDOW_DAYS } from "./spyBenchmark";

const bars = [
  { date: "2026-05-06", close: 730.13 },
  { date: "2026-09-18", close: 761.69 },
  { date: "2026-09-21", close: 773.5 },
  { date: "2026-10-19", close: 766.5 },
  { date: "2026-10-20", close: 768.41 },
];

test("since-inception and 30-day windows: account vs SPY close-to-close, gap in points", () => {
  // the real 2026-10-20 endpoints: -6.90% vs +5.24% (Yahoo adjusted closes)
  const curve = [
    { date: "2026-10-20", value: 99373.36 },
    { date: "2026-05-06", value: 106742.68 },
    { date: "2026-09-20", value: 102560.45 }, // a Sunday — pairs with Friday's close
  ];
  const b = benchmarkVsSpy(curve, bars);
  assert.deepEqual(b.since, { from: "2026-05-06", to: "2026-10-20", days: 167, accountPct: -6.9, spyPct: 5.24, gapPp: -12.15 });
  assert.equal(b.d30?.from, "2026-09-20");
  assert.equal(b.d30?.days, BENCH_SHORT_WINDOW_DAYS);
  assert.equal(b.d30?.spyPct, Math.round((768.41 / 761.69 - 1) * 10000) / 100, "non-session day uses the latest earlier close");
  assert.deepEqual(b.spyLast, { date: "2026-10-20", pct: 0.25 });
  assert.equal(b.barsThrough, "2026-10-20");
});

test("nothing is inferred: missing closes or a short curve give null windows", () => {
  assert.equal(benchmarkVsSpy([{ date: "2026-10-20", value: 1 }], bars).since, null, "one point is no window");
  assert.equal(benchmarkVsSpy([{ date: "2026-01-02", value: 100 }, { date: "2026-10-20", value: 90 }], bars).since, null,
    "curve starts before the first SPY bar");
  const noBars = benchmarkVsSpy([{ date: "2026-05-06", value: 100 }, { date: "2026-10-20", value: 90 }], []);
  assert.deepEqual(noBars, { since: null, d30: null, spyLast: null, barsThrough: null });
  const young = benchmarkVsSpy([{ date: "2026-10-19", value: 100 }, { date: "2026-10-20", value: 101 }], bars);
  assert.equal(young.d30, null, "a curve shorter than the window has no 30-day comparison");
  assert.equal(young.since?.gapPp, Math.round((1 - (768.41 / 766.5 - 1) * 100) * 100) / 100);
});

test("bad rows are skipped, not compared", () => {
  const b = benchmarkVsSpy([
    { date: "2026-05-06", value: 106742.68 }, { date: "2026-10-20", value: 99373.36 },
    { date: "2026-10-21", value: 0 }, { date: "2026-10-22", value: Number.NaN },
  ], [...bars, { date: "2026-10-21", close: -1 }]);
  assert.equal(b.since?.to, "2026-10-20");
  assert.equal(b.barsThrough, "2026-10-20");
});

test("closeOnOrBefore: exact, between, before-first", () => {
  assert.equal(closeOnOrBefore(bars, "2026-09-21")?.close, 773.5);
  assert.equal(closeOnOrBefore(bars, "2026-09-20")?.date, "2026-09-18");
  assert.equal(closeOnOrBefore(bars, "2026-01-01"), null);
});
