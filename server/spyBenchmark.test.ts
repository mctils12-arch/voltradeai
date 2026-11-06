// spyBenchmark — the account vs buy-and-hold SPY (GOAL priority 3's own
// yardstick). Run: npx tsx --test server/spyBenchmark.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { benchmarkVsSpy, closeOnOrBefore, BENCH_SHORT_WINDOW_DAYS, SPY_BENCH_ADJUSTMENT, SPY_BENCH_BASIS } from "./spyBenchmark";

const here = path.dirname(fileURLToPath(import.meta.url));
/** the source of one of bot.ts's route-level helpers (indented two spaces), from its signature to its closing brace */
const fnSrc = (bot: string, signature: string): string => {
  const i = bot.indexOf(signature);
  assert.ok(i > 0, `${signature} exists`);
  return bot.slice(i, bot.indexOf("\n  }\n", i));
};

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

// ── SPY's basis: PRICE RETURN ([RULE-REVIEW] 2026-11-05, human-ordered "fix the dividend mismatch") ──
// The paper account is credited no dividends, so SPY is compared without them: split-adjusted bars.

test("the SPY series is price return: split-adjusted, and labelled with its basis", () => {
  assert.equal(SPY_BENCH_ADJUSTMENT, "split");
  assert.equal(SPY_BENCH_BASIS, "price return: SPY without dividends, because the paper account receives none (2026-11-05)");
});

test("wiring ratchet: every benchmark reader shares ONE split-adjusted SPY fetch, and the payloads name the basis", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.match(bot, /import \{ benchmarkVsSpy, SPY_BENCH_CACHE_MS, SPY_BENCH_ADJUSTMENT, SPY_BENCH_BASIS, type DailyClose, type Benchmark \} from "\.\/spyBenchmark"/);
  // the one fetch fills the one cache (the benchmark + its after-tax leg, the scoreboard, the TODAY tile)
  const bars = fnSrc(bot, "async function spyBenchBars(");
  assert.match(bars, /spyBenchCache = \{ at: Date\.now\(\), from, bars: await fetchDailyBarsRange\("SPY", from, new Date\(\)\.toISOString\(\), SPY_BENCH_ADJUSTMENT\) \};/);
  assert.equal(bot.split("spyBenchCache = ").length - 1, 1, "spyBenchBars is the only writer of the SPY cache");
  assert.equal(bot.split('fetchDailyBarsRange("SPY"').length - 1, 1, "one SPY fetch, never a second series on another adjustment");
  // to the end of the statement ([^)]* would stop at toISOString() and never see the old "all")
  assert.doesNotMatch(bot, /fetchDailyBarsRange\("SPY",[^;\n]*"all"/, "no dividend-adjusted SPY fetch");
  // the owner route and the "benchmark" diag probe (one body) carry the adjustment and the basis
  const body = fnSrc(bot, "async function benchmarkBody(");
  assert.match(body, /const spy = await spyBenchBars\(oldest\);/);
  assert.match(body, /benchmarkVsSpy\(equityCurve, spy\.bars\)/);
  assert.match(body, /adjustment: SPY_BENCH_ADJUSTMENT, basis: SPY_BENCH_BASIS, afterTax: afterTaxFor\(bench, rates\)/);
  assert.doesNotMatch(body, /adjustment: "all"/);
  // the scoreboard reads the same cache and states the same basis in its sources
  const score = fnSrc(bot, "async function scoreboardBody(");
  assert.match(score, /const spy = await spyBenchBars\(oldest\);/);
  assert.ok(score.includes("spy: `SPY daily closes, ${SPY_BENCH_BASIS} (Alpaca bars, adjustment=${SPY_BENCH_ADJUSTMENT}, feed=iex)`"), "scoreboard sources.spy names the basis");
  assert.doesNotMatch(score, /dividend-adjusted|adjustment=all/);
  // the TODAY tile reads that cache without fetching (its own ratchet is in todayTile.test.ts)
  assert.match(fnSrc(bot, "function todayFor("), /spyBars: spyBenchCache\?\.bars \?\? null/);
});

test("restatement on identical inputs (2026-11-05): price-only SPY moves the since-May-6 window by the two dividends and leaves the 30-day window alone", () => {
  // Yahoo SPY closes: 05-06 733.83 split-adjusted / 730.13 dividend-adjusted (ex-dates after it: 06-18 $1.904,
  // 09-18 $1.889); 10-05 774.83, 10-06 779.09 and 11-04 766.81 are after the last ex-date (the same on both bases);
  // 11-05 740.82 = 11-04 × (1 − 3.39 %), the live probe's intraday spyLast at its 17:19Z fetch. Equity: the live
  // curve (equity_curve probe, 18:12Z; 11-05 is the latest hourly snapshot, intraday).
  const curve = [{ date: "2026-05-06", value: 106742.68 }, { date: "2026-10-06", value: 105773.85 }, { date: "2026-11-05", value: 90072.76 }];
  const priceBars = [{ date: "2026-05-06", close: 733.83 }, { date: "2026-10-06", close: 779.09 }, { date: "2026-11-04", close: 766.81 }, { date: "2026-11-05", close: 740.82 }];
  const totalBars = priceBars.map((b) => (b.date === "2026-05-06" ? { ...b, close: 730.13 } : b));
  // the dividend-adjusted start is the split-adjusted close times the two ex-date factors (1 − dividend ÷ prior close)
  assert.ok(Math.abs(733.83 * (1 - 1.904 / 740.96) * (1 - 1.889 / 762.6) - 730.13) < 0.01);
  const before = benchmarkVsSpy(curve, totalBars); // adjustment=all, until 2026-11-05
  const after = benchmarkVsSpy(curve, priceBars); // adjustment=split
  assert.deepEqual(before.since, { from: "2026-05-06", to: "2026-11-05", days: 183, accountPct: -15.62, spyPct: 1.46, gapPp: -17.08 });
  assert.deepEqual(after.since, { from: "2026-05-06", to: "2026-11-05", days: 183, accountPct: -15.62, spyPct: 0.95, gapPp: -16.57 });
  assert.deepEqual(after.d30, before.d30, "no ex-date between 10-06 and 11-05: the 30-day window is identical");
  assert.deepEqual(after.d30, { from: "2026-10-06", to: "2026-11-05", days: 30, accountPct: -14.84, spyPct: -4.91, gapPp: -9.93 });
  assert.equal(Math.round((before.since!.spyPct - after.since!.spyPct) * 100) / 100, 0.51, "the account reads 0.51 pt better: SPY's dividends over the window");
  assert.deepEqual(after.spyLast, before.spyLast, "the last bar's move is after the last ex-date: the same on both bases");
  // the same restatement through the last COMPLETE session (the 11-04 close and the curve's after-close 11-04 row)
  const curve4 = [{ date: "2026-05-06", value: 106742.68 }, { date: "2026-10-05", value: 105592.66 }, { date: "2026-11-04", value: 93765 }];
  const price4 = [{ date: "2026-05-06", close: 733.83 }, { date: "2026-10-05", close: 774.83 }, { date: "2026-11-04", close: 766.81 }];
  const before4 = benchmarkVsSpy(curve4, price4.map((b) => (b.date === "2026-05-06" ? { ...b, close: 730.13 } : b)));
  const after4 = benchmarkVsSpy(curve4, price4);
  assert.deepEqual(before4.since, { from: "2026-05-06", to: "2026-11-04", days: 182, accountPct: -12.16, spyPct: 5.02, gapPp: -17.18 });
  assert.deepEqual(after4.since, { from: "2026-05-06", to: "2026-11-04", days: 182, accountPct: -12.16, spyPct: 4.49, gapPp: -16.65 });
  assert.deepEqual(after4.d30, before4.d30);
  assert.deepEqual(after4.d30, { from: "2026-10-05", to: "2026-11-04", days: 30, accountPct: -11.2, spyPct: -1.04, gapPp: -10.17 });
});
