// scoreboard — the daily bot-vs-SPY table on the bot page (synthetic data).
// Run: npx tsx --test server/scoreboard.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildScoreboard, maxDrawdownPct, SCOREBOARD_CACHE_MS } from "./scoreboard";

// Mon 10-19 .. Fri 10-23, then Mon 10-26 (weekend 24/25 has no bars)
const bars = [
  { date: "2026-10-19", close: 100 },
  { date: "2026-10-20", close: 101 },
  { date: "2026-10-21", close: 99.99 },
  { date: "2026-10-22", close: 102 },
  { date: "2026-10-23", close: 102 },
  { date: "2026-10-26", close: 104.04 },
];
const r2 = (x: number) => Math.round(x * 100) / 100;

test("one row per session, newest first: daily returns, difference, cumulative since the first point", () => {
  const curve = [
    { date: "2026-10-20", value: 1020 }, // any order in
    { date: "2026-10-19", value: 1000 },
    { date: "2026-10-21", value: 1009.8 },
  ];
  const s = buildScoreboard(curve, bars, "2026-10-28");
  assert.deepEqual(s.rows.map((r) => r.date), ["2026-10-21", "2026-10-20"]);
  const [d21, d20] = s.rows;
  assert.deepEqual(d20, { date: "2026-10-20", prevDate: "2026-10-19", sessions: 1, botPct: 2, spyPct: 1, diffPp: 1, botCumPct: 2, spyCumPct: 1, cumDiffPp: 1, provisional: false });
  assert.equal(d21.botPct, -1);
  assert.equal(d21.spyPct, r2((99.99 / 101 - 1) * 100));
  assert.equal(d21.diffPp, r2(-1 - (99.99 / 101 - 1) * 100));
  assert.equal(d21.botCumPct, 0.98);
  assert.equal(d21.spyCumPct, -0.01);
  assert.equal(d21.cumDiffPp, 0.99);
  assert.equal(s.summary?.from, "2026-10-19");
  assert.equal(s.summary?.to, "2026-10-21");
  assert.equal(s.summary?.daysBeat, 1, "beat SPY on 10-20, lost on 10-21");
  assert.equal(s.summary?.daysCompared, 2);
  assert.equal(s.barsThrough, "2026-10-26");
  assert.equal(s.curveThrough, "2026-10-21");
});

test("cumulative is the compounded daily returns (not their sum)", () => {
  const curve = [1000, 1100, 990, 1089, 1200, 1150].map((value, k) => ({ date: bars[k].date, value }));
  const s = buildScoreboard(curve, bars, "2026-10-28");
  const chrono = [...s.rows].reverse();
  let wealth = 1, spyWealth = 1, sum = 0;
  for (let k = 0; k < chrono.length; k++) {
    const raw = curve[k + 1].value / curve[k].value;
    wealth *= raw; sum += (raw - 1) * 100;
    spyWealth *= bars[k + 1].close / bars[k].close;
    assert.equal(chrono[k].botCumPct, r2((wealth - 1) * 100), `${chrono[k].date} bot`);
    assert.equal(chrono[k].spyCumPct, r2((spyWealth - 1) * 100), `${chrono[k].date} spy`);
  }
  assert.notEqual(chrono[chrono.length - 1].botCumPct, r2(sum), "a sum of daily returns would differ");
  assert.equal(s.summary?.botCumPct, 15);
  assert.equal(s.summary?.spyCumPct, 4.04);
  assert.equal(s.summary?.cumDiffPp, 10.96);
});

test("weekend/holiday points are skipped; a missed session makes one row spanning both, flagged", () => {
  const curve = [
    { date: "2026-10-19", value: 1000 },
    { date: "2026-10-20", value: 1010 },
    // 10-21 missing (server down), 10-22 present
    { date: "2026-10-22", value: 1030.2 },
    { date: "2026-10-23", value: 1030.2 },
    { date: "2026-10-24", value: 1030.2 }, // Saturday: no bar → no row
    { date: "2026-10-25", value: 999 },    // Sunday junk value never compared
    { date: "2026-10-26", value: 1040 },
  ];
  const s = buildScoreboard(curve, bars, "2026-10-28");
  assert.deepEqual(s.rows.map((r) => r.date), ["2026-10-26", "2026-10-23", "2026-10-22", "2026-10-20"]);
  const gap = s.rows.find((r) => r.date === "2026-10-22")!;
  assert.equal(gap.prevDate, "2026-10-20");
  assert.equal(gap.sessions, 2, "10-21 and 10-22 both inside the row");
  assert.equal(gap.botPct, 2);
  assert.equal(gap.spyPct, r2((102 / 101 - 1) * 100), "SPY measured over the SAME two sessions");
  const mon = s.rows[0];
  assert.equal(mon.prevDate, "2026-10-23", "Monday compares with Friday, not the weekend points");
  assert.equal(mon.sessions, 1);
  assert.equal(mon.botPct, r2((1040 / 1030.2 - 1) * 100));
  assert.equal(s.summary?.gapRows, 1);
  assert.equal(s.summary?.daysCompared, 4);
});

test("no lookahead: data after a row never changes that row", () => {
  const early = [{ date: "2026-10-19", value: 1000 }, { date: "2026-10-20", value: 990 }, { date: "2026-10-21", value: 1001 }];
  const a = buildScoreboard(early, bars.slice(0, 3), "2026-10-28");
  const b = buildScoreboard([...early, { date: "2026-10-22", value: 500 }, { date: "2026-10-23", value: 2000 }], bars, "2026-10-28");
  for (const row of a.rows) assert.deepEqual(b.rows.find((r) => r.date === row.date), row, row.date);
});

test("provisional: only the latest row, only while its date is still today", () => {
  const curve = [{ date: "2026-10-22", value: 1000 }, { date: "2026-10-23", value: 1005 }, { date: "2026-10-26", value: 1010 }];
  const live = buildScoreboard(curve, bars, "2026-10-26");
  assert.deepEqual(live.rows.map((r) => r.provisional), [true, false]);
  const later = buildScoreboard(curve, bars, "2026-10-27");
  assert.deepEqual(later.rows.map((r) => r.provisional), [false, false]);
});

test("nothing is inferred: too little data or no bars gives no rows and no summary; bad rows are skipped", () => {
  const empty = { rows: [], summary: null, barsThrough: null, curveThrough: null };
  assert.deepEqual(buildScoreboard([], [], "2026-10-28"), empty);
  assert.deepEqual(buildScoreboard([{ date: "2026-10-19", value: 1 }, { date: "2026-10-20", value: 2 }], [], "2026-10-28"), { ...empty, curveThrough: "2026-10-20" });
  const one = buildScoreboard([{ date: "2026-10-19", value: 1000 }], bars, "2026-10-28");
  assert.equal(one.rows.length, 0);
  assert.equal(one.summary, null);
  const bad = buildScoreboard([
    { date: "2026-10-19", value: 1000 }, { date: "2026-10-20", value: 0 }, { date: "2026-10-21", value: Number.NaN }, { date: "2026-10-22", value: 1010 },
  ], [...bars.slice(0, 3), { date: "2026-10-22", close: 102 }, { date: "2026-10-23", close: -5 }], "2026-10-28");
  assert.deepEqual(bad.rows.map((r) => [r.prevDate, r.date, r.sessions]), [["2026-10-19", "2026-10-22", 3]]);
  assert.equal(bad.barsThrough, "2026-10-22", "a non-positive close is not a bar");
});

test("max drawdown: worst peak-to-trough on the session points, both legs", () => {
  assert.equal(maxDrawdownPct([100, 120, 90, 130, 117]), -25);
  assert.equal(maxDrawdownPct([1, 2, 3]), 0);
  assert.equal(maxDrawdownPct([]), 0);
  const curve = [1000, 1100, 990, 1089, 1200, 1150].map((value, k) => ({ date: bars[k].date, value }));
  const s = buildScoreboard(curve, bars, "2026-10-28");
  assert.equal(s.summary?.botMaxDdPct, -10);
  assert.equal(s.summary?.spyMaxDdPct, r2((99.99 / 101 - 1) * 100));
});

test("the server cache window sits inside the 5–15 min the page is built for", () => {
  assert.ok(SCOREBOARD_CACHE_MS >= 5 * 60_000 && SCOREBOARD_CACHE_MS <= 15 * 60_000);
});
