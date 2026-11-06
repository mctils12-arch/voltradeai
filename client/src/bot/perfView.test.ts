// perfView — the overview's main chart model: after-tax lines vs SPY from the scoreboard's series, the
// before-tax fallback, the "sold now" ring, drawdowns and the scales. Run: npx tsx --test client/src/bot/perfView.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { perfModel, perfHeadline, drawdowns, moveWords, yDomain, niceTicks, tickLabel, monthTicks, linePath, gapBands, nearestIndex } from "./perfView";
import type { Scoreboard, Benchmark, AfterTaxSeries, ScoreRow } from "./useBotData";

const row = (date: string, prevDate: string, botCumPct: number, spyCumPct: number): ScoreRow => ({ date, prevDate, sessions: 1, botPct: 0, spyPct: 0, diffPp: 0, botCumPct, spyCumPct, cumDiffPp: botCumPct - spyCumPct, provisional: false });
// newest first, like the server
const rows = [row("2026-06-03", "2026-06-02", -3, 2), row("2026-06-02", "2026-06-01", 1, 1), row("2026-06-01", "2026-05-29", 2, -1)];
const series: AfterTaxSeries = {
  rates: { shortTerm: 0.24, longTerm: 0.15, state: 0, niit: 0 }, from: "2026-05-29", to: "2026-06-03", equityStart: 100_000, notes: ["n"],
  points: [
    { date: "2026-05-29", botPct: 0, spyPct: 0, botTaxPct: 0, spyTaxPct: 0, botAfterTaxPct: 0, spyAfterTaxPct: 0, gapPp: 0, spyTerm: "short" },
    { date: "2026-06-01", botPct: 2, spyPct: -1, botTaxPct: 0.48, spyTaxPct: 0, botAfterTaxPct: 1.52, spyAfterTaxPct: -1, gapPp: 2.52, spyTerm: "short" },
    { date: "2026-06-02", botPct: 1, spyPct: 1, botTaxPct: 0.48, spyTaxPct: 0.24, botAfterTaxPct: 0.52, spyAfterTaxPct: 0.76, gapPp: -0.24, spyTerm: "short" },
    { date: "2026-06-03", botPct: -3, spyPct: 2, botTaxPct: 0.48, spyTaxPct: 0.48, botAfterTaxPct: -3.48, spyAfterTaxPct: 1.52, gapPp: -5, spyTerm: "short" },
  ],
};
const board = (o: Partial<Scoreboard> = {}): Scoreboard => ({ rows, summary: { from: "2026-05-29", to: "2026-06-03", botCumPct: -3, spyCumPct: 2, cumDiffPp: -5, daysBeat: 1, daysCompared: 3, gapRows: 0, botMaxDdPct: -4.9, spyMaxDdPct: -2 }, barsThrough: "2026-06-03", curveThrough: "2026-06-03", fetchedAt: "2026-06-03T21:00:00Z", cashFlows: null, ...o });
const bench = (bot: number, spy: number): Benchmark => ({ since: null, d30: null, spyLast: null, barsThrough: null,
  afterTax: { rates: { shortTerm: 0.24, longTerm: 0.15, state: 0, niit: 0 }, window: { from: "2026-05-29", to: "2026-06-03", days: 5 }, spy: { pct: 2, term: "short", taxPct: 0.48, afterTaxPct: spy }, account: { pct: -3, taxPct: 0.6, afterTaxPct: bot, equityStart: 100_000, realizedTax: 480, liquidationTax: 600, openPnl: 500, openTaxIfSold: 120 }, gapPp: Math.round((bot - spy) * 100) / 100, notes: [] } });
const ok = { loading: false, error: null };

test("after-tax lines from the server's series, oldest first, with the before-tax values beside them and the tile's 'sold now' ring", () => {
  const m = perfModel(board({ afterTax: series }), bench(-3.6, 1.52), ok);
  assert.equal(m.state, "ready");
  assert.equal(m.afterTax, true);
  assert.deepEqual(m.points.map((p) => p.date), ["2026-05-29", "2026-06-01", "2026-06-02", "2026-06-03"]);
  assert.deepEqual(m.points.map((p) => p.bot), [0, 1.52, 0.52, -3.48]);
  assert.deepEqual(m.points.map((p) => p.botPre), [0, 2, 1, -3]);
  assert.deepEqual(m.soldNow, { bot: -3.6, spy: 1.52, gap: -5.12, to: "2026-06-03" });
  assert.deepEqual(m.rates, { shortTerm: 0.24, longTerm: 0.15 });
  assert.deepEqual(m.maxDd, { bot: -4.9, spy: -2 }); // the scoreboard summary's, measured on unrounded values
  assert.equal(m.note, null);
  assert.equal(perfHeadline(m), "After tax, the portfolio is down 3.5% since May 29; SPY is up 1.5%. The bot is behind SPY by 5.0 points.");
});

test("no series: the scoreboard's before-tax columns, the start at zero, and the reason in the note — never a guessed tax", () => {
  const old = perfModel(board(), undefined, ok); // a server that predates the field
  assert.equal(old.afterTax, false);
  assert.deepEqual(old.points.map((p) => [p.date, p.bot, p.spy]), [["2026-05-29", 0, 0], ["2026-06-01", 2, -1], ["2026-06-02", 1, 1], ["2026-06-03", -3, 2]]);
  assert.match(old.note ?? "", /predates the after-tax series/);
  assert.equal(old.soldNow, null, "no ring beside before-tax lines");
  const warming = perfModel(board({ afterTax: null, afterTaxReason: "sleeve ledger not loaded yet; the next poll fills it" }), bench(-3.6, 1.52), ok);
  assert.equal(warming.afterTax, false);
  assert.match(warming.note ?? "", /not loaded yet \(sleeve ledger not loaded yet/);
  assert.match(perfHeadline(warming), /^Before tax, the portfolio is down 3\.0%/);
  // a series with a non-finite value is not drawn as after tax
  const bad = perfModel(board({ afterTax: { ...series, points: series.points.map((p, i) => (i === 2 ? { ...p, botAfterTaxPct: Number.NaN } : p)) } }), undefined, ok);
  assert.equal(bad.afterTax, false);
});

test("loading, error and empty are designed states with no points", () => {
  assert.equal(perfModel(undefined, undefined, { loading: true, error: null }).state, "loading");
  const e = perfModel(undefined, undefined, { loading: false, error: "502: upstream" });
  assert.equal(e.state, "error");
  assert.equal(e.note, "502: upstream");
  const none = perfModel(board({ rows: [], summary: null }), undefined, ok);
  assert.equal(none.state, "empty");
  assert.equal(none.points.length, 0);
  assert.match(none.note ?? "", /not enough history/);
  const err = perfModel(board({ rows: [], summary: null, error: "SPY bars 403" }), undefined, ok);
  assert.equal(err.state, "error");
  assert.match(err.note ?? "", /SPY bars 403 · nothing is drawn rather than a guess/);
  assert.equal(perfHeadline(none), "");
});

test("drawdowns: from the running peak of the cumulative return, never above zero", () => {
  assert.deepEqual(drawdowns([0, 10, 5, 12, 0]), [0, 0, -4.55, 0, -10.71]);
  assert.deepEqual(drawdowns([0, -5, -10]), [0, -5, -10]);
  const m = perfModel(board({ afterTax: series }), undefined, ok);
  assert.deepEqual(m.points.map((p) => p.botDd), [0, 0, -0.98, -4.9]);
});

test("words: moves as up / down / flat with one decimal", () => {
  assert.equal(moveWords(2.44), "up 2.4%");
  assert.equal(moveWords(-11.31), "down 11.3%");
  assert.equal(moveWords(0.04), "flat");
});

test("scales: the domain holds zero and every value with padding; ticks are round and inside; labels carry a true minus", () => {
  const [lo, hi] = yDomain([-11.3, 3.3, 2.5]);
  assert.ok(lo < -11.3 && hi > 3.3);
  assert.ok(Math.abs(lo + 11.3 - -(hi - 3.3)) < 1e-9, "the same pad both ways");
  const [z0, z1] = yDomain([0.1, 0.2]);
  assert.ok(z0 < 0 && z1 > 0.2, "a flat series still gets a range");
  assert.deepEqual(niceTicks(-12.5, 4.5, 5), [-10, -5, 0]);
  assert.deepEqual(niceTicks(-1, 1, 4), [-1, -0.5, 0, 0.5, 1]);
  assert.deepEqual(niceTicks(2, 1), []);
  assert.equal(tickLabel(5), "+5%");
  assert.equal(tickLabel(-10), "−10%");
  assert.equal(tickLabel(0), "0%");
  assert.equal(tickLabel(-2.5), "−2.5%");
});

test("month ticks: the first session of each new month", () => {
  assert.deepEqual(monthTicks(["2026-05-29", "2026-06-01", "2026-06-02", "2026-07-01"]), [{ i: 1, label: "Jun" }, { i: 3, label: "Jul" }]);
  assert.deepEqual(monthTicks(["2026-05-29"]), []);
});

test("paths: a line through every point; the gap between two lines splits where they cross, each piece one sign", () => {
  const x = (i: number) => i * 10, y = (v: number) => -v;
  assert.equal(linePath([0, 1, -1], x, y), "M0.0 0.0 L10.0 -1.0 L20.0 1.0");
  const bands = gapBands([1, 2, -1, -2], [0, 0, 0, 0], x, y);
  assert.deepEqual(bands.map((b) => b.sign), [1, -1]);
  // the crossing sits where a − b changes sign, between x = 10 and 20 at fraction 2 / 3
  assert.match(bands[0].d, /16\.7 -?0\.0/);
  assert.match(bands[1].d, /^M16\.7 -?0\.0/);
  assert.deepEqual(gapBands([1, 1], [1, 1], x, y), [], "no gap, no band");
});

test("hover: the nearest point to the pointer, clamped to the series", () => {
  assert.equal(nearestIndex(46, 101, 46, 1046), 0);
  assert.equal(nearestIndex(1046, 101, 46, 1046), 100);
  assert.equal(nearestIndex(551, 101, 46, 1046), 51);
  assert.equal(nearestIndex(-50, 10, 0, 100), 0);
  assert.equal(nearestIndex(500, 10, 0, 100), 9);
  assert.equal(nearestIndex(5, 1, 0, 100), 0);
});
