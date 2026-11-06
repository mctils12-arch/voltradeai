// afterTaxSeries — the vs-SPY comparison after tax, one point per session (the
// bot page's main chart, 2026-11-06). Run: npx tsx --test server/afterTaxSeries.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterTaxSeries } from "./afterTaxSeries";
import { afterTaxBenchmark } from "./afterTaxBenchmark";
import { buildScoreboard } from "./scoreboard";
import { taxLedger } from "./taxLedger";
import type { RealizedEvent } from "./sleeveLedger";
import type { FillRow } from "./killSwitchAutoResume";
import type { CurvePoint, DailyClose } from "./spyBenchmark";

const here = path.dirname(fileURLToPath(import.meta.url));
/** source with // and /* comments removed, so a comment naming a pattern never counts as the pattern */
const codeOnly = (src: string) => src.replace(/\/\*[\s\S]*?\*\//g, "").split("\n").filter((l) => !/^\s*\/\//.test(l)).join("\n");
const D = 86_400_000;
const T0 = Date.parse("2026-05-06T00:00:00Z");
const day = (k: number) => new Date(T0 + k * D).toISOString().slice(0, 10);
const close = (a: number, b: number, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} ≠ ${b}`);
// a close on day k at 15:00Z (a session hour), opened on day `open`
const ev = (symbol: string, k: number, pnl: number, open: number, qty = 10): RealizedEvent => ({ symbol, t: T0 + k * D + 15 * 3_600_000, pnl, openedAt: T0 + open * D, qty, closeDir: -1, basis: qty * 100 });
const buy = (symbol: string, k: number, qty = 10): FillRow => ({ symbol, dir: 1, qty, price: 10, t: T0 + k * D + 16 * 3_600_000 });

// five sessions: day 0 (start) .. day 4, plus a weekend-like curve point on day 5 with no SPY bar
const curve: CurvePoint[] = [
  { date: day(0), value: 100_000 }, { date: day(1), value: 101_000 }, { date: day(2), value: 99_000 },
  { date: day(3), value: 102_000 }, { date: day(4), value: 103_000 }, { date: day(5), value: 103_500 },
];
const bars: DailyClose[] = [0, 1, 2, 3, 4].map((k) => ({ date: day(k), close: [500, 505, 510, 495, 520][k] }));

test("one point per session date, oldest first: the start is all zeros, a curve date without a SPY bar is skipped — the scoreboard's own dates", () => {
  const s = afterTaxSeries(curve, bars, { events: [], fills: [] });
  assert.ok(s);
  assert.deepEqual(s.points.map((p) => p.date), [day(0), day(1), day(2), day(3), day(4)]);
  const board = buildScoreboard(curve, bars, day(9));
  assert.deepEqual(s.points.slice(1).map((p) => p.date), board.rows.map((r) => r.date).reverse());
  // before tax, the line IS the scoreboard's cumulative columns
  for (const r of board.rows) {
    const p = s.points.find((x) => x.date === r.date)!;
    assert.equal(p.botPct, r.botCumPct);
    assert.equal(p.spyPct, r.spyCumPct);
  }
  assert.deepEqual(s.points[0], { date: day(0), botPct: 0, spyPct: 0, botTaxPct: 0, spyTaxPct: 0, botAfterTaxPct: 0, spyAfterTaxPct: 0, gapPp: 0, spyTerm: "short" });
  assert.equal(s.from, day(0));
  assert.equal(s.to, day(4));
  assert.equal(s.equityStart, 100_000);
  assert.equal(s.rates.shortTerm, 0.24);
});

test("fewer than two session points: no series (never a line of zeros)", () => {
  assert.equal(afterTaxSeries([], bars, { events: [], fills: [] }), null);
  assert.equal(afterTaxSeries([{ date: day(0), value: 100_000 }], bars, { events: [], fills: [] }), null);
  assert.equal(afterTaxSeries(curve, [], { events: [], fills: [] }), null);
});

test("SPY leg: its gain since the start taxed once at the short-term rate, a loss owes nothing", () => {
  const s = afterTaxSeries(curve, bars, { events: [], fills: [] })!;
  const d1 = s.points[1]; // SPY 505 / 500 = +1 %
  close(d1.spyPct, 1);
  close(d1.spyTaxPct, 0.24);
  close(d1.spyAfterTaxPct, 0.76);
  const d3 = s.points[3]; // 495 / 500 = −1 %
  close(d3.spyTaxPct, 0);
  close(d3.spyAfterTaxPct, -1);
  assert.equal(d3.spyTerm, "short");
});

test("account leg: the tax on the trades closed BY that day, point-in-time — a later close never changes an earlier point", () => {
  const events = [ev("A", 1, 1000, 0), ev("B", 3, 500, 2)];
  const s = afterTaxSeries(curve, bars, { events, fills: [] })!;
  close(s.points[1].botTaxPct, 0.24); // 1000 × 24 % of 100,000
  close(s.points[2].botTaxPct, 0.24); // nothing new closed on day 2
  close(s.points[3].botTaxPct, 0.36); // (1000 + 500) × 24 %
  close(s.points[1].botAfterTaxPct, 1 - 0.24);
  close(s.points[3].gapPp, s.points[3].botAfterTaxPct - s.points[3].spyAfterTaxPct);
  // appending a later close (day 4) leaves days 0–3 exactly as they were
  const more = afterTaxSeries(curve, bars, { events: [...events, ev("C", 4, 2000, 3)], fills: [] })!;
  assert.deepEqual(more.points.slice(0, 4), s.points.slice(0, 4));
  close(more.points[4].botTaxPct, 0.84); // (1000 + 500 + 2000) × 24 %
});

test("a wash sale counts from the day its replacement fills, never before (no lookahead into later fills)", () => {
  // a −1000 loss on day 1 (gain of 1500 on day 1 too) re-bought on day 3: from day 3 the loss is disallowed
  const events = [ev("W", 1, -1000, 0), ev("G", 1, 1500, 0)];
  const s = afterTaxSeries(curve, bars, { events, fills: [buy("W", 3)] })!;
  close(s.points[1].botTaxPct, 0.12); // (1500 − 1000) × 24 %
  close(s.points[2].botTaxPct, 0.12);
  close(s.points[3].botTaxPct, 0.36); // the replacement on day 3 holds the loss back: 1500 × 24 %
  close(s.points[4].botTaxPct, 0.36);
});

test("each point IS afterTaxBenchmark for start → that day without open positions (one method, the tile's)", () => {
  const events = [ev("A", 1, 1000, 0), ev("W", 2, -300, 1), ev("B", 4, 800, 2)];
  const fills = [buy("W", 3)];
  const s = afterTaxSeries(curve, bars, { events, fills, rates: { shortTerm: 0.32, state: 0.05 } })!;
  assert.equal(s.rates.shortTerm, 0.32);
  const last = s.points[4];
  const want = afterTaxBenchmark(
    { from: day(0), to: day(4), days: 4, accountPct: 3, spyPct: 4, gapPp: -1 },
    taxLedger(events, fills, { windowStartMs: T0, rates: { shortTerm: 0.32, state: 0.05 } }),
    { equityStart: 100_000, openPnl: null },
  )!;
  close(last.botPct, 3);
  close(last.spyPct, 4);
  close(last.botTaxPct, want.account.taxPct);
  close(last.botAfterTaxPct, want.account.afterTaxPct);
  close(last.spyTaxPct, want.spy.taxPct);
  close(last.spyAfterTaxPct, want.spy.afterTaxPct);
  close(last.gapPp, want.gapPp);
  // a carried-forward realized loss owes nothing: the account's after-tax line is its pre-tax line
  const loss = afterTaxSeries(curve, bars, { events: [ev("L", 1, -5000, 0)], fills: [] })!;
  for (const p of loss.points) close(p.botTaxPct, 0);
});

test("past the first year SPY's sale is long-term", () => {
  const long: CurvePoint[] = [{ date: day(0), value: 100_000 }, { date: day(400), value: 110_000 }];
  const lb: DailyClose[] = [{ date: day(0), close: 500 }, { date: day(400), close: 600 }];
  const s = afterTaxSeries(long, lb, { events: [], fills: [] })!;
  assert.equal(s.points[1].spyTerm, "long");
  close(s.points[1].spyTaxPct, 3); // 20 % × 15 %
});

test("the notes state the framing, the point-in-time rule and what the line leaves out (open positions)", () => {
  const s = afterTaxSeries(curve, bars, { events: [], fills: [] })!;
  const text = s.notes.join(" ");
  assert.match(text, /both sold at that day's close/);
  assert.match(text, /point-in-time/);
  assert.match(text, /open positions/);
  assert.match(text, /paper account pays no tax/);
});

test("source ratchet: every point is afterTaxBenchmark over a point-in-time taxLedger — no second tax formula", () => {
  const src = codeOnly(fs.readFileSync(path.join(here, "afterTaxSeries.ts"), "utf8"));
  assert.match(src, /import \{ afterTaxBenchmark \} from "\.\/afterTaxBenchmark"/);
  assert.match(src, /taxLedger\(events\.slice\(0, ie\), fills\.slice\(0, ifl\), \{ windowStartMs: startMs, rates \}\)/);
  assert.match(src, /openPnl: null/, "no point taxes open positions (their past P&L is not recorded)");
  assert.doesNotMatch(src, /\* \(?rates\.shortTerm|shortTerm \*/, "the rates are applied inside afterTaxBenchmark / taxLedger only");
});

test("wiring ratchet: /api/bot/scoreboard carries the series from the sleeve ledger's tax inputs, null with its reason before that cache loads", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.match(bot, /import \{ afterTaxSeries \} from "\.\/afterTaxSeries"/);
  const start = bot.indexOf("async function scoreboardBody(");
  assert.ok(start > 0, "the scoreboard body exists");
  const body = codeOnly(bot.slice(start, bot.indexOf('app.get("/api/bot/scoreboard"', start)));
  assert.match(body, /const afterTax = sleeveCache \? afterTaxSeries\(equityCurve, spy\.bars, \{ events: sleeveCache\.taxInputs\.events, fills: sleeveCache\.taxInputs\.fills \}\) : null;/);
  assert.match(body, /sources, cashFlows, afterTax, afterTaxReason \}/, "the computed body carries the series and its reason");
  assert.match(body, /"sleeve ledger not loaded yet; the next poll fills it"/);
  // every early answer carries the field too, so the page never reads `undefined` as "server predates it"
  assert.equal((body.match(/afterTax: null, afterTaxReason:/g) ?? []).length, 2);
});
