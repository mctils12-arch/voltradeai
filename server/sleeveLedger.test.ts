// sleeveLedger — P&L by sleeve from Alpaca's fill ledger ([RULE-REVIEW]
// 2026-10-20). Run: npx tsx --test server/sleeveLedger.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { sleeveLedger, realizedEvents, sleeveOf } from "./sleeveLedger";
import { realizedPnlFromFills, type FillRow, type PositionRow } from "./killSwitchAutoResume";

const FLOOR = new Set(["QQQ", "SVXY", "SPY", "SMH", "KWEB", "VXUS", "GLD"]);
const T = (iso: string) => Date.parse(iso);
const fill = (symbol: string, dir: 1 | -1, qty: number, price: number, iso: string): FillRow => ({ symbol, dir, qty, price, t: T(iso) });
const pos = (symbol: string, qty: number, avgEntry: number | null, unrealizedPl: number | null): PositionRow =>
  ({ symbol, qty, avgEntry, unrealizedPl, option: symbol.length > 8 });

// one realistic month: a floor round trip, a stock loser, the live TSLA condor
// (opened for a 4.59 credit, closed for 5.25 — its four legs), a CSP still held
const FILLS: FillRow[] = [
  fill("QQQ", 1, 10, 600, "2026-08-03T14:00:00Z"), fill("QQQ", -1, 10, 590, "2026-08-20T14:00:00Z"),
  fill("AMD", 1, 20, 150, "2026-08-05T14:00:00Z"), fill("AMD", -1, 20, 140, "2026-09-02T14:00:00Z"),
  fill("TSLA261030C00470000", -1, 1, 6.10, "2026-10-19T13:36:13Z"), fill("TSLA261030C00485000", 1, 1, 3.20, "2026-10-19T13:36:13Z"),
  fill("TSLA261030P00420000", -1, 1, 5.40, "2026-10-19T13:36:13Z"), fill("TSLA261030P00405000", 1, 1, 3.71, "2026-10-19T13:36:13Z"),
  fill("TSLA261030C00470000", 1, 1, 6.45, "2026-10-19T13:40:38Z"), fill("TSLA261030C00485000", -1, 1, 3.05, "2026-10-19T13:40:38Z"),
  fill("TSLA261030P00420000", 1, 1, 5.70, "2026-10-19T13:40:38Z"), fill("TSLA261030P00405000", -1, 1, 3.85, "2026-10-19T13:40:38Z"),
  fill("BAC261120P00048000", -1, 1, 0.39, "2026-10-19T13:48:32Z"),
];
const POSITIONS: PositionRow[] = [pos("BAC261120P00048000", -1, 0.39, -17), pos("SMH", 30, 240, -55.5), pos("NVDA", 5, 180, 12.25)];

test("sleeves: OCC symbols are options, the floor/leg ETFs are floor, the rest are stocks", () => {
  assert.equal(sleeveOf("TSLA261030C00470000", FLOOR), "options");
  assert.equal(sleeveOf("kweb", FLOOR), "floor");
  assert.equal(sleeveOf("AMD", FLOOR), "stocks");
  assert.equal(sleeveOf("SVXY", FLOOR), "floor", "the VRP leg is in bot.ts's passive set");
});

test("realized P&L per sleeve, per month, and the condor's own −$66", () => {
  const L = sleeveLedger(FILLS, POSITIONS, FLOOR);
  const by = Object.fromEntries(L.rows.map((r) => [r.sleeve, r]));
  assert.equal(by.floor.realized, -100);
  assert.deepEqual(by.floor.byMonth, { "2026-08": -100 });
  assert.equal(by.stocks.realized, -200);
  assert.deepEqual(by.stocks.byMonth, { "2026-09": -200 });
  // credit 6.10-3.20+5.40-3.71 = 4.59; debit 6.45-3.05+5.70-3.85 = 5.25 → −0.66 × 100
  assert.equal(by.options.realized, -66);
  assert.equal(by.options.fills, 9);
  assert.equal(by.options.symbols, 5);
  assert.equal(L.realizedTotal, -366);
  assert.equal(by.options.unrealized, -17);
  assert.equal(by.floor.unrealized, -55.5);
  assert.equal(by.stocks.unrealized, 12.25);
  assert.equal(L.unrealizedTotal, -60.25);
  assert.equal(L.firstFill, "2026-08-03T14:00:00.000Z");
});

test("same ruler as the kill switch: totals equal realizedPnlFromFills on the same inputs", () => {
  const cases: Array<[FillRow[], PositionRow[]]> = [
    [FILLS, POSITIONS],
    // a position opened before the window, partly closed inside it, still held
    [[fill("XOM", -1, 5, 120, "2026-09-10T14:00:00Z"), fill("XOM", 1, 2, 110, "2026-09-11T14:00:00Z")], [pos("XOM", 7, 100, 70)]],
    // a flip through zero and an unpriced close (pre-window, no longer held)
    [[fill("IWM", 1, 3, 200, "2026-09-01T14:00:00Z"), fill("IWM", -1, 5, 210, "2026-09-02T14:00:00Z"), fill("IWM", 1, 2, 205, "2026-09-03T14:00:00Z"),
      fill("TLT", -1, 4, 90, "2026-09-05T14:00:00Z")], []],
  ];
  for (const [f, p] of cases) {
    const sorted = [...f].sort((a, b) => a.t - b.t);
    const ref = realizedPnlFromFills(sorted, p);
    const { events, unpriced } = realizedEvents(sorted, p);
    assert.equal(Math.round(events.reduce((s, e) => s + e.pnl, 0) * 100) / 100, ref.realized);
    assert.equal([...unpriced.values()].reduce((s, q) => s + q, 0), ref.unpricedCloses);
    assert.equal(sleeveLedger(f, p, FLOOR).realizedTotal, ref.realized);
  }
});

test("closes against an unknown basis are counted, never guessed; a missing unrealized is null, not zero", () => {
  const L = sleeveLedger([fill("TLT", -1, 4, 90, "2026-09-05T14:00:00Z")], [pos("SPY", 1, 700, null)], FLOOR);
  const by = Object.fromEntries(L.rows.map((r) => [r.sleeve, r]));
  assert.equal(by.stocks.unpricedCloses, 4);
  assert.equal(by.stocks.realized, 0);
  assert.equal(by.floor.unrealized, null);
  assert.equal(L.unrealizedTotal, null);
  assert.deepEqual(sleeveLedger([], [], FLOOR).rows.map((r) => r.realized), [0, 0, 0]);
});
