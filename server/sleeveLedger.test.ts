// sleeveLedger — P&L by sleeve from Alpaca's fill ledger ([RULE-REVIEW]
// 2026-10-20). Run: npx tsx --test server/sleeveLedger.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { sleeveLedger, realizedEvents, sleeveOf, occRoot, SLEEVE_UNDERLYING_CAP } from "./sleeveLedger";
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

test("the options sleeve by underlying: losers first, the condor's −$66 under TSLA, open P&L per root, capped", () => {
  const L = sleeveLedger(FILLS, POSITIONS, FLOOR);
  assert.equal(L.optionsUnderlyings, 2);
  assert.deepEqual(L.optionsByUnderlying.map((u) => u.root), ["TSLA", "BAC"], "losers first");
  const tsla = L.optionsByUnderlying[0];
  assert.equal(tsla.realized, -66);
  assert.equal(tsla.fills, 8);
  assert.deepEqual(tsla.byMonth, { "2026-10": -66 });
  assert.equal(tsla.open, 0);
  const bac = L.optionsByUnderlying[1];
  assert.equal(bac.realized, 0, "still open: nothing realized");
  assert.equal(bac.open, -17);
  assert.equal(bac.openPositions, 1);
  assert.equal(occRoot("TSLA261030C00470000"), "TSLA");
  assert.equal(occRoot("BRK.B"), "BRK.B");
  assert.equal(occRoot("IONQ1261120P00037000"), "IONQ1", "adjusted roots keep their digit");
  // cap: 45 distinct roots -> 40 reported, the count stays 45
  const many = Array.from({ length: 45 }, (_, i) => {
    const root = `R${String(i).padStart(2, "0")}`;
    const occ = `${root}261120P00050000`;
    return [fill(occ, -1, 1, 1.0, "2026-09-01T14:00:00Z"), fill(occ, 1, 1, 1.0 + i / 100, "2026-09-02T14:00:00Z")];
  }).flat();
  const C = sleeveLedger(many, [], FLOOR);
  assert.equal(C.optionsUnderlyings, 45);
  assert.equal(C.optionsByUnderlying.length, SLEEVE_UNDERLYING_CAP);
  assert.equal(C.optionsByUnderlying[0].root, "R44", "the biggest loser first");
});

test("window start: a lot held at the window start is priced at its REAL pre-window basis, and only in-window fills, events and months are counted", () => {
  // QQQ: 20 bought in April at 500 (pre-window), 10 sold in August at 590 (in window), 10 still held
  // with Alpaca's CURRENT avg_entry at 560 (set by later churn in the live book — the old approximation's basis).
  // AMD: a full round trip in April (pre-window): must not appear in the window at all.
  const fills: FillRow[] = [
    fill("QQQ", 1, 20, 500, "2026-04-01T14:00:00Z"), fill("QQQ", -1, 10, 590, "2026-08-20T14:00:00Z"),
    fill("AMD", 1, 20, 150, "2026-04-05T14:00:00Z"), fill("AMD", -1, 20, 140, "2026-04-20T14:00:00Z"),
    fill("TSLA261030C00470000", -1, 1, 6.10, "2026-10-19T13:36:13Z"), fill("TSLA261030C00470000", 1, 1, 6.45, "2026-10-19T13:40:38Z"),
  ];
  const positions: PositionRow[] = [pos("QQQ", 10, 560, 300)];
  const W = T("2026-05-05T00:00:00Z");
  const L = sleeveLedger(fills, positions, FLOOR, W);
  const by = Object.fromEntries(L.rows.map((r) => [r.sleeve, r]));
  assert.equal(by.floor.realized, 900, "10 × (590 − 500): the April basis, not the current 560 avg_entry");
  assert.deepEqual(by.floor.byMonth, { "2026-08": 900 }, "no April month in the window");
  assert.equal(by.floor.fills, 1, "the April buy is basis only");
  assert.equal(by.stocks.realized, 0, "AMD's pre-window round trip is outside the window");
  assert.equal(by.stocks.fills, 0);
  assert.deepEqual(by.stocks.byMonth, {});
  assert.equal(by.options.realized, -35);
  assert.equal(L.realizedTotal, 865);
  assert.equal(L.preWindowFills, 3);
  assert.equal(L.windowStart, "2026-05-05T00:00:00.000Z");
  assert.equal(L.firstFill, "2026-04-01T14:00:00.000Z", "the history's first fill, not the window's");
  assert.equal(L.unrealizedTotal, 300);
  assert.deepEqual(L.optionsByUnderlying.map((u) => [u.root, u.realized, u.fills]), [["TSLA", -35, 2]]);
  // the OLD reading of the same window (only in-window fills known) prices the QQQ lot at the current avg_entry
  const old = sleeveLedger(fills.filter((f) => f.t >= W), positions, FLOOR);
  assert.equal(Object.fromEntries(old.rows.map((r) => [r.sleeve, r])).floor.realized, 300, "10 × (590 − 560): the approximation this replaces");
  assert.equal(old.windowStart, null);
  assert.equal(old.preWindowFills, 0);
  // no window: byte-identical to before (every fill is in the window)
  const none = sleeveLedger(fills, positions, FLOOR);
  assert.equal(Object.fromEntries(none.rows.map((r) => [r.sleeve, r])).stocks.realized, -200);
  assert.equal(none.realizedTotal, 665);
});
