// expectancy by sleeve ([PRODUCT] 2026-11-03, item B2) — shared/expectancy.ts
// against the ruler it decomposes, server/edgeLedger.ts. Run:
// npx tsx --test server/expectancyBySleeve.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { expectancyBySleeve, lotSleeve, sleeveStats, VRP_LEG_TICKERS, SLEEVE_IDS, LOT_GROUPS, type SleeveStats } from "../shared/expectancy";
import { edgeLedger, brokerBlock, type EdgeRow } from "./edgeLedger";
import { realizedEvents, sleeveOf, SLEEVE_CACHE_MS, type RealizedEvent } from "./sleeveLedger";
import { isOptionPosition } from "./reconstructPnl";
import type { FillRow } from "./killSwitchAutoResume";

// bot.ts FLOOR_AND_LEG_TICKERS — the passive set the sleeve ledger treats as the floor (SVXY inside it)
const FLOOR = new Set(["QQQ", "SVXY", "SPY", "SMH", "KWEB", "VXUS", "GLD"]);
const T = (iso: string) => Date.parse(iso);
const r2 = (x: number) => Math.round(x * 100) / 100;
const fill = (symbol: string, dir: 1 | -1, qty: number, price: number, iso: string): FillRow => ({ symbol, dir, qty, price, t: T(iso) });

// the edge ledger's own synthetic book (edgeLedger.test.ts) plus an SVXY round trip — the VRP leg
const FILLS: FillRow[] = [
  fill("QQQ", 1, 10, 600, "2026-08-03T14:00:00Z"), fill("QQQ", -1, 10, 612, "2026-08-20T14:00:00Z"),                         // floor +120 on 6,000 = +2 %
  fill("AMD", 1, 20, 150, "2026-08-05T14:00:00Z"), fill("AMD", -1, 5, 140, "2026-09-02T14:00:00Z"), fill("AMD", -1, 15, 135, "2026-09-03T14:00:00Z"), // −6.67 %, −10 %
  fill("AMD261120P00120000", -1, 2, 2.00, "2026-09-01T14:00:00Z"), fill("AMD261120P00120000", 1, 2, 1.00, "2026-09-10T14:00:00Z"),   // credit kept: +50 %
  fill("XOM261120P00050000", -1, 1, 1.00, "2026-09-05T14:00:00Z"), fill("XOM261120P00050000", 1, 1, 2.50, "2026-09-06T14:00:00Z"),   // credit blown: −150 %
  fill("GE261120C00150000", 1, 1, 3.00, "2026-09-15T14:00:00Z"), fill("GE261120C00150000", -1, 1, 3.00, "2026-09-16T14:00:00Z"),     // flat: a loss of 0
  fill("TLT", -1, 4, 90, "2026-09-05T14:00:00Z"),                                                                                 // closes a lot older than the history
  fill("SVXY", 1, 50, 40, "2026-09-20T14:00:00Z"), fill("SVXY", -1, 50, 38, "2026-09-25T14:00:00Z"),                            // VRP leg: −100 on 2,000 = −5 %
];

const SHARED_KEYS = ["n", "wins", "losses", "winRate", "avgWinPct", "avgLossPct", "expectancyPct", "realizedUsd", "lastCloseAt", "unpriced"] as const;
function sameRow(a: SleeveStats, b: EdgeRow, label: string) {
  for (const k of SHARED_KEYS) assert.deepEqual(a[k], b[k], `${label}.${k}: ${String(a[k])} vs edge ledger ${String(b[k])}`);
}

test("IDENTITY: active (options + stocks) equals the edge ledger's headline row field for field; options and stocks equal their rows; floor + vrp partition the ledger's floor", () => {
  const { events, unpriced } = realizedEvents([...FILLS].sort((a, b) => a.t - b.t), []);
  const edge = edgeLedger(events, unpriced, FLOOR);
  const x = edge.bySleeve;
  sameRow(x.active, edge.active, "active");
  sameRow(x.options, edge.options, "options");
  sameRow(x.stocks, edge.stocks, "stocks");
  // the VRP leg is carved out of the passive set: the ledger's floor row is floor + vrp here
  assert.equal(x.floor.n + x.vrp.n, edge.floor.n);
  assert.equal(r2(x.floor.realizedUsd + x.vrp.realizedUsd), edge.floor.realizedUsd);
  assert.equal(x.vrp.n, 1);
  assert.equal(x.vrp.expectancyPct, -5);
  assert.equal(x.floor.n, 1);
  assert.equal(x.floor.expectancyPct, 2);
  // computed from the same inputs by the standalone function too (the server folds it in; the client never recomputes)
  assert.deepEqual(expectancyBySleeve(events, FLOOR, VRP_LEG_TICKERS, undefined, unpriced), x);
  assert.deepEqual(LOT_GROUPS, ["options", "floor", "stocks", "vrp", "active", "all"]);
  assert.deepEqual(SLEEVE_IDS, ["options", "floor", "stocks", "vrp"]);
});

test("PARTITION: the four sleeves' n sum to all.n and their n-weighted expectancies reproduce all (and active) to rounding; unpriced adds up the same way", () => {
  const { events, unpriced } = realizedEvents([...FILLS].sort((a, b) => a.t - b.t), []);
  const x = expectancyBySleeve(events, FLOOR, VRP_LEG_TICKERS, undefined, unpriced);
  const parts = SLEEVE_IDS.map((s) => x[s]);
  assert.equal(parts.reduce((s, r) => s + r.n, 0), x.all.n);
  assert.equal(x.options.n + x.stocks.n, x.active.n);
  assert.equal(parts.reduce((s, r) => s + r.unpriced, 0), x.all.unpriced);
  assert.equal(x.options.unpriced + x.stocks.unpriced, x.active.unpriced);
  const weighted = (rows: SleeveStats[]) => rows.reduce((s, r) => s + r.n * (r.expectancyPct ?? 0), 0) / rows.reduce((s, r) => s + r.n, 0);
  assert.ok(Math.abs(weighted(parts) - x.all.expectancyPct!) <= 0.02, `all: ${weighted(parts)} vs ${x.all.expectancyPct}`);
  assert.ok(Math.abs(weighted([x.options, x.stocks]) - x.active.expectancyPct!) <= 0.02);
  assert.equal(r2(parts.reduce((s, r) => s + r.realizedUsd, 0)), x.all.realizedUsd);
  assert.equal(x.unclassified, 0);
  // all = 7 priced lots: +2, −6.67, −10, +50, −150, 0, −5
  assert.equal(x.all.n, 7);
  assert.equal(x.all.expectancyPct, r2((2 - 6.67 - 10 + 50 - 150 + 0 - 5) / 7));
  assert.equal(x.all.lastCloseAt, "2026-09-25T14:00:00.000Z", "the VRP close is the newest");
});

test("CLASSIFICATION: the sleeve ledger's rule (options = symbol longer than 8, isOptionPosition) with the VRP leg carved out; case-insensitive sets; an empty symbol is unclassified, never guessed", () => {
  for (const s of ["QQQ", "SPY", "GLD", "KWEB", "AMD", "TLT", "X", "BRK.B", "GOOGL261127P00300000", "GE261120C00150000", "CRWV261120P00055000", "ABCDEFGHI"]) {
    const mine = lotSleeve(s, FLOOR);
    assert.equal(mine === "options", isOptionPosition(s), `${s}: option rule`);
    if (mine !== "vrp") assert.equal(mine, sleeveOf(s, FLOOR), `${s}: the sleeve ledger's sleeve`);
  }
  assert.equal(lotSleeve("SVXY", FLOOR), "vrp");
  assert.equal(lotSleeve("svxy", FLOOR), "vrp");
  assert.equal(sleeveOf("SVXY", FLOOR), "floor", "the ledger keeps SVXY in its floor; here it is the VRP leg");
  assert.equal(lotSleeve("qqq", FLOOR), "floor");
  assert.equal(lotSleeve("", FLOOR), null);
  assert.equal(lotSleeve("SVXY", FLOOR, new Set()), "floor", "no VRP set: SVXY stays in the passive set");
  assert.deepEqual([...VRP_LEG_TICKERS], ["SVXY"], "bot_engine.py LEG3_VRP_TICKER");
  // an unclassified lot is counted and sits in all, in no sleeve
  const lots: RealizedEvent[] = [
    { symbol: "", t: 1, pnl: 10, basis: 100, openedAt: null, qty: 1, closeDir: -1 },
    { symbol: "AMD", t: 2, pnl: -10, basis: 100, openedAt: null, qty: 1, closeDir: -1 },
  ];
  const x = expectancyBySleeve(lots, FLOOR);
  assert.equal(x.unclassified, 1);
  assert.equal(x.all.n, 2);
  assert.equal(x.stocks.n, 1);
  assert.equal(x.active.n, 1);
  assert.equal(SLEEVE_IDS.reduce((s, k) => s + x[k].n, 0), 1);
});

test("FORMULAS: payoff = avg win ÷ |avg loss|, breakeven = |avg loss| ÷ (avg win + |avg loss|) × 100, and winRate × avgWin + lossRate × avgLoss = expectancy", () => {
  // 2 wins of +30 %, 3 losses of −10 %: WR 40, payoff 3, breakeven 25 %, expectancy (60 − 30) / 5 = +6 %
  const lot = (symbol: string, pnl: number, t: number): RealizedEvent => ({ symbol, t, pnl, basis: 100, openedAt: null, qty: 1, closeDir: -1 });
  const lots = [lot("AMD", 30, 1), lot("NVDA", 30, 2), lot("GE", -10, 3), lot("F", -10, 4), lot("T", -10, 5)];
  const s = sleeveStats(lots);
  assert.deepEqual([s.n, s.wins, s.losses, s.winRate, s.avgWinPct, s.avgLossPct, s.payoffRatio, s.breakevenWinRate, s.expectancyPct, s.realizedUsd], [5, 2, 3, 40, 30, -10, 3, 25, 6, 30]);
  assert.ok(Math.abs((s.winRate! / 100) * s.avgWinPct! + (1 - s.winRate! / 100) * s.avgLossPct! - s.expectancyPct!) <= 0.02);
  // at exactly the breakeven win rate the expectancy is zero: 1 win of +30, 3 losses of −10 → WR 25 = breakeven, expectancy 0
  const be = sleeveStats([lot("AMD", 30, 1), lot("GE", -10, 2), lot("F", -10, 3), lot("T", -10, 4)]);
  assert.equal(be.winRate, be.breakevenWinRate);
  assert.equal(be.expectancyPct, 0);
  // the live 2026-10-15 audit numbers (format.test.ts edgeStats): avg win 9.08, avg loss −5.16 → payoff 1.76, breakeven 36.2 %
  const live = sleeveStats([lot("A", 9.08, 1), lot("B", -5.16, 2)]);
  assert.equal(live.payoffRatio, 1.76);
  assert.equal(live.breakevenWinRate, 36.24);
  assert.equal(live.lastCloseAt, new Date(2).toISOString());
});

test("GUARDS: no lots, all wins, all losses and flat-only closes give nulls with the reason in the shape — never NaN or Infinity", () => {
  const lot = (symbol: string, pnl: number, t: number, basis = 100): RealizedEvent => ({ symbol, t, pnl, basis, openedAt: null, qty: 1, closeDir: -1 });
  const empty = sleeveStats([]);
  assert.deepEqual(empty, { n: 0, wins: 0, losses: 0, winRate: null, avgWinPct: null, avgLossPct: null, payoffRatio: null, breakevenWinRate: null, expectancyPct: null, realizedUsd: 0, lastCloseAt: null, unpriced: 0 });
  const wins = sleeveStats([lot("A", 10, 1), lot("B", 20, 2)]);
  assert.deepEqual([wins.winRate, wins.avgWinPct, wins.avgLossPct, wins.payoffRatio, wins.breakevenWinRate, wins.expectancyPct], [100, 15, null, null, null, 15]);
  const losses = sleeveStats([lot("A", -10, 1), lot("B", -20, 2)]);
  assert.deepEqual([losses.winRate, losses.avgWinPct, losses.avgLossPct, losses.payoffRatio, losses.breakevenWinRate, losses.expectancyPct], [0, null, -15, null, null, -15]);
  // flat closes are losses of 0: a payoff would divide by zero → null; breakeven 0 % (any win pays)
  const flat = sleeveStats([lot("A", 10, 1), lot("B", 0, 2)]);
  assert.deepEqual([flat.winRate, flat.avgWinPct, flat.avgLossPct, flat.payoffRatio, flat.breakevenWinRate, flat.expectancyPct], [50, 10, 0, null, 0, 5]);
  const allFlat = sleeveStats([lot("A", 0, 1)]);
  assert.deepEqual([allFlat.winRate, allFlat.avgWinPct, allFlat.avgLossPct, allFlat.payoffRatio, allFlat.breakevenWinRate, allFlat.expectancyPct], [0, null, 0, null, null, 0]);
  // a zero basis cannot price a return: unpriced by quantity, not divided
  const unpriced = sleeveStats([lot("A", 10, 1, 0), lot("B", 5, 2)], 3);
  assert.equal(unpriced.n, 1);
  assert.equal(unpriced.unpriced, 4);
  for (const s of [empty, wins, losses, flat, allFlat, unpriced]) for (const v of Object.values(s)) if (typeof v === "number") assert.ok(Number.isFinite(v), `finite: ${v}`);
  const x = expectancyBySleeve([], FLOOR);
  for (const g of LOT_GROUPS) { assert.equal(x[g].n, 0); assert.equal(x[g].expectancyPct, null); assert.equal(x[g].payoffRatio, null); }
  assert.equal(x.unclassified, 0);
});

test("WINDOW: only lots closed from the window start count, exactly as the edge ledger windows them; unpriced stays as the history reports it", () => {
  const { events, unpriced } = realizedEvents([...FILLS].sort((a, b) => a.t - b.t), []);
  const from = T("2026-09-03T00:00:00Z");
  const edge = edgeLedger(events, unpriced, FLOOR, from);
  const x = edge.bySleeve;
  sameRow(x.active, edge.active, "active (windowed)");
  sameRow(x.stocks, edge.stocks, "stocks (windowed)");
  assert.equal(x.stocks.n, 1, "the 09-02 partial close is before the window");
  assert.equal(x.stocks.expectancyPct, -10);
  assert.equal(x.floor.n, 0, "QQQ closed 08-20, before the window");
  assert.equal(x.vrp.n, 1);
  assert.equal(x.stocks.unpriced, 4, "TLT's 4 shares, not windowed");
  assert.equal(x.all.unpriced, 4);
});

test("WIRING: the broker block of /api/bot/performance carries bySleeve beside the existing rows (the spread), so the page gets the split from the one sleeve cache", () => {
  const { events, unpriced } = realizedEvents([...FILLS].sort((a, b) => a.t - b.t), []);
  const edge = edgeLedger(events, unpriced, FLOOR);
  const at = T("2026-11-03T14:00:00Z");
  const { broker } = brokerBlock({ at, edge }, at + 1000, null, SLEEVE_CACHE_MS);
  assert.ok(broker);
  assert.deepEqual(broker!.bySleeve, edge.bySleeve);
  assert.equal(broker!.bySleeve.active.expectancyPct, broker!.active.expectancyPct, "the table's total row IS the tile's headline");
  // the existing rows are untouched by the addition (MEASUREMENT INTEGRITY: before = after on identical inputs)
  assert.equal(broker!.active.expectancyPct, r2((50 - 150 + 0 - 6.67 - 10) / 5));
  assert.equal(broker!.active.n, 5);
  assert.equal(broker!.floor.n, 2, "the ledger's floor still holds SVXY");
});
