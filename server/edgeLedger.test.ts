// edgeLedger — expectancy per closed lot from the broker's fills ([RULE-REVIEW]
// 2026-10-29). Run: npx tsx --test server/edgeLedger.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { edgeLedger, brokerBlock, BROKER_SOURCE, EDGE_GROUPS, type EdgeLedger } from "./edgeLedger";
import { realizedEvents, SLEEVE_CACHE_MS } from "./sleeveLedger";
import type { FillRow } from "./killSwitchAutoResume";

const FLOOR = new Set(["QQQ", "SVXY", "SPY", "SMH", "KWEB", "VXUS", "GLD"]);
const T = (iso: string) => Date.parse(iso);
const r2 = (x: number) => Math.round(x * 100) / 100;
const fill = (symbol: string, dir: 1 | -1, qty: number, price: number, iso: string): FillRow => ({ symbol, dir, qty, price, t: T(iso) });

// synthetic fills: a floor round trip, a stock loser closed in two parts, a sold
// put bought back cheaper (credit kept), a sold put bought back dearer (loses
// more than its basis), a bought call sold flat, a close with no basis in the history
const FILLS: FillRow[] = [
  fill("QQQ", 1, 10, 600, "2026-08-03T14:00:00Z"), fill("QQQ", -1, 10, 612, "2026-08-20T14:00:00Z"),                         // floor +120 on 6,000 = +2 %
  fill("AMD", 1, 20, 150, "2026-08-05T14:00:00Z"), fill("AMD", -1, 5, 140, "2026-09-02T14:00:00Z"), fill("AMD", -1, 15, 135, "2026-09-03T14:00:00Z"), // −50 on 750 (−6.67 %), −225 on 2,250 (−10 %)
  fill("AMD261120P00120000", -1, 2, 2.00, "2026-09-01T14:00:00Z"), fill("AMD261120P00120000", 1, 2, 1.00, "2026-09-10T14:00:00Z"),   // credit 400, bought back for 200: +200 = +50 %
  fill("XOM261120P00050000", -1, 1, 1.00, "2026-09-05T14:00:00Z"), fill("XOM261120P00050000", 1, 1, 2.50, "2026-09-06T14:00:00Z"),   // credit 100, bought back for 250: −150 = −150 %
  fill("GE261120C00150000", 1, 1, 3.00, "2026-09-15T14:00:00Z"), fill("GE261120C00150000", -1, 1, 3.00, "2026-09-16T14:00:00Z"),     // flat: a loss of 0
  fill("TLT", -1, 4, 90, "2026-09-05T14:00:00Z"),                                                                                 // closes a lot older than the history
];

test("per sleeve and for the active sleeves: n, win rate, average win/loss as pnl/basis, expectancy, dollars, dates, unpriced", () => {
  const { events, unpriced } = realizedEvents([...FILLS].sort((a, b) => a.t - b.t), []);
  const L = edgeLedger(events, unpriced, FLOOR);
  assert.deepEqual(EDGE_GROUPS, ["active", "options", "stocks", "floor"]);
  // floor: one lot, +2 %
  assert.equal(L.floor.n, 1);
  assert.equal(L.floor.winRate, 100);
  assert.equal(L.floor.avgWinPct, 2);
  assert.equal(L.floor.avgLossPct, null, "no losing lot: null, never 0");
  assert.equal(L.floor.expectancyPct, 2);
  assert.equal(L.floor.expectancyUsd, 120);
  assert.equal(L.floor.realizedUsd, 120);
  assert.equal(L.floor.profitFactor, null, "no gross loss: null, never Infinity");
  // stocks: the two partial closes are two lots (−6.67 %, −10 %); TLT's close is unpriced, not a lot
  assert.equal(L.stocks.n, 2);
  assert.equal(L.stocks.wins, 0);
  assert.equal(L.stocks.losses, 2);
  assert.equal(L.stocks.winRate, 0);
  assert.equal(L.stocks.avgWinPct, null);
  assert.equal(L.stocks.avgLossPct, -8.33);
  assert.equal(L.stocks.expectancyPct, -8.33);
  assert.equal(L.stocks.expectancyUsd, -137.5);
  assert.equal(L.stocks.realizedUsd, -275);
  assert.equal(L.stocks.unpriced, 4, "TLT: 4 shares closed against no basis — counted, never guessed");
  assert.equal(L.stocks.firstCloseAt, "2026-09-02T14:00:00.000Z");
  assert.equal(L.stocks.lastCloseAt, "2026-09-03T14:00:00.000Z");
  // options: +50 % (credit kept), −150 % (credit lot blown), 0 % (flat = a loss of 0)
  assert.equal(L.options.n, 3);
  assert.equal(L.options.wins, 1);
  assert.equal(L.options.losses, 2);
  assert.equal(L.options.winRate, 33.33);
  assert.equal(L.options.avgWinPct, 50);
  assert.equal(L.options.avgLossPct, -75);
  assert.equal(L.options.expectancyPct, -33.33);
  assert.equal(L.options.expectancyUsd, 16.67);
  assert.equal(L.options.realizedUsd, 50);
  assert.equal(L.options.profitFactor, 1.33, "200 / 150");
  // active = options + stocks, the floor left out
  assert.equal(L.active.n, 5);
  assert.equal(L.active.wins, 1);
  assert.equal(L.active.winRate, 20);
  assert.equal(L.active.realizedUsd, -225);
  assert.equal(L.active.expectancyUsd, -45);
  assert.equal(L.active.expectancyPct, r2((50 - 150 + 0 - 6.67 - 10) / 5), "mean of the five returns");
  assert.equal(L.active.unpriced, 4);
  assert.equal(L.active.lastCloseAt, "2026-09-16T14:00:00.000Z");
  assert.equal(L.windowStart, null);
  // the identity the tile relies on: winRate × avgWin + lossRate × avgLoss = the mean return (to the rounding of the parts)
  const a = L.active;
  assert.ok(Math.abs((a.winRate! / 100) * a.avgWinPct! + (1 - a.winRate! / 100) * a.avgLossPct! - a.expectancyPct!) <= 0.02);
});

test("the window counts only lots closed from its start; unpriced closes stay as the history reports them; empty inputs are nulls, not zeros", () => {
  const { events, unpriced } = realizedEvents([...FILLS].sort((a, b) => a.t - b.t), []);
  const L = edgeLedger(events, unpriced, FLOOR, T("2026-09-03T00:00:00Z"));
  assert.equal(L.windowStart, "2026-09-03T00:00:00.000Z");
  assert.equal(L.stocks.n, 1, "the 09-02 partial close is before the window");
  assert.equal(L.stocks.expectancyPct, -10);
  assert.equal(L.floor.n, 0);
  assert.equal(L.floor.expectancyPct, null);
  assert.equal(L.floor.winRate, null);
  assert.equal(L.floor.lastCloseAt, null);
  assert.equal(L.stocks.unpriced, 4);
  const empty = edgeLedger([], new Map(), FLOOR);
  for (const g of EDGE_GROUPS) {
    assert.equal(empty[g].n, 0);
    assert.equal(empty[g].expectancyPct, null);
    assert.equal(empty[g].realizedUsd, 0);
    assert.equal(empty[g].unpriced, 0);
  }
});

test("a credit lot's sign convention: bought back cheaper is a win worth the share of the credit kept; dearer can lose more than 100 %", () => {
  const win = realizedEvents([fill("AMD261120P00120000", -1, 2, 2.00, "2026-09-01T14:00:00Z"), fill("AMD261120P00120000", 1, 2, 1.00, "2026-09-10T14:00:00Z")], []);
  const W = edgeLedger(win.events, win.unpriced, FLOOR);
  assert.deepEqual([W.options.n, W.options.winRate, W.options.avgWinPct, W.options.expectancyUsd], [1, 100, 50, 200]);
  const loss = realizedEvents([fill("XOM261120P00050000", -1, 1, 1.00, "2026-09-05T14:00:00Z"), fill("XOM261120P00050000", 1, 1, 2.50, "2026-09-06T14:00:00Z")], []);
  const X = edgeLedger(loss.events, loss.unpriced, FLOOR);
  assert.deepEqual([X.options.n, X.options.winRate, X.options.avgLossPct, X.options.expectancyUsd], [1, 0, -150, -150]);
});

test("brokerBlock: the cached ledger with its age and source, or null with the reason while the cache is cold or failed", () => {
  const edge: EdgeLedger = edgeLedger([], new Map(), FLOOR);
  const at = T("2026-10-29T14:00:00Z");
  const warm = brokerBlock({ at, edge }, at + 90_000, null, SLEEVE_CACHE_MS);
  assert.equal(warm.brokerReason, null);
  assert.equal(warm.broker!.asOf, "2026-10-29T14:00:00.000Z");
  assert.equal(warm.broker!.source, BROKER_SOURCE);
  assert.equal(warm.broker!.cached_s, 90);
  assert.equal(warm.broker!.stale, false);
  assert.equal(warm.broker!.lastError, null);
  assert.equal(warm.broker!.active.n, 0);
  const old = brokerBlock({ at, edge }, at + SLEEVE_CACHE_MS + 1000, "boom", SLEEVE_CACHE_MS);
  assert.equal(old.broker!.stale, true);
  assert.equal(old.broker!.lastError, "boom", "a stale cache still serves, and says why it is stale");
  const cold = brokerBlock(null, at, null, SLEEVE_CACHE_MS);
  assert.equal(cold.broker, null);
  assert.match(cold.brokerReason!, /not loaded yet/);
  const failed = brokerBlock(null, at, "activities unreadable", SLEEVE_CACHE_MS);
  assert.equal(failed.broker, null);
  assert.match(failed.brokerReason!, /^sleeve ledger failed: activities unreadable/);
});

// ── the route wiring (source pins, as peakEquityRpc.test.ts pins its routes) ──
const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
function slice(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

test("/api/bot/performance serves `broker` from the ONE sleeve cache and annotates the bot-log number with what it excludes", () => {
  const route = slice('app.get("/api/bot/performance"', 'app.get("/api/bot/benchmark"');
  assert.ok(route.includes("brokerBlock(sleeveCache ? { at: sleeveCache.at, edge: sleeveCache.edge } : null, Date.now(), sleeveError, SLEEVE_CACHE_MS)"), "the broker block reads the sleeve cache the sleeves route fills");
  assert.ok(!route.includes("edgeLedger("), "never computed in the route: the cache computes it once per refresh");
  assert.ok(!/refreshSleeves\(\)/.test(route), "the route never fetches; a cold cache is warmed through sleevesBody for the next poll");
  assert.ok(/if \(!sleeveCache\) void sleevesBody\(\)/.test(route));
  assert.ok(/res\.json\(\{\s*\.\.\.perf,\s*\.\.\.brokerEdge,/.test(route), "broker + brokerReason ride beside every existing field");
  // the inline Python keeps every existing field and adds the annotations
  for (const field of ["'totalTrades': len(feedback)", "'winRate': round(win_rate, 1)", "'avgGain': round(avg_win, 2)", "'avgLoss': round(avg_loss, 2)", "'profitFactor':", "'recentTrades': feedback[-20:][::-1]", "'slippageBasis': fills_slippage_basis(feedback)"]) {
    assert.ok(route.includes(field), `existing field kept: ${field}`);
  }
  assert.ok(route.includes("'lastClosedAt': last_closed_at"));
  assert.ok(route.includes("'excluded': {'orphanExits': orphan_exits, 'openEntries': open_entries}"));
  assert.ok(route.includes("'basis': 'trigger price, not fill'"));
});

test("refreshSleeves computes the edge ledger from the same walk the tax inputs use, and stores it on the cache", () => {
  const refresh = slice("async function refreshSleeves()", "async function sleevesBody(");
  assert.ok(refresh.includes("const walked = realizedEvents(fills.rows, positions);"));
  assert.ok(refresh.includes("edge: edgeLedger(walked.events, walked.unpriced, FLOOR_AND_LEG_TICKERS, windowStartMs)"));
  assert.ok(refresh.includes("const taxInputs = { events: walked.events, fills: fills.rows, windowStartMs };"));
  assert.equal((bot.match(/edgeLedger\(/g) || []).length, 1, "one computation site in bot.ts");
});
