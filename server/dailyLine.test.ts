// DAILY LINE (2026-11-05, human-directed): "Add a daily line to the
// logs/report: account vs SPY (same-basis), floor exposure (x SPY), option
// P&L split into fills/slippage vs direction, and number of option exits
// blocked by the 9:30–9:45 window." Measurement only.
// Run: npx tsx --test server/dailyLine.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";
import {
  DAILY_LINE_KEEP, OPTION_FILL_LEDGER_FILENAME,
  appendOptionFillLedger, buildDailyLine, dailyLineDue, dailyLineMessage, dailyLinePrecheck, fillsSlippage, floorExposure,
  lastLineDay, loadDailyLines, openingWindowBlocks, optionFillsInWindow, optionLedgerRow, optionPnl, optionWindowStart,
  parseCalendarSession, parseFloorConfig, parseLedger, parsePositionsLite, readOptionFillLedger, realizedOptionPnl,
  sameBasis, saveDailyLine, type DailyLine, type DailyLineInputs, type FloorConfig, type LedgerRow, type OptionContract,
} from "./dailyLine";
import type { RealizedEvent } from "./sleeveLedger";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const tmpDir = () => fs.mkdtempSync(path.join(os.tmpdir(), "daily-line-"));
const ms = (iso: string) => Date.parse(iso);

// ── the gate ────────────────────────────────────────────────────────────────

const NOV5 = { date: "2026-11-05", closeMin: 16 * 60 };

test("gate: due from the broker close + 20 min to ET midnight, once per day, DST-correct", () => {
  assert.equal(dailyLineDue(ms("2026-11-05T21:19:00Z"), NOV5, null).due, false, "16:19 EST: ETF options still print");
  const g = dailyLineDue(ms("2026-11-05T21:20:00Z"), NOV5, null);
  assert.deepEqual([g.due, g.partialOk, g.day], [true, false, "2026-11-05"]);
  assert.equal(dailyLineDue(ms("2026-11-05T22:30:00Z"), NOV5, null).partialOk, true, "from the close + 90 min a partial line is allowed");
  assert.equal(dailyLineDue(ms("2026-11-06T04:59:00Z"), NOV5, null).due, true, "23:59 EST is still the same ET day");
  assert.equal(dailyLineDue(ms("2026-11-06T05:00:00Z"), NOV5, null).due, false, "after ET midnight the day is over (a new ET date)");
  assert.equal(dailyLineDue(ms("2026-11-05T21:20:00Z"), NOV5, "2026-11-05").due, false, "already written");
  assert.equal(dailyLineDue(ms("2026-11-05T21:20:00Z"), null, null).due, false, "no session: the market was not open");
  // summer (EDT): 16:20 ET is 20:20Z
  assert.equal(dailyLineDue(ms("2026-07-06T20:20:00Z"), { date: "2026-07-06", closeMin: 960 }, null).due, true);
  assert.equal(dailyLineDue(ms("2026-07-06T20:19:00Z"), { date: "2026-07-06", closeMin: 960 }, null).due, false);
  // a 13:00 half day: due at 13:20
  assert.equal(dailyLineDue(ms("2026-11-27T18:20:00Z"), { date: "2026-11-27", closeMin: 780 }, null).due, true);
});

test("gate: the precheck spares the calendar call on weekends, before 13:20 ET and once written", () => {
  assert.equal(dailyLinePrecheck(ms("2026-11-07T21:30:00Z"), null).due, false, "Saturday");
  assert.equal(dailyLinePrecheck(ms("2026-11-05T18:19:00Z"), null).due, false, "13:19 EST");
  assert.equal(dailyLinePrecheck(ms("2026-11-05T18:20:00Z"), null).due, true);
  assert.equal(dailyLinePrecheck(ms("2026-11-05T21:30:00Z"), "2026-11-05").due, false);
});

test("gate: the broker calendar's session (holiday = null, half day = 13:00); an unreadable answer throws for a retry", () => {
  assert.deepEqual(parseCalendarSession([{ date: "2026-11-05", open: "09:30", close: "16:00" }], "2026-11-05"), NOV5);
  assert.deepEqual(parseCalendarSession([{ date: "2026-11-27", open: "09:30", close: "13:00" }], "2026-11-27"), { date: "2026-11-27", closeMin: 780 });
  assert.equal(parseCalendarSession([], "2026-11-26"), null, "Thanksgiving: no session");
  assert.equal(parseCalendarSession([{ date: "2026-11-27", close: "13:00" }], "2026-11-26"), null);
  assert.throws(() => parseCalendarSession({ message: "forbidden" }, "2026-11-05"));
  assert.throws(() => parseCalendarSession([{ date: "2026-11-05", close: "4pm" }], "2026-11-05"));
});

test("gate across restarts: the saved line is the claim — the same day is not written twice, the next day is", () => {
  const dir = tmpDir();
  try {
    const nowMs = ms("2026-11-05T21:25:00Z");
    assert.equal(dailyLineDue(nowMs, NOV5, lastLineDay(loadDailyLines(dir))).due, true, "a fresh volume: due");
    assert.ok(saveDailyLine(dir, fullLine()));
    // a new process: nothing in memory, the file says the day is done
    assert.equal(dailyLineDue(nowMs + 3_600_000, NOV5, lastLineDay(loadDailyLines(dir))).due, false);
    const nov6 = { date: "2026-11-06", closeMin: 960 };
    assert.equal(dailyLineDue(ms("2026-11-06T21:25:00Z"), nov6, lastLineDay(loadDailyLines(dir))).due, true, "and the next session is due again");
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ── 1. account vs SPY, same basis ───────────────────────────────────────────

const SPLIT_BARS = [
  { date: "2026-04-30", close: 700 }, { date: "2026-05-06", close: 710 }, { date: "2026-10-05", close: 740 },
  { date: "2026-11-04", close: 750 }, { date: "2026-11-05", close: 753 },
];
// newest first, as bot.ts keeps it: a junk next-day row, tier 3's earlier reading of today, and a pre-SINCE row
const CURVE = [
  { date: "2026-11-06", value: 1 }, { date: "2026-11-05", value: 94000 }, { date: "2026-11-04", value: 95000 },
  { date: "2026-10-05", value: 98000 }, { date: "2026-05-06", value: 100000 }, { date: "2026-05-01", value: 50000 },
];

test("same basis: the account's day vs SPY's price-only day, and since 05-06 / 30d from the curve ending at the same account read", () => {
  const b = sameBasis({ equity: "93445.63", lastEquity: "95205.49", spyBars: SPLIT_BARS, curve: CURVE, day: "2026-11-05", since: "2026-05-06" });
  assert.deepEqual([b.day.account_pct, b.day.spy_pct, b.day.gap_pp], [-1.85, 0.4, -2.25]);
  assert.equal(b.day.spy_bar_date, "2026-11-05");
  assert.deepEqual(b.since, { from: "2026-05-06", to: "2026-11-05", days: 183, accountPct: -6.55, spyPct: 6.06, gapPp: -12.61 },
    "today's end point is the equity just read (93,445.63), not tier 3's 94,000; pre-SINCE and next-day rows ignored");
  assert.deepEqual(b.d30, { from: "2026-10-05", to: "2026-11-05", days: 31, accountPct: -4.65, spyPct: 1.76, gapPp: -6.4 });
  assert.equal(b.windows_reason, null);
  assert.match(b.basis, /PRICE-ONLY/);
});

test("same basis: no SPY bar for today -> no SPY day and no windows (never yesterday's close against today's equity)", () => {
  const b = sameBasis({ equity: 93445.63, lastEquity: 95205.49, spyBars: SPLIT_BARS.slice(0, -1), curve: CURVE, day: "2026-11-05", since: "2026-05-06" });
  assert.equal(b.day.account_pct, -1.85);
  assert.equal(b.day.spy_pct, null);
  assert.equal(b.day.gap_pp, null);
  assert.equal(b.since, null);
  assert.match(String(b.windows_reason), /no SPY bar for today/);
  const bad = sameBasis({ equity: "0", lastEquity: 95205.49, spyBars: SPLIT_BARS, curve: CURVE, day: "2026-11-05", since: "2026-05-06" });
  assert.deepEqual([bad.day.account_pct, bad.since, bad.d30], [null, null, null]);
  assert.match(String(bad.day.reason), /account read invalid/);
});

// ── 2. floor exposure ───────────────────────────────────────────────────────

const VM_CFG: FloorConfig = { mode: "vol_managed_spy", base: "SPY", levered: "SSO", owned: ["SPY", "SSO"], legacy: ["GLD", "KWEB", "QQQ", "SMH", "VXUS"] };
const POSITIONS = parsePositionsLite([
  { symbol: "SPY", qty: "80", side: "long", market_value: "60000", unrealized_pl: "100", asset_class: "us_equity" },
  { symbol: "SSO", qty: "200", side: "long", market_value: "20000", unrealized_pl: "50", asset_class: "us_equity" },
  { symbol: "QQQ", qty: "8", side: "long", market_value: "5000", unrealized_pl: "0", asset_class: "us_equity" },
  { symbol: "AAPL", qty: "10", side: "long", market_value: "3000", unrealized_pl: "0", asset_class: "us_equity" },
  { symbol: "HOLD261218C00050000", qty: "1", side: "long", market_value: "280", unrealized_pl: "30", asset_class: "us_option" },
  { symbol: "QQQ261218P00500000", qty: "3", side: "long", market_value: "570", unrealized_pl: "-30", asset_class: "us_option" },
]);
/** Python's datetime.now().isoformat(): local wall time, no offset. */
const naiveLocal = (t: number) => new Date(t - new Date(t).getTimezoneOffset() * 60_000).toISOString().slice(0, 23);

test("floor: vol-managed = (SPY + 2 x SSO) / equity in SPY terms, legacy reported apart, the target from fresh floor state", () => {
  const nowMs = ms("2026-11-05T21:20:00Z");
  const vmState = { at: naiveLocal(nowMs - 600_000), equity: 100000, ok: true, target_exposure: 1.2, capital: 85000 };
  const f = floorExposure({ positions: POSITIONS, equity: 100000, cfg: VM_CFG, vmState, nowMs });
  assert.deepEqual([f.exposure_x, f.base_mv, f.levered_mv, f.legacy_mv, f.target_x], [1, 60000, 20000, 5000, 1.02]);
  assert.match(String(f.basis), /SPY \+ 2 x SSO/);
  const stale = floorExposure({ positions: POSITIONS, equity: 100000, cfg: VM_CFG, vmState: { ...vmState, at: naiveLocal(nowMs - 7 * 3_600_000) }, nowMs });
  assert.equal(stale.target_x, null, "a stale floor state shows no target");
  assert.equal(floorExposure({ positions: POSITIONS, equity: 100000, cfg: VM_CFG, vmState: { ...vmState, ok: false }, nowMs }).target_x, null);
});

test("floor: basket mode is floor market value / equity, labelled not SPY-equivalent; unknowns are null with a reason", () => {
  const basket: FloorConfig = { mode: "basket", base: "SPY", levered: "SSO", owned: ["GLD", "KWEB", "QQQ", "SMH", "VXUS"], legacy: [] };
  const f = floorExposure({ positions: POSITIONS, equity: 100000, cfg: basket, vmState: null, nowMs: 0 });
  assert.deepEqual([f.mode, f.exposure_x, f.floor_mv], ["basket", 0.05, 5000]);
  assert.match(String(f.basis), /not SPY-equivalent/);
  assert.match(String(floorExposure({ positions: POSITIONS, equity: 100000, cfg: null, vmState: null, nowMs: 0 }).reason), /FLOOR_MODE unreadable/);
  assert.match(String(floorExposure({ positions: null, equity: 100000, cfg: VM_CFG, vmState: null, nowMs: 0 }).reason), /positions unreadable/);
  const noMv = parsePositionsLite([{ symbol: "SSO", qty: "1", side: "long", market_value: null, unrealized_pl: "0" }]);
  const nullMv = floorExposure({ positions: noMv, equity: 100000, cfg: VM_CFG, vmState: null, nowMs: 0 });
  assert.deepEqual([nullMv.exposure_x, nullMv.reason], [null, "SSO held with no market value"]);
  assert.equal(parseFloorConfig({ error: "ModuleNotFoundError" }), null);
  assert.deepEqual(parseFloorConfig({ mode: "vol_managed_spy", base: "SPY", levered: "SSO", owned: ["SPY", "SSO"], legacy: ["QQQ"] }),
    { mode: "vol_managed_spy", base: "SPY", levered: "SSO", owned: ["SPY", "SSO"], legacy: ["QQQ"] });
});

// ── 3. option fills/slippage vs direction ───────────────────────────────────

const FROM = ms("2026-11-04T21:20:00Z");
const TO = ms("2026-11-05T21:20:00Z");
const ORDERS = [
  { id: "o0", symbol: "XLF261127P00055500", asset_class: "us_option", side: "sell", status: "filled", filled_qty: "1", filled_avg_price: "0.60", filled_at: "2026-11-04T20:00:00Z" },
  { id: "o1", symbol: "XLF261127P00055500", asset_class: "us_option", side: "sell", status: "filled", filled_qty: "2", filled_avg_price: "0.45", filled_at: "2026-11-05T15:00:00Z" },
  { id: "o2", symbol: "IREN261211P00018000", asset_class: "us_option", side: "buy", status: "filled", filled_qty: "1", filled_avg_price: "1.80", filled_at: "2026-11-05T16:00:00Z" },
  { id: "o3", symbol: "GOOGL261127P00300000", asset_class: "us_option", side: "sell", status: "filled", filled_qty: "1", filled_avg_price: "10.65", filled_at: "2026-11-05T14:50:00Z" },
  { id: "o4", symbol: "QQQ261218P00500000", asset_class: "us_option", side: "buy", status: "filled", filled_qty: "3", filled_avg_price: "2.00", filled_at: "2026-11-05T15:30:00Z" },
  { id: "o5", symbol: "TLT261218C00090000", asset_class: "us_option", side: "buy", status: "canceled", filled_qty: "1", filled_avg_price: "1.10", filled_at: "2026-11-05T14:45:00Z" },
  { id: "o6", symbol: "TLT261218C00091000", asset_class: "us_option", side: "buy", status: "canceled", filled_qty: "0", filled_avg_price: null, filled_at: null },
  { id: "m1", order_class: "mleg", symbol: "", side: "", status: "filled", filled_qty: "2", filled_avg_price: "-0.55", filled_at: "2026-11-05T17:00:00Z", legs: [
    { id: "m1a", symbol: "SPY261218C00560000", side: "sell", filled_qty: "2", filled_avg_price: "1.02", filled_at: "2026-11-05T17:00:00Z" },
    { id: "m1b", symbol: "SPY261218C00570000", side: "buy", filled_qty: "2", filled_avg_price: "0.47", filled_at: "2026-11-05T17:00:00Z" },
  ] },
  { id: "m2", order_class: "mleg", symbol: "", side: "", status: "filled", filled_qty: "3", filled_avg_price: "0.96", filled_at: "2026-11-05T19:57:41Z" },
  { id: "s1", symbol: "SPY", asset_class: "us_equity", side: "buy", status: "filled", filled_qty: "10", filled_avg_price: "750", filled_at: "2026-11-05T15:00:00Z" },
];
const row = (order_id: string, symbol: string, side: string, mid: number | null, source: string): LedgerRow =>
  ({ order_id, symbol, side, qty: 1, bid: mid === null ? null : mid - 0.02, ask: mid === null ? null : mid + 0.02, mid, source });
const LEDGER = {
  rows: [
    row("o1", "XLF261127P00055500", "sell", 0.49, "options_manager._submit_close_order"),
    row("o2", "IREN261211P00018000", "buy", 1.70, "options_manager._submit_close_order"),
    row("o3", "GOOGL261127P00300000", "sell", null, "bot.ts:POS-KILL"),
    row("o5", "TLT261218C00090000", "buy", 1.00, "options_execution.submit_options_order"),
    row("m1", "SPY261218C00560000", "sell", 1.05, "options_execution._submit_mleg_order"),
    row("m1", "SPY261218C00570000", "buy", 0.45, "options_execution._submit_mleg_order"),
  ],
  malformed: 1,
};

test("fills: the window's option fills — single legs, a partly filled cancel, mleg legs — never stocks, never outside the window", () => {
  const { fills, mleg_no_leg_fills } = optionFillsInWindow(ORDERS, FROM, TO);
  assert.deepEqual(fills.map((f) => `${f.order_id}:${f.symbol}:${f.side}:${f.qty}@${f.price}`), [
    "o1:XLF261127P00055500:sell:2@0.45", "o2:IREN261211P00018000:buy:1@1.8", "o3:GOOGL261127P00300000:sell:1@10.65",
    "o4:QQQ261218P00500000:buy:3@2", "o5:TLT261218C00090000:buy:1@1.1",
    "m1:SPY261218C00560000:sell:2@1.02", "m1:SPY261218C00570000:buy:2@0.47",
  ]);
  assert.equal(mleg_no_leg_fills, 1, "an mleg without per-leg fill prices is counted, not split by a guessed sign convention");
});

test("fills: (fill − mid) × qty × 100 signed as a cost; no ledger row / null mid / no leg fills are unattributed, never guessed", () => {
  const of = optionFillsInWindow(ORDERS, FROM, TO);
  const f = fillsSlippage(of.fills, LEDGER, of.mleg_no_leg_fills);
  // o1 sold 0.04 under the mid x2 = $8, o2 bought 0.10 over = $10, o5 $10, m1 legs $6 + $4
  assert.deepEqual([f.fills, f.attributed, f.cost, f.pnl], [8, 5, 38, -38]);
  assert.deepEqual(f.unattributed, { count: 3, no_row: 1, no_mid: 1, mleg_no_leg_fills: 1, notional: 1665 });
  assert.deepEqual(f.worst.map((w) => [w.order_id, w.cost]), [["o2", 10], ["o5", 10], ["o1", 8], ["m1", 6], ["m1", 4]]);
  assert.equal(f.worst[0].source, "options_manager._submit_close_order");
  assert.equal(f.ledger_malformed, 1);
  // a fill ABOVE the mid on a sell is a gain (negative cost), not hidden
  const better = fillsSlippage([{ order_id: "x", symbol: "A261218C00010000", side: "sell", qty: 1, price: 1.10, filled_at: null, mleg: false }],
    { rows: [row("x", "A261218C00010000", "sell", 1.00, "t")], malformed: 0 });
  assert.equal(better.pnl, 10);
});

test("ledger: both generations are read, Python and bot.ts rows alike; malformed lines are counted", () => {
  const dir = tmpDir();
  try {
    fs.writeFileSync(path.join(dir, `${OPTION_FILL_LEDGER_FILENAME}.1`),
      `${JSON.stringify({ ts: "2026-11-04T15:00:00+00:00", order_id: "py-1", symbol: "XLF261127P00055500", side: "buy", qty: 1.0, bid: 0.47, ask: 0.51, mid: 0.49, source: "options_manager._submit_close_order", limit: 0.51, quote_age_s: 2.1 })}\n{broken\n`);
    assert.ok(appendOptionFillLedger(dir, optionLedgerRow("ts-1", "GOOGL261127P00300000", "SELL", 1, "bot.ts:POS-KILL", ms("2026-11-05T14:50:00Z"))));
    const got = readOptionFillLedger(dir);
    assert.deepEqual(got.rows.map((r) => [r.order_id, r.side, r.mid, r.source]),
      [["py-1", "buy", 0.49, "options_manager._submit_close_order"], ["ts-1", "sell", null, "bot.ts:POS-KILL"]]);
    assert.equal(got.malformed, 1);
    // rotation keeps one old generation
    for (let i = 0; i < 20; i++) appendOptionFillLedger(dir, optionLedgerRow(`r-${i}`, "GOOGL261127P00300000", "sell", 1, "t"), 400);
    assert.ok(fs.statSync(path.join(dir, OPTION_FILL_LEDGER_FILENAME)).size < 400 + 300);
    assert.ok(readOptionFillLedger(dir).rows.some((r) => r.order_id === "r-19"));
    assert.equal(appendOptionFillLedger(path.join(dir, `${OPTION_FILL_LEDGER_FILENAME}.1`, "x"), optionLedgerRow("z", "Q", "buy", 1, "t")), false, "an unwritable dir returns false, never throws");
    assert.deepEqual(parseLedger(`${JSON.stringify({ order_id: "", symbol: "X" })}\n\n`), { rows: [], malformed: 1 });
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

const START: OptionContract[] = [
  { symbol: "XLF261127P00055500", qty: 2, mv: 100, upl: -20 },    // sold today (o1)
  { symbol: "KMI261107P00025000", qty: 1, mv: 5, upl: -95 },      // expired worthless: left with no fill
  { symbol: "IREN261211P00018000", qty: -1, mv: -150, upl: 18 },  // short, bought back today (o2)
  { symbol: "HOLD261218C00050000", qty: 1, mv: 300, upl: 50 },    // still held
];
const EVENTS: RealizedEvent[] = [
  { symbol: "XLF261127P00055500", t: ms("2026-11-05T15:00:00Z"), pnl: -40, openedAt: null, qty: 2, closeDir: -1, basis: 130 },
  { symbol: "IREN261211P00018000", t: ms("2026-11-05T16:00:00Z"), pnl: -12, openedAt: null, qty: 1, closeDir: 1, basis: 168 },
  { symbol: "AAPL", t: ms("2026-11-05T16:00:00Z"), pnl: 500, openedAt: null, qty: 1, closeDir: -1, basis: 100 },
  { symbol: "XLF261127P00055500", t: ms("2026-11-04T20:00:00Z"), pnl: 99, openedAt: null, qty: 1, closeDir: -1, basis: 50 },
];

test("option P&L: TOTAL = realized + unrealized now − at the last line + a no-fill exit's value to zero; DIRECTION = TOTAL − FILLS", () => {
  assert.deepEqual(realizedOptionPnl(EVENTS, FROM, TO), { pnl: -52, lots: 2 }, "option lots closed in the window only");
  const of = optionFillsInWindow(ORDERS, FROM, TO);
  const fills = fillsSlippage(of.fills, LEDGER, of.mleg_no_leg_fills);
  const end = (POSITIONS ?? []).filter((p) => p.option).map((p) => ({ symbol: p.symbol, qty: p.qty, mv: p.marketValue, upl: p.unrealizedPl }));
  const o = optionPnl({ fromMs: FROM, toMs: TO, fromLine: "2026-11-04", events: EVENTS, start: START, end, fills, filledSymbols: new Set(of.fills.map((f) => f.symbol)) });
  // U_start −47, U_end 0, KMI −(5 − (−95)) = −100: −52 − 100 + 0 + 47 = −105; fills −38 -> direction −67
  assert.deepEqual([o.realized, o.unrealized_start, o.unrealized_end, o.total, o.direction], [-52, -47, 0, -105, -67]);
  assert.deepEqual(o.closed_without_fill, [{ symbol: "KMI261107P00025000", qty: 1, adjustment: -100 }]);
  // a short that expired worthless keeps its credit: liability 10 -> 0 is +10
  const shortGone = optionPnl({ fromMs: FROM, toMs: TO, fromLine: "2026-11-04", events: [], start: [{ symbol: "S261107P00010000", qty: -1, mv: -10, upl: 190 }],
    end: [], fills: fillsSlippage([], { rows: [], malformed: 0 }), filledSymbols: new Set() });
  assert.equal(shortGone.total, 10);
  const first = optionPnl({ fromMs: FROM, toMs: TO, fromLine: null, events: EVENTS, start: null, end, fills, filledSymbols: new Set() });
  assert.deepEqual([first.total, first.direction], [null, null]);
  assert.match(String(first.reason), /no previous line/);
  assert.equal(optionPnl({ fromMs: FROM, toMs: TO, fromLine: "2026-11-04", events: EVENTS, start: START, end: null, fills, filledSymbols: new Set() }).total, null);
});

test("option window: opens at the previous line (within 7 days), else at ET midnight with no start", () => {
  const prev = { ...fullLine(), day: "2026-11-04", written_at: "2026-11-04T21:20:05.000Z" };
  assert.deepEqual(optionWindowStart(prev, "2026-11-05", TO), { fromMs: ms("2026-11-04T21:20:05.000Z"), prev });
  const old = { ...prev, day: "2026-10-20", written_at: "2026-10-20T20:20:00.000Z" };
  assert.deepEqual(optionWindowStart(old, "2026-11-05", TO), { fromMs: ms("2026-11-05T05:00:00Z"), prev: null });
  assert.deepEqual(optionWindowStart(null, "2026-11-05", TO), { fromMs: ms("2026-11-05T05:00:00Z"), prev: null });
});

// ── 4. option exits held by the opening window ──────────────────────────────

const DEFERRALS = { version: 1, days: {
  "2026-11-05": { single_leg: ["XLF261127P00055500|loss_limit", "GOOGL261127P00300000|dte_close_bought"], group: ["IREN261204C00024000,IREN261204P00020000|strategy_ic_max_loss_exit"] },
  "2026-11-04": { single_leg: ["OLD261127P00010000|x"], group: [] },
} };
const POS_KILL_ROWS = [
  { time: "2026-11-05T14:31:00.000Z", message: "GOOGL261127P00300000: option down -80.0% (threshold: -75%) in the opening window — no option exit before 09:45 ET" },
  { time: "2026-11-05T14:32:00.000Z", message: "GOOGL261127P00300000: option down -81.0% ..." },
  { time: "2026-11-04T14:31:00.000Z", message: "XLK261219P00100000: option down -80.0% ..." },
  { time: "2026-11-05T14:40:00.000Z", message: "KWEB261218C00030000: option down -76.0% ..." },
];

test("opening window: the manager's (contract, rule) set for the ET day + the day's POS-KILL-OPTION-OPENING tickers", () => {
  const b = openingWindowBlocks(DEFERRALS, "2026-11-05", POS_KILL_ROWS);
  assert.deepEqual([b.total, b.manager_single_leg, b.manager_group, b.pos_kill], [5, 2, 1, 2]);
  assert.deepEqual(b.pos_kill_tickers, ["GOOGL261127P00300000", "KWEB261218C00030000"]);
  assert.deepEqual(openingWindowBlocks(null, "2026-11-05", []).total, 0, "no file: nothing was held");
  assert.match(String(openingWindowBlocks(DEFERRALS, "2026-11-05", null).reason), /audit log unreadable/);
});

// ── the line ────────────────────────────────────────────────────────────────

function inputs(over: Partial<DailyLineInputs> = {}): DailyLineInputs {
  const nowMs = ms("2026-11-05T21:20:30Z");
  const prev = { ...fullLineShell(), day: "2026-11-04", written_at: "2026-11-04T21:20:00.000Z" };
  prev.options.contracts_end = START;
  return {
    day: "2026-11-05", nowMs, session: NOV5,
    account: { equity: "100000", lastEquity: "101000" },
    positions: POSITIONS, spyBars: SPLIT_BARS, curve: CURVE, floorCfg: VM_CFG,
    vmState: { at: naiveLocal(nowMs - 600_000), equity: 100000, ok: true, target_exposure: 1.1, capital: 85000 },
    prev, windowFromMs: ms("2026-11-04T21:20:00.000Z"), events: EVENTS, orders: ORDERS, ledger: LEDGER,
    deferrals: DEFERRALS, posKillRows: POS_KILL_ROWS, missing: [],
    ...over,
  };
}

/** A minimal stored line (the shape loadDailyLines returns). */
function fullLineShell(): DailyLine {
  return buildDailyLine({
    day: "2026-11-05", nowMs: ms("2026-11-05T21:20:30Z"), session: NOV5, account: null, positions: null, spyBars: null, curve: [],
    floorCfg: null, vmState: null, prev: null, windowFromMs: ms("2026-11-05T05:00:00Z"), events: null, orders: null,
    ledger: { rows: [], malformed: 0 }, deferrals: null, posKillRows: [], missing: [],
  });
}
function fullLine(): DailyLine {
  return buildDailyLine(inputs());
}

test("the line: one compact audit message carrying all four answers", () => {
  const l = fullLine();
  assert.equal(l.day, "2026-11-05");
  assert.equal(l.session_close, "16:00");
  assert.deepEqual([l.account_vs_spy.day.account_pct, l.account_vs_spy.day.spy_pct], [-0.99, 0.4]);
  assert.equal(l.floor.exposure_x, 1);
  assert.equal(l.options.total, -105);
  assert.equal(l.opening_window.total, 5);
  assert.equal(l.message,
    "2026-11-05 acct -0.99% vs SPY +0.40% (-1.39pp) | since 05-06 +0.00% vs +6.06% (-6.06pp); 30d +2.04% vs +1.76% (+0.28pp)"
    + " | floor 1.00x SPY (target 0.94x) | options -$105 = fills -$38 + direction -$67 (5/8 fills priced)"
    + " | opening-window blocks 5 (manager 3, POS-KILL 2)");
  assert.ok(l.message.length <= 490, "persistAudit keeps 500 chars");
  assert.equal(dailyLineMessage(l), l.message);
  assert.equal(l.options.window.single_session, true, "the previous line is the previous session's (the SPY bar before today)");
  // a missed line (the previous one is two sessions back): the option window says where it starts
  const base = inputs();
  const missed = buildDailyLine({ ...base, prev: base.prev ? { ...base.prev, day: "2026-11-03" } : null });
  assert.equal(missed.options.window.single_session, false);
  assert.match(missed.message, /\| options since 11-03 line -\$105 = /);
});

test("the line: a partial line names what was missing, and an unreadable orders list voids the option total", () => {
  const l = buildDailyLine(inputs({ orders: null, spyBars: null, missing: ["orders: Alpaca 500", "spy bars: timeout"] }));
  assert.deepEqual([l.options.total, l.options.direction], [null, null]);
  assert.match(String(l.options.reason), /orders are unreadable/);
  assert.match(l.message, /acct -0\.99% vs SPY n\/a/);
  assert.match(l.message, /\| partial: orders, spy bars$/);
  assert.deepEqual(l.missing, ["orders: Alpaca 500", "spy bars: timeout"]);
});

test("store: saveDailyLine replaces the same day, keeps the newest 60, and loads oldest first", () => {
  const dir = tmpDir();
  try {
    const base = fullLine();
    for (let i = 0; i < DAILY_LINE_KEEP + 5; i++) {
      const d = new Date(ms("2026-06-01T12:00:00Z") + i * 86_400_000).toISOString().slice(0, 10);
      assert.ok(saveDailyLine(dir, { ...base, day: d }));
    }
    saveDailyLine(dir, { ...base, day: "2026-08-04", message: "replaced" });
    const lines = loadDailyLines(dir);
    assert.equal(lines.length, DAILY_LINE_KEEP);
    assert.equal(lines[0].day, "2026-06-06");
    assert.equal(lastLineDay(lines), "2026-08-04");
    assert.equal(lines.filter((l) => l.day === "2026-08-04").length, 1);
    assert.equal(lines[lines.length - 1].message, "replaced");
    fs.writeFileSync(path.join(dir, "voltrade_daily_line.json"), "{not json");
    assert.deepEqual(loadDailyLines(dir), [], "an unreadable file is no lines (the day is then due)");
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ── bot.ts wiring ───────────────────────────────────────────────────────────

const slice = (from: string, to: string) => {
  const a = bot.indexOf(from);
  assert.ok(a > 0, `marker not found: ${from}`);
  const b = bot.indexOf(to, a);
  assert.ok(b > a, `end marker not found: ${to}`);
  return bot.slice(a, b);
};

test("wiring: once per day — saved BEFORE the one DAILY-LINE audit row; SPY price-only; a 5-minute poll; the probe is sanitized", () => {
  const check = slice("async function checkDailyLine(): Promise<void> {", "// ── TIER 0: Real-Time Streaming");
  assert.ok(check.indexOf("saveDailyLine(dir, line)") > 0 && check.indexOf("saveDailyLine(dir, line)") < check.indexOf('audit("DAILY-LINE", line.message)'),
    "the line is the claim: saved before its audit row");
  assert.ok(check.includes("dailyLineDue(Date.now(), session, written)"));
  const compute = slice("async function computeDailyLine(", "async function checkDailyLine(");
  assert.ok(compute.includes('fetchDailyBarsRange("SPY", from, new Date(startMs).toISOString(), "split")'), "SPY is price-only (adjustment=split)");
  assert.ok(compute.indexOf("if (missing.length && !gate.partialOk) return { line: null, missing };") < compute.indexOf("refreshSleeves()"),
    "a missing cheap input never costs the full fill walk");
  assert.ok(bot.includes("setInterval(() => { checkDailyLine()") && bot.includes("}, DAILY_LINE_CHECK_MS);"));
  const probe = slice('case "daily_line": {', 'case "regime": {');
  assert.ok(probe.includes("sanitizeDiag({") && probe.includes('probe: "daily_line"'));
  assert.ok(!/alpaca\(/.test(probe), "the probe reads the file only, never the broker");
});

test("wiring: every bot.ts option order path writes its ledger row only AFTER its POST returned, with the POST unchanged", () => {
  const sites: Array<[string, string, string]> = [
    ['app.post("/api/bot/trade"', "// Close a position", '"bot.ts:/api/bot/trade"'],
    ['audit("TIME-EXIT", `${ticker}', 'audit("TIME-EXIT-ERROR"', '"bot.ts:TIME-EXIT"'],
    ["const killOrderResult = await alpaca(", "// STALE-ORDER-SWEEP FIX (KNOWN BROKEN #34)", '"bot.ts:POS-KILL"'],
    ["Scale-out would leave", "CC-UNWIND-ERROR", '"bot.ts:CC-UNWIND scale-out"'],
    ["before stock exit`);", "CC-UNWIND-ERROR", '"bot.ts:CC-UNWIND stock exit"'],
  ];
  for (const [from, to, source] of sites) {
    const s = slice(from, to);
    const post = s.indexOf('await alpaca("/v2/orders", {');
    const write = s.indexOf(`ledgerOptionOrder(`);
    assert.ok(post >= 0 && write > post, `${source}: the ledger write must follow the order POST`);
    assert.ok(s.includes(source), `${source}: the row names its path`);
  }
  // the helper can never throw into an order branch
  const helper = slice("function ledgerOptionOrder(", "// POS-KILL SCOPE FIX 2026-10-13");
  assert.ok(helper.includes("try {") && helper.includes("} catch (e: unknown) {") && helper.includes("if (!isOptionSymbol(symbol)) return;"));
});

// The REAL per-position loop body of syncMonitoredPositions (the cut the
// POS-KILL harnesses use) and the REAL ledgerOptionOrder, with only the file
// append stubbed: a confirmed option kill and an option TIME-EXIT each write
// one row with the order id the POST returned; a stock kill writes none.
test("wiring (run): the POS-KILL and TIME-EXIT option orders write one ledger row each, keyed by the returned order id; stocks write none", async () => {
  const matchBrace = (src: string, open: number): number => {
    let depth = 0;
    for (let i = open; i < src.length; i++) {
      const c = src[i];
      if (c === '"' || c === "'" || c === "`") { for (i++; i < src.length && src[i] !== c; i++) if (src[i] === "\\") i++; continue; }
      if (c === "{") depth++;
      else if (c === "}" && --depth === 0) return i + 1;
    }
    throw new Error("unbalanced braces");
  };
  const fnSrc = (sig: string) => { const s = bot.indexOf(sig); return s < 0 ? "" : bot.slice(s, matchBrace(bot, bot.indexOf("{", s + sig.length))); };
  const syncStart = bot.indexOf("async function syncMonitoredPositions()");
  const loopStart = bot.indexOf("for (const pos of positions) {", syncStart);
  const loopEnd = bot.indexOf("// Read stop state for this ticker", loopStart);
  const helpers = [
    "function isOptionSymbol(ticker: string): boolean",
    "function isPosKillExemptOption(symbol: string, assetClass?: string): boolean",
    'function posKillThresholdPct(isOption: boolean, side: "long" | "short", stockPct: number,',
    "function posKillHasValidMark(currentPriceRaw: unknown, plpcRaw: unknown): boolean",
    "function posKillObserveBreach(pending: Map<string, PosKillBreach>, symbol: string,",
    "function ledgerOptionOrder(order: unknown, symbol: string, side: string, qty: number, source: string): void",
  ].map(fnSrc).join("\n");
  assert.ok(helpers.includes("function ledgerOptionOrder("), "the real ledger helper is cut from bot.ts");
  const js = transformSync(`${helpers}
async function runTick(positions, env) {
  const { FLOOR_AND_LEG_TICKERS, LEGACY_DEFENSIVE_CANDIDATE_TICKERS, audit, alpaca, state, openOrders, getOrderParams,
          notify, pkgVersion, posKillPendingBreach, posKillNoMarkAuditDay } = env;
  const posKillMarketOpenCurrent = true;
  const posKillOptionOpeningWindowCurrent = false;
  const activeTickers = new Set();
  ${bot.slice(loopStart, loopEnd)}
}
}`, { loader: "ts" }).code;
  let nowMs = ms("2026-11-05T15:00:00Z");
  class FakeDate extends Date {
    constructor(v?: number | string | Date) { if (v === undefined) super(nowMs); else super(v); }
    static now(): number { return nowMs; }
  }
  const rows: Array<Record<string, unknown>> = [];
  const runTick = new Function("Date", "appendOptionFillLedger", "dailyLineDir", "optionLedgerRow", "process", `${js}\nreturn runTick;`)(
    FakeDate, (_dir: string, r: Record<string, unknown>) => { rows.push(r); return true; }, () => "/nowhere", optionLedgerRow, { env: {} },
  ) as (p: unknown[], e: unknown) => Promise<void>;
  const posts: string[] = [];
  const env = {
    FLOOR_AND_LEG_TICKERS: new Set(["SPY", "SSO"]), LEGACY_DEFENSIVE_CANDIDATE_TICKERS: new Set<string>(),
    audit: () => undefined, notify: () => undefined, pkgVersion: "test", state: {}, openOrders: [] as unknown[],
    getOrderParams: () => ({ type: "market", time_in_force: "day" }),
    alpaca: async (p: string, init?: { method?: string; body?: string }) => {
      if (init?.method === "POST") { posts.push(JSON.parse(init.body ?? "{}").symbol); return { id: `ord-${posts.length}` }; }
      return [];
    },
    posKillPendingBreach: new Map(), posKillNoMarkAuditDay: new Map(),
  };
  const pick = (r: Record<string, unknown>) => ({ id: r.order_id, symbol: r.symbol, side: r.side, qty: r.qty, mid: r.mid, source: r.source });
  const tick = async (p: unknown[]) => { await runTick(p, env); nowMs += 45_000; };
  const longOpt = (pnlPct: number, symbol: string, createdAt = "2026-11-03T15:00:00Z") => ({
    symbol, asset_class: "us_option", side: "long", qty: "2", avg_entry_price: "4", current_price: (4 * (1 + pnlPct / 100)).toFixed(2),
    unrealized_plpc: String(pnlPct / 100), created_at: createdAt,
  });
  await tick([longOpt(-80, "XLK261219P00100000")]);
  await tick([longOpt(-80, "XLK261219P00100000")]);
  assert.deepEqual(posts, ["XLK261219P00100000"], "the confirmed option kill POSTs once");
  assert.deepEqual(rows.map(pick), [{ id: "ord-1", symbol: "XLK261219P00100000", side: "sell", qty: 2, mid: null, source: "bot.ts:POS-KILL" }],
    "no quote on this path: the mid is null (unattributed), never the position's mark");
  await tick([longOpt(1, "OLD261219C00050000", "2026-10-01T15:00:00Z")]);
  assert.deepEqual(rows.map(pick)[1], { id: "ord-2", symbol: "OLD261219C00050000", side: "sell", qty: 2, mid: null, source: "bot.ts:TIME-EXIT" });
  const stock = { symbol: "AXTI", asset_class: "us_equity", side: "long", qty: "5", avg_entry_price: "20", current_price: "14", unrealized_plpc: "-0.3", created_at: "2026-11-03T15:00:00Z" };
  await tick([stock]);
  await tick([stock]);
  assert.equal(posts[posts.length - 1], "AXTI", "the stock kill still POSTs exactly as before");
  assert.equal(rows.length, 2, "a stock order writes no fill-ledger row");
});
