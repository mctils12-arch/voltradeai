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
  DAILY_LINE_KEEP, DAILY_LINE_SPY_MAX_AGE_MS, OPTION_FILL_LEDGER_FILENAME,
  appendOptionFillLedger, buildDailyLine, dailyLineBody, dailyLineDue, dailyLineMessage, dailyLinePrecheck, dailyLineSchedule, fillsSlippage, floorExposure,
  lastLineDay, loadDailyLines, openingWindowBlocks, optionFillsInWindow, optionLedgerRow, optionPnl, optionWindowStart, readDailyLines,
  parseCalendarSession, parseFloorConfig, parseLedger, parsePositionsLite, readOptionFillLedger, realizedOptionPnl,
  sameBasis, saveDailyLine, type DailyLine, type DailyLineInputs, type FloorConfig, type LedgerRow, type OptionContract,
  // the close reading ([RULE-REVIEW] 2026-11-06)
  DAILY_LINE_CLOSE_FILENAME, DAILY_LINE_CLOSE_KEEP, DAILY_LINE_CLOSE_LEAD_MS, DAILY_LINE_CLOSE_MAX_AGE_MS, DAILY_LINE_CLOSE_TICK_MS,
  SAME_BASIS_TEXT, SAME_BASIS_TEXT_WRITE_TIME, closeCheck, closeReadingFor, closeReadingFrom, closeReadingWindow, readCloseReadings,
  saveCloseReading, usableCloseReading, type CloseReading,
} from "./dailyLine";
import type { RealizedEvent } from "./sleeveLedger";
import { runLadder, LADDER_DEFAULTS } from "./limitLadder";

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
    // the normal day ([RULE-REVIEW] 2026-11-06): a close reading 10 s before the bell (here equal to the evening read,
    // so the pins below are the ones they were; the tests under "the close reading" move it)
    close: { day: "2026-11-05", at: "2026-11-05T20:59:50.000Z", equity: 100000, last_equity: 101000, balance_asof: null, reads: 6 },
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
  assert.ok(compute.includes("spyBars = (await spyBenchBars(DAILY_LINE_SINCE, DAILY_LINE_SPY_MAX_AGE_MS)).bars;"),
    "SPY is the benchmark's own price-only series (one fetch, one cache), fresh enough for the day's final bar");
  assert.ok(!compute.includes('fetchDailyBarsRange("SPY"'), "no second SPY fetch");
  assert.ok(DAILY_LINE_SPY_MAX_AGE_MS > 0 && DAILY_LINE_SPY_MAX_AGE_MS <= 20 * 60_000, "younger than the gate's close + 20 min");
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
    ["const send = async (order: Record<string, string>): Promise<LadderSubmit> => {", "submitLimit: (q, price) =>", '"bot.ts:POS-KILL ladder"'],
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
test("wiring (run): a confirmed option kill goes to the backstop ladder, whose order writes one ledger row keyed by the returned id; TIME-EXIT never closes an option; stocks write none", async () => {
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
  const starter = fnSrc("async function posKillStartOptionLadder(");
  assert.ok(starter.includes("ledgerOptionOrder(placed, ticker, closeSide"), "the ladder's send writes the ledger row");
  assert.ok(helpers.includes("function ledgerOptionOrder("), "the real ledger helper is cut from bot.ts");
  const js = transformSync(`${helpers}
async function runTick(positions, env) {
  const { FLOOR_AND_LEG_TICKERS, LEGACY_DEFENSIVE_CANDIDATE_TICKERS, audit, alpaca, state, openOrders, getOrderParams,
          notify, pkgVersion, posKillPendingBreach, posKillNoMarkAuditDay, posKillLadderInFlight, runLadder } = env;
  const posKillMarketOpenCurrent = true;
  const posKillOptionOpeningWindowCurrent = false;
  const BACKSTOP_LADDER_CONFIG = env.ladderConfig;
  const posKillOptionQuote = async () => ({ bid: 0.75, ask: 0.85 });
  const posKillLadderSleep = () => new Promise(() => undefined); // the ladder stops after its first step here
  ${starter}
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
    posKillLadderInFlight: new Map<string, number>(), runLadder, ladderConfig: LADDER_DEFAULTS,
  };
  const pick = (r: Record<string, unknown>) => ({ id: r.order_id, symbol: r.symbol, side: r.side, qty: r.qty, mid: r.mid, source: r.source });
  const flush = async () => { for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r)); };
  const tick = async (p: unknown[]) => { await runTick(p, env); await flush(); nowMs += 45_000; };
  const longOpt = (pnlPct: number, symbol: string, createdAt = "2026-11-03T15:00:00Z") => ({
    symbol, asset_class: "us_option", side: "long", qty: "2", avg_entry_price: "4", current_price: (4 * (1 + pnlPct / 100)).toFixed(2),
    unrealized_plpc: String(pnlPct / 100), created_at: createdAt,
  });
  await tick([longOpt(-80, "XLK261219P00100000")]);
  await tick([longOpt(-80, "XLK261219P00100000")]);
  assert.deepEqual(posts, ["XLK261219P00100000"], "the confirmed option kill: the ladder's first limit, one POST");
  assert.deepEqual(rows.map(pick), [{ id: "ord-1", symbol: "XLK261219P00100000", side: "sell", qty: 2, mid: null, source: "bot.ts:POS-KILL ladder" }],
    "the ladder's order is in the ledger; no quote is held on this path, so the mid is null (unattributed), never the position's mark");
  await tick([longOpt(1, "OLD261219C00050000", "2026-10-01T15:00:00Z")]);
  assert.equal(posts.length, 1, "an aged, flat option is not time-exited (options_manager owns option exits, #1647)");
  assert.equal(rows.length, 1);
  const stock = { symbol: "AXTI", asset_class: "us_equity", side: "long", qty: "5", avg_entry_price: "20", current_price: "14", unrealized_plpc: "-0.3", created_at: "2026-11-03T15:00:00Z" };
  await tick([stock]);
  await tick([stock]);
  assert.equal(posts[posts.length - 1], "AXTI", "the stock kill still POSTs exactly as before");
  assert.equal(rows.length, 1, "a stock order writes no fill-ledger row");
});

// ── the owner route ([PRODUCT] 2026-11-06): GET /api/bot/daily-line and the diag probe serve ONE body ──────────

const CAL = { holidays: new Set(["2026-11-26"]), halfDays: new Set(["2026-11-27"]) };

test("readDailyLines: the job's reader, saying why when the file cannot be read; a missing file is no lines and no error", () => {
  const dir = tmpDir();
  try {
    assert.deepEqual(readDailyLines(dir), { lines: [], error: null }, "a fresh volume: nothing written yet is not an error");
    const base = fullLine();
    saveDailyLine(dir, { ...base, day: "2026-11-05" });
    saveDailyLine(dir, { ...base, day: "2026-11-04" });
    const ok = readDailyLines(dir);
    assert.equal(ok.error, null);
    assert.deepEqual(ok.lines.map((l) => l.day), ["2026-11-04", "2026-11-05"], "oldest first, as loadDailyLines");
    assert.deepEqual(loadDailyLines(dir), ok.lines, "loadDailyLines is the same read");
    fs.writeFileSync(path.join(dir, "voltrade_daily_line.json"), "{not json");
    const bad = readDailyLines(dir);
    assert.deepEqual(bad.lines, []);
    assert.match(bad.error ?? "", /^the daily line file is unreadable \(/);
    assert.deepEqual(loadDailyLines(dir), [], "the job still sees no lines (the day is then due)");
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("dailyLineSchedule: today's line by the NYSE calendar — due at the close + 20 min (13:20 ET on a half day); no session on a weekend or holiday", () => {
  assert.deepEqual(dailyLineSchedule(ms("2026-11-05T15:00:00Z"), CAL), { day: "2026-11-05", session: true, due_at: "2026-11-05T21:20:00.000Z" });
  assert.deepEqual(dailyLineSchedule(ms("2026-11-06T04:30:00Z"), CAL), { day: "2026-11-05", session: true, due_at: "2026-11-05T21:20:00.000Z" }, "23:30 ET is still the 5th");
  assert.deepEqual(dailyLineSchedule(ms("2026-11-27T15:00:00Z"), CAL), { day: "2026-11-27", session: true, due_at: "2026-11-27T18:20:00.000Z" });
  assert.deepEqual(dailyLineSchedule(ms("2026-11-26T15:00:00Z"), CAL), { day: "2026-11-26", session: false, due_at: null });
  assert.deepEqual(dailyLineSchedule(ms("2026-11-07T15:00:00Z"), CAL), { day: "2026-11-07", session: false, due_at: null });
  assert.equal(dailyLineSchedule(ms("2026-07-06T15:00:00Z"), CAL).due_at, "2026-07-06T20:20:00.000Z", "EDT: 16:20 ET is 20:20Z");
});

test("dailyLineBody: the stored lines as written, newest first, ?days 1–60 (default 10), the newest, the count, the last attempt, today's schedule and the read error", () => {
  const base = fullLine();
  const lines = Array.from({ length: 14 }, (_, i) => ({ ...base, day: new Date(ms("2026-10-20T12:00:00Z") + i * 86_400_000).toISOString().slice(0, 10) }));
  const attempt = { at: "2026-11-05T21:25:00.000Z", day: "2026-11-05", status: "waiting" as const, detail: "spy bars: no 2026-11-05 bar yet (last 2026-11-04)" };
  const now = ms("2026-11-05T21:30:00Z");
  const b = dailyLineBody({ lines, error: null }, undefined, attempt, now, CAL);
  assert.equal(b.count, 14);
  assert.equal(b.recent.length, 10);
  assert.deepEqual(b.recent.map((l) => l.day).slice(0, 2), ["2026-11-02", "2026-11-01"], "newest first");
  assert.equal(b.latest?.day, "2026-11-02");
  assert.strictEqual(b.recent[0], lines[13], "the line exactly as stored: nothing recomputed");
  assert.deepEqual(b.lastAttempt, attempt);
  assert.deepEqual(b.schedule, { day: "2026-11-05", session: true, due_at: "2026-11-05T21:20:00.000Z" });
  assert.equal(b.keep, DAILY_LINE_KEEP);
  assert.equal(b.error, null);
  for (const [raw, n] of [["3", 3], [3, 3], ["0", 10], ["-4", 1], ["500", 14], ["junk", 10], [null, 10]] as const) {
    assert.equal(dailyLineBody({ lines, error: null }, raw, null, now, CAL).recent.length, n, `days=${String(raw)}`);
  }
  const empty = dailyLineBody({ lines: [], error: "the daily line file is unreadable (Unexpected token)" }, "10", null, now, CAL);
  assert.deepEqual([empty.count, empty.latest, empty.recent, empty.lastAttempt, empty.error], [0, null, [], null, "the daily line file is unreadable (Unexpected token)"]);
});

test("wiring: GET /api/bot/daily-line is owner-gated and serves the diag probe's own body — the job's reader, no broker call, a bounded 500", () => {
  const route = slice('app.get("/api/bot/daily-line", requireOwner, (req, res) => {', "// ── REGIME ENGINE daily report");
  assert.ok(route.includes("res.json(dailyLineBodyNow(req.query.days));"));
  assert.ok(route.includes("res.status(500).json({ error: String(err instanceof Error ? err.message : err).slice(0, 200) });"));
  assert.equal(bot.split('app.get("/api/bot/daily-line"').length - 1, 1, "one route");
  const helper = slice("function dailyLineBodyNow(days: unknown) {", 'app.get("/api/bot/daily-line"');
  assert.ok(helper.includes("dailyLineBody(readDailyLines(dailyLineDir(process.env)), days, dailyLineLastAttempt, Date.now(), { holidays: marketHolidays(), halfDays: marketHalfDays() })"));
  assert.ok(!/alpaca\(/.test(helper + route), "the route reads the file only, never the broker");
  const probe = slice('case "daily_line": {', 'case "regime": {');
  assert.ok(probe.includes('sanitizeDiag({ probe: "daily_line", ...dailyLineBodyNow(req.query.days) })'), "the probe serves the same body");
});

// ── the close reading ([RULE-REVIEW] 2026-11-06): the account's day ends at the close, not at the write time ─────────
// The 2026-11-05 line read −4.07 % from equity 89,814.42 at 20:12 ET (after-hours marks in it) while Alpaca's own close
// to close was −3.99 %: last_equity rolled overnight from 93,623.46 to 89,888.90 ("equity as of previous trading day at
// 16:00:00 ET"); SPY's side was −4.34 % close to close. These pin the new basis, its fallback and the next-day check.

const NOV5_CLOSE_MS = ms("2026-11-05T21:00:00Z");
const reading = (over: Partial<CloseReading> = {}): CloseReading =>
  ({ day: "2026-11-05", at: "2026-11-05T20:59:52.000Z", equity: 89888.9, last_equity: 93623.46, balance_asof: "2026-11-04", reads: 6, ...over });

test("close reading: the window is the last minute before the NYSE close (13:00 ET on a half day); none on a weekend or holiday", () => {
  assert.equal(closeReadingWindow(ms("2026-11-05T20:58:59.999Z"), CAL), null, "15:58:59 EST: not yet");
  assert.deepEqual(closeReadingWindow(ms("2026-11-05T20:59:00Z"), CAL), { day: "2026-11-05", closeMs: NOV5_CLOSE_MS });
  assert.deepEqual(closeReadingWindow(ms("2026-11-05T20:59:59.999Z"), CAL), { day: "2026-11-05", closeMs: NOV5_CLOSE_MS });
  assert.equal(closeReadingWindow(NOV5_CLOSE_MS, CAL), null, "the bell itself is past the window: never a post-close read");
  assert.deepEqual(closeReadingWindow(ms("2026-07-06T19:59:30Z"), CAL), { day: "2026-07-06", closeMs: ms("2026-07-06T20:00:00Z") }, "EDT: 15:59:30 ET is 19:59:30Z");
  assert.deepEqual(closeReadingWindow(ms("2026-11-27T17:59:30Z"), CAL), { day: "2026-11-27", closeMs: ms("2026-11-27T18:00:00Z") }, "the half day closes at 13:00");
  assert.equal(closeReadingWindow(ms("2026-11-27T20:59:30Z"), CAL), null, "15:59 on a half day is after its close");
  assert.equal(closeReadingWindow(ms("2026-11-26T20:59:30Z"), CAL), null, "Thanksgiving");
  assert.equal(closeReadingWindow(ms("2026-11-07T20:59:30Z"), CAL), null, "Saturday");
  assert.equal(closeReadingWindow(ms("2026-11-05T20:58:30Z"), CAL, 2 * 60_000)?.day, "2026-11-05", "the lead is a parameter");
  // the timing the line relies on: a tick inside the window lands at most one tick before the bell, and stays inside MAX_AGE
  assert.ok(DAILY_LINE_CLOSE_TICK_MS * 3 <= DAILY_LINE_CLOSE_LEAD_MS, "several reads per window (a failed one is retried)");
  assert.ok(DAILY_LINE_CLOSE_LEAD_MS < DAILY_LINE_CLOSE_MAX_AGE_MS, "every reading the job keeps is young enough for the line");
});

test("close reading: an account answer becomes the day's reading (the latest replaces the earlier), kept for the newest 10 days", () => {
  const first = closeReadingFrom({ equity: "89901.10", last_equity: "93623.46", balance_asof: "2026-11-04" }, "2026-11-05", ms("2026-11-05T20:59:02Z"), null);
  assert.deepEqual(first, { day: "2026-11-05", at: "2026-11-05T20:59:02.000Z", equity: 89901.1, last_equity: 93623.46, balance_asof: "2026-11-04", reads: 1 });
  const last = closeReadingFrom({ equity: "89888.90", last_equity: "93623.46" }, "2026-11-05", ms("2026-11-05T20:59:52Z"), first);
  assert.deepEqual([last?.equity, last?.at, last?.reads, last?.balance_asof], [89888.9, "2026-11-05T20:59:52.000Z", 2, null]);
  assert.equal(closeReadingFrom({ equity: "89888.90" }, "2026-11-06", ms("2026-11-06T20:59:52Z"), first)?.reads, 1, "a new day counts from one");
  for (const bad of [{ equity: "0" }, { equity: "-5" }, { equity: "junk" }, {}, null, "x"]) {
    assert.equal(closeReadingFrom(bad, "2026-11-05", 0, null), null, `no reading from ${JSON.stringify(bad)}`);
  }
  assert.equal(closeReadingFrom({ equity: "1", last_equity: "0", balance_asof: "2026-11-04T00:00:00Z" }, "2026-11-05", 0, null)?.last_equity, null);
  assert.equal(closeReadingFrom({ equity: "1", balance_asof: "2026-11-04T00:00:00Z" }, "2026-11-05", 0, null)?.balance_asof, "2026-11-04", "Alpaca's date, as a date");
  const dir = tmpDir();
  try {
    assert.deepEqual(readCloseReadings(dir), [], "none yet");
    assert.ok(saveCloseReading(dir, first!));
    assert.ok(saveCloseReading(dir, last!));
    assert.deepEqual(readCloseReadings(dir), [last], "the day's latest read replaced its earlier one");
    for (let i = 0; i < DAILY_LINE_CLOSE_KEEP + 3; i++) {
      const d = new Date(ms("2026-10-01T12:00:00Z") + i * 86_400_000).toISOString().slice(0, 10);
      saveCloseReading(dir, reading({ day: d, at: `${d}T19:59:52.000Z` }));
    }
    const kept = readCloseReadings(dir);
    assert.equal(kept.length, DAILY_LINE_CLOSE_KEEP);
    assert.equal(kept[kept.length - 1].day, "2026-11-05", "the newest days are kept, oldest first");
    assert.equal(closeReadingFor(kept, "2026-11-05")?.equity, 89888.9);
    assert.equal(closeReadingFor(kept, "2026-10-01"), null, "rotated out");
    fs.writeFileSync(path.join(dir, DAILY_LINE_CLOSE_FILENAME), "{not json");
    assert.deepEqual(readCloseReadings(dir), [], "an unreadable file is no readings (the line then falls back, labelled)");
    assert.equal(saveCloseReading(path.join(dir, DAILY_LINE_CLOSE_FILENAME, "x"), last!), false, "an unwritable dir returns false, never throws");
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("close reading: usable only when it is this session's close — before the broker close, within 3 min of it, the same prior close", () => {
  assert.deepEqual(usableCloseReading(reading(), "2026-11-05", NOV5_CLOSE_MS, 93623.46), { ok: true, equity: 89888.9, at: "2026-11-05T20:59:52.000Z" });
  assert.deepEqual(usableCloseReading(reading({ last_equity: null }), "2026-11-05", NOV5_CLOSE_MS, 93623.46).ok, true, "a reading without its prior close is still the close");
  const no = (r: CloseReading | null, closeMs: number | null = NOV5_CLOSE_MS, le: number | null = 93623.46) => {
    const u = usableCloseReading(r, "2026-11-05", closeMs, le);
    return u.ok ? "ok" : u.reason;
  };
  assert.equal(no(null), "no close reading for 2026-11-05");
  assert.equal(no(reading({ day: "2026-11-04" })), "no close reading for 2026-11-05");
  assert.equal(no(reading(), null), "the session's close time is unknown");
  assert.equal(no(reading({ at: "2026-11-05T21:00:00.000Z" })), "the close reading was taken at 16:00 ET, not before the 16:00 ET close");
  // the broker's calendar says 13:00 (a half day market_calendar.py did not know): the 15:59 reading is after that close
  assert.equal(no(reading(), ms("2026-11-05T18:00:00Z")), "the close reading was taken at 15:59 ET, not before the 13:00 ET close");
  assert.equal(no(reading({ at: new Date(NOV5_CLOSE_MS - DAILY_LINE_CLOSE_MAX_AGE_MS - 60_000).toISOString() })), "the close reading is 4 min before the close (over 3)");
  assert.equal(no(reading({ at: new Date(NOV5_CLOSE_MS - DAILY_LINE_CLOSE_MAX_AGE_MS).toISOString() })), "ok", "exactly 3 min is still the close");
  assert.equal(no(reading({ last_equity: 95205.49 })), "the close reading's prior close (95205.49) is not the line's (93623.46)", "another session's read");
  assert.equal(no(reading({ at: "junk" })), "the close reading is unreadable");
});

// The 2026-11-05 line's own inputs (the daily_line diag probe, read 2026-11-06): equity 89,814.42 at 20:12 ET, last_equity
// 93,623.46, the equity curve's 05-06 and 10-06 points (the equity_curve probe: 106,742.68 and 105,773.85; the 11-05 and
// 11-06 rows are tier 3's evening reads, after the day, and ignored), and SPY bars rebuilt to the line's own SPY numbers
// (−4.34 % day, −0.05 % since 05-06, −5.86 % over 30 days; the raw bars are not stored, and SPY's side does not change).
const LIVE_CURVE = [
  { date: "2026-11-06", value: 89564.13 }, { date: "2026-11-05", value: 89800.22 }, { date: "2026-11-04", value: 93765 },
  { date: "2026-10-06", value: 105773.85 }, { date: "2026-05-06", value: 106742.68 },
];
const LIVE_SPY = [
  { date: "2026-05-06", close: 95.70785 }, { date: "2026-10-06", close: 101.61462 }, { date: "2026-11-04", close: 100 }, { date: "2026-11-05", close: 95.66 },
];

test("2026-11-05 before / after on identical inputs: write-time equity −4.07 % (+0.27 pp vs SPY) vs the close −3.99 % (+0.35 pp); SPY unchanged", () => {
  const base = { equity: 89814.42, lastEquity: 93623.46, spyBars: LIVE_SPY, curve: LIVE_CURVE, day: "2026-11-05", since: "2026-05-06", closeMs: NOV5_CLOSE_MS, nowMs: ms("2026-11-06T01:12:26.339Z") };
  // BEFORE: no close reading existed — the line ends at the equity read when written, exactly the stored line, now labelled
  const before = sameBasis({ ...base, close: null });
  assert.deepEqual([before.day.account_pct, before.day.spy_pct, before.day.gap_pp], [-4.07, -4.34, 0.27], "the stored 11-05 line, reproduced");
  assert.deepEqual([before.since?.accountPct, before.since?.spyPct, before.since?.gapPp], [-15.86, -0.05, -15.81]);
  assert.deepEqual([before.d30?.from, before.d30?.accountPct, before.d30?.spyPct, before.d30?.gapPp], ["2026-10-06", -15.09, -5.86, -9.23]);
  assert.deepEqual([before.day.account_source, before.day.equity_end, before.day.equity_end_at, before.day.close_reason],
    ["write_time", 89814.42, "2026-11-06T01:12:26.339Z", "no close reading for 2026-11-05"]);
  assert.equal(before.basis, SAME_BASIS_TEXT_WRITE_TIME);
  // AFTER: a close reading at Alpaca's own 16:00 ET equity (what last_equity rolled to overnight)
  const after = sameBasis({ ...base, close: reading() });
  assert.deepEqual([after.day.account_pct, after.day.spy_pct, after.day.gap_pp], [-3.99, -4.34, 0.35], "Alpaca's close to close");
  assert.deepEqual([after.since?.accountPct, after.since?.spyPct, after.since?.gapPp], [-15.79, -0.05, -15.74], "the windows end at the same close");
  assert.deepEqual([after.d30?.accountPct, after.d30?.spyPct, after.d30?.gapPp], [-15.02, -5.86, -9.16]);
  assert.deepEqual([after.day.account_source, after.day.equity_end, after.day.equity_end_at, after.day.close_reason], ["close", 89888.9, "2026-11-05T20:59:52.000Z", null]);
  assert.deepEqual([after.day.equity, after.day.last_equity], [89814.42, 93623.46], "equity stays the read when written; last_equity the prior close");
  assert.equal(after.basis, SAME_BASIS_TEXT);
  assert.match(after.basis, /PRICE-ONLY/);
  // the old method's error on this day: the after-hours drift from the close to 20:12 ET, −$74.48 = −0.08 pp of the day
  assert.equal(Math.round((before.day.account_pct! - after.day.account_pct!) * 100) / 100, -0.08);
  // an invalid evening read voids the day as before (the close reading never stands in for the prior close)
  const bad = sameBasis({ ...base, equity: "0", close: reading() });
  assert.deepEqual([bad.day.account_pct, bad.day.account_source, bad.day.equity_end, bad.since], [null, null, null, null]);
  assert.match(String(bad.day.reason), /account read invalid/);
});

test("the line: the account day ends at the close reading; the floor still divides by the equity read with the positions; the fallback is labelled", () => {
  // a close reading of 99,500 against the evening's 100,000 (last_equity 101,000)
  const l = buildDailyLine(inputs({ close: { day: "2026-11-05", at: "2026-11-05T20:59:50.000Z", equity: 99500, last_equity: 101000, balance_asof: null, reads: 6 } }));
  assert.deepEqual([l.account_vs_spy.day.account_pct, l.account_vs_spy.day.account_source, l.account_vs_spy.day.equity_end], [-1.49, "close", 99500]);
  assert.equal(l.floor.exposure_x, 1, "(60,000 + 2 x 20,000) / 100,000: the evening read, the same moment as the positions");
  assert.equal(l.account_vs_spy.since?.to, "2026-11-05");
  assert.ok(l.message.startsWith("2026-11-05 acct -1.49% vs SPY +0.40% (-1.89pp) | since 05-06 -0.50% vs +6.06% (-6.56pp)"), l.message);
  // a reading taken after the broker's 13:00 close (a half day the NYSE file missed) is not the close: the fallback, labelled
  const half = buildDailyLine(inputs({ session: { date: "2026-11-05", closeMin: 13 * 60 } }));
  assert.deepEqual([half.account_vs_spy.day.account_source, half.account_vs_spy.day.account_pct], ["write_time", -0.99]);
  assert.equal(half.session_close, "13:00");
  // no reading at all: the exact message, marked
  const none = buildDailyLine(inputs({ close: null }));
  assert.equal(none.account_vs_spy.day.close_reason, "no close reading for 2026-11-05");
  assert.equal(none.message,
    "2026-11-05 acct -0.99% vs SPY +0.40% (-1.39pp) [acct at write time] | since 05-06 +0.00% vs +6.06% (-6.06pp); 30d +2.04% vs +1.76% (+0.28pp)"
    + " | floor 1.00x SPY (target 0.94x) | options -$105 = fills -$38 + direction -$67 (5/8 fills priced)"
    + " | opening-window blocks 5 (manager 3, POS-KILL 2)");
  assert.equal(inputs().close?.equity, 100000);
  assert.ok(!fullLine().message.includes("[acct at write time]"), "the normal day carries no marker");
});

test("close check: the next line checks the last one against Alpaca's rolled last_equity — the 11-05 line read 0.08 pp low", () => {
  // the stored 11-05 line (written before this change: no account_source, so its day ended at the equity read when written)
  const nov5 = { ...fullLine(), day: "2026-11-05", written_at: "2026-11-06T01:12:26.339Z" } as DailyLine;
  nov5.account_vs_spy = { ...nov5.account_vs_spy, day: { account_pct: -4.07, spy_pct: -4.34, gap_pp: 0.27, equity: 89814.42, last_equity: 93623.46, spy_bar_date: "2026-11-05", reason: null } };
  const c = closeCheck(nov5, "89888.90", "2026-11-05", "2026-11-05");
  assert.deepEqual(c, {
    day: "2026-11-05", line_pct: -4.07, line_equity: 89814.42, line_source: "write_time",
    alpaca_close: 89888.9, alpaca_pct: -3.99, error_pp: -0.08, error_usd: -74.48, balance_asof: "2026-11-05", reason: null,
  });
  // which close last_equity is: balance_asof decides when present, else the previous line must be the previous session's
  assert.equal(closeCheck(nov5, "89888.90", null, "2026-11-05")?.error_pp, -0.08, "no balance_asof: the SPY bar before today says 11-05");
  assert.equal(closeCheck(nov5, "93623.46", "2026-11-04", "2026-11-05")?.reason, "last_equity is Alpaca's 2026-11-04 close, not 2026-11-05's", "not rolled yet");
  assert.equal(closeCheck(nov5, "89888.90", null, "2026-11-06")?.reason, "the previous line (2026-11-05) is not the previous session's (2026-11-06)", "a missed line in between");
  assert.equal(closeCheck(nov5, "89888.90", null, null)?.reason, "which session's close last_equity is cannot be told (no balance_asof, no SPY bars)");
  assert.equal(closeCheck(nov5, "junk", "2026-11-05", null)?.reason, "last_equity unreadable");
  assert.equal(closeCheck(null, "89888.90", "2026-11-05", "2026-11-05"), null, "no previous line: nothing to check");
  assert.equal(closeCheck(fullLineShell(), "89888.90", "2026-11-05", "2026-11-05")?.reason, "the 2026-11-05 line recorded no account day");
  // a line on the close reading is checked at the equity it ended at
  const onClose = { ...nov5, account_vs_spy: { ...nov5.account_vs_spy, day: { ...nov5.account_vs_spy.day, account_pct: -3.98, account_source: "close" as const, equity_end: 89901.1 } } };
  assert.deepEqual(closeCheck(onClose, 89888.9, "2026-11-05", null), { ...c, line_pct: -3.98, line_equity: 89901.1, line_source: "close", error_pp: 0.01, error_usd: 12.2 });
  // end to end: the 11-06 line carries it, and its audit row says it
  const nov6 = buildDailyLine(inputs({
    day: "2026-11-06", nowMs: ms("2026-11-06T21:20:30Z"), session: { date: "2026-11-06", closeMin: 960 },
    account: { equity: "89700", lastEquity: "89888.90", balanceAsof: "2026-11-05" },
    close: { day: "2026-11-06", at: "2026-11-06T20:59:55.000Z", equity: 89650, last_equity: 89888.9, balance_asof: "2026-11-05", reads: 6 },
    spyBars: [...SPLIT_BARS, { date: "2026-11-06", close: 758 }], prev: nov5, windowFromMs: ms(nov5.written_at),
  }));
  assert.deepEqual([nov6.close_check?.day, nov6.close_check?.error_pp], ["2026-11-05", -0.08]);
  assert.match(nov6.message, / \| 11-05 line -4\.07% vs Alpaca close -3\.99% \(-0\.08pp\)$/);
  assert.equal(nov6.account_vs_spy.day.account_pct, -0.27, "(89,650 − 89,888.90) / 89,888.90 at the close, not the evening's 89,700");
});

// The REAL captureCloseEquity from bot.ts, run on a fake clock across the bell with the real dailyLine.ts helpers and a
// temp DATA_DIR: no read outside the window, the last read before the bell is what the line gets, a failed read is
// retried, and nothing but GET /v2/account is ever sent.
test("wiring (run): captureCloseEquity reads the account only in the last minute before the close and keeps the last read before the bell", async () => {
  const start = bot.indexOf("async function captureCloseEquity(): Promise<void> {");
  assert.ok(start > 0, "captureCloseEquity is in bot.ts");
  let depth = 0, end = -1;
  for (let i = bot.indexOf("{", start); i < bot.length; i++) {
    if (bot[i] === "{") depth++;
    else if (bot[i] === "}" && --depth === 0) { end = i + 1; break; }
  }
  const src = bot.slice(start, end);
  assert.ok(!/state\.|tier1|killSwitch|halt/i.test(src), "independent of trading state: no Tier 1 / kill-switch / halt gate");
  assert.deepEqual(src.match(/alpaca\([^)]*\)/g), ['alpaca("/v2/account")'], "one broker call, a GET of the account");
  const js = transformSync(`let closeReadingRunning = false;\n${src}\nreturn captureCloseEquity;`, { loader: "ts" }).code;
  const dir = tmpDir();
  try {
    let nowMs = 0;
    class FakeDate extends Date {
      constructor(v?: number | string | Date) { if (v === undefined) super(nowMs); else super(v); }
      static now(): number { return nowMs; }
    }
    const calls: Array<{ path: string; at: string }> = [];
    let equity = 89950;
    let failNext = false;
    const alpaca = async (p: string) => {
      calls.push({ path: p, at: new Date(nowMs).toISOString() });
      if (failNext) { failNext = false; throw new Error("Alpaca 503"); }
      return { equity: String(equity), last_equity: "93623.46", balance_asof: "2026-11-04" };
    };
    const warn = console.warn;
    console.warn = () => undefined;
    const capture = new Function("Date", "alpaca", "closeReadingWindow", "marketHolidays", "marketHalfDays", "dailyLineDir", "process",
      "closeReadingFrom", "closeReadingFor", "readCloseReadings", "saveCloseReading", js)(
      FakeDate, alpaca, closeReadingWindow, () => CAL.holidays, () => CAL.halfDays, () => dir, { env: {} },
      closeReadingFrom, closeReadingFor, readCloseReadings, saveCloseReading,
    ) as () => Promise<void>;
    try {
      for (let t = ms("2026-11-05T20:58:00Z"); t <= ms("2026-11-05T21:01:00Z"); t += DAILY_LINE_CLOSE_TICK_MS) {
        nowMs = t;
        if (t === ms("2026-11-05T20:59:50Z")) failNext = true; // the read 10 s before the bell fails
        equity -= 10;
        await capture();
      }
    } finally {
      console.warn = warn;
    }
    assert.deepEqual(calls.map((c) => c.at.slice(11, 19)), ["20:59:00", "20:59:10", "20:59:20", "20:59:30", "20:59:40", "20:59:50"],
      "every tick of the last minute and none outside it (none at or after the bell)");
    assert.ok(calls.every((c) => c.path === "/v2/account"));
    const kept = readCloseReadings(dir);
    assert.equal(kept.length, 1);
    // the 20:59:50 read failed, so the 20:59:40 read (equity 89,950 − 10 x 11 = 89,840) is the day's close reading
    assert.deepEqual(kept[0], { day: "2026-11-05", at: "2026-11-05T20:59:40.000Z", equity: 89840, last_equity: 93623.46, balance_asof: "2026-11-04", reads: 5 });
    assert.equal(usableCloseReading(kept[0], "2026-11-05", NOV5_CLOSE_MS, 93623.46).ok, true, "and the line takes it");
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("wiring: the close reading runs on its own 10 s timer; the line reads it (and balance_asof) and never waits for it", () => {
  assert.ok(bot.includes("setInterval(() => { captureCloseEquity().catch((e: unknown) => console.error(\"[DAILY-LINE-CLOSE]\", e instanceof Error ? e.message : e)); }, DAILY_LINE_CLOSE_TICK_MS);"));
  assert.equal(bot.split("captureCloseEquity().catch(").length - 1, 1, "one timer");
  assert.equal(bot.split("captureCloseEquity(").length - 1, 2, "the declaration and the timer: no other caller");
  const job = slice("async function captureCloseEquity(): Promise<void> {", "// ── TIER 0: Real-Time Streaming");
  assert.ok(job.indexOf("closeReadingWindow(sentMs") < job.indexOf('alpaca("/v2/account")'), "the window is decided before any broker read");
  assert.ok(job.includes("saveCloseReading(dir, reading)"));
  const compute = slice("async function computeDailyLine(", "async function checkDailyLine(");
  assert.ok(compute.includes("account = { equity: a.equity, lastEquity: a.last_equity, balanceAsof: a.balance_asof };"));
  assert.ok(compute.includes("close: closeReadingFor(readCloseReadings(dir), gate.day)"));
  assert.ok(!/missing\.push\(`close/.test(compute), "a missing close reading is a labelled fallback, never an input the line waits for");
});
