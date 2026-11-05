// BROKER CLOCK ORDER GATE (2026-11-05, human-directed) — server/brokerClockGate.ts.
//
// The rule: no order is sent while Alpaca's clock says the market is closed,
// so nothing is queued for the next opening print and options never trade
// outside regular hours. The one exception (human, same day: "I want as much
// on as possible ... we have a paper account so use it"): an extended-hours
// SELL of a held long stock/ETF position — the limit orderParams.ts builds for
// stop_loss / trailing_stop / take_profit outside 09:30–16:00 — passes while a
// REAL extended session runs: a trading day, 04:00–09:30 ET or after the close
// until 20:00 ET (17:00 ET after an early close). Default ON;
// VOLTRADE_EXTENDED_HOURS_STOCK_EXITS=0 turns it off.
//
// The trading calendar is the real market_calendar.py (parsed the way bot.ts
// reads it); the clock bodies are the shape Alpaca's /v2/clock returns.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  createOrderGate, clockStateFrom, describeOrder, extendedSession, extendedStockExitsEnabled,
  emptySessionEvidence, observeReading, parseClockReading,
  ORDER_BLOCKED_AUDIT_TYPE, ORDER_BLOCKED_PREFIX, EXTENDED_HOURS_STOCK_EXITS_ENV,
} from "./brokerClockGate";
import { parseMarketHolidays } from "./optionsChainArchive";
import { getOrderParams } from "./orderParams";
import type { MarketCalendar } from "./jobHeartbeat";

const here = path.dirname(fileURLToPath(import.meta.url));
const calendarPy = fs.readFileSync(path.join(here, "..", "market_calendar.py"), "utf8");
const CAL: MarketCalendar = {
  holidays: parseMarketHolidays(calendarPy),
  halfDays: parseMarketHolidays(calendarPy, "HALF_DAYS"),
};

// Thu 2026-11-05: a normal session (EST, UTC-5). Fri 2026-11-27: early close
// 13:00 ET. Thu 2026-11-26: Thanksgiving. Sat 2026-11-07: weekend.
const CLOSED_AFTER_THU = { is_open: false, timestamp: "2026-11-05T17:30:00-05:00", next_open: "2026-11-06T09:30:00-05:00", next_close: "2026-11-06T16:00:00-05:00" };
const CLOSED_PRE_THU = { is_open: false, timestamp: "2026-11-05T08:00:00-05:00", next_open: "2026-11-05T09:30:00-05:00", next_close: "2026-11-05T16:00:00-05:00" };
const OPEN_THU = { is_open: true, timestamp: "2026-11-05T11:00:00.123456789-05:00", next_open: "2026-11-06T09:30:00-05:00", next_close: "2026-11-05T16:00:00-05:00" };
const CLOSED_AFTER_HALF = { is_open: false, timestamp: "2026-11-27T17:30:00-05:00", next_open: "2026-11-30T09:30:00-05:00", next_close: "2026-11-30T16:00:00-05:00" };
const OPEN_HALF = { is_open: true, timestamp: "2026-11-27T12:00:00-05:00", next_open: "2026-11-30T09:30:00-05:00", next_close: "2026-11-27T13:00:00-05:00" };
const CLOSED_SAT = { is_open: false, timestamp: "2026-11-07T17:30:00-05:00", next_open: "2026-11-09T09:30:00-05:00", next_close: "2026-11-09T16:00:00-05:00" };
const CLOSED_THANKSGIVING = { is_open: false, timestamp: "2026-11-26T08:00:00-05:00", next_open: "2026-11-27T09:30:00-05:00", next_close: "2026-11-27T13:00:00-05:00" };

const LONG_AXTI = { symbol: "AXTI", side: "long", qty: "5" };

/** The body bot.ts POSTs: orderParams.ts's own params for this context and ET hour. */
function body(side: "buy" | "sell", context: "stop_loss" | "take_profit" | "trailing_stop" | "new_entry" | "options_exit",
              etHour: number, symbol = "AXTI", qty = 5): string {
  return JSON.stringify({ symbol, qty: String(qty), side, ...getOrderParams(20, context, etHour) });
}

function harness(opts: { nowIso: string; clock: unknown; position?: unknown; env?: Record<string, string | undefined>; cal?: MarketCalendar }) {
  let nowMs = Date.parse(opts.nowIso);
  let clockBody: unknown = opts.clock;
  let position: unknown = opts.position;
  const audits: Array<{ type: string; msg: string }> = [];
  const positionReads: string[] = [];
  let clockReads = 0;
  const gate = createOrderGate({
    readClock: async () => { clockReads++; return { at: nowMs, clock: clockBody }; },
    audit: (type: string, msg: string) => { audits.push({ type, msg }); },
    getPosition: async (symbol: string) => {
      positionReads.push(symbol);
      if (position === undefined) throw new Error(`Alpaca 404: {"code":40410000,"message":"position does not exist"}`);
      return position;
    },
    calendar: () => opts.cal ?? CAL,
    isOption: (symbol: string) => symbol.length > 10 || /^[A-Z]+\d{6}[CP]\d{8}$/.test(symbol),
    now: () => nowMs,
    env: () => opts.env ?? {},
  });
  return {
    gate, audits, positionReads,
    clockReads: () => clockReads,
    at(iso: string) { nowMs = Date.parse(iso); },
    serve(b: unknown) { clockBody = b; },
    hold(p: unknown) { position = p; },
    post: (b: string) => gate.enforce("/v2/orders", "POST", b),
  };
}

async function refused(p: Promise<void>, ...fragments: string[]): Promise<string> {
  let message = "";
  await assert.rejects(p, (err: unknown) => {
    message = err instanceof Error ? err.message : String(err);
    return true;
  });
  assert.ok(message.startsWith(`${ORDER_BLOCKED_PREFIX}: market closed`), message);
  for (const f of fragments) assert.ok(message.includes(f), `expected "${f}" in: ${message}`);
  return message;
}

// ── the human's cases (extended-hours stock exits, default ON) ─────────────

test("premise: market_calendar.py knows the dates these tests use", () => {
  assert.ok(CAL.holidays.has("2026-11-26"), "Thanksgiving");
  assert.ok(CAL.halfDays.has("2026-11-27"), "the day after Thanksgiving closes at 13:00 ET");
  assert.ok(!CAL.holidays.has("2026-11-05") && !CAL.halfDays.has("2026-11-05"));
  assert.equal(getOrderParams(20, "stop_loss", 17.5).extended_hours, true, "orderParams builds an extended-hours limit after 16:00");
});

test("an after-hours stop exit at 17:30 ET on a normal trading day passes", async () => {
  const h = harness({ nowIso: "2026-11-05T17:30:00-05:00", clock: CLOSED_AFTER_THU, position: LONG_AXTI });
  await h.post(body("sell", "stop_loss", 17.5));
  await h.post(body("sell", "take_profit", 17.5));
  await h.post(body("sell", "trailing_stop", 17.5));
  assert.deepEqual(h.positionReads, ["AXTI", "AXTI", "AXTI"], "each exit is checked against the held position");
  assert.equal(h.audits.length, 0);
});

test("the same exit at 17:30 ET after a half-day early close is refused (that after-hours session ended at 17:00)", async () => {
  const h = harness({ nowIso: "2026-11-27T17:30:00-05:00", clock: CLOSED_AFTER_HALF, position: LONG_AXTI });
  await refused(h.post(body("sell", "stop_loss", 17.5)), "after-hours 13:00–17:00");
  assert.equal(h.positionReads.length, 0, "no session: the position is not even read");
  // ...while inside that day's 13:00–17:00 session an extended-hours exit passes
  h.at("2026-11-27T16:30:00-05:00");
  await h.post(body("sell", "stop_loss", 16.5));
  h.at("2026-11-27T13:30:00-05:00");
  await h.post(body("sell", "stop_loss", 16.5)); // an extended-hours limit at 13:30 would pass too...
  // ...but orderParams.ts reads the time of day only, so at 13:30 it still
  // builds a plain MARKET stop, which cannot trade after the close: refused.
  assert.equal(getOrderParams(20, "stop_loss", 13.5).type, "market");
  await refused(h.post(body("sell", "stop_loss", 13.5)));
});

test("an early close the broker reported is honoured even if market_calendar.py missed it", async () => {
  const noHalfDays: MarketCalendar = { holidays: CAL.holidays, halfDays: new Set() };
  const h = harness({ nowIso: "2026-11-27T12:00:00-05:00", clock: OPEN_HALF, position: LONG_AXTI, cal: noHalfDays });
  await h.post(body("sell", "stop_loss", 12)); // open: everything passes
  h.serve(CLOSED_AFTER_HALF);
  h.at("2026-11-27T17:30:00-05:00");
  await refused(h.post(body("sell", "stop_loss", 17.5)), "after-hours 13:00–17:00");
});

test("at 21:00 ET (overnight) the exit is refused", async () => {
  const h = harness({ nowIso: "2026-11-05T21:00:00-05:00", clock: CLOSED_AFTER_THU, position: LONG_AXTI });
  await refused(h.post(body("sell", "stop_loss", 21)), "no extended session at 21:00 ET");
  h.at("2026-11-06T03:59:00-05:00");
  await refused(h.post(body("sell", "stop_loss", 3.99)), "no extended session at 03:59 ET");
});

test("on a Saturday the exit is refused, morning or evening", async () => {
  const h = harness({ nowIso: "2026-11-07T17:30:00-05:00", clock: CLOSED_SAT, position: LONG_AXTI });
  await refused(h.post(body("sell", "stop_loss", 17.5)), "2026-11-07 is not a trading day");
  h.at("2026-11-07T08:00:00-05:00");
  await refused(h.post(body("sell", "stop_loss", 8)), "2026-11-07 is not a trading day");
});

test("a pre-market ENTRY is refused; a pre-market exit on a trading day passes", async () => {
  const h = harness({ nowIso: "2026-11-05T08:00:00-05:00", clock: CLOSED_PRE_THU, position: LONG_AXTI });
  assert.equal(JSON.parse(body("buy", "new_entry", 8)).extended_hours, true, "premise: the entry is an extended-hours limit too");
  await refused(h.post(body("buy", "new_entry", 8)), "only a sell of a held long stock position");
  await h.post(body("sell", "stop_loss", 8));
  assert.ok(h.positionReads.length === 1, "only the exit read the position");
});

test("holidays: Thanksgiving pre-market is refused; a holiday missing from the calendar is caught by the broker's next open", async () => {
  const h = harness({ nowIso: "2026-11-26T08:00:00-05:00", clock: CLOSED_THANKSGIVING, position: LONG_AXTI });
  await refused(h.post(body("sell", "stop_loss", 8)), "2026-11-26 is not a trading day");
  const missing: MarketCalendar = { holidays: new Set(), halfDays: CAL.halfDays };
  const h2 = harness({ nowIso: "2026-11-26T08:00:00-05:00", clock: CLOSED_THANKSGIVING, position: LONG_AXTI, cal: missing });
  await refused(h2.post(body("sell", "stop_loss", 8)), "the broker clock's next open is 2026-11-27");
});

test("a day the broker showed closed during regular hours has no after-hours session", async () => {
  const missing: MarketCalendar = { holidays: new Set(), halfDays: CAL.halfDays };
  const h = harness({ nowIso: "2026-11-26T11:00:00-05:00", clock: { ...CLOSED_THANKSGIVING, timestamp: "2026-11-26T11:00:00-05:00" }, position: LONG_AXTI, cal: missing });
  await refused(h.post(body("sell", "stop_loss", 11)));
  h.at("2026-11-26T17:30:00-05:00");
  await refused(h.post(body("sell", "stop_loss", 17.5)), "showed the market closed during 2026-11-26's regular hours");
});

test("only a sell of a held long STOCK position passes: options, short covers, oversells, unheld symbols and plain orders are refused", async () => {
  const h = harness({ nowIso: "2026-11-05T17:30:00-05:00", clock: CLOSED_AFTER_THU, position: LONG_AXTI });
  const optionExit = JSON.stringify({ symbol: "XLF261127P00055500", qty: "1", side: "sell", ...getOrderParams(1.6, "stop_loss", 17.5), extended_hours: true });
  await refused(h.post(optionExit), "options never trade outside regular hours");
  await refused(h.post(body("buy", "stop_loss", 17.5)), "only a sell of a held long stock position"); // a short cover
  await refused(h.post(body("sell", "stop_loss", 17.5, "AXTI", 6)), "not an exit: 5 long AXTI share(s) held for a sell of 6");
  h.hold({ symbol: "AXTI", side: "short", qty: "-5" });
  await refused(h.post(body("sell", "stop_loss", 17.5)), "not an exit: 0 long AXTI");
  h.hold(undefined);
  await refused(h.post(body("sell", "stop_loss", 17.5)), "not an exit: no long AXTI");
  // TIME-EXIT / CC-UNWIND / LIQUIDATE send plain market orders: no extended_hours, no exception
  const marketSell = JSON.stringify({ symbol: "AXTI", qty: "5", side: "sell", type: "market", time_in_force: "day" });
  h.hold(LONG_AXTI);
  const msg = await refused(h.post(marketSell));
  assert.ok(!msg.includes("; "), "a plain order's refusal carries only the clock detail");
});

test("the switch: VOLTRADE_EXTENDED_HOURS_STOCK_EXITS=0 (or false/off/no) refuses the 17:30 exit; unset or 1 allows it", async () => {
  assert.equal(EXTENDED_HOURS_STOCK_EXITS_ENV, "VOLTRADE_EXTENDED_HOURS_STOCK_EXITS");
  assert.equal(extendedStockExitsEnabled({}), true, "default ON");
  for (const off of ["0", "false", "OFF", " no "]) assert.equal(extendedStockExitsEnabled({ [EXTENDED_HOURS_STOCK_EXITS_ENV]: off }), false, off);
  for (const on of ["1", "true", "on", ""]) assert.equal(extendedStockExitsEnabled({ [EXTENDED_HOURS_STOCK_EXITS_ENV]: on }), true, on);
  const h = harness({ nowIso: "2026-11-05T17:30:00-05:00", clock: CLOSED_AFTER_THU, position: LONG_AXTI, env: { [EXTENDED_HOURS_STOCK_EXITS_ENV]: "0" } });
  await refused(h.post(body("sell", "stop_loss", 17.5)), "extended-hours stock exits are off");
  assert.equal(h.positionReads.length, 0);
  const snap = await h.gate.snapshot();
  assert.equal(snap.extendedExitsOn, false);
  assert.equal(snap.longStockExitOpen, false);
});

// ── the rule itself ─────────────────────────────────────────────────────────

test("open: every order passes (options included) with no audit; GET-style reads are never routed here", async () => {
  const h = harness({ nowIso: "2026-11-05T11:00:00-05:00", clock: OPEN_THU });
  await h.post(JSON.stringify({ symbol: "XLF261127P00055500", qty: "1", side: "buy", type: "limit", limit_price: "1.6", time_in_force: "day" }));
  await h.post(body("buy", "new_entry", 11));
  await h.gate.enforce("/v2/positions/AXTI", "DELETE", undefined);
  assert.equal(h.audits.length, 0);
  assert.equal(h.positionReads.length, 0);
});

test("closed: refused with one ORDER-BLOCKED-CLOSED audit per symbol per ET day", async () => {
  const h = harness({ nowIso: "2026-11-07T11:00:00-05:00", clock: CLOSED_SAT });
  const entry = JSON.stringify({ symbol: "QQQ", qty: "3", side: "buy", type: "limit", limit_price: "600", time_in_force: "day" });
  const m = await refused(h.post(entry), "buy 3 QQQ not sent", "closed until Mon 2026-11-09 09:30 ET");
  await refused(h.post(entry));
  await refused(h.post(entry));
  assert.equal(h.audits.length, 1, "one line per symbol per day, however often the scan retries");
  assert.equal(h.audits[0].type, ORDER_BLOCKED_AUDIT_TYPE);
  assert.ok(h.audits[0].msg.startsWith(`QQQ: ${m}`), h.audits[0].msg);
  await refused(h.post(JSON.stringify({ symbol: "GLD", qty: "1", side: "sell", type: "market", time_in_force: "day" })));
  assert.equal(h.audits.length, 2, "another symbol gets its own line");
  h.at("2026-11-08T11:00:00-05:00");
  await refused(h.post(entry));
  assert.equal(h.audits.length, 3, "the next ET day logs again");
  await refused(h.gate.enforce("/v2/positions/QQQ", "DELETE", undefined), "close of the QQQ position not sent");
});

test("unknown clock: a failed read with no earlier reading passes (a clock outage never blocks an exit)", async () => {
  for (const clock of [null, { is_open: "yes" }, ["x"], { message: "forbidden" }]) {
    const h = harness({ nowIso: "2026-11-07T11:00:00-05:00", clock });
    await h.post(body("buy", "new_entry", 11));
    assert.equal(h.audits.length, 0);
    assert.equal((await h.gate.snapshot()).state, "unknown");
  }
});

test("a failed read is extrapolated from the last good reading", async () => {
  const h = harness({ nowIso: "2026-11-07T11:00:00-05:00", clock: CLOSED_SAT });
  await refused(h.post(body("buy", "new_entry", 11)));
  h.serve(null);
  h.at("2026-11-08T11:00:00-05:00");
  const m = await refused(h.post(body("buy", "new_entry", 11)), "clock read failed", "closed until Mon 2026-11-09 09:30 ET");
  assert.ok(m.includes("last good reading"), m);
  h.at("2026-11-09T10:00:00-05:00");
  await h.post(body("buy", "new_entry", 10)); // past its next_open, before its next_close: open on its schedule
  const g = harness({ nowIso: "2026-11-05T11:00:00-05:00", clock: OPEN_THU });
  await g.post(body("buy", "new_entry", 11));
  g.serve(null);
  g.at("2026-11-05T15:59:00-05:00");
  await g.post(body("buy", "new_entry", 15.98)); // said open until 16:00
  g.at("2026-11-05T16:30:00-05:00");
  await refused(g.post(body("buy", "new_entry", 16.5)), "closed at Thu 2026-11-05 16:00 ET"); // past its close, before its next open
  g.at("2026-11-06T17:00:00-05:00");
  await g.post(body("buy", "new_entry", 17)); // past both of its times: unknown, so allowed
});

test("a reading that said open is read as closed once its own next_close has passed (the 13:00 half-day close)", () => {
  const r = parseClockReading(OPEN_HALF, Date.parse("2026-11-27T12:59:30-05:00"));
  assert.ok(r);
  assert.equal(clockStateFrom(r, r, Date.parse("2026-11-27T12:59:59-05:00")).state, "open");
  const after = clockStateFrom(r, r, Date.parse("2026-11-27T13:00:20-05:00"));
  assert.equal(after.state, "closed");
  assert.equal(after.source, "live");
});

test("parseClockReading: is_open must be a boolean; times need an offset; nanoseconds are fine", () => {
  const r = parseClockReading(OPEN_THU, 1);
  assert.ok(r && r.isOpen === true && r.nextCloseMs === Date.parse("2026-11-05T21:00:00Z"));
  assert.equal(parseClockReading({ is_open: "true" }, 1), null);
  assert.equal(parseClockReading([OPEN_THU], 1), null);
  assert.equal(parseClockReading(null, 1), null);
  const noOffset = parseClockReading({ is_open: false, next_open: "2026-11-06T09:30:00" }, 1);
  assert.ok(noOffset && noOffset.nextOpenMs === null, "an offset-less time is unreadable, never guessed in local time");
  assert.equal(clockStateFrom(null, null, 1).state, "unknown");
});

test("session evidence: an open reading records today's close; a closed one inside 09:35–13:00 marks a day without a session", () => {
  let ev = emptySessionEvidence();
  ev = observeReading(ev, parseClockReading(OPEN_HALF, Date.parse("2026-11-27T12:00:00-05:00"))!);
  assert.equal(ev.openSeenDay, "2026-11-27");
  assert.equal(ev.openSeenCloseMs, Date.parse("2026-11-27T13:00:00-05:00"));
  ev = observeReading(ev, parseClockReading(CLOSED_AFTER_HALF, Date.parse("2026-11-27T13:30:00-05:00"))!);
  assert.equal(ev.closedInSessionDay, null, "closed after the open was seen: not a holiday");
  const s = extendedSession(Date.parse("2026-11-27T16:30:00-05:00"), { holidays: new Set(), halfDays: new Set() }, null, ev);
  assert.equal(s.session, "post", "the broker's 13:00 close sets the 13:00–17:00 session");
  ev = observeReading(emptySessionEvidence(), parseClockReading({ is_open: false }, Date.parse("2026-11-26T10:00:00-05:00"))!);
  assert.equal(ev.closedInSessionDay, "2026-11-26");
});

test("snapshot: longStockExitOpen says what a sell of a held long stock would get right now", async () => {
  const cases: Array<[string, unknown, boolean, string]> = [
    ["2026-11-05T11:00:00-05:00", OPEN_THU, true, "open"],
    ["2026-11-05T17:30:00-05:00", CLOSED_AFTER_THU, true, "closed"],
    ["2026-11-05T21:00:00-05:00", CLOSED_AFTER_THU, false, "closed"],
    ["2026-11-07T17:30:00-05:00", CLOSED_SAT, false, "closed"],
    ["2026-11-27T17:30:00-05:00", CLOSED_AFTER_HALF, false, "closed"],
    ["2026-11-07T11:00:00-05:00", null, true, "unknown"],
  ];
  for (const [iso, clock, open, state] of cases) {
    const snap = await harness({ nowIso: iso, clock }).gate.snapshot();
    assert.equal(snap.state, state, iso);
    assert.equal(snap.longStockExitOpen, open, iso);
  }
});

test("describeOrder reads the JSON body bot.ts sends, a position DELETE's symbol from its path, and never throws", () => {
  const o = describeOrder("/v2/orders", "post", body("sell", "stop_loss", 17.5));
  assert.deepEqual([o.method, o.symbol, o.side, o.qty, o.extendedHours], ["POST", "AXTI", "sell", 5, true]);
  const mleg = describeOrder("/v2/orders", "POST", JSON.stringify({ order_class: "mleg", qty: "1", legs: [{ symbol: "GOOGL261127P00300000", side: "sell" }] }));
  assert.equal(mleg.symbol, "GOOGL261127P00300000");
  assert.equal(describeOrder("/v2/positions/brk.b", "DELETE", undefined).symbol, "BRK.B");
  assert.equal(describeOrder("/v2/positions", "DELETE", undefined).symbol, "ALL POSITIONS");
  assert.equal(describeOrder("/v2/orders", "POST", "{not json").symbol, "?");
});
