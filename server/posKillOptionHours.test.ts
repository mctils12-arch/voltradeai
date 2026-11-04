// POS-KILL OPTION HOURS (2026-11-04, [REPAIR]; options-bleed diagnosis bug 2).
//
// syncMonitoredPositions (server/bot.ts) runs every minute around the clock,
// so an option's kill breach was observed — and confirmed — on pre-market /
// after-hours marks, when options do not trade: the confirmed kill POSTed
// getOrderParams(current, 'stop_loss'), an extended_hours limit outside
// regular hours that Alpaca rejects for options (125 POS-KILL-ERROR rows on
// 2026-10-28/29), and a breach left pending at the close confirmed on the
// first tick of the next session — at the opening print.
//
// The fix: while the clock says CLOSED, an option position's breach is never
// kept (pending cleared before and after the observation) and the kill
// branch leaves before any order with one POS-KILL-OPTION-OFFHOURS line per
// ticker per UTC day. Stocks are unchanged; an unknown clock counts as open.
//
// Same harness as posKillMarkGuard.test.ts: the REAL per-position loop body is
// cut out of bot.ts, transpiled and run tick by tick against stubs, with the
// sync's market-open reading injected as posKillMarketOpenCurrent.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

/** Index just past the `}` closing the brace opened at `open` (quotes skipped). */
function matchBrace(src: string, open: number): number {
  assert.equal(src[open], "{");
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    const c = src[i];
    if (c === '"' || c === "'" || c === "`") {
      for (i++; i < src.length && src[i] !== c; i++) if (src[i] === "\\") i++;
      continue;
    }
    if (c === "{") depth++;
    else if (c === "}" && --depth === 0) return i + 1;
  }
  throw new Error("unbalanced braces");
}

/** Source of a top-level function in bot.ts, cut at its own closing brace; "" when absent. */
function functionSourceIfPresent(signaturePrefix: string): string {
  const start = bot.indexOf(signaturePrefix);
  if (start < 0) return "";
  return bot.slice(start, matchBrace(bot, bot.indexOf("{", start + signaturePrefix.length)));
}

function transpile(ts: string): string {
  return transformSync(ts, { loader: "ts" }).code;
}

// The per-position loop of syncMonitoredPositions, from its `for` through the
// kill/warn chain (the stop-state read that follows is outside the fix).
const syncStart = bot.indexOf("async function syncMonitoredPositions()");
assert.ok(syncStart > 0, "syncMonitoredPositions not found in bot.ts");
const loopStart = bot.indexOf("for (const pos of positions) {", syncStart);
assert.ok(loopStart > syncStart, "the per-position loop was not found inside syncMonitoredPositions");
const loopEnd = bot.indexOf("// Read stop state for this ticker", loopStart);
assert.ok(loopEnd > loopStart, "the end of the kill/warn chain was not found");
const loopBody = bot.slice(loopStart, loopEnd) + "\n}\n";

const HELPERS = [
  "function isOptionSymbol(ticker: string): boolean",
  "function isPosKillExemptOption(symbol: string, assetClass?: string): boolean",
  'function posKillThresholdPct(isOption: boolean, side: "long" | "short", stockPct: number,',
  "function posKillHasValidMark(currentPriceRaw: unknown, plpcRaw: unknown): boolean",
  "function posKillObserveBreach(pending: Map<string, PosKillBreach>, symbol: string,",
].map(functionSourceIfPresent).join("\n");

type Row = {
  symbol: string; asset_class?: string; side: string; qty: string; avg_entry_price?: string;
  current_price?: string; unrealized_plpc?: string; created_at?: string;
};
type Audit = { type: string; msg: string };
type Post = { path: string; body: Record<string, unknown> };

/** A controllable clock that stands in for `Date` inside the loop body. */
function makeClock(startIso: string) {
  let nowMs = Date.parse(startIso);
  class FakeDate extends Date {
    constructor(value?: number | string | Date) {
      if (value === undefined) super(nowMs);
      else super(value);
    }
    static now(): number { return nowMs; }
  }
  return { FakeDate, advance(ms: number) { nowMs += ms; } };
}

function makeHarness(startIso = "2026-10-29T13:30:09Z") {
  const clock = makeClock(startIso);
  const audits: Audit[] = [];
  const posts: Post[] = [];
  const gets: string[] = [];
  const openOrders: Array<Record<string, unknown>> = [];
  const env = {
    FLOOR_AND_LEG_TICKERS: new Set(["QQQ", "SVXY", "SPY", "SMH", "KWEB", "VXUS", "GLD"]),
    LEGACY_DEFENSIVE_CANDIDATE_TICKERS: new Set(["VTI", "IWM", "TLT", "IEF", "SCHP"]),
    audit: (type: string, msg: string) => { audits.push({ type, msg }); },
    alpaca: async (p: string, init?: { method?: string; body?: string }) => {
      if (init?.method === "POST") {
        posts.push({ path: p, body: JSON.parse(init.body ?? "{}") });
        return { id: `ord-${posts.length}` };
      }
      gets.push(p);
      return [];
    },
    state: {} as Record<string, unknown>,
    openOrders,
    getOrderParams: () => ({ type: "market", time_in_force: "day" }),
    notify: () => undefined,
    pkgVersion: "test",
    posKillPendingBreach: new Map<string, { firstSeenMs: number; pnlPct: number }>(),
    posKillNoMarkAuditDay: new Map<string, string>(),
    marketOpen: null as boolean | null,
  };
  const js = transpile(`
${HELPERS}
async function runTick(positions, env) {
  const { FLOOR_AND_LEG_TICKERS, LEGACY_DEFENSIVE_CANDIDATE_TICKERS, audit, alpaca, state, openOrders,
          getOrderParams, notify, pkgVersion, posKillPendingBreach, posKillNoMarkAuditDay } = env;
  const posKillMarketOpenCurrent = env.marketOpen;
  const activeTickers = new Set();
  ${loopBody}
}
`);
  const runTick = new Function("Date", `${js}\nreturn runTick;`)(clock.FakeDate) as
    (positions: Row[], e: typeof env) => Promise<void>;
  const tick = async (positions: Row[]) => { await runTick(positions, env); clock.advance(45_000); };
  const audited = (type: string) => audits.filter((a) => a.type === type);
  return { tick, clock, audits, posts, gets, openOrders, env, audited };
}

// The live rows, as /v2/positions returned them (qty/entry are illustrative).
const GOOGL_PUT_NOMARK: Row = {
  symbol: "GOOGL261127P00300000", asset_class: "us_option", side: "long", qty: "2",
  avg_entry_price: "9.41", current_price: "0", unrealized_plpc: "-1", created_at: "2026-10-27T15:12:00Z",
};
const GOOGL_CALL_REAL: Row = {
  symbol: "GOOGL261127C00355000", asset_class: "us_option", side: "long", qty: "1",
  avg_entry_price: "5.00", current_price: "0.06", unrealized_plpc: "-0.988", created_at: "2026-10-27T15:12:00Z",
};
function longOptionAt(pnlPct: number, symbol = "XLK261219P00100000"): Row {
  const entry = 4.0;
  const mark = entry * (1 + pnlPct / 100);
  return {
    symbol, asset_class: "us_option", side: "long", qty: "2", avg_entry_price: String(entry),
    current_price: mark.toFixed(2), unrealized_plpc: String(pnlPct / 100), created_at: "2026-10-27T15:12:00Z",
  };
}
function stockAt(pnlPct: number, symbol = "AXTI", createdAt = "2026-10-27T15:12:00Z"): Row {
  const entry = 20.0;
  return {
    symbol, asset_class: "us_equity", side: "long", qty: "5", avg_entry_price: String(entry),
    current_price: (entry * (1 + pnlPct / 100)).toFixed(2), unrealized_plpc: String(pnlPct / 100), created_at: createdAt,
  };
}


function shortPutAt(pnlPct: number, symbol = "IREN261211P00018000"): Row {
  const entry = 1.68;
  return {
    symbol, asset_class: "us_option", side: "short", qty: "-1", avg_entry_price: String(entry),
    current_price: (entry * (1 - pnlPct / 100)).toFixed(2), unrealized_plpc: String(pnlPct / 100), created_at: "2026-11-04T18:41:56Z",
  };
}

test("market closed: an option breach on valid marks never confirms and never reaches the POST; one OFFHOURS line per day", async () => {
  const h = makeHarness("2026-11-05T12:00:00Z"); // 07:00 ET
  h.env.marketOpen = false;
  for (let i = 0; i < 6; i++) await h.tick([longOptionAt(-80)]);
  assert.equal(h.posts.length, 0, "no liquidation order while the market is closed");
  assert.equal(h.audited("POS-KILL").length, 0);
  assert.equal(h.audited("POS-KILL-PENDING").length, 0, "an off-hours tick is not a breach observation");
  assert.equal(h.env.posKillPendingBreach.size, 0, "nothing is left pending off-hours");
  const off = h.audited("POS-KILL-OPTION-OFFHOURS");
  assert.equal(off.length, 1, "exactly one OFFHOURS line per ticker per UTC day");
  assert.ok(off[0].msg.startsWith("XLK261219P00100000: option down -80.0% (threshold: -75%) while the market is closed"), off[0].msg);
  // a sold put far past its -150% backstop: same treatment
  await h.tick([shortPutAt(-200)]);
  assert.equal(h.posts.length, 0);
  assert.equal(h.audited("POS-KILL-OPTION-OFFHOURS").length, 2, "a second ticker gets its own line");
  // next UTC day the line is allowed again
  h.clock.advance(24 * 3600_000);
  await h.tick([longOptionAt(-80)]);
  assert.equal(h.audited("POS-KILL-OPTION-OFFHOURS").length, 3);
  assert.equal(h.posts.length, 0);
});

test("a breach pending at the close does not confirm on the next session's first tick — two regular-hours ticks are needed", async () => {
  const h = makeHarness("2026-11-04T20:59:00Z"); // 15:59 ET
  h.env.marketOpen = true;
  await h.tick([longOptionAt(-80)]);
  assert.equal(h.audited("POS-KILL-PENDING").length, 1, "the last regular-hours tick observes the breach");
  assert.equal(h.posts.length, 0);
  h.env.marketOpen = false; // the close
  for (let i = 0; i < 3; i++) await h.tick([longOptionAt(-80)]);
  assert.equal(h.env.posKillPendingBreach.size, 0, "the off-hours ticks forget the pending breach");
  assert.equal(h.posts.length, 0);
  h.env.marketOpen = true; // the next open
  await h.tick([longOptionAt(-80)]);
  assert.equal(h.posts.length, 0, "the first regular-hours tick only re-observes (PENDING), it does not liquidate at the opening print");
  assert.equal(h.audited("POS-KILL-PENDING").length, 2);
  await h.tick([longOptionAt(-80)]);
  assert.equal(h.posts.length, 1, "the second consecutive regular-hours breach liquidates as before");
  assert.equal(h.posts[0].body.symbol, "XLK261219P00100000");
});

test("stocks are unchanged off-hours: two valid-mark breach ticks still liquidate", async () => {
  const h = makeHarness("2026-11-05T12:00:00Z");
  h.env.marketOpen = false;
  await h.tick([stockAt(-30)]);
  await h.tick([stockAt(-30)]);
  assert.equal(h.posts.length, 1, "a stock's kill is not gated by the option hours rule");
  assert.equal(h.posts[0].body.symbol, "AXTI");
  assert.equal(h.audited("POS-KILL-OPTION-OFFHOURS").length, 0);
});

test("an unknown clock (null) keeps today's behaviour for options: two valid breach ticks liquidate", async () => {
  const h = makeHarness("2026-11-05T15:00:00Z");
  h.env.marketOpen = null;
  await h.tick([longOptionAt(-80)]);
  await h.tick([longOptionAt(-80)]);
  assert.equal(h.posts.length, 1, "a kill is never withheld by a clock outage");
  assert.equal(h.audited("POS-KILL-OPTION-OFFHOURS").length, 0);
});

test("posKillReadMarketOpen: is_open boolean passes through, cached 60 s; a failed or malformed read is null", async () => {
  const src = functionSourceIfPresent("async function posKillReadMarketOpen(): Promise<boolean | null>");
  assert.ok(src, "posKillReadMarketOpen must exist in bot.ts");
  const make = (alpaca: (p: string) => Promise<unknown>) => {
    let nowMs = 1_000_000;
    const FakeDate = { now: () => nowMs } as unknown as DateConstructor;
    const fn = new Function("alpaca", "Date", "console",
      `const POS_KILL_CLOCK_TTL_MS = 60000; let posKillClockCache = null;\n${transpile(src)}\nreturn posKillReadMarketOpen;`,
    )(alpaca, FakeDate, { warn: () => undefined }) as () => Promise<boolean | null>;
    return { read: fn, advance: (ms: number) => { nowMs += ms; } };
  };
  let calls = 0;
  const open = make(async () => { calls++; return { is_open: false }; });
  assert.equal(await open.read(), false);
  assert.equal(await open.read(), false);
  assert.equal(calls, 1, "the clock is read once per 60 s");
  open.advance(61_000);
  await open.read();
  assert.equal(calls, 2);
  assert.equal(await make(async () => ({ is_open: "yes" })).read(), null, "a malformed is_open is unknown");
  assert.equal(await make(async () => { throw new Error("503"); }).read(), null, "a failed read is unknown");
  assert.equal(await make(async () => ({ is_open: true })).read(), true);
});

test("wiring: the clock is read once per sync before the loop, and the off-hours gate leaves the kill branch before every order path", () => {
  const sync = bot.slice(syncStart, loopStart);
  assert.ok(sync.includes("posKillMarketOpenCurrent = await posKillReadMarketOpen();"), "the sync reads the clock once, before the per-position loop");
  const loop = bot.slice(loopStart, loopEnd);
  const killAt = loop.indexOf("if (pnlPct <= POSITION_KILL_LOSS_PCT) {");
  const kill = loop.slice(killAt);
  const offAt = kill.indexOf("if (optionOffHours) {");
  const nomarkAt = kill.indexOf('if (breach === "nomark") {');
  const postAt = kill.indexOf("const killOrderResult = await alpaca(");
  assert.ok(offAt > 0 && nomarkAt > offAt && postAt > nomarkAt, "order: off-hours gate, no-mark gate, …, POST");
  const gate = kill.slice(offAt, nomarkAt);
  assert.ok(/continue;/.test(gate), "the off-hours gate leaves the branch");
  assert.ok(!/alpaca\(/.test(gate) && !/openOrders\.push/.test(gate), "the off-hours gate never calls Alpaca or registers an order");
  assert.ok(loop.includes('typeof posKillMarketOpenCurrent !== "undefined"'), "the loop body stays runnable in harnesses that do not define the reading");
});
