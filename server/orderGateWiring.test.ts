// BROKER CLOCK ORDER GATE (2026-11-05, human-directed) — the bot.ts wiring.
//
// server/brokerClockGate.test.ts pins the gate's decisions. Here, on bot.ts
// itself:
//   1. alpaca() — the one function every Node order goes through — runs the
//      gate BEFORE the request for every order submission (POST/PATCH on the
//      orders endpoint, DELETE of a position) and never for reads or cancels;
//      a refused order throws before anything is sent;
//   2. the gate reads the clock through POS-KILL's existing 60 s cache (one
//      reader, which now also keeps the clock body), and the calendar through
//      market_calendar.py;
//   3. the 60 s position sync (POS-KILL backstop, TIME-EXIT) asks the gate
//      once per sync and does not attempt what it would refuse: a stock kill
//      is held (no POS-KILL line, which the client shows as an exit) with one
//      POS-KILL-STOCK-CLOSED line per ticker per day, and sent on the first
//      tick the gate allows; TIME-EXIT (a market order) waits for the open.
// The loop body is the REAL one, cut out of bot.ts and run tick by tick
// against stubs (the posKillOptionHours.test.ts harness).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");

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

/** Source of a top-level function, from its signature to the brace closing its body. */
function functionSource(signaturePrefix: string, bodyOpener = ") {"): string {
  const start = bot.indexOf(signaturePrefix);
  assert.ok(start >= 0, `${signaturePrefix} not found in bot.ts`);
  const at = bot.indexOf(bodyOpener, start);
  assert.ok(at > start, `the body of ${signaturePrefix} was not found`);
  const open = at + bodyOpener.indexOf("{");
  return bot.slice(start, matchBrace(bot, open));
}

function transpile(ts: string): string {
  return transformSync(ts, { loader: "ts" }).code;
}

// ── 1. alpaca() runs the gate first, for order submissions only ────────────

type Enforce = (p: string, m: string, b: unknown) => Promise<void>;

function makeAlpaca(enforce: Enforce) {
  const calls: string[] = [];
  const fetchStub = async (url: string, init: { method?: string }) => {
    calls.push(`fetch ${init.method ?? "GET"} ${url.replace("https://paper-api.alpaca.markets", "")}`);
    return { ok: true, statusText: "OK", json: async () => ({ id: "o-1" }), text: async () => "" };
  };
  const orderGate = {
    enforce: async (p: string, m: string, b: unknown) => { calls.push(`gate ${m} ${p}`); await enforce(p, m, b); },
  };
  const src = functionSource("function isGatedOrderRequest(", "boolean {") + "\n" + functionSource("async function alpaca(", ") {\n");
  const alpaca = new Function("orderGate", "fetch", "ALPACA_BASE", "ALPACA_KEY", "ALPACA_SECRET",
    `${transpile(src)}\nreturn alpaca;`,
  )(orderGate, fetchStub, "https://paper-api.alpaca.markets", "key", "secret") as
    (p: string, o?: { method?: string; body?: string }) => Promise<unknown>;
  return { alpaca, calls };
}

test("alpaca(): an order submission passes the gate before anything is sent", async () => {
  const seen: unknown[] = [];
  const { alpaca, calls } = makeAlpaca(async (p, m, b) => { seen.push([p, m, b]); });
  const body = JSON.stringify({ symbol: "AXTI", qty: "5", side: "sell", type: "market", time_in_force: "day" });
  assert.deepEqual(await alpaca("/v2/orders", { method: "POST", body }), { id: "o-1" });
  assert.deepEqual(calls, ["gate POST /v2/orders", "fetch POST /v2/orders"], "the gate runs first");
  assert.deepEqual(seen, [["/v2/orders", "POST", body]], "the gate sees the exact body that would be sent");
});

test("alpaca(): a refused order throws the gate's error and nothing is sent", async () => {
  const { alpaca, calls } = makeAlpaca(async () => { throw new Error("ORDER-BLOCKED: market closed — sell 5 AXTI not sent (broker clock: closed)"); });
  await assert.rejects(alpaca("/v2/orders", { method: "POST", body: "{}" }), /^Error: ORDER-BLOCKED: market closed/);
  await assert.rejects(alpaca("/v2/positions/AXTI", { method: "DELETE" }), /ORDER-BLOCKED/);
  assert.deepEqual(calls, ["gate POST /v2/orders", "gate DELETE /v2/positions/AXTI"], "no request left the process");
});

test("alpaca(): reads and cancels never touch the gate; PATCH replaces and position closes do", async () => {
  const { alpaca, calls } = makeAlpaca(async () => undefined);
  for (const [p, m] of [["/v2/orders?status=open", undefined], ["/v2/orders/abc", undefined], ["/v2/clock", undefined],
                        ["/v2/positions", undefined], ["/v2/orders", "DELETE"], ["/v2/orders/abc", "DELETE"],
                        ["/v2/ordersx", "POST"]] as Array<[string, string | undefined]>) {
    await alpaca(p, m ? { method: m } : {});
  }
  assert.ok(calls.every((c) => c.startsWith("fetch ")), `reads and cancels are never gated: ${calls.join(" | ")}`);
  calls.length = 0;
  await alpaca("/v2/orders/abc", { method: "PATCH", body: "{}" });
  await alpaca("/v2/positions/AXTI", { method: "DELETE" });
  await alpaca("/v2/positions", { method: "DELETE" });
  assert.deepEqual(calls.filter((c) => c.startsWith("gate")), ["gate PATCH /v2/orders/abc", "gate DELETE /v2/positions/AXTI", "gate DELETE /v2/positions"]);
});

test("every Node order goes through alpaca(): no raw fetch submits an order in bot.ts or routes.ts", () => {
  const rawOrderFetch = /fetch\(\s*`[^`]*\/v2\/(orders|positions)[^`]*`\s*,\s*\{[^}]*method:\s*"(POST|PATCH|DELETE)"/;
  assert.ok(!rawOrderFetch.test(bot), "bot.ts: an order submission bypasses alpaca()");
  assert.ok(!rawOrderFetch.test(routes), "routes.ts: an order submission bypasses alpaca()");
  for (const m of routes.matchAll(/\/v2\/orders/g)) {
    assert.ok(!/method:\s*"(POST|PATCH)"/.test(routes.slice(m.index ?? 0, (m.index ?? 0) + 200)), "routes.ts only reads orders");
  }
  const alpacaFn = functionSource("async function alpaca(", ") {\n");
  const firstStatement = alpacaFn.slice(alpacaFn.indexOf("{\n") + 2).trimStart().split("\n")[0];
  assert.equal(firstStatement, "if (isGatedOrderRequest(path, opts.method)) await orderGate.enforce(path, String(opts.method), opts.body);");
});

// ── 2. one clock reader, shared with POS-KILL ──────────────────────────────

test("posKillReadMarketOpen keeps the clock body for the gate; its own answers are unchanged", async () => {
  const src = functionSource("async function posKillReadMarketOpen(", "null> {\n");
  const make = (alpaca: (p: string) => Promise<unknown>) => new Function("alpaca", "Date", "console",
    `const POS_KILL_CLOCK_TTL_MS = 60000; let posKillClockCache = null;\n${transpile(src)}\nreturn { read: posKillReadMarketOpen, cache: () => posKillClockCache };`,
  )(alpaca, { now: () => 5_000_000 }, { warn: () => undefined }) as { read: () => Promise<boolean | null>; cache: () => { at: number; isOpen: boolean | null; clock?: unknown } };
  const body = { is_open: false, timestamp: "2026-11-07T11:00:00-05:00", next_open: "2026-11-09T09:30:00-05:00", next_close: "2026-11-09T16:00:00-05:00" };
  const ok = make(async () => body);
  assert.equal(await ok.read(), false);
  assert.deepEqual(ok.cache(), { at: 5_000_000, isOpen: false, clock: body });
  const failed = make(async () => { throw new Error("Alpaca 503"); });
  assert.equal(await failed.read(), null);
  assert.deepEqual(failed.cache(), { at: 5_000_000, isOpen: null, clock: null }, "a failed read leaves no body to trust");
  const malformed = make(async () => ({ is_open: "yes" }));
  assert.equal(await malformed.read(), null);
  assert.equal(malformed.cache().clock, null);
});

test("bot.ts builds ONE order gate, on POS-KILL's cache and market_calendar.py", () => {
  assert.ok(bot.includes('import { createOrderGate } from "./brokerClockGate";'));
  assert.equal(bot.split("createOrderGate({").length - 1, 1, "one gate");
  const decl = bot.slice(bot.indexOf("const orderGate = createOrderGate({"), bot.indexOf("});", bot.indexOf("const orderGate = createOrderGate({")));
  for (const wire of [
    "readClock: async () => { await posKillReadMarketOpen(); return posKillClockCache; }",
    "calendar: () => ({ holidays: marketHolidays(), halfDays: marketHalfDays() })",
    "isOption: (symbol: string) => isOptionSymbol(symbol)",
    "audit: (type: string, message: string) => audit(type, message)",
    "getPosition: (symbol: string) => alpaca(`/v2/positions/${encodeURIComponent(symbol)}`)",
  ]) {
    assert.ok(decl.includes(wire), `missing: ${wire}`);
  }
  assert.ok(bot.indexOf("const orderGate = createOrderGate({") > bot.indexOf("async function posKillReadMarketOpen("),
    "declared after the reader it uses");
});

// ── 3. the position sync asks the gate once per sync ───────────────────────

const syncStart = bot.indexOf("async function syncMonitoredPositions()");
const loopStart = bot.indexOf("for (const pos of positions) {", syncStart);
const loopEnd = bot.indexOf("// Read stop state for this ticker", loopStart);
assert.ok(syncStart > 0 && loopStart > syncStart && loopEnd > loopStart, "the sync's per-position loop was not found");
const loopBody = bot.slice(loopStart, loopEnd) + "\n}\n";

test("wiring: the sync reads the gate once, after the clock, before the loop", () => {
  const sync = bot.slice(syncStart, loopStart);
  const clockAt = sync.indexOf("posKillMarketOpenCurrent = await posKillReadMarketOpen();");
  const gateAt = sync.indexOf("const posKillGate = await orderGate.snapshot();");
  assert.ok(clockAt > 0 && gateAt > clockAt, "the gate snapshot follows the (cached) clock read");
  assert.ok(sync.includes('posKillOrdersClosedCurrent = posKillGate.state === "closed";'));
  assert.ok(sync.includes("posKillLongStockExitOpenCurrent = posKillGate.longStockExitOpen\n" +
    '        && (!posKillOrdersClosedCurrent || getOrderParams(1, "stop_loss").extended_hours === true);'),
    "while closed, a kill passes only as getOrderParams' extended-hours limit (not the market stop it builds 09:30-16:00)");
  const loop = bot.slice(loopStart, loopEnd);
  assert.ok(loop.includes('typeof posKillOrdersClosedCurrent !== "undefined"'), "harness-safe read");
  assert.ok(loop.includes('typeof posKillLongStockExitOpenCurrent !== "undefined"'), "harness-safe read");
});

test("wiring: the stock hold sits after the option gates and before the no-mark gate, and never orders", () => {
  const loop = bot.slice(loopStart, loopEnd);
  const kill = loop.slice(loop.indexOf("if (pnlPct <= POSITION_KILL_LOSS_PCT) {"));
  const owAt = kill.indexOf("if (optionOpeningWindow) {");
  const holdAt = kill.indexOf("if (stockKillRefused) {");
  const nomarkAt = kill.indexOf('if (breach === "nomark") {');
  const postAt = kill.indexOf("const killOrderResult = await alpaca(");
  assert.ok(owAt > 0 && holdAt > owAt && nomarkAt > holdAt && postAt > nomarkAt);
  const hold = kill.slice(holdAt, nomarkAt);
  assert.ok(/continue;/.test(hold) && !/alpaca\(/.test(hold) && !/openOrders\.push/.test(hold));
  assert.ok(loop.includes("if (entryTime && !ordersClosedNow) {"), "TIME-EXIT waits while the gate refuses orders");
});

type Row = {
  symbol: string; asset_class?: string; side: string; qty: string; avg_entry_price?: string;
  current_price?: string; unrealized_plpc?: string; created_at?: string;
};

const HELPERS = [
  "function isOptionSymbol(ticker: string): boolean",
  "function isPosKillExemptOption(symbol: string, assetClass?: string): boolean",
  'function posKillThresholdPct(isOption: boolean, side: "long" | "short", stockPct: number,',
  "function posKillHasValidMark(currentPriceRaw: unknown, plpcRaw: unknown): boolean",
  "function posKillObserveBreach(pending: Map<string, PosKillBreach>, symbol: string,",
].map((sig) => functionSource(sig, "{\n").replace(/\n$/, "")).join("\n");

function makeHarness(startIso: string) {
  let nowMs = Date.parse(startIso);
  class FakeDate extends Date {
    constructor(value?: number | string | Date) {
      if (value === undefined) super(nowMs);
      else super(value);
    }
    static now(): number { return nowMs; }
  }
  const audits: Array<{ type: string; msg: string }> = [];
  const posts: Array<{ path: string; body: Record<string, unknown> }> = [];
  const env = {
    FLOOR_AND_LEG_TICKERS: new Set(["QQQ", "SVXY", "SPY", "SSO", "SMH", "KWEB", "VXUS", "GLD"]),
    LEGACY_DEFENSIVE_CANDIDATE_TICKERS: new Set(["VTI", "IWM", "TLT", "IEF", "SCHP"]),
    audit: (type: string, msg: string) => { audits.push({ type, msg }); },
    alpaca: async (p: string, init?: { method?: string; body?: string }) => {
      if (init?.method === "POST") {
        posts.push({ path: p, body: JSON.parse(init.body ?? "{}") });
        return { id: `ord-${posts.length}` };
      }
      return [];
    },
    state: {} as Record<string, unknown>,
    openOrders: [] as Array<Record<string, unknown>>,
    getOrderParams: () => ({ type: "market", time_in_force: "day" }),
    notify: () => undefined,
    pkgVersion: "test",
    posKillPendingBreach: new Map<string, { firstSeenMs: number; pnlPct: number }>(),
    posKillNoMarkAuditDay: new Map<string, string>(),
    marketOpen: null as boolean | null,
    ordersClosed: false,
    longStockExitOpen: true,
  };
  const js = transpile(`
${HELPERS}
async function runTick(positions, env) {
  const { FLOOR_AND_LEG_TICKERS, LEGACY_DEFENSIVE_CANDIDATE_TICKERS, audit, alpaca, state, openOrders,
          getOrderParams, notify, pkgVersion, posKillPendingBreach, posKillNoMarkAuditDay } = env;
  const posKillMarketOpenCurrent = env.marketOpen;
  const posKillOptionOpeningWindowCurrent = false;
  const posKillOrdersClosedCurrent = env.ordersClosed;
  const posKillLongStockExitOpenCurrent = env.longStockExitOpen;
  const activeTickers = new Set();
  ${loopBody}
}
`);
  const runTick = new Function("Date", `${js}\nreturn runTick;`)(FakeDate) as (rows: Row[], e: typeof env) => Promise<void>;
  return {
    env, audits, posts,
    tick: async (rows: Row[]) => { await runTick(rows, env); nowMs += 60_000; },
    audited: (type: string) => audits.filter((a) => a.type === type),
  };
}

function stockAt(pnlPct: number, opts: { symbol?: string; side?: "long" | "short"; createdAt?: string } = {}): Row {
  const entry = 20.0;
  const side = opts.side ?? "long";
  const mark = side === "long" ? entry * (1 + pnlPct / 100) : entry * (1 - pnlPct / 100);
  return {
    symbol: opts.symbol ?? "AXTI", asset_class: "us_equity", side, qty: side === "long" ? "5" : "-5",
    avg_entry_price: String(entry), current_price: mark.toFixed(2), unrealized_plpc: String(pnlPct / 100),
    created_at: opts.createdAt ?? "2026-11-02T15:12:00Z",
  };
}

const optionAt = (pnlPct: number): Row => ({
  symbol: "XLK261219P00100000", asset_class: "us_option", side: "long", qty: "2", avg_entry_price: "4",
  current_price: (4 * (1 + pnlPct / 100)).toFixed(2), unrealized_plpc: String(pnlPct / 100), created_at: "2026-10-27T15:12:00Z",
});

test("closed, no extended session: a stock kill is held — no order, no POS-KILL line, one hold line per day — and sent on the first tick the gate allows", async () => {
  const h = makeHarness("2026-11-07T15:00:00Z"); // Saturday
  h.env.marketOpen = false;
  h.env.ordersClosed = true;
  h.env.longStockExitOpen = false;
  for (let i = 0; i < 4; i++) await h.tick([stockAt(-30)]);
  assert.equal(h.posts.length, 0, "the gate would refuse it: nothing is attempted");
  assert.equal(h.audited("POS-KILL").length, 0, "no POS-KILL line (the client renders those as exits)");
  assert.equal(h.audited("POS-KILL-ERROR").length, 0);
  const held = h.audited("POS-KILL-STOCK-CLOSED");
  assert.equal(held.length, 1, "one line per ticker per day");
  assert.ok(held[0].msg.startsWith("AXTI: long stock down -30.0% (threshold: -25%) while the market is closed"), held[0].msg);
  h.env.marketOpen = true; // the open
  h.env.ordersClosed = false;
  h.env.longStockExitOpen = true;
  await h.tick([stockAt(-30)]);
  assert.equal(h.posts.length, 1, "the breach seen while closed was kept: the first allowed tick kills");
  assert.equal(h.posts[0].body.symbol, "AXTI");
  assert.equal(h.audited("POS-KILL").length, 1);
});

test("an extended session the gate allows: a long stock kill goes out as before (two valid-mark ticks)", async () => {
  const h = makeHarness("2026-11-05T22:30:00Z"); // Thu 17:30 ET
  h.env.marketOpen = false;
  h.env.ordersClosed = true;
  h.env.longStockExitOpen = true;
  await h.tick([stockAt(-30)]);
  await h.tick([stockAt(-30)]);
  assert.equal(h.posts.length, 1);
  assert.equal(h.audited("POS-KILL-STOCK-CLOSED").length, 0);
});

test("a short stock position is held even inside an extended session (only sells of longs pass the gate)", async () => {
  const h = makeHarness("2026-11-05T22:30:00Z");
  h.env.marketOpen = false;
  h.env.ordersClosed = true;
  h.env.longStockExitOpen = true;
  await h.tick([stockAt(-30, { side: "short" })]);
  await h.tick([stockAt(-30, { side: "short" })]);
  assert.equal(h.posts.length, 0);
  assert.equal(h.audited("POS-KILL-STOCK-CLOSED").length, 1);
  assert.ok(h.audited("POS-KILL-STOCK-CLOSED")[0].msg.startsWith("AXTI: short stock down"));
});

test("options keep their own off-hours rule; the stock hold never touches them", async () => {
  const h = makeHarness("2026-11-07T15:00:00Z");
  h.env.marketOpen = false;
  h.env.ordersClosed = true;
  h.env.longStockExitOpen = false;
  await h.tick([optionAt(-80)]);
  await h.tick([optionAt(-80)]);
  assert.equal(h.posts.length, 0);
  assert.equal(h.audited("POS-KILL-OPTION-OFFHOURS").length, 1);
  assert.equal(h.audited("POS-KILL-STOCK-CLOSED").length, 0);
});

test("TIME-EXIT (a market order) waits while the gate refuses orders, and closes the stale position once open", async () => {
  const h = makeHarness("2026-11-07T15:00:00Z");
  const stale = stockAt(0.5, { createdAt: "2026-10-01T15:00:00Z" });
  h.env.ordersClosed = true;
  h.env.longStockExitOpen = true; // even inside an extended session: a market order never qualifies
  await h.tick([stale]);
  await h.tick([stale]);
  assert.equal(h.posts.length, 0);
  assert.equal(h.audited("TIME-EXIT").length, 0, "no 'closing stale' line for an order that cannot be sent");
  h.env.ordersClosed = false;
  await h.tick([stale]);
  assert.equal(h.posts.length, 1);
  assert.equal(h.audited("TIME-EXIT").length, 1);
});

test("harnesses that never define the gate flags keep today's behaviour (a stock kill on two ticks)", async () => {
  const h = makeHarness("2026-11-07T15:00:00Z");
  const js = transpile(`
${HELPERS}
async function runTick(positions, env) {
  const { FLOOR_AND_LEG_TICKERS, LEGACY_DEFENSIVE_CANDIDATE_TICKERS, audit, alpaca, state, openOrders,
          getOrderParams, notify, pkgVersion, posKillPendingBreach, posKillNoMarkAuditDay } = env;
  const activeTickers = new Set();
  ${loopBody}
}
`);
  const runTick = new Function(`${js}\nreturn runTick;`)() as (rows: Row[], e: typeof h.env) => Promise<void>;
  await runTick([stockAt(-30)], h.env);
  await runTick([stockAt(-30)], h.env);
  assert.equal(h.posts.length, 1);
});
