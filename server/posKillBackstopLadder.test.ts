// BACKSTOP LADDER (2026-11-05, [RULE-REVIEW]; human-approved change to the
// frozen order path): POS-KILL's option exit in syncMonitoredPositions
// (server/bot.ts) no longer posts one market order. It runs
// posKillStartOptionLadder -> limitLadder.runLadder: a limit at the bid (sell)
// / ask (buy), one tick further every step, and ONE market order for the
// remainder only after the hard timeout. Stocks keep their single order.
//
// Same harness as posKillMarkGuard.test.ts — the REAL per-position loop body
// and the REAL ladder starter are cut out of bot.ts, transpiled and run against
// stubs — plus a simulated broker behind the alpaca() stub (orders, fills,
// cancels with alpaca()'s real throw on Alpaca's 204) and a fake clock that
// drives the ladder's sleeps, so whole ladders run end to end in milliseconds.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";
import { runLadder, ladderQuoteFromAlpaca, LADDER_DEFAULTS, BACKSTOP_LADDER_QUOTE_TIMEOUT_MS } from "./limitLadder";

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
function functionSource(signaturePrefix: string): string {
  const start = bot.indexOf(signaturePrefix);
  assert.ok(start >= 0, `${signaturePrefix} not found in bot.ts`);
  return bot.slice(start, matchBrace(bot, bot.indexOf("{", start + signaturePrefix.length)));
}
const transpile = (ts: string): string => transformSync(ts, { loader: "ts" }).code;

const syncStart = bot.indexOf("async function syncMonitoredPositions()");
const loopStart = bot.indexOf("for (const pos of positions) {", syncStart);
const loopEnd = bot.indexOf("// Read stop state for this ticker", loopStart);
assert.ok(syncStart > 0 && loopStart > syncStart && loopEnd > loopStart, "the per-position loop was not found");
const loopBody = bot.slice(loopStart, loopEnd) + "\n}\n";
const HELPERS = [
  "function isOptionSymbol(ticker: string): boolean",
  "function isPosKillExemptOption(symbol: string, assetClass?: string): boolean",
  'function posKillThresholdPct(isOption: boolean, side: "long" | "short", stockPct: number,',
  "function posKillHasValidMark(currentPriceRaw: unknown, plpcRaw: unknown): boolean",
  "function posKillObserveBreach(pending: Map<string, PosKillBreach>, symbol: string,",
].map(functionSource).join("\n");
const STARTER_SIG = "async function posKillStartOptionLadder(";
const STARTER = functionSource(STARTER_SIG);

type Row = { symbol: string; asset_class?: string; side: string; qty: string; avg_entry_price?: string; current_price?: string; unrealized_plpc?: string; created_at?: string };
type SimOrder = { id: string; symbol: string; side: string; qty: number; type: string; limit_price?: string; status: string; filled_qty: number; filled_avg_price: number | null };
type Op = { op: "POST" | "DELETE"; t: number; id: string; body?: Record<string, string> };
const TERMINAL = new Set(["filled", "canceled", "expired", "rejected"]);

function makeHarness(startIso = "2026-11-05T15:00:00Z") {
  let nowMs = Date.parse(startIso);
  class FakeDate extends Date {
    constructor(value?: number | string | Date) { if (value === undefined) super(nowMs); else super(value); }
    static now(): number { return nowMs; }
  }
  const t0 = nowMs;
  const timers: Array<{ at: number; resolve: () => void }> = [];
  const audits: Array<{ type: string; msg: string }> = [];
  const notes: Array<{ type: string; msg: string }> = [];
  const ops: Op[] = [];
  const orders = new Map<string, SimOrder>();
  const openOrders: Array<Record<string, unknown>> = [];
  let n = 0;
  let maxLive = 0;
  const copy = (o: SimOrder) => ({
    id: o.id, symbol: o.symbol, side: o.side, qty: String(o.qty), type: o.type, limit_price: o.limit_price, status: o.status,
    filled_qty: String(o.filled_qty), filled_avg_price: o.filled_avg_price === null ? null : String(o.filled_avg_price),
  });
  const env = {
    FLOOR_AND_LEG_TICKERS: new Set(["QQQ", "SVXY", "SPY", "SMH", "KWEB", "VXUS", "GLD", "SSO"]),
    LEGACY_DEFENSIVE_CANDIDATE_TICKERS: new Set(["VTI", "IWM", "TLT", "IEF", "SCHP"]),
    audit: (type: string, msg: string) => { audits.push({ type, msg }); },
    notify: (type: string, msg: string) => { notes.push({ type, msg }); },
    state: {} as Record<string, unknown>,
    openOrders,
    getOrderParams: () => ({ type: "market", time_in_force: "day" }),
    pkgVersion: "test",
    posKillPendingBreach: new Map<string, { firstSeenMs: number; pnlPct: number }>(),
    posKillNoMarkAuditDay: new Map<string, string>(),
    posKillLadderInFlight: new Map<string, number>(),
    runLadder,
    ladderConfig: LADDER_DEFAULTS,
    quotes: new Map<string, { bid: number; ask: number }>(),
    /** Live positions, Alpaca's signed qty (short is negative); a missing symbol answers 404. */
    positions: new Map<string, number>(),
    /** When set, every quote read waits for it (pauses a ladder between steps). */
    quoteGate: null as Promise<void> | null,
    /** Simulated market: called on every order read and cancel request. */
    react: null as ((o: SimOrder, why: "read" | "cancel") => void) | null,
    /** When it returns an Error for a POST body, alpaca() throws it instead of placing the order. */
    postError: null as ((body: Record<string, string>) => Error | null) | null,
    sleep: (ms: number) => new Promise<void>((resolve) => { timers.push({ at: nowMs + ms, resolve }); }),
    alpaca: async (p: string, init?: { method?: string; body?: string }): Promise<unknown> => {
      if (init?.method === "POST") {
        const body = JSON.parse(init.body ?? "{}") as Record<string, string>;
        const err = env.postError?.(body);
        const id = err ? "" : `ord-${++n}`;
        ops.push({ op: "POST", t: nowMs, id, body });
        if (err) throw err;
        const o: SimOrder = { id, symbol: body.symbol, side: body.side, qty: Number(body.qty), type: body.type, limit_price: body.limit_price, status: "new", filled_qty: 0, filled_avg_price: null };
        orders.set(id, o);
        maxLive = Math.max(maxLive, [...orders.values()].filter((x) => x.symbol === o.symbol && !TERMINAL.has(x.status)).length);
        return copy(o);
      }
      if (init?.method === "DELETE") {
        const id = p.split("/").pop() ?? "";
        ops.push({ op: "DELETE", t: nowMs, id });
        const o = orders.get(id);
        if (!o) throw new Error("Alpaca 404: order not found");
        env.react?.(o, "cancel");
        if (o.status === "filled") throw new Error("Alpaca 422: order is not cancelable");
        o.status = "canceled";
        throw new SyntaxError("Unexpected end of JSON input"); // alpaca() on Alpaca's 204 No Content
      }
      if (p.startsWith("/v2/orders?")) {
        const q = new URLSearchParams(p.slice(p.indexOf("?") + 1));
        return [...orders.values()]
          .filter((o) => o.symbol === q.get("symbols") && o.side === q.get("side") && !TERMINAL.has(o.status))
          .map(copy);
      }
      const pos = /^\/v2\/positions\/([^/?]+)$/.exec(p);
      if (pos) {
        const q = env.positions.get(pos[1]);
        if (q === undefined) throw new Error('Alpaca 404: {"code":40410000,"message":"position does not exist"}');
        return { symbol: pos[1], qty: String(q) };
      }
      const m = /^\/v2\/orders\/([^/?]+)$/.exec(p);
      if (m) {
        const o = orders.get(m[1]);
        if (!o) throw new Error("Alpaca 404: order not found");
        env.react?.(o, "read");
        return copy(o);
      }
      return [];
    },
  };
  const js = transpile(`
${HELPERS}
async function runTick(positions, env) {
  const { FLOOR_AND_LEG_TICKERS, LEGACY_DEFENSIVE_CANDIDATE_TICKERS, audit, alpaca, state, openOrders,
          getOrderParams, notify, pkgVersion, posKillPendingBreach, posKillNoMarkAuditDay,
          posKillLadderInFlight, runLadder } = env;
  const BACKSTOP_LADDER_CONFIG = env.ladderConfig;
  const posKillOptionQuote = async (symbol) => { if (env.quoteGate) await env.quoteGate; return env.quotes.get(symbol) ?? null; };
  const posKillLadderSleep = env.sleep;
  ${STARTER}
  const activeTickers = new Set();
  ${loopBody}
}
`);
  const runTick = new Function("Date", `${js}\nreturn runTick;`)(FakeDate) as (positions: Row[], e: typeof env) => Promise<void>;
  const flush = async () => { for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r)); };
  const tick = async (positions: Row[]) => { await runTick(positions, env); await flush(); };
  /** Advance the fake clock up to `ms`, waking each ladder sleep in time order. */
  const run = async (ms: number) => {
    const end = nowMs + ms;
    await flush();
    for (;;) {
      timers.sort((a, b) => a.at - b.at);
      const next = timers[0];
      if (!next || next.at > end) break;
      timers.shift();
      nowMs = Math.max(nowMs, next.at);
      next.resolve();
      await flush();
    }
    nowMs = Math.max(nowMs, end);
  };
  const posts = () => ops.filter((o) => o.op === "POST");
  const audited = (type: string) => audits.filter((a) => a.type === type);
  return { env, tick, run, ops, posts, orders, openOrders, audits, audited, notes, t0, maxLive: () => maxLive };
}

const PUT = "XLK261219P00100000";
function longPutAt(pnlPct: number): Row {
  const entry = 4.0;
  return { symbol: PUT, asset_class: "us_option", side: "long", qty: "2", avg_entry_price: String(entry), current_price: (entry * (1 + pnlPct / 100)).toFixed(2), unrealized_plpc: String(pnlPct / 100), created_at: "2026-11-02T15:12:00Z" };
}
const SHORT_PUT = "XLF261127P00055500";
function shortPutAt(pnlPct: number): Row {
  const entry = 0.72;
  return { symbol: SHORT_PUT, asset_class: "us_option", side: "short", qty: "-1", avg_entry_price: String(entry), current_price: (entry * (1 - pnlPct / 100)).toFixed(2), unrealized_plpc: String(pnlPct / 100), created_at: "2026-11-02T15:12:00Z" };
}
function stockAt(pnlPct: number): Row {
  return { symbol: "AXTI", asset_class: "us_equity", side: "long", qty: "5", avg_entry_price: "20", current_price: (20 * (1 + pnlPct / 100)).toFixed(2), unrealized_plpc: String(pnlPct / 100), created_at: "2026-11-02T15:12:00Z" };
}
async function killPut(h: ReturnType<typeof makeHarness>): Promise<void> {
  h.env.positions.set(PUT, 2);
  await h.tick([longPutAt(-80)]); // PENDING
  await h.tick([longPutAt(-80)]); // confirmed: the ladder starts
}

test("an option kill opens with a limit SELL at the bid — never a direct market order — registered for the sweeper with the ML exit-fill contract", async () => {
  const h = makeHarness();
  h.env.quotes.set(PUT, { bid: 0.78, ask: 0.86 });
  await killPut(h);
  assert.equal(h.posts().length, 1);
  assert.deepEqual(h.posts()[0].body, { symbol: PUT, side: "sell", time_in_force: "day", qty: "2", type: "limit", limit_price: "0.78" });
  assert.equal(h.audited("POS-KILL").length, 1, "the kill line is unchanged");
  assert.ok(h.audited("POS-KILL")[0].msg.includes("position down -80.0% — forcing liquidation (threshold: -75%)"));
  assert.deepEqual(h.audited("POS-KILL-LADDER").map((a) => a.msg), [`${PUT}: step 0: limit sell 2 @ $0.78 (bid 0.78 / ask 0.86) — order ord-1`]);
  assert.equal(h.openOrders.length, 1);
  const tracked = h.openOrders[0] as Record<string, unknown>;
  assert.deepEqual(
    { orderId: tracked.orderId, ticker: tracked.ticker, side: tracked.side, qty: tracked.qty, limitPrice: tracked.limitPrice, isExit: tracked.isExit, score: tracked.score },
    { orderId: "ord-1", ticker: PUT, side: "sell", qty: 2, limitPrice: 0.78, isExit: true, score: 0 },
  );
  assert.deepEqual(tracked.pendingExitFill, { ticker: PUT, exitSide: "sell", pnlPct: -80, exitReason: "position_kill", codeVersion: "test" },
    "the sweeper records the fill at Alpaca's confirmed price once this order fills (KNOWN BROKEN #35 contract, unchanged)");
  assert.ok(h.env.posKillLadderInFlight.has(PUT), "the ladder is in flight");
  assert.equal(h.notes.filter((x) => x.type === "alert").length, 1);
  assert.ok(h.notes[0].msg.startsWith(`POSITION KILL: ${PUT} at -80.0% — closing with a limit ladder (market order after 120s unfilled)`));

  // a short option is bought back from the ask
  const s = makeHarness();
  s.env.quotes.set(SHORT_PUT, { bid: 1.71, ask: 1.88 });
  await s.tick([shortPutAt(-160)]);
  await s.tick([shortPutAt(-160)]);
  assert.deepEqual(s.posts()[0].body, { symbol: SHORT_PUT, side: "buy", time_in_force: "day", qty: "1", type: "limit", limit_price: "1.88" });
});

test("TIMEOUT FALLBACK end to end: an option kill that never fills walks 8 limit steps and sends exactly ONE market order, at the 120 s hard timeout", async () => {
  const h = makeHarness();
  h.env.quotes.set(PUT, { bid: 0.78, ask: 0.86 });
  await killPut(h);
  const start = h.posts()[0].t;
  await h.run(10 * 60_000);
  const posts = h.posts();
  const limits = posts.filter((p) => p.body.type === "limit");
  const markets = posts.filter((p) => p.body.type === "market");
  assert.deepEqual(limits.map((p) => p.body.limit_price), ["0.78", "0.73", "0.68", "0.63", "0.58", "0.53", "0.48", "0.43"]);
  assert.deepEqual(limits.map((p) => (p.t - start) / 1000), [0, 15, 30, 45, 60, 75, 90, 105]);
  assert.equal(markets.length, 1, "exactly one market order");
  assert.deepEqual(markets[0].body, { symbol: PUT, side: "sell", time_in_force: "day", qty: "2", type: "market" });
  assert.ok(markets[0].t - start >= 120_000, "never before the hard timeout");
  // strictly alternating: every limit is cancelled before the next order goes out
  assert.deepEqual(h.ops.map((o) => o.op), [...Array(8).fill(["POST", "DELETE"]).flat(), "POST"]);
  assert.equal(h.maxLive(), 1, "never two live orders for the position");
  assert.equal(h.openOrders.length, 9, "every ladder order is registered for the sweeper");
  assert.ok(h.openOrders.every((o) => o.isExit === true && (o.pendingExitFill as { exitReason: string }).exitReason === "position_kill"));
  assert.equal(h.openOrders[8].limitPrice, 0, "the market order is registered like the old market kill (limitPrice 0)");
  const ladder = h.audited("POS-KILL-LADDER").map((a) => a.msg);
  assert.equal(ladder[ladder.length - 1], `${PUT}: no full fill in 120s: market order for the remaining 2 after 8 limit step(s); 0/2 filled`);
  assert.ok(ladder.includes(`${PUT}: hard timeout 120s: market sell 2 — order ord-9`));
  assert.equal(h.env.posKillLadderInFlight.size, 0, "the in-flight entry is released when the ladder ends");
  assert.equal(h.audited("POS-KILL-ERROR").length, 0);
});

test("a sync tick while the ladder works never starts a second ladder or order — even between a cancel and the next step, when the broker shows no open order", async () => {
  const h = makeHarness();
  h.env.quotes.set(PUT, { bid: 0.78, ask: 0.86 });
  await killPut(h);
  let release: () => void = () => undefined;
  h.env.quoteGate = new Promise<void>((r) => { release = r; });
  await h.run(15_000); // step 0 rests, is cancelled at its deadline; step 1 waits on its quote
  assert.equal(h.orders.get("ord-1")?.status, "canceled");
  assert.equal(h.posts().length, 1, "step 1 has not gone out yet");
  h.openOrders.length = 0; // the sweeper pruned the cancelled order; the broker lists nothing open
  await h.tick([longPutAt(-80)]);
  const skip = h.audited("POS-KILL-SKIP");
  assert.equal(skip.length, 1);
  assert.ok(skip[0].msg.startsWith(`${PUT}: down -80.0% but its backstop limit ladder is still working (started 15s ago)`), skip[0].msg);
  assert.equal(h.posts().length, 1, "no second order and no second ladder");
  assert.equal(h.audited("POS-KILL").length, 1, "no second POS-KILL line");
  release();
  h.env.quoteGate = null;
  await h.run(1);
  assert.equal(h.posts().length, 2);
  assert.equal(h.posts()[1].body.limit_price, "0.73", "the same ladder continues at step 1");
});

test("a ladder that fills ends cleanly: the outcome is audited, the in-flight entry is released, no market order", async () => {
  const h = makeHarness();
  h.env.quotes.set(PUT, { bid: 0.78, ask: 0.86 });
  h.env.react = (o) => { o.status = "filled"; o.filled_qty = o.qty; o.filled_avg_price = 0.79; };
  await killPut(h);
  await h.run(60_000);
  assert.equal(h.posts().length, 1);
  assert.equal(h.ops.filter((o) => o.op === "DELETE").length, 0);
  const ladder = h.audited("POS-KILL-LADDER").map((a) => a.msg);
  assert.deepEqual(ladder.slice(1), [
    `${PUT}: step 0: order ord-1 filled, 2/2 filled at $0.79`,
    `${PUT}: 2/2 filled at avg $0.79 by limit after 1 step(s); no market order`,
  ]);
  assert.equal(h.env.posKillLadderInFlight.size, 0);
});

test("the alpaca() adapter: an 'Alpaca 4xx' throw is a rejection (that step is skipped, the walk continues); a timeout is an unknown outcome (POS-KILL-ERROR, alert, no market order)", async () => {
  const r = makeHarness();
  r.env.quotes.set(PUT, { bid: 0.78, ask: 0.86 });
  r.env.postError = (b) => (b.limit_price === "0.78" ? new Error("Alpaca 422: {\"message\":\"invalid limit_price\"}") : null);
  r.env.react = (o) => { if (o.limit_price === "0.73") { o.status = "filled"; o.filled_qty = o.qty; o.filled_avg_price = 0.75; } };
  await killPut(r);
  assert.equal(r.openOrders.length, 0, "a rejected order is never registered");
  await r.run(60_000);
  assert.deepEqual(r.posts().map((p) => [p.body.limit_price, p.id]), [["0.78", ""], ["0.73", "ord-1"]]);
  assert.ok(r.audited("POS-KILL-LADDER").some((a) => a.msg.startsWith(`${PUT}: step 0: limit sell 2 @ $0.78 rejected (Alpaca 422:`)));
  assert.equal(r.audited("POS-KILL-LADDER").pop()?.msg, `${PUT}: 2/2 filled at avg $0.75 by limit after 2 step(s); no market order`);

  const u = makeHarness();
  u.env.quotes.set(PUT, { bid: 0.78, ask: 0.86 });
  u.env.postError = (b) => (b.limit_price === "0.73" ? new Error("This operation was aborted") : null);
  await killPut(u);
  await u.run(10 * 60_000);
  assert.deepEqual(u.posts().map((p) => p.body.type), ["limit", "limit"], "no order after the unknown outcome — and no market order");
  const err = u.audited("POS-KILL-ERROR");
  assert.equal(err.length, 1);
  assert.ok(err[0].msg.startsWith(`${PUT}: backstop ladder stopped — step 1: limit sell 2 @ $0.73 failed with an unknown outcome (This operation was aborted)`), err[0].msg);
  assert.ok(u.notes.some((x) => x.type === "alert" && x.msg.startsWith(`POSITION KILL: ${PUT} backstop ladder stopped — step 1:`)));
  assert.equal(u.env.posKillLadderInFlight.size, 0);
});

test("the live position caps every later order: closed elsewhere between steps -> nothing more is sent; partly closed -> the next order shrinks", async () => {
  const gone = makeHarness();
  gone.env.quotes.set(PUT, { bid: 0.78, ask: 0.86 });
  await killPut(gone);
  gone.env.positions.delete(PUT); // e.g. options_manager's own close filled while step 0 rested
  await gone.run(10 * 60_000);
  assert.equal(gone.posts().length, 1, "no order for a position that is gone: a sell could open a short put");
  const err = gone.audited("POS-KILL-ERROR");
  assert.equal(err.length, 1);
  assert.ok(err[0].msg.startsWith(`${PUT}: backstop ladder stopped — step 1: the position is no longer held — closed outside the ladder; nothing more sent`), err[0].msg);
  assert.equal(gone.env.posKillLadderInFlight.size, 0);

  const part = makeHarness();
  part.env.quotes.set(PUT, { bid: 0.78, ask: 0.86 });
  await killPut(part);
  part.env.positions.set(PUT, 1);
  await part.run(10 * 60_000);
  assert.deepEqual(part.posts().map((x) => x.body.qty), ["2", "1", "1", "1", "1", "1", "1", "1", "1"], "steps 1-7 and the market order carry the live 1");
  assert.equal(part.posts()[8].body.type, "market");
});

test("stocks keep the single getOrderParams(current, 'stop_loss') order: no ladder", async () => {
  const h = makeHarness();
  await h.tick([stockAt(-30)]);
  await h.tick([stockAt(-30)]);
  assert.equal(h.posts().length, 1);
  assert.deepEqual(h.posts()[0].body, { symbol: "AXTI", qty: "5", side: "sell", type: "market", time_in_force: "day" });
  assert.equal(h.audited("POS-KILL-LADDER").length, 0);
  assert.equal(h.env.posKillLadderInFlight.size, 0);
  assert.equal((h.openOrders[0].pendingExitFill as { exitReason: string }).exitReason, "position_kill");
});

test("wiring: the option branch runs after every guard and the POS-KILL line, before the stock order, and holds no order of its own", () => {
  const loop = bot.slice(loopStart, loopEnd);
  const kill = loop.slice(loop.indexOf("if (pnlPct <= POSITION_KILL_LOSS_PCT) {"), loop.indexOf("} else if (pnlPct <= POSITION_WARN_LOSS_PCT) {"));
  const at = (s: string) => { const i = kill.indexOf(s); assert.ok(i > 0, `${s} not found in the kill branch`); return i; };
  const order = [
    'if (breach === "pending") {',
    "const trackedExit = openOrders.find(",
    "const ladderSince = typeof posKillLadderInFlight !== \"undefined\" ? posKillLadderInFlight.get(ticker) : undefined;",
    "await alpaca(`/v2/orders?status=open&symbols=${ticker}&side=${closeSide}`)",
    'audit("POS-KILL", ',
    "if (isOptionPos) {",
    "getOrderParams(current, 'stop_loss')",
  ].map(at);
  assert.deepEqual([...order].sort((a, b) => a - b), order, "pending gate < tracked-order guard < ladder guard < broker guard < POS-KILL line < option branch < stock order");
  const optionBranch = kill.slice(at("if (isOptionPos) {"), at("getOrderParams(current, 'stop_loss')"));
  const code = optionBranch.split("\n").filter((l) => !l.trim().startsWith("//")).join("\n");
  assert.match(code, /if \(isOptionPos\) \{\s*await posKillStartOptionLadder\(ticker, closeSide, qty, current, pnlPct\);\s*continue;\s*\}/);
  assert.ok(!/alpaca\(|"market"|getOrderParams/.test(code), "the option branch sends nothing itself");
  assert.equal(bot.split("if (pnlPct <= POSITION_KILL_LOSS_PCT) {").length - 1, 1, "the kill-branch opener is untouched (sibling tests slice on it)");
});

test("wiring: the starter's only market order is the ladder's timeout fallback; the config is read once from env at module load", () => {
  const markets = STARTER.match(/type: "market"/g) ?? [];
  assert.equal(markets.length, 1, "one market order body in the starter");
  assert.match(STARTER, /submitMarket: \(q\) => send\(\{ qty: String\(q\), type: "market" \}\)/);
  assert.match(STARTER, /\}, \{ side: closeSide, qty, fallbackPrice: mark \}, BACKSTOP_LADDER_CONFIG\)/);
  const moduleLevel = bot.slice(0, bot.indexOf("export function registerBotRoutes("));
  assert.ok(moduleLevel.includes("const { config: BACKSTOP_LADDER_CONFIG, rejected: backstopLadderEnvRejected } = ladderConfigFromEnv(process.env);"));
  assert.ok(moduleLevel.includes("if (backstopLadderEnvRejected.length > 0) {"), "an ignored override is logged, not silent");
  assert.ok(bot.indexOf("const posKillLadderInFlight: Map<string, number> = new Map();") < syncStart);
});

test("posKillOptionQuote: Alpaca's latest option quote on the indicative feed (ALPACA_OPTIONS_FEED overrides); any failure is null", async () => {
  const src = functionSource("async function posKillOptionQuote(occ: string): Promise<LadderQuote | null>");
  const make = (env: Record<string, string>, fetchImpl: (url: string, init: { headers: Record<string, string> }) => Promise<unknown>) => {
    const warned: string[] = [];
    const fn = new Function("process", "fetch", "ALPACA_KEY", "ALPACA_SECRET", "AbortSignal", "BACKSTOP_LADDER_QUOTE_TIMEOUT_MS", "ladderQuoteFromAlpaca", "console",
      `${transpile(src)}\nreturn posKillOptionQuote;`,
    )({ env }, fetchImpl, "k", "s", AbortSignal, BACKSTOP_LADDER_QUOTE_TIMEOUT_MS, ladderQuoteFromAlpaca, { warn: (m: string) => warned.push(m) }) as (occ: string) => Promise<unknown>;
    return { fn, warned };
  };
  const seen: Array<{ url: string; headers: Record<string, string> }> = [];
  const ok = make({}, async (url, init) => {
    seen.push({ url, headers: init.headers });
    return { ok: true, status: 200, json: async () => ({ quotes: { [PUT]: { bp: 0.78, ap: 0.86, bs: 5, as: 9 } } }) };
  });
  assert.deepEqual(await ok.fn(PUT), { bid: 0.78, ask: 0.86 });
  assert.equal(seen[0].url, `https://data.alpaca.markets/v1beta1/options/quotes/latest?symbols=${PUT}&feed=indicative`);
  assert.deepEqual(seen[0].headers, { "APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s" });
  const opra = make({ ALPACA_OPTIONS_FEED: "opra" }, async (url) => { seen.push({ url, headers: {} }); return { ok: true, status: 200, json: async () => ({ quotes: {} }) }; });
  assert.equal(await opra.fn(PUT), null, "no quote for the symbol");
  assert.ok(seen[1].url.endsWith("&feed=opra"));
  const denied = make({}, async () => ({ ok: false, status: 403, json: async () => ({}) }));
  assert.equal(await denied.fn(PUT), null);
  assert.ok(denied.warned[0].includes("HTTP 403"), "a failed read is logged");
  const thrown = make({}, async () => { throw new Error("socket hang up"); });
  assert.equal(await thrown.fn(PUT), null);
  assert.ok(thrown.warned[0].includes("socket hang up"));
});
