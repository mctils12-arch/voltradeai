// ENTRY SHADOW MODE (human directive 2026-10-28, paper account): "move the
// options sleeve and stock entries to shadow mode. Keep the floor basket and
// all exits live." The flag reader and the day book are pure and tested
// directly; the bot.ts gates live inside the registerBotRoutes closure and run
// against Alpaca, so their wiring is pinned on the source text (same
// convention as shadowGateRelabel.test.ts). Python half: test_entry_shadow.py
// (which also pins this module's default against system_config BASE_CONFIG).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  ENTRY_SHADOW_DEFAULTS, entryShadowRaw, parseShadowSleeves, isEntryShadowed, entryShadowStatus,
  ShadowEntryBook, etDay, shadowEntryAuditLine, shadowEntryRow, SHADOW_ROWS_CAP,
} from "./entryShadow";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function slice(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

// ── flag ─────────────────────────────────────────────────────────────────────
test("default (env absent) shadows stocks and options, source config", () => {
  assert.equal(ENTRY_SHADOW_DEFAULTS.VOLTRADE_SHADOW_ENTRIES, "stocks,options");
  assert.deepEqual(entryShadowStatus({}), { stocks: true, options: true, source: "config" });
  assert.equal(isEntryShadowed("stocks", {}), true);
  assert.equal(isEntryShadowed("options", {}), true);
});

test("a PRESENT env var wins — every off value turns both sleeves live", () => {
  for (const v of ["off", "0", "", "none", " OFF ", "false"]) {
    const env = { VOLTRADE_SHADOW_ENTRIES: v };
    assert.deepEqual(entryShadowRaw(env), { value: v, source: "env" });
    assert.equal(isEntryShadowed("stocks", env), false, v);
    assert.equal(isEntryShadowed("options", env), false, v);
    assert.deepEqual(entryShadowStatus(env), { stocks: false, options: false, source: "env" });
  }
});

test("partial values narrow; unknown tokens (e.g. floor) shadow nothing", () => {
  assert.deepEqual([...parseShadowSleeves("stocks")], ["stocks"]);
  assert.deepEqual([...parseShadowSleeves("options")], ["options"]);
  assert.deepEqual([...parseShadowSleeves("Options, Stocks")].sort(), ["options", "stocks"]);
  assert.deepEqual([...parseShadowSleeves("all")].sort(), ["options", "stocks"]);
  assert.deepEqual([...parseShadowSleeves("floor")], []);
  assert.equal(isEntryShadowed("options", { VOLTRADE_SHADOW_ENTRIES: "stocks" }), false);
});

test("the day book is per sleeve+ticker and resets on a new ET day", () => {
  const book = new ShadowEntryBook();
  const t0 = Date.parse("2026-10-28T15:00:00Z"); // 11:00 ET
  assert.equal(book.add("stocks", "nvda", t0), true, "first time today -> audit");
  assert.equal(book.add("stocks", "NVDA", t0 + 60_000), false, "rate-limited per ticker per day");
  assert.equal(book.has("options", "NVDA", t0), false, "sleeves are separate");
  assert.equal(book.has("stocks", "NVDA", Date.parse("2026-10-29T03:30:00Z")), true, "23:30 ET is still 10-28");
  assert.equal(book.has("stocks", "NVDA", Date.parse("2026-10-29T14:00:00Z")), false, "next ET day");
  assert.equal(etDay(t0), "2026-10-28");
});

// ── Bot page: today's would-be entries (GET /api/bot/status) ────────────────
test("shadowEntryRow shapes partial input honestly (null, never a fake zero)", () => {
  const t0 = Date.parse("2026-10-28T15:00:00Z");
  assert.deepEqual(shadowEntryRow("stocks", "nvda", t0, { side: "BUY", notional: 1500.456, source: "executeTrades.etf", instrument: "nvdl" }), {
    time: "2026-10-28T15:00:00.000Z", sleeve: "stocks", ticker: "NVDA", side: "buy", notional: 1500.46, source: "executeTrades.etf", instrument: "NVDL",
  });
  const bare = shadowEntryRow("options", "SVXY", t0, { source: "bot_engine" });
  assert.equal(bare.side, null);
  assert.equal(bare.notional, null, "Python reported no size -> null, not $0");
  assert.equal(shadowEntryRow("options", "SPY", t0, { notional: 0 }).notional, null);
  assert.equal(shadowEntryRow("options", "SPY", t0, { notional: "x" }).notional, null);
  assert.equal(shadowEntryRow("stocks", "AAPL", t0, { instrument: "aapl" }).instrument, null, "same as ticker -> null");
  assert.equal(shadowEntryRow("stocks", "AAPL", t0).source, "unknown");
});

test("the day book lists today's first-time entries newest first, capped, exact count, ET reset", () => {
  const book = new ShadowEntryBook();
  const t0 = Date.parse("2026-10-28T14:00:00Z");
  assert.deepEqual(book.today(t0), { day: "2026-10-28", count: 0, entries: [] });
  book.add("stocks", "NVDA", t0, { side: "buy", notional: 1000, source: "executeTrades.stock" });
  book.add("stocks", "NVDA", t0 + 1000, { side: "buy", notional: 9999, source: "dup" }); // repeat: not a new row
  book.add("options", "NVDA", t0 + 2000, { side: "sell", notional: 5000, source: "tier_dispatcher.SELL_CSP" });
  const t = book.today(t0 + 3000);
  assert.equal(t.count, 2);
  assert.deepEqual(t.entries.map(e => `${e.sleeve}:${e.ticker}:${e.notional}`), ["options:NVDA:5000", "stocks:NVDA:1000"]);
  for (let i = 0; i < SHADOW_ROWS_CAP + 10; i++) book.add("stocks", `T${i}`, t0 + 10_000 + i);
  const big = book.today(t0 + 100_000);
  assert.equal(big.count, SHADOW_ROWS_CAP + 12, "count stays exact past the cap");
  assert.equal(big.entries.length, SHADOW_ROWS_CAP);
  assert.equal(big.entries[0].ticker, `T${SHADOW_ROWS_CAP + 9}`, "newest first");
  assert.deepEqual(book.today(Date.parse("2026-10-29T14:00:00Z")), { day: "2026-10-29", count: 0, entries: [] });
});

test("/api/bot/status carries entry_shadow + today's would-be entries; both book writers pass detail", () => {
  const route = slice('app.get("/api/bot/status"', "});");
  assert.ok(route.includes("entry_shadow: entryShadowStatus(process.env),"));
  assert.ok(route.includes("shadow_entries_today: shadowEntryBook.today(),"));
  assert.ok(bot.includes("shadowEntryBook.add(e.sleeve, e.ticker, Date.now(), e)"), "noteShadowEntry passes the full entry");
  assert.ok(/shadowEntryBook\.add\(e\.sleeve, String\(e\.ticker\), Date\.now\(\), \{ source:/.test(bot), "Python-side entries pass their source");
});

test("audit line names sleeve, size, instrument and the source, and says no order", () => {
  const line = shadowEntryAuditLine({ sleeve: "stocks", ticker: "NVDA", source: "executeTrades.etf", side: "buy", qty: 12, notional: 1500.4, instrument: "NVDL" });
  assert.match(line, /^stocks: would BUY 12 NVDL for NVDA ~\$1500 via executeTrades\.etf — logged, no order/);
});

// ── bot.ts gates: before the POST, and the POST still there for flag off ────
const execLoop = slice("for (const trade of trades) {", "// ── Batch fill confirmation");

function assertGateBeforePost(section: string, gateMarker: string, postMarker: string, label: string) {
  const g = section.indexOf(gateMarker);
  assert.ok(g >= 0, `${label}: gate not found (${gateMarker})`);
  const p = section.indexOf(postMarker, g);
  assert.ok(p > g, `${label}: the order POST must follow the gate (flag off = unchanged path)`);
  assert.ok(section.slice(g, p).includes("continue;"), `${label}: the shadowed branch must skip the POST`);
}

test("executeTrades: stock and ETF buys are gated before their POST and logged with relabel", () => {
  assertGateBeforePost(execLoop, 'source: "executeTrades.stock"', 'const orderResult = await alpaca("/v2/orders"', "stock");
  assertGateBeforePost(execLoop, 'source: "executeTrades.etf"', 'const etfOrderResult = await alpaca("/v2/orders"', "ETF");
  const stockGate = execLoop.slice(execLoop.indexOf('if (isEntryShadowed("stocks")) {\n          // ENTRY SHADOW MODE'), execLoop.indexOf('const orderParams = getOrderParams(trade.price || 0);'));
  assert.ok(stockGate.includes("relabel: true") && stockGate.includes("shadowEntries"), "one record per decision (relabel deep_score's)");
  assert.ok(stockGate.includes("slotsUsed++") && stockGate.includes("totalDeployed +="), "slot/heat consumed as live would");
});

test("executeTrades: a shadowed options open never falls back to a stock buy", () => {
  const at = execLoop.indexOf("optExec.order?.status === SHADOW_ORDER_STATUS");
  assert.ok(at > 0);
  const win = execLoop.slice(at, execLoop.indexOf('["submitted", "filled", "pending_new", "accepted"].includes(optExec.order?.status)', at));
  assert.ok(win.includes("continue;"), "shadowed -> next candidate, not the stock path");
  assert.ok(!win.includes("openOrders.push"), "nothing rests at the broker");
});

test("executeTrades: a candidate already shadow-entered today is treated as held", () => {
  const at = execLoop.indexOf("if ((trade as any)._consumed) continue;");
  const win = execLoop.slice(at, at + 700);
  assert.ok(win.includes("shadowEntryBook.has(tradeSleeve, trade.ticker)"));
  assert.ok(win.includes("noteGate(trade, SHADOW_REPEAT_DECISION"), "its deep_score record is relabelled, not left taken");
  const noteAt = win.indexOf("noteGate(trade, SHADOW_REPEAT_DECISION");
  assert.ok(win.indexOf("continue;", noteAt) > noteAt, "noted before it continues");
});

test("executeTrades: one shadow-log write after the loop, before the gate relabel flush", () => {
  const flushAt = execLoop.indexOf('await flushShadowEntries(shadowEntries, "executeTrades");');
  assert.ok(flushAt > 0 && flushAt < execLoop.indexOf("if (gateRejections.length > 0) {"));
  assert.equal((execLoop.match(/flushShadowEntries\(/g) || []).length, 1);
});

test("morning queue: gated before its POST, flushed once after the loop", () => {
  const mq = slice("async function executeMorningQueue(", "// ── Three-Tier Engine");
  assertGateBeforePost(mq, 'source: "executeMorningQueue"', 'const order = await alpaca("/v2/orders"', "morning queue");
  assert.ok(mq.indexOf('await flushShadowEntries(mqShadowEntries, "morning queue");') > mq.lastIndexOf('audit("MORNING-ERROR"'));
});

test("tier dispatcher: T3 BUY gated before its POST; SELL_CSP / BUY_PUT shadow status consumes slot+cash, never registers an order", () => {
  const d = slice("for (const action of result.tier_actions) {", "if (tierActionsDeferred > 0) {");
  const t3 = d.indexOf('source: "tier_dispatcher.T3_BUY"');
  assert.ok(t3 > 0 && t3 < d.indexOf("const t3OrderResult = await alpaca("), "T3 gate before the POST");
  assert.ok(d.slice(d.indexOf("if (shares > 0 && isEntryShadowed(\"stocks\")) {"), d.indexOf("const t3OrderResult")).includes("} else if (shares > 0) {"));
  for (const [src, okMarker] of [["tier_dispatcher.SELL_CSP", 'audit("T" + action.tier, `SELL_CSP'], ["tier_dispatcher.BUY_PUT", 'audit("T4", `BUY_PUT']]) {
    const at = d.indexOf(`source: "${src}", side`);
    assert.ok(at > 0, src);
    const branchStart = d.lastIndexOf("if (r.status === SHADOW_ORDER_STATUS) {", at);
    const branch = d.slice(branchStart, d.indexOf(okMarker, branchStart));
    assert.ok(branch.includes('} else if (r.status === "submitted" || r.status === "filled") {'), `${src}: live branch unchanged`);
    assert.ok(!branch.slice(0, branch.indexOf("} else if")).includes("openOrders.push"), `${src}: nothing to sweep`);
  }
  const csp = d.slice(d.indexOf("if (r.status === SHADOW_ORDER_STATUS) {"), d.indexOf('audit("T" + action.tier, `SELL_CSP'));
  assert.ok(csp.includes("tierOptionsSlotsUsed++") && csp.includes("tierCashAvailable -="), "the batch sees the book live would have");
  assert.ok(d.includes("shadowEntryBook.has(tierSleeve, String(action.ticker))"), "already-shadowed entries are not re-dispatched");
  assert.ok(d.indexOf('await flushShadowEntries(tierShadowEntries, "tier dispatcher");') < d.indexOf("if (cspGateRejections.length > 0) {"));
});

test("Python-side shadow entries from the scan are audited once per sleeve+ticker per day", () => {
  const at = bot.indexOf("result.entry_shadow?.entries");
  assert.ok(at > 0);
  const win = bot.slice(at, at + 500);
  assert.ok(win.includes("!e.new") && win.includes("shadowEntryBook.add(e.sleeve") && win.includes('audit("SHADOW-ENTRY"'));
});

// ── exits, liquidations, the manual route and the floor are never gated ─────
test("no exit path consults the switch", () => {
  const exitSections: Array<[string, string]> = [
    ["async function checkPositionOnTick(", "function startStreaming("],
    ["async function syncMonitoredPositions(", "function addPositionToMonitor("],
    ["async function checkDrawdownKillSwitch(", "async function executeMorningQueue("],
    ["async function tier1ProtectiveSync(", "async function tier1Reflex("],
    ['app.post("/api/bot/trade"', 'app.post("/api/bot/close"'],
  ];
  for (const [from, to] of exitSections) {
    const s = slice(from, to);
    assert.ok(!/isEntryShadowed|noteShadowEntry|SHADOW_ORDER_STATUS/.test(s), `${from} must stay live`);
  }
});

test("the gate set is exactly the reviewed entry sites", () => {
  // stock: morning queue, T3 BUY, ETF, stock; plus the two sleeve lookups
  // (executeTrades repeat check, tier repeat check)
  assert.equal((bot.match(/isEntryShadowed\("stocks"\)/g) || []).length, 4);
  assert.equal((bot.match(/isEntryShadowed\(tradeSleeve\)/g) || []).length, 1);
  assert.equal((bot.match(/isEntryShadowed\(tierSleeve\)/g) || []).length, 1);
  // options opens are gated in the Python each one passes through
  assert.equal((bot.match(/if entry_shadow\.is_shadowed\('options'\):/g) || []).length, 3);
});

test("the shadow flush routes daemon-first with a temp-file fallback to the same entry point", () => {
  const f = slice("async function flushShadowEntries(", "// ── Per-Ticker Order Attempt Counter");
  assert.ok(f.includes('"entry_shadow_record"'));
  assert.ok(f.includes("from entry_shadow import record_node_entries"));
  assert.ok(f.includes("json.load(open('${shadowTmpPath}'))"));
  assert.ok(/try \{[\s\S]*catch \(shadowErr\)[\s\S]*finally \{[\s\S]*fs\.unlinkSync\(shadowTmpPath\)/.test(f));
  const daemon = fs.readFileSync(path.join(here, "..", "voltrade_daemon.py"), "utf8");
  assert.ok(daemon.includes('"entry_shadow_record": ("entry_shadow", "record_node_entries")'));
});

// ── visibility, never gating ─────────────────────────────────────────────────
test("/api/health bot block and the sleeves body carry entry_shadow; the deploy gate is unchanged", () => {
  const health = slice('app.get("/api/health"', "// Check 5b:");
  const botBlock = health.slice(health.indexOf("checks.checks.bot = {"));
  assert.ok(botBlock.includes("entry_shadow: entryShadowStatus(process.env),"));
  const sleeves = slice("async function sleevesBody(", 'app.get("/api/bot/sleeves"');
  assert.equal((sleeves.match(/entry_shadow: entryShadowStatus\(process\.env\)/g) || []).length, 3, "every branch of the body");
  const gate = fs.readFileSync(path.join(here, "healthGate.ts"), "utf8");
  assert.ok(!gate.includes("entry_shadow"), "healthGate.ts SERVING_CHECKS untouched");
});
