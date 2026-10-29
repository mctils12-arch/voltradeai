// POS-KILL MARK GUARD (2026-10-29) — the per-position kill in
// `syncMonitoredPositions()` (server/bot.ts) read `pnlPct` and `current`
// straight off Alpaca's /v2/positions row and liquidated on one reading.
//
// LIVE NEAR-MISS 2026-10-29 13:30:09Z (paper): in the first seconds of the
// session the row for GOOGL261127P00300000 — a LONG put actually UP +291%
// (~$3,680; positions-detail 40 min later: unrealized_plpc 2.91) — carried
// current_price 0 (no print yet) and so unrealized_plpc -1. The branch logged
// "position down -100.0% — forcing liquidation (threshold: -75%)" and
// submitted a market SELL of the hedge. It survived only because Alpaca
// rejected the order ("insufficient options buying power for cash-secured
// put" — the short 310P leg's buy-back had filled 6 s earlier). The SAME tick
// correctly killed GOOGL261127C00355000 at -98.8% on a real $0.06 mark.
//
// The fix (threshold unchanged — only WHEN the exit fires):
//   1. no valid mark (current_price missing/0/NaN or unrealized_plpc missing)
//      -> no liquidation, one POS-KILL-NOMARK audit line per ticker per UTC
//      day; the same `hasMark` gates the TIME-EXIT band test, which a
//      missing plpc (reads 0.0%) would otherwise satisfy;
//   2. a breach on a valid mark liquidates only when the previous valid-mark
//      tick saw it too (in-memory map; first sighting audits
//      POS-KILL-PENDING). A no-mark tick neither confirms nor clears.
//   3. the in-flight / open-order SKIP guards and the POST are untouched and
//      run after confirmation.
//
// bot.ts is a side-effecting orchestrator and cannot be imported, so — like
// shadowBackfillExitAligned.test.ts — this cuts the REAL per-position loop
// body out of the source, transpiles it with esbuild and RUNS it tick by tick
// against stubs. Tests (a) and (b) below fail against the pre-fix bot.ts on
// their behavioural assertions (the first tick POSTs), not on a missing
// marker: the harness tolerates the helpers being absent on purpose.
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
  };
  const js = transpile(`
${HELPERS}
async function runTick(positions, env) {
  const { FLOOR_AND_LEG_TICKERS, LEGACY_DEFENSIVE_CANDIDATE_TICKERS, audit, alpaca, state, openOrders,
          getOrderParams, notify, pkgVersion, posKillPendingBreach, posKillNoMarkAuditDay } = env;
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

test("posKillHasValidMark: zero / missing / NaN marks are invalid, a real $0.06 mark is valid", () => {
  const src = functionSourceIfPresent("function posKillHasValidMark(currentPriceRaw: unknown, plpcRaw: unknown): boolean");
  assert.ok(src, "posKillHasValidMark must exist in bot.ts");
  const valid = new Function(`${transpile(src)}\nreturn posKillHasValidMark;`)() as (p: unknown, l: unknown) => boolean;
  // The near-miss row.
  assert.equal(valid("0", "-1"), false, "current_price 0 with plpc -1 is the open-tick artifact, not a mark");
  assert.equal(valid(0, -1), false);
  assert.equal(valid(undefined, "-0.5"), false, "missing current_price");
  assert.equal(valid(null, "-0.5"), false);
  assert.equal(valid("", "-0.5"), false);
  assert.equal(valid("NaN", "-0.5"), false);
  assert.equal(valid("abc", "-0.5"), false);
  assert.equal(valid("-1", "-0.5"), false, "a negative price is not a mark");
  assert.equal(valid("1.88", undefined), false, "missing plpc (the loop would read it as 0.0%)");
  assert.equal(valid("1.88", null), false);
  assert.equal(valid("1.88", ""), false);
  assert.equal(valid("1.88", "n/a"), false);
  // Real marks, however small.
  assert.equal(valid("0.06", "-0.988"), true, "the GOOGL 355C kill on the same tick was on a real $0.06 mark");
  assert.equal(valid("0.01", "-0.998"), true);
  assert.equal(valid("1.88", "0"), true, "a flat position has a mark");
  assert.equal(valid(0.06, -0.988), true, "numeric fields are accepted too");
});

test("(a) the near-miss row: -100.0% on current_price 0 never submits, audits POS-KILL-NOMARK once per day", async () => {
  const h = makeHarness();
  await h.tick([GOOGL_PUT_NOMARK]);
  assert.equal(h.posts.length, 0, "a -100% reading off a zero mark must never reach the liquidation POST");
  assert.equal(h.audited("POS-KILL").length, 0, "no 'forcing liquidation' line");
  const nomark = h.audited("POS-KILL-NOMARK");
  assert.equal(nomark.length, 1, "exactly one POS-KILL-NOMARK line on the first no-mark tick");
  assert.ok(nomark[0].msg.startsWith("GOOGL261127P00300000: no valid mark from Alpaca ("), nomark[0].msg);
  assert.ok(nomark[0].msg.includes("current_price=0") && nomark[0].msg.includes("plpc=-1"), "the raw Alpaca fields must be in the line");
  assert.ok(nomark[0].msg.endsWith("— kill deferred"), nomark[0].msg);
  assert.equal(h.openOrders.length, 0, "nothing registered for the sweeper");

  // Same day, every subsequent sync (~45 s): still no order, no repeat line.
  for (let i = 0; i < 5; i++) await h.tick([GOOGL_PUT_NOMARK]);
  assert.equal(h.posts.length, 0);
  assert.equal(h.audited("POS-KILL-NOMARK").length, 1, "the NOMARK line is once per ticker per UTC day");
  assert.equal(h.audited("POS-KILL-PENDING").length, 0, "a no-mark tick is not a breach observation");
  assert.equal(h.env.posKillPendingBreach.size, 0, "a no-mark tick must not start a pending breach");

  // A missing plpc reads 0.0% through parseFloat(... || "0") — equally no order.
  await h.tick([{ ...GOOGL_PUT_NOMARK, unrealized_plpc: undefined }]);
  assert.equal(h.posts.length, 0);

  // Next UTC day the line is allowed again (dedupe is per calendar day, not forever).
  h.clock.advance(24 * 3600_000);
  await h.tick([GOOGL_PUT_NOMARK]);
  assert.equal(h.audited("POS-KILL-NOMARK").length, 2, "a new day gets its own line");
  assert.equal(h.posts.length, 0);
});

test("(b) a valid -80% breach: PENDING on the first tick, liquidated through the unchanged submit path on the next", async () => {
  const h = makeHarness();
  await h.tick([longOptionAt(-80)]);
  assert.equal(h.posts.length, 0, "the first observation of a breach must not submit");
  const pending = h.audited("POS-KILL-PENDING");
  assert.equal(pending.length, 1);
  assert.ok(pending[0].msg.startsWith("XLK261219P00100000: position down -80.0%"), pending[0].msg);
  assert.ok(pending[0].msg.endsWith("— confirming on the next tick"), pending[0].msg);
  assert.equal(h.audited("POS-KILL").length, 0);
  assert.equal(h.env.posKillPendingBreach.get("XLK261219P00100000")?.firstSeenMs, Date.parse("2026-10-29T13:30:09Z"));

  await h.tick([longOptionAt(-80)]);
  assert.equal(h.posts.length, 1, "the second consecutive valid-mark breach liquidates");
  assert.equal(h.posts[0].path, "/v2/orders");
  assert.deepEqual(
    { symbol: h.posts[0].body.symbol, qty: h.posts[0].body.qty, side: h.posts[0].body.side, type: h.posts[0].body.type },
    { symbol: "XLK261219P00100000", qty: "2", side: "sell", type: "market" },
    "the existing getOrderParams(current, 'stop_loss') market SELL, unchanged",
  );
  const kill = h.audited("POS-KILL");
  assert.equal(kill.length, 1);
  assert.ok(kill[0].msg.includes("position down -80.0% — forcing liquidation (threshold: -75%)"), kill[0].msg);
  assert.ok(
    h.gets.some((p) => p === "/v2/orders?status=open&symbols=XLK261219P00100000&side=sell"),
    "the broker open-order guard (#35) still runs before the POST, after confirmation",
  );
  assert.equal(h.openOrders.length, 1, "the kill is registered for the sweeper (#34)");
  assert.equal(h.openOrders[0].isExit, true);
  assert.equal((h.openOrders[0].pendingExitFill as { exitReason: string }).exitReason, "position_kill");
  assert.equal(h.audited("POS-KILL-PENDING").length, 1, "no second PENDING line once confirmed");

  // Stocks go through the same two-tick path against their own -25% rule.
  const s = makeHarness();
  await s.tick([stockAt(-30)]);
  assert.equal(s.posts.length, 0);
  assert.equal(s.audited("POS-KILL-PENDING").length, 1);
  await s.tick([stockAt(-30)]);
  assert.equal(s.posts.length, 1);
  assert.equal(s.posts[0].body.symbol, "AXTI");
  assert.ok(s.audited("POS-KILL")[0].msg.includes("(threshold: -25%)"));
});

test("(c) -80% then -10% then -80%: the recovery tick resets the confirmation, no order", async () => {
  const h = makeHarness();
  await h.tick([longOptionAt(-80)]);
  await h.tick([longOptionAt(-10)]);
  assert.equal(h.env.posKillPendingBreach.size, 0, "a valid-mark tick with no breach clears the pending breach");
  await h.tick([longOptionAt(-80)]);
  assert.equal(h.posts.length, 0, "two non-consecutive breaches must not liquidate");
  assert.equal(h.audited("POS-KILL").length, 0);
  assert.equal(h.audited("POS-KILL-PENDING").length, 2, "the third tick is a fresh first observation");
  assert.equal(
    h.env.posKillPendingBreach.get("XLK261219P00100000")?.firstSeenMs,
    Date.parse("2026-10-29T13:30:09Z") + 2 * 45_000,
    "firstSeenMs is the third tick's, not the first's",
  );
  // One more breach tick confirms it.
  await h.tick([longOptionAt(-80)]);
  assert.equal(h.posts.length, 1);
});

test("(d) the genuine kill from the same live tick: -98.8% on a real $0.06 mark is killed on its second tick", async () => {
  const h = makeHarness();
  await h.tick([GOOGL_CALL_REAL, GOOGL_PUT_NOMARK]);
  assert.equal(h.posts.length, 0);
  assert.equal(h.audited("POS-KILL-PENDING").length, 1);
  assert.ok(h.audited("POS-KILL-PENDING")[0].msg.startsWith("GOOGL261127C00355000: position down -98.8%"));
  await h.tick([GOOGL_CALL_REAL, GOOGL_PUT_NOMARK]);
  assert.equal(h.posts.length, 1, "a tiny but real mark is a real loss — the guard must not block it");
  assert.equal(h.posts[0].body.symbol, "GOOGL261127C00355000");
  assert.equal(h.posts[0].body.side, "sell");
  assert.ok(h.audited("POS-KILL")[0].msg.includes("GOOGL261127C00355000: position down -98.8% — forcing liquidation (threshold: -75%)"));
  // ...while the winning put next to it was never touched.
  assert.ok(!h.posts.some((p) => p.body.symbol === "GOOGL261127P00300000"));
  assert.equal(h.audited("POS-KILL-NOMARK").length, 1);
});

test("a no-mark tick between two valid breaches neither confirms nor clears", async () => {
  const h = makeHarness();
  await h.tick([longOptionAt(-80)]);
  await h.tick([{ ...longOptionAt(-80), current_price: "0", unrealized_plpc: "-1" }]);
  assert.equal(h.posts.length, 0, "the no-mark tick itself never submits");
  assert.equal(h.env.posKillPendingBreach.size, 1, "the pending breach is left exactly as it was");
  await h.tick([longOptionAt(-80)]);
  assert.equal(h.posts.length, 1, "the second VALID observation confirms");
});

test("a pending breach does not outlive its position: a re-entry in the same symbol starts the count again", async () => {
  // The pruning after the loop belongs to the sync, not the loop body, so
  // it is pinned statically; the loop-level contract is checked here.
  const pruneAt = bot.indexOf("for (const ticker of Array.from(posKillPendingBreach.keys())) {", syncStart);
  assert.ok(pruneAt > loopEnd, "pending breaches for tickers no longer held must be pruned after the loop");
  assert.ok(/if \(!activeTickers\.has\(ticker\)\) posKillPendingBreach\.delete\(ticker\);/.test(bot.slice(pruneAt, pruneAt + 200)));
});

test("TIME-EXIT shares the mark check: a >14d position with no mark is not closed on a phantom 0.0%", async () => {
  const h = makeHarness();
  const old = "2026-09-01T15:12:00Z";
  // Missing plpc reads 0.0% through parseFloat(... || "0") — inside [-5, +2].
  await h.tick([{ ...stockAt(0, "AXTI", old), current_price: "0", unrealized_plpc: undefined }]);
  assert.equal(h.posts.length, 0, "no mark, no time exit");
  assert.equal(h.audited("TIME-EXIT").length, 0);
  // TIME-EXIT's own logic is untouched: a real +0.5% after 14d still closes.
  await h.tick([stockAt(0.5, "AXTI", old)]);
  assert.equal(h.posts.length, 1);
  assert.equal(h.audited("TIME-EXIT").length, 1);
  assert.ok(h.audited("TIME-EXIT")[0].msg.includes("closing stale"));
});

test("wiring: hasMark is computed from the Alpaca row before TIME-EXIT, and gates its band test", () => {
  const loop = bot.slice(loopStart, loopEnd);
  const hasMarkAt = loop.indexOf("const hasMark = posKillHasValidMark(pos.current_price, pos.unrealized_plpc);");
  const timeExitAt = loop.indexOf('audit("TIME-EXIT"');
  assert.ok(hasMarkAt > 0 && timeExitAt > hasMarkAt, "hasMark must be computed once, before the TIME-EXIT block");
  assert.ok(loop.includes("if (hasMark && ageDays > 14 && pnlPct >= -5.0 && pnlPct <= 2.0) {"), "TIME-EXIT's band test must be gated on hasMark and otherwise unchanged");
});

test("wiring: the breach is observed only on a valid mark, and the two new gates sit before the duplicate guard and the POST", () => {
  const loop = bot.slice(loopStart, loopEnd);
  assert.ok(
    /const breach: "nomark" \| "clear" \| "pending" \| "confirmed" = hasMark\s*\?\s*posKillObserveBreach\(posKillPendingBreach, ticker, pnlPct <= POSITION_KILL_LOSS_PCT, pnlPct, Date\.now\(\)\)\s*:\s*"nomark";/.test(loop),
    "posKillObserveBreach must be reached only when hasMark is true, against this position's own threshold",
  );
  const killAt = loop.indexOf("if (pnlPct <= POSITION_KILL_LOSS_PCT) {");
  const kill = loop.slice(killAt);
  const nomarkAt = kill.indexOf('if (breach === "nomark") {');
  const pendingAt = kill.indexOf('if (breach === "pending") {');
  const guardAt = kill.indexOf("const trackedExit = openOrders.find(");
  const postAt = kill.indexOf("const killOrderResult = await alpaca(");
  assert.ok(nomarkAt > 0 && pendingAt > nomarkAt && guardAt > pendingAt && postAt > guardAt,
    "order must be: no-mark gate, confirmation gate, duplicate guard, POST");
  for (const [label, from, to] of [["no-mark gate", nomarkAt, pendingAt], ["confirmation gate", pendingAt, guardAt]] as const) {
    const gate = kill.slice(from, to);
    assert.ok(/continue;/.test(gate), `${label} must leave the branch, not fall through to the POST`);
    assert.ok(!/alpaca\(/.test(gate) && !/openOrders\.push/.test(gate), `${label} must not call Alpaca or register an order`);
  }
  assert.ok(/audit\("POS-KILL-NOMARK", `\$\{ticker\}: no valid mark from Alpaca \(current_price=/.test(kill), "the no-mark line names the raw Alpaca fields");
  assert.ok(/toISOString\(\)\.split\("T"\)\[0\]/.test(kill.slice(nomarkAt, pendingAt)), "the NOMARK dedupe key is a calendar day");
  assert.ok(/audit\("POS-KILL-PENDING", `\$\{ticker\}: position down \$\{pnlPct\.toFixed\(1\)\}%/.test(kill), "the first observation is visible in the audit log");
});

test("wiring: the confirmation state is an in-memory map declared beside monitoredPositions, never persisted", () => {
  assert.ok(bot.includes("const posKillPendingBreach: Map<string, PosKillBreach> = new Map();"), "pending breaches live in a plain in-memory Map");
  assert.ok(bot.includes("const posKillNoMarkAuditDay: Map<string, string> = new Map();"));
  const decl = bot.indexOf("const posKillPendingBreach: Map<string, PosKillBreach> = new Map();");
  assert.ok(decl > bot.indexOf("const monitoredPositions: Record<string, MonitoredPosition> = {};") && decl < syncStart,
    "declared with the other position-monitor state, before syncMonitoredPositions");
  assert.ok(!/posKillPendingBreach[\s\S]{0,80}(JSON\.stringify|writeFile|saveState)/.test(bot), "a restart must forget it — nothing writes it anywhere");
});
