// CSP SWEEP/RESUBMIT LOOP FIX (2026-10-13) — regression tests.
//
// Live 2026-10-12: 51 canceled CSP limit orders and ZERO SWEEP audit lines.
// sweepStaleOrders() DELETEs a stale limit; Alpaca answers 204 No Content;
// alpaca() calls r.json() on every 2xx, which throws after the cancel already
// succeeded — so the audit never ran. Meanwhile the SELL_CSP dispatcher
// re-submitted the same non-marketable limit on the same underlying every
// scan, so the loop never settled. These tests pin both halves: the
// 204-then-GET-canceled audit path, and the dispatcher's sweep memory +
// open sell-to-open guard.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  confirmCancelAfterDeleteError,
  isEmptyBodyParseError,
  isOpenSellToOpenOn,
  occUnderlying,
  shouldDispatchCsp,
} from "./cspSweepCooldown";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function slice(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

const MIN = 60_000;
const NOW = Date.UTC(2026, 9, 13, 15, 0, 0);
const COOLDOWN = 60;

// ── shouldDispatchCsp ───────────────────────────────────────────────────────

test("shouldDispatchCsp: underlying swept 10 min ago → skip (swept_cooldown)", () => {
  const d = shouldDispatchCsp("TLT", { now: NOW, sweptAt: NOW - 10 * MIN, openOrders: [], cooldownMinutes: COOLDOWN });
  assert.equal(d.dispatch, false);
  assert.equal(d.dispatch === false && d.reason, "swept_cooldown");
});

test("shouldDispatchCsp: underlying swept 2h ago → allow", () => {
  const d = shouldDispatchCsp("TLT", { now: NOW, sweptAt: NOW - 120 * MIN, openOrders: [], cooldownMinutes: COOLDOWN });
  assert.deepEqual(d, { dispatch: true });
});

test("shouldDispatchCsp: never swept, no open orders → allow", () => {
  assert.deepEqual(shouldDispatchCsp("TLT", { now: NOW, openOrders: [], cooldownMinutes: COOLDOWN }), { dispatch: true });
});

test("shouldDispatchCsp: open sell-to-open option order on the same underlying → skip", () => {
  const d = shouldDispatchCsp("TLT", {
    now: NOW,
    openOrders: [{ symbol: "TLT261120P00085000", side: "sell", position_intent: "sell_to_open" }],
    cooldownMinutes: COOLDOWN,
  });
  assert.equal(d.dispatch, false);
  assert.equal(d.dispatch === false && d.reason, "open_sell_to_open");
});

test("shouldDispatchCsp: open option orders on OTHER underlyings, a buy, or a sell_to_close do not block", () => {
  const d = shouldDispatchCsp("TLT", {
    now: NOW,
    openOrders: [
      { symbol: "TLTW261120P00025000", side: "sell", position_intent: "sell_to_open" }, // different root
      { symbol: "TLT261120P00085000", side: "buy", position_intent: "buy_to_close" },
      { symbol: "TLT261120C00095000", side: "sell", position_intent: "sell_to_close" },
      { symbol: "TLT", side: "sell" }, // the stock itself, not an option
    ],
    cooldownMinutes: COOLDOWN,
  });
  assert.deepEqual(d, { dispatch: true });
});

test("shouldDispatchCsp: a sell option order with no position_intent counts as opening (matches countOpenOptionsOpeningOrders)", () => {
  const d = shouldDispatchCsp("xlp", {
    now: NOW,
    openOrders: [{ symbol: "XLP261120P00078000", side: "sell" }],
    cooldownMinutes: COOLDOWN,
  });
  assert.equal(d.dispatch, false);
});

test("occUnderlying parses the OCC root and rejects non-option symbols", () => {
  assert.equal(occUnderlying("XLK260424P00149000"), "XLK");
  assert.equal(occUnderlying("AAPL1260424C00100000"), "AAPL1");
  assert.equal(occUnderlying("SPY"), null);
  assert.equal(occUnderlying(""), null);
  assert.equal(isOpenSellToOpenOn({ symbol: "IBIT261120P00050000", side: "sell" }, "IBIT"), true);
});

// ── 204-then-GET-canceled audit path ────────────────────────────────────────

// Reproduce exactly what alpaca() does on a 2xx: `return r.json()` — on a
// real 204 No Content Response that throws a SyntaxError.
async function alpacaLikeJsonOn204(): Promise<unknown> {
  const r = new Response(null, { status: 204 });
  assert.ok(r.ok);
  return r.json();
}

test("a real 204 No Content r.json() failure is classified as a probable success", async () => {
  let err: unknown = null;
  try {
    await alpacaLikeJsonOn204();
  } catch (e) {
    err = e;
  }
  assert.ok(err, "r.json() on a 204 must throw — this is the bug's trigger");
  assert.equal(isEmptyBodyParseError(err), true);
});

test("204 on DELETE, then GET says canceled → confirmed (SWEEP audit is written)", async () => {
  let err: unknown = null;
  try {
    await alpacaLikeJsonOn204();
  } catch (e) {
    err = e;
  }
  let gets = 0;
  const res = await confirmCancelAfterDeleteError(err, async () => {
    gets++;
    return { id: "abc", status: "canceled" };
  });
  assert.equal(gets, 1, "must confirm with exactly one read-only GET");
  assert.deepEqual(res, { confirmed: true, status: "canceled" });
});

test("204 on DELETE, but GET says filled (raced a fill) → not confirmed", async () => {
  const res = await confirmCancelAfterDeleteError(new SyntaxError("Unexpected end of JSON input"), async () => ({ status: "filled" }));
  assert.deepEqual(res, { confirmed: false, status: "filled" });
});

test("a real HTTP error from alpaca() (e.g. 422 not cancelable) or a timeout is NOT treated as success, and no GET is made", async () => {
  let gets = 0;
  const fetchOrder = async () => {
    gets++;
    return { status: "canceled" };
  };
  const httpErr = await confirmCancelAfterDeleteError(new Error("Alpaca 422: order is not cancelable"), fetchOrder);
  const abort = Object.assign(new Error("This operation was aborted"), { name: "AbortError" });
  const timeoutErr = await confirmCancelAfterDeleteError(abort, fetchOrder);
  assert.equal(httpErr.confirmed, false);
  assert.equal(timeoutErr.confirmed, false);
  assert.equal(gets, 0);
});

test("a failing confirm GET leaves the cancel unconfirmed rather than throwing", async () => {
  const res = await confirmCancelAfterDeleteError(new SyntaxError("Unexpected end of JSON input"), async () => {
    throw new Error("Alpaca 500: boom");
  });
  assert.deepEqual(res, { confirmed: false, status: null });
});

// ── bot.ts wiring ───────────────────────────────────────────────────────────

const sweepStaleOrders = slice("async function sweepStaleOrders() {", "async function replaceIfBetter(");

test("sweepStaleOrders' DELETE-catch path confirms via GET /v2/orders/{id} and writes the SWEEP audit + sweep memory when confirmed", () => {
  assert.ok(
    /catch \(e[^)]*\) \{[\s\S]*?const cancel = await confirmCancelAfterDeleteError\(e, \(\) => alpaca\(`\/v2\/orders\/\$\{stale\.orderId\}`\)\);[\s\S]*?if \(cancel\.confirmed\) \{[\s\S]*?audit\("SWEEP"[\s\S]*?sweptCspAt\[stale\.cspUnderlying\] = Date\.now\(\);/.test(sweepStaleOrders),
    "the 204 throw must be confirmed with a read-only GET and then audited as SWEEP, recording the swept CSP underlying",
  );
  assert.ok(
    !/alpaca\(`\/v2\/orders\/\$\{stale\.orderId\}`,\s*\{\s*method:\s*"(POST|PATCH)"/.test(sweepStaleOrders),
    "the confirm must be a read-only GET, never a write",
  );
});

test("SELL_CSP dispatch consults shouldDispatchCsp before submitting, and audits one line when skipping", () => {
  const loop = slice("for (const action of result.tier_actions) {", "// ── TIER 3: BUY SPY/QQQ at 2x");
  const gateIdx = loop.indexOf("shouldDispatchCsp(action.ticker");
  const submitIdx = loop.indexOf("submit_options_order(contract)");
  assert.ok(gateIdx > 0 && submitIdx > gateIdx, "the sweep/open-order gate must precede the CSP submission");
  assert.ok(/sweptAt:\s*sweptCspAt\[/.test(loop), "the gate must read the sweep memory");
  assert.ok(/openOrders:\s*Array\.isArray\(freshOpenOrdersForTiers\)/.test(loop), "the gate must see the live open-orders list");
  // 2026-11-19 (price ladder): was /cooldownMinutes:\s*SWEPT_CSP_COOLDOWN_MINUTES/. The gate's cooldown now
  // comes from the ladder step, which is itself built from the named constant (its value when the
  // ladder is off or spent); both halves are pinned.
  assert.ok(/cooldownMinutes:\s*cspLadder\.cooldownMinutes/.test(loop), "the gate's cooldown comes from the ladder step");
  assert.ok(/cspLadderStep\([\s\S]{0,300}?cspLadderOn, SWEPT_CSP_COOLDOWN_MINUTES\)/.test(loop), "the ladder's legacy cooldown is the named constant");
  assert.ok(/if \(!cspGate\.dispatch\) \{\s*audit\("CSP-SKIP"[\s\S]*?continue;/.test(loop), "a skip writes one CSP-SKIP audit line then moves on");
  assert.ok(/cspUnderlying:\s*String\(action\.ticker\)\.toUpperCase\(\)/.test(loop), "the tracked CSP order carries its underlying so the sweep can record it");
});

test("SWEPT_CSP_COOLDOWN_MINUTES is declared next to STALE_ORDER_MINUTES and the sweep memory resets daily with orderAttemptCounts", () => {
  assert.ok(/const STALE_ORDER_MINUTES = 12;[^\n]*\n[\s\S]{0,800}?const SWEPT_CSP_COOLDOWN_MINUTES = 60;/.test(bot));
  assert.ok(/orderAttemptCounts = \{\};\s*\n\s*sweptCspAt = \{\};/.test(bot), "sweptCspAt must reset in the 4am daily reset alongside orderAttemptCounts");
});

// KB #46 — restart-proof brake: a fresh process (no sweep memory, empty open
// book) still skips an underlying the BROKER shows rested unfilled.
test("broker closed order rested 12m, canceled 3m ago => skip with no in-process memory", () => {
  const now = Date.parse("2026-10-13T18:30:00Z");
  const closed = [{
    symbol: "XLF261120P00050000", side: "sell", status: "canceled", filled_qty: "0",
    submitted_at: "2026-10-13T18:15:00Z", canceled_at: "2026-10-13T18:27:00Z",
  }];
  const d = shouldDispatchCsp("XLF", { now, cooldownMinutes: 60, closedOrders: closed, staleMinutes: 12 });
  assert.equal(d.dispatch, false);
  assert.equal((d as { reason?: string }).reason, "broker_unfilled_rested");
});

test("deploy cancel-all 36s after submit is NOT evidence of non-fill", () => {
  const now = Date.parse("2026-10-13T17:35:00Z");
  const closed = [{
    symbol: "XLF261120P00050000", side: "sell", status: "canceled", filled_qty: "0",
    submitted_at: "2026-10-13T17:33:09Z", canceled_at: "2026-10-13T17:33:45Z",
  }];
  assert.equal(shouldDispatchCsp("XLF", { now, cooldownMinutes: 60, closedOrders: closed, staleMinutes: 12 }).dispatch, true);
});

test("filled, old, other-underlying and buy-to-close closed orders do not brake", () => {
  const now = Date.parse("2026-10-13T18:30:00Z");
  const base = { symbol: "XLF261120P00050000", side: "sell", status: "canceled", filled_qty: "0",
    submitted_at: "2026-10-13T18:15:00Z", canceled_at: "2026-10-13T18:27:00Z" };
  const run = (o: Record<string, unknown>, t = "XLF") => shouldDispatchCsp(t, { now, cooldownMinutes: 60, closedOrders: [o], staleMinutes: 12 }).dispatch;
  assert.equal(run({ ...base, filled_qty: "1" }), true);
  assert.equal(run({ ...base, canceled_at: "2026-10-13T16:00:00Z", submitted_at: "2026-10-13T15:45:00Z" }), true);
  assert.equal(run(base, "HPE"), true);
  assert.equal(run({ ...base, side: "buy" }), true);
  assert.equal(run({ ...base, status: "filled" }), true);
});

test("bot.ts feeds the dispatcher gate the broker's closed orders", () => {
  assert.match(bot, /closedOrders: Array\.isArray\(recentClosedOrdersForCsp\)/);
  assert.match(bot, /status=closed&after=/);
});

// ── PRICE LADDER (2026-11-19): the AAL puts of that day, replayed ────────────
import { countUnfilledRestedCspSince, cspLadderStep, cspPriceLadderEnabled, CSP_PRICE_WALKS, CSP_LADDER_COOLDOWN_MINUTES } from "./cspSweepCooldown";
import fsLadder from "node:fs";
import pathLadder from "node:path";
import { fileURLToPath as urlLadder } from "node:url";

const DAY = Date.parse("2026-11-19T14:30:00Z");
const order = (sym: string, submitted: string, ended: string, extra: Record<string, unknown> = {}) => ({
  symbol: sym, side: "sell", position_intent: "sell_to_open", status: "canceled", filled_qty: "0",
  submitted_at: submitted, canceled_at: ended, ...extra,
});
const AAL1 = order("AAL261231P00011500", "2026-11-19T15:30:08Z", "2026-11-19T15:42:54Z");
const AAL2 = order("AAL261231P00011500", "2026-11-19T18:50:20Z", "2026-11-19T19:02:54Z");

test("ladder: the day's declined attempts are counted from the broker's closed orders", () => {
  assert.equal(countUnfilledRestedCspSince([AAL1, AAL2], "AAL", DAY, 12), 2);
  assert.equal(countUnfilledRestedCspSince([AAL1, AAL2], "aal", DAY, 12), 2);
  assert.equal(countUnfilledRestedCspSince([AAL1], "NU", DAY, 12), 0);                 // another underlying
  assert.equal(countUnfilledRestedCspSince(null, "AAL", DAY, 12), 0);
});

test("ladder: a fill, a quick cancel, a close, a call on another root and yesterday's order are not declined attempts", () => {
  const filled = order("AAL261231P00011500", "2026-11-19T15:30:08Z", "2026-11-19T15:42:54Z", { filled_qty: "1" });
  const quick = order("AAL261231P00011500", "2026-11-19T16:00:00Z", "2026-11-19T16:03:00Z");   // a deploy's cancel-all
  const closing = order("AAL261231P00011500", "2026-11-19T16:00:00Z", "2026-11-19T16:20:00Z", { position_intent: "sell_to_close" });
  const yesterday = order("AAL261231P00011500", "2026-11-18T15:30:00Z", "2026-11-18T15:45:00Z");
  const buy = order("AAL261231P00011500", "2026-11-19T16:00:00Z", "2026-11-19T16:20:00Z", { side: "buy" });
  const stillOpen = order("AAL261231P00011500", "2026-11-19T16:00:00Z", "", { status: "new", canceled_at: null });
  assert.equal(countUnfilledRestedCspSince([filled, quick, closing, yesterday, buy, stillOpen], "AAL", DAY, 12), 0);
  const expired = order("AAL261231P00011500", "2026-11-19T15:00:00Z", "", { status: "expired", canceled_at: null, expired_at: "2026-11-19T21:00:00Z" });
  assert.equal(countUnfilledRestedCspSince([expired], "AAL", DAY, 12), 1);
});

test("ladder: three prices a day — near mid, 65% toward the bid, the bid — then nothing until tomorrow", () => {
  assert.deepEqual([...CSP_PRICE_WALKS], [0.3, 0.65, 1.0]);
  assert.deepEqual(cspLadderStep(0, true, 60), { step: 0, walk: 0.3, exhausted: false, cooldownMinutes: CSP_LADDER_COOLDOWN_MINUTES });
  assert.deepEqual(cspLadderStep(1, true, 60), { step: 1, walk: 0.65, exhausted: false, cooldownMinutes: 20 });
  assert.deepEqual(cspLadderStep(2, true, 60), { step: 2, walk: 1.0, exhausted: false, cooldownMinutes: 20 });
  assert.deepEqual(cspLadderStep(3, true, 60), { step: 3, walk: null, exhausted: true, cooldownMinutes: 60 });
  assert.equal(cspLadderStep(9, true, 60).exhausted, true);
  assert.equal(cspLadderStep(-1, true, 60).step, 0);
  assert.equal(cspLadderStep(NaN, true, 60).step, 0);
  assert.ok(CSP_PRICE_WALKS.every((w, i) => w > 0 && w <= 1 && (i === 0 || w > CSP_PRICE_WALKS[i - 1])), "each step moves toward the bid and never past it");
});

test("ladder: one setting restores one price and the 60-minute cooldown", () => {
  assert.equal(cspPriceLadderEnabled({}), true);
  for (const v of ["0", "false", "off", "no", " OFF "]) assert.equal(cspPriceLadderEnabled({ VOLTRADE_CSP_PRICE_LADDER: v }), false);
  assert.equal(cspPriceLadderEnabled({ VOLTRADE_CSP_PRICE_LADDER: "1" }), true);
  for (const n of [0, 1, 2, 3, 7]) {
    assert.deepEqual(cspLadderStep(n, false, 60), { step: n, walk: 0.3, exhausted: false, cooldownMinutes: 60 });
  }
});

test("ladder: 11-19 replayed — the second AAL attempt comes 20 minutes after the first sweep, not 60, and asks a lower price", () => {
  // 15:30 first limit, swept 15:42:54. At 15:46:54 the dispatcher ran again.
  const at = Date.parse("2026-11-19T15:46:54Z");
  const step = cspLadderStep(countUnfilledRestedCspSince([AAL1], "AAL", DAY, 12), true, 60);
  assert.equal(step.walk, 0.65);
  const tooSoon = shouldDispatchCsp("AAL", { now: at, openOrders: [], closedOrders: [AAL1], cooldownMinutes: step.cooldownMinutes, staleMinutes: 12 });
  assert.equal(tooSoon.dispatch, false);                                             // 4 min after the sweep
  const later = shouldDispatchCsp("AAL", { now: Date.parse("2026-11-19T16:03:00Z"), openOrders: [], closedOrders: [AAL1], cooldownMinutes: step.cooldownMinutes, staleMinutes: 12 });
  assert.equal(later.dispatch, true);                                                // 20 min after: step 2 may go
  const legacy = shouldDispatchCsp("AAL", { now: Date.parse("2026-11-19T16:03:00Z"), openOrders: [], closedOrders: [AAL1], cooldownMinutes: 60, staleMinutes: 12 });
  assert.equal(legacy.dispatch, false);                                              // what happened on 11-19
});

test("ladder wiring: the dispatcher counts today's declined puts, stops when the ladder is spent, and sends the walk only from the second attempt", () => {
  const bot = fsLadder.readFileSync(pathLadder.join(pathLadder.dirname(urlLadder(import.meta.url)), "bot.ts"), "utf8");
  assert.match(bot, /const cspLadderSinceMs = Date\.now\(\) - 7 \* 3600_000;/);
  assert.match(bot, /cspLadder = cspLadderStep\(\s*countUnfilledRestedCspSince\(/);
  assert.match(bot, /if \(cspLadder\.exhausted\) \{[\s\S]{0,400}noteCspGate\(action, "rejected_csp_resubmit"[\s\S]{0,200}continue;/);
  assert.match(bot, /cooldownMinutes: cspLadder\.cooldownMinutes,/);
  assert.match(bot, /if \(cspLadder\.step > 0 && cspLadder\.walk != null\) \{\s*cspPayload\.metadata = \{ \.\.\.\(action\.metadata \?\? \{\}\), price_walk: cspLadder\.walk, price_step: cspLadder\.step \};/);
});
