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
  assert.ok(/cooldownMinutes:\s*SWEPT_CSP_COOLDOWN_MINUTES/.test(loop), "the cooldown comes from the named constant");
  assert.ok(/if \(!cspGate\.dispatch\) \{\s*audit\("CSP-SKIP"[\s\S]*?continue;/.test(loop), "a skip writes one CSP-SKIP audit line then moves on");
  assert.ok(/cspUnderlying:\s*String\(action\.ticker\)\.toUpperCase\(\)/.test(loop), "the tracked CSP order carries its underlying so the sweep can record it");
});

test("SWEPT_CSP_COOLDOWN_MINUTES is declared next to STALE_ORDER_MINUTES and the sweep memory resets daily with orderAttemptCounts", () => {
  assert.ok(/const STALE_ORDER_MINUTES = 12;[^\n]*\n[\s\S]{0,800}?const SWEPT_CSP_COOLDOWN_MINUTES = 60;/.test(bot));
  assert.ok(/orderAttemptCounts = \{\};\s*\n\s*sweptCspAt = \{\};/.test(bot), "sweptCspAt must reset in the 4am daily reset alongside orderAttemptCounts");
});
