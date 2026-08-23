// STALE-ORDER-SWEEP FIX (KNOWN BROKEN #32, research/open_questions.md) —
// FINAL THREE SITES. Every prior session in this chain named the same
// remaining scope explicitly: "the manual API route, morning-queue
// execution, Tier 3 BUY" order-submission call sites in server/bot.ts
// never registered a TrackedOrder with `openOrders`, so sweepStaleOrders()
// (tier1Reflex, every ~45s) had nothing to cancel if any of them placed a
// resting limit order that never filled — the same dead-mechanism class
// already fixed for options entries, stock/ETF entries, and exits.
//
// - Manual API route (`POST /api/bot/trade`): defaults to type: "market"
//   but a caller can pass any `type`; a non-market order can rest and is
//   now tracked (a market order fills immediately and gains nothing from
//   tracking, matching the file's existing TIME-EXIT/CC-UNWIND precedent
//   of skipping market-only submissions). A "sell" via this route is
//   always closing an existing long (shorting is hard-blocked above), so
//   it's tagged isExit; a "buy" is an entry.
// - Morning queue (`executeMorningQueue`): getOrderParams' 'new_entry'
//   default (orderParams.ts) is ALWAYS a DAY limit order, in and out of
//   regular hours — a real, not theoretical, gap for every queued
//   overnight-research trade.
// - Tier 3 BUY dispatcher (the `action.action === "BUY"` branch, same
//   dispatcher as the already-fixed SELL_CSP/BUY_PUT branches): its order
//   submission's response was discarded entirely (not even captured into
//   a variable) — same always-limit getOrderParams default as above.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function slice(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

test("openOrders.push now appears at least 9 times (2 options-entry + 2 stock/ETF-entry + 2 exit + manual-route + morning-queue + T3-BUY)", () => {
  const pushCalls = bot.match(/openOrders\.push\(/g) || [];
  assert.ok(
    pushCalls.length >= 9,
    "expected the 6 pre-existing pushes plus 3 new pushes (manual route, morning queue, T3 BUY) — fewer means one of the new registrations is missing",
  );
});

test("manual /api/bot/trade route registers a non-market order with openOrders, tagged isExit by side", () => {
  const section = slice('app.post("/api/bot/trade"', "// Close a position");
  assert.ok(
    /const order\s*=\s*await alpaca\(\s*"\/v2\/orders"/.test(section),
    "the manual route's order submission must capture its response",
  );
  assert.ok(
    /if\s*\(order\?\.id\s*&&\s*type\s*!==\s*"market"\)\s*\{[\s\S]*?openOrders\.push\(/.test(section),
    "the push must be gated on a real order id AND skip market orders (which fill immediately and gain nothing from tracking)",
  );
  assert.ok(
    /orderId:\s*order\.id/.test(section) && /isExit:\s*orderSide\s*===\s*"sell"/.test(section),
    "the tracked entry must carry the real returned orderId and tag isExit based on which side this manual order actually is",
  );
});

test("morning queue registers its order submission with openOrders", () => {
  const section = slice(
    "const _qtyToSubmit = Math.floor(trade.shares);",
    "await new Promise(r => setTimeout(r, 500));",
  );
  assert.ok(
    /const order\s*=\s*await alpaca\(\s*"\/v2\/orders"/.test(section),
    "the morning-queue order submission must capture its response",
  );
  assert.ok(
    /if\s*\(order\?\.id\)\s*\{[\s\S]*?openOrders\.push\(/.test(section),
    "the push must be gated on order actually carrying an id",
  );
  assert.ok(
    /orderId:\s*order\.id/.test(section),
    "the tracked entry must carry the real returned orderId",
  );
});

test("Tier 3 BUY dispatcher captures its order response and registers it with openOrders", () => {
  const section = slice(
    'else if (action.action === "BUY") {',
    "// ── TIER 4: BUY OTM SPY PUT",
  );
  assert.ok(
    /const t3OrderResult\s*=\s*await alpaca\(\s*"\/v2\/orders"/.test(section),
    "the T3 BUY order submission's response was previously discarded entirely — it must now be captured into a variable",
  );
  assert.ok(
    /if\s*\(t3OrderResult\?\.id\)\s*\{[\s\S]*?openOrders\.push\(/.test(section),
    "the push must be gated on t3OrderResult actually carrying an id",
  );
  assert.ok(
    /orderId:\s*t3OrderResult\.id/.test(section),
    "the tracked entry must carry the real returned orderId",
  );
});
