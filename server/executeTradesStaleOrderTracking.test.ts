// STALE-ORDER-SWEEP FIX continuation (KNOWN BROKEN #32, research/open_questions.md):
// the 2026-08-22 session shipped the options-entry half of this fix (SELL_CSP +
// BUY_PUT tier-dispatch branches now register with `openOrders`, pinned by
// staleOrderSweepOptionsTracking.test.ts) and explicitly filed the rest — every
// other order-submission call site in this file — as its own follow-up, naming
// `executeTrades`' entry paths as the highest-value next target since that is
// where the bulk of daily order volume flows.
//
// This pins that follow-up: executeTrades' two stock-side entry paths (the ETF
// 2x-leverage branch and the main stock/fallback-from-options branch) now
// register a successful order submission with `openOrders` too, so
// sweepStaleOrders() (tier1Reflex, every ~45s) can actually cancel a DAY limit
// entry that never fills — getOrderParams' 'new_entry' context is always a
// limit order, never a market order, so this is a real (not theoretical) gap
// for every stock/ETF entry the bot places.
//
// Source-scraping style matching this file's own staleOrderSweepOptionsTracking
// .test.ts precedent, since `openOrders`/`executeTrades` are closures inside the
// un-exported `registerBotRoutes()` function with no export surface.
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

test("openOrders.push now appears at least 4 times (SELL_CSP, BUY_PUT, ETF entry, main stock entry)", () => {
  const pushCalls = bot.match(/openOrders\.push\(/g) || [];
  assert.ok(
    pushCalls.length >= 4,
    "expected the two pre-existing options-entry pushes plus two new stock/ETF-entry pushes — fewer means one of the new registrations is missing",
  );
});

test("ETF 2x-leverage entry branch registers its order with openOrders", () => {
  const section = slice(
    'if (trade.instrument === "etf" && trade.instrument_ticker',
    "addPositionToMonitor(etfTicker,",
  );
  assert.ok(
    /const etfOrderResult\s*=\s*await alpaca\(\s*"\/v2\/orders"/.test(section),
    "the ETF order submission must capture its response so the returned order id is available to register",
  );
  assert.ok(
    /if\s*\(etfOrderResult\?\.id\)\s*\{[\s\S]*?openOrders\.push\(/.test(section),
    "the push must be gated on etfOrderResult actually carrying an id",
  );
  assert.ok(
    /orderId:\s*etfOrderResult\.id/.test(section) && /ticker:\s*etfTicker/.test(section),
    "the tracked entry must carry the real returned orderId and the ETF ticker (not the underlying trade.ticker)",
  );
});

test("main stock entry branch (default or fallback-from-options) registers its order with openOrders", () => {
  const section = slice(
    "// ── Stock execution (default or fallback from options) ──",
    "// ── Batch fill confirmation",
  );
  assert.ok(
    section.includes("openOrders.push("),
    "the main stock entry path must push the returned order onto openOrders so sweepStaleOrders() can cancel it if it never fills",
  );
  assert.ok(
    /if\s*\(orderId\)\s*\{[\s\S]*?openOrders\.push\(/.test(section),
    "the push must be gated on orderId actually being present",
  );
  assert.ok(
    /orderId,[\s\S]{0,40}ticker:\s*trade\.ticker,[\s\S]{0,40}score:\s*trade\.score/.test(section),
    "the tracked entry must carry the real orderId, ticker, and the trade's actual score (unlike the options branches, a real score is available here — useful for replaceIfBetter's weakest-score comparison)",
  );
  // the new push must come after the pre-existing pendingOrderIds collection,
  // not replace or precede the batch-fill-confirmation bookkeeping it relies on.
  const pendingIdx = section.indexOf("pendingOrderIds.push(");
  const openOrdersIdx = section.indexOf("openOrders.push(");
  assert.ok(pendingIdx >= 0 && pendingIdx < openOrdersIdx, "pendingOrderIds (batch fill confirmation) must remain intact and precede the new openOrders registration");
});

test("both new registrations use a real limitPrice sourced from the actual order params, not a hardcoded value", () => {
  const etfSection = slice(
    'if (trade.instrument === "etf" && trade.instrument_ticker',
    "addPositionToMonitor(etfTicker,",
  );
  const stockSection = slice(
    "// ── Stock execution (default or fallback from options) ──",
    "// ── Batch fill confirmation",
  );
  assert.ok(/limitPrice:\s*Number\(etfOrderParams\.limit_price\)/.test(etfSection));
  assert.ok(/limitPrice:\s*Number\(orderParams\.limit_price\)/.test(stockSection));
});
