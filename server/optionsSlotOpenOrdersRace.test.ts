// OPEN-ORDERS-SLOT-RACE-FIX (KNOWN BROKEN #30, third recurrence, 2026-08-13
// found / 2026-08-15 fixed): live audit + order evidence showed the real
// filled-option-position count crossing 6→7 one full scan cycle after
// crossing 5→6 — the account's DAY LIMIT CSP orders (options_execution.py
// submit_options_order) return "submitted" with no fill confirmation, so a
// still-resting order is a real slot commitment invisible to both
// enforcement points' /v2/positions re-fetch (that re-fetch only reflects
// FILLED orders — it's the exact fix the 2026-08-03 race already shipped,
// which structurally cannot see this hole). Root cause + three candidate
// structural fixes filed in wishlist.md under "OPTIONS-SLOT CAP: THIRD
// RECURRENCE"; this pins the shipped fix, Option 1 ("count open orders
// too"): both options-opening paths now also fetch /v2/orders?status=open
// and add still-open SELL (opening) option orders to the live slot count
// via the new countOpenOptionsOpeningOrders() helper, mirroring
// countOptionsPositions()'s existing QQQ-convexity-overlay exclusion.
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

test("countOpenOptionsOpeningOrders is a single module-scope helper that only counts SELL (opening) option orders", () => {
  const defs = bot.match(/function countOpenOptionsOpeningOrders\(/g) || [];
  assert.equal(
    defs.length,
    1,
    "countOpenOptionsOpeningOrders must be defined exactly once (module scope) — duplicating this filter per call site is exactly the drift class that already caused KNOWN BROKEN #3 once",
  );
  const helperSection = slice(
    "function countOpenOptionsOpeningOrders(",
    "// ─── ET Hour Helper",
  );
  assert.ok(
    /o\.side\s*!==\s*"sell"/.test(helperSection),
    'the helper must exclude anything that is not a "sell" (opening) order — a "buy" order closes an existing short and frees a slot rather than consuming one',
  );
  assert.ok(
    helperSection.includes("isOptionSymbol(sym)"),
    "the helper must restrict to actual option symbols, not any open equity order",
  );
  assert.ok(
    helperSection.includes('sym.startsWith("QQQ")'),
    "the helper must exclude QQQ convexity-overlay puts, same as countOptionsPositions, so the two stay conceptually paired",
  );
});

test("executeTrades adds open sell-side option orders to its live options-slot count", () => {
  const execSection = slice("async function executeTrades(", "const pendingOrderIds");
  assert.ok(
    execSection.includes('await alpaca("/v2/orders?status=open")'),
    "executeTrades must fetch live open orders, not rely on /v2/positions (filled-only) alone",
  );
  assert.ok(
    execSection.includes("countOpenOptionsOpeningOrders("),
    "executeTrades must run the open-orders fetch through the shared helper",
  );
  assert.ok(
    /optionsSlotsUsed\s*=\s*optionsPositions\s*\+\s*optionsOrdersPending/.test(execSection),
    "executeTrades' live options-slot count must be filled positions PLUS still-open opening orders, not filled positions alone",
  );
});

test("tier dispatcher adds open sell-side option orders to its live options-slot count before dispatching SELL_CSP", () => {
  const dispatcherSection = slice(
    "TIER DISPATCHER — routes tier_actions to the right execution path",
    "// ── TIER 3: BUY SPY/QQQ at 2x",
  );
  assert.ok(
    dispatcherSection.includes('await alpaca("/v2/orders?status=open")'),
    "tier dispatcher must fetch live open orders, mirroring executeTrades' fix",
  );
  assert.ok(
    dispatcherSection.includes("countOpenOptionsOpeningOrders("),
    "tier dispatcher must run the open-orders fetch through the shared helper",
  );
  assert.ok(
    /tierOptionsSlotsUsed\s*=\s*countOptionsPositions\(/.test(dispatcherSection) &&
      dispatcherSection.includes("+ countOpenOptionsOpeningOrders("),
    "tier dispatcher's live options-slot count must be filled positions PLUS still-open opening orders",
  );
  // the (now-widened) cap check must still precede the loop body's SELL_CSP execution
  const gateIdx = dispatcherSection.search(/tierOptionsSlotsUsed\s*>=\s*MAX_OPTIONS_POSITIONS/);
  const openOrdersFetchIdx = dispatcherSection.indexOf('await alpaca("/v2/orders?status=open")');
  const submitIdx = dispatcherSection.indexOf("submit_options_order(contract)");
  assert.ok(
    openOrdersFetchIdx > 0 && gateIdx > openOrdersFetchIdx && submitIdx > gateIdx,
    "the open-orders fetch must happen before the slot-cap gate, which must still precede SELL_CSP submission",
  );
});

test("MAX_OPTIONS_POSITIONS is still declared exactly once — this fix widens what counts toward the cap, not the cap itself", () => {
  const decls = bot.match(/const MAX_OPTIONS_POSITIONS = \d+;/g) || [];
  assert.equal(
    decls.length,
    1,
    "the cap VALUE (6) is unchanged by this fix — only what counts against it changed — so no second declaration should appear",
  );
});
