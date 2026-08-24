// STALE-ORDER-SWEEP FIX (KNOWN BROKEN #34, research/open_questions.md) —
// the standalone POS-KILL per-position forced-liquidation branch in
// `syncMonitoredPositions()` (server/bot.ts). This is NOT the exit path
// KNOWN BROKEN #32 closed: #32's exit-side half covers
// `checkPositionOnTick()`'s stop-loss/trailing-stop/take-profit/time-stop
// branch and the scale-out branch. POS-KILL is a separate risk check with
// its own `alpaca("/v2/orders", ...)` call whose response was discarded, so
// `sweepStaleOrders()` (tier1Reflex, ~45s) had nothing to cancel.
//
// Why it can rest: `getOrderParams(current, 'stop_loss')` returns
// `type: "market"` during regular hours (fills immediately) but
// `type: "limit", extended_hours: true` outside them (orderParams.ts) — so a
// position crossing the -25% kill threshold pre-market or after-hours could
// leave an unfilled resting order invisible to the sweeper, on exactly the
// forced-liquidation order where a stuck fill matters most.
//
// Pushing unconditionally on a real order id (rather than gating on
// `type !== "market"` the way the manual /api/bot/trade route does) matches
// the two sibling EXIT sites, whose contexts are likewise market-during-RTH:
// sweepStaleOrders' own Alpaca reconciliation prunes any already-filled
// order on its next pass, so a tracked market order is harmless.
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

// The POS-KILL branch, from its threshold test through to the warn branch
// that immediately follows it.
const posKill = slice(
  "if (pnlPct <= POSITION_KILL_LOSS_PCT) {",
  "} else if (pnlPct <= POSITION_WARN_LOSS_PCT) {",
);

test("POS-KILL forced-liquidation captures its order response instead of discarding it", () => {
  assert.ok(
    /const killOrderResult\s*=\s*await alpaca\(\s*"\/v2\/orders"/.test(posKill),
    "the POS-KILL order submission's response was previously discarded entirely — it must now be captured into a variable",
  );
});

test("POS-KILL registers its order with openOrders, gated on a real order id", () => {
  assert.ok(
    /if\s*\(killOrderResult\?\.id\)\s*\{[\s\S]*?openOrders\.push\(/.test(posKill),
    "the push must be gated on the submission actually carrying an id — a failed/malformed response must not enter the sweeper's list",
  );
  assert.ok(
    /orderId:\s*killOrderResult\.id/.test(posKill),
    "the tracked entry must carry the real returned orderId, not a synthesized one",
  );
});

test("POS-KILL's tracked order is tagged isExit so replaceIfBetter cannot cancel it", () => {
  assert.ok(
    /isExit:\s*true/.test(posKill),
    "a forced liquidation is an exit — without isExit its score: 0 would make it look like the weakest ENTRY order, and replaceIfBetter would strip a dying position's only liquidation order to free buying power that cancelling a SELL never actually frees",
  );
  // Guard the mechanism the tag depends on, in the same test: if
  // replaceIfBetter ever stopped filtering on isExit, the tag above would be
  // decorative and this branch would silently regress.
  const replace = slice("async function replaceIfBetter(", "async function executeMorningQueue(");
  assert.ok(
    /openOrders\.filter\(o\s*=>\s*!o\.isExit\)/.test(replace),
    "replaceIfBetter must keep excluding isExit orders for the tag above to protect anything",
  );
});

test("POS-KILL's tracked order carries the side/qty/limit the sweeper's audit line reports", () => {
  assert.ok(
    /side:\s*closeSide/.test(posKill) && /\bqty,/.test(posKill),
    "side and qty must come from the branch's own computed close side and position size, so the SWEEP audit line names the real order",
  );
  assert.ok(
    /limitPrice:\s*Number\(orderParams\.limit_price\)\s*\|\|\s*0/.test(posKill),
    "limitPrice must read the submitted params (0 for the regular-hours market case), matching the sibling exit sites",
  );
});

test("openOrders.push now appears at least 10 times (the 9 pinned by the #32 chain plus POS-KILL)", () => {
  const pushCalls = bot.match(/openOrders\.push\(/g) || [];
  assert.ok(
    pushCalls.length >= 10,
    `expected the 9 registrations pinned by finalOrderSitesStaleTracking.test.ts plus this one, found ${pushCalls.length}`,
  );
});
