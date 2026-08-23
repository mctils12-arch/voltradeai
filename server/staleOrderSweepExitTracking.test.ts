// STALE-ORDER-SWEEP FIX (KNOWN BROKEN #32, research/open_questions.md) —
// EXIT-SIDE HALF. Prior sessions this same date fixed the options-entry
// branches (staleOrderSweepOptionsTracking.test.ts) and the stock/ETF entry
// branches (executeTradesStaleOrderTracking.test.ts) and explicitly filed
// the exit side as the concrete NEXT: checkPositionOnTick()'s scale-out
// path (~line 5680) reads `openOrders` as its stated PRIMARY defense
// against submitting a duplicate scale-out sell order, but the scale-out
// order itself never registered with `openOrders` — so that guard has been
// dead code since its own introduction, resting entirely on its live-Alpaca
// backup query. The full stop-loss/trailing-stop/take-profit/time-stop exit
// a few hundred lines later has the identical gap (its own `orderResult`
// was captured but never used for anything).
//
// This pins both registrations, AND a new safety guard they required:
// `replaceIfBetter()` picks the lowest-score open order to cancel to free
// buying power for a better new ENTRY. Every exit pushed here carries
// score: 0 (no natural score in scope, same as the options-entry branches)
// — without exclusion, replaceIfBetter would treat a live protective
// stop-loss/take-profit order as the "weakest" candidate and cancel it to
// make room for an unrelated new trade, which doesn't even free the buying
// power it thinks it does (cancelling a SELL order doesn't return buying
// power) while stripping the position's only protective order. The new
// `TrackedOrder.isExit` flag marks these so replaceIfBetter excludes them.
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

test("openOrders.push now appears at least 6 times (2 options-entry + 2 stock/ETF-entry + 2 exit)", () => {
  const pushCalls = bot.match(/openOrders\.push\(/g) || [];
  assert.ok(
    pushCalls.length >= 6,
    "expected the 4 pre-existing entry-side pushes plus 2 new exit-side pushes — fewer means one of the new registrations is missing",
  );
});

test("TrackedOrder interface declares isExit so replaceIfBetter can distinguish exits from entries", () => {
  const section = slice("interface TrackedOrder {", "const openOrders: TrackedOrder[] = [];");
  assert.ok(/isExit\?\s*:\s*boolean/.test(section), "TrackedOrder must declare an optional isExit field");
});

test("replaceIfBetter excludes isExit orders before picking the weakest candidate to cancel", () => {
  const section = slice("async function replaceIfBetter(", "// ── Morning Queue Execution");
  assert.ok(
    /openOrders\.filter\(\s*o\s*=>\s*!o\.isExit\s*\)/.test(section),
    "replaceIfBetter must filter out isExit orders before reducing to the weakest score — otherwise a score:0 protective exit order looks like the weakest ENTRY and gets cancelled to make room for an unrelated new trade",
  );
  assert.ok(
    !/openOrders\.reduce\(/.test(section),
    "the weakest-order reduce must run over the filtered entry-only list, not openOrders directly",
  );
});

test("scale-out exit registers its order submission with openOrders, gated on a real order id, tagged isExit", () => {
  const section = slice(
    "// ── DUPLICATE SELL ORDER GUARD",
    "pos.scalesCompleted++;",
  );
  assert.ok(
    /const scaleOrderResult\s*=\s*await alpaca\(\s*"\/v2\/orders"/.test(section),
    "the scale-out order submission must capture its response so the returned order id is available to register",
  );
  assert.ok(
    /if\s*\(scaleOrderResult\?\.id\)\s*\{[\s\S]*?openOrders\.push\(/.test(section),
    "the push must be gated on scaleOrderResult actually carrying an id",
  );
  assert.ok(
    /orderId:\s*scaleOrderResult\.id/.test(section) && /isExit:\s*true/.test(section),
    "the tracked entry must carry the real returned orderId and be tagged isExit: true",
  );
});

test("full exit (stop-loss/trailing-stop/take-profit/time-stop) registers its order submission with openOrders, tagged isExit", () => {
  const section = slice(
    "// Submit sell/cover order for remaining shares",
    "const scaleNote",
  );
  assert.ok(
    section.includes("openOrders.push("),
    "the full exit path must push the returned order onto openOrders so sweepStaleOrders() can cancel a resting take-profit/extended-hours-stop limit that never fills",
  );
  assert.ok(
    /if\s*\(orderResult\?\.id\)\s*\{[\s\S]*?openOrders\.push\(/.test(section),
    "the push must be gated on orderResult actually carrying an id",
  );
  assert.ok(
    /orderId:\s*orderResult\.id/.test(section) && /isExit:\s*true/.test(section),
    "the tracked entry must carry the real returned orderId and be tagged isExit: true",
  );
});
