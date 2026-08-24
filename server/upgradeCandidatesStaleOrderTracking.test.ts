// STALE-ORDER-SWEEP FIX (KNOWN BROKEN #33, research/open_questions.md) —
// runUpgradeCandidates()'s upgrade-buy order (server/bot.ts, inside
// runOvernightResearch) is the last of the #32-class order-registration
// gaps found while closing KNOWN BROKEN #32 and deliberately left open one
// logical change at a time (#33 here, #34 already fixed).
//
// When a higher-scoring overnight-research candidate is found for an
// existing lower-scoring position, this branch sells the old position and
// buys the new one via getOrderParams(betterPick.price || 0) — the same
// always-DAY-limit 'new_entry' default (orderParams.ts) confirmed for
// KNOWN BROKEN #32's morning-queue and Tier 3 BUY sites. The order response
// was discarded entirely (only used to call addPositionToMonitor), so
// sweepStaleOrders() had nothing to act on if the upgrade-buy never fills.
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

// The upgrade-buy branch, from its qty computation through the end of
// runUpgradeCandidates()'s try block.
const upgradeBuy = slice(
  "const upgradeQty = Math.floor(candidate.market_value / betterPick.price);",
  "} catch (e: any) { audit(\"UPGRADE-ERROR\", `Failed to upgrade: ${e.message}`); }",
);

test("upgrade-buy captures its order response instead of discarding it", () => {
  assert.ok(
    /const upgradeOrderResult\s*=\s*await alpaca\(\s*"\/v2\/orders"/.test(upgradeBuy),
    "the upgrade-buy order submission's response was previously discarded entirely — it must now be captured into a variable",
  );
});

test("upgrade-buy registers its order with openOrders, gated on a real order id", () => {
  assert.ok(
    /if\s*\(upgradeOrderResult\?\.id\)\s*\{[\s\S]*?openOrders\.push\(/.test(upgradeBuy),
    "the push must be gated on the submission actually carrying an id — a failed/malformed response must not enter the sweeper's list",
  );
  assert.ok(
    /orderId:\s*upgradeOrderResult\.id/.test(upgradeBuy),
    "the tracked entry must carry the real returned orderId, not a synthesized one",
  );
});

test("upgrade-buy's tracked order is NOT tagged isExit (it is an entry, not an exit)", () => {
  assert.ok(
    !/isExit:/.test(upgradeBuy),
    "this branch buys the replacement position — like the T3 BUY dispatcher, it must not carry isExit, or replaceIfBetter would treat it as unswappable buying power it never needs to protect",
  );
});

test("upgrade-buy's tracked order carries the ticker/score/qty/limit the sweeper's audit line reports", () => {
  assert.ok(
    /ticker:\s*betterPick\.ticker/.test(upgradeBuy) && /score:\s*betterPick\.score\s*\|\|\s*0/.test(upgradeBuy),
    "ticker and score must come from betterPick, the position actually being bought, not the sold candidate",
  );
  assert.ok(
    /qty:\s*upgradeQty/.test(upgradeBuy),
    "qty must be the computed upgrade quantity actually submitted",
  );
  assert.ok(
    /limitPrice:\s*Number\(upgradeOrderParams\.limit_price\)\s*\|\|\s*0/.test(upgradeBuy),
    "limitPrice must read the submitted params, matching every sibling entry site",
  );
});

test("openOrders.push now appears at least 11 times (the 9 pinned by #32, POS-KILL's #34, plus this one)", () => {
  const pushCalls = bot.match(/openOrders\.push\(/g) || [];
  assert.ok(
    pushCalls.length >= 11,
    `expected the 9 registrations pinned by finalOrderSitesStaleTracking.test.ts plus POS-KILL (#34) plus this one, found ${pushCalls.length}`,
  );
});
