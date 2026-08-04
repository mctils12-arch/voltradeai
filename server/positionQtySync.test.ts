// REPAIR 2026-08-04 (scheduled-routine session, live evidence from
// /api/diag/audit): syncMonitoredPositions() computed a monitored position's
// remainingQty by taking the min of the in-memory tracked value and the
// value persisted in voltrade_stop_state.json — but never bounded either
// against the broker's actual live qty. Nothing ever deletes a closed
// ticker's entry from voltrade_stop_state.json, so re-trading the same
// ticker later inherits the PRIOR position's stale remaining_qty.
//
// Live evidence (this session, /api/diag/audit?limit=200): AXTI's tracked
// remainingQty was stuck at 12 while the live Alpaca position held only 5
// shares (17 bought, 12 already sold) — 72 of the last 200 audit entries
// were AXTI POS-MONITOR-SYNC/WS-EXIT-ERROR noise, spanning at least
// 2026-08-04T14:17Z through 16:03Z (1h45m+) with every WS exit attempt
// failing with Alpaca 403 "insufficient qty available for order
// (requested: 12, existing_qty: 5)" and the position never actually
// closing — a live risk-management malfunction, not just log spam.
//
// This test pins the fix (cap remainingQty at the live qty, self-healing a
// stale persisted value on the next sync) and its accompanying audit
// visibility line, using the same source-text-extraction technique as
// wsPositionFeed.test.ts since bot.ts's position-sync closures aren't
// independently importable.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function syncMonitoredPositionsFn(): string {
  const start = bot.indexOf("async function syncMonitoredPositions()");
  assert.ok(start > 0, "syncMonitoredPositions() not found in bot.ts");
  const end = bot.indexOf("\n  function addPositionToMonitor(", start);
  assert.ok(end > start, "syncMonitoredPositions() body not found (addPositionToMonitor marker missing)");
  return bot.slice(start, end);
}

test("remainingQty is capped at the live broker qty, never the stale persisted/in-memory value", () => {
  const block = syncMonitoredPositionsFn();
  const remainingQtyLine = block.slice(block.indexOf("const remainingQty ="));
  assert.ok(
    /const remainingQty = Math\.min\(rawRemainingQty, qty\)/.test(remainingQtyLine),
    "remainingQty must be capped at `qty` (the live Alpaca-reported position size) — otherwise a stale voltrade_stop_state.json entry can force an exit order for more shares than are actually held, which Alpaca rejects with a 403 forever"
  );
});

test("a corrected (stale > live) remainingQty is audited, not silently clamped", () => {
  const block = syncMonitoredPositionsFn();
  assert.ok(
    block.includes('if (rawRemainingQty > qty) {') && block.includes('audit("POS-QTY-CORRECTED"'),
    "the correction must be visible in the audit log — this is exactly the kind of silent state drift that let the AXTI bug run for 1h45m+ before being noticed"
  );
});

test("originalQty is untouched by the fix — capping it at live qty would break normal scale-out sizing", () => {
  // originalQty legitimately exceeds the live qty any time scale-outs have
  // already reduced the position during ITS OWN lifecycle (e.g. originalQty
  // 17, one scale-out sold, live qty 12) — that is correct, not stale state.
  // Only remainingQty (shares still owed an exit right now) can never
  // exceed live qty; capping originalQty the same way would corrupt the
  // 1/3-of-original scale-out math for every normal multi-scale position.
  const block = syncMonitoredPositionsFn();
  assert.ok(
    block.includes("const originalQty = ps.original_qty || qty;"),
    "originalQty's computation must stay a plain fallback, not gain a live-qty cap"
  );
});

test("the remainingQty fix runs numerically as expected against the live AXTI evidence", () => {
  // Re-derive the exact expression from the source (not a hand-copied
  // reimplementation) and evaluate it against the real numbers from this
  // session's /api/diag/positions-detail + /api/diag/orders + /api/diag/audit:
  // bought 17, sold 12 already, live qty 5; stale tracked remainingQty 12.
  const block = syncMonitoredPositionsFn();
  const exprMatch = block.match(/const rawRemainingQty = existingPos\s*\n\s*\? Math\.min\(existingPos\.remainingQty, ps\.remaining_qty \|\| qty\)\s*\n\s*: \(ps\.remaining_qty \|\| qty\);\s*\n\s*const remainingQty = Math\.min\(rawRemainingQty, qty\);/);
  assert.ok(exprMatch, "expected exact rawRemainingQty/remainingQty expression not found — extraction regex is stale relative to the source");

  function computeRemainingQty(existingPos: { remainingQty: number } | undefined, psRemainingQty: number | undefined, qty: number) {
    const rawRemainingQty = existingPos
      ? Math.min(existingPos.remainingQty, psRemainingQty || qty)
      : (psRemainingQty || qty);
    return Math.min(rawRemainingQty, qty);
  }

  // First sync after a container restart with no in-memory state: only the
  // stale persisted value (12) is available, live qty is 5.
  assert.equal(computeRemainingQty(undefined, 12, 5), 5, "a stale persisted remaining_qty must be clamped to the live qty on the first sync");

  // Subsequent syncs with the stale value already latched into memory —
  // this is the actual stuck state observed live before the fix.
  assert.equal(computeRemainingQty({ remainingQty: 12 }, 12, 5), 5, "a stale in-memory remainingQty must also be clamped, not just the persisted fallback");

  // Normal case: no staleness, remainingQty legitimately equals live qty.
  assert.equal(computeRemainingQty({ remainingQty: 5 }, 5, 5), 5, "the fix must be a no-op when tracked and live qty already agree");
});
