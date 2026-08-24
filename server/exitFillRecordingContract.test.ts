// FILL-RECORDING CONTRACT FIX (KNOWN BROKEN #35, half two,
// research/open_questions.md) — both exit sites that call recordExitFill
// (the standalone POS-KILL forced-liquidation branch and
// checkPositionOnTick()'s stop-loss/trailing-stop/take-profit/time-stop
// exit) used to write an ML feedback fill record immediately after the
// order POST returned, using the WS `current`/`currentPrice` snapshot as
// `fillPrice` — unconditionally, whether or not the order had actually
// filled. Per orderParams.ts, both sites can submit a RESTING
// extended_hours limit rather than a market order, so a fill record was
// written for orders that might never fill, or that fill later at a
// different price — a synthetic-price record for exactly the
// worst-drawdown trades (POS-KILL) the ML loop should learn from most
// honestly. This is the PRIORITY-2 (MEASUREMENT INTEGRITY) half of #35;
// half one (the duplicate-submission guard) shipped separately in v1.0.779
// per PROMOTION RULE 5 (one logical change per PR).
//
// THE FIX: recording is deferred. Each exit push now carries a
// `pendingExitFill` (everything buildExitFillPayload() needs except
// fillPrice/qty), and sweepStaleOrders()'s new resolvePendingExitFill()
// queries Alpaca for the order's real terminal status before the order is
// ever dropped from openOrders — recordExitFill fires only on a confirmed
// "filled" status, at the confirmed filled_avg_price/filled_qty. A status
// query that itself fails leaves the order tracked for the next sweep
// rather than silently losing the fill.
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

const resolvePendingExitFill = slice(
  "async function resolvePendingExitFill(",
  "async function sweepStaleOrders() {",
);

const sweepStaleOrders = slice(
  "async function sweepStaleOrders() {",
  "async function replaceIfBetter(",
);

const posKill = slice(
  "if (pnlPct <= POSITION_KILL_LOSS_PCT) {",
  "} else if (pnlPct <= POSITION_WARN_LOSS_PCT) {",
);

const wsExit = slice(
  "// Submit sell/cover order for remaining shares",
  "// Write stop-loss cooldown",
);

test("resolvePendingExitFill records a fill only on a confirmed \"filled\" status from Alpaca, at the confirmed price/qty", () => {
  assert.ok(
    /const final\s*=\s*await alpaca\(`\/v2\/orders\/\$\{tracked\.orderId\}`\)/.test(resolvePendingExitFill),
    "must query the order's own status endpoint, not infer from the open-orders list alone",
  );
  assert.ok(
    /if\s*\(final\?\.status\s*===\s*"filled"\)\s*\{[\s\S]*?recordExitFill\(/.test(resolvePendingExitFill),
    "recordExitFill must be gated on status === \"filled\" — a resting/cancelled/expired order must record nothing",
  );
  assert.ok(
    /fillPrice:\s*Number\(final\.filled_avg_price\)\s*\|\|\s*tracked\.limitPrice/.test(resolvePendingExitFill),
    "the recorded fill price must come from Alpaca's confirmed filled_avg_price, not a WS current-price snapshot taken at submit time",
  );
  assert.ok(
    /qty:\s*Number\(final\.filled_qty\)\s*\|\|\s*tracked\.qty/.test(resolvePendingExitFill),
    "the recorded qty must come from Alpaca's confirmed filled_qty",
  );
});

test("resolvePendingExitFill is a no-op (returns true, records nothing) for a TrackedOrder with no pendingExitFill", () => {
  assert.ok(
    /if\s*\(!tracked\.pendingExitFill\)\s*return true;/.test(resolvePendingExitFill),
    "entry orders (and anything not migrated to this contract) must pass through unaffected — this keeps the function backward-compatible with every non-exit TrackedOrder",
  );
});

test("resolvePendingExitFill returns false (does not resolve) when the status query itself throws — a fill must never be silently dropped", () => {
  assert.ok(
    /catch\s*\{\s*return false;\s*\}/.test(resolvePendingExitFill),
    "a failed status query must return false so the caller retries next sweep instead of dropping a possibly-filled order unrecorded",
  );
});

test("sweepStaleOrders calls resolvePendingExitFill in both the stale-cancel-failure path and the Alpaca-reconciliation path, and only drops a tracked order once resolved", () => {
  assert.ok(
    /catch \(e[^)]*\) \{[\s\S]*?const resolved = await resolvePendingExitFill\(stale\);[\s\S]*?if \(!resolved\) continue;/.test(sweepStaleOrders),
    "when cancelling a stale order fails (it may have just filled), the fill must be resolved before the entry is dropped, and dropping must be skipped if resolution itself failed",
  );
  assert.ok(
    /const resolved = await resolvePendingExitFill\(tracked\);[\s\S]*?if \(resolved\) openOrders\.splice\(i, 1\)/.test(sweepStaleOrders),
    "an order that quietly left Alpaca's open list (the common case — it filled) must be resolved before removal, and removal must not happen if resolution failed",
  );
});

test("POS-KILL no longer calls recordExitFill directly at submit time — it defers via pendingExitFill", () => {
  assert.ok(
    !/recordExitFill\(buildExitFillPayload\(/.test(posKill),
    "the old unconditional recordExitFill(buildExitFillPayload(...)) call at submit time must be gone — recording now happens only in resolvePendingExitFill, on a confirmed fill",
  );
  assert.ok(
    /pendingExitFill:\s*\{[\s\S]*?exitReason:\s*"position_kill"/.test(posKill),
    "the openOrders.push for POS-KILL must attach a pendingExitFill carrying exitReason position_kill",
  );
});

test("POS-KILL warns via audit (not a silent no-op) when the liquidation order carries no id, since it then cannot be tracked for ML fill recording", () => {
  assert.ok(
    /audit\("POS-KILL-WARN"/.test(posKill),
    "an id-less response must be visible in the audit log — losing trackability for a forced liquidation should never be silent",
  );
});

test("the WS exit (stop-loss/trailing-stop/take-profit/time-stop) no longer calls recordExitFill directly at submit time — it defers via pendingExitFill", () => {
  assert.ok(
    !/recordExitFill\(buildExitFillPayload\(/.test(wsExit),
    "the old unconditional recordExitFill(buildExitFillPayload(...)) call at submit time must be gone from the WS exit path too",
  );
  assert.ok(
    /pendingExitFill:\s*\{[\s\S]*?entryDate:\s*pos\.entryDate/.test(wsExit),
    "the openOrders.push for the WS exit must attach a pendingExitFill carrying entryDate (used for days_held)",
  );
  assert.ok(
    /audit\("WS-EXIT-WARN"/.test(wsExit),
    "an id-less response must be visible in the audit log here too",
  );
});

test("recordExitFill itself is invoked from exactly one call site now: resolvePendingExitFill", () => {
  // recordExitFill's own declaration ("function recordExitFill(payload...")
  // also matches a bare /recordExitFill\(/ scan, so match the actual call
  // shape (always recordExitFill(buildExitFillPayload({ ... in this file)
  // to count invocations only, not the declaration.
  const calls = bot.match(/recordExitFill\(buildExitFillPayload\(/g) || [];
  assert.equal(
    calls.length, 1,
    `expected exactly one recordExitFill(buildExitFillPayload(...)) call site (inside resolvePendingExitFill) now that both submit-time call sites were removed, found ${calls.length}`,
  );
});
