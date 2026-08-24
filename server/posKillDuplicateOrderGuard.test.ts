// DUPLICATE LIQUIDATION GUARD (KNOWN BROKEN #35, research/open_questions.md) —
// the POS-KILL per-position forced-liquidation branch in
// `syncMonitoredPositions()` (server/bot.ts) had no open-order pre-check.
//
// `syncMonitoredPositions()` is invoked from three sites, one of them
// `tier1Reflex` (~45s cadence). `getOrderParams(current, 'stop_loss')` returns
// a market order during regular hours (fills immediately, so the next sync
// sees no position) but a RESTING `extended_hours` limit outside them
// (orderParams.ts) — so a position sitting below -25% pre-market or
// after-hours got a fresh liquidation order submitted on every sync until one
// filled, each submission also writing another `recordExitFill` record at a
// synthetic price for a fill that may never have happened. #34's sweeper
// registration bounds the pile-up at ~12 minutes' worth; only a pre-check
// prevents it.
//
// The guard mirrors the scale-out DUPLICATE SELL ORDER GUARD in the same file:
// the free in-memory `openOrders` check first, the authoritative Alpaca query
// second (in-memory is empty after a restart — `sweepStaleOrders()` only ever
// prunes that list against the broker, never repopulates it).
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

const posKill = slice(
  "if (pnlPct <= POSITION_KILL_LOSS_PCT) {",
  "} else if (pnlPct <= POSITION_WARN_LOSS_PCT) {",
);

// Everything before the submission — a guard that runs AFTER the POST guards
// nothing, so every assertion below is made against this prefix only.
const beforeSubmit = posKill.slice(0, posKill.indexOf("const killOrderResult"));

test("POS-KILL checks the in-memory openOrders list for a same-side order before submitting", () => {
  assert.ok(
    /openOrders\.find\(o\s*=>\s*o\.ticker\s*===\s*ticker\s*&&\s*o\.side\s*===\s*closeSide\)/.test(beforeSubmit),
    "the branch must consult the tracked-order list (ticker + close side) before submitting a second liquidation",
  );
  assert.ok(
    /if\s*\(trackedExit\)\s*\{[\s\S]*?continue;/.test(beforeSubmit),
    "a tracked same-side order must short-circuit the branch — not merely be logged",
  );
});

test("POS-KILL also asks Alpaca directly, because openOrders is empty after a restart", () => {
  assert.ok(
    /await alpaca\(`\/v2\/orders\?status=open&symbols=\$\{ticker\}&side=\$\{closeSide\}`\)/.test(beforeSubmit),
    "the authoritative broker query must run before submitting (the in-memory list is process-local and never repopulated from Alpaca)",
  );
  assert.ok(
    /if\s*\(brokerExitOpen\s*>\s*0\)\s*\{[\s\S]*?continue;/.test(beforeSubmit),
    "an existing open same-side order at the broker must short-circuit the branch",
  );
});

test("a failed broker query does not leave a -25% position unliquidated", () => {
  // The catch must NOT set the skip condition: losing the check is not a
  // reason to stop liquidating, only a reason to fall back to the in-memory
  // guard that already ran.
  const catchAt = beforeSubmit.indexOf("} catch (_) {");
  assert.ok(catchAt > 0, "the broker query must be wrapped in its own catch — an unhandled throw here would abort the whole sync loop");
  const catchBlock = beforeSubmit.slice(catchAt);
  assert.ok(
    !/brokerExitOpen\s*=/.test(catchBlock.slice(0, catchBlock.indexOf("}"))),
    "the catch must leave brokerExitOpen at 0 so submission still proceeds when the broker check itself fails",
  );
});

test("the skip path records why it skipped and never writes an exit fill", () => {
  assert.ok(
    /audit\("POS-KILL-SKIP"/.test(beforeSubmit),
    "a suppressed liquidation must be visible in the audit log — a silent skip on a -25% position is indistinguishable from a dead code path",
  );
  // The integrity half of #35: each repeat submission also called
  // recordExitFill unconditionally, feeding the ML loop duplicate exits at a
  // synthetic price for exactly the worst-drawdown trades. Skipping the
  // submission must skip that too.
  assert.ok(
    !/recordExitFill/.test(beforeSubmit),
    "no exit fill may be recorded on the skip path — the duplicate-exit records are the PRIORITY-2 half of this bug",
  );
});

test("the guard runs before the submission, not after it", () => {
  const guardAt = posKill.indexOf("trackedExit");
  const submitAt = posKill.indexOf("const killOrderResult");
  assert.ok(guardAt > 0 && submitAt > 0, "both the guard and the submission must exist in this branch");
  assert.ok(
    guardAt < submitAt,
    "a duplicate-order guard that runs after the order POST prevents nothing",
  );
});
