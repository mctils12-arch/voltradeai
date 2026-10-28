// MORNING-QUEUE NAKED-SHORT FIX (2026-10-27, audit §6.8 N2) — regression tests.
//
// deep_score "SELL OPTIONS" trades carry trade_type="options", side="sell",
// shares>0. Off-hours they are queued verbatim; executeMorningQueue's stock-
// short block skipped trade_type "options", so the trade reached the STOCK
// order body with side:"sell" (a naked short) and was monitored as "long".
// These tests pin the pure decision and its wiring into bot.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { decideMorningQueueEntry, isOptionsIntended } from "./morningQueueEntry";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function slice(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start);
  assert.ok(end > start, `end marker not found after ${fromMarker}: ${toMarker}`);
  return bot.slice(start, end);
}

// The exact shape bot_engine._scan_market_inner emits for a deep_score
// SELL OPTIONS candidate (trade_type/side/action from deep_score ~1862-1866).
const SELL_OPTIONS = {
  ticker: "XYZ", side: "sell", trade_type: "options", action: "SELL OPTIONS",
  action_label: "SELL OPTIONS", shares: 40, use_options: false, options_strategy: "stock",
};

test("an options-typed SELL is never submitted from the morning queue", () => {
  const v = decideMorningQueueEntry(SELL_OPTIONS);
  assert.equal(v.submit, false);
  if (v.submit) return;
  assert.equal(v.auditTag, "MORNING-OPTIONS-SKIP");
  assert.match(v.reason, /XYZ/);
  assert.match(v.reason, /never submit it as a stock order/);
});

test("any trade_type 'options' entry is dropped, whatever its side", () => {
  for (const side of ["buy", "sell", "short", undefined]) {
    const v = decideMorningQueueEntry({ ticker: "Q", trade_type: "options", side });
    assert.equal(v.submit, false, `side=${side} must not reach the stock order body`);
  }
});

test("the extreme-mover queue's bearish options setups are dropped, not flipped long", () => {
  // bot.ts runOvernightResearch pushes {side:'sell', use_options:true,
  // options_strategy:'buy_put'|'sell_cash_secured_put'} with NO trade_type.
  for (const strat of ["buy_put", "sell_cash_secured_put"]) {
    const t = { ticker: "MOV", side: "sell", use_options: true, options_strategy: strat, action_label: "BUY PUT" };
    assert.equal(isOptionsIntended(t), true);
    assert.equal(decideMorningQueueEntry(t).submit, false);
  }
});

test("a plain stock BUY submits buy and is monitored long", () => {
  const v = decideMorningQueueEntry({ ticker: "AAA", side: "buy", trade_type: "stock" });
  assert.deepEqual(v, { submit: true, side: "buy", monitorSide: "long", convertedFrom: null });
});

test("a stock BUY flagged use_options still submits (in-hours falls back to stock too)", () => {
  const v = decideMorningQueueEntry({ ticker: "AAA", side: "buy", trade_type: "stock", use_options: true });
  assert.equal(v.submit, true);
});

test("a stock sell/short keeps PR #53's convert-to-BUY and is tracked with the submitted side", () => {
  for (const side of ["sell", "short", "SELL"]) {
    const v = decideMorningQueueEntry({ ticker: "S", side, trade_type: "stock" });
    assert.equal(v.submit, true);
    if (!v.submit) continue;
    assert.equal(v.side, "buy");
    assert.equal(v.monitorSide, "long", "monitor side must match the side actually submitted");
    assert.equal(v.convertedFrom, side.toLowerCase());
  }
});

test("missing side / trade_type defaults to a stock buy", () => {
  const v = decideMorningQueueEntry({ ticker: "D" });
  assert.equal(v.submit, true);
});

// ── bot.ts wiring ────────────────────────────────────────────────────────────

const mq = slice("async function executeMorningQueue(", "// Clear the queue");

test("executeMorningQueue consults the verdict before any order work and drops on submit=false", () => {
  const call = mq.indexOf("decideMorningQueueEntry(trade)");
  assert.ok(call > 0, "executeMorningQueue must call decideMorningQueueEntry(trade)");
  const after = mq.slice(call, call + 400);
  assert.match(after, /if \(!mqVerdict\.submit\) \{\s*audit\(mqVerdict\.auditTag, mqVerdict\.reason\);\s*continue;/);
  assert.ok(call < mq.indexOf("Pre-market price check"), "decide before the snapshot gap check");
  assert.ok(call < mq.indexOf('alpaca("/v2/orders"'), "decide before the order submission");
});

test("the old trade_type-skipping short block is gone", () => {
  assert.doesNotMatch(mq, /trade\.trade_type !== "options" && \(trade\.side === "short" \|\| trade\.side === "sell"\)/);
});

test("the stock order body, the open-order record and the monitor all use the verdict's sides", () => {
  const orderBody = mq.slice(mq.indexOf('alpaca("/v2/orders"'), mq.indexOf("MORNING-TRADE"));
  assert.match(orderBody, /side: mqVerdict\.side,/);
  assert.doesNotMatch(orderBody, /side: trade\.side/);
  const push = mq.slice(mq.indexOf("openOrders.push("), mq.indexOf("openOrders.push(") + 300);
  assert.match(push, /side: mqVerdict\.side,/);
  assert.match(mq, /addPositionToMonitor\(trade\.ticker, mqVerdict\.monitorSide,/);
  assert.doesNotMatch(mq, /addPositionToMonitor\(trade\.ticker, "long",/);
});

test("bot.ts imports the helper", () => {
  assert.match(bot, /import \{ decideMorningQueueEntry \} from "\.\/morningQueueEntry";/);
});
