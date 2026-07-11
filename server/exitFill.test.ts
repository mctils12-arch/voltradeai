// REPAIR 2026-07-06 (R12-D2): the exit payload must satisfy the exact
// contract ml_model_v2.track_fill's exit path keys on — exit_context
// present (=> _is_exit_fill true), a quantity field v1.0.153 accepts, and
// the bot-accounted pnl_pct inside exit_context (it wins over the price
// recomputation). A payload drifting from this contract silently reverts
// the loop to "no live outcome ever recorded".
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildExitFillPayload } from "./exitFill";

const FIXED_NOW = Date.parse("2026-07-06T18:00:00Z");

test("full exit payload carries the exit-detection contract", () => {
  const p = buildExitFillPayload({
    ticker: "VXUS", exitSide: "sell", qty: 40, fillPrice: 62.31,
    pnlPct: -1.4, exitReason: "time_stop",
    entryDate: "2026-06-29T14:30:00Z", nowMs: FIXED_NOW,
    codeVersion: "1.0.269",
  });
  // _is_exit_fill fires on exit_context — the load-bearing key
  assert.ok(p.exit_context, "exit_context must be present");
  assert.equal(p.exit_context.pnl_pct, -1.4);
  assert.equal(p.exit_context.exit_reason, "time_stop");
  assert.equal(p.exit_context.days_held, 7);
  assert.equal(p.exit_reason, "time_stop");
  // v1.0.153 qty resolution: qty | qty_filled | qty_requested
  assert.equal(p.qty_filled, 40);
  assert.equal(p.ticker, "VXUS");
  assert.equal(p.side, "sell");
  assert.equal(p.fill_price, 62.31);
  assert.equal(p.code_version, "1.0.269");
});

// REPAIR 2026-07-11 (MEASUREMENT INTEGRITY): track_fill's Python side used
// to hardcode every record's code_version to the literal "1.0.34" forever,
// no matter what caller or app version produced it — verified live via
// /api/diag/ml on 2026-07-11 (feedback records dated 2026-07-10 tagged
// "1.0.34" while the deployed app was v1.0.269). This payload is the one
// place that can carry the real running version across the bot.ts ->
// track_fill boundary; a payload missing code_version silently reverts to
// that permanently-stale attribution.
test("exit payload always carries the caller's real code_version", () => {
  const p = buildExitFillPayload({
    ticker: "SMH", exitSide: "sell", qty: 12, fillPrice: 611.7,
    pnlPct: -0.1, exitReason: "stop_loss", nowMs: FIXED_NOW,
    codeVersion: "1.0.269",
  });
  assert.equal(p.code_version, "1.0.269");
});

test("short cover exits use side=buy and still carry exit_context", () => {
  const p = buildExitFillPayload({
    ticker: "XYZ", exitSide: "buy", qty: 15, fillPrice: 9.8,
    pnlPct: 3.2, exitReason: "take_profit", nowMs: FIXED_NOW,
    codeVersion: "1.0.269",
  });
  assert.equal(p.side, "buy");
  assert.ok(p.exit_context);
  assert.equal(p.exit_context.days_held, 0, "no entryDate -> 0, never NaN");
});

test("bad entryDate degrades to days_held=0, never NaN", () => {
  const p = buildExitFillPayload({
    ticker: "ABC", exitSide: "sell", qty: 1, fillPrice: 10,
    pnlPct: 0.5, exitReason: "stop_loss", entryDate: "not-a-date",
    nowMs: FIXED_NOW, codeVersion: "1.0.269",
  });
  assert.equal(p.exit_context.days_held, 0);
  assert.ok(!Number.isNaN(p.exit_context.days_held));
});

test("position_kill liquidation is a recordable exit reason", () => {
  const p = buildExitFillPayload({
    ticker: "DEF", exitSide: "sell", qty: 30, fillPrice: 4.1,
    pnlPct: -26.3, exitReason: "position_kill", nowMs: FIXED_NOW,
    codeVersion: "1.0.269",
  });
  assert.equal(p.exit_context.exit_reason, "position_kill");
  assert.equal(p.exit_context.pnl_pct, -26.3);
});
