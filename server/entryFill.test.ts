// REPAIR 2026-07-31 (KNOWN BROKEN #12(c) contributor): the ETF entry
// payload must NOT trip ml_model_v2._is_exit_fill (no exit_context /
// exit_reason / is_close keys) so track_fill takes the entry branch and
// appends an outcome=None record — the exact record a later WS exit's
// _find_entry_record needs to find. A payload that accidentally carried
// any exit-detection key would silently revert this fix (the entry would
// itself be misfiled as an orphan exit).
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildEntryFillPayload, confirmedEntryFill, isTerminalOrderStatus, syntheticSlippagePct } from "./entryFill";

const FIXED_NOW = Date.parse("2026-07-31T14:30:00Z");

test("ETF entry payload never carries an exit-detection key", () => {
  const p = buildEntryFillPayload({
    ticker: "TQQQ", side: "buy", qty: 20, fillPrice: 85.5,
    session: "regular", volume: 5000000, score: 78,
    instrument: "etf", nowMs: FIXED_NOW, codeVersion: "1.0.565",
  });
  assert.equal(p.exit_context, undefined, "exit_context must be absent — entries are not exits");
  assert.equal(p.exit_reason, undefined);
  assert.equal(p.is_close, undefined);
});

test("ETF entry payload carries the fields track_fill's entry branch reads", () => {
  const p = buildEntryFillPayload({
    ticker: "TQQQ", side: "buy", qty: 20, fillPrice: 85.5,
    session: "regular", volume: 5000000, score: 78,
    instrument: "etf", nowMs: FIXED_NOW, codeVersion: "1.0.565",
  });
  assert.equal(p.ticker, "TQQQ");
  assert.equal(p.side, "buy");
  assert.equal(p.qty, 20);
  assert.equal(p.fill_price, 85.5);
  assert.equal(p.expected_price, 85.5);
  assert.equal(p.session, "regular");
  assert.equal(p.volume, 5000000);
  assert.equal(p.score, 78);
  assert.equal(p.code_version, "1.0.565");
  assert.equal(p.time_placed, "2026-07-31T14:30:00.000Z");
});

test("defaults: volume falls back to 1,000,000 when omitted", () => {
  const p = buildEntryFillPayload({
    ticker: "SOXL", side: "buy", qty: 5, fillPrice: 40,
    session: "regular", instrument: "etf",
    nowMs: FIXED_NOW, codeVersion: "1.0.565",
  });
  assert.equal(p.volume, 1000000);
});

// ── [RULE-REVIEW] 2026-10-27 REAL BROKER FILLS ─────────────────────────────

test("entry payload records the broker fill as fill_price and the old haircut only as synthetic_fill_price", () => {
  const p = buildEntryFillPayload({
    ticker: "ABC", side: "buy", qty: 10, fillPrice: 100.02,
    expectedPrice: 100, qtyRequested: 12,
    syntheticSlippagePct: 0.1, syntheticBase: "fill",
    orderStatus: "filled", orderId: "oid-1",
    session: "regular", volume: 3000000, score: 70,
    instrument: "stock", nowMs: FIXED_NOW, codeVersion: "1.0.1169",
  });
  assert.equal(p.fill_price, 100.02, "fill_price must be the broker's filled_avg_price");
  assert.equal(p.expected_price, 100);
  assert.equal(p.fill_source, "broker");
  assert.equal(p.synthetic_fill_price, 100.12, "regular path: the haircut sits on the broker fill (prior formula)");
  assert.equal(p.qty, 10);
  assert.equal(p.qty_requested, 12);
  assert.equal(p.partial_fill, true);
  assert.equal(p.order_status, "filled");
  assert.equal(p.order_id, "oid-1");
});

test("morning-queue synthetic estimate keeps the prior base (the queued quote, not the fill)", () => {
  const p = buildEntryFillPayload({
    ticker: "ABC", side: "buy", qty: 5, fillPrice: 101,
    expectedPrice: 100, syntheticSlippagePct: 0.2, syntheticBase: "expected",
    session: "morning_queue", instrument: "stock", nowMs: FIXED_NOW, codeVersion: "1.0.1169",
  });
  assert.equal(p.fill_price, 101);
  assert.equal(p.synthetic_fill_price, 100.2);
});

test("expectedPrice 0 is kept (record stays out of slippage stats) instead of defaulting to the fill", () => {
  const p = buildEntryFillPayload({
    ticker: "TQQQ", side: "buy", qty: 3, fillPrice: 85.5, expectedPrice: 0,
    session: "regular", instrument: "etf", nowMs: FIXED_NOW, codeVersion: "1.0.1169",
  });
  assert.equal(p.expected_price, 0);
  assert.equal(p.fill_price, 85.5);
});

test("syntheticSlippagePct reproduces the prior volume tiers exactly", () => {
  const lo = () => 0, hi = () => 1;
  assert.equal(syntheticSlippagePct(30e6, "regular", lo), 0.02);
  assert.equal(syntheticSlippagePct(30e6, "regular", hi), 0.05);
  assert.equal(syntheticSlippagePct(6e6, "regular", lo), 0.05);
  assert.ok(Math.abs(syntheticSlippagePct(2e6, "regular", hi) - 0.15) < 1e-12);
  assert.equal(syntheticSlippagePct(5e5, "regular", lo), 0.12);
  assert.equal(syntheticSlippagePct(undefined, "regular", lo), 0.12, "missing volume -> 1M -> lowest tier, as before");
  assert.equal(syntheticSlippagePct(30e6, "morning_queue", lo), 0.05);
  assert.equal(syntheticSlippagePct(6e6, "morning_queue", lo), 0.08);
  assert.ok(Math.abs(syntheticSlippagePct(2e6, "morning_queue", hi) - 0.25) < 1e-12);
});

test("confirmedEntryFill returns the broker fill only when shares actually filled", () => {
  assert.deepEqual(confirmedEntryFill({ status: "filled", filled_qty: "10", filled_avg_price: "100.02" }), { qty: 10, price: 100.02 });
  assert.deepEqual(confirmedEntryFill({ status: "canceled", filled_qty: "4", filled_avg_price: "99.5" }), { qty: 4, price: 99.5 },
    "a partial fill that was then cancelled still holds those shares");
  assert.equal(confirmedEntryFill({ status: "canceled", filled_qty: "0", filled_avg_price: null }), null,
    "an unfilled order must never produce an entry record");
  assert.equal(confirmedEntryFill({ status: "new", filled_qty: "0" }), null);
  assert.equal(confirmedEntryFill(null), null);
});

test("isTerminalOrderStatus separates working orders from finished ones", () => {
  for (const s of ["filled", "canceled", "expired", "rejected", "done_for_day"]) assert.equal(isTerminalOrderStatus(s), true, s);
  for (const s of ["new", "accepted", "pending_new", "partially_filled", "pending_cancel", "", undefined]) {
    assert.equal(isTerminalOrderStatus(s), false, String(s));
  }
});
