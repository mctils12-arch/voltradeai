// REPAIR 2026-07-31 (KNOWN BROKEN #12(c) contributor): the ETF entry
// payload must NOT trip ml_model_v2._is_exit_fill (no exit_context /
// exit_reason / is_close keys) so track_fill takes the entry branch and
// appends an outcome=None record — the exact record a later WS exit's
// _find_entry_record needs to find. A payload that accidentally carried
// any exit-detection key would silently revert this fix (the entry would
// itself be misfiled as an orphan exit).
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildEntryFillPayload } from "./entryFill";

const FIXED_NOW = Date.parse("2026-07-31T14:30:00Z");

test("ETF entry payload never carries an exit-detection key", () => {
  const p: any = buildEntryFillPayload({
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
