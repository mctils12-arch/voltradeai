import { test } from "node:test";
import assert from "node:assert/strict";
import { shadowSleeves, shadowHeadline, shadowSourceNote, shadowCountWord, shadowRowWhat, shadowRowTime } from "./shadowView";
import type { ShadowEntryRow } from "./useBotData";

const row = (o: Partial<ShadowEntryRow> = {}): ShadowEntryRow => ({ time: "2026-10-28T14:05:00.000Z", sleeve: "stocks", ticker: "NVDA", side: "buy", notional: 1500.4, source: "executeTrades.stock", instrument: null, ...o });

test("nothing shadowed or not loaded -> no indicator", () => {
  assert.deepEqual(shadowSleeves(undefined), []);
  assert.deepEqual(shadowSleeves({ stocks: false, options: false, source: "env" }), []);
  assert.equal(shadowHeadline({ stocks: false, options: false, source: "config" }), null);
  assert.equal(shadowSourceNote(undefined), null);
});

test("headline names the shadowed sleeves and what stays live", () => {
  assert.equal(shadowHeadline({ stocks: true, options: true, source: "config" }), "New entries: SHADOW — stocks · options (logged, not traded). Floor + exits live.");
  assert.equal(shadowHeadline({ stocks: true, options: false, source: "config" }), "New entries: SHADOW — stocks (logged, not traded). Options, floor + exits live.");
  assert.equal(shadowHeadline({ stocks: false, options: true, source: "config" }), "New entries: SHADOW — options (logged, not traded). Stocks, floor + exits live.");
  assert.match(shadowSourceNote({ stocks: true, options: true, source: "env" })!, /VOLTRADE_SHADOW_ENTRIES/);
  assert.equal(shadowSourceNote({ stocks: true, options: true, source: "config" }), "system_config default");
});

test("count words and an honest empty state", () => {
  assert.equal(shadowCountWord(0), "no would-be entries yet today");
  assert.equal(shadowCountWord(undefined), "no would-be entries yet today");
  assert.equal(shadowCountWord(1), "1 would-be entry today");
  assert.equal(shadowCountWord(7), "7 would-be entries today");
});

test("a row says side, instrument, size — never a fake $0", () => {
  assert.equal(shadowRowWhat(row()), "BUY · ~$1,500");
  assert.equal(shadowRowWhat(row({ instrument: "NVDL" })), "BUY NVDL · ~$1,500");
  assert.equal(shadowRowWhat(row({ sleeve: "options", ticker: "AMD", side: "sell", notional: 15500, instrument: "AMD271119P00155000" })), "SELL AMD 155P 19-Nov-27 · ~$15,500");
  assert.equal(shadowRowWhat(row({ side: null, notional: null, sleeve: "options", ticker: "SVXY" })), "ENTRY · size not reported");
  assert.equal(shadowRowWhat(row({ notional: 0 })), "BUY · size not reported");
  assert.equal(shadowRowTime(row()), "10:05");
});
