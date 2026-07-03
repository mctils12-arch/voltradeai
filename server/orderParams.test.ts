import { test } from "node:test";
import assert from "node:assert/strict";
import { getOrderParams } from "./orderParams";

// KNOWN BROKEN #8: stock/ETF orders priced for the extended-hours session
// (4am-9:30am, 4pm-8pm ET) must carry extended_hours: true, or Alpaca queues
// them as ordinary day-limit orders that only become fillable at the next
// regular session open — silently defeating the wider-buffer pricing this
// function already computes for that window (e.g. a WS stop-loss firing at
// 6am would never actually attempt to fill until 9:30am).

test("extended-hours stop_loss sets extended_hours: true", () => {
  const params = getOrderParams(100, "stop_loss", 6.0); // 6:00 AM ET premarket
  assert.equal(params.type, "limit");
  assert.equal(params.extended_hours, true);
});

test("extended-hours trailing_stop sets extended_hours: true", () => {
  const params = getOrderParams(100, "trailing_stop", 17.0); // 5:00 PM ET after-hours
  assert.equal(params.type, "limit");
  assert.equal(params.extended_hours, true);
});

test("extended-hours take_profit sets extended_hours: true", () => {
  const params = getOrderParams(100, "take_profit", 4.5); // 4:30 AM ET
  assert.equal(params.type, "limit");
  assert.equal(params.extended_hours, true);
});

test("extended-hours new_entry sets extended_hours: true", () => {
  const params = getOrderParams(100, "new_entry", 19.0); // 7:00 PM ET
  assert.equal(params.type, "limit");
  assert.equal(params.extended_hours, true);
});

test("regular-hours orders never set extended_hours", () => {
  for (const ctx of ["stop_loss", "trailing_stop", "take_profit", "new_entry"] as const) {
    const params = getOrderParams(100, ctx, 12.0); // 12:00 PM ET — regular session
    assert.equal(params.extended_hours, undefined, `${ctx} should not set extended_hours during regular hours`);
  }
});

test("options orders never set extended_hours, even outside regular hours", () => {
  for (const hour of [6.0, 12.0, 17.0]) {
    const entry = getOrderParams(1.5, "options_entry", hour);
    const exit = getOrderParams(1.5, "options_exit", hour);
    assert.equal(entry.extended_hours, undefined, `options_entry@${hour} should never set extended_hours`);
    assert.equal(exit.extended_hours, undefined, `options_exit@${hour} should never set extended_hours`);
    assert.equal(entry.type, "limit");
  }
});
