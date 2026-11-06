import { test } from "node:test";
import assert from "node:assert/strict";
import {
  IDLE_BACKDROP,
  getTradeBackdrop,
  resetTradeBackdrop,
  setTradeBackdrop,
  subscribeTradeBackdrop,
} from "./tradeBackdrop";

test("starts idle: no loading, no data, no ticker", () => {
  resetTradeBackdrop();
  assert.deepEqual(getTradeBackdrop(), IDLE_BACKDROP);
});

test("setTradeBackdrop merges partial updates and notifies once per real change", () => {
  resetTradeBackdrop();
  let fired = 0;
  const off = subscribeTradeBackdrop(() => { fired += 1; });

  setTradeBackdrop({ isLoading: true });
  assert.deepEqual(getTradeBackdrop(), { isLoading: true, hasData: false, ticker: null });
  assert.equal(fired, 1);

  setTradeBackdrop({ isLoading: true }); // no change → no notification
  assert.equal(fired, 1);

  setTradeBackdrop({ isLoading: false, hasData: true, ticker: "SPY" });
  assert.deepEqual(getTradeBackdrop(), { isLoading: false, hasData: true, ticker: "SPY" });
  assert.equal(fired, 2);

  off();
  setTradeBackdrop({ ticker: "QQQ" });
  assert.equal(fired, 2, "an unsubscribed listener must not fire");
  assert.equal(getTradeBackdrop().ticker, "QQQ");
});

test("a ticker of null clears the beacon; undefined leaves it alone", () => {
  resetTradeBackdrop();
  setTradeBackdrop({ ticker: "NVDA", hasData: true });
  setTradeBackdrop({ isLoading: true }); // ticker undefined → kept
  assert.equal(getTradeBackdrop().ticker, "NVDA");
  setTradeBackdrop({ ticker: null });
  assert.equal(getTradeBackdrop().ticker, null);
});

test("resetTradeBackdrop returns to idle and notifies", () => {
  setTradeBackdrop({ isLoading: true, hasData: true, ticker: "AAPL" });
  let fired = 0;
  const off = subscribeTradeBackdrop(() => { fired += 1; });
  resetTradeBackdrop();
  assert.deepEqual(getTradeBackdrop(), IDLE_BACKDROP);
  assert.equal(fired, 1);
  off();
});
