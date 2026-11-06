import { test } from "node:test";
import assert from "node:assert/strict";
import {
  getActiveTicker,
  normalizeTicker,
  setActiveTicker,
  subscribeActiveTicker,
} from "./activeTicker";

test("normalizeTicker upper-cases, trims, and rejects garbage", () => {
  assert.equal(normalizeTicker(" spy "), "SPY");
  assert.equal(normalizeTicker("brk.b"), "BRK.B");
  assert.equal(normalizeTicker("rds-a"), "RDS-A");
  assert.equal(normalizeTicker(""), null);
  assert.equal(normalizeTicker(null), null);
  assert.equal(normalizeTicker("12345"), null, "must start with a letter");
  assert.equal(normalizeTicker("A".repeat(11)), null, "max 10 characters");
  assert.equal(normalizeTicker("<script>"), null);
});

test("setActiveTicker stores a normalized symbol and notifies on change only", () => {
  setActiveTicker(null);
  let fired = 0;
  const off = subscribeActiveTicker(() => { fired += 1; });

  setActiveTicker("nvda");
  assert.equal(getActiveTicker(), "NVDA");
  assert.equal(fired, 1);

  setActiveTicker("NVDA"); // same symbol → silent
  assert.equal(fired, 1);

  setActiveTicker("not a ticker!"); // garbage never replaces a good symbol
  assert.equal(getActiveTicker(), "NVDA");
  assert.equal(fired, 1);

  setActiveTicker(null); // explicit clear
  assert.equal(getActiveTicker(), null);
  assert.equal(fired, 2);

  off();
  setActiveTicker("SPY");
  assert.equal(fired, 2, "an unsubscribed listener must not fire");
  setActiveTicker(null);
});
