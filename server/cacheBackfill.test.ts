import { test } from "node:test";
import assert from "node:assert/strict";
import { resolveCacheItems } from "./cacheBackfill";

test("resolveCacheItems: non-empty live items win even with an existing cache", () => {
  let backfillCalls = 0;
  const backfill = () => { backfillCalls++; return [999]; };
  const out = resolveCacheItems(true, [1, 2, 3], backfill);
  assert.deepEqual(out, [1, 2, 3]);
  assert.equal(backfillCalls, 0);
});

test("resolveCacheItems: non-empty live items win with no prior cache, backfill untouched", () => {
  let backfillCalls = 0;
  const backfill = () => { backfillCalls++; return [999]; };
  const out = resolveCacheItems(false, [1], backfill);
  assert.deepEqual(out, [1]);
  assert.equal(backfillCalls, 0);
});

test("resolveCacheItems: empty live items with an existing cache leaves the cache alone (never overwrites a good cache with a stale archive read)", () => {
  let backfillCalls = 0;
  const backfill = () => { backfillCalls++; return [999]; };
  const out = resolveCacheItems(true, [], backfill);
  assert.equal(out, null);
  assert.equal(backfillCalls, 0);
});

test("resolveCacheItems: empty live items with no cache and a non-empty archive backfills", () => {
  let backfillCalls = 0;
  const backfill = () => { backfillCalls++; return [7, 8]; };
  const out = resolveCacheItems(false, [], backfill);
  assert.deepEqual(out, [7, 8]);
  assert.equal(backfillCalls, 1);
});

test("resolveCacheItems: empty live items with no cache and an empty archive stays null", () => {
  const out = resolveCacheItems(false, [], () => []);
  assert.equal(out, null);
});

test("resolveCacheItems: models the thrown-fetch-error call site ([] standing in for 'fetch threw', not 'feed returned zero rows')", () => {
  const out = resolveCacheItems(false, [], () => [42]);
  assert.deepEqual(out, [42]);
});
