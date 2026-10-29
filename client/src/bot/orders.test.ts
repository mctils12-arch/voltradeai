import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { unwrapOpenOrders } from "./orders";

type O = { id: string; symbol: string; side: string };
const A: O = { id: "1", symbol: "XYZ", side: "buy" };

test("the /api/orders/open wrapper is unwrapped to the order list", () => {
  // The exact shape server/routes.ts sends: { orders }.
  assert.deepEqual(unwrapOpenOrders<O>({ orders: [A] }), [A]);
  assert.equal(unwrapOpenOrders<O>({ orders: [A] }).length, 1, "the panel's orders.length must be the real count, not undefined");
  assert.deepEqual(unwrapOpenOrders<O>({ orders: [] }), []);
});

test("a bare array is accepted unchanged", () => {
  assert.deepEqual(unwrapOpenOrders<O>([A]), [A]);
});

test("an unexpected shape throws (query error state) rather than reading as 'none working'", () => {
  assert.throws(() => unwrapOpenOrders<O>({ error: "x" }));
  assert.throws(() => unwrapOpenOrders<O>(null));
});

test("useBotData routes /api/orders/open through unwrapOpenOrders", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const src = fs.readFileSync(path.join(here, "useBotData.ts"), "utf8");
  assert.ok(/getJson<unknown>\("\/api\/orders\/open"\)\.then\(unwrapOpenOrders<OpenOrder>\)/.test(src));
});
