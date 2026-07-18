// gpCache policy — the politeness core (fresh cache = zero network) and the
// honest stale note. IDB itself is browser-only; the POLICY is pure.
import { test } from "node:test";
import assert from "node:assert/strict";
import { catalogPlan, staleCatalogNote } from "./gpCache";

test("catalogPlan: fresh cache is authoritative (zero network inside TTL)", () => {
  assert.equal(catalogPlan(10_000_000, 9_000_000, 6 * 3_600_000), "use-cached");
  assert.equal(catalogPlan(10_000_000, null, 6 * 3_600_000), "refetch");
  assert.equal(catalogPlan(10_000_000, 10_000_000 - 7 * 3_600_000, 6 * 3_600_000), "refetch");
});

test("staleCatalogNote: age stated plainly, uncertainty claim matches the card's element-age honesty", () => {
  const h5 = staleCatalogNote(5.4 * 3_600_000);
  assert.ok(/5\.4h old/.test(h5), h5);
  assert.ok(/CelesTrak unreachable/.test(h5));
  assert.ok(/uncertainty grows with element age/.test(h5));
  assert.ok(/3\.5d old/.test(staleCatalogNote(3.5 * 24 * 3_600_000)));
});
