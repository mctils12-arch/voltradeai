// Hermetic tests for the /api/v1/meta shape guard. Fixtures only.
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseApiMetaDoc, tierLimit } from "./apiMetaGuard.ts";

// Mirrors the REAL apiMeta() shape (server/apiProduct.ts, verified 2026-07-12).
const GOOD_META = {
  version: "v1",
  auth: "x-api-key header (or ?api_key=).",
  endpoints: [
    { path: "/api/v1/meta", params: "", desc: "this document" },
    { path: "/api/v1/stats/portdwell", params: "port, days", desc: "median dwell", preview: "/api/data/portdwell" },
  ],
  coming_gated: ["/api/v1/signals/tankfill"],
  limits: {
    dev: { perMinute: 60, perDay: 10_000 },
    pro: { perMinute: 600, perDay: 100_000 },
    enterprise: { perMinute: 3_000, perDay: 1_000_000 },
  },
  license_marks: { aircraft: { license: "ODbL", attribution: "adsb.lol", resell: "share-alike" } },
  disclaimer: "data as-is",
};

test("parseApiMetaDoc: accepts the real meta shape", () => {
  const doc = parseApiMetaDoc(true, GOOD_META);
  assert.ok(doc);
  assert.equal(doc!.endpoints.length, 2);
  assert.equal(doc!.limits.dev.perMinute, 60);
});

test("parseApiMetaDoc: a non-ok response is rejected even with a JSON body", () => {
  // Regression: our routes return JSON {error} bodies on 4xx/5xx; parsing one
  // as meta crashed /developers at meta.limits[tier].perMinute.
  assert.equal(parseApiMetaDoc(false, GOOD_META), null);
  assert.equal(parseApiMetaDoc(false, { error: "internal error" }), null);
});

test("parseApiMetaDoc: JSON error body / wrong shapes -> null, never a throw", () => {
  assert.equal(parseApiMetaDoc(true, { error: "invalid or missing API key" }), null);
  assert.equal(parseApiMetaDoc(true, null), null);
  assert.equal(parseApiMetaDoc(true, "string body"), null);
  assert.equal(parseApiMetaDoc(true, []), null);
  assert.equal(parseApiMetaDoc(true, {}), null);
});

test("parseApiMetaDoc: version-skewed meta without limits (pre-P2 server) -> null", () => {
  const { limits: _drop, ...older } = GOOD_META;
  assert.equal(parseApiMetaDoc(true, older), null);
});

test("parseApiMetaDoc: limits present but no valid tier entry -> null", () => {
  assert.equal(parseApiMetaDoc(true, { ...GOOD_META, limits: { dev: { perMinute: "60", perDay: null } } }), null);
});

test("tierLimit: returns the tier's limits, null for a missing/renamed tier", () => {
  const doc = parseApiMetaDoc(true, GOOD_META)!;
  assert.deepEqual(tierLimit(doc, "pro"), { perMinute: 600, perDay: 100_000 });
  assert.equal(tierLimit(doc, "hobby"), null); // future rename must not crash the page
});
