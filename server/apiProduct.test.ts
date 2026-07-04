// API product foundation — pure-module tests (no express, no db, no network).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  parseApiKeys, makeRateLimiter, keyId, meterUsage, apiMeta,
  LICENSE_MARKS, TIER_LIMITS,
} from "./apiProduct";

const here = path.dirname(fileURLToPath(import.meta.url));

test("keys are env-seeded only, tiered, and trivially short keys are refused", () => {
  const m = parseApiKeys({ API_PRODUCT_KEYS: "supersecretkey123:alice:pro, k2:short:dev, anotherlongkey456:bob:nonsense" } as any);
  assert.equal(m.size, 2, "short key refused");
  assert.deepEqual(m.get("supersecretkey123"), { label: "alice", tier: "pro" });
  assert.equal(m.get("anotherlongkey456")!.tier, "dev", "unknown tier falls back to dev");
  assert.equal(parseApiKeys({} as any).size, 0, "no env -> no keys -> the API is fully closed");
});

test("rate limiter: per-minute window enforces tier limits and recovers", () => {
  const rl = makeRateLimiter();
  const t0 = 1_000_000_000_000;
  for (let i = 0; i < TIER_LIMITS.dev.perMinute; i++) {
    assert.ok(rl.allow("k", "dev", t0 + i * 10).ok);
  }
  const denied = rl.allow("k", "dev", t0 + 59_000);
  assert.equal(denied.ok, false);
  assert.equal(denied.retryAfterSec, 60);
  assert.ok(rl.allow("k", "dev", t0 + 61_000 + TIER_LIMITS.dev.perMinute * 10).ok, "window slides");
});

test("metering: raw keys NEVER reach the usage archive — only sha256 prefixes", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "vt-api-"));
  const NOW = Date.UTC(2026, 6, 4, 12, 0, 0);
  meterUsage({ key: "supersecretkey123", endpoint: "/api/v1/stats/archive", status: 200, tier: "dev" }, base, NOW);
  const day = fs.readFileSync(path.join(base, "apiusage", "2026-07-04.jsonl"), "utf8");
  assert.ok(!day.includes("supersecretkey123"), "raw key leaked into the archive");
  const rec = JSON.parse(day.trim());
  assert.equal(rec.k, keyId("supersecretkey123"));
  assert.equal(rec.k.length, 12);
  assert.equal(rec.e, "/api/v1/stats/archive");
});

test("license marks: aircraft-derived endpoints carry ODbL share-alike; OWM is absent from the API entirely", () => {
  assert.equal(LICENSE_MARKS["tracks/aircraft"].resell, "share-alike");
  assert.ok(LICENSE_MARKS["tracks/aircraft"].license.includes("ODbL"));
  assert.equal(LICENSE_MARKS["tracks/vessels"].resell, "conditional");
  assert.equal(LICENSE_MARKS["stats/portdwell"].resell, "conditional", "AIS-derived stats inherit conditionality");
  const meta = apiMeta();
  assert.ok(!JSON.stringify(meta).toLowerCase().includes("openweathermap"),
    "OWM tiles are a display product — they may not appear on the data API");
});

test("meta honesty: gated products listed as coming, never as live endpoints", () => {
  const meta = apiMeta();
  const paths = meta.endpoints.map((e: any) => e.path).join(" ");
  assert.ok(!paths.includes("tank"), "tank-fill must not be a live endpoint before gate 2");
  assert.ok(!paths.includes("timeline"), "entity timelines must not be live before Graph v1");
  assert.ok(meta.coming_gated.length >= 2);
  assert.ok(meta.disclaimer.includes("safety-of-life"));
});

test("wiring pinned: /api/v1 routes registered behind requireApiKey; meta is the only public one", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  for (const p of ["/api/v1/meta", "/api/v1/tracks/:kind/:id", "/api/v1/stats/portdwell", "/api/v1/stats/shadow", "/api/v1/stats/archive"]) {
    assert.ok(routes.includes(`"${p}"`), `route ${p} missing`);
  }
  const v1Block = routes.slice(routes.indexOf("/api/v1 — the DATA PRODUCT"));
  const guarded = (v1Block.match(/requireApiKey\(req, res\)/g) || []).length;
  assert.ok(guarded >= 4, `expected >=4 key-guarded endpoints, found ${guarded}`);
  assert.ok(routes.includes("meterUsage"), "metering must be wired");
});
