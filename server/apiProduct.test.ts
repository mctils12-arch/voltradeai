// API product foundation — pure-module tests (no express, no db, no network).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  parseApiKeys, makeRateLimiter, keyId, meterUsage, apiMeta, agentToolSpec,
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

test("meta honesty: gated products listed as coming, never as live endpoints; Graph v1 now IS live", () => {
  const meta = apiMeta();
  const paths = meta.endpoints.map((e: any) => e.path).join(" ");
  assert.ok(!paths.includes("tank"), "tank-fill must not be a live endpoint before gate 2");
  assert.ok(paths.includes("/api/v1/graph"), "Everything Graph v1 shipped — its keyed mirror must be a live endpoint");
  assert.ok(paths.includes("/api/v1/stats/plant-operations"), "EPA CAMD keyed mirror shipped — must be a live endpoint");
  assert.ok(paths.includes("/api/v1/stats/secftd"), "SEC FTD keyed mirror shipped — must be a live endpoint");
  assert.ok(meta.coming_gated.length >= 1, "tank-fill remains the one still-gated product");
  assert.ok(!meta.coming_gated.join(" ").includes("Everything Graph"), "graph must not be listed as coming once live");
  assert.ok(meta.disclaimer.includes("safety-of-life"));
});

test("wiring pinned: /api/v1 routes registered behind requireApiKey; meta is the only public one", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  for (const p of ["/api/v1/meta", "/api/v1/tracks/:kind/:id", "/api/v1/stats/portdwell", "/api/v1/stats/shadow", "/api/v1/stats/archive", "/api/v1/graph", "/api/v1/stats/plant-operations", "/api/v1/stats/secftd"]) {
    assert.ok(routes.includes(`"${p}"`), `route ${p} missing`);
  }
  const v1Block = routes.slice(routes.indexOf("/api/v1 — the DATA PRODUCT"));
  const guarded = (v1Block.match(/requireApiKey\(req, res\)/g) || []).length;
  assert.ok(guarded >= 7, `expected >=7 key-guarded endpoints, found ${guarded}`);
  assert.ok(routes.includes("meterUsage"), "metering must be wired");
});

test("graph license mark: conditional resell, inherits AIS conditionality like the port stats", () => {
  assert.equal(LICENSE_MARKS["graph"].resell, "conditional");
  assert.ok(LICENSE_MARKS["graph"].license.includes("aisstream"));
});

test("plant-operations license mark: public-domain US-gov data resells freely, unlike the AIS-derived stats", () => {
  assert.equal(LICENSE_MARKS["stats/plant-operations"].resell, "ok");
  assert.ok(LICENSE_MARKS["stats/plant-operations"].license.includes("public domain"));
  assert.ok(LICENSE_MARKS["stats/plant-operations"].license.includes("EPA"));
});

test("secftd license mark: public-domain US-gov data resells freely; agent tool documents it", () => {
  assert.equal(LICENSE_MARKS["stats/secftd"].resell, "ok");
  assert.ok(LICENSE_MARKS["stats/secftd"].license.includes("public domain"));
  assert.ok(LICENSE_MARKS["stats/secftd"].license.includes("SEC"));
  const spec = agentToolSpec();
  const tool = spec.tools.find((t: any) => t.name === "voltrade_secftd_stats");
  assert.ok(tool, "voltrade_secftd_stats tool must exist");
  assert.deepEqual(tool.returns_provenance, ["stats/secftd"]);
});

test("every v1 endpoint documents a preview (or states it needs a live id), so /developers can't silently drift", () => {
  const meta = apiMeta();
  for (const e of meta.endpoints as any[]) {
    if (e.path === "/api/v1/tracks/:kind/:id") continue; // needs a real id, no static preview possible
    assert.ok(typeof e.preview === "string" && e.preview.length > 0, `${e.path} missing a preview route for the docs explorer`);
  }
});

test("honesty: every v1 response and its public preview mirror carry generated_at, so freshness is never silently omitted", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(/const v1Envelope = .*generated_at: new Date\(generatedAt/s.test(routes), "v1Envelope must stamp generated_at");
  for (const call of ['v1Envelope("stats/portdwell", dwellCache.data, dwellCache.at)',
                       'v1Envelope("stats/shadow", shadowCache.data, shadowCache.at)']) {
    assert.ok(routes.includes(call), `expected cache-timestamped envelope call: ${call}`);
  }
  for (const previewSnippet of ['res.json({ ...shadowCache.data, generated_at:',
                                 'res.json({ ...dwellCache.data, generated_at:',
                                 'res.json({ ...archiveStats(), generated_at:']) {
    assert.ok(routes.includes(previewSnippet), `public preview route missing generated_at: ${previewSnippet}`);
  }
});

test("agent tool spec: one tool per LIVE endpoint (meta excluded), gated signals never leak in", () => {
  const spec = agentToolSpec();
  // meta is docs-about-docs and is not a tool; every other live endpoint is.
  const liveDataEndpoints = apiMeta().endpoints.filter((e: any) => e.path !== "/api/v1/meta");
  assert.equal(spec.tools.length, liveDataEndpoints.length, "tool count must track the live data endpoints");
  // Gated signals may be NAMED in excluded_gated (honest roadmap) but must
  // never appear as a callable tool — scan the tools array specifically.
  const toolBlob = JSON.stringify(spec.tools).toLowerCase();
  assert.ok(!toolBlob.includes("tank"), "gated tank-fill signal must never appear as a tool");
  assert.ok(!toolBlob.includes("timeline"), "gated entity timelines must never appear as a tool");
  assert.ok(!toolBlob.includes("openweathermap"), "OWM is a display product, not on the data API");
  // The gated roadmap is surfaced honestly as excluded, not silently dropped.
  assert.deepEqual(spec.excluded_gated, apiMeta().coming_gated);
});

test("agent tool spec: every tool is valid JSON-Schema and its provenance keys resolve to real license marks", () => {
  const spec = agentToolSpec();
  for (const tool of spec.tools as any[]) {
    assert.ok(/^voltrade_[a-z_]+$/.test(tool.name), `tool name not agent-safe: ${tool.name}`);
    assert.equal(tool.input_schema.type, "object", `${tool.name} input_schema must be an object`);
    assert.ok(tool.input_schema.properties && typeof tool.input_schema.properties === "object");
    assert.ok(Array.isArray(tool.returns_provenance) && tool.returns_provenance.length > 0,
      `${tool.name} must declare what it returns`);
    for (const mark of tool.returns_provenance) {
      assert.ok(LICENSE_MARKS[mark], `${tool.name} references unknown license mark ${mark}`);
    }
  }
  // Provenance travels WITH the spec so an agent can cite it, not just the number.
  assert.deepEqual(spec.license_marks, LICENSE_MARKS);
  assert.ok(spec.ground_truth_note.toLowerCase().includes("provenance"));
  assert.ok(apiMeta().agent_tools === "/api/v1/agent-tools", "meta must point agents at the tool spec");
});

test("agent tool spec: /api/v1/agent-tools is wired and public (docs, not data — like /meta)", () => {
  const routes = fs.readFileSync(path.join(here, "routes.ts"), "utf8");
  assert.ok(routes.includes('"/api/v1/agent-tools"'), "agent-tools route missing");
  assert.ok(routes.includes("agentToolSpec()"), "agent-tools must serve agentToolSpec()");
  // It sits with the public meta doc, NOT behind requireApiKey (it's a spec, not data).
  const specLine = routes.split("\n").find((l) => l.includes('"/api/v1/agent-tools"')) || "";
  assert.ok(!specLine.includes("requireApiKey"), "the spec itself is public documentation");
});
