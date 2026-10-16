// PRODUCT 2026-10-16 (bot page v2): /api/bot/insight is owner-gated, runs
// ml_insight.py, caches the heavy band report, and never leaks a stack trace.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function routeBlock(): string {
  const start = bot.indexOf('app.get("/api/bot/insight"');
  assert.ok(start > 0, "insight route not found");
  const end = bot.indexOf('app.get("/api/bot/ml-status"', start);
  assert.ok(end > start);
  return bot.slice(start, end);
}

test("owner-gated and backed by ml_insight.py", () => {
  const b = routeBlock();
  assert.ok(b.includes('requireOwner'), "must be owner-gated");
  assert.ok(b.includes('from ml_insight import ml_insight'), "must call the read-only insight module");
});

test("the heavy band report is cached and skippable", () => {
  const b = routeBlock();
  assert.ok(bot.includes("INSIGHT_CACHE_MS"), "cache constant declared");
  assert.ok(b.includes('req.query.bands'), "?bands=0 must skip the report");
  assert.ok(b.includes("insightCache = { at: Date.now()"), "result must be cached");
});

test("failure returns 503 with a bounded message and the stale body, never a 200 that looks healthy", () => {
  const b = routeBlock();
  assert.ok(b.includes("res.status(503)"), "errors must not answer 200");
  assert.ok(b.includes("msg.slice(0, 300)"), "error text bounded");
  assert.ok(b.includes("stale: insightCache?.body ?? null"), "last-known body returned with the error");
});
