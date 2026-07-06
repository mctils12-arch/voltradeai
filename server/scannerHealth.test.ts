// Tier2 scan-failure visibility (REPAIR 2026-07-06) — the regression test
// that would have caught the production blind spot: Tier2 scanned and
// errored every cycle for ~1.5h (10+ consecutive TIER2-BACKOFF failures,
// all "Could not fetch market data from Alpaca") while /api/health stayed
// "ok" the whole time, and the audit log never said WHY the fetch failed.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { scannerDegraded, SCANNER_DEGRADED_FAILURE_THRESHOLD } from "./scannerHealth";

const here = path.dirname(fileURLToPath(import.meta.url));

test("scannerDegraded: healthy below threshold, degraded at and above it", () => {
  assert.equal(scannerDegraded(0), false);
  assert.equal(scannerDegraded(SCANNER_DEGRADED_FAILURE_THRESHOLD - 1), false);
  assert.equal(scannerDegraded(SCANNER_DEGRADED_FAILURE_THRESHOLD), true);
  assert.equal(scannerDegraded(SCANNER_DEGRADED_FAILURE_THRESHOLD + 12), true, "stays degraded, not just at the boundary");
});

test("threshold matches the scheduler's own backoff cap (bot.ts scheduleTier2)", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  // Backoff caps at n=5 (Math.min(tier2ConsecutiveFailures, 5)); the
  // degrade threshold must sit AT OR PAST that cap so it only fires once
  // the retry interval has stopped shrinking and a failure is persistent,
  // not a transient blip still within its first few retries.
  assert.ok(bot.includes("Math.min(tier2ConsecutiveFailures, 5)"), "backoff cap constant moved — re-check the threshold");
  assert.ok(SCANNER_DEGRADED_FAILURE_THRESHOLD >= 5);
});

test("wiring pinned: /api/health consumes scannerDegraded and degrades status", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes('import { scannerDegraded } from "./scannerHealth"'), "bot.ts must import the pure check");
  assert.ok(bot.includes("checks.checks.scanner"), "health payload must expose a scanner check");
  assert.ok(
    /if \(scannerDegraded\(tier2ConsecutiveFailures\)\) checks\.status = "degraded"/.test(bot),
    "a degraded scanner must degrade overall /api/health status",
  );
});

test("wiring pinned: Tier2 failure detail is captured and surfaced via the token-gated diag probe, not swallowed", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes("tier2LastFailureDetail"), "bot.ts must track the last failure's reason");
  assert.ok(bot.includes("result?.debug_detail"), "the Python scan error's debug_detail must be read");
  const scannerProbeStart = bot.indexOf('case "scanner":');
  const scannerProbeEnd = bot.indexOf("default:", scannerProbeStart);
  assert.ok(scannerProbeStart > 0 && scannerProbeEnd > scannerProbeStart, "scanner diag probe not found");
  const scannerProbe = bot.slice(scannerProbeStart, scannerProbeEnd);
  assert.ok(scannerProbe.includes("lastFailureDetail: tier2LastFailureDetail"), "the token-gated scanner probe must expose the failure detail");
});

test("public /api/health never carries the free-text failure detail — only status + a bare count", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  const healthStart = bot.indexOf('app.get("/api/health"');
  const scannerCheckStart = bot.indexOf("checks.checks.scanner", healthStart);
  const scannerCheckEnd = bot.indexOf("};", scannerCheckStart);
  assert.ok(healthStart > 0 && scannerCheckStart > healthStart && scannerCheckEnd > scannerCheckStart, "scanner check not found in /api/health");
  const scannerCheck = bot.slice(scannerCheckStart, scannerCheckEnd);
  assert.ok(!scannerCheck.includes("tier2LastFailureDetail"), "free-text failure detail must stay off the unauthenticated /api/health endpoint");
});

test("wiring pinned: bot_engine.py's scan error carries debug_detail (root cause, not just a generic message)", () => {
  const engine = fs.readFileSync(path.join(here, "..", "bot_engine.py"), "utf8");
  assert.ok(engine.includes('"debug_detail": _snap_diag["detail"]'), "the empty-scan error path must attach the diagnosed reason");
  assert.ok(engine.includes("def _parse_snapshot_batch"), "the pure, unit-testable parse+diagnose function must exist");
});
