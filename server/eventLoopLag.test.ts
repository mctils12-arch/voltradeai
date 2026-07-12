// See server/eventLoopLag.ts header for the full KNOWN BROKEN #18
// correlation this monitor tests (TIER2-ERROR daemon timeouts landing
// within tens of ms of STREAM-DISCONNECT entries on 2026-07-12).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  computeLagMs,
  lagExceedsThreshold,
  EVENTLOOP_LAG_CHECK_MS,
  EVENTLOOP_LAG_ALERT_THRESHOLD_MS,
} from "./eventLoopLag";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

test("computeLagMs: a tick firing exactly on schedule has zero lag", () => {
  assert.equal(computeLagMs(2000, 2000), 0);
});

test("computeLagMs: a tick firing late reports the overshoot as lag", () => {
  assert.equal(computeLagMs(2000, 2750), 750);
});

test("computeLagMs: a tick firing early (clock skew / test double) clamps to zero, never negative", () => {
  assert.equal(computeLagMs(2000, 1500), 0);
});

test("lagExceedsThreshold: below the default threshold does not alert", () => {
  assert.equal(lagExceedsThreshold(EVENTLOOP_LAG_ALERT_THRESHOLD_MS - 1), false);
});

test("lagExceedsThreshold: at or above the default threshold alerts", () => {
  assert.equal(lagExceedsThreshold(EVENTLOOP_LAG_ALERT_THRESHOLD_MS), true);
  assert.equal(lagExceedsThreshold(EVENTLOOP_LAG_ALERT_THRESHOLD_MS + 1000), true);
});

test("lagExceedsThreshold: a custom threshold overrides the default", () => {
  assert.equal(lagExceedsThreshold(100, 200), false);
  assert.equal(lagExceedsThreshold(300, 200), true);
});

test("wiring pinned: bot.ts imports the event-loop lag helpers from eventLoopLag.ts", () => {
  assert.ok(
    /import\s*\{[^}]*computeLagMs[^}]*\}\s*from\s*"\.\/eventLoopLag"/.test(bot),
    "bot.ts must import computeLagMs (and siblings) from ./eventLoopLag rather than reimplementing the calculation inline",
  );
});

test("wiring pinned: bot.ts arms a setInterval at EVENTLOOP_LAG_CHECK_MS that audits EVENTLOOP-LAG when threshold is exceeded", () => {
  const monitorIdx = bot.indexOf("EVENT-LOOP LAG MONITOR");
  assert.ok(monitorIdx > 0, "event-loop lag monitor block not found in bot.ts");
  const block = bot.slice(monitorIdx, monitorIdx + 1200);
  assert.ok(block.includes("setInterval("), "monitor must be wired via setInterval");
  assert.ok(block.includes("EVENTLOOP_LAG_CHECK_MS"), "monitor must tick at EVENTLOOP_LAG_CHECK_MS");
  assert.ok(block.includes("lagExceedsThreshold(lag)"), "monitor must gate the audit call on lagExceedsThreshold");
  assert.ok(block.includes('audit("EVENTLOOP-LAG"'), "monitor must log to the audit trail under a distinct EVENTLOOP-LAG type so it's filterable from existing TIER2-ERROR/STREAM-DISCONNECT entries");
});

test("wiring pinned: the monitor tracks its own last-tick timestamp rather than relying on setInterval's nominal delay", () => {
  const monitorIdx = bot.indexOf("EVENT-LOOP LAG MONITOR");
  const block = bot.slice(monitorIdx, monitorIdx + 1200);
  assert.ok(/eventLoopLagLastTick\s*=\s*Date\.now\(\)/.test(block), "must seed a last-tick timestamp — comparing against actual elapsed wall-clock time (not the nominal interval) is the entire point of the measurement");
});
