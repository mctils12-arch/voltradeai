import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import {
  snapshotFromHealthChecks,
  recordHealthSnapshot,
  readHealthHistory,
  summarizePipelineHealth,
  summarizeWindow,
  resetHealthHistoryThrottleForTest,
} from "./pipelineHealthHistory";

const OK_CHECKS = {
  status: "ok",
  checks: {
    server: { heap_used_mb: 300, rss_mb: 900 },
    database: { status: "ok" },
    alpaca: { status: "ok" },
    python: { status: "ok" },
    bot: { status: "active", liveness: { dark: false } },
    scanner: { status: "ok", consecutiveFailures: 0 },
    licensing: { status: "ok" },
  },
};

const DEGRADED_CHECKS = {
  status: "degraded",
  checks: {
    server: { heap_used_mb: 500, rss_mb: 1200 },
    database: { status: "ok" },
    alpaca: { status: "error" },
    python: { status: "ok" },
    bot: { status: "active", liveness: { dark: true, marketHours: 3 } },
    scanner: { status: "degraded", consecutiveFailures: 12 },
    licensing: { status: "warning" },
  },
};

test("snapshotFromHealthChecks: flattens an ok /api/health payload", () => {
  const s = snapshotFromHealthChecks(OK_CHECKS, Date.parse("2026-07-31T12:00:00Z"));
  assert.equal(s.status, "ok");
  assert.equal(s.database_ok, true);
  assert.equal(s.alpaca_ok, true);
  assert.equal(s.scanner_ok, true);
  assert.equal(s.scanner_consecutive_failures, 0);
  assert.equal(s.bot_status, "active");
  assert.equal(s.bot_liveness_dark, false);
  assert.equal(s.t, Math.floor(Date.parse("2026-07-31T12:00:00Z") / 1000));
});

test("snapshotFromHealthChecks: flattens a degraded payload, every failing check readable", () => {
  const s = snapshotFromHealthChecks(DEGRADED_CHECKS);
  assert.equal(s.status, "degraded");
  assert.equal(s.database_ok, true);
  assert.equal(s.alpaca_ok, false);
  assert.equal(s.scanner_ok, false);
  assert.equal(s.scanner_consecutive_failures, 12);
  assert.equal(s.licensing_ok, false);
  assert.equal(s.bot_liveness_dark, true);
});

test("snapshotFromHealthChecks: missing/malformed sub-objects never throw, read as unhealthy-safe", () => {
  const s = snapshotFromHealthChecks({});
  assert.equal(s.status, "degraded");
  assert.equal(s.database_ok, false);
  assert.equal(s.bot_status, "unknown");
  assert.equal(s.scanner_consecutive_failures, 0);
});

test("recordHealthSnapshot: throttles to one write per window, writes again after it elapses", () => {
  resetHealthHistoryThrottleForTest();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "phh-"));
  const t0 = Date.parse("2026-07-31T00:00:00Z");
  assert.equal(recordHealthSnapshot(OK_CHECKS, base, t0, 300_000), true);
  assert.equal(recordHealthSnapshot(OK_CHECKS, base, t0 + 60_000, 300_000), false, "inside the throttle window");
  assert.equal(recordHealthSnapshot(OK_CHECKS, base, t0 + 300_000, 300_000), true, "window elapsed");
  const rows = readHealthHistory(1, base, t0 + 300_000);
  assert.equal(rows.length, 2);
});

test("recordHealthSnapshot: appends across calls, never overwrites the day file", () => {
  resetHealthHistoryThrottleForTest();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "phh-"));
  const t0 = Date.parse("2026-07-31T00:00:00Z");
  recordHealthSnapshot(OK_CHECKS, base, t0, 0);
  recordHealthSnapshot(DEGRADED_CHECKS, base, t0 + 1000, 0);
  recordHealthSnapshot(OK_CHECKS, base, t0 + 2000, 0);
  const rows = readHealthHistory(1, base, t0 + 2000);
  assert.equal(rows.length, 3);
  assert.deepEqual(rows.map((r) => r.status), ["ok", "degraded", "ok"]);
});

test("readHealthHistory: spans multiple day files and returns oldest-first", () => {
  resetHealthHistoryThrottleForTest();
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "phh-"));
  const day1 = Date.parse("2026-07-29T23:00:00Z");
  const day2 = Date.parse("2026-07-30T01:00:00Z");
  const day3 = Date.parse("2026-07-31T01:00:00Z");
  recordHealthSnapshot(OK_CHECKS, base, day1, 0);
  recordHealthSnapshot(OK_CHECKS, base, day2, 0);
  recordHealthSnapshot(DEGRADED_CHECKS, base, day3, 0);
  const rows = readHealthHistory(3, base, day3);
  assert.equal(rows.length, 3);
  assert.ok(rows[0].t < rows[1].t && rows[1].t < rows[2].t);
});

test("readHealthHistory: missing directory returns empty, never throws", () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), "phh-"));
  assert.deepEqual(readHealthHistory(7, base), []);
});

test("summarizePipelineHealth: honest null uptime with zero samples, never fabricated as 100%", () => {
  const s = summarizePipelineHealth([], 24);
  assert.equal(s.uptime_pct, null);
  assert.equal(s.sample_count, 0);
  assert.equal(s.current, null);
});

test("summarizePipelineHealth: uptime_pct + per-check degraded counts + current = last row", () => {
  const rows = [
    snapshotFromHealthChecks(OK_CHECKS, 1000),
    snapshotFromHealthChecks(OK_CHECKS, 2000),
    snapshotFromHealthChecks(DEGRADED_CHECKS, 3000),
    snapshotFromHealthChecks(OK_CHECKS, 4000),
  ];
  const s = summarizePipelineHealth(rows, 24);
  assert.equal(s.sample_count, 4);
  assert.equal(s.uptime_pct, 75);
  assert.equal(s.degraded_counts.alpaca, 1);
  assert.equal(s.degraded_counts.scanner, 1);
  assert.equal(s.degraded_counts.licensing, 1);
  assert.equal(s.degraded_counts.database, 0);
  assert.equal(s.current?.status, "ok");
  assert.equal(s.current?.t, rows[3].t);
});

test("summarizePipelineHealth: timeline is capped and evenly downsampled for large row counts", () => {
  const rows = Array.from({ length: 5000 }, (_, i) => snapshotFromHealthChecks(OK_CHECKS, i * 1000));
  const s = summarizePipelineHealth(rows, 168);
  assert.ok(s.timeline.length <= 200);
  assert.ok(s.timeline.length > 0);
});

test("summarizeWindow: filters to the trailing N hours before summarizing", () => {
  const now = Date.parse("2026-07-31T12:00:00Z");
  const rows = [
    snapshotFromHealthChecks(OK_CHECKS, now - 30 * 3600_000),       // 30h ago: outside a 24h window
    snapshotFromHealthChecks(DEGRADED_CHECKS, now - 2 * 3600_000),  // 2h ago: inside
    snapshotFromHealthChecks(OK_CHECKS, now),
  ];
  const s24 = summarizeWindow(rows, 24, now);
  assert.equal(s24.sample_count, 2, "the 30h-old row falls outside a 24h window");
  const s48 = summarizeWindow(rows, 48, now);
  assert.equal(s48.sample_count, 3);
});
