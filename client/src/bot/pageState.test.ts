import { test } from "node:test";
import assert from "node:assert/strict";
import { derivePageState, parseScanSummary, parseScanInterval, type AuditRow } from "./pageState";

// The real 15 Oct 2026 audit rows (research/bot_cockpit_audit/live_snapshot).
const R = (time: string, type: string, message: string): AuditRow => ({ time, type, message });
const base = (killSwitch = false) => ({ killSwitch, active: true, marketOpen: true, modelAgeHours: 9.7, fallbackAtHours: 24 });

test("parsers read the real TIER2 wording", () => {
  assert.deepEqual(parseScanSummary("Scanned 12138 stocks, 2 trade candidates (via daemon)"), { scanned: 12138, candidates: 2 });
  assert.deepEqual(parseScanSummary("nope"), { scanned: null, candidates: null });
  assert.equal(parseScanInterval("Starting scan (interval: 5min based on market time)"), 5);
});

test("scanning: a Starting scan newer than the last Scanned", () => {
  const audit = [R("2026-10-15T12:23:53Z", "TIER2", "Starting scan (interval: 5min based on market time)"), R("2026-10-15T12:18:53Z", "TIER2", "Scanned 12138 stocks, 2 trade candidates (via daemon)")];
  const d = derivePageState({ ...base(), audit, nowMs: Date.parse("2026-10-15T12:24:15Z") });
  assert.equal(d.state, "scanning");
  assert.equal(d.scan.elapsedMs, 22_000);
  assert.equal(d.scan.nextAtMs, Date.parse("2026-10-15T12:28:53Z"));
});

test("candidate: the latest scan found candidates / a QUEUE row", () => {
  const audit = [R("2026-10-15T13:28:55Z", "QUEUE", "KWEB queued for market open (score 75)"), R("2026-10-15T13:28:55Z", "TIER2", "Scanned 12138 stocks, 2 trade candidates (via daemon)"), R("2026-10-15T13:28:07Z", "TIER2", "Starting scan (interval: 5min based on market time)")];
  const d = derivePageState({ ...base(), audit, nowMs: Date.parse("2026-10-15T13:29:30Z") });
  assert.equal(d.state, "candidate");
  assert.equal(d.scan.candidates, 2);
});

test("incident: the latched kill switch wins over everything, with the DRAWDOWN-KILL row", () => {
  const audit = [R("2026-10-15T13:30:12Z", "DRAWDOWN-KILL", "MAX DRAWDOWN KILL SWITCH: Equity $99008 is -10.6% below peak $110727"), R("2026-10-15T13:28:55Z", "QUEUE", "KWEB queued for market open (score 75)")];
  const d = derivePageState({ ...base(true), audit, nowMs: Date.parse("2026-10-15T17:58:00Z") });
  assert.equal(d.state, "incident");
  assert.equal(d.incident?.kind, "kill");
  assert.equal(d.since, "2026-10-15T13:30:12Z");
});

test("incident: a model older than the fallback line, even with nothing in the log", () => {
  const d = derivePageState({ ...base(), modelAgeHours: 25.5, audit: [], nowMs: Date.now() });
  assert.equal(d.state, "incident");
  assert.equal(d.incident?.kind, "model");
});

test("closed beats idle; idle when the last scan is old and found nothing", () => {
  const audit = [R("2026-10-15T12:18:53Z", "TIER2", "Scanned 12138 stocks, 0 trade candidates (via daemon)"), R("2026-10-15T12:18:18Z", "TIER2", "Starting scan (interval: 5min based on market time)")];
  assert.equal(derivePageState({ ...base(), marketOpen: false, audit, nowMs: Date.parse("2026-10-15T21:00:00Z") }).state, "closed");
  assert.equal(derivePageState({ ...base(), audit, nowMs: Date.parse("2026-10-15T12:19:30Z") }).state, "idle");
});

test("placed and exit windows", () => {
  const trade = R("2026-10-14T14:14:31Z", "TRADE", "LIMIT BUY 12 QQQ @ ~$736 | Score: 75 | Rules: 75 ML: 60 Blend: 71");
  assert.equal(derivePageState({ ...base(), audit: [trade], nowMs: Date.parse("2026-10-14T14:20:00Z") }).state, "placed");
  const ex = R("2026-10-14T14:30:00Z", "WS-EXIT", "QQQ stop_loss");
  assert.equal(derivePageState({ ...base(), audit: [trade, ex], nowMs: Date.parse("2026-10-14T14:31:00Z") }).state, "exit");
  assert.equal(derivePageState({ ...base(), audit: [trade, ex], nowMs: Date.parse("2026-10-14T15:31:00Z") }).state, "idle");
});
