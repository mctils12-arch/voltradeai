import { test } from "node:test";
import assert from "node:assert/strict";
import { cadenceBoxes, deriveStages, deriveIssues, deriveDayReview, type DeriveInput } from "./derive";
import type { AuditRow } from "./pageState";

const R = (time: string, type: string, message: string): AuditRow => ({ time, type, message });
const NOW = Date.parse("2026-10-15T17:58:00Z");
const base: DeriveInput = {
  audit: [], nowMs: NOW, killSwitch: false, active: true, marketOpen: true, equity: 100172, equityPeak: 110727, maxDrawdownPct: 10,
  modelAgeHours: 9.7, fallbackAtHours: 24, constantFeatures: 11, featureCount: 34, openOrders: 0, totalTrades: 36, slotsAvailable: 0, modelTypes: { lightgbm_neutral: 120 },
};

test("cadence boxes: present ticks ok, the open bucket late, older gaps missing", () => {
  const iv = 60_000;
  const times = [NOW - 30_000, NOW - 90_000, NOW - 4 * iv - 10];
  assert.deepEqual(cadenceBoxes(times, NOW, iv, 5), ["ok", "miss", "miss", "ok", "ok"]);
  assert.deepEqual(cadenceBoxes([], NOW, iv, 3), ["miss", "miss", "late"]);
});

test("stages read the real rows: kill latched, constant features, sizing refusals", () => {
  const audit = [
    R("2026-10-15T17:57:40Z", "SIZE-BLOCK", "AMD [stocks]: Negative expectancy for strategy 'stocks' (WR=28%); refusing to size (score 74.3)"),
    R("2026-10-15T13:30:12Z", "DRAWDOWN-KILL", "MAX DRAWDOWN KILL SWITCH: Equity $99008 is -10.6% below peak $110727"),
    R("2026-10-15T13:28:55Z", "TIER2", "Scanned 12138 stocks, 2 trade candidates (via daemon)"),
    R("2026-10-15T13:28:07Z", "TIER2", "Starting scan (interval: 5min based on market time)"),
    R("2026-10-15T13:28:05Z", "STREAM", "Real-time feed live — 43 tickers"),
  ];
  const s = deriveStages({ ...base, audit, killSwitch: true });
  const by = Object.fromEntries(s.map((x) => [x.id, x]));
  assert.equal(by.risk.level, "bad");
  assert.match(by.risk.value, /latched/);
  assert.equal(by.features.level, "bad");
  assert.equal(by.features.value, "11 / 34 constant");
  assert.equal(by.sizing.level, "warn");
  assert.match(by.feeds.value, /^WS 43 · /);
  assert.equal(by.rules.level, "warn");
  assert.equal(by.scans.boxes.length, 10);
});

test("issues: ordered by level, the latch first with its resolution numbers; sizing refusals; thin sample", () => {
  const audit = [
    R("2026-10-15T16:14:37Z", "KILL-AUTO-RESUME-EVAL", "EVALUATED UNRECONCILED (0/3): drawdown -9.65% not above -8.00%; paper re-baseline after 13 market hours (2.8 so far)"),
    R("2026-10-15T13:30:12Z", "DRAWDOWN-KILL", "MAX DRAWDOWN KILL SWITCH: Equity $99008 is -10.6% below peak $110727"),
    R("2026-10-15T17:50:00Z", "SIZE-BLOCK", "AMD [stocks]: Negative expectancy (score 74.3)"),
  ];
  const iss = deriveIssues({ ...base, audit, killSwitch: true, totalTrades: 20 });
  assert.equal(iss[0].id, "kill");
  assert.match(iss[0].clears, /-9.65% not above -8.00%/);
  assert.equal(iss[0].action?.kind, "resume");
  assert.ok(iss.some((x) => x.id === "sizing" && /stocks/.test(x.condition)));
  assert.ok(iss.some((x) => x.id === "features"));
  assert.equal(iss[iss.length - 1].level, "L1");
});

test("issues: model past the fallback line is L3 even with an empty log; nothing wrong → empty", () => {
  assert.equal(deriveIssues({ ...base, modelAgeHours: 26 })[0].id, "model");
  assert.deepEqual(deriveIssues({ ...base, constantFeatures: 0, totalTrades: 40 }), []);
});

test("day review counts today's rows in ET", () => {
  const audit = [
    R("2026-10-15T13:30:12Z", "DRAWDOWN-KILL", "x"), R("2026-10-15T13:27:48Z", "SHUTDOWN", "All open orders cancelled"), R("2026-10-15T16:13:46Z", "SHUTDOWN", "All open orders cancelled"),
    R("2026-10-15T12:18:53Z", "TIER2", "Scanned 12138 stocks, 2 trade candidates (via daemon)"), R("2026-10-15T12:25:16Z", "TIER2", "Scanned 12138 stocks, 2 trade candidates (via daemon)"),
    R("2026-10-15T16:18:04Z", "EVENTLOOP-LAG", "Node event loop stalled ~5151ms past its 2000ms tick"), R("2026-10-15T17:13:51Z", "TIER3", "ML model fresh (9h old) — skipping retrain"),
    R("2026-10-14T20:00:00Z", "TIER2", "Scanned 1 stocks, 9 trade candidates (via daemon)"),
  ];
  const d = deriveDayReview(audit, NOW);
  assert.equal(d.scans, 2);
  assert.equal(d.candidatesMax, 2);
  assert.equal(d.deploys, 2);
  assert.equal(d.stallMaxMs, 5151);
  assert.equal(d.freshChecks, 1);
  assert.equal(d.killAt, "2026-10-15T13:30:12Z");
});
