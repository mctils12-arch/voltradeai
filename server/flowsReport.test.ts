// The institutional flow model's report (2026-11-06; flows_model.py writes the
// file, server/flowsReport.ts shapes it). Pins the GET /api/bot/flows contract
// the UI reads, the FLOWS audit row, and the wiring (owner route, diag probe,
// Tier 3 refresh in the mutex lane). Run: npx tsx --test server/flowsReport.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { flowsBody, readFlows, flowsAuditMessage, flowsAgeDays, FLOWS_FILE, FLOWS_REFRESH_CMD, FLOWS_REFRESH_TIMEOUT_MS, FLOWS_STALE_DAYS } from "./flowsReport";
import { planPythonExec } from "./pythonExec";
import { DIAG_PROBES } from "./diag";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const daemon = fs.readFileSync(path.join(here, "..", "voltrade_daemon.py"), "utf8");
const model = fs.readFileSync(path.join(here, "..", "flows_model.py"), "utf8");

const DOC = {
  ok: true, label: "ESTIMATE", asOf: "2026-11-20", computedAt: "2026-11-20T22:00:00Z", horizonDays: 5,
  equity: {
    pressure: -0.31, direction: "sell", confidence: 0.22,
    drivers: [
      { name: "Pension/balanced-fund rebalancing", key: "rebalance", value: -0.51, weight: 0.45, reason: "a 60/40 fund is 1.9 points overweight stocks" },
      { name: "Month-end cash needs", key: "month_end", value: -0.2, weight: 0.2, reason: "3 sessions in the cash raise" },
      { name: "S&P 500 quarterly rebalance", key: "index_event", value: 0, weight: 0, reason: "no direction" },
    ],
    sessions: [{ date: "2026-11-23", pressure: -0.4, phase: "cash_sell" }],
  },
  sectors: [
    { sector: "Energy", etf: "XLE", pressure: -0.4, direction: "sell", confidence: 0.2, ytd: -0.08, drivers: [{ name: "Tax-loss selling", key: "tax_loss", value: -1, weight: 0.3, reason: "XLE -8.0% on the year" }] },
    { sector: "Technology", etf: "XLK", pressure: 0.02, direction: "neutral", drivers: [] },
    { sector: "Financials", etf: "XLF", pressure: 0.3, direction: "buy", drivers: [] },
  ],
  calendar: [{ date: "2026-11-19", end: "2026-11-25", event: "Month-end cash raise", effect: "Pensions sell", kind: "cash_sell", source: "Etula et al. 2020" }],
  notes: ["ESTIMATE, not a forecast of record."],
  floor: { multiplier: 0.977, applied: true, reason: "Expected net equity selling" },
};

test("the body has the contract shape the UI reads", () => {
  const b = flowsBody(DOC, new Date("2026-11-21T15:00:00Z"));
  assert.deepEqual(Object.keys(b).sort(), ["ageDays", "asOf", "calendar", "computedAt", "equity", "floor", "horizonDays", "label", "notes", "sectors", "sessions", "stale"]);
  assert.equal(b.asOf, "2026-11-20");
  assert.equal(b.horizonDays, 5);
  assert.deepEqual(Object.keys(b.equity).sort(), ["confidence", "direction", "drivers", "pressure"]);
  assert.equal(b.equity.direction, "sell");
  assert.deepEqual(b.equity.drivers[0], { name: "Pension/balanced-fund rebalancing", value: -0.51, reason: "a 60/40 fund is 1.9 points overweight stocks", key: "rebalance", weight: 0.45 });
  for (const s of b.sectors) {
    for (const k of ["sector", "etf", "pressure", "direction", "drivers"]) assert.ok(k in s, k);
    assert.ok(["buy", "sell", "neutral"].includes(s.direction));
  }
  assert.deepEqual(b.calendar[0], { date: "2026-11-19", event: "Month-end cash raise", effect: "Pensions sell", end: "2026-11-25", kind: "cash_sell", source: "Etula et al. 2020" });
  assert.ok(Array.isArray(b.notes) && b.notes.every((n) => typeof n === "string"));
  assert.equal(b.label, "ESTIMATE");
  assert.equal(b.stale, false);
  assert.deepEqual(b.floor, { multiplier: 0.977, applied: true, reason: "Expected net equity selling" });
});

test("bad numbers are clamped and directions normalized, never passed through", () => {
  const b = flowsBody({ ...DOC, equity: { ...DOC.equity, pressure: 7, direction: "dump", confidence: 3, drivers: [{ name: "x", value: "nan", reason: 5 }] },
    floor: { multiplier: 1.4, applied: true, reason: "?" } }, new Date("2026-11-21T00:00:00Z"));
  assert.equal(b.equity.pressure, 1);
  assert.equal(b.equity.direction, "neutral");
  assert.equal(b.equity.confidence, 1);
  assert.deepEqual(b.equity.drivers[0], { name: "x", value: 0, reason: "5" });
  assert.equal(b.floor?.multiplier, 1, "the served floor multiplier is never above 1");
});

test("no file, a failed estimate and a stale one each say so", () => {
  for (const doc of [null, { ok: false, error: "need SPY daily closes, have 3" }]) {
    const b = flowsBody(doc as Record<string, unknown> | null, new Date());
    assert.equal(b.asOf, null);
    assert.equal(b.stale, true);
    assert.equal(b.equity.pressure, 0);
    assert.equal(b.equity.direction, "neutral");
    assert.deepEqual(b.sectors, []);
    assert.match(b.notes[0], /^No flows estimate: /);
  }
  const old = flowsBody(DOC, new Date("2026-11-30T12:00:00Z"));
  assert.equal(old.ageDays, 10);
  assert.ok(old.ageDays! > FLOWS_STALE_DAYS && old.stale);
  assert.match(old.notes[0], /^Stale: this estimate is 10 days old/);
  assert.equal(flowsAgeDays("garbage", new Date()), null);
});

test("readFlows reads the file and tolerates a missing or torn one", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "flows-"));
  assert.equal(readFlows(dir), null);
  fs.writeFileSync(path.join(dir, FLOWS_FILE), JSON.stringify(DOC));
  assert.equal(readFlows(dir)?.asOf, "2026-11-20");
  fs.writeFileSync(path.join(dir, FLOWS_FILE), "{\"ok\": tr");
  assert.equal(readFlows(dir), null);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("one FLOWS audit row per new estimate, one FLOWS-ERROR per failure, nothing when cached", () => {
  const body = flowsBody(DOC, new Date("2026-11-21T00:00:00Z"));
  const row = flowsAuditMessage({ status: "computed", asOf: "2026-11-20" }, body);
  assert.equal(row?.type, "FLOWS");
  assert.match(row!.message, /^2026-11-20 ESTIMATE: equities -0\.31 \(sell, confidence 0\.22\); floor x0\.98/);
  assert.match(row!.message, /main driver: a 60\/40 fund is 1\.9 points overweight stocks/);
  assert.match(row!.message, /selling: XLE -0\.40/);
  assert.match(row!.message, /buying: XLF \+0\.30/);
  assert.deepEqual(flowsAuditMessage({ status: "error", reason: "ConnectionError: down" }, body), { type: "FLOWS-ERROR", message: "flows estimate failed: ConnectionError: down" });
  assert.equal(flowsAuditMessage({ status: "cached", asOf: "2026-11-20" }, body), null);
  assert.equal(flowsAuditMessage({ status: "cached_recent" }, body), null);
  assert.equal(flowsAuditMessage({ status: "off" }, body), null);
});

test("wiring: owner route, diag probe, Tier 3 refresh in the mutex lane, daemon routes", () => {
  assert.ok(bot.includes('import { flowsBody, readFlows, flowsAuditMessage, FLOWS_REFRESH_CMD, FLOWS_REFRESH_TIMEOUT_MS } from "./flowsReport";'));
  assert.ok(bot.includes('app.get("/api/bot/flows", requireOwner, (_req, res) => {'), "owner-only route");
  assert.ok(bot.includes('res.json(flowsBody(readFlows(regimeDataDir(process.env)), new Date()));'));
  assert.ok((DIAG_PROBES as readonly string[]).includes("flows"));
  assert.ok(bot.includes('case "flows": {') && bot.includes('sanitizeDiag({ probe: "flows", ...flowsBody('));
  const t3 = bot.slice(bot.indexOf("async function tier3Strategic() {"), bot.indexOf('audit("TIER3", "Strategic scan complete");'));
  assert.ok(t3.includes("execPythonSerialized(FLOWS_REFRESH_CMD, { timeout: FLOWS_REFRESH_TIMEOUT_MS })"));
  assert.ok(t3.includes("flowsAuditMessage(flowsResult,") && t3.includes('noteJobError("tier3_strategic", `flows:'));
  const plan = planPythonExec(FLOWS_REFRESH_CMD, FLOWS_REFRESH_TIMEOUT_MS);
  assert.equal(plan.lane, "mutex");
  assert.equal(plan.timeoutMs, 60_000);
  assert.ok(FLOWS_REFRESH_CMD.includes("from flows_model import refresh_daily"));
  assert.ok(model.includes("def refresh_daily(") && model.includes("def report("));
  assert.ok(daemon.includes('"flows_refresh": ("flows_model", "refresh_daily")'));
  assert.ok(daemon.includes('"flows_report": ("flows_model", "report")'));
  assert.ok(model.includes('FLOWS_FILE = "voltrade_flows.json"') && FLOWS_FILE === "voltrade_flows.json");
});
