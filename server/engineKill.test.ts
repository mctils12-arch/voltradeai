// ENGINE KILL VISIBILITY (2026-11-10, KNOWN BROKEN #49) — see engineKill.ts.
// Half 1 pins the pure functions on the live reason strings risk_kill_switch.py
// produces. Half 2 pins HOW bot.ts wires them (the regression the live incident
// exposed was a state that existed but was never surfaced).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  classifyEngineKill, engineKillSnapshot, engineKillHealth, shouldAuditEngineKill,
  engineKillAuditLine, ENGINE_KILL_AUDIT_INTERVAL_MS,
} from "./engineKill";

const NOW = Date.UTC(2026, 10, 10, 20, 0, 0);

// the exact reason strings risk_kill_switch.check_kill_switches persists
const LIVE = {
  dd: "Portfolio DD -20.0% <= -20%",
  daily: "Daily loss limit hit: -3.0%",
  consec: "5 consecutive losses; pause 24h",
  manual: "MANUAL kill file present",
};

test("classifyEngineKill maps the live reason strings and nothing is parsed for numbers", () => {
  assert.equal(classifyEngineKill(LIVE.dd), "PORTFOLIO_DD");
  assert.equal(classifyEngineKill("Portfolio DD kill: -25.3% <= -20%"), "PORTFOLIO_DD");
  assert.equal(classifyEngineKill(LIVE.daily), "DAILY_LOSS");
  assert.equal(classifyEngineKill(LIVE.consec), "CONSECUTIVE_LOSS");
  assert.equal(classifyEngineKill(LIVE.manual), "MANUAL");
  assert.equal(classifyEngineKill("Regime stress: VXX 1.45x"), "OTHER");
  assert.equal(classifyEngineKill(undefined), "OTHER");
});

test("engineKillSnapshot: no kill_status -> null; latched -> kind + tier block carried", () => {
  assert.equal(engineKillSnapshot({}, NOW), null);
  assert.equal(engineKillSnapshot({ kill_status: null }, NOW), null);
  const ok = engineKillSnapshot({ kill_status: { killed: false, kill_reason: "", warnings: [] } }, NOW)!;
  assert.equal(ok.killed, false);
  assert.equal(ok.kind, null);
  const s = engineKillSnapshot({
    kill_status: { killed: true, kill_reason: LIVE.dd },
    tier_kill_status: { killed: true, kill_reason: "Portfolio DD kill: -18.5% <= -20%" },
  }, NOW)!;
  assert.equal(s.killed, true);
  assert.equal(s.kind, "PORTFOLIO_DD");
  assert.equal(s.tierKilled, true);
  assert.equal(s.at, NOW);
});

test("engineKillHealth: the live incident shape (engine latched, Node off) degrades; both latched does not double-alarm", () => {
  const latched = engineKillSnapshot({ kill_status: { killed: true, kill_reason: LIVE.dd } }, NOW - 5 * 60_000)!;
  const h = engineKillHealth(latched, false, NOW);
  assert.equal(h.status, "latched");
  assert.equal(h.kind, "PORTFOLIO_DD");
  assert.equal(h.ageMinutes, 5);
  assert.equal(h.disagree, true);
  assert.equal(h.degrade, true);
  // Node also latched: the existing killed/liveness path owns it
  const both = engineKillHealth(latched, true, NOW);
  assert.equal(both.status, "latched");
  assert.equal(both.disagree, false);
  assert.equal(both.degrade, false);
});

test("engineKillHealth: not latched -> ok; no scan yet -> unknown and never degrades", () => {
  const ok = engineKillSnapshot({ kill_status: { killed: false, kill_reason: "" } }, NOW)!;
  assert.deepEqual(engineKillHealth(ok, false, NOW), {
    status: "ok", kind: null, ageMinutes: 0, nodeKillSwitch: false, disagree: false, degrade: false,
  });
  const unknown = engineKillHealth(null, false, NOW);
  assert.equal(unknown.status, "unknown");
  assert.equal(unknown.degrade, false);
});

test("the public health block carries no dollar figure — kind and age only, the reason stays on diag", () => {
  const latched = engineKillSnapshot({ kill_status: { killed: true, kill_reason: LIVE.dd } }, NOW)!;
  const h = engineKillHealth(latched, false, NOW);
  assert.doesNotMatch(JSON.stringify(h), /\$|\d{4,}/);
});

test("shouldAuditEngineKill: once per hour while the sides disagree, never otherwise", () => {
  const latched = engineKillSnapshot({ kill_status: { killed: true, kill_reason: LIVE.dd } }, NOW)!;
  const h = engineKillHealth(latched, false, NOW);
  assert.equal(shouldAuditEngineKill(h, 0, NOW), true);
  assert.equal(shouldAuditEngineKill(h, NOW - 10 * 60_000, NOW), false);
  assert.equal(shouldAuditEngineKill(h, NOW - ENGINE_KILL_AUDIT_INTERVAL_MS, NOW), true);
  assert.equal(shouldAuditEngineKill(engineKillHealth(latched, true, NOW), 0, NOW), false);
  const ok = engineKillSnapshot({ kill_status: { killed: false, kill_reason: "" } }, NOW)!;
  assert.equal(shouldAuditEngineKill(engineKillHealth(ok, false, NOW), 0, NOW), false);
});

test("engineKillAuditLine names the kind, the reason, the consequence and the probe", () => {
  const s = engineKillSnapshot({
    kill_status: { killed: true, kill_reason: LIVE.dd },
    tier_kill_status: { killed: true, kill_reason: "Portfolio DD kill: -18.5% <= -20%" },
  }, NOW)!;
  const line = engineKillAuditLine(s, engineKillHealth(s, false, NOW));
  assert.match(line, /PORTFOLIO_DD: Portfolio DD -20\.0%/);
  assert.match(line, /Node's kill switch is OFF/);
  assert.match(line, /every scanner entry suppressed/);
  assert.match(line, /\/api\/diag\/kill_state/);
  assert.match(line, /tier engine also blocked/);
  assert.match(line, /KNOWN BROKEN #49/);
});

// ── wiring half: bot.ts must use the module where the state is produced and read ──
const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

test("bot.ts records the snapshot where it audits KILL-SWITCH FIRED and audits the disagreement", () => {
  const fired = bot.indexOf('audit("KILL-SWITCH", `FIRED: ${result.kill_status.kill_reason}`)');
  assert.ok(fired > 0, "the FIRED audit site must still exist (frozen-adjacent behavior unchanged)");
  const around = bot.slice(fired - 400, fired + 1600);
  assert.match(around, /lastEngineKill = engineKillSnapshot\(result, /);
  assert.match(around, /shouldAuditEngineKill\(/);
  assert.match(around, /audit\("ENGINE-KILL-LATCHED", engineKillAuditLine\(/);
});

test("bot.ts /api/health bot block exposes engineKill and degrades on disagreement — without touching the HTTP gate", () => {
  const i = bot.indexOf("checks.checks.bot = {");
  assert.ok(i > 0);
  const block = bot.slice(i, i + 4000);
  assert.match(block, /engineKill: engineKillNow,/);
  assert.match(bot, /const engineKillNow = engineKillHealth\(lastEngineKill, state\.killSwitch, Date\.now\(\)\);/);
  assert.match(block, /if \(engineKillNow\.degrade\) checks\.status = "degraded";/);
  // the deploy gate must not learn about it (healthGate.ts SERVING_CHECKS unchanged)
  const gate = fs.readFileSync(path.join(here, "healthGate.ts"), "utf8");
  assert.doesNotMatch(gate, /engineKill/);
});

test("bot.ts serves the token-gated kill_state probe through pythonCall with a subprocess fallback", () => {
  const i = bot.indexOf('case "kill_state": {');
  assert.ok(i > 0, "kill_state probe missing");
  const block = bot.slice(i, i + 2500);
  assert.match(block, /pythonCall\(\s*"kill_state"/);
  assert.match(block, /from kill_state_probe import kill_state_report/);
  assert.match(block, /engine_last_scan: lastEngineKill/);
  const diag = fs.readFileSync(path.join(here, "diag.ts"), "utf8");
  assert.match(diag, /"kill_state"/);
  const daemon = fs.readFileSync(path.join(here, "..", "voltrade_daemon.py"), "utf8");
  assert.match(daemon, /"kill_state": \("kill_state_probe", "kill_state_report"\)/);
});
