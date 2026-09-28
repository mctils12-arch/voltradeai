// Source-level wiring pins for the paper drawdown-kill AUTO-RESUME
// (human-directed 2026-09-28, KNOWN BROKEN #42/#43), same style as
// tier1DrawdownKillWiring.test.ts. The pure decision is unit-tested in
// killSwitchAutoResume.test.ts; these pin HOW bot.ts uses it: the kill trip
// and the owner toggle are untouched, resume goes through the same activation
// path as /api/bot/start, the latch reasons bot.ts writes are the ones the
// classifier recognizes, and the health HTTP gate is not involved.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { classifyLatchReason, isPaperAlpacaUrl } from "./killSwitchAutoResume";
import { SERVING_CHECKS } from "./healthGate";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function slice(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

const tier1Interval = slice("// TIER 1: Reflex (every 45 seconds)", "// TIER 2: Intelligence");
const performFn = slice("async function performKillSwitchAutoResume(", "// TIER 1: Reflex (every 45 seconds)");
const tickFn = slice("async function killSwitchAutoResumeTick(", "async function performKillSwitchAutoResume(");
const startRoute = slice('app.post("/api/bot/start"', 'app.post("/api/bot/stop"');
const killRoute = slice('app.post("/api/bot/kill"', "// Reset circuit breaker");
const drawdownFn = slice("async function checkDrawdownKillSwitch(", "// ── Morning Queue Execution");
const healthBot = slice("checks.checks.bot = {", "// Check 5b:");

test("every latch reason bot.ts writes is classified the way the auto-resume relies on", () => {
  const literals = [...bot.matchAll(/saveKillSwitch\("([^"]+)"/g)].map((m) => m[1]);
  assert.ok(literals.includes("drawdown-kill (tier1)"));
  assert.ok(literals.includes("drawdown-kill (account route)"));
  for (const r of literals) {
    assert.equal(classifyLatchReason(r), "DRAWDOWN_KILL", `trip reason ${r} must stay auto-resumable`);
  }
  assert.ok(bot.includes('saveKillSwitch(state.killSwitch ? "owner toggle ON" : "owner toggle OFF")'), "owner toggle must keep persisting both directions");
  assert.equal(classifyLatchReason("owner toggle ON"), "OWNER_MANUAL", "an owner ON latch must never auto-clear");
  assert.notEqual(classifyLatchReason("auto-resume: DATA_ANOMALY"), "DRAWDOWN_KILL");
});

test("the kill trip itself is unchanged: validated read, -10% latch, order cancel, -25% mercy rule", () => {
  assert.ok(/evaluateDrawdown\(acct\.equity, state\.equityPeak, state\.maxDrawdownPct\)/.test(drawdownFn));
  assert.ok(/if \(t1dd\.valid && t1dd\.kill && !state\.killSwitch\) \{\s*state\.killSwitch = true;\s*saveKillSwitch\("drawdown-kill \(tier1\)"\);/.test(drawdownFn));
  assert.ok(drawdownFn.includes('await alpaca("/v2/orders", { method: "DELETE" })'));
  assert.ok(/VOLTRADE_LIQUIDATE_ON_KILL/.test(drawdownFn) && /t1Drawdown <= -25\.0/.test(drawdownFn));
  assert.ok(/maxDrawdownPct: -10,/.test(bot), "the -10% threshold is not touched by this change");
});

test("the owner toggle remains and is not routed through the auto-resume", () => {
  assert.ok(killRoute.includes("state.killSwitch = !state.killSwitch;"));
  assert.ok(!killRoute.includes("performKillSwitchAutoResume"));
  assert.ok(!killRoute.includes("killSwitchAutoResumeTick"));
});

test("tier-1 interval: while latched it evaluates the auto-resume instead of tier1Reflex", () => {
  const call = tier1Interval.indexOf("if (state.killSwitch) { await killSwitchAutoResumeTick(); return; }");
  const gate = tier1Interval.indexOf("if (!state.active || state.killSwitch) return;");
  const reflex = tier1Interval.indexOf("await tier1Reflex();");
  assert.ok(call > 0, "auto-resume tick must be wired into the tier-1 interval");
  assert.ok(gate > call && reflex > gate, "tier1Reflex still never runs while killed");
  // exactly one caller of each
  assert.equal(bot.split("await killSwitchAutoResumeTick()").length - 1, 1);
  assert.equal(bot.split("await performKillSwitchAutoResume(").length - 1, 1);
  assert.ok(tickFn.includes("if (d.resume) await performKillSwitchAutoResume(d);"));
});

test("paper gate covers every effective base URL, and the hardcoded one is paper", () => {
  const base = /const ALPACA_BASE = "([^"]+)";/.exec(bot);
  assert.ok(base, "ALPACA_BASE constant not found");
  assert.equal(isPaperAlpacaUrl(base[1]), true);
  const paperExpr = "allPaperAlpacaUrls([ALPACA_BASE, process.env.ALPACA_BASE_URL || ALPACA_BASE])";
  assert.ok(tickFn.includes(paperExpr), "tick must check both the Node and the env (Python/routes) base URL");
  assert.ok(performFn.includes(paperExpr), "eligibility is re-checked at resume time");
  assert.ok(tickFn.includes("enabled: autoResumeEnabled(process.env)"), "env rollback lever is honored by the tick");
  assert.ok(performFn.includes("enabled: autoResumeEnabled(process.env)"), "and re-checked at resume time");
});

test("resume: persists the clear, audits + notifies + emails, re-activates via the /api/bot/start path", () => {
  assert.ok(/state\.killSwitch = false;\s*saveKillSwitch\(`auto-resume: \$\{basis\}`, \{ autoResume: true \}\);/.test(performFn));
  assert.ok(performFn.includes('audit("KILL SWITCH AUTO-RESUME", msg);'));
  assert.ok(performFn.includes('notify("alert", msg);'));
  assert.ok(performFn.includes("sendEmailAlert("));
  assert.ok(performFn.includes('"paper re-baseline after real loss"'), "PAPER_REBASE is loudly labeled");
  assert.ok(performFn.includes('startBotActivity("kill-switch auto-resume")'));
  assert.ok(startRoute.includes('startBotActivity("")'), "/api/bot/start uses the same activation function");
  assert.ok(!performFn.includes("state.active = true"), "no bypass of the start path's preconditions");
  // peak re-baseline is persisted on the same line (test_audit_critical style)
  assert.ok(performFn.includes("state.equityPeak = d.rebaselinePeakTo; saveEquityPeak();"));
  assert.ok(performFn.includes("RE-BASELINED $${priorPeak.toFixed(2)}"), "prior peak is logged");
  // Python-side halts go through the registered daemon route w/ subprocess fallback
  assert.ok(performFn.includes('"paper_resume_sync"') && performFn.includes("from paper_resume_sync import rebaseline_python_halts"));
  // the Python sync runs BEFORE the latch clears
  assert.ok(performFn.indexOf("pythonCall(") < performFn.indexOf("state.killSwitch = false;"));
});

test("start-path precondition: activation refuses while the kill switch is ON", () => {
  const fn = slice("function startBotActivity(", "// Start bot");
  assert.ok(fn.includes('if (state.killSwitch) return { ok: false, error: "Kill switch is ON. Disable it first." };'));
  assert.ok(fn.includes("ownerStopped"), "an explicit owner stop is not overridden by auto-resume");
});

test("health: autoResume is payload-only — never flips status, never reaches the HTTP gate", () => {
  assert.ok(healthBot.includes("autoResume: {"));
  assert.ok(!/checks\.status\s*=.*autoResume/.test(bot));
  assert.ok(!healthBot.includes("autoResumeStatus.numbers") && !healthBot.includes("priorPeak"), "no dollar figures on the unauthenticated endpoint");
  assert.deepEqual([...SERVING_CHECKS], ["server", "database"]);
  assert.ok(bot.includes("const httpCode = healthHttpCode(checks);"));
});

test("kill file stays backward compatible and records the anti-flap stamp", () => {
  assert.ok(bot.includes("state.killSwitch = loadKillSwitch()"));
  assert.ok(bot.includes("parseKillSwitchRecord(JSON.parse(fs.readFileSync(p, \"utf8\")))"));
  assert.ok(/JSON\.stringify\(\{ killSwitch: state\.killSwitch, reason, savedAt: .*lastAutoResumeAt \}\)/.test(bot));
});
