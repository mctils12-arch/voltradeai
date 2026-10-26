// [RULE-REVIEW] 2026-10-26: the tier dispatcher's SELL_CSP gates were the
// last rules on the counterfactual program with no measured cost. A dropped
// CSP never had a shadow record (tier1_csp_core proposes it; deep_score never
// sees it), so the dispatcher collects its rejections and ONE
// tiered_strategy.log_csp_gate_shadow call after the loop logs them. Same
// source-inspection convention as shadowGateRelabel.test.ts: the loop runs
// against Alpaca, so the wiring is pinned on the source text.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function slice(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

const dispatcher = slice("for (const action of result.tier_actions) {", "if (tierActionsDeferred > 0) {");
const collector = slice("const cspGateRejections: Array<", "for (const action of result.tier_actions) {");

const GATES: Array<{ anchor: string; decision: string; exits: "continue" | "break" }> = [
  { anchor: "if (state.killSwitch) {", decision: "rejected_killswitch", exits: "break" },
  { anchor: 'audit("OPTIONS-SLOT-FULL", `${action.ticker}: tier dispatcher', decision: "rejected_options_slots", exits: "continue" },
  { anchor: 'audit("CSP-SKIP"', decision: "rejected_csp_resubmit", exits: "continue" },
];

for (const g of GATES) {
  test(`dispatcher gate ${g.decision} notes its rejection before it ${g.exits}s`, () => {
    const at = dispatcher.indexOf(g.anchor);
    assert.ok(at >= 0, `anchor not found in the dispatcher loop: ${g.anchor}`);
    const window = dispatcher.slice(at, at + 600);
    const noteAt = window.indexOf(`"${g.decision}"`);
    assert.ok(noteAt >= 0, `noteCspGate(..., "${g.decision}") must follow ${g.anchor}`);
    assert.ok(window.slice(0, noteAt).includes("noteCspGate("), "the bucket must be passed to noteCspGate");
    const exitAt = window.indexOf(`${g.exits};`, noteAt);
    assert.ok(exitAt > noteAt, `the note must come BEFORE the ${g.exits}`);
  });
}

test("the kill-switch break labels every later SELL_CSP action, not just the current one", () => {
  const at = dispatcher.indexOf("if (state.killSwitch) {");
  const window = dispatcher.slice(at, at + 400);
  assert.ok(window.includes("result.tier_actions.slice(result.tier_actions.indexOf(action))"));
});

test("only SELL_CSP actions are noted; the market-closed deferral is not a rule and is not noted", () => {
  assert.ok(collector.includes('a.action === "SELL_CSP"'), "floor ETF buys and tail hedges are not shadow candidates");
  assert.ok(collector.includes(".slice(0, 200)"), "reason truncated to the record field");
  const deferAt = dispatcher.indexOf("tierActionsDeferred++;");
  const deferWindow = dispatcher.slice(deferAt - 200, deferAt + 60);
  assert.ok(!deferWindow.includes("noteCspGate("), "a deferral is recomputed next in-hours scan, it is not a rejection");
});

test("buckets are rejected_-prefixed, and the slot bucket is the admission loop's (same rule, two enforcement points)", () => {
  for (const g of GATES) assert.match(g.decision, /^rejected_[a-z_]+$/);
  assert.ok(bot.includes('noteGate(trade, "rejected_options_slots"'), "executeTrades' OPTIONS-SLOT-FULL uses the same bucket");
});

test("one shadow-log call per dispatcher loop, via pythonCall with a temp-file fallback, never able to break the loop", () => {
  const flush = slice("if (cspGateRejections.length > 0) {", "if (tierActionsDeferred > 0) {");
  assert.ok(flush.includes('await pythonCall(\n              "csp_gate_shadow"'), "daemon route first");
  assert.ok(flush.includes("from tiered_strategy import log_csp_gate_shadow"), "fallback calls the same entry point");
  assert.ok(flush.includes("json.load(open('${cspGateTmpPath}'))"), "payload via temp file — reasons and metadata carry quotes");
  assert.ok(/try \{[\s\S]*catch \(cspGateErr\)[\s\S]*finally \{[\s\S]*fs\.unlinkSync\(cspGateTmpPath\)/.test(flush));
  assert.ok(flush.includes('audit("SHADOW-GATES", `csp:'), "visible in the audit log");
  assert.equal((bot.match(/pythonCall\(\s*"csp_gate_shadow"/g) || []).length, 1);
  // The flush sits after the loop body (whose last statement is the
  // TIER-DISPATCH-ERR catch) and before the per-scan TIER-DEFER line.
  const dispatchErrAt = dispatcher.indexOf('audit("TIER-DISPATCH-ERR"');
  const flushAt = dispatcher.indexOf("if (cspGateRejections.length > 0) {");
  assert.ok(dispatchErrAt > 0 && flushAt > dispatchErrAt, "the loop body never pays for the flush");
});

test("the daemon route is registered for the CSP gate logger", () => {
  const daemon = fs.readFileSync(path.join(here, "..", "voltrade_daemon.py"), "utf8");
  assert.ok(daemon.includes('"csp_gate_shadow": ("tiered_strategy", "log_csp_gate_shadow")'));
});
