// [RULE-REVIEW] 2026-10-26: executeTrades' admission loop applies ~15 gates
// after scan_market returns, and every candidate they dropped stayed
// labelled "taken" in the shadow log (deep_score logs the record before
// Node ever sees it). Each gate now notes its rejection and ONE
// shadow_update_decisions call after the loop relabels the records, so
// every Node rule earns a prevention-P&L in win_rate_by_decision. Same
// source-inspection convention as cspCapitalCounterfactualLogging.test.ts:
// the loop lives inside bot.ts's closure and runs against Alpaca, so the
// wiring is pinned on the source text.
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

const loop = slice("for (const trade of trades) {", "// ── Batch fill confirmation");

// Every gate that drops a candidate, the audit tag it already writes (or
// the code it already runs), the bucket it must relabel to, and whether it
// ends the loop (a break drops every later trade, so the remainder is
// labelled too).
const GATES: Array<{ anchor: string; decision: string; exits: "continue" | "break" }> = [
  { anchor: "if (state.killSwitch) {", decision: "rejected_killswitch", exits: "break" },
  { anchor: 'audit("OPTIONS-SLOT-FULL"', decision: "rejected_options_slots", exits: "continue" },
  { anchor: "if (slotsUsed >= MAX_POSITIONS) {", decision: "rejected_stock_slots", exits: "break" },
  { anchor: 'audit("SKIP", `${trade.ticker}: score', decision: "rejected_node_score", exits: "continue" },
  { anchor: "if (held.includes(trade.ticker))", decision: "rejected_held", exits: "continue" },
  { anchor: 'audit("SKIP", `${trade.ticker}: blocked for today', decision: "rejected_daily_block", exits: "continue" },
  { anchor: 'audit("SIZE-CAP"', decision: "rejected_size_cap", exits: "continue" },
  { anchor: 'audit("HEAT-CAP"', decision: "rejected_exposure_cap", exits: "break" },
  { anchor: 'audit("CORR-CAP"', decision: "rejected_corr_cap", exits: "continue" },
  { anchor: 'audit("HALT-SKIP"', decision: "rejected_stock_halted", exits: "continue" },
  { anchor: 'audit("MAX-RETRY"', decision: "rejected_max_retry", exits: "continue" },
  { anchor: 'audit("DUP-SKIP"', decision: "rejected_dup_order", exits: "continue" },
  { anchor: "const replaced = await replaceIfBetter(trade);", decision: "rejected_buying_power", exits: "continue" },
  { anchor: 'audit("OPTIONS-SKIP"', decision: "rejected_options_nocontract", exits: "continue" },
  { anchor: 'audit("OPTIONS-ERROR"', decision: "rejected_options_error", exits: "continue" },
];

for (const g of GATES) {
  test(`gate ${g.decision} notes its rejection before it ${g.exits}s`, () => {
    const at = loop.indexOf(g.anchor);
    assert.ok(at >= 0, `anchor not found in the admission loop: ${g.anchor}`);
    const window = loop.slice(at, at + 700);
    const note = g.exits === "break" ? `noteRemaining(trade, "${g.decision}"` : `noteGate(trade, "${g.decision}"`;
    const noteAt = window.indexOf(note);
    assert.ok(noteAt >= 0, `${note} must follow ${g.anchor}`);
    const exitAt = window.indexOf(`${g.exits};`, noteAt);
    assert.ok(exitAt > noteAt, `the note must come BEFORE the ${g.exits} that drops the candidate`);
  });
}

test("every Node bucket is distinct, rejected_-prefixed, and never reuses a Python-side bucket", () => {
  const decisions = GATES.map((g) => g.decision);
  assert.equal(new Set(decisions).size, decisions.length, "two gates share a bucket — attribution dies");
  for (const d of decisions) assert.match(d, /^rejected_[a-z_]+$/);
  // scan_market's own relabels (bot_engine.py) and the CSP dispatch bucket:
  // reusing one would blend two different rules into one win rate. In
  // particular HEAT-CAP (total exposure) must not become rejected_heat
  // (the sector/correlation block).
  const pythonSide = ["rejected_heat", "rejected_other", "rejected_sizing", "rejected_score", "rejected_masterkill", "rejected_capital", "taken"];
  for (const p of pythonSide) assert.ok(!decisions.includes(p), `${p} is a Python-side bucket`);
});

test("noteGate skips a trade consumed by the upgrade path and the remainder keeps held tickers honest", () => {
  const collector = slice("const gateRejections: Array<", "for (const trade of trades) {");
  assert.ok(collector.includes("!t._consumed"), "a consumed trade was ordered via UPGRADE — it stays taken");
  assert.ok(collector.includes('held.includes(t.ticker) ? "rejected_held" : decision'), "the remainder after a break must label already-held tickers rejected_held, not with the break reason");
  assert.ok(collector.includes(".slice(0, 200)"), "reason is truncated to the shadow record's 200-char field");
});

test("one relabel call per loop, via pythonCall with the daemon route and a temp-file subprocess fallback, never able to break the loop", () => {
  const flush = slice("if (gateRejections.length > 0) {", "// ── Batch fill confirmation");
  assert.ok(flush.includes('await pythonCall(\n          "shadow_update_decisions"'), "must route through pythonCall so the daemon is tried first");
  assert.ok(flush.includes("max_age_seconds: SHADOW_GATE_RELABEL_MAX_AGE_S"), "daemon args must carry the window");
  assert.ok(flush.includes("from shadow_portfolio import update_decisions_batch"), "subprocess fallback must call the same entry point");
  assert.ok(flush.includes("json.load(open('${gateTmpPath}'))"), "fallback reads the payload from a temp file — reasons carry quotes, never inline them into python3 -c");
  assert.ok(/try \{[\s\S]*catch \(gateErr\)[\s\S]*finally \{[\s\S]*fs\.unlinkSync\(gateTmpPath\)/.test(flush), "try/catch/finally: a relabel failure is audited and the temp file always removed");
  assert.ok(flush.includes('audit("SHADOW-GATES"'), "the relabel result is visible in the audit log");
  // The flush sits AFTER the loop closes and BEFORE fill confirmation —
  // the loop body never pays for it.
  assert.equal((loop.match(/pythonCall\(\s*"shadow_update_decisions"/g) || []).length, 1);
});

test("the relabel window constant matches shadow_portfolio.NODE_GATE_RELABEL_MAX_AGE_S", () => {
  const py = fs.readFileSync(path.join(here, "..", "shadow_portfolio.py"), "utf8");
  const pyVal = py.match(/^NODE_GATE_RELABEL_MAX_AGE_S = ([\d.]+)/m)?.[1];
  const tsVal = bot.match(/^const SHADOW_GATE_RELABEL_MAX_AGE_S = (\d+);/m)?.[1];
  assert.ok(pyVal && tsVal, "both constants must exist");
  assert.equal(Number(pyVal), Number(tsVal));
});

test("the daemon route is registered for the batch relabel", () => {
  const daemon = fs.readFileSync(path.join(here, "..", "voltrade_daemon.py"), "utf8");
  assert.ok(daemon.includes('"shadow_update_decisions": ("shadow_portfolio", "update_decisions_batch")'));
});

test("scan_market's own batch relabel counts reach the audit log from the scan result", () => {
  const at = bot.indexOf("result.shadow_gate_relabels");
  assert.ok(at > 0, "bot.ts must read shadow_gate_relabels off the Tier2 scan result");
  const window = bot.slice(at, at + 600);
  assert.ok(window.includes('audit("SHADOW-GATES"'), "a silently failing scan-side wiring must be visible");
  assert.ok(window.includes("g.by_decision"), "per-bucket counts are what makes the line readable");
});
