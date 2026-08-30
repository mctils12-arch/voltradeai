// CSP CAPITAL ALLOCATION counterfactual logging (research/open_questions.md,
// filed 2026-07-28, correction 2026-08-06): "would reserving a cash sleeve
// for CSP collateral raise long-run compound growth?" is a RULE-REVIEW
// question CLAUDE.md says needs evidence, not vibes — and the 2026-08-06
// session's own queued follow-up was to build the shadow_portfolio
// counterfactual bucket for a capital-starved CSP candidate, the same
// pattern tiered_strategy.py's log_masterkill_csp_shadow already uses for
// the master-kill-switch rejection bucket. This was never built — until
// this PR, a capital-starved SELL_CSP dispatch left only a T-FAIL audit
// line, never a labeled shadow record a future session could read
// win_rate_by_decision["rejected_capital"] off of.
//
// The SELL_CSP dispatch itself is an inline Python subprocess (bot.ts has
// no runtime for it), so — same convention as optionsCapitalCheckFix.test.ts
// — these tests assert on the raw embedded-Python source text rather than
// executing it from Node.
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

function cspDispatchSnippet(): string {
  return slice(
    "contract = select_contract(data['ticker'], data['strategy']",
    'print(json.dumps(result))',
  );
}

test("a capital-insufficient select_contract error triggers a shadow_portfolio counterfactual log", () => {
  const snippet = cspDispatchSnippet();
  assert.ok(
    /if\s+'budget'\s+in\s+_err\s+or\s+'capital'\s+in\s+_err:/.test(snippet),
    "the capital-insufficient signature check must gate the shadow log call — it must not fire for data/liquidity failures like 'No liquid options contracts' or 'Could not determine current price'",
  );
  assert.ok(snippet.includes("import shadow_portfolio"), "must use the existing shadow_portfolio module, not a new logging path");
  assert.ok(
    snippet.includes("decision='rejected_capital'"),
    "must log under its own decision bucket, distinct from rejected_masterkill/rejected_heat/rejected_other",
  );
});

test("the counterfactual log still fires (unconditionally) even when it would fail", () => {
  const snippet = cspDispatchSnippet();
  const budgetIdx = snippet.search(/if\s+'budget'\s+in\s+_err\s+or\s+'capital'\s+in\s+_err:/);
  const tryIdx = snippet.indexOf("try:", budgetIdx);
  const exceptIdx = snippet.indexOf("except Exception:", tryIdx);
  const printIdx = snippet.indexOf("print(json.dumps({'status': 'error'", exceptIdx);
  assert.ok(
    budgetIdx > 0 && tryIdx > budgetIdx && exceptIdx > tryIdx && printIdx > exceptIdx,
    "the shadow-log call must be wrapped in its own try/except so a logging failure can never suppress or alter the real error response the dispatcher already returns to the caller",
  );
});

test("the shadow log carries the price select_contract resolved, not the placeholder 0 sent to it", () => {
  const snippet = cspDispatchSnippet();
  assert.ok(
    snippet.includes("entry_price=contract.get('price', 0.0)"),
    "the CSP dispatch payload always sends price=0 ('Python will fetch current price') — the shadow record needs the price options_execution actually resolved internally, or backfill_outcomes can never label this record",
  );
});

test("the shadow log reads regime context off the action's own metadata, not a value bot.ts doesn't have here", () => {
  const snippet = cspDispatchSnippet();
  assert.ok(snippet.includes("_meta = data.get('metadata') or {}"), "must read from the JSON payload's metadata field");
  assert.ok(snippet.includes("vxx_ratio=_meta.get('vxx_ratio', 1.0)"), "must read vxx_ratio from action metadata, matching tier1_csp_core's new metadata keys");
  assert.ok(snippet.includes("regime_label=_meta.get('regime_label', 'unknown')"), "must read regime_label from action metadata, matching tier1_csp_core's new metadata keys");
});

test("cspPayload still threads action.metadata through, unchanged, so the new metadata keys actually reach the subprocess", () => {
  const dispatchSection = slice("const cspPayload = {", "cash_available=data.get('cash')");
  assert.ok(dispatchSection.includes("metadata: action.metadata,"), "the CSP dispatch payload must keep forwarding action.metadata verbatim");
});
