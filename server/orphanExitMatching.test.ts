// [RULE-REVIEW] 2026-10-27 — ORPHAN EXITS (audit §7.2.4): 342/398 live
// trade-feedback records are orphan_exit. Node-side halves of the fix (the
// Python halves are in test_orphan_exit_matching.py):
//   - exit records for an OCC option symbol key by the UNDERLYING, the key
//     every options entry record uses (POS-KILL used to write the OCC
//     symbol, which _find_entry_record can never match);
//   - the tier dispatcher's SELL_CSP / BUY_PUT inline Python registers the
//     pending entry marker after a submitted order.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildExitFillPayload, feedbackTickerFor } from "./exitFill";

test("an OCC option symbol maps to its underlying; equity symbols are unchanged", () => {
  assert.equal(feedbackTickerFor("MUU261120P00010000"), "MUU");
  assert.equal(feedbackTickerFor("NVDA261218C00150000"), "NVDA");
  assert.equal(feedbackTickerFor("BRK.B261218C00450000"), "BRK.B");
  assert.equal(feedbackTickerFor("nvda"), "NVDA");
  assert.equal(feedbackTickerFor("TQQQ"), "TQQQ");
  assert.equal(feedbackTickerFor("SPY"), "SPY");
});

test("a POS-KILL exit of an option position is keyed by the underlying, matching the entry record", () => {
  const p = buildExitFillPayload({
    ticker: "MUU261120P00010000", exitSide: "buy", qty: 2, fillPrice: 3.1,
    pnlPct: -160, exitReason: "position_kill", codeVersion: "1.0.1169",
  });
  assert.equal(p.ticker, "MUU", "the entry (options_manager pending entry) is keyed by the underlying");
});

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

test("the tier dispatcher's SELL_CSP and BUY_PUT branches register the pending entry marker", () => {
  const calls = bot.match(/register_pending_entry_feedback\(contract\.get\('occ_symbol', ''\), data\['ticker'\], contract\.get\('side', '(sell|buy)'\)\)/g) || [];
  assert.equal(calls.length, 2, `expected SELL_CSP + BUY_PUT, found ${calls.length}`);
  const csp = bot.slice(bot.indexOf('if (action.action === "SELL_CSP") {\n              // Delegate'), bot.indexOf("// ── TIER 4: BUY OTM SPY PUT"));
  assert.ok(/result = submit_options_order\(contract\)[\s\S]*?register_pending_entry_feedback/.test(csp),
    "the marker is written only after submit_options_order returns");
  assert.ok(/if result\.get\('status'\) in \('submitted', 'filled', 'pending_new', 'accepted'\):/.test(csp));
});
