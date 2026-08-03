// OPTIONS-SLOT-RACE-FIX (2026-08-03): live audit logs showed OPTIONS-SLOT-FULL
// firing at "(7/6)" — one MORE than system_config.py's documented
// MAX_OPTIONS_POSITIONS cap of 6 — and /api/diag/positions-detail confirmed
// the account genuinely held 7 real us_option positions. Root cause traced
// this session (read-before-write, not the same mechanism as KNOWN BROKEN #3):
// within one Tier-2 scan cycle, server/bot.ts calls executeTrades() (the
// scanner-path options opener) FIRST, then dispatches result.tier_actions
// (SELL_CSP, computed by tiered_strategy.py's tier1_csp_core against a
// positions snapshot taken BEFORE executeTrades ran) with ZERO re-check of
// live state at dispatch time. If executeTrades filled an options slot in
// between, the tier dispatcher's SELL_CSP actions — planned against the
// now-stale slots_available — could push the real count one past the cap.
// This pins the fix: the tier dispatcher re-fetches positions and
// re-enforces MAX_OPTIONS_POSITIONS via the same countOptionsPositions()
// helper executeTrades uses, immediately before dispatching any SELL_CSP,
// and increments its local counter after each successful submit — so a
// second SELL_CSP in the same batch can't also slip through.
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

test("countOptionsPositions is a single module-scope helper, not duplicated per-function filter logic", () => {
  const defs = bot.match(/function countOptionsPositions\(/g) || [];
  assert.equal(defs.length, 1, "countOptionsPositions must be defined exactly once (module scope) — re-duplicating the filter per function is what let the two enforcement points drift before");
  // executeTrades must call the shared helper, not re-inline the us_option/QQQ-overlay filter itself
  const execSection = slice("async function executeTrades(", "const pendingOrderIds");
  assert.ok(execSection.includes("countOptionsPositions(positions)"), "executeTrades must compute optionsPositions via the shared countOptionsPositions() helper");
});

test("tier dispatcher re-fetches live positions and re-enforces MAX_OPTIONS_POSITIONS before dispatching SELL_CSP", () => {
  const dispatcherSection = slice(
    "TIER DISPATCHER — routes tier_actions to the right execution path",
    "// ── TIER 3: BUY SPY/QQQ at 2x",
  );
  assert.ok(
    dispatcherSection.includes('await alpaca("/v2/positions")'),
    "tier dispatcher must fetch a fresh positions snapshot (not reuse the pre-executeTrades snapshot tier_actions was computed against)",
  );
  assert.ok(
    dispatcherSection.includes("countOptionsPositions("),
    "tier dispatcher must compute its options-slot count via the shared helper, consistent with executeTrades",
  );
  assert.ok(
    /tierOptionsSlotsUsed\s*>=\s*MAX_OPTIONS_POSITIONS/.test(dispatcherSection),
    "tier dispatcher must gate SELL_CSP against MAX_OPTIONS_POSITIONS immediately before dispatch",
  );
  assert.ok(
    dispatcherSection.includes("tierOptionsSlotsUsed++"),
    "tier dispatcher must increment its live counter after each successful SELL_CSP submit, so a second SELL_CSP in the same batch is also capped",
  );
  // the cap check must precede the loop body's SELL_CSP execution, not follow it
  const gateIdx = dispatcherSection.search(/tierOptionsSlotsUsed\s*>=\s*MAX_OPTIONS_POSITIONS/);
  const submitIdx = dispatcherSection.indexOf("submit_options_order(contract)");
  assert.ok(gateIdx > 0 && submitIdx > gateIdx, "the slot-cap gate must sit before the SELL_CSP order submission in source order");
});

test("MAX_OPTIONS_POSITIONS is declared exactly once, at module scope, shared by executeTrades and the tier dispatcher", () => {
  const decls = bot.match(/const MAX_OPTIONS_POSITIONS = \d+;/g) || [];
  assert.equal(decls.length, 1, "MAX_OPTIONS_POSITIONS must be declared exactly once — a second per-function local constant is exactly the drift class KNOWN BROKEN #3 already burned us on");
});
