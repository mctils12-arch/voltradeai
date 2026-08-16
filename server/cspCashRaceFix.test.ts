// CSP-CASH-RACE-FIX (2026-08-13): live audit logs (/api/diag/audit?type=T2-FAIL)
// showed the same two tickers (TLT, CRWV) rejected by Alpaca seconds apart,
// repeatedly across 24h+, with "Alpaca rejected: insufficient options buying
// power" — meaning OUR OWN pre-submission cash_available check passed, then
// Alpaca's real-time check failed — making up 47/100 recent T2-FAIL entries.
// Root cause traced this session (read-before-write): the tier dispatcher's
// `cashAvailable` was captured ONCE, before executeTrades() ran, and reused
// unchanged for every SELL_CSP action in result.tier_actions — the exact
// same TOCTOU shape as the options-slot race fixed 2026-08-03
// (optionsSlotRaceFix.test.ts), just on the dollar budget instead of the
// position-count budget. This pins the fix: the tier dispatcher re-fetches
// account cash immediately before dispatching any SELL_CSP (mirroring the
// existing freshPositionsForTiers re-fetch), and decrements a local running
// total after each successful submit by that order's actual collateral
// requirement — so a second SELL_CSP in the same batch is checked against
// what is genuinely left, not what was left before executeTrades and any
// earlier CSPs in this very loop.
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

test("tier dispatcher re-fetches live account cash before dispatching SELL_CSP, not the pre-executeTrades snapshot", () => {
  const dispatcherSection = slice(
    "TIER DISPATCHER — routes tier_actions to the right execution path",
    "// ── TIER 3: BUY SPY/QQQ at 2x",
  );
  assert.ok(
    dispatcherSection.includes('await alpaca("/v2/account")'),
    "tier dispatcher must fetch a fresh account snapshot for cash, not reuse the pre-executeTrades `cashAvailable`",
  );
  assert.ok(
    /let tierCashAvailable\s*=/.test(dispatcherSection),
    "tier dispatcher must track its own live-refreshed cash variable, separate from the outer cashAvailable",
  );
  // the CSP payload sent to Python must use the fresh per-cycle variable
  assert.ok(
    dispatcherSection.includes("cash: tierCashAvailable"),
    "SELL_CSP payload must be built from tierCashAvailable, not the stale outer cashAvailable",
  );
  assert.ok(
    !/cash:\s*cashAvailable/.test(dispatcherSection),
    "SELL_CSP payload must not fall back to reusing the stale pre-executeTrades cashAvailable directly",
  );
});

test("tier dispatcher decrements tierCashAvailable after each successful SELL_CSP submit, so a second SELL_CSP in the same batch sees the reduced budget", () => {
  const dispatcherSection = slice(
    "TIER DISPATCHER — routes tier_actions to the right execution path",
    "// ── TIER 3: BUY SPY/QQQ at 2x",
  );
  assert.ok(
    dispatcherSection.includes("tierCashAvailable -="),
    "a successful SELL_CSP submit must decrement the running tierCashAvailable total",
  );
  assert.ok(
    dispatcherSection.includes("result['cash_required']") || dispatcherSection.includes("cash_required"),
    "the inline python submission must surface the contract's actual cash_required back to the dispatcher so it can decrement the running total",
  );
  // decrement must happen inside the same success branch as the slot-count increment
  const successBranch = slice("if (r.status === \"submitted\" || r.status === \"filled\") {", "} else {");
  assert.ok(
    successBranch.includes("tierOptionsSlotsUsed++") && successBranch.includes("tierCashAvailable -="),
    "the slot-count increment and cash decrement must both fire in the same successful-submit branch",
  );
});

test("tierCashAvailable falls back to the pre-executeTrades cashAvailable if the fresh account re-fetch fails, never leaving the budget undefined", () => {
  const dispatcherSection = slice(
    "TIER DISPATCHER — routes tier_actions to the right execution path",
    "// ── TIER 3: BUY SPY/QQQ at 2x",
  );
  assert.ok(
    /freshAcctForTiers\s*\?\s*parseFloat\(freshAcctForTiers\.cash[^)]*\)\s*:\s*cashAvailable/.test(dispatcherSection),
    "on a failed account re-fetch, tierCashAvailable must fall back to the outer cashAvailable rather than becoming NaN/undefined",
  );
});
