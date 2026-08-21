// OPTIONS-CAPITAL-CHECK-FIX (2026-08-06, revived + extended 2026-08-21):
// live /api/diag/audit showed the SAME symptom the 2026-07-31 LIVE-CAPITAL
// CAP fix was written to resolve still firing 6 days later — repeated
// T2-FAIL "insufficient options buying power for cash-secured put"
// rejections (DRAM, VZ) alongside KILL-WARN "Free BP below 10%: 0.0%", on
// live capital that our own affordability pre-check had judged sufficient.
// Root cause: the 07-31 fix threaded acct.cash into options_execution's
// cash_available cap, but Alpaca's account object carries a DEDICATED
// options_buying_power field (confirmed against Alpaca's API docs) that
// reflects collateral already committed to open short-option positions the
// way raw cash does not — Alpaca's own rejection message names exactly this
// field. Using cash let the internal check pass on capital Alpaca's real
// order-submission check would reject, submitting a doomed order every
// cycle. This pins the fix: prefer options_buying_power, falling back to
// cash only when the account response omits the field.
// 2026-08-21 EXTENSION: this original PR (#708) never merged in 6 days; by
// the time it was revived, a later 2026-08-13 "CSP-CASH-RACE FIX" session
// had added a SEPARATE fresh-refetch (`tierCashAvailable`/
// `freshAcctForTiers`) that is the value actually threaded into the live
// SELL_CSP dispatch payload — and it re-read `.cash`, silently
// reintroducing the exact same wrong-field bug on the path that matters.
// `cashAvailable` (this file's original subject) is now only the fallback
// used when that fresh fetch fails. Both are covered below.
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

test("cashAvailable prefers acct.options_buying_power over acct.cash", () => {
  const section = slice(
    "LIVE-CAPITAL CAP 2026-07-31",
    "const lastEquity = parseFloat(acct.last_equity",
  );
  assert.ok(
    /parseFloat\(acct\.options_buying_power\)/.test(section),
    "the capital-available computation must read acct.options_buying_power — the field Alpaca's own CSP rejection message actually checks",
  );
  const cashAssignIdx = section.search(/const cashAvailable\s*=/);
  const optionsBpIdx = section.search(/const optionsBp\s*=\s*parseFloat\(acct\.options_buying_power\)/);
  assert.ok(optionsBpIdx > 0 && cashAssignIdx > optionsBpIdx, "optionsBp must be computed before cashAvailable is assigned from it");
});

test("cashAvailable falls back to acct.cash only when options_buying_power is absent/non-numeric", () => {
  const section = slice(
    "const optionsBp = parseFloat(acct.options_buying_power);",
    "const lastEquity = parseFloat(acct.last_equity",
  );
  assert.ok(
    /Number\.isFinite\(optionsBp\)\s*\?\s*optionsBp\s*:\s*parseFloat\(acct\.cash \|\| "0"\)/.test(section),
    "cashAvailable must fall back to acct.cash only when options_buying_power failed to parse to a finite number, not silently prefer cash",
  );
});

test("tierCashAvailable (not raw freshAcctForTiers.cash) feeds the SELL_CSP dispatch payload's cash field", () => {
  const dispatchSection = slice(
    "const cspPayload = {",
    "cash_available=data.get('cash')",
  );
  assert.ok(dispatchSection.includes("cash: tierCashAvailable,"), "the CSP dispatch payload must thread tierCashAvailable through to options_execution's cash_available parameter");
});

test("tierCashAvailable prefers freshAcctForTiers.options_buying_power over .cash", () => {
  const section = slice(
    "const freshAcctForTiers = await alpaca(\"/v2/account\").catch(() => null);",
    "for (const action of result.tier_actions)",
  );
  assert.ok(
    /parseFloat\(freshAcctForTiers\.options_buying_power\)/.test(section),
    "the fresh per-batch refetch must read freshAcctForTiers.options_buying_power, not just .cash — this is the value actually dispatched to options_execution",
  );
  const optionsBpIdx = section.search(/const freshOptionsBp\s*=/);
  const tierCashIdx = section.search(/let tierCashAvailable\s*=/);
  assert.ok(optionsBpIdx > 0 && tierCashIdx > optionsBpIdx, "freshOptionsBp must be computed before tierCashAvailable is assigned from it");
});

test("tierCashAvailable falls back to freshAcctForTiers.cash only when options_buying_power is absent/non-numeric, and to cashAvailable only when the refetch itself fails", () => {
  const section = slice(
    "const freshAcctForTiers = await alpaca(\"/v2/account\").catch(() => null);",
    "for (const action of result.tier_actions)",
  );
  assert.ok(
    /Number\.isFinite\(freshOptionsBp\)\s*\?\s*freshOptionsBp\s*:\s*parseFloat\(freshAcctForTiers\.cash \|\| "0"\)/.test(section),
    "tierCashAvailable must fall back to freshAcctForTiers.cash only when options_buying_power failed to parse to a finite number",
  );
  assert.ok(
    /:\s*cashAvailable;/.test(section),
    "tierCashAvailable must still fall back to the outer cashAvailable when the fresh account refetch itself fails (freshAcctForTiers is null)",
  );
});
