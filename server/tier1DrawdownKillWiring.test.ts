// REPAIR 2026-09-09 (KNOWN BROKEN #42 follow-up, live production incident):
// the -10%-from-peak drawdown kill switch (drawdownGuard.ts's evaluateDrawdown,
// docs call it "the live Tier-1 kill switch") used to live ONLY inside
// executeMorningQueue() — which itself only runs on the first market-open
// Tier-1 cycle of the day, and only when morningQueue.length > 0. On any day
// with an empty morning queue (or after that once-daily gate fired), the
// check never ran again for the rest of the day, despite its own comment
// claiming "every Tier 1 cycle".
//
// Live evidence this session (`/api/health`, `/api/diag/audit`): production
// equity sat at -17.9% below peak (equityPeak $110,727.04, equity ~$90,800)
// — already ~8 points past the -10% threshold — with ZERO DRAWDOWN-KILL and
// ZERO EQUITY-READ-INVALID audit lines. The check was simply unreachable
// that day: Tier2's independent daily-loss halt (evaluateDailyPnl, -3%
// threshold, unaffected by this bug) was firing every cycle and blocking new
// entries, which masked the fact that the -10% kill switch itself never ran.
//
// Fix: extracted the check into its own `checkDrawdownKillSwitch()` and
// call it unconditionally as tier1Reflex()'s own first step, independent of
// the morning-queue gate. Behavior of the check itself (evaluateDrawdown,
// the -10% threshold, the -25% liquidate-on-kill mercy rule) is unchanged —
// only WHEN it runs changed.
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

const drawdownFn = slice(
  "async function checkDrawdownKillSwitch(",
  "// ── Morning Queue Execution",
);
const tier1 = slice(
  "async function tier1Reflex() {",
  "  // Once per day, backfill shadow portfolio outcomes.",
);
const morningQueueFn = slice(
  "async function executeMorningQueue(",
  "// ── Three-Tier Engine",
);

test("checkDrawdownKillSwitch exists as its own function, independent of morningQueue state", () => {
  assert.ok(
    /async function checkDrawdownKillSwitch\(\)/.test(bot),
    "the drawdown-kill check must be its own callable function, not inlined only inside executeMorningQueue",
  );
  // The extracted function must still fetch the account and apply the same
  // validated-read/threshold/kill/liquidate logic that used to live inline.
  assert.ok(/const acct = await alpaca\("\/v2\/account"\);/.test(drawdownFn));
  assert.ok(/evaluateDrawdown\(acct\.equity, state\.equityPeak, state\.maxDrawdownPct\)/.test(drawdownFn));
  assert.ok(/saveKillSwitch\("drawdown-kill \(tier1\)"\)/.test(drawdownFn));
  assert.ok(/audit\("DRAWDOWN-KILL"/.test(drawdownFn));
  assert.ok(
    /VOLTRADE_LIQUIDATE_ON_KILL/.test(drawdownFn) && /t1Drawdown <= -25\.0/.test(drawdownFn),
    "the -25% liquidate-on-kill mercy rule must survive the extraction unchanged",
  );
});

test("tier1Reflex() calls checkDrawdownKillSwitch() unconditionally, before the morning-queue gate", () => {
  const callIdx = tier1.indexOf("checkDrawdownKillSwitch()");
  assert.ok(callIdx > 0, "tier1Reflex must call checkDrawdownKillSwitch()");
  const gateIdx = tier1.indexOf("!state.morningQueueExecuted && morningQueue.length > 0");
  assert.ok(gateIdx > 0, "morning-queue gate marker not found in tier1Reflex");
  assert.ok(
    callIdx < gateIdx,
    "checkDrawdownKillSwitch() must run BEFORE (and independent of) the " +
    "morning-queue-length/once-per-day gate — this is the exact bug: the " +
    "check must not be reachable only through that gate",
  );
});

test("checkDrawdownKillSwitch() call in tier1Reflex is not nested inside the morning-queue if-block", () => {
  // Regression guard against re-introducing the original bug in a new
  // shape: the call must appear textually before the `if (clockT1.is_open`
  // conditional even opens, not just before its body's inner call.
  const ifIdx = tier1.indexOf("if (clockT1.is_open");
  const callIdx = tier1.indexOf("checkDrawdownKillSwitch()");
  assert.ok(ifIdx > 0 && callIdx > 0 && callIdx < ifIdx);
});

test("executeMorningQueue no longer performs its own duplicate /v2/account fetch", () => {
  assert.ok(
    /async function executeMorningQueue\(equity: number\)/.test(bot),
    "executeMorningQueue must take the already-validated equity as a parameter instead of re-fetching the account",
  );
  assert.ok(
    !/alpaca\("\/v2\/account"\)/.test(morningQueueFn),
    "executeMorningQueue must not re-fetch /v2/account — that duplicate fetch was the original bug's home and must not come back",
  );
  // Cheap safety net should remain: don't proceed to place orders if the
  // kill switch (evaluated upstream this same cycle) is already on.
  assert.ok(/if \(state\.killSwitch\) return;/.test(morningQueueFn));
});

test("the morning-queue call site passes tier1Reflex's own validated equity through", () => {
  assert.ok(
    /await executeMorningQueue\(t1Equity\)/.test(tier1),
    "tier1Reflex must pass the equity value checkDrawdownKillSwitch() already computed this cycle, not re-derive it",
  );
});
