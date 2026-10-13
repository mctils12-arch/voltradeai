// OFF-HOURS SELL_CSP FIX (2026-10-13): the Tier-2 scan window is 04:00-20:00
// ET (getETHour() is weekday-blind), so tier2Intelligence runs pre-market and
// after-hours with isMarketOpen=false. executeTrades() already routes
// off-hours stock entries to morningQueue, but the TIER DISPATCHER had no
// market-open check and submitted opening SELL_CSP (plus Tier-3 BUY / Tier-4
// BUY_PUT) orders whenever the scan ran. Equity options trade only in regular
// hours, so those orders rested at stale prices until a deploy's cancel-all
// or filled at the open into POS-KILL. This pins the fix: a pure gate
// (server/tierDispatchGate.ts) defers every order-placing tier action while
// the market is closed — not queued, the next in-hours scan recomputes — and
// the dispatcher loop consults it before any select_contract/submit, emitting
// ONE TIER-DEFER audit line per scan rather than one per action.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  shouldDispatchTierAction,
  tierDeferAuditMessage,
  TIER_ORDER_ACTIONS,
} from "./tierDispatchGate";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function slice(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

// ── pure gate ────────────────────────────────────────────────────────────────

test("market closed: opening SELL_CSP, Tier-3 BUY and Tier-4 BUY_PUT are all deferred", () => {
  for (const kind of ["SELL_CSP", "BUY", "BUY_PUT"]) {
    assert.equal(
      shouldDispatchTierAction({ action: kind }, { isMarketOpen: false }),
      false,
      `${kind} places a broker order and must not be dispatched while the market is closed`,
    );
  }
});

test("market open: every order-placing tier action is dispatchable (the gate never blocks in-hours trading)", () => {
  for (const kind of ["SELL_CSP", "BUY", "BUY_PUT"]) {
    assert.equal(shouldDispatchTierAction({ action: kind }, { isMarketOpen: true }), true, kind);
  }
});

test("TIER_ORDER_ACTIONS covers exactly the dispatcher's three order-placing branches", () => {
  assert.deepEqual([...TIER_ORDER_ACTIONS].sort(), ["BUY", "BUY_PUT", "SELL_CSP"]);
  // every branch the dispatcher actually runs must be covered by the gate
  const loop = slice("for (const action of result.tier_actions)", "TIER-DEFER");
  const branchKinds = [...loop.matchAll(/action\.action === "([A-Z_]+)"\)\s*\{/g)].map((m) => m[1]);
  assert.ok(branchKinds.length >= 3, `expected the dispatcher's action branches, found ${branchKinds.join(",")}`);
  for (const k of branchKinds) {
    assert.ok(TIER_ORDER_ACTIONS.has(k), `dispatcher branch ${k} places an order but is not gated off-hours`);
  }
});

test("unknown/missing action kinds pass through unchanged (the dispatcher's if/else chain already ignores them)", () => {
  assert.equal(shouldDispatchTierAction({ action: "HOLD" }, { isMarketOpen: false }), true);
  assert.equal(shouldDispatchTierAction({}, { isMarketOpen: false }), true);
  assert.equal(shouldDispatchTierAction(null, { isMarketOpen: false }), true);
});

test("TIER-DEFER message reports the deferred count and says market closed", () => {
  assert.match(tierDeferAuditMessage(3), /^3 tier actions deferred — market closed/);
  assert.match(tierDeferAuditMessage(1), /^1 tier action deferred — market closed/);
});

// ── dispatcher wiring (static source) ────────────────────────────────────────

test("dispatcher loop consults shouldDispatchTierAction with tier2Intelligence's isMarketOpen before any select_contract/submit", () => {
  const loop = slice("for (const action of result.tier_actions)", "// ── TIER 3: BUY SPY/QQQ at 2x");
  const gateIdx = loop.search(/if\s*\(!shouldDispatchTierAction\(action,\s*\{\s*isMarketOpen\s*\}\)\)\s*\{/);
  assert.ok(gateIdx > 0, "the dispatcher loop must call shouldDispatchTierAction(action, { isMarketOpen }) and skip on false");
  const selectIdx = loop.indexOf("select_contract(");
  const submitIdx = loop.indexOf("submit_options_order(contract)");
  const slotGateIdx = loop.search(/tierOptionsSlotsUsed\s*>=\s*MAX_OPTIONS_POSITIONS/);
  assert.ok(selectIdx > gateIdx && submitIdx > gateIdx, "the off-hours gate must precede SELL_CSP contract selection and submission");
  assert.ok(slotGateIdx > gateIdx, "the off-hours gate must precede the slot-cap gate so a deferred CSP never logs a misleading OPTIONS-SLOT-FULL");
  // deferred means skipped (continue), not queued for the morning
  const gateBlockMatch = loop.slice(gateIdx).match(/^if\s*\(!shouldDispatchTierAction\(action,\s*\{\s*isMarketOpen\s*\}\)\)\s*\{([^}]*)\}/);
  assert.ok(gateBlockMatch, "could not isolate the off-hours gate's if-block");
  const gateBlock = gateBlockMatch[1];
  assert.ok(gateBlock.includes("continue"), "a deferred tier action must be skipped with continue");
  assert.ok(!gateBlock.includes("morningQueue"), "deferred tier actions must NOT be queued — the next in-hours scan recomputes them");
  assert.ok(!gateBlock.includes("audit("), "the per-action skip must not audit — one TIER-DEFER line per scan only");
});

test("TIER-DEFER is audited once per scan, after the dispatcher loop, not per action", () => {
  const section = slice(
    "TIER DISPATCHER — routes tier_actions to the right execution path",
    "// Surface kill-switch warnings",
  );
  const deferAudits = section.match(/audit\("TIER-DEFER"/g) || [];
  assert.equal(deferAudits.length, 1, "exactly one TIER-DEFER audit call in the dispatcher section");
  const loopStart = section.indexOf("for (const action of result.tier_actions)");
  const dispatchErrIdx = section.indexOf('audit("TIER-DISPATCH-ERR"');
  const deferIdx = section.indexOf('audit("TIER-DEFER"');
  assert.ok(loopStart > 0 && dispatchErrIdx > loopStart && deferIdx > dispatchErrIdx, "TIER-DEFER must be emitted after the loop body closes");
  assert.ok(/if\s*\(tierActionsDeferred\s*>\s*0\)/.test(section), "TIER-DEFER only fires when something was actually deferred");
});

test("the off-hours gate is imported from the pure module, not re-implemented inline in bot.ts", () => {
  assert.ok(/import\s*\{[^}]*shouldDispatchTierAction[^}]*\}\s*from\s*"\.\/tierDispatchGate"/.test(bot));
  assert.equal((bot.match(/function shouldDispatchTierAction\(/g) || []).length, 0);
});
