// Position management wiring in server/bot.ts (2026-11-06, refocus spec
// section 4): the cash reserve gates every Node entry path (executeTrades, the
// scanner CSP path, the tier dispatcher's SELL_CSP, the morning queue), and the
// once-dead runUpgradeCandidates now has a caller and places
// position_mgmt.plan_rotations' swaps through its original transmission calls.
// Source-inspection, the style of executeTradesStaleOrderTracking.test.ts:
// these are closures inside registerBotRoutes with no export surface.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const daemon = fs.readFileSync(path.join(here, "..", "voltrade_daemon.py"), "utf8");

function section(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start + fromMarker.length);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

test("the reserve snapshot comes from Python: daemon route first, python3 -c fallback to the same function", () => {
  const fn = section("async function fetchReserveBudget(", "function noteReserve(");
  assert.match(fn, /pythonCall\("cash_reserve_snapshot", args,/);
  assert.match(fn, /from position_mgmt import snapshot_rpc/);
  assert.match(fn, /new ReserveBudget\(snap, equity\)/);
  assert.match(fn, /audit\("CASH-RESERVE-ERROR"/, "an unreadable reserve is audited, never silent");
  assert.ok(daemon.includes('"cash_reserve_snapshot": ("position_mgmt", "snapshot_rpc"),'), "the daemon method table registers the route");
});

test("executeTrades: one budget per loop, fitted as the last gate before an order, a block relabels the shadow record", () => {
  const fn = section("async function executeTrades(", "// ── Tier 3: Strategic");
  const iBudget = fn.indexOf('const reserve = await fetchReserveBudget(');
  const iLoop = fn.indexOf("for (const trade of trades) {");
  assert.ok(iBudget > 0 && iBudget < iLoop, "the budget is read once, before the loop");
  const iBp = fn.indexOf("// Buying power check");
  const iFit = fn.indexOf("const reserveFit = fitTradeToReserve(reserve, trade, isOptionsTrade);");
  const iAttempt = fn.indexOf("orderAttemptCounts[trade.ticker] = (orderAttemptCounts[trade.ticker] || 0) + 1;");
  assert.ok(iBp < iFit && iFit < iAttempt, "after every other gate, before the attempt counter and the order");
  assert.match(fn.slice(iFit, iAttempt), /noteGate\(trade, "rejected_cash_reserve", reserveFit\.reason\);\s*continue;/);
  assert.equal((fn.match(/reserve\.spend\(Number\(optExec\.contract\?\.cash_required \|\| optExec\.contract\?\.max_cost \|\| trade\.position_value\) \|\| 0\);/g) || []).length, 2,
    "an option open spends its real cost, shadowed and live");
  assert.match(fn, /cash: reserve\.active \? reserve\.remaining : null/);
  assert.match(fn, /select_contract\(data\['ticker'\], data\['strategy'\], data\['price'\], data\['equity'\], cash_available=data\.get\('cash'\)\)/,
    "the scanner CSP's collateral is capped by the reserve's room");
});

test("tier dispatcher: SELL_CSP collateral = min(options buying power, reserve room); none left = skipped and shadow-logged", () => {
  const fn = section("const tierReserve = await fetchReserveBudget(", "// ── TIER 3: BUY SPY/QQQ at 2x");
  const iLoop = fn.indexOf("for (const action of result.tier_actions) {");
  assert.match(fn.slice(0, iLoop), /if \(tierReserve\.active && tierReserve\.remaining < tierCashAvailable\) \{[\s\S]*?tierCashAvailable = tierReserve\.remaining;/,
    "the batch's cash is capped once, before the loop, so the payload's tierCashAvailable carries the cap");
  assert.match(fn, /const cspFit: ReserveFit = tierReserve\.fitOpen\(\);/);
  assert.match(fn, /noteCspGate\(action, "rejected_cash_reserve", cspFit\.reason\);\s*continue;/);
  assert.match(fn, /cash: tierCashAvailable,/);
  assert.equal((fn.match(/tierReserve\.spend\(parseFloat\(r\.cash_required\) \|\| 0\);/g) || []).length, 2);
});

test("morning queue: queued entries are fitted to the reserve before the shadow gate and the order", () => {
  const fn = section("async function executeMorningQueue(", "// ── Three-Tier Engine");
  const iBudget = fn.indexOf('const mqReserve = await fetchReserveBudget(');
  const iFit = fn.indexOf("const mqFit = mqReserve.fitShares(Number(trade.price) || 0, Math.floor(trade.shares));");
  const iQty = fn.indexOf("const _qtyToSubmit = Math.floor(trade.shares);");
  const iShadow = fn.indexOf('if (isEntryShadowed("stocks")) {');
  const iPost = fn.indexOf('const order = await alpaca("/v2/orders", {');
  assert.ok(iBudget > 0 && iBudget < iFit && iFit < iQty && iQty < iShadow && iShadow < iPost,
    "fitted before the qty is taken, so a trim flows into the unchanged qty guard, shadow gate and order");
  assert.match(fn, /if \(mqFit\.action === "block"\) continue;/);
  assert.match(fn, /if \(mqFit\.action === "trim"\) trade\.shares = mqFit\.qty;/);
});

test("runUpgradeCandidates is no longer dead: tier2Intelligence calls it after executeTrades, in market hours", () => {
  const calls = bot.match(/runUpgradeCandidates\(/g) || [];
  assert.equal(calls.length, 2, "one definition + one call site");
  assert.match(bot, /await executeTrades\(result\.new_trades \|\| \[\], equity\);\s*\/\/ CAPITAL ROTATION[^\n]*\n\s*await runUpgradeCandidates\(parseRotationPlan\(result\.rotation_plan\)\);/);
});

test("runUpgradeCandidates re-checks each swap, keeps its original transmission calls, and records both fills", () => {
  const fn = section("async function runUpgradeCandidates(", "// ── Overnight Research Engine");
  assert.match(fn, /if \(state\.killSwitch \|\| !haltAllows\(currentHaltState\(\), "entry"\)\) break;/);
  assert.match(fn, /const pre = rotationPreflight\(rot, \{/);
  assert.match(fn, /isEntryShadowed\("stocks"\)/);
  // the transmission lines are the function's originals, byte for byte
  assert.ok(fn.includes('const closeOrder = await alpaca(`/v2/positions/${candidate.ticker}`, { method: "DELETE" });'));
  assert.ok(fn.includes('const upgradeOrderParams = getOrderParams(betterPick.price || 0);'));
  assert.ok(fn.includes('body: JSON.stringify({ symbol: betterPick.ticker, qty: String(upgradeQty), side: "buy", ...upgradeOrderParams }),  // Always buy (short blocked)'));
  assert.equal((fn.match(/await alpaca\("\/v2\/orders", \{/g) || []).length, 1, "exactly one order POST: the entry");
  assert.match(fn, /pendingExitFill: \{ ticker: candidate\.ticker, exitSide: "sell", pnlPct: pre\.pnlPct, exitReason: "rotation"/);
  assert.match(fn, /settleEntryFill\(upgradeOrderResult\.id, upgradeOrderResult, \{/);
  assert.match(fn, /appendPositionMgmtLog\(\{ kind: "rotation", action: "rotate"/);
  assert.match(fn, /if \(!swapPlaced\) \{\s*appendPositionMgmtLog\(\{ kind: "rotation", action: "failed"/);
  assert.ok(fn.includes('} catch (e: any) { audit("UPGRADE-ERROR", `Failed to upgrade: ${e.message}`); }'), "the original catch line is unchanged");
  assert.match(fn, /audit\("ROTATE-SKIP"/);
  assert.match(fn, /break; \/\/ one swap per scan/);
});
