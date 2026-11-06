// Scale-in by thirds + the scale-out fixes, wired in server/bot.ts (2026-11-06).
// Source-inspection (the closures live inside registerBotRoutes):
//   - executeTrades (stock + ETF) and the morning queue buy the FIRST third
//     after the shadow gate, and register the plan once the order is accepted;
//   - executeScaleInTranches runs each in-hours scan after rotation, with the
//     entry gates, the tranche spending its own reserve, one order call;
//   - syncMonitoredPositions counts the whole live position before the first
//     scale-out, and keeps pendingExit only while an exit order is working.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function section(fromMarker: string, toMarker: string): string {
  const start = bot.indexOf(fromMarker);
  assert.ok(start > 0, `marker not found: ${fromMarker}`);
  const end = bot.indexOf(toMarker, start + fromMarker.length);
  assert.ok(end > start, `end marker not found after start: ${toMarker}`);
  return bot.slice(start, end);
}

test("executeTrades stock entry: whole entry to the shadow gate, first third to the broker, plan after the order", () => {
  const s = section("// ── Stock execution (default or fallback from options) ──", "// Log with sizing details");
  const iShadow = s.indexOf('if (isEntryShadowed("stocks")) {');
  const iPlan = s.indexOf("const scaleIn = planScaleIn({ ticker: trade.ticker, qty,");
  const iAssign = s.indexOf("qty = scaleIn.firstQty;");
  const iPost = s.indexOf('const orderResult = await alpaca("/v2/orders", {');
  const iReg = s.indexOf("registerScaleIn(scaleIn.plan, orderId, scaleIn.reason);");
  assert.ok(s.includes("let qty = Math.floor(trade.shares);"));
  assert.ok(iShadow > 0 && iShadow < iPlan && iPlan < iAssign && iAssign < iPost && iPost < iReg);
  assert.match(s, /stopPrice: Number\(trade\.stop_loss\) \|\| 0/);
});

test("ETF substitution: the ETF's own shares are split, with a stop estimated from the stock's", () => {
  const s = section('if (trade.instrument === "etf" && trade.instrument_ticker', "addPositionToMonitor(etfTicker,");
  assert.ok(s.includes("let etfShares = etfPlan.qty;"));
  const iShadow = s.indexOf('if (isEntryShadowed("stocks")) {');
  const iSplit = s.indexOf("etfShares = etfScaleIn.firstQty;");
  const iPost = s.indexOf('const etfOrderResult = await alpaca("/v2/orders", {');
  const iReg = s.indexOf("registerScaleIn(etfScaleIn.plan, etfOrderResult?.id || null, etfScaleIn.reason);");
  assert.ok(iShadow > 0 && iShadow < iSplit && iSplit < iPost && iPost < iReg);
  assert.match(s, /stopPrice: etfStopFromStock\(Number\(trade\.price\) \|\| 0, Number\(trade\.stop_loss\) \|\| 0, etfPrice\)/);
});

test("morning queue: the first third is what is ordered, monitored and recorded", () => {
  const s = section("const mqOrderQty = mqScaleIn.firstQty;", "} catch (e: any) {");
  for (const want of ["qty: String(mqOrderQty),", "${mqOrderQty} ${trade.ticker} @ market", "qty: mqOrderQty,",
    "qtyRequested: mqOrderQty,", "addPositionToMonitor(trade.ticker, mqVerdict.monitorSide, trade.price || 0, mqOrderQty);",
    "registerScaleIn(mqScaleIn.plan, order?.id || null, mqScaleIn.reason);"]) {
    assert.ok(s.includes(want), want);
  }
  assert.ok(!s.includes("_qtyToSubmit"), "after the split nothing orders the whole queued quantity");
  const mq = section("async function executeMorningQueue(", "// ── Three-Tier Engine");
  assert.ok(mq.indexOf('if (isEntryShadowed("stocks")) {') < mq.indexOf("const mqScaleIn = planScaleIn("));
});

test("executeScaleInTranches runs after rotation in the market-open branch", () => {
  assert.match(bot, /await runUpgradeCandidates\(parseRotationPlan\(result\.rotation_plan\)\);\s*\/\/ SCALE-IN BY THIRDS[^\n]*\n\s*await executeScaleInTranches\(equity\);/);
  assert.equal((bot.match(/executeScaleInTranches\(/g) || []).length, 2, "one definition + one call");
});

test("a due tranche passes the entry gates and spends its own reservation before its one order", () => {
  const fn = section("async function executeScaleInTranches(", "// ── Overnight Research Engine");
  const order = [
    "if (!scaleInEnabled(process.env)) return;",
    "if (!Array.isArray(livePositions) || !Array.isArray(liveOrders)) return;",
    "const d = decideTranche(plan, ctx);",
    "if (!inWindow) continue;",
    'if (state.killSwitch || !haltAllows(currentHaltState(), "entry")) break;',
    'if (isEntryShadowed("stocks")) {',
    'budget = budget ?? await fetchReserveBudget(livePositions, liveOrders, equity, "scale-in");',
    "budget.release(d.tranche.qty * plan.ref_price);",
    "const fit = budget.fitShares(price, d.tranche.qty);",
    'const trancheOrder = await alpaca("/v2/orders", {',
    "applyTrancheDecision(plan, d, ctx, { qty: trancheQty, price, orderId: trancheOrder?.id ?? null });",
    "openOrders.push({ orderId: trancheOrder.id,",
    "saveScaleInState(file, st);",
  ];
  let at = -1;
  for (const marker of order) {
    const i = fn.indexOf(marker, at + 1);
    assert.ok(i > at, `out of order or missing: ${marker}`);
    at = i;
  }
  assert.equal((fn.match(/await alpaca\("\/v2\/orders", \{/g) || []).length, 1);
  assert.match(fn, /audit\("SCALE-IN-CANCEL"/);
  assert.ok(!fn.includes("settleEntryFill("), "a tranche fill is not a new ML entry record");
  assert.ok(!fn.includes("addPositionToMonitor("), "the position sync grows the monitored position; nothing resets it");
});

test("the position sync: whole live position before the first scale-out; pendingExit only while an exit works", () => {
  const fn = section("async function syncMonitoredPositions()", "\n  function addPositionToMonitor(");
  assert.match(fn, /const preScaleOut = scalesCompleted === 0;/);
  assert.match(fn, /originalQty: preScaleOut \? qty : originalQty,\s*remainingQty: preScaleOut \? qty : remainingQty,/);
  assert.match(fn, /openOrders\.some\(\(o\) => o\.ticker === ticker && o\.isExit === true\)/);
  assert.match(fn, /if \(existingPos\?\.pendingExit && exitStillWorking\) \{\s*monitoredPositions\[ticker\]\.pendingExit = true;/);
  // the 2026-08-04 live-qty cap stays as it was (positionQtySync.test.ts pins it)
  assert.ok(fn.includes("const remainingQty = Math.min(rawRemainingQty, qty);"));
  assert.ok(fn.includes("const originalQty = ps.original_qty || qty;"));
});
