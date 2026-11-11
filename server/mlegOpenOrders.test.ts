// RESTING MULTI-LEG ORDERS (2026-11-06) — a resting multi-leg OPENING order is
// seen everywhere a single-leg one is: the options slot count, the
// same-underlying duplicate check, the tier dispatcher's resting-order gate and
// the 12-minute stale-order sweep (the cash reserve's half is Python:
// test_mleg_resting_visible.py).
//
// An Alpaca mleg order has an EMPTY top-level side and symbol (live
// /api/diag/orders); only a nested read (nested=true) carries its legs. Until
// the mleg limit-sign fix every credit entry filled at once; since then a credit
// condor rests whenever the market moved against it (3 of the 5 live credit
// entries would have), and the mid-first debit entries rest by design.
//
// Harness: the REAL countOpenOptionsOpeningOrders / followMlegOrder /
// sweepStaleOrders are cut out of bot.ts, transpiled and run against stubs.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { transformSync } from "esbuild";
import { followReplacements, hasOpenMlegOpeningOn, mlegOpeningLegs, mlegOpeningUnderlyings, openMlegUnderlyings } from "./mlegOpenOrders";
import { confirmCancelAfterDeleteError, shouldDispatchCsp } from "./cspSweepCooldown";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

function between(from: string, to: string): string {
  const start = bot.indexOf(from);
  assert.ok(start >= 0, `marker not found: ${from}`);
  const end = bot.indexOf(to, start + from.length);
  assert.ok(end > start, `end marker not found: ${to}`);
  return bot.slice(start, end);
}

const leg = (symbol: string, side: "buy" | "sell", intent: string) => ({ symbol, side, position_intent: intent, ratio_qty: "1" });
// a resting credit iron condor on IREN, 3 units, as GET /v2/orders?status=open&nested=true returns it
const CONDOR = {
  id: "mleg-iren", order_class: "mleg", symbol: "", side: "", type: "limit", qty: "3", filled_qty: "0", limit_price: "-0.71", status: "new",
  legs: [leg("IREN261218C00016000", "sell", "sell_to_open"), leg("IREN261218C00017000", "buy", "buy_to_open"),
         leg("IREN261218P00012000", "sell", "sell_to_open"), leg("IREN261218P00011000", "buy", "buy_to_open")],
};
// the group close of a condor (the #1655 mid-first close rests too): closing legs only
const CONDOR_CLOSE = { ...CONDOR, id: "mleg-close", limit_price: "0.96",
  legs: CONDOR.legs.map((l) => ({ ...l, side: l.side === "sell" ? "buy" : "sell", position_intent: l.side === "sell" ? "buy_to_close" : "sell_to_close" })) };
const CSP = { id: "csp-1", symbol: "XYZ261218P00040000", side: "sell", type: "limit", qty: "1", position_intent: "sell_to_open" };

test("mleg helpers: the opening legs and underlyings of a resting condor; a close and a single leg have none", () => {
  assert.equal(mlegOpeningLegs(CONDOR).length, 4);
  assert.deepEqual(mlegOpeningUnderlyings(CONDOR), ["IREN"]);
  assert.deepEqual(mlegOpeningLegs(CONDOR_CLOSE), [], "a group close frees slots, it never consumes one");
  assert.deepEqual(mlegOpeningLegs(CSP), []);
  assert.deepEqual(mlegOpeningLegs({ ...CONDOR, legs: undefined }), [], "a non-nested read carries no legs");
  assert.deepEqual(mlegOpeningLegs(null), []);
  assert.deepEqual(openMlegUnderlyings([CSP, CONDOR, CONDOR_CLOSE]), ["IREN"]);
  assert.equal(hasOpenMlegOpeningOn([CONDOR], "iren"), true);
  assert.equal(hasOpenMlegOpeningOn([CONDOR], "AMD"), false);
  assert.equal(hasOpenMlegOpeningOn([CONDOR_CLOSE], "IREN"), false);
});

test("followReplacements: the mid-first ladder's replaced orders lead to the one working now", async () => {
  const orders: Record<string, unknown> = { A: { status: "replaced", replaced_by: "B" }, B: { status: "replaced", replaced_by: "C" }, C: { status: "new" } };
  assert.equal(await followReplacements("A", async (id) => orders[id]), "C");
  assert.equal(await followReplacements("C", async (id) => orders[id]), "C");
  const loop: Record<string, unknown> = { X: { status: "replaced", replaced_by: "X" } };
  assert.equal(await followReplacements("X", async (id) => loop[id], 3), "X", "bounded by maxHops");
});

// ── the slot count: the REAL bot.ts helper ──────────────────────────────────
const slotSrc = between("function isOptionSymbol(", "// DAILY LINE 2026-11-05") + "\n" +
  between("function countOpenOptionsOpeningOrders(", "// ─── ET Hour Helper");
const countOpen = new Function("mlegOpeningLegs",
  `${transformSync(slotSrc, { loader: "ts" }).code}\nreturn countOpenOptionsOpeningOrders;`)(mlegOpeningLegs) as (o: unknown[]) => number;

test("countOpenOptionsOpeningOrders: a resting credit condor counts its four legs, as its filled legs will", () => {
  assert.equal(countOpen([CONDOR]), 4);
  assert.equal(countOpen([CONDOR, CSP]), 5, "plus a resting single-leg CSP, counted as before");
  assert.equal(countOpen([CONDOR_CLOSE]), 0, "a resting group close is not an opening");
  assert.equal(countOpen([{ ...CSP, side: "buy" }]), 0, "a single-leg buy closes a short (unchanged)");
  const qqq = { ...CONDOR, legs: [leg("QQQ261218C00600000", "sell", "sell_to_open"), leg("QQQ261218C00610000", "buy", "buy_to_open"),
    leg("QQQ261218P00500000", "sell", "sell_to_open"), leg("QQQ261218P00490000", "buy", "buy_to_open")] };
  assert.equal(countOpen([qqq]), 2, "QQQ puts are excluded the same way countOptionsPositions excludes them");
  assert.equal(countOpen([{ ...CONDOR, legs: undefined }]), 0, "without nested legs it cannot be seen (hence nested=true at every caller)");
});

test("both options-opening paths read the open orders nested, so the legs reach the count and the cash reserve", () => {
  const exec = between("async function executeTrades(", "const pendingOrderIds");
  assert.ok(exec.includes('await alpaca("/v2/orders?status=open&nested=true")'));
  const tiers = between("TIER DISPATCHER — routes tier_actions to the right execution path", "// ── TIER 3: BUY SPY/QQQ at 2x");
  assert.ok(tiers.includes('await alpaca("/v2/orders?status=open&nested=true")'));
  assert.match(bot, /const mqOrders = await alpaca\("\/v2\/orders\?status=open&nested=true"\)/, "morning queue's reserve read");
  assert.match(bot, /const liveOrders = await alpaca\("\/v2\/orders\?status=open&nested=true"\)\.catch\(\(\) => null\)/, "scale-in's reserve read");
});

// ── the same-underlying duplicate check ──────────────────────────────────────
test("DUP-SKIP: an options trade skips an underlying with a resting multi-leg opening order", () => {
  const dup = between("// Duplicate order check — skip if we already have a pending order for this ticker", "// Buying power check");
  assert.ok(dup.includes('await alpaca("/v2/orders?status=open&nested=true")'), "the legs are needed");
  assert.match(dup, /const restingStructure = isOptionsTrade && hasOpenMlegOpeningOn\(openOrders, trade\.ticker\);/);
  assert.match(dup, /openOrders\.some\(\(o[^)]*\) => o\.symbol === trade\.ticker\) \|\| restingStructure/, "the single-symbol check is kept");
  assert.match(dup, /noteGate\(trade, "rejected_dup_order", dupWhy\);\s*continue;/);
  assert.equal(hasOpenMlegOpeningOn([CONDOR], "IREN"), true, "so a second IREN structure is skipped while the first rests");
});

// ── the tier dispatcher's resting-order gate ─────────────────────────────────
test("tier dispatcher: no SELL_CSP on an underlying with a resting multi-leg opening order", () => {
  const ctx = { now: Date.now(), cooldownMinutes: 60, openOrders: [CONDOR], openStructureUnderlyings: openMlegUnderlyings([CONDOR]) };
  const d = shouldDispatchCsp("IREN", ctx);
  assert.equal(d.dispatch === false && d.reason, "open_structure");
  assert.equal(shouldDispatchCsp("AMD", ctx).dispatch, true);
  assert.equal(shouldDispatchCsp("IREN", { ...ctx, openStructureUnderlyings: openMlegUnderlyings([CONDOR_CLOSE]) }).dispatch, true);
  assert.ok(between("const cspGate = shouldDispatchCsp(action.ticker, {", "if (!cspGate.dispatch)")
    .includes("openStructureUnderlyings: openMlegUnderlyings(freshOpenOrdersForTiers),"));
});

// ── the 12-minute sweep: executeTrades registers a multi-leg entry ───────────
test("executeTrades registers an accepted multi-leg entry with the stale-order sweeper", () => {
  const branch = between("// 2026-11-06: \"new\" (Alpaca's status for an accepted, routed order)", 'audit("OPTIONS-TRADE"');
  assert.match(branch, /\["submitted", "filled", "pending_new", "accepted"\]\.includes\(optExec\.order\?\.status\) \|\| optExec\.order\?\.status === "new"/);
  assert.match(branch, /if \(entryLegs\.length > 1 && optExec\.order\?\.order_id\) \{\s*openOrders\.push\(\{[\s\S]*?side: "mleg",[\s\S]*?mleg: true,\s*\}\);/);
  const scanner = between("SCANNER PATH: Direct to select_contract + submit", "OPTIONS-SCANNER-EXEC");
  assert.ok(scanner.includes("'legs':_legs}"), "the scanner path reports the legs it registered");
});

// ── the 12-minute sweep: the REAL loop ───────────────────────────────────────
type Tracked = { orderId: string; ticker: string; score: number; placedAt: number; side: string; qty: number; limitPrice: number; mleg?: boolean };
function runSweepHarness(tracked: Tracked[], orders: Record<string, { status: string; replaced_by?: string }>) {
  const src = between("async function followMlegOrder(", "// ── Score-Based Order Replacement");
  const calls: string[] = [];
  const audits: string[] = [];
  const alpaca = async (p: string, opts?: { method?: string }) => {
    calls.push(`${opts?.method || "GET"} ${p}`);
    if (opts?.method === "DELETE") return {};
    if (p.startsWith("/v2/orders?status=open")) return Object.entries(orders).filter(([, o]) => ["new", "accepted"].includes(o.status)).map(([id]) => ({ id }));
    const id = p.split("/").pop() as string;
    return { id, ...(orders[id] || { status: "not_found" }) };
  };
  const fns = new Function("ctx", `const { openOrders, STALE_ORDER_MINUTES, alpaca, audit, sweptCspAt, resolvePendingEntryFill, resolvePendingExitFill, monitoredPositions, confirmCancelAfterDeleteError, followReplacements } = ctx;
${transformSync(src, { loader: "ts" }).code}
return { sweepStaleOrders, followMlegOrder };`)({
    openOrders: tracked, STALE_ORDER_MINUTES: 12, alpaca, audit: (_t: string, d: string) => audits.push(d), sweptCspAt: {},
    resolvePendingEntryFill: async () => true, resolvePendingExitFill: async () => true, monitoredPositions: {},
    confirmCancelAfterDeleteError, followReplacements,
  }) as { sweepStaleOrders: () => Promise<void> };
  return { sweep: fns.sweepStaleOrders, calls, audits };
}
const minutesAgo = (m: number) => Date.now() - m * 60_000;
const entry = (id: string, placedMinutesAgo: number, extra: Partial<Tracked> = {}): Tracked => ({
  orderId: id, ticker: "IREN", score: 72, placedAt: minutesAgo(placedMinutesAgo), side: "mleg", qty: 3, limitPrice: 0.71, mleg: true, ...extra });

test("sweep: a multi-leg entry resting 12+ minutes is cancelled like a single-leg one", async () => {
  const tracked = [entry("mleg-iren", 13), entry("csp-1", 13, { ticker: "XYZ", side: "sell", qty: 1, mleg: undefined })];
  const h = runSweepHarness(tracked, { "mleg-iren": { status: "new" }, "csp-1": { status: "new" } });
  await h.sweep();
  assert.deepEqual(h.calls.filter((c) => c.startsWith("DELETE")), ["DELETE /v2/orders/mleg-iren", "DELETE /v2/orders/csp-1"]);
  assert.ok(h.audits.some((a) => a.startsWith("Cancelled stale order: MLEG 3 IREN")), h.audits.join(" | "));
  assert.equal(tracked.length, 0, "both dropped from tracking once cancelled");
});

test("sweep: a re-priced (replaced) mleg entry is cancelled under the id working now", async () => {
  const tracked = [entry("A", 13)];
  const h = runSweepHarness(tracked, { A: { status: "replaced", replaced_by: "B" }, B: { status: "replaced", replaced_by: "C" }, C: { status: "new" } });
  await h.sweep();
  assert.deepEqual(h.calls.filter((c) => c.startsWith("DELETE")), ["DELETE /v2/orders/C"]);
});

test("sweep: a young replaced mleg entry stays tracked under its new id; a filled one is dropped", async () => {
  const young = [entry("A", 2)];
  let h = runSweepHarness(young, { A: { status: "replaced", replaced_by: "B" }, B: { status: "new" } });
  await h.sweep();
  assert.deepEqual(young.map((t) => t.orderId), ["B"], "still working: tracked, re-keyed");
  assert.equal(h.calls.filter((c) => c.startsWith("DELETE")).length, 0);
  const filled = [entry("F", 2)];
  h = runSweepHarness(filled, { F: { status: "filled" } });
  await h.sweep();
  assert.equal(filled.length, 0);
});
