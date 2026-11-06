// Floor passes become audit rows (2026-11-05): the scan's floor trades were
// never audited, and the kill-switch reduce-only pass must log every action
// (human decision 2026-11-05). Run: npx tsx --test server/floorAudit.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { floorAuditRows } from "./floorAudit";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const slice = (from: string, to: string) => {
  const i = bot.indexOf(from);
  assert.ok(i >= 0, `missing: ${from}`);
  const j = bot.indexOf(to, i + from.length);
  assert.ok(j > i, `missing end: ${to}`);
  return bot.slice(i, j);
};

const PLAN = { mode: "vol_managed_spy", sigma: 0.1546, target_exposure: 1.035, current_exposure: 0.47, capital: 76472.83, rebalance: true };

test("a scan rebalance: one summary row, then one row per order with its limit and order id", () => {
  const rows = floorAuditRows({
    ...PLAN, status: "ok",
    actions: [
      { type: "floor_vm_legacy_sell", ticker: "QQQ", qty: 24, limit: 655.1, ok: true, order_id: "a1", error: null },
      { type: "floor_vm_buy", ticker: "SPY", qty: 97, limit: 758.2, ok: true, order_id: "a2", error: null },
      { type: "floor_vm_buy", ticker: "SSO", qty: 20, limit: 99.5, ok: false, order_id: null, error: "insufficient buying power" },
    ],
  }, "scan");
  assert.equal(rows.length, 4);
  assert.ok(rows.every((r) => r.type === "FLOOR-VM"));
  assert.match(rows[0].message, /rebalance — sigma20 15\.5%, target 1\.03x, now 0\.47x of capital \$76473/);
  assert.equal(rows[1].message, "SELL legacy 24 QQQ @ 655.10 limit — order a1");
  assert.equal(rows[2].message, "BUY 97 SPY @ 758.20 limit — order a2");
  assert.match(rows[3].message, /^BUY 20 SSO @ 99\.50 limit — REJECTED: insufficient buying power/);
});

test("a flows cut (2026-11-06) shows the rule's target and the multiplier, and one reason row", () => {
  const cut = {
    ...PLAN, target_exposure: 0.932, rule_target_exposure: 1.035, exposure_multiplier: 0.9, status: "ok",
    flows: { multiplier: 0.9, applied: true, reason: "Expected net equity selling (pressure -1.00, estimate): floor target x0.90." },
    actions: [{ type: "floor_vm_buy", ticker: "SPY", qty: 90, limit: 758.2, ok: true, order_id: "f1", error: null }],
  };
  const rows = floorAuditRows(cut, "scan");
  assert.equal(rows.length, 3);
  assert.match(rows[0].message, /target 0\.93x \(rule 1\.03x x flows 0\.90\), now 0\.47x/);
  assert.equal(rows[1].message, "flows ESTIMATE: Expected net equity selling (pressure -1.00, estimate): floor target x0.90.");
  assert.equal(rows[2].message, "BUY 90 SPY @ 758.20 limit — order f1");
  const kill = floorAuditRows({ ...cut, actions: [{ type: "floor_vm_sell", ticker: "SSO", qty: 5, limit: 99.9, ok: true, order_id: "k" }] }, "kill");
  assert.equal(kill[1].type, "FLOOR-KILL-REDUCE");
  assert.match(kill[1].message, /^flows ESTIMATE: /);
  const neutral = floorAuditRows({ ...cut, exposure_multiplier: 1, flows: { multiplier: 1, applied: false, reason: "no adjustment" } }, "scan");
  assert.equal(neutral.length, 2, "no flows row when nothing was cut");
  assert.doesNotMatch(neutral[0].message, /flows/);
});

test("a scan pass that only deferred or waited writes nothing (it repeats every scan)", () => {
  assert.deepEqual(floorAuditRows({ ...PLAN, status: "deferred", actions: [{ type: "floor_vm_deferred", reason: "outside 09:45-15:54 ET" }] }, "scan"), []);
  assert.deepEqual(floorAuditRows({ ...PLAN, status: "ok", actions: [{ type: "floor_vm_wait", ticker: "SPY", reason: "order working" }] }, "scan"), []);
  assert.deepEqual(floorAuditRows({ ...PLAN, rebalance: false, status: "ok", actions: [] }, "scan"), []);
  assert.deepEqual(floorAuditRows({ actions: [], status: "ok", target_pct: 0 }, "scan"), [], "the basket floor's result is not this formatter's");
});

test("a failed floor pass is one row in either context", () => {
  assert.deepEqual(floorAuditRows({ mode: "vol_managed_spy", status: "error: broker down", actions: [] }, "scan"),
    [{ type: "FLOOR-VM", message: "floor pass failed: error: broker down" }]);
  assert.equal(floorAuditRows({ status: "error: x" }, "kill")[0].type, "FLOOR-KILL-REDUCE-ERROR");
});

test("the kill pass logs every action, under its own type", () => {
  const rows = floorAuditRows({
    ...PLAN, target_exposure: 0.49, current_exposure: 1.5, status: "ok", reduce_only: true,
    actions: [
      { type: "floor_vm_sell", ticker: "SSO", qty: 425, limit: 99.95, ok: true, order_id: "k1" },
      { type: "floor_vm_sell", ticker: "SPY", qty: 43, limit: 499.9, ok: true, order_id: "k2" },
      { type: "cc_unwind", symbol: "QQQ261218C00700000", qty: 1, ticker: "QQQ" },
      { type: "floor_vm_legacy_sell", ticker: "QQQ", qty: 10, limit: 599.9, ok: true, order_id: "k3" },
    ],
  }, "kill");
  assert.equal(rows.length, 5);
  assert.ok(rows.every((r) => r.type === "FLOOR-KILL-REDUCE"));
  assert.match(rows[0].message, /^kill switch tripped, reduce-only \(sells only: SSO, then SPY, then legacy\)/);
  assert.deepEqual(rows.slice(1).map((r) => r.message.split(" — ")[0]),
    ["SELL 425 SSO @ 99.95 limit", "SELL 43 SPY @ 499.90 limit", "BUY TO CLOSE covered call 1 QQQ261218C00700000", "SELL legacy 10 QQQ @ 599.90 limit"]);
});

test("a kill pass that did nothing says so once a day, with why", () => {
  const idle = { ...PLAN, rebalance: false, status: "ok", actions: [] };
  const first = floorAuditRows(idle, "kill", false);
  assert.equal(first.length, 1);
  assert.match(first[0].message, /pass ran, nothing to reduce \(within the band of its target\)/);
  assert.deepEqual(floorAuditRows(idle, "kill", true), [], "already said today");
  const deferred = floorAuditRows({ ...PLAN, status: "deferred", actions: [{ type: "floor_vm_deferred", reason: "broker clock: market closed" }] }, "kill");
  assert.match(deferred[0].message, /deferred \(broker clock: market closed\)/);
  const basket = floorAuditRows({ mode: "basket", actions: [], status: "skipped: the basket floor has no reduce-only pass (it holds while killed)" }, "kill");
  assert.match(basket[0].message, /^kill switch tripped: skipped: the basket floor has no reduce-only pass/);
});

test("bot.ts: the scan audits its floor pass, and the halted pass runs the kill floor only under the kill latch", () => {
  assert.ok(bot.includes('import { floorAuditRows } from "./floorAudit";'));
  assert.ok(bot.includes('for (const row of floorAuditRows(result.spy_floor, "scan")) audit(row.type, row.message);'));
  const halted = slice("async function tier2HaltedExits(): Promise<void> {", "let killFloorIdleDay");
  assert.ok(halted.indexOf("if (state.killSwitch) await killSwitchFloorReduce();") > halted.indexOf("formatOptionsMgmt(r.result)"),
    "the options exits run first; a failed options pass no longer skips the floor");
  const kill = slice("async function killSwitchFloorReduce(): Promise<void> {", "async function scheduleTier2() {");
  assert.ok(kill.includes('"floor_reduce_only"') && kill.includes("floor_reduce_only_pass()"));
  assert.ok(kill.includes('floorAuditRows(r.result, "kill", killFloorIdleDay === today)'));
  assert.ok(kill.includes('audit("FLOOR-KILL-REDUCE-ERROR"'));
  const heavy = slice("const HEAVY_DAEMON_ONLY = new Set([", "]);");
  assert.ok(heavy.includes('"floor_reduce_only"'), "never a fresh interpreter for a bot_engine import");
});
