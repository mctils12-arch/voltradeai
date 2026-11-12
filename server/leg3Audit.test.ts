// THIRD-LEG AUDIT (2026-11-06, rule call "SVXY 10% stop / 24h cooldown",
// decided KEEP): the SVXY leg's pass (scan_market's `third_leg`) becomes audit
// rows, so the stop's and the cooldown's rollback triggers can be checked.
// bot.ts never read `third_leg` before this.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { leg3AuditRows } from "./leg3Audit";

const here = path.dirname(fileURLToPath(import.meta.url));

test("a quiet pass (no action) writes nothing; junk input writes nothing", () => {
  assert.deepEqual(leg3AuditRows({ actions: [], status: "ok", regime: "BULL" }), []);
  assert.deepEqual(leg3AuditRows(undefined), []);
  assert.deepEqual(leg3AuditRows("x"), []);
  assert.deepEqual(leg3AuditRows({ actions: [null, 3, { type: "shadow_entry", ticker: "SVXY" }], status: "ok" }), []);
});

test("buy, protective exit and take-profit exit rows carry price, size, trigger and the broker's answer", () => {
  const rows = leg3AuditRows({
    status: "ok",
    actions: [
      { type: "vrp_harvest", symbol: "SVXY", shares: 300, price: 50, reason: "VXX ratio 1.100 (elevated+declining) — VRP harvest", regime: "CAUTION", order_id: "ord-1", ok: true },
      { type: "third_leg_exit", trigger: "stop_loss", symbol: "SVXY", reason: "SVXY -11.0% from entry <= -10% stop", pnl_pct: -11, price: 45.1, qty: "300", ok: true },
      { type: "third_leg_exit", trigger: "regime_recovery", symbol: "SVXY", reason: "Regime recovered to BULL — exiting third leg", pnl_pct: 6.2, price: 53.2, qty: "300", ok: false },
    ],
  });
  assert.deepEqual(rows.map((r) => r.type), ["LEG3-BUY", "LEG3-EXIT", "LEG3-EXIT"]);
  assert.equal(rows[0].message, "SVXY BUY 300 @ ~$50.00 limit — VXX ratio 1.100 (elevated+declining) — VRP harvest · regime CAUTION · order ord-1 accepted");
  assert.equal(rows[1].message, "SVXY SELL 300 @ ~$45.10 market · stop_loss (protective: stop loss, re-entry cooldown starts) — SVXY -11.0% from entry <= -10% stop · P&L -11.00% · accepted");
  assert.match(rows[2].message, /regime_recovery \(take profit: regime recovered\).*REJECTED by the broker$/);
});

test("a cooldown-blocked buy names the cooldown's end and the price it would have paid", () => {
  const until = Date.parse("2026-11-07T15:00:00Z") / 1000;
  const [row] = leg3AuditRows({ status: "ok", actions: [{ type: "leg3_cooldown_block", symbol: "SVXY", price: 48.25, vxx_ratio: 1.12, cooldown_until: until, reason: "buy blocked by the 24h re-entry cooldown" }] });
  assert.equal(row.type, "LEG3-COOLDOWN-BLOCK");
  assert.equal(row.message, "SVXY buy NOT placed: re-entry cooldown until 2026-11-07T15:00:00.000Z — would have bought at ~$48.25 · VXX ratio 1.120 · buy blocked by the 24h re-entry cooldown");
});

test("reserve cuts, order failures and a failed pass are reported; an unknown broker answer is never called accepted", () => {
  const rows = leg3AuditRows({
    status: "error", error: "KeyError: 'equity'",
    actions: [
      { type: "cash_reserve_trim", symbol: "SVXY", reason: "5,000 of room above the 15% reserve" },
      { type: "third_leg_error", stage: "exit (vol_spike)", symbol: "SVXY", error: "connection reset" },
      { type: "vrp_harvest", symbol: "SVXY", shares: 10, price: 50, ok: null },
    ],
  });
  assert.deepEqual(rows.map((r) => r.type), ["LEG3-RESERVE", "LEG3-ERROR", "LEG3-BUY", "LEG3-ERROR"]);
  assert.equal(rows[0].message, "SVXY buy cut by the regime cash reserve — 5,000 of room above the 15% reserve");
  assert.equal(rows[1].message, "SVXY exit (vol_spike) failed: connection reset");
  assert.match(rows[2].message, /broker answer unknown$/);
  assert.equal(rows[3].message, "third leg pass failed: KeyError: 'equity'");
});

test("bot.ts audits the scan's third_leg right after the floor rows", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes('import { leg3AuditRows } from "./leg3Audit";'));
  const floor = bot.indexOf('for (const row of floorAuditRows(result.spy_floor, "scan")) audit(row.type, row.message);');
  const leg3 = bot.indexOf("for (const row of leg3AuditRows(result.third_leg)) audit(row.type, row.message);");
  assert.ok(floor > 0 && leg3 > floor && leg3 - floor < 400, "the third-leg rows follow the floor rows in the scan handler");
});
