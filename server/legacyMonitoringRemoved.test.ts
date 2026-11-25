// legacyMonitoringRemoved.test.ts — the three pre-refocus monitoring routes stay removed (owner, 2026-11-25:
// "if the old monitoring routes dont work get rid of them"), and nothing the server serves reads the kill
// switch's losing streak or day loss from risk_kill_switch.get_kill_switch_status(), whose two source files
// (consec_losses.json, daily_pnl.json) no code writes. Source pins.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const read = (rel: string) => fs.readFileSync(path.join(here, rel), "utf8");
const bot = read("bot.ts");

test("the three routes are not registered; the cache inventory (operator plumbing) still is", () => {
  for (const name of ["overview", "tier-pnl", "drawdown"]) {
    assert.ok(!bot.includes(`"/api/monitoring/${name}"`), `/api/monitoring/${name} is back`);
  }
  assert.ok(bot.includes('app.get("/api/monitoring/caches", requireOwner,'));
});

test("no server source calls the status function whose figures are never written", () => {
  for (const f of fs.readdirSync(here).filter((n) => n.endsWith(".ts") && !n.endsWith(".test.ts"))) {
    const src = read(f);
    assert.ok(!/get_kill_switch_status\s*\(/.test(src), `${f} calls get_kill_switch_status()`);
    assert.ok(!/["'`]risk_status["'`]/.test(src), `${f} asks the daemon for risk_status`);
  }
  const daemon = read("../voltrade_daemon.py");
  assert.ok(!daemon.includes('"risk_status"') && !daemon.includes('"get_peak_equity"'), "the daemon still maps a method nothing calls");
});

test("the owner snapshot's kill-switch block is the real state (safety_limits_view), not the removed status", () => {
  const a = bot.indexOf('app.get("/api/system/snapshot"');
  assert.ok(a > 0);
  const route = bot.slice(a, bot.indexOf("\n  });", a));
  assert.ok(route.includes("from safety_limits_view import safety_limits") && route.includes("result['kill_switches'] = safety_limits()"));
});

test("the helpers only those routes used are gone with them", () => {
  assert.ok(!fs.existsSync(path.join(here, "peakEquityRpc.ts")));
  assert.ok(!/drawdownStatus/.test(read("drawdownGuard.ts")));
  // the two helpers the live kill switches use are untouched
  assert.ok(/export function evaluateDrawdown\(/.test(read("drawdownGuard.ts")) && /export function evaluateDailyPnl\(/.test(read("drawdownGuard.ts")));
});
