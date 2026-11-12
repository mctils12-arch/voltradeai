// [REPAIR] 2026-11-06: the boot RULES banner describes the live state.
// Run: npx tsx --test server/rulesBanner.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { killSwitchBannerLine, satelliteExposureBannerLine, minScoreBannerLine } from "./rulesBanner";

test("a latched kill switch is reported ON with its reason, never OFF", () => {
  const line = killSwitchBannerLine(true, "Max drawdown -10.2% hit", -20);
  assert.match(line, /^Kill switch: ON \(latched: Max drawdown -10\.2% hit\)/);
  assert.doesNotMatch(line, /OFF/);
  assert.match(killSwitchBannerLine(true, null, -20), /^Kill switch: ON \(latched\) —/);
  assert.match(killSwitchBannerLine(true, "   ", -20), /^Kill switch: ON \(latched\) —/);
});

test("an unlatched kill switch is reported OFF with the threshold it trips at", () => {
  assert.equal(killSwitchBannerLine(false, "stale reason", -20),
    "Kill switch: OFF (armed: trips at -20% from the equity peak; manual stop available)");
});

test("a long reason is capped so the banner line stays one readable row", () => {
  const line = killSwitchBannerLine(true, "x".repeat(500), -20);
  assert.ok(line.length < 320, `line is ${line.length} chars`);
});

test("the exposure line names the floor generically, not QQQ", () => {
  const line = satelliteExposureBannerLine(0.3);
  assert.equal(line, "Max active satellite exposure: 30% (the floor + VRP excluded) | Total portfolio cap: 95% (regime-adaptive)");
  assert.doesNotMatch(line, /QQQ/);
});

test("the score line reads the live threshold", () => {
  assert.equal(minScoreBannerLine(65), "Minimum score to trade: 65/100 combined edge score");
  assert.equal(minScoreBannerLine(70), "Minimum score to trade: 70/100 combined edge score");
});

test("bot.ts's banner uses the live-state lines, and the fixed strings are gone", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const src = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(src.includes("killSwitchBannerLine(state.killSwitch, killSwitchMeta.reason, state.maxDrawdownPct)"));
  assert.ok(src.includes("satelliteExposureBannerLine(MAX_TOTAL_EXPOSURE)"));
  assert.ok(src.includes("minScoreBannerLine(state.minScoreThreshold)"));
  assert.ok(!src.includes("Kill switch: OFF (manual emergency stop available)"));
  assert.ok(!src.includes("(QQQ floor + VRP excluded)"));
});
