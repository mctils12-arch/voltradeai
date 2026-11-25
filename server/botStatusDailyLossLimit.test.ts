// botStatusDailyLossLimit.test.ts — /api/bot/status reports the daily-loss limit the loop enforces (source pin).
// Moved unchanged on 2026-11-25 from peakEquityRpc.test.ts, whose other four tests covered a helper and a route
// (/api/monitoring/drawdown) that were removed that day.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

test("/api/bot/status reports the enforced daily-loss limit, not the dead 5% config", () => {
  const start = bot.indexOf('app.get("/api/bot/status"');
  const route = bot.slice(start, bot.indexOf("});", start));
  assert.ok(!/dailyLossLimitPct:\s*5\b/.test(route));
  assert.ok(/dailyLossLimitPct:\s*Math\.abs\(DAILY_LOSS_LIMIT\)/.test(route));
});
