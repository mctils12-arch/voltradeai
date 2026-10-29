import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { peakEquityFromRpc } from "./peakEquityRpc";

test("daemon shape (raw float) is read as the peak", () => {
  // voltrade_daemon.py maps get_peak_equity straight to the function's float.
  assert.equal(peakEquityFromRpc(111613.42), 111613.42);
});

test("subprocess shape ({peak}) is read as the peak", () => {
  assert.equal(peakEquityFromRpc({ peak: 111613.42 }), 111613.42);
});

test("missing / zero / garbage peaks are unknown, never the current equity", () => {
  assert.equal(peakEquityFromRpc(0), null, "get_peak_equity returns 0.0 when no peak file exists");
  assert.equal(peakEquityFromRpc({ peak: 0 }), null);
  assert.equal(peakEquityFromRpc(undefined), null);
  assert.equal(peakEquityFromRpc({ error: "boom" }), null);
  assert.equal(peakEquityFromRpc("abc"), null);
});

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

test("/api/monitoring/drawdown reads the peak through peakEquityFromRpc", () => {
  const start = bot.indexOf('app.get("/api/monitoring/drawdown"');
  const route = bot.slice(start, bot.indexOf('app.get("/api/monitoring/caches"', start));
  assert.ok(/peakEquityFromRpc\(peakCall\.result\)/.test(route));
  assert.ok(!/peakCall\.result\.peak \|\| equity/.test(route), "the shape-blind .peak read that fell back to equity must be gone");
});

test("/api/bot/status reports the enforced daily-loss limit, not the dead 5% config", () => {
  const start = bot.indexOf('app.get("/api/bot/status"');
  const route = bot.slice(start, bot.indexOf("});", start));
  assert.ok(!/dailyLossLimitPct:\s*5\b/.test(route));
  assert.ok(/dailyLossLimitPct:\s*Math\.abs\(DAILY_LOSS_LIMIT\)/.test(route));
});
