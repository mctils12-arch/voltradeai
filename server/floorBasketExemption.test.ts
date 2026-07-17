// R-FIX 2026-07-17 (found via live /api/diag/audit query): the passive
// floor-basket + legacy leg ticker exemption in this file was hardcoded to
// {"QQQ", "SVXY", "SPY"} in three separate places and never updated when
// system_config.py's FLOOR_BASKET (SMH/KWEB/VXUS) shipped 2026-04-22, nor
// for DEFENSIVE_FLOOR_TICKER (GLD). Two effects, both confirmed live in
// production this session:
//   1. syncMonitoredPositions() let SMH/KWEB/VXUS/GLD into monitoredPositions,
//      so checkPositionOnTick's WS-driven active-exit logic (stop-loss/
//      take-profit/trailing-stop/time-stop) applied to them — /api/diag/audit
//      showed 5x erroneous "WS TIME STOP" sells of VXUS and 1x "WS STOP LOSS"
//      sell of SMH in a single session, each immediately re-bought by the
//      floor rebalancer (_manage_spy_floor) on its next cycle.
//   2. executeTrades()'s totalDeployed heat-cap counted floor-basket market
//      value as active-satellite-trade capital, inflating it toward
//      MAX_TOTAL_EXPOSURE and risking premature HEAT-CAP throttling of
//      legitimate new active trades.
// bot_engine.py's mirror of this list (FLOOR_AND_LEG_TICKERS) had the exact
// same staleness bug — see test_floor_basket_stops.py for the Python side.
// These tests pin the fix: a single shared FLOOR_AND_LEG_TICKERS constant,
// used at all three call sites, containing every live floor-basket member.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

const EXPECTED_FLOOR_TICKERS = ["QQQ", "SVXY", "SPY", "SMH", "KWEB", "VXUS", "GLD"];

test("a single FLOOR_AND_LEG_TICKERS constant exists and includes every live floor-basket member", () => {
  const declMatch = bot.match(/const FLOOR_AND_LEG_TICKERS = new Set\(\[([^\]]*)\]\);/);
  assert.ok(declMatch, "FLOOR_AND_LEG_TICKERS constant not found in bot.ts");
  const declared = declMatch![1].split(",").map((s) => s.trim().replace(/["']/g, "")).filter(Boolean);
  for (const ticker of EXPECTED_FLOOR_TICKERS) {
    assert.ok(
      declared.includes(ticker),
      `FLOOR_AND_LEG_TICKERS is missing "${ticker}" — a live floor-basket/defensive-floor ` +
      `ticker would be subjected to active stop-loss/take-profit/time-stop logic again`
    );
  }
});

test("no stale QQQ/SVXY/SPY-only managed-ticker literal remains at any call site", () => {
  // The exact pre-fix pattern (three independent `new Set(["QQQ", "SVXY", "SPY"])`
  // literals) must not reappear — every call site must reference the shared const.
  assert.ok(
    !/new Set\(\["QQQ",\s*"SVXY",\s*"SPY"\]\)/.test(bot),
    "found a re-introduced stale floor-ticker literal — use the shared FLOOR_AND_LEG_TICKERS constant instead"
  );
});

test("syncMonitoredPositions skips floor-basket tickers via the shared constant", () => {
  const start = bot.indexOf("async function syncMonitoredPositions()");
  assert.ok(start > 0, "syncMonitoredPositions() not found in bot.ts");
  const end = bot.indexOf("\n  async function", start + 1);
  const block = bot.slice(start, end > start ? end : start + 4000);
  assert.ok(
    /if \(FLOOR_AND_LEG_TICKERS\.has\(ticker\)\) continue;/.test(block),
    "syncMonitoredPositions must skip FLOOR_AND_LEG_TICKERS before adding a position to monitoredPositions"
  );
});

test("executeTrades excludes floor-basket tickers from the active-trade heat-cap total", () => {
  const start = bot.indexOf("async function executeTrades(");
  assert.ok(start > 0, "executeTrades() not found in bot.ts");
  const end = bot.indexOf("\n  // ── Tier 2", start + 1);
  const block = bot.slice(start, end > start ? end : start + 6000);
  assert.ok(
    /if \(FLOOR_AND_LEG_TICKERS\.has\(p\.symbol\)\) return sum;/.test(block),
    "executeTrades' totalDeployed heat-cap must exclude FLOOR_AND_LEG_TICKERS market value"
  );
});
