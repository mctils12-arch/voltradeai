// LIVENESS ALARM runtime half (Amendment 2) — the regression test that
// would have caught the production incident: the bot sat PAUSED and
// /api/health stayed "ok". Pure-module tests + a wiring pin on bot.ts.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  nextLiveness, marketHoursBetween, loopDark,
  LOOP_DARK_MARKET_HOURS, LOOP_DARK_WALLCLOCK_HOURS,
} from "./liveness";

const here = path.dirname(fileURLToPath(import.meta.url));

// Fixed timestamps (July 2026, EDT = UTC-4). Wed 2026-07-08.
const WED_10ET = Date.UTC(2026, 6, 8, 14, 0);   // Wed 10:00 ET — market open
const WED_1130ET = Date.UTC(2026, 6, 8, 15, 30);
const WED_13ET = Date.UTC(2026, 6, 8, 17, 0);
const FRI_1630ET = Date.UTC(2026, 6, 10, 20, 30); // Fri after close
const MON_09ET = Date.UTC(2026, 6, 13, 13, 0);    // Mon 9:00 ET — pre-open

test("heartbeat: active stamps now; fresh boot seeds now (no false alarm); inactive keeps the last stamp across restarts", () => {
  const now = WED_10ET;
  assert.equal(nextLiveness(null, true, now).lastActiveAt, now);
  assert.equal(nextLiveness(null, false, now).lastActiveAt, now, "fresh install seeds, never alarms instantly");
  const prev = { lastActiveAt: WED_10ET };
  assert.equal(nextLiveness(prev, false, WED_13ET), prev, "restart must NOT reset the darkness clock");
});

test("market hours: intraday counts, weekend does not", () => {
  const intraday = marketHoursBetween(WED_10ET, WED_13ET);
  assert.ok(Math.abs(intraday - 3) < 0.2, `Wed 10:00→13:00 ET should be ~3 market hours (got ${intraday})`);
  const weekend = marketHoursBetween(FRI_1630ET, MON_09ET);
  assert.ok(weekend < 0.1, `Fri 16:30 → Mon 9:00 ET spans no session (got ${weekend})`);
});

test("alarm: >2 market hours dark degrades; 1.5h does not; active never does", () => {
  const pausedAt = { lastActiveAt: WED_10ET };
  const dark = loopDark(pausedAt, false, WED_13ET); // 3 market hours
  assert.equal(dark.dark, true);
  assert.ok(dark.detail.includes("LIVENESS ALARM"), "detail must name the alarm");
  const ok = loopDark(pausedAt, false, WED_1130ET); // 1.5 market hours
  assert.equal(ok.dark, false);
  assert.equal(loopDark(pausedAt, true, WED_13ET).dark, false, "active loop is never dark");
});

test("alarm: weekend-long halt trips the 24h wall-clock ceiling before Monday", () => {
  const pausedFri = { lastActiveAt: FRI_1630ET };
  const monday = loopDark(pausedFri, false, MON_09ET);
  assert.ok(monday.wallHours > LOOP_DARK_WALLCLOCK_HOURS);
  assert.equal(monday.dark, true, "weekend halts must surface before the Monday open");
  // but a Saturday check ~14h in stays quiet — no market hours, under 24h wall
  const satNight = loopDark(pausedFri, false, FRI_1630ET + 14 * 3_600_000);
  assert.equal(satNight.dark, false);
});

test("wiring pinned: /api/health consumes the assessment and degrades on dark", () => {
  const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
  assert.ok(bot.includes("nextLiveness"), "bot.ts must maintain the liveness heartbeat");
  assert.ok(bot.includes("loopDark"), "bot.ts health check must assess loop darkness");
  assert.ok(/lv\.dark.*\n?.*degraded|degraded.*lv\.dark/.test(bot) || bot.includes('if (lv.dark) checks.status = "degraded"'),
    "a dark loop must degrade overall /api/health status");
  assert.equal(LOOP_DARK_MARKET_HOURS, 2, "approved threshold: 2 market hours");
  assert.equal(LOOP_DARK_WALLCLOCK_HOURS, 24, "approved ceiling: 24h wall-clock");
});
