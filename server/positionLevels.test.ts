// shared/positionLevels.ts pinned by hand-worked numbers. STOCKS: the ladder
// the engine executes (shared/stockExitLadder.ts, the mirror of server/bot.ts
// checkPositionOnTick + syncMonitoredPositions; server/stockExitLadder.test.ts
// runs that code as written against it). Until 2026-11-12 (KNOWN BROKEN #50a)
// these tests pinned bot_engine.manage_positions' formulas instead — a regime
// scalar on every stop, a 10 % phase-1 clamp, a 60-minute and a 4-day
// suppression, a −20 % hard floor — which Tier 1 never executes. OPTIONS:
// options_manager.py, further down.
import { test } from "node:test";
import assert from "node:assert/strict";
import * as positionLevelsModule from "../shared/positionLevels";
import {
  positionLevels, optionLegLevels, regimeMults, pricePct, parseEngineTime, OPTIONS_PROFIT_TARGET_FRAC,
  type StopStateRow,
} from "../shared/positionLevels";
import {
  optionExitPlan, optionManagement,
  OPTIONS_DTE_EXIT_THRESHOLD, OPTIONS_DTE_CRITICAL, OPTIONS_LOSS_LIMIT_MULTIPLIER, OPTIONS_BOUGHT_LOSS_LIMIT_FRAC, OPTIONS_IC_MAX_LOSS_EXIT_FRAC,
  OPTIONS_GAMMA_THRESHOLD, OPTIONS_GAMMA_DTE, OPTIONS_ASSIGNMENT_DELTA, OPTIONS_ASSIGNMENT_DTE, OPTIONS_MIN_HOLD_MINUTES,
  OPTIONS_MULTI_LEG_STRATEGIES, OPTIONS_CREDIT_STRATEGIES, OPTIONS_DEBIT_STRATEGIES, CONVEXITY_HEDGE_STRATEGY, CONVEXITY_ROLL_DTE,
  OPTION_LONG_KILL_LOSS_PCT, OPTION_SHORT_KILL_LOSS_PCT,
  OPTIONS_PT_SCALE_OUT_GAIN, OPTIONS_PT_SCALE_OUT_FRACTION, OPTIONS_PT_TRAIL_ARM_GAIN, OPTIONS_PT_TRAIL_GIVEBACK,
} from "../shared/positionLevels";
import { readFileSync } from "node:fs";
import path from "node:path";

const NOW = Date.parse("2026-10-21T15:00:00Z");
const close = (a: number | null, b: number, eps = 1e-6) => { assert.ok(a !== null && Math.abs(a - b) < eps, `${a} ≠ ${b}`); };

test("regime multipliers mirror checkPositionOnTick's isBullish / isBearish branches", () => {
  assert.deepEqual(regimeMults("BULL"), { bullish: true, bearish: false, trailAtrMult: 3, scaleOut2R: 3 });
  assert.deepEqual(regimeMults("NEUTRAL_BULL"), { bullish: true, bearish: false, trailAtrMult: 3, scaleOut2R: 3 });
  assert.deepEqual(regimeMults("CAUTION"), { bullish: false, bearish: true, trailAtrMult: 1.5, scaleOut2R: 1.5 });
  assert.deepEqual(regimeMults("NEUTRAL"), { bullish: false, bearish: false, trailAtrMult: 2, scaleOut2R: 2 });
  assert.deepEqual(regimeMults(null), { bullish: false, bearish: false, trailAtrMult: 2, scaleOut2R: 2 });
});

test("phase 1 long: stop 2×ATR from entry (clamped 1.5–8), TP max(4, min(3×ATR, 15)) armed only after ⅔ out, thirds at +1R and the regime's 2R, the −25 % backstop, nothing suppressed", () => {
  const stop: StopStateRow = { phase: 1, initial_risk_pct: 5.0, current_stop_pct: 2.5, highest_pnl: 0.4, highest_price: 101, original_qty: 30, remaining_qty: 30, scales_completed: 0, breakeven_active: false, regime: "NEUTRAL", days_held: 5, entry_date: "2026-10-16", entry_timestamp: "2026-10-16T14:31:00.000000" };
  const lv = positionLevels({ entry: 100, current: 100.4, qty: 30, pnlPct: 0.4, stop, nowMs: NOW });
  assert.equal(lv.recorded, true);
  assert.equal(lv.side, "long");
  assert.equal(lv.phase, 1);
  close(lv.stopPct, 5.0); // max(1.5, min(2.5×2, 8))
  close(lv.stopPrice, 95.0);
  assert.equal(lv.stopFrom, "entry");
  assert.equal(lv.stopHit, false);
  close(lv.takeProfitPct, 7.5); // max(4, min(2.5×3, 15))
  close(lv.takeProfitPrice, 107.5);
  assert.equal(lv.takeProfitArmed, false);
  assert.equal(lv.scaleOuts.length, 2);
  close(lv.scaleOuts[0].price, 105.0); // +1R = +5 %
  assert.equal(lv.scaleOuts[0].done, false);
  assert.equal(lv.scaleOuts[1].rMultiple, 2); // NEUTRAL → 2R
  close(lv.scaleOuts[1].price, 110.0);
  // 2026-11-12: was HARD_STOP_PCT_DEFAULT (20) → 80.0, manage_positions' phase-1 hard stop, which nothing executes.
  // 25 is POS-KILL's STOCK_KILL_LOSS_PCT (server/stockExitLadder.test.ts ties the number to server/bot.ts).
  assert.equal(lv.hardFloorPct, 25);
  close(lv.hardFloorPrice, 75.0);
  // 2026-11-12: was `stopSuppressed === null` ("day 5, past the short-hold window") — the engine has no such window
  assert.equal("stopSuppressed" in lv, false);
  assert.equal(lv.timeStop.day, 7); // was TIME_STOP_DAYS_ENGINE, the same 7
  assert.equal(lv.timeStop.withinPct, 2);
  assert.equal(lv.timeStop.armed, false);
  assert.equal(lv.daysHeld, 5);
  close(lv.rMultiple, 0.08);
});

test("phase 2 long (peak ≥ 1R): stop trails 1.5×ATR from the P&L high-water mark, no regime scalar; first third done; TP still pending; the backstop stays", () => {
  // entry_date added 2026-11-12: the engine counts the time stop's days from it (a real row always carries it)
  const stop: StopStateRow = { phase: 2, initial_risk_pct: 4.0, current_stop_pct: 2.0, highest_pnl: 6.0, highest_price: 106, original_qty: 30, remaining_qty: 20, scales_completed: 1, breakeven_active: true, regime: "BEAR", days_held: 9, entry_date: "2026-10-12" };
  const lv = positionLevels({ entry: 100, current: 104, qty: 20, pnlPct: 4.0, stop, nowMs: NOW });
  assert.equal(lv.phase, 2); // peak 6 / 4 = 1.5R
  // 2026-11-12: was 3.6 = max(1.5, 2.0 × 1.5 × 1.2), with BEAR's STOP_SCALAR — the engine applies none
  close(lv.stopPct, 3.0); // max(1.5, 2.0 × 1.5)
  // 2026-11-12: was 102.4
  close(lv.stopPrice, 103.0); // entry × (1 + (6.0 − 3.0) / 100) — above entry, so breakeven does not raise it
  assert.equal(lv.stopFrom, "peak_pnl");
  assert.equal(lv.stopHit, false); // 2.0 off the peak, under the 3.0 trail
  assert.equal(lv.breakevenActive, true);
  assert.equal(lv.scaleOuts[0].done, true);
  assert.equal(lv.scaleOuts[1].rMultiple, 1.5); // BEAR
  close(lv.scaleOuts[1].price, 106.0);
  // 2026-11-12: was null ("no hard floor" in phase 2) — POS-KILL judges a stock in every phase
  close(lv.hardFloorPrice, 75.0);
  assert.equal(lv.takeProfitArmed, false);
  close(lv.takeProfitPrice, 106.0); // max(4, 6) = 6 %, under +2R (8 %)
  assert.equal(lv.timeStop.armed, true); // held 9 ≥ 7
  assert.equal(lv.daysHeld, 9);
});

test("breakeven raises a trailing stop that sits below entry; the live price can lift the peak (and the phase) past the stored one", () => {
  const stop: StopStateRow = { phase: 1, initial_risk_pct: 4.0, current_stop_pct: 2.0, highest_pnl: 0.9, highest_price: 100.9, original_qty: 30, remaining_qty: 20, scales_completed: 1, breakeven_active: true, regime: "NEUTRAL", days_held: 3 };
  // live P&L +4.2 % → peak 4.2 / 4 = 1.05R → phase 2; stop = 4.2 − max(1.5, 3.0) = +1.2 % → 101.2 (above entry)
  const live = positionLevels({ entry: 100, current: 104.2, qty: 20, pnlPct: 4.2, stop, nowMs: NOW });
  assert.equal(live.phase, 2);
  close(live.stopPrice, 101.2);
  // live P&L +2.0 % with a stored peak of 4.1 % → stop = 4.1 − 3.0 = +1.1 → 101.1; breakeven irrelevant
  const held = positionLevels({ entry: 100, current: 102, qty: 20, pnlPct: 2.0, stop: { ...stop, highest_pnl: 4.1 }, nowMs: NOW });
  close(held.stopPrice, 101.1);
  // stored peak 4.1 with ATR 3 → stop width max(1.5, 4.5) = 4.5 → 4.1 − 4.5 = −0.4 % → breakeven lifts it to entry
  const be = positionLevels({ entry: 100, current: 102, qty: 20, pnlPct: 2.0, stop: { ...stop, highest_pnl: 4.1, current_stop_pct: 3.0 }, nowMs: NOW });
  close(be.stopPrice, 100.0);
  assert.equal(be.stopFrom, "breakeven");
  assert.equal(be.stopHit, false);
  // under entry with breakeven on: the engine's breakeven test is met
  const under = positionLevels({ entry: 100, current: 99.9, qty: 20, pnlPct: -0.1, stop: { ...stop, highest_pnl: 4.1, current_stop_pct: 3.0 }, nowMs: NOW });
  assert.equal(under.stopHit, true);
  // the stored phase stands when it is ahead of what the rounded peak gives (the sync takes it as written)
  assert.equal(positionLevels({ entry: 100, current: 103, qty: 20, pnlPct: 3.0, stop: { ...stop, phase: 2, highest_pnl: 3.99 }, nowMs: NOW }).phase, 2);
});

test("phase 4 trails the highest PRICE by trail×ATR (BULL → 3×), no regime scalar", () => {
  const stop: StopStateRow = { phase: 4, initial_risk_pct: 3.0, current_stop_pct: 2.0, highest_pnl: 12.0, highest_price: 112, original_qty: 30, remaining_qty: 10, scales_completed: 2, breakeven_active: true, regime: "BULL", days_held: 20 };
  const lv = positionLevels({ entry: 100, current: 110, qty: 10, pnlPct: 10, stop, nowMs: NOW });
  assert.equal(lv.phase, 4); // 12 / 3 = 4R
  // 2026-11-12: was 5.4 = 2.0 × 3 × 0.9, with BULL's STOP_SCALAR — the engine applies none
  close(lv.stopPct, 6.0); // 2.0 × 3
  // 2026-11-12: was 112 × (1 − 0.054)
  close(lv.stopPrice, 112 * (1 - 0.06));
  assert.equal(lv.stopFrom, "peak_price");
  assert.equal(lv.takeProfitPct, null); // no fixed target in phase 3+
  assert.equal(lv.takeProfitArmed, true);
  close(lv.hardFloorPrice, 75.0);
});

// 2026-11-12: this test was "suppression: the first hour only lets a −10 % loss out; days 0–3 in phase 1 only a
// −15 % loss". It pinned manage_positions' 60-minute minimum hold and 4-day short-hold grace (stopSuppressed.until
// "60min" / "day4", overridePct 10 / 15, minutesLeft 40, daysLeft 2). The engine has neither: its phase-1 stop is
// live from the first tick. Same three positions, the engine's answer.
test("no suppression: 20 minutes in, or on day 2, the phase-1 stop is live and a price past it is a hit", () => {
  const base: StopStateRow = { phase: 1, initial_risk_pct: 5.0, current_stop_pct: 2.5, highest_pnl: 0, original_qty: 10, remaining_qty: 10, scales_completed: 0, regime: "NEUTRAL" };
  const fresh = positionLevels({ entry: 100, current: 99, qty: 10, pnlPct: -1, stop: { ...base, days_held: 0, entry_date: "2026-10-21", entry_timestamp: "2026-10-21T14:40:00.000000" }, nowMs: NOW });
  assert.equal("stopSuppressed" in fresh, false);
  close(fresh.stopPrice, 95.0);
  assert.equal(fresh.stopHit, false);
  close(fresh.heldMinutes, 20);
  const young = positionLevels({ entry: 100, current: 99, qty: 10, pnlPct: -1, stop: { ...base, days_held: 2, entry_date: "2026-10-19", entry_timestamp: "2026-10-19T14:40:00.000000" }, nowMs: NOW });
  assert.equal("stopSuppressed" in young, false);
  close(young.stopPrice, 95.0);
  assert.equal(young.daysHeld, 2);
  // −6 % twenty minutes after entry: manage_positions would wait for −10 %; the engine's stop test is already met
  const freshPast = positionLevels({ entry: 100, current: 94, qty: 10, pnlPct: -6, stop: { ...base, days_held: 0, entry_date: "2026-10-21", entry_timestamp: "2026-10-21T14:40:00.000000" }, nowMs: NOW });
  assert.equal(freshPast.stopHit, true);
  close(freshPast.stopPrice, 95.0);
  // −7 % on day 2: manage_positions would wait for −15 % or day 4
  const youngPast = positionLevels({ entry: 100, current: 93, qty: 10, pnlPct: -7, stop: { ...base, days_held: 2, entry_date: "2026-10-19", entry_timestamp: "2026-10-19T14:40:00.000000" }, nowMs: NOW });
  assert.equal(youngPast.stopHit, true);
  // exactly at the stop counts (the engine's test is <=); a hair inside does not
  assert.equal(positionLevels({ entry: 100, current: 95, qty: 10, pnlPct: -5, stop: base, nowMs: NOW }).stopHit, true);
  assert.equal(positionLevels({ entry: 100, current: 95.01, qty: 10, pnlPct: -4.99, stop: base, nowMs: NOW }).stopHit, false);
  // phase 2 was never suppressed, and still is not
  const p2 = positionLevels({ entry: 100, current: 105, qty: 10, pnlPct: 5, stop: { ...base, highest_pnl: 5.5, days_held: 2, entry_timestamp: "2026-10-19T14:40:00.000000" }, nowMs: NOW });
  assert.equal(p2.phase, 2);
  assert.equal("stopSuppressed" in p2, false);
});

test("KNOWN BROKEN #50a's example: a 7 % ATR in BULL stops at −8 % on the first tick past it (the Book said −10 %, waiting until day 4)", () => {
  const stop: StopStateRow = { phase: 1, initial_risk_pct: 14.0, current_stop_pct: 7.0, highest_pnl: 0.3, highest_price: 20.06, original_qty: 150, remaining_qty: 150, scales_completed: 0, breakeven_active: false, regime: "BULL", days_held: 0, entry_date: "2026-10-21", entry_timestamp: "2026-10-21T14:40:00.000000" };
  const lv = positionLevels({ entry: 20, current: 19.5, qty: 150, pnlPct: -2.5, stop, nowMs: NOW });
  close(lv.stopPct, 8.0); // max(1.5, min(7 × 2, 8)) — not min(7 × 2 × 0.9, 10)
  close(lv.stopPrice, 18.4);
  assert.equal(lv.stopFrom, "entry");
  assert.equal(lv.stopHit, false);
  assert.equal("stopSuppressed" in lv, false);
  close(lv.hardFloorPrice, 15.0); // −25 %, not −20 % (16.00)
  close(lv.riskPct, 14.0);
  close(lv.scaleOuts[0].price, 22.8); // +1R
  assert.equal(lv.scaleOuts[1].rMultiple, 3); // BULL
  close(lv.scaleOuts[1].price, 28.4); // +3R = +42 %
  close(lv.takeProfitPrice, 23.0); // max(4, min(21, 15)) = 15 %
  // the first tick past −8 %: the engine sells; nothing waits
  const past = positionLevels({ entry: 20, current: 18.3, qty: 150, pnlPct: -8.5, stop, nowMs: NOW });
  assert.equal(past.stopHit, true);
  close(past.stopPrice, 18.4);
});

test("short side: levels mirror below / above entry; a single-share lot cannot scale out, so it has no target", () => {
  const stop: StopStateRow = { phase: 1, initial_risk_pct: 4.0, current_stop_pct: 2.0, highest_pnl: 0, original_qty: 1, remaining_qty: 1, scales_completed: 0, regime: "NEUTRAL", days_held: 6 };
  const lv = positionLevels({ entry: 50, current: 49, qty: -1, pnlPct: 2, stop, nowMs: NOW });
  assert.equal(lv.side, "short");
  close(lv.stopPrice, 52.0); // +4 % against a short
  // 2026-11-12: was 60.0 ("engine default 20 %")
  close(lv.hardFloorPrice, 62.5); // the −25 % backstop, above a short's entry
  assert.equal(lv.scaleOuts[0].applicable, false);
  // 2026-11-12: was takeProfitPrice 47.0 — the target fires only after both thirds, which one share never sells
  assert.equal(lv.takeProfitPrice, null);
  assert.equal(lv.takeProfitPct, null);
  // a short lot that can split keeps the target, below entry
  const lot = positionLevels({ entry: 50, current: 49, qty: -30, pnlPct: 2, stop: { ...stop, original_qty: 30, remaining_qty: 30 }, nowMs: NOW });
  close(lot.takeProfitPrice, 47.0); // max(4, 6) = 6 % → 50 × 0.94
  assert.equal(lot.scaleOuts[0].applicable, true);
  close(lot.scaleOuts[0].price, 48.0);
  assert.equal(pricePct(50, -4, "short"), 52);
});

test("a lot of two cannot sell a third either (floor(2 / 3) = 0): no scale-out, no target; three shares can", () => {
  const stop: StopStateRow = { phase: 1, initial_risk_pct: 5.0, current_stop_pct: 2.5, highest_pnl: 0, scales_completed: 0, regime: "NEUTRAL" };
  const two = positionLevels({ entry: 100, current: 100, qty: 2, pnlPct: 0, stop, nowMs: NOW });
  assert.deepEqual(two.scaleOuts.map((s) => s.applicable), [false, false]);
  assert.equal(two.takeProfitPrice, null);
  assert.deepEqual([two.originalQty, two.remainingQty], [2, 2]);
  const three = positionLevels({ entry: 100, current: 100, qty: 3, pnlPct: 0, stop, nowMs: NOW });
  assert.deepEqual(three.scaleOuts.map((s) => s.applicable), [true, true]);
  close(three.takeProfitPrice, 107.5);
});

test("the time stop counts days from the entry date, as the engine does; days_held alone never arms it", () => {
  const base: StopStateRow = { phase: 1, initial_risk_pct: 5.0, current_stop_pct: 2.5, highest_pnl: 0.4, scales_completed: 0, regime: "NEUTRAL" };
  const at = (row: StopStateRow) => positionLevels({ entry: 100, current: 100.4, qty: 30, pnlPct: 0.4, stop: row, nowMs: NOW });
  assert.deepEqual([at({ ...base, entry_date: "2026-10-14" }).daysHeld, at({ ...base, entry_date: "2026-10-14" }).timeStop.armed], [7, true]);
  assert.deepEqual([at({ ...base, entry_date: "2026-10-15" }).daysHeld, at({ ...base, entry_date: "2026-10-15" }).timeStop.armed], [6, false]);
  // a stale days_held (manage_positions stopped ticking) does not hold the count back
  assert.deepEqual([at({ ...base, entry_date: "2026-10-12", days_held: 3 }).daysHeld, at({ ...base, entry_date: "2026-10-12", days_held: 3 }).timeStop.armed], [9, true]);
  // no entry date: the engine counts from today, so its time stop cannot arm; the row's days_held still names the age
  assert.deepEqual([at({ ...base, days_held: 9 }).daysHeld, at({ ...base, days_held: 9 }).timeStop.armed], [9, false]);
});

test("no stop state → not recorded: no stop, target, thirds or backstop, phase 1 by default", () => {
  const lv = positionLevels({ entry: 100, current: 101, qty: 5, pnlPct: 1, stop: null, nowMs: NOW });
  assert.equal(lv.recorded, false);
  assert.equal(lv.stopPrice, null);
  assert.equal(lv.stopHit, false);
  assert.equal(lv.takeProfitPrice, null);
  assert.deepEqual(lv.scaleOuts, []);
  assert.equal(lv.hardFloorPrice, null);
  assert.equal(lv.phase, 1);
  // a row without an ATR is not recorded either (bot.ts used to invent 2.5 %)
  assert.equal(positionLevels({ entry: 100, current: 101, qty: 5, pnlPct: 1, stop: { phase: 2 }, nowMs: NOW }).recorded, false);
  // nor is a row whose ATR is zero: the engine falls back to its 2 % default there, and the Book draws nothing
  // rather than the 1.5 % stop a zero ATR would give
  const zero = positionLevels({ entry: 100, current: 101, qty: 5, pnlPct: 1, stop: { phase: 1, current_stop_pct: 0, initial_risk_pct: 0 }, nowMs: NOW });
  assert.equal(zero.recorded, false);
  assert.equal(zero.stopPrice, null);
});

test("engine timestamps are the container's clock (UTC), never the reader's local zone", () => {
  assert.equal(parseEngineTime("2026-10-21T14:40:00.000000"), Date.parse("2026-10-21T14:40:00.000Z"));
  assert.equal(parseEngineTime("2026-10-21T14:40:00Z"), Date.parse("2026-10-21T14:40:00Z"));
  assert.equal(parseEngineTime(""), null);
  assert.equal(parseEngineTime("not a time"), null);
});

test("option legs: breakeven at expiry and the keep-the-premium side, from the leg's own strike and premium", () => {
  const sp = optionLegLevels({ optionType: "PUT", strike: 55, qty: -1, premium: 2.47 });
  assert.ok(sp);
  assert.equal(sp.side, "short");
  close(sp.breakevenAtExpiry, 52.53);
  assert.equal(sp.maxProfitZone, "above_strike");
  assert.equal(sp.profitTargetFrac, OPTIONS_PROFIT_TARGET_FRAC);
  const sc = optionLegLevels({ optionType: "CALL", strike: 30, qty: -2, premium: 0.8 });
  close(sc!.breakevenAtExpiry, 30.8);
  assert.equal(sc!.maxProfitZone, "below_strike");
  const lc = optionLegLevels({ optionType: "CALL", strike: 100, qty: 1, premium: 3 });
  close(lc!.breakevenAtExpiry, 103);
  assert.equal(lc!.maxProfitZone, null);
  assert.equal(lc!.profitTargetFrac, null, "a bought leg has no manager profit target — the chart must not claim one");
  assert.equal(optionLegLevels({ optionType: "PUT", strike: 0, qty: -1, premium: 1 }), null);
});

// 2026-11-12: this test was "the regime's scalar is picked from BASE_CONFIG; …" — six assertions on
// levelConfigFromBaseConfig (STOP_SCALAR_<regime>, POSITION_HARD_STOP_PCT / _ENABLED → LevelConfig). The function is
// deleted: the ladder the engine executes reads no config, so there is nothing for the Book to resolve.
test("the stock ladder reads no config: a stop scalar or a hard-stop setting cannot move a level, and the route sends none", () => {
  const stop: StopStateRow = { phase: 2, initial_risk_pct: 4.0, current_stop_pct: 2.0, highest_pnl: 6.0, highest_price: 106, original_qty: 30, remaining_qty: 20, scales_completed: 1, breakeven_active: true, regime: "BEAR", days_held: 9, entry_date: "2026-10-12" };
  const plain = { entry: 100, current: 104, qty: 20, pnlPct: 4.0, stop, nowMs: NOW };
  // what /api/bot/positions passed until 2026-11-12 (BEAR's scalar 1.2; and with the hard stop switched off)
  const withScalar = { ...plain, config: { stopScalar: 1.2, hardStopPct: 20, hardStopEnabled: true } };
  const withFloorOff = { ...plain, config: { stopScalar: 0.7, hardStopPct: 12, hardStopEnabled: false } };
  assert.deepEqual(positionLevels(withScalar), positionLevels(plain));
  assert.deepEqual(positionLevels(withFloorOff), positionLevels(plain));
  close(positionLevels(withScalar).stopPct, 3.0);
  close(positionLevels(withFloorOff).hardFloorPrice, 75.0);
  assert.equal("levelConfigFromBaseConfig" in positionLevelsModule, false, "the config resolver is gone");
  for (const gone of ["HARD_STOP_PCT_DEFAULT", "SHORT_HOLD_DAYS", "SHORT_HOLD_DEEP_LOSS_PCT", "MIN_HOLD_MINUTES", "MIN_HOLD_CATASTROPHE_PCT"]) {
    assert.equal(gone in positionLevelsModule, false, `${gone}: a manage_positions constant nothing executes`);
  }
  const here = path.dirname(new URL(import.meta.url).pathname);
  const bot = readFileSync(path.join(here, "bot.ts"), "utf8");
  const body = bot.slice(bot.indexOf("async function positionsBody(): Promise<unknown[]>"), bot.indexOf('app.get("/api/bot/positions"'));
  assert.ok(body.length > 2000, "positionsBody not found");
  assert.ok(body.includes("const levels = positionLevels({ entry, current, qty: parseFloat(p.qty), pnlPct: Number.isFinite(pnlPct) ? pnlPct : null, stop: ss, nowMs: Date.now() });"), "the route passes the stop state and the position, nothing else");
  assert.ok(!/STOP_SCALAR|POSITION_HARD_STOP|levelConfigFromBaseConfig/.test(body.replace(/^\s*\/\/.*$/gm, "")), "the route no longer reads manage_positions' stop scalar or hard stop");
  assert.ok(body.includes("levels: isOption ? null : levels,"), "share lots carry the ladder; option legs do not");
  const shared = readFileSync(path.join(here, "..", "shared", "positionLevels.ts"), "utf8");
  assert.ok(shared.includes('} from "./stockExitLadder";'), "the stock half is the executed ladder's module");
  assert.ok(!/stopScalar|stopSuppressed|SHORT_HOLD|hardStopEnabled/.test(shared), "no manage_positions-only rule is left in the Book's levels");
});

// ── option exits: the manager's own rules (options_manager.py, read 2026-11-02) ──
const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const pyNum = (src: string, re: RegExp): number => { const m = re.exec(src); assert.ok(m, `pattern not found: ${re}`); return Number(m![1]); };
const pyStrings = (src: string, re: RegExp): string[] => { const m = re.exec(src); assert.ok(m, `pattern not found: ${re}`); return [...m![1].matchAll(/"([a-z_]+)"/g)].map((x) => x[1]); };

test("the option-exit mirrors equal the engine's own constants (options_manager.py, bot_engine.py, server/bot.ts)", () => {
  const om = readFileSync(path.join(REPO, "options_manager.py"), "utf8");
  assert.equal(pyNum(om, /^DTE_EXIT_THRESHOLD = (\d+)/m), OPTIONS_DTE_EXIT_THRESHOLD);
  assert.equal(pyNum(om, /^DTE_CRITICAL = (\d+)/m), OPTIONS_DTE_CRITICAL);
  assert.equal(pyNum(om, /^PROFIT_TARGET_PCT = ([\d.]+)/m), OPTIONS_PROFIT_TARGET_FRAC);
  assert.equal(pyNum(om, /^LOSS_LIMIT_MULTIPLIER = ([\d.]+)/m), OPTIONS_LOSS_LIMIT_MULTIPLIER);
  assert.equal(pyNum(om, /^IC_MAX_LOSS_EXIT_PCT = ([\d.]+)/m), OPTIONS_IC_MAX_LOSS_EXIT_FRAC);
  assert.equal(pyNum(om, /^GAMMA_THRESHOLD = ([\d.]+)/m), OPTIONS_GAMMA_THRESHOLD);
  assert.equal(pyNum(om, /^ASSIGNMENT_DELTA_THRESHOLD = ([\d.]+)/m), OPTIONS_ASSIGNMENT_DELTA);
  assert.equal(pyNum(om, /^MIN_HOLD_MINUTES = (\d+)/m), OPTIONS_MIN_HOLD_MINUTES);
  // the bought option's loss limit is a literal in the standalone path, and the debit group rule the same number
  assert.equal(pyNum(om, /# Bought option: close if lost more than 50% of premium paid[\s\S]*?if loss_pct >= ([\d.]+):/), OPTIONS_BOUGHT_LOSS_LIMIT_FRAC);
  assert.equal(-pyNum(om, /if pnl_of_debit <= (-[\d.]+):\s*# Lost 50% of debit/), OPTIONS_BOUGHT_LOSS_LIMIT_FRAC);
  // the group loss limits are measured against the NET premium (2026-11-04): the denominators the mirror's words name
  assert.ok(/pnl_of_credit = total_unrealized_pnl \/ net_credit\n\s*if pnl_of_credit <= -LOSS_LIMIT_MULTIPLIER:/.test(om), "credit group stop: loss vs the net credit");
  assert.ok(/pnl_of_debit = total_unrealized_pnl \/ net_debit\n/.test(om), "debit group stop: loss vs the net debit");
  assert.equal(pyNum(om, /abs\(current_gamma\) > GAMMA_THRESHOLD and dte <= (\d+)/), OPTIONS_GAMMA_DTE);
  assert.equal(pyNum(om, /abs\(current_delta\) > ASSIGNMENT_DELTA_THRESHOLD and\s*\n\s*dte <= (\d+)\)/), OPTIONS_ASSIGNMENT_DTE);
  assert.deepEqual(pyStrings(om, /^MULTI_LEG_STRATEGIES = \(([\s\S]*?)\)/m).sort(), [...OPTIONS_MULTI_LEG_STRATEGIES].sort());
  assert.deepEqual(pyStrings(om, /is_credit_strategy = strategy in \(([\s\S]*?)\)/).sort(), [...OPTIONS_CREDIT_STRATEGIES].sort());
  assert.deepEqual(pyStrings(om, /is_debit_strategy = strategy in \(([^)]*)\)/).sort(), [...OPTIONS_DEBIT_STRATEGIES].sort());
  const be = readFileSync(path.join(REPO, "bot_engine.py"), "utf8");
  assert.equal(pyNum(be, /convexity_cfg\.get\("roll_dte", (\d+)\)/), CONVEXITY_ROLL_DTE);
  assert.ok(be.includes(`'strategy': '${CONVEXITY_HEDGE_STRATEGY}'`), "the overlay marks its puts with this strategy name");
  // 2026-11-06 BOUGHT-OPTION PROFIT TAKING: system_config's OPTIONS_PROFIT_TAKING and the manager's defaults
  const sc = readFileSync(path.join(REPO, "system_config.py"), "utf8");
  const block = /"OPTIONS_PROFIT_TAKING": \{([\s\S]*?)\}/.exec(sc);
  assert.ok(block, "system_config OPTIONS_PROFIT_TAKING");
  const defaults = /^PROFIT_TAKING_DEFAULTS = \{([\s\S]*?)\}/m.exec(om);
  assert.ok(defaults, "options_manager PROFIT_TAKING_DEFAULTS");
  for (const src of [block![1], defaults![1]]) {
    assert.equal(pyNum(src, /"scale_out_gain": ([\d.]+)/), OPTIONS_PT_SCALE_OUT_GAIN);
    assert.equal(pyNum(src, /"scale_out_fraction": ([\d.]+)/), OPTIONS_PT_SCALE_OUT_FRACTION);
    assert.equal(pyNum(src, /"trail_arm_gain": ([\d.]+)/), OPTIONS_PT_TRAIL_ARM_GAIN);
    assert.equal(pyNum(src, /"trail_giveback": ([\d.]+)/), OPTIONS_PT_TRAIL_GIVEBACK);
  }
  const bot = readFileSync(path.join(REPO, "server", "bot.ts"), "utf8");
  assert.equal(pyNum(bot, /const OPTION_LONG_KILL_LOSS_PCT = (-[\d.]+);/), OPTION_LONG_KILL_LOSS_PCT);
  assert.equal(pyNum(bot, /const OPTION_SHORT_KILL_LOSS_PCT = (-[\d.]+);/), OPTION_SHORT_KILL_LOSS_PCT);
});

test("management: multi-leg strategies group, the convexity hedge is the overlay's, everything else (an unrecorded strategy included) is a standalone leg", () => {
  assert.equal(optionManagement("bull_put_credit_spread"), "group");
  assert.equal(optionManagement("qqq_iron_condor"), "group");
  assert.equal(optionManagement("convexity_hedge"), "hedge");
  assert.equal(optionManagement("sell_cash_secured_put"), "single");
  assert.equal(optionManagement("covered_call"), "single");
  assert.equal(optionManagement("unknown"), "single");
  assert.equal(optionManagement(null), "single");
});

// 2026-11-06 (BOUGHT-OPTION PROFIT TAKING): this pin was "no profit target … never a partial exit". It now
// holds with profit taking switched off (profitTakingOn: false, the rollback); on, the next test pins it.
test("a bought single leg, profit taking off: no profit target; close at 21 DTE, −50% of premium, 5 DTE force, gamma inside 30 DTE, the Tier-1 −75% floor; never a partial exit", () => {
  const plan = optionExitPlan({ qty: 1, entryPrice: 3.2, currentPrice: 12.8, daysToExpiry: 47, strategy: null, profitTakingOn: false });
  assert.equal(plan.side, "long");
  assert.equal(plan.management, "single");
  assert.equal(plan.strategy, null);
  assert.equal(plan.profitTarget, null);
  assert.equal(plan.partial, false);
  assert.equal(plan.minHoldMinutes, 60);
  assert.deepEqual(plan.rules.map((r) => r.key), ["dte_critical", "dte_exit", "loss_limit", "gamma", "backstop"]);
  const by = Object.fromEntries(plan.rules.map((r) => [r.key, r]));
  assert.deepEqual(by.dte_critical, { key: "dte_critical", price: null, inDays: 42, hit: false });
  assert.deepEqual(by.dte_exit, { key: "dte_exit", price: null, inDays: 26, hit: false });
  close(by.loss_limit.price, 1.6); assert.equal(by.loss_limit.hit, false);
  assert.deepEqual(by.gamma, { key: "gamma", price: null, inDays: 17, hit: false });
  close(by.backstop.price, 0.8); assert.equal(by.backstop.hit, false);
  // the same leg after a collapse: the −50% stop and the floor both read hit; "unknown" reads as not recorded
  const down = optionExitPlan({ qty: 1, entryPrice: 3.2, currentPrice: 0.7, daysToExpiry: 20, strategy: "unknown", profitTakingOn: false });
  assert.equal(down.strategy, null);
  const d = Object.fromEntries(down.rules.map((r) => [r.key, r]));
  assert.equal(d.loss_limit.hit, true);
  assert.equal(d.backstop.hit, true);
  assert.equal(d.dte_exit.hit, true); assert.equal(d.dte_exit.inDays, -1);
  assert.equal(d.dte_critical.hit, false);
  // a leg the manager blocks (it may cover a sold option of its type) reads the same
  assert.deepEqual(optionExitPlan({ qty: 2, entryPrice: 3.2, currentPrice: 12.8, daysToExpiry: 47, strategy: null, scaleOut: "blocked" }).rules.map((r) => r.key),
    ["dte_critical", "dte_exit", "loss_limit", "gamma", "backstop"]);
});

test("a bought single leg, profit taking on (2026-11-06): half at +100% of the premium, the rest on a 35% trail off the peak once +50%, both before the −50% stop", () => {
  const two = optionExitPlan({ qty: 2, entryPrice: 3.2, currentPrice: 5.0, daysToExpiry: 47, strategy: "buy_call" });
  assert.deepEqual(two.rules.map((r) => r.key), ["dte_critical", "dte_exit", "scale_out", "trail", "loss_limit", "gamma", "backstop"]);
  const by = Object.fromEntries(two.rules.map((r) => [r.key, r]));
  assert.deepEqual(by.scale_out, { key: "scale_out", price: 6.4, inDays: null, hit: false });
  assert.deepEqual({ ...by.trail, price: null }, { key: "trail", price: null, inDays: null, hit: false, state: "not_armed" });
  close(by.trail.price, 4.8); // no peak recorded: the trail arms at +50% ($4.80)
  assert.equal(two.partial, true);
  assert.ok(two.profitTarget === by.scale_out);
  // up 300% with the manager's peak at 4.1× the premium: the half is due, the trail armed at 4.1 × 0.65 × $3.20
  const up = optionExitPlan({ qty: 2, entryPrice: 3.2, currentPrice: 12.8, daysToExpiry: 47, strategy: "buy_call", peakRatio: 4.1 });
  const u = Object.fromEntries(up.rules.map((r) => [r.key, r]));
  assert.equal(u.scale_out.hit, true);
  assert.equal(u.trail.state, "armed"); close(u.trail.price, 3.2 * 4.1 * 0.65); assert.equal(u.trail.hit, false);
  // fell back under the trail: due
  const back = Object.fromEntries(optionExitPlan({ qty: 1, entryPrice: 3.2, currentPrice: 8.0, daysToExpiry: 47, strategy: "buy_call", peakRatio: 4.1, scaleOut: "filled" }).rules.map((r) => [r.key, r]));
  assert.equal(back.trail.hit, true);
  assert.deepEqual(back.scale_out, { key: "scale_out", price: 6.4, inDays: null, hit: false, state: "done" }, "the half already sold");
  // a lot of 1 cannot be halved: the trail is its profit rule, no partial exit
  const one = optionExitPlan({ qty: 1, entryPrice: 3.2, currentPrice: 12.8, daysToExpiry: 47, strategy: null });
  const o = Object.fromEntries(one.rules.map((r) => [r.key, r]));
  assert.equal(o.scale_out.state, "not_possible"); assert.equal(o.scale_out.hit, false);
  assert.equal(one.partial, false);
  assert.ok(one.profitTarget === o.trail);
  // sold legs and the hedge are untouched by profit taking
  assert.ok(!optionExitPlan({ qty: -2, entryPrice: 2.47, currentPrice: 1.0, daysToExpiry: 30, strategy: "sell_cash_secured_put" }).rules.some((r) => r.key === "scale_out" || r.key === "trail"));
  assert.ok(!optionExitPlan({ qty: 3, entryPrice: 1.5, currentPrice: 4.0, daysToExpiry: 60, strategy: "convexity_hedge" }).rules.some((r) => r.key === "scale_out" || r.key === "trail"));
});

test("a sold single leg: 50% of the credit kept is the target, 2× the credit the stop, roll-else-close at 21 DTE, deep-ITM roll inside 10 DTE, the −150% floor", () => {
  const plan = optionExitPlan({ qty: -1, entryPrice: 2.47, currentPrice: 2.67, daysToExpiry: 30, strategy: "sell_cash_secured_put" });
  assert.equal(plan.side, "short");
  assert.equal(plan.management, "single");
  assert.deepEqual(plan.rules.map((r) => r.key), ["dte_critical", "assignment", "dte_exit", "profit_target", "loss_limit", "gamma", "backstop"]);
  const by = Object.fromEntries(plan.rules.map((r) => [r.key, r]));
  assert.ok(plan.profitTarget && plan.profitTarget === by.profit_target);
  close(by.profit_target.price, 1.235); assert.equal(by.profit_target.hit, false);
  close(by.loss_limit.price, 4.94); assert.equal(by.loss_limit.hit, false);
  assert.equal(by.assignment.inDays, 20);
  assert.equal(by.dte_exit.inDays, 9);
  close(by.backstop.price, 6.175);
  const won = optionExitPlan({ qty: -2, entryPrice: 2.47, currentPrice: 1.1, daysToExpiry: 30, strategy: "csp_normal_market" });
  assert.equal(Object.fromEntries(won.rules.map((r) => [r.key, r])).profit_target.hit, true);
  const blown = optionExitPlan({ qty: -1, entryPrice: 2.47, currentPrice: 7.0, daysToExpiry: 30, strategy: "covered_call" });
  const b = Object.fromEntries(blown.rules.map((r) => [r.key, r]));
  assert.equal(b.loss_limit.hit, true); assert.equal(b.backstop.hit, true); assert.equal(b.profit_target.hit, false);
});

test("a multi-leg strategy is judged on its combined P&L (no per-leg prices): credit → 50% of net credit target, 2× stop, 21 DTE; condors add the half-max-loss stop; debit → −50% of the debit", () => {
  const spread = optionExitPlan({ qty: -1, entryPrice: 4.1, currentPrice: 3.2, daysToExpiry: 18, strategy: "bull_put_credit_spread", legsInGroup: 2 });
  assert.equal(spread.management, "group");
  assert.equal(spread.legsInGroup, 2);
  assert.deepEqual(spread.rules.map((r) => r.key), ["dte_critical", "profit_target", "loss_limit", "dte_exit", "edge", "backstop"]);
  assert.ok(spread.profitTarget && spread.profitTarget.price === null);
  assert.ok(spread.rules.filter((r) => r.key !== "backstop").every((r) => r.price === null), "group rules carry no per-leg price");
  assert.equal(spread.rules.find((r) => r.key === "dte_exit")!.hit, true);
  const ic = optionExitPlan({ qty: -1, entryPrice: 1, currentPrice: 1, daysToExpiry: 4, strategy: "qqq_iron_condor", legsInGroup: 4 });
  assert.deepEqual(ic.rules.map((r) => r.key), ["dte_critical", "profit_target", "ic_max_loss", "loss_limit", "dte_exit", "edge", "backstop"]);
  assert.equal(ic.rules[0].hit, true);
  // 2026-11-06 (BOUGHT-OPTION PROFIT TAKING): a debit group takes profit on its combined value (was: no
  // profit target, rules ["dte_critical", "loss_limit", "dte_exit", "backstop"] — still so with it off)
  const straddle = optionExitPlan({ qty: 1, entryPrice: 5, currentPrice: 5, daysToExpiry: 40, strategy: "buy_straddle", legsInGroup: 2 });
  assert.deepEqual(straddle.rules.map((r) => r.key), ["dte_critical", "loss_limit", "dte_exit", "scale_out", "trail", "backstop"]);
  const st = Object.fromEntries(straddle.rules.map((r) => [r.key, r]));
  assert.deepEqual(st.scale_out, { key: "scale_out", price: null, inDays: null, hit: false, state: "not_possible" }, "1 unit cannot be halved");
  assert.ok(straddle.profitTarget === st.trail && st.trail.price === null);
  const off = optionExitPlan({ qty: 1, entryPrice: 5, currentPrice: 5, daysToExpiry: 40, strategy: "buy_straddle", legsInGroup: 2, profitTakingOn: false });
  assert.equal(off.profitTarget, null);
  assert.deepEqual(off.rules.map((r) => r.key), ["dte_critical", "loss_limit", "dte_exit", "backstop"]);
  const vertical = optionExitPlan({ qty: 2, entryPrice: 5, currentPrice: 9, daysToExpiry: 40, strategy: "bull_call_spread", legsInGroup: 2 });
  assert.equal(vertical.partial, true);
  assert.ok(vertical.profitTarget?.key === "scale_out" && vertical.profitTarget.price === null, "group rules carry no per-leg price");
});

test("the convexity hedge is insurance: rolled under 21 DTE, no target, no stop — only the Tier-1 floor", () => {
  const h = optionExitPlan({ qty: 3, entryPrice: 1.5, currentPrice: 0.9, daysToExpiry: 60, strategy: "convexity_hedge" });
  assert.equal(h.management, "hedge");
  assert.equal(h.profitTarget, null);
  assert.deepEqual(h.rules.map((r) => r.key), ["hedge_roll", "backstop"]);
  assert.deepEqual(h.rules[0], { key: "hedge_roll", price: null, inDays: 39, hit: false });
  close(h.rules[1].price, 0.375);
});

test("missing entry premium or DTE degrades to null prices / days, never to a guess", () => {
  const p = optionExitPlan({ qty: 1, entryPrice: null, currentPrice: 2, daysToExpiry: null, strategy: "buy_put" });
  assert.ok(p.rules.every((r) => r.price === null && r.inDays === null && r.hit === false));
});
