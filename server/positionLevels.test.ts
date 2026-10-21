// shared/positionLevels.ts pinned against bot_engine.py manage_positions'
// formulas (read 2026-10-21). Every expected number below is worked by hand
// from the engine's own arithmetic, so a drift in either file fails here.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  positionLevels, optionLegLevels, regimeMults, pricePct, parseEngineTime, levelConfigFromBaseConfig,
  HARD_STOP_PCT_DEFAULT, TIME_STOP_DAYS_ENGINE, SHORT_HOLD_DEEP_LOSS_PCT, MIN_HOLD_CATASTROPHE_PCT, OPTIONS_PROFIT_TARGET_FRAC,
  type StopStateRow,
} from "../shared/positionLevels";

const NOW = Date.parse("2026-10-21T15:00:00Z");
const cfg = { stopScalar: 1.0, hardStopPct: 20, hardStopEnabled: true };
const close = (a: number | null, b: number, eps = 1e-6) => { assert.ok(a !== null && Math.abs(a - b) < eps, `${a} ≠ ${b}`); };

test("regime multipliers mirror the engine's is_bullish / is_bearish branches", () => {
  assert.deepEqual(regimeMults("BULL"), { bullish: true, bearish: false, trailAtrMult: 3, scaleOut2R: 3 });
  assert.deepEqual(regimeMults("NEUTRAL_BULL"), { bullish: true, bearish: false, trailAtrMult: 3, scaleOut2R: 3 });
  assert.deepEqual(regimeMults("CAUTION"), { bullish: false, bearish: true, trailAtrMult: 1.5, scaleOut2R: 1.5 });
  assert.deepEqual(regimeMults("NEUTRAL"), { bullish: false, bearish: false, trailAtrMult: 2, scaleOut2R: 2 });
  assert.deepEqual(regimeMults(null), { bullish: false, bearish: false, trailAtrMult: 2, scaleOut2R: 2 });
});

test("phase 1 long: stop 2×ATR from entry (clamped 1.5–10), TP max(4, min(3×ATR, 15)) armed only after ⅔ out, thirds at +1R and the regime's 2R, hard floor −20", () => {
  const stop: StopStateRow = { phase: 1, initial_risk_pct: 5.0, current_stop_pct: 2.5, highest_pnl: 0.4, highest_price: 101, original_qty: 30, remaining_qty: 30, scales_completed: 0, breakeven_active: false, regime: "NEUTRAL", days_held: 5, entry_date: "2026-10-16", entry_timestamp: "2026-10-16T14:31:00.000000" };
  const lv = positionLevels({ entry: 100, current: 100.4, qty: 30, pnlPct: 0.4, stop, config: cfg, nowMs: NOW });
  assert.equal(lv.recorded, true);
  assert.equal(lv.side, "long");
  assert.equal(lv.phase, 1);
  close(lv.stopPct, 5.0); // max(1.5, min(2.5×2×1.0, 10))
  close(lv.stopPrice, 95.0);
  assert.equal(lv.stopFrom, "entry");
  close(lv.takeProfitPct, 7.5); // max(4, min(2.5×3, 15))
  close(lv.takeProfitPrice, 107.5);
  assert.equal(lv.takeProfitArmed, false);
  assert.equal(lv.scaleOuts.length, 2);
  close(lv.scaleOuts[0].price, 105.0); // +1R = +5 %
  assert.equal(lv.scaleOuts[0].done, false);
  assert.equal(lv.scaleOuts[1].rMultiple, 2); // NEUTRAL → 2R
  close(lv.scaleOuts[1].price, 110.0);
  assert.equal(lv.hardFloorPct, HARD_STOP_PCT_DEFAULT);
  close(lv.hardFloorPrice, 80.0);
  assert.equal(lv.stopSuppressed, null); // day 5, past the short-hold window
  assert.equal(lv.timeStop.day, TIME_STOP_DAYS_ENGINE);
  assert.equal(lv.timeStop.armed, false);
  close(lv.rMultiple, 0.08);
});

test("phase 2 long (peak ≥ 1R): stop trails 1.5×ATR×scalar from the P&L high-water mark; first third done; TP still pending; no hard floor", () => {
  const stop: StopStateRow = { phase: 2, initial_risk_pct: 4.0, current_stop_pct: 2.0, highest_pnl: 6.0, highest_price: 106, original_qty: 30, remaining_qty: 20, scales_completed: 1, breakeven_active: true, regime: "BEAR", days_held: 9 };
  const lv = positionLevels({ entry: 100, current: 104, qty: 20, pnlPct: 4.0, stop, config: { stopScalar: 1.2, hardStopPct: 20, hardStopEnabled: true }, nowMs: NOW });
  assert.equal(lv.phase, 2); // peak 6 / 4 = 1.5R
  close(lv.stopPct, 3.6); // max(1.5, 2.0 × 1.5 × 1.2)
  close(lv.stopPrice, 102.4); // entry × (1 + (6.0 − 3.6) / 100) — above entry, so breakeven does not raise it
  assert.equal(lv.stopFrom, "peak_pnl");
  assert.equal(lv.breakevenActive, true);
  assert.equal(lv.scaleOuts[0].done, true);
  assert.equal(lv.scaleOuts[1].rMultiple, 1.5); // BEAR
  close(lv.scaleOuts[1].price, 106.0);
  assert.equal(lv.hardFloorPrice, null);
  assert.equal(lv.takeProfitArmed, false);
  assert.equal(lv.timeStop.armed, true); // held 9 ≥ 7
});

test("breakeven raises a trailing stop that sits below entry; the live price can lift the peak (and the phase) past the stored one", () => {
  const stop: StopStateRow = { phase: 1, initial_risk_pct: 4.0, current_stop_pct: 2.0, highest_pnl: 0.9, highest_price: 100.9, original_qty: 30, remaining_qty: 20, scales_completed: 1, breakeven_active: true, regime: "NEUTRAL", days_held: 3 };
  // live P&L +4.2 % → peak 4.2 / 4 = 1.05R → phase 2; stop = 4.2 − max(1.5, 3.0) = +1.2 % → 101.2 (above entry)
  const live = positionLevels({ entry: 100, current: 104.2, qty: 20, pnlPct: 4.2, stop, config: cfg, nowMs: NOW });
  assert.equal(live.phase, 2);
  close(live.stopPrice, 101.2);
  // live P&L +2.0 % with a stored peak of 4.1 % → stop = 4.1 − 3.0 = +1.1 → 101.1; breakeven irrelevant
  const held = positionLevels({ entry: 100, current: 102, qty: 20, pnlPct: 2.0, stop: { ...stop, highest_pnl: 4.1 }, config: cfg, nowMs: NOW });
  close(held.stopPrice, 101.1);
  // stored peak 4.1 with ATR 3 → stop width max(1.5, 4.5) = 4.5 → 4.1 − 4.5 = −0.4 % → breakeven lifts it to entry
  const be = positionLevels({ entry: 100, current: 102, qty: 20, pnlPct: 2.0, stop: { ...stop, highest_pnl: 4.1, current_stop_pct: 3.0 }, config: cfg, nowMs: NOW });
  close(be.stopPrice, 100.0);
  assert.equal(be.stopFrom, "breakeven");
});

test("phase 4 trails the highest PRICE by trail×ATR×scalar (BULL → 3×)", () => {
  const stop: StopStateRow = { phase: 4, initial_risk_pct: 3.0, current_stop_pct: 2.0, highest_pnl: 12.0, highest_price: 112, original_qty: 30, remaining_qty: 10, scales_completed: 2, breakeven_active: true, regime: "BULL", days_held: 20 };
  const lv = positionLevels({ entry: 100, current: 110, qty: 10, pnlPct: 10, stop, config: { stopScalar: 0.9, hardStopPct: 20, hardStopEnabled: true }, nowMs: NOW });
  assert.equal(lv.phase, 4); // 12 / 3 = 4R
  close(lv.stopPct, 5.4); // 2.0 × 3 × 0.9
  close(lv.stopPrice, 112 * (1 - 0.054));
  assert.equal(lv.stopFrom, "peak_price");
  assert.equal(lv.takeProfitPct, null); // no fixed target in phase 3+
  assert.equal(lv.takeProfitArmed, true);
});

test("suppression: the first hour only lets a −10 % loss out; days 0–3 in phase 1 only a −15 % loss", () => {
  const base: StopStateRow = { phase: 1, initial_risk_pct: 5.0, current_stop_pct: 2.5, highest_pnl: 0, original_qty: 10, remaining_qty: 10, scales_completed: 0, regime: "NEUTRAL" };
  const fresh = positionLevels({ entry: 100, current: 99, qty: 10, pnlPct: -1, stop: { ...base, days_held: 0, entry_timestamp: "2026-10-21T14:40:00.000000" }, config: cfg, nowMs: NOW });
  assert.equal(fresh.stopSuppressed?.until, "60min");
  assert.equal(fresh.stopSuppressed?.overridePct, MIN_HOLD_CATASTROPHE_PCT);
  close(fresh.stopSuppressed?.minutesLeft ?? null, 40);
  const young = positionLevels({ entry: 100, current: 99, qty: 10, pnlPct: -1, stop: { ...base, days_held: 2, entry_timestamp: "2026-10-19T14:40:00.000000" }, config: cfg, nowMs: NOW });
  assert.equal(young.stopSuppressed?.until, "day4");
  assert.equal(young.stopSuppressed?.overridePct, SHORT_HOLD_DEEP_LOSS_PCT);
  assert.equal(young.stopSuppressed?.daysLeft, 2);
  // phase 2 is never suppressed by the day-4 grace
  const p2 = positionLevels({ entry: 100, current: 105, qty: 10, pnlPct: 5, stop: { ...base, highest_pnl: 5.5, days_held: 2, entry_timestamp: "2026-10-19T14:40:00.000000" }, config: cfg, nowMs: NOW });
  assert.equal(p2.phase, 2);
  assert.equal(p2.stopSuppressed, null);
});

test("short side: levels mirror below / above entry; a single-share lot cannot scale out; missing config falls back to the engine defaults", () => {
  const stop: StopStateRow = { phase: 1, initial_risk_pct: 4.0, current_stop_pct: 2.0, highest_pnl: 0, original_qty: 1, remaining_qty: 1, scales_completed: 0, regime: "NEUTRAL", days_held: 6 };
  const lv = positionLevels({ entry: 50, current: 49, qty: -1, pnlPct: 2, stop, config: null, nowMs: NOW });
  assert.equal(lv.side, "short");
  close(lv.stopPrice, 52.0); // +4 % against a short
  close(lv.takeProfitPrice, 47.0); // max(4, 6) = 6 % → 50 × 0.94
  close(lv.hardFloorPrice, 60.0); // engine default 20 %
  assert.equal(lv.scaleOuts[0].applicable, false);
  assert.equal(pricePct(50, -4, "short"), 52);
});

test("no stop state → not recorded: no stop, target, thirds or floor, phase 1 by default", () => {
  const lv = positionLevels({ entry: 100, current: 101, qty: 5, pnlPct: 1, stop: null, config: cfg, nowMs: NOW });
  assert.equal(lv.recorded, false);
  assert.equal(lv.stopPrice, null);
  assert.equal(lv.takeProfitPrice, null);
  assert.deepEqual(lv.scaleOuts, []);
  assert.equal(lv.hardFloorPrice, null);
  assert.equal(lv.phase, 1);
  // a row without an ATR is not recorded either (bot.ts used to invent 2.5 %)
  assert.equal(positionLevels({ entry: 100, current: 101, qty: 5, pnlPct: 1, stop: { phase: 2 }, config: cfg, nowMs: NOW }).recorded, false);
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
  assert.equal(optionLegLevels({ optionType: "PUT", strike: 0, qty: -1, premium: 1 }), null);
});

test("the regime's scalar is picked from BASE_CONFIG; an unknown regime (NEUTRAL_BULL has no key) reads 1.0 as the engine's .get default; a failed read is null", () => {
  const cfg = { STOP_SCALAR_BULL: 0.9, STOP_SCALAR_BEAR: 1.2, POSITION_HARD_STOP_PCT: 20, POSITION_HARD_STOP_ENABLED: true };
  assert.deepEqual(levelConfigFromBaseConfig(cfg, "BEAR"), { stopScalar: 1.2, hardStopPct: 20, hardStopEnabled: true });
  assert.deepEqual(levelConfigFromBaseConfig(cfg, "NEUTRAL_BULL"), { stopScalar: 1.0, hardStopPct: 20, hardStopEnabled: true });
  assert.deepEqual(levelConfigFromBaseConfig({ STOP_SCALAR_BULL: 0.9, POSITION_HARD_STOP_ENABLED: false }, "BULL"), { stopScalar: 0.9, hardStopPct: null, hardStopEnabled: false });
  assert.equal(levelConfigFromBaseConfig({}, "BULL"), null);
  assert.equal(levelConfigFromBaseConfig({ error: "import failed" }, "BULL"), null);
  assert.equal(levelConfigFromBaseConfig(null, "BULL"), null);
});
