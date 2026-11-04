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
import {
  optionExitPlan, optionManagement,
  OPTIONS_DTE_EXIT_THRESHOLD, OPTIONS_DTE_CRITICAL, OPTIONS_LOSS_LIMIT_MULTIPLIER, OPTIONS_BOUGHT_LOSS_LIMIT_FRAC, OPTIONS_IC_MAX_LOSS_EXIT_FRAC,
  OPTIONS_GAMMA_THRESHOLD, OPTIONS_GAMMA_DTE, OPTIONS_ASSIGNMENT_DELTA, OPTIONS_ASSIGNMENT_DTE, OPTIONS_MIN_HOLD_MINUTES,
  OPTIONS_MULTI_LEG_STRATEGIES, OPTIONS_CREDIT_STRATEGIES, OPTIONS_DEBIT_STRATEGIES, CONVEXITY_HEDGE_STRATEGY, CONVEXITY_ROLL_DTE,
  OPTION_LONG_KILL_LOSS_PCT, OPTION_SHORT_KILL_LOSS_PCT,
} from "../shared/positionLevels";
import { readFileSync } from "node:fs";
import path from "node:path";

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
  assert.equal(lc!.profitTargetFrac, null, "a bought leg has no manager profit target — the chart must not claim one");
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

test("a bought single leg: no profit target; close at 21 DTE, −50% of premium, 5 DTE force, gamma inside 30 DTE, the Tier-1 −75% floor; never a partial exit", () => {
  const plan = optionExitPlan({ qty: 1, entryPrice: 3.2, currentPrice: 12.8, daysToExpiry: 47, strategy: null });
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
  const down = optionExitPlan({ qty: 1, entryPrice: 3.2, currentPrice: 0.7, daysToExpiry: 20, strategy: "unknown" });
  assert.equal(down.strategy, null);
  const d = Object.fromEntries(down.rules.map((r) => [r.key, r]));
  assert.equal(d.loss_limit.hit, true);
  assert.equal(d.backstop.hit, true);
  assert.equal(d.dte_exit.hit, true); assert.equal(d.dte_exit.inDays, -1);
  assert.equal(d.dte_critical.hit, false);
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
  const straddle = optionExitPlan({ qty: 1, entryPrice: 5, currentPrice: 5, daysToExpiry: 40, strategy: "buy_straddle", legsInGroup: 2 });
  assert.equal(straddle.profitTarget, null);
  assert.deepEqual(straddle.rules.map((r) => r.key), ["dte_critical", "loss_limit", "dte_exit", "backstop"]);
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
