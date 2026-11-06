// client/src/bot/exitPlan.ts — the Book's words for WHAT a position is and
// WHEN the engine sells it (2026-11-02). Runtime pins over the shared plan:
// every number below is worked by hand from options_manager.py's thresholds
// (21 / 5 DTE, 50 % target, 2× credit, −50 % bought, −75 % / −150 % floor)
// and manage_positions' ladder as /api/bot/positions reports it.
import { test } from "node:test";
import assert from "node:assert/strict";
import { optionExitPlan, type PositionLevels } from "../../../shared/positionLevels";
import { optionWhatLine, optionExitSegments, stockExitSegments, strategyWords, joinSegments } from "./exitPlan";

test("strategy names read as plain words; an unrecorded or unknown one says so instead of guessing", () => {
  assert.equal(strategyWords("sell_cash_secured_put"), "cash-secured put");
  assert.equal(strategyWords("csp_normal_market"), "cash-secured put");
  assert.equal(strategyWords("bull_put_credit_spread"), "bull put credit spread");
  assert.equal(strategyWords("convexity_hedge"), "QQQ tail hedge");
  assert.equal(strategyWords("unknown"), "setup not recorded");
  assert.equal(strategyWords(null), "setup not recorded");
  assert.equal(strategyWords(""), "setup not recorded");
  assert.equal(strategyWords("some_new_thing"), "some new thing");
});

test("line 3: bought/sold · call/put · days left · setup · single leg or the legs closed as one", () => {
  assert.equal(joinSegments(optionWhatLine({ qty: 1, optionType: "CALL", daysToExpiry: 47, strategy: null })), "bought call · 47d to expiry · setup not recorded · single leg");
  assert.equal(joinSegments(optionWhatLine({ qty: -1, optionType: "PUT", daysToExpiry: 30, strategy: "sell_cash_secured_put" })), "sold put · 30d to expiry · cash-secured put · single leg");
  assert.equal(joinSegments(optionWhatLine({ qty: -1, optionType: "PUT", daysToExpiry: 18, strategy: "bull_put_credit_spread", legsInGroup: 2 })), "sold put · 18d to expiry · bull put credit spread · 2 legs held, closed as one");
  assert.equal(joinSegments(optionWhatLine({ qty: 1, optionType: "PUT", daysToExpiry: 18, strategy: "bull_put_credit_spread", legsInGroup: 1 })), "bought put · 18d to expiry · bull put credit spread · 1 leg held, closed as one");
  assert.equal(joinSegments(optionWhatLine({ qty: 3, optionType: "PUT", daysToExpiry: 60, strategy: "convexity_hedge" })), "bought put · 60d to expiry · QQQ tail hedge · overlay rolls it");
  assert.equal(joinSegments(optionWhatLine({ qty: -2, optionType: "CALL", daysToExpiry: 0, strategy: "covered_call" })), "sold call · expires today · covered call · single leg");
  assert.equal(joinSegments(optionWhatLine({ qty: 1, optionType: null, daysToExpiry: null, strategy: "unknown" })), "bought option · expiry not recorded · setup not recorded · single leg");
});

// 2026-11-06 (BOUGHT-OPTION PROFIT TAKING): the "NO profit target" line now holds with profit taking
// switched off (profitTakingOn: false); on, the next test pins the scale-out and the trail.
test("line 4, a bought call up 300 %, profit taking off: whole lot, NO profit target, the 21 DTE close and the −50 % stop at their prices, the floor", () => {
  const plan = optionExitPlan({ qty: 1, entryPrice: 3.2, currentPrice: 12.8, daysToExpiry: 47, strategy: null, profitTakingOn: false });
  const segs = optionExitSegments(plan);
  assert.equal(joinSegments(segs), "whole lot at once · no profit target · force close ≤5 DTE in 42d · close at 21 DTE in 26d · stop: −50% at $1.60 · gamma >0.08 inside 30 DTE · floor −75% at $0.80");
  assert.ok(segs.every((s) => !s.hit), "nothing is due on a healthy leg");
  const dying = optionExitSegments(optionExitPlan({ qty: 1, entryPrice: 3.2, currentPrice: 0.7, daysToExpiry: 20, strategy: "unknown", profitTakingOn: false }));
  assert.equal(joinSegments(dying), "whole lot at once · no profit target · force close ≤5 DTE in 15d · close at 21 DTE · due now · stop: −50% at $1.60 · hit · gamma >0.08 inside 30 DTE · floor −75% at $0.80 · hit");
  assert.deepEqual(dying.filter((s) => s.hit).map((s) => s.text), ["close at 21 DTE · due now", "stop: −50% at $1.60 · hit", "floor −75% at $0.80 · hit"]);
});

test("line 4, a bought call, profit taking on (2026-11-06): half at +100 %, the rest on the trail, in the manager's order", () => {
  assert.equal(joinSegments(optionExitSegments(optionExitPlan({ qty: 2, entryPrice: 3.2, currentPrice: 5.0, daysToExpiry: 47, strategy: "buy_call" }))),
    "half first, then the rest · force close ≤5 DTE in 42d · close at 21 DTE in 26d · half at +100% at $6.40 · trail 35% off the peak once +50% ($4.80) · stop: −50% at $1.60 · gamma >0.08 inside 30 DTE · floor −75% at $0.80");
  const up = optionExitSegments(optionExitPlan({ qty: 2, entryPrice: 3.2, currentPrice: 12.8, daysToExpiry: 47, strategy: "buy_call", peakRatio: 4.1 }));
  assert.deepEqual(up.filter((s) => s.hit).map((s) => s.text), ["half at +100% at $6.40 · hit"]);
  assert.ok(up.some((s) => s.text === "trail: rest below $8.53 (35% off the peak)"));
  assert.equal(joinSegments(optionExitSegments(optionExitPlan({ qty: 1, entryPrice: 3.2, currentPrice: 8.0, daysToExpiry: 47, strategy: "buy_call", peakRatio: 4.1, scaleOut: "filled" }))),
    "whole lot at once · force close ≤5 DTE in 42d · close at 21 DTE in 26d · half sold at +100% · trail: rest below $8.53 (35% off the peak) · hit · stop: −50% at $1.60 · gamma >0.08 inside 30 DTE · floor −75% at $0.80");
  assert.equal(joinSegments(optionExitSegments(optionExitPlan({ qty: 1, entryPrice: 3.2, currentPrice: 12.8, daysToExpiry: 47, strategy: null }))),
    "whole lot at once · force close ≤5 DTE in 42d · close at 21 DTE in 26d · 1 contract, no half to sell · trail 35% off the peak once +50% ($4.80) · stop: −50% at $1.60 · gamma >0.08 inside 30 DTE · floor −75% at $0.80");
});

test("line 4, a sold put: keep 50 % of the credit is the target, 2× the credit the stop, roll/close at 21 DTE, force at 5", () => {
  const plan = optionExitPlan({ qty: -1, entryPrice: 2.47, currentPrice: 2.67, daysToExpiry: 30, strategy: "sell_cash_secured_put" });
  assert.equal(joinSegments(optionExitSegments(plan)), "whole lot at once · force close ≤5 DTE in 25d · deep-ITM roll ≤10 DTE · roll/close at 21 DTE in 9d · target: keep 50% at $1.24 · stop: 2× credit at $4.94 · gamma >0.08 inside 30 DTE · floor −150% at $6.18");
  const won = optionExitSegments(optionExitPlan({ qty: -1, entryPrice: 2.47, currentPrice: 1.1, daysToExpiry: 30, strategy: "sell_cash_secured_put" }));
  assert.deepEqual(won.filter((s) => s.hit).map((s) => s.text), ["target: keep 50% at $1.24 · hit"]);
});

test("line 4, a multi-leg strategy and the hedge: the unit, the group rules without per-leg prices; insurance with no target and no stop", () => {
  assert.equal(joinSegments(optionExitSegments(optionExitPlan({ qty: -1, entryPrice: 4.1, currentPrice: 3.2, daysToExpiry: 18, strategy: "bull_put_credit_spread", legsInGroup: 2 }))),
    "all 2 legs as one · force close ≤5 DTE in 13d · target: 50% of net credit · stop: loss of 2× net credit · close all at 21 DTE · due now · edge gone (VRP turns negative) · floor −150% at $10.25");
  assert.equal(joinSegments(optionExitSegments(optionExitPlan({ qty: -1, entryPrice: 1, currentPrice: 1, daysToExpiry: 4, strategy: "qqq_iron_condor", legsInGroup: 4 }))),
    "all 4 legs as one · force close ≤5 DTE · due now · target: 50% of net credit · stop: 50% of max loss · stop: loss of 2× net credit · close all at 21 DTE · due now · edge gone (VRP turns negative) · floor −150% at $2.50");
  // 2026-11-06: a debit strategy now takes profit on its combined value (with it off: "no profit target" as before)
  assert.equal(joinSegments(optionExitSegments(optionExitPlan({ qty: 1, entryPrice: 5, currentPrice: 5, daysToExpiry: 40, strategy: "buy_straddle", legsInGroup: 2 }))),
    "all 2 legs as one · force close ≤5 DTE in 35d · stop: −50% of net debit · close all at 21 DTE in 19d · no half to sell (units cannot be split) · trail 35% off the peak once +50% · floor −75% at $1.25");
  assert.equal(joinSegments(optionExitSegments(optionExitPlan({ qty: 2, entryPrice: 5, currentPrice: 5, daysToExpiry: 40, strategy: "bull_call_spread", legsInGroup: 2 }))),
    "all 2 legs as one · force close ≤5 DTE in 35d · stop: −50% of net debit · close all at 21 DTE in 19d · half at +100% of net debit · trail 35% off the peak once +50% · floor −75% at $1.25");
  assert.equal(joinSegments(optionExitSegments(optionExitPlan({ qty: 1, entryPrice: 5, currentPrice: 5, daysToExpiry: 40, strategy: "buy_straddle", legsInGroup: 2, profitTakingOn: false }))),
    "all 2 legs as one · no profit target · force close ≤5 DTE in 35d · stop: −50% of net debit · close all at 21 DTE in 19d · floor −75% at $1.25");
  assert.equal(joinSegments(optionExitSegments(optionExitPlan({ qty: 3, entryPrice: 1.5, currentPrice: 0.9, daysToExpiry: 60, strategy: "convexity_hedge" }))),
    "insurance, not a trade · no target, no stop · rolled under 21 DTE in 39d · floor −75% at $0.38");
});

const lv = (over: Partial<PositionLevels> = {}): PositionLevels => ({
  recorded: true, side: "long", phase: 1, regime: "NEUTRAL", atrPct: 1.5, riskPct: 3, pnlPct: -2.42, rMultiple: -0.81, peakR: 0.2, highestPnl: 0.6, highestPrice: 745.8,
  stopPct: 3, stopPrice: 719.08, stopFrom: "entry", breakevenActive: false, stopSuppressed: null, hardFloorPct: 20, hardFloorPrice: 593.06,
  takeProfitPct: 4.5, takeProfitPrice: 774.68, takeProfitArmed: false,
  scaleOuts: [{ n: 1, rMultiple: 1, pct: 3, price: 763.56, done: false, applicable: true }, { n: 2, rMultiple: 2, pct: 6, price: 785.8, done: false, applicable: true }],
  trailAtrMult: 2, scaleOut2R: 2, originalQty: 61, remainingQty: 61, scalesCompleted: 0, daysHeld: 3, heldMinutes: 4320, timeStop: { day: 7, withinPct: 2, armed: false }, ...over,
});

test("line 4, a share lot: the stop and what it trails, the floor, the two thirds (the portion), the target after ⅔ out, the time stop", () => {
  assert.equal(joinSegments(stockExitSegments(lv())), "stop $719.08 from entry · floor $593.06 (−20%) · ⅓ at $763.56 (+1.0R) · ⅓ at $785.80 (+2.0R) · target $774.68 after ⅔ out · time stop d7 if ±2%");
  const phase2 = stockExitSegments(lv({ phase: 2, stopPrice: 551.43, stopFrom: "breakeven", breakevenActive: true, hardFloorPct: null, hardFloorPrice: null, takeProfitPrice: 576.24, takeProfitPct: 4.8, scaleOuts: [{ n: 1, rMultiple: 1, pct: 3.2, price: 569.08, done: true, applicable: true }, { n: 2, rMultiple: 2, pct: 6.4, price: 586.72, done: false, applicable: true }], scalesCompleted: 1, timeStop: { day: 7, withinPct: 2, armed: true } }));
  assert.equal(joinSegments(phase2), "stop $551.43 breakeven · ⅓ sold at $569.08 · ⅓ at $586.72 (+2.0R) · target $576.24 after ⅔ out · time stop d7 if ±2% · armed");
  assert.deepEqual(phase2.filter((s) => s.hit).map((s) => s.text), ["time stop d7 if ±2% · armed"]);
  assert.equal(joinSegments(stockExitSegments(lv({ stopPrice: 25.47, stopSuppressed: { until: "day4", reason: "short-hold grace", overridePct: 15, minutesLeft: null, daysLeft: 2 }, hardFloorPrice: 21.22 }))).split(" · ").slice(0, 3).join(" · "), "stop $25.47 from entry · waits 2d (grace to day 4) · exit now only below −15% · floor $21.22 (−20%)".split(" · ").slice(0, 3).join(" · "));
  assert.equal(joinSegments(stockExitSegments(lv({ phase: 4, stopFrom: "peak_price", stopPrice: 801.2, hardFloorPct: null, hardFloorPrice: null, takeProfitPct: null, takeProfitPrice: null, scaleOuts: [{ n: 1, rMultiple: 1, pct: 3, price: 763.56, done: true, applicable: true }, { n: 2, rMultiple: 2, pct: 6, price: 785.8, done: true, applicable: true }], scalesCompleted: 2 }))),
    "stop $801.20 trails the high · ⅓ sold at $763.56 · ⅓ sold at $785.80 · trail takes the rest · time stop d7 if ±2%");
  assert.equal(joinSegments(stockExitSegments(lv({ scaleOuts: [{ n: 1, rMultiple: 1, pct: 3, price: 763.56, done: false, applicable: false }, { n: 2, rMultiple: 2, pct: 6, price: 785.8, done: false, applicable: false }], originalQty: 1, remainingQty: 1 }))),
    "stop $719.08 from entry · floor $593.06 (−20%) · no scale-out (lot of 1) · target $774.68 after ⅔ out · time stop d7 if ±2%");
  assert.equal(joinSegments(stockExitSegments(null)), "stop state not recorded yet · the engine writes it on its first manage tick after entry");
  assert.equal(joinSegments(stockExitSegments(lv({ recorded: false }))), "stop state not recorded yet · the engine writes it on its first manage tick after entry");
});
