// shared/positionLevels.ts — the engine's exit levels for ONE position.
//
// STOCKS (2026-11-12, KNOWN BROKEN #50a): the ladder the engine executes —
// server/bot.ts checkPositionOnTick on the levels syncMonitoredPositions
// builds from the stop state, plus the sync's POS-KILL backstop. The numbers
// and the arithmetic live in shared/stockExitLadder.ts, which
// server/stockExitLadder.test.ts pins against server/bot.ts and
// live_exit_policy.py. Before that date this file followed
// bot_engine.manage_positions' formulas (a regime scalar on every stop, a 10 %
// phase-1 clamp, a 60-minute and a 4-day suppression, a −20 % hard floor).
// Tier 1 never executes those, so the Book showed exits that would not happen.
//
// OPTIONS: options_manager.py's rules, further down.
//
// One source of truth for two readers: server/bot.ts /api/bot/positions
// (stopPrice, takeProfitPrice and the `levels` block the Bot page chart draws)
// and the chart's own option-leg arithmetic. Pure; no I/O; no clock of its own
// (nowMs is an input); pinned by server/positionLevels.test.ts.
import {
  stockLadder, regimeClass, ladderDaysHeld, pricePct,
  SCALE2_R, TRAIL4_ATR_MULT, TIME_STOP_DAYS, TIME_STOP_BAND_PCT,
  type Side, type StopBasis,
} from "./stockExitLadder";

export { pricePct };
export type { Side, StopBasis };

/** One row of voltrade_stop_state.json (written by manage_positions every
 *  tick; older rows may lack newer fields, so everything is optional). */
export interface StopStateRow {
  phase?: number;
  initial_risk_pct?: number;
  current_stop_pct?: number; // the raw ATR % at the last tick ("bot.ts computes final stop")
  highest_pnl?: number;
  highest_price?: number;
  r_multiple?: number;
  original_qty?: number;
  remaining_qty?: number;
  scales_completed?: number;
  breakeven_active?: boolean;
  regime?: string;
  days_held?: number;
  entry_timestamp?: string; // naive ISO, the container's clock (UTC on Railway)
  entry_date?: string;
}

export interface LevelInput {
  entry: number;
  current: number;
  qty: number; // negative = short
  pnlPct: number | null; // Alpaca unrealized_plpc × 100 (the engine's pnl_pct); recomputed from prices when null
  stop: StopStateRow | null | undefined;
  nowMs?: number;
}

export interface ScaleOut {
  n: 1 | 2;
  rMultiple: number; // 1 for the first third; the regime's 2R threshold for the second
  pct: number; // P&L % at the trigger
  price: number;
  done: boolean;
  applicable: boolean; // done, or the engine can still sell it: more than 1 share left and a third (floor(original / 3)) of at least 1 share
}

export interface PositionLevels {
  recorded: boolean; // false until the stop state carries this position's ATR (manage_positions' first tick after entry); no level is drawn before
  side: Side;
  phase: number;
  regime: string | null;
  atrPct: number | null;
  riskPct: number | null; // 1R, in %
  pnlPct: number | null;
  rMultiple: number | null;
  peakR: number | null;
  highestPnl: number | null;
  highestPrice: number | null;
  stopPct: number | null; // the phase's stop width in %
  stopPrice: number | null; // the engine's trigger price for the phase, breakeven-raised when active
  stopFrom: StopBasis | null;
  stopHit: boolean; // the current reading already meets the engine's stop test: it sells on its next 1-minute bar for the symbol
  breakevenActive: boolean;
  /** The stock BACKSTOP, every phase: server/bot.ts POS-KILL closes a stock that is this % below entry on two
   *  valid readings in a row (STOCK_KILL_LOSS_PCT, the 60 s position sync). The field names date from 2026-10-21,
   *  when they carried manage_positions' −20 % phase-1 hard stop, which nothing executes; they are kept so a Bot
   *  page already open on the previous bundle still reads a number. */
  hardFloorPct: number | null;
  hardFloorPrice: number | null;
  takeProfitPct: number | null;
  takeProfitPrice: number | null;
  takeProfitArmed: boolean; // the engine's TP fires only after both scale-outs
  scaleOuts: ScaleOut[];
  trailAtrMult: number;
  scaleOut2R: number;
  originalQty: number | null;
  remainingQty: number | null;
  scalesCompleted: number;
  daysHeld: number | null;
  heldMinutes: number | null;
  timeStop: { day: number; withinPct: number; armed: boolean };
}

/** options_manager.PROFIT_TARGET_PCT — the manager closes sold premium at
 *  this share of max profit; an option-price level, so the chart prints it
 *  in the legend rather than drawing it on the underlying. */
export const OPTIONS_PROFIT_TARGET_FRAC = 0.5;

const num = (v: unknown): number | null => {
  if (v === null || v === undefined || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
};

/** The regime-adaptive multipliers checkPositionOnTick derives (isBullish / isBearish). */
export function regimeMults(regime: string | null | undefined): { bullish: boolean; bearish: boolean; trailAtrMult: number; scaleOut2R: number } {
  const cls = regimeClass(regime);
  return { bullish: cls === "bullish", bearish: cls === "bearish", trailAtrMult: TRAIL4_ATR_MULT[cls], scaleOut2R: SCALE2_R[cls] };
}

/** The engine's naive ISO timestamps are the container's clock (UTC on
 *  Railway); a timestamp without a zone is read as UTC, never as the
 *  reader's local time. */
export function parseEngineTime(ts: string | null | undefined): number | null {
  if (!ts || typeof ts !== "string") return null;
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(ts);
  const ms = Date.parse(hasZone ? ts : ts + "Z");
  return Number.isFinite(ms) ? ms : null;
}

export function positionLevels(i: LevelInput): PositionLevels {
  const nowMs = i.nowMs ?? Date.now();
  const side: Side = i.qty < 0 ? "short" : "long";
  const s = i.stop ?? null;
  const entry = num(i.entry);
  const current = num(i.current);
  const atr = num(s?.current_stop_pct);
  // A row with no ATR (or a zero one) is not recorded: the engine runs its 2 % default there, and the Book
  // draws nothing rather than a level worked from a number the stop state does not hold.
  const recorded = !!s && atr !== null && atr > 0 && entry !== null && entry > 0;
  const regime = typeof s?.regime === "string" && s.regime ? s.regime : null;
  const { trailAtrMult, scaleOut2R } = regimeMults(regime);
  const pnl = num(i.pnlPct) ?? (entry && current ? (side === "long" ? current / entry - 1 : 1 - current / entry) * 100 : null);
  const entryMs = parseEngineTime(s?.entry_timestamp);
  const heldMinutes = entryMs !== null ? Math.max(0, (nowMs - entryMs) / 60_000) : null;
  // the engine's own day count (from the entry date); the row's days_held only names the age when there is no date
  const engineDays = ladderDaysHeld(s?.entry_date, nowMs);
  const daysHeld = engineDays ?? num(s?.days_held);
  const timeStop = { day: TIME_STOP_DAYS, withinPct: TIME_STOP_BAND_PCT, armed: engineDays !== null && engineDays >= TIME_STOP_DAYS };
  const qtyAbs = Number.isFinite(i.qty) ? Math.abs(i.qty) : null;

  if (!recorded || entry === null) {
    const storedPeak = num(s?.highest_pnl);
    const storedHigh = num(s?.highest_price);
    const scalesCompleted = Math.max(0, Math.round(num(s?.scales_completed) ?? 0));
    return {
      recorded: false, side, phase: Math.max(1, Math.min(4, Math.round(num(s?.phase) ?? 1))), regime, atrPct: atr, riskPct: null, pnlPct: pnl, rMultiple: null, peakR: null,
      highestPnl: storedPeak === null && pnl === null ? null : Math.max(storedPeak ?? -Infinity, pnl ?? -Infinity),
      highestPrice: storedHigh === null && current === null ? null : Math.max(storedHigh ?? -Infinity, current ?? -Infinity),
      stopPct: null, stopPrice: null, stopFrom: null, stopHit: false, breakevenActive: !!s?.breakeven_active, hardFloorPct: null, hardFloorPrice: null,
      takeProfitPct: null, takeProfitPrice: null, takeProfitArmed: scalesCompleted >= 2, scaleOuts: [], trailAtrMult, scaleOut2R,
      originalQty: num(s?.original_qty) ?? qtyAbs, remainingQty: num(s?.remaining_qty) ?? qtyAbs, scalesCompleted, daysHeld, heldMinutes, timeStop,
    };
  }

  const L = stockLadder({ row: s, side, entry, current, qty: qtyAbs ?? 0, pnlPct: pnl, nowMs });
  return {
    recorded: true, side, phase: L.phase, regime, atrPct: L.atrPct, riskPct: L.riskPct, pnlPct: L.pnlPct, rMultiple: L.rMultiple, peakR: L.peakR,
    highestPnl: L.highestPnl, highestPrice: L.highestPrice,
    stopPct: L.stopPct, stopPrice: L.stopPrice, stopFrom: L.stopFrom, stopHit: L.stopHit, breakevenActive: L.breakevenActive,
    hardFloorPct: L.backstopPct, hardFloorPrice: L.backstopPrice,
    takeProfitPct: L.takeProfitPct, takeProfitPrice: L.takeProfitPrice, takeProfitArmed: L.takeProfitArmed, scaleOuts: L.scaleOuts, trailAtrMult, scaleOut2R,
    originalQty: L.originalQty, remainingQty: L.remainingQty, scalesCompleted: L.scalesCompleted, daysHeld, heldMinutes, timeStop,
  };
}

// ── option legs (manage_positions skips these; options_manager owns them) ──

export interface OptionLegInput {
  optionType: "CALL" | "PUT";
  strike: number;
  qty: number; // negative = sold (short premium)
  premium: number; // the leg's average entry price per share
}

export interface OptionLegLevels {
  side: Side;
  strike: number;
  /** underlying price at expiry where the leg's own P&L is zero */
  breakevenAtExpiry: number;
  /** where the sold leg keeps its whole premium at expiry (short legs only) */
  maxProfitZone: "above_strike" | "below_strike" | null;
  /** options_manager.PROFIT_TARGET_PCT for a sold leg; null for a bought leg — its profit taking is the scale-out + trail (OPTIONS_PT_*), not a share of max profit */
  profitTargetFrac: number | null;
}

export function optionLegLevels(o: OptionLegInput): OptionLegLevels | null {
  const k = num(o.strike), prem = num(o.premium);
  if (k === null || prem === null || k <= 0) return null;
  const side: Side = o.qty < 0 ? "short" : "long";
  const breakevenAtExpiry = o.optionType === "PUT" ? k - prem : k + prem;
  const maxProfitZone = side === "short" ? (o.optionType === "PUT" ? "above_strike" : "below_strike") : null;
  return { side, strike: k, breakevenAtExpiry, maxProfitZone, profitTargetFrac: side === "short" ? OPTIONS_PROFIT_TARGET_FRAC : null };
}

// ── option exits (options_manager.py manage_options_positions, read 2026-11-02) ──
// The manager's own thresholds, named after its identifiers so a drift in
// either file fails server/positionLevels.test.ts (which reads the Python
// source). None of these is read from system_config; the engine hard-codes
// them at the top of options_manager.py.
export const OPTIONS_DTE_EXIT_THRESHOLD = 21; // DTE_EXIT_THRESHOLD — close (bought) or roll-else-close (sold) at ≤ 21 DTE
export const OPTIONS_DTE_CRITICAL = 5; // DTE_CRITICAL — force close at ≤ 5 DTE, ignoring the hold gate
export const OPTIONS_LOSS_LIMIT_MULTIPLIER = 2.0; // LOSS_LIMIT_MULTIPLIER — sold: close when the cost to close reaches 2× the credit; a credit strategy group: when its loss reaches 2× its NET credit
export const OPTIONS_BOUGHT_LOSS_LIMIT_FRAC = 0.5; // the standalone path's `loss_pct >= 0.50` for a bought option (and the debit-strategy group rule, on the NET debit)
export const OPTIONS_IC_MAX_LOSS_EXIT_FRAC = 0.5; // IC_MAX_LOSS_EXIT_PCT — iron condors close at half their max loss (recorded at entry, else worked from the legs: widest wing × 100 × qty − net credit)
export const OPTIONS_GAMMA_THRESHOLD = 0.08; // GAMMA_THRESHOLD, judged only at ≤ OPTIONS_GAMMA_DTE
export const OPTIONS_GAMMA_DTE = 30;
export const OPTIONS_ASSIGNMENT_DELTA = 0.8; // ASSIGNMENT_DELTA_THRESHOLD — a sold leg at |delta| > 0.80 and ≤ 10 DTE is rolled, else closed
export const OPTIONS_ASSIGNMENT_DTE = 10;
export const OPTIONS_MIN_HOLD_MINUTES = 60; // MIN_HOLD_MINUTES — no exit but the critical-DTE one inside the first hour
/** BOUGHT-OPTION PROFIT TAKING (2026-11-06): system_config OPTIONS_PROFIT_TAKING's
 *  defaults (server/positionLevels.test.ts reads them). A bought standalone leg
 *  sells floor(qty × fraction) at +100% of its premium, then the rest once the
 *  mark is 35% below its peak, armed after the peak has been +50%; a debit
 *  strategy group the same on its combined value against the net debit. */
export const OPTIONS_PT_SCALE_OUT_GAIN = 1.0;
export const OPTIONS_PT_SCALE_OUT_FRACTION = 0.5;
export const OPTIONS_PT_TRAIL_ARM_GAIN = 0.5;
export const OPTIONS_PT_TRAIL_GIVEBACK = 0.35;
/** MULTI_LEG_STRATEGIES — legs the manager groups by underlying + strategy
 *  and closes as ONE unit (_manage_strategy_group). Everything else is a
 *  standalone leg on the single-leg rules — including a leg whose state
 *  never recorded a strategy ("unknown"). */
export const OPTIONS_MULTI_LEG_STRATEGIES: readonly string[] = ["buy_straddle", "short_straddle", "iron_condor", "bull_call_spread", "bear_put_spread", "bull_put_credit_spread", "qqq_iron_condor"];
/** _manage_strategy_group's is_credit_strategy / is_debit_strategy sets */
export const OPTIONS_CREDIT_STRATEGIES: readonly string[] = ["short_straddle", "iron_condor", "csp_normal_market", "bull_put_credit_spread", "sell_cash_secured_put", "covered_call", "qqq_iron_condor"];
export const OPTIONS_DEBIT_STRATEGIES: readonly string[] = ["buy_straddle", "bull_call_spread", "bear_put_spread"];
/** The multi-leg strategies whose worst case is bounded by their own legs
   *  (wings cap a condor / vertical; a long straddle can lose only its debit).
   *  server/bot.ts POS-KILL (2026-11-02) never judges such a leg on its own:
   *  a wing at −99 % is the structure working, a short leg at −150 % of its
   *  credit is the GROUP's loss, which options_manager manages as a unit. A
   *  short straddle is undefined-risk and keeps the per-leg backstop. */
  export const OPTIONS_DEFINED_RISK_STRATEGIES: readonly string[] = ["iron_condor", "qqq_iron_condor", "bull_put_credit_spread", "bull_call_spread", "bear_put_spread", "buy_straddle"];
  /** bot_engine._run_convexity_overlay owns these QQQ puts (the manager skips
 *  them): held as insurance, sold to close and re-bought ~60 DTE out once
 *  under CONVEXITY_OVERLAY.roll_dte. No target, no stop. */
export const CONVEXITY_HEDGE_STRATEGY = "convexity_hedge";
export const CONVEXITY_ROLL_DTE = 21;
/** server/bot.ts Tier-1 POS-KILL backstop for option positions, as % of the
 *  premium — the one exit that still runs while the Python side is halted.
 *  Confirmed on two consecutive valid-mark ticks (~45 s apart). */
export const OPTION_LONG_KILL_LOSS_PCT = -75;
export const OPTION_SHORT_KILL_LOSS_PCT = -150;

export type OptionManagement = "single" | "group" | "hedge";

/** How the engine manages an option leg with this recorded strategy. */
export function optionManagement(strategy: string | null | undefined): OptionManagement {
  const s = (strategy ?? "").trim();
  if (s === CONVEXITY_HEDGE_STRATEGY) return "hedge";
  if (OPTIONS_MULTI_LEG_STRATEGIES.includes(s)) return "group";
  return "single";
}

export interface OptionExitInput {
  qty: number; // negative = sold (short premium)
  entryPrice: number | null; // premium per share at entry (a sold leg's credit)
  currentPrice: number | null; // the leg's mark now, per share
  daysToExpiry: number | null;
  strategy: string | null | undefined; // the state's recorded strategy; null / "" / "unknown" = not recorded
  /** legs of the same underlying + strategy held right now, this one included (1 when alone) */
  legsInGroup?: number;
  /** the manager's peak mark ÷ premium (options state profit_taking.peak_ratio); null until it has one */
  peakRatio?: number | null;
  /** the manager's scale-out record status ("submitted" | "filled" | "not_possible" | …), "blocked" when the
   *  manager will not take profit on this leg (it may cover a sold option of its type); null when none */
  scaleOut?: string | null;
  /** system_config OPTIONS_PROFIT_TAKING.enabled as the server read it (absent = the default, on) */
  profitTakingOn?: boolean;
}

export type OptionExitRuleKey = "dte_critical" | "assignment" | "dte_exit" | "profit_target" | "scale_out" | "trail" | "loss_limit" | "ic_max_loss" | "gamma" | "edge" | "hedge_roll" | "backstop";

export interface OptionExitRule {
  key: OptionExitRuleKey;
  /** the leg's price (per share) the rule fires at, for a price rule on a standalone leg; null for DTE rules and for group rules (judged on the strategy's combined P&L) */
  price: number | null;
  /** days until a DTE rule arms (≤ 0 = due on the next manage tick); null for a price rule */
  inDays: number | null;
  /** the current reading already satisfies the trigger — the manager acts on its next tick (after the 60-minute hold, critical DTE excepted) */
  hit: boolean;
  /** scale_out: the half is already sold ("done") or the lot cannot be split ("not_possible"); trail: "armed" (price = where it sells the rest) or "not_armed" (price = the mark that arms it); absent otherwise */
  state?: "done" | "not_possible" | "armed" | "not_armed";
}

export interface OptionExitPlan {
  side: Side;
  management: OptionManagement;
  /** the recorded strategy, or null when the state never recorded one */
  strategy: string | null;
  /** the manager's checks in the order it runs them */
  rules: OptionExitRule[];
  /** true when a scale-out can still sell part of the position (a bought leg or debit group of 2+); every other option exit is the whole lot */
  partial: boolean;
  /** the rule that takes profit, or null — a sold leg's / credit group's target; a bought leg's or debit group's scale-out (the trail when it cannot be split); the tail hedge has none */
  profitTarget: OptionExitRule | null;
  minHoldMinutes: number;
  legsInGroup: number;
}

/** The manager's exit rules for one option leg, with the prices and days
 *  they fire at worked from the leg's own entry premium and DTE. Pure. */
export function optionExitPlan(i: OptionExitInput): OptionExitPlan {
  const side: Side = i.qty < 0 ? "short" : "long";
  const raw = (i.strategy ?? "").trim();
  const strategy = raw && raw !== "unknown" ? raw : null;
  const management = optionManagement(strategy);
  const entry = num(i.entryPrice);
  const cur = num(i.currentPrice);
  const dte = num(i.daysToExpiry);
  const legs = Math.max(1, Math.round(num(i.legsInGroup) ?? 1));
  const until = (d: number) => (dte === null ? null : dte - d);
  const dteRule = (key: OptionExitRuleKey, d: number): OptionExitRule => ({ key, price: null, inDays: until(d), hit: dte !== null && dte <= d });
  const priceRule = (key: OptionExitRuleKey, price: number | null, hit: boolean): OptionExitRule => ({ key, price, inDays: null, hit });
  const rules: OptionExitRule[] = [];
  let profitTarget: OptionExitRule | null = null;
  let partial = false;
  // BOUGHT-OPTION PROFIT TAKING (2026-11-06): the scale-out and the trail, priced
  // for a standalone leg from its premium and the manager's peak; group rules carry no price.
  const takeProfit = (perLeg: boolean): void => {
    if (i.profitTakingOn === false || i.scaleOut === "blocked") return;
    const lots = Math.floor(Math.abs(i.qty) * OPTIONS_PT_SCALE_OUT_FRACTION + 1e-9);
    const done = i.scaleOut === "filled";
    const canSplit = lots >= 1 && i.scaleOut !== "not_possible";
    const soPrice = perLeg && entry !== null && entry > 0 ? entry * (1 + OPTIONS_PT_SCALE_OUT_GAIN) : null;
    const so: OptionExitRule = priceRule("scale_out", soPrice, !done && canSplit && soPrice !== null && cur !== null && cur >= soPrice);
    if (done) so.state = "done";
    else if (!canSplit) so.state = "not_possible";
    const peak = num(i.peakRatio);
    const armed = peak !== null && peak >= 1 + OPTIONS_PT_TRAIL_ARM_GAIN;
    const trailRatio = armed && peak !== null ? peak * (1 - OPTIONS_PT_TRAIL_GIVEBACK) : 1 + OPTIONS_PT_TRAIL_ARM_GAIN;
    const trailPrice = perLeg && entry !== null && entry > 0 ? entry * trailRatio : null;
    const tr: OptionExitRule = { ...priceRule("trail", trailPrice, armed && trailPrice !== null && cur !== null && cur > 0 && cur <= trailPrice), state: armed ? "armed" : "not_armed" };
    rules.push(so, tr);
    partial = !done && canSplit;
    profitTarget = so.state === "not_possible" ? tr : so;
  };
  if (management === "hedge") {
    rules.push(dteRule("hedge_roll", CONVEXITY_ROLL_DTE));
  } else if (management === "group") {
    const credit = OPTIONS_CREDIT_STRATEGIES.includes(strategy ?? "");
    const debit = OPTIONS_DEBIT_STRATEGIES.includes(strategy ?? "");
    rules.push(dteRule("dte_critical", OPTIONS_DTE_CRITICAL));
    if (credit) { profitTarget = priceRule("profit_target", null, false); rules.push(profitTarget); }
    if (strategy === "iron_condor" || strategy === "qqq_iron_condor") rules.push(priceRule("ic_max_loss", null, false));
    if (credit || debit) rules.push(priceRule("loss_limit", null, false));
    rules.push(dteRule("dte_exit", OPTIONS_DTE_EXIT_THRESHOLD));
    if (debit) takeProfit(false);
    if (credit) rules.push(priceRule("edge", null, false));
  } else {
    rules.push(dteRule("dte_critical", OPTIONS_DTE_CRITICAL));
    if (side === "short") rules.push({ key: "assignment", price: null, inDays: until(OPTIONS_ASSIGNMENT_DTE), hit: false });
    rules.push(dteRule("dte_exit", OPTIONS_DTE_EXIT_THRESHOLD));
    if (side === "short") {
      const target = entry !== null && entry > 0 ? entry * (1 - OPTIONS_PROFIT_TARGET_FRAC) : null;
      profitTarget = priceRule("profit_target", target, target !== null && cur !== null && cur > 0 && cur <= target);
      rules.push(profitTarget);
      const stop = entry !== null && entry > 0 ? entry * OPTIONS_LOSS_LIMIT_MULTIPLIER : null;
      rules.push(priceRule("loss_limit", stop, stop !== null && cur !== null && cur >= stop));
    } else {
      takeProfit(true);
      const stop = entry !== null && entry > 0 ? entry * (1 - OPTIONS_BOUGHT_LOSS_LIMIT_FRAC) : null;
      rules.push(priceRule("loss_limit", stop, stop !== null && cur !== null && cur > 0 && cur <= stop));
    }
    rules.push({ key: "gamma", price: null, inDays: until(OPTIONS_GAMMA_DTE), hit: false });
  }
  const killPct = side === "short" ? OPTION_SHORT_KILL_LOSS_PCT : OPTION_LONG_KILL_LOSS_PCT;
  const kill = entry !== null && entry > 0 ? entry * (1 + (side === "short" ? -killPct : killPct) / 100) : null;
  rules.push(priceRule("backstop", kill, kill !== null && cur !== null && cur > 0 && (side === "short" ? cur >= kill : cur <= kill)));
  return { side, management, strategy, rules, partial, profitTarget, minHoldMinutes: OPTIONS_MIN_HOLD_MINUTES, legsInGroup: legs };
}
