// shared/positionLevels.ts — the engine's exit levels for ONE position,
// computed from its stop state with the SAME formulas bot_engine.py
// manage_positions applies (read 2026-10-21). One source of truth for two
// readers: server/bot.ts /api/bot/positions (stopPrice, takeProfitPrice and
// the `levels` block the Bot page chart draws) and the chart's own option-
// leg arithmetic. Pure; no I/O; no clock of its own (nowMs is an input);
// pinned by server/positionLevels.test.ts.
//
// If manage_positions changes a formula, this file changes with it — or the
// chart draws a level the engine will not honour. Every constant below that
// is NOT read from system_config is hard-coded in the engine itself, and is
// named after the engine's own identifier.

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
  short_hold_suppression?: string;
}

/** The system_config values the engine reads on this path, resolved for the
 *  position's regime by the caller (STOP_SCALAR_<regime>, missing → 1.0 as in
 *  the engine's `.get(key, 1.0)`; POSITION_HARD_STOP_*). */
export interface LevelConfig {
  stopScalar: number;
  hardStopPct: number | null;
  hardStopEnabled: boolean;
}

export interface LevelInput {
  entry: number;
  current: number;
  qty: number; // negative = short
  pnlPct: number | null; // Alpaca unrealized_plpc × 100 (the engine's pnl_pct); recomputed from prices when null
  stop: StopStateRow | null | undefined;
  config: LevelConfig | null | undefined;
  nowMs?: number;
}

export type Side = "long" | "short";
export type StopBasis = "entry" | "peak_pnl" | "peak_price" | "breakeven";

export interface ScaleOut {
  n: 1 | 2;
  rMultiple: number; // 1 for the first third; the regime's 2R threshold for the second
  pct: number; // P&L % at the trigger
  price: number;
  done: boolean;
  applicable: boolean; // the engine skips scale-outs when the lot cannot be split (original_qty ≤ 1)
}

export interface StopSuppression {
  until: "60min" | "day4";
  reason: string;
  /** the loss that still closes the position while the suppression holds */
  overridePct: number;
  minutesLeft: number | null;
  daysLeft: number | null;
}

export interface PositionLevels {
  recorded: boolean; // false until manage_positions has ticked once for this position (no stop state row / no ATR)
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
  breakevenActive: boolean;
  stopSuppressed: StopSuppression | null;
  hardFloorPct: number | null; // phase 1 only
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

// Engine constants that live in bot_engine.py itself (not system_config):
export const SHORT_HOLD_DAYS = 4; // _in_short_hold_window = days_held < 4
export const SHORT_HOLD_DEEP_LOSS_PCT = 15; // SHORT_HOLD_DEEP_LOSS_OVERRIDE_PCT
export const MIN_HOLD_MINUTES = 60; // _MIN_EQUITY_HOLD_MINUTES
export const MIN_HOLD_CATASTROPHE_PCT = 10; // _is_catastrophic = pnl_pct <= -10.0
export const TIME_STOP_DAYS_ENGINE = 7; // `days_held >= 7` — hard-coded; TIME_STOP_DAYS in system_config is unread on this path (open_questions.md 2026-10-21)
export const TIME_STOP_WITHIN_PCT = 2; // abs(pnl_pct) < 2.0
export const HARD_STOP_PCT_DEFAULT = 20; // BASE_CONFIG.get("POSITION_HARD_STOP_PCT", 20.0)
/** options_manager.PROFIT_TARGET_PCT — the manager closes sold premium at
 *  this share of max profit; an option-price level, so the chart prints it
 *  in the legend rather than drawing it on the underlying. */
export const OPTIONS_PROFIT_TARGET_FRAC = 0.5;

const num = (v: unknown): number | null => {
  if (v === null || v === undefined || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
};

/** The regime-adaptive multipliers manage_positions derives before the loop. */
export function regimeMults(regime: string | null | undefined): { bullish: boolean; bearish: boolean; trailAtrMult: number; scaleOut2R: number } {
  const r = (regime ?? "").toUpperCase();
  const bullish = r === "BULL" || r === "NEUTRAL_BULL";
  const bearish = r === "BEAR" || r === "PANIC" || r === "CAUTION";
  const m = bullish ? 3.0 : bearish ? 1.5 : 2.0;
  return { bullish, bearish, trailAtrMult: m, scaleOut2R: m };
}

/** Price at a P&L % from entry, for either side. */
export function pricePct(entry: number, pct: number, side: Side): number {
  return side === "long" ? entry * (1 + pct / 100) : entry * (1 - pct / 100);
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

function daysSince(dateYmd: string | undefined, nowMs: number): number | null {
  if (!dateYmd || !/^\d{4}-\d{2}-\d{2}$/.test(dateYmd)) return null;
  const t = Date.parse(dateYmd + "T00:00:00Z");
  if (!Number.isFinite(t)) return null;
  return Math.floor((nowMs - t) / 86_400_000);
}

export function positionLevels(i: LevelInput): PositionLevels {
  const nowMs = i.nowMs ?? Date.now();
  const side: Side = i.qty < 0 ? "short" : "long";
  const s = i.stop ?? null;
  const entry = num(i.entry);
  const current = num(i.current);
  const atr = num(s?.current_stop_pct);
  const recorded = !!s && atr !== null && entry !== null && entry > 0;
  const regime = typeof s?.regime === "string" && s.regime ? s.regime : null;
  const { trailAtrMult, scaleOut2R } = regimeMults(regime);
  const scalar = num(i.config?.stopScalar) ?? 1.0;

  const pnl = num(i.pnlPct) ?? (entry && current ? (side === "long" ? current / entry - 1 : 1 - current / entry) * 100 : null);
  const risk = num(s?.initial_risk_pct) ?? (atr !== null ? atr * 2.0 : null);
  const storedPeak = num(s?.highest_pnl);
  const highestPnl = storedPeak === null && pnl === null ? null : Math.max(storedPeak ?? -Infinity, pnl ?? -Infinity);
  const storedHigh = num(s?.highest_price);
  const highestPrice = storedHigh === null && current === null ? null : Math.max(storedHigh ?? -Infinity, current ?? -Infinity);
  const rMultiple = risk && pnl !== null ? pnl / risk : null;
  const peakR = risk && highestPnl !== null ? highestPnl / risk : null;
  // the engine re-derives the phase from peak_r every tick; the stored phase
  // is the fallback when the peak cannot be computed
  const phase = peakR === null ? Math.max(1, Math.min(4, Math.round(num(s?.phase) ?? 1))) : peakR >= 3 ? 4 : peakR >= 2 ? 3 : peakR >= 1 ? 2 : 1;

  let stopPct: number | null = null;
  let stopFrom: StopBasis | null = null;
  let stopPrice: number | null = null;
  if (recorded && atr !== null && entry !== null) {
    if (phase >= 4) {
      stopPct = atr * trailAtrMult * scalar;
      stopFrom = "peak_price";
      // the engine trails the highest PRICE in phase 4 (long-centric); a
      // short falls back to the P&L form it would use in phases 2–3
      stopPrice = side === "long" && highestPrice !== null ? highestPrice * (1 - stopPct / 100) : highestPnl !== null ? pricePct(entry, highestPnl - stopPct, side) : null;
    } else if (phase === 3) {
      stopPct = Math.max(1.0, atr * 1.0 * scalar);
      stopFrom = "peak_pnl";
      stopPrice = highestPnl !== null ? pricePct(entry, highestPnl - stopPct, side) : null;
    } else if (phase === 2) {
      stopPct = Math.max(1.5, atr * 1.5 * scalar);
      stopFrom = "peak_pnl";
      stopPrice = highestPnl !== null ? pricePct(entry, highestPnl - stopPct, side) : null;
    } else {
      stopPct = Math.max(1.5, Math.min(atr * 2.0 * scalar, 10.0));
      stopFrom = "entry";
      stopPrice = pricePct(entry, -stopPct, side);
    }
  }
  const breakevenActive = !!s?.breakeven_active;
  if (breakevenActive && phase >= 2 && entry !== null && (stopPrice === null || (side === "long" ? entry > stopPrice : entry < stopPrice))) {
    stopPrice = entry;
    stopFrom = "breakeven";
  }

  const hardEnabled = i.config ? !!i.config.hardStopEnabled : true;
  const hardPct = i.config && i.config.hardStopPct !== null ? i.config.hardStopPct : HARD_STOP_PCT_DEFAULT;
  const hardFloorPct = recorded && phase < 2 && hardEnabled ? hardPct : null;
  const hardFloorPrice = hardFloorPct !== null && entry !== null ? pricePct(entry, -hardFloorPct, side) : null;

  const daysHeld = num(s?.days_held) ?? daysSince(s?.entry_date, nowMs);
  const entryMs = parseEngineTime(s?.entry_timestamp);
  const heldMinutes = entryMs !== null ? Math.max(0, (nowMs - entryMs) / 60_000) : null;
  let stopSuppressed: StopSuppression | null = null;
  if (recorded && heldMinutes !== null && heldMinutes < MIN_HOLD_MINUTES) {
    stopSuppressed = { until: "60min", reason: "60-minute minimum hold: no exit yet unless the loss reaches the catastrophe line", overridePct: MIN_HOLD_CATASTROPHE_PCT, minutesLeft: MIN_HOLD_MINUTES - heldMinutes, daysLeft: null };
  } else if (recorded && daysHeld !== null && daysHeld < SHORT_HOLD_DAYS && phase < 2 && !breakevenActive) {
    stopSuppressed = { until: "day4", reason: "short-hold grace: the phase-1 stop waits until day 4 unless the loss reaches the deep-loss line", overridePct: SHORT_HOLD_DEEP_LOSS_PCT, minutesLeft: null, daysLeft: SHORT_HOLD_DAYS - daysHeld };
  }

  const scalesCompleted = Math.max(0, Math.round(num(s?.scales_completed) ?? 0));
  const takeProfitPct = recorded && atr !== null && phase < 3 ? Math.max(4.0, Math.min(atr * 3.0, 15.0)) : null;
  const takeProfitPrice = takeProfitPct !== null && entry !== null ? pricePct(entry, takeProfitPct, side) : null;
  const originalQty = num(s?.original_qty) ?? (Number.isFinite(i.qty) ? Math.abs(i.qty) : null);
  const remainingQty = num(s?.remaining_qty) ?? (Number.isFinite(i.qty) ? Math.abs(i.qty) : null);
  const applicable = (originalQty ?? 0) > 1;
  const scaleOuts: ScaleOut[] = recorded && risk !== null && entry !== null ? [
    { n: 1, rMultiple: 1, pct: risk, price: pricePct(entry, risk, side), done: scalesCompleted >= 1, applicable },
    { n: 2, rMultiple: scaleOut2R, pct: risk * scaleOut2R, price: pricePct(entry, risk * scaleOut2R, side), done: scalesCompleted >= 2, applicable },
  ] : [];

  return {
    recorded, side, phase, regime, atrPct: atr, riskPct: risk, pnlPct: pnl, rMultiple, peakR, highestPnl, highestPrice,
    stopPct, stopPrice, stopFrom, breakevenActive, stopSuppressed, hardFloorPct, hardFloorPrice,
    takeProfitPct, takeProfitPrice, takeProfitArmed: scalesCompleted >= 2, scaleOuts, trailAtrMult, scaleOut2R,
    originalQty, remainingQty, scalesCompleted, daysHeld, heldMinutes,
    timeStop: { day: TIME_STOP_DAYS_ENGINE, withinPct: TIME_STOP_WITHIN_PCT, armed: daysHeld !== null && daysHeld >= TIME_STOP_DAYS_ENGINE },
  };
}

/** The LevelConfig for a position's regime from the BASE_CONFIG subset the
 *  positions route reads (STOP_SCALAR_* and POSITION_HARD_STOP_*). An empty
 *  or failed read gives null, and positionLevels falls back to the engine's
 *  own defaults (scalar 1.0, hard stop 20 % on). */
export function levelConfigFromBaseConfig(cfg: Record<string, unknown> | null | undefined, regime: unknown): LevelConfig | null {
  if (!cfg || typeof cfg !== "object" || !Object.keys(cfg).length || "error" in cfg) return null;
  const sc = num(cfg[`STOP_SCALAR_${String(regime ?? "").toUpperCase()}`]);
  const hp = num(cfg.POSITION_HARD_STOP_PCT);
  const en = cfg.POSITION_HARD_STOP_ENABLED;
  return { stopScalar: sc ?? 1.0, hardStopPct: hp, hardStopEnabled: en === undefined || en === null ? true : !!en };
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
  /** options_manager.PROFIT_TARGET_PCT for a sold leg; null for a bought leg — the manager has no profit-taking rule for one (it rides to a DTE or loss rule) */
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
}

export type OptionExitRuleKey = "dte_critical" | "assignment" | "dte_exit" | "profit_target" | "loss_limit" | "ic_max_loss" | "gamma" | "edge" | "hedge_roll" | "backstop";

export interface OptionExitRule {
  key: OptionExitRuleKey;
  /** the leg's price (per share) the rule fires at, for a price rule on a standalone leg; null for DTE rules and for group rules (judged on the strategy's combined P&L) */
  price: number | null;
  /** days until a DTE rule arms (≤ 0 = due on the next manage tick); null for a price rule */
  inDays: number | null;
  /** the current reading already satisfies the trigger — the manager acts on its next tick (after the 60-minute hold, critical DTE excepted) */
  hit: boolean;
}

export interface OptionExitPlan {
  side: Side;
  management: OptionManagement;
  /** the recorded strategy, or null when the state never recorded one */
  strategy: string | null;
  /** the manager's checks in the order it runs them */
  rules: OptionExitRule[];
  /** every option exit is the whole lot — a single leg's full qty, or the whole multi-leg strategy via one order; the manager never sells part of an option position */
  partial: false;
  /** the rule that takes profit, or null — a bought leg and the tail hedge have none (they ride until a DTE or loss rule) */
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
      const stop = entry !== null && entry > 0 ? entry * (1 - OPTIONS_BOUGHT_LOSS_LIMIT_FRAC) : null;
      rules.push(priceRule("loss_limit", stop, stop !== null && cur !== null && cur > 0 && cur <= stop));
    }
    rules.push({ key: "gamma", price: null, inDays: until(OPTIONS_GAMMA_DTE), hit: false });
  }
  const killPct = side === "short" ? OPTION_SHORT_KILL_LOSS_PCT : OPTION_LONG_KILL_LOSS_PCT;
  const kill = entry !== null && entry > 0 ? entry * (1 + (side === "short" ? -killPct : killPct) / 100) : null;
  rules.push(priceRule("backstop", kill, kill !== null && cur !== null && cur > 0 && (side === "short" ? cur >= kill : cur <= kill)));
  return { side, management, strategy, rules, partial: false, profitTarget, minHoldMinutes: OPTIONS_MIN_HOLD_MINUTES, legsInGroup: legs };
}
