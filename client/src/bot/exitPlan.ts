// Bot page Book — the words under each position for WHAT it is and WHEN the
// engine sells it (2026-11-02, human: "how do I know what type of option I
// have, did I sell a put or buy, was it a single option or a strategy … and
// when would you sell the position or a portion, is that shown"). Pure
// presentation over shared/positionLevels: every threshold is the engine
// mirror there, never retyped here. Pinned by exitPlan.test.ts.
import {
  optionManagement, type OptionExitPlan, type OptionExitRule, type PositionLevels,
  OPTIONS_PROFIT_TARGET_FRAC, OPTIONS_LOSS_LIMIT_MULTIPLIER, OPTIONS_BOUGHT_LOSS_LIMIT_FRAC, OPTIONS_DTE_EXIT_THRESHOLD, OPTIONS_DTE_CRITICAL,
  OPTIONS_IC_MAX_LOSS_EXIT_FRAC, OPTIONS_GAMMA_THRESHOLD, OPTIONS_GAMMA_DTE, OPTIONS_ASSIGNMENT_DTE, OPTIONS_CREDIT_STRATEGIES,
  OPTION_LONG_KILL_LOSS_PCT, OPTION_SHORT_KILL_LOSS_PCT, CONVEXITY_ROLL_DTE,
} from "../../../shared/positionLevels";
import { MINUS } from "./format";

export interface PlanSegment { text: string; hit?: boolean }
export const joinSegments = (segs: readonly PlanSegment[]): string => segs.map((s) => s.text).join(" · ");

/** The engine's strategy names (options_execution / options_scanner /
 *  instrument_selector / bot_engine write them into the options state) in
 *  plain words. An unlisted name prints as written; null and "unknown" (a leg
 *  the Node tier dispatcher opened — its state never records a strategy)
 *  read "setup not recorded". */
export const STRATEGY_WORDS: Readonly<Record<string, string>> = {
  sell_cash_secured_put: "cash-secured put", csp_normal_market: "cash-secured put", csp_options: "cash-secured put", csp: "cash-secured put",
  covered_call: "covered call", sell_covered_call: "covered call",
  buy_call: "long call", long_call: "long call", buy_put: "long put", long_put: "long put",
  bull_put_credit_spread: "bull put credit spread", qqq_iron_condor: "QQQ weekly iron condor", iron_condor: "iron condor",
  bull_call_spread: "bull call spread", bear_put_spread: "bear put spread", buy_straddle: "long straddle", short_straddle: "short straddle",
  calendar_spread: "calendar spread", diagonal_spread: "diagonal spread", convexity_hedge: "QQQ tail hedge",
};
export function strategyWords(strategy: string | null | undefined): string {
  const s = (strategy ?? "").trim();
  if (!s || s === "unknown") return "setup not recorded";
  return STRATEGY_WORDS[s] ?? s.replace(/_/g, " ");
}

const money = (v: number) => `$${v.toFixed(2)}`;
const pct = (f: number) => `${Math.round(f * 100)}%`;

export interface OptionWhatInput {
  qty: number; // negative = sold
  optionType: string | null | undefined; // "CALL" | "PUT" from the OCC symbol
  daysToExpiry: number | null | undefined;
  strategy: string | null | undefined;
  /** legs of the same underlying + strategy held right now, this one included */
  legsInGroup?: number;
}

/** "sold put · 30d to expiry · cash-secured put · single leg" — what the
 *  position IS: side, type, time left, the recorded setup, and whether the
 *  manager runs it alone or as one unit with its sibling legs. */
export function optionWhatLine(i: OptionWhatInput): PlanSegment[] {
  const side = i.qty < 0 ? "sold" : "bought";
  const type = i.optionType === "PUT" ? "put" : i.optionType === "CALL" ? "call" : "option";
  const dte = i.daysToExpiry;
  const hasDte = dte !== null && dte !== undefined && Number.isFinite(dte);
  const legs = Math.max(1, Math.round(i.legsInGroup ?? 1));
  const mgmt = optionManagement(i.strategy);
  return [
    { text: `${side} ${type}` },
    { text: !hasDte ? "expiry not recorded" : dte <= 0 ? "expires today" : `${Math.round(dte)}d to expiry` },
    { text: strategyWords(i.strategy) },
    { text: mgmt === "hedge" ? "overlay rolls it" : mgmt === "group" ? `${legs} ${legs === 1 ? "leg" : "legs"} held, closed as one` : "single leg" },
  ];
}

function ruleWords(r: OptionExitRule, plan: OptionExitPlan): PlanSegment {
  const short = plan.side === "short";
  const timed = (label: string, due: string) => (r.inDays === null ? label : r.hit ? `${label} · ${due}` : `${label} in ${Math.ceil(r.inDays)}d`);
  const at = r.price !== null ? ` at ${money(r.price)}` : "";
  const hit = r.hit ? " · hit" : "";
  switch (r.key) {
    case "dte_critical": return { text: timed(`force close ≤${OPTIONS_DTE_CRITICAL} DTE`, "due now"), hit: r.hit };
    case "assignment": return { text: `deep-ITM roll ≤${OPTIONS_ASSIGNMENT_DTE} DTE` };
    case "dte_exit": return { text: timed(plan.management === "group" ? `close all at ${OPTIONS_DTE_EXIT_THRESHOLD} DTE` : short ? `roll/close at ${OPTIONS_DTE_EXIT_THRESHOLD} DTE` : `close at ${OPTIONS_DTE_EXIT_THRESHOLD} DTE`, "due now"), hit: r.hit };
    case "profit_target": return { text: r.price === null ? `target: ${pct(OPTIONS_PROFIT_TARGET_FRAC)} of net credit` : `target: keep ${pct(OPTIONS_PROFIT_TARGET_FRAC)}${at}${hit}`, hit: r.hit };
    case "ic_max_loss": return { text: `stop: ${pct(OPTIONS_IC_MAX_LOSS_EXIT_FRAC)} of max loss` };
    case "loss_limit":
      if (plan.management === "group") return { text: OPTIONS_CREDIT_STRATEGIES.includes(plan.strategy ?? "") ? `stop: loss of ${OPTIONS_LOSS_LIMIT_MULTIPLIER}× the premium` : `stop: ${MINUS}${pct(OPTIONS_BOUGHT_LOSS_LIMIT_FRAC)} of the debit` };
      return { text: short ? `stop: ${OPTIONS_LOSS_LIMIT_MULTIPLIER}× credit${at}${hit}` : `stop: ${MINUS}${pct(OPTIONS_BOUGHT_LOSS_LIMIT_FRAC)}${at}${hit}`, hit: r.hit };
    case "gamma": return { text: `gamma >${OPTIONS_GAMMA_THRESHOLD} inside ${OPTIONS_GAMMA_DTE} DTE` };
    case "edge": return { text: "edge gone (VRP turns negative)" };
    case "hedge_roll": return { text: timed(`rolled under ${CONVEXITY_ROLL_DTE} DTE`, "roll due"), hit: r.hit };
    case "backstop": return { text: `floor ${MINUS}${Math.abs(short ? OPTION_SHORT_KILL_LOSS_PCT : OPTION_LONG_KILL_LOSS_PCT)}%${at}${hit}`, hit: r.hit };
  }
}

/** "whole lot at once · no profit target · force close ≤5 DTE in 25d · close
 *  at 21 DTE in 9d · stop: −50% at $1.60 · gamma … · floor −75% at $0.80" —
 *  WHEN the engine sells, in the manager's own order, each price worked from
 *  the leg's entry premium; a rule the current reading already satisfies is
 *  flagged hit. The first segment answers "or a portion": never. */
export function optionExitSegments(plan: OptionExitPlan): PlanSegment[] {
  const segs: PlanSegment[] = [];
  segs.push({ text: plan.management === "group" ? (plan.legsInGroup > 1 ? `all ${plan.legsInGroup} legs as one` : "strategy closed as one unit") : plan.management === "hedge" ? "insurance, not a trade" : "whole lot at once" });
  if (plan.management === "hedge") segs.push({ text: "no target, no stop" });
  else if (!plan.profitTarget) segs.push({ text: "no profit target" });
  for (const r of plan.rules) segs.push(ruleWords(r, plan));
  return segs;
}

const STOP_FROM: Record<string, string> = { entry: "from entry", peak_pnl: "trails peak P&L", peak_price: "trails the high", breakeven: "breakeven" };

/** "stop $719.08 from entry · floor $593.06 (−20%) · ⅓ at $763.56 (+1.0R) ·
 *  ⅓ at $785.80 (+2.0R) · target $774.68 after ⅔ out · time stop d7 if ±2%"
 *  — manage_positions' ladder for a share lot, from the levels the server
 *  already computes (shared/positionLevels). The thirds answer "or a
 *  portion": yes, two of them, then the trail takes the rest. */
export function stockExitSegments(lv: PositionLevels | null | undefined): PlanSegment[] {
  if (!lv || !lv.recorded) return [{ text: "stop state not recorded yet" }, { text: "the engine writes it on its first manage tick after entry" }];
  const segs: PlanSegment[] = [];
  if (lv.stopPrice !== null) {
    const sup = lv.stopSuppressed;
    const waits = sup ? (sup.until === "60min" ? ` · waits ${Math.ceil(sup.minutesLeft ?? 0)} min (minimum hold)` : ` · waits ${sup.daysLeft ?? "?"}d (grace to day 4)`) : "";
    segs.push({ text: `stop ${money(lv.stopPrice)} ${STOP_FROM[lv.stopFrom ?? "entry"] ?? "from entry"}${waits}` });
    if (sup) segs.push({ text: `exit now only below ${MINUS}${sup.overridePct}%` });
  } else segs.push({ text: "stop not set" });
  if (lv.hardFloorPrice !== null && lv.hardFloorPct !== null) segs.push({ text: `floor ${money(lv.hardFloorPrice)} (${MINUS}${lv.hardFloorPct}%)` });
  if (lv.scaleOuts.length && !lv.scaleOuts[0].applicable) segs.push({ text: "no scale-out (lot of 1)" });
  else for (const so of lv.scaleOuts) segs.push({ text: so.done ? `⅓ sold at ${money(so.price)}` : `⅓ at ${money(so.price)} (+${so.rMultiple.toFixed(1)}R)` });
  if (lv.takeProfitPrice !== null && lv.takeProfitPct !== null) segs.push({ text: `target ${money(lv.takeProfitPrice)}${lv.takeProfitArmed ? " armed" : " after ⅔ out"}` });
  else if (lv.phase >= 3) segs.push({ text: "trail takes the rest" });
  segs.push({ text: `time stop d${lv.timeStop.day} if ±${lv.timeStop.withinPct}%${lv.timeStop.armed ? " · armed" : ""}`, hit: lv.timeStop.armed });
  return segs;
}
