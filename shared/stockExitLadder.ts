// shared/stockExitLadder.ts — the stock exit ladder the engine EXECUTES, as
// numbers and pure arithmetic (KNOWN BROKEN #50a, 2026-11-12).
//
// The only code that sells a stock position is server/bot.ts:
//   - checkPositionOnTick, on every 1-minute bar close of a held stock, against
//     the levels syncMonitoredPositions rebuilds every 60 s from
//     voltrade_stop_state.json and the broker's position;
//   - the sync's own POS-KILL backstop (a stock 25 % below entry on two valid
//     readings in a row).
// live_exit_policy.py is that ladder's Python mirror, the stock exit policy of
// record (KNOWN BROKEN #48 step 3). This module is the same ladder for the
// Book: shared/positionLevels.ts turns it into the levels /api/bot/positions
// serves, so the chart, the exit line and the pick cards show what will sell.
//
// bot_engine.manage_positions is NOT mirrored here. It writes the stop state
// this ladder reads, and it also computes its own exits (a regime scalar on
// every stop, a 10 % phase-1 clamp, a 60-minute and a 4-day suppression, a
// −20 % hard stop). Tier 1 never executes those: they are recommendations.
//
// server/bot.ts does not import this module. Its ladder literals are pinned
// verbatim by test_backtest_v2_exit_model.py (the drift guard for
// live_exit_policy.py), and its order paths are frozen. Instead,
// server/stockExitLadder.test.ts guards the copy three ways: it runs
// live_exit_policy.py and compares every constant; it checks that each
// server/bot.ts expression still carries its number; and it cuts the sync's
// level-building block and the tick's decision blocks out of server/bot.ts,
// runs them as written, and compares them with stockLadder() over 4,000
// seeded stop states. A change to the engine's ladder fails that test.
//
// WHAT THE BOOK DOES NOT SHOW (each is a rule of the tick path, not a level):
//   - a tax hold (the stop state's tax_defer_until) defers the scale-outs, the
//     take-profit and the time stop, never a stop;
//   - a leveraged substitute ETF is closed past its maximum hold
//     (server/etfSubstitution.ts), whatever its P&L;
//   - an owner stop turns the tick exits off; a halt does not;
//   - without a stop-state row the engine runs this ladder on ATR_DEFAULT_PCT
//     and no entry date. positionLevels draws nothing until the row exists.
// Short lots: the engine's phase-4 test is written for a long (it measures the
// fall from the highest price). New short stock entries are blocked, so a
// short falls back to the phase 2–3 form here, as it did before.

// Named after live_exit_policy.py's constants; each comment quotes the
// server/bot.ts expression it copies.
export const RISK_ATR_MULT = 2.0; // `ps.initial_risk_pct || atrPct * 2.0` — 1R, written once at entry
export const P1_STOP_ATR_MULT = 2.0; // `Math.max(1.5, Math.min(atrPct * 2.0, 8.0))`
export const P1_STOP_MIN_PCT = 1.5; //   ... lower clamp
export const P1_STOP_MAX_PCT = 8.0; //   ... upper clamp
export const PHASE2_R = 1.0; // `if (peakR >= 3 && pos.phase < 4) pos.phase = 4; else if (peakR >= 2 ...) ... else if (peakR >= 1 ...)`
export const PHASE3_R = 2.0; //   ... phases only move up, from the PEAK R multiple
export const PHASE4_R = 3.0;
export const SCALE1_R = 1.0; // `const scaleOut1R = 1.0`
export const SCALE2_R = { bullish: 3.0, bearish: 1.5, neutral: 2.0 }; // `scaleOut2R = isBullish ? 3.0 : (isBearish ? 1.5 : 2.0)`
export const TRAIL4_ATR_MULT = { bullish: 3.0, bearish: 1.5, neutral: 2.0 }; // `trailAtrMult`, from the highest PRICE
export const P2_TRAIL_MIN_PCT = 1.5; // `Math.max(1.5, atr * 1.5)`, from peak P&L
export const P2_TRAIL_ATR_MULT = 1.5;
export const P3_TRAIL_MIN_PCT = 1.0; // `Math.max(1.0, atr * 1.0)`, from peak P&L
export const P3_TRAIL_ATR_MULT = 1.0;
export const TP_MIN_PCT = 4.0; // `Math.max(4.0, Math.min(atrPct * 3.0, 15.0))`
export const TP_ATR_MULT = 3.0;
export const TP_MAX_PCT = 15.0;
export const TIME_STOP_DAYS = 7; // `daysHeld >= 7 && Math.abs(pnlPct) < 2.0` — calendar days from the entry date
export const TIME_STOP_BAND_PCT = 2.0; //   ... the |P&L| band
export const ATR_DEFAULT_PCT = 2.0; // `ps.current_stop_pct ? ps.current_stop_pct : 2.0`
/** POS-KILL, syncMonitoredPositions' per-position backstop for a stock:
 *  `const STOCK_KILL_LOSS_PCT = -25.0`. Every phase; floor tickers and options
 *  are not judged by it. */
export const STOCK_KILL_LOSS_PCT = -25.0;

const DAY_MS = 86_400_000;

export type Side = "long" | "short";
export type RegimeClass = "bullish" | "bearish" | "neutral";
export type StopBasis = "entry" | "peak_pnl" | "peak_price" | "breakeven";

/** bot.ts isBullish / isBearish on `ps.regime || "NEUTRAL"` (exact labels). */
export function regimeClass(regime: string | null | undefined): RegimeClass {
  const r = regime || "NEUTRAL";
  if (r === "BULL" || r === "NEUTRAL_BULL") return "bullish";
  if (r === "BEAR" || r === "PANIC" || r === "CAUTION") return "bearish";
  return "neutral";
}

/** Price at a P&L % from entry, for either side. */
export function pricePct(entry: number, pct: number, side: Side): number {
  return side === "long" ? entry * (1 + pct / 100) : entry * (1 - pct / 100);
}

/** `peakR >= 3 ? 4 : peakR >= 2 ? 3 : peakR >= 1 ? 2 : 1` */
export function phaseFromPeakR(peakR: number): 1 | 2 | 3 | 4 {
  return peakR >= PHASE4_R ? 4 : peakR >= PHASE3_R ? 3 : peakR >= PHASE2_R ? 2 : 1;
}

/** The stop's width in a phase, in %: 2×ATR clamped 1.5–8 from entry (1),
 *  max(1.5, 1.5×ATR) from peak P&L (2), max(1.0, ATR) from peak P&L (3), the
 *  regime's multiple of ATR from the highest price (4). No regime scalar. */
export function stopWidthPct(phase: number, atrPct: number, cls: RegimeClass): number {
  if (phase >= 4) return atrPct * TRAIL4_ATR_MULT[cls];
  if (phase >= 3) return Math.max(P3_TRAIL_MIN_PCT, atrPct * P3_TRAIL_ATR_MULT);
  if (phase >= 2) return Math.max(P2_TRAIL_MIN_PCT, atrPct * P2_TRAIL_ATR_MULT);
  return Math.max(P1_STOP_MIN_PCT, Math.min(atrPct * P1_STOP_ATR_MULT, P1_STOP_MAX_PCT));
}

/** The fixed target of phases 1–2, in %. It fires only after both scale-outs
 *  and only while the phase is below 3 (see stockLadder for when it exists). */
export function takeProfitPct(atrPct: number): number {
  return Math.max(TP_MIN_PCT, Math.min(atrPct * TP_ATR_MULT, TP_MAX_PCT));
}

/** The tick's `Math.floor((Date.now() - new Date(pos.entryDate).getTime()) / 86400000)`.
 *  null without a usable entry date: the engine then counts from today, so its
 *  time stop cannot arm. */
export function ladderDaysHeld(entryDate: string | null | undefined, nowMs: number): number | null {
  if (!entryDate || typeof entryDate !== "string") return null;
  const t = new Date(entryDate).getTime();
  return Number.isFinite(t) ? Math.floor((nowMs - t) / DAY_MS) : null;
}

/** The stop-state fields syncMonitoredPositions reads (one row of
 *  voltrade_stop_state.json; older rows may lack newer fields). */
export interface LadderRow {
  current_stop_pct?: number; // the raw ATR % ("bot.ts computes final stop")
  initial_risk_pct?: number;
  highest_pnl?: number;
  highest_price?: number;
  phase?: number;
  regime?: string;
  original_qty?: number;
  remaining_qty?: number;
  scales_completed?: number;
  breakeven_active?: boolean;
  entry_date?: string;
}

export interface LadderInput {
  row: LadderRow | null | undefined;
  side: Side;
  entry: number; // the broker's average entry price
  current: number | null; // the last price (the tick's close); null = no mark
  qty: number; // shares held at the broker, unsigned
  pnlPct: number | null; // unrealized P&L in % of entry; worked from the prices when null
  nowMs: number;
}

export interface LadderScaleOut {
  n: 1 | 2;
  rMultiple: number; // 1 for the first third; the regime's second threshold for the other
  pct: number; // P&L % at the trigger
  price: number;
  done: boolean;
  applicable: boolean; // done, or the engine can still sell it (more than 1 share left and a third of at least 1 share)
}

export interface StockLadder {
  atrPct: number;
  atrDefaulted: boolean; // the stop state carried no ATR: the engine's 2 % default
  riskPct: number; // 1R, in %
  pnlPct: number | null;
  rMultiple: number | null;
  highestPnl: number;
  highestPrice: number | null;
  peakR: number;
  phase: 1 | 2 | 3 | 4;
  regimeClass: RegimeClass;
  trailAtrMult: number;
  scaleOut2R: number;
  stopPct: number;
  stopFrom: StopBasis;
  stopPrice: number | null;
  stopHit: boolean; // the current reading already meets the engine's stop test: its next tick sells
  breakevenActive: boolean;
  takeProfitPct: number | null; // null in phase 3+, for a lot that cannot sell both thirds first, and at or beyond +2R
  takeProfitPrice: number | null;
  takeProfitArmed: boolean;
  scaleOuts: LadderScaleOut[];
  thirdQty: number;
  originalQty: number;
  remainingQty: number;
  scalesCompleted: number;
  daysHeld: number | null;
  timeStopArmed: boolean;
  backstopPct: number; // positive: the loss, in %, that POS-KILL closes at
  backstopPrice: number;
}

/** JS `x || fallback` on a JSON number: a finite, non-zero number or null. */
const truthy = (v: unknown): number | null => {
  const n = typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
  return Number.isFinite(n) && n !== 0 ? n : null;
};

/** The ladder as the engine holds it right now: the monitored position
 *  syncMonitoredPositions would build from this stop-state row and broker
 *  position, advanced by checkPositionOnTick's own high-water marks and phase
 *  at the current price, with the price each test fires at. Pure. */
export function stockLadder(i: LadderInput): StockLadder {
  const ps = i.row ?? {};
  const side = i.side;
  const long = side === "long";
  const entry = i.entry;
  const current = i.current;
  const storedAtr = truthy(ps.current_stop_pct);
  const atrPct = storedAtr ?? ATR_DEFAULT_PCT;
  const riskPct = truthy(ps.initial_risk_pct) ?? atrPct * RISK_ATR_MULT;
  const pnlPct = i.pnlPct !== null && Number.isFinite(i.pnlPct) ? i.pnlPct
    : current !== null && entry > 0 ? (long ? (current - entry) / entry : (entry - current) / entry) * 100 : null;

  // high-water marks: `Math.max(ps.highest_pnl || 0, pnlPct)`, `Math.max(ps.highest_price || current, current)`
  const highestPnl = Math.max(truthy(ps.highest_pnl) ?? 0, pnlPct ?? -Infinity);
  const storedHigh = truthy(ps.highest_price);
  const highestPrice = current === null ? storedHigh : Math.max(storedHigh ?? current, current);
  const rMultiple = pnlPct === null ? null : riskPct > 0 ? pnlPct / riskPct : 0;
  const peakR = riskPct > 0 ? highestPnl / riskPct : 0;
  // the sync takes `ps.phase || <derived>`; the tick only ever moves it up
  const derived = phaseFromPeakR(peakR);
  const stored = truthy(ps.phase);
  const phase = Math.min(4, Math.max(stored !== null ? Math.floor(stored) : derived, derived)) as 1 | 2 | 3 | 4;

  const cls = regimeClass(typeof ps.regime === "string" ? ps.regime : null);
  const trailAtrMult = TRAIL4_ATR_MULT[cls];
  const scaleOut2R = SCALE2_R[cls];

  // share counts: before the first scale-out the whole live position is the position
  const qty = Math.trunc(Math.abs(i.qty));
  const scalesCompleted = Math.max(0, Math.round(truthy(ps.scales_completed) ?? 0));
  const originalQty = scalesCompleted === 0 ? qty : truthy(ps.original_qty) ?? qty;
  const remainingQty = scalesCompleted === 0 ? qty : Math.min(truthy(ps.remaining_qty) ?? qty, qty);
  const breakevenActive = !!ps.breakeven_active;

  // scale-outs: `pos.scalesCompleted < 2 && pos.remainingQty > 1`, `thirdQty = Math.floor(pos.originalQty / 3)` >= 1
  const thirdQty = Math.floor(originalQty / 3);
  const canScaleOut = scalesCompleted < 2 && remainingQty > 1 && thirdQty >= 1;
  const scaleOut = (n: 1 | 2, r: number): LadderScaleOut => ({
    n, rMultiple: r, pct: riskPct * r, price: pricePct(entry, riskPct * r, side), done: scalesCompleted >= n, applicable: scalesCompleted >= n || canScaleOut,
  });
  const scaleOuts = [scaleOut(1, SCALE1_R), scaleOut(2, scaleOut2R)];

  // the stop, and the price its test fires at
  const stopPct = stopWidthPct(phase, atrPct, cls);
  let stopFrom: StopBasis;
  let stopPrice: number | null;
  let stopHit: boolean;
  if (phase >= 4 && long) {
    // `((pos.highestPrice - currentPrice) / pos.highestPrice) * 100 >= stopPct`
    stopFrom = "peak_price";
    stopPrice = highestPrice !== null ? highestPrice * (1 - stopPct / 100) : null;
    stopHit = current !== null && highestPrice !== null && highestPrice > 0 && ((highestPrice - current) / highestPrice) * 100 >= stopPct;
  } else if (phase >= 2) {
    // `pos.highestPnl - pnlPct >= stopPct` (also a short's phase 4 here, see the header)
    stopFrom = phase >= 4 ? "peak_price" : "peak_pnl";
    stopPrice = pricePct(entry, highestPnl - stopPct, side);
    stopHit = pnlPct !== null && highestPnl - pnlPct >= stopPct;
  } else {
    // `pnlPct <= -stopPct`
    stopFrom = "entry";
    stopPrice = pricePct(entry, -stopPct, side);
    stopHit = pnlPct !== null && pnlPct <= -stopPct;
  }
  // breakeven, phases 2+ only: `!shouldStop && breakevenFloor && pnlPct < 0`
  if (breakevenActive && phase >= 2) {
    if (stopPrice === null || (long ? entry > stopPrice : entry < stopPrice)) {
      stopPrice = entry;
      stopFrom = "breakeven";
    }
    if (pnlPct !== null && pnlPct < 0) stopHit = true;
  }

  // take-profit: `pnlPct >= pos.tpPct && pos.phase < 3 && pos.scalesCompleted >= 2`. Two cases can never
  // meet that test, so they have no target: a lot that cannot sell both thirds, and a target at or beyond
  // +2R (a tick that high has already moved the phase to 3).
  const tpPct = takeProfitPct(atrPct);
  const tp = phase < 3 && (scalesCompleted >= 2 || canScaleOut) && tpPct < PHASE3_R * riskPct ? tpPct : null;
  // time stop: `daysHeld >= 7 && Math.abs(pnlPct) < 2.0`
  const daysHeld = ladderDaysHeld(ps.entry_date, i.nowMs);

  return {
    atrPct, atrDefaulted: storedAtr === null, riskPct, pnlPct, rMultiple, highestPnl, highestPrice, peakR, phase,
    regimeClass: cls, trailAtrMult, scaleOut2R, stopPct, stopFrom, stopPrice, stopHit, breakevenActive,
    takeProfitPct: tp, takeProfitPrice: tp !== null ? pricePct(entry, tp, side) : null, takeProfitArmed: scalesCompleted >= 2,
    scaleOuts, thirdQty, originalQty, remainingQty, scalesCompleted,
    daysHeld, timeStopArmed: daysHeld !== null && daysHeld >= TIME_STOP_DAYS,
    backstopPct: -STOCK_KILL_LOSS_PCT, backstopPrice: pricePct(entry, STOCK_KILL_LOSS_PCT, side),
  };
}
