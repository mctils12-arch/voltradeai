// server/scaleIn.ts — scale-in by thirds (2026-11-06, refocus spec section 4:
// "scale in systematically (e.g. thirds)").
//
// A new stock/ETF satellite entry (server/bot.ts executeTrades, morning queue)
// buys its FIRST third now. The plan for the other two is persisted
// (DATA_DIR/voltrade_scale_in_state.json; position_mgmt.py counts its unbought
// tranches as pending cash) and executeScaleInTranches checks it every
// in-hours scan:
//   tranche 2 — CONFIRMATION: a later session than the entry, the price within
//               ±CONFIRM_BAND_PCT of the first fill, above the plan's stop;
//   tranche 3 — a FURTHER CONDITION, after tranche 2: a pullback toward support
//               (at least PULLBACK_FRAC of the way from the first fill to the
//               stop, still above it) or a breakout continuation (a new high
//               since entry, at least BREAKOUT_PCT over the first fill).
// The plan is cancelled when the stop hits (the position is gone, or the price
// is at or below the plan's stop) or the thesis breaks (a BEAR/PANIC regime, a
// scale-out has begun — the trade is taking profit — or MAX_AGE_DAYS passed).
// NEUTRAL (no new stock trades) only makes it wait. Every tranche and
// cancellation is logged (positionMgmt.appendPositionMgmtLog) by the caller.
//
// Pure except loadScaleInState / saveScaleInState.

import fs from "node:fs";
import path from "node:path";

export const SCALE_IN_STATE_FILE = "voltrade_scale_in_state.json";
/** Tunables (2026-11-06, new; no prior values — entries were bought whole). */
export const SCALE_IN = {
  /** env VOLTRADE_SCALE_IN = off / 0 / false / no turns it off (default ON, paper account) */
  ENV: "VOLTRADE_SCALE_IN",
  CONFIRM_BAND_PCT: 3,
  PULLBACK_FRAC: 0.5,
  BREAKOUT_PCT: 4,
  MAX_AGE_DAYS: 5, // = etfSubstitution ETF_MAX_HOLD_DAYS, so an ETF plan never outlives its hold
  /** each tranche must be worth at least this share of equity, else the entry is bought whole */
  MIN_TRANCHE_PCT_OF_EQUITY: 0.005,
  /** a tranche order is placed only inside this ET window (the floor's and rotation's) */
  WINDOW_ET: [9.75, 15.9] as readonly [number, number],
} as const;

export type TrancheStatus = "bought" | "pending" | "ordered" | "cancelled";
export interface Tranche { n: 1 | 2 | 3; qty: number; status: TrancheStatus; condition?: string; at?: string; price?: number; orderId?: string | null; reason?: string }
export interface ScaleInPlan {
  ticker: string;            // the order symbol (the ETF itself for a substitution)
  underlying: string;
  instrument: "stock" | "etf";
  status: "active" | "complete" | "cancelled";
  created_at: string;        // ISO
  entry_session: string;     // ET date of tranche 1
  ref_price: number;         // tranche 1's order price
  stop_price: number;        // the entry plan's stop for this symbol
  high: number;              // highest price seen since entry
  planned_qty: number;
  score: number | null;
  regime: string | null;
  tranches: Tranche[];
  cancel_reason: string | null;
  updated_at: string;
}
export interface ScaleInState { version: 1; plans: Record<string, ScaleInPlan> }

const OFF = new Set(["off", "0", "false", "no"]);
export function scaleInEnabled(env: Record<string, string | undefined> = process.env): boolean {
  return !OFF.has(String(env[SCALE_IN.ENV] ?? "").trim().toLowerCase());
}

/** [first, second, third] — the remainder goes to the first tranches; [] below 3 shares. */
export function splitThirds(qty: number): number[] {
  const q = Math.floor(qty);
  if (!(q >= 3)) return [];
  const base = Math.floor(q / 3), rem = q % 3;
  return [base + (rem > 0 ? 1 : 0), base + (rem > 1 ? 1 : 0), base];
}

/** The thesis stop for an ETF substitution, from the stock's: the stock stop's percent
 *  doubled for a 2x fund, held to instrument_selector's ETF stop bounds (4%-9%). An
 *  estimate (the scan sizes the stop on the stock); 0 when the stock stop is unusable. */
export function etfStopFromStock(stockPrice: number, stockStop: number, etfPrice: number): number {
  if (!(stockPrice > 0 && stockStop > 0 && stockStop < stockPrice && etfPrice > 0)) return 0;
  const stockPct = (stockPrice - stockStop) / stockPrice;
  const etfPct = Math.min(0.09, Math.max(0.04, 2 * stockPct));
  return Math.round(etfPrice * (1 - etfPct) * 100) / 100;
}

export interface PlanInput {
  ticker: string; underlying?: string; instrument?: "stock" | "etf";
  qty: number; price: number; stopPrice: number; equity: number;
  score?: number | null; regime?: string | null; nowMs: number; etDate: string;
}

/** {firstQty, plan} — plan is null (buy whole) when off, under 3 shares, a tranche would be
 *  worth less than MIN_TRANCHE_PCT_OF_EQUITY, or there is no usable stop below the price. */
export function planScaleIn(i: PlanInput, env: Record<string, string | undefined> = process.env): { firstQty: number; plan: ScaleInPlan | null; reason: string } {
  const qty = Math.floor(i.qty);
  if (!scaleInEnabled(env)) return { firstQty: qty, plan: null, reason: "scale-in is switched off" };
  const parts = splitThirds(qty);
  if (parts.length === 0) return { firstQty: qty, plan: null, reason: `${qty} share(s) cannot be split in thirds` };
  const minValue = Math.max(0, i.equity) * SCALE_IN.MIN_TRANCHE_PCT_OF_EQUITY;
  if (parts[2] * i.price < minValue) return { firstQty: qty, plan: null, reason: `a third would be under $${Math.round(minValue)}` };
  if (!(i.stopPrice > 0 && i.stopPrice < i.price)) return { firstQty: qty, plan: null, reason: "no stop below the entry price to judge the thesis by" };
  const iso = new Date(i.nowMs).toISOString();
  const plan: ScaleInPlan = {
    ticker: i.ticker.toUpperCase(), underlying: (i.underlying || i.ticker).toUpperCase(), instrument: i.instrument || "stock",
    status: "active", created_at: iso, entry_session: i.etDate, ref_price: i.price, stop_price: i.stopPrice, high: i.price,
    planned_qty: qty, score: i.score ?? null, regime: i.regime ?? null,
    tranches: [
      { n: 1, qty: parts[0], status: "bought", at: iso, price: i.price },
      { n: 2, qty: parts[1], status: "pending", condition: "confirm" },
      { n: 3, qty: parts[2], status: "pending", condition: "pullback_or_breakout" },
    ],
    cancel_reason: null, updated_at: iso,
  };
  return { firstQty: parts[0], plan, reason: `buying ${parts[0]} of ${qty} now; ${parts[1]} on confirmation, ${parts[2]} on a pullback or breakout` };
}

export interface TrancheContext {
  held: { qty: number; price: number } | null; // the live long position, null when not held
  orderWorking: boolean;                          // any open order on the symbol
  scalesCompleted: number;                        // the WS monitor's scale-outs so far
  regime: string;
  etDate: string;
  nowMs: number;
}
export type TrancheDecision =
  | { action: "buy"; tranche: Tranche; reason: string }
  | { action: "wait"; reason: string }
  | { action: "cancel"; reason: string }
  | { action: "complete"; reason: string };

const pct = (a: number, b: number) => (b > 0 ? (a / b - 1) * 100 : 0);
const usd = (v: number) => `$${v.toFixed(2)}`;

/** What to do with one active plan now. Updates nothing; see applyTrancheDecision. */
export function decideTranche(plan: ScaleInPlan, c: TrancheContext): TrancheDecision {
  if (!c.held || !(c.held.qty > 0)) {
    // the first third's order can still be working seconds after the plan was made
    if (c.orderWorking) return { action: "wait", reason: `${plan.ticker} is not held yet: its order is still working` };
    return { action: "cancel", reason: `${plan.ticker} is no longer held (its stop or another exit closed it)` };
  }
  if (c.scalesCompleted > 0) return { action: "cancel", reason: "a scale-out has begun, so the trade is taking profit, not adding" };
  const regime = String(c.regime || "").toUpperCase();
  if (regime === "BEAR" || regime === "PANIC") return { action: "cancel", reason: `the regime turned ${regime}, where no new stock longs are opened` };
  const price = c.held.price;
  if (price > 0 && price <= plan.stop_price) return { action: "cancel", reason: `the price ${usd(price)} reached the entry plan's stop ${usd(plan.stop_price)}` };
  const ageDays = (c.nowMs - Date.parse(plan.created_at)) / 86_400_000;
  if (ageDays > SCALE_IN.MAX_AGE_DAYS) return { action: "cancel", reason: `the plan expired after ${SCALE_IN.MAX_AGE_DAYS} days without its conditions` };
  const next = plan.tranches.find((t) => t.status === "pending");
  if (!next) return { action: "complete", reason: "every tranche is placed" };
  if (c.orderWorking) return { action: "wait", reason: `an order for ${plan.ticker} is still working` };
  if (regime === "NEUTRAL") return { action: "wait", reason: "the NEUTRAL regime opens no new stock trades; the tranche waits" };
  if (!(price > 0)) return { action: "wait", reason: "no live price" };
  if (next.n === 2) {
    if (c.etDate <= plan.entry_session) return { action: "wait", reason: "confirmation needs a later session than the entry" };
    const off = pct(price, plan.ref_price);
    if (Math.abs(off) <= SCALE_IN.CONFIRM_BAND_PCT) {
      return { action: "buy", tranche: next, reason: `confirmed: a later session with the price ${off >= 0 ? "+" : ""}${off.toFixed(1)}% from the first fill (within ±${SCALE_IN.CONFIRM_BAND_PCT}%) and above the stop` };
    }
    return { action: "wait", reason: `the price is ${off.toFixed(1)}% from the first fill, outside the ±${SCALE_IN.CONFIRM_BAND_PCT}% confirmation band` };
  }
  const t2 = plan.tranches.find((t) => t.n === 2);
  if (!t2 || (t2.status !== "bought" && t2.status !== "ordered")) return { action: "wait", reason: "the third tranche waits for the second" };
  const support = plan.ref_price - SCALE_IN.PULLBACK_FRAC * (plan.ref_price - plan.stop_price);
  if (price > plan.stop_price && price <= support) {
    return { action: "buy", tranche: next, reason: `a pullback toward support: ${usd(price)} is at least half-way from the first fill ${usd(plan.ref_price)} to the stop ${usd(plan.stop_price)}` };
  }
  if (pct(price, plan.ref_price) >= SCALE_IN.BREAKOUT_PCT && price >= plan.high) {
    return { action: "buy", tranche: next, reason: `a breakout continuation: a new high ${usd(price)} since entry, ${pct(price, plan.ref_price).toFixed(1)}% over the first fill` };
  }
  return { action: "wait", reason: "neither a pullback toward support nor a breakout continuation yet" };
}

/** Record a decision on the plan (status, tranche, high-water mark). */
export function applyTrancheDecision(plan: ScaleInPlan, d: TrancheDecision, c: TrancheContext, order?: { qty: number; price: number; orderId: string | null }): void {
  const iso = new Date(c.nowMs).toISOString();
  plan.updated_at = iso;
  if (c.held && c.held.price > plan.high) plan.high = c.held.price;
  if (d.action === "cancel") {
    plan.status = "cancelled";
    plan.cancel_reason = d.reason;
    for (const t of plan.tranches) if (t.status === "pending") { t.status = "cancelled"; t.reason = d.reason; t.at = iso; }
  } else if (d.action === "complete") {
    plan.status = "complete";
  } else if (d.action === "buy" && order) {
    d.tranche.status = "ordered";
    d.tranche.qty = order.qty;
    d.tranche.price = order.price;
    d.tranche.orderId = order.orderId;
    d.tranche.at = iso;
    d.tranche.reason = d.reason;
    if (!plan.tranches.some((t) => t.status === "pending")) plan.status = "complete";
  }
}

/** Unbought tranche dollars across active plans (what position_mgmt.py reserves). */
export function pendingTrancheDollars(state: ScaleInState): number {
  let total = 0;
  for (const p of Object.values(state.plans)) {
    if (p.status !== "active") continue;
    for (const t of p.tranches) if (t.status === "pending") total += t.qty * p.ref_price;
  }
  return total;
}

export function scaleInStatePath(env: Record<string, string | undefined> = process.env): string {
  const dir = env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
  return path.join(dir, SCALE_IN_STATE_FILE);
}

/** The persisted plans; a missing or unreadable file is an empty state (logged by the caller). */
export function loadScaleInState(file: string): ScaleInState {
  try {
    const raw = JSON.parse(fs.readFileSync(file, "utf8"));
    if (raw && typeof raw === "object" && raw.plans && typeof raw.plans === "object") return { version: 1, plans: raw.plans };
  } catch (e: unknown) {
    if (!(e instanceof Error && "code" in e && (e as NodeJS.ErrnoException).code === "ENOENT")) {
      console.warn(`[scale-in] state unreadable (${file}): ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  return { version: 1, plans: {} };
}

/** Atomic write; finished plans older than keepDays are dropped. */
export function saveScaleInState(file: string, state: ScaleInState, nowMs: number = Date.now(), keepDays = 14): void {
  for (const [k, p] of Object.entries(state.plans)) {
    if (p.status !== "active" && nowMs - Date.parse(p.updated_at) > keepDays * 86_400_000) delete state.plans[k];
  }
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(state));
  fs.renameSync(tmp, file);
}
