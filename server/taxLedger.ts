// taxLedger.ts — what the same trades would cost in a TAXABLE account
// ([RULE-REVIEW] measurement, Bot page program slice E, 2026-10-21). The
// paper account pays no tax; the human asked whether the bot's manoeuvres
// still make sense once short-term gains are taxed as income and year-end
// loss harvesting / wash-sale rules apply. This computes, from the SAME
// realized events the sleeve ledger walks (average-cost basis, one event per
// priced close):
//   • realized P&L by holding-period TERM (short ≤ 1 year, long > 1 year,
//     unknown when the lot predates the fetched history), per calendar year;
//   • WASH SALES: a loss closed on a symbol that is re-entered on the same
//     side within the 30 days AFTER the close — the loss is disallowed in
//     proportion to the replacement quantity (deferred into the new lot's
//     basis). Only exact-symbol replacements after the close are counted
//     (a LOWER bound: the rule also covers purchases in the 30 days before
//     and "substantially identical" contracts, which an average-cost walk
//     cannot attribute honestly);
//   • an ESTIMATED tax at the caller's assumed rates after standard netting
//     (net short vs net long; a remaining net loss carries forward, the
//     $3,000-per-year ordinary offset is not modelled), and the after-tax
//     realized result.
// Every number is an ESTIMATE at assumed rates and is labelled so by the
// caller. Pure; no I/O; pinned by taxLedger.test.ts.

import type { FillRow } from "./killSwitchAutoResume";
import type { RealizedEvent } from "./sleeveLedger";

export interface TaxRates { shortTerm: number; longTerm: number; state: number; niit: number }
/** Assumed federal 24 % ordinary / 15 % long-term, no state, no NIIT —
 *  the middle brackets, overridable per call; the UI prints whatever was
 *  used. */
export const DEFAULT_TAX_RATES: TaxRates = { shortTerm: 0.24, longTerm: 0.15, state: 0, niit: 0 };
export const LONG_TERM_MS = 365 * 86_400_000; // "more than one year"
export const WASH_WINDOW_MS = 30 * 86_400_000;

export type Term = "short" | "long" | "unknown";
export interface TermRow { term: Term; events: number; gains: number; losses: number; net: number; washDisallowed: number }
export interface WashSale { symbol: string; closedAt: string; loss: number; disallowed: number; replacementQty: number; replacedAt: string; term: Term }
export interface YearRow { year: number; short: number; long: number; unknown: number; washDisallowed: number; estimatedTax: number }
export interface TaxLedger {
  rates: TaxRates;
  windowStart: string | null;
  byTerm: TermRow[];
  byYear: YearRow[];
  washSales: { count: number; disallowed: number; examples: WashSale[] };
  netting: { shortNet: number; longNet: number; shortAfterNetting: number; longAfterNetting: number; carryforward: number };
  estimatedTax: { shortTerm: number; longTerm: number; niit: number; total: number };
  afterTax: { realizedNet: number; taxDrag: number; afterTaxNet: number };
  notes: string[];
}

export function termOf(e: RealizedEvent): Term {
  if (e.openedAt === null) return "unknown";
  return e.t - e.openedAt > LONG_TERM_MS ? "long" : "short";
}

/** The net share of a losing close that a same-side re-entry within 30
 *  days after it disallows. */
export function washSalesFor(events: readonly RealizedEvent[], fills: readonly FillRow[], windowStartMs = -Infinity): WashSale[] {
  const bySymbol = new Map<string, FillRow[]>();
  for (const f of fills) { const a = bySymbol.get(f.symbol); if (a) a.push(f); else bySymbol.set(f.symbol, [f]); }
  const out: WashSale[] = [];
  for (const e of events) {
    if (!(e.pnl < 0) || e.t < windowStartMs || !(e.qty > 0)) continue;
    const list = bySymbol.get(e.symbol);
    if (!list) continue;
    // the replacement re-enters on the SIDE the loss was on: a sold long is
    // replaced by a buy, a covered short by a new short sale
    const replacementDir = -e.closeDir;
    let qty = 0; let firstAt: number | null = null;
    for (const f of list) {
      if (f.dir !== replacementDir || f.t <= e.t || f.t > e.t + WASH_WINDOW_MS) continue;
      qty += f.qty; if (firstAt === null) firstAt = f.t;
    }
    if (qty <= 0 || firstAt === null) continue;
    const share = Math.min(1, qty / e.qty);
    out.push({ symbol: e.symbol, closedAt: new Date(e.t).toISOString(), loss: e.pnl, disallowed: -e.pnl * share, replacementQty: qty, replacedAt: new Date(firstAt).toISOString(), term: termOf(e) });
  }
  return out;
}

const r2 = (v: number) => Math.round(v * 100) / 100;

export function taxLedger(events: readonly RealizedEvent[], fills: readonly FillRow[], opts: { windowStartMs?: number; rates?: Partial<TaxRates> } = {}): TaxLedger {
  const rates: TaxRates = { ...DEFAULT_TAX_RATES, ...(opts.rates ?? {}) };
  const windowStartMs = opts.windowStartMs ?? -Infinity;
  const inWindow = events.filter((e) => e.t >= windowStartMs);
  const wash = washSalesFor(events, fills, windowStartMs);
  const washByEvent = new Map<string, number>();
  for (const w of wash) washByEvent.set(`${w.symbol}|${w.closedAt}|${w.loss}`, (washByEvent.get(`${w.symbol}|${w.closedAt}|${w.loss}`) ?? 0) + w.disallowed);
  const terms: Record<Term, TermRow> = {
    short: { term: "short", events: 0, gains: 0, losses: 0, net: 0, washDisallowed: 0 },
    long: { term: "long", events: 0, gains: 0, losses: 0, net: 0, washDisallowed: 0 },
    unknown: { term: "unknown", events: 0, gains: 0, losses: 0, net: 0, washDisallowed: 0 },
  };
  const years = new Map<number, YearRow>();
  for (const e of inWindow) {
    const term = termOf(e);
    const row = terms[term];
    row.events++;
    if (e.pnl >= 0) row.gains += e.pnl; else row.losses += e.pnl;
    row.net += e.pnl;
    const dis = washByEvent.get(`${e.symbol}|${new Date(e.t).toISOString()}|${e.pnl}`) ?? 0;
    row.washDisallowed += dis;
    const y = new Date(e.t).getUTCFullYear();
    const yr = years.get(y) ?? { year: y, short: 0, long: 0, unknown: 0, washDisallowed: 0, estimatedTax: 0 };
    yr[term] += e.pnl; yr.washDisallowed += dis;
    years.set(y, yr);
  }
  const washTotal = wash.reduce((s, w) => s + w.disallowed, 0);
  // netting: disallowed losses are added back (not deductible now), then
  // net short against net long; what is left negative carries forward
  const shortNet = terms.short.net + terms.short.washDisallowed + terms.unknown.net + terms.unknown.washDisallowed; // unknown-age lots are treated as short (the conservative term)
  const longNet = terms.long.net + terms.long.washDisallowed;
  let st = shortNet, lt = longNet;
  if (st < 0 && lt > 0) { const off = Math.min(-st, lt); lt -= off; st += off; }
  else if (lt < 0 && st > 0) { const off = Math.min(-lt, st); st -= off; lt += off; }
  const carryforward = Math.min(0, st + lt);
  const stRate = rates.shortTerm + rates.state, ltRate = rates.longTerm + rates.state;
  const taxST = Math.max(0, st) * stRate, taxLT = Math.max(0, lt) * ltRate;
  const niit = (Math.max(0, st) + Math.max(0, lt)) * rates.niit;
  const total = taxST + taxLT + niit;
  for (const yr of years.values()) {
    // per-year estimate with the same netting, for the year-end view
    let ys = yr.short + yr.unknown + yr.washDisallowed, yl = yr.long;
    if (ys < 0 && yl > 0) { const off = Math.min(-ys, yl); yl -= off; ys += off; } else if (yl < 0 && ys > 0) { const off = Math.min(-yl, ys); ys -= off; yl += off; }
    yr.estimatedTax = r2(Math.max(0, ys) * stRate + Math.max(0, yl) * ltRate + (Math.max(0, ys) + Math.max(0, yl)) * rates.niit);
    yr.short = r2(yr.short); yr.long = r2(yr.long); yr.unknown = r2(yr.unknown); yr.washDisallowed = r2(yr.washDisallowed);
  }
  const realizedNet = inWindow.reduce((s, e) => s + e.pnl, 0);
  const notes: string[] = [
    "estimate at the assumed rates shown; the paper account pays no tax",
    "terms from the average-cost lot walk the sleeve ledger uses (not specific-lot or FIFO); lots older than the fetched history read as 'unknown' and are taxed as short-term",
    "wash sales: exact-symbol re-entries on the same side within 30 days AFTER a loss only — a lower bound (purchases before the loss and substantially identical contracts are not evaluated)",
    "a remaining net loss carries forward; the $3,000-per-year ordinary-income offset is not modelled",
  ];
  if (terms.unknown.events) notes.push(`${terms.unknown.events} close(s) on lots older than the fetched history`);
  return {
    rates, windowStart: Number.isFinite(windowStartMs) ? new Date(windowStartMs).toISOString() : null,
    byTerm: (["short", "long", "unknown"] as Term[]).map((t) => ({ ...terms[t], gains: r2(terms[t].gains), losses: r2(terms[t].losses), net: r2(terms[t].net), washDisallowed: r2(terms[t].washDisallowed) })),
    byYear: [...years.values()].sort((a, b) => a.year - b.year),
    washSales: { count: wash.length, disallowed: r2(washTotal), examples: wash.sort((a, b) => b.disallowed - a.disallowed).slice(0, 10) },
    netting: { shortNet: r2(shortNet), longNet: r2(longNet), shortAfterNetting: r2(st), longAfterNetting: r2(lt), carryforward: r2(carryforward) },
    estimatedTax: { shortTerm: r2(taxST), longTerm: r2(taxLT), niit: r2(niit), total: r2(total) },
    afterTax: { realizedNet: r2(realizedNet), taxDrag: r2(total), afterTaxNet: r2(realizedNet - total) },
    notes,
  };
}

/** Rates from query parameters (?st=0.32&lt=0.15&state=0.05&niit=0.038),
 *  each a fraction in [0, 0.6]; anything else keeps the default. */
export function ratesFromQuery(q: Record<string, unknown>): Partial<TaxRates> {
  const pick = (k: string): number | undefined => { const v = Number(q[k]); return Number.isFinite(v) && v >= 0 && v <= 0.6 ? v : undefined; };
  const out: Partial<TaxRates> = {};
  const st = pick("st"), lt = pick("lt"), state = pick("state"), niit = pick("niit");
  if (st !== undefined) out.shortTerm = st;
  if (lt !== undefined) out.longTerm = lt;
  if (state !== undefined) out.state = state;
  if (niit !== undefined) out.niit = niit;
  return out;
}
