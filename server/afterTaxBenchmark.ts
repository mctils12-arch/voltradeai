// afterTaxBenchmark.ts — the vs-SPY comparison after tax ([RULE-REVIEW]
// 2026-10-22). Pure; server/bot.ts composes it from the benchmark window
// (server/spyBenchmark.ts), the tax ledger for that same window
// (server/taxLedger.ts) and the sleeve ledger's open P&L.
//
// FRAMING — one honest, symmetric view: both legs are SOLD at the window's
// end. Buy-and-hold SPY defers its tax until that sale (one event, at the
// holding term's rate); the account pays the ledger's estimated tax on every
// trade it closed in the window, plus what selling its open positions would
// add, netted the way the ledger nets.
//
// BIAS, STATED (MEASUREMENT INTEGRITY):
//   - SPY's total return includes dividends a real holder is taxed on each
//     year at the qualified rate; taxing the whole return at the short-term
//     rate in a sub-year window OVERSTATES SPY's tax by about
//     (short − long) × dividend yield (≈ 0.05 pt over five months) — that
//     much flatters the account. Called out in the notes.
//   - Open positions are taxed as short-term (the conservative term for a
//     bot whose lots rarely age a year) — the account reads worse, never
//     better, for it.
//   - The paper account pays nothing; every figure is an estimate at the
//     rates shown.
import type { BenchWindow } from "./spyBenchmark";
import { LONG_TERM_MS, type TaxLedger, type TaxRates, type Term } from "./taxLedger";

export interface AfterTaxLeg { pct: number; taxPct: number; afterTaxPct: number }
export interface AfterTaxBenchmark {
  rates: TaxRates;
  window: { from: string; to: string; days: number };
  /** buy-and-hold SPY sold at `to`: the whole total return taxed once at the holding term's rate */
  spy: AfterTaxLeg & { term: Term };
  /** the account: the ledger's estimated tax on the trades closed in the window plus what selling every
   *  open position at `to` would add (short-term, netted like the ledger); taxPct is of the start equity */
  account: AfterTaxLeg & { equityStart: number; realizedTax: number; liquidationTax: number; openPnl: number | null; openTaxIfSold: number | null };
  /** account.afterTaxPct − spy.afterTaxPct */
  gapPp: number;
  notes: string[];
}

const r2 = (x: number) => Math.round(x * 100) / 100;

/** The holding term of a window: more than one year is long (the ledger's own rule). */
export function windowTerm(days: number): Term { return days * 86_400_000 > LONG_TERM_MS ? "long" : "short"; }

/** Pure. `ledger` is the tax ledger computed for the benchmark window's own start; `openPnl` the sleeve
 *  ledger's open P&L at `to` (null when any position is unpriced). Null without a start equity. */
export function afterTaxBenchmark(since: BenchWindow, ledger: TaxLedger, inputs: { equityStart: number; openPnl: number | null }): AfterTaxBenchmark | null {
  if (!since || !(inputs.equityStart > 0) || !Number.isFinite(since.accountPct) || !Number.isFinite(since.spyPct)) return null;
  const { rates } = ledger;
  const stRate = rates.shortTerm + rates.state, ltRate = rates.longTerm + rates.state;
  // SPY: one sale at `to`
  const term = windowTerm(since.days);
  const spyTaxPct = Math.max(0, since.spyPct) * ((term === "long" ? ltRate : stRate) + rates.niit);
  // the account: the ledger's netted short/long from the closed trades, then the open P&L sold as
  // short-term and netted the same way (an open loss shelters a realized gain; a carried-forward
  // realized loss shelters an open gain)
  const realizedTax = ledger.estimatedTax.total;
  let liquidationTax = realizedTax;
  let openTaxIfSold: number | null = null;
  const open = inputs.openPnl;
  if (typeof open === "number" && Number.isFinite(open)) {
    let st = ledger.netting.shortAfterNetting + open, lt = ledger.netting.longAfterNetting;
    if (st < 0 && lt > 0) { const off = Math.min(-st, lt); lt -= off; st += off; }
    else if (lt < 0 && st > 0) { const off = Math.min(-lt, st); st -= off; lt += off; }
    liquidationTax = Math.max(0, st) * stRate + Math.max(0, lt) * ltRate + (Math.max(0, st) + Math.max(0, lt)) * rates.niit;
    openTaxIfSold = liquidationTax - realizedTax;
  }
  const accountTaxPct = liquidationTax / inputs.equityStart * 100;
  const spy = { pct: since.spyPct, term, taxPct: r2(spyTaxPct), afterTaxPct: r2(since.spyPct - spyTaxPct) };
  const account = {
    pct: since.accountPct, equityStart: r2(inputs.equityStart), realizedTax: r2(realizedTax), liquidationTax: r2(liquidationTax),
    openPnl: typeof open === "number" && Number.isFinite(open) ? r2(open) : null, openTaxIfSold: openTaxIfSold === null ? null : r2(openTaxIfSold),
    taxPct: r2(accountTaxPct), afterTaxPct: r2(since.accountPct - accountTaxPct),
  };
  const notes = [
    `both sold at ${since.to}: an estimate at the rates shown — the paper account pays no tax`,
    `SPY: its whole total return taxed once at the ${term}-term rate; the dividends inside it would really be taxed each year at the qualified rate, so SPY's tax is slightly overstated (about (short − long) × yield), which flatters the account by that much`,
    "account: the ledger's estimated tax on the trades closed in the window, plus what selling every open position would add — open P&L taxed as short-term (the conservative term) and netted the way the ledger nets, so open losses reduce the tax",
  ];
  if (openTaxIfSold === null) notes.push("open positions are unpriced, so their tax if sold is not included");
  return { rates, window: { from: since.from, to: since.to, days: since.days }, spy, account, gapPp: r2(account.afterTaxPct - spy.afterTaxPct), notes };
}
