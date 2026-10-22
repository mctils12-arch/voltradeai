import { test } from "node:test";
import assert from "node:assert/strict";
import { afterTaxBenchmark, windowTerm } from "./afterTaxBenchmark";
import { taxLedger, type TaxRates } from "./taxLedger";
import type { RealizedEvent } from "./sleeveLedger";
import type { BenchWindow } from "./spyBenchmark";

const D = 86_400_000;
const T0 = Date.parse("2026-05-06T00:00:00Z");
const ev = (symbol: string, closeDay: number, pnl: number, openDay: number | null, qty = 10, closeDir: 1 | -1 = -1): RealizedEvent => ({ symbol, t: T0 + closeDay * D, pnl, openedAt: openDay === null ? null : T0 + openDay * D, qty, closeDir });
const win = (o: Partial<BenchWindow> = {}): BenchWindow => ({ from: "2026-05-06", to: "2026-10-20", days: 167, accountPct: -6.9, spyPct: 5.24, gapPp: -12.14, ...o });
const ledger = (events: RealizedEvent[], rates?: Partial<TaxRates>) => taxLedger(events, [], { windowStartMs: T0, rates });
const close = (a: number, b: number, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} ≠ ${b}`);

test("window term: more than one year is long, a year to the day is short (the ledger's rule)", () => {
  assert.equal(windowTerm(167), "short");
  assert.equal(windowTerm(365), "short");
  assert.equal(windowTerm(366), "long");
});

test("SPY: one sale at the window's end — a gain taxed at the holding term's rate, a loss not at all", () => {
  const r = afterTaxBenchmark(win(), ledger([]), { equityStart: 100_000, openPnl: 0 });
  assert.ok(r);
  assert.equal(r.spy.term, "short");
  close(r.spy.taxPct, 1.26); // 5.24 × 24 %
  close(r.spy.afterTaxPct, 3.98);
  const long = afterTaxBenchmark(win({ days: 400, to: "2027-06-10" }), ledger([]), { equityStart: 100_000, openPnl: 0 });
  assert.equal(long?.spy.term, "long");
  close(long!.spy.taxPct, 0.79); // 5.24 × 15 %
  const loss = afterTaxBenchmark(win({ spyPct: -3 }), ledger([]), { equityStart: 100_000, openPnl: 0 });
  close(loss!.spy.taxPct, 0);
  close(loss!.spy.afterTaxPct, -3);
});

test("account: the ledger's realized tax plus what selling the open P&L adds as short-term; the gap is after-tax minus after-tax", () => {
  const r = afterTaxBenchmark(win(), ledger([ev("A", 10, 1000, 2)]), { equityStart: 100_000, openPnl: 500 });
  assert.ok(r);
  close(r.account.realizedTax, 240); // 1000 × 24 %
  close(r.account.liquidationTax, 360); // (1000 + 500) × 24 %
  close(r.account.openTaxIfSold, 120);
  close(r.account.taxPct, 0.36); // of the start equity
  close(r.account.afterTaxPct, -7.26);
  close(r.gapPp, -11.24); // −7.26 − 3.98
  assert.equal(r.window.from, "2026-05-06");
  assert.equal(r.window.days, 167);
});

test("netting: an open loss reduces the realized tax; a carried-forward realized loss shelters an open gain; a long-term loss shelters an open short-term gain", () => {
  const openLoss = afterTaxBenchmark(win(), ledger([ev("A", 10, 1000, 2)]), { equityStart: 100_000, openPnl: -400 });
  close(openLoss!.account.liquidationTax, 144); // (1000 − 400) × 24 %
  close(openLoss!.account.openTaxIfSold, -96);
  const carry = afterTaxBenchmark(win(), ledger([ev("B", 10, -1000, 2)]), { equityStart: 100_000, openPnl: 400 });
  close(carry!.account.realizedTax, 0);
  close(carry!.account.liquidationTax, 0); // −1000 carried forward shelters the +400
  close(carry!.account.openTaxIfSold, 0);
  close(carry!.account.afterTaxPct, -6.9);
  const longLoss = afterTaxBenchmark(win({ days: 420, to: "2027-06-30" }), ledger([ev("C", 400, -1000, 0)]), { equityStart: 100_000, openPnl: 1500 });
  close(longLoss!.account.liquidationTax, 120); // (1500 − 1000) × 24 %
});

test("unpriced open positions: the tax if sold is not included and the note says so", () => {
  const r = afterTaxBenchmark(win(), ledger([ev("A", 10, 1000, 2)]), { equityStart: 100_000, openPnl: null });
  assert.equal(r!.account.openPnl, null);
  assert.equal(r!.account.openTaxIfSold, null);
  close(r!.account.liquidationTax, 240);
  assert.ok(r!.notes.some((n) => /unpriced/.test(n)));
});

test("no start equity → null; nothing is inferred", () => {
  assert.equal(afterTaxBenchmark(win(), ledger([]), { equityStart: 0, openPnl: 0 }), null);
  assert.equal(afterTaxBenchmark(win(), ledger([]), { equityStart: Number.NaN, openPnl: 0 }), null);
});

test("rates pass through the ledger: short-term, state and NIIT all reach the SPY sale and the account", () => {
  const st = afterTaxBenchmark(win(), ledger([], { shortTerm: 0.32 }), { equityStart: 100_000, openPnl: 0 });
  close(st!.spy.taxPct, 1.68); // 5.24 × 32 %
  assert.equal(st!.rates.shortTerm, 0.32);
  const all = afterTaxBenchmark(win(), ledger([ev("A", 10, 1000, 2)], { state: 0.05, niit: 0.038 }), { equityStart: 100_000, openPnl: 0 });
  close(all!.spy.taxPct, 1.72); // 5.24 × (24 + 5 + 3.8) %
  close(all!.account.realizedTax, 328); // 1000 × (24 + 5 + 3.8) %
});

test("the notes state the bias in both directions and the liquidation framing", () => {
  const r = afterTaxBenchmark(win(), ledger([]), { equityStart: 100_000, openPnl: 0 });
  assert.ok(r!.notes.some((n) => /both sold at 2026-10-20/.test(n)));
  assert.ok(r!.notes.some((n) => /flatters the account/.test(n)));
  assert.ok(r!.notes.some((n) => /short-term \(the conservative term\)/.test(n)));
});
