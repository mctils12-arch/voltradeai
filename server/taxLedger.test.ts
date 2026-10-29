import { test } from "node:test";
import assert from "node:assert/strict";
import { taxLedger, washSalesFor, termOf, ratesFromQuery, DEFAULT_TAX_RATES, LONG_TERM_MS, WASH_WINDOW_MS } from "./taxLedger";
import { realizedEvents, type RealizedEvent } from "./sleeveLedger";
import type { FillRow } from "./killSwitchAutoResume";

const D = 86_400_000;
const T0 = Date.parse("2026-01-05T15:00:00Z");
const fill = (symbol: string, dir: 1 | -1, qty: number, price: number, day: number): FillRow => ({ symbol, dir, qty, price, t: T0 + day * D });
const ev = (symbol: string, closeDay: number, pnl: number, openDay: number | null, qty = 10, closeDir: 1 | -1 = -1): RealizedEvent => ({ symbol, t: T0 + closeDay * D, pnl, openedAt: openDay === null ? null : T0 + openDay * D, qty, closeDir, basis: qty * 100 });
const close = (a: number, b: number, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} ≠ ${b}`);

test("term: more than one year is long, a year to the day is short, an unknown-age lot is unknown", () => {
  assert.equal(termOf(ev("A", 400, 1, 0)), "long");
  assert.equal(termOf(ev("A", 365, 1, 0)), "short");
  assert.equal(termOf(ev("A", 10, 1, null)), "unknown");
  assert.equal(LONG_TERM_MS, 365 * D);
});

test("wash sale: a same-side re-entry within 30 days after a loss disallows the loss in proportion to the replacement quantity; a re-entry on day 31, a gain, or the other side does not", () => {
  const fills = [fill("NVDA", 1, 10, 100, 0), fill("NVDA", -1, 10, 90, 5), fill("NVDA", 1, 4, 88, 12), fill("NVDA", -1, 4, 95, 50), fill("NVDA", 1, 10, 80, 90)];
  const events = [ev("NVDA", 5, -100, 0, 10, -1), ev("NVDA", 50, 28, 12, 4, -1), ev("NVDA", 95, -50, 90, 10, -1)];
  const w = washSalesFor(events, fills);
  assert.equal(w.length, 1);
  assert.equal(w[0].symbol, "NVDA");
  close(w[0].disallowed, 40); // 4 of 10 shares re-bought within 30 days → 40 % of the $100 loss
  assert.equal(w[0].replacementQty, 4);
  assert.equal(w[0].term, "short");
  // the day-90 loss has no re-entry after it; the day-50 close was a gain
  // a covered short's loss is washed by a NEW short sale, not a buy
  const shortFills = [fill("TSLA", -1, 5, 100, 0), fill("TSLA", 1, 5, 110, 3), fill("TSLA", -1, 5, 112, 10), fill("TSLA", 1, 2, 111, 20)];
  const shortEvents = [ev("TSLA", 3, -50, 0, 5, 1)];
  const ws = washSalesFor(shortEvents, shortFills);
  assert.equal(ws.length, 1);
  close(ws[0].disallowed, 50);
  assert.equal(ws[0].replacementQty, 5);
  // the window is exactly 30 days
  const edge = [fill("X", 1, 1, 10, 0), fill("X", -1, 1, 9, 1), fill("X", 1, 1, 9, 1 + WASH_WINDOW_MS / D + 0.01)];
  assert.equal(washSalesFor([ev("X", 1, -1, 0, 1, -1)], edge).length, 0);
});

test("the ledger nets short against long, adds disallowed losses back, estimates tax at the assumed rates and reports the after-tax result", () => {
  const events = [
    ev("A", 10, 1000, 0),            // short-term gain
    ev("B", 400, 2000, 0),           // long-term gain
    ev("C", 20, -600, 1),            // short-term loss, no re-entry
    ev("D", 30, -400, 2, 10, -1),    // short-term loss, fully washed by the re-buy below
    ev("E", 40, 50, null),           // unknown-age lot → short
  ];
  const fills = [fill("D", 1, 10, 100, 2), fill("D", -1, 10, 60, 30), fill("D", 1, 10, 58, 35)];
  const l = taxLedger(events, fills, { rates: { shortTerm: 0.3, longTerm: 0.2, state: 0.05, niit: 0 } });
  const short = l.byTerm.find((r) => r.term === "short")!, long = l.byTerm.find((r) => r.term === "long")!, unk = l.byTerm.find((r) => r.term === "unknown")!;
  assert.equal(short.events, 3); close(short.gains, 1000); close(short.losses, -1000); close(short.net, 0); close(short.washDisallowed, 400);
  assert.equal(long.events, 1); close(long.net, 2000);
  assert.equal(unk.events, 1); close(unk.net, 50);
  assert.equal(l.washSales.count, 1); close(l.washSales.disallowed, 400);
  // short net = 0 + 400 disallowed + 50 unknown = 450; long net = 2000; no cross-netting needed
  close(l.netting.shortNet, 450); close(l.netting.longNet, 2000); close(l.netting.carryforward, 0);
  close(l.estimatedTax.shortTerm, 450 * 0.35); close(l.estimatedTax.longTerm, 2000 * 0.25); close(l.estimatedTax.total, 157.5 + 500);
  close(l.afterTax.realizedNet, 2050); close(l.afterTax.afterTaxNet, 2050 - 657.5);
  assert.equal(l.byYear.length, 2); // the long-term close lands in the next calendar year
  assert.ok(l.notes.some((n) => n.startsWith("estimate at the assumed rates")));
});

test("cross-netting: a short-term loss offsets a long-term gain before the lower rate applies; a remaining net loss carries forward with no tax", () => {
  const l = taxLedger([ev("A", 10, -300, 0), ev("B", 400, 500, 0)], [], { rates: { shortTerm: 0.3, longTerm: 0.2, state: 0, niit: 0 } });
  close(l.netting.shortAfterNetting, 0); close(l.netting.longAfterNetting, 200);
  close(l.estimatedTax.total, 40);
  const loss = taxLedger([ev("A", 10, -300, 0), ev("B", 400, 100, 0)], []);
  close(loss.estimatedTax.total, 0); close(loss.netting.carryforward, -200);
  // NIIT applies to positive investment income after netting
  const n = taxLedger([ev("A", 10, 1000, 0)], [], { rates: { shortTerm: 0.2, longTerm: 0.15, state: 0, niit: 0.038 } });
  close(n.estimatedTax.niit, 38); close(n.estimatedTax.total, 238);
});

test("the window start excludes earlier closes; rates parse from query strings with bounds; the walk carries qty and closeDir without changing totals", () => {
  const events = [ev("A", 1, 100, 0), ev("A", 50, 200, 10)];
  const l = taxLedger(events, [], { windowStartMs: T0 + 20 * D });
  close(l.afterTax.realizedNet, 200);
  assert.equal(l.windowStart, new Date(T0 + 20 * D).toISOString());
  assert.deepEqual(ratesFromQuery({ st: "0.32", lt: "0.2", state: "0.05", niit: "0.038" }), { shortTerm: 0.32, longTerm: 0.2, state: 0.05, niit: 0.038 });
  assert.deepEqual(ratesFromQuery({ st: "1.5", lt: "abc" }), {});
  assert.deepEqual(DEFAULT_TAX_RATES, { shortTerm: 0.24, longTerm: 0.15, state: 0, niit: 0 });
  const fills = [fill("Q", 1, 10, 100, 0), fill("Q", -1, 4, 110, 1), fill("Q", -1, 6, 90, 2)];
  const { events: walked } = realizedEvents(fills, []);
  assert.deepEqual(walked.map((e) => [e.qty, e.closeDir, e.pnl]), [[4, -1, 40], [6, -1, -60]]);
});
