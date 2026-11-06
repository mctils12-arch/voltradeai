// server/scaleIn.ts — scale-in by thirds (2026-11-06): the split, the plan,
// each tranche's condition, the cancellations, and the persisted state.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  splitThirds, planScaleIn, decideTranche, applyTrancheDecision, etfStopFromStock, pendingTrancheDollars,
  loadScaleInState, saveScaleInState, scaleInEnabled, SCALE_IN, type ScaleInPlan, type TrancheContext,
} from "./scaleIn";

const DAY = 86_400_000;
const T0 = Date.parse("2026-11-09T15:00:00Z");      // Monday 10:00 ET
const ON = {};

const plan = (over: Partial<Parameters<typeof planScaleIn>[0]> = {}): ScaleInPlan =>
  planScaleIn({ ticker: "aapl", qty: 30, price: 100, stopPrice: 94, equity: 100_000, score: 72, regime: "BULL",
    nowMs: T0, etDate: "2026-11-09", ...over }, ON).plan!;
const ctx = (over: Partial<TrancheContext> = {}): TrancheContext => ({
  held: { qty: 10, price: 100 }, orderWorking: false, scalesCompleted: 0, regime: "BULL",
  etDate: "2026-11-10", nowMs: T0 + DAY, ...over,
});

test("splitThirds: remainder to the first tranches; under 3 shares there is no split", () => {
  assert.deepEqual(splitThirds(30), [10, 10, 10]);
  assert.deepEqual(splitThirds(10), [4, 3, 3]);
  assert.deepEqual(splitThirds(11), [4, 4, 3]);
  assert.deepEqual(splitThirds(3), [1, 1, 1]);
  assert.deepEqual(splitThirds(2), []);
  assert.deepEqual(splitThirds(Number.NaN), []);
});

test("planScaleIn buys a third now and plans the rest — or buys whole when it cannot", () => {
  const r = planScaleIn({ ticker: "aapl", qty: 30, price: 100, stopPrice: 94, equity: 100_000, nowMs: T0, etDate: "2026-11-09" }, ON);
  assert.equal(r.firstQty, 10);
  assert.ok(r.plan);
  assert.equal(r.plan.ticker, "AAPL");
  assert.deepEqual(r.plan.tranches.map((t) => [t.n, t.qty, t.status]), [[1, 10, "bought"], [2, 10, "pending"], [3, 10, "pending"]]);
  assert.match(r.reason, /buying 10 of 30 now; 10 on confirmation, 10 on a pullback or breakout/);
  const whole = (over: object, why: RegExp, env: Record<string, string> = {}) => {
    const w = planScaleIn({ ticker: "X", qty: 30, price: 100, stopPrice: 94, equity: 100_000, nowMs: T0, etDate: "2026-11-09", ...over }, env);
    assert.equal(w.plan, null);
    assert.equal(w.firstQty, (over as { qty?: number }).qty ?? 30);
    assert.match(w.reason, why);
  };
  whole({ qty: 2 }, /cannot be split/);
  whole({ qty: 3, price: 100 }, /a third would be under \$500/);              // 1 share x $100 < 0.5% of $100k
  whole({ stopPrice: 0 }, /no stop below/);
  whole({}, /switched off/, { VOLTRADE_SCALE_IN: "off" });
  assert.equal(scaleInEnabled({}), true);
  assert.equal(scaleInEnabled({ VOLTRADE_SCALE_IN: "0" }), false);
});

test("tranche 2 needs a later session and a price within the confirmation band", () => {
  const p = plan();
  assert.deepEqual(decideTranche(p, ctx({ etDate: "2026-11-09" })), { action: "wait", reason: "confirmation needs a later session than the entry" });
  const d = decideTranche(p, ctx({ held: { qty: 10, price: 102 } }));
  assert.equal(d.action, "buy");
  assert.equal(d.action === "buy" && d.tranche.n, 2);
  assert.match(d.action === "buy" ? d.reason : "", /\+2\.0% from the first fill \(within ±3%\)/);
  const far = decideTranche(p, ctx({ held: { qty: 10, price: 104 } }));
  assert.equal(far.action, "wait");
  assert.match(far.action === "wait" ? far.reason : "", /outside the ±3% confirmation band/);
});

test("tranche 3 follows tranche 2 on a pullback toward support or a breakout to a new high", () => {
  const p = plan();
  const c2 = ctx({ held: { qty: 10, price: 101 } });
  const d2 = decideTranche(p, c2);
  applyTrancheDecision(p, d2, c2, { qty: 10, price: 101, orderId: "o2" });
  assert.equal(p.tranches[1].status, "ordered");
  assert.equal(p.status, "active");
  // support = 100 - 0.5 x (100 - 94) = 97
  assert.equal(decideTranche(p, ctx({ held: { qty: 20, price: 99 } })).action, "wait");
  const pull = decideTranche(p, ctx({ held: { qty: 20, price: 96.5 } }));
  assert.equal(pull.action, "buy");
  assert.match(pull.action === "buy" ? pull.reason : "", /pullback toward support/);
  // breakout: >= +4% over the first fill AND a new high since entry
  p.high = 106;
  assert.equal(decideTranche(p, ctx({ held: { qty: 20, price: 105 } })).action, "wait", "+5% but under the 106 high");
  const brk = decideTranche(p, ctx({ held: { qty: 20, price: 106.5 } }));
  assert.equal(brk.action, "buy");
  assert.match(brk.action === "buy" ? brk.reason : "", /breakout continuation/);
  applyTrancheDecision(p, brk, ctx({ held: { qty: 20, price: 106.5 } }), { qty: 10, price: 106.5, orderId: "o3" });
  assert.equal(p.status, "complete");
  assert.equal(p.high, 106.5);
});

test("the stop, a broken thesis or age cancel what is left; NEUTRAL and working orders only wait", () => {
  const cases: Array<[Partial<TrancheContext>, RegExp]> = [
    [{ held: null }, /no longer held/],
    [{ scalesCompleted: 1 }, /scale-out has begun/],
    [{ regime: "BEAR" }, /regime turned BEAR/],
    [{ regime: "panic" }, /regime turned PANIC/],
    [{ held: { qty: 10, price: 93.9 } }, /reached the entry plan's stop/],
    [{ nowMs: T0 + (SCALE_IN.MAX_AGE_DAYS + 1) * DAY }, /expired/],
  ];
  for (const [over, why] of cases) {
    const d = decideTranche(plan(), ctx(over));
    assert.equal(d.action, "cancel", String(why));
    assert.match(d.action === "cancel" ? d.reason : "", why);
  }
  // a plan judged in the same pass it was made: its first order still working is a wait, not "no longer held"
  const fresh = decideTranche(plan(), ctx({ held: null, orderWorking: true, etDate: "2026-11-09", nowMs: T0 + 5_000 }));
  assert.deepEqual(fresh, { action: "wait", reason: "AAPL is not held yet: its order is still working" });
  assert.match((decideTranche(plan(), ctx({ regime: "NEUTRAL" })) as { reason: string }).reason, /NEUTRAL regime opens no new stock trades/);
  assert.match((decideTranche(plan(), ctx({ orderWorking: true })) as { reason: string }).reason, /still working/);
  const p = plan();
  const c = ctx({ regime: "BEAR" });
  applyTrancheDecision(p, decideTranche(p, c), c);
  assert.equal(p.status, "cancelled");
  assert.deepEqual(p.tranches.map((t) => t.status), ["bought", "cancelled", "cancelled"]);
  assert.match(p.cancel_reason || "", /BEAR/);
});

test("etfStopFromStock doubles the stock stop percent for a 2x fund, held to 4%-9%", () => {
  assert.equal(etfStopFromStock(100, 97, 20), 18.8);   // 3% -> 6%
  assert.equal(etfStopFromStock(100, 99, 20), 19.2);   // 1% -> 2% -> floored at 4%
  assert.equal(etfStopFromStock(100, 90, 20), 18.2);   // 10% -> 20% -> capped at 9%
  assert.equal(etfStopFromStock(100, 0, 20), 0);
  assert.equal(etfStopFromStock(100, 101, 20), 0);
});

test("pending dollars count only unbought tranches of active plans", () => {
  const a = plan();
  const b = plan({ ticker: "MSFT", qty: 9, price: 400, stopPrice: 380 });
  b.status = "cancelled";
  assert.equal(pendingTrancheDollars({ version: 1, plans: { AAPL: a, MSFT: b } }), 20 * 100);
});

test("state round-trips, prunes old finished plans, and reads a missing or torn file as empty", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "scale-in-"));
  const file = path.join(dir, "state.json");
  assert.deepEqual(loadScaleInState(file), { version: 1, plans: {} });
  const live = plan();
  const old = plan({ ticker: "OLD" });
  old.status = "complete";
  old.updated_at = new Date(T0 - 30 * DAY).toISOString();
  saveScaleInState(file, { version: 1, plans: { AAPL: live, OLD: old } }, T0);
  const back = loadScaleInState(file);
  assert.deepEqual(Object.keys(back.plans), ["AAPL"]);
  assert.equal(back.plans.AAPL.tranches[1].status, "pending");
  fs.writeFileSync(file, "{torn");
  assert.deepEqual(loadScaleInState(file), { version: 1, plans: {} });
});
