// Regression tests — audit §6.8 N6 (ETF substitution), [REPAIR] 2026-10-27.
// executeTrades bought Math.floor(trade.shares / 2) ETF shares (half the
// UNDERLYING's share count) at getOrderParams(trade.price) (the UNDERLYING's
// price). It now places exactly the scan's dollar-sized ETF plan, priced off
// the ETF's own ask, and falls back to the stock when there is no plan.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { planEtfOrder } from "./etfSubstitution";

const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");

const base = { ticker: "XLF", instrument: "etf", instrument_ticker: "FAS" };

test("uses the scan's ETF share count and ETF price, not the underlying's", () => {
  const plan = planEtfOrder({ ...base, etf_shares: 33, etf_price: 150, etf_position_value: 4950 });
  assert.deepEqual(plan, { ok: true, symbol: "FAS", qty: 33, price: 150, value: 4950 });
});

test("no plan from the scan -> not ok (caller trades the stock), never a guess", () => {
  for (const t of [
    { ...base },
    { ...base, etf_shares: 0, etf_price: 150 },
    { ...base, etf_shares: 2.5, etf_price: 150 },
    { ...base, etf_shares: 10, etf_price: 0 },
    { ...base, etf_shares: 10, etf_price: null },
  ]) {
    assert.equal(planEtfOrder(t).ok, false, JSON.stringify(t));
  }
});

test("a non-ETF or self-referencing trade is not an ETF substitution", () => {
  assert.equal(planEtfOrder({ ticker: "XLF", instrument: "stock", instrument_ticker: "XLF" }).ok, false);
  assert.equal(planEtfOrder({ ticker: "XLF", instrument: "etf", instrument_ticker: "XLF", etf_shares: 5, etf_price: 40 }).ok, false);
});

test("value falls back to qty x price when the scan omitted it", () => {
  const plan = planEtfOrder({ ...base, etf_shares: 4, etf_price: 25 });
  assert.ok(plan.ok && plan.value === 100);
});

test("bot.ts ETF branch is wired to the plan (no underlying shares/2, no underlying price)", () => {
  const start = bot.indexOf("── ETF execution (instrument selector chose");
  const end = bot.indexOf("── Options vs Stock execution ──", start);
  assert.ok(start > 0 && end > start, "ETF branch markers not found");
  const branch = bot.slice(start, end);
  assert.ok(branch.includes("planEtfOrder(trade)"), "ETF branch must use planEtfOrder");
  assert.ok(!/trade\.shares\s*\/\s*2/.test(branch), "ETF qty must not be derived from the underlying's share count");
  assert.ok(!/getOrderParams\(trade\.price/.test(branch), "ETF limit must not be priced off the underlying");
  assert.ok(branch.includes("getOrderParams(etfPrice)"), "ETF limit must be priced off the ETF plan price");
  assert.ok(branch.includes("fillPrice: etfPrice"), "entry fill record must carry the ETF price");
  assert.ok(branch.includes("totalDeployed += etfPlan.value"), "heat tracking must add the ETF dollars");
});
