// manualOrderCheck.test.ts — the manual order's size check ([REPAIR] 2026-11-25): the order is valued at its own
// price, the entry checks run on buys only, and a check that cannot run refuses the buy. The second half pins the
// route's wiring in server/bot.ts (source pins, like finalOrderSitesStaleTracking.test.ts), including that the
// order POST itself is exactly as it was.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { manualOrderValue, stockPriceFromLatestTrade, optionPriceFromQuote, isManualEntry, sizeCheckBlocker, OCC_SYMBOL_RE, OPTION_MULTIPLIER } from "./manualOrderCheck";

test("an order is worth quantity x its own price; an option contract is 100 shares", () => {
  assert.equal(manualOrderValue(25, 10, false), 250);
  assert.equal(manualOrderValue("15", "2000", false), 30000);
  assert.equal(manualOrderValue(2, 1.35, true), 2 * 1.35 * OPTION_MULTIPLIER);
  assert.equal(manualOrderValue(0.5, 700, false), 350);
});

test("the defect, as numbers: on an $87,600 account the old estimate priced every share at $876", () => {
  const equity = 87_600, limit = 0.2;
  const old = (qty: number) => qty * (equity / 100);
  // 25 shares of a $10 stock: $250 (0.3% of the account) was refused; it now passes
  assert.ok(old(25) / equity > limit);
  assert.ok(manualOrderValue(25, 10, false)! / equity < limit);
  // 15 shares of a $2,000 stock: $30,000 (34%) passed; it is now over the limit
  assert.ok(old(15) / equity < limit);
  assert.ok(manualOrderValue(15, 2000, false)! / equity > limit);
});

test("no usable quantity or price is null, never a guess", () => {
  for (const [q, p] of [[0, 10], [-1, 10], [10, 0], [10, -5], [NaN, 10], [10, null], [null, 10], ["", 10], ["abc", 10], [10, undefined], [Infinity, 10]] as Array<[unknown, unknown]>) {
    assert.equal(manualOrderValue(q, p, false), null, `qty=${String(q)} price=${String(p)}`);
  }
});

test("prices are read from the broker's own shapes; anything else is null", () => {
  assert.equal(stockPriceFromLatestTrade({ symbol: "AAPL", trade: { t: "2026-11-25T14:31:00Z", p: 231.42, s: 100 } }), 231.42);
  assert.equal(stockPriceFromLatestTrade({ trade: { p: "231.42" } }), 231.42);
  for (const bad of [null, undefined, {}, { trade: null }, { trade: {} }, { trade: { p: 0 } }, { trade: { p: -1 } }, { message: "not found" }, "x"]) assert.equal(stockPriceFromLatestTrade(bad), null);
  assert.equal(optionPriceFromQuote({ bid: 1.2, ask: 1.5 }), 1.35);
  assert.equal(optionPriceFromQuote({ bid: 0, ask: 1.5 }), 1.5);
  assert.equal(optionPriceFromQuote({ bid: 1.2, ask: 0 }), 1.2);
  assert.equal(optionPriceFromQuote({ bid: 0, ask: 0 }), null);
  assert.equal(optionPriceFromQuote(null), null);
  assert.ok(OCC_SYMBOL_RE.test("AAL261231P00011500") && !OCC_SYMBOL_RE.test("AAL"));
});

test("entry checks are for buys; a check that cannot run refuses the buy, in plain words", () => {
  assert.equal(isManualEntry("buy"), true);
  assert.equal(isManualEntry(" BUY "), true);
  for (const s of ["sell", "short", "", null, undefined, "hold"]) assert.equal(isManualEntry(s), false);
  assert.equal(sizeCheckBlocker("AAPL", 87000, 2300), null);
  assert.equal(sizeCheckBlocker("AAPL", 87000, null), "No current price for AAPL, so the size check could not run. Order not sent.");
  for (const pv of [NaN, 0, -1, null, undefined, "x"]) assert.equal(sizeCheckBlocker("AAPL", pv, 2300), "The account's value could not be read, so the size check could not run. Order not sent.");
});

// ── the route ────────────────────────────────────────────────────────────────
const here = path.dirname(fileURLToPath(import.meta.url));
const bot = fs.readFileSync(path.join(here, "bot.ts"), "utf8");
const a = bot.indexOf('app.post("/api/bot/trade"');
const route = bot.slice(a, bot.indexOf("// Close a position", a));

test("route: the placeholder price is gone and the check runs on the instrument's own price, for buys only", () => {
  assert.ok(a > 0 && route.length > 500);
  assert.ok(!/last_equity\)\s*\/\s*100/.test(route) && !route.includes("currentPrice || 1"), "the equity/100 placeholder must not come back");
  const gate = route.indexOf("if (isManualEntry(side)) {");
  const check = route.indexOf("await checkTradeAllowed(portfolioValue, orderValue as number)");
  const post = route.indexOf('await alpaca("/v2/orders", {');
  assert.ok(gate > 0 && check > gate && post > check, "entry checks sit inside the buy branch, before the order");
  assert.equal((route.match(/checkTradeAllowed\(/g) || []).length, 1, "one entry check, in the buy branch");
  assert.ok(route.includes("orderValue = manualOrderValue(qty, await manualOrderPrice(sym), isOptionSymbol(sym));"));
});

test("route: a buy with no price or no account value is refused and logged, never sent unchecked", () => {
  const blockAt = route.indexOf("const blocker = sizeCheckBlocker(sym, portfolioValue, orderValue);");
  const refuse = route.indexOf("return res.status(400).json({ error: blocker });");
  const post = route.indexOf('await alpaca("/v2/orders", {');
  assert.ok(blockAt > 0 && refuse > blockAt && post > refuse);
  assert.ok(route.includes('audit("BLOCKED", `Manual BUY ${qty} ${sym} not sent: ${blocker}`);'));
});

test("route: how the order is built and sent is exactly as before (frozen transmission path)", () => {
  assert.ok(route.includes(`      const order = await alpaca("/v2/orders", {
        method: "POST",
        body: JSON.stringify({
          symbol: ticker.toUpperCase(),
          qty: String(qty),
          side: orderSide,
          type,
          time_in_force: "day",
        }),
      });`));
  // the kill refusal and the no-shorting rule are untouched
  assert.ok(route.includes('if (state.killSwitch) return res.status(400).json({ error: "Kill switch is ON" });'));
  assert.ok(route.includes('return res.status(400).json({ error: "Stock shorting is disabled. Use options for bearish plays." });'));
});

test("the price reader asks the data API for the latest trade (stocks) or the option quote, and never throws", () => {
  const p = bot.indexOf("async function manualOrderPrice(symbol: string): Promise<number | null> {");
  assert.ok(p > 0);
  const fn = bot.slice(p, bot.indexOf("\n  }\n", p));
  assert.ok(fn.includes("if (isOptionSymbol(symbol)) return optionPriceFromQuote(await posKillOptionQuote(symbol));"));
  assert.ok(fn.includes("/trades/latest?feed=iex") && fn.includes("AbortSignal.timeout(8000)") && fn.includes("return stockPriceFromLatestTrade(await r.json());"));
  assert.ok(/catch \(err\) \{[\s\S]*return null;/.test(fn));
});
