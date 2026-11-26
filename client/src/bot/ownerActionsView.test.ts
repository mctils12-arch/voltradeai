// ownerActionsView.test — what each owner button says before it acts, from
// the page's own data; and that the two server constants the wording quotes
// are still the server's.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  breakerView, BREAKER_STOPS, BREAKER_RESET_CONFIRM, scanConfirm, scanResult, closeChoices, closeConfirm, reviewOrder, orderResult,
  lastPriceFromBars, ownerActionsSentence, MANUAL_POSITION_LIMIT, refusalWords,
} from "./ownerActionsView.ts";

test("circuit breaker: off, a streak, on with its end time; a server that does not report it says so", () => {
  assert.deepEqual(breakerView({ circuitBreakerActive: false, circuitBreakerUntil: null, consecutiveStopLosses: 0 }),
    { known: true, active: false, until: null, stops: 0, sentence: "Circuit breaker off · no stop-loss streak", canReset: false });
  const streak = breakerView({ circuitBreakerActive: false, circuitBreakerUntil: null, consecutiveStopLosses: 2 });
  assert.equal(streak.sentence, "Circuit breaker off · 2 stop-losses in a row (it trips at 3)");
  assert.equal(streak.canReset, true);
  assert.equal(breakerView({ circuitBreakerActive: false, consecutiveStopLosses: 1 }).sentence, "Circuit breaker off · 1 stop-loss in a row (it trips at 3)");
  const on = breakerView({ circuitBreakerActive: true, circuitBreakerUntil: "2026-11-25T16:30:00Z", consecutiveStopLosses: 3 });
  assert.equal(on.sentence, "Circuit breaker ON until 11:30 ET: new entries are paused · 3 stop-losses in a row (it trips at 3)");
  assert.deepEqual([on.active, on.canReset], [true, true]);
  for (const s of [null, undefined, {}, { active: true }, "x"]) {
    assert.deepEqual(breakerView(s), { known: false, active: false, until: null, stops: null, sentence: "Circuit breaker: state not reported by this server", canReset: false });
  }
  assert.match(BREAKER_RESET_CONFIRM, /Exits were never paused\.$/);
});

test("scan now: the consequence by market state, and the server's two answers", () => {
  assert.match(scanConfirm(true), /can place orders exactly as a scheduled scan does\.$/);
  assert.match(scanConfirm(false), /The market is closed: the order gate refuses orders until it opens\.$/);
  assert.match(scanConfirm(null), /Market state not known\.$/);
  assert.equal(scanResult({ message: "Tier 2 intelligence scan starting..." }), "Scan started. Its rows appear in the timeline as it runs.");
  assert.equal(scanResult({ message: "Tier 2 scan already running..." }), "A scan is already running: nothing new was started.");
  assert.equal(scanResult({}), "The server answered without a message.");
});

const POSITIONS = [
  { ticker: "SPY", qty: 72, side: "long", marketValue: 49250.4, isOption: false },
  { ticker: "BOXX", qty: 287, side: "long", marketValue: 32718.1, isOption: false },
  { ticker: "NVDA", qty: 12, side: "long", marketValue: 2174.4, isOption: false },
  { ticker: "AAL261231P00011500", qty: -1, side: "short", marketValue: -135, isOption: true, optionMeta: { underlyingTicker: "AAL", strategy: "cash_secured_put" } },
  { ticker: "QQQ261218C00520000", qty: 1, side: "long", marketValue: 410, asset_class: "us_option", optionMeta: { underlyingTicker: "QQQ", strategy: "bull_call_spread" } },
  { ticker: "QQQ261218C00540000", qty: -1, side: "short", marketValue: -180, asset_class: "us_option", optionMeta: { underlyingTicker: "QQQ", strategy: "bull_call_spread" } },
  { ticker: "ZERO", qty: 0, marketValue: 0 }, null, { qty: 3 },
];

test("close: every held position with what closing it means; the base holding, short options and spread legs carry their warning", () => {
  const c = closeChoices(POSITIONS, ["SPY", "SSO", "boxx", null]);
  assert.deepEqual(c.map((x) => x.symbol), ["SPY", "BOXX", "NVDA", "AAL261231P00011500", "QQQ261218C00520000", "QQQ261218C00540000"]);
  assert.equal(c[0].label, "SPY · 72 sh · $49,250");
  assert.deepEqual(c[0].warnings, ["Part of the base holding: the bot buys it back at its next rebalance."]);
  assert.deepEqual(c[1].warnings, ["Part of the base holding: the bot buys it back at its next rebalance."]);
  assert.deepEqual([c[2].label, c[2].warnings], ["NVDA · 12 sh · $2,174", []]);
  assert.match(c[3].label, /^AAL 11\.5P 31-Dec(-26)? · 1 ct short · \$135$/);
  assert.deepEqual(c[3].warnings, ["A short option: closing buys it back at the market price.", "Option quotes here are indicative: a market order can fill far from the price shown."]);
  assert.deepEqual(c[4].warnings, ["One of 2 legs the bot manages together: closing it alone leaves the rest open.", "Option quotes here are indicative: a market order can fill far from the price shown."]);
  assert.equal(c[5].warnings.length, 3);
  assert.equal(closeConfirm(c[2], true), "Closes NVDA · 12 sh · $2,174: one market order for the whole position, sent now.");
  assert.match(closeConfirm(c[2], false), /The market is closed: the order gate refuses it, except a held stock inside an extended session\.$/);
  assert.match(closeConfirm(c[3], false), /The market is closed: the order gate refuses it until the open\.$/);
  assert.deepEqual(closeChoices(null), []);
  assert.deepEqual(closeChoices({ error: "x" }), []);
});

const OPEN = { price: 231.42, priceAt: "2026-11-25T15:04:00Z", heldQty: 0, equity: 87000, killSwitch: false, marketOpen: true };

test("order: a clean buy reads back its size, price and share of the account", () => {
  const r = reviewOrder({ ticker: " aapl ", side: "buy", qty: "10" }, OPEN);
  assert.equal(r.ok, true);
  assert.deepEqual(r.problems, []);
  assert.equal(r.sentence, "BUY 10 AAPL at market · about $2,314 at the last trade ($231.42, 10:04 ET) · 2.7% of the account");
  assert.deepEqual(r.notes, ["A market order fills at the next available price, which can differ from the last trade."]);
  assert.deepEqual([r.symbol, r.qty, Math.round(r.value!)], ["AAPL", 10, 2314]);
});

test("order: every refusal the server would give is said before the click", () => {
  const p = (d: Parameters<typeof reviewOrder>[0], ctx = OPEN) => reviewOrder(d, ctx).problems;
  assert.deepEqual(p({ ticker: "", side: "buy", qty: "1" }), ["Enter a ticker."]);
  assert.deepEqual(p({ ticker: "AAL261231P00011500", side: "buy", qty: "1" }), ["Stocks and ETFs only here: a ticker of up to six letters. Options are not placed from this form."]);
  for (const q of ["", "0", "-3", "1.5", "ten", "1e3"]) assert.ok(p({ ticker: "AAPL", side: "buy", qty: q }).includes("Quantity must be a whole number of shares, 1 or more."), q);
  // over the server's 20% position ceiling: 100 x $231.42 = $23,142 = 27% of $87,000
  assert.deepEqual(p({ ticker: "AAPL", side: "buy", qty: "100" }), ["About 27% of the account: the server refuses a manual buy over 20%."]);
  assert.deepEqual(p({ ticker: "AAPL", side: "buy", qty: "10" }, { ...OPEN, price: null }), ["No last price for AAPL: the server's size check refuses a buy it cannot price."]);
  assert.deepEqual(p({ ticker: "AAPL", side: "buy", qty: "10" }, { ...OPEN, marketOpen: false }), ["The market is closed: the order gate refuses a buy until it opens."]);
  assert.deepEqual(p({ ticker: "AAPL", side: "buy", qty: "10" }, { ...OPEN, killSwitch: true }), ["The kill switch is latched: the server refuses every manual order until it is cleared. Closing a position still works."]);
  // sells: only what is held, never short
  assert.deepEqual(p({ ticker: "AAPL", side: "sell", qty: "5" }), ["You hold no AAPL: this route does not sell short."]);
  assert.deepEqual(p({ ticker: "AAPL", side: "sell", qty: "50" }, { ...OPEN, heldQty: 12 }), ["You hold 12 AAPL: a sell cannot exceed it."]);
  assert.deepEqual(p({ ticker: "AAPL", side: "sell", qty: "5" }, { ...OPEN, heldQty: null }), ["Positions are not loaded, so the holding cannot be checked."]);
  // a sell is not sized against the 20% entry ceiling, and a closed market is a note (an extended session may take it)
  const big = reviewOrder({ ticker: "SPY", side: "sell", qty: "60" }, { ...OPEN, price: 684, heldQty: 72, marketOpen: false });
  assert.equal(big.ok, true);
  assert.equal(big.notes[0], "The market is closed: the order gate refuses the sale unless an extended session is open.");
  assert.equal(big.sentence, "SELL 60 SPY at market · about $41,040 at the last trade ($684.00, 10:04 ET) · 47.2% of the account");
  // no price: a sell is still reviewable, and says so
  assert.equal(reviewOrder({ ticker: "SPY", side: "sell", qty: "1" }, { ...OPEN, price: null, heldQty: 72 }).sentence, "SELL 1 SPY at market · no last price");
  assert.ok(reviewOrder({ ticker: "AAPL", side: "buy", qty: "1" }, { ...OPEN, marketOpen: null }).notes.includes("Market state not known: the server decides when the order arrives."));
});

test("answers: the broker's order object, the last bar, the block's one line", () => {
  assert.equal(orderResult({ id: "61e69015-8549-4bfd-b9c3-01e75843f47d", status: "accepted" }), 'Order sent: the broker says "accepted" · id 61e69015. It appears under working orders until it fills.');
  assert.equal(orderResult({ id: "abc", status: "pending_new" }), 'Order sent: the broker says "pending new" · id abc. It appears under working orders until it fills.');
  assert.equal(orderResult({}), "The server answered, but not with an order: check the timeline and the working orders.");
  assert.deepEqual(lastPriceFromBars({ bars: [{ time: 1795530000, close: 230.1 }, { time: 1795530240, close: 231.42 }] }), { price: 231.42, at: new Date(1795530240000).toISOString() });
  for (const bad of [null, {}, { bars: [] }, { bars: [{ time: 1, close: 0 }] }, { bars: [{ close: 5 }] }, { error: "x" }]) assert.equal(lastPriceFromBars(bad), null);
  const b = breakerView({ circuitBreakerActive: false, consecutiveStopLosses: 0 });
  assert.equal(ownerActionsSentence(b, 6), "Circuit breaker off · no stop-loss streak · scan now, close one of 6 positions, or place an order: each asks before it acts");
  assert.equal(ownerActionsSentence(b, 1), "Circuit breaker off · no stop-loss streak · scan now, close one of 1 position, or place an order: each asks before it acts");
  assert.equal(ownerActionsSentence(b, null), "Circuit breaker off · no stop-loss streak · scan now, close a position, or place an order: each asks before it acts");
});

test("the two server figures the wording quotes are still the server's", () => {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const bot = fs.readFileSync(path.join(here, "../../../server/bot.ts"), "utf8");
  assert.ok(bot.includes(`const MAX_POSITION_SIZE = ${MANUAL_POSITION_LIMIT.toFixed(2)};`), "the manual buy ceiling moved: update MANUAL_POSITION_LIMIT");
  assert.ok(bot.includes(`if (state.consecutiveStopLosses >= ${BREAKER_STOPS}) {`), "the breaker's trip count moved: update BREAKER_STOPS");
});

test("a refusal is the server's own reason, and a timeout warns the action may have gone through", () => {
  assert.equal(refusalWords(new Error('400: {"error":"Kill switch is ON"}')), "Not done. The server refused: Kill switch is ON");
  assert.equal(refusalWords(new Error('400: {"error":"No current price for ZZZZ, so the size check could not run. Order not sent."}')), "Not done. The server refused: No current price for ZZZZ, so the size check could not run. Order not sent.");
  assert.equal(refusalWords(new Error('500: {"error":"ORDER-BLOCKED: market closed"}')), "Not done. Server error 500: ORDER-BLOCKED: market closed");
  assert.equal(refusalWords(new Error("502: Bad Gateway")), "Not done. Server error 502: Bad Gateway");
  assert.equal(refusalWords(new Error("Failed to fetch")), "Not done. Failed to fetch");
  assert.equal(refusalWords(new Error("timeout: no answer from /api/bot/close in 20000 ms")), "No answer from the server in time. Check the timeline before trying again: the action may have gone through.");
  assert.equal(refusalWords(null), "Not done. no reason given");
});
