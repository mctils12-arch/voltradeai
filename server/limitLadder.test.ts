// BACKSTOP LADDER (2026-11-05, [RULE-REVIEW], human-approved frozen-path
// change): server/limitLadder.ts replaces POS-KILL's market order for options
// with a marketable-limit ladder — the bid (sell) / ask (buy), one tickStep
// further every stepSeconds, and ONE market order for the remainder only after
// hardTimeoutSeconds. These tests run the pure module against a fake broker on
// a fake clock (every sleep advances it), including the human's named
// requirement: the timeout fallback.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  runLadder, ladderPrice, roundOptionPrice, ladderConfigFromEnv, ladderQuoteFromAlpaca,
  LADDER_DEFAULTS, LADDER_ENV, BACKSTOP_LADDER_STEP_SECONDS, BACKSTOP_LADDER_TICK_STEP,
  BACKSTOP_LADDER_HARD_TIMEOUT_SECONDS, BACKSTOP_LADDER_POLL_SECONDS, BACKSTOP_LADDER_MIN_PRICE,
  type LadderConfig, type LadderDeps, type LadderEvent, type LadderQuote, type LadderSubmit,
} from "./limitLadder";

// ── Fake broker on a fake clock ─────────────────────────────────────────────
type Order = { id: string; kind: "limit" | "market"; qty: number; price: number | null; status: string; filled_qty: number; filled_avg_price: number | null };
type Call = { t: number; op: "limit" | "market" | "cancel"; qty?: number; price?: number; id?: string };

function makeBroker(opts: {
  quote?: LadderQuote | null;
  /** Called on every order read and after every cancel request: mutate the order to simulate fills. */
  react?: (o: Order, t: number, why: "read" | "cancel") => void;
  /** Overrides the limit submission (rejections, unknown outcomes). */
  submitLimit?: (qty: number, price: number, t: number) => LadderSubmit | null;
  /** What the cancel request does: the default is Alpaca's real 204 quirk (cancel works, alpaca() throws on the empty body). */
  cancel?: (o: Order) => void;
  /** The live position size per read (supplies getPositionQty); "throw" makes the read fail. */
  position?: (t: number) => number | null | "throw";
} = {}) {
  let now = 1_000_000;
  let quote: LadderQuote | null = opts.quote === undefined ? { bid: 1.23, ask: 1.31 } : opts.quote;
  const orders = new Map<string, Order>();
  const calls: Call[] = [];
  const events: LadderEvent[] = [];
  let n = 0;
  const live = (): Order[] => [...orders.values()].filter((o) => !["filled", "canceled", "expired", "rejected"].includes(o.status));
  let maxLive = 0;
  const place = (kind: "limit" | "market", qty: number, price: number | null): LadderSubmit => {
    const id = `o${++n}`;
    const o: Order = { id, kind, qty, price, status: "new", filled_qty: 0, filled_avg_price: null };
    orders.set(id, o);
    maxLive = Math.max(maxLive, live().length);
    return { ok: true, order: { id, status: "new", filled_qty: "0", filled_avg_price: null } };
  };
  const deps: LadderDeps = {
    submitLimit: async (qty, price) => {
      calls.push({ t: now, op: "limit", qty, price });
      const override = opts.submitLimit?.(qty, price, now);
      return override ?? place("limit", qty, price);
    },
    submitMarket: async (qty) => {
      calls.push({ t: now, op: "market", qty });
      return place("market", qty, null);
    },
    cancel: async (id) => {
      calls.push({ t: now, op: "cancel", id });
      const o = orders.get(id)!;
      opts.react?.(o, now, "cancel");
      if (o.status === "filled") throw new Error("Alpaca 422: order is not cancelable");
      if (opts.cancel) return opts.cancel(o);
      o.status = "canceled";
      throw new SyntaxError("Unexpected end of JSON input"); // alpaca() on Alpaca's 204 No Content
    },
    getOrder: async (id) => {
      const o = orders.get(id)!;
      opts.react?.(o, now, "read");
      return { id: o.id, status: o.status, filled_qty: String(o.filled_qty), filled_avg_price: o.filled_avg_price === null ? null : String(o.filled_avg_price) };
    },
    getQuote: async () => quote,
    now: () => now,
    sleep: async (ms) => { now += ms; },
    onEvent: (e) => { events.push(e); },
  };
  let positionReads = 0;
  if (opts.position) {
    const position = opts.position;
    deps.getPositionQty = async () => {
      positionReads += 1;
      const v = position(now);
      if (v === "throw") throw new Error("Alpaca 500: upstream");
      return v;
    };
  }
  return {
    positionReads: () => positionReads,
    deps, calls, orders, events,
    t0: now,
    setQuote: (q: LadderQuote | null) => { quote = q; },
    jump: (ms: number) => { now += ms; },
    maxLive: () => maxLive,
    limits: () => calls.filter((c) => c.op === "limit"),
    markets: () => calls.filter((c) => c.op === "market"),
  };
}
const fill = (o: Order, qty: number, price: number): void => {
  o.filled_qty = qty;
  o.filled_avg_price = price;
  if (qty >= o.qty) o.status = "filled";
  else o.status = "partially_filled";
};

// ── Prices ──────────────────────────────────────────────────────────────────
test("ladderPrice: a sell starts at the bid and walks DOWN one tickStep per step; a buy starts at the ask and walks UP", () => {
  assert.deepEqual([0, 1, 2, 3].map((k) => ladderPrice("sell", 1.23, 1.31, k)), [1.23, 1.18, 1.13, 1.08]);
  assert.deepEqual([0, 1, 2, 3].map((k) => ladderPrice("buy", 1.23, 1.31, k)), [1.31, 1.36, 1.41, 1.46]);
  // float noise never moves a price across a cent: 0.80 - 7 * 0.05 is 0.44999999999999996 in floating point
  assert.equal(ladderPrice("sell", 0.8, 0.9, 7), 0.45);
  assert.equal(ladderPrice("buy", 0.8, 1.15, 0), 1.15, "1.15 * 100 is 114.99999999999999 in floating point");
});

test("ladderPrice: $0.01 increments below $3.00, $0.05 at or above; sells round down and buys round up (both stay marketable)", () => {
  assert.equal(ladderPrice("sell", 3.07, 3.2, 0), 3.05, "a 3.07 bid offers 3.05, never 3.10");
  assert.equal(ladderPrice("sell", 3.07, 3.2, 1), 3.0, "3.02 rounds down to the nickel");
  assert.equal(ladderPrice("sell", 3.07, 3.2, 2), 2.97, "below $3.00 the penny increment applies");
  assert.equal(ladderPrice("buy", 2.9, 3.02, 0), 3.05, "a 3.02 ask bids 3.05, never 3.00");
  assert.equal(ladderPrice("buy", 2.9, 2.98, 1), 3.05, "2.98 + 0.05 = 3.03 crosses $3.00 and rounds up to the nickel");
  assert.equal(ladderPrice("buy", 2.9, 2.931, 0), 2.94, "a fractional-cent ask rounds up");
  assert.equal(ladderPrice("sell", 2.939, 2.95, 0), 2.93, "a fractional-cent bid rounds down");
  assert.equal(roundOptionPrice(3, "sell"), 3);
  assert.equal(roundOptionPrice(2.995, "buy"), 3);
  assert.equal(roundOptionPrice(2.995, "sell"), 2.99);
  assert.equal(roundOptionPrice(12.42, "sell"), 12.4);
  assert.equal(roundOptionPrice(12.42, "buy"), 12.45);
});

test("ladderPrice: a sell never goes below minPrice; a quote without the needed side prices nothing", () => {
  assert.deepEqual([0, 1, 2, 3, 10].map((k) => ladderPrice("sell", 0.1, 0.2, k)), [0.1, 0.05, 0.01, 0.01, 0.01]);
  const cfg: LadderConfig = { ...LADDER_DEFAULTS, minPrice: 0.05 };
  assert.equal(ladderPrice("sell", 0.1, 0.2, 5, cfg), 0.05);
  assert.equal(ladderPrice("sell", 0, 0.2, 0), null, "no bid: a sell has no reference");
  assert.equal(ladderPrice("buy", 0.1, 0, 0), null, "no ask: a buy has no reference");
  assert.equal(ladderPrice("buy", 0.1, Number.NaN, 0), null);
  assert.equal(ladderPrice("sell", 1.0, 1.1, -3), 1.0, "a negative step is step 0");
});

test("defaults are the documented tunables; env overrides apply inside their bounds, anything else keeps the default and is reported", () => {
  assert.deepEqual({ ...LADDER_DEFAULTS }, { stepSeconds: 15, tickStep: 0.05, hardTimeoutSeconds: 120, pollSeconds: 5, minPrice: 0.01 });
  assert.equal(BACKSTOP_LADDER_STEP_SECONDS, 15);
  assert.equal(BACKSTOP_LADDER_TICK_STEP, 0.05);
  assert.equal(BACKSTOP_LADDER_HARD_TIMEOUT_SECONDS, 120);
  assert.equal(BACKSTOP_LADDER_POLL_SECONDS, 5);
  assert.equal(BACKSTOP_LADDER_MIN_PRICE, 0.01);
  assert.deepEqual(ladderConfigFromEnv({}), { config: { ...LADDER_DEFAULTS }, rejected: [] });
  const ok = ladderConfigFromEnv({
    VOLTRADE_BACKSTOP_LADDER_STEP_S: "10", VOLTRADE_BACKSTOP_LADDER_TICK: " 0.10 ", VOLTRADE_BACKSTOP_LADDER_TIMEOUT_S: "90",
    VOLTRADE_BACKSTOP_LADDER_POLL_S: "2", VOLTRADE_BACKSTOP_LADDER_MIN_PRICE: "0.021",
  });
  assert.deepEqual(ok, { config: { stepSeconds: 10, tickStep: 0.1, hardTimeoutSeconds: 90, pollSeconds: 2, minPrice: 0.03 }, rejected: [] },
    "minPrice is an order price: rounded up to whole cents");
  const bad = ladderConfigFromEnv({
    VOLTRADE_BACKSTOP_LADDER_STEP_S: "abc", VOLTRADE_BACKSTOP_LADDER_TICK: "0", VOLTRADE_BACKSTOP_LADDER_TIMEOUT_S: "601",
    VOLTRADE_BACKSTOP_LADDER_POLL_S: "-1", VOLTRADE_BACKSTOP_LADDER_MIN_PRICE: "NaN",
  });
  assert.deepEqual(bad.config, { ...LADDER_DEFAULTS }, "every invalid value keeps its default");
  assert.deepEqual(bad.rejected, [
    "VOLTRADE_BACKSTOP_LADDER_STEP_S=abc", "VOLTRADE_BACKSTOP_LADDER_TICK=0", "VOLTRADE_BACKSTOP_LADDER_TIMEOUT_S=601",
    "VOLTRADE_BACKSTOP_LADDER_POLL_S=-1", "VOLTRADE_BACKSTOP_LADDER_MIN_PRICE=NaN",
  ]);
  assert.deepEqual(ladderConfigFromEnv({ VOLTRADE_BACKSTOP_LADDER_STEP_S: "  " }).rejected, [], "a blank value is unset, not invalid");
  assert.ok(LADDER_ENV.stepSeconds.max * 1000 < 12 * 60_000, "a step never outlives bot.ts's 12-minute stale-order sweep");
});

test("ladderQuoteFromAlpaca: bp/ap become bid/ask; a side that is not a positive number reads 0; nothing usable is null", () => {
  assert.deepEqual(ladderQuoteFromAlpaca({ bp: 1.2, ap: "1.35", bs: 3 }), { bid: 1.2, ask: 1.35 });
  assert.deepEqual(ladderQuoteFromAlpaca({ bp: 0, ap: 0.4 }), { bid: 0, ask: 0.4 }, "no bid on the book");
  assert.equal(ladderQuoteFromAlpaca({ bp: "x", ap: -1 }), null);
  assert.equal(ladderQuoteFromAlpaca({}), null);
  assert.equal(ladderQuoteFromAlpaca(null), null);
  assert.equal(ladderQuoteFromAlpaca("1.2"), null);
});

// ── The ladder ──────────────────────────────────────────────────────────────
test("filled at step 0: one limit at the bid, no cancel, no market order", async () => {
  const b = makeBroker({ react: (o) => { if (o.kind === "limit") fill(o, o.qty, 1.24); } });
  const r = await runLadder(b.deps, { side: "sell", qty: 2 });
  assert.equal(r.status, "filled");
  assert.deepEqual(b.calls.map((c) => [c.op, c.qty, c.price]), [["limit", 2, 1.23]]);
  assert.equal(r.filledQty, 2);
  assert.equal(r.avgPrice, 1.24);
  assert.equal(r.steps, 1);
  assert.match(r.summary, /^2\/2 filled at avg \$1\.24 by limit after 1 step\(s\); no market order$/);
  assert.deepEqual(b.events.map((e) => e.type), ["submitted", "settled"]);
  assert.equal(b.events[0].line, "step 0: limit sell 2 @ $1.23 (bid 1.23 / ask 1.31) — order o1");
});

test("TIMEOUT FALLBACK: never filled — 8 limit steps walking down, then exactly ONE market order for the remainder, at or after the hard timeout and none before", async () => {
  const b = makeBroker({ quote: { bid: 0.8, ask: 0.9 } });
  const r = await runLadder(b.deps, { side: "sell", qty: 3 });
  const limits = b.limits();
  assert.deepEqual(limits.map((c) => c.price), [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45], "one tickStep further every step");
  assert.deepEqual(limits.map((c) => (c.t - b.t0) / 1000), [0, 15, 30, 45, 60, 75, 90, 105], "one step every stepSeconds");
  assert.ok(limits.every((c) => c.qty === 3));
  const markets = b.markets();
  assert.equal(markets.length, 1, "exactly one market order");
  assert.equal(markets[0].qty, 3, "for the whole unfilled remainder");
  assert.ok(markets[0].t - b.t0 >= 120_000, `the market order went out at +${(markets[0].t - b.t0) / 1000}s, before the 120 s timeout`);
  const lastLimitCancel = b.calls.filter((c) => c.op === "cancel").pop()!;
  assert.ok(b.calls.indexOf(lastLimitCancel) < b.calls.indexOf(markets[0]), "the last limit is cancelled before the market order");
  assert.ok(b.calls.slice(0, b.calls.indexOf(markets[0])).every((c) => c.op !== "market"), "no market order before the timeout");
  assert.equal(b.maxLive(), 1, "never two live orders");
  assert.equal(r.status, "market_fallback");
  assert.equal(r.steps, 8);
  assert.equal(r.orders.length, 9);
  assert.equal(r.orders[8].kind, "market");
  assert.match(r.summary, /^no full fill in 120s: market order for the remaining 3 after 8 limit step\(s\); 0\/3 filled$/);
});

test("TIMEOUT FALLBACK with a custom config: steps, ticks and the timeout all follow the config", async () => {
  const cfg: LadderConfig = { stepSeconds: 20, tickStep: 0.1, hardTimeoutSeconds: 50, pollSeconds: 5, minPrice: 0.01 };
  const b = makeBroker({ quote: { bid: 2.5, ask: 2.7 } });
  const r = await runLadder(b.deps, { side: "sell", qty: 1 }, cfg);
  assert.deepEqual(b.limits().map((c) => [(c.t - b.t0) / 1000, c.price]), [[0, 2.5], [20, 2.4], [40, 2.3]]);
  assert.equal(b.markets().length, 1);
  assert.ok(b.markets()[0].t - b.t0 >= 50_000);
  assert.equal(r.status, "market_fallback");
});

test("TIMEOUT FALLBACK after a partial fill: the market order is for the remainder only", async () => {
  const b = makeBroker({ react: (o) => { if (o.id === "o1" && o.filled_qty === 0) fill(o, 2, 1.23); } });
  const r = await runLadder(b.deps, { side: "sell", qty: 5 });
  assert.deepEqual(b.limits().map((c) => c.qty), [5, 3, 3, 3, 3, 3, 3, 3], "every later step offers only the unfilled 3");
  assert.deepEqual(b.markets().map((c) => c.qty), [3]);
  assert.equal(r.filledQty, 2);
  assert.equal(r.status, "market_fallback");
});

test("partial fill: the step is cancelled at its deadline and the REMAINING qty is offered one tick lower, then fills", async () => {
  const b = makeBroker({
    react: (o) => {
      if (o.id === "o1" && o.filled_qty === 0) fill(o, 2, 1.23);
      if (o.id === "o2") fill(o, o.qty, 1.18);
    },
  });
  const r = await runLadder(b.deps, { side: "sell", qty: 5 });
  assert.deepEqual(b.calls.map((c) => [c.op, c.qty ?? c.id, c.price]), [["limit", 5, 1.23], ["cancel", "o1", undefined], ["limit", 3, 1.18]]);
  assert.equal(r.status, "filled");
  assert.equal(r.filledQty, 5);
  assert.equal(r.avgPrice, (2 * 1.23 + 3 * 1.18) / 5, "volume-weighted over both orders");
  assert.equal(b.markets().length, 0);
  assert.equal(b.maxLive(), 1);
});

test("cancel race: the order fills just as the ladder cancels it — the cancel fails, the re-read shows the fill, and the ladder stops", async () => {
  const b = makeBroker({ react: (o, _t, why) => { if (why === "cancel" && o.id === "o1") fill(o, o.qty, 1.23); } });
  const r = await runLadder(b.deps, { side: "sell", qty: 2 });
  assert.deepEqual(b.calls.map((c) => c.op), ["limit", "cancel"], "no further order after the race");
  assert.equal(r.status, "filled");
  assert.equal(r.filledQty, 2);
  assert.equal(r.avgPrice, 1.23);
  assert.equal(r.orders[0].canceledByLadder, false, "a filled order is not reported as cancelled");
});

test("a cancel that cannot be confirmed stops the ladder: never a second live order, never a market order", async () => {
  const b = makeBroker({ cancel: (o) => { o.status = "pending_cancel"; } });
  const r = await runLadder(b.deps, { side: "sell", qty: 2 });
  assert.equal(r.status, "error");
  assert.match(r.reason, /^step 0: order o1 still pending_cancel after its cancel — it may still be working$/);
  assert.deepEqual(b.calls.map((c) => c.op), ["limit", "cancel"]);
  assert.equal(b.markets().length, 0);
});

test("a fresh quote prices each step; a missing quote reuses the last one; with no quote ever, the position mark is the reference", async () => {
  const quotes: Array<LadderQuote | null> = [{ bid: 1.0, ask: 1.1 }, { bid: 0.9, ask: 1.0 }, null, { bid: 0, ask: 0.95 }];
  const b = makeBroker();
  let i = 0;
  b.deps.getQuote = async () => (i < quotes.length ? quotes[i++] : { bid: 0.5, ask: 0.6 });
  const r = await runLadder(b.deps, { side: "sell", qty: 1, fallbackPrice: 2.0 });
  assert.deepEqual(b.limits().slice(0, 5).map((c) => c.price), [1.0, 0.85, 0.8, 0.75, 0.3],
    "step 1 = fresh 0.90 - 0.05; steps 2-3 = last usable 0.90 minus 2 and 3 ticks (no quote, then no bid); step 4 = fresh 0.50 - 0.20");
  assert.equal(r.status, "market_fallback");
  const refs = b.events.filter((e): e is Extract<LadderEvent, { type: "submitted" }> => e.type === "submitted").map((e) => e.ref?.source ?? "none");
  assert.deepEqual(refs.slice(0, 5), ["quote", "quote", "last_quote", "last_quote", "quote"]);

  const m = makeBroker({ quote: null });
  await runLadder(m.deps, { side: "sell", qty: 1, fallbackPrice: 0.8 });
  assert.deepEqual(m.limits().slice(0, 3).map((c) => c.price), [0.8, 0.75, 0.7], "no quote at all: the ladder walks down from the mark");
  assert.equal(m.events[0].line, "step 0: limit sell 1 @ $0.80 (position mark $0.80, no quote) — order o1");
  const thrown = makeBroker();
  thrown.deps.getQuote = async () => { throw new Error("quote feed down"); };
  await runLadder(thrown.deps, { side: "buy", qty: 1, fallbackPrice: 0.8 });
  assert.equal(thrown.limits()[0].price, 0.8, "a failed quote read falls back to the mark too");
});

test("no quote and no mark: nothing is sent until the hard timeout, then the one market order", async () => {
  const b = makeBroker({ quote: null });
  const r = await runLadder(b.deps, { side: "sell", qty: 2 });
  assert.equal(b.limits().length, 0);
  assert.deepEqual(b.markets().map((c) => [(c.t - b.t0) / 1000, c.qty]), [[120, 2]]);
  assert.equal(r.status, "market_fallback");
  assert.equal(r.steps, 0);
  assert.ok(b.events.some((e) => e.type === "skipped" && e.line === "step 0: no quote and no mark to price a limit — nothing sent this step"));
});

test("a buy (closing a short option) starts at the ask, walks UP, and falls back to one market buy", async () => {
  const b = makeBroker({ quote: { bid: 2.8, ask: 2.96 } });
  const r = await runLadder(b.deps, { side: "buy", qty: 1 });
  assert.deepEqual(b.limits().map((c) => c.price), [2.96, 3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35],
    "3.01 rounds up to the $3+ nickel, then one nickel per step");
  assert.equal(b.markets().length, 1);
  assert.equal(r.status, "market_fallback");
});

test("a definite broker rejection sends nothing that step; the walk continues on schedule and the timeout still applies", async () => {
  const b = makeBroker({
    quote: { bid: 1.0, ask: 1.1 },
    submitLimit: (_q, price) => (price >= 0.95 ? { ok: false, rejected: true, error: "Alpaca 422: invalid limit_price" } : null),
  });
  const r = await runLadder(b.deps, { side: "sell", qty: 1 });
  assert.deepEqual(b.limits().slice(0, 3).map((c) => [(c.t - b.t0) / 1000, c.price]), [[0, 1.0], [15, 0.95], [30, 0.9]],
    "the rejected steps are not retried early");
  assert.equal(b.orders.size, 7, "6 accepted limits + 1 market");
  assert.equal(r.steps, 8);
  assert.equal(r.status, "market_fallback");
  const rej = b.events.filter((e) => e.type === "rejected");
  assert.equal(rej.length, 2);
  assert.equal(rej[0].line, "step 0: limit sell 1 @ $1.00 rejected (Alpaca 422: invalid limit_price) — next step at its time");
});

test("an unknown submit outcome (timeout, 5xx, a throw) stops the ladder: no second limit, no market order", async () => {
  for (const submit of [
    (): LadderSubmit => ({ ok: false, rejected: false, error: "This operation was aborted" }),
    (): LadderSubmit => { throw new Error("socket hang up"); },
  ]) {
    const b = makeBroker({ submitLimit: (_q, _p, t) => (t > 1_000_000 ? submit() : null) });
    const r = await runLadder(b.deps, { side: "sell", qty: 1 });
    assert.equal(r.status, "error");
    assert.match(r.reason, /^step 1: limit sell 1 @ \$1\.18 failed with an unknown outcome \((This operation was aborted|socket hang up)\) — stopped rather than risk a second live order$/);
    assert.deepEqual(b.calls.map((c) => c.op), ["limit", "cancel", "limit"]);
    assert.equal(b.markets().length, 0);
  }
  const noId = makeBroker({ submitLimit: () => ({ ok: true, order: { status: "new" } }) });
  const r = await runLadder(noId.deps, { side: "sell", qty: 1 });
  assert.equal(r.status, "error");
  assert.match(r.reason, /returned no id/);
  assert.equal(noId.calls.length, 1);
});

test("an order cancelled or expired by someone else stops the ladder; it never resubmits against another actor", async () => {
  const b = makeBroker({ react: (o, t) => { if (o.id === "o1" && t >= 1_005_000) { o.filled_qty = 1; o.filled_avg_price = 1.23; o.status = "canceled"; } } });
  const r = await runLadder(b.deps, { side: "sell", qty: 3 });
  assert.equal(r.status, "error");
  assert.equal(r.reason, "step 0: order o1 ended canceled outside the ladder — not resubmitting");
  assert.equal(r.filledQty, 1);
  assert.deepEqual(b.calls.map((c) => c.op), ["limit"]);
});

test("an asynchronous rejection (status rejected after acceptance) is treated like a submit rejection", async () => {
  const b = makeBroker({ react: (o) => { if (o.id === "o1") o.status = "rejected"; if (o.id === "o2") fill(o, o.qty, 1.18); } });
  const r = await runLadder(b.deps, { side: "sell", qty: 1 });
  assert.deepEqual(b.limits().map((c) => [(c.t - b.t0) / 1000, c.price]), [[0, 1.23], [15, 1.18]], "the next step waits for its time");
  assert.equal(r.status, "filled");
});

test("a stalled clock: the concession follows elapsed time instead of firing the missed steps back to back", async () => {
  let stalled = false; // one 50 s stall, on step 0's first status read
  const b = makeBroker({ react: (o, _t, why) => { if (o.id === "o1" && why === "read" && !stalled) { stalled = true; b.jump(50_000); } } });
  await runLadder(b.deps, { side: "sell", qty: 1 });
  const limits = b.limits();
  assert.equal(limits[1].price, 1.08, "after a 50+ s stall the next order is step 3 (bid - 3 ticks), not step 1");
  assert.equal(b.markets().length, 1);
  assert.ok(b.markets()[0].t - b.t0 >= 120_000);
});

test("a throwing event observer never strands an order: the ladder still runs to its end", async () => {
  const b = makeBroker({ react: (o) => { if (o.id === "o2") fill(o, o.qty, 1.18); } });
  const original = console.error;
  const logged: string[] = [];
  console.error = (msg: string) => { logged.push(msg); };
  try {
    b.deps.onEvent = () => { throw new Error("audit db locked"); };
    const r = await runLadder(b.deps, { side: "sell", qty: 1 });
    assert.equal(r.status, "filled");
  } finally {
    console.error = original;
  }
  assert.ok(logged.some((l) => l.includes("[limit-ladder] event observer failed: audit db locked")));
});

test("every order after the first is capped at the live position (someone else closed part of it); the first order does not wait for a read", async () => {
  const b = makeBroker({ position: () => 1 });
  const r = await runLadder(b.deps, { side: "sell", qty: 3 });
  assert.deepEqual(b.limits().map((c) => c.qty), [3, 1, 1, 1, 1, 1, 1, 1]);
  assert.deepEqual(b.markets().map((c) => c.qty), [1], "the market fallback is capped too");
  assert.equal(b.positionReads(), 8, "steps 1-7 and the market order read the position; step 0 does not");
  assert.equal(r.status, "market_fallback");
});

test("a position closed elsewhere between steps stops the ladder: nothing more is sent (a sell for a gone long could open a short)", async () => {
  const b = makeBroker({ position: () => 0 });
  const r = await runLadder(b.deps, { side: "sell", qty: 2 });
  assert.equal(r.status, "error");
  assert.equal(r.reason, "step 1: the position is no longer held — closed outside the ladder; nothing more sent");
  assert.deepEqual(b.calls.map((c) => c.op), ["limit", "cancel"]);

  const late = makeBroker({ position: (t) => (t - 1_000_000 >= 120_000 ? 0 : 2) });
  const r2 = await runLadder(late.deps, { side: "sell", qty: 2 });
  assert.equal(late.markets().length, 0, "gone by the timeout: no market order");
  assert.equal(r2.reason, "hard timeout: the position is no longer held — closed outside the ladder; no market order sent");
  assert.equal(late.limits().length, 8);
});

test("an unreadable or unknown position keeps the ladder's own count", async () => {
  for (const position of [(): "throw" => "throw", (): null => null]) {
    const b = makeBroker({ position });
    await runLadder(b.deps, { side: "sell", qty: 3 });
    assert.deepEqual(b.limits().map((c) => c.qty), [3, 3, 3, 3, 3, 3, 3, 3]);
    assert.deepEqual(b.markets().map((c) => c.qty), [3]);
  }
});

test("nothing to close: qty 0 sends nothing", async () => {
  const b = makeBroker();
  const r = await runLadder(b.deps, { side: "sell", qty: 0 });
  assert.equal(r.status, "error");
  assert.equal(b.calls.length, 0);
});
