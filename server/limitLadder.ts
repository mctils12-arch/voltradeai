/**
 * server/limitLadder.ts — the marketable-limit ladder POS-KILL uses to exit an
 * OPTION position (2026-11-05, [RULE-REVIEW]). Human-approved change to the
 * frozen order path, verbatim: "Option backstop/emergency exit: I approve
 * changing this protected path. Replace market orders with a marketable-limit
 * ladder: start at the bid (for sells) or ask (for buys), widen by a set tick
 * step every N seconds, and fall back to a market order only after a hard
 * timeout. Keep it configurable and unit-test the timeout fallback."
 * Context: research/wishlist.md "POS-KILL option order type (frozen path) —
 * 2026-11-04" (a market order on a wide option spread is the worst case of
 * the fills research/options_bleed_2026-11-04.md measured).
 *
 * THE LADDER (runLadder):
 *   step 0   a limit at the bid (sell) / ask (buy) of a fresh quote;
 *   step k   every stepSeconds the unfilled remainder is cancelled and offered
 *            again k * tickStep further through the market (sell: lower, buy:
 *            higher), from a fresh quote when one is available;
 *   timeout  after hardTimeoutSeconds, ONE market order for whatever is still
 *            unfilled. It is the only market order the ladder ever sends.
 * Prices round to a valid option increment ($0.01 below $3.00, $0.05 at or
 * above) toward the marketable side: sells round down, buys round up. A sell is
 * never offered below minPrice.
 *
 * SAFETY RULES (each one has a test in limitLadder.test.ts):
 *   - Never two live orders. The next order goes out only after an order read
 *     shows the previous one terminal. A cancel that fails because the order
 *     just filled is settled by that read, and the ladder stops.
 *   - Never more than the position. Each new order is for qty minus every
 *     confirmed fill, and a "filled" status counts the order's whole qty.
 *     Every order after the first is also capped at the live position
 *     (getPositionQty, when supplied): another actor may close some or all of
 *     it between steps, and a sell for a position that is gone could OPEN a
 *     short option. A live position of 0 stops the ladder.
 *   - An order whose outcome is unknown (transport error, 5xx, no order id)
 *     stops the ladder: sending another could double the exit. A definite
 *     broker rejection (submitted with ok:false, rejected:true — a 4xx) sends
 *     nothing that step; the walk continues, and the timeout still applies.
 *   - An order cancelled or expired by anyone else stops the ladder. It never
 *     fights another actor.
 *   - No usable quote: the last usable quote, else params.fallbackPrice (POS-KILL
 *     passes the position's mark, which the mark guard has already validated).
 *     With neither, that step sends nothing, and the timeout still applies.
 *
 * Pure and dependency-injected: no network, no clock and no broker endpoint
 * strings live here. server/bot.ts supplies submit/cancel/read through its
 * unchanged order transport, the quote, the clock and the sleep, and audits
 * every event this module emits.
 */

export type LadderSide = "buy" | "sell";

export interface LadderConfig {
  /** Seconds each limit price rests before it is cancelled and re-priced. */
  stepSeconds: number;
  /** Dollars each step concedes past the reference price (sell: lower, buy: higher). */
  tickStep: number;
  /** Seconds from the ladder's start to its market-order fallback. */
  hardTimeoutSeconds: number;
  /** Seconds between order-status reads while an order rests. */
  pollSeconds: number;
  /** Lowest price a sell is ever offered at, in dollars (whole cents). */
  minPrice: number;
}

// ── Tunables (2026-11-05, [RULE-REVIEW] backstop ladder, human-approved) ────
// 15 s per step: a limit at the current quote is marketable, so it should
// fill on its first status reads; 15 s (three reads) leaves room for a quote
// that moved, and 8 price steps still fit inside the 2-minute timeout.
export const BACKSTOP_LADDER_STEP_SECONDS = 15;
// $0.05 per step: one nickel (the $3.00+ option increment; five pennies below
// $3.00). Seven concessions reach $0.35 past the current quote before the
// market fallback.
export const BACKSTOP_LADDER_TICK_STEP = 0.05;
// 120 s to the market fallback: the human required a hard timeout, and it
// bounds the extra exposure. The kill has already waited for two confirming
// sync ticks before the ladder starts.
export const BACKSTOP_LADDER_HARD_TIMEOUT_SECONDS = 120;
// 5 s between status reads: 3 reads per step and at most ~24 per 2-minute
// ladder. bot.ts's alpaca() has no client-side rate limiter, and the account's
// Alpaca limit (200 requests/minute, alpaca_rate_limiter.py) is shared by the
// whole bot.
export const BACKSTOP_LADDER_POLL_SECONDS = 5;
// $0.01: the smallest valid option limit price. A sell never goes below it.
export const BACKSTOP_LADDER_MIN_PRICE = 0.01;
// After a cancel request: read the order up to 5 times, 1 s apart, until it is
// terminal. Alpaca normally settles a cancel within a second. An order still
// live after that stops the ladder rather than risk a second live order.
export const BACKSTOP_LADDER_CANCEL_CONFIRM_READS = 5;
export const BACKSTOP_LADDER_CANCEL_CONFIRM_INTERVAL_MS = 1000;
// Timeout for one option-quote read in bot.ts (posKillOptionQuote). A slow
// quote must not hold the kill tick: on timeout the ladder uses its last
// quote or the mark.
export const BACKSTOP_LADDER_QUOTE_TIMEOUT_MS = 5000;

export const LADDER_DEFAULTS: Readonly<LadderConfig> = Object.freeze({
  stepSeconds: BACKSTOP_LADDER_STEP_SECONDS,
  tickStep: BACKSTOP_LADDER_TICK_STEP,
  hardTimeoutSeconds: BACKSTOP_LADDER_HARD_TIMEOUT_SECONDS,
  pollSeconds: BACKSTOP_LADDER_POLL_SECONDS,
  minPrice: BACKSTOP_LADDER_MIN_PRICE,
});

/**
 * Env override for each tunable and the inclusive bounds a value must sit in.
 * A value that is not a finite number inside its bounds is ignored (default
 * kept) and reported by ladderConfigFromEnv. stepSeconds <= 300 keeps every
 * ladder order younger than bot.ts's 12-minute stale-order sweep, so the
 * sweeper never cancels an order the ladder is still managing.
 */
export const LADDER_ENV: Readonly<Record<keyof LadderConfig, { name: string; min: number; max: number }>> = Object.freeze({
  stepSeconds: { name: "VOLTRADE_BACKSTOP_LADDER_STEP_S", min: 1, max: 300 },
  tickStep: { name: "VOLTRADE_BACKSTOP_LADDER_TICK", min: 0.01, max: 1 },
  hardTimeoutSeconds: { name: "VOLTRADE_BACKSTOP_LADDER_TIMEOUT_S", min: 1, max: 600 },
  pollSeconds: { name: "VOLTRADE_BACKSTOP_LADDER_POLL_S", min: 1, max: 60 },
  minPrice: { name: "VOLTRADE_BACKSTOP_LADDER_MIN_PRICE", min: 0.01, max: 1 },
});

/** The ladder config with env overrides applied; `rejected` lists each ignored `NAME=value`. */
export function ladderConfigFromEnv(env: Record<string, string | undefined>): { config: LadderConfig; rejected: string[] } {
  const config: LadderConfig = { ...LADDER_DEFAULTS };
  const rejected: string[] = [];
  for (const key of Object.keys(LADDER_ENV) as Array<keyof LadderConfig>) {
    const { name, min, max } = LADDER_ENV[key];
    const raw = env[name];
    if (raw === undefined || raw.trim() === "") continue;
    const value = Number(raw.trim());
    if (!Number.isFinite(value) || value < min || value > max) {
      rejected.push(`${name}=${raw}`);
      continue;
    }
    // minPrice is an order price, so it is held to whole cents (rounded up).
    config[key] = key === "minPrice" ? Math.ceil(value * 100 - 1e-9) / 100 : value;
  }
  return { config, rejected };
}

/** Round to a valid option increment toward the marketable side: sells down, buys up. */
export function roundOptionPrice(price: number, side: LadderSide): number {
  // Work in 1/100 cents so float noise (0.80 - 0.35 = 0.44999999999999996)
  // cannot push a price across an increment boundary.
  const units = Math.round(price * 10000);
  const inc = price >= 3 ? 500 : 100;
  const n = side === "sell" ? Math.floor(units / inc) : Math.ceil(units / inc);
  return (n * inc) / 10000;
}

/**
 * The limit price of ladder step `step`: a sell starts at the bid and walks
 * down tickStep per step, a buy starts at the ask and walks up. Rounded to a
 * valid increment (sells down, buys up), never below minPrice. null when the
 * side the order needs (bid for a sell, ask for a buy) is not a positive number.
 */
export function ladderPrice(side: LadderSide, bid: number, ask: number, step: number,
                            cfg: LadderConfig = LADDER_DEFAULTS): number | null {
  const ref = side === "sell" ? bid : ask;
  if (!Number.isFinite(ref) || ref <= 0) return null;
  const k = Number.isFinite(step) && step > 0 ? Math.floor(step) : 0;
  const raw = side === "sell" ? ref - k * cfg.tickStep : ref + k * cfg.tickStep;
  return Math.max(roundOptionPrice(raw, side), cfg.minPrice);
}

export interface LadderQuote { bid: number; ask: number }

/** An Alpaca option quote ({bp, ap}) as a LadderQuote; a side that is not a positive number reads 0. null when both are. */
export function ladderQuoteFromAlpaca(raw: unknown): LadderQuote | null {
  if (!raw || typeof raw !== "object") return null;
  const q = raw as { bp?: unknown; ap?: unknown };
  const bid = Number(q.bp);
  const ask = Number(q.ap);
  const b = Number.isFinite(bid) && bid > 0 ? bid : 0;
  const a = Number.isFinite(ask) && ask > 0 ? ask : 0;
  if (b === 0 && a === 0) return null;
  return { bid: b, ask: a };
}

/** The fields of a broker order the ladder reads (Alpaca's order object). */
export interface LadderBrokerOrder {
  id?: unknown;
  status?: unknown;
  filled_qty?: unknown;
  filled_avg_price?: unknown;
}

/** What a submit returns: the accepted order, or a failure that says whether the broker definitely rejected it. */
export type LadderSubmit =
  | { ok: true; order: LadderBrokerOrder }
  | { ok: false; rejected: boolean; error: string };

export interface LadderOrder {
  id: string;
  kind: "limit" | "market";
  /** The ladder step this order was sent at (the market order carries the step the timeout fell in). */
  step: number;
  /** Limit price; null for the market order. */
  price: number | null;
  /** Quantity submitted (the remainder at the time). */
  qty: number;
  submittedAt: number;
  /** Last status read from the broker. */
  status: string;
  filledQty: number;
  avgPrice: number | null;
  /** True once the ladder's own cancel request ended this order unfilled or partly filled. */
  canceledByLadder: boolean;
}

export interface LadderRef { source: "quote" | "last_quote" | "mark"; bid: number; ask: number }

export type LadderEvent =
  | { type: "submitted"; step: number; order: LadderOrder; ref: LadderRef | null; line: string }
  | { type: "rejected"; step: number; price: number; qty: number; error: string; line: string }
  | { type: "skipped"; step: number; line: string }
  | { type: "settled"; step: number; order: LadderOrder; line: string };

export interface LadderDeps {
  submitLimit(qty: number, price: number): Promise<LadderSubmit>;
  submitMarket(qty: number): Promise<LadderSubmit>;
  /** Request a cancel. May throw (Alpaca's 204, an order that already filled): the ladder re-reads the order either way. */
  cancel(orderId: string): Promise<unknown>;
  getOrder(orderId: string): Promise<LadderBrokerOrder>;
  getQuote(): Promise<LadderQuote | null>;
  /** The live size of the position being closed (0 = no longer held); null or a throw = unknown. Optional. */
  getPositionQty?(): Promise<number | null>;
  now(): number;
  sleep(ms: number): Promise<void>;
  /** Every submission, rejection, skipped step and settled order, with a one-line description. */
  onEvent?(event: LadderEvent): void;
}

export interface LadderParams {
  side: LadderSide;
  qty: number;
  /** Price to ladder from when no quote is usable (POS-KILL: the position's mark). */
  fallbackPrice?: number;
}

export type LadderStatus = "filled" | "market_fallback" | "error";

export interface LadderResult {
  status: LadderStatus;
  orders: LadderOrder[];
  filledQty: number;
  /** Volume-weighted fill price over every order; null when nothing filled or a fill had no price. */
  avgPrice: number | null;
  /** Limit steps attempted (an order accepted or rejected). */
  steps: number;
  /** Why the ladder ended. */
  reason: string;
  /** One audit line describing the outcome. */
  summary: string;
}

const TERMINAL = new Set(["filled", "canceled", "expired", "rejected", "done_for_day", "replaced", "stopped", "suspended"]);
const isTerminal = (status: string): boolean => TERMINAL.has(status);
const errText = (err: unknown): string => (err instanceof Error ? err.message : String(err)).slice(0, 160);
const usd = (n: number): string => `$${n.toFixed(2)}`;

function refText(ref: LadderRef | null): string {
  if (!ref) return "no price reference";
  if (ref.source === "mark") return `position mark ${usd(ref.bid)}, no quote`;
  const q = `bid ${ref.bid.toFixed(2)} / ask ${ref.ask.toFixed(2)}`;
  return ref.source === "quote" ? q : `last quote ${q}, no fresh quote`;
}

/**
 * Run the ladder to its end. Never rejects: every failure comes back as
 * status "error" with the orders sent so far, so the caller can tell what may
 * still be working at the broker.
 */
export async function runLadder(deps: LadderDeps, params: LadderParams,
                                cfg: LadderConfig = LADDER_DEFAULTS): Promise<LadderResult> {
  const side = params.side;
  const qty = Math.floor(params.qty);
  const orders: LadderOrder[] = [];
  let steps = 0;
  let lastQuote: LadderQuote | null = null;
  let lastReadError = "";
  const t0 = deps.now();
  const stepMs = cfg.stepSeconds * 1000;
  const pollMs = cfg.pollSeconds * 1000;
  const hardDeadline = t0 + cfg.hardTimeoutSeconds * 1000;
  const maxSteps = Math.max(1, Math.ceil(cfg.hardTimeoutSeconds / cfg.stepSeconds));
  const maxPolls = Math.ceil(stepMs / pollMs) + 2; // bounds every wait even if the clock stalls

  const emit = (event: LadderEvent): void => {
    if (!deps.onEvent) return;
    try {
      deps.onEvent(event);
    } catch (err) {
      // An observer failure (an audit write) must never strand a live order.
      console.error(`[limit-ladder] event observer failed: ${errText(err)}`);
    }
  };
  const filled = (): number => orders.reduce((s, o) => s + o.filledQty, 0);
  const remaining = (): number => Math.max(0, qty - filled());

  const finish = (status: LadderStatus, reason: string): LadderResult => {
    const filledQty = filled();
    const fills = orders.filter((o) => o.filledQty > 0);
    const priced = fills.every((o) => o.avgPrice !== null);
    const avgPrice = filledQty > 0 && priced
      ? fills.reduce((s, o) => s + o.filledQty * (o.avgPrice ?? 0), 0) / filledQty
      : null;
    const fillText = `${filledQty}/${qty} filled${avgPrice !== null ? ` at avg ${usd(avgPrice)}` : ""}`;
    const summary = status === "filled"
      ? `${fillText} by limit after ${steps} step(s); no market order`
      : status === "market_fallback"
        ? `${reason}: market order for the remaining ${orders[orders.length - 1]?.qty ?? 0} after ${steps} limit step(s); ${fillText}`
        : `${reason}; ${fillText} after ${steps} limit step(s)`;
    return { status, orders, filledQty, avgPrice, steps, reason, summary };
  };

  const apply = (order: LadderOrder, b: LadderBrokerOrder | null | undefined): void => {
    if (!b) return;
    if (b.status !== undefined && b.status !== null) order.status = String(b.status);
    const fq = Number(b.filled_qty);
    // Cumulative per order: never let a stale read lower it.
    if (Number.isFinite(fq) && fq > order.filledQty) order.filledQty = Math.min(order.qty, fq);
    // A filled order filled its whole qty — counting it all can only make the next order smaller.
    if (order.status === "filled") order.filledQty = order.qty;
    const ap = Number(b.filled_avg_price);
    if (Number.isFinite(ap) && ap > 0) order.avgPrice = ap;
  };
  const read = async (order: LadderOrder): Promise<void> => {
    try {
      apply(order, await deps.getOrder(order.id));
    } catch (err) {
      lastReadError = errText(err); // one failed read is retried on the next poll
    }
  };
  const sleepUntil = async (t: number): Promise<void> => {
    const wait = t - deps.now();
    if (wait > 0) await deps.sleep(wait);
  };
  const submit = async (fn: () => Promise<LadderSubmit>): Promise<LadderSubmit> => {
    try {
      return await fn();
    } catch (err) {
      return { ok: false, rejected: false, error: errText(err) }; // a throw is an unknown outcome, never a rejection
    }
  };
  const openOrder = (kind: "limit" | "market", step: number, price: number | null, q: number, b: LadderBrokerOrder): LadderOrder | null => {
    const id = b && b.id !== undefined && b.id !== null ? String(b.id) : "";
    if (!id) return null;
    const order: LadderOrder = {
      id, kind, step, price, qty: q, submittedAt: deps.now(), status: "new", filledQty: 0, avgPrice: null, canceledByLadder: false,
    };
    orders.push(order);
    apply(order, b);
    return order;
  };
  /** Poll until the order is terminal or the deadline passes (at least one read when time remains). */
  const pollUntil = async (order: LadderOrder, deadline: number): Promise<void> => {
    for (let polls = 0; polls < maxPolls && !isTerminal(order.status); polls++) {
      const now = deps.now();
      if (now >= deadline) return;
      await deps.sleep(Math.min(pollMs, deadline - now));
      await read(order);
    }
  };
  /** Cancel, then read until terminal. False when the order cannot be shown terminal. */
  const cancelAndSettle = async (order: LadderOrder): Promise<{ ok: boolean; cancelError: string }> => {
    let cancelError = "";
    try {
      await deps.cancel(order.id);
    } catch (err) {
      cancelError = errText(err); // a 204 parse error or "not cancelable" (just filled): the read decides
    }
    for (let i = 0; i < BACKSTOP_LADDER_CANCEL_CONFIRM_READS; i++) {
      await read(order);
      if (isTerminal(order.status)) {
        if (order.status !== "filled") order.canceledByLadder = true;
        return { ok: true, cancelError };
      }
      if (i < BACKSTOP_LADDER_CANCEL_CONFIRM_READS - 1) await deps.sleep(BACKSTOP_LADDER_CANCEL_CONFIRM_INTERVAL_MS);
    }
    return { ok: false, cancelError };
  };
  /** remaining(), capped at the live position when it can be read. 0 with fills still owed = closed elsewhere. */
  const heldRemaining = async (): Promise<number> => {
    const want = remaining();
    if (!deps.getPositionQty) return want;
    let held: number | null = null;
    try {
      held = await deps.getPositionQty();
    } catch (err) {
      lastReadError = errText(err); // unknown: the ladder keeps its own count
    }
    return held === null || !Number.isFinite(held) ? want : Math.min(want, Math.max(0, Math.floor(held)));
  };
  const settledLine = (o: LadderOrder): string =>
    `step ${o.step}: order ${o.id} ${o.status}, ${o.filledQty}/${o.qty} filled${o.avgPrice !== null ? ` at ${usd(o.avgPrice)}` : ""}`;
  /** The price reference for a step: a fresh usable quote, else the last one, else the fallback price. */
  const reference = async (): Promise<LadderRef | null> => {
    let q: LadderQuote | null = null;
    try {
      q = await deps.getQuote();
    } catch (err) {
      lastReadError = errText(err); // a failed quote read falls back below
    }
    const sideOk = (x: LadderQuote | null): x is LadderQuote =>
      !!x && Number.isFinite(side === "sell" ? x.bid : x.ask) && (side === "sell" ? x.bid : x.ask) > 0;
    if (sideOk(q)) {
      lastQuote = q;
      return { source: "quote", bid: q.bid, ask: q.ask };
    }
    if (lastQuote) return { source: "last_quote", bid: lastQuote.bid, ask: lastQuote.ask };
    const fb = Number(params.fallbackPrice);
    if (Number.isFinite(fb) && fb > 0) return { source: "mark", bid: fb, ask: fb };
    return null;
  };

  if (!(qty > 0)) return finish("error", `nothing to close (qty ${params.qty})`);

  try {
    let step = 0;
    while (step < maxSteps) {
      const now = deps.now();
      if (now >= hardDeadline) break;
      // The concession follows the clock: after a stalled event loop the
      // ladder skips steps instead of firing them back to back.
      step = Math.max(step, Math.floor((now - t0) / stepMs));
      if (step >= maxSteps) break;
      const stepDeadline = Math.min(t0 + (step + 1) * stepMs, hardDeadline);
      const ref = await reference();
      const price = ref ? ladderPrice(side, ref.bid, ref.ask, step, cfg) : null;
      if (price === null) {
        emit({ type: "skipped", step, line: `step ${step}: no quote and no mark to price a limit — nothing sent this step` });
        await sleepUntil(stepDeadline);
        step += 1;
        continue;
      }
      // The first order follows the caller's own position read; every later
      // one re-checks the live position.
      const want = steps === 0 ? remaining() : await heldRemaining();
      if (want === 0) {
        return finish("error", `step ${step}: the position is no longer held — closed outside the ladder; nothing more sent`);
      }
      steps += 1;
      const sub = await submit(() => deps.submitLimit(want, price));
      if (!sub.ok) {
        if (!sub.rejected) {
          return finish("error", `step ${step}: limit ${side} ${want} @ ${usd(price)} failed with an unknown outcome (${sub.error}) — stopped rather than risk a second live order`);
        }
        emit({ type: "rejected", step, price, qty: want, error: sub.error,
          line: `step ${step}: limit ${side} ${want} @ ${usd(price)} rejected (${sub.error}) — next step at its time` });
        await sleepUntil(stepDeadline);
        step += 1;
        continue;
      }
      const order = openOrder("limit", step, price, want, sub.order);
      if (!order) return finish("error", `step ${step}: the broker accepted a limit order but returned no id — it cannot be tracked or cancelled`);
      emit({ type: "submitted", step, order, ref,
        line: `step ${step}: limit ${side} ${want} @ ${usd(price)} (${refText(ref)}) — order ${order.id}` });
      await pollUntil(order, stepDeadline);
      if (!isTerminal(order.status)) {
        const settled = await cancelAndSettle(order);
        if (!settled.ok) {
          return finish("error", `step ${step}: order ${order.id} still ${order.status} after its cancel${settled.cancelError ? ` (${settled.cancelError})` : ""}${lastReadError ? `; last read error: ${lastReadError}` : ""} — it may still be working`);
        }
      }
      emit({ type: "settled", step, order, line: settledLine(order) });
      if (remaining() === 0) return finish("filled", "filled");
      if (order.status === "rejected") {
        await sleepUntil(stepDeadline); // an asynchronous rejection is treated like a submit rejection
      } else if (!order.canceledByLadder) {
        return finish("error", `step ${step}: order ${order.id} ended ${order.status} outside the ladder — not resubmitting`);
      }
      step += 1;
    }

    // ── Hard timeout: ONE market order for the remainder ───────────────────
    if (remaining() === 0) return finish("filled", "filled");
    await sleepUntil(hardDeadline); // never before the timeout, whatever ended the loop
    const rest = await heldRemaining();
    if (rest === 0) {
      return finish("error", "hard timeout: the position is no longer held — closed outside the ladder; no market order sent");
    }
    const sub = await submit(() => deps.submitMarket(rest));
    if (!sub.ok) return finish("error", `hard timeout: market ${side} ${rest} failed (${sub.error})`);
    const market = openOrder("market", step, null, rest, sub.order);
    if (!market) return finish("error", "hard timeout: the broker accepted the market order but returned no id");
    emit({ type: "submitted", step, order: market, ref: null,
      line: `hard timeout ${cfg.hardTimeoutSeconds}s: market ${side} ${rest} — order ${market.id}` });
    await pollUntil(market, deps.now() + stepMs);
    if (isTerminal(market.status)) emit({ type: "settled", step, order: market, line: settledLine(market) });
    if (isTerminal(market.status) && market.status !== "filled") {
      return finish("error", `hard timeout: market order ${market.id} ended ${market.status}`);
    }
    return finish("market_fallback", `no full fill in ${cfg.hardTimeoutSeconds}s`);
  } catch (err) {
    return finish("error", `ladder failed: ${errText(err)}`);
  }
}
