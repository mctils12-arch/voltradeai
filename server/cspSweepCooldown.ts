// CSP SWEEP/RESUBMIT LOOP FIX (2026-10-13) — pure helpers for bot.ts's
// sweepStaleOrders() and the tier dispatcher's SELL_CSP branch.
//
// THE LOOP (live 2026-10-12: 51 canceled orders, 0 SWEEP audit lines):
//   1. The SELL_CSP dispatcher submits a non-marketable DAY limit put.
//   2. sweepStaleOrders() (tier1Reflex) cancels it after STALE_ORDER_MINUTES
//      via the broker's order-cancel DELETE. Alpaca answers 204 No
//      Content; the shared alpaca() helper calls r.json() on every 2xx,
//      which throws AFTER the cancel has already succeeded — so the
//      success audit never ran and the cancel was invisible.
//   3. Next scan, the dispatcher re-submits the same limit on the same
//      underlying with no memory of the sweep, and the loop never settles.
//
// alpaca()'s request/transmission path is shared with every order POST and is
// deliberately NOT changed (FROZEN-adjacent); instead the sweep confirms a
// probable empty-body success with a read-only GET of the order, and the
// dispatcher consults a per-underlying sweep memory + the live open-orders
// list before re-submitting.

// Statuses that mean "the cancel request was accepted". pending_cancel is the
// short-lived state Alpaca reports between the DELETE and the final canceled.
const CANCEL_ACCEPTED_STATUSES = new Set(["canceled", "pending_cancel"]);

/**
 * True when `err` looks like the JSON parse failure alpaca() throws on a 2xx
 * response with an empty body (DELETE → 204 No Content), i.e. the request
 * itself succeeded. A real HTTP failure from alpaca() is thrown as
 * `Error("Alpaca <status>: ...")` and a timeout as an AbortError — neither is
 * a probable success.
 */
export function isEmptyBodyParseError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { name?: unknown; message?: unknown };
  const name = String(e.name ?? "");
  const msg = String(e.message ?? "");
  if (/^Alpaca \d{3}\b/.test(msg)) return false;
  if (name === "AbortError") return false;
  return name === "SyntaxError" || /JSON|Unexpected end/i.test(msg);
}

export interface CancelConfirmation {
  confirmed: boolean;
  status: string | null;
}

/**
 * After the order-cancel DELETE threw, decide whether the cancel actually
 * went through. Only an empty-body parse failure is treated as a probable
 * success, and it is CONFIRMED by a read-only status fetch (`fetchOrder`,
 * a read-only order-status GET made by the caller) — never assumed. Any
 * fetch failure → unconfirmed.
 */
export async function confirmCancelAfterDeleteError(
  err: unknown,
  fetchOrder: () => Promise<unknown>,
): Promise<CancelConfirmation> {
  if (!isEmptyBodyParseError(err)) return { confirmed: false, status: null };
  try {
    const order = (await fetchOrder()) as { status?: unknown } | null;
    const status = order && order.status != null ? String(order.status) : null;
    return { confirmed: status !== null && CANCEL_ACCEPTED_STATUSES.has(status), status };
  } catch (fetchErr) {
    console.error("[bot] sweep cancel-confirm GET failed:", (fetchErr as Error)?.message || fetchErr);
    return { confirmed: false, status: null };
  }
}

// OCC option symbol tail: YYMMDD + C/P + 8-digit strike (Alpaca omits the
// space padding, e.g. "XLK260424P00149000" → root "XLK").
const OCC_TAIL = /\d{6}[CP]\d{8}$/;
const OCC_TAIL_LEN = 15;

/** Underlying root of an OCC option symbol, or null if `symbol` is not one. */
export function occUnderlying(symbol: string | null | undefined): string | null {
  const s = String(symbol || "").trim().toUpperCase();
  if (s.length <= OCC_TAIL_LEN || !OCC_TAIL.test(s)) return null;
  return s.slice(0, -OCC_TAIL_LEN);
}

export interface OpenOrderLike {
  symbol?: string;
  side?: string;
  position_intent?: string;
  // Present on closed orders returned by GET /v2/orders?status=closed.
  status?: string;
  submitted_at?: string | null;
  canceled_at?: string | null;
  expired_at?: string | null;
  filled_qty?: string | number | null;
}

/**
 * True when `order` is an open sell-to-open option order on `underlying`.
 * An explicit sell_to_close intent (closing a long option) does not count;
 * a sell with no position_intent is treated as opening, matching
 * countOpenOptionsOpeningOrders()'s convention in bot.ts.
 */
export function isOpenSellToOpenOn(order: OpenOrderLike, underlying: string): boolean {
  if (!order || order.side !== "sell") return false;
  const intent = order.position_intent;
  if (intent && intent !== "sell_to_open") return false;
  const root = occUnderlying(order.symbol);
  return root !== null && root === String(underlying).toUpperCase();
}

export interface CspDispatchContext {
  now: number; // epoch ms
  sweptAt?: number; // epoch ms of the last confirmed sweep of a CSP on this underlying
  openOrders?: OpenOrderLike[];
  cooldownMinutes: number;
  // RESTART-PROOF BRAKE (KB #46): recent CLOSED broker orders. The sweep memory
  // above lives in process memory and a deploy's cancel-all empties the open
  // book, so a fresh container forgets both. The broker remembers.
  closedOrders?: OpenOrderLike[];
  staleMinutes?: number; // an order that rested at least this long "would not fill"
}

export type CspDispatchDecision =
  | { dispatch: true }
  | { dispatch: false; reason: "swept_cooldown" | "open_sell_to_open" | "broker_unfilled_rested"; detail: string };

/**
 * True when `order` is a closed, never-filled sell-to-open on `underlying`
 * that RESTED >= staleMinutes before it was canceled/expired, and that ended
 * within `cooldownMinutes` of `now`. Indifferent to WHO canceled it (sweep,
 * deploy cancel-all, manual) — but a cancel soon after submit proves nothing
 * about fillability, so it does not count.
 */
export function isRecentUnfilledRestedCsp(
  order: OpenOrderLike,
  underlying: string,
  now: number,
  cooldownMinutes: number,
  staleMinutes: number,
): boolean {
  if (!isOpenSellToOpenOn(order, underlying)) return false;
  if (order.status !== "canceled" && order.status !== "expired") return false;
  if (Number(order.filled_qty ?? 0) > 0) return false;
  const start = Date.parse(String(order.submitted_at ?? ""));
  const end = Date.parse(String(order.canceled_at ?? order.expired_at ?? ""));
  if (!Number.isFinite(start) || !Number.isFinite(end)) return false;
  if ((end - start) / 60000 < staleMinutes) return false;
  const sinceEnd = (now - end) / 60000;
  return sinceEnd >= 0 && sinceEnd < cooldownMinutes;
}

/**
 * Decide whether the tier dispatcher may submit a SELL_CSP on `ticker` now.
 * Skips (a) an underlying whose CSP limit was swept within `cooldownMinutes`
 * — re-submitting the same non-marketable limit just restarts the
 * submit/cancel loop — and (b) an underlying that already has a resting
 * sell-to-open option order (a second one would double the commitment).
 */
export function shouldDispatchCsp(ticker: string, ctx: CspDispatchContext): CspDispatchDecision {
  const sym = String(ticker).toUpperCase();
  if (typeof ctx.sweptAt === "number" && Number.isFinite(ctx.sweptAt)) {
    const ageMin = (ctx.now - ctx.sweptAt) / 60000;
    if (ageMin >= 0 && ageMin < ctx.cooldownMinutes) {
      return {
        dispatch: false,
        reason: "swept_cooldown",
        detail: `CSP limit swept ${Math.round(ageMin)} min ago (cooldown ${ctx.cooldownMinutes} min)`,
      };
    }
  }
  const resting = (Array.isArray(ctx.openOrders) ? ctx.openOrders : []).find((o) => isOpenSellToOpenOn(o, sym));
  if (resting) {
    return {
      dispatch: false,
      reason: "open_sell_to_open",
      detail: `open sell-to-open order already resting (${resting.symbol})`,
    };
  }
  if (typeof ctx.staleMinutes === "number") {
    const stale = (Array.isArray(ctx.closedOrders) ? ctx.closedOrders : []).find((o) =>
      isRecentUnfilledRestedCsp(o, sym, ctx.now, ctx.cooldownMinutes, ctx.staleMinutes as number),
    );
    if (stale) {
      return {
        dispatch: false,
        reason: "broker_unfilled_rested",
        detail: `broker shows ${stale.symbol} rested >= ${ctx.staleMinutes} min unfilled, ended ${stale.canceled_at ?? stale.expired_at} (cooldown ${ctx.cooldownMinutes} min)`,
      };
    }
  }
  return { dispatch: true };
}
