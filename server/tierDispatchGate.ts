/**
 * tierDispatchGate.ts — decides whether the Tier-2 scan's TIER DISPATCHER
 * (server/bot.ts, inside tier2Intelligence) may submit a given tier action
 * NOW. Pure (node:test safe — no fs, no network, no sqlite).
 *
 * WHY THIS MODULE EXISTS (off-hours SELL_CSP, found 2026-10-13):
 * the Tier-2 scan window is 04:00-20:00 ET (and getETHour() is
 * weekday-blind), so tier2Intelligence runs pre-market and after-hours with
 * isMarketOpen=false (Alpaca /v2/clock is_open). executeTrades() already
 * routes off-hours stock entries to morningQueue, but the tier dispatcher
 * had NO market-open check: it submitted opening SELL_CSP orders (and the
 * Tier-3 BUY / Tier-4 BUY_PUT orders) whenever the scan ran. Equity options
 * only trade in regular hours, so those limit orders rested on the book at
 * stale off-hours prices until either a deploy's cancel-all swept them or
 * they filled at the open into an immediate POS-KILL loss.
 *
 * THE CONTRACT: when the market is closed, every order-placing tier action
 * is DEFERRED — not queued. The next in-hours Tier-2 scan recomputes
 * tier_actions against live quotes, slots and buying power, which is the
 * only state a dispatch should ever be priced against. Unknown action kinds
 * are passed through unchanged (the dispatcher's if/else chain already
 * ignores them, so gating them would change nothing).
 */

/** Tier actions whose dispatch branch submits an order to the broker. */
export const TIER_ORDER_ACTIONS: ReadonlySet<string> = new Set([
  "SELL_CSP", // Tier 1/2 — opening short put (options: regular hours only)
  "BUY",      // Tier 3 — leveraged SPY/QQQ stock/ETF buy
  "BUY_PUT",  // Tier 4 — tail-hedge long put (options: regular hours only)
]);

export interface TierDispatchContext {
  isMarketOpen: boolean;
}

/**
 * true  -> the dispatcher may run this action's branch now.
 * false -> defer: market is closed and this action would place an order.
 */
export function shouldDispatchTierAction(
  action: { action?: string } | null | undefined,
  ctx: TierDispatchContext,
): boolean {
  if (ctx.isMarketOpen === true) return true;
  const kind = action?.action;
  if (typeof kind !== "string") return true;
  return !TIER_ORDER_ACTIONS.has(kind);
}

/** The one audit line per scan summarising off-hours deferrals. */
export function tierDeferAuditMessage(deferred: number): string {
  return `${deferred} tier action${deferred === 1 ? "" : "s"} deferred — market closed (next in-hours scan recomputes against live quotes)`;
}
