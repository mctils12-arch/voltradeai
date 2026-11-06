// tradeBackdrop — the one animation state behind every Trade screen.
//
// Human 2026-10-29: "I want this moving map like this but more modern as
// well for some of the screens." The moving world map (DataWorldMap) used
// to be mounted inside the Analyze page only, re-created on every visit and
// absent from Scanner, Research, News, Watchlist, Planner and Taxes. Now the
// app shell mounts it once for the whole Trade world (TradeBackdrop.tsx)
// and the Analyze page drives its state through this tiny store instead of
// owning the canvas: idle particles on every Trade tab, the loading burst
// and the ticker beacon only while Analyze is working on a name.
//
// Same shape as lib/units.ts: a module-level value, a setter, a
// subscription. No React import, so the store is unit-testable.

export interface TradeBackdropState {
  isLoading: boolean;
  hasData: boolean;
  ticker: string | null;
}

export const IDLE_BACKDROP: TradeBackdropState = { isLoading: false, hasData: false, ticker: null };

let current: TradeBackdropState = IDLE_BACKDROP;
const listeners = new Set<() => void>();

export function getTradeBackdrop(): TradeBackdropState {
  return current;
}

/** Merge `next` into the current state; listeners fire only on a real change. */
export function setTradeBackdrop(next: Partial<TradeBackdropState>): void {
  const merged: TradeBackdropState = {
    isLoading: next.isLoading ?? current.isLoading,
    hasData: next.hasData ?? current.hasData,
    ticker: next.ticker === undefined ? current.ticker : next.ticker,
  };
  if (
    merged.isLoading === current.isLoading &&
    merged.hasData === current.hasData &&
    merged.ticker === current.ticker
  ) return;
  current = merged;
  listeners.forEach((fn) => fn());
}

/** Back to idle particles (the Analyze page unmounted, or a tab without a ticker). */
export function resetTradeBackdrop(): void {
  setTradeBackdrop(IDLE_BACKDROP);
}

export function subscribeTradeBackdrop(fn: () => void): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}
