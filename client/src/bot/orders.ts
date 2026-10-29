// /api/orders/open (server/routes.ts) answers `{ orders: [...] }` — the
// orphan TradingActivity component reads it that way — but the Bot page typed
// the response as a bare array and passed it through. `orders.length` on the
// wrapper is undefined, so the Orders panel always said "none working" and the
// stage row read "? open" even with working orders (audit §8.2 D4, fixed
// 2026-10-27). Unwrap here; an unexpected shape throws so the query lands in
// its error state ("orders not loaded") instead of a confident wrong "none".
export function unwrapOpenOrders<T>(raw: unknown): T[] {
  if (Array.isArray(raw)) return raw as T[];
  if (raw && typeof raw === "object" && Array.isArray((raw as { orders?: unknown }).orders)) {
    return (raw as { orders: T[] }).orders;
  }
  throw new Error("unexpected /api/orders/open response shape");
}
