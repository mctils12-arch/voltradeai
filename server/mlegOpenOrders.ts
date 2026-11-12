// RESTING MULTI-LEG ORDERS (2026-11-06): what an open mleg order commits to.
//
// An Alpaca multi-leg order (order_class "mleg") has an EMPTY top-level side
// and symbol (live /api/diag/orders); its legs only come back when the orders
// are read with nested=true. Every check that looked at side/symbol therefore
// missed a resting condor: the options slot count, the same-underlying
// duplicate check, the cash reserve's working-order total and the tier
// dispatcher's resting-order check. Until the mleg limit-sign fix every credit
// entry filled at once, so nothing rested long enough to matter. Since then a
// credit entry rests whenever the market moved against it, and the mid-first
// debit entries rest by design. These helpers read the legs; the underlying
// comes from each leg's OCC symbol (cspSweepCooldown.occUnderlying, the one
// OCC parse the order checks use).
import { occUnderlying } from "./cspSweepCooldown";

export interface MlegLegLike {
  symbol?: string;
  side?: string;
  position_intent?: string;
  ratio_qty?: string | number;
}

/** The opening legs (position_intent *_to_open) of a multi-leg order; [] for
 *  any other order, and for a closing mleg (a group close: *_to_close legs). */
export function mlegOpeningLegs(order: unknown): MlegLegLike[] {
  if (!order || typeof order !== "object") return [];
  const o = order as { order_class?: unknown; legs?: unknown };
  if (String(o.order_class ?? "").toLowerCase() !== "mleg" || !Array.isArray(o.legs)) return [];
  return (o.legs as unknown[]).filter((l): l is MlegLegLike =>
    !!l && typeof l === "object" && String((l as MlegLegLike).position_intent ?? "").toLowerCase().endsWith("_to_open"));
}

/** The underlyings an open multi-leg OPENING order commits to (sorted, unique). */
export function mlegOpeningUnderlyings(order: unknown): string[] {
  const out = new Set<string>();
  for (const leg of mlegOpeningLegs(order)) {
    const root = occUnderlying(leg.symbol);
    if (root) out.add(root);
  }
  return [...out].sort();
}

/** Every underlying with a multi-leg opening order still open in `orders`. */
export function openMlegUnderlyings(orders: unknown): string[] {
  const out = new Set<string>();
  for (const o of Array.isArray(orders) ? orders : []) for (const u of mlegOpeningUnderlyings(o)) out.add(u);
  return [...out].sort();
}

/** True when some open order is a multi-leg opening order on `underlying`. */
export function hasOpenMlegOpeningOn(orders: unknown, underlying: string): boolean {
  const u = String(underlying || "").toUpperCase();
  return !!u && openMlegUnderlyings(orders).includes(u);
}

/** The order working now: an order the mid-first ladder re-priced is REPLACED
 *  (Alpaca replace: a new order id per step), so follow replaced_by. Stops at
 *  the first order that was not replaced (or after maxHops). */
export async function followReplacements(orderId: string, getOrder: (id: string) => Promise<unknown>,
  maxHops = 12): Promise<string> {
  let id = String(orderId);
  for (let hop = 0; hop < maxHops; hop++) {
    const o = await getOrder(id);
    const rec = o && typeof o === "object" ? (o as { status?: unknown; replaced_by?: unknown }) : {};
    if (String(rec.status ?? "") !== "replaced" || !rec.replaced_by) return id;
    id = String(rec.replaced_by);
  }
  return id;
}
