// MORNING-QUEUE ENTRY DECISION (2026-10-27, audit §6.8 N2) — pure helper for
// bot.ts's executeMorningQueue().
//
// THE BUG: deep_score's "SELL OPTIONS" candidates carry trade_type="options",
// side="sell" and shares > 0. Off-hours, Tier 2 pushes them into morningQueue
// verbatim. executeMorningQueue's stock-short hard block only converted
// sell/short when trade_type !== "options", so an options-typed SELL fell
// through to the STOCK order body with side:"sell" — a naked stock short —
// and was then monitored as "long". The overnight extreme-mover queue has the
// sibling shape: bearish "BUY PUT" / "SELL OPTIONS" setups queued with
// side="sell", use_options=true and no trade_type, which the block flipped
// into a long STOCK buy (the opposite exposure).
//
// THE FIX: the morning queue has no options execution path, so an options-
// intended entry is dropped with an audit line (the next in-hours Tier 2 scan
// re-scores the name and, if it still qualifies, executeTrades routes it
// through the options path it would have taken anyway). A plain stock
// sell/short keeps PR #53's convert-to-BUY safety net, and the position
// monitor side is derived from the side actually submitted — never assumed.
// This decides WHAT is submitted; how the order is transmitted is unchanged.

export interface MorningQueueTradeLike {
  ticker?: string;
  side?: string;
  trade_type?: string;
  use_options?: boolean;
  options_strategy?: string;
  action_label?: string;
}

export type MorningQueueVerdict =
  | { submit: false; auditTag: "MORNING-OPTIONS-SKIP"; reason: string }
  | {
      submit: true;
      side: "buy";
      monitorSide: "long";
      convertedFrom: string | null; // original side when PR #53's block flipped it
    };

const BEARISH_SIDES = new Set(["sell", "short"]);

/** True when the queued entry was meant to be executed as an options trade. */
export function isOptionsIntended(trade: MorningQueueTradeLike): boolean {
  if (trade.trade_type === "options") return true;
  // A bearish entry the producer flagged for options (the extreme-mover
  // queue's BUY PUT / SELL OPTIONS setups) has no stock equivalent: flipping
  // it to a stock BUY takes the opposite exposure.
  const side = String(trade.side || "buy").toLowerCase();
  return trade.use_options === true && BEARISH_SIDES.has(side);
}

/**
 * Decide what (if anything) executeMorningQueue may submit for `trade`.
 * Never returns a sell side: the morning queue only opens long stock.
 */
export function decideMorningQueueEntry(trade: MorningQueueTradeLike): MorningQueueVerdict {
  const side = String(trade.side || "buy").toLowerCase();
  if (isOptionsIntended(trade)) {
    const what = trade.action_label || trade.options_strategy || trade.trade_type || "options";
    return {
      submit: false,
      auditTag: "MORNING-OPTIONS-SKIP",
      reason:
        `${trade.ticker ?? "?"}: options-intended entry (${what}, side='${side}') dropped — ` +
        `the morning queue has no options path and must never submit it as a stock order; ` +
        `the next in-hours scan re-evaluates it`,
    };
  }
  return {
    submit: true,
    side: "buy",
    monitorSide: "long",
    convertedFrom: BEARISH_SIDES.has(side) ? side : null,
  };
}
