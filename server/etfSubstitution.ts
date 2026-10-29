// etfSubstitution.ts — the leveraged-ETF substitution's order inputs.
//
// [REPAIR] 2026-10-27 (trading-engine audit §6.8 N6). executeTrades' ETF
// branch used to buy Math.floor(trade.shares / 2) ETF shares — half the
// UNDERLYING's share count, unrelated to dollars — at
// getOrderParams(trade.price), i.e. a limit derived from the UNDERLYING's
// price: XLF→FAS (≈3x the price) never filled and most others were
// effectively market orders. bot_engine._resolve_etf_execution now sends the
// ETF's own dollar-sized share count and ask (instrument_selector
// .plan_etf_execution); this module only validates them. A trade without
// them (a scan from older code, or a planning failure) is NOT guessed at:
// the caller falls back to the stock, the branch's existing failure path.

export interface EtfTradeFields {
  ticker?: string;
  instrument?: string;
  instrument_ticker?: string;
  etf_shares?: number | null;
  etf_price?: number | null;
  etf_position_value?: number | null;
}

export type EtfOrderPlan =
  | { ok: true; symbol: string; qty: number; price: number; value: number }
  | { ok: false; reason: string };

/** The ETF order to place for an instrument-selector "etf" trade, or why not. */
export function planEtfOrder(trade: EtfTradeFields): EtfOrderPlan {
  const symbol = String(trade.instrument_ticker || "").toUpperCase();
  if (trade.instrument !== "etf" || !symbol || symbol === String(trade.ticker || "").toUpperCase()) {
    return { ok: false, reason: "not an ETF substitution" };
  }
  const qty = Number(trade.etf_shares);
  const price = Number(trade.etf_price);
  if (!Number.isInteger(qty) || qty < 1) {
    return { ok: false, reason: `no ETF share count from the scan (etf_shares=${trade.etf_shares})` };
  }
  if (!Number.isFinite(price) || price <= 0) {
    return { ok: false, reason: `no ETF price from the scan (etf_price=${trade.etf_price})` };
  }
  const value = Number(trade.etf_position_value);
  return {
    ok: true, symbol, qty, price,
    value: Number.isFinite(value) && value > 0 ? value : qty * price,
  };
}

// ── ETF max hold (audit §6.8 N6, [REPAIR] 2026-10-27) ──────────────────────
// instrument_selector.ETF_MAX_HOLD_DAYS ("never hold leveraged ETF longer
// than 5 days" — daily-rebalance decay) gated only the ENTRY choice; no exit
// enforced it (the live time stop is 7 days and only when |P&L| < 2%).
// checkPositionOnTick now exits any held substitute ETF past the limit.
// Mirrors (pinned by test_etf_max_hold_n6.py): ETF_MAX_HOLD_DAYS =
// instrument_selector.ETF_MAX_HOLD_DAYS; SUBSTITUTE_LEVERAGED_ETFS = every
// bull/bear value of analyze.LEVERAGED_ETFS.
export const ETF_MAX_HOLD_DAYS = 5;
export const SUBSTITUTE_LEVERAGED_ETFS: ReadonlySet<string> = new Set([
  "TSLL", "TSLS", "NVDL", "NVDS", "SSO", "SDS", "QLD", "QID", "AAPU", "AAPD",
  "AMZU", "AMZD", "MSFU", "MSFD", "METU", "METD", "GGLL", "GGLS", "AMDU", "AMDS",
  "MSTX", "MSTS", "TNA", "TZA", "UGL", "GLL", "UBT", "TBT", "ERX", "ERY", "FAS", "FAZ",
]);

/** Calendar days held (the live time stop's floor((now - entryDate) / 1 day))
 *  and whether a substitute ETF is past ETF_MAX_HOLD_DAYS. */
export function etfMaxHoldExceeded(
  ticker: string, entryDate: string, nowMs: number,
): { exceeded: boolean; daysHeld: number } {
  if (!SUBSTITUTE_LEVERAGED_ETFS.has(String(ticker || "").toUpperCase())) {
    return { exceeded: false, daysHeld: 0 };
  }
  const entryMs = new Date(entryDate).getTime();
  if (!Number.isFinite(entryMs)) return { exceeded: false, daysHeld: 0 };
  const daysHeld = Math.floor((nowMs - entryMs) / 86400000);
  return { exceeded: daysHeld > ETF_MAX_HOLD_DAYS, daysHeld };
}
