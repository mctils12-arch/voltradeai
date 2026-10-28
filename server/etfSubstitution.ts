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
