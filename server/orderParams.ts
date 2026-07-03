// ─── ET Hour Helper (DST-aware) ───────────────────────────────────────────────
export function getETHour(): number {
  // Proper ET that handles DST automatically (EST = UTC-5, EDT = UTC-4)
  const now = new Date();
  const et = new Date(now.toLocaleString("en-US", { timeZone: "America/New_York" }));
  return et.getHours() + et.getMinutes() / 60;
}

// ─── Market Hours Helper ──────────────────────────────────────────────────────
export type OrderContext = 'stop_loss' | 'trailing_stop' | 'take_profit' | 'new_entry' | 'options_entry' | 'options_exit';

export interface OrderParams {
  type: string;
  limit_price?: string;
  time_in_force: string;
  extended_hours?: boolean;
}

export function getOrderParams(
  price: number,
  context: OrderContext = 'new_entry',
  etHourOverride?: number
): OrderParams {
  const etTime = etHourOverride ?? getETHour();
  const isRegularHours = etTime >= 9.5 && etTime < 16.0;

  // Options: ALWAYS limit, no exceptions (wide bid-ask spreads). Options have
  // no extended-hours session on Alpaca (or anywhere) — extended_hours never applies.
  if (context === 'options_entry' || context === 'options_exit') {
    const limitPrice = Math.round(price * 100) / 100;
    return { type: "limit", limit_price: String(limitPrice), time_in_force: "day" };
  }

  if (isRegularHours) {
    switch (context) {
      case 'stop_loss':
      case 'trailing_stop':
        // Speed matters — get out NOW. A limit that doesn't fill while price drops is catastrophic.
        return { type: "market", time_in_force: "day" };
      case 'take_profit': {
        // Not in a rush — want the exact target price
        const tpPrice = Math.round(price * 100) / 100;
        return { type: "limit", limit_price: String(tpPrice), time_in_force: "day" };
      }
      case 'new_entry':
      default: {
        // Limit at ask + 0.1% — fill priority while capping worst case
        const entryPrice = Math.round(price * 1.001 * 100) / 100;
        return { type: "limit", limit_price: String(entryPrice), time_in_force: "day" };
      }
    }
  } else {
    // Extended hours (4am-9:30am, 4pm-8pm ET): Alpaca requires limit orders AND
    // extended_hours: true — without the flag, a day-limit order submitted here
    // is simply queued for the next regular session and never attempts to fill
    // during the pre-market/after-hours session it was priced for (KNOWN BROKEN #8).
    switch (context) {
      case 'stop_loss':
      case 'trailing_stop': {
        // Bid - 0.5% to ensure fill in thin liquidity
        const stopPrice = Math.round(price * 0.995 * 100) / 100;
        return { type: "limit", limit_price: String(stopPrice), time_in_force: "day", extended_hours: true };
      }
      case 'take_profit': {
        const tpPrice = Math.round(price * 100) / 100;
        return { type: "limit", limit_price: String(tpPrice), time_in_force: "day", extended_hours: true };
      }
      case 'new_entry':
      default: {
        // Ask + 0.5% — wider buffer for thinner extended hours liquidity
        const entryPrice = Math.round(price * 1.005 * 100) / 100;
        return { type: "limit", limit_price: String(entryPrice), time_in_force: "day", extended_hours: true };
      }
    }
  }
}
