// manualOrderCheck.ts — what a manual order (POST /api/bot/trade) is worth,
// for the size check that runs before it is sent ([REPAIR] 2026-11-25).
//
// THE DEFECT: the route valued every order at quantity x (account's prior
// equity / 100), a placeholder written as "rough estimate" and never
// replaced. On an $87,600 account that prices every share of every stock at
// about $876: 25 shares of a $10 stock ($250) read as $21,900 and were
// refused as over the 20% position limit, while 15 shares of a $2,000 stock
// ($30,000, 34% of the account) read as $13,140 (15%) and passed. The same
// check also ran on
// SELLS, so reducing a held position by more than about 20 shares was
// refused as "exceeds max position size", and on a day past the daily loss
// limit every manual sell was refused: entry rules applied to an exit. And
// when the account read failed the check was skipped and the order sent.
//
// NOW: a BUY is valued at quantity x the instrument's own last price (x 100
// for an option contract) and the entry checks run on that; with no price or
// no account reading the buy is refused (fail closed: a size check that
// cannot run does not pass). A SELL that reduces a held long is not an entry
// and is not put through the entry checks (the route's own no-shorting rule
// and the kill switch's refusal are unchanged). How the order is built and
// sent is untouched. Pure; pinned by manualOrderCheck.test.ts.

/** OCC option symbol: root, yymmdd, C/P, strike x 1000 in 8 digits. */
export const OCC_SYMBOL_RE = /^[A-Z]{1,6}\d{6}[CP]\d{8}$/;
/** shares per option contract */
export const OPTION_MULTIPLIER = 100;

const num = (v: unknown): number | null => {
  const n = typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
};

/** The order's dollar value at `price`, or null when the quantity or the price is not a positive number. */
export function manualOrderValue(qty: unknown, price: unknown, isOption: boolean): number | null {
  const q = num(qty), p = num(price);
  if (q === null || p === null || q <= 0 || p <= 0) return null;
  return q * p * (isOption ? OPTION_MULTIPLIER : 1);
}

/** A stock's last price from Alpaca's `GET /v2/stocks/{symbol}/trades/latest` body ({trade: {p}}); null when absent. */
export function stockPriceFromLatestTrade(body: unknown): number | null {
  const t = body && typeof body === "object" ? (body as { trade?: unknown }).trade : null;
  const p = t && typeof t === "object" ? num((t as { p?: unknown }).p) : null;
  return p !== null && p > 0 ? p : null;
}

/** An option's price per share from its quote: the mid when both sides are quoted, else the one side that is; null with neither. */
export function optionPriceFromQuote(q: { bid: number; ask: number } | null | undefined): number | null {
  if (!q) return null;
  const bid = q.bid > 0 ? q.bid : 0, ask = q.ask > 0 ? q.ask : 0;
  if (bid > 0 && ask > 0) return (bid + ask) / 2;
  return bid > 0 ? bid : ask > 0 ? ask : null;
}

/** Entry checks (position size, daily loss) apply to a buy. A sell on this route can only reduce a held long. */
export function isManualEntry(side: unknown): boolean {
  return String(side ?? "").trim().toLowerCase() === "buy";
}

/** Why a buy is refused before any check can run, in the owner's words; null when the check can run. */
export function sizeCheckBlocker(symbol: string, portfolioValue: unknown, value: number | null): string | null {
  const pv = num(portfolioValue);
  if (pv === null || pv <= 0) return "The account's value could not be read, so the size check could not run. Order not sent.";
  if (value === null) return `No current price for ${symbol}, so the size check could not run. Order not sent.`;
  return null;
}
