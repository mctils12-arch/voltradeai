// ── Entry-fill feedback payloads ────────────────────────────────────────
// REPAIR 2026-07-31 (KNOWN BROKEN #12(c) contributor, found while
// live-verifying #12(b)'s open gate): the ETF entry path (bot.ts's "ETF
// execution" branch — 2x-leveraged-ETF trades the instrument selector
// chose over the underlying stock) calls addPositionToMonitor but never
// track_fill. Both other entry paths (regular-hours, morning-queue) DO
// call track_fill on entry, so their later WS exits can find a matching
// open record. The ETF path's positions get monitored and eventually
// exited via the exact same WS mechanism (recordExitFill, same ticker),
// but with no entry record ever written, ml_model_v2._find_entry_record
// always misses -> every ETF exit becomes a permanent, unrecoverable
// orphan_exit. Confirmed live via /api/diag/ml this session: 70/70
// non-seeded feedback records dated 2026-07-10..30 are orphan_exit, zero
// matched win/loss ever recorded — direct evidence against KNOWN BROKEN
// #12(b)'s working assumption that D2's WS exit path would soon produce
// real outcomes on its own. This payload gives the ETF path parity with
// the two already-working entry sites.
//
// Deliberately NOT extended to the options-entry monitor call in the same
// function (options positions are monitored via the underlying ticker's
// price for stop-loss purposes) — an entry/exit pair recorded there would
// price pnl_pct off the UNDERLYING's move, not the option's actual
// premium P&L, which is exactly the "options fill realism" gap
// open_questions.md #12(c) already gates behind a dedicated design
// decision (quote-based premium fills). Writing a stock-priced entry
// record under an "options" instrument tag would mislabel training data
// with the wrong P&L basis — left open, filed as a follow-up.

export interface EntryFillArgs {
  ticker: string;
  side: string;
  qty: number;
  fillPrice: number;
  session: string;
  volume?: number;
  score?: number;
  instrument: string;
  /** Injected clock for tests. */
  nowMs?: number;
  codeVersion: string;
  /** Decision-time quote. Defaults to fillPrice (zero slippage). Pass 0 when
   *  no comparable quote exists, so track_fill excludes the record from the
   *  slippage stats instead of reporting a fake 0%. */
  expectedPrice?: number;
  qtyRequested?: number;
  /** The prior synthetic haircut (percent), kept as a comparison field only. */
  syntheticSlippagePct?: number;
  /** What the prior code applied the haircut to: the broker fill (regular
   *  path) or the decision-time quote (morning-queue path). */
  syntheticBase?: "fill" | "expected";
  orderStatus?: string;
  orderId?: string;
  entryFeatures?: unknown;
}

// ── REAL BROKER FILLS (2026-10-27, [RULE-REVIEW], audit §7.1 / §7.2.5) ─────
// Entry records used to carry a SYNTHETIC fill_price. The regular-hours path
// added a random volume-tiered haircut on top of Alpaca's filled_avg_price,
// and the morning-queue path never read a fill at all (trade.price times a
// random 0.05-0.25%). ml_model_v2.track_fill stored only that number, so
// slippage_pct, fills_slippage_stats and realisticPnlPct measured the bot's
// own random draw plus quote drift, and ROOT LADDER gate 5 (execution) was
// not measurable live. Now every entry record is written only once Alpaca
// confirms a fill, with fill_price = filled_avg_price and qty = filled_qty.
// The old haircut is still computed (the same formula per path) and stored
// as synthetic_fill_price, so old and new rulers can be compared on the same
// live trades.

const round2 = (x: number) => Math.round(x * 100) / 100;

/** The prior volume-tiered haircut, unchanged (percent). "morning_queue"
 *  selects the wider opening table; anything else the regular-hours table. */
export function syntheticSlippagePct(volume: number | undefined, session: string,
                                     rand: () => number = Math.random): number {
  const vol = volume || 1000000;
  if (session === "morning_queue") {
    if (vol > 20000000) return 0.05 + rand() * 0.05;
    if (vol > 5000000) return 0.08 + rand() * 0.07;
    return 0.15 + rand() * 0.10;
  }
  if (vol > 20000000) return 0.02 + rand() * 0.03;
  if (vol > 5000000) return 0.05 + rand() * 0.05;
  if (vol > 1000000) return 0.08 + rand() * 0.07;
  return 0.12 + rand() * 0.13;
}

/** The fields of an Alpaca order object the fill helpers read. */
export interface AlpacaOrderFill {
  status?: unknown;
  filled_qty?: unknown;
  filled_avg_price?: unknown;
}

/** The broker-confirmed fill on an Alpaca order object, or null when nothing
 *  has filled. A canceled/expired order with a partial fill still returns
 *  its filled part, because those shares are really held. */
export function confirmedEntryFill(order: AlpacaOrderFill | null | undefined): { qty: number; price: number } | null {
  const qty = Number(order?.filled_qty) || 0;
  const price = Number(order?.filled_avg_price) || 0;
  if (qty <= 0 || price <= 0) return null;
  return { qty, price };
}

const TERMINAL_ORDER_STATUSES = new Set([
  "filled", "canceled", "expired", "rejected", "done_for_day", "replaced", "stopped", "suspended",
]);

/** True once an Alpaca order can no longer fill further. */
export function isTerminalOrderStatus(status: unknown): boolean {
  return TERMINAL_ORDER_STATUSES.has(String(status || ""));
}

export function buildEntryFillPayload(a: EntryFillArgs) {
  const nowMs = a.nowMs ?? Date.now();
  const expected = a.expectedPrice ?? a.fillPrice;
  const payload: Record<string, unknown> = {
    ticker: a.ticker,
    side: a.side,
    qty: a.qty,
    expected_price: expected,
    fill_price: a.fillPrice,
    fill_source: "broker",
    time_placed: new Date(nowMs).toISOString(),
    session: a.session,
    volume: a.volume ?? 1000000,
    score: a.score,
    instrument: a.instrument,
    code_version: a.codeVersion,
  };
  if (a.qtyRequested != null) {
    payload.qty_requested = a.qtyRequested;
    payload.partial_fill = a.qty > 0 && a.qty < a.qtyRequested;
  }
  if (a.orderStatus) payload.order_status = a.orderStatus;
  if (a.orderId) payload.order_id = a.orderId;
  if (a.entryFeatures !== undefined) payload.entry_features = a.entryFeatures;
  if (a.syntheticSlippagePct != null) {
    const base = a.syntheticBase === "expected" ? expected : a.fillPrice;
    const dir = a.side === "buy" ? 1 : -1;
    if (base > 0) {
      payload.synthetic_fill_price = round2(base * (1 + dir * a.syntheticSlippagePct / 100));
      payload.synthetic_slippage_applied_pct = a.syntheticSlippagePct;
    }
  }
  return payload;
}
