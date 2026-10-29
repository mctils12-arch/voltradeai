// edgeLedger.ts — expectancy per CLOSED LOT from the broker's own fills
// ([RULE-REVIEW] 2026-10-29). The bot page's "Expectancy / trade" tile read
// −1.20 % for weeks: /api/bot/performance averages trade_feedback's pnl_pct,
// which only 36 of the bot's closed trades carry (351 exits are orphan_exit
// with pnl_pct None, 85 of them options; the last non-floor stock exit at the
// broker was 2026-10-14, and with entry shadow mode on — v1.0.1210 — that
// number can never move again on its own). Alpaca's fill ledger is complete,
// so this computes the same scorecard from sleeveLedger.realizedEvents — the
// average-cost walk the kill switch's reconciliation and /api/bot/sleeves
// already use — one row per sleeve plus one for the sleeves the bot actually
// trades ("active" = options + stocks; the floor is rebalancing, not a trade
// decision). Pure; bot.ts serves it inside /api/bot/performance from the one
// sleeve cache, never from a second fetch.
//
// SIGN CONVENTION. pnl is signed from the position's side (sleeveLedger). A
// lot's return is pnl / basis, basis = closing qty × avg cost × mult > 0: for
// a long lot the cost paid; for a SHORT (credit) lot the credit received — a
// sold put bought back cheaper is a WIN with basis = credit × 100 × qty and
// a return equal to the share of the credit kept; bought back dearer its
// return can sit below −100 % (a credit lot can lose more than its basis).
// A flat close (pnl 0) counts as a loss of 0, as the bot log's own filter
// does, so winRate × avgWin + lossRate × avgLoss equals the mean return.
// Lots closed against a basis the fetched history does not contain are
// counted in `unpriced` (quantity) and excluded — never guessed.
import type { RealizedEvent, Sleeve } from "./sleeveLedger";
import { sleeveOf } from "./sleeveLedger";

export type EdgeGroup = "active" | Sleeve;
export const EDGE_GROUPS: readonly EdgeGroup[] = ["active", "options", "stocks", "floor"];

export interface EdgeRow {
  /** priced closed lots in the window */
  n: number;
  wins: number;
  losses: number;
  /** percent of lots with pnl > 0 (0–100); null without lots */
  winRate: number | null;
  /** mean pnl / basis of the winning lots, percent (> 0); null without wins */
  avgWinPct: number | null;
  /** mean pnl / basis of the losing lots, percent (≤ 0, signed); null without losses */
  avgLossPct: number | null;
  /** mean pnl / basis over every lot, percent — the headline */
  expectancyPct: number | null;
  /** mean pnl per lot, dollars */
  expectancyUsd: number | null;
  /** sum of pnl, dollars */
  realizedUsd: number;
  /** gross profit ÷ gross loss in dollars; null without a losing lot */
  profitFactor: number | null;
  firstCloseAt: string | null;
  lastCloseAt: string | null;
  /** closed quantity against a basis the history does not contain (shares / contracts), excluded above */
  unpriced: number;
}
export interface EdgeLedger {
  /** the window the lots are counted from (ISO); null = the whole history */
  windowStart: string | null;
  active: EdgeRow;
  options: EdgeRow;
  stocks: EdgeRow;
  floor: EdgeRow;
}

const r2 = (x: number) => Math.round(x * 100) / 100;
const r2n = (x: number | null) => (x === null ? null : r2(x));
const mean = (xs: readonly number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null);

function edgeRow(lots: readonly RealizedEvent[], unpricedQty: number): EdgeRow {
  // a basis of 0 cannot price a return (a zero fill price never happens at the broker; kept honest, not divided)
  const priced = lots.filter((e) => e.basis > 0);
  const unpriced = unpricedQty + lots.filter((e) => !(e.basis > 0)).reduce((s, e) => s + e.qty, 0);
  const n = priced.length;
  const wins = priced.filter((e) => e.pnl > 0);
  const losses = priced.filter((e) => e.pnl <= 0);
  const ret = (e: RealizedEvent) => (e.pnl / e.basis) * 100;
  const grossProfit = wins.reduce((s, e) => s + e.pnl, 0);
  const grossLoss = -losses.reduce((s, e) => s + e.pnl, 0);
  let first = Infinity;
  let last = -Infinity;
  for (const e of priced) { if (e.t < first) first = e.t; if (e.t > last) last = e.t; }
  return {
    n,
    wins: wins.length,
    losses: losses.length,
    winRate: n ? r2((wins.length / n) * 100) : null,
    avgWinPct: r2n(mean(wins.map(ret))),
    avgLossPct: r2n(mean(losses.map(ret))),
    expectancyPct: r2n(mean(priced.map(ret))),
    expectancyUsd: r2n(mean(priced.map((e) => e.pnl))),
    realizedUsd: r2(priced.reduce((s, e) => s + e.pnl, 0)),
    profitFactor: grossLoss > 0 ? r2(grossProfit / grossLoss) : null,
    firstCloseAt: n ? new Date(first).toISOString() : null,
    lastCloseAt: n ? new Date(last).toISOString() : null,
    unpriced,
  };
}

/** Pure: the realized events (any order) + the unpriced closes from
 *  sleeveLedger.realizedEvents → the scorecard per sleeve and for the active
 *  sleeves. `windowStartMs` (optional) counts only lots closed from there on,
 *  the same window /api/bot/sleeves counts; `unpriced` is not windowed (those
 *  closes are the history's first lots, as sleeveLedger reports them). */
export function edgeLedger(events: readonly RealizedEvent[], unpriced: ReadonlyMap<string, number>, floorTickers: ReadonlySet<string>, windowStartMs?: number): EdgeLedger {
  const inWindow = windowStartMs === undefined ? events : events.filter((e) => e.t >= windowStartMs);
  const lots = new Map<Sleeve, RealizedEvent[]>([["options", []], ["floor", []], ["stocks", []]]);
  for (const e of inWindow) lots.get(sleeveOf(e.symbol, floorTickers))!.push(e);
  const unp = new Map<Sleeve, number>([["options", 0], ["floor", 0], ["stocks", 0]]);
  for (const [symbol, q] of unpriced) { const s = sleeveOf(symbol, floorTickers); unp.set(s, (unp.get(s) ?? 0) + q); }
  return {
    windowStart: windowStartMs === undefined ? null : new Date(windowStartMs).toISOString(),
    active: edgeRow([...lots.get("options")!, ...lots.get("stocks")!], unp.get("options")! + unp.get("stocks")!),
    options: edgeRow(lots.get("options")!, unp.get("options")!),
    stocks: edgeRow(lots.get("stocks")!, unp.get("stocks")!),
    floor: edgeRow(lots.get("floor")!, unp.get("floor")!),
  };
}

export const BROKER_SOURCE = "alpaca fills via sleeve ledger";
export interface BrokerBlock extends EdgeLedger {
  /** when the sleeve cache this was computed from was fetched (ISO) */
  asOf: string;
  source: typeof BROKER_SOURCE;
  /** seconds since that fetch */
  cached_s: number;
  /** older than the cache's refresh interval (a refresh is running or failed; lastError says) */
  stale: boolean;
  lastError: string | null;
}
/** The `broker` half of /api/bot/performance: the cached ledger with its age,
 *  or null and the reason while the sleeve cache is cold (the route never
 *  waits for it — the next poll fills it) or has only ever failed. */
export function brokerBlock(cache: { at: number; edge: EdgeLedger } | null, nowMs: number, lastError: string | null, cacheMs: number): { broker: BrokerBlock | null; brokerReason: string | null } {
  if (!cache) return { broker: null, brokerReason: lastError ? `sleeve ledger failed: ${lastError}` : "sleeve ledger not loaded yet; the next poll fills it" };
  return {
    broker: { ...cache.edge, asOf: new Date(cache.at).toISOString(), source: BROKER_SOURCE, cached_s: Math.max(0, Math.round((nowMs - cache.at) / 1000)), stale: nowMs - cache.at > cacheMs, lastError },
    brokerReason: null,
  };
}
