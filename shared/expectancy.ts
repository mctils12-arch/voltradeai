// shared/expectancy.ts — expectancy per CLOSED LOT split by sleeve ([PRODUCT]
// 2026-11-03, human-approved item B2: "Split 'Expectancy / closed lot ·
// broker' by sleeve: options, ETF floor, stocks, VRP leg. For each sleeve
// show win rate, average win, average loss, payoff ratio and breakeven win
// rate"). Pure and dependency-free so the server computes it once per sleeve-
// cache refresh (server/edgeLedger.ts folds it into the broker block of
// /api/bot/performance) and the client only renders.
//
// THE RULER IS UNCHANGED. A lot's return is pnl / basis, basis = closing qty ×
// average cost × multiplier > 0 (a credit lot's basis is the credit
// received); a flat close counts as a loss of 0; a lot closed against a basis
// the history does not contain is `unpriced`, never guessed — exactly
// server/edgeLedger.ts, whose `active` row (options + stocks) is the
// Expectancy tile's headline. `active` here is pinned EQUAL to it on identical
// inputs (server/expectancyBySleeve.test.ts); the four sleeves are a
// partition of the same lots, so their n sum to all.n and the n-weighted mean
// of their expectancies is all.expectancyPct.
//
// CLASSIFICATION — one rule, the sleeve ledger's, with the VRP leg carved out
// of the passive set:
//   options : symbol longer than 8 characters (server/reconstructPnl.ts
//             isOptionPosition — the same test sleeveLedger.sleeveOf applies)
//   VRP leg : VRP_LEG_TICKERS (bot_engine.py LEG3_VRP_TICKER — SVXY, the third
//             leg's VRP-harvest instrument; sits inside bot.ts's
//             FLOOR_AND_LEG_TICKERS, so the ETF floor row here = that set
//             minus the VRP leg, and floor + vrp = the sleeve ledger's floor)
//   floor   : the floor ticker set the caller passes (FLOOR_AND_LEG_TICKERS)
//   stocks  : everything else
// An empty symbol cannot be classified: it is counted in `unclassified` and
// in `all`, and sits in no sleeve — reported, never guessed.
//
// FORMULAS (percent, 0–100 for the two rates):
//   winRate          = wins / n × 100                      (wins: pnl > 0)
//   avgWinPct        = mean(pnl / basis) over wins × 100
//   avgLossPct       = mean(pnl / basis) over losses × 100  (≤ 0, signed)
//   payoffRatio      = avgWinPct / |avgLossPct|
//   breakevenWinRate = |avgLossPct| / (avgWinPct + |avgLossPct|) × 100 — the
//                      win rate at which winRate × avgWin + lossRate × avgLoss
//                      is exactly zero
//   expectancyPct    = mean(pnl / basis) over every lot × 100 — the headline
// Every ratio is null (with the reason visible in the UI) when its inputs do
// not exist: no lots, no winning lot, no losing lot, or losses averaging 0.

export type SleeveId = "options" | "floor" | "stocks" | "vrp";
export const SLEEVE_IDS: readonly SleeveId[] = ["options", "floor", "stocks", "vrp"];
/** the four sleeves plus the two totals: `active` = options + stocks (the
 *  Expectancy tile's headline), `all` = every priced lot, unclassified included */
export type LotGroup = SleeveId | "active" | "all";
export const LOT_GROUPS: readonly LotGroup[] = ["options", "floor", "stocks", "vrp", "active", "all"];

/** bot_engine.py LEG3_VRP_TICKER. MUST mirror it; the Python side owns the value. */
export const VRP_LEG_TICKERS: ReadonlySet<string> = new Set(["SVXY"]);

/** One closed lot of the sleeve ledger's walk (a structural subset of
 *  server/sleeveLedger.ts RealizedEvent). `qty` only feeds the unpriced count. */
export interface SleeveLot { symbol: string; t: number; pnl: number; basis: number; qty?: number }

export interface SleeveStats {
  /** priced closed lots */
  n: number;
  wins: number;
  losses: number;
  /** percent of lots with pnl > 0 (0–100); null without lots */
  winRate: number | null;
  /** mean pnl / basis of the winning lots, percent (> 0); null without a winning lot */
  avgWinPct: number | null;
  /** mean pnl / basis of the losing lots, percent (≤ 0, signed); null without a losing lot */
  avgLossPct: number | null;
  /** avgWinPct ÷ |avgLossPct|; null without both, or when the losses average 0 */
  payoffRatio: number | null;
  /** |avgLossPct| ÷ (avgWinPct + |avgLossPct|) × 100; null without both a winning and a losing lot */
  breakevenWinRate: number | null;
  /** mean pnl / basis over every lot, percent — the headline */
  expectancyPct: number | null;
  /** sum of pnl, dollars */
  realizedUsd: number;
  /** ISO time of the newest close; null without lots */
  lastCloseAt: string | null;
  /** closed quantity against a basis the history does not contain (shares / contracts), excluded above */
  unpriced: number;
}

export interface SleeveExpectancy {
  options: SleeveStats;
  floor: SleeveStats;
  stocks: SleeveStats;
  vrp: SleeveStats;
  active: SleeveStats;
  all: SleeveStats;
  /** priced lots whose symbol could not be classified (empty): in `all`, in no sleeve */
  unclassified: number;
}

/** The one classification rule (header comment). null = cannot be classified. */
export function lotSleeve(symbol: string, floorTickers: ReadonlySet<string>, vrpTickers: ReadonlySet<string> = VRP_LEG_TICKERS): SleeveId | null {
  if (typeof symbol !== "string" || symbol.length === 0) return null;
  if (symbol.length > 8) return "options";
  const u = symbol.toUpperCase();
  if (vrpTickers.has(u)) return "vrp";
  if (floorTickers.has(u)) return "floor";
  return "stocks";
}

const r2 = (x: number) => Math.round(x * 100) / 100;
const r2n = (x: number | null) => (x === null ? null : r2(x));
const mean = (xs: readonly number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null);

export function sleeveStats(lots: readonly SleeveLot[], unpricedQty = 0): SleeveStats {
  // a basis of 0 cannot price a return (kept honest, not divided) — the edge ledger's own rule
  const priced = lots.filter((e) => e.basis > 0);
  const unpriced = unpricedQty + lots.filter((e) => !(e.basis > 0)).reduce((s, e) => s + (Number.isFinite(e.qty) ? (e.qty as number) : 1), 0);
  const n = priced.length;
  const ret = (e: SleeveLot) => (e.pnl / e.basis) * 100;
  const wins = priced.filter((e) => e.pnl > 0);
  const losses = priced.filter((e) => e.pnl <= 0);
  const avgWin = mean(wins.map(ret));
  const avgLoss = mean(losses.map(ret));
  const absLoss = avgLoss === null ? null : Math.abs(avgLoss);
  const payoff = avgWin !== null && absLoss !== null && absLoss > 0 ? avgWin / absLoss : null;
  const breakeven = avgWin !== null && absLoss !== null && avgWin + absLoss > 0 ? (absLoss / (avgWin + absLoss)) * 100 : null;
  let last = -Infinity;
  for (const e of priced) if (e.t > last) last = e.t;
  return {
    n,
    wins: wins.length,
    losses: losses.length,
    winRate: n ? r2((wins.length / n) * 100) : null,
    avgWinPct: r2n(avgWin),
    avgLossPct: r2n(avgLoss),
    payoffRatio: r2n(payoff),
    breakevenWinRate: r2n(breakeven),
    expectancyPct: r2n(mean(priced.map(ret))),
    realizedUsd: r2(priced.reduce((s, e) => s + e.pnl, 0)),
    lastCloseAt: n && Number.isFinite(last) ? new Date(last).toISOString() : null,
    unpriced,
  };
}

/** Pure: the closed lots (any order) → the scorecard per sleeve and for the
 *  two totals. `windowStartMs` counts only lots closed from there on (the
 *  same window the edge ledger and /api/bot/sleeves count); `unpriced`
 *  (symbol → closed quantity with no basis in the history) is not windowed,
 *  as the sleeve ledger reports it. */
export function expectancyBySleeve(
  lots: readonly SleeveLot[],
  floorTickers: ReadonlySet<string>,
  vrpTickers: ReadonlySet<string> = VRP_LEG_TICKERS,
  windowStartMs?: number,
  unpriced?: ReadonlyMap<string, number>,
): SleeveExpectancy {
  const inWindow = windowStartMs === undefined ? lots : lots.filter((e) => e.t >= windowStartMs);
  const buckets: Record<SleeveId, SleeveLot[]> = { options: [], floor: [], stocks: [], vrp: [] };
  const unp: Record<SleeveId, number> = { options: 0, floor: 0, stocks: 0, vrp: 0 };
  let unclassified = 0;
  let unclassifiedUnpriced = 0;
  for (const e of inWindow) {
    const s = lotSleeve(e.symbol, floorTickers, vrpTickers);
    if (s === null) { if (e.basis > 0) unclassified++; continue; }
    buckets[s].push(e);
  }
  if (unpriced) {
    for (const [symbol, q] of unpriced) {
      const s = lotSleeve(symbol, floorTickers, vrpTickers);
      if (s === null) unclassifiedUnpriced += q; else unp[s] += q;
    }
  }
  const options = sleeveStats(buckets.options, unp.options);
  const floor = sleeveStats(buckets.floor, unp.floor);
  const stocks = sleeveStats(buckets.stocks, unp.stocks);
  const vrp = sleeveStats(buckets.vrp, unp.vrp);
  const active = sleeveStats([...buckets.options, ...buckets.stocks], unp.options + unp.stocks);
  const all = sleeveStats(inWindow, unp.options + unp.floor + unp.stocks + unp.vrp + unclassifiedUnpriced);
  return { options, floor, stocks, vrp, active, all, unclassified };
}
