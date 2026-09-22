/**
 * reconstructPnl.ts — pure, testable aggregator behind the "reconstruct_pnl"
 * diag probe (server/bot.ts). Ports scripts/reconstruct_position_pnl.py's
 * reconstruct() logic (built 2026-09-10 for KNOWN BROKEN #42 — see
 * research/open_questions.md) into TypeScript, ELIMINATING the Python
 * subprocess entirely rather than adding one: this session's own KNOWN
 * BROKEN #44 finding is that ANY diag probe invoking a scripts/*.py module
 * 500s in production, because `scripts/` is never COPYed into the
 * Dockerfile's production stage (fixed separately, pending human approval
 * — Dockerfile is FROZEN). Reusing the Python script here would hit the
 * identical trap; this module sidesteps it by never shelling out at all.
 *
 * WHY THIS EXISTS: KNOWN BROKEN #42's own queued NEXT step (2) — "a
 * `/api/diag/reconstruct_pnl` live probe (auto-pulling current positions +
 * date ...) would remove the manual position-JSON step" the 2026-09-10
 * incident needed to run the CLI script by hand. This makes that check
 * reusable for any FUTURE equity/last_equity-vs-reality incident without
 * re-deriving anything.
 *
 * Pure module (node:test safe, no fs/network) — `bars` per position must
 * already be fetched by the caller (production: fetchDailyBarsRange's
 * Alpaca-backed daily closes in bot.ts; tests: synthetic fixtures).
 */

export interface Bar { date: string; close: number } // "YYYY-MM-DD", ascending, trading days only

export interface PositionInput {
  symbol: string;
  qty: number;
  assetClass?: string;
  /** Pre-fetched daily bars spanning at least one trading day before
   *  `date` through `date` itself. null = bars unavailable (fetch failed,
   *  or deliberately not fetched for an option leg). */
  bars: readonly Bar[] | null;
}

export interface Leg {
  symbol: string;
  qty: number;
  prev_close: number;
  close: number;
  delta: number;
  contribution: number;
}

export interface ExcludedNoData { symbol: string; reason: string }

export interface ReconstructResult {
  date: string;
  reconstructed_pnl: number;
  legs: Leg[];
  excluded_options: string[];
  excluded_no_data: ExcludedNoData[];
  reported_pnl?: number;
  gap?: number;
}

/** Same OCC-option detection heuristic diag.ts's positionsSummary already
 *  uses (symbol length > 8, or an explicit us_option asset_class) — kept
 *  consistent with the codebase's one existing option/equity split rather
 *  than introducing a second classifier. Matches the Python script's own
 *  OCC-regex intent (root 1-6 letters + YYMMDD + C/P + 8-digit strike is
 *  always >8 chars; every real equity ticker on this platform is not). */
export function isOptionPosition(symbol: string, assetClass?: string): boolean {
  return symbol.length > 8 || assetClass === "us_option";
}

function round2(n: number): number { return Math.round(n * 100) / 100; }
function round4(n: number): number { return Math.round(n * 10000) / 10000; }

/** For each non-option position with usable bars, finds `date`'s close and
 *  the immediately preceding bar's close and sums qty * (close - prevClose)
 *  across the book. Exact-date match only — a `date` that isn't a trading
 *  day (weekend/holiday) simply excludes that symbol rather than silently
 *  substituting a nearby day (mirrors the Python script's own documented
 *  behavior). Options are detected and excluded, stated honestly in the
 *  response rather than mistreated as equities (no free historical
 *  options-quote source exists — research/wishlist.md). */
export function reconstructPortfolioPnl(
  positions: readonly PositionInput[],
  date: string,
  reportedPnl?: number | null,
): ReconstructResult {
  const legs: Leg[] = [];
  const excludedOptions: string[] = [];
  const excludedNoData: ExcludedNoData[] = [];
  let total = 0;

  for (const p of positions) {
    if (isOptionPosition(p.symbol, p.assetClass)) {
      excludedOptions.push(p.symbol);
      continue;
    }
    if (!p.symbol || !Number.isFinite(p.qty) || p.qty === 0) continue;
    if (!p.bars) {
      excludedNoData.push({ symbol: p.symbol, reason: "bars fetch failed" });
      continue;
    }
    const i = p.bars.findIndex((b) => b.date === date);
    if (i < 0) {
      excludedNoData.push({ symbol: p.symbol, reason: "date not in fetched bars" });
      continue;
    }
    if (i === 0) {
      excludedNoData.push({ symbol: p.symbol, reason: "no prior-day bar in lookback window" });
      continue;
    }
    const closeToday = p.bars[i].close;
    const closePrev = p.bars[i - 1].close;
    if (!(closeToday > 0) || !(closePrev > 0)) {
      excludedNoData.push({ symbol: p.symbol, reason: "non-positive close in fetched bars" });
      continue;
    }
    const delta = closeToday - closePrev;
    const contribution = p.qty * delta;
    total += contribution;
    legs.push({
      symbol: p.symbol, qty: p.qty,
      prev_close: closePrev, close: closeToday,
      delta: round4(delta), contribution: round2(contribution),
    });
  }

  const result: ReconstructResult = {
    date,
    reconstructed_pnl: round2(total),
    legs,
    excluded_options: excludedOptions,
    excluded_no_data: excludedNoData,
  };
  if (reportedPnl !== undefined && reportedPnl !== null && Number.isFinite(reportedPnl)) {
    result.reported_pnl = reportedPnl;
    result.gap = round2(total - reportedPnl);
  }
  return result;
}
