// Trade cockpit — which panels may fetch on their own (2026-11-06). Pure (tested).
//
// KEEP THE SYSTEM ALIVE (CLAUDE.md GOAL #1; coordinator 2026-11-06: the
// server's Node heap is ~2.7 GB, RSS ~3.4 GB, and the container OOM-restarted
// once tonight). Four of the page's symbol-keyed reads run a Python process
// per call on the server (server/routes.ts):
//   /api/analyze/<SYM>   analyze.py              the strip, Options & Volatility, Stock · Fundamentals / Earnings
//   /api/research/<SYM>  python -m alphadesk     Stock · Research
//   /api/insights/<SYM>  insights.py             Smart Money and Structure (one response, two filters)
//   /api/etf/<SYM>       the ETF builder's script ETF Builder
// The old Trade pages started one of them per look-up. The cockpit shows every
// panel at once, so a look-up runs only the first (shared by every panel that
// reads it); each of the other three waits for its own "Load … for <SYM>"
// press. A press belongs to the symbol open when it was made; the page clears
// every press on a new look-up, so a later look-up never runs one by itself.
// The News panel's symbol filter is the one other read a look-up makes: one
// HTTP call to Alpaca, no Python.

export type HeavyLoad = "research" | "insights" | "etf";
export const HEAVY_LOADS: readonly HeavyLoad[] = ["research", "insights", "etf"];

/** the symbol each heavy panel was asked to load for (null: not asked) */
export type HeavyRequests = Readonly<Record<HeavyLoad, string | null>>;
export const NO_REQUESTS: HeavyRequests = { research: null, insights: null, etf: null };

/** a press on a panel's "Load … for <SYM>" control */
export function requestLoad(r: HeavyRequests, what: HeavyLoad, sym: string): HeavyRequests {
  return { ...r, [what]: sym };
}

/** the panel may mount its fetching body: it was asked to, for the symbol open now */
export function isRequested(r: HeavyRequests, what: HeavyLoad, sym: string | null): boolean {
  return !!sym && r[what] === sym;
}

/** One symbol-keyed server read and whether the route runs a Python process. */
export interface SymbolFetch { route: string; python: boolean }

/** What the page reads for `sym` given the presses so far. The look-up alone:
 *  /api/analyze (the one Python process) and the News filter (no Python). */
export function symbolFetches(sym: string | null, r: HeavyRequests): SymbolFetch[] {
  if (!sym) return [];
  const out: SymbolFetch[] = [
    { route: `/api/analyze/${sym}`, python: true },
    { route: `/api/market/news?ticker=${sym}`, python: false },
  ];
  if (isRequested(r, "research", sym)) out.push({ route: `/api/research/${sym}`, python: true });
  if (isRequested(r, "insights", sym)) out.push({ route: `/api/insights/${sym}`, python: true });
  if (isRequested(r, "etf", sym)) out.push({ route: `/api/etf/${sym}`, python: true });
  return out;
}
