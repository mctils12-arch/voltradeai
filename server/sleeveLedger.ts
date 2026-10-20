// sleeveLedger.ts — where the account's P&L comes from, split by sleeve
// ([RULE-REVIEW] measurement, 2026-10-20). The account trailed SPY by ~12 pt
// since 2026-05-06; a top-down read attributed ~5 pt to the ETF floor's
// composition and left ~6 pt UNATTRIBUTED, because nothing recorded P&L per
// part of the strategy (trade_feedback had P&L on 55 of 386 exits and none
// on options). Alpaca's own fill ledger (/v2/account/activities/FILL) is
// complete, so this computes, from it alone:
//   realized P&L per sleeve (and per calendar month, UTC) + current
//   unrealized P&L per sleeve from /v2/positions.
// Sleeves: "options" (OCC symbols), "floor" (the passive floor + leg ETFs —
// the same FLOOR_AND_LEG_TICKERS set bot.ts already treats as passive),
// "stocks" (everything else). Pure; bot.ts serves GET /api/bot/sleeves.
//
// SAME RULER as the kill switch's reconciliation: realizedEvents() is the
// average-cost walk of killSwitchAutoResume.realizedPnlFromFills, emitting
// each close as a timed event; its total is pinned EQUAL to that function's
// on the same inputs (sleeveLedger.test.ts). Closes against a basis the
// window does not contain (opened before the first fetched fill, no longer
// held) are counted in `unpricedCloses`, never guessed. No fees are in
// Alpaca's paper fills, so none are modelled.

import type { FillRow, PositionRow } from "./killSwitchAutoResume";
import { isOptionPosition } from "./reconstructPnl";

export type Sleeve = "options" | "floor" | "stocks";
export const SLEEVES: readonly Sleeve[] = ["options", "floor", "stocks"];

export function sleeveOf(symbol: string, floorTickers: ReadonlySet<string>, option?: boolean): Sleeve {
  if (option ?? isOptionPosition(symbol)) return "options";
  return floorTickers.has(symbol.toUpperCase()) ? "floor" : "stocks";
}

export interface RealizedEvent { symbol: string; t: number; pnl: number }

/** The realizedPnlFromFills walk, one event per priced close. */
export function realizedEvents(fills: readonly FillRow[], positions: readonly PositionRow[]): { events: RealizedEvent[]; unpriced: Map<string, number> } {
  const bySymbol = new Map<string, FillRow[]>();
  for (const f of fills) {
    const arr = bySymbol.get(f.symbol);
    if (arr) arr.push(f); else bySymbol.set(f.symbol, [f]);
  }
  const held = new Map(positions.map((p) => [p.symbol, p] as const));
  const events: RealizedEvent[] = [];
  const unpriced = new Map<string, number>();
  for (const [symbol, list] of bySymbol) {
    const mult = isOptionPosition(symbol) ? 100 : 1;
    const cur = held.get(symbol);
    const netFilled = list.reduce((s, f) => s + f.dir * f.qty, 0);
    let pos = (cur ? cur.qty : 0) - netFilled;
    if (Math.abs(pos) < 1e-9) pos = 0;
    let avg: number | null =
      pos !== 0 && cur && cur.avgEntry !== null && Math.sign(cur.qty) === Math.sign(pos) ? cur.avgEntry : null;
    for (const f of list) {
      if (pos === 0 || Math.sign(pos) === f.dir) {
        const absPos = Math.abs(pos);
        avg = pos === 0 ? f.price : avg === null ? null : (absPos * avg + f.qty * f.price) / (absPos + f.qty);
        pos += f.dir * f.qty;
        continue;
      }
      const closing = Math.min(f.qty, Math.abs(pos));
      if (avg === null) unpriced.set(symbol, (unpriced.get(symbol) ?? 0) + closing);
      else events.push({ symbol, t: f.t, pnl: closing * (f.price - avg) * Math.sign(pos) * mult });
      pos += f.dir * closing;
      if (Math.abs(pos) < 1e-9) pos = 0;
      const rest = f.qty - closing;
      if (rest > 1e-9) { pos = f.dir * rest; avg = f.price; }
      else if (pos === 0) avg = null;
    }
  }
  return { events, unpriced };
}

export interface SleeveRow {
  sleeve: Sleeve;
  fills: number;
  symbols: number;
  realized: number;
  unpricedCloses: number;
  unrealized: number | null;   // null = a held position reported no unrealized P&L
  openPositions: number;
  byMonth: Record<string, number>;
}
export interface SleeveLedger { firstFill: string | null; lastFill: string | null; rows: SleeveRow[]; realizedTotal: number; unrealizedTotal: number | null }

const r2 = (x: number) => Math.round(x * 100) / 100;

/** Pure: fills (any order) + current positions → P&L by sleeve. */
export function sleeveLedger(fills: readonly FillRow[], positions: readonly PositionRow[], floorTickers: ReadonlySet<string>): SleeveLedger {
  const sorted = [...fills].sort((a, b) => a.t - b.t);
  const { events, unpriced } = realizedEvents(sorted, positions);
  const rows = new Map<Sleeve, SleeveRow & { syms: Set<string> }>(SLEEVES.map((s) => [s, {
    sleeve: s, fills: 0, symbols: 0, realized: 0, unpricedCloses: 0, unrealized: 0, openPositions: 0, byMonth: {}, syms: new Set<string>(),
  }]));
  for (const f of sorted) { const r = rows.get(sleeveOf(f.symbol, floorTickers))!; r.fills++; r.syms.add(f.symbol); }
  for (const e of events) {
    const r = rows.get(sleeveOf(e.symbol, floorTickers))!;
    r.realized += e.pnl;
    const m = new Date(e.t).toISOString().slice(0, 7);
    r.byMonth[m] = (r.byMonth[m] ?? 0) + e.pnl;
  }
  for (const [sym, q] of unpriced) rows.get(sleeveOf(sym, floorTickers))!.unpricedCloses += q;
  for (const p of positions) {
    const r = rows.get(sleeveOf(p.symbol, floorTickers, p.option))!;
    r.openPositions++;
    r.unrealized = r.unrealized === null || p.unrealizedPl === null ? null : r.unrealized + p.unrealizedPl;
  }
  const out: SleeveRow[] = SLEEVES.map((s) => {
    const { syms, ...r } = rows.get(s)!;
    return {
      ...r, symbols: syms.size, realized: r2(r.realized), unrealized: r.unrealized === null ? null : r2(r.unrealized),
      byMonth: Object.fromEntries(Object.entries(r.byMonth).sort().map(([k, v]) => [k, r2(v)])),
    };
  });
  const unrealizedTotal = out.some((r) => r.unrealized === null) ? null : r2(out.reduce((s, r) => s + (r.unrealized ?? 0), 0));
  return {
    firstFill: sorted.length ? new Date(sorted[0].t).toISOString() : null,
    lastFill: sorted.length ? new Date(sorted[sorted.length - 1].t).toISOString() : null,
    rows: out, realizedTotal: r2(out.reduce((s, r) => s + r.realized, 0)), unrealizedTotal,
  };
}

/** bounded fills read: 100 per page (Alpaca's max) */
export const SLEEVE_FILLS_PAGE_SIZE = 100;
export const SLEEVE_FILLS_MAX_PAGES = 60;
/** the ledger is recomputed at most this often */
export const SLEEVE_CACHE_MS = 30 * 60_000;
/** a cold request waits this long for the first computation, then answers "pending" */
export const SLEEVE_COLD_WAIT_MS = 12_000;
