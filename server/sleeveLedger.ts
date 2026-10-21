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

/** `openedAt`: when the lot being closed was opened (the fill that took the
 *  position from flat); null when the lot predates the fetched history.
 *  `qty` / `closeDir` (2026-10-21, taxLedger.ts): the quantity this event
 *  closed and the direction of the closing fill (−1 sold a long, +1 covered
 *  a short) — the wash-sale rule needs both; totals are unchanged. */
export interface RealizedEvent { symbol: string; t: number; pnl: number; openedAt: number | null; qty: number; closeDir: 1 | -1 }

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
    let openedAt: number | null = null; // a lot held before the first fetched fill has no known open time
    for (const f of list) {
      if (pos === 0 || Math.sign(pos) === f.dir) {
        const absPos = Math.abs(pos);
        if (pos === 0) { avg = f.price; openedAt = f.t; }
        else avg = avg === null ? null : (absPos * avg + f.qty * f.price) / (absPos + f.qty);
        pos += f.dir * f.qty;
        continue;
      }
      const closing = Math.min(f.qty, Math.abs(pos));
      if (avg === null) unpriced.set(symbol, (unpriced.get(symbol) ?? 0) + closing);
      else events.push({ symbol, t: f.t, pnl: closing * (f.price - avg) * Math.sign(pos) * mult, openedAt, qty: closing, closeDir: f.dir });
      pos += f.dir * closing;
      if (Math.abs(pos) < 1e-9) pos = 0;
      const rest = f.qty - closing;
      if (rest > 1e-9) { pos = f.dir * rest; avg = f.price; openedAt = f.t; }
      else if (pos === 0) { avg = null; openedAt = null; }
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
/** One options underlying (OCC root): where the options sleeve's P&L came
 *  from. Symbols are within the 2026-07-07 human-authorized diag widening. */
export interface UnderlyingRow { root: string; fills: number; realized: number; byMonth: Record<string, number>; open: number | null; openPositions: number }
export interface SleeveLedger {
  firstFill: string | null; lastFill: string | null; rows: SleeveRow[]; realizedTotal: number; unrealizedTotal: number | null;
  /** the P&L window's start (ISO) when one was given — every lot's BASIS comes from the whole fill history,
   *  but fills, realized P&L and months are counted from here ([RULE-REVIEW] 2026-10-20: the +$4.7k residual) */
  windowStart: string | null;
  /** fills before the window that only served as basis */
  preWindowFills: number;
  /** options sleeve by underlying, losers first, capped at SLEEVE_UNDERLYING_CAP */
  optionsByUnderlying: UnderlyingRow[];
  /** how many underlyings the options sleeve touched (the list above may be shorter) */
  optionsUnderlyings: number;
  /** options realized P&L by hold time of the closed lot, every bucket present */
  optionsByHold: HoldRow[];
  /** the same split per CLOSE month (ISO yyyy-mm, oldest first) so the months after a fix read on their own */
  optionsByHoldMonth: Array<{ month: string; rows: HoldRow[] }>;
}
export const SLEEVE_UNDERLYING_CAP = 40;

/** Options realized P&L by how long the lot was held ([RULE-REVIEW]
 *  2026-10-20): the research round's churn question — August's NVDA 133
 *  fills for −$1,126, the entry/exit DTE conflict's instant closes (fixed
 *  v1.0.1117) — needs the instant round trips separated from the adverse
 *  moves. Same realized events, bucketed by close − open. */
export const HOLD_BUCKETS = ["<1h", "<1d", "1–7d", "8–30d", ">30d", "unknown"] as const;
export type HoldBucket = (typeof HOLD_BUCKETS)[number];
export interface HoldRow { bucket: HoldBucket; events: number; realized: number }
export function holdBucket(openedAt: number | null, closedAt: number): HoldBucket {
  if (openedAt === null) return "unknown";
  const h = (closedAt - openedAt) / 3_600_000;
  if (h < 1) return "<1h";
  if (h < 24) return "<1d";
  if (h <= 24 * 7) return "1–7d";
  if (h <= 24 * 30) return "8–30d";
  return ">30d";
}

/** "TSLA261030C00470000" → "TSLA"; a stock symbol is its own root. */
export function occRoot(symbol: string): string {
  return symbol.length > 15 && /\d{6}[CP]\d{8}$/.test(symbol) ? symbol.slice(0, -15) : symbol;
}

const r2 = (x: number) => Math.round(x * 100) / 100;

/** Pure: fills (any order) + current positions → P&L by sleeve.
 *  `windowStartMs` (optional): the whole fill history still prices every lot
 *  (a position held at the window start gets its REAL basis instead of the
 *  current avg_entry_price), but fills, realized events and months are
 *  counted only from the window start. Without it, every fill is in the window. */
export function sleeveLedger(fills: readonly FillRow[], positions: readonly PositionRow[], floorTickers: ReadonlySet<string>, windowStartMs?: number): SleeveLedger {
  const all = [...fills].sort((a, b) => a.t - b.t);
  const { events: allEvents, unpriced } = realizedEvents(all, positions);
  const inWindow = (t: number) => windowStartMs === undefined || t >= windowStartMs;
  const sorted = all.filter((f) => inWindow(f.t));
  const events = allEvents.filter((e) => inWindow(e.t));
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
  // the options sleeve by underlying (where its P&L came from)
  const und = new Map<string, UnderlyingRow>();
  const row = (root: string) => { let u = und.get(root); if (!u) { u = { root, fills: 0, realized: 0, byMonth: {}, open: 0, openPositions: 0 }; und.set(root, u); } return u; };
  for (const f of sorted) if (sleeveOf(f.symbol, floorTickers) === "options") row(occRoot(f.symbol)).fills++;
  for (const e of events) {
    if (sleeveOf(e.symbol, floorTickers) !== "options") continue;
    const u = row(occRoot(e.symbol));
    u.realized += e.pnl;
    const m = new Date(e.t).toISOString().slice(0, 7);
    u.byMonth[m] = (u.byMonth[m] ?? 0) + e.pnl;
  }
  for (const p of positions) {
    if (sleeveOf(p.symbol, floorTickers, p.option) !== "options") continue;
    const u = row(occRoot(p.symbol));
    u.openPositions++;
    u.open = u.open === null || p.unrealizedPl === null ? null : u.open + p.unrealizedPl;
  }
  const optionsByUnderlying = [...und.values()]
    .map((u) => ({ ...u, realized: r2(u.realized), open: u.open === null ? null : r2(u.open),
      byMonth: Object.fromEntries(Object.entries(u.byMonth).sort().map(([k, v]) => [k, r2(v)])) }))
    .sort((a, b) => a.realized - b.realized || a.root.localeCompare(b.root))
    .slice(0, SLEEVE_UNDERLYING_CAP);
  const hold = new Map<HoldBucket, HoldRow>(HOLD_BUCKETS.map((b) => [b, { bucket: b, events: 0, realized: 0 }]));
  for (const e of events) {
    if (sleeveOf(e.symbol, floorTickers) !== "options") continue;
    const h = hold.get(holdBucket(e.openedAt, e.t))!;
    h.events++;
    h.realized += e.pnl;
  }
  const optionsByHold = HOLD_BUCKETS.map((b) => ({ ...hold.get(b)!, realized: r2(hold.get(b)!.realized) }));
  const holdMonth = new Map<string, Map<HoldBucket, HoldRow>>();
  for (const e of events) {
    if (sleeveOf(e.symbol, floorTickers) !== "options") continue;
    const mo = new Date(e.t).toISOString().slice(0, 7);
    if (!holdMonth.has(mo)) holdMonth.set(mo, new Map(HOLD_BUCKETS.map((b) => [b, { bucket: b, events: 0, realized: 0 }])));
    const h = holdMonth.get(mo)!.get(holdBucket(e.openedAt, e.t))!;
    h.events++;
    h.realized += e.pnl;
  }
  const optionsByHoldMonth = [...holdMonth.keys()].sort().map((month) => ({
    month, rows: HOLD_BUCKETS.map((b) => ({ ...holdMonth.get(month)!.get(b)!, realized: r2(holdMonth.get(month)!.get(b)!.realized) })),
  }));
  return {
    firstFill: all.length ? new Date(all[0].t).toISOString() : null,
    lastFill: all.length ? new Date(all[all.length - 1].t).toISOString() : null,
    rows: out, realizedTotal: r2(out.reduce((s, r) => s + r.realized, 0)), unrealizedTotal,
    windowStart: windowStartMs === undefined ? null : new Date(windowStartMs).toISOString(),
    preWindowFills: all.length - sorted.length,
    optionsByUnderlying, optionsUnderlyings: und.size, optionsByHold, optionsByHoldMonth,
  };
}

/** bounded activity read: 100 per page (Alpaca's max), newest first; the cap
 *  cuts the OLDEST rows (basis only), never the newest. 150 pages = 15k rows ≈
 *  the whole paper account's history with headroom ([RULE-REVIEW] 2026-10-20;
 *  was 60 when only the window's fills were read). */
export const SLEEVE_FILLS_PAGE_SIZE = 100;
export const SLEEVE_FILLS_MAX_PAGES = 150;
/** the ledger is recomputed at most this often */
export const SLEEVE_CACHE_MS = 30 * 60_000;
/** a cold request waits this long for the first computation, then answers "pending" */
export const SLEEVE_COLD_WAIT_MS = 12_000;
