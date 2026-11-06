// afterTaxSeries.ts — the vs-SPY comparison AFTER TAX, one point per trading
// day: the bot page's main chart (human refocus spec 2026-11-05 §8: "larger
// main chart: portfolio vs SPY, after tax"). Pure; server/bot.ts attaches it
// to GET /api/bot/scoreboard from the SAME inputs the daily scoreboard and the
// vs-SPY tile's after-tax line read: the daily equity curve, SPY's daily
// closes (the shared benchmark cache) and the sleeve ledger's realized events
// and fills.
//
// FRAMING — afterTaxBenchmark's own, applied at every session close: both legs
// SOLD at that day's close. SPY's whole gain since the start is taxed once at
// its holding term; the account pays the tax ledger's estimate on the trades it
// had closed by that day (netted short against long, wash sales held back the
// way the ledger holds them). Each point IS afterTaxBenchmark for the window
// start → that day, so the method can never drift from the tile's.
//
// HONESTY RULES (pinned by afterTaxSeries.test.ts):
//   - POINT-IN-TIME: a day's ledger reads only the events and fills up to the
//     end of that UTC date (the equity curve's own date key), so a later close
//     or a later re-entry (a wash sale's replacement) never changes an earlier
//     point — appending data never rewrites history.
//   - OPEN POSITIONS: their P&L at past closes is not recorded, so no point
//     includes the tax selling them would add. The vs-SPY tile's after-tax line
//     (afterTaxBenchmark with today's open P&L) is the "sold now" number the
//     page draws beside the line's end. Leaving it out of the line flatters the
//     account by that tax while its open P&L is a gain — stated in the notes.
//   - The same session points as the scoreboard (a curve date with a SPY bar,
//     one value per date, the later write winning); the first is the start,
//     all zeros. Nothing is interpolated.
//   - An estimate at the rates shown; the paper account pays no tax.
import { afterTaxBenchmark } from "./afterTaxBenchmark";
import { taxLedger, DEFAULT_TAX_RATES, type TaxRates, type Term } from "./taxLedger";
import type { CurvePoint, DailyClose } from "./spyBenchmark";
import type { RealizedEvent } from "./sleeveLedger";
import type { FillRow } from "./killSwitchAutoResume";

export interface AfterTaxPoint {
  /** the session date (YYYY-MM-DD) */
  date: string;
  /** before tax, since the start (the scoreboard's cumulative columns) */
  botPct: number; spyPct: number;
  /** the estimated tax if sold at this close, percent of the start value */
  botTaxPct: number; spyTaxPct: number;
  botAfterTaxPct: number; spyAfterTaxPct: number;
  /** botAfterTaxPct − spyAfterTaxPct, percentage points */
  gapPp: number;
  /** SPY's holding term at this close (more than a year is long) */
  spyTerm: Term;
}

export interface AfterTaxSeries {
  rates: TaxRates;
  from: string; to: string;
  /** the account's equity at the start point (the tax percents' base) */
  equityStart: number;
  /** oldest first; the first point is the start */
  points: AfterTaxPoint[];
  notes: string[];
}

const DAY_MS = 86_400_000;
const r2 = (x: number) => Math.round(x * 100) / 100;
const byDate = <T extends { date: string }>(a: T, b: T) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0);

/** Pure. `curve` and `spyBars` in any order; `events` / `fills` the sleeve ledger's tax inputs (any order).
 *  Null with fewer than two session points. */
export function afterTaxSeries(
  curve: readonly CurvePoint[],
  spyBars: readonly DailyClose[],
  tax: { events: readonly RealizedEvent[]; fills: readonly FillRow[]; rates?: Partial<TaxRates> },
): AfterTaxSeries | null {
  const close = new Map<string, number>();
  for (const b of spyBars) if (b && typeof b.date === "string" && Number.isFinite(b.close) && b.close > 0) close.set(b.date, b.close);
  // one point per session date; a duplicate keeps the later row (the scoreboard's rule)
  const sess = new Map<string, number>();
  for (const p of curve.filter((x) => x && typeof x.date === "string" && Number.isFinite(x.value) && x.value > 0).slice().sort(byDate)) {
    if (close.has(p.date)) sess.set(p.date, p.value);
  }
  const pts = [...sess.entries()].map(([date, value]) => ({ date, value })).sort(byDate);
  if (pts.length < 2) return null;
  const rates: TaxRates = { ...DEFAULT_TAX_RATES, ...(tax.rates ?? {}) };
  const base = pts[0];
  const baseClose = close.get(base.date)!;
  const startMs = Date.parse(`${base.date}T00:00:00Z`);
  const events = tax.events.filter((e) => Number.isFinite(e.t)).slice().sort((a, b) => a.t - b.t);
  const fills = tax.fills.filter((f) => Number.isFinite(f.t)).slice().sort((a, b) => a.t - b.t);
  const points: AfterTaxPoint[] = [{ date: base.date, botPct: 0, spyPct: 0, botTaxPct: 0, spyTaxPct: 0, botAfterTaxPct: 0, spyAfterTaxPct: 0, gapPp: 0, spyTerm: "short" }];
  let ie = 0, ifl = 0;
  for (let i = 1; i < pts.length; i++) {
    const p = pts[i];
    const endMs = Date.parse(`${p.date}T00:00:00Z`) + DAY_MS;
    while (ie < events.length && events[ie].t < endMs) ie++;
    while (ifl < fills.length && fills[ifl].t < endMs) ifl++;
    const ledger = taxLedger(events.slice(0, ie), fills.slice(0, ifl), { windowStartMs: startMs, rates });
    const accountPct = r2((p.value / base.value - 1) * 100);
    const spyPct = r2((close.get(p.date)! / baseClose - 1) * 100);
    const days = Math.round((Date.parse(`${p.date}T00:00:00Z`) - startMs) / DAY_MS);
    const at = afterTaxBenchmark({ from: base.date, to: p.date, days, accountPct, spyPct, gapPp: r2(accountPct - spyPct) }, ledger, { equityStart: base.value, openPnl: null });
    if (!at) return null; // unreachable with a positive start value and finite returns
    points.push({
      date: p.date, botPct: accountPct, spyPct,
      botTaxPct: at.account.taxPct, spyTaxPct: at.spy.taxPct,
      botAfterTaxPct: at.account.afterTaxPct, spyAfterTaxPct: at.spy.afterTaxPct,
      gapPp: at.gapPp, spyTerm: at.spy.term,
    });
  }
  const last = points[points.length - 1];
  const notes = [
    "each point: both sold at that day's close — an estimate at the rates shown; the paper account pays no tax",
    "SPY: its whole gain since the start taxed once at its holding term's rate (a loss owes nothing)",
    "account: the tax ledger's estimate on the trades it had closed by that day, point-in-time (later fills never change an earlier point); wash-sale losses are held back the way the ledger holds them",
    "open positions: their P&L at past closes is not recorded, so the line leaves out the tax selling them would add — the vs-SPY tile's after-tax figure (today's open P&L included) is the 'sold now' number",
  ];
  return { rates, from: base.date, to: last.date, equityStart: r2(base.value), points, notes };
}
