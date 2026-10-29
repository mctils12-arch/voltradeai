// scoreboard.ts — the account vs buy-and-hold SPY, ONE ROW PER TRADING DAY
// (human-ordered 2026-10-28: "a daily bot-vs-SPY scoreboard on the bot
// page"). Pure; server/bot.ts serves it at GET /api/bot/scoreboard from the
// SAME inputs as /api/bot/benchmark (server/spyBenchmark.ts): the bot's own
// daily equity curve (Alpaca /v2/account portfolio_value, one row per UTC
// date, written by the hourly tier-3 job, last write before 00:00Z = after
// the close) and SPY's dividend-adjusted (total-return) daily closes.
//
// HONESTY RULES (each pinned by server/scoreboard.test.ts):
//   - A row exists only for a SESSION date — an equity point whose date has
//     a SPY bar. Weekend/holiday points are skipped, never compared.
//   - A row's two returns cover the SAME span: from the previous session
//     point to this one. When the curve missed sessions in between (server
//     down), the row says so (`sessions` > 1) — nothing is interpolated.
//   - Cumulative = this point / the first session point − 1, which equals
//     compounding the daily returns; it never uses a later point (no
//     lookahead: adding data after a row never changes that row).
//   - Equity change day over day is NOT adjusted for deposits/withdrawals;
//     bot.ts attaches the non-trade ledger's journals/transfers so the page
//     can say whether that caveat bites.

import type { DailyClose, CurvePoint } from "./spyBenchmark";
export type { DailyClose, CurvePoint };

export interface ScoreRow {
  /** the session date (YYYY-MM-DD) this row closes */
  date: string;
  /** the previous session point the returns are measured from */
  prevDate: string;
  /** SPY sessions the row spans: 1 normally; > 1 when the curve missed days */
  sessions: number;
  botPct: number;
  spyPct: number;
  /** bot − SPY, percentage points */
  diffPp: number;
  botCumPct: number;
  spyCumPct: number;
  cumDiffPp: number;
  /** the latest row while its UTC date is still today: equity may still move */
  provisional: boolean;
}

export interface ScoreSummary {
  from: string; to: string;
  botCumPct: number; spyCumPct: number; cumDiffPp: number;
  /** rows where the bot's return beat SPY's (strictly) / rows compared */
  daysBeat: number; daysCompared: number;
  /** rows that span more than one session (a curve gap) */
  gapRows: number;
  /** worst peak-to-trough on the session points, percent (≤ 0) */
  botMaxDdPct: number; spyMaxDdPct: number;
}

export interface Scoreboard {
  rows: ScoreRow[];            // newest first
  summary: ScoreSummary | null;
  barsThrough: string | null;
  curveThrough: string | null;
}

/** How long bot.ts keeps a computed scoreboard (the inputs move hourly at most). */
export const SCOREBOARD_CACHE_MS = 10 * 60_000;

const r2 = (x: number) => Math.round(x * 100) / 100;
const byDate = <T extends { date: string }>(a: T, b: T) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0);

/** Worst drawdown (percent, ≤ 0) of a value series. */
export function maxDrawdownPct(values: readonly number[]): number {
  let peak = -Infinity, worst = 0;
  for (const v of values) {
    if (v > peak) peak = v;
    if (peak > 0) worst = Math.min(worst, (v / peak - 1) * 100);
  }
  return r2(worst);
}

/** Pure: the daily scoreboard from the equity curve (any order) and SPY's
 *  daily closes (any order). `today` is the UTC date of the request. */
export function buildScoreboard(curve: readonly CurvePoint[], spyBars: readonly DailyClose[], today: string): Scoreboard {
  const bars = spyBars
    .filter((b) => b && typeof b.date === "string" && Number.isFinite(b.close) && b.close > 0)
    .slice().sort(byDate);
  const close = new Map(bars.map((b) => [b.date, b.close]));
  const sessionDates = bars.map((b) => b.date);
  const pts = curve
    .filter((p) => p && typeof p.date === "string" && Number.isFinite(p.value) && p.value > 0)
    .slice().sort(byDate);
  const curveThrough = pts.length ? pts[pts.length - 1].date : null;
  const barsThrough = bars.length ? bars[bars.length - 1].date : null;
  // one point per session date (the curve keeps one row per date; a
  // duplicate keeps the later row, as recordDailyEquity's rewrite would)
  const sess = new Map<string, number>();
  for (const p of pts) if (close.has(p.date)) sess.set(p.date, p.value);
  const sp = [...sess.entries()].map(([date, value]) => ({ date, value })).sort(byDate);
  if (sp.length < 2) return { rows: [], summary: null, barsThrough, curveThrough };

  const base = sp[0], baseClose = close.get(base.date)!;
  const rows: ScoreRow[] = [];
  let beat = 0, gapRows = 0;
  for (let i = 1; i < sp.length; i++) {
    const a = sp[i - 1], b = sp[i];
    const ca = close.get(a.date)!, cb = close.get(b.date)!;
    const sessions = sessionDates.filter((d) => d > a.date && d <= b.date).length;
    const bot = (b.value / a.value - 1) * 100;
    const spy = (cb / ca - 1) * 100;
    const botCum = (b.value / base.value - 1) * 100;
    const spyCum = (cb / baseClose - 1) * 100;
    if (bot > spy) beat++;
    if (sessions > 1) gapRows++;
    rows.push({
      date: b.date, prevDate: a.date, sessions,
      botPct: r2(bot), spyPct: r2(spy), diffPp: r2(bot - spy),
      botCumPct: r2(botCum), spyCumPct: r2(spyCum), cumDiffPp: r2(botCum - spyCum),
      provisional: i === sp.length - 1 && b.date >= today,
    });
  }
  const last = rows[rows.length - 1];
  const summary: ScoreSummary = {
    from: base.date, to: last.date,
    botCumPct: last.botCumPct, spyCumPct: last.spyCumPct, cumDiffPp: last.cumDiffPp,
    daysBeat: beat, daysCompared: rows.length, gapRows,
    botMaxDdPct: maxDrawdownPct(sp.map((p) => p.value)),
    spyMaxDdPct: maxDrawdownPct(sp.map((p) => close.get(p.date)!)),
  };
  return { rows: rows.reverse(), summary, barsThrough, curveThrough };
}
