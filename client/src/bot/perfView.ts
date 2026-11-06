// Bot page OVERVIEW — the main chart's model (human refocus spec 2026-11-05 §8:
// "larger main chart: portfolio vs SPY, after tax"). Pure; pinned by
// perfView.test.ts. The lines come from GET /api/bot/scoreboard's `afterTax`
// series (server/afterTaxSeries.ts: both legs sold at each close, the account's
// tax from the ledger point-in-time); a server without it, or one whose sleeve
// ledger has not loaded, draws the scoreboard's own before-tax columns and says
// so — never a guessed tax. The "sold now" marker is the vs-SPY tile's figure
// (server/afterTaxBenchmark.ts with today's open positions), drawn beside the
// line's end, so the two numbers on the page are never silently different.
import type { Scoreboard, Benchmark } from "./useBotData";
import { fmtMonDay, MINUS } from "./format";

export interface PerfPoint {
  date: string;
  /** the drawn lines: after tax when the server sent the series (model.afterTax), else before tax */
  bot: number; spy: number;
  /** before tax — the scoreboard's cumulative columns */
  botPre: number; spyPre: number;
  /** drawdown from the running peak, before tax, percent (≤ 0) */
  botDd: number; spyDd: number;
}
export type PerfState = "loading" | "error" | "empty" | "ready";
export interface PerfModel {
  state: PerfState;
  /** the lines are after tax (false: the before-tax fallback; `note` says why) */
  afterTax: boolean;
  points: PerfPoint[];
  last: PerfPoint | null;
  /** the vs-SPY tile's after-tax figure — both sold now, open positions included; null without it */
  soldNow: { bot: number; spy: number; gap: number; to: string } | null;
  /** worst drawdown over the drawn span, before tax (the scoreboard summary's when present) */
  maxDd: { bot: number; spy: number } | null;
  rates: { shortTerm: number; longTerm: number } | null;
  from: string | null; to: string | null;
  /** why the lines are before tax, or why nothing is drawn */
  note: string | null;
}

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Running-peak drawdown of cumulative percent returns, percent (≤ 0). */
export function drawdowns(cumPct: readonly number[]): number[] {
  let peak = -Infinity;
  return cumPct.map((c) => {
    const w = 1 + c / 100;
    if (w > peak) peak = w;
    return peak > 0 ? Math.min(0, Math.round((w / peak - 1) * 10_000) / 100) : 0;
  });
}

export function perfModel(board: Scoreboard | undefined, bench: Benchmark | undefined, status: { loading: boolean; error: string | null }): PerfModel {
  const empty = (state: PerfState, note: string | null): PerfModel => ({ state, afterTax: false, points: [], last: null, soldNow: null, maxDd: null, rates: null, from: null, to: null, note });
  if (!board) return status.error ? empty("error", status.error) : empty("loading", null);
  let afterTax = false;
  let raw: Array<{ date: string; bot: number; spy: number; botPre: number; spyPre: number }> = [];
  const series = board.afterTax;
  if (series && Array.isArray(series.points) && series.points.length >= 2 && series.points.every((p) => fin(p.botAfterTaxPct) && fin(p.spyAfterTaxPct) && fin(p.botPct) && fin(p.spyPct))) {
    afterTax = true;
    raw = series.points.map((p) => ({ date: p.date, bot: p.botAfterTaxPct, spy: p.spyAfterTaxPct, botPre: p.botPct, spyPre: p.spyPct }));
  } else if (Array.isArray(board.rows) && board.rows.length) {
    // the scoreboard's rows are newest first; the oldest row's prevDate is the start (all zeros)
    const rows = [...board.rows].filter((r) => fin(r.botCumPct) && fin(r.spyCumPct)).reverse();
    if (rows.length) raw = [{ date: rows[0].prevDate, bot: 0, spy: 0, botPre: 0, spyPre: 0 }, ...rows.map((r) => ({ date: r.date, bot: r.botCumPct, spy: r.spyCumPct, botPre: r.botCumPct, spyPre: r.spyCumPct }))];
  }
  if (raw.length < 2) {
    return empty(board.error ? "error" : "empty", board.error ? `${board.error} · nothing is drawn rather than a guess` : "not enough history yet — a point needs two trading days with both an equity snapshot and a SPY close");
  }
  const bdd = drawdowns(raw.map((p) => p.botPre)), sdd = drawdowns(raw.map((p) => p.spyPre));
  const points: PerfPoint[] = raw.map((p, i) => ({ ...p, botDd: bdd[i], spyDd: sdd[i] }));
  const s = board.summary;
  // the scoreboard summary's worst drawdowns are measured on the same session points from the unrounded values
  const maxDd = s && fin(s.botMaxDdPct) && fin(s.spyMaxDdPct) ? { bot: s.botMaxDdPct, spy: s.spyMaxDdPct } : { bot: Math.min(0, ...bdd), spy: Math.min(0, ...sdd) };
  const at = bench?.afterTax ?? null;
  const soldNow = afterTax && at && fin(at.account?.afterTaxPct) && fin(at.spy?.afterTaxPct) && fin(at.gapPp) ? { bot: at.account.afterTaxPct, spy: at.spy.afterTaxPct, gap: at.gapPp, to: at.window.to } : null;
  const rates = series?.rates && fin(series.rates.shortTerm) && fin(series.rates.longTerm) ? { shortTerm: series.rates.shortTerm, longTerm: series.rates.longTerm } : null;
  const note = afterTax ? null : series === undefined ? "this server predates the after-tax series — the lines are before tax" : `after-tax series not loaded yet (${board.afterTaxReason ?? "no reason given"}) — the lines are before tax`;
  return { state: "ready", afterTax, points, last: points[points.length - 1], soldNow, maxDd, rates, from: points[0].date, to: points[points.length - 1].date, note };
}

/** "up 2.5%" · "down 11.3%" · "flat" — a signed percent as words, one decimal. */
export function moveWords(v: number): string {
  const r = Math.round(v * 10) / 10;
  return r > 0 ? `up ${r.toFixed(1)}%` : r < 0 ? `down ${Math.abs(r).toFixed(1)}%` : "flat";
}

/** The chart's plain-English headline: where both stand since the start and the gap, in points. */
export function perfHeadline(m: PerfModel): string {
  if (m.state !== "ready" || !m.last) return "";
  const basis = m.afterTax ? "After tax" : "Before tax";
  const gap = Math.round((m.last.bot - m.last.spy) * 10) / 10;
  const since = m.from ? fmtMonDay(`${m.from.slice(0, 10)}T12:00:00Z`) : "the start";
  const verdict = gap > 0 ? `ahead of SPY by ${gap.toFixed(1)} points` : gap < 0 ? `behind SPY by ${Math.abs(gap).toFixed(1)} points` : "level with SPY";
  return `${basis}, the portfolio is ${moveWords(m.last.bot)} since ${since}; SPY is ${moveWords(m.last.spy)}. The bot is ${verdict}.`;
}

// ── scales (pure; the SVG in overview.tsx draws what these return) ──────────

/** The y range over every drawn value and zero, padded 8 % so no line touches an edge. */
export function yDomain(values: readonly number[]): [number, number] {
  const v = values.filter(fin);
  let lo = Math.min(0, ...v), hi = Math.max(0, ...v);
  if (hi - lo < 1) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

/** Round ticks inside [lo, hi]: a 1 / 2 / 2.5 / 5 × 10^k step giving at most `max` ticks. */
export function niceTicks(lo: number, hi: number, max = 5): number[] {
  if (!(hi > lo) || !fin(lo) || !fin(hi)) return [];
  const raw = (hi - lo) / Math.max(1, max);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => (hi - lo) / s <= max) ?? 10 * mag;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.round(t * 1e6) / 1e6);
  return out;
}

/** "+5%" · "0%" · "−10%" — an axis label (a true minus, like every number on the page). */
export function tickLabel(v: number): string {
  const r = Math.round(v * 100) / 100;
  const body = String(Math.abs(r));
  return r > 0 ? `+${body}%` : r < 0 ? `${MINUS}${body}%` : "0%";
}

/** The first point of each calendar month (x-axis ticks), labelled "Jun"; the start's own month is skipped
 *  when its first point is the start itself (the axis begins there). */
export function monthTicks(dates: readonly string[]): Array<{ i: number; label: string }> {
  const out: Array<{ i: number; label: string }> = [];
  const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  for (let i = 1; i < dates.length; i++) {
    if (dates[i].slice(0, 7) !== dates[i - 1].slice(0, 7)) out.push({ i, label: MON[Number(dates[i].slice(5, 7)) - 1] ?? dates[i].slice(5, 7) });
  }
  return out;
}

export type Scale = (v: number) => number;
/** An SVG path through the values at x(i), y(v). */
export function linePath(values: readonly number[], x: Scale, y: Scale): string {
  return values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
}

/** The shaded gap between two lines, split where they cross so each piece carries one sign:
 *  +1 where `a` is above `b` (the bot ahead of SPY), −1 below. */
export function gapBands(a: readonly number[], b: readonly number[], x: Scale, y: Scale): Array<{ d: string; sign: 1 | -1 }> {
  const out: Array<{ d: string; sign: 1 | -1 }> = [];
  const n = Math.min(a.length, b.length);
  let top: Array<[number, number]> = [], bot: Array<[number, number]> = [], sign: 1 | -1 | 0 = 0;
  const flush = () => {
    if (top.length >= 2 && sign !== 0) out.push({ d: `M${top.map(([px, py]) => `${px.toFixed(1)} ${py.toFixed(1)}`).join(" L")} L${[...bot].reverse().map(([px, py]) => `${px.toFixed(1)} ${py.toFixed(1)}`).join(" L")} Z`, sign });
    top = []; bot = [];
  };
  for (let i = 0; i < n; i++) {
    const s: 1 | -1 | 0 = a[i] > b[i] ? 1 : a[i] < b[i] ? -1 : 0;
    if (i > 0 && s !== 0 && sign !== 0 && s !== sign) {
      // the crossing between i − 1 and i: both lines meet at fraction f
      const d0 = a[i - 1] - b[i - 1], d1 = a[i] - b[i];
      const f = d0 / (d0 - d1);
      const xi = x(i - 1) + (x(i) - x(i - 1)) * f, vi = a[i - 1] + (a[i] - a[i - 1]) * f;
      top.push([xi, y(vi)]); bot.push([xi, y(vi)]);
      flush();
      top.push([xi, y(vi)]); bot.push([xi, y(vi)]);
    }
    top.push([x(i), y(a[i])]); bot.push([x(i), y(b[i])]);
    if (s !== 0) sign = s;
  }
  flush();
  return out;
}

/** The point nearest a pointer x inside the plot [x0, x1] across n evenly spaced points. */
export function nearestIndex(px: number, n: number, x0: number, x1: number): number {
  if (n <= 1 || !(x1 > x0)) return 0;
  return Math.max(0, Math.min(n - 1, Math.round(((px - x0) / (x1 - x0)) * (n - 1))));
}
