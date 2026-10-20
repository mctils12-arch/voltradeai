// spyBenchmark.ts — the account measured against buy-and-hold SPY, the
// GOAL's own yardstick (CLAUDE.md priority 3). Pure; server/bot.ts fetches
// the bars and serves GET /api/bot/benchmark.
//
// MEASUREMENT NOTES (2026-10-20, [RULE-REVIEW]):
//   - SPY bars are DIVIDEND-adjusted (Alpaca adjustment=all), so SPY is a
//     total-return benchmark. A split-only series would understate SPY by
//     its ~1.2%/yr yield and flatter the account — the wrong direction for
//     an honesty metric.
//   - The equity curve keeps one row per UTC date, last written by the
//     hourly tier-3 job before 00:00Z (~19-20 ET), i.e. after the close; each
//     row is compared with the SPY close of the same date, or the latest
//     earlier close when that date was not a session.
//   - Nothing is inferred: a window whose ends lack a SPY close is null.

export interface DailyClose { date: string; close: number }
export interface CurvePoint { date: string; value: number }
export interface BenchWindow {
  from: string; to: string; days: number;
  accountPct: number; spyPct: number; gapPp: number;
}
export interface Benchmark {
  since: BenchWindow | null;
  d30: BenchWindow | null;
  spyLast: { date: string; pct: number } | null;
  barsThrough: string | null;
}

/** Window length for the short comparison. */
export const BENCH_SHORT_WINDOW_DAYS = 30;
/** How long the server keeps fetched SPY bars (the series moves daily). */
export const SPY_BENCH_CACHE_MS = 30 * 60_000;

const r2 = (x: number) => Math.round(x * 100) / 100;
const dayMs = (d: string) => Date.parse(`${d}T00:00:00Z`);

/** Latest close on or before `date` (bars sorted ascending), else null. */
export function closeOnOrBefore(bars: DailyClose[], date: string): DailyClose | null {
  let lo = 0, hi = bars.length - 1, hit: DailyClose | null = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (bars[mid].date <= date) { hit = bars[mid]; lo = mid + 1; } else hi = mid - 1;
  }
  return hit;
}

function window(a: CurvePoint, b: CurvePoint, bars: DailyClose[]): BenchWindow | null {
  const sa = closeOnOrBefore(bars, a.date);
  const sb = closeOnOrBefore(bars, b.date);
  if (!sa || !sb || !(a.value > 0) || !(sa.close > 0) || a.date >= b.date) return null;
  const accountPct = (b.value / a.value - 1) * 100;
  const spyPct = (sb.close / sa.close - 1) * 100;
  return {
    from: a.date, to: b.date, days: Math.round((dayMs(b.date) - dayMs(a.date)) / 86_400_000),
    accountPct: r2(accountPct), spyPct: r2(spyPct), gapPp: r2(accountPct - spyPct),
  };
}

/** Pure: compare the equity curve (any order) with SPY daily closes. */
export function benchmarkVsSpy(curve: CurvePoint[], spyBars: DailyClose[]): Benchmark {
  const pts = curve
    .filter((p) => p && typeof p.date === "string" && Number.isFinite(p.value) && p.value > 0)
    .sort((x, y) => (x.date < y.date ? -1 : x.date > y.date ? 1 : 0));
  const bars = spyBars
    .filter((b) => b && typeof b.date === "string" && Number.isFinite(b.close) && b.close > 0)
    .sort((x, y) => (x.date < y.date ? -1 : x.date > y.date ? 1 : 0));
  const last = pts[pts.length - 1];
  let since: BenchWindow | null = null, d30: BenchWindow | null = null;
  if (pts.length >= 2 && bars.length) {
    since = window(pts[0], last, bars);
    const cutoff = new Date(dayMs(last.date) - BENCH_SHORT_WINDOW_DAYS * 86_400_000).toISOString().slice(0, 10);
    const start = [...pts].reverse().find((p) => p.date <= cutoff);
    d30 = start ? window(start, last, bars) : null;
  }
  const n = bars.length;
  const spyLast = n >= 2 ? { date: bars[n - 1].date, pct: r2((bars[n - 1].close / bars[n - 2].close - 1) * 100) } : null;
  return { since, d30, spyLast, barsThrough: n ? bars[n - 1].date : null };
}
