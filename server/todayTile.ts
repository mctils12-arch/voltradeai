// todayTile.ts — the Bot page's "Today" tile (B1, human-approved 2026-11-03:
// "show equity now vs. prior-close equity, in $ and %, plus SPY's % today and
// the difference between them"). Pure; server/bot.ts's /api/bot/account route
// feeds it the account read it already makes and the SPY bars the benchmark
// already caches, and serves the result as `today`.
//
// REUSE, NOT RE-DERIVATION. The percent is evaluateDailyPnl (drawdownGuard.ts)
// — the repo's validated daily P&L from Alpaca's equity and last_equity, the
// Node twin of bot_engine.compute_daily_pnl_pct (test_daily_pnl_pct.py) — so
// this tile can never disagree with the daily-loss halt about what "today"
// made. SPY's move is benchmarkVsSpy(...).spyLast (spyBenchmark.ts), the same
// last-bar-vs-prior-close number the vs-SPY tile and the scoreboard read, taken
// only when that last bar IS today's ET session (the daily bars end at "now",
// so during the session the last bar is the session in progress; before the
// open, on a weekend or a holiday it is an older day and is NOT shown as
// today's). Nothing is inferred: a side that is missing is null with its
// reason, never a zero.
import { evaluateDailyPnl } from "./drawdownGuard";
import { benchmarkVsSpy, type DailyClose } from "./spyBenchmark";
import { etDay } from "./entryShadow";

export interface TodayTile {
  /** the ET calendar day these numbers describe */
  day: string;
  /** Alpaca `equity` now, when a credible read */
  equity_now: number | null;
  /** Alpaca `last_equity`: the account's equity at the prior session's close */
  equity_prev_close: number | null;
  /** equity_now − equity_prev_close, in dollars (cents) */
  pnl_usd: number | null;
  /** the same change in percent of the prior close (evaluateDailyPnl, 2 dp) */
  pnl_pct: number | null;
  /** why pnl_usd / pnl_pct are null */
  pnl_reason: string | null;
  /** SPY today: today's daily bar against the prior close, percent (2 dp) */
  spy_pct_today: number | null;
  /** when the SPY bars were fetched (the cache's age is the number's age) */
  spy_asof: string | null;
  /** the date of the last SPY bar the server holds */
  spy_bar_date: string | null;
  /** why spy_pct_today is null */
  spy_reason: string | null;
  /** pnl_pct − spy_pct_today, percentage points (2 dp); null when either side is */
  excess_pct: number | null;
  /** when the account was read */
  asof: string;
  source: string;
}

export interface TodayInput {
  equity: unknown;
  lastEquity: unknown;
  /** the benchmark's cached SPY bars (any order), or null when nothing is cached */
  spyBars: readonly DailyClose[] | null | undefined;
  /** when those bars were fetched (ms), or null */
  spyFetchedAt: number | null | undefined;
  nowMs: number;
}

export const TODAY_SOURCE =
  "Alpaca paper account: equity now vs last_equity (the prior session's 16:00 ET close) · SPY: today's daily bar vs the prior close, dividend-adjusted (Alpaca bars, feed iex)";

const r2 = (x: number) => Math.round(x * 100) / 100;
/** the same parse evaluateDailyPnl applies to the raw account fields */
const asNum = (raw: unknown): number => (typeof raw === "number" ? raw : parseFloat(String(raw ?? "")));
const credible = (x: number): boolean => Number.isFinite(x) && x > 0;

export interface SpyToday { pct: number | null; asof: string | null; barDate: string | null; reason: string | null }

/** SPY's move on `day` (an ET date) from the cached daily bars, or null and why. */
export function spyToday(bars: readonly DailyClose[] | null | undefined, fetchedAtMs: number | null | undefined, day: string): SpyToday {
  const asof = typeof fetchedAtMs === "number" && Number.isFinite(fetchedAtMs) ? new Date(fetchedAtMs).toISOString() : null;
  const b = benchmarkVsSpy([], bars ? [...bars] : []);
  if (!b.barsThrough) return { pct: null, asof, barDate: null, reason: "no SPY bars fetched yet" };
  if (b.barsThrough !== day) return { pct: null, asof, barDate: b.barsThrough, reason: `no SPY bar for today yet (last ${b.barsThrough})` };
  if (!b.spyLast) return { pct: null, asof, barDate: b.barsThrough, reason: "SPY prior close missing" };
  return { pct: b.spyLast.pct, asof, barDate: b.barsThrough, reason: null };
}

/** Pure: the tile from one account read and the cached SPY bars. */
export function todayTile(i: TodayInput): TodayTile {
  const nowMs = Number.isFinite(i.nowMs) ? i.nowMs : Date.now();
  const day = etDay(nowMs);
  const equity = asNum(i.equity);
  const prev = asNum(i.lastEquity);
  const ev = evaluateDailyPnl(i.equity, i.lastEquity);
  const pnlPct = ev.valid ? r2(ev.dailyPnlPct!) : null;
  const spy = spyToday(i.spyBars, i.spyFetchedAt, day);
  return {
    day,
    equity_now: credible(equity) ? equity : null,
    equity_prev_close: credible(prev) ? prev : null,
    pnl_usd: ev.valid ? r2(equity - prev) : null,
    pnl_pct: pnlPct,
    pnl_reason: ev.valid ? null : "account read invalid (equity or prior close not a positive number)",
    spy_pct_today: spy.pct,
    spy_asof: spy.asof,
    spy_bar_date: spy.barDate,
    spy_reason: spy.reason,
    // from the two ROUNDED sides, so the three numbers the tile prints always add up
    excess_pct: pnlPct !== null && spy.pct !== null ? r2(pnlPct - spy.pct) : null,
    asof: new Date(nowMs).toISOString(),
    source: TODAY_SOURCE,
  };
}
