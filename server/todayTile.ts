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
// only when that last bar IS the session's own bar. Nothing is inferred: a side
// that is missing is null with its reason, never a zero.
//
// BETWEEN SESSIONS ([REPAIR] 2026-11-06, open_questions.md 2026-11-06 "[UI]
// Two gaps": "The Today tile labels the overnight 'today' with the last
// session's move, a stale prior close overnight"). The tile used to describe
// the ET calendar day of the read. After midnight ET that day has no session
// yet, so the tile printed the LAST session's move (equity vs last_equity,
// which Alpaca rolls only overnight) under "today", or, once Alpaca had rolled
// last_equity, the after-hours drift since the close, beside "no SPY bar for
// today". Live on 2026-11-06 05:52 ET: "today" −$232 · −0.26 % (the marks since
// Thursday's close) while Thursday's session itself was −3.99 % against SPY's
// −4.34 %. Now the tile describes ONE session, named by tileSession():
//   - during a session (the broker clock's shared 60 s reading; the
//     market_calendar.py calendar when no reading is usable): today, against
//     last_equity — the prior session's close — exactly as before;
//   - between sessions (after the close, overnight, pre-market, weekends,
//     market holidays): the last session that ran. Its prior close is
//     last_equity while Alpaca still holds it (the session's own ET day), or
//     that session's daily line's record of it (server/dailyLine.ts writes
//     last_equity from the close + 20 min, before midnight ET — on 2026-11-05
//     it had not rolled at 20:12 ET). The change runs to equity now (after-
//     hours marks included) until Alpaca rolls last_equity, and from then to
//     last_equity itself: the session's own 16:00 ET close, Alpaca's snapshot,
//     so the account side is close to close like SPY's bar. With neither
//     record, the account side is null with the reason.
//   SPY's side is that session's bar (bars after it, e.g. a pre-market bar of
//   the next day, are left out) and says whether it was fetched after the
//   session's close, i.e. is the final close.
import { evaluateDailyPnl } from "./drawdownGuard";
import { benchmarkVsSpy, type DailyClose } from "./spyBenchmark";
import { isTradingDay, zonedInstant, zonedParts, type MarketCalendar } from "./jobHeartbeat";
import { EARLY_CLOSE_MIN_ET, PREMARKET_START_MIN_ET, REGULAR_CLOSE_MIN_ET, REGULAR_OPEN_MIN_ET, type ClockStateName } from "./brokerClockGate";

/** "session": a regular session is running and the tile is today's; "between": it is the last session's */
export type TilePhase = "session" | "between";
/** where the account side's prior close came from */
export type PrevCloseSource = "last_equity" | "daily_line";
/** what the change runs to: equity now, or the session's 16:00 ET close (Alpaca's last_equity once rolled to it) */
export type EndBasis = "now" | "close";

export interface TodayTile {
  /** the ET date of the session these numbers describe: today during a session, the last session between sessions */
  day: string;
  /** a session running now ("session"), or between sessions ("between") */
  phase: TilePhase;
  /** when the read falls: "during the session" · "after the close" · "overnight" · "pre-market" · "weekend" · "market holiday" */
  when: string;
  /** what decided the phase: the broker clock, its last reading extrapolated, or the market calendar */
  phase_source: string;
  /** the trading day before `day`: the session whose close is the prior close (market_calendar.py) */
  prev_day: string | null;
  /** Alpaca `equity` now, when a credible read */
  equity_now: number | null;
  /** the close before `day`'s session (Alpaca last_equity, or the daily line's record of it between sessions) */
  equity_prev_close: number | null;
  prev_close_source: PrevCloseSource | null;
  /** what the change runs to: equity now, or `day`'s own 16:00 ET close once Alpaca has rolled last_equity to it */
  equity_end: number | null;
  end_basis: EndBasis | null;
  /** equity_end − equity_prev_close, in dollars (cents) */
  pnl_usd: number | null;
  /** the same change in percent of the prior close (evaluateDailyPnl, 2 dp) */
  pnl_pct: number | null;
  /** why pnl_usd / pnl_pct are null */
  pnl_reason: string | null;
  /** SPY on `day`: that session's daily bar against the prior close, percent (2 dp) */
  spy_pct_today: number | null;
  /** when the SPY bars were fetched (the cache's age is the number's age) */
  spy_asof: string | null;
  /** the date of the SPY bar used (or the last bar held, when none is `day`'s) */
  spy_bar_date: string | null;
  /** why spy_pct_today is null */
  spy_reason: string | null;
  /** the bars were fetched after `day`'s close: spy_pct_today is the session's final close, not an intraday reading */
  spy_final: boolean;
  /** pnl_pct − spy_pct_today, percentage points (2 dp); null when either side is */
  excess_pct: number | null;
  /** when the account was read */
  asof: string;
  source: string;
}

/** The broker clock's state (server/brokerClockGate.ts clockStateFrom) and whether its reading is live. */
export interface TileClock { state: ClockStateName; source: "live" | "extrapolated" | "none" }

/** What the tile reads from one stored daily line (server/dailyLine.ts DailyLine). */
export interface TodayLineRecord {
  /** the session the line describes */
  day: string;
  /** Alpaca last_equity as the line read it after that session's close: the close before the session */
  lastEquity: number | null;
  writtenAt: string | null;
}

export interface TodayInput {
  equity: unknown;
  lastEquity: unknown;
  /** the benchmark's cached SPY bars (any order), or null when nothing is cached */
  spyBars: readonly DailyClose[] | null | undefined;
  /** when those bars were fetched (ms), or null */
  spyFetchedAt: number | null | undefined;
  nowMs: number;
  /** the broker clock now; absent or "unknown" = the market calendar decides the phase */
  clock?: TileClock | null;
  /** market_calendar.py's holidays and half days (absent = weekends only) */
  calendar?: MarketCalendar | null;
  /** the stored daily lines (any order; todayLineRecord), for the prior close between sessions */
  lines?: readonly TodayLineRecord[] | null;
}

// SPY's side is the benchmark's own series: price return, split-adjusted with no dividends, like the
// paper account, which is credited none (spyBenchmark.ts SPY_BENCH_ADJUSTMENT; 2026-11-05, was dividend-adjusted)
export const TODAY_SOURCE =
  "Alpaca paper account: equity now vs last_equity (the prior session's 16:00 ET close) · SPY: today's daily bar vs the prior close, price only, no dividends, like the paper account (Alpaca bars, adjustment=split, feed iex)";
/** the same sentence between sessions (2026-11-06): the last session, against the close before it */
export const TODAY_SOURCE_BETWEEN =
  "Between sessions, the last session: Alpaca paper account equity now, or that session's own 16:00 ET close once Alpaca has rolled last_equity to it, vs the close before it (last_equity while Alpaca still holds it, else the session's daily line's record of it) · SPY: that session's daily bar vs the prior close, price only, no dividends, like the paper account (Alpaca bars, adjustment=split, feed iex)";

/** last_equity differs from the line's record by at least this (half a cent): Alpaca has rolled it to the session's close */
export const LAST_EQUITY_ROLL_EPS = 0.005;
// bot.ts todayFor's inputs (2026-11-06):
/** a broker clock reading at most this old is live (the position sync refreshes the shared 60 s cache every minute);
 *  an older one is extrapolated from the times it states (brokerClockGate.clockStateFrom) */
export const TODAY_CLOCK_LIVE_MS = 2 * 60_000;
/** the stored daily lines are re-read at most this often (one is written a day, from the close + 20 min) */
export const TODAY_LINES_TTL_MS = 60_000;
/** how many of the newest stored lines the tile reads (it needs the last session's) */
export const TODAY_LINES_KEEP = 10;

const ET = "America/New_York";
const NO_CALENDAR: MarketCalendar = { holidays: new Set<string>(), halfDays: new Set<string>() };
/** how far back the trading-day walk looks (the longest NYSE closure is a few days around a weekend) */
const TRADING_DAY_WALK = 10;

const r2 = (x: number) => Math.round(x * 100) / 100;
/** the same parse evaluateDailyPnl applies to the raw account fields */
const asNum = (raw: unknown): number => (typeof raw === "number" ? raw : parseFloat(String(raw ?? "")));
const credible = (x: number): boolean => Number.isFinite(x) && x > 0;

function shiftDay(day: string, days: number): string {
  const [y, mo, d] = day.split("-").map(Number);
  return new Date(Date.UTC(y, mo - 1, d + days)).toISOString().slice(0, 10);
}

/** The last trading day strictly before `day` (a weekday that is not a market_calendar.py holiday). */
export function previousTradingDay(day: string, calendar: MarketCalendar | null | undefined): string | null {
  const cal = calendar ?? NO_CALENDAR;
  for (let k = 1; k <= TRADING_DAY_WALK; k++) {
    const d = shiftDay(day, -k);
    if (isTradingDay(d, cal)) return d;
  }
  return null;
}

/** minutes after midnight ET at which `day`'s session closes (13:00 on a market_calendar.py half day) */
const closeMinOf = (day: string, cal: MarketCalendar) => (cal.halfDays.has(day) ? EARLY_CLOSE_MIN_ET : REGULAR_CLOSE_MIN_ET);

export interface TileSession {
  phase: TilePhase;
  /** the session the tile describes */
  day: string;
  prevDay: string | null;
  when: string;
  source: string;
  /** that session's close (ms) */
  closeMs: number;
}

/** Which session the tile describes at nowMs. A known broker clock decides "session" vs "between"; without one
 *  (no reading, or one too old to extrapolate) the calendar does: a trading day, 09:30 ET to its close. Between
 *  sessions it is the last session that ran: today's once today's session has opened, else the trading day before. */
export function tileSession(nowMs: number, clock: TileClock | null | undefined, calendar: MarketCalendar | null | undefined): TileSession {
  const cal = calendar ?? NO_CALENDAR;
  const p = zonedParts(nowMs, ET);
  const today = p.date;
  const trading = isTradingDay(today, cal);
  const closeMin = closeMinOf(today, cal);
  const known = !!clock && clock.state !== "unknown";
  const open = known ? clock!.state === "open" : trading && p.minutes >= REGULAR_OPEN_MIN_ET && p.minutes < closeMin;
  const source = !known
    ? "market calendar (no usable broker clock reading)"
    : clock!.source === "live" ? "broker clock" : "broker clock, extrapolated from its last reading";
  const at = (day: string, when: string, phase: TilePhase): TileSession =>
    ({ phase, day, prevDay: previousTradingDay(day, cal), when, source, closeMs: zonedInstant(day, closeMinOf(day, cal), ET) });
  if (open) return at(today, "during the session", "session");
  if (trading && p.minutes >= REGULAR_OPEN_MIN_ET) return at(today, "after the close", "between");
  const when = trading
    ? (p.minutes < PREMARKET_START_MIN_ET ? "overnight" : "pre-market")
    : (p.weekday === "Sat" || p.weekday === "Sun" ? "weekend" : "market holiday");
  return at(previousTradingDay(today, cal) ?? shiftDay(today, -1), when, "between");
}

/** One stored daily line -> what the tile reads from it; null when it is not a line. */
export function todayLineRecord(raw: unknown): TodayLineRecord | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const o = raw as Record<string, unknown>;
  if (typeof o.day !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(o.day)) return null;
  const avs = o.account_vs_spy && typeof o.account_vs_spy === "object" ? (o.account_vs_spy as Record<string, unknown>) : {};
  const d = avs.day && typeof avs.day === "object" ? (avs.day as Record<string, unknown>) : {};
  const le = typeof d.last_equity === "number" && credible(d.last_equity) ? d.last_equity : null;
  return { day: o.day, lastEquity: le, writtenAt: typeof o.written_at === "string" ? o.written_at : null };
}

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

/** Pure: the tile from one account read, the cached SPY bars, the clock, the calendar and the stored daily lines. */
export function todayTile(i: TodayInput): TodayTile {
  const nowMs = Number.isFinite(i.nowMs) ? i.nowMs : Date.now();
  const s = tileSession(nowMs, i.clock, i.calendar);
  const today = zonedParts(nowMs, ET).date;
  const equity = asNum(i.equity);
  const last = asNum(i.lastEquity);
  // the read itself, by the daily-loss halt's own guard: an impossible equity or last_equity voids the account side
  const ev = evaluateDailyPnl(i.equity, i.lastEquity);
  // between sessions: the daily line written after the session the tile describes
  const line = s.phase === "between" ? (i.lines ?? []).find((l) => !!l && l.day === s.day && l.lastEquity !== null) ?? null : null;
  let from: number | null = null;
  let fromSource: PrevCloseSource | null = null;
  let to: number | null = null;
  let toBasis: EndBasis | null = null;
  let reason: string | null = null;
  if (s.phase === "session" || (!line && s.day === today)) {
    // during the session, and after the close on the session's own ET day (Alpaca rolls last_equity overnight)
    from = credible(last) ? last : null; fromSource = from === null ? null : "last_equity";
    to = credible(equity) ? equity : null; toBasis = to === null ? null : "now";
  } else if (line) {
    from = line.lastEquity; fromSource = "daily_line";
    const rolled = credible(last) && Math.abs(last - line.lastEquity!) >= LAST_EQUITY_ROLL_EPS;
    to = rolled ? last : credible(equity) ? equity : null;
    toBasis = to === null ? null : rolled ? "close" : "now";
  } else {
    reason = `the close before the ${s.day} session is not recorded (no daily line for ${s.day}), and Alpaca's last_equity may already be that session's own close`;
  }
  const move = ev.valid && from !== null && to !== null ? evaluateDailyPnl(to, from) : null;
  const pnlPct = move && move.valid ? r2(move.dailyPnlPct!) : null;
  if (!ev.valid) reason = "account read invalid (equity or prior close not a positive number)";
  // SPY: the session's own bar; a later bar (the next day's pre-market) is not this session's
  const bars = i.spyBars ? i.spyBars.filter((b) => !!b && typeof b.date === "string" && b.date <= s.day) : i.spyBars;
  const spy = spyToday(bars, i.spyFetchedAt, s.day);
  const spyReason = s.phase === "between" && spy.barDate && spy.barDate !== s.day ? `no SPY bar for ${s.day} yet (last ${spy.barDate})` : spy.reason;
  const spyFinal = spy.pct !== null && typeof i.spyFetchedAt === "number" && Number.isFinite(i.spyFetchedAt) && i.spyFetchedAt >= s.closeMs;
  return {
    day: s.day,
    phase: s.phase,
    when: s.when,
    phase_source: s.source,
    prev_day: s.prevDay,
    equity_now: credible(equity) ? equity : null,
    equity_prev_close: from,
    prev_close_source: fromSource,
    equity_end: to,
    end_basis: toBasis,
    pnl_usd: pnlPct !== null ? r2(to! - from!) : null,
    pnl_pct: pnlPct,
    pnl_reason: pnlPct !== null ? null : reason,
    spy_pct_today: spy.pct,
    spy_asof: spy.asof,
    spy_bar_date: spy.barDate,
    spy_reason: spyReason,
    spy_final: spyFinal,
    // from the two ROUNDED sides, so the three numbers the tile prints always add up
    excess_pct: pnlPct !== null && spy.pct !== null ? r2(pnlPct - spy.pct) : null,
    asof: new Date(nowMs).toISOString(),
    source: s.phase === "session" ? TODAY_SOURCE : TODAY_SOURCE_BETWEEN,
  };
}
