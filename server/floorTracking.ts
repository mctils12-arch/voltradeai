// floorTracking.ts — the volatility-managed floor's WATCH LIST, compiled
// ([RULE-REVIEW] measurement, 2026-11-12). research/open_questions.md
// "2026-11-05 — the volatility-managed floor: what to watch" names three
// things a session had to work out by hand:
//   1. TRACKING: the floor's realized return against what its target exposure
//      times SPY's return would have delivered, per session and over the
//      rolling 20 sessions.
//   2. The PRE-REGISTERED ROLLBACK TRIGGER (research/floor_study_2026-11-05.md,
//      quoted verbatim in FLOOR_TRACKING_TRIGGER_TEXT), evaluated mechanically.
//      Both legs since 2026-11-22: tracking, and the account's drawdown
//      against the backtest's 95th-percentile 3-month drawdown
//      (FLOOR_DRAWDOWN_LIMIT_PCT; `account_drawdown` in the definitions).
//   3. TURNOVER: the floor's orders and dollars traded per session and over the
//      window, beside the study's pace.
//
// MEASUREMENT ONLY. Pure: no I/O, no network, no order. server/bot.ts gathers
// the inputs (floorTrackingBody), serves the report as the token-gated
// `floor_tracking` diag probe and, once a day right after the daily line,
// writes the report's one-line summary to the audit log (FLOOR-TRACKING).
// Nothing is stored: every call recomputes from the inputs.
//
// INPUTS (existing records only)
//   - the stored daily lines (server/dailyLine.ts, the newest DAILY_LINE_KEEP). Per
//     session: the floor's target (floor.target_x, read from the floor's own
//     state file after the close), the prior close's equity (Alpaca
//     last_equity) and the floor's market value when the line was written;
//   - the broker's orders in the floor's tickers (filled quantity and average
//     fill price per order) and the positions held now;
//   - daily closes, PRICE ONLY (Alpaca bars, adjustment=split, feed iex): the
//     paper account receives no dividends, so both sides are price return;
//   - system_config: FLOOR_MODE, the three tickers, FLOOR_VM_CAPITAL_PCT,
//     FLOOR_VM_BAND.
//
// DEFINITIONS, for a session t whose previous session is p
//   capital          FLOOR_VM_CAPITAL_PCT x the account's equity at the close
//                    of p (the last_equity that line t recorded).
//   target_exposure  E*, the target the floor aimed at during t, in multiples
//                    of its capital: the daily line's target_x (equity terms)
//                    / FLOOR_VM_CAPITAL_PCT. The floor decides it from closes
//                    up to p (never today's bar); it carries the flows
//                    multiplier when one applied.
//   spy_pct          the base ticker's close of t / its close of p - 1.
//   rule_pct         target_exposure x spy_pct: the target exposure held from
//                    close to close, as a return on the capital. It has no
//                    financing above 1x and no yield on idle cash.
//   floor_pct        the core's realized P&L over the session / capital. The
//                    core is the base and the levered ticker. P&L = shares
//                    held at the close of p x (close t - close p), plus, for
//                    every fill in (close p, close t], signed shares x
//                    (close t - fill price). Shares at a past close = shares
//                    held now - the signed fills since.
//   tracking_pp      floor_pct - rule_pct.
//   held_exposure    (base value + 2 x levered value at the close of p) /
//                    capital. The levered ticker is a 2x DAILY fund, so this
//                    holds for one session only: the comparison is made
//                    session by session, then compounded.
//   exposure_pp      (held_exposure - target_exposure) x spy_pct: the part of
//                    tracking_pp that comes from entering the session away
//                    from the target (inside the band, or a rebalance that
//                    waits for 09:45 ET).
//   other_pp         tracking_pp - exposure_pp: fills against the close, the
//                    levered fund's own cost against exactly 2x, rounding.
//   cash_fund_pnl    the same P&L for the T-bill fund the idle cash sits in.
//                    Reported beside the core, NOT inside floor_pct: the
//                    rule's return has no cash yield to set it against.
//   a window         over consecutive sessions, in percentage points:
//                    prod(1 + floor_pct) - prod(1 + rule_pct).
//
// THE ENTRY SESSION. A session the floor enters holding no base or levered
// shares (2026-11-06: its first fill came at 09:47 ET) has no close-to-close
// floor return to compare. It is reported under `entry`, with its shortfall
// against the rule, and is in NO window: the windows measure the floor
// running, not its launch.
//
// BIAS, stated so no number is over-read
//   - Closes on both sides. The rule's return holds the new target from the
//     previous close; the floor can only trade 09:45-15:54 ET. On a rebalance
//     day the overnight gap is taken at the old exposure. That is real
//     tracking and is counted (it sits in exposure_pp); its sign follows the
//     market's.
//   - The band. The floor may sit up to FLOOR_VM_BAND from its target without
//     trading. Against E* that reads as tracking of up to band x SPY's move,
//     in either direction.
//   - No financing in the rule. A levered floor (E* > 1) reads lower than the
//     study's rule would, by the levered fund's financing and fee on the
//     excess (the study charged the T-bill rate + 1% a year on it).
//   - IEX daily bars. The closes come from the IEX feed, which need not print
//     the consolidated close. The rule and the base ticker use the same
//     series; the levered ticker's own series adds noise to other_pp.
//   - Leaving the entry session out makes the first windows read better than a
//     measure that counted it; its shortfall is shown, never dropped.
//   - Shares at a past close are rebuilt from the orders. A change no order
//     explains (a split, a transfer) would misstate them, so each session's
//     rebuilt value is checked against the market value its daily line read,
//     and the session is void when the two differ by more than
//     FLOOR_TRACKING_MV_CHECK_TOL. A difference inside that tolerance passes
//     unseen. Positions and orders are two reads taken together; a fill
//     landing between them would misstate that one answer.
//
// Nothing is inferred: a number whose input is missing is null, with a reason.
import type { DailyLine, PositionLite } from "./dailyLine";
import type { DailyClose } from "./spyBenchmark";
import { zonedInstant, type MarketCalendar } from "./jobHeartbeat";

const ET = "America/New_York";

/** The floor mode this report is defined for (system_config FLOOR_MODE). */
export const FLOOR_TRACKING_MODE = "vol_managed_spy";
/** The pre-registered trigger, verbatim (floorTracking.test.ts pins it to the study's text). */
export const FLOOR_TRACKING_TRIGGER_TEXT =
  "rollback trigger: live floor tracking vs its own rule > 3 pp over any 20 sessions, or account drawdown beyond the backtest's 95th-percentile 3-month drawdown";
export const FLOOR_TRACKING_TRIGGER_SOURCE = "research/floor_study_2026-11-05.md, Pre-registration";
/** "> 3 pp over any 20 sessions": the two numbers of the trigger's tracking leg. Pre-registered, not tunable. */
export const FLOOR_TRACKING_LIMIT_PP = 3;
export const FLOOR_TRACKING_WINDOW_SESSIONS = 20;
/** The trigger's second leg, "the backtest's 95th-percentile 3-month drawdown": of every 63-session window of the
 *  study's V1 backtest (1993-02-01 to 2026-11-04, 8,437 windows), the fall from a high to a later low inside the
 *  window that 95% of the windows stayed above. Computed 2026-11-22 from the study's own code and inputs
 *  (scripts/floor_drawdown_limit.py; research/floor_study_drawdown_limit_2026-11-22.json, where two other readings
 *  of the same words are recorded beside it). test_floor_drawdown_limit.py pins both numbers to that file. Not tunable. */
export const FLOOR_DRAWDOWN_LIMIT_PCT = -14.64;
export const FLOOR_DRAWDOWN_WINDOW_SESSIONS = 63;
export const FLOOR_DRAWDOWN_LIMIT_SOURCE = "research/floor_study_drawdown_limit_2026-11-22.json (V1_volmanaged.p95_pct)";
/** The levered ticker is a 2x daily fund (floor_vol_managed.py LEVERAGE_OF_LEVERED; the test pins the two together). */
export const FLOOR_TRACKING_LEVERED_MULTIPLE = 2;
/** A session whose rebuilt core value is further than this from its daily line's reading is void. The line reads its
 *  value after the close, so the two differ by the after-hours move; a split halves or doubles the rebuilt value, and
 *  a missed rebalance moves it by the band or more (0.10 of capital). */
export const FLOOR_TRACKING_MV_CHECK_TOL = 0.05;
/** The orders read starts this many days before the oldest stored line (floor orders are DAY orders). */
export const FLOOR_TRACKING_ORDERS_LOOKBACK_DAYS = 10;
/** The study's V1 pace (research/floor_study_2026-11-05.json: exposure.V1_rebalances over window.sessions). */
export const FLOOR_TRACKING_STUDY_REBALANCES = 800;
export const FLOOR_TRACKING_STUDY_SESSIONS = 8499;
/** Fills listed per leg (a rebalance is one or two; `fills_count` is always exact). */
export const FLOOR_TRACKING_FILLS_LISTED = 20;

const EPS = 1e-9;
// "+ 0": a rounded negative zero is written as 0
const r2 = (x: number) => Math.round(x * 100) / 100 + 0;
const r4 = (x: number) => Math.round(x * 10_000) / 10_000 + 0;
const r2n = (x: number | null) => (x === null ? null : r2(x));
const r4n = (x: number | null) => (x === null ? null : r4(x));

function rec(x: unknown): Record<string, unknown> {
  return x && typeof x === "object" && !Array.isArray(x) ? (x as Record<string, unknown>) : {};
}

function num(x: unknown): number | null {
  if (x === null || x === undefined || x === "") return null;
  const v = typeof x === "number" ? x : parseFloat(String(x));
  return Number.isFinite(v) ? v : null;
}

function str(x: unknown): string {
  return typeof x === "string" ? x : x === null || x === undefined ? "" : String(x);
}

function msOf(x: unknown): number | null {
  const t = Date.parse(str(x));
  return Number.isFinite(t) ? t : null;
}

// ── inputs ──────────────────────────────────────────────────────────────────

/** What the report needs from system_config (bot.ts reads the FLOOR_* values from Python once per report). */
export interface FloorTrackingConfig {
  mode: string;
  base: string;
  levered: string;
  /** the T-bill fund; null when unset or the same as a core ticker */
  cash: string | null;
  /** FLOOR_VM_CAPITAL_PCT, in (0, 1] */
  capitalPct: number;
  /** FLOOR_VM_BAND; null when unreadable (the band counts are then null) */
  band: number | null;
}

/** system_config's FLOOR_* values -> the config. No value is assumed: a missing mode, ticker or capital share is an
 *  error, never a default. */
export function parseFloorTrackingConfig(raw: unknown): { cfg: FloorTrackingConfig | null; error: string | null } {
  const o = rec(raw);
  if (o.error) return { cfg: null, error: str(o.error).slice(0, 200) };
  const mode = str(o.FLOOR_MODE).trim();
  const base = str(o.FLOOR_VM_TICKER).trim().toUpperCase();
  const levered = str(o.FLOOR_VM_LEVERED_TICKER).trim().toUpperCase();
  const pct = num(o.FLOOR_VM_CAPITAL_PCT);
  if (!mode) return { cfg: null, error: "system_config has no FLOOR_MODE" };
  if (!base || !levered) return { cfg: null, error: "system_config has no FLOOR_VM_TICKER or FLOOR_VM_LEVERED_TICKER" };
  if (base === levered) return { cfg: null, error: `FLOOR_VM_TICKER and FLOOR_VM_LEVERED_TICKER are both ${base}` };
  if (pct === null || !(pct > 0) || pct > 1) return { cfg: null, error: "system_config has no FLOOR_VM_CAPITAL_PCT in (0, 1]" };
  const cash = str(o.FLOOR_VM_CASH_TICKER).trim().toUpperCase();
  const band = num(o.FLOOR_VM_BAND);
  return {
    cfg: { mode, base, levered, cash: cash && cash !== base && cash !== levered ? cash : null, capitalPct: pct, band: band !== null && band >= 0 ? band : null },
    error: null,
  };
}

/** One filled floor order. `at_ms` is its fill time; an order that filled in part and was then cancelled carries no
 *  fill time, so its submission time stands in (a DAY order fills in the session it was sent in). */
export interface FloorFill {
  order_id: string;
  symbol: string;
  dir: 1 | -1;
  qty: number;
  price: number;
  at_ms: number;
  time_basis: "filled_at" | "submitted_at";
}

export interface FloorOrders {
  /** oldest first */
  fills: FloorFill[];
  /** every order sent in the floor's tickers, filled or not */
  submitted: Array<{ symbol: string; at_ms: number }>;
  /** filled orders with no readable time, side or price: past positions cannot be rebuilt past them */
  unplaced: number;
}

/** The broker's order rows -> the floor's fills and submissions. Null when the answer is not a list. */
export function floorOrders(orders: unknown, tickers: ReadonlySet<string>): FloorOrders | null {
  if (!Array.isArray(orders)) return null;
  const out: FloorOrders = { fills: [], submitted: [], unplaced: 0 };
  for (const raw of orders) {
    const o = rec(raw);
    const symbol = str(o.symbol).toUpperCase();
    if (!tickers.has(symbol)) continue;
    const sentMs = msOf(o.submitted_at);
    if (sentMs !== null) out.submitted.push({ symbol, at_ms: sentMs });
    const qty = num(o.filled_qty);
    if (qty === null || !(qty > 0)) continue;
    const price = num(o.filled_avg_price);
    const side = str(o.side).toLowerCase();
    const dir = side === "buy" ? 1 : side.startsWith("sell") ? -1 : 0;
    const filledMs = msOf(o.filled_at);
    const atMs = filledMs ?? sentMs;
    if (dir === 0 || price === null || !(price > 0) || atMs === null) { out.unplaced++; continue; }
    out.fills.push({ order_id: str(o.id), symbol, dir, qty, price, at_ms: atMs, time_basis: filledMs !== null ? "filled_at" : "submitted_at" });
  }
  out.fills.sort((a, b) => a.at_ms - b.at_ms);
  return out;
}

/** Daily closes of one ticker and when they were read (a bar read before its session closed is not a close). */
export interface FloorTrackingBars { bars: readonly DailyClose[]; read_at_ms: number }

export type FloorTrackingInputName = "config" | "lines" | "base_bars" | "levered_bars" | "cash_bars" | "positions" | "orders";

export interface FloorTrackingInputs {
  nowMs: number;
  cfg: FloorTrackingConfig | null;
  /** the stored daily lines, any order */
  lines: readonly DailyLine[];
  bars: { base: FloorTrackingBars | null; levered: FloorTrackingBars | null; cash: FloorTrackingBars | null };
  /** the positions held now (dailyLine.parsePositionsLite); null when unreadable */
  positions: readonly PositionLite[] | null;
  /** the broker's order rows in the floor's tickers, every status; anything but a list is unreadable */
  orders: unknown;
  /** the orders read holds every order sent after this instant */
  ordersFromMs: number | null;
  calendar: MarketCalendar;
  /** read failures the caller saw, by input */
  errors: Partial<Record<FloorTrackingInputName, string>>;
}

// ── the report ──────────────────────────────────────────────────────────────

export type FloorLegRole = "base" | "levered" | "cash_fund";

/** One ticker's part of a session, with everything its P&L is made of. */
export interface FloorTrackingLeg {
  role: FloorLegRole;
  ticker: string;
  shares_prev: number;
  close_prev: number | null;
  close: number | null;
  /** fills in (previous close, this close]: `fills_count` is exact, `fills` lists at most FLOOR_TRACKING_FILLS_LISTED */
  fills_count: number;
  fills: Array<{ side: "buy" | "sell"; qty: number; price: number; at: string; order_id: string }>;
  pnl: number | null;
  reason: string | null;
}

/** orders = sent in the session, any status; filled = orders with a fill in it; dollars = filled shares x average fill price */
export interface FloorTurnoverCell { orders: number; filled: number; dollars: number }

export interface FloorTrackingSession {
  day: string;
  prev_day: string;
  /** a session the floor entered with no core shares (see the header): never in a window */
  entry: boolean;
  spy_pct: number | null;
  capital: number | null;
  target_exposure: number | null;
  held_exposure: number | null;
  /** |target_exposure - held_exposure| */
  drift: number | null;
  rule_pnl: number | null;
  rule_pct: number | null;
  floor_pnl: number | null;
  floor_pct: number | null;
  tracking_pp: number | null;
  exposure_pp: number | null;
  other_pp: number | null;
  cash_fund_pnl: number | null;
  cash_fund_pct: number | null;
  /** the FLOOR_TRACKING_WINDOW_SESSIONS-session window ending here; null until that many sessions in a row have a tracking_pp */
  window_pp: number | null;
  /** the core's value at this close rebuilt from the orders, against the market value the daily line read */
  mv_check: { rebuilt: number; line: number; diff_pct: number; ok: boolean } | null;
  core: FloorTurnoverCell;
  cash_fund: FloorTurnoverCell;
  /** a session other than the entry with at least one filled core order */
  rebalance: boolean;
  legs: FloorTrackingLeg[];
  /** why tracking_pp is null, or a note on a number beside it */
  reason: string | null;
}

export interface FloorTrackingWindow {
  sessions: number;
  from: string | null;
  to: string | null;
  floor_pct: number | null;
  rule_pct: number | null;
  tracking_pp: number | null;
}

export type FloorTriggerLegStatus = "met" | "not_met" | "not_enough_history" | "not_evaluable";

export interface FloorRollbackTrigger {
  text: string;
  source: string;
  /** met when either leg is met; not_met only when BOTH legs are evaluated and inside their limits; otherwise
   *  undetermined (one leg inside its limit does not clear a two-leg trigger) */
  status: "met" | "not_met" | "undetermined";
  summary: string;
  tracking: {
    status: FloorTriggerLegStatus;
    limit_pp: number;
    window_sessions: number;
    /** sessions with a tracking_pp in the stored history */
    sessions_measured: number;
    windows_evaluated: number;
    /** the evaluated window furthest from zero */
    worst: { from: string; to: string; tracking_pp: number } | null;
    /** before the first full window: what the measured sessions add up to, and whether that alone is past the limit */
    so_far: (FloorTrackingWindow & { beyond_limit: boolean }) | null;
    reason: string | null;
  };
  drawdown: FloorDrawdownLeg;
}

/** One closing value of the account in the drawdown leg's series. */
export interface FloorEquityPoint { day: string; equity: number }

export interface FloorDrawdownLeg {
  /** met = the fall is past the limit (a partial window can already be); not_met = a full window, every close read,
   *  inside the limit; not_enough_history = fewer than window_sessions since the floor entered, or a close missing */
  status: FloorTriggerLegStatus;
  limit_pct: number | null;
  limit_source: string;
  window_sessions: number;
  /** sessions the measured closes span: the newest window_sessions since the floor entered, at most */
  sessions_measured: number;
  /** the largest fall from a high to a later low among those closes, in percent of the high (0 or negative) */
  drawdown_pct: number | null;
  peak: FloorEquityPoint | null;
  trough: FloorEquityPoint | null;
  from: string | null;
  to: string | null;
  /** sessions in the span whose closing equity no daily line recorded: a fall through one of them is not seen */
  missing: string[];
  beyond_limit: boolean;
  reason: string | null;
}

export interface FloorTurnoverSummary {
  sessions: number;
  from: string | null;
  to: string | null;
  core: FloorTurnoverCell;
  cash_fund: FloorTurnoverCell;
  /** core dollars traded / capital, summed over the sessions with a capital, in percent */
  core_turnover_pct: number | null;
  rebalance_sessions: number;
  /** the longest run of rebalance sessions on consecutive sessions */
  longest_rebalance_run: number;
  /** rebalance sessions that began within the band of their target, by the previous close's marks: a trade the
   *  intraday marks alone caused. Null when the band is unreadable */
  rebalances_inside_band: number | null;
  /** sessions that began outside the band, by the previous close's marks, and filled no core order */
  outside_band_no_trade: number | null;
}

export interface FloorTurnover {
  basis: string;
  /** the last FLOOR_TRACKING_WINDOW_SESSIONS sessions (fewer while the history is shorter), entry sessions left out */
  window: FloorTurnoverSummary;
  /** every stored session, entry sessions left out */
  stored: FloorTurnoverSummary;
  /** after the last stored session's close (the session in progress): counted nowhere above */
  pending: { core: FloorTurnoverCell; cash_fund: FloorTurnoverCell };
  study: { rebalances: number; sessions: number; per_window: number; source: string };
}

export interface FloorTrackingEntry {
  day: string;
  first_fill_at: string | null;
  capital: number | null;
  target_exposure: number | null;
  floor_pnl: number | null;
  rule_pnl: number | null;
  shortfall_pp: number | null;
  core: FloorTurnoverCell;
  note: string;
}

export interface FloorTrackingReport {
  version: 1;
  generated_at: string;
  status: "ok" | "not_applicable" | "unavailable";
  reason: string | null;
  mode: string | null;
  config: { base: string; levered: string; cash_fund: string | null; capital_pct: number; band: number | null } | null;
  definitions: Record<string, string>;
  inputs: {
    lines: number;
    first_line: string | null;
    last_line: string | null;
    /** stored sessions before the floor held or traded its core */
    sessions_before_floor: number;
    orders: number;
    fills: number;
    bars: Record<FloorLegRole, { first: string; last: string; count: number } | null>;
    errors: Record<string, string>;
  };
  /** the latest entry session in the stored history, or null */
  entry: FloorTrackingEntry | null;
  /** the latest session with a tracking_pp */
  latest: FloorTrackingSession | null;
  /** the last FLOOR_TRACKING_WINDOW_SESSIONS sessions; `missing` are those, other than an entry, without a tracking_pp */
  rolling: FloorTrackingWindow & { window_sessions: number; complete: boolean; missing: string[] };
  /** every measured session in the stored history */
  stored: FloorTrackingWindow;
  rollback_trigger: FloorRollbackTrigger;
  turnover: FloorTurnover;
  /** newest first */
  sessions: FloorTrackingSession[];
  /** the audit row: one compact line */
  message: string;
}

export const FLOOR_TRACKING_DEFINITIONS: Record<string, string> = {
  capital: "FLOOR_VM_CAPITAL_PCT x the account's equity at the previous session's close (Alpaca last_equity, as the session's daily line recorded it)",
  target_exposure: "E*, the floor's target during the session, in multiples of its capital: the daily line's floor.target_x / FLOOR_VM_CAPITAL_PCT",
  held_exposure: "(base value + 2 x levered value at the previous close) / capital",
  spy_pct: "the base ticker's close / its previous close - 1, in percent; price only (Alpaca bars, adjustment=split, feed iex)",
  rule_pct: "target_exposure x spy_pct: the target held from close to close, on the capital; no financing above 1x, no yield on idle cash",
  floor_pct: "realized P&L of the base and levered tickers over the session / capital, in percent: shares at the previous close x the close-to-close move, plus each fill's signed shares x (close - fill price)",
  tracking_pp: "floor_pct - rule_pct, in percentage points of the floor's capital",
  exposure_pp: "(held_exposure - target_exposure) x spy_pct: tracking from entering the session away from the target",
  other_pp: "tracking_pp - exposure_pp: fills against the close, the levered fund's cost against exactly 2x, rounding",
  cash_fund: "the T-bill fund's P&L, measured the same way; reported beside the core and not inside floor_pct",
  window: "over consecutive sessions: prod(1 + floor_pct) - prod(1 + rule_pct), in percentage points",
  entry: "a session the floor entered with no base or levered shares: shown with its shortfall against the rule, in no window",
  account_drawdown: "the trigger's second leg. The account's closing equity, session by session, from the close before the floor's first fill: each session's close is the next daily line's last_equity (Alpaca's own 16:00 ET close), and the newest session's is its own line's close reading. Over the newest 63 sessions of that series: the largest fall from a high to a later low, in percent of the high. The limit is the same measure's 95th percentile over every 63-session window of the study's backtest. Sessions before the floor are not in it: they were another floor's",
  turnover: "orders = sent in the session, any status; filled = orders with a fill in the session; dollars = filled shares x average fill price. Legacy floor tickers are not counted",
};

function closeMap(b: FloorTrackingBars | null): Map<string, number> {
  const m = new Map<string, number>();
  for (const bar of b?.bars ?? []) {
    if (bar && typeof bar.date === "string" && typeof bar.close === "number" && Number.isFinite(bar.close) && bar.close > 0) m.set(bar.date, bar.close);
  }
  return m;
}

const emptyCell = (): FloorTurnoverCell => ({ orders: 0, filled: 0, dollars: 0 });
const roundCell = (c: FloorTurnoverCell): FloorTurnoverCell => ({ orders: c.orders, filled: c.filled, dollars: r2(c.dollars) });
const addCell = (a: FloorTurnoverCell, b: FloorTurnoverCell): FloorTurnoverCell => ({ orders: a.orders + b.orders, filled: a.filled + b.filled, dollars: a.dollars + b.dollars });

/** One session before rounding: what the windows compound. floorRet / ruleRet are null when it has no tracking_pp. */
interface RawSession {
  row: FloorTrackingSession;
  floorRet: number | null;
  ruleRet: number | null;
  capital: number | null;
  coreDollars: number;
  /** an entry session's shortfall against the rule, in percentage points; null on every other session */
  entryShortfall: number | null;
  /** the account's equity at the previous session's close (the line's last_equity); null when the line holds none */
  equityPrev: number | null;
  /** the account's equity at this session's close as its own line read it (equity_end, else equity) */
  equityClose: number | null;
}

function compound(rs: readonly RawSession[]): { floor: number; rule: number } {
  let f = 1, r = 1;
  for (const s of rs) { f *= 1 + (s.floorRet ?? 0); r *= 1 + (s.ruleRet ?? 0); }
  return { floor: f - 1, rule: r - 1 };
}

const NO_WINDOW: FloorTrackingWindow = { sessions: 0, from: null, to: null, floor_pct: null, rule_pct: null, tracking_pp: null };

function windowOf(rs: readonly RawSession[]): FloorTrackingWindow {
  if (!rs.length) return { ...NO_WINDOW };
  const c = compound(rs);
  return {
    sessions: rs.length, from: rs[0].row.day, to: rs[rs.length - 1].row.day,
    floor_pct: r4(c.floor * 100), rule_pct: r4(c.rule * 100), tracking_pp: r4((c.floor - c.rule) * 100),
  };
}

function turnoverSummary(rs: readonly RawSession[], band: number | null): FloorTurnoverSummary {
  const used = rs.filter((s) => !s.row.entry);
  let core = emptyCell(), cash = emptyCell();
  let turnover: number | null = null, rebalances = 0, run = 0, longest = 0;
  let inside: number | null = band === null ? null : 0;
  let outside: number | null = band === null ? null : 0;
  for (const s of used) {
    core = addCell(core, s.row.core);
    cash = addCell(cash, s.row.cash_fund);
    if (s.capital !== null && s.capital > 0) turnover = (turnover ?? 0) + (s.coreDollars / s.capital) * 100;
    if (s.row.rebalance) { rebalances++; run++; longest = Math.max(longest, run); } else run = 0;
    if (band !== null && inside !== null && outside !== null && s.row.drift !== null) {
      if (s.row.rebalance && s.row.drift <= band + EPS) inside++;
      if (!s.row.rebalance && s.row.drift > band + EPS) outside++;
    }
  }
  return {
    sessions: used.length, from: used.length ? used[0].row.day : null, to: used.length ? used[used.length - 1].row.day : null,
    core: roundCell(core), cash_fund: roundCell(cash), core_turnover_pct: r4n(turnover),
    rebalance_sessions: rebalances, longest_rebalance_run: longest, rebalances_inside_band: inside, outside_band_no_trade: outside,
  };
}

function studyPace(): FloorTurnover["study"] {
  return {
    rebalances: FLOOR_TRACKING_STUDY_REBALANCES, sessions: FLOOR_TRACKING_STUDY_SESSIONS,
    per_window: r2((FLOOR_TRACKING_STUDY_REBALANCES / FLOOR_TRACKING_STUDY_SESSIONS) * FLOOR_TRACKING_WINDOW_SESSIONS),
    source: "research/floor_study_2026-11-05.json (exposure.V1_rebalances over window.sessions)",
  };
}

/** The largest fall from a high to a later low in a series of closes. Null with fewer than two closes. */
export function largestFall(points: readonly FloorEquityPoint[]): { pct: number; peak: FloorEquityPoint; trough: FloorEquityPoint } | null {
  if (points.length < 2) return null;
  let peak = points[0];
  let worst = { pct: 0, peak: points[0], trough: points[0] };
  for (const p of points) {
    if (p.equity > peak.equity) peak = p;
    const pct = (p.equity / peak.equity - 1) * 100;
    if (pct < worst.pct) worst = { pct, peak, trough: p };
  }
  return worst;
}

const noDrawdown = (reason: string): FloorDrawdownLeg => ({
  status: "not_evaluable", limit_pct: FLOOR_DRAWDOWN_LIMIT_PCT, limit_source: FLOOR_DRAWDOWN_LIMIT_SOURCE, window_sessions: FLOOR_DRAWDOWN_WINDOW_SESSIONS,
  sessions_measured: 0, drawdown_pct: null, peak: null, trough: null, from: null, to: null, missing: [], beyond_limit: false, reason,
});

/** The trigger's second leg: the account's largest fall from a high to a later low over the newest
 *  FLOOR_DRAWDOWN_WINDOW_SESSIONS sessions since the floor entered, against FLOOR_DRAWDOWN_LIMIT_PCT. A fall already
 *  past the limit is "met" whatever the window's length (a longer window can only hold a larger fall); "not met" is
 *  said only of a full window with every close read. Sessions before the floor's first fill are another floor's and
 *  are not in the series. */
function drawdownLeg(raws: readonly RawSession[]): FloorDrawdownLeg {
  if (!raws.length) return noDrawdown("no session of the floor is stored");
  // the close of every session from the one before the floor's first: session k's close is session k+1's
  // last_equity (Alpaca's own close), else what session k's own line read at its close
  const closes: Array<{ day: string; equity: number | null }> = [{ day: raws[0].row.prev_day, equity: raws[0].equityPrev }];
  for (let k = 0; k < raws.length; k++) {
    const next = k + 1 < raws.length ? raws[k + 1].equityPrev : null;
    closes.push({ day: raws[k].row.day, equity: next ?? raws[k].equityClose });
  }
  const W = FLOOR_DRAWDOWN_WINDOW_SESSIONS;
  const span = closes.slice(-(W + 1));
  const read = span.filter((c): c is FloorEquityPoint => c.equity !== null && c.equity > 0);
  const missing = span.filter((c) => !(c.equity !== null && c.equity > 0)).map((c) => c.day);
  const fall = largestFall(read);
  if (!fall) return { ...noDrawdown("fewer than two closing equities are recorded since the floor entered"), missing };
  const sessions = span.length - 1;
  // judged on the figure the report prints (two decimals), so the status never disagrees with the number beside it
  const pct = r2(fall.pct);
  const beyond = pct < FLOOR_DRAWDOWN_LIMIT_PCT;
  const full = sessions >= W && !missing.length;
  return {
    status: beyond ? "met" : full ? "not_met" : "not_enough_history",
    limit_pct: FLOOR_DRAWDOWN_LIMIT_PCT, limit_source: FLOOR_DRAWDOWN_LIMIT_SOURCE, window_sessions: W, sessions_measured: sessions,
    drawdown_pct: pct,
    peak: { day: fall.peak.day, equity: r2(fall.peak.equity) }, trough: { day: fall.trough.day, equity: r2(fall.trough.equity) },
    from: span[0].day, to: span[span.length - 1].day, missing, beyond_limit: beyond,
    reason: beyond || full ? null
      : missing.length ? `${missing.length} session${missing.length === 1 ? "" : "s"} with no closing equity on record (${missing.slice(0, 3).join(", ")}${missing.length > 3 ? ", ..." : ""}): a fall through ${missing.length === 1 ? "it" : "them"} is not seen`
        : `${sessions} session${sessions === 1 ? "" : "s"} since the floor entered, a window needs ${W}`,
  };
}

const signed = (x: number | null, unit: string) => (x === null ? "n/a" : `${x >= 0 ? "+" : ""}${x.toFixed(2)}${unit}`);

/** The audit row: one compact line (persistAudit keeps 500 chars). */
export function floorTrackingMessage(r: Omit<FloorTrackingReport, "message">): string {
  if (r.status !== "ok") return `floor tracking ${r.status === "not_applicable" ? "not applicable" : "unavailable"}: ${r.reason ?? "unknown"}`.slice(0, 490);
  const t = r.rollback_trigger;
  const parts: string[] = [];
  if (t.status === "met") parts.push(`ROLLBACK TRIGGER MET: ${t.summary}`);
  const s = r.latest;
  parts.push(s
    ? `${s.day} floor ${signed(s.floor_pct, "%")} vs rule ${signed(s.rule_pct, "%")} (${s.target_exposure === null ? "n/a" : `${s.target_exposure.toFixed(2)}x`} of SPY ${signed(s.spy_pct, "%")}): tracking ${signed(s.tracking_pp, "pp")}`
    : `no session measured yet${r.reason ? ` (${r.reason})` : ""}`);
  const w = r.rolling;
  parts.push(`last ${w.sessions} of ${w.window_sessions} sessions ${signed(w.tracking_pp, "pp")}${w.missing.length ? ` (${w.missing.length} unmeasured)` : ""}`);
  if (r.entry) parts.push(`entry ${r.entry.day.slice(5)} ${signed(r.entry.shortfall_pp, "pp")} (in no window)`);
  if (t.status !== "met") parts.push(`rollback trigger ${t.status}: ${t.summary}`);
  const tv = r.turnover.window;
  parts.push(`turnover: ${tv.rebalance_sessions} rebalance session${tv.rebalance_sessions === 1 ? "" : "s"} in ${tv.sessions} (study pace ${r.turnover.study.per_window.toFixed(1)} per ${w.window_sessions}), $${Math.round(tv.core.dollars)} core traded`);
  return parts.join(" | ").slice(0, 490);
}

/** The report with nothing measured: every block present, every number null. */
function shell(i: FloorTrackingInputs, status: FloorTrackingReport["status"], reason: string | null): Omit<FloorTrackingReport, "message"> {
  const errors: Record<string, string> = {};
  for (const [k, v] of Object.entries(i.errors)) if (v) errors[k] = String(v).slice(0, 200);
  const days = i.lines.map((l) => l?.day).filter((d): d is string => typeof d === "string").sort();
  const span = (b: FloorTrackingBars | null) => {
    const ds = [...closeMap(b).keys()].sort();
    return ds.length ? { first: ds[0], last: ds[ds.length - 1], count: ds.length } : null;
  };
  const noTurnover = turnoverSummary([], i.cfg?.band ?? null);
  const why = reason ?? "no session measured";
  return {
    version: 1, generated_at: new Date(i.nowMs).toISOString(), status, reason, mode: i.cfg?.mode ?? null,
    config: i.cfg ? { base: i.cfg.base, levered: i.cfg.levered, cash_fund: i.cfg.cash, capital_pct: i.cfg.capitalPct, band: i.cfg.band } : null,
    definitions: FLOOR_TRACKING_DEFINITIONS,
    inputs: {
      lines: i.lines.length, first_line: days[0] ?? null, last_line: days[days.length - 1] ?? null, sessions_before_floor: 0,
      orders: 0, fills: 0, bars: { base: span(i.bars.base), levered: span(i.bars.levered), cash_fund: span(i.bars.cash) }, errors,
    },
    entry: null, latest: null,
    rolling: { ...NO_WINDOW, window_sessions: FLOOR_TRACKING_WINDOW_SESSIONS, complete: false, missing: [] },
    stored: { ...NO_WINDOW },
    rollback_trigger: {
      text: FLOOR_TRACKING_TRIGGER_TEXT, source: FLOOR_TRACKING_TRIGGER_SOURCE, status: "undetermined", summary: `not evaluated (${why})`,
      tracking: {
        status: "not_evaluable", limit_pp: FLOOR_TRACKING_LIMIT_PP, window_sessions: FLOOR_TRACKING_WINDOW_SESSIONS,
        sessions_measured: 0, windows_evaluated: 0, worst: null, so_far: null, reason: why,
      },
      drawdown: noDrawdown(why),
    },
    turnover: {
      basis: FLOOR_TRACKING_DEFINITIONS.turnover, window: noTurnover, stored: noTurnover,
      pending: { core: emptyCell(), cash_fund: emptyCell() }, study: studyPace(),
    },
    sessions: [],
  };
}

// ── rebuilding a session ────────────────────────────────────────────────────

/** What every session is rebuilt from: the fills, the shares held now, the closes and when each set was read. */
interface Rebuild {
  book: FloorOrders;
  positions: readonly PositionLite[];
  closes: Record<FloorLegRole, Map<string, number>>;
  bars: Record<FloorLegRole, FloorTrackingBars | null>;
  ordersFromMs: number | null;
}

/** Shares of `ticker` held at an instant: the shares held now less the signed fills since. */
function sharesAt(rb: Rebuild, ticker: string, ms: number): number {
  let q = rb.positions.find((p) => p.symbol.toUpperCase() === ticker)?.qty ?? 0;
  for (const f of rb.book.fills) if (f.symbol === ticker && f.at_ms > ms) q -= f.dir * f.qty;
  return Math.abs(q) < EPS ? 0 : q;
}

/** One ticker's part of a session, before rounding (`leg` is the row as printed). */
interface LegRaw {
  leg: FloorTrackingLeg;
  pnl: number | null;
  cell: FloorTurnoverCell;
  sharesPrev: number;
  sharesEnd: number;
  closePrev: number | null;
  closeEnd: number | null;
}

/** A ticker over the session (fromMs, toMs]: shares at the previous close, the fills, the P&L marked to the close. */
function legOf(rb: Rebuild, role: FloorLegRole, ticker: string, prev: string, day: string, fromMs: number, toMs: number): LegRaw {
  const inSession = (ms: number) => ms > fromMs && ms <= toMs;
  const fills = rb.book.fills.filter((f) => f.symbol === ticker && inSession(f.at_ms));
  const sharesPrev = sharesAt(rb, ticker, fromMs), sharesEnd = sharesAt(rb, ticker, toMs);
  const closes = rb.closes[role];
  const readAt = rb.bars[role]?.read_at_ms ?? null;
  const closePrev = closes.get(prev) ?? null;
  // a bar read before its session closed is an intraday price, not a close
  const closeEnd = readAt !== null && readAt > toMs ? closes.get(day) ?? null : null;
  let pnl: number | null = null, reason: string | null = null;
  if (rb.ordersFromMs === null || rb.ordersFromMs >= fromMs) reason = `the orders read does not reach back to the ${prev} close`;
  else if (sharesPrev < 0 || sharesEnd < 0) reason = `${ticker}: the orders do not explain the shares held now (a rebuilt position is negative)`;
  else if (sharesPrev === 0 && !fills.length) pnl = 0;
  else if (!rb.bars[role]) reason = `${ticker} daily closes are unreadable`;
  else if (closeEnd === null) reason = closes.has(day) ? `the ${ticker} bar for ${day} was read before that session closed` : `no ${ticker} close for ${day}`;
  else if (sharesPrev !== 0 && closePrev === null) reason = `no ${ticker} close for ${prev}`;
  else pnl = sharesPrev * (closeEnd - (closePrev ?? closeEnd)) + fills.reduce((s, f) => s + f.dir * f.qty * (closeEnd - f.price), 0);
  return {
    pnl, sharesPrev, sharesEnd, closePrev, closeEnd,
    cell: {
      orders: rb.book.submitted.filter((o) => o.symbol === ticker && inSession(o.at_ms)).length,
      filled: fills.length, dollars: fills.reduce((s, f) => s + f.qty * f.price, 0),
    },
    leg: {
      role, ticker, shares_prev: r4(sharesPrev), close_prev: closePrev, close: closeEnd, fills_count: fills.length,
      fills: fills.slice(0, FLOOR_TRACKING_FILLS_LISTED).map((f) => ({
        side: f.dir === 1 ? "buy" as const : "sell" as const, qty: f.qty, price: f.price, at: new Date(f.at_ms).toISOString(), order_id: f.order_id,
      })),
      pnl: r2n(pnl), reason,
    },
  };
}

/** The core's value at the session's close, rebuilt from the orders, against what the daily line read then (its
 *  write-time marks). Null when the line read no value or a held ticker has no close. */
function mvCheckOf(base: LegRaw, lev: LegRaw, lineMv: number | null): FloorTrackingSession["mv_check"] {
  const priced = (l: LegRaw) => l.sharesEnd === 0 || l.closeEnd !== null;
  if (lineMv === null || !priced(base) || !priced(lev)) return null;
  const rebuilt = base.sharesEnd * (base.closeEnd ?? 0) + lev.sharesEnd * (lev.closeEnd ?? 0);
  const scale = Math.max(Math.abs(lineMv), Math.abs(rebuilt));
  const diff = scale > 0 ? Math.abs(rebuilt - lineMv) / scale : 0;
  return { rebuilt: r2(rebuilt), line: r2(lineMv), diff_pct: r4(diff * 100), ok: diff <= FLOOR_TRACKING_MV_CHECK_TOL };
}

/** One session measured: the legs against what the session's daily line recorded (its target, the prior close's
 *  equity, the floor's market value). `skipped`: a stored line's day the bars have no close for, between `prev` and
 *  `day` — the row would then span two sessions against one session's target, so it is void. */
function measureSession(cfg: FloorTrackingConfig, day: string, prev: string, line: DailyLine | null, base: LegRaw, lev: LegRaw, cash: LegRaw | null, skipped: string | null): RawSession {
  const coreCell = addCell(base.cell, lev.cell);
  const heldNothing = base.sharesPrev === 0 && lev.sharesPrev === 0;
  const entry = heldNothing && coreCell.filled > 0;
  const floorBlock = rec(line?.floor);
  const lineMode = str(floorBlock.mode);
  const vmLine = line !== null && lineMode === FLOOR_TRACKING_MODE;
  const lastEquity = num(rec(rec(line?.account_vs_spy).day).last_equity);
  const capital = lastEquity !== null && lastEquity > 0 ? cfg.capitalPct * lastEquity : null;
  const dayPair = rec(rec(line?.account_vs_spy).day);
  const closeEquity = num(dayPair.equity_end) ?? num(dayPair.equity);
  const targetX = vmLine ? num(floorBlock.target_x) : null;
  const target = targetX !== null && targetX >= 0 ? targetX / cfg.capitalPct : null;
  const spyRet = base.closeEnd !== null && base.closePrev !== null ? base.closeEnd / base.closePrev - 1 : null;
  const corePnl = base.pnl !== null && lev.pnl !== null ? base.pnl + lev.pnl : null;
  const pricedPrev = (l: LegRaw) => l.sharesPrev === 0 || l.closePrev !== null;
  const heldValue = pricedPrev(base) && pricedPrev(lev)
    ? base.sharesPrev * (base.closePrev ?? 0) + FLOOR_TRACKING_LEVERED_MULTIPLE * lev.sharesPrev * (lev.closePrev ?? 0) : null;
  const held = capital !== null && heldValue !== null ? heldValue / capital : null;
  const mvCheck = vmLine ? mvCheckOf(base, lev, num(floorBlock.floor_mv)) : null;

  // every reason the session cannot be measured; none = sound
  const reasons: string[] = [];
  if (skipped) reasons.push(`the ${cfg.base} closes read skip the ${skipped} session, so ${prev} is not the close before ${day}`);
  if (!line) reasons.push("no daily line is stored for this session (no recorded target or prior-close equity)");
  else {
    if (!vmLine) reasons.push(`the floor's mode on this session was ${lineMode ? `"${lineMode}"` : "unreadable"}`);
    else if (target === null) reasons.push("the daily line recorded no floor target (the floor's state was stale or held no plan)");
    if (capital === null) reasons.push("the daily line recorded no prior-close equity");
  }
  if (base.leg.reason) reasons.push(base.leg.reason);
  if (lev.leg.reason && lev.leg.reason !== base.leg.reason) reasons.push(lev.leg.reason);
  if (spyRet === null && !base.leg.reason) {
    reasons.push(base.closeEnd === null ? `the ${cfg.base} bar for ${day} was read before that session closed` : `no ${cfg.base} close for ${prev}`);
  }
  if (mvCheck && !mvCheck.ok) {
    reasons.push(`the core's value at the ${day} close rebuilt from the orders ($${mvCheck.rebuilt.toFixed(0)}) is not the daily line's reading ($${mvCheck.line.toFixed(0)})`);
  }
  const sound = reasons.length === 0;
  const rulePnl = capital !== null && target !== null && spyRet !== null ? target * capital * spyRet : null;
  const floorRet = sound && corePnl !== null && capital !== null ? corePnl / capital : null;
  const ruleRet = sound && target !== null && spyRet !== null ? target * spyRet : null;
  const measured = !entry && floorRet !== null && ruleRet !== null;
  const tracking = measured ? (floorRet - ruleRet) * 100 : null;
  const exposurePp = measured && held !== null && target !== null && spyRet !== null ? (held - target) * spyRet * 100 : null;
  if (entry) reasons.unshift("entry session: the floor held no base or levered shares at the previous close, so it has no close-to-close return (see `entry`)");
  if (cash?.leg.reason) reasons.push(`cash fund: ${cash.leg.reason}`);
  const cashPnl = cash ? cash.pnl : null;
  return {
    floorRet: measured ? floorRet : null, ruleRet: measured ? ruleRet : null, capital, coreDollars: coreCell.dollars,
    // the entry's shortfall against the rule: its P&L from the fills to the close, less the target held from the previous close
    entryShortfall: entry && sound && corePnl !== null && rulePnl !== null && capital !== null ? ((corePnl - rulePnl) / capital) * 100 : null,
    equityPrev: lastEquity !== null && lastEquity > 0 ? lastEquity : null,
    equityClose: closeEquity !== null && closeEquity > 0 ? closeEquity : null,
    row: {
      day, prev_day: prev, entry,
      spy_pct: r4n(spyRet === null ? null : spyRet * 100),
      capital: r2n(capital), target_exposure: r4n(target), held_exposure: r4n(held),
      drift: target !== null && held !== null ? r4(Math.abs(target - held)) : null,
      rule_pnl: r2n(rulePnl), rule_pct: r4n(rulePnl !== null && capital !== null ? (rulePnl / capital) * 100 : null),
      floor_pnl: r2n(corePnl), floor_pct: r4n(corePnl !== null && capital !== null ? (corePnl / capital) * 100 : null),
      tracking_pp: r4n(tracking), exposure_pp: r4n(exposurePp),
      other_pp: tracking !== null && exposurePp !== null ? r4(tracking - exposurePp) : null,
      cash_fund_pnl: r2n(cashPnl), cash_fund_pct: cashPnl !== null && capital !== null ? r4((cashPnl / capital) * 100) : null,
      window_pp: null, mv_check: mvCheck,
      core: roundCell(coreCell), cash_fund: roundCell(cash ? cash.cell : emptyCell()),
      rebalance: !entry && coreCell.filled > 0,
      legs: [base.leg, lev.leg, ...(cash ? [cash.leg] : [])],
      reason: reasons.length ? reasons.join("; ") : null,
    },
  };
}

// ── the trigger ─────────────────────────────────────────────────────────────

/** The tracking leg: every run of FLOOR_TRACKING_WINDOW_SESSIONS consecutive measured sessions is a window (its
 *  figure is written on the row that ends it); the leg is met when any window is more than FLOOR_TRACKING_LIMIT_PP
 *  from zero, on either side. */
function trackingLeg(raws: readonly RawSession[]): FloorRollbackTrigger["tracking"] {
  const W = FLOOR_TRACKING_WINDOW_SESSIONS;
  let worst: { from: string; to: string; tracking_pp: number } | null = null;
  let windows = 0, met = false;
  for (let k = W - 1; k < raws.length; k++) {
    const win = raws.slice(k - W + 1, k + 1);
    if (win.some((s) => s.floorRet === null)) continue;
    const c = compound(win);
    // judged on the figure the report prints (four decimals), so the status never disagrees with the number beside it
    const pp = r4((c.floor - c.rule) * 100);
    raws[k].row.window_pp = pp;
    windows++;
    if (Math.abs(pp) > FLOOR_TRACKING_LIMIT_PP) met = true;
    if (!worst || Math.abs(pp) > Math.abs(worst.tracking_pp)) worst = { from: win[0].row.day, to: win[W - 1].row.day, tracking_pp: pp };
  }
  const measured = raws.filter((s) => s.floorRet !== null);
  const soFar = windowOf(measured.slice(-W));
  return {
    status: met ? "met" : windows > 0 ? "not_met" : measured.length < W ? "not_enough_history" : "not_evaluable",
    limit_pp: FLOOR_TRACKING_LIMIT_PP, window_sessions: W, sessions_measured: measured.length, windows_evaluated: windows, worst,
    so_far: windows === 0 && measured.length
      ? { ...soFar, beyond_limit: soFar.tracking_pp !== null && Math.abs(soFar.tracking_pp) > FLOOR_TRACKING_LIMIT_PP } : null,
    reason: windows > 0 ? null
      : measured.length < W ? `${measured.length} measured session${measured.length === 1 ? "" : "s"}, a window needs ${W} in a row`
        : `no ${W} sessions in a row all have a tracking number`,
  };
}

/** The drawdown leg in a few words (the audit row is 490 characters for the whole report). */
function drawdownWords(d: FloorDrawdownLeg): string {
  if (d.drawdown_pct === null) return `drawdown not evaluated (${d.reason ?? "no closes"})`;
  const pct = `${d.drawdown_pct.toFixed(2)}%`, limit = `${(d.limit_pct ?? 0).toFixed(2)}%`;
  if (d.status === "met") return `account drawdown ${pct} BEYOND ${limit} (${d.peak?.day} to ${d.trough?.day})`;
  if (d.status === "not_met") return `drawdown ${pct} within ${limit} over ${d.window_sessions} sessions`;
  return `drawdown ${pct} in ${d.sessions_measured} of ${d.window_sessions} sessions (limit ${limit}${d.missing.length ? `; ${d.missing.length} unread` : ""})`;
}

function triggerOf(tracking: FloorRollbackTrigger["tracking"], drawdown: FloorDrawdownLeg): FloorRollbackTrigger {
  const { worst, so_far: soFar } = tracking;
  const limitTxt = `${tracking.limit_pp} pp over ${tracking.window_sessions} sessions`;
  const soFarTxt = soFar ? `; ${signed(soFar.tracking_pp, "pp")} so far${soFar.beyond_limit ? ` (already past ${tracking.limit_pp} pp)` : ""}` : "";
  const trackingTxt = worst
    ? `tracking ${tracking.status === "met" ? "BEYOND" : "within"} ${limitTxt} (worst ${signed(worst.tracking_pp, "pp")}, ${worst.from} to ${worst.to}; ${tracking.windows_evaluated} window${tracking.windows_evaluated === 1 ? "" : "s"})`
    : `tracking: ${tracking.reason}${soFarTxt}`;
  return {
    text: FLOOR_TRACKING_TRIGGER_TEXT, source: FLOOR_TRACKING_TRIGGER_SOURCE,
    status: tracking.status === "met" || drawdown.status === "met" ? "met" : tracking.status === "not_met" && drawdown.status === "not_met" ? "not_met" : "undetermined",
    summary: `${trackingTxt}; ${drawdownWords(drawdown)}`,
    tracking, drawdown,
  };
}

// ── the report ──────────────────────────────────────────────────────────────

function finish(body: Omit<FloorTrackingReport, "message">): FloorTrackingReport {
  return { ...body, message: floorTrackingMessage(body) };
}

/** The report. Pure: the same inputs give the same answer. */
export function floorTracking(i: FloorTrackingInputs): FloorTrackingReport {
  const cfg = i.cfg;
  if (!cfg) return finish(shell(i, "unavailable", `the floor's config is unreadable (${i.errors.config ?? "system_config"})`));
  if (cfg.mode !== FLOOR_TRACKING_MODE) {
    return finish(shell(i, "not_applicable", `FLOOR_MODE is "${cfg.mode}": this report is defined for "${FLOOR_TRACKING_MODE}" only`));
  }
  if (i.errors.lines) return finish(shell(i, "unavailable", `the daily lines are unreadable (${i.errors.lines})`));
  const lineByDay = new Map<string, DailyLine>();
  for (const l of i.lines) if (l && typeof l.day === "string") lineByDay.set(l.day, l);
  const lineDays = [...lineByDay.keys()].sort();
  if (!lineDays.length) return finish(shell(i, "unavailable", "no daily line is stored yet (the first is written after the next close)"));
  const firstLine = lineDays[0], lastLine = lineDays[lineDays.length - 1];
  if (!i.bars.base) return finish(shell(i, "unavailable", `${cfg.base} daily closes are unreadable (${i.errors.base_bars ?? "no bars"})`));
  const closes: Record<FloorLegRole, Map<string, number>> = { base: closeMap(i.bars.base), levered: closeMap(i.bars.levered), cash_fund: closeMap(i.bars.cash) };
  // the sessions are the base ticker's bars: it trades every session
  const sessionDays = [...closes.base.keys()].sort();
  if (!sessionDays.length || sessionDays[0] >= firstLine) {
    return finish(shell(i, "unavailable", `the ${cfg.base} closes read do not reach back before the first stored line (${firstLine})`));
  }
  if (!i.positions) return finish(shell(i, "unavailable", `positions are unreadable (${i.errors.positions ?? "the answer was not a list"})`));
  const book = floorOrders(i.orders, new Set([cfg.base, cfg.levered, ...(cfg.cash ? [cfg.cash] : [])]));
  if (!book) return finish(shell(i, "unavailable", `the floor's orders are unreadable (${i.errors.orders ?? "the answer was not a list"})`));
  if (book.unplaced) {
    return finish(shell(i, "unavailable", `${book.unplaced} filled floor order(s) carry no readable time, side or price: past positions cannot be rebuilt`));
  }

  const body = shell(i, "ok", null);
  body.inputs.orders = book.submitted.length;
  body.inputs.fills = book.fills.length;
  const noBar = lineDays.filter((d) => !closes.base.has(d));
  if (noBar.length) body.inputs.errors.lines_without_bar = `no ${cfg.base} bar for the stored line(s) of ${noBar.join(", ")}: not measured`.slice(0, 200);

  const rb: Rebuild = {
    book, positions: i.positions, closes, ordersFromMs: i.ordersFromMs,
    bars: { base: i.bars.base, levered: i.bars.levered, cash_fund: i.bars.cash },
  };
  // a session's close: its daily line's own (13:00 on a half day), else the calendar's
  const closeMs = (day: string): number => {
    const m = /^(\d{1,2}):(\d{2})$/.exec(str(lineByDay.get(day)?.session_close));
    return zonedInstant(day, m ? Number(m[1]) * 60 + Number(m[2]) : i.calendar.halfDays.has(day) ? 13 * 60 : 16 * 60, ET);
  };

  const raws: RawSession[] = [];
  for (let k = 1; k < sessionDays.length; k++) {
    const day = sessionDays[k], prev = sessionDays[k - 1];
    if (day < firstLine || day > lastLine) continue;
    const fromMs = closeMs(prev), toMs = closeMs(day);
    const base = legOf(rb, "base", cfg.base, prev, day, fromMs, toMs);
    const lev = legOf(rb, "levered", cfg.levered, prev, day, fromMs, toMs);
    // the stored sessions before the floor first held or traded its core are not the floor's
    if (!raws.length && base.sharesPrev === 0 && lev.sharesPrev === 0 && base.cell.filled + lev.cell.filled === 0) { body.inputs.sessions_before_floor++; continue; }
    const cash = cfg.cash ? legOf(rb, "cash_fund", cfg.cash, prev, day, fromMs, toMs) : null;
    const s = measureSession(cfg, day, prev, lineByDay.get(day) ?? null, base, lev, cash, noBar.find((d) => d > prev && d < day) ?? null);
    raws.push(s);
    if (s.row.entry) {
      const first = book.fills.find((f) => (f.symbol === cfg.base || f.symbol === cfg.levered) && f.at_ms > fromMs && f.at_ms <= toMs) ?? null;
      body.entry = {
        day, first_fill_at: first ? new Date(first.at_ms).toISOString() : null, capital: s.row.capital, target_exposure: s.row.target_exposure,
        floor_pnl: s.row.floor_pnl, rule_pnl: s.row.rule_pnl, shortfall_pp: r4n(s.entryShortfall), core: s.row.core,
        note: "floor_pnl runs from the fills to the close; rule_pnl is the target held from the previous close. The gap is what entering after the open cost or earned: shown once, in no window",
      };
    }
  }
  if (!raws.length) body.reason = `the floor held no ${cfg.base} or ${cfg.levered} and filled no order in the stored sessions (${firstLine} to ${lastLine})`;

  body.rollback_trigger = triggerOf(trackingLeg(raws), drawdownLeg(raws));
  const W = FLOOR_TRACKING_WINDOW_SESSIONS;
  const measured = raws.filter((s) => s.floorRet !== null);
  const tail = raws.slice(-W);
  const tailMeasured = tail.filter((s) => s.floorRet !== null);
  body.latest = measured.length ? measured[measured.length - 1].row : null;
  body.rolling = {
    ...windowOf(tailMeasured), window_sessions: W, complete: tail.length === W && tailMeasured.length === W,
    missing: tail.filter((s) => s.floorRet === null && !s.row.entry).map((s) => s.row.day),
  };
  body.stored = windowOf(measured);

  // fills after the last stored session's close belong to the session in progress
  const lastClose = raws.length ? closeMs(raws[raws.length - 1].row.day) : null;
  const pending = (tickers: ReadonlyArray<string | null>): FloorTurnoverCell => {
    if (lastClose === null) return emptyCell();
    const fills = book.fills.filter((f) => tickers.includes(f.symbol) && f.at_ms > lastClose);
    return roundCell({
      orders: book.submitted.filter((o) => tickers.includes(o.symbol) && o.at_ms > lastClose).length,
      filled: fills.length, dollars: fills.reduce((sum, f) => sum + f.qty * f.price, 0),
    });
  };
  body.turnover = {
    basis: FLOOR_TRACKING_DEFINITIONS.turnover, window: turnoverSummary(tail, cfg.band), stored: turnoverSummary(raws, cfg.band),
    pending: { core: pending([cfg.base, cfg.levered]), cash_fund: pending([cfg.cash]) }, study: studyPace(),
  };
  body.sessions = raws.map((s) => s.row).reverse();
  return finish(body);
}
