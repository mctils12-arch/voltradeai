// dailyLine.ts — ONE line a day that says how the account did and why.
// Human-directed 2026-11-05: "Add a daily line to the logs/report: account vs
// SPY (same-basis), floor exposure (x SPY), option P&L split into
// fills/slippage vs direction, and number of option exits blocked by the
// 9:30–9:45 window."
//
// MEASUREMENT ONLY: nothing here, and nothing in its bot.ts wiring, changes an
// order, a price, a size or a decision. server/bot.ts runs the job once per
// trading day after the close (first 5-minute tick at/after the broker
// calendar's close + DAILY_LINE_DELAY_MIN), gathers the inputs, and calls
// buildDailyLine(). The line goes to the audit log as ONE "DAILY-LINE" row
// (its `message`) and, structured, to DATA_DIR/voltrade_daily_line.json (the
// last DAILY_LINE_KEEP lines), which the `daily_line` diag probe serves. The
// file is also the once-a-day record: its newest line's day is the last day
// written, so a restart neither writes a day twice nor skips one.
//
//   1. ACCOUNT vs SPY, SAME BASIS. The paper account receives no dividends,
//      so SPY is PRICE-ONLY here (Alpaca bars, adjustment=split, feed iex).
//      Day: equity vs last_equity (evaluateDailyPnl, the daily-loss halt's own
//      number) against today's SPY bar vs the prior close (spyToday). Since
//      2026-05-06 and the last 30 days: benchmarkVsSpy over the equity curve
//      with today's point taken from the same account read, on the same
//      split-adjusted bars. No SPY bar for today = no windows (a window
//      ending on yesterday's SPY close would not be the same basis).
//   2. FLOOR EXPOSURE in SPY terms: (SPY + 2 x SSO market value) / equity
//      under FLOOR_MODE "vol_managed_spy" (SSO is a 2x fund); under "basket",
//      floor market value / equity, labelled as such. The mode comes from
//      Python's system_config (bot.ts asks it once per line); the target is
//      the vol-managed floor's own state file (target_exposure x capital /
//      equity), shown only when that state is fresh.
//   3. OPTION P&L = FILLS/SLIPPAGE + DIRECTION over the window since the
//      previous line (same session after a normal day; it says so when it
//      spans more). TOTAL = realized option P&L of the window (the sleeve
//      ledger's closed-lot walk) + unrealized now - unrealized at the previous
//      line, plus, for a contract that left the book with no fill (expiry,
//      assignment, exercise), its value going to zero in the options sleeve.
//      FILLS = -Σ (fill - mid at submission) x qty x 100 (a buy above the mid
//      or a sell below it is a cost) over the window's filled option orders,
//      joined by order id (and leg symbol for mleg) to the option fill ledger
//      the order paths write (option_fill_ledger.py; bot.ts). A fill with no
//      ledger row, a null mid, or an mleg without per-leg fills is
//      UNATTRIBUTED — counted and reported, never given a guessed mid; its
//      slippage then sits inside DIRECTION = TOTAL - FILLS.
//   4. OPTION EXITS HELD BY THE 09:30–09:45 WINDOW: options_manager's per-day
//      set of (contract, rule) deferrals (voltrade_option_opening_deferrals
//      .json) + the day's POS-KILL-OPTION-OPENING audit rows (one per ticker
//      per day by construction).
// Nothing is inferred: a piece whose input is missing is null with a reason.
import fs from "node:fs";
import path from "node:path";
import { evaluateDailyPnl } from "./drawdownGuard";
import { benchmarkVsSpy, type BenchWindow, type CurvePoint, type DailyClose } from "./spyBenchmark";
import { spyToday } from "./todayTile";
import { zonedParts, zonedInstant } from "./jobHeartbeat";
import { isOptionPosition } from "./reconstructPnl";
import type { RealizedEvent } from "./sleeveLedger";

const ET = "America/New_York";
const DAY_MS = 86_400_000;

/** Minutes after the session close before the line is due: ETF options (SPY,
 *  QQQ, IWM) print until 16:15, and a DAY order on them can fill until then. */
export const DAILY_LINE_DELAY_MIN = 20;
/** Minutes after the close past which a line is written even when an input is
 *  still unreadable (before that the job retries every tick for a whole line). */
export const DAILY_LINE_PARTIAL_AFTER_MIN = 90;
/** The earliest a line can be due: a 13:00 half-day close + the delay. */
export const DAILY_LINE_EARLIEST_MIN = 13 * 60 + DAILY_LINE_DELAY_MIN;
export const DAILY_LINE_CHECK_MS = 5 * 60_000;
export const DAILY_LINE_KEEP = 60;
/** The human's "since" date: the equity curve's first day. */
export const DAILY_LINE_SINCE = "2026-05-06";
/** A previous line older than this does not open the option window (the
 *  orders and walk reads would grow without bound); the total is then null. */
export const DAILY_LINE_OPTION_WINDOW_MAX_DAYS = 7;
/** The vol-managed floor's state is rewritten every scan; older than this, its target is not shown. */
export const DAILY_LINE_VM_STATE_MAX_AGE_MS = 6 * 3_600_000;
export const DAILY_LINE_WORST_FILLS = 15;
export const DAILY_LINE_FILENAME = "voltrade_daily_line.json";
export const OPTION_FILL_LEDGER_FILENAME = "voltrade_option_fill_ledger.jsonl";
export const OPTION_FILL_LEDGER_MAX_BYTES = 2_000_000;
export const OPENING_DEFERRALS_FILENAME = "voltrade_option_opening_deferrals.json";
export const FLOOR_VM_STATE_FILENAME = "voltrade_floor_vm_state.json";

const r2 = (x: number) => Math.round(x * 100) / 100;
const r4 = (x: number) => Math.round(x * 10_000) / 10_000;

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

// ── 0. the once-a-day gate ──────────────────────────────────────────────────

/** The line reads the benchmark's shared SPY cache (server/bot.ts spyBenchBars: one fetch, one
 *  cache) but no older than this: the line is due from the close + 20 min, so a cache this young
 *  was fetched after the close and holds the day's final bar. */
export const DAILY_LINE_SPY_MAX_AGE_MS = 5 * 60_000;

export interface DailyLineSession { date: string; closeMin: number }
export interface DailyLineGate { due: boolean; partialOk: boolean; day: string; reason: string }

/** Alpaca's calendar rows -> the session on `day`, or null when the market
 *  was not open that day. Throws on an unreadable answer (the caller retries). */
export function parseCalendarSession(raw: unknown, day: string): DailyLineSession | null {
  if (!Array.isArray(raw)) throw new Error("calendar answer was not a list");
  for (const row of raw) {
    const o = rec(row);
    if (str(o.date) !== day) continue;
    const m = /^(\d{1,2}):(\d{2})$/.exec(str(o.close));
    if (!m) throw new Error(`calendar close unreadable for ${day}: ${str(o.close)}`);
    return { date: day, closeMin: Number(m[1]) * 60 + Number(m[2]) };
  }
  return null;
}

/** Cheap pre-check before the broker calendar is asked: a weekday, past the
 *  earliest possible due time, and not written yet. */
export function dailyLinePrecheck(nowMs: number, lastWritten: string | null): DailyLineGate {
  const p = zonedParts(nowMs, ET);
  const no = (reason: string): DailyLineGate => ({ due: false, partialOk: false, day: p.date, reason });
  if (p.weekday === "Sat" || p.weekday === "Sun") return no(`${p.date} is a weekend`);
  if (lastWritten === p.date) return no(`already written for ${p.date}`);
  if (p.minutes < DAILY_LINE_EARLIEST_MIN) return no("before 13:20 ET (the earliest close + 20 min)");
  return { due: true, partialOk: false, day: p.date, reason: "ask the broker calendar" };
}

/** Is the line due now? `session`: the broker calendar's session for today's
 *  ET date (null = the market was not open). Once per ET day, from the close
 *  + DAILY_LINE_DELAY_MIN to midnight ET; partialOk past the close +
 *  DAILY_LINE_PARTIAL_AFTER_MIN. A process down all evening skips that day. */
export function dailyLineDue(nowMs: number, session: DailyLineSession | null, lastWritten: string | null): DailyLineGate {
  const p = zonedParts(nowMs, ET);
  const no = (reason: string): DailyLineGate => ({ due: false, partialOk: false, day: p.date, reason });
  if (lastWritten === p.date) return no(`already written for ${p.date}`);
  if (!session || session.date !== p.date) return no(`the market was not open on ${p.date}`);
  const dueMin = session.closeMin + DAILY_LINE_DELAY_MIN;
  if (p.minutes < dueMin) return no(`due at ${hhmm(dueMin)} ET`);
  return { due: true, partialOk: p.minutes >= session.closeMin + DAILY_LINE_PARTIAL_AFTER_MIN, day: p.date, reason: "due" };
}

function hhmm(min: number): string {
  return `${String(Math.floor(min / 60)).padStart(2, "0")}:${String(min % 60).padStart(2, "0")}`;
}

// ── 1. account vs SPY, same basis ───────────────────────────────────────────

export interface DayPair {
  account_pct: number | null;
  spy_pct: number | null;
  gap_pp: number | null;
  equity: number | null;
  last_equity: number | null;
  spy_bar_date: string | null;
  reason: string | null;
}
export interface SameBasisBlock {
  basis: string;
  day: DayPair;
  since: BenchWindow | null;
  d30: BenchWindow | null;
  since_requested: string;
  windows_reason: string | null;
}

export const SAME_BASIS_TEXT =
  "account: Alpaca equity vs last_equity (no dividends in the paper account) · SPY: PRICE-ONLY daily closes (Alpaca bars, adjustment=split, feed iex)";

export function sameBasis(i: {
  equity: unknown; lastEquity: unknown; spyBars: readonly DailyClose[] | null;
  curve: readonly CurvePoint[]; day: string; since: string;
}): SameBasisBlock {
  const ev = evaluateDailyPnl(i.equity, i.lastEquity);
  const equity = ev.valid ? num(i.equity) : null;
  const accountPct = ev.valid && ev.dailyPnlPct !== null ? r2(ev.dailyPnlPct) : null;
  const spy = spyToday(i.spyBars, null, i.day);
  const reasons = [ev.valid ? null : "account read invalid (equity or last_equity not a positive number)", spy.reason].filter(Boolean);
  const day: DayPair = {
    account_pct: accountPct,
    spy_pct: spy.pct,
    // from the two ROUNDED sides, so the three printed numbers always add up
    gap_pp: accountPct !== null && spy.pct !== null ? r2(accountPct - spy.pct) : null,
    equity,
    last_equity: ev.valid ? num(i.lastEquity) : null,
    spy_bar_date: spy.barDate,
    reason: reasons.length ? reasons.join("; ") : null,
  };
  let since: BenchWindow | null = null;
  let d30: BenchWindow | null = null;
  let windowsReason: string | null = null;
  if (equity === null) windowsReason = "no valid account read";
  else if (!i.spyBars || spy.barDate !== i.day) windowsReason = spy.reason ?? `no SPY bar for ${i.day}`;
  else {
    const pts: CurvePoint[] = i.curve
      .filter((p) => p && typeof p.date === "string" && p.date >= i.since && p.date < i.day && Number.isFinite(p.value) && p.value > 0)
      .map((p) => ({ date: p.date, value: p.value }));
    pts.push({ date: i.day, value: equity });
    const b = benchmarkVsSpy(pts, [...i.spyBars]);
    since = b.since;
    d30 = b.d30;
    if (!since) windowsReason = `no equity-curve point on or after ${i.since}`;
    else if (!d30) windowsReason = "the equity curve has no point 30 days back";
  }
  return { basis: SAME_BASIS_TEXT, day, since, d30, since_requested: i.since, windows_reason: windowsReason };
}

// ── 2. floor exposure ───────────────────────────────────────────────────────

/** What bot.ts learns from Python's system_config once per line. */
export interface FloorConfig { mode: string; base: string; levered: string; owned: string[]; legacy: string[] }

export function parseFloorConfig(raw: unknown): FloorConfig | null {
  const o = rec(raw);
  const mode = str(o.mode);
  if (!mode || o.error) return null;
  const list = (x: unknown) => (Array.isArray(x) ? x.map(str).filter(Boolean) : []);
  return { mode, base: str(o.base) || "SPY", levered: str(o.levered) || "SSO", owned: list(o.owned), legacy: list(o.legacy) };
}

/** One /v2/positions row, the fields the line reads. */
export interface PositionLite { symbol: string; qty: number; marketValue: number | null; unrealizedPl: number | null; option: boolean }

/** /v2/positions -> rows; null when the answer is not a list (never "no positions"). */
export function parsePositionsLite(raw: unknown): PositionLite[] | null {
  if (!Array.isArray(raw)) return null;
  const out: PositionLite[] = [];
  for (const p of raw) {
    const o = rec(p);
    const symbol = str(o.symbol);
    const q = num(o.qty);
    if (!symbol || q === null || q === 0) continue;
    const qty = o.side === "short" ? -Math.abs(q) : o.side === "long" ? Math.abs(q) : q;
    out.push({ symbol, qty, marketValue: num(o.market_value), unrealizedPl: num(o.unrealized_pl), option: isOptionPosition(symbol, str(o.asset_class)) });
  }
  return out;
}

export interface FloorBlock {
  mode: string | null;
  exposure_x: number | null;
  basis: string | null;
  base_mv: number | null;
  levered_mv: number | null;
  floor_mv: number | null;
  legacy_mv: number | null;
  target_x: number | null;
  target_at: string | null;
  reason: string | null;
}

export function floorExposure(i: { positions: PositionLite[] | null; equity: number | null; cfg: FloorConfig | null; vmState: unknown; nowMs: number }): FloorBlock {
  const out: FloorBlock = { mode: i.cfg?.mode ?? null, exposure_x: null, basis: null, base_mv: null, levered_mv: null, floor_mv: null, legacy_mv: null, target_x: null, target_at: null, reason: null };
  if (!i.cfg) return { ...out, reason: "FLOOR_MODE unreadable (system_config)" };
  if (!i.positions) return { ...out, reason: "positions unreadable" };
  if (!(i.equity !== null && i.equity > 0)) return { ...out, reason: "no valid equity" };
  const bySym = new Map(i.positions.map((p) => [p.symbol, p] as const));
  let unpriced = "";
  // `counts`: the ticker enters the exposure, so a held row with no market
  // value voids it; a legacy ticker is only reported
  const mv = (sym: string, counts = true): number => {
    const p = bySym.get(sym);
    if (!p) return 0;
    if (p.marketValue === null && counts) unpriced = sym;
    return p.marketValue ?? 0;
  };
  const legacy = i.cfg.legacy.reduce((s, t) => s + mv(t, false), 0);
  if (i.cfg.mode === "vol_managed_spy") {
    const base = mv(i.cfg.base);
    const lev = mv(i.cfg.levered);
    Object.assign(out, {
      basis: `(${i.cfg.base} + 2 x ${i.cfg.levered} market value) / equity`,
      base_mv: r2(base), levered_mv: r2(lev), floor_mv: r2(base + lev), legacy_mv: r2(legacy),
      exposure_x: r4((base + 2 * lev) / i.equity),
    });
    const s = rec(i.vmState);
    const at = str(s.at);
    const atMs = msOf(at.slice(0, 23));
    const te = num(s.target_exposure), cap = num(s.capital), eq = num(s.equity);
    if (s.ok === true && te !== null && cap !== null && eq !== null && eq > 0 && atMs !== null && Math.abs(i.nowMs - atMs) <= DAILY_LINE_VM_STATE_MAX_AGE_MS) {
      out.target_x = r4((te * cap) / eq);
      out.target_at = at;
    }
  } else {
    const owned = i.cfg.owned.reduce((s, t) => s + mv(t), 0);
    Object.assign(out, {
      basis: `floor market value / equity (${i.cfg.mode} mode: ${i.cfg.owned.join(", ") || "no floor tickers"}; not SPY-equivalent)`,
      floor_mv: r2(owned), exposure_x: r4(owned / i.equity),
    });
  }
  if (unpriced) return { ...out, exposure_x: null, reason: `${unpriced} held with no market value` };
  return out;
}

// ── 3. option P&L: fills/slippage vs direction ──────────────────────────────

/** One fill-ledger row (option_fill_ledger.py / bot.ts write the same shape). */
export interface LedgerRow { order_id: string; symbol: string; side: string; qty: number | null; bid: number | null; ask: number | null; mid: number | null; source: string }

export function parseLedger(text: string): { rows: LedgerRow[]; malformed: number } {
  const rows: LedgerRow[] = [];
  let malformed = 0;
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    let o: Record<string, unknown>;
    try {
      o = rec(JSON.parse(line));
    } catch {
      malformed++;
      continue;
    }
    const orderId = str(o.order_id);
    const symbol = str(o.symbol);
    if (!orderId || !symbol) { malformed++; continue; }
    rows.push({ order_id: orderId, symbol, side: str(o.side).toLowerCase(), qty: num(o.qty), bid: num(o.bid), ask: num(o.ask), mid: num(o.mid), source: str(o.source) });
  }
  return { rows, malformed };
}

/** One filled option order (or one leg of a filled mleg order) in the window. */
export interface OptionFill { order_id: string; symbol: string; side: "buy" | "sell"; qty: number; price: number; filled_at: string | null; mleg: boolean }

/** Alpaca's orders (nested: an mleg parent carries its legs) -> the option
 *  fills whose fill time falls in (fromMs, toMs]. An mleg parent that filled
 *  but carries no per-leg fill prices is counted, not split by a guessed sign
 *  convention of its net price. */
export function optionFillsInWindow(orders: unknown, fromMs: number, toMs: number): { fills: OptionFill[]; mleg_no_leg_fills: number } {
  const fills: OptionFill[] = [];
  let mlegNoLegFills = 0;
  const inWindow = (t: number | null) => t !== null && t > fromMs && t <= toMs;
  const sideOf = (x: unknown): "buy" | "sell" | null => {
    const s = str(x).toLowerCase();
    return s.startsWith("buy") ? "buy" : s.startsWith("sell") ? "sell" : null;
  };
  for (const raw of Array.isArray(orders) ? orders : []) {
    const o = rec(raw);
    const id = str(o.id);
    const legs = Array.isArray(o.legs) ? o.legs : [];
    // an order with no symbol is an option combo (every stock order names one)
    const isMleg = str(o.order_class) === "mleg" || !str(o.symbol);
    if (isMleg) {
      const parentQty = num(o.filled_qty) ?? 0;
      let legFills = 0;
      for (const lraw of legs) {
        const l = rec(lraw);
        const qty = num(l.filled_qty) ?? 0;
        const price = num(l.filled_avg_price);
        const side = sideOf(l.side);
        const symbol = str(l.symbol);
        const at = str(l.filled_at) || str(o.filled_at) || null;
        if (!(qty > 0) || price === null || !(price > 0) || !side || !symbol) continue;
        legFills++;
        if (inWindow(msOf(at))) fills.push({ order_id: id, symbol, side, qty, price, filled_at: at, mleg: true });
      }
      if (parentQty > 0 && legFills === 0 && inWindow(msOf(o.filled_at))) mlegNoLegFills++;
      continue;
    }
    const symbol = str(o.symbol);
    if (!symbol || !isOptionPosition(symbol, str(o.asset_class))) continue;
    const qty = num(o.filled_qty) ?? 0;
    const price = num(o.filled_avg_price);
    const side = sideOf(o.side);
    const at = str(o.filled_at) || null;
    if (!(qty > 0) || price === null || !(price > 0) || !side || !inWindow(msOf(at))) continue;
    fills.push({ order_id: id, symbol, side, qty, price, filled_at: at, mleg: false });
  }
  return { fills, mleg_no_leg_fills: mlegNoLegFills };
}

export interface FillsBlock {
  /** filled option orders (mleg: legs) in the window */
  fills: number;
  attributed: number;
  /** Σ (fill − mid) × qty × 100 over the attributed fills, signed as a cost (> 0 = paid away) */
  cost: number;
  /** −cost: the fills/slippage part of the option P&L */
  pnl: number;
  unattributed: { count: number; no_row: number; no_mid: number; mleg_no_leg_fills: number; notional: number };
  worst: Array<{ order_id: string; symbol: string; side: string; qty: number; fill: number; mid: number; cost: number; source: string }>;
  ledger_rows: number;
  ledger_malformed: number;
}

export function fillsSlippage(fills: readonly OptionFill[], ledger: { rows: readonly LedgerRow[]; malformed: number }, mlegNoLegFills = 0): FillsBlock {
  const byKey = new Map<string, LedgerRow>();
  for (const r of ledger.rows) {
    const k = `${r.order_id}|${r.symbol}`;
    if (!byKey.has(k)) byKey.set(k, r);
  }
  let cost = 0, attributed = 0, noRow = 0, noMid = 0, notional = 0;
  const priced: FillsBlock["worst"] = [];
  for (const f of fills) {
    const row = byKey.get(`${f.order_id}|${f.symbol}`);
    if (!row || row.mid === null || !(row.mid > 0)) {
      if (!row) noRow++; else noMid++;
      notional += Math.abs(f.price * f.qty * 100);
      continue;
    }
    const c = (f.side === "buy" ? f.price - row.mid : row.mid - f.price) * f.qty * 100;
    cost += c;
    attributed++;
    priced.push({ order_id: f.order_id, symbol: f.symbol, side: f.side, qty: f.qty, fill: f.price, mid: row.mid, cost: r2(c), source: row.source });
  }
  priced.sort((a, b) => b.cost - a.cost);
  return {
    fills: fills.length + mlegNoLegFills, attributed, cost: r2(cost), pnl: r2(-cost),
    unattributed: { count: noRow + noMid + mlegNoLegFills, no_row: noRow, no_mid: noMid, mleg_no_leg_fills: mlegNoLegFills, notional: r2(notional) },
    worst: priced.slice(0, DAILY_LINE_WORST_FILLS), ledger_rows: ledger.rows.length, ledger_malformed: ledger.malformed,
  };
}

/** One option contract held at a line's time (kept so the next line can see
 *  what left the book without a fill). */
export interface OptionContract { symbol: string; qty: number; mv: number | null; upl: number | null }

export function optionContracts(positions: readonly PositionLite[]): OptionContract[] {
  return positions.filter((p) => p.option).map((p) => ({ symbol: p.symbol, qty: p.qty, mv: p.marketValue, upl: p.unrealizedPl }));
}

/** Realized option P&L of the closed lots whose closing fill is in (fromMs, toMs]. */
export function realizedOptionPnl(events: readonly RealizedEvent[], fromMs: number, toMs: number): { pnl: number; lots: number } {
  let pnl = 0, lots = 0;
  for (const e of events) {
    if (!isOptionPosition(e.symbol) || !(e.t > fromMs && e.t <= toMs)) continue;
    pnl += e.pnl;
    lots++;
  }
  return { pnl: r2(pnl), lots };
}

export interface OptionsBlock {
  /** single_session: the window is exactly the day's session (the previous
   *  line is the previous session's; null when no SPY bars tell) */
  window: { from: string; to: string; from_line: string | null; single_session: boolean | null };
  realized: number | null;
  realized_lots: number | null;
  unrealized_start: number | null;
  unrealized_end: number | null;
  closed_without_fill: Array<{ symbol: string; qty: number; adjustment: number | null }>;
  total: number | null;
  fills: FillsBlock;
  direction: number | null;
  contracts_end: OptionContract[] | null;
  reason: string | null;
}

const sumUpl = (cs: readonly OptionContract[] | null): number | null => {
  if (!cs) return null;
  let s = 0;
  for (const c of cs) {
    if (c.upl === null) return null;
    s += c.upl;
  }
  return r2(s);
};

/** TOTAL = realized + (unrealized now − unrealized at the previous line) +
 *  for each contract that left the book with no fill in the window, −(its
 *  cost basis) (its value went to zero in the options sleeve: an expiry, an
 *  assignment or an exercise); DIRECTION = TOTAL − FILLS. */
export function optionPnl(i: {
  fromMs: number; toMs: number; fromLine: string | null;
  events: readonly RealizedEvent[] | null;
  start: readonly OptionContract[] | null;
  end: OptionContract[] | null;
  fills: FillsBlock; filledSymbols: ReadonlySet<string>;
}): OptionsBlock {
  const win = { from: new Date(i.fromMs).toISOString(), to: new Date(i.toMs).toISOString(), from_line: i.fromLine, single_session: i.fromLine ? null : true };
  const realized = i.events ? realizedOptionPnl(i.events, i.fromMs, i.toMs) : null;
  const uEnd = sumUpl(i.end);
  const uStart = sumUpl(i.start);
  const gone: OptionsBlock["closed_without_fill"] = [];
  if (i.start && i.end) {
    const held = new Set(i.end.map((c) => c.symbol));
    for (const c of i.start) {
      if (held.has(c.symbol) || i.filledSymbols.has(c.symbol)) continue;
      gone.push({ symbol: c.symbol, qty: c.qty, adjustment: c.mv !== null && c.upl !== null ? r2(-(c.mv - c.upl)) : null });
    }
  }
  const reasons: string[] = [];
  if (!realized) reasons.push("the sleeve ledger's closed-lot walk is unavailable");
  if (!i.fromLine) reasons.push(`no previous line within ${DAILY_LINE_OPTION_WINDOW_MAX_DAYS} days to start from (the option total starts with the next line)`);
  else if (!i.start) reasons.push("the previous line recorded no option positions (positions were unreadable then)");
  else if (uStart === null) reasons.push("the previous line has an option position with no unrealized P&L");
  if (!i.end) reasons.push("positions unreadable");
  else if (uEnd === null) reasons.push("an option position reported no unrealized P&L");
  if (gone.some((g) => g.adjustment === null)) reasons.push("a contract left the book with no fill and no recorded value");
  const total = !reasons.length && realized && uEnd !== null && uStart !== null
    ? r2(realized.pnl + gone.reduce((s, g) => s + (g.adjustment ?? 0), 0) + uEnd - uStart)
    : null;
  return {
    window: win, realized: realized ? realized.pnl : null, realized_lots: realized ? realized.lots : null,
    unrealized_start: uStart, unrealized_end: uEnd, closed_without_fill: gone, total, fills: i.fills,
    direction: total !== null ? r2(total - i.fills.pnl) : null,
    contracts_end: i.end, reason: reasons.length ? reasons.join("; ") : null,
  };
}

/** Where the option window opens: the previous line's write time when it is
 *  at most DAILY_LINE_OPTION_WINDOW_MAX_DAYS old (it carries the unrealized
 *  P&L the window starts from), else today's ET midnight (and no start). */
export function optionWindowStart(prev: DailyLine | null, day: string, nowMs: number): { fromMs: number; prev: DailyLine | null } {
  const at = prev ? msOf(prev.written_at) : null;
  if (prev && at !== null && prev.day < day && nowMs - at <= DAILY_LINE_OPTION_WINDOW_MAX_DAYS * DAY_MS) return { fromMs: at, prev };
  return { fromMs: zonedInstant(day, 0, ET), prev: null };
}

// ── 4. option exits held by the opening window ──────────────────────────────

export interface OpeningWindowBlock {
  total: number;
  manager_single_leg: number;
  manager_group: number;
  pos_kill: number;
  pos_kill_tickers: string[];
  keys: string[];
  reason: string | null;
}

export function openingWindowBlocks(doc: unknown, day: string, posKillRows: ReadonlyArray<{ time: string; message: string }> | null): OpeningWindowBlock {
  const d = rec(rec(rec(doc).days)[day]);
  const list = (x: unknown) => (Array.isArray(x) ? Array.from(new Set(x.map(str).filter(Boolean))) : []);
  const single = list(d.single_leg);
  const group = list(d.group);
  const tickers = new Set<string>();
  for (const r of posKillRows ?? []) {
    const t = msOf(r.time);
    if (t === null || zonedParts(t, ET).date !== day) continue;
    const m = /^([^:\s]+):/.exec(str(r.message));
    if (m) tickers.add(m[1]);
  }
  const posKill = Array.from(tickers).sort();
  return {
    total: single.length + group.length + posKill.length,
    manager_single_leg: single.length, manager_group: group.length, pos_kill: posKill.length,
    pos_kill_tickers: posKill, keys: [...single, ...group],
    reason: posKillRows === null ? "audit log unreadable: POS-KILL-OPTION-OPENING rows not counted" : null,
  };
}

// ── the line ────────────────────────────────────────────────────────────────

export interface DailyLine {
  version: 1;
  day: string;
  written_at: string;
  session_close: string;
  account_vs_spy: SameBasisBlock;
  floor: FloorBlock;
  options: OptionsBlock;
  opening_window: OpeningWindowBlock;
  /** inputs that could not be read (the line was written partial) */
  missing: string[];
  message: string;
}

export interface DailyLineInputs {
  day: string;
  nowMs: number;
  session: DailyLineSession;
  account: { equity: unknown; lastEquity: unknown } | null;
  positions: PositionLite[] | null;
  spyBars: DailyClose[] | null;
  curve: readonly CurvePoint[];
  floorCfg: FloorConfig | null;
  vmState: unknown;
  /** the previous line that opens the option window (optionWindowStart), or null */
  prev: DailyLine | null;
  windowFromMs: number;
  events: readonly RealizedEvent[] | null;
  orders: unknown[] | null;
  ledger: { rows: LedgerRow[]; malformed: number };
  deferrals: unknown;
  posKillRows: ReadonlyArray<{ time: string; message: string }> | null;
  missing: string[];
}

export function buildDailyLine(i: DailyLineInputs): DailyLine {
  const ab = sameBasis({ equity: i.account?.equity, lastEquity: i.account?.lastEquity, spyBars: i.spyBars, curve: i.curve, day: i.day, since: DAILY_LINE_SINCE });
  const floor = floorExposure({ positions: i.positions, equity: ab.day.equity, cfg: i.floorCfg, vmState: i.vmState, nowMs: i.nowMs });
  const of = i.orders ? optionFillsInWindow(i.orders, i.windowFromMs, i.nowMs) : { fills: [], mleg_no_leg_fills: 0 };
  const fills = fillsSlippage(of.fills, i.ledger, of.mleg_no_leg_fills);
  const options = optionPnl({
    fromMs: i.windowFromMs, toMs: i.nowMs, fromLine: i.prev ? i.prev.day : null,
    events: i.events,
    start: i.prev ? i.prev.options?.contracts_end ?? null : null,
    end: i.positions ? optionContracts(i.positions) : null,
    fills, filledSymbols: new Set(of.fills.map((f) => f.symbol)),
  });
  // the previous session is the SPY bar before the day's (bars exist for every session)
  const prevSession = (i.spyBars ?? []).map((b) => b.date).filter((d) => d < i.day).sort().pop() ?? null;
  if (i.prev) options.window.single_session = prevSession ? i.prev.day === prevSession : null;
  if (!i.orders) {
    options.total = options.direction = null;
    options.reason = [options.reason, "the window's orders are unreadable (fills not joined)"].filter(Boolean).join("; ");
  }
  const line: DailyLine = {
    version: 1, day: i.day, written_at: new Date(i.nowMs).toISOString(), session_close: hhmm(i.session.closeMin),
    account_vs_spy: ab, floor, options, opening_window: openingWindowBlocks(i.deferrals, i.day, i.posKillRows),
    missing: [...i.missing], message: "",
  };
  line.message = dailyLineMessage(line);
  return line;
}

const pctTxt = (x: number | null) => (x === null ? "n/a" : `${x >= 0 ? "+" : ""}${x.toFixed(2)}%`);
const ppTxt = (x: number | null) => (x === null ? "n/a" : `${x >= 0 ? "+" : ""}${x.toFixed(2)}pp`);
const usdTxt = (x: number | null) => (x === null ? "n/a" : `${x < 0 ? "-" : "+"}$${Math.abs(x).toFixed(0)}`);
const xTxt = (x: number | null) => (x === null ? "n/a" : `${x.toFixed(2)}x`);

/** The audit row: one compact line (persistAudit keeps 500 chars). */
export function dailyLineMessage(l: DailyLine): string {
  const a = l.account_vs_spy;
  const parts = [`${l.day} acct ${pctTxt(a.day.account_pct)} vs SPY ${pctTxt(a.day.spy_pct)} (${ppTxt(a.day.gap_pp)})`];
  const w = (label: string, b: BenchWindow | null) => (b ? `${label} ${pctTxt(b.accountPct)} vs ${pctTxt(b.spyPct)} (${ppTxt(b.gapPp)})` : `${label} n/a`);
  parts.push(`${w(`since ${a.since?.from.slice(5) ?? a.since_requested.slice(5)}`, a.since)}; ${w("30d", a.d30)}`);
  const f = l.floor;
  if (f.exposure_x === null) parts.push(`floor n/a (${f.reason ?? "unknown"})`);
  else if (f.mode === "vol_managed_spy") parts.push(`floor ${xTxt(f.exposure_x)} SPY${f.target_x !== null ? ` (target ${xTxt(f.target_x)})` : ""}`);
  else parts.push(`floor ${xTxt(f.exposure_x)} equity (${f.mode}, not SPY-equivalent)`);
  const o = l.options;
  // a window wider than the day's session (a missed line) says where it starts
  const span = o.window.from_line && o.window.single_session !== true ? ` since ${o.window.from_line.slice(5)} line` : "";
  const priced = `${o.fills.attributed}/${o.fills.fills} fills priced`;
  parts.push(o.total !== null
    ? `options${span} ${usdTxt(o.total)} = fills ${usdTxt(o.fills.pnl)} + direction ${usdTxt(o.direction)} (${priced})`
    : `options total n/a; fills ${usdTxt(o.fills.pnl)} (${priced})`);
  const ow = l.opening_window;
  parts.push(`opening-window blocks ${ow.total} (manager ${ow.manager_single_leg + ow.manager_group}, POS-KILL ${ow.pos_kill})`);
  if (l.missing.length) parts.push(`partial: ${l.missing.map((m) => m.split(":")[0]).join(", ")}`);
  return parts.join(" | ").slice(0, 490);
}

// ── files (DATA_DIR, the same directory Python's storage_config uses) ───────

export function dailyLineDir(env: Record<string, string | undefined> = process.env): string {
  return env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
}

export function readJsonFile(p: string): unknown {
  try {
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch (e: unknown) {
    if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.warn(`[daily-line] ${p} unreadable:`, e instanceof Error ? e.message : e);
    return null;
  }
}

/** The stored lines, oldest first ([] when none or unreadable). */
export function loadDailyLines(dir: string): DailyLine[] {
  const doc = rec(readJsonFile(path.join(dir, DAILY_LINE_FILENAME)));
  const lines = Array.isArray(doc.lines) ? doc.lines : [];
  return lines.filter((l): l is DailyLine => !!l && typeof l === "object" && typeof (l as { day?: unknown }).day === "string")
    .sort((a, b) => (a.day < b.day ? -1 : a.day > b.day ? 1 : 0));
}

export function lastLineDay(lines: readonly DailyLine[]): string | null {
  return lines.length ? lines[lines.length - 1].day : null;
}

/** Add (or replace) the day's line, keep the newest `keep`, write-then-rename.
 *  Returns the path written, or null when the write failed. */
export function saveDailyLine(dir: string, line: DailyLine, keep = DAILY_LINE_KEEP): string | null {
  const p = path.join(dir, DAILY_LINE_FILENAME);
  try {
    const lines = loadDailyLines(dir).filter((l) => l.day !== line.day);
    lines.push(line);
    fs.mkdirSync(dir, { recursive: true });
    const tmp = `${p}.${process.pid}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify({ version: 1, lines: lines.slice(-keep) }));
    fs.renameSync(tmp, p);
    return p;
  } catch (e: unknown) {
    console.error("[daily-line] could not save:", e instanceof Error ? e.message : e);
    return null;
  }
}

/** The fill-ledger row server/bot.ts writes after one of ITS option order
 *  POSTs returned. Those paths hold no quote, so bid/ask/mid are null and the
 *  daily line counts their fills as unattributed. */
export function optionLedgerRow(orderId: string, symbol: string, side: string, qty: number, source: string, nowMs = Date.now()): Record<string, unknown> {
  return {
    ts: new Date(nowMs).toISOString(), order_id: orderId, symbol, side: side.toLowerCase(),
    qty: Number.isFinite(qty) ? qty : null, bid: null, ask: null, mid: null, source, limit: null, quote_age_s: null,
  };
}

/** Append one row to the option fill ledger (rotated to .1 above maxBytes).
 *  Never throws: false when the write failed. */
export function appendOptionFillLedger(dir: string, row: Record<string, unknown>, maxBytes = OPTION_FILL_LEDGER_MAX_BYTES): boolean {
  const p = path.join(dir, OPTION_FILL_LEDGER_FILENAME);
  try {
    fs.mkdirSync(dir, { recursive: true });
    if (fs.existsSync(p) && fs.statSync(p).size > maxBytes) fs.renameSync(p, `${p}.1`);
    fs.appendFileSync(p, `${JSON.stringify(row)}\n`);
    return true;
  } catch (e: unknown) {
    console.error("[option-fill-ledger] row not written:", e instanceof Error ? e.message : e);
    return false;
  }
}

/** Both generations of the ledger, older first. */
export function readOptionFillLedger(dir: string): { rows: LedgerRow[]; malformed: number } {
  const out = { rows: [] as LedgerRow[], malformed: 0 };
  for (const f of [`${OPTION_FILL_LEDGER_FILENAME}.1`, OPTION_FILL_LEDGER_FILENAME]) {
    const p = path.join(dir, f);
    let text = "";
    try {
      text = fs.readFileSync(p, "utf8");
    } catch (e: unknown) {
      if ((e as NodeJS.ErrnoException)?.code !== "ENOENT") console.warn(`[daily-line] ${p} unreadable:`, e instanceof Error ? e.message : e);
      continue;
    }
    const parsed = parseLedger(text);
    out.rows.push(...parsed.rows);
    out.malformed += parsed.malformed;
  }
  return out;
}
