/**
 * brokerClockGate.ts — BROKER CLOCK ORDER GATE (2026-11-05, human-directed).
 *
 * "Market-hours check: apply the Alpaca clock/calendar check (is_open,
 * holidays, half-day early close) to every order path, including the old
 * floor (so a rollback is safe), options exits and the backstop. Make the
 * 4am–8pm scan unable to send orders when the market is closed."
 *
 * THE RULE. No order is sent while Alpaca's clock says the market is closed,
 * so nothing is ever queued for the next opening print and options never
 * trade outside regular hours. server/bot.ts's alpaca() calls enforce() before
 * every order submission (POST/PATCH on the orders endpoint, and DELETE of a
 * position, which Alpaca turns into a market order); a refused order throws an
 * Error whose message starts with ORDER_BLOCKED_PREFIX, which every caller
 * already handles as a rejected order. Cancels are never gated. The Python
 * engine's order paths pass the same rule in broker_clock.py.
 *
 * THE CLOCK. One cached reading of Alpaca /v2/clock, shared with POS-KILL
 * (bot.ts posKillReadMarketOpen, 60 s): open -> allowed; closed -> refused;
 * a failed read is extrapolated from the last good reading (said open and
 * before its next_close -> open; said closed and before its next_open ->
 * closed; said open, past its next_close and before its next_open -> closed;
 * otherwise unknown); unknown -> allowed, because a clock outage must never
 * block a protective exit. A reading that said open is read as closed once its
 * own next_close has passed, so a 13:00 half-day close takes effect at 13:00.
 *
 * THE ONE EXCEPTION — EXTENDED-HOURS STOCK EXITS. Human, 2026-11-05: "I want
 * as much on as possible to see if it works and not gate everything ... we
 * have a paper account so use it." A SELL of a held long stock/ETF position
 * carrying extended_hours: true (what orderParams.ts builds for stop_loss /
 * trailing_stop / take_profit outside 09:30–16:00) passes while a REAL
 * extended session is running:
 *   - a trading day: not a weekend, not a market_calendar.py holiday, and not
 *     a day the broker clock showed closed during regular hours;
 *   - pre-market 04:00–09:30 ET, and only when the broker's next open is that
 *     same day;
 *   - after-hours from the close for 4 hours: 16:00–20:00 ET, or 13:00–17:00
 *     ET after an early close (market_calendar.py half days, or the close time
 *     the broker's clock gave that day).
 * Outside a real session (weekends, holidays, overnight 20:00–04:00) a stock
 * exit is refused like every other order. Entries, options, short covers and
 * every order without extended_hours: true stay refused while the market is
 * closed. Default ON; EXTENDED_HOURS_STOCK_EXITS_ENV=0 (also false/off/no)
 * turns it off, and then every order is refused while the clock says closed.
 */
import { isTradingDay, zonedParts, type MarketCalendar } from "./jobHeartbeat";

export const ORDER_BLOCKED_PREFIX = "ORDER-BLOCKED";
export const ORDER_BLOCKED_AUDIT_TYPE = "ORDER-BLOCKED-CLOSED";
/** Env switch for the extended-hours stock-exit exception (default ON). */
export const EXTENDED_HOURS_STOCK_EXITS_ENV = "VOLTRADE_EXTENDED_HOURS_STOCK_EXITS";
/** Alpaca's sessions, minutes after midnight ET. */
export const PREMARKET_START_MIN_ET = 4 * 60;
export const REGULAR_OPEN_MIN_ET = 9 * 60 + 30;
export const REGULAR_CLOSE_MIN_ET = 16 * 60;
export const EARLY_CLOSE_MIN_ET = 13 * 60;
/** After-hours runs this long after the close (16:00–20:00; 13:00–17:00 after an early close). */
export const AFTER_HOURS_SESSION_MIN = 4 * 60;

const ET = "America/New_York";

export interface ClockReading {
  isOpen: boolean;
  nextOpenMs: number | null;
  nextCloseMs: number | null;
  readAtMs: number;
}

export type ClockStateName = "open" | "closed" | "unknown";

export interface ClockState {
  state: ClockStateName;
  source: "live" | "extrapolated" | "none";
  detail: string;
  /** The reading the state came from (live or last good); null when none. */
  reading: ClockReading | null;
}

/** An Alpaca clock timestamp as epoch ms; null when missing, unreadable or without an offset. */
function parseClockTime(raw: unknown): number | null {
  if (typeof raw !== "string" || !/(Z|[+-]\d{2}:?\d{2})$/.test(raw)) return null;
  const ms = Date.parse(raw);
  return Number.isFinite(ms) ? ms : null;
}

/** A usable reading needs a boolean is_open; next_open / next_close are optional. */
export function parseClockReading(body: unknown, readAtMs: number): ClockReading | null {
  if (!body || typeof body !== "object" || Array.isArray(body)) return null;
  const b = body as Record<string, unknown>;
  if (typeof b.is_open !== "boolean") return null;
  return { isOpen: b.is_open, nextOpenMs: parseClockTime(b.next_open), nextCloseMs: parseClockTime(b.next_close), readAtMs };
}

function hhmm(minutes: number): string {
  return `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;
}

function etLabel(ms: number): string {
  const p = zonedParts(ms, ET);
  return `${p.weekday} ${p.date} ${hhmm(p.minutes)} ET`;
}

/** The market state a reading implies at nowMs. A live reading's is_open is
 *  trusted as read; an old one (the read failed) only within the times it states. */
function stateFromReading(r: ClockReading, nowMs: number, live: boolean): ClockState {
  const source = live ? "live" : "extrapolated";
  const st = (state: ClockStateName, detail: string): ClockState => ({ state, source, detail, reading: r });
  if (r.isOpen) {
    if (r.nextCloseMs === null) {
      return live ? st("open", "broker clock: open") : st("unknown", "the last reading said open but gave no close time");
    }
    if (nowMs < r.nextCloseMs) return st("open", `broker clock: open until ${etLabel(r.nextCloseMs)}`);
    if (r.nextOpenMs !== null && nowMs < r.nextOpenMs) {
      return st("closed", `broker clock: closed at ${etLabel(r.nextCloseMs)}, next open ${etLabel(r.nextOpenMs)}`);
    }
    return st("unknown", "the last reading is out of date");
  }
  if (r.nextOpenMs === null) {
    return live ? st("closed", "broker clock: closed") : st("unknown", "the last reading said closed but gave no open time");
  }
  if (nowMs < r.nextOpenMs) return st("closed", `broker clock: closed until ${etLabel(r.nextOpenMs)}`);
  if (r.nextCloseMs !== null && nowMs < r.nextCloseMs) {
    return st("open", `broker clock: opened at ${etLabel(r.nextOpenMs)} on its own schedule`);
  }
  return st("unknown", "the last reading is out of date");
}

/** live = this cache window's reading (null when the read failed); lastGood = the last usable reading. */
export function clockStateFrom(live: ClockReading | null, lastGood: ClockReading | null, nowMs: number): ClockState {
  if (live) return stateFromReading(live, nowMs, true);
  if (!lastGood) return { state: "unknown", source: "none", detail: "no usable broker clock reading", reading: null };
  const st = stateFromReading(lastGood, nowMs, false);
  return { ...st, detail: `clock read failed; ${st.detail} (last good reading ${etLabel(lastGood.readAtMs)})` };
}

/** Default ON; "0" / "false" / "off" / "no" turns the exception off. */
export function extendedStockExitsEnabled(env: Record<string, string | undefined>): boolean {
  const raw = env[EXTENDED_HOURS_STOCK_EXITS_ENV];
  if (raw === undefined) return true;
  return !["0", "false", "off", "no"].includes(raw.trim().toLowerCase());
}

/** What the gate has seen of today's session from the broker's own readings. */
export interface SessionEvidence {
  /** ET date of the last reading that said open, and that day's close time from it (null if not that day). */
  openSeenDay: string | null;
  openSeenCloseMs: number | null;
  /** ET date on which a reading inside 09:35–13:00 ET said closed and none said open: no session that day. */
  closedInSessionDay: string | null;
}

export function emptySessionEvidence(): SessionEvidence {
  return { openSeenDay: null, openSeenCloseMs: null, closedInSessionDay: null };
}

/** Fold one live reading into the evidence (pure: returns the new evidence). */
export function observeReading(ev: SessionEvidence, r: ClockReading): SessionEvidence {
  const p = zonedParts(r.readAtMs, ET);
  if (r.isOpen) {
    const closeToday = r.nextCloseMs !== null && zonedParts(r.nextCloseMs, ET).date === p.date ? r.nextCloseMs : null;
    return { ...ev, openSeenDay: p.date, openSeenCloseMs: closeToday };
  }
  if (p.minutes >= REGULAR_OPEN_MIN_ET + 5 && p.minutes < EARLY_CLOSE_MIN_ET && ev.openSeenDay !== p.date) {
    return { ...ev, closedInSessionDay: p.date };
  }
  return ev;
}

export type ExtendedSessionName = "pre" | "post";

/** Is a real extended session running at nowMs? (Asked only while the clock says closed.) */
export function extendedSession(nowMs: number, cal: MarketCalendar, reading: ClockReading | null,
                                ev: SessionEvidence): { session: ExtendedSessionName | null; why: string } {
  const p = zonedParts(nowMs, ET);
  if (!isTradingDay(p.date, cal)) {
    return { session: null, why: `${p.date} is not a trading day (weekend or market_calendar.py holiday)` };
  }
  if (ev.closedInSessionDay === p.date) {
    return { session: null, why: `the broker clock showed the market closed during ${p.date}'s regular hours` };
  }
  if (p.minutes >= PREMARKET_START_MIN_ET && p.minutes < REGULAR_OPEN_MIN_ET) {
    if (reading && reading.nextOpenMs !== null) {
      const nextOpenDay = zonedParts(reading.nextOpenMs, ET).date;
      if (nextOpenDay !== p.date) {
        return { session: null, why: `the broker clock's next open is ${nextOpenDay}, so ${p.date} has no pre-market session` };
      }
    }
    return { session: "pre", why: `pre-market session (${hhmm(PREMARKET_START_MIN_ET)}–${hhmm(REGULAR_OPEN_MIN_ET)} ET)` };
  }
  const closeMin = ev.openSeenDay === p.date && ev.openSeenCloseMs !== null
    ? zonedParts(ev.openSeenCloseMs, ET).minutes
    : (cal.halfDays.has(p.date) ? EARLY_CLOSE_MIN_ET : REGULAR_CLOSE_MIN_ET);
  const endMin = closeMin + AFTER_HOURS_SESSION_MIN;
  if (p.minutes >= closeMin && p.minutes < endMin) {
    return { session: "post", why: `after-hours session (${hhmm(closeMin)}–${hhmm(endMin)} ET)` };
  }
  return {
    session: null,
    why: `no extended session at ${hhmm(p.minutes)} ET (pre-market ${hhmm(PREMARKET_START_MIN_ET)}–${hhmm(REGULAR_OPEN_MIN_ET)}, after-hours ${hhmm(closeMin)}–${hhmm(endMin)})`,
  };
}

export interface GatedOrder {
  method: string;
  symbol: string;
  side: string;
  qty: number;
  type: string;
  extendedHours: boolean;
}

/** The order an alpaca() call would submit: its JSON body, or for a position
 *  DELETE (a close) the symbol from the path. Never throws. */
export function describeOrder(path: string, method: string, body: unknown): GatedOrder {
  let b: Record<string, unknown> = {};
  if (typeof body === "string" && body) {
    try {
      const parsed: unknown = JSON.parse(body);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) b = parsed as Record<string, unknown>;
    } catch (err) {
      b = { unparsed: err instanceof Error ? err.message : String(err) };
    }
  }
  const legs = Array.isArray(b.legs) ? b.legs : [];
  const firstLeg = legs.length > 0 && legs[0] && typeof legs[0] === "object" ? (legs[0] as Record<string, unknown>) : {};
  const lastSegment = decodeURIComponent(String(path).split("?")[0].split("/").filter(Boolean).pop() || "?");
  const fromPath = lastSegment.toLowerCase() === "positions" ? "ALL POSITIONS" : lastSegment;
  const symbol = typeof b.symbol === "string" && b.symbol ? b.symbol
    : typeof firstLeg.symbol === "string" && firstLeg.symbol ? firstLeg.symbol
    : String(method).toUpperCase() === "DELETE" ? fromPath : "?";
  return {
    method: String(method || "GET").toUpperCase(),
    symbol: symbol.toUpperCase(),
    side: typeof b.side === "string" ? b.side.toLowerCase() : (String(method).toUpperCase() === "DELETE" ? "close" : ""),
    qty: Number(b.qty) || 0,
    type: typeof b.type === "string" ? b.type : "",
    extendedHours: b.extended_hours === true,
  };
}

export interface OrderGateDeps {
  /** The shared cached clock read: { at, clock } where clock is the last response body (null when that read failed). */
  readClock: () => Promise<{ at: number; clock?: unknown } | null>;
  audit: (type: string, message: string) => void;
  /** The account's position in a symbol (rejects when there is none). */
  getPosition: (symbol: string) => Promise<unknown>;
  calendar: () => MarketCalendar;
  isOption: (symbol: string) => boolean;
  now?: () => number;
  env?: () => Record<string, string | undefined>;
}

export interface GateSnapshot {
  state: ClockStateName;
  detail: string;
  extendedExitsOn: boolean;
  extendedSession: ExtendedSessionName | null;
  sessionWhy: string;
  /** Would a sell of a held long stock position carrying extended_hours: true pass right now? */
  longStockExitOpen: boolean;
}

export interface OrderGate {
  snapshot(): Promise<GateSnapshot>;
  enforce(path: string, method: string, body: unknown): Promise<void>;
}

export function createOrderGate(deps: OrderGateDeps): OrderGate {
  const now = deps.now ?? (() => Date.now());
  const env = deps.env ?? (() => process.env);
  let lastGood: ClockReading | null = null;
  let evidence = emptySessionEvidence();
  const blockedAuditDay = new Map<string, string>();

  async function snapshot(): Promise<GateSnapshot> {
    const nowMs = now();
    let live: ClockReading | null = null;
    try {
      const cached = await deps.readClock();
      live = cached ? parseClockReading(cached.clock, cached.at) : null;
    } catch (err) {
      console.warn(`[order-gate] clock read failed: ${err instanceof Error ? err.message : String(err)}`);
    }
    if (live) {
      lastGood = live;
      evidence = observeReading(evidence, live);
    }
    const clock = clockStateFrom(live, lastGood, nowMs);
    const extendedExitsOn = extendedStockExitsEnabled(env());
    let session: ExtendedSessionName | null = null;
    let sessionWhy = "";
    if (clock.state === "closed") {
      const s = extendedSession(nowMs, deps.calendar(), clock.reading, evidence);
      session = s.session;
      sessionWhy = s.why;
    }
    return {
      state: clock.state, detail: clock.detail, extendedExitsOn, extendedSession: session, sessionWhy,
      longStockExitOpen: clock.state !== "closed" || (extendedExitsOn && session !== null),
    };
  }

  async function heldLongQty(symbol: string): Promise<number | null> {
    try {
      const pos = await deps.getPosition(symbol);
      if (!pos || typeof pos !== "object") return null;
      const p = pos as Record<string, unknown>;
      return p.side === "long" ? Math.abs(Number(p.qty) || 0) : 0;
    } catch (err) {
      console.warn(`[order-gate] position read for ${symbol} failed: ${err instanceof Error ? err.message : String(err)}`);
      return null;
    }
  }

  /** May this order use the extended session while the clock says closed? why is "" for an ordinary order. */
  async function extendedExitVerdict(order: GatedOrder, snap: GateSnapshot): Promise<{ allowed: boolean; why: string }> {
    if (!order.extendedHours || order.method !== "POST") return { allowed: false, why: "" };
    if (!snap.extendedExitsOn) return { allowed: false, why: `extended-hours stock exits are off (${EXTENDED_HOURS_STOCK_EXITS_ENV})` };
    if (deps.isOption(order.symbol)) return { allowed: false, why: "options never trade outside regular hours" };
    if (order.side !== "sell") return { allowed: false, why: "only a sell of a held long stock position may use an extended session" };
    if (!snap.extendedSession) return { allowed: false, why: snap.sessionWhy };
    const held = await heldLongQty(order.symbol);
    if (held === null || held <= 0 || order.qty <= 0 || order.qty > held) {
      return { allowed: false, why: `not an exit: ${held === null ? "no" : held} long ${order.symbol} share(s) held for a sell of ${order.qty}` };
    }
    return { allowed: true, why: snap.sessionWhy };
  }

  function auditBlockedOnce(order: GatedOrder, message: string, nowMs: number): void {
    const day = zonedParts(nowMs, ET).date;
    if (blockedAuditDay.get(order.symbol) === day) return;
    for (const [sym, d] of Array.from(blockedAuditDay.entries())) if (d !== day) blockedAuditDay.delete(sym);
    blockedAuditDay.set(order.symbol, day);
    deps.audit(ORDER_BLOCKED_AUDIT_TYPE, message);
  }

  async function enforce(path: string, method: string, body: unknown): Promise<void> {
    const snap = await snapshot();
    if (snap.state !== "closed") return; // open, or unknown: a clock outage never blocks an exit
    const order = describeOrder(path, method, body);
    const verdict = await extendedExitVerdict(order, snap);
    if (verdict.allowed) return;
    const what = order.method === "DELETE" ? `close of the ${order.symbol} position` : `${order.side || "order"} ${order.qty} ${order.symbol}`;
    const message = `${ORDER_BLOCKED_PREFIX}: market closed — ${what} not sent (${snap.detail}${verdict.why ? `; ${verdict.why}` : ""})`;
    auditBlockedOnce(order, `${order.symbol}: ${message} — one line per symbol per day`, now());
    throw new Error(message);
  }

  return { snapshot, enforce };
}
