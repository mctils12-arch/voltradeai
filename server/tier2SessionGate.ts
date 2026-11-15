/**
 * tier2SessionGate.ts — may the Tier-2 scan run on this tick? (2026-11-15)
 *
 * WHY THIS EXISTS. scheduleTier2 (server/bot.ts) scanned whenever the broker
 * clock was open OR the ET hour was inside 04:00-20:00 — with no trading-day
 * test. The weekday pre-market / after-hours research cadence therefore ran
 * on every Saturday, Sunday and market holiday too: a full 12k-stock scan
 * every ~3.5 min (about 270 a closed day) on bars that cannot change, through
 * the daemon (the rss sawtooth and the #18-class timeouts continued all
 * weekend), each one re-logging Friday's candidates to the shadow log.
 * Measured live 2026-11-15 (`/api/diag/shadow.session_days`, v1.0.1406):
 * 6,434 of the log's 20,000 records — 32% of its cap — were written on days
 * with no session, evicting real history that much sooner. v1.0.1405 stopped
 * those copies COUNTING at read time; this stops them at the source.
 *
 * THE RULE. Inside the 04:00-20:00 ET window on a day with no session
 * (weekend, or a full-closure holiday in the calendar) the scan runs only
 * while this container holds no scan result at all — one scan after a
 * closed-day boot, so the picks pages have something to show (the result is
 * in-memory) and the caches a scan writes exist. After that the tick is
 * skipped until the next session day. Everything else is unchanged:
 *   - the broker clock reading OPEN always scans (a calendar table that is
 *     wrong must never stop a live session — believe the broker);
 *   - outside 04:00-20:00 ET with the market closed nothing scans (as before);
 *   - a half day is a session day.
 *
 * OFF SWITCH (one setting): VOLTRADE_TIER2_CLOSED_DAY_SCANS=1 restores the
 * prior every-calendar-day cadence exactly.
 *
 * Pure module (node:test safe).
 */
import { isTradingDay, type MarketCalendar } from "./jobHeartbeat";

export const TIER2_SCAN_START_ET_HOUR = 4;
export const TIER2_SCAN_END_ET_HOUR = 20;
export const CLOSED_DAY_SCANS_ENV = "VOLTRADE_TIER2_CLOSED_DAY_SCANS";

/** The skip text the scheduler has always beaten outside its window. */
export const OUTSIDE_WINDOW_WHY = "outside the scan window (market closed and outside 04:00-20:00 ET)";

export function closedDayScansEnabled(env: Record<string, string | undefined> = process.env): boolean {
  return ["1", "true", "on", "yes"].includes(String(env[CLOSED_DAY_SCANS_ENV] ?? "").trim().toLowerCase());
}

export interface Tier2GateInput {
  /** Alpaca /v2/clock is_open, read this tick. */
  clockIsOpen: boolean;
  /** Fractional ET hour (orderParams.getETHour). */
  etHour: number;
  /** ET calendar date, YYYY-MM-DD. */
  etDate: string;
  calendar: MarketCalendar;
  /** Does this container already hold a completed scan's result? */
  hasScanResult: boolean;
  /** The off switch (closedDayScansEnabled()). */
  closedDayScans: boolean;
}

export interface Tier2GateDecision {
  scan: boolean;
  /** True when the tick falls inside the window on a day with no session. */
  closedDay: boolean;
  /** Why the tick is skipped (heartbeat detail); null when it scans. */
  why: string | null;
}

export function tier2ScanDecision(i: Tier2GateInput): Tier2GateDecision {
  if (i.clockIsOpen) return { scan: true, closedDay: false, why: null };
  const inWindow = i.etHour >= TIER2_SCAN_START_ET_HOUR && i.etHour < TIER2_SCAN_END_ET_HOUR;
  if (!inWindow) return { scan: false, closedDay: false, why: OUTSIDE_WINDOW_WHY };
  // An unreadable date is treated as a session day: this rule only removes a
  // scan it can positively place on a closed day.
  const validDate = /^\d{4}-\d{2}-\d{2}$/.test(i.etDate);
  if (!validDate || isTradingDay(i.etDate, i.calendar)) return { scan: true, closedDay: false, why: null };
  if (i.closedDayScans) return { scan: true, closedDay: true, why: null };
  if (!i.hasScanResult) return { scan: true, closedDay: true, why: null };
  return {
    scan: false, closedDay: true,
    why: `no session on ${i.etDate} (weekend or market holiday): the scan this container holds is kept, the next scan is the next session day's 04:00 ET`,
  };
}

/** One audit line per closed ET day (not one per skipped tick). */
export function closedDayAuditLine(etDate: string, lastScanAt: string | null): string {
  const held = lastScanAt ? `the scan of ${lastScanAt} is the one shown` : "one scan was run for this container";
  return `No session on ${etDate} (weekend or market holiday) — Tier 2 scans are skipped until the next session day; ${held}. Set ${CLOSED_DAY_SCANS_ENV}=1 to scan on closed days again.`;
}
