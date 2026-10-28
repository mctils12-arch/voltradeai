// REPAIR 2026-10-27 (trading-engine audit §1.3 / defect D2): the once-a-day
// "4am branch" — daily reset of the per-day trading counters
// (dailyBlockedTickers, consecutiveStopLosses, morningQueueExecuted,
// orderAttemptCounts, sweptCspAt), the daily ML retrain and the nightly
// backup — lived inside runOvernightResearch(), behind TWO gates that can
// never both pass:
//
//   1. scheduleTier2 only enters tier2Intelligence when
//      `clock.is_open || (4 <= ET < 20)`;
//   2. tier2Intelligence only calls runOvernightResearch when
//      `!isMarketOpen && (ET >= 20 || ET < 4)`;
//   3. and the branch itself needs `Math.floor(ET) === 4`, which (2)
//      excludes outright (ET < 4 or ET >= 20 is never hour 4).
//
// So the branch was dead code; morningQueueExecuted/orderAttemptCounts were
// only ever reset by a process restart. This module is the pure decision
// ("should the daily branch run now?") plus a tiny persisted record of the
// last ET date it ran, so the branch fires exactly once per ET calendar day:
// a deploy inside the window neither repeats it (the date is claimed on disk
// BEFORE the work starts) nor skips it (a process that boots anywhere inside
// the window and finds no record for today runs it).
//
// Window: [04:00, 09:30) ET — the original 4am intent, held open until the
// market opens so a restart mid-window still catches up. A process that is
// down for the WHOLE window skips that day: a fresh boot already starts with
// reset counters, and the hourly Tier-3 retrain covers a model older than 24h.
// Every calendar day (weekends too), as the original branch would have.
//
// Persistence follows bot.ts's existing state-file pattern (equity peak,
// kill-switch latch): /data/voltrade volume first, /tmp fallback.
import fs from "fs";
import path from "path";

export const DAILY_MAINTENANCE_START_HOUR_ET = 4;
export const DAILY_MAINTENANCE_END_HOUR_ET = 9.5;
/** Poll cadence for the unconditional scheduler interval in bot.ts. */
export const DAILY_MAINTENANCE_CHECK_MS = 5 * 60 * 1000;
export const DAILY_MAINTENANCE_PATHS: readonly string[] = [
  "/data/voltrade/voltrade_daily_maintenance.json",
  "/tmp/voltrade_daily_maintenance.json",
];

const ET_DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

const etFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

/** The ET calendar date (YYYY-MM-DD) and fractional ET hour for an instant. DST-correct. */
export function etClock(nowMs: number): { etDate: string; etHour: number } {
  const parts: Record<string, string> = {};
  for (const p of etFormatter.formatToParts(new Date(nowMs))) parts[p.type] = p.value;
  const hour = Number(parts.hour) % 24;
  return {
    etDate: `${parts.year}-${parts.month}-${parts.day}`,
    etHour: hour + Number(parts.minute) / 60,
  };
}

export type DailyMaintenanceReason = "before-window" | "after-window" | "already-ran-today" | "due";

export interface DailyMaintenanceDecision {
  run: boolean;
  reason: DailyMaintenanceReason;
  etDate: string;
  etHour: number;
}

/**
 * Should the daily branch run now? Pure: `lastRunEtDate` is the ET date of
 * the last claimed run (from disk and/or memory), or null when none is known.
 */
export function shouldRunDailyMaintenance(nowMs: number, lastRunEtDate: string | null): DailyMaintenanceDecision {
  const { etDate, etHour } = etClock(nowMs);
  if (etHour < DAILY_MAINTENANCE_START_HOUR_ET) return { run: false, reason: "before-window", etDate, etHour };
  if (etHour >= DAILY_MAINTENANCE_END_HOUR_ET) return { run: false, reason: "after-window", etDate, etHour };
  if (lastRunEtDate === etDate) return { run: false, reason: "already-ran-today", etDate, etHour };
  return { run: true, reason: "due", etDate, etHour };
}

/** Later of two ET dates (YYYY-MM-DD sorts lexicographically); nulls ignored. */
export function latestEtDate(a: string | null, b: string | null): string | null {
  if (!a) return b;
  if (!b) return a;
  return a >= b ? a : b;
}

/** The lastRunEtDate in a persisted record, or null if the record is not a valid one. */
export function parseDailyMaintenanceRecord(raw: unknown): string | null {
  if (!raw || typeof raw !== "object") return null;
  const d = (raw as { lastRunEtDate?: unknown }).lastRunEtDate;
  return typeof d === "string" && ET_DATE_RE.test(d) ? d : null;
}

/**
 * Latest valid lastRunEtDate across every path (not just the first file
 * found): if a volume write ever failed over to /tmp, the newer /tmp record
 * must still win, or the branch would run twice that day.
 */
export function loadLastRunEtDate(paths: readonly string[] = DAILY_MAINTENANCE_PATHS): string | null {
  let latest: string | null = null;
  for (const p of paths) {
    try {
      if (!fs.existsSync(p)) continue;
      latest = latestEtDate(latest, parseDailyMaintenanceRecord(JSON.parse(fs.readFileSync(p, "utf8"))));
    } catch (e: unknown) {
      console.error(`[daily-maintenance] could not read ${p}:`, e instanceof Error ? e.message : e);
    }
  }
  return latest;
}

/** Persist the claim (write-then-rename). Returns the path written, or null if every path failed. */
export function saveLastRunEtDate(
  etDate: string, nowMs: number, paths: readonly string[] = DAILY_MAINTENANCE_PATHS,
): string | null {
  const body = JSON.stringify({ lastRunEtDate: etDate, claimedAt: new Date(nowMs).toISOString() });
  for (const p of paths) {
    try {
      fs.mkdirSync(path.dirname(p), { recursive: true });
      const tmp = `${p}.${process.pid}.tmp`;
      fs.writeFileSync(tmp, body);
      fs.renameSync(tmp, p);
      return p;
    } catch (e: unknown) {
      console.error(`[daily-maintenance] could not save to ${p}:`, e instanceof Error ? e.message : e);
    }
  }
  return null;
}

/**
 * The whole "check" step bot.ts runs on every poll: read the persisted claim,
 * combine it with the in-process memory of the last run (which still guards
 * the day if every state path is unwritable), decide, and — only when due —
 * persist the claim BEFORE the caller starts any work. `persistedTo` is null
 * when the decision was not to run, or when every path failed to write.
 */
export function claimDailyMaintenance(
  nowMs: number, memLastRunEtDate: string | null, paths: readonly string[] = DAILY_MAINTENANCE_PATHS,
): { decision: DailyMaintenanceDecision; persistedTo: string | null } {
  const decision = shouldRunDailyMaintenance(nowMs, latestEtDate(memLastRunEtDate, loadLastRunEtDate(paths)));
  if (!decision.run) return { decision, persistedTo: null };
  return { decision, persistedTo: saveLastRunEtDate(decision.etDate, nowMs, paths) };
}
