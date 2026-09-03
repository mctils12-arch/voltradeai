/**
 * portDwellWeekly.ts — durable weekly-snapshot accumulator for
 * port_dwell_maritime_transit's GATE 2 (SIGNAL) test.
 *
 * WHY THIS EXISTS: `computePortDwellAsync`/the `portdwell_window` diag probe
 * can only ever see a ROLLING raw-retention window (`RAW_RETENTION_DAYS`,
 * currently 30 days -- see `datacoreArchive.ts`'s `oldestRawHour`). That
 * window rolls forward with "now" forever, so an on-demand query taken today
 * will read a materially shorter history a month from now, not a longer one
 * -- raw retention can never accumulate on its own. GATE 2's own test plan
 * (research/open_questions.md, PORT DWELL ANALYTICS) needs a weekly {median
 * dwell, in-port count} series spanning enough weeks for the anomaly-vs-
 * forward-returns test to have any statistical power (REASONING STANDARD
 * #4 -- a handful of points is not evidence). The only way to get there is
 * to CAPTURE one small, aggregate-only snapshot per completed week into a
 * durable file BEFORE that week ages out of raw retention, and never
 * overwrite a week once captured -- turning a permanently-thin rolling
 * window into an ever-growing archive-derived series, the same
 * "accumulation substitutes for purchase" principle CLAUDE.md's BUILD-FIRST
 * RULE names for other roots (flight-track history, position archive).
 *
 * Weeks are anchored to the archive's own start (ARCHIVE_START_MS, the same
 * 2026-07-03 date every other archive-boundary comment in this codebase
 * already cites) rather than to "today", so week boundaries are the same
 * fixed grid no matter which day a session happens to run this -- two
 * sessions a week apart append two DIFFERENT weeks, never the same one
 * re-computed on a shifted boundary.
 *
 * Snapshots carry only the same aggregate per-port fields the live
 * /api/data/portdwell route and the portdwell_window probe already expose
 * (dwell_median_h/dwell_p90_h/dwell_max_h/visits_completed/unique_vessels/
 * in_port_now) -- no per-vessel MMSI, name, or position, matching the
 * anomaly_examples-stripping posture other diag-fed research artifacts in
 * this repo already use.
 *
 * Pure module: no fs/network here (that lives in the calling script), so
 * the merge/extract logic is unit-testable without a live archive.
 */
import type { PortDwellStats } from "./portDwell";

// Same date every archive-boundary comment in this repo already cites
// (datacoreArchive.ts, portDwell.ts's own module header, diag.ts's
// portdwell_window case). Not re-derived from any live source -- it is the
// archive's own documented start.
export const ARCHIVE_START_MS = Date.UTC(2026, 6, 3, 0, 0, 0);
const WEEK_MS = 7 * 24 * 3600_000;

export interface WeekBounds { index: number; startMs: number; endMs: number }

/** The fixed weekly grid: week k spans [start + k*7d, start + (k+1)*7d). */
export function weekBounds(index: number): WeekBounds {
  return { index, startMs: ARCHIVE_START_MS + index * WEEK_MS, endMs: ARCHIVE_START_MS + (index + 1) * WEEK_MS };
}

/** The highest week index that has fully elapsed by `nowMs` (its end is not
 *  in the future) -- the most recent week eligible for capture. */
export function lastCompletedWeekIndex(nowMs: number): number {
  return Math.floor((nowMs - ARCHIVE_START_MS) / WEEK_MS) - 1;
}

export interface WeeklyPortSnapshot {
  id: string; name: string;
  dwell_median_h: number | null;
  dwell_p90_h: number | null;
  dwell_max_h: number | null;
  visits_completed: number;
  unique_vessels: number;
  in_port_now: number;
}

export interface WeeklySnapshot {
  week_index: number;
  week_start: string;  // ISO
  week_end: string;    // ISO
  captured_at: string; // ISO, when this snapshot was taken (may be well after week_end)
  ports: WeeklyPortSnapshot[];
}

/** A read where every port shows zero completed AND zero ongoing visits,
 *  despite `vessels_seen` being nonzero, is not a plausible "quiet week" --
 *  it is the exact signature the 2026-08-19 GATE 1 session found for a
 *  broken reader (see portDwell.ts's own header) and, separately, what a
 *  week fully swallowed by an already-diagnosed feed outage (e.g. the
 *  2026-08-05..08-12 aisstream.io outage, wishlist.md "AIS VESSEL FEED
 *  DARK") also produces. Whichever cause, MEASUREMENT INTEGRITY says a read
 *  this degenerate must not be persisted as if it were a genuine zero-
 *  activity week -- the caller should skip it, not average it in. */
export function isDegenerateAllZeroRead(stats: PortDwellStats): boolean {
  if (stats.vessels_seen <= 0) return false; // an empty archive window is a coverage question, not this one
  return stats.ports.every((p) => p.visits_completed === 0 && p.in_port_now === 0);
}

/** Strips a live PortDwellStats read down to the aggregate-only, per-vessel-
 *  identity-free fields worth persisting long-term. Caller is responsible
 *  for having confirmed `full_raw_coverage` (no coverage_caveat) and that
 *  `isDegenerateAllZeroRead` is false before calling this -- this function
 *  does not itself check either. */
export function extractWeeklySnapshot(stats: PortDwellStats, week: WeekBounds, capturedAtMs: number): WeeklySnapshot {
  return {
    week_index: week.index,
    week_start: new Date(week.startMs).toISOString(),
    week_end: new Date(week.endMs).toISOString(),
    captured_at: new Date(capturedAtMs).toISOString(),
    ports: stats.ports.map((p) => ({
      id: p.id, name: p.name,
      dwell_median_h: p.dwell_median_h, dwell_p90_h: p.dwell_p90_h, dwell_max_h: p.dwell_max_h,
      visits_completed: p.visits_completed, unique_vessels: p.unique_vessels, in_port_now: p.in_port_now,
    })),
  };
}

/** Appends `next` into `existing`, keyed by week_index. A week already
 *  present is NEVER overwritten -- once a week is captured inside raw
 *  retention, a later re-capture of the SAME week can only see a shorter or
 *  equal raw tail (retention rolls forward, never backward), so the first
 *  capture is always the most complete one available. Returns a NEW array,
 *  sorted by week_index ascending; does not mutate `existing`. */
export function mergeWeeklySnapshot(existing: WeeklySnapshot[], next: WeeklySnapshot): WeeklySnapshot[] {
  if (existing.some((s) => s.week_index === next.week_index)) return existing.slice();
  return [...existing, next].sort((a, b) => a.week_index - b.week_index);
}

/** Week indices in [earliestIndex, lastCompletedWeekIndex(nowMs)] not yet
 *  present in `existing` -- the set a backfill run still needs to attempt.
 *  `earliestIndex` is normally derived from the live `raw_vessel_archive_from`
 *  boundary by the caller (the earliest week whose full 7 days sit at or
 *  after that boundary), never assumed to be week 0 (see this module's own
 *  header: raw retention rolls, it does not reach back to the archive start). */
export function missingWeekIndices(existing: WeeklySnapshot[], earliestIndex: number, nowMs: number): number[] {
  const have = new Set(existing.map((s) => s.week_index));
  const out: number[] = [];
  for (let k = earliestIndex; k <= lastCompletedWeekIndex(nowMs); k++) {
    if (!have.has(k)) out.push(k);
  }
  return out;
}
