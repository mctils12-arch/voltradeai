/**
 * portDwellCapture.ts — IN-PROCESS, server-side capture of
 * port_dwell_maritime_transit's weekly snapshot series (server/portDwellWeekly.ts
 * defines the pure grid/merge/extraction logic this module drives).
 *
 * WHY THIS EXISTS (2026-09-07 PRODUCT session; full account in
 * research/experiments.md this date): the two immediately preceding
 * 2026-09-06 sessions measured that `scripts/portdwell_weekly_snapshot.ts`'s
 * capture path — fetch `/api/diag/portdwell_window?hours=168` over HTTP —
 * now reliably exceeds Railway's edge-proxy response timeout (46-71s
 * before a 502/connection-reset) as the vessel archive has grown, and
 * confirmed the bottleneck is CPU-bound (a flat ~11.7-17.1us/point cost
 * that does NOT amortize down as the window grows), not I/O-bound. That
 * session's own explicit RECOMMENDATION: "move this fold OFF the
 * synchronous request/response path entirely — a scheduled background
 * computation ... writing weekly stats to a durable file the diag probe/
 * weekly-snapshot script then reads cheaply." This module IS that
 * background computation: it runs the exact same
 * `computePortDwellAsyncTimed` fold `portdwell_window` already uses (no
 * new aggregation logic), but IN-PROCESS on the server's own Tier-3
 * (hourly) clock instead of over HTTP — so there is no client-facing
 * proxy timeout to exceed, regardless of how large the archive grows.
 *
 * PERSISTS TO THE RAILWAY VOLUME (`DATA_DIR`/"/data/voltrade", `/tmp`
 * fallback — same convention as tiles3dBudget.ts's own state dir: this is
 * operational capture state, not a manifested datacore stream, so it
 * resolves the dir directly rather than via archiveBaseDir()), NOT the
 * git-tracked `datacore/port_dwell_weekly.json` research artifact a
 * session commits by hand. The two stay deliberately separate: this file
 * is always a superset (or equal) of the committed one, and a future
 * session reconciles them by reading the new `portdwell_weekly_captured`
 * diag probe (cheap — returns this file's contents, no live fold) instead
 * of re-running the expensive per-week HTTP loop
 * `portdwell_weekly_snapshot.ts` still contains. Migrating that script to
 * prefer the cheap probe is deliberately left for a follow-up session —
 * one logical change per PR; this one only builds the capture path and
 * the read surface.
 *
 * Bounded cost by construction: `captureIfDue` attempts AT MOST ONE
 * 168h fold per call — the single oldest week that is neither already
 * captured nor already recorded as permanently skipped (coverage-gapped
 * weeks can never become reachable again, since raw retention only rolls
 * forward; a degenerate all-zero read is recorded once and not retried
 * every tick either, since the same historical window reads the same
 * archive content on every future attempt). A tick with nothing new due
 * costs a couple of small JSON reads and a Set lookup — no fold, no
 * archive scan.
 *
 * The compute step is injected (defaults to the real
 * `computePortDwellAsyncTimed`) so the orchestration logic here — which
 * week to pick, when to skip, what to persist — is unit-testable with a
 * canned `PortDwellStats` fixture, without a live vessel archive.
 */
import fs from "node:fs";
import path from "node:path";
import {
  weekBounds, missingWeekIndices, mergeWeeklySnapshot, extractWeeklySnapshot,
  isDegenerateAllZeroRead, ARCHIVE_START_MS, type WeeklySnapshot,
} from "./portDwellWeekly";
import { computePortDwellAsyncTimed, type PortDef, type PortDwellStats } from "./portDwell";

// Same DATA_DIR-or-volume-or-/tmp convention as tiles3dBudget.ts's own
// state dir — durable across the process lifetime and across redeploys
// (the Railway volume), degrading gracefully to /tmp locally.
export function portDwellCaptureStateDir(env: NodeJS.ProcessEnv = process.env): string {
  return env.DATA_DIR || (fs.existsSync("/data") ? "/data/voltrade" : "/tmp");
}
const SNAPSHOTS_FILE = "voltrade_port_dwell_weekly_captured.json";
const SKIPPED_FILE = "voltrade_port_dwell_weekly_skipped.json";

interface SkippedRecord { result: "skipped_degenerate"; at: string; detail?: string }

export function loadCapturedSnapshots(dir = portDwellCaptureStateDir()): WeeklySnapshot[] {
  try {
    const raw = JSON.parse(fs.readFileSync(path.join(dir, SNAPSHOTS_FILE), "utf8"));
    return Array.isArray(raw) ? raw : [];
  } catch { return []; }
}

function writeCapturedSnapshots(dir: string, snaps: WeeklySnapshot[]): void {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, SNAPSHOTS_FILE), JSON.stringify(snaps, null, 2) + "\n");
}

function loadSkipped(dir: string): Map<number, SkippedRecord> {
  try {
    const raw = JSON.parse(fs.readFileSync(path.join(dir, SKIPPED_FILE), "utf8")) as Record<string, SkippedRecord>;
    return new Map(Object.entries(raw).map(([k, v]) => [Number(k), v]));
  } catch { return new Map(); }
}

function writeSkipped(dir: string, skipped: Map<number, SkippedRecord>): void {
  fs.mkdirSync(dir, { recursive: true });
  const obj: Record<string, SkippedRecord> = {};
  for (const [k, v] of skipped) obj[String(k)] = v;
  fs.writeFileSync(path.join(dir, SKIPPED_FILE), JSON.stringify(obj, null, 2) + "\n");
}

/** Pure: the earliest week index worth attempting, derived from the live
 *  raw-retention floor — mirrors `portdwell_weekly_snapshot.ts`'s own
 *  identical formula (never assume week 0/archive-start is reachable; raw
 *  retention is a rolling window, not a growing one). `null` (no known
 *  floor, e.g. an empty archive) falls back to week 0. */
export function earliestAttemptableWeekIndex(oldestRawHourMs: number | null): number {
  if (oldestRawHourMs == null) return 0;
  return Math.max(0, Math.ceil((oldestRawHourMs - ARCHIVE_START_MS) / (7 * 24 * 3600_000)));
}

/** Pure: the single oldest week that is neither captured nor already
 *  recorded as permanently skipped, or null if none is due. Reuses
 *  portDwellWeekly.ts's own `missingWeekIndices` (EDGE DOCTRINE #3 —
 *  compile once) rather than re-deriving the same range logic. */
export function nextCaptureTarget(existing: WeeklySnapshot[], skippedIndices: ReadonlySet<number>,
                                   oldestRawHourMs: number | null, nowMs: number): number | null {
  const earliestIndex = earliestAttemptableWeekIndex(oldestRawHourMs);
  const missing = missingWeekIndices(existing, earliestIndex, nowMs);
  const idx = missing.find((k) => !skippedIndices.has(k));
  return idx === undefined ? null : idx;
}

export interface CaptureResult {
  action: "captured" | "skipped_degenerate" | "no_missing_week";
  week_index?: number;
  detail?: string;
}

type ComputeFn = (ports: PortDef[], windowHours: number, baseDir: string | undefined, nowMs: number) =>
  Promise<PortDwellStats & { pointsScanned: number; elapsedMs: number }>;

/** The one impure entry point — called from bot.ts's Tier 3 hourly tick.
 *  `ports`/`oldestRawHourMs`/`nowMs` are caller-supplied (bot.ts already
 *  computes all three for the `portdwell_window` diag probe), so this
 *  module needs no import of datacoreArchive.ts or the sites registry
 *  itself. `computeFn` defaults to the real fold; tests inject a fake to
 *  exercise the orchestration logic against a canned PortDwellStats
 *  without a live archive. */
export async function captureIfDue(ports: PortDef[], oldestRawHourMs: number | null, nowMs: number,
                                    dir = portDwellCaptureStateDir(),
                                    computeFn: ComputeFn = computePortDwellAsyncTimed): Promise<CaptureResult> {
  const existing = loadCapturedSnapshots(dir);
  const skipped = loadSkipped(dir);
  const idx = nextCaptureTarget(existing, new Set(skipped.keys()), oldestRawHourMs, nowMs);
  if (idx == null) return { action: "no_missing_week" };

  // No separate "is this week within raw retention" check here:
  // nextCaptureTarget already derives its search range from
  // earliestAttemptableWeekIndex(oldestRawHourMs), which guarantees any
  // index it returns satisfies weekBounds(idx).startMs >= oldestRawHourMs
  // — a week whose start predates the retention floor is never a
  // candidate in the first place, so there is nothing left to guard here.
  const week = weekBounds(idx);
  const stats = await computeFn(ports, 168, undefined, week.endMs);
  if (isDegenerateAllZeroRead(stats)) {
    const detail = `all-zero read despite vessels_seen=${stats.vessels_seen}`;
    skipped.set(idx, { result: "skipped_degenerate", at: new Date(nowMs).toISOString(), detail });
    writeSkipped(dir, skipped);
    return { action: "skipped_degenerate", week_index: idx, detail };
  }

  const snap = extractWeeklySnapshot(stats, week, nowMs);
  writeCapturedSnapshots(dir, mergeWeeklySnapshot(existing, snap));
  return { action: "captured", week_index: idx };
}
