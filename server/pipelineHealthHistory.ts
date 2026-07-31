/**
 * pipelineHealthHistory.ts — MAP V2 ROADMAP R6(c) PIPELINE-HEALTH dashboard,
 * the time-series half. (a) SIGNAL-STRENGTH and (b) DATA-QUALITY (both
 * shipped 2026-07-30) were pure joins over state that already existed
 * elsewhere; this one is NOT — /api/health (server/bot.ts) only ever
 * reports the current instant, so a "checks history / provider backoff
 * states / compliance status" panel needs its own capture, per the R6(c)
 * roadmap note this session found unbuilt in research/open_questions.md.
 *
 * Design choice: no new poller/timer. /api/health is already the single
 * place that computes every check (database/alpaca/python/bot/scanner/
 * licensing) each time it is hit, and Railway (plus any external monitor)
 * already polls it on a steady cadence for platform health — so
 * recordSnapshot() is called from INSIDE that existing handler and simply
 * throttles itself (module-level last-write clock, same tolerance pattern
 * as datacoreArchive.ts's per-entity lastWrite map: a process restart just
 * writes one extra sample early, harmless) to avoid writing on every hit.
 * Zero new network calls, zero new timers, one append per throttle window.
 *
 * Storage: one flat JSONL line per snapshot, one file per UTC day, at
 * `<archiveBaseDir>/pipeline_health/YYYY-MM-DD.jsonl` — the same day-file
 * convention as cftcTff.ts/edgarForm4.ts/etc. Small (one ~150-byte line
 * per throttle window; default 5 min -> ~288 rows/day, ~45KB/day plain,
 * no gzip needed at that size).
 */
import fs from "fs";
import path from "path";
import { archiveBaseDir } from "./datacoreArchive";

export interface PipelineHealthSnapshot {
  t: number;                 // unix seconds
  status: "ok" | "degraded";
  database_ok: boolean;
  alpaca_ok: boolean;
  python_ok: boolean;
  scanner_ok: boolean;
  scanner_consecutive_failures: number;
  licensing_ok: boolean;
  bot_status: string;        // "active" | "stopped" | "killed"
  bot_liveness_dark: boolean;
  heap_used_mb: number;
  rss_mb: number;
}

function healthDir(baseDir?: string): string {
  return path.join(baseDir || archiveBaseDir(), "pipeline_health");
}

function dayFile(dir: string, when: Date): string {
  return path.join(dir, `${when.toISOString().slice(0, 10)}.jsonl`);
}

/** Builds the flat snapshot record from the exact `checks` object /api/health
 *  already computes — no re-derivation of any check, just a flatten. */
export function snapshotFromHealthChecks(checks: any, nowMs?: number): PipelineHealthSnapshot {
  const c = checks?.checks || {};
  return {
    t: Math.floor((nowMs ?? Date.now()) / 1000),
    status: checks?.status === "ok" ? "ok" : "degraded",
    database_ok: c.database?.status === "ok",
    alpaca_ok: c.alpaca?.status === "ok",
    python_ok: c.python?.status === "ok",
    scanner_ok: c.scanner?.status === "ok",
    scanner_consecutive_failures: Number(c.scanner?.consecutiveFailures) || 0,
    licensing_ok: c.licensing?.status === "ok",
    bot_status: c.bot?.status ?? "unknown",
    bot_liveness_dark: c.bot?.liveness?.dark === true,
    heap_used_mb: Number(c.server?.heap_used_mb) || 0,
    rss_mb: Number(c.server?.rss_mb) || 0,
  };
}

// Throttle clock: one real disk write per window regardless of how often
// /api/health itself is hit. Reset seam for hermetic tests below.
let lastRecordedAt = 0;

export function resetHealthHistoryThrottleForTest(): void {
  lastRecordedAt = 0;
}

export const DEFAULT_MIN_INTERVAL_MS = 5 * 60_000;

/** Appends a snapshot iff at least `minIntervalMs` has passed since the last
 *  one. Returns true iff a line was actually written (useful for tests). */
export function recordHealthSnapshot(
  checks: any,
  baseDir?: string,
  nowMs?: number,
  minIntervalMs = DEFAULT_MIN_INTERVAL_MS,
): boolean {
  const now = nowMs ?? Date.now();
  if (now - lastRecordedAt < minIntervalMs) return false;
  lastRecordedAt = now;
  try {
    const dir = healthDir(baseDir);
    fs.mkdirSync(dir, { recursive: true });
    const snap = snapshotFromHealthChecks(checks, now);
    fs.appendFileSync(dayFile(dir, new Date(now)), JSON.stringify(snap) + "\n");
    return true;
  } catch (e: any) {
    console.error("[pipeline-health] record:", e?.message || e);
    return false;
  }
}

/** Reads every snapshot from the last `days` UTC calendar days (today
 *  inclusive), oldest first. Missing/corrupt files are skipped, not fatal —
 *  matches every other archive reader's per-file try/catch discipline. */
export function readHealthHistory(days: number, baseDir?: string, nowMs?: number): PipelineHealthSnapshot[] {
  const dir = healthDir(baseDir);
  const now = nowMs ?? Date.now();
  const out: PipelineHealthSnapshot[] = [];
  for (let i = days - 1; i >= 0; i--) {
    const fp = dayFile(dir, new Date(now - i * 86400_000));
    try {
      const text = fs.readFileSync(fp, "utf8");
      for (const line of text.split("\n")) {
        if (!line) continue;
        try { out.push(JSON.parse(line)); } catch {}
      }
    } catch {}
  }
  out.sort((a, b) => a.t - b.t);
  return out;
}

export interface PipelineHealthSummary {
  window_hours: number;
  sample_count: number;
  uptime_pct: number | null;   // null when there is no data yet (honest, never fabricated as 100%)
  current: PipelineHealthSnapshot | null;
  degraded_counts: {
    database: number; alpaca: number; python: number;
    scanner: number; licensing: number; bot_liveness_dark: number;
  };
  timeline: Array<{ t: number; status: "ok" | "degraded" }>;
}

const MAX_TIMELINE_POINTS = 200;

/** Pure aggregation over an already-loaded snapshot list (caller filters to
 *  the desired window) — sample-and-hold semantics: a window with rows is
 *  measured on those rows only, never padded with assumed-healthy gaps. */
export function summarizePipelineHealth(
  rows: PipelineHealthSnapshot[],
  windowHours: number,
): PipelineHealthSummary {
  const degraded_counts = { database: 0, alpaca: 0, python: 0, scanner: 0, licensing: 0, bot_liveness_dark: 0 };
  let okCount = 0;
  for (const r of rows) {
    if (r.status === "ok") okCount++;
    if (!r.database_ok) degraded_counts.database++;
    if (!r.alpaca_ok) degraded_counts.alpaca++;
    if (!r.python_ok) degraded_counts.python++;
    if (!r.scanner_ok) degraded_counts.scanner++;
    if (!r.licensing_ok) degraded_counts.licensing++;
    if (r.bot_liveness_dark) degraded_counts.bot_liveness_dark++;
  }
  const step = Math.max(1, Math.ceil(rows.length / MAX_TIMELINE_POINTS));
  const timeline = rows.filter((_, idx) => idx % step === 0).map((r) => ({ t: r.t, status: r.status }));
  return {
    window_hours: windowHours,
    sample_count: rows.length,
    uptime_pct: rows.length > 0 ? Math.round((okCount / rows.length) * 1000) / 10 : null,
    current: rows.length > 0 ? rows[rows.length - 1] : null,
    degraded_counts,
    timeline,
  };
}

/** Convenience: filters an already-read history list to the trailing
 *  `hours` window ending at nowMs, then summarizes it. */
export function summarizeWindow(
  rows: PipelineHealthSnapshot[], hours: number, nowMs?: number,
): PipelineHealthSummary {
  const now = nowMs ?? Date.now();
  const cutoff = Math.floor(now / 1000) - hours * 3600;
  return summarizePipelineHealth(rows.filter((r) => r.t >= cutoff), hours);
}
