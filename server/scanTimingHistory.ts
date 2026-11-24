// 2026-11-24 (KNOWN BROKEN #60 recurrence; research/open_questions.md (al3)).
//
// /api/diag/timings served ONE scan, the latest. A reading of "how long is
// what follows the options stage" needed a session to poll it, and a scan the
// daemon abandoned at 300 s left no timings at all once the next scan started.
// scan_timing_log.py now keeps one line per scan by UTC day; this module reads
// a day of them and says what a reader needs to judge the 300 s budget:
// how many scans ran past it, and how long the steps after the options stage
// took, split by whether the CSP scores were recomputed that scan.
//
// A record of our own scans. Nothing on a trade path calls this.
import fs from "node:fs";
import path from "node:path";

export const SCAN_TIMING_LOG_DIR = "scan_timing_log";
export const SCAN_TIMEOUT_SEC = 300;     // the daemon's run_full_scan timeout (server/bot.ts)
export const HISTORY_ROWS_SERVED = 60;   // newest rows returned beside the summary

export interface ScanTimingRow {
  scan_started: number;
  day: string;
  status: "completed" | "unfinished";
  total_sec: number | null;
  reached_sec: number | null;
  last_phase: string | null;
  options_stage_end_sec: number | null;
  tier_engine_start_sec: number | null;
  after_options_sec: number | null;
  options: { deadline_sec?: number | null; candidates?: number | null; not_reached?: number | null; loop_sec?: number | null };
  /** the options step's seconds outside its per-ticker loop: no deadline covers them */
  options_outside_loop_sec?: number | null;
  tier: Record<string, number>;
  layer2: { cache_hit?: boolean | null; elapsed_sec?: number | null; completed?: number | null; total?: number | null };
  phase_sec: Record<string, number>;
}

export interface Spread { n: number; min: number; median: number; p90: number; max: number }

export interface ScanDaySummary {
  day: string;
  scans: number;
  completed: number;
  unfinished: number;
  /** completed with total_sec at or over the timeout, plus every unfinished scan (abandoned by the daemon and never finished, or cut by a restart: the last phase says where) */
  past_timeout: number;
  past_timeout_last_phases: Record<string, number>;
  total_sec: Spread | null;
  /** scans on which the options stage's deadline left candidates unread */
  deadline_bound: number;
  not_reached_when_bound: Spread | null;
  options_stage_end_sec: Spread | null;
  options_outside_loop_sec: Spread | null;
  /** seconds from the end of the options stage to the end of the scan, completed scans only */
  after_options_sec: { all: Spread | null; scores_recomputed: Spread | null; scores_cached: Spread | null };
  tier_engine_sec: Spread | null;
  malformed_lines: number;
}

function fin(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

/** min / median / p90 / max by nearest rank; null for an empty list. */
export function spread(values: number[]): Spread | null {
  const v = values.filter(fin).sort((a, b) => a - b);
  if (!v.length) return null;
  const rank = (q: number) => v[Math.min(v.length - 1, Math.max(0, Math.ceil(q * v.length) - 1))];
  const r2 = (x: number) => Math.round(x * 100) / 100;
  return { n: v.length, min: r2(v[0]), median: r2(rank(0.5)), p90: r2(rank(0.9)), max: r2(v[v.length - 1]) };
}

/** One row per well-formed line; lines that are not a scan row are counted, not guessed at. */
export function parseScanTimingLines(text: string): { rows: ScanTimingRow[]; malformed: number } {
  const rows: ScanTimingRow[] = [];
  let malformed = 0;
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(line);
    } catch (e) {
      malformed += 1;   // a line cut by a restart mid-write
      continue;
    }
    const o = (parsed && typeof parsed === "object" ? parsed : {}) as Record<string, unknown>;
    if (!fin(o.scan_started) || (o.status !== "completed" && o.status !== "unfinished")) {
      malformed += 1;
      continue;
    }
    const obj = <T>(v: unknown): T => (v && typeof v === "object" ? v : {}) as T;
    rows.push({
      ...(o as unknown as ScanTimingRow),
      options: obj<ScanTimingRow["options"]>(o.options),
      tier: obj<ScanTimingRow["tier"]>(o.tier),
      layer2: obj<ScanTimingRow["layer2"]>(o.layer2),
      phase_sec: obj<ScanTimingRow["phase_sec"]>(o.phase_sec),
    });
  }
  return { rows, malformed };
}

export function summarizeScanDay(day: string, rows: ScanTimingRow[], malformed = 0): ScanDaySummary {
  const completed = rows.filter((r) => r.status === "completed" && fin(r.total_sec));
  const unfinished = rows.filter((r) => !(r.status === "completed" && fin(r.total_sec)));
  const past = rows.filter((r) => !(r.status === "completed" && fin(r.total_sec)) || (r.total_sec as number) >= SCAN_TIMEOUT_SEC);
  const lastPhases: Record<string, number> = {};
  for (const r of past) {
    const k = r.status === "completed" ? "finished late" : String(r.last_phase || "unknown");
    lastPhases[k] = (lastPhases[k] || 0) + 1;
  }
  const bound = rows.filter((r) => fin(r.options.not_reached) && (r.options.not_reached as number) > 0);
  const after = completed.filter((r) => fin(r.after_options_sec));
  const nums = (rs: ScanTimingRow[], f: (r: ScanTimingRow) => unknown) => rs.map(f).filter(fin) as number[];
  return {
    day,
    scans: rows.length,
    completed: completed.length,
    unfinished: unfinished.length,
    past_timeout: past.length,
    past_timeout_last_phases: lastPhases,
    total_sec: spread(nums(completed, (r) => r.total_sec)),
    deadline_bound: bound.length,
    not_reached_when_bound: spread(nums(bound, (r) => r.options.not_reached)),
    options_stage_end_sec: spread(nums(rows, (r) => r.options_stage_end_sec)),
    options_outside_loop_sec: spread(nums(rows, (r) => r.options_outside_loop_sec)),
    after_options_sec: {
      all: spread(nums(after, (r) => r.after_options_sec)),
      scores_recomputed: spread(nums(after.filter((r) => r.layer2.cache_hit === false), (r) => r.after_options_sec)),
      scores_cached: spread(nums(after.filter((r) => r.layer2.cache_hit === true), (r) => r.after_options_sec)),
    },
    tier_engine_sec: spread(nums(completed, (r) => r.tier.total_sec)),
    malformed_lines: malformed,
  };
}

/** "2026-11-24" exactly, else null: the value becomes part of a file name. */
export function parseHistoryDay(raw: unknown): string | null {
  return typeof raw === "string" && /^\d{4}-\d{2}-\d{2}$/.test(raw) ? raw : null;
}

export function scanTimingDataDirs(env: Record<string, string | undefined> = process.env): string[] {
  return env.DATA_DIR ? [env.DATA_DIR] : ["/data/voltrade", "/tmp"];
}

export interface ScanTimingHistory {
  day: string;
  found: boolean;
  days_held: string[];
  summary: ScanDaySummary | null;
  rows_total: number;
  rows: ScanTimingRow[];
}

/**
 * A day of the log: the summary over every row, and the newest
 * HISTORY_ROWS_SERVED rows with `rows_total` beside them so a cut list says
 * that it is cut. `day` null means the newest day held.
 */
export function readScanTimingHistory(dataDirs: string[], day: string | null): ScanTimingHistory {
  for (const dir of dataDirs) {
    const logDir = path.join(dir, SCAN_TIMING_LOG_DIR);
    if (!fs.existsSync(logDir)) continue;
    const days = fs.readdirSync(logDir)
      .filter((n) => /^\d{4}-\d{2}-\d{2}\.jsonl$/.test(n))
      .map((n) => n.slice(0, 10))
      .sort();
    const want = day ?? days[days.length - 1] ?? null;
    if (!want || !days.includes(want)) {
      return { day: want ?? "", found: false, days_held: days, summary: null, rows_total: 0, rows: [] };
    }
    const { rows, malformed } = parseScanTimingLines(fs.readFileSync(path.join(logDir, `${want}.jsonl`), "utf8"));
    return {
      day: want,
      found: true,
      days_held: days,
      summary: summarizeScanDay(want, rows, malformed),
      rows_total: rows.length,
      rows: rows.slice(-HISTORY_ROWS_SERVED),
    };
  }
  return { day: day ?? "", found: false, days_held: [], summary: null, rows_total: 0, rows: [] };
}
